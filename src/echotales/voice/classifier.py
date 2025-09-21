import numpy as np
import librosa
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torchaudio
from transformers import pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import joblib
import pickle

from ..core.models import VoiceProfile, Character, Gender, AgeGroup

logger = logging.getLogger(__name__)


class VoiceFeatureExtractor:
    """Advanced voice feature extraction using multiple techniques"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def extract_acoustic_features(self, audio_path: str) -> Dict[str, float]:
        """Extract comprehensive acoustic features from audio"""
        try:
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Fundamental frequency and pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = pitches[pitches > 0]
            pitch_mean = np.mean(pitch_values) if len(pitch_values) > 0 else 0.0
            pitch_std = np.std(pitch_values) if len(pitch_values) > 0 else 0.0
            pitch_range = np.max(pitch_values) - np.min(pitch_values) if len(pitch_values) > 0 else 0.0
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            
            # MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_features = {f'mfcc_{i}': float(np.mean(mfcc)) for i, mfcc in enumerate(mfccs)}
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            
            # Tempo and rhythm
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Energy and dynamics
            rms_energy = librosa.feature.rms(y=y)[0]
            energy_variance = np.var(rms_energy)
            
            # Harmonics and percussive separation
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            harmonic_ratio = np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8)
            
            features = {
                # Pitch features
                'pitch_mean': float(pitch_mean),
                'pitch_std': float(pitch_std),
                'pitch_range': float(pitch_range),
                
                # Spectral features
                'spectral_centroid_mean': float(np.mean(spectral_centroids)),
                'spectral_centroid_std': float(np.std(spectral_centroids)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
                'spectral_contrast_mean': float(np.mean(spectral_contrast)),
                
                # Rhythm features
                'tempo': float(tempo),
                'zcr_mean': float(np.mean(zcr)),
                'zcr_std': float(np.std(zcr)),
                
                # Energy features
                'rms_energy_mean': float(np.mean(rms_energy)),
                'energy_variance': float(energy_variance),
                'harmonic_ratio': float(harmonic_ratio),
                
                # Chroma features
                **{f'chroma_{i}': float(val) for i, val in enumerate(chroma_mean)},
                
                # Duration
                'duration': float(len(y) / sr)
            }
            
            # Add MFCC features
            features.update(mfcc_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            return {}
    
    def extract_wav2vec_features(self, audio_path: str, processor, model) -> np.ndarray:
        """Extract features using Wav2Vec2 model"""
        try:
            audio_input, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                audio_input = resampler(audio_input)
            
            # Process audio
            inputs = processor(audio_input.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting Wav2Vec2 features from {audio_path}: {e}")
            return np.array([])


class GeminiVoiceClassifier:
    """Primary classifier using Gemini API for voice analysis"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize Gemini client here when available
        self.client = None
        
    def classify_voice_characteristics(self, audio_path: str) -> Dict[str, str]:
        """Classify voice using Gemini API"""
        # Implementation would use Gemini API to analyze audio
        # This is a placeholder for the actual implementation
        try:
            # Would call Gemini API here
            return {
                'gender': 'unknown',
                'age_group': 'adult',
                'emotion': 'neutral',
                'accent': 'neutral',
                'speaking_style': 'conversational'
            }
        except Exception as e:
            logger.error(f"Gemini API classification failed: {e}")
            return {}


class MLVoiceClassifier:
    """Machine learning-based voice classifier as fallback"""
    
    def __init__(self, models_path: str):
        self.models_path = Path(models_path)
        self.feature_extractor = VoiceFeatureExtractor()
        
        # Initialize models
        self.gender_classifier = None
        self.age_classifier = None
        self.emotion_classifier = None
        self.scaler = None
        
        # Wav2Vec2 for deep features
        self.wav2vec_processor = None
        self.wav2vec_model = None
        
        self._load_or_train_models()
    
    def _load_or_train_models(self):
        """Load pre-trained models or train new ones"""
        try:
            # Load Wav2Vec2 model for feature extraction
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2ForSequenceClassification.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Try to load existing models
            gender_model_path = self.models_path / "gender_classifier.pkl"
            age_model_path = self.models_path / "age_classifier.pkl"
            emotion_model_path = self.models_path / "emotion_classifier.pkl"
            scaler_path = self.models_path / "feature_scaler.pkl"
            
            if all(path.exists() for path in [gender_model_path, age_model_path, scaler_path]):
                self.gender_classifier = joblib.load(gender_model_path)
                self.age_classifier = joblib.load(age_model_path)
                self.emotion_classifier = joblib.load(emotion_model_path)
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded pre-trained ML models")
            else:
                logger.info("Pre-trained models not found, will train on first use")
                
        except Exception as e:
            logger.error(f"Error loading ML models: {e}")
    
    def train_models(self, training_data: List[Tuple[str, Dict[str, str]]]):
        """Train ML models on labeled voice data"""
        if not training_data:
            logger.warning("No training data provided")
            return
            
        logger.info(f"Training ML models on {len(training_data)} samples")
        
        # Extract features from all audio files
        features_list = []
        gender_labels = []
        age_labels = []
        emotion_labels = []
        
        for audio_path, labels in training_data:
            features = self.feature_extractor.extract_acoustic_features(audio_path)
            if features:
                # Convert features dict to array
                feature_vector = list(features.values())
                features_list.append(feature_vector)
                
                gender_labels.append(labels.get('gender', 'unknown'))
                age_labels.append(labels.get('age_group', 'adult'))
                emotion_labels.append(labels.get('emotion', 'neutral'))
        
        if not features_list:
            logger.error("No valid features extracted from training data")
            return
            
        # Convert to numpy arrays
        X = np.array(features_list)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train gender classifier
        self.gender_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.gender_classifier.fit(X_scaled, gender_labels)
        
        # Train age classifier  
        self.age_classifier = SVC(kernel='rbf', probability=True, random_state=42)
        self.age_classifier.fit(X_scaled, age_labels)
        
        # Train emotion classifier
        self.emotion_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.emotion_classifier.fit(X_scaled, emotion_labels)
        
        # Save models
        self.models_path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.gender_classifier, self.models_path / "gender_classifier.pkl")
        joblib.dump(self.age_classifier, self.models_path / "age_classifier.pkl") 
        joblib.dump(self.emotion_classifier, self.models_path / "emotion_classifier.pkl")
        joblib.dump(self.scaler, self.models_path / "feature_scaler.pkl")
        
        logger.info("ML models trained and saved successfully")
    
    def classify_voice(self, audio_path: str) -> Dict[str, str]:
        """Classify voice characteristics using ML models"""
        if not all([self.gender_classifier, self.age_classifier, self.scaler]):
            logger.warning("ML models not trained, returning default values")
            return {
                'gender': 'unknown',
                'age_group': 'adult', 
                'emotion': 'neutral'
            }
            
        # Extract features
        features = self.feature_extractor.extract_acoustic_features(audio_path)
        if not features:
            return {'gender': 'unknown', 'age_group': 'adult', 'emotion': 'neutral'}
            
        # Convert to array and scale
        feature_vector = np.array(list(features.values())).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make predictions
        gender_pred = self.gender_classifier.predict(feature_vector_scaled)[0]
        age_pred = self.age_classifier.predict(feature_vector_scaled)[0]
        emotion_pred = self.emotion_classifier.predict(feature_vector_scaled)[0]
        
        return {
            'gender': gender_pred,
            'age_group': age_pred,
            'emotion': emotion_pred
        }


class TransformerVoiceClassifier:
    """Transformer-based voice classifier as secondary fallback"""
    
    def __init__(self):
        # Initialize pre-trained models for voice analysis
        try:
            # Use speech emotion recognition model
            self.emotion_classifier = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er"
            )
            
            # Use speaker verification for characteristics
            self.speaker_classifier = pipeline(
                "audio-classification", 
                model="microsoft/wavlm-base-plus-sv"
            )
            
            logger.info("Loaded transformer models for voice classification")
            
        except Exception as e:
            logger.error(f"Error loading transformer models: {e}")
            self.emotion_classifier = None
            self.speaker_classifier = None
    
    def classify_voice(self, audio_path: str) -> Dict[str, str]:
        """Classify voice using transformer models"""
        results = {
            'gender': 'unknown',
            'age_group': 'adult',
            'emotion': 'neutral'
        }
        
        try:
            # Emotion classification
            if self.emotion_classifier:
                emotion_result = self.emotion_classifier(audio_path)
                if emotion_result:
                    results['emotion'] = emotion_result[0]['label'].lower()
            
            # Additional analysis using speaker classifier
            if self.speaker_classifier:
                speaker_result = self.speaker_classifier(audio_path)
                # Process speaker results to infer gender/age characteristics
                
        except Exception as e:
            logger.error(f"Transformer classification failed: {e}")
            
        return results


class VoiceDatasetClassifier:
    """Advanced voice classification and character matching system with multiple fallbacks"""
    
    def __init__(self, dataset_path: str, models_path: str, gemini_api_key: Optional[str] = None):
        self.dataset_path = Path(dataset_path)
        self.male_actors_path = self.dataset_path / "male_actors"
        self.female_actors_path = self.dataset_path / "female_actors"
        self.processed_path = self.dataset_path / "processed_samples"
        self.models_path = Path(models_path)
        
        # Initialize classifiers in order of preference
        self.gemini_classifier = GeminiVoiceClassifier(gemini_api_key) if gemini_api_key else None
        self.ml_classifier = MLVoiceClassifier(str(self.models_path))
        self.transformer_classifier = TransformerVoiceClassifier()
        
        # Voice profiles database
        self.voice_profiles = {}
        self.actor_features = {}
        
        logger.info("Voice dataset classifier initialized with multiple fallback models")
    
    def analyze_voice_features(self, audio_path: str) -> Dict[str, float]:
        """Extract detailed voice features for classification"""
        return VoiceFeatureExtractor().extract_acoustic_features(audio_path)
    
    def classify_voice_with_fallbacks(self, audio_path: str) -> Dict[str, str]:
        """Classify voice using primary model with fallbacks"""
        
        # Try Gemini API first
        if self.gemini_classifier:
            try:
                result = self.gemini_classifier.classify_voice_characteristics(audio_path)
                if result and result.get('gender') != 'unknown':
                    logger.info("Voice classified using Gemini API")
                    return result
            except Exception as e:
                logger.warning(f"Gemini classification failed, using fallback: {e}")
        
        # Try ML classifier
        try:
            result = self.ml_classifier.classify_voice(audio_path)
            if result and result.get('gender') != 'unknown':
                logger.info("Voice classified using ML models")
                return result
        except Exception as e:
            logger.warning(f"ML classification failed, using transformer fallback: {e}")
        
        # Try transformer classifier as final fallback
        try:
            result = self.transformer_classifier.classify_voice(audio_path)
            logger.info("Voice classified using transformer models")
            return result
        except Exception as e:
            logger.error(f"All classification methods failed: {e}")
            return {
                'gender': 'unknown',
                'age_group': 'adult',
                'emotion': 'neutral'
            }
    
    def process_actor_dataset(self) -> Dict[str, List[VoiceProfile]]:
        """Process all 24 actors and their 44 samples each"""
        logger.info("Processing voice actor dataset...")
        
        actor_profiles = {'male': [], 'female': []}
        
        # Process male actors
        if self.male_actors_path.exists():
            for actor_dir in self.male_actors_path.iterdir():
                if actor_dir.is_dir():
                    profiles = self._process_actor_samples(actor_dir, Gender.MALE)
                    actor_profiles['male'].extend(profiles)
        
        # Process female actors  
        if self.female_actors_path.exists():
            for actor_dir in self.female_actors_path.iterdir():
                if actor_dir.is_dir():
                    profiles = self._process_actor_samples(actor_dir, Gender.FEMALE)
                    actor_profiles['female'].extend(profiles)
        
        logger.info(f"Processed {len(actor_profiles['male'])} male and {len(actor_profiles['female'])} female voice profiles")
        
        return actor_profiles
    
    def _process_actor_samples(self, actor_dir: Path, gender: Gender) -> List[VoiceProfile]:
        """Process all samples for a single actor"""
        actor_id = actor_dir.name
        sample_files = list(actor_dir.glob("*.wav")) + list(actor_dir.glob("*.mp3")) + list(actor_dir.glob("*.flac"))
        
        if not sample_files:
            logger.warning(f"No audio samples found for actor {actor_id}")
            return []
        
        profiles = []
        
        for sample_file in sample_files[:44]:  # Limit to 44 samples as specified
            try:
                # Extract features
                features = self.analyze_voice_features(str(sample_file))
                
                # Classify voice characteristics
                classification = self.classify_voice_with_fallbacks(str(sample_file))
                
                # Determine age group based on voice features and classification
                age_group = AgeGroup(classification.get('age_group', 'adult'))
                
                # Create voice profile
                profile = VoiceProfile(
                    actor_id=actor_id,
                    gender=gender,
                    age_group=age_group,
                    pitch=features.get('pitch_mean', 0.5) / 500.0,  # Normalize pitch
                    tempo=min(features.get('tempo', 120.0) / 200.0, 1.0),  # Normalize tempo
                    emotion=classification.get('emotion', 'neutral'),
                    quality_score=self._calculate_quality_score(features),
                    sample_paths=[str(sample_file)]
                )
                
                profiles.append(profile)
                
            except Exception as e:
                logger.error(f"Error processing sample {sample_file}: {e}")
                continue
        
        return profiles
    
    def _calculate_quality_score(self, features: Dict[str, float]) -> float:
        """Calculate voice quality score based on features"""
        try:
            # Simple quality heuristic based on various factors
            noise_score = 1.0 - min(features.get('zcr_std', 0.1) / 0.2, 1.0)
            energy_score = min(features.get('rms_energy_mean', 0.1) * 10, 1.0)
            pitch_stability = 1.0 - min(features.get('pitch_std', 50.0) / 100.0, 1.0)
            
            quality_score = (noise_score + energy_score + pitch_stability) / 3.0
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def match_character_to_voice(self, character: Character, available_profiles: List[VoiceProfile]) -> Optional[VoiceProfile]:
        """Advanced character-voice matching algorithm"""
        if not available_profiles:
            return None
        
        # Filter by gender first
        gender_matches = [p for p in available_profiles if p.gender == character.gender]
        if not gender_matches:
            gender_matches = available_profiles  # Fallback to all if no gender match
        
        # Filter by age group if possible
        age_matches = [p for p in gender_matches if p.age_group == character.age_group]
        if not age_matches:
            age_matches = gender_matches  # Fallback to gender matches
        
        # Score each profile based on personality compatibility
        scored_profiles = []
        for profile in age_matches:
            score = self._calculate_character_voice_compatibility(character, profile)
            scored_profiles.append((profile, score))
        
        # Return the best match
        if scored_profiles:
            best_match = max(scored_profiles, key=lambda x: x[1])
            return best_match[0]
        
        return None
    
    def _calculate_character_voice_compatibility(self, character: Character, voice_profile: VoiceProfile) -> float:
        """Calculate compatibility score between character and voice profile"""
        score = 0.0
        
        # Gender match (high weight)
        if character.gender == voice_profile.gender:
            score += 0.4
        
        # Age group match (medium weight)
        if character.age_group == voice_profile.age_group:
            score += 0.3
        
        # Personality-voice matching (medium weight)
        personality_score = self._match_personality_to_voice(character.personality, voice_profile)
        score += personality_score * 0.2
        
        # Voice quality (low weight but important)
        score += voice_profile.quality_score * 0.1
        
        return score
    
    def _match_personality_to_voice(self, personality: 'PersonalityVector', voice_profile: VoiceProfile) -> float:
        """Match personality traits to voice characteristics"""
        score = 0.0
        
        # Extraversion correlates with tempo and energy
        if personality.extraversion > 0.6 and voice_profile.tempo > 0.6:
            score += 0.3
        elif personality.extraversion < 0.4 and voice_profile.tempo < 0.4:
            score += 0.3
        
        # Confidence correlates with pitch stability and energy
        if personality.confidence > 0.7 and voice_profile.quality_score > 0.7:
            score += 0.2
        
        # Agreeableness might correlate with softer, warmer voices
        # This would require more sophisticated voice feature analysis
        
        return min(score, 1.0)
    
    def train_on_dataset(self):
        """Train ML models using the voice actor dataset"""
        logger.info("Training ML models on voice actor dataset...")
        
        training_data = []
        
        # Collect training data from male actors
        for actor_dir in self.male_actors_path.iterdir():
            if actor_dir.is_dir():
                for sample_file in actor_dir.glob("*.wav"):
                    labels = {'gender': 'male', 'age_group': 'adult'}  # Default labels
                    training_data.append((str(sample_file), labels))
        
        # Collect training data from female actors
        for actor_dir in self.female_actors_path.iterdir():
            if actor_dir.is_dir():
                for sample_file in actor_dir.glob("*.wav"):
                    labels = {'gender': 'female', 'age_group': 'adult'}  # Default labels
                    training_data.append((str(sample_file), labels))
        
        # Train the ML classifier
        self.ml_classifier.train_models(training_data)