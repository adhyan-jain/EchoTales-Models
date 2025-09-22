"""
Voice Dataset Handler for EchoTales

This module handles voice dataset operations including:
- Audio ingestion with consistent character classifications
- Format normalization and quality control
- Noise reduction pipeline
- Character-voice matching with gender/age priority
- Comprehensive fallback mechanisms
"""

import os
import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .postprocessing import AudioProcessor, AudioProcessingConfig
from ..core.environment import get_config
from ..models.schemas import Character

logger = logging.getLogger(__name__)


class Gender(Enum):
    """Standardized gender classifications"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class AgeGroup(Enum):
    """Standardized age group classifications"""
    CHILD = "child"          # 5-12 years
    TEENAGER = "teenager"    # 13-19 years
    YOUNG_ADULT = "young_adult"  # 20-35 years
    ADULT = "adult"          # 36-55 years
    MIDDLE_AGED = "middle_aged"  # 56-70 years
    ELDERLY = "elderly"      # 70+ years
    UNKNOWN = "unknown"


class PersonalityTrait(Enum):
    """Standardized personality traits for both characters and voices"""
    # Primary emotional traits
    CONFIDENT = "confident"
    SHY = "shy"
    AGGRESSIVE = "aggressive"
    GENTLE = "gentle"
    CHEERFUL = "cheerful"
    MELANCHOLIC = "melancholic"
    CALM = "calm"
    ANXIOUS = "anxious"
    
    # Social traits
    OUTGOING = "outgoing"
    INTROVERTED = "introverted"
    FRIENDLY = "friendly"
    STERN = "stern"
    WARM = "warm"
    COLD = "cold"
    
    # Communication style traits
    ARTICULATE = "articulate"
    MUMBLING = "mumbling"
    FAST_SPEAKING = "fast_speaking"
    SLOW_SPEAKING = "slow_speaking"
    LOUD = "loud"
    SOFT_SPOKEN = "soft_spoken"
    
    # Character archetype traits
    HEROIC = "heroic"
    VILLAINOUS = "villainous"
    MYSTERIOUS = "mysterious"
    WISE = "wise"
    NAIVE = "naive"
    SARCASTIC = "sarcastic"
    SERIOUS = "serious"
    HUMOROUS = "humorous"


@dataclass
class VoiceProfile:
    """Standardized voice profile matching character classifications"""
    voice_id: str
    actor_id: str
    gender: Gender
    age_group: AgeGroup
    
    # Personality traits with confidence scores
    personality_traits: Dict[PersonalityTrait, float] = field(default_factory=dict)
    
    # Voice characteristics
    pitch_range: Tuple[float, float] = (80.0, 300.0)  # Hz
    speaking_rate: float = 150.0  # words per minute
    voice_quality: Dict[str, float] = field(default_factory=dict)  # breathiness, roughness, etc.
    
    # Audio file information
    audio_files: List[str] = field(default_factory=list)
    sample_rate: int = 22050
    total_duration: float = 0.0  # seconds
    quality_score: float = 0.0  # 0-1 quality rating
    
    # Metadata
    accent: str = "neutral"
    language: str = "en"
    recording_environment: str = "studio"
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "voice_id": self.voice_id,
            "actor_id": self.actor_id,
            "gender": self.gender.value,
            "age_group": self.age_group.value,
            "personality_traits": {trait.value: score for trait, score in self.personality_traits.items()},
            "pitch_range": list(self.pitch_range),
            "speaking_rate": self.speaking_rate,
            "voice_quality": self.voice_quality,
            "audio_files": self.audio_files,
            "sample_rate": self.sample_rate,
            "total_duration": self.total_duration,
            "quality_score": self.quality_score,
            "accent": self.accent,
            "language": self.language,
            "recording_environment": self.recording_environment,
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VoiceProfile":
        """Create from dictionary"""
        # Convert string enums back to enum objects
        gender = Gender(data.get("gender", "unknown"))
        age_group = AgeGroup(data.get("age_group", "unknown"))
        
        # Convert personality traits
        personality_traits = {}
        for trait_str, score in data.get("personality_traits", {}).items():
            try:
                trait = PersonalityTrait(trait_str)
                personality_traits[trait] = float(score)
            except ValueError:
                logger.warning(f"Unknown personality trait: {trait_str}")
        
        # Convert datetime
        created_at = datetime.fromisoformat(data.get("created_at", datetime.utcnow().isoformat()))
        
        return cls(
            voice_id=data.get("voice_id", ""),
            actor_id=data.get("actor_id", ""),
            gender=gender,
            age_group=age_group,
            personality_traits=personality_traits,
            pitch_range=tuple(data.get("pitch_range", [80.0, 300.0])),
            speaking_rate=data.get("speaking_rate", 150.0),
            voice_quality=data.get("voice_quality", {}),
            audio_files=data.get("audio_files", []),
            sample_rate=data.get("sample_rate", 22050),
            total_duration=data.get("total_duration", 0.0),
            quality_score=data.get("quality_score", 0.0),
            accent=data.get("accent", "neutral"),
            language=data.get("language", "en"),
            recording_environment=data.get("recording_environment", "studio"),
            created_at=created_at
        )


@dataclass
class VoiceDatasetConfig:
    """Configuration for voice dataset handling"""
    # Dataset paths
    male_voices_path: str = "data/voice_dataset/male_actors"
    female_voices_path: str = "data/voice_dataset/female_actors"
    processed_voices_path: str = "data/voice_dataset/processed"
    metadata_file: str = "data/voice_dataset/voice_profiles.json"
    
    # Audio processing
    target_sample_rate: int = 22050
    target_format: str = "wav"
    min_duration: float = 1.0  # seconds
    max_duration: float = 30.0  # seconds
    
    # Quality thresholds
    min_quality_score: float = 0.6
    noise_threshold_db: float = -40.0
    
    # Fallback settings
    create_fallback_voices: bool = True
    fallback_voice_duration: float = 3.0


class VoiceDatasetHandler:
    """Main voice dataset handler with comprehensive fallbacks"""
    
    def __init__(self, config: Optional[VoiceDatasetConfig] = None):
        self.config = config or VoiceDatasetConfig()
        self.app_config = get_config()
        
        # Initialize audio processor
        audio_config = AudioProcessingConfig(
            target_sample_rate=self.config.target_sample_rate,
            target_format=self.config.target_format
        )
        self.audio_processor = AudioProcessor(audio_config)
        
        # Voice profiles cache
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.load_voice_profiles()
    
    def load_voice_profiles(self):
        """Load voice profiles from metadata file with fallbacks"""
        try:
            metadata_path = Path(self.config.metadata_file)
            
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for voice_id, profile_data in data.items():
                    try:
                        profile = VoiceProfile.from_dict(profile_data)
                        self.voice_profiles[voice_id] = profile
                    except Exception as e:
                        logger.warning(f"Failed to load voice profile {voice_id}: {e}")
                
                logger.info(f"Loaded {len(self.voice_profiles)} voice profiles")
            else:
                logger.info("No existing voice profiles found, will create new ones")
                
        except Exception as e:
            logger.error(f"Failed to load voice profiles: {e}")
            # Start with empty profiles
            self.voice_profiles = {}
    
    def save_voice_profiles(self):
        """Save voice profiles to metadata file"""
        try:
            metadata_path = Path(self.config.metadata_file)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert profiles to dict format
            data = {}
            for voice_id, profile in self.voice_profiles.items():
                data[voice_id] = profile.to_dict()
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.voice_profiles)} voice profiles")
            
        except Exception as e:
            logger.error(f"Failed to save voice profiles: {e}")
    
    def ingest_voice_dataset(self, force_reingest: bool = False) -> Dict[str, Any]:
        """
        Ingest and process voice dataset with comprehensive error handling
        
        Args:
            force_reingest: Whether to force re-ingestion of existing files
            
        Returns:
            Dictionary with ingestion results and statistics
        """
        
        results = {
            "success": False,
            "ingested_voices": 0,
            "failed_voices": 0,
            "errors": [],
            "warnings": [],
            "voice_profiles_created": [],
            "stats": {}
        }
        
        try:
            # Process male voices
            male_results = self._ingest_gender_voices(
                Gender.MALE, 
                self.config.male_voices_path,
                force_reingest
            )
            
            # Process female voices
            female_results = self._ingest_gender_voices(
                Gender.FEMALE,
                self.config.female_voices_path, 
                force_reingest
            )
            
            # Combine results
            results["ingested_voices"] = male_results["ingested"] + female_results["ingested"]
            results["failed_voices"] = male_results["failed"] + female_results["failed"]
            results["errors"].extend(male_results["errors"])
            results["warnings"].extend(female_results["warnings"])
            results["voice_profiles_created"].extend(male_results["profiles"])
            results["voice_profiles_created"].extend(female_results["profiles"])
            
            # Save updated profiles
            self.save_voice_profiles()
            
            # Create fallback voices if needed
            if self.config.create_fallback_voices:
                self._create_fallback_voices()
            
            results["success"] = True
            results["stats"] = self._generate_dataset_stats()
            
            logger.info(f"Voice dataset ingestion completed: {results['ingested_voices']} voices ingested")
            
        except Exception as e:
            logger.error(f"Voice dataset ingestion failed: {e}")
            results["errors"].append(f"Dataset ingestion failed: {e}")
        
        return results
    
    def _ingest_gender_voices(self, 
                             gender: Gender, 
                             base_path: str,
                             force_reingest: bool) -> Dict[str, Any]:
        """Ingest voices for a specific gender"""
        
        results = {
            "ingested": 0,
            "failed": 0,
            "errors": [],
            "warnings": [],
            "profiles": []
        }
        
        try:
            base_path = Path(base_path)
            
            if not base_path.exists():
                logger.warning(f"Voice path does not exist: {base_path}")
                # Create fallback directory structure
                base_path.mkdir(parents=True, exist_ok=True)
                self._create_sample_voice_structure(base_path, gender)
                results["warnings"].append(f"Created fallback structure for {gender.value} voices")
            
            # Process each actor directory
            for actor_dir in base_path.iterdir():
                if actor_dir.is_dir():
                    try:
                        actor_results = self._process_actor_directory(
                            actor_dir, 
                            gender, 
                            force_reingest
                        )
                        
                        results["ingested"] += actor_results["ingested"]
                        results["failed"] += actor_results["failed"]
                        results["errors"].extend(actor_results["errors"])
                        results["profiles"].extend(actor_results["profiles"])
                        
                    except Exception as e:
                        logger.error(f"Failed to process actor directory {actor_dir}: {e}")
                        results["failed"] += 1
                        results["errors"].append(f"Actor {actor_dir.name}: {e}")
        
        except Exception as e:
            logger.error(f"Failed to ingest {gender.value} voices: {e}")
            results["errors"].append(f"{gender.value} ingestion failed: {e}")
        
        return results
    
    def _process_actor_directory(self, 
                                actor_dir: Path, 
                                gender: Gender,
                                force_reingest: bool) -> Dict[str, Any]:
        """Process a single actor directory"""
        
        results = {
            "ingested": 0,
            "failed": 0,
            "errors": [],
            "profiles": []
        }
        
        try:
            actor_id = f"{gender.value}_{actor_dir.name}"
            
            # Check if already processed
            existing_profiles = [p for p in self.voice_profiles.values() if p.actor_id == actor_id]
            
            if existing_profiles and not force_reingest:
                logger.info(f"Actor {actor_id} already processed, skipping")
                return results
            
            # Find audio files
            audio_files = self._find_audio_files(actor_dir)
            
            if not audio_files:
                logger.warning(f"No audio files found for actor {actor_id}")
                # Create fallback audio
                fallback_file = self._create_fallback_audio(actor_dir, gender)
                if fallback_file:
                    audio_files = [fallback_file]
                    results["errors"].append(f"Created fallback audio for {actor_id}")
                else:
                    results["failed"] += 1
                    results["errors"].append(f"No audio files for {actor_id}")
                    return results
            
            # Process audio files and extract characteristics
            processed_files = self._process_audio_files(audio_files, actor_id)
            
            if not processed_files:
                logger.error(f"Failed to process any audio files for {actor_id}")
                results["failed"] += 1
                results["errors"].append(f"Audio processing failed for {actor_id}")
                return results
            
            # Analyze voice characteristics
            voice_characteristics = self._analyze_voice_characteristics(processed_files)
            
            # Infer personality traits and age from audio
            personality_traits, age_group = self._infer_voice_personality_and_age(
                voice_characteristics, 
                gender
            )
            
            # Create voice profile
            voice_profile = VoiceProfile(
                voice_id=f"{actor_id}_profile",
                actor_id=actor_id,
                gender=gender,
                age_group=age_group,
                personality_traits=personality_traits,
                pitch_range=voice_characteristics.get("pitch_range", (80.0, 300.0)),
                speaking_rate=voice_characteristics.get("speaking_rate", 150.0),
                voice_quality=voice_characteristics.get("quality", {}),
                audio_files=processed_files,
                sample_rate=self.config.target_sample_rate,
                total_duration=voice_characteristics.get("total_duration", 0.0),
                quality_score=voice_characteristics.get("quality_score", 0.5)
            )
            
            # Store profile
            self.voice_profiles[voice_profile.voice_id] = voice_profile
            
            results["ingested"] += 1
            results["profiles"].append(voice_profile.voice_id)
            
            logger.info(f"Successfully processed actor {actor_id}")
            
        except Exception as e:
            logger.error(f"Failed to process actor directory {actor_dir}: {e}")
            results["failed"] += 1
            results["errors"].append(f"Processing failed: {e}")
        
        return results
    
    def _find_audio_files(self, directory: Path) -> List[str]:
        """Find audio files in directory with fallbacks"""
        
        audio_extensions = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'}
        audio_files = []
        
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file() and file_path.suffix.lower() in audio_extensions:
                    audio_files.append(str(file_path))
            
            # Sort for consistent processing order
            audio_files.sort()
            
        except Exception as e:
            logger.error(f"Error finding audio files in {directory}: {e}")
        
        return audio_files
    
    def _process_audio_files(self, audio_files: List[str], actor_id: str) -> List[str]:
        """Process audio files with normalization and quality control"""
        
        try:
            # Create processing directory
            output_dir = Path(self.config.processed_voices_path) / actor_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Process with audio pipeline
            pipeline_results = self.audio_processor.process_audio_pipeline(
                input_files=audio_files,
                output_dir=str(output_dir),
                normalize_volume=True,
                trim_silence=True,
                target_format=self.config.target_format
            )
            
            if pipeline_results["success"]:
                # Filter by quality and duration
                processed_files = []
                
                for file_path in pipeline_results["processed_files"]:
                    if self._validate_audio_file(file_path):
                        processed_files.append(file_path)
                
                return processed_files
            else:
                logger.error(f"Audio processing pipeline failed for {actor_id}")
                # Create fallback
                return self._create_processed_fallbacks(audio_files, output_dir)
                
        except Exception as e:
            logger.error(f"Audio processing failed for {actor_id}: {e}")
            return []
    
    def _validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file quality and duration"""
        
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False
            
            # Check file size (basic validation)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb < 0.01:  # Less than 10KB
                logger.warning(f"Audio file too small: {file_path}")
                return False
            
            # Try to get duration if possible
            try:
                duration = self._get_audio_duration(file_path)
                if duration < self.config.min_duration or duration > self.config.max_duration:
                    logger.warning(f"Audio duration out of range: {file_path} ({duration}s)")
                    return False
            except:
                # If we can't get duration, assume it's valid
                pass
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed for {file_path}: {e}")
            return False
    
    def _get_audio_duration(self, file_path: Path) -> float:
        """Get audio duration with fallbacks"""
        
        try:
            # Try librosa first
            if self.audio_processor.audio_backend == "librosa":
                import librosa
                duration = librosa.get_duration(filename=str(file_path))
                return duration
        except:
            pass
        
        try:
            # Try pydub
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(file_path))
            return len(audio) / 1000.0  # Convert ms to seconds
        except:
            pass
        
        try:
            # Try scipy for WAV files
            if file_path.suffix.lower() == '.wav':
                from scipy.io import wavfile
                sample_rate, audio = wavfile.read(str(file_path))
                return len(audio) / sample_rate
        except:
            pass
        
        # Fallback: estimate from file size (very rough)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        estimated_duration = file_size_mb / 0.17  # Rough estimate for compressed audio
        return estimated_duration
    
    def _analyze_voice_characteristics(self, audio_files: List[str]) -> Dict[str, Any]:
        """Analyze voice characteristics from audio files"""
        
        characteristics = {
            "pitch_range": (80.0, 300.0),
            "speaking_rate": 150.0,
            "quality": {},
            "total_duration": 0.0,
            "quality_score": 0.5
        }
        
        try:
            total_duration = 0.0
            pitch_values = []
            
            for audio_file in audio_files[:5]:  # Analyze first 5 files max
                try:
                    duration = self._get_audio_duration(Path(audio_file))
                    total_duration += duration
                    
                    # Try to extract pitch information
                    if self.audio_processor.audio_backend == "librosa":
                        pitch_info = self._extract_pitch_librosa(audio_file)
                        if pitch_info:
                            pitch_values.extend(pitch_info)
                
                except Exception as e:
                    logger.warning(f"Failed to analyze {audio_file}: {e}")
            
            characteristics["total_duration"] = total_duration
            
            # Calculate pitch range
            if pitch_values:
                pitch_values = [p for p in pitch_values if 50 < p < 500]  # Filter reasonable range
                if pitch_values:
                    characteristics["pitch_range"] = (
                        float(np.percentile(pitch_values, 10)),
                        float(np.percentile(pitch_values, 90))
                    )
            
            # Estimate quality score based on file count and duration
            if len(audio_files) >= 3 and total_duration >= 10.0:
                characteristics["quality_score"] = min(0.8, 0.5 + (len(audio_files) * 0.1))
            else:
                characteristics["quality_score"] = 0.4
            
        except Exception as e:
            logger.error(f"Voice characteristic analysis failed: {e}")
        
        return characteristics
    
    def _extract_pitch_librosa(self, audio_file: str) -> Optional[List[float]]:
        """Extract pitch using librosa"""
        
        try:
            import librosa
            
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.config.target_sample_rate)
            
            # Extract pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            
            # Get pitch values where magnitude is high enough
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(float(pitch))
            
            return pitch_values[:100]  # Return first 100 values
            
        except Exception as e:
            logger.warning(f"Pitch extraction failed for {audio_file}: {e}")
            return None
    
    def _infer_voice_personality_and_age(self, 
                                        voice_characteristics: Dict[str, Any],
                                        gender: Gender) -> Tuple[Dict[PersonalityTrait, float], AgeGroup]:
        """Infer personality traits and age group from voice characteristics"""
        
        personality_traits = {}
        age_group = AgeGroup.ADULT  # Default
        
        try:
            pitch_range = voice_characteristics.get("pitch_range", (80.0, 300.0))
            avg_pitch = (pitch_range[0] + pitch_range[1]) / 2
            speaking_rate = voice_characteristics.get("speaking_rate", 150.0)
            quality_score = voice_characteristics.get("quality_score", 0.5)
            
            # Age inference based on pitch and gender
            if gender == Gender.MALE:
                if avg_pitch > 200:
                    age_group = AgeGroup.TEENAGER
                    personality_traits[PersonalityTrait.CHEERFUL] = 0.7
                elif avg_pitch < 100:
                    age_group = AgeGroup.ELDERLY
                    personality_traits[PersonalityTrait.WISE] = 0.6
                else:
                    age_group = AgeGroup.ADULT
            
            elif gender == Gender.FEMALE:
                if avg_pitch > 280:
                    age_group = AgeGroup.TEENAGER
                    personality_traits[PersonalityTrait.CHEERFUL] = 0.7
                elif avg_pitch < 180:
                    age_group = AgeGroup.ELDERLY
                    personality_traits[PersonalityTrait.WISE] = 0.6
                else:
                    age_group = AgeGroup.ADULT
            
            # Personality inference from voice characteristics
            if speaking_rate > 180:
                personality_traits[PersonalityTrait.FAST_SPEAKING] = 0.8
                personality_traits[PersonalityTrait.OUTGOING] = 0.6
            elif speaking_rate < 120:
                personality_traits[PersonalityTrait.SLOW_SPEAKING] = 0.8
                personality_traits[PersonalityTrait.CALM] = 0.6
            
            # Quality-based traits
            if quality_score > 0.7:
                personality_traits[PersonalityTrait.ARTICULATE] = 0.8
                personality_traits[PersonalityTrait.CONFIDENT] = 0.6
            elif quality_score < 0.4:
                personality_traits[PersonalityTrait.SHY] = 0.5
            
            # Pitch-based personality
            pitch_variability = pitch_range[1] - pitch_range[0]
            if pitch_variability > 100:
                personality_traits[PersonalityTrait.OUTGOING] = 0.7
                personality_traits[PersonalityTrait.CHEERFUL] = 0.6
            else:
                personality_traits[PersonalityTrait.CALM] = 0.6
            
        except Exception as e:
            logger.error(f"Personality/age inference failed: {e}")
        
        return personality_traits, age_group
    
    def match_character_to_voice(self, character: Character) -> Optional[VoiceProfile]:
        """
        Match character to best voice profile with gender/age priority
        
        Args:
            character: Character object to match
            
        Returns:
            Best matching VoiceProfile or None
        """
        
        if not self.voice_profiles:
            logger.warning("No voice profiles available for matching")
            return None
        
        try:
            # Convert character attributes to our standard enums
            char_gender = self._normalize_gender(character.gender)
            char_age = self._normalize_age_group(character.age_group)
            char_personality = self._normalize_personality_traits(character.personality)
            
            best_match = None
            best_score = -1.0
            
            for voice_profile in self.voice_profiles.values():
                score = self._calculate_character_voice_match_score(
                    char_gender, char_age, char_personality,
                    voice_profile
                )
                
                if score > best_score:
                    best_score = score
                    best_match = voice_profile
            
            if best_match and best_score > 0.3:  # Minimum match threshold
                logger.info(f"Matched character {character.name} to voice {best_match.voice_id} (score: {best_score:.2f})")
                return best_match
            else:
                logger.warning(f"No suitable voice match found for character {character.name}")
                # Return fallback voice of same gender if available
                return self._get_fallback_voice(char_gender, char_age)
            
        except Exception as e:
            logger.error(f"Character-voice matching failed: {e}")
            return None
    
    def _calculate_character_voice_match_score(self,
                                              char_gender: Gender,
                                              char_age: AgeGroup,
                                              char_personality: Dict[PersonalityTrait, float],
                                              voice_profile: VoiceProfile) -> float:
        """Calculate match score between character and voice profile"""
        
        score = 0.0
        
        # Gender matching (highest priority)
        if char_gender == voice_profile.gender:
            score += 0.5  # 50% weight for gender match
        elif char_gender == Gender.UNKNOWN or voice_profile.gender == Gender.UNKNOWN:
            score += 0.2  # Partial credit for unknown
        else:
            return 0.0  # No match if gender mismatch (unless unknown)
        
        # Age group matching (second priority)
        age_score = self._calculate_age_match_score(char_age, voice_profile.age_group)
        score += age_score * 0.3  # 30% weight for age
        
        # Personality matching (lower priority but important for quality)
        personality_score = self._calculate_personality_match_score(
            char_personality, 
            voice_profile.personality_traits
        )
        score += personality_score * 0.2  # 20% weight for personality
        
        return score
    
    def _calculate_age_match_score(self, char_age: AgeGroup, voice_age: AgeGroup) -> float:
        """Calculate age group compatibility score"""
        
        if char_age == voice_age:
            return 1.0
        
        # Age compatibility matrix
        age_compatibility = {
            (AgeGroup.CHILD, AgeGroup.TEENAGER): 0.6,
            (AgeGroup.TEENAGER, AgeGroup.YOUNG_ADULT): 0.8,
            (AgeGroup.YOUNG_ADULT, AgeGroup.ADULT): 0.9,
            (AgeGroup.ADULT, AgeGroup.MIDDLE_AGED): 0.8,
            (AgeGroup.MIDDLE_AGED, AgeGroup.ELDERLY): 0.7,
        }
        
        # Check both directions
        pair = (char_age, voice_age)
        reverse_pair = (voice_age, char_age)
        
        if pair in age_compatibility:
            return age_compatibility[pair]
        elif reverse_pair in age_compatibility:
            return age_compatibility[reverse_pair]
        elif char_age == AgeGroup.UNKNOWN or voice_age == AgeGroup.UNKNOWN:
            return 0.5
        else:
            return 0.2  # Distant age groups
    
    def _calculate_personality_match_score(self,
                                          char_traits: Dict[PersonalityTrait, float],
                                          voice_traits: Dict[PersonalityTrait, float]) -> float:
        """Calculate personality trait compatibility score"""
        
        if not char_traits or not voice_traits:
            return 0.5  # Neutral score if no traits available
        
        # Find common traits
        common_traits = set(char_traits.keys()) & set(voice_traits.keys())
        
        if not common_traits:
            return 0.3  # Low score if no common traits
        
        # Calculate weighted similarity
        total_score = 0.0
        total_weight = 0.0
        
        for trait in common_traits:
            char_strength = char_traits[trait]
            voice_strength = voice_traits[trait]
            
            # Weight by average strength
            weight = (char_strength + voice_strength) / 2
            
            # Similarity score (1 - absolute difference)
            similarity = 1.0 - abs(char_strength - voice_strength)
            
            total_score += similarity * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.3
    
    def _normalize_gender(self, gender_str: str) -> Gender:
        """Normalize gender string to Gender enum"""
        
        gender_str = gender_str.lower().strip()
        
        gender_mapping = {
            "male": Gender.MALE,
            "m": Gender.MALE,
            "man": Gender.MALE,
            "boy": Gender.MALE,
            "female": Gender.FEMALE,
            "f": Gender.FEMALE,
            "woman": Gender.FEMALE,
            "girl": Gender.FEMALE,
            "neutral": Gender.NEUTRAL,
            "unknown": Gender.UNKNOWN,
            "": Gender.UNKNOWN,
        }
        
        return gender_mapping.get(gender_str, Gender.UNKNOWN)
    
    def _normalize_age_group(self, age_str: str) -> AgeGroup:
        """Normalize age string to AgeGroup enum"""
        
        age_str = age_str.lower().strip()
        
        age_mapping = {
            "child": AgeGroup.CHILD,
            "kid": AgeGroup.CHILD,
            "children": AgeGroup.CHILD,
            "teenager": AgeGroup.TEENAGER,
            "teen": AgeGroup.TEENAGER,
            "adolescent": AgeGroup.TEENAGER,
            "young_adult": AgeGroup.YOUNG_ADULT,
            "young adult": AgeGroup.YOUNG_ADULT,
            "young": AgeGroup.YOUNG_ADULT,
            "adult": AgeGroup.ADULT,
            "middle_aged": AgeGroup.MIDDLE_AGED,
            "middle aged": AgeGroup.MIDDLE_AGED,
            "middle-aged": AgeGroup.MIDDLE_AGED,
            "elderly": AgeGroup.ELDERLY,
            "old": AgeGroup.ELDERLY,
            "senior": AgeGroup.ELDERLY,
            "unknown": AgeGroup.UNKNOWN,
            "": AgeGroup.UNKNOWN,
        }
        
        return age_mapping.get(age_str, AgeGroup.UNKNOWN)
    
    def _normalize_personality_traits(self, personality_dict: Dict[str, Any]) -> Dict[PersonalityTrait, float]:
        """Normalize personality traits to standard format"""
        
        normalized_traits = {}
        
        if not personality_dict:
            return normalized_traits
        
        # Handle different personality dict formats
        traits_dict = {}
        
        if isinstance(personality_dict, dict):
            if "traits" in personality_dict:
                # Format: {"traits": ["confident", "friendly"], "confidence_scores": {...}}
                trait_list = personality_dict.get("traits", [])
                confidence_scores = personality_dict.get("confidence_scores", {})
                
                for trait in trait_list:
                    score = confidence_scores.get(trait, 0.5)
                    traits_dict[trait] = float(score)
                    
            else:
                # Direct trait -> score mapping
                traits_dict = personality_dict
        
        # Normalize trait names and scores
        for trait_str, score in traits_dict.items():
            try:
                trait_str = trait_str.lower().strip()
                
                # Map common variations to standard traits
                trait_mapping = {
                    "confident": PersonalityTrait.CONFIDENT,
                    "confidence": PersonalityTrait.CONFIDENT,
                    "shy": PersonalityTrait.SHY,
                    "timid": PersonalityTrait.SHY,
                    "aggressive": PersonalityTrait.AGGRESSIVE,
                    "assertive": PersonalityTrait.AGGRESSIVE,
                    "gentle": PersonalityTrait.GENTLE,
                    "kind": PersonalityTrait.GENTLE,
                    "cheerful": PersonalityTrait.CHEERFUL,
                    "happy": PersonalityTrait.CHEERFUL,
                    "positive": PersonalityTrait.CHEERFUL,
                    "melancholic": PersonalityTrait.MELANCHOLIC,
                    "sad": PersonalityTrait.MELANCHOLIC,
                    "depressed": PersonalityTrait.MELANCHOLIC,
                    "calm": PersonalityTrait.CALM,
                    "peaceful": PersonalityTrait.CALM,
                    "anxious": PersonalityTrait.ANXIOUS,
                    "nervous": PersonalityTrait.ANXIOUS,
                    "worried": PersonalityTrait.ANXIOUS,
                    "outgoing": PersonalityTrait.OUTGOING,
                    "extroverted": PersonalityTrait.OUTGOING,
                    "social": PersonalityTrait.OUTGOING,
                    "introverted": PersonalityTrait.INTROVERTED,
                    "quiet": PersonalityTrait.INTROVERTED,
                    "friendly": PersonalityTrait.FRIENDLY,
                    "warm": PersonalityTrait.WARM,
                    "stern": PersonalityTrait.STERN,
                    "strict": PersonalityTrait.STERN,
                    "cold": PersonalityTrait.COLD,
                    "distant": PersonalityTrait.COLD,
                    "articulate": PersonalityTrait.ARTICULATE,
                    "eloquent": PersonalityTrait.ARTICULATE,
                    "wise": PersonalityTrait.WISE,
                    "intelligent": PersonalityTrait.WISE,
                    "mysterious": PersonalityTrait.MYSTERIOUS,
                    "enigmatic": PersonalityTrait.MYSTERIOUS,
                    "heroic": PersonalityTrait.HEROIC,
                    "brave": PersonalityTrait.HEROIC,
                    "villainous": PersonalityTrait.VILLAINOUS,
                    "evil": PersonalityTrait.VILLAINOUS,
                    "sarcastic": PersonalityTrait.SARCASTIC,
                    "witty": PersonalityTrait.SARCASTIC,
                    "serious": PersonalityTrait.SERIOUS,
                    "humorous": PersonalityTrait.HUMOROUS,
                    "funny": PersonalityTrait.HUMOROUS,
                    "naive": PersonalityTrait.NAIVE,
                    "innocent": PersonalityTrait.NAIVE,
                }
                
                if trait_str in trait_mapping:
                    trait_enum = trait_mapping[trait_str]
                    score_float = float(score) if isinstance(score, (int, float, str)) else 0.5
                    score_float = max(0.0, min(1.0, score_float))  # Clamp to [0, 1]
                    normalized_traits[trait_enum] = score_float
                    
            except (ValueError, TypeError):
                logger.warning(f"Could not normalize trait: {trait_str} = {score}")
        
        return normalized_traits
    
    def _get_fallback_voice(self, gender: Gender, age_group: AgeGroup) -> Optional[VoiceProfile]:
        """Get fallback voice profile matching gender and age"""
        
        # First try exact match
        for profile in self.voice_profiles.values():
            if profile.gender == gender and profile.age_group == age_group:
                return profile
        
        # Try gender match only
        for profile in self.voice_profiles.values():
            if profile.gender == gender:
                return profile
        
        # Last resort: any profile
        if self.voice_profiles:
            return list(self.voice_profiles.values())[0]
        
        return None
    
    def _create_fallback_voices(self):
        """Create fallback voice profiles if none exist"""
        
        try:
            # Check if we have at least one voice for each gender
            male_voices = [p for p in self.voice_profiles.values() if p.gender == Gender.MALE]
            female_voices = [p for p in self.voice_profiles.values() if p.gender == Gender.FEMALE]
            
            if not male_voices:
                self._create_synthetic_voice_profile(Gender.MALE, AgeGroup.ADULT)
            
            if not female_voices:
                self._create_synthetic_voice_profile(Gender.FEMALE, AgeGroup.ADULT)
            
        except Exception as e:
            logger.error(f"Failed to create fallback voices: {e}")
    
    def _create_synthetic_voice_profile(self, gender: Gender, age_group: AgeGroup):
        """Create a synthetic voice profile with fallback audio"""
        
        try:
            actor_id = f"synthetic_{gender.value}_{age_group.value}"
            voice_id = f"{actor_id}_profile"
            
            # Create synthetic audio directory
            audio_dir = Path(self.config.processed_voices_path) / actor_id
            audio_dir.mkdir(parents=True, exist_ok=True)
            
            # Create fallback audio file
            fallback_audio = audio_dir / "fallback_voice.wav"
            self.audio_processor._create_silent_fallback(
                fallback_audio, 
                duration=self.config.fallback_voice_duration
            )
            
            # Create basic personality traits
            personality_traits = {
                PersonalityTrait.NEUTRAL: 0.7,
                PersonalityTrait.CALM: 0.6,
                PersonalityTrait.ARTICULATE: 0.5
            }
            
            # Adjust traits based on gender and age
            if gender == Gender.MALE:
                if age_group == AgeGroup.ADULT:
                    personality_traits[PersonalityTrait.CONFIDENT] = 0.6
                elif age_group == AgeGroup.ELDERLY:
                    personality_traits[PersonalityTrait.WISE] = 0.7
            elif gender == Gender.FEMALE:
                if age_group == AgeGroup.ADULT:
                    personality_traits[PersonalityTrait.FRIENDLY] = 0.6
                elif age_group == AgeGroup.ELDERLY:
                    personality_traits[PersonalityTrait.WISE] = 0.7
            
            # Set appropriate pitch range
            if gender == Gender.MALE:
                pitch_range = (80.0, 200.0)
            else:
                pitch_range = (150.0, 300.0)
            
            # Create voice profile
            voice_profile = VoiceProfile(
                voice_id=voice_id,
                actor_id=actor_id,
                gender=gender,
                age_group=age_group,
                personality_traits=personality_traits,
                pitch_range=pitch_range,
                speaking_rate=150.0,
                audio_files=[str(fallback_audio)],
                sample_rate=self.config.target_sample_rate,
                total_duration=self.config.fallback_voice_duration,
                quality_score=0.3,  # Low quality for synthetic
                recording_environment="synthetic"
            )
            
            self.voice_profiles[voice_id] = voice_profile
            
            logger.info(f"Created synthetic voice profile: {voice_id}")
            
        except Exception as e:
            logger.error(f"Failed to create synthetic voice profile: {e}")
    
    def _create_sample_voice_structure(self, base_path: Path, gender: Gender):
        """Create sample voice directory structure"""
        
        try:
            # Create a few sample actor directories
            for i in range(1, 4):  # Create 3 sample actors
                actor_dir = base_path / f"actor_{i}"
                actor_dir.mkdir(exist_ok=True)
                
                # Create a placeholder audio file
                audio_file = actor_dir / "sample.wav"
                self.audio_processor._create_silent_fallback(
                    audio_file, 
                    duration=self.config.fallback_voice_duration
                )
            
            logger.info(f"Created sample voice structure for {gender.value}")
            
        except Exception as e:
            logger.error(f"Failed to create sample structure: {e}")
    
    def _create_fallback_audio(self, actor_dir: Path, gender: Gender) -> Optional[str]:
        """Create fallback audio file for actor"""
        
        try:
            fallback_file = actor_dir / "fallback_voice.wav"
            self.audio_processor._create_silent_fallback(
                fallback_file,
                duration=self.config.fallback_voice_duration
            )
            
            if fallback_file.exists():
                return str(fallback_file)
        except Exception as e:
            logger.error(f"Failed to create fallback audio: {e}")
        
        return None
    
    def _create_processed_fallbacks(self, audio_files: List[str], output_dir: Path) -> List[str]:
        """Create processed fallback files"""
        
        fallback_files = []
        
        try:
            for i, original_file in enumerate(audio_files[:3]):  # Max 3 fallbacks
                fallback_file = output_dir / f"fallback_{i}.wav"
                
                try:
                    # Try to copy original
                    self.audio_processor._safe_copy_audio(Path(original_file), fallback_file)
                except:
                    # Create silent fallback
                    self.audio_processor._create_silent_fallback(
                        fallback_file,
                        duration=self.config.fallback_voice_duration
                    )
                
                if fallback_file.exists():
                    fallback_files.append(str(fallback_file))
        
        except Exception as e:
            logger.error(f"Failed to create processed fallbacks: {e}")
        
        return fallback_files
    
    def _generate_dataset_stats(self) -> Dict[str, Any]:
        """Generate comprehensive dataset statistics"""
        
        stats = {
            "total_voices": len(self.voice_profiles),
            "gender_distribution": {},
            "age_distribution": {},
            "personality_distribution": {},
            "quality_distribution": {},
            "total_duration": 0.0,
            "average_quality": 0.0
        }
        
        try:
            if not self.voice_profiles:
                return stats
            
            # Gender distribution
            for gender in Gender:
                count = len([p for p in self.voice_profiles.values() if p.gender == gender])
                if count > 0:
                    stats["gender_distribution"][gender.value] = count
            
            # Age distribution
            for age_group in AgeGroup:
                count = len([p for p in self.voice_profiles.values() if p.age_group == age_group])
                if count > 0:
                    stats["age_distribution"][age_group.value] = count
            
            # Personality trait distribution
            trait_counts = {}
            for profile in self.voice_profiles.values():
                for trait in profile.personality_traits.keys():
                    trait_counts[trait.value] = trait_counts.get(trait.value, 0) + 1
            
            stats["personality_distribution"] = dict(sorted(
                trait_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10])  # Top 10 traits
            
            # Quality distribution
            quality_ranges = {"high": 0, "medium": 0, "low": 0}
            total_duration = 0.0
            total_quality = 0.0
            
            for profile in self.voice_profiles.values():
                quality = profile.quality_score
                total_quality += quality
                total_duration += profile.total_duration
                
                if quality >= 0.7:
                    quality_ranges["high"] += 1
                elif quality >= 0.5:
                    quality_ranges["medium"] += 1
                else:
                    quality_ranges["low"] += 1
            
            stats["quality_distribution"] = quality_ranges
            stats["total_duration"] = total_duration
            stats["average_quality"] = total_quality / len(self.voice_profiles)
            
        except Exception as e:
            logger.error(f"Failed to generate stats: {e}")
        
        return stats
    
    def get_voice_profiles_by_criteria(self, 
                                     gender: Optional[Gender] = None,
                                     age_group: Optional[AgeGroup] = None,
                                     personality_traits: Optional[List[PersonalityTrait]] = None,
                                     min_quality: float = 0.0) -> List[VoiceProfile]:
        """Get voice profiles matching specific criteria"""
        
        matching_profiles = []
        
        try:
            for profile in self.voice_profiles.values():
                # Check gender
                if gender and profile.gender != gender:
                    continue
                
                # Check age group
                if age_group and profile.age_group != age_group:
                    continue
                
                # Check quality
                if profile.quality_score < min_quality:
                    continue
                
                # Check personality traits
                if personality_traits:
                    profile_traits = set(profile.personality_traits.keys())
                    required_traits = set(personality_traits)
                    
                    # Require at least 50% trait overlap
                    overlap = len(profile_traits & required_traits)
                    if overlap < len(required_traits) * 0.5:
                        continue
                
                matching_profiles.append(profile)
            
            # Sort by quality score (highest first)
            matching_profiles.sort(key=lambda p: p.quality_score, reverse=True)
            
        except Exception as e:
            logger.error(f"Failed to filter voice profiles: {e}")
        
        return matching_profiles


def main():
    """Test the voice dataset handler"""
    
    try:
        # Create handler
        handler = VoiceDatasetHandler()
        
        print(f"Initialized voice dataset handler")
        print(f"Audio backend: {handler.audio_processor.audio_backend}")
        print(f"Existing voice profiles: {len(handler.voice_profiles)}")
        
        # Generate stats
        stats = handler._generate_dataset_stats()
        print(f"\nDataset statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # Test ingestion (dry run)
        print(f"\nTesting voice dataset structure...")
        results = handler.ingest_voice_dataset(force_reingest=False)
        
        print(f"Ingestion results:")
        print(f"  Success: {results['success']}")
        print(f"  Voices ingested: {results['ingested_voices']}")
        print(f"  Failed voices: {results['failed_voices']}")
        
        if results['errors']:
            print(f"  Errors: {results['errors'][:3]}")  # Show first 3 errors
        
        if results['warnings']:
            print(f"  Warnings: {results['warnings'][:3]}")  # Show first 3 warnings
    
    except Exception as e:
        print(f"Test failed: {e}")


if __name__ == "__main__":
    main()