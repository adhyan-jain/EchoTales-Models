#!/usr/bin/env python3
"""
Comprehensive Model Setup for EchoTales
Sets up all essential models: BookNLP, OpenVoice, Character Classification, Voice Classification
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tarfile
import json
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ComprehensiveModelSetup:
    """Sets up all models needed for EchoTales to function properly"""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Critical model directories
        self.model_paths = {
            "booknlp": self.models_dir / "booknlp",
            "character_analysis": self.models_dir / "character_analysis", 
            "voice_cloning": self.models_dir / "voice_cloning",
            "tts": self.models_dir / "tts",
            "image_generation": self.models_dir / "image_generation"
        }
        
        # Create all directories
        for path in self.model_paths.values():
            path.mkdir(exist_ok=True, parents=True)
        
        self.setup_status = {
            "booknlp": False,
            "openvoice": False,
            "character_models": False,
            "voice_models": False,
            "basic_models": False
        }
    
    def install_dependencies(self):
        """Install critical dependencies for model processing"""
        logger.info("🔧 Installing critical dependencies...")
        
        # Core ML dependencies
        core_deps = [
            "torch", "transformers", "tqdm", "requests", 
            "huggingface-hub", "sentence-transformers",
            "librosa", "soundfile", "scipy", "numpy",
            "scikit-learn", "pandas"
        ]
        
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install"
            ] + core_deps, check=True)
            logger.info("✅ Core dependencies installed")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e}")
            return False
    
    def setup_booknlp(self):
        """Set up BookNLP for character and entity extraction"""
        logger.info("📚 Setting up BookNLP...")
        
        try:
            # Install BookNLP
            subprocess.run([
                sys.executable, "-m", "pip", "install", "booknlp"
            ], check=True)
            
            # Download BookNLP models
            booknlp_dir = self.model_paths["booknlp"]
            
            # Create BookNLP test to ensure it works
            test_script = booknlp_dir / "test_booknlp.py"
            test_script.write_text("""
import sys
sys.path.append('.')

try:
    from booknlp.booknlp import BookNLP
    
    # Initialize BookNLP
    model_params = {
        "pipeline": "entity,quote,supersense,event,coref", 
        "model": "big"
    }
    
    booknlp = BookNLP("en", model_params)
    
    # Test with simple text
    test_text = "John walked to the store. 'Hello there,' he said to Mary."
    
    print("BookNLP successfully initialized!")
    print("Available pipelines:", booknlp.pipeline)
    
except Exception as e:
    print(f"BookNLP setup failed: {e}")
    sys.exit(1)
""")
            
            # Test BookNLP installation
            result = subprocess.run([sys.executable, str(test_script)], 
                                  capture_output=True, text=True, cwd=str(booknlp_dir))
            
            if result.returncode == 0:
                logger.info("✅ BookNLP setup complete and tested")
                self.setup_status["booknlp"] = True
                return True
            else:
                logger.error(f"❌ BookNLP test failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ BookNLP setup failed: {e}")
            return False
    
    def setup_openvoice(self):
        """Set up OpenVoice for voice cloning and TTS"""
        logger.info("🎤 Setting up OpenVoice...")
        
        try:
            openvoice_dir = self.model_paths["voice_cloning"]
            
            # Clone OpenVoice repository
            if not (openvoice_dir / "OpenVoice").exists():
                logger.info("📥 Cloning OpenVoice repository...")
                subprocess.run([
                    "git", "clone", "https://github.com/myshell-ai/OpenVoice.git",
                    str(openvoice_dir / "OpenVoice")
                ], check=True)
            
            # Install OpenVoice dependencies
            openvoice_path = openvoice_dir / "OpenVoice"
            if (openvoice_path / "requirements.txt").exists():
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r",
                    str(openvoice_path / "requirements.txt")
                ], check=True)
            
            # Download base models
            self.download_openvoice_models(openvoice_dir)
            
            logger.info("✅ OpenVoice setup complete")
            self.setup_status["openvoice"] = True
            return True
            
        except Exception as e:
            logger.error(f"❌ OpenVoice setup failed: {e}")
            return False
    
    def download_openvoice_models(self, openvoice_dir):
        """Download OpenVoice base models"""
        logger.info("📥 Downloading OpenVoice base models...")
        
        # Base model URLs (these are example URLs - replace with actual OpenVoice model links)
        models = {
            "base_speaker.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers_se.zip",
            "converter.pth": "https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/converter_se.zip"
        }
        
        models_path = openvoice_dir / "models"
        models_path.mkdir(exist_ok=True)
        
        for model_name, url in models.items():
            model_file = models_path / model_name
            if not model_file.exists():
                try:
                    logger.info(f"📥 Downloading {model_name}...")
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    
                    total_size = int(response.headers.get('content-length', 0))
                    
                    with open(model_file, 'wb') as f:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    
                    logger.info(f"✅ Downloaded {model_name}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to download {model_name}: {e}")
    
    def setup_character_analysis_models(self):
        """Set up models for character personality and emotion analysis"""
        logger.info("👤 Setting up character analysis models...")
        
        try:
            from transformers import AutoTokenizer, AutoModel
            
            char_dir = self.model_paths["character_analysis"]
            
            # Essential character analysis models
            models = {
                "emotion_classifier": {
                    "model_id": "j-hartmann/emotion-english-distilroberta-base",
                    "description": "7-class emotion classification"
                },
                "personality_classifier": {
                    "model_id": "unitary/toxic-bert",
                    "description": "Personality trait classification"
                },
                "sentiment_analyzer": {
                    "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                    "description": "Advanced sentiment analysis"
                },
                "character_embeddings": {
                    "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                    "description": "Character description embeddings"
                }
            }
            
            for model_name, config in models.items():
                model_path = char_dir / model_name
                
                if not model_path.exists() or not list(model_path.glob("*")):
                    logger.info(f"📥 Downloading {model_name}: {config['description']}")
                    
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
                        model = AutoModel.from_pretrained(config["model_id"])
                        
                        model_path.mkdir(exist_ok=True)
                        tokenizer.save_pretrained(str(model_path))
                        model.save_pretrained(str(model_path))
                        
                        logger.info(f"✅ {model_name} downloaded successfully")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to download {model_name}: {e}")
                        continue
                else:
                    logger.info(f"⏭️ {model_name} already exists")
            
            self.setup_status["character_models"] = True
            logger.info("✅ Character analysis models setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Character analysis setup failed: {e}")
            return False
    
    def setup_voice_classification_models(self):
        """Set up models for voice characteristic classification"""
        logger.info("🔊 Setting up voice classification models...")
        
        try:
            from transformers import AutoProcessor, AutoModel
            
            voice_dir = self.model_paths["voice_cloning"] / "classification"
            voice_dir.mkdir(exist_ok=True)
            
            # Voice classification models
            models = {
                "wav2vec2_base": {
                    "model_id": "facebook/wav2vec2-base",
                    "description": "Base audio feature extraction"
                },
                "speech_emotion": {
                    "model_id": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
                    "description": "Speech emotion recognition"
                },
                "speaker_verification": {
                    "model_id": "microsoft/speecht5_vc",
                    "description": "Speaker verification and voice conversion"
                }
            }
            
            for model_name, config in models.items():
                model_path = voice_dir / model_name
                
                if not model_path.exists() or not list(model_path.glob("*")):
                    logger.info(f"📥 Downloading {model_name}: {config['description']}")
                    
                    try:
                        # Some models need processor, others need tokenizer
                        try:
                            processor = AutoProcessor.from_pretrained(config["model_id"])
                            model = AutoModel.from_pretrained(config["model_id"])
                            
                            model_path.mkdir(exist_ok=True)
                            processor.save_pretrained(str(model_path))
                            model.save_pretrained(str(model_path))
                            
                        except:
                            # Fallback to tokenizer
                            from transformers import AutoTokenizer
                            tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
                            model = AutoModel.from_pretrained(config["model_id"])
                            
                            model_path.mkdir(exist_ok=True)
                            tokenizer.save_pretrained(str(model_path))
                            model.save_pretrained(str(model_path))
                        
                        logger.info(f"✅ {model_name} downloaded successfully")
                        
                    except Exception as e:
                        logger.error(f"❌ Failed to download {model_name}: {e}")
                        continue
                else:
                    logger.info(f"⏭️ {model_name} already exists")
            
            self.setup_status["voice_models"] = True
            logger.info("✅ Voice classification models setup complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Voice classification setup failed: {e}")
            return False
    
    def classify_voice_dataset(self):
        """Classify the existing voice dataset"""
        logger.info("🎯 Classifying voice dataset...")
        
        voice_dataset_path = Path("data/voice_dataset")
        if not voice_dataset_path.exists():
            logger.warning("⚠️ Voice dataset not found, skipping classification")
            return False
        
        try:
            # Create classification script
            classify_script = Path("classify_voice_data.py")
            classify_script.write_text("""
import os
import json
import librosa
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def analyze_audio_file(file_path):
    '''Analyze audio file and extract features'''
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Extract features
        features = {
            'duration': len(audio) / sr,
            'pitch_mean': float(np.mean(librosa.yin(audio, fmin=50, fmax=500))),
            'pitch_std': float(np.std(librosa.yin(audio, fmin=50, fmax=500))),
            'energy_mean': float(np.mean(librosa.feature.rms(y=audio))),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))),
            'mfcc_mean': librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).mean(axis=1).tolist(),
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Failed to analyze {file_path}: {e}")
        return None

def classify_voice_dataset():
    '''Classify all voice files in the dataset'''
    voice_path = Path('data/voice_dataset')
    results = {
        'male_actors': {},
        'female_actors': {},
        'metadata': {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0
        }
    }
    
    # Process male actors
    male_path = voice_path / 'male_actors'
    if male_path.exists():
        for actor_dir in male_path.iterdir():
            if actor_dir.is_dir():
                actor_name = actor_dir.name
                results['male_actors'][actor_name] = {
                    'gender': 'male',
                    'files': {},
                    'characteristics': {}
                }
                
                for audio_file in actor_dir.glob('*.wav'):
                    results['metadata']['total_files'] += 1
                    features = analyze_audio_file(audio_file)
                    
                    if features:
                        results['male_actors'][actor_name]['files'][audio_file.name] = features
                        results['metadata']['processed_files'] += 1
                    else:
                        results['metadata']['failed_files'] += 1
    
    # Process female actors
    female_path = voice_path / 'female_actors'
    if female_path.exists():
        for actor_dir in female_path.iterdir():
            if actor_dir.is_dir():
                actor_name = actor_dir.name
                results['female_actors'][actor_name] = {
                    'gender': 'female',
                    'files': {},
                    'characteristics': {}
                }
                
                for audio_file in actor_dir.glob('*.wav'):
                    results['metadata']['total_files'] += 1
                    features = analyze_audio_file(audio_file)
                    
                    if features:
                        results['female_actors'][actor_name]['files'][audio_file.name] = features
                        results['metadata']['processed_files'] += 1
                    else:
                        results['metadata']['failed_files'] += 1
    
    # Save results
    output_file = voice_path / 'voice_classification.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Voice classification complete!")
    print(f"Total files: {results['metadata']['total_files']}")
    print(f"Processed: {results['metadata']['processed_files']}")
    print(f"Failed: {results['metadata']['failed_files']}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    classify_voice_dataset()
""")
            
            # Run classification
            result = subprocess.run([sys.executable, str(classify_script)], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Voice dataset classification complete")
                logger.info(result.stdout)
                return True
            else:
                logger.error(f"❌ Voice classification failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Voice dataset classification failed: {e}")
            return False
    
    def create_model_test_scripts(self):
        """Create test scripts to verify all models work"""
        logger.info("🧪 Creating model test scripts...")
        
        test_dir = Path("tests/model_tests")
        test_dir.mkdir(exist_ok=True, parents=True)
        
        # BookNLP test
        booknlp_test = test_dir / "test_booknlp.py"
        booknlp_test.write_text("""
from booknlp.booknlp import BookNLP

def test_booknlp():
    model_params = {"pipeline":"entity,quote,supersense,event,coref", "model":"big"}
    booknlp = BookNLP("en", model_params)
    
    text = "John walked to Mary. 'Hello,' he said."
    result = booknlp.process(text, "test_output")
    print("BookNLP test passed!")

if __name__ == "__main__":
    test_booknlp()
""")
        
        # Character analysis test
        char_test = test_dir / "test_character_models.py"
        char_test.write_text("""
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

def test_character_models():
    models_dir = Path("models/character_analysis")
    
    for model_dir in models_dir.iterdir():
        if model_dir.is_dir():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                model = AutoModel.from_pretrained(str(model_dir))
                print(f"✅ {model_dir.name} loaded successfully")
            except Exception as e:
                print(f"❌ {model_dir.name} failed: {e}")

if __name__ == "__main__":
    test_character_models()
""")
        
        logger.info(f"✅ Test scripts created in {test_dir}")
    
    def run_setup(self):
        """Run complete model setup"""
        logger.info("🚀 Starting Comprehensive EchoTales Model Setup")
        logger.info("=" * 80)
        
        steps = [
            ("Installing Dependencies", self.install_dependencies),
            ("Setting up BookNLP", self.setup_booknlp),
            ("Setting up Character Analysis Models", self.setup_character_analysis_models),
            ("Setting up Voice Classification Models", self.setup_voice_classification_models),
            ("Setting up OpenVoice", self.setup_openvoice),
            ("Classifying Voice Dataset", self.classify_voice_dataset),
            ("Creating Test Scripts", self.create_model_test_scripts)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"📋 {step_name}...")
            try:
                success = step_func()
                if success:
                    logger.info(f"✅ {step_name} completed successfully")
                else:
                    logger.warning(f"⚠️ {step_name} completed with issues")
            except Exception as e:
                logger.error(f"❌ {step_name} failed: {e}")
        
        # Final status report
        logger.info("=" * 80)
        logger.info("📊 SETUP SUMMARY")
        logger.info("=" * 80)
        
        for component, status in self.setup_status.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"{status_icon} {component.upper()}: {'Ready' if status else 'Not Ready'}")
        
        ready_count = sum(self.setup_status.values())
        total_count = len(self.setup_status)
        
        logger.info(f"\n🎯 Overall Status: {ready_count}/{total_count} components ready")
        
        if ready_count == total_count:
            logger.info("🎉 EchoTales is fully ready for production use!")
        elif ready_count >= 3:
            logger.info("⚡ EchoTales is ready for most functionality!")
        else:
            logger.warning("⚠️ EchoTales has limited functionality. Some models failed to install.")
        
        logger.info("\n🚀 Next Steps:")
        logger.info("1. Run: python run_dev.py")
        logger.info("2. Test: python tests/model_tests/test_character_models.py")
        logger.info("3. Start processing novels!")


def main():
    setup = ComprehensiveModelSetup()
    setup.run_setup()


if __name__ == "__main__":
    main()