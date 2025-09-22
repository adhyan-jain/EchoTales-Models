#!/usr/bin/env python3
"""
Simplified Model Initialization for EchoTales

Downloads essential AI models for the EchoTales system with minimal dependencies.
"""

import os
import sys
import logging
from pathlib import Path
import requests
from tqdm import tqdm
import zipfile
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SimpleModelDownloader:
    """Simple model downloader with progress bars"""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Create model type directories
        self.subdirs = {
            "character_analysis": self.models_dir / "character_analysis",
            "text_processing": self.models_dir / "text_processing",
            "image_generation": self.models_dir / "image_generation", 
            "voice_processing": self.models_dir / "voice_processing"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True, parents=True)
    
    def download_huggingface_model(self, model_id, local_path, model_name):
        """Download model from HuggingFace using transformers"""
        try:
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            
            logger.info(f"Downloading {model_name} from HuggingFace: {model_id}")
            
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Download tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModel.from_pretrained(model_id)
            
            # Save locally
            tokenizer.save_pretrained(str(local_dir))
            model.save_pretrained(str(local_dir))
            
            logger.info(f"✓ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
            return False
    
    def download_url_model(self, url, local_path, model_name):
        """Download model from URL with progress bar"""
        try:
            logger.info(f"Downloading {model_name} from URL")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            local_dir = Path(local_path)
            local_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine filename
            filename = url.split('/')[-1] or f"{model_name}.zip"
            file_path = local_dir / filename
            
            # Download with progress bar
            with open(file_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=model_name) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract if it's a zip file
            if filename.endswith('.zip'):
                logger.info(f"Extracting {filename}")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(local_dir)
                # Remove zip file after extraction
                file_path.unlink()
            
            logger.info(f"✓ Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"✗ Failed to download {model_name}: {e}")
            return False
    
    def check_model_exists(self, local_path):
        """Check if model already exists"""
        path = Path(local_path)
        if path.is_dir():
            # Check if directory has files
            return len(list(path.rglob("*"))) > 0
        return path.exists()


def main():
    """Main function to download all models"""
    logger.info("🚀 Starting EchoTales Model Setup")
    logger.info("=" * 60)
    
    downloader = SimpleModelDownloader()
    
    # Define models to download (essential ones only)
    models = {
        "character_analysis": [
            {
                "name": "emotion_classifier",
                "source": "huggingface",
                "model_id": "j-hartmann/emotion-english-distilroberta-base",
                "local_path": "models/character_analysis/emotion_classifier",
                "description": "Emotion classification for character analysis"
            },
            {
                "name": "sentiment_analyzer",
                "source": "huggingface", 
                "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "local_path": "models/character_analysis/sentiment_analyzer",
                "description": "Sentiment analysis for character emotions"
            }
        ],
        "text_processing": [
            {
                "name": "sentence_transformer",
                "source": "huggingface",
                "model_id": "sentence-transformers/all-MiniLM-L6-v2",
                "local_path": "models/text_processing/sentence_transformer",
                "description": "Sentence embeddings for text similarity"
            }
        ],
        "image_generation": [
            {
                "name": "prompt_enhancer",
                "source": "huggingface",
                "model_id": "gpt2",
                "local_path": "models/image_generation/prompt_enhancer", 
                "description": "Text generation for prompt enhancement"
            }
        ],
        "voice_processing": [
            {
                "name": "speech_embeddings",
                "source": "huggingface",
                "model_id": "microsoft/speecht5_vc",
                "local_path": "models/voice_processing/speech_embeddings",
                "description": "Speech embeddings for voice classification"
            }
        ]
    }
    
    total_models = sum(len(category_models) for category_models in models.values())
    downloaded = 0
    skipped = 0
    failed = 0
    
    logger.info(f"📦 Found {total_models} models to download")
    logger.info("")
    
    # Process each category
    for category, category_models in models.items():
        logger.info(f"🔧 Processing {category.upper()} models...")
        
        for model in category_models:
            model_name = model["name"]
            local_path = model["local_path"]
            
            # Check if already exists
            if downloader.check_model_exists(local_path):
                logger.info(f"⏭️  {model_name} already exists, skipping")
                skipped += 1
                continue
            
            logger.info(f"📥 Downloading {model_name}...")
            logger.info(f"   Description: {model['description']}")
            
            # Download based on source
            success = False
            if model["source"] == "huggingface":
                success = downloader.download_huggingface_model(
                    model["model_id"], local_path, model_name
                )
            elif model["source"] == "url":
                success = downloader.download_url_model(
                    model["url"], local_path, model_name
                )
            
            if success:
                downloaded += 1
            else:
                failed += 1
            
            logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("📊 DOWNLOAD SUMMARY")
    logger.info("=" * 60)
    logger.info(f"✅ Successfully downloaded: {downloaded} models")
    logger.info(f"⏭️  Already existed: {skipped} models")
    logger.info(f"❌ Failed downloads: {failed} models")
    logger.info(f"📁 Models directory: {downloader.models_dir.absolute()}")
    
    if failed == 0:
        logger.info("🎉 All models are ready! EchoTales is now fully functional.")
    else:
        logger.warning("⚠️  Some models failed to download. EchoTales may have limited functionality.")
        logger.info("💡 You can re-run this script to retry failed downloads.")
    
    logger.info("")
    logger.info("🚀 Next steps:")
    logger.info("   1. Run: python run_dev.py")
    logger.info("   2. Visit: http://127.0.0.1:5000/health")
    logger.info("   3. Test the API endpoints")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n⏹️  Download interrupted by user")
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        sys.exit(1)