"""
Model Manager for EchoTales

This module handles downloading, caching, and managing AI models used throughout the system.
It organizes models by type and provides a unified interface for model access.
"""

import os
import logging
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable
from urllib.parse import urlparse
import requests
from tqdm import tqdm
import shutil
import zipfile
import tarfile

# HuggingFace Hub for model downloads
try:
    from huggingface_hub import hf_hub_download, snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelInfo:
    """Information about a model"""
    def __init__(
        self,
        name: str,
        type: str,
        source: str,
        url_or_id: str,
        description: str = "",
        size_mb: int = 0,
        dependencies: Optional[List[str]] = None,
        license: str = "unknown",
        local_path: Optional[str] = None,
        checksum: Optional[str] = None
    ):
        self.name = name
        self.type = type  # "nlp", "image", "voice", "character_analysis", etc.
        self.source = source  # "huggingface", "url", "local"
        self.url_or_id = url_or_id
        self.description = description
        self.size_mb = size_mb
        self.dependencies = dependencies or []
        self.license = license
        self.local_path = local_path
        self.checksum = checksum


class ModelManager:
    """Manages AI model downloads and caching"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        
        # Create subdirectories for different model types
        self.subdirs = {
            "nlp": self.models_dir / "nlp",
            "image": self.models_dir / "image_generation", 
            "voice": self.models_dir / "voice_cloning",
            "character": self.models_dir / "character_analysis",
            "scene": self.models_dir / "scene_analysis",
            "emotion": self.models_dir / "emotion_detection",
            "cache": self.models_dir / "cache"
        }
        
        for subdir in self.subdirs.values():
            subdir.mkdir(exist_ok=True, parents=True)
        
        self.model_registry_file = self.models_dir / "model_registry.json"
        self.registered_models = self._load_registry()
        
        logger.info(f"ModelManager initialized with models directory: {self.models_dir}")
    
    def _load_registry(self) -> Dict[str, Dict]:
        """Load the model registry from disk"""
        if self.model_registry_file.exists():
            try:
                with open(self.model_registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load model registry: {e}")
        return {}
    
    def _save_registry(self):
        """Save the model registry to disk"""
        try:
            with open(self.model_registry_file, 'w', encoding='utf-8') as f:
                json.dump(self.registered_models, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_info: ModelInfo) -> bool:
        """Register a model in the registry"""
        try:
            self.registered_models[model_info.name] = {
                "name": model_info.name,
                "type": model_info.type,
                "source": model_info.source,
                "url_or_id": model_info.url_or_id,
                "description": model_info.description,
                "size_mb": model_info.size_mb,
                "dependencies": model_info.dependencies,
                "license": model_info.license,
                "local_path": model_info.local_path,
                "checksum": model_info.checksum,
                "downloaded": False
            }
            self._save_registry()
            logger.info(f"Registered model: {model_info.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register model {model_info.name}: {e}")
            return False
    
    def download_model(self, model_name: str, force_redownload: bool = False) -> Optional[str]:
        """Download a model if not already present"""
        if model_name not in self.registered_models:
            logger.error(f"Model {model_name} not registered")
            return None
        
        model_info = self.registered_models[model_name]
        local_path = model_info.get("local_path")
        
        # Check if already downloaded
        if not force_redownload and local_path and Path(local_path).exists():
            logger.info(f"Model {model_name} already downloaded: {local_path}")
            return local_path
        
        try:
            source = model_info["source"]
            url_or_id = model_info["url_or_id"]
            model_type = model_info["type"]
            
            # Determine download path
            target_dir = self.subdirs.get(model_type, self.models_dir) / model_name
            target_dir.mkdir(exist_ok=True, parents=True)
            
            if source == "huggingface":
                local_path = self._download_huggingface_model(url_or_id, target_dir)
            elif source == "url":
                local_path = self._download_url_model(url_or_id, target_dir, model_name)
            else:
                logger.error(f"Unsupported model source: {source}")
                return None
            
            if local_path:
                # Update registry with local path
                self.registered_models[model_name]["local_path"] = str(local_path)
                self.registered_models[model_name]["downloaded"] = True
                self._save_registry()
                
                logger.info(f"Successfully downloaded model {model_name} to {local_path}")
                return str(local_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {model_name}: {e}")
        
        return None
    
    def _download_huggingface_model(self, model_id: str, target_dir: Path) -> Optional[str]:
        """Download model from HuggingFace Hub"""
        if not HF_HUB_AVAILABLE:
            logger.error("HuggingFace Hub not available. Install with: pip install huggingface-hub")
            return None
        
        try:
            # Download entire model repository
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=target_dir,
                local_dir_use_symlinks=False
            )
            return str(local_path)
            
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return None
    
    def _download_url_model(self, url: str, target_dir: Path, model_name: str) -> Optional[str]:
        """Download model from URL"""
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Determine file extension
            parsed_url = urlparse(url)
            filename = parsed_url.path.split('/')[-1] or f"{model_name}.bin"
            filepath = target_dir / filename
            
            # Download with progress bar
            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f, tqdm(
                desc=f"Downloading {model_name}",
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            # Extract if it's an archive
            if filename.endswith(('.zip', '.tar.gz', '.tar')):
                self._extract_archive(filepath, target_dir)
                filepath.unlink()  # Remove archive after extraction
                return str(target_dir)
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"URL download failed: {e}")
            return None
    
    def _extract_archive(self, archive_path: Path, extract_dir: Path):
        """Extract archive files"""
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_path.suffix in ['.tar', '.tar.gz']:
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_dir)
    
    def get_model_path(self, model_name: str) -> Optional[str]:
        """Get the local path of a model"""
        if model_name in self.registered_models:
            local_path = self.registered_models[model_name].get("local_path")
            if local_path and Path(local_path).exists():
                return local_path
        return None
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is downloaded"""
        return self.get_model_path(model_name) is not None
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict]:
        """List all registered models"""
        models = []
        for name, info in self.registered_models.items():
            if model_type is None or info["type"] == model_type:
                info_copy = info.copy()
                info_copy["downloaded"] = self.is_model_downloaded(name)
                models.append(info_copy)
        return models
    
    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model"""
        local_path = self.get_model_path(model_name)
        if local_path:
            try:
                path_obj = Path(local_path)
                if path_obj.is_file():
                    path_obj.unlink()
                elif path_obj.is_dir():
                    shutil.rmtree(path_obj)
                
                # Update registry
                self.registered_models[model_name]["local_path"] = None
                self.registered_models[model_name]["downloaded"] = False
                self._save_registry()
                
                logger.info(f"Deleted model {model_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete model {model_name}: {e}")
        
        return False
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics for models directory"""
        usage = {}
        for model_type, subdir in self.subdirs.items():
            if subdir.exists():
                total_size = sum(f.stat().st_size for f in subdir.rglob('*') if f.is_file())
                usage[model_type] = total_size / (1024 * 1024)  # Convert to MB
        return usage


# Predefined model configurations for EchoTales
ECHOTALES_MODELS = {
    # Character Analysis Models
    "personality_analyzer": ModelInfo(
        name="personality_analyzer",
        type="character",
        source="huggingface",
        url_or_id="j-hartmann/emotion-english-distilroberta-base",
        description="Character personality and emotion analysis",
        size_mb=250,
        license="apache-2.0"
    ),
    
    "physical_description_extractor": ModelInfo(
        name="physical_description_extractor",
        type="character", 
        source="huggingface",
        url_or_id="microsoft/DialoGPT-medium",
        description="Extracts physical descriptions from text",
        size_mb=350,
        license="mit"
    ),
    
    # Scene Analysis Models
    "scene_classifier": ModelInfo(
        name="scene_classifier",
        type="scene",
        source="huggingface",
        url_or_id="facebook/bart-large-mnli",
        description="Classifies and analyzes scenes in novels",
        size_mb=1600,
        license="mit"
    ),
    
    "setting_detector": ModelInfo(
        name="setting_detector",
        type="scene",
        source="huggingface",
        url_or_id="sentence-transformers/all-MiniLM-L6-v2",
        description="Detects setting changes and transitions",
        size_mb=90,
        license="apache-2.0"
    ),
    
    # NLP Models
    "text_summarizer": ModelInfo(
        name="text_summarizer",
        type="nlp",
        source="huggingface",
        url_or_id="facebook/bart-large-cnn",
        description="Summarizes text for scene analysis",
        size_mb=1600,
        license="mit"
    ),
    
    "sentiment_analyzer": ModelInfo(
        name="sentiment_analyzer",
        type="emotion",
        source="huggingface",
        url_or_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        description="Analyzes sentiment and emotion in text",
        size_mb=500,
        license="mit"
    ),
    
    # Voice Classification (placeholder - would be custom trained)
    "voice_classifier": ModelInfo(
        name="voice_classifier",
        type="voice",
        source="local",
        url_or_id="custom_voice_model",
        description="Classifies and matches character voices",
        size_mb=100,
        license="custom"
    ),
}


def setup_echotales_models(model_manager: ModelManager) -> List[str]:
    """Register all EchoTales models in the model manager"""
    registered = []
    
    for model_name, model_info in ECHOTALES_MODELS.items():
        if model_manager.register_model(model_info):
            registered.append(model_name)
    
    logger.info(f"Registered {len(registered)} EchoTales models")
    return registered


def create_model_manager() -> ModelManager:
    """Create and configure the model manager for EchoTales"""
    manager = ModelManager()
    setup_echotales_models(manager)
    return manager


if __name__ == "__main__":
    import asyncio
    
    async def test_model_manager():
        """Test the model manager functionality"""
        manager = create_model_manager()
        
        # List all models
        print("\n=== Available Models ===")
        for model in manager.list_models():
            status = "✓ Downloaded" if model["downloaded"] else "○ Not Downloaded"
            print(f"{status} {model['name']} ({model['type']}) - {model['description']}")
        
        # Show disk usage
        print(f"\n=== Disk Usage ===")
        usage = manager.get_disk_usage()
        total_size = sum(usage.values())
        for model_type, size_mb in usage.items():
            print(f"{model_type}: {size_mb:.1f} MB")
        print(f"Total: {total_size:.1f} MB")
        
        # Download a small model for testing
        print(f"\n=== Test Download ===")
        test_model = "setting_detector"  # Small model for testing
        print(f"Downloading {test_model}...")
        
        path = manager.download_model(test_model)
        if path:
            print(f"✓ Successfully downloaded to: {path}")
        else:
            print("✗ Download failed")
    
    asyncio.run(test_model_manager())