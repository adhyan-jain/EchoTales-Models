#!/usr/bin/env python3
"""
Model Initialization Script for EchoTales

This script downloads and initializes all required AI models for the EchoTales system
with comprehensive error handling and fallback mechanisms.

Usage:
    python scripts/processing/initialize_models.py
    python scripts/processing/initialize_models.py --download-all
    python scripts/processing/initialize_models.py --model-type character_analysis
    python scripts/processing/initialize_models.py --force-download
"""

import sys
import os
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import hashlib
import platform

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.echotales.core.environment import get_config, init_environment
    from src.echotales.ai.model_manager import ModelManager
except ImportError as e:
    print(f"Error importing EchoTales modules: {e}")
    print("Make sure you're running this script from the project root directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelInitializer:
    """Manages model downloads and initialization with comprehensive fallbacks"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.model_manager = ModelManager()
        
        # Initialize GPU/device information
        self.device_info = self._detect_compute_devices()
        logger.info(f"Compute devices detected: {self.device_info}")
        
        self.download_stats = {
            "total_models": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "skipped_models": 0,
            "total_size_mb": 0,
            "start_time": None,
            "end_time": None,
            "device_info": self.device_info
        }
        
        # Define model priorities and requirements
        self.model_definitions = {
            "character_analysis": {
                "priority": 1,
                "required": True,
                "models": [
                    {
                        "name": "personality_classifier",
                        "source": "huggingface",
                        "model_id": "cardiffnlp/twitter-roberta-base-emotion",
                        "local_path": "models/character_analysis/personality_classifier",
                        "size_mb": 450,
                        "description": "Personality trait classification model",
                        "fallback_models": [
                            "j-hartmann/emotion-english-distilroberta-base",
                            "SamLowe/roberta-base-go_emotions"
                        ]
                    },
                    {
                        "name": "sentiment_analyzer", 
                        "source": "huggingface",
                        "model_id": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                        "local_path": "models/character_analysis/sentiment_analyzer",
                        "size_mb": 450,
                        "description": "Sentiment analysis for character emotions",
                        "fallback_models": [
                            "cardiffnlp/twitter-roberta-base-sentiment",
                            "distilbert-base-uncased-finetuned-sst-2-english"
                        ]
                    }
                ]
            },
            "text_processing": {
                "priority": 2,
                "required": True,
                "models": [
                    {
                        "name": "booknlp_model",
                        "source": "url",
                        "model_url": "https://github.com/booknlp/booknlp/releases/download/v1.0.4/model_v1.0.4.zip",
                        "local_path": "models/booknlp/model_v1.0.4",
                        "size_mb": 2000,
                        "description": "BookNLP model for character and entity extraction",
                        "fallback_models": [],
                        "optional": True  # BookNLP can work without pre-downloaded models
                    }
                ]
            },
            "image_generation": {
                "priority": 3,
                "required": False,
                "models": [
                    {
                        "name": "prompt_enhancer",
                        "source": "huggingface", 
                        "model_id": "microsoft/DialoGPT-medium",
                        "local_path": "models/image_generation/prompt_enhancer",
                        "size_mb": 800,
                        "description": "Model for enhancing image generation prompts",
                        "fallback_models": [
                            "microsoft/DialoGPT-small",
                            "gpt2"
                        ]
                    }
                ]
            },
            "voice_processing": {
                "priority": 4,
                "required": False,
                "models": [
                    {
                        "name": "voice_classifier",
                        "source": "huggingface",
                        "model_id": "facebook/wav2vec2-base",
                        "local_path": "models/voice_processing/voice_classifier",
                        "size_mb": 350,
                        "description": "Voice characteristic classification",
                        "fallback_models": [
                            "facebook/wav2vec2-large-960h"
                        ]
                    }
                ]
            }
        }
    
    def initialize_all_models(self, 
                            force_download: bool = False,
                            model_types: Optional[List[str]] = None,
                            skip_optional: bool = False) -> Dict[str, Any]:
        """
        Initialize all models with comprehensive error handling
        
        Args:
            force_download: Whether to force re-download existing models
            model_types: Specific model types to download (None = all)
            skip_optional: Whether to skip optional models
            
        Returns:
            Dictionary with initialization results
        """
        
        self.download_stats["start_time"] = datetime.utcnow()
        logger.info("Starting model initialization...")
        
        results = {
            "success": False,
            "downloaded_models": [],
            "failed_models": [],
            "skipped_models": [],
            "errors": [],
            "warnings": [],
            "stats": {}
        }
        
        try:
            # Create model directories
            self._create_model_directories()
            
            # Filter model types if specified
            models_to_process = self.model_definitions
            if model_types:
                models_to_process = {
                    k: v for k, v in self.model_definitions.items() 
                    if k in model_types
                }
            
            # Sort by priority (lower number = higher priority)
            sorted_model_types = sorted(
                models_to_process.items(),
                key=lambda x: x[1]["priority"]
            )
            
            # Process each model type
            for model_type, model_config in sorted_model_types:
                logger.info(f"Processing {model_type} models...")
                
                # Skip optional models if requested
                if skip_optional and not model_config["required"]:
                    logger.info(f"Skipping optional model type: {model_type}")
                    continue
                
                # Download models in this category
                for model_info in model_config["models"]:
                    try:
                        # Get device-optimized model config
                        optimized_model_info = self._get_device_specific_model_config(model_info)
                        
                        success = self._download_model(
                            optimized_model_info, 
                            force_download,
                            skip_optional
                        )
                        
                        if success:
                            results["downloaded_models"].append({
                                "type": model_type,
                                "name": model_info["name"],
                                "path": model_info["local_path"]
                            })
                        else:
                            results["failed_models"].append({
                                "type": model_type,
                                "name": model_info["name"],
                                "error": "Download failed"
                            })
                            
                    except Exception as e:
                        logger.error(f"Error processing model {model_info['name']}: {e}")
                        results["failed_models"].append({
                            "type": model_type,
                            "name": model_info["name"],
                            "error": str(e)
                        })
                        results["errors"].append(f"Model {model_info['name']}: {e}")
            
            # Check if we have minimum required models
            required_models = self._get_required_models()
            downloaded_names = [m["name"] for m in results["downloaded_models"]]
            missing_required = [m for m in required_models if m not in downloaded_names]
            
            if missing_required:
                results["warnings"].append(f"Missing required models: {missing_required}")
                logger.warning(f"Some required models are missing: {missing_required}")
            
            # Update statistics
            self.download_stats["end_time"] = datetime.utcnow()
            results["stats"] = self._generate_download_stats()
            
            # Determine overall success
            if results["downloaded_models"] or not missing_required:
                results["success"] = True
                logger.info("Model initialization completed successfully")
            else:
                results["success"] = False
                logger.error("Model initialization failed - no models downloaded")
            
            # Save initialization report
            self._save_initialization_report(results)
            
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            results["errors"].append(f"Initialization failed: {e}")
            results["success"] = False
        
        return results
    
    def _detect_compute_devices(self) -> Dict[str, Any]:
        """Detect available compute devices (GPU/CPU) for model optimization"""
        
        device_info = {
            "cpu": True,
            "cuda_available": False,
            "cuda_devices": 0,
            "mps_available": False,  # Apple Metal Performance Shaders
            "directml_available": False,  # Windows DirectML
            "recommended_device": "cpu",
            "gpu_memory_gb": 0,
            "platform": platform.system()
        }
        
        try:
            # Check for CUDA (NVIDIA GPUs)
            try:
                import torch
                if torch.cuda.is_available():
                    device_info["cuda_available"] = True
                    device_info["cuda_devices"] = torch.cuda.device_count()
                    device_info["recommended_device"] = "cuda"
                    
                    # Get GPU memory info
                    if device_info["cuda_devices"] > 0:
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory
                        device_info["gpu_memory_gb"] = gpu_memory / (1024**3)
                        logger.info(f"CUDA GPU detected: {torch.cuda.get_device_name(0)} with {device_info['gpu_memory_gb']:.1f}GB memory")
                    
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    # Apple Silicon GPU support
                    device_info["mps_available"] = True
                    device_info["recommended_device"] = "mps"
                    logger.info("Apple MPS GPU detected")
                    
            except ImportError:
                logger.info("PyTorch not available, checking other GPU options...")
            
            # Check for DirectML on Windows (AMD/Intel GPUs)
            if platform.system() == "Windows":
                try:
                    import torch_directml
                    device_info["directml_available"] = True
                    if device_info["recommended_device"] == "cpu":
                        device_info["recommended_device"] = "dml"
                    logger.info("DirectML GPU support detected")
                except ImportError:
                    pass
            
            # Check for TensorFlow GPU support
            try:
                import tensorflow as tf
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus and not device_info["cuda_available"]:
                    logger.info(f"TensorFlow GPU support detected: {len(gpus)} device(s)")
                    if device_info["recommended_device"] == "cpu":
                        device_info["recommended_device"] = "gpu"
            except ImportError:
                pass
            
        except Exception as e:
            logger.warning(f"Error detecting GPU devices: {e}")
        
        # Log final device recommendation
        if device_info["recommended_device"] != "cpu":
            logger.info(f"Recommended compute device: {device_info['recommended_device']}")
        else:
            logger.info("Using CPU for model processing (no GPU acceleration available)")
        
        return device_info
    
    def _get_device_specific_model_config(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get device-optimized model configuration"""
        
        config = model_info.copy()
        
        # Adjust model variants based on available compute
        if self.device_info["recommended_device"] != "cpu":
            # For GPU setups, prefer larger/more accurate models if available
            if "fallback_models" in config:
                # Reorder fallbacks to prioritize better models for GPU
                fallbacks = config["fallback_models"]
                
                # Move larger models to front if we have GPU with sufficient memory
                if self.device_info.get("gpu_memory_gb", 0) >= 8:
                    # Sufficient GPU memory - prefer larger models
                    larger_models = [m for m in fallbacks if "large" in m]
                    other_models = [m for m in fallbacks if "large" not in m]
                    config["fallback_models"] = larger_models + other_models
        else:
            # For CPU setups, prefer smaller/faster models
            if "fallback_models" in config:
                fallbacks = config["fallback_models"]
                
                # Prioritize smaller models for CPU
                smaller_models = [m for m in fallbacks if any(x in m for x in ["small", "base", "distil"])]
                other_models = [m for m in fallbacks if not any(x in m for x in ["small", "base", "distil"])]
                config["fallback_models"] = smaller_models + other_models
        
        return config
    
    def _download_model(self,
                       model_info: Dict[str, Any], 
                       force_download: bool,
                       skip_optional: bool) -> bool:
        """Download a single model with fallback mechanisms"""
        
        model_name = model_info["name"]
        local_path = Path(model_info["local_path"])
        
        try:
            # Check if model already exists
            if local_path.exists() and not force_download:
                if self._verify_model_integrity(local_path, model_info):
                    logger.info(f"Model {model_name} already exists and is valid")
                    self.download_stats["skipped_models"] += 1
                    return True
                else:
                    logger.warning(f"Model {model_name} exists but appears corrupted, re-downloading...")
            
            # Skip optional models if requested
            if skip_optional and model_info.get("optional", False):
                logger.info(f"Skipping optional model: {model_name}")
                self.download_stats["skipped_models"] += 1
                return True
            
            # Attempt to download the model
            logger.info(f"Downloading {model_name} (~{model_info['size_mb']}MB)...")
            
            success = False
            
            # Try primary model first
            if model_info["source"] == "huggingface":
                success = self._download_huggingface_model(model_info)
            elif model_info["source"] == "url":
                success = self._download_url_model(model_info)
            
            # Try fallback models if primary failed
            if not success and model_info.get("fallback_models"):
                logger.warning(f"Primary model {model_name} failed, trying fallbacks...")
                
                for fallback_id in model_info["fallback_models"]:
                    try:
                        fallback_info = model_info.copy()
                        fallback_info["model_id"] = fallback_id
                        fallback_info["local_path"] = f"{model_info['local_path']}_fallback"
                        
                        logger.info(f"Trying fallback model: {fallback_id}")
                        
                        if model_info["source"] == "huggingface":
                            success = self._download_huggingface_model(fallback_info)
                        
                        if success:
                            logger.info(f"Successfully downloaded fallback model: {fallback_id}")
                            break
                            
                    except Exception as e:
                        logger.warning(f"Fallback model {fallback_id} failed: {e}")
                        continue
            
            if success:
                self.download_stats["successful_downloads"] += 1
                self.download_stats["total_size_mb"] += model_info["size_mb"]
                logger.info(f"Successfully downloaded {model_name}")
            else:
                self.download_stats["failed_downloads"] += 1
                logger.error(f"Failed to download {model_name}")
            
            self.download_stats["total_models"] += 1
            return success
            
        except Exception as e:
            logger.error(f"Error downloading model {model_name}: {e}")
            self.download_stats["failed_downloads"] += 1
            self.download_stats["total_models"] += 1
            return False
    
    def _download_huggingface_model(self, model_info: Dict[str, Any]) -> bool:
        """Download model from HuggingFace with GPU optimization"""
        
        try:
            # Try using the model manager first with device info
            result = self.model_manager.download_model(
                model_id=model_info["model_id"],
                local_path=model_info["local_path"],
                source="huggingface",
                device_info=self.device_info
            )
            
            if result["success"]:
                return True
                
        except Exception as e:
            logger.warning(f"Model manager download failed: {e}")
        
        # Fallback: Try direct HuggingFace download with device optimization
        try:
            from transformers import AutoTokenizer, AutoModel, AutoConfig
            
            model_id = model_info["model_id"]
            local_path = Path(model_info["local_path"])
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Prepare device-specific download arguments
            download_kwargs = self._get_download_kwargs()
            
            # Download with timeout and retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    device_str = self.device_info["recommended_device"]
                    logger.info(f"Downloading {model_id} for {device_str} (attempt {attempt + 1}/{max_retries})...")
                    
                    # Download tokenizer and model with device optimization
                    tokenizer = AutoTokenizer.from_pretrained(model_id, **download_kwargs)
                    
                    # For GPU setups, try to download with appropriate precision
                    model_kwargs = download_kwargs.copy()
                    if self.device_info["recommended_device"] != "cpu":
                        # Try mixed precision for GPU if available
                        try:
                            model_kwargs["torch_dtype"] = self._get_optimal_dtype()
                        except:
                            pass
                    
                    model = AutoModel.from_pretrained(model_id, **model_kwargs)
                    
                    # Save locally with device-optimized configuration
                    tokenizer.save_pretrained(str(local_path))
                    model.save_pretrained(str(local_path))
                    
                    # Save device configuration for runtime
                    self._save_device_config(local_path)
                    
                    logger.info(f"Successfully downloaded {model_id} optimized for {device_str}")
                    return True
                    
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt < max_retries - 1:
                        time.sleep(5)  # Wait before retry
                    continue
            
            return False
            
        except ImportError:
            logger.error("transformers library not available for HuggingFace downloads")
            return False
        except Exception as e:
            logger.error(f"HuggingFace download failed: {e}")
            return False
    
    def _get_download_kwargs(self) -> Dict[str, Any]:
        """Get device-optimized download arguments"""
        
        kwargs = {}
        
        # Add device-specific optimizations
        if self.device_info["recommended_device"] != "cpu":
            # For GPU setups, enable optimizations
            try:
                import torch
                kwargs["use_fast_tokenizer"] = True
                
                # Memory-efficient loading for large models
                if self.device_info.get("gpu_memory_gb", 0) < 8:
                    kwargs["low_cpu_mem_usage"] = True
                    
            except ImportError:
                pass
        
        return kwargs
    
    def _get_optimal_dtype(self):
        """Get optimal data type for model based on device capabilities"""
        
        try:
            import torch
            
            if self.device_info["recommended_device"] == "cuda":
                # Use half precision for CUDA if supported
                if self.device_info.get("gpu_memory_gb", 0) >= 6:
                    return torch.float16
                else:
                    # For lower memory GPUs, use bfloat16 if available
                    if hasattr(torch, 'bfloat16'):
                        return torch.bfloat16
            
            # Default to float32
            return torch.float32
            
        except ImportError:
            return None
    
    def _save_device_config(self, model_path: Path):
        """Save device configuration alongside model"""
        
        try:
            config_file = model_path / "device_config.json"
            
            device_config = {
                "recommended_device": self.device_info["recommended_device"],
                "cuda_available": self.device_info["cuda_available"],
                "mps_available": self.device_info["mps_available"],
                "directml_available": self.device_info["directml_available"],
                "optimization_timestamp": datetime.utcnow().isoformat(),
                "platform": self.device_info["platform"]
            }
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(device_config, f, indent=2)
                
            logger.debug(f"Device config saved to {config_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save device config: {e}")
    
    def _download_url_model(self, model_info: Dict[str, Any]) -> bool:
        """Download model from URL with fallbacks"""
        
        try:
            import requests
            import zipfile
            import tempfile
            
            model_url = model_info["model_url"]
            local_path = Path(model_info["local_path"])
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download with progress
            logger.info(f"Downloading from {model_url}...")
            
            response = requests.get(model_url, stream=True, timeout=300)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        tmp_file.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded_size % (10 * 1024 * 1024) < 8192:
                            progress = (downloaded_size / total_size * 100) if total_size else 0
                            logger.info(f"Download progress: {progress:.1f}%")
                
                tmp_file.flush()
                
                # Extract if it's a zip file
                if model_url.endswith('.zip'):
                    logger.info("Extracting downloaded archive...")
                    with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                        zip_ref.extractall(str(local_path))
                else:
                    # Move file to destination
                    import shutil
                    shutil.move(tmp_file.name, str(local_path / Path(model_url).name))
            
            # Clean up temp file
            try:
                os.unlink(tmp_file.name)
            except:
                pass
            
            logger.info(f"Successfully downloaded and extracted model")
            return True
            
        except ImportError:
            logger.error("requests library not available for URL downloads")
            return False
        except Exception as e:
            logger.error(f"URL download failed: {e}")
            return False
    
    def _verify_model_integrity(self, model_path: Path, model_info: Dict[str, Any]) -> bool:
        """Verify that a downloaded model is complete and valid"""
        
        try:
            if not model_path.exists():
                return False
            
            # Basic checks
            if model_path.is_dir():
                # Check if directory has reasonable content
                files = list(model_path.rglob("*"))
                if len(files) < 2:  # Should have at least a few files
                    return False
                
                # Check for common model files
                has_model_files = any(
                    f.name in ["config.json", "pytorch_model.bin", "tf_model.h5", "model.safetensors"]
                    for f in files
                )
                
                if not has_model_files:
                    logger.warning(f"Model directory {model_path} lacks expected model files")
                
                return True
            else:
                # Single file - check size
                file_size_mb = model_path.stat().st_size / (1024 * 1024)
                expected_size_mb = model_info.get("size_mb", 0)
                
                # Allow 20% variance in file size
                if expected_size_mb > 0:
                    size_ratio = file_size_mb / expected_size_mb
                    if size_ratio < 0.8 or size_ratio > 1.2:
                        logger.warning(f"Model size {file_size_mb}MB differs significantly from expected {expected_size_mb}MB")
                        return False
                
                return True
                
        except Exception as e:
            logger.warning(f"Model integrity check failed: {e}")
            return False
    
    def _create_model_directories(self):
        """Create all necessary model directories"""
        
        try:
            base_dirs = [
                "models",
                "models/character_analysis", 
                "models/text_processing",
                "models/booknlp",
                "models/image_generation",
                "models/voice_processing"
            ]
            
            for dir_path in base_dirs:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
                
            logger.info("Model directories created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create model directories: {e}")
            raise
    
    def _get_required_models(self) -> List[str]:
        """Get list of required model names"""
        
        required_models = []
        
        for model_type, config in self.model_definitions.items():
            if config["required"]:
                for model_info in config["models"]:
                    if not model_info.get("optional", False):
                        required_models.append(model_info["name"])
        
        return required_models
    
    def _generate_download_stats(self) -> Dict[str, Any]:
        """Generate comprehensive download statistics"""
        
        stats = self.download_stats.copy()
        
        if stats["start_time"] and stats["end_time"]:
            duration = stats["end_time"] - stats["start_time"]
            stats["total_time_seconds"] = duration.total_seconds()
            
            if stats["total_time_seconds"] > 0:
                stats["download_speed_mbps"] = stats["total_size_mb"] / (stats["total_time_seconds"] / 60)
            else:
                stats["download_speed_mbps"] = 0
        
        stats["success_rate"] = (
            stats["successful_downloads"] / max(stats["total_models"], 1)
        )
        
        return stats
    
    def _save_initialization_report(self, results: Dict[str, Any]):
        """Save detailed initialization report"""
        
        try:
            report_dir = Path("models")
            report_dir.mkdir(exist_ok=True)
            
            report_file = report_dir / "initialization_report.json"
            
            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "results": results,
                "model_definitions": self.model_definitions,
                "system_info": {
                    "python_version": sys.version,
                    "platform": sys.platform
                },
                "device_info": self.device_info
            }
            
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Initialization report saved to {report_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save initialization report: {e}")
    
    def list_available_models(self) -> Dict[str, Any]:
        """List all available models and their status"""
        
        models_status = {}
        
        for model_type, config in self.model_definitions.items():
            models_status[model_type] = {
                "priority": config["priority"],
                "required": config["required"],
                "models": []
            }
            
            for model_info in config["models"]:
                local_path = Path(model_info["local_path"])
                
                status = {
                    "name": model_info["name"],
                    "description": model_info["description"],
                    "local_path": str(local_path),
                    "size_mb": model_info["size_mb"],
                    "downloaded": local_path.exists(),
                    "valid": False
                }
                
                if status["downloaded"]:
                    status["valid"] = self._verify_model_integrity(local_path, model_info)
                
                models_status[model_type]["models"].append(status)
        
        return models_status
    
    def cleanup_failed_downloads(self):
        """Clean up partially downloaded or corrupted models"""
        
        logger.info("Cleaning up failed downloads...")
        cleaned_count = 0
        
        try:
            for model_type, config in self.model_definitions.items():
                for model_info in config["models"]:
                    local_path = Path(model_info["local_path"])
                    
                    if local_path.exists():
                        if not self._verify_model_integrity(local_path, model_info):
                            logger.info(f"Removing corrupted model: {local_path}")
                            
                            if local_path.is_dir():
                                import shutil
                                shutil.rmtree(str(local_path))
                            else:
                                local_path.unlink()
                            
                            cleaned_count += 1
            
            logger.info(f"Cleaned up {cleaned_count} corrupted models")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(
        description="Initialize AI models for EchoTales with comprehensive fallbacks"
    )
    
    parser.add_argument(
        "--download-all",
        action="store_true",
        help="Download all models including optional ones"
    )
    
    parser.add_argument(
        "--model-type",
        choices=["character_analysis", "text_processing", "image_generation", "voice_processing"],
        help="Download only models of specified type"
    )
    
    parser.add_argument(
        "--force-download",
        action="store_true", 
        help="Force re-download of existing models"
    )
    
    parser.add_argument(
        "--skip-optional",
        action="store_true",
        help="Skip optional models to save time and space"
    )
    
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and their status"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Clean up corrupted or failed downloads"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize environment
        config = init_environment()
        initializer = ModelInitializer(config)
        
        if args.list_models:
            # List models and their status
            models_status = initializer.list_available_models()
            
            print("\nEchoTales Model Status:")
            print("=" * 80)
            
            for model_type, type_info in models_status.items():
                print(f"\n{model_type.upper()} (Priority: {type_info['priority']}, Required: {type_info['required']})")
                print("-" * 60)
                
                for model in type_info["models"]:
                    status_icon = "✓" if model["downloaded"] and model["valid"] else "✗"
                    print(f"{status_icon} {model['name']} ({model['size_mb']}MB)")
                    print(f"    {model['description']}")
                    print(f"    Path: {model['local_path']}")
                    
                    if model["downloaded"]:
                        validity = "Valid" if model["valid"] else "Corrupted"
                        print(f"    Status: Downloaded, {validity}")
                    else:
                        print(f"    Status: Not downloaded")
                    print()
            
            return
        
        if args.cleanup:
            # Clean up corrupted downloads
            initializer.cleanup_failed_downloads()
            return
        
        # Determine model types to process
        model_types = None
        if args.model_type:
            model_types = [args.model_type]
        
        # Run initialization
        results = initializer.initialize_all_models(
            force_download=args.force_download,
            model_types=model_types,
            skip_optional=args.skip_optional and not args.download_all
        )
        
        # Display results
        print("\n" + "=" * 80)
        print("MODEL INITIALIZATION SUMMARY")
        print("=" * 80)
        
        if results["success"]:
            print("✓ Initialization completed successfully!")
        else:
            print("⚠ Initialization completed with issues")
        
        print(f"\nDownloaded models: {len(results['downloaded_models'])}")
        for model in results["downloaded_models"]:
            print(f"  ✓ {model['name']} ({model['type']})")
        
        if results["failed_models"]:
            print(f"\nFailed models: {len(results['failed_models'])}")
            for model in results["failed_models"]:
                print(f"  ✗ {model['name']} ({model['type']}): {model['error']}")
        
        if results["skipped_models"]:
            print(f"\nSkipped models: {len(results['skipped_models'])}")
        
        if results["warnings"]:
            print("\nWarnings:")
            for warning in results["warnings"]:
                print(f"  ⚠ {warning}")
        
        if results["errors"]:
            print("\nErrors:")
            for error in results["errors"]:
                print(f"  ✗ {error}")
        
        # Display statistics
        stats = results["stats"]
        print(f"\nStatistics:")
        print(f"  Total models processed: {stats['total_models']}")
        print(f"  Successful downloads: {stats['successful_downloads']}")
        print(f"  Failed downloads: {stats['failed_downloads']}")
        print(f"  Skipped models: {stats['skipped_models']}")
        print(f"  Total size downloaded: {stats['total_size_mb']:.1f} MB")
        
        # Display device information
        device_info = stats.get("device_info", {})
        if device_info:
            device = device_info.get("recommended_device", "cpu").upper()
            print(f"  Compute device: {device}")
            
            if device_info.get("cuda_available"):
                gpu_count = device_info.get("cuda_devices", 0)
                gpu_memory = device_info.get("gpu_memory_gb", 0)
                print(f"  GPU acceleration: {gpu_count} CUDA device(s), {gpu_memory:.1f}GB memory")
            elif device_info.get("mps_available"):
                print(f"  GPU acceleration: Apple Metal Performance Shaders")
            elif device_info.get("directml_available"):
                print(f"  GPU acceleration: DirectML (Windows)")
            else:
                print(f"  GPU acceleration: Not available")
        
        if stats.get("total_time_seconds"):
            minutes = stats["total_time_seconds"] / 60
            print(f"  Total time: {minutes:.1f} minutes")
            
            if stats.get("download_speed_mbps"):
                print(f"  Average speed: {stats['download_speed_mbps']:.1f} MB/min")
        
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print("=" * 80)
        
        if results["success"]:
            print("\nModels are ready! You can now run EchoTales.")
        else:
            print("\nSome models failed to download. EchoTales may have limited functionality.")
            print("Try running with --force-download or check your internet connection.")
        
        sys.exit(0 if results["success"] else 1)
        
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()