"""
Environment configuration management for EchoTales.
Handles loading environment variables, validation, and configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    url: str = "postgresql://localhost:5432/echotales"
    redis_url: str = "redis://localhost:6379/0"


@dataclass 
class APIKeysConfig:
    """API keys for external services"""
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    stability_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None


@dataclass
class ImageGenerationConfig:
    """Image generation settings"""
    enabled: bool = True
    cache_dir: str = "data/output/images"
    max_size: str = "1024x1024"
    default_style: str = "cinematic"
    
    # Provider priorities (1=highest)
    gemini_priority: int = 1
    huggingface_priority: int = 2
    openai_priority: int = 3
    stability_priority: int = 4
    
    # Rate limits (requests per minute)
    gemini_rate_limit: int = 10
    openai_rate_limit: int = 5
    stability_rate_limit: int = 3
    huggingface_rate_limit: int = 20


@dataclass
class VoiceConfig:
    """Voice generation settings"""
    enabled: bool = True
    cache_dir: str = "data/output/audio"
    default_engine: str = "openvoice"
    sample_rate: int = 22050
    format: str = "wav"
    male_voices_path: str = "data/voice_dataset/male_actors"
    female_voices_path: str = "data/voice_dataset/female_actors"


@dataclass
class ProcessingConfig:
    """Text processing settings"""
    booknlp_enabled: bool = True
    booknlp_model_path: str = "models/booknlp"
    character_analysis_enabled: bool = True


@dataclass
class CacheConfig:
    """Caching and storage settings"""
    ttl: int = 3600
    max_size: str = "1GB"
    persistent: bool = True
    max_upload_size: str = "50MB"
    allowed_file_types: List[str] = field(default_factory=lambda: ["txt", "epub", "pdf", "docx"])


@dataclass
class MonitoringConfig:
    """Monitoring and logging settings"""
    log_level: str = "INFO"
    log_file: str = "logs/echotales.log"
    error_tracking: bool = True
    cuda_enabled: bool = True
    max_gpu_memory: str = "4GB"


@dataclass
class SecurityConfig:
    """Security settings"""
    secret_key: str = "dev-secret-key-change-in-production"
    jwt_secret_key: str = "dev-jwt-secret-key"
    cors_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:3001"])


@dataclass
class AWSConfig:
    """AWS configuration"""
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    bucket_name: Optional[str] = None
    region: str = "us-east-1"


@dataclass
class DevelopmentConfig:
    """Development-specific settings"""
    mock_apis: bool = False
    mock_response_delay: int = 1
    debug_sql: bool = False
    api_docs_enabled: bool = True


@dataclass
class AppConfig:
    """Flask application settings"""
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = True
    env: str = "development"


@dataclass
class EchoTalesConfig:
    """Main configuration class containing all settings"""
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api_keys: APIKeysConfig = field(default_factory=APIKeysConfig)
    image_generation: ImageGenerationConfig = field(default_factory=ImageGenerationConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    development: DevelopmentConfig = field(default_factory=DevelopmentConfig)


class EnvironmentManager:
    """Manages environment configuration loading and validation"""
    
    def __init__(self, env_file: Optional[str] = None):
        self.env_file = env_file or ".env"
        self.config: Optional[EchoTalesConfig] = None
        self._load_environment()
    
    def _load_environment(self):
        """Load environment variables from .env file"""
        # Try to load from the specified path
        env_path = Path(self.env_file)
        
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from {env_path}")
        else:
            logger.warning(f"Environment file {env_path} not found. Using system environment variables only.")
    
    def get_config(self) -> EchoTalesConfig:
        """Get the complete configuration object"""
        if self.config is None:
            self.config = self._build_config()
        return self.config
    
    def _build_config(self) -> EchoTalesConfig:
        """Build configuration from environment variables"""
        return EchoTalesConfig(
            app=AppConfig(
                host=os.getenv("FLASK_HOST", "127.0.0.1"),
                port=int(os.getenv("FLASK_PORT", "5000")),
                debug=os.getenv("FLASK_DEBUG", "True").lower() == "true",
                env=os.getenv("FLASK_ENV", "development")
            ),
            database=DatabaseConfig(
                url=os.getenv("DATABASE_URL", "postgresql://localhost:5432/echotales"),
                redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0")
            ),
            api_keys=APIKeysConfig(
                openai_api_key=os.getenv("OPENAI_API_KEY"),
                gemini_api_key=os.getenv("GEMINI_API_KEY"),
                stability_api_key=os.getenv("STABILITY_API_KEY"),
                huggingface_token=os.getenv("HUGGINGFACE_API_TOKEN"),
                elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY")
            ),
            image_generation=ImageGenerationConfig(
                enabled=os.getenv("IMAGE_GENERATION_ENABLED", "True").lower() == "true",
                cache_dir=os.getenv("IMAGE_CACHE_DIR", "data/output/images"),
                max_size=os.getenv("MAX_IMAGE_SIZE", "1024x1024"),
                default_style=os.getenv("DEFAULT_IMAGE_STYLE", "cinematic"),
                gemini_priority=int(os.getenv("GEMINI_PRIORITY", "1")),
                huggingface_priority=int(os.getenv("HUGGINGFACE_PRIORITY", "2")),
                openai_priority=int(os.getenv("OPENAI_PRIORITY", "3")),
                stability_priority=int(os.getenv("STABILITY_PRIORITY", "4")),
                gemini_rate_limit=int(os.getenv("GEMINI_RATE_LIMIT", "10")),
                openai_rate_limit=int(os.getenv("OPENAI_RATE_LIMIT", "5")),
                stability_rate_limit=int(os.getenv("STABILITY_RATE_LIMIT", "3")),
                huggingface_rate_limit=int(os.getenv("HUGGINGFACE_RATE_LIMIT", "20"))
            ),
            voice=VoiceConfig(
                enabled=os.getenv("VOICE_GENERATION_ENABLED", "True").lower() == "true",
                cache_dir=os.getenv("VOICE_CACHE_DIR", "data/output/audio"),
                default_engine=os.getenv("DEFAULT_VOICE_ENGINE", "openvoice"),
                sample_rate=int(os.getenv("AUDIO_SAMPLE_RATE", "22050")),
                format=os.getenv("AUDIO_FORMAT", "wav"),
                male_voices_path=os.getenv("MALE_VOICES_PATH", "data/voice_dataset/male_actors"),
                female_voices_path=os.getenv("FEMALE_VOICES_PATH", "data/voice_dataset/female_actors")
            ),
            processing=ProcessingConfig(
                booknlp_enabled=os.getenv("BOOKNLP_ENABLED", "True").lower() == "true",
                booknlp_model_path=os.getenv("BOOKNLP_MODEL_PATH", "models/booknlp"),
                character_analysis_enabled=os.getenv("CHARACTER_ANALYSIS_ENABLED", "True").lower() == "true"
            ),
            cache=CacheConfig(
                ttl=int(os.getenv("CACHE_TTL", "3600")),
                max_size=os.getenv("MAX_CACHE_SIZE", "1GB"),
                persistent=os.getenv("ENABLE_PERSISTENT_CACHE", "True").lower() == "true",
                max_upload_size=os.getenv("MAX_UPLOAD_SIZE", "50MB"),
                allowed_file_types=os.getenv("ALLOWED_FILE_TYPES", "txt,epub,pdf,docx").split(",")
            ),
            monitoring=MonitoringConfig(
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                log_file=os.getenv("LOG_FILE", "logs/echotales.log"),
                error_tracking=os.getenv("ENABLE_ERROR_TRACKING", "True").lower() == "true",
                cuda_enabled=os.getenv("CUDA_ENABLED", "True").lower() == "true",
                max_gpu_memory=os.getenv("MAX_GPU_MEMORY", "4GB")
            ),
            security=SecurityConfig(
                secret_key=os.getenv("SECRET_KEY", "dev-secret-key-change-in-production"),
                jwt_secret_key=os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-key"),
                cors_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")
            ),
            aws=AWSConfig(
                access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                bucket_name=os.getenv("AWS_BUCKET_NAME"),
                region=os.getenv("AWS_REGION", "us-east-1")
            ),
            development=DevelopmentConfig(
                mock_apis=os.getenv("ENABLE_MOCK_APIS", "False").lower() == "true",
                mock_response_delay=int(os.getenv("MOCK_RESPONSE_DELAY", "1")),
                debug_sql=os.getenv("DEBUG_SQL", "False").lower() == "true",
                api_docs_enabled=os.getenv("ENABLE_API_DOCS", "True").lower() == "true"
            )
        )
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        config = self.get_config()
        
        # Check required API keys for enabled features
        if config.image_generation.enabled:
            if not any([
                config.api_keys.openai_api_key,
                config.api_keys.gemini_api_key, 
                config.api_keys.stability_api_key,
                config.api_keys.huggingface_token
            ]):
                issues.append("Image generation enabled but no API keys configured")
        
        if config.voice.enabled:
            if not any([
                config.api_keys.openai_api_key,
                config.api_keys.elevenlabs_api_key
            ]):
                issues.append("Voice generation enabled but no TTS API keys configured")
        
        # Check paths exist
        required_dirs = [
            config.image_generation.cache_dir,
            config.voice.cache_dir,
        ]
        
        for dir_path in required_dirs:
            if not Path(dir_path).exists():
                issues.append(f"Required directory does not exist: {dir_path}")
        
        # Check log directory
        log_dir = Path(config.monitoring.log_file).parent
        if not log_dir.exists():
            issues.append(f"Log directory does not exist: {log_dir}")
        
        return issues
    
    def create_required_directories(self):
        """Create required directories if they don't exist"""
        config = self.get_config()
        
        dirs_to_create = [
            config.image_generation.cache_dir,
            config.voice.cache_dir,
            Path(config.monitoring.log_file).parent,
            "models",
            "data/processed/chapters",
            "data/output/character_portraits",
            config.voice.male_voices_path,
            config.voice.female_voices_path
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        config = self.get_config()
        
        return {
            "app": {
                "host": config.app.host,
                "port": config.app.port,
                "environment": config.app.env
            },
            "features": {
                "image_generation": config.image_generation.enabled,
                "voice_generation": config.voice.enabled,
                "booknlp": config.processing.booknlp_enabled,
                "character_analysis": config.processing.character_analysis_enabled
            },
            "api_keys_configured": {
                "openai": bool(config.api_keys.openai_api_key),
                "gemini": bool(config.api_keys.gemini_api_key),
                "stability": bool(config.api_keys.stability_api_key),
                "huggingface": bool(config.api_keys.huggingface_token),
                "elevenlabs": bool(config.api_keys.elevenlabs_api_key)
            },
            "paths": {
                "image_cache": config.image_generation.cache_dir,
                "voice_cache": config.voice.cache_dir,
                "log_file": config.monitoring.log_file
            }
        }


# Global environment manager instance
_env_manager: Optional[EnvironmentManager] = None


def get_env_manager() -> EnvironmentManager:
    """Get the global environment manager instance"""
    global _env_manager
    if _env_manager is None:
        _env_manager = EnvironmentManager()
    return _env_manager


def get_config() -> EchoTalesConfig:
    """Get the global configuration object"""
    return get_env_manager().get_config()


def init_environment(env_file: Optional[str] = None) -> EchoTalesConfig:
    """Initialize the environment configuration"""
    global _env_manager
    _env_manager = EnvironmentManager(env_file)
    
    # Create required directories
    _env_manager.create_required_directories()
    
    # Validate configuration
    issues = _env_manager.validate_config()
    if issues:
        logger.warning("Configuration issues found:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    config = _env_manager.get_config()
    logger.info("Environment configuration initialized successfully")
    return config


if __name__ == "__main__":
    # Test the environment manager
    env_manager = EnvironmentManager()
    config = env_manager.get_config()
    
    print("Configuration Summary:")
    summary = env_manager.get_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\nValidation Issues:")
    issues = env_manager.validate_config()
    for issue in issues:
        print(f"- {issue}")