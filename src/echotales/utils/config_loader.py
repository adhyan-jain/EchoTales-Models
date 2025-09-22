import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AppConfig:
    name: str
    version: str
    environment: str
    debug: bool
    host: str
    port: int


@dataclass
class DatabaseConfig:
    type: str
    url: str
    echo: bool


@dataclass
class RedisConfig:
    host: str
    port: int
    db: int
    decode_responses: bool


@dataclass
class VoiceDatasetConfig:
    path: str
    male_actors: int
    female_actors: int
    samples_per_actor: int
    supported_formats: list


@dataclass
class CachingConfig:
    enabled: bool
    ttl: int
    storage: str
    keys: Dict[str, str]


class ConfigLoader:
    """Load and manage configuration from YAML files"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default to development config
            config_path = "configs/development.yaml"
        
        self.config_path = Path(config_path)
        self._raw_config = {}
        self._load_config()
        
        # Create typed config objects
        self.app = self._create_app_config()
        self.database = self._create_database_config()
        self.redis = self._create_redis_config()
        self.voice_dataset = self._create_voice_dataset_config()
        self.caching = self._create_caching_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Replace environment variables
                content = self._substitute_env_vars(content)
                
                # Parse YAML
                self._raw_config = yaml.safe_load(content)
                
                logger.info(f"Configuration loaded from {self.config_path}")
                
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            # Load default configuration
            self._raw_config = self._get_default_config()
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration"""
        import re
        
        # Find all ${VAR_NAME} patterns
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if env var not found
        
        return re.sub(pattern, replace_var, content)
    
    def _create_app_config(self) -> AppConfig:
        """Create typed app configuration"""
        app_config = self._raw_config.get('app', {})
        return AppConfig(
            name=app_config.get('name', 'EchoTales-Enhanced'),
            version=app_config.get('version', '2.0.0'),
            environment=app_config.get('environment', 'development'),
            debug=app_config.get('debug', True),
            host=app_config.get('host', 'localhost'),
            port=app_config.get('port', 5000)
        )
    
    def _create_database_config(self) -> DatabaseConfig:
        """Create typed database configuration"""
        db_config = self._raw_config.get('database', {})
        return DatabaseConfig(
            type=db_config.get('type', 'sqlite'),
            url=db_config.get('url', 'sqlite:///data/cache/echotales_dev.db'),
            echo=db_config.get('echo', True)
        )
    
    def _create_redis_config(self) -> RedisConfig:
        """Create typed Redis configuration"""
        redis_config = self._raw_config.get('redis', {})
        return RedisConfig(
            host=redis_config.get('host', 'localhost'),
            port=redis_config.get('port', 6379),
            db=redis_config.get('db', 0),
            decode_responses=redis_config.get('decode_responses', True)
        )
    
    def _create_voice_dataset_config(self) -> VoiceDatasetConfig:
        """Create typed voice dataset configuration"""
        voice_config = self._raw_config.get('voice_dataset', {})
        return VoiceDatasetConfig(
            path=voice_config.get('path', 'data/voice_dataset/'),
            male_actors=voice_config.get('male_actors', 12),
            female_actors=voice_config.get('female_actors', 12),
            samples_per_actor=voice_config.get('samples_per_actor', 44),
            supported_formats=voice_config.get('supported_formats', ['wav', 'mp3', 'flac'])
        )
    
    def _create_caching_config(self) -> CachingConfig:
        """Create typed caching configuration"""
        cache_config = self._raw_config.get('caching', {})
        return CachingConfig(
            enabled=cache_config.get('enabled', True),
            ttl=cache_config.get('ttl', 3600),
            storage=cache_config.get('storage', 'redis'),
            keys=cache_config.get('keys', {
                'novels': 'novel:{book_id}',
                'characters': 'characters:{book_id}',
                'audio': 'audio:{book_id}:{chapter}',
                'images': 'images:{book_id}:{character_id}'
            })
        )
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration when file loading fails"""
        return {
            'app': {
                'name': 'EchoTales-Enhanced',
                'version': '2.0.0',
                'environment': 'development',
                'debug': True,
                'host': 'localhost',
                'port': 5000
            },
            'database': {
                'type': 'sqlite',
                'url': 'sqlite:///data/cache/echotales_dev.db',
                'echo': True
            },
            'redis': {
                'host': 'localhost',
                'port': 6379,
                'db': 0,
                'decode_responses': True
            },
            'voice_dataset': {
                'path': 'data/voice_dataset/',
                'male_actors': 12,
                'female_actors': 12,
                'samples_per_actor': 44,
                'supported_formats': ['wav', 'mp3', 'flac']
            },
            'caching': {
                'enabled': True,
                'ttl': 3600,
                'storage': 'redis',
                'keys': {
                    'novels': 'novel:{book_id}',
                    'characters': 'characters:{book_id}',
                    'audio': 'audio:{book_id}:{chapter}',
                    'images': 'images:{book_id}:{character_id}'
                }
            }
        }
    
    def get_raw_config(self) -> Dict[str, Any]:
        """Get raw configuration dictionary"""
        return self._raw_config
    
    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'app.debug')"""
        keys = key_path.split('.')
        current = self._raw_config
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current