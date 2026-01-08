"""
Configuration utilities for AutoJudge
"""
import yaml
import os
from pathlib import Path

class Config:
    """Configuration manager"""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Load configuration from YAML file"""
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            cls._config = yaml.safe_load(f)
    
    @classmethod
    def get(cls, key_path, default=None):
        """
        Get configuration value by key path
        
        Args:
            key_path: Dot-separated path (e.g., 'model.classification.type')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        if cls._config is None:
            cls._load_config()
        
        keys = key_path.split('.')
        value = cls._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    @classmethod
    def get_all(cls):
        """Get entire configuration dictionary"""
        if cls._config is None:
            cls._load_config()
        return cls._config
    
    @classmethod
    def reload(cls):
        """Reload configuration from file"""
        cls._load_config()


# Global config instance
config = Config()
