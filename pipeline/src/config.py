"""
Configuration management for the arthropod classification pipeline.

This module provides a centralized configuration system using YAML files.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import re


class Config:
    """
    Configuration manager with environment variable expansion and path resolution.

    Supports:
    - Loading from YAML files
    - Environment variable substitution (${VAR_NAME})
    - Path reference expansion (${paths.data_root})
    - Nested key access with dot notation

    Example:
        >>> config = Config()
        >>> data_path = config.get('paths.data_root')
        >>> batch_size = config.get('classification.training.batch_size')
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to custom config file. If None, uses default_config.yaml
        """
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / "config" / "default_config.yaml"

        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        self.config = self._load_config()
        self._expand_variables()
        self._resolve_paths()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _expand_variables(self):
        """
        Expand environment variables and internal references.

        Supports:
        - ${ENV_VAR} - Environment variables
        - ${paths.data_root} - Internal config references
        """
        self.config = self._expand_dict(self.config, self.config)

    def _expand_dict(self, obj: Any, root_config: Dict) -> Any:
        """Recursively expand variables in dictionaries and lists."""
        if isinstance(obj, dict):
            return {k: self._expand_dict(v, root_config) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._expand_dict(item, root_config) for item in obj]
        elif isinstance(obj, str):
            return self._expand_string(obj, root_config)
        else:
            return obj

    def _expand_string(self, value: str, root_config: Dict) -> str:
        """
        Expand variables in a string.

        Examples:
            "${HOME}/data" -> "/home/user/data"
            "${paths.data_root}/models" -> "./data/models"
        """
        # Pattern: ${variable.name}
        pattern = r'\$\{([^}]+)\}'

        def replace_var(match):
            var_name = match.group(1)

            # Check if it's an environment variable
            if var_name in os.environ:
                return os.environ[var_name]

            # Check if it's an internal config reference (dot notation)
            if '.' in var_name:
                keys = var_name.split('.')
                val = root_config
                try:
                    for key in keys:
                        val = val[key]
                    return str(val)
                except (KeyError, TypeError):
                    # Variable not found, return as-is
                    return match.group(0)

            # Variable not found, return as-is
            return match.group(0)

        return re.sub(pattern, replace_var, value)

    def _resolve_paths(self):
        """Convert path strings to Path objects and resolve relative paths."""
        if 'paths' in self.config:
            base_dir = Path(__file__).parent.parent  # Project root

            for key, value in self.config['paths'].items():
                if isinstance(value, str):
                    path = Path(value)

                    # Resolve relative paths relative to project root
                    if not path.is_absolute():
                        path = (base_dir / path).resolve()

                    self.config['paths'][key] = str(path)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key: Configuration key in dot notation (e.g., 'paths.data_root')
            default: Default value if key not found

        Returns:
            Configuration value or default

        Example:
            >>> config.get('classification.training.batch_size')
            16
            >>> config.get('non.existent.key', 42)
            42
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any):
        """
        Set configuration value using dot notation.

        Args:
            key: Configuration key in dot notation
            value: Value to set

        Example:
            >>> config.set('classification.training.batch_size', 32)
        """
        keys = key.split('.')
        target = self.config

        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]

        target[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update multiple configuration values.

        Args:
            updates: Dictionary of updates (can be nested)
        """
        self._deep_update(self.config, updates)

    def _deep_update(self, target: Dict, updates: Dict):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    def save(self, output_path: Optional[str] = None):
        """
        Save configuration to YAML file.

        Args:
            output_path: Output file path. If None, overwrites original file.
        """
        if output_path is None:
            output_path = self.config_path

        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        return f"Config(path={self.config_path})"


# Global config instance
# Users can override by calling: config = Config('path/to/custom_config.yaml')
config = Config()


# Convenience functions
def get_config(key: str, default: Any = None) -> Any:
    """
    Get configuration value (convenience function).

    Args:
        key: Configuration key in dot notation
        default: Default value if not found

    Returns:
        Configuration value
    """
    return config.get(key, default)


def load_custom_config(config_path: str) -> Config:
    """
    Load a custom configuration file.

    Args:
        config_path: Path to custom YAML config file

    Returns:
        Config instance

    Example:
        >>> from src.config import load_custom_config
        >>> config = load_custom_config('my_config.yaml')
    """
    return Config(config_path)
