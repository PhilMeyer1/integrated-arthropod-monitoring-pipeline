"""
Tests for configuration system.

Tests YAML config loading, variable expansion, and path resolution.
"""

import pytest
from pathlib import Path
import yaml
import os

from src.config import Config


@pytest.fixture
def sample_config_dict():
    """Create sample configuration dictionary."""
    return {
        'paths': {
            'data_root': './data',
            'models': '${paths.data_root}/models',
            'raw': '${paths.data_root}/raw',
            'processed': '${paths.data_root}/processed'
        },
        'device': 'cpu',
        'image_processing': {
            'detection': {
                'confidence_threshold': 0.25,
                'iou_threshold': 0.2,
                'model_path': '${paths.models}/detection.pt'
            },
            'segmentation': {
                'confidence_threshold': 0.30,
                'dilation_factor': 1.02
            }
        },
        'training': {
            'epochs': 50,
            'batch_size': 16,
            'image_size': 512
        }
    }


@pytest.fixture
def sample_config_file(tmp_path, sample_config_dict):
    """Create sample YAML config file."""
    config_path = tmp_path / "test_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def config_with_env_vars(tmp_path):
    """Create config with environment variable references."""
    config_dict = {
        'paths': {
            'data_root': '${DATA_ROOT}',
            'models': '${DATA_ROOT}/models'
        },
        'api_key': '${API_KEY}'
    }
    config_path = tmp_path / "env_config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f)
    return config_path


class TestConfigLoading:
    """Test configuration loading."""

    def test_load_from_file(self, sample_config_file):
        """Test loading config from YAML file."""
        config = Config(config_path=sample_config_file)

        assert config is not None
        assert 'paths' in config._config
        assert 'device' in config._config

    def test_load_from_dict(self, sample_config_dict):
        """Test loading config from dictionary."""
        config = Config(config_dict=sample_config_dict)

        assert config is not None
        assert config.get('device') == 'cpu'

    def test_default_config_loading(self):
        """Test loading default config."""
        # Should try to load from config/default_config.yaml
        try:
            config = Config()
            assert config is not None
        except FileNotFoundError:
            # OK if default config doesn't exist in test environment
            pytest.skip("Default config file not found")

    def test_invalid_config_path(self):
        """Test loading from invalid path."""
        with pytest.raises(FileNotFoundError):
            Config(config_path=Path("nonexistent_config.yaml"))


class TestVariableExpansion:
    """Test variable expansion in config."""

    def test_basic_variable_expansion(self, sample_config_file):
        """Test expansion of ${variable} references."""
        config = Config(config_path=sample_config_file)

        # ${paths.data_root} should be expanded
        models_path = config.get('paths.models')
        assert '${' not in models_path
        assert 'data' in models_path
        assert 'models' in models_path

    def test_nested_variable_expansion(self, sample_config_file):
        """Test nested variable expansion."""
        config = Config(config_path=sample_config_file)

        # ${paths.models} contains ${paths.data_root}
        model_path = config.get('image_processing.detection.model_path')
        assert '${' not in model_path
        assert 'data' in model_path
        assert 'models' in model_path
        assert 'detection.pt' in model_path

    def test_environment_variable_expansion(self, config_with_env_vars):
        """Test expansion of environment variables."""
        # Set environment variables
        os.environ['DATA_ROOT'] = '/test/data'
        os.environ['API_KEY'] = 'test_key_12345'

        config = Config(config_path=config_with_env_vars)

        data_root = config.get('paths.data_root')
        assert data_root == '/test/data'

        api_key = config.get('api_key')
        assert api_key == 'test_key_12345'

        # Clean up
        del os.environ['DATA_ROOT']
        del os.environ['API_KEY']

    def test_missing_environment_variable(self, config_with_env_vars):
        """Test behavior with missing environment variable."""
        # Ensure env var doesn't exist
        if 'MISSING_VAR' in os.environ:
            del os.environ['MISSING_VAR']

        config_dict = {
            'test_var': '${MISSING_VAR}'
        }
        config_path = config_with_env_vars.parent / "missing_var.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        config = Config(config_path=config_path)

        # Should keep unexpanded or use default
        test_var = config.get('test_var')
        # Either unexpanded or empty/None
        assert test_var is None or '${' in test_var or test_var == ''


class TestDotNotationAccess:
    """Test dot notation access to config values."""

    def test_simple_key_access(self, sample_config_file):
        """Test accessing simple keys."""
        config = Config(config_path=sample_config_file)

        device = config.get('device')
        assert device == 'cpu'

    def test_nested_key_access(self, sample_config_file):
        """Test accessing nested keys with dot notation."""
        config = Config(config_path=sample_config_file)

        confidence = config.get('image_processing.detection.confidence_threshold')
        assert confidence == 0.25

        dilation = config.get('image_processing.segmentation.dilation_factor')
        assert dilation == 1.02

    def test_deeply_nested_access(self, sample_config_file):
        """Test deeply nested key access."""
        config = Config(config_path=sample_config_file)

        # Three levels deep
        model_path = config.get('image_processing.detection.model_path')
        assert model_path is not None

    def test_nonexistent_key(self, sample_config_file):
        """Test accessing nonexistent key."""
        config = Config(config_path=sample_config_file)

        result = config.get('nonexistent.key')
        assert result is None

    def test_default_value(self, sample_config_file):
        """Test default value when key doesn't exist."""
        config = Config(config_path=sample_config_file)

        result = config.get('nonexistent.key', default='default_value')
        assert result == 'default_value'


class TestPathResolution:
    """Test path resolution in config."""

    def test_relative_path_resolution(self, sample_config_file):
        """Test resolution of relative paths."""
        config = Config(config_path=sample_config_file)

        # Paths starting with ./ should be resolved
        data_root = config.get('paths.data_root')
        assert data_root == './data' or Path(data_root).parts[-1] == 'data'

    def test_absolute_path_preservation(self, tmp_path):
        """Test that absolute paths are preserved."""
        config_dict = {
            'paths': {
                'absolute': str(tmp_path / 'absolute'),
                'relative': './relative'
            }
        }
        config_path = tmp_path / "path_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        config = Config(config_path=config_path)

        absolute = config.get('paths.absolute')
        # Should preserve absolute path
        assert Path(absolute).is_absolute() or str(tmp_path) in absolute

    def test_path_concatenation(self, sample_config_file):
        """Test path concatenation through variable expansion."""
        config = Config(config_path=sample_config_file)

        # ${paths.data_root}/models should concatenate properly
        models_path = config.get('paths.models')
        assert 'data' in models_path
        assert 'models' in models_path
        # Should have path separator
        assert '/' in models_path or '\\' in models_path


class TestConfigValidation:
    """Test configuration validation."""

    def test_required_keys_present(self, sample_config_file):
        """Test that required keys are present."""
        config = Config(config_path=sample_config_file)

        required_keys = [
            'paths',
            'device',
            'image_processing'
        ]

        for key in required_keys:
            assert config.get(key) is not None

    def test_device_values(self, sample_config_file):
        """Test valid device values."""
        config = Config(config_path=sample_config_file)

        device = config.get('device')
        assert device in ['cpu', 'cuda', 'mps']

    def test_threshold_ranges(self, sample_config_file):
        """Test that thresholds are in valid range."""
        config = Config(config_path=sample_config_file)

        confidence = config.get('image_processing.detection.confidence_threshold')
        assert 0.0 <= confidence <= 1.0

        iou = config.get('image_processing.detection.iou_threshold')
        assert 0.0 <= iou <= 1.0

    def test_training_parameters(self, sample_config_file):
        """Test training parameter validity."""
        config = Config(config_path=sample_config_file)

        epochs = config.get('training.epochs')
        assert epochs > 0

        batch_size = config.get('training.batch_size')
        assert batch_size > 0

        image_size = config.get('training.image_size')
        assert image_size > 0


class TestConfigUpdates:
    """Test configuration updates."""

    def test_update_simple_value(self, sample_config_file):
        """Test updating a simple value."""
        config = Config(config_path=sample_config_file)

        original_device = config.get('device')
        assert original_device == 'cpu'

        config.set('device', 'cuda')
        assert config.get('device') == 'cuda'

    def test_update_nested_value(self, sample_config_file):
        """Test updating a nested value."""
        config = Config(config_path=sample_config_file)

        original_confidence = config.get('image_processing.detection.confidence_threshold')
        assert original_confidence == 0.25

        config.set('image_processing.detection.confidence_threshold', 0.35)
        assert config.get('image_processing.detection.confidence_threshold') == 0.35

    def test_add_new_key(self, sample_config_file):
        """Test adding a new key."""
        config = Config(config_path=sample_config_file)

        assert config.get('new_key') is None

        config.set('new_key', 'new_value')
        assert config.get('new_key') == 'new_value'


class TestIntegration:
    """Integration tests for config system."""

    def test_config_in_pipeline(self, sample_config_file):
        """Test using config throughout pipeline."""
        config = Config(config_path=sample_config_file)

        # Detection should use config values
        confidence = config.get('image_processing.detection.confidence_threshold')
        assert confidence is not None

        # Training should use config values
        epochs = config.get('training.epochs')
        assert epochs is not None

        # Paths should be resolved
        models_path = config.get('paths.models')
        assert models_path is not None

    def test_config_with_multiple_modules(self, sample_config_file):
        """Test config used across multiple modules."""
        config = Config(config_path=sample_config_file)

        # Each module should access different sections
        detection_conf = config.get('image_processing.detection.confidence_threshold')
        segmentation_conf = config.get('image_processing.segmentation.confidence_threshold')
        training_epochs = config.get('training.epochs')

        # All should be accessible
        assert detection_conf is not None
        assert segmentation_conf is not None
        assert training_epochs is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
