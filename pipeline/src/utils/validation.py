"""
Validation utilities.

Functions for validating inputs, files, and data.
"""

from pathlib import Path
from typing import List, Union, Optional
import numpy as np


def validate_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """
    Validate that file exists.

    Args:
        file_path: Path to file
        description: Description for error message

    Returns:
        Path object

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"{description} is not a file: {file_path}")

    return file_path


def validate_directory_exists(dir_path: Union[str, Path], create: bool = False) -> Path:
    """
    Validate that directory exists.

    Args:
        dir_path: Path to directory
        create: If True, create directory if it doesn't exist

    Returns:
        Path object

    Raises:
        FileNotFoundError: If directory doesn't exist and create=False
    """
    dir_path = Path(dir_path)

    if not dir_path.exists():
        if create:
            dir_path.mkdir(parents=True, exist_ok=True)
        else:
            raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise ValueError(f"Path is not a directory: {dir_path}")

    return dir_path


def validate_image_file(file_path: Union[str, Path]) -> Path:
    """
    Validate that file is an image.

    Args:
        file_path: Path to image file

    Returns:
        Path object

    Raises:
        ValueError: If file is not a valid image format
    """
    file_path = validate_file_exists(file_path, "Image file")

    valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    if file_path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Invalid image format: {file_path.suffix}. "
            f"Supported: {', '.join(valid_extensions)}"
        )

    return file_path


def validate_size_fraction(size_fraction: str) -> str:
    """
    Validate size fraction code.

    Args:
        size_fraction: Size fraction code (k1, 1, 2, 7, etc.)

    Returns:
        Validated size fraction

    Raises:
        ValueError: If invalid size fraction
    """
    valid_fractions = {'k1', '1', '1.5', '2', '3', '5', '7', 'A'}

    if size_fraction not in valid_fractions:
        raise ValueError(
            f"Invalid size fraction: {size_fraction}. "
            f"Valid options: {', '.join(sorted(valid_fractions))}"
        )

    return size_fraction


def validate_bounding_box(
    bbox: tuple,
    image_width: Optional[int] = None,
    image_height: Optional[int] = None
) -> tuple:
    """
    Validate bounding box format and values.

    Args:
        bbox: Bounding box as (x1, y1, x2, y2)
        image_width: Optional image width for bounds checking
        image_height: Optional image height for bounds checking

    Returns:
        Validated bounding box

    Raises:
        ValueError: If invalid bounding box
    """
    if len(bbox) != 4:
        raise ValueError(f"Bounding box must have 4 values, got {len(bbox)}")

    x1, y1, x2, y2 = bbox

    # Check order
    if x2 <= x1:
        raise ValueError(f"Invalid bbox: x2 ({x2}) must be > x1 ({x1})")

    if y2 <= y1:
        raise ValueError(f"Invalid bbox: y2 ({y2}) must be > y1 ({y1})")

    # Check bounds if image dimensions provided
    if image_width is not None:
        if x1 < 0 or x2 > image_width:
            raise ValueError(f"Bbox x coordinates out of image bounds (0, {image_width})")

    if image_height is not None:
        if y1 < 0 or y2 > image_height:
            raise ValueError(f"Bbox y coordinates out of image bounds (0, {image_height})")

    return bbox


def validate_confidence_score(confidence: float) -> float:
    """
    Validate confidence score is in valid range.

    Args:
        confidence: Confidence score

    Returns:
        Validated confidence

    Raises:
        ValueError: If confidence not in [0, 1]
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"Confidence must be in [0, 1], got {confidence}")

    return confidence


def validate_model_file(model_path: Union[str, Path]) -> Path:
    """
    Validate YOLO model file.

    Args:
        model_path: Path to model (.pt file)

    Returns:
        Path object

    Raises:
        ValueError: If not a valid model file
    """
    model_path = validate_file_exists(model_path, "Model file")

    if model_path.suffix != '.pt':
        raise ValueError(f"Model file must be .pt format, got {model_path.suffix}")

    return model_path


def validate_sample_id(sample_id: str) -> str:
    """
    Validate sample ID format.

    Sample IDs should contain only alphanumeric characters, underscores, and hyphens.

    Args:
        sample_id: Sample identifier

    Returns:
        Validated sample ID

    Raises:
        ValueError: If invalid format
    """
    if not sample_id:
        raise ValueError("Sample ID cannot be empty")

    # Check for valid characters
    import re
    if not re.match(r'^[A-Za-z0-9_-]+$', sample_id):
        raise ValueError(
            f"Sample ID '{sample_id}' contains invalid characters. "
            "Use only letters, numbers, underscores, and hyphens."
        )

    return sample_id


def validate_detections(detections: List[dict]) -> List[dict]:
    """
    Validate list of detection dictionaries.

    Args:
        detections: List of detection dicts

    Returns:
        Validated detections

    Raises:
        ValueError: If invalid detection format
    """
    required_keys = {'bounding_box', 'confidence'}

    for i, det in enumerate(detections):
        # Check required keys
        missing = required_keys - set(det.keys())
        if missing:
            raise ValueError(f"Detection {i} missing keys: {missing}")

        # Validate bbox
        validate_bounding_box(det['bounding_box'])

        # Validate confidence
        validate_confidence_score(det['confidence'])

    return detections


def check_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available.

    Returns:
        True if GPU available, False otherwise
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def validate_config_paths(config_dict: dict) -> dict:
    """
    Validate that all path values in config exist.

    Args:
        config_dict: Configuration dictionary

    Returns:
        Validated config

    Raises:
        FileNotFoundError: If any path doesn't exist
    """
    def check_paths(d, prefix=""):
        for key, value in d.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                check_paths(value, full_key)
            elif isinstance(value, str):
                # Check if it looks like a path
                if any(x in value for x in ['/', '\\', '.pt', '.yaml', '.json']):
                    path = Path(value)
                    # Only check if not a placeholder or template
                    if not any(x in value for x in ['${', '{', '<', '>']):
                        if not path.exists() and not str(path).startswith('./data'):
                            print(f"Warning: Path not found in config '{full_key}': {path}")

    check_paths(config_dict)
    return config_dict
