"""
Image utility functions.

Helper functions for image processing operations.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional, Union


def load_image(image_path: Union[str, Path]) -> np.ndarray:
    """
    Load image as numpy array (RGB format).

    Args:
        image_path: Path to image file

    Returns:
        Image as numpy array in RGB format

    Raises:
        FileNotFoundError: If image doesn't exist
        ValueError: If image cannot be loaded
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Try OpenCV first (faster)
    img = cv2.imread(str(image_path))

    if img is None:
        # Fallback to PIL (handles more formats)
        try:
            img = np.array(Image.open(image_path))
            if img.ndim == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img
        except Exception as e:
            raise ValueError(f"Could not load image {image_path}: {e}")

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def save_image(image: np.ndarray, output_path: Union[str, Path], quality: int = 95):
    """
    Save image to file.

    Args:
        image: Image as numpy array (RGB format)
        output_path: Path to save image
        quality: JPEG quality (0-100), only for .jpg files

    Raises:
        ValueError: If save fails
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert RGB to BGR for OpenCV
    if image.ndim == 3 and image.shape[2] == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = image

    # Set quality for JPEG
    if output_path.suffix.lower() in ['.jpg', '.jpeg']:
        success = cv2.imwrite(str(output_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        success = cv2.imwrite(str(output_path), img_bgr)

    if not success:
        raise ValueError(f"Failed to save image to {output_path}")


def get_image_dimensions(image_path: Union[str, Path]) -> Tuple[int, int]:
    """
    Get image dimensions without loading full image.

    Args:
        image_path: Path to image

    Returns:
        Tuple of (width, height)
    """
    with Image.open(image_path) as img:
        return img.size


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    interpolation: int = cv2.INTER_LANCZOS4
) -> np.ndarray:
    """
    Resize image.

    Args:
        image: Input image
        target_size: Target (width, height). If None, uses scale_factor.
        scale_factor: Scale factor (e.g., 0.5 for half size). Ignored if target_size provided.
        interpolation: OpenCV interpolation method

    Returns:
        Resized image

    Example:
        >>> img = load_image('large.jpg')
        >>> small = resize_image(img, target_size=(800, 600))
        >>> half = resize_image(img, scale_factor=0.5)
    """
    if target_size is None and scale_factor is None:
        raise ValueError("Must provide either target_size or scale_factor")

    if target_size is not None:
        width, height = target_size
    else:
        height, width = image.shape[:2]
        width = int(width * scale_factor)
        height = int(height * scale_factor)

    return cv2.resize(image, (width, height), interpolation=interpolation)


def calculate_image_hash(image_path: Union[str, Path]) -> str:
    """
    Calculate perceptual hash of image.

    Useful for detecting duplicate or near-duplicate images.

    Args:
        image_path: Path to image

    Returns:
        Hash string

    Example:
        >>> hash1 = calculate_image_hash('img1.jpg')
        >>> hash2 = calculate_image_hash('img2.jpg')
        >>> if hash1 == hash2:
        ...     print("Images are identical or very similar")
    """
    import hashlib

    img = load_image(image_path)

    # Resize to small size and convert to grayscale
    small = resize_image(img, target_size=(8, 8))
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)

    # Calculate hash
    img_bytes = gray.tobytes()
    return hashlib.md5(img_bytes).hexdigest()


def crop_to_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop image to bounding box.

    Args:
        image: Input image
        bbox: Bounding box as (x1, y1, x2, y2)

    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = map(int, bbox)

    # Ensure within bounds
    height, width = image.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    return image[y1:y2, x1:x2]


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calculate image sharpness using Laplacian variance.

    Higher values = sharper image.

    Args:
        image: Input image (RGB or grayscale)

    Returns:
        Sharpness score

    Example:
        >>> img = load_image('photo.jpg')
        >>> sharpness = calculate_sharpness(img)
        >>> print(f"Sharpness: {sharpness:.2f}")
    """
    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Calculate Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Return variance
    return laplacian.var()


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: int = 8) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    Enhances contrast in images, useful for improving detection/segmentation.

    Args:
        image: Input image (RGB)
        clip_limit: Threshold for contrast limiting
        tile_size: Size of grid for histogram equalization

    Returns:
        Enhanced image

    Example:
        >>> img = load_image('dark_image.jpg')
        >>> enhanced = apply_clahe(img)
        >>> save_image(enhanced, 'enhanced.jpg')
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge([l, a, b])

    # Convert back to RGB
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def create_thumbnail(
    image_path: Union[str, Path],
    output_path: Union[str, Path],
    max_size: int = 200
):
    """
    Create thumbnail of image.

    Args:
        image_path: Input image path
        output_path: Output thumbnail path
        max_size: Maximum dimension (width or height)

    Example:
        >>> create_thumbnail('large.jpg', 'thumb.jpg', max_size=150)
    """
    img = Image.open(image_path)

    # Calculate new size maintaining aspect ratio
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Save
    img.save(output_path, optimize=True, quality=85)


def batch_resize_images(
    input_dir: Path,
    output_dir: Path,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    pattern: str = "*.jpg"
):
    """
    Batch resize all images in directory.

    Args:
        input_dir: Directory with input images
        output_dir: Directory for output images
        target_size: Target (width, height)
        scale_factor: Scale factor
        pattern: File pattern (e.g., "*.jpg", "*.png")

    Example:
        >>> batch_resize_images(
        ...     Path('./originals'),
        ...     Path('./resized'),
        ...     scale_factor=0.5
        ... )
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list(input_dir.glob(pattern))

    for img_path in image_paths:
        img = load_image(img_path)
        resized = resize_image(img, target_size=target_size, scale_factor=scale_factor)

        output_path = output_dir / img_path.name
        save_image(resized, output_path)

    print(f"Resized {len(image_paths)} images to {output_dir}")
