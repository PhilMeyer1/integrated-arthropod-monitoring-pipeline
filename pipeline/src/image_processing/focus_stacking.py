"""
Focus Stacking Module

Open-source alternative to Helicon Focus for combining multiple focal planes
into a single sharp image.

Note: The original pipeline used Helicon Focus Pro 8.3.563. This module provides
an open-source OpenCV-based implementation using the Laplacian variance method.
Results may differ slightly from the commercial software.

See EXTERNAL_TOOLS.md for details on the original software and alternatives.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Union, Optional, Tuple
from PIL import Image


class FocusStacker:
    """
    Focus stacking using Laplacian variance method.

    Combines multiple images with different focal planes into a single
    all-in-focus image by selecting the sharpest pixels from each image.

    Algorithm:
    1. Load all images in z-stack
    2. Convert to grayscale
    3. Calculate Laplacian (edge detection) for each image
    4. Create focus map: For each pixel, select image with highest Laplacian variance
    5. Blend images based on focus map

    Attributes:
        method: Stacking method ('laplacian' or 'pyramid')
    """

    def __init__(self, method: str = 'laplacian'):
        """
        Initialize focus stacker.

        Args:
            method: Stacking method. Currently supported:
                - 'laplacian': Simple Laplacian variance method (fast)
                - 'pyramid': Laplacian pyramid blending (higher quality, slower)
        """
        self.method = method

    def stack_images(
        self,
        image_paths: List[Path],
        output_path: Optional[Path] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Stack multiple focal planes into single sharp image.

        Args:
            image_paths: List of paths to images in z-stack (near to far focus)
            output_path: Path to save result. If None, doesn't save.
            verbose: Print progress messages

        Returns:
            Stacked image as numpy array (BGR format)

        Example:
            >>> stacker = FocusStacker()
            >>> z_stack = [Path(f'img_{i:03d}.jpg') for i in range(10)]
            >>> result = stacker.stack_images(z_stack, 'stacked.jpg')
        """
        if len(image_paths) == 0:
            raise ValueError("No images provided for stacking")

        if verbose:
            print(f"Focus stacking {len(image_paths)} images using {self.method} method...")

        # Load images
        images = self._load_images(image_paths, verbose=verbose)

        if self.method == 'laplacian':
            result = self._stack_laplacian(images, verbose=verbose)
        elif self.method == 'pyramid':
            result = self._stack_pyramid(images, verbose=verbose)
        else:
            raise ValueError(f"Unknown stacking method: {self.method}")

        # Save if output path provided
        if output_path is not None:
            cv2.imwrite(str(output_path), result)
            if verbose:
                print(f"Stacked image saved to {output_path}")

        return result

    def _load_images(
        self,
        image_paths: List[Path],
        verbose: bool = False
    ) -> List[np.ndarray]:
        """Load and validate images."""
        images = []

        for i, path in enumerate(image_paths):
            img = cv2.imread(str(path))

            if img is None:
                raise ValueError(f"Could not load image: {path}")

            # Check all images have same dimensions
            if i == 0:
                reference_shape = img.shape
            elif img.shape != reference_shape:
                # Resize to match first image
                if verbose:
                    print(f"Warning: Image {path.name} has different size, resizing...")
                img = cv2.resize(img, (reference_shape[1], reference_shape[0]))

            images.append(img)

            if verbose and (i + 1) % 5 == 0:
                print(f"  Loaded {i + 1}/{len(image_paths)} images...")

        return images

    def _stack_laplacian(
        self,
        images: List[np.ndarray],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Stack images using Laplacian variance method.

        For each pixel, selects the image with the highest Laplacian variance
        (sharpest edge).
        """
        if verbose:
            print("Calculating Laplacian variance for each image...")

        # Convert to grayscale and calculate Laplacian
        laplacians = []
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacians.append(np.abs(laplacian))

            if verbose and (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(images)} images...")

        # Stack laplacians to find max focus per pixel
        laplacian_stack = np.array(laplacians)

        # Create focus map: index of image with max Laplacian for each pixel
        focus_map = np.argmax(laplacian_stack, axis=0)

        if verbose:
            print("Blending images based on focus map...")

        # Create result by selecting pixels from sharpest image
        result = np.zeros_like(images[0])

        for i in range(len(images)):
            mask = (focus_map == i).astype(np.uint8)
            mask = cv2.merge([mask, mask, mask])  # Convert to 3-channel
            result += images[i] * mask

        return result

    def _stack_pyramid(
        self,
        images: List[np.ndarray],
        verbose: bool = False
    ) -> np.ndarray:
        """
        Stack images using Laplacian pyramid blending.

        Higher quality but slower than simple Laplacian method.
        Uses multi-scale blending to avoid visible seams.
        """
        if verbose:
            print("Building Laplacian pyramids...")

        # Parameters
        num_levels = 5  # Pyramid levels

        # Build Gaussian pyramids for each image
        gaussian_pyramids = []
        for img in images:
            pyramid = self._build_gaussian_pyramid(img, num_levels)
            gaussian_pyramids.append(pyramid)

        # Build Laplacian pyramids
        laplacian_pyramids = []
        for pyramid in gaussian_pyramids:
            lap_pyramid = self._build_laplacian_pyramid(pyramid)
            laplacian_pyramids.append(lap_pyramid)

        # Calculate focus measure pyramids
        focus_pyramids = []
        for lap_pyramid in laplacian_pyramids:
            focus_pyramid = []
            for level in lap_pyramid:
                gray = cv2.cvtColor(level, cv2.COLOR_BGR2GRAY) if level.ndim == 3 else level
                focus = cv2.Laplacian(gray, cv2.CV_64F)
                focus = np.abs(focus)
                focus_pyramid.append(focus)
            focus_pyramids.append(focus_pyramid)

        # Blend pyramids
        if verbose:
            print("Blending pyramids...")

        blended_pyramid = []
        for level_idx in range(num_levels):
            # Get focus measures for this level
            level_focus_measures = [fp[level_idx] for fp in focus_pyramids]
            focus_stack = np.array(level_focus_measures)

            # Find best image for each pixel
            best_indices = np.argmax(focus_stack, axis=0)

            # Blend Laplacian levels
            level_shape = laplacian_pyramids[0][level_idx].shape
            blended_level = np.zeros(level_shape, dtype=np.float64)

            for img_idx in range(len(images)):
                mask = (best_indices == img_idx).astype(np.float64)
                if len(level_shape) == 3:  # Color image
                    mask = np.stack([mask] * 3, axis=2)

                blended_level += laplacian_pyramids[img_idx][level_idx] * mask

            blended_pyramid.append(blended_level)

        # Reconstruct image from pyramid
        if verbose:
            print("Reconstructing final image...")

        result = self._reconstruct_from_pyramid(blended_pyramid)

        # Clip and convert to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def _build_gaussian_pyramid(
        self,
        image: np.ndarray,
        num_levels: int
    ) -> List[np.ndarray]:
        """Build Gaussian pyramid."""
        pyramid = [image]

        for _ in range(num_levels - 1):
            image = cv2.pyrDown(image)
            pyramid.append(image)

        return pyramid

    def _build_laplacian_pyramid(
        self,
        gaussian_pyramid: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Build Laplacian pyramid from Gaussian pyramid."""
        laplacian_pyramid = []

        for i in range(len(gaussian_pyramid) - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)

        # Last level is the smallest Gaussian image
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    def _reconstruct_from_pyramid(
        self,
        laplacian_pyramid: List[np.ndarray]
    ) -> np.ndarray:
        """Reconstruct image from Laplacian pyramid."""
        image = laplacian_pyramid[-1]

        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            image = cv2.pyrUp(image, dstsize=size)
            image = cv2.add(image, laplacian_pyramid[i])

        return image

    def stack_from_directory(
        self,
        directory: Path,
        pattern: str = "*.jpg",
        output_path: Optional[Path] = None,
        sort_key: Optional[callable] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Stack all images matching pattern in directory.

        Args:
            directory: Directory containing z-stack images
            pattern: Glob pattern for image files (e.g., "*.jpg", "img_*.png")
            output_path: Path to save result
            sort_key: Function to sort filenames. If None, uses alphabetical sort.
            verbose: Print progress

        Returns:
            Stacked image

        Example:
            >>> stacker = FocusStacker()
            >>> result = stacker.stack_from_directory(
            ...     Path('./z_stacks/sample_001'),
            ...     pattern='*.jpg',
            ...     output_path='stacked_001.jpg'
            ... )
        """
        directory = Path(directory)

        # Find all matching images
        image_paths = sorted(directory.glob(pattern), key=sort_key)

        if len(image_paths) == 0:
            raise ValueError(f"No images found matching pattern '{pattern}' in {directory}")

        if verbose:
            print(f"Found {len(image_paths)} images in {directory}")

        return self.stack_images(image_paths, output_path=output_path, verbose=verbose)

    @staticmethod
    def compare_with_single_image(
        stacked_image: np.ndarray,
        single_image_path: Path
    ) -> Dict:
        """
        Compare stacked image sharpness with a single image.

        Useful for validating that stacking improved image quality.

        Args:
            stacked_image: Result from stack_images()
            single_image_path: Path to a single image from the stack

        Returns:
            Dictionary with comparison metrics:
                - stacked_sharpness: Laplacian variance of stacked image
                - single_sharpness: Laplacian variance of single image
                - improvement_ratio: How much sharper the stacked image is
        """
        single_image = cv2.imread(str(single_image_path))

        stacked_gray = cv2.cvtColor(stacked_image, cv2.COLOR_BGR2GRAY)
        single_gray = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)

        stacked_lap = cv2.Laplacian(stacked_gray, cv2.CV_64F)
        single_lap = cv2.Laplacian(single_gray, cv2.CV_64F)

        stacked_sharpness = stacked_lap.var()
        single_sharpness = single_lap.var()

        improvement = stacked_sharpness / single_sharpness if single_sharpness > 0 else 0

        return {
            'stacked_sharpness': stacked_sharpness,
            'single_sharpness': single_sharpness,
            'improvement_ratio': improvement
        }
