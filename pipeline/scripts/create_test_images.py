#!/usr/bin/env python3
"""
Create synthetic test images for pipeline testing.

This script generates simple composite images with specimen-like shapes
for testing the detection, segmentation, and extraction pipeline.

Usage:
    python scripts/create_test_images.py
"""

from PIL import Image, ImageDraw
import random
from pathlib import Path


def create_composite_image(
    width: int,
    height: int,
    num_specimens: int,
    min_size: int,
    max_size: int,
    output_path: Path
):
    """
    Create a synthetic composite image with specimen-like shapes.

    Args:
        width: Image width in pixels
        height: Image height in pixels
        num_specimens: Number of specimen shapes to add
        min_size: Minimum specimen size
        max_size: Maximum specimen size
        output_path: Path to save image
    """
    # Create near-white background (RGB: 248, 248, 248) - matching actual composites
    img = Image.new('RGB', (width, height), color=(248, 248, 248))
    draw = ImageDraw.Draw(img)

    # Track positions to avoid too much overlap
    positions = []

    for i in range(num_specimens):
        # Find non-overlapping position
        attempts = 0
        while attempts < 50:
            x = random.randint(max_size, width - max_size)
            y = random.randint(max_size, height - max_size)
            size = random.randint(min_size, max_size)

            # Check overlap with existing specimens
            overlap = False
            for px, py, ps in positions:
                dist = ((x - px)**2 + (y - py)**2)**0.5
                if dist < (size + ps) * 1.2:  # Allow some overlap
                    overlap = True
                    break

            if not overlap:
                positions.append((x, y, size))
                break

            attempts += 1

        if attempts < 50:  # Successfully found position
            # Random dark color (specimens are usually dark)
            color = (
                random.randint(30, 100),
                random.randint(30, 100),
                random.randint(30, 100)
            )

            # Random shape type
            shape_type = random.choice(['circle', 'ellipse'])

            if shape_type == 'circle':
                # Draw circle
                bbox = [x - size, y - size, x + size, y + size]
                draw.ellipse(bbox, fill=color, outline=color)

            elif shape_type == 'ellipse':
                # Draw ellipse with random orientation
                width_factor = random.uniform(0.5, 1.5)
                bbox = [
                    x - size,
                    y - int(size * width_factor),
                    x + size,
                    y + int(size * width_factor)
                ]
                draw.ellipse(bbox, fill=color, outline=color)

    # Save image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    print(f"Created: {output_path} ({num_specimens} specimens)")


def main():
    """Create example composite images for different size fractions."""

    print("=" * 70)
    print("CREATING SYNTHETIC TEST IMAGES")
    print("=" * 70)

    examples_dir = Path(__file__).parent.parent / 'data' / 'examples'
    examples_dir.mkdir(parents=True, exist_ok=True)

    # Size fraction k1 (<1mm): Many small specimens
    print("\nCreating k1 fraction (many small specimens)...")
    create_composite_image(
        width=4000,
        height=3000,
        num_specimens=50,
        min_size=15,
        max_size=40,
        output_path=examples_dir / 'composite_S003_k1.png'
    )

    # Size fraction 1 (1-2mm): Medium number of medium specimens
    print("Creating 1mm fraction (medium specimens)...")
    create_composite_image(
        width=4000,
        height=3000,
        num_specimens=30,
        min_size=30,
        max_size=80,
        output_path=examples_dir / 'composite_S001_1mm.png'
    )

    # Size fraction 2 (2-3mm): Fewer larger specimens
    print("Creating 2mm fraction (larger specimens)...")
    create_composite_image(
        width=4000,
        height=3000,
        num_specimens=20,
        min_size=60,
        max_size=120,
        output_path=examples_dir / 'composite_S002_2mm.png'
    )

    print("\n" + "=" * 70)
    print("IMAGES CREATED SUCCESSFULLY")
    print("=" * 70)
    print(f"\nOutput directory: {examples_dir}")
    print("\nNext steps:")
    print("1. Install dependencies:")
    print("   pip install -r requirements.txt")
    print("\n2. Download YOLO models:")
    print("   python scripts/download_models.py")
    print("\n3. Test the pipeline:")
    print("   python scripts/02_process_images.py \\")
    print("       --metadata data/examples/metadata_example.csv \\")
    print("       --output-dir ./output/test_run")


if __name__ == '__main__':
    main()
