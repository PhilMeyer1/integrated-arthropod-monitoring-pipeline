"""
Insert mock specimen data for testing the pipeline.

This creates fake specimen images and classifications to test
the training and inference pipeline without real data.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import (
    get_session, SingleImage, CompositeImage,
    SampleAssignment, SamplingRound, Location, Project
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def create_mock_image(path: Path, size=(512, 512)):
    """Create a simple mock image."""
    # Create random noise image
    arr = np.random.randint(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(path)
    logger.info(f"Created mock image: {path}")


def insert_mock_data():
    """Insert mock specimens into database."""

    db_path = Path('data/arthropod_pipeline.db')
    db_url = f"sqlite:///{db_path}"

    logger.info("="*70)
    logger.info("INSERTING MOCK TEST DATA")
    logger.info("="*70)
    logger.info(f"Database: {db_path}")
    logger.info("")

    session = get_session(db_url)

    # Create directories
    mock_images_dir = Path('data/test_images')
    mock_images_dir.mkdir(parents=True, exist_ok=True)

    # Get first project and location
    project = session.query(Project).first()
    if not project:
        logger.error("No project found. Run scripts/01_setup_database.py --example first")
        return False

    location = session.query(Location).first()
    if not location:
        logger.error("No location found")
        return False

    # Get first sampling round
    sampling_round = session.query(SamplingRound).first()
    if not sampling_round:
        logger.error("No sampling round found")
        return False

    # Get first sample assignment
    sample = session.query(SampleAssignment).first()
    if not sample:
        logger.error("No sample assignment found")
        return False

    logger.info(f"Using sample: {sample.sample_id}")

    # Create a composite image entry
    composite_path = mock_images_dir / "composite_001.png"
    create_mock_image(composite_path, size=(2000, 2000))

    composite = CompositeImage(
        sample_assignment_id=sample.id,
        file_path=str(composite_path),
        width=2000,
        height=2000,
        megapixels=4.0,
        processed=True
    )
    session.add(composite)
    session.flush()

    logger.info(f"Created composite image: {composite.id}")
    logger.info("")

    # Create mock single images for different taxa
    # We'll create specimens for: Coleoptera, Diptera, Hymenoptera, Hemiptera
    taxa_specimens = {
        'COLEOPTERA': 15,  # Beetles
        'DIPTERA': 12,     # Flies
        'HYMENOPTERA': 10, # Wasps/Ants
        'HEMIPTERA': 8     # True bugs
    }

    total_created = 0

    for taxon_id, count in taxa_specimens.items():
        logger.info(f"Creating {count} specimens for {taxon_id}...")

        for i in range(count):
            # Create mock image
            img_path = mock_images_dir / f"specimen_{taxon_id}_{i+1:03d}.png"
            create_mock_image(img_path)

            # Create database entry
            specimen = SingleImage(
                composite_image_id=composite.id,
                sample_assignment_id=sample.id,
                work_images_path=str(img_path),
                original_image_path=str(img_path),
                bbox_x1=100.0 + i*10,
                bbox_y1=100.0 + i*10,
                bbox_x2=600.0 + i*10,
                bbox_y2=600.0 + i*10,
                detection_confidence=0.95,
                manual_determination=taxon_id,  # This is the taxon ID from Taxa table
                is_debris=False,
                exclude_from_training=False
            )
            session.add(specimen)
            total_created += 1

        logger.info(f"  Created {count} specimens")

    # Add a few debris specimens
    logger.info("Creating 5 debris specimens...")
    for i in range(5):
        img_path = mock_images_dir / f"specimen_DEBRIS_{i+1:03d}.png"
        create_mock_image(img_path)

        specimen = SingleImage(
            composite_image_id=composite.id,
            sample_assignment_id=sample.id,
            work_images_path=str(img_path),
            original_image_path=str(img_path),
            bbox_x1=700.0 + i*10,
            bbox_y1=700.0 + i*10,
            bbox_x2=1200.0 + i*10,
            bbox_y2=1200.0 + i*10,
            detection_confidence=0.85,
            manual_determination=None,
            is_debris=True,
            exclude_from_training=False
        )
        session.add(specimen)
        total_created += 1

    logger.info(f"  Created 5 debris specimens")
    logger.info("")

    # Commit all changes
    session.commit()
    session.close()

    logger.info("="*70)
    logger.info("MOCK DATA INSERTION COMPLETE")
    logger.info("="*70)
    logger.info(f"Total specimens created: {total_created}")
    logger.info(f"Images directory: {mock_images_dir}")
    logger.info("")
    logger.info("Breakdown:")
    for taxon, count in taxa_specimens.items():
        logger.info(f"  {taxon}: {count} specimens")
    logger.info(f"  DEBRIS: 5 specimens")
    logger.info("="*70)
    logger.info("")

    return True


if __name__ == '__main__':
    success = insert_mock_data()
    sys.exit(0 if success else 1)
