"""
Initialize SQLite database for arthropod classification pipeline.

This script:
1. Creates database schema (all tables)
2. Loads taxonomic hierarchy from Catalogue of Life (COL)
3. Initializes size fraction definitions
4. Optionally loads example data

Usage:
    # Create empty database with schema only
    python scripts/01_setup_database.py

    # Create database and load example data
    python scripts/01_setup_database.py --example

    # Specify custom database path
    python scripts/01_setup_database.py --db-path data/my_database.db

    # Load taxonomy from custom JSON file
    python scripts/01_setup_database.py --taxonomy-file data/taxonomy/custom_taxa.json
"""

import argparse
import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import (
    init_database, get_session, Taxa, SizeFraction,
    Project, Location, SamplingRound, SampleAssignment,
    CompositeImage, SingleImage
)
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def setup_size_fractions(session):
    """
    Initialize size fraction definitions.

    Based on Methods section: Different sieve sizes for arthropod sorting.
    """
    logger.info("Creating size fraction definitions...")

    size_fractions = [
        {
            'code': 'k1',
            'min_size_mm': 0.0,
            'max_size_mm': 1.0,
            'description': 'Very small arthropods (mites, springtails)'
        },
        {
            'code': '1',
            'min_size_mm': 1.0,
            'max_size_mm': 2.0,
            'description': 'Small beetles and flies'
        },
        {
            'code': '1.5',
            'min_size_mm': 1.5,
            'max_size_mm': 2.0,
            'description': 'Intermediate fraction'
        },
        {
            'code': '2',
            'min_size_mm': 2.0,
            'max_size_mm': 3.0,
            'description': 'Medium-sized beetles'
        },
        {
            'code': '3',
            'min_size_mm': 3.0,
            'max_size_mm': 5.0,
            'description': 'Large beetles'
        },
        {
            'code': '5',
            'min_size_mm': 5.0,
            'max_size_mm': 7.0,
            'description': 'Very large beetles'
        },
        {
            'code': '7',
            'min_size_mm': 7.0,
            'max_size_mm': 10.0,
            'description': 'Ground beetles and large specimens'
        },
        {
            'code': 'A',
            'min_size_mm': 10.0,
            'max_size_mm': None,
            'description': 'Extra large specimens (>10mm)'
        }
    ]

    for sf_data in size_fractions:
        sf = SizeFraction(**sf_data)
        session.add(sf)

    session.commit()
    logger.info(f"✓ Created {len(size_fractions)} size fractions")


def load_taxonomy(session, taxonomy_file: Path):
    """
    Load taxonomic hierarchy from Catalogue of Life (COL) JSON.

    Expected JSON format:
    [
        {
            "taxon_id": "RT",
            "name": "Arthropoda",
            "rank": "phylum",
            "parent_id": null,
            "col_id": "...",
            "col_name": "..."
        },
        ...
    ]

    Args:
        session: SQLAlchemy session
        taxonomy_file: Path to JSON file with taxonomy data
    """
    logger.info(f"Loading taxonomy from {taxonomy_file}...")

    if not taxonomy_file.exists():
        logger.warning(f"Taxonomy file not found: {taxonomy_file}")
        logger.info("Creating minimal taxonomy (root taxon only)...")

        # Create minimal root taxon
        root_taxon = Taxa(
            taxon_id='RT',
            name='Arthropoda',
            rank='phylum',
            parent_id=None,
            col_id='Arthropoda',
            col_name='Arthropoda'
        )
        session.add(root_taxon)
        session.commit()
        logger.info("✓ Created root taxon: Arthropoda (ID: RT)")
        return 1

    # Load from JSON
    with open(taxonomy_file, 'r', encoding='utf-8') as f:
        taxa_data = json.load(f)

    logger.info(f"Found {len(taxa_data)} taxa in file")

    # Insert taxa
    count = 0
    for taxon_data in taxa_data:
        taxon = Taxa(
            taxon_id=taxon_data.get('taxon_id'),
            name=taxon_data.get('name'),
            rank=taxon_data.get('rank'),
            parent_id=taxon_data.get('parent_id'),
            col_id=taxon_data.get('col_id'),
            col_name=taxon_data.get('col_name')
        )
        session.add(taxon)
        count += 1

        # Commit in batches of 1000
        if count % 1000 == 0:
            session.commit()
            logger.info(f"  Loaded {count}/{len(taxa_data)} taxa...")

    session.commit()
    logger.info(f"✓ Loaded {count} taxa into database")

    return count


def load_example_data(session):
    """
    Load minimal example data for testing the pipeline.

    This creates:
    - 1 project
    - 2 locations
    - 3 sampling rounds
    - 5 sample assignments (trays)

    Note: Actual composite images and single images should be created
    separately using scripts/00_create_example_data.py
    """
    logger.info("Creating example data...")

    # Create project
    project = Project(
        name='Example Arthropod Study',
        description='Minimal example dataset for testing the classification pipeline'
    )
    session.add(project)
    session.flush()  # Get project ID

    # Create locations
    location1 = Location(
        project_id=project.id,
        name='Site A',
        latitude=50.9375,
        longitude=6.9603,
        description='Agricultural field with hedge'
    )
    location2 = Location(
        project_id=project.id,
        name='Site B',
        latitude=50.9400,
        longitude=6.9650,
        description='Control site without hedge'
    )
    session.add_all([location1, location2])
    session.flush()

    # Create sampling rounds
    rounds = []
    for year in [2020, 2021, 2022]:
        round_loc1 = SamplingRound(
            sampling_round_id=f'R{year}_A',
            location_id=location1.id,
            year=year,
            description=f'Sampling round {year} at Site A'
        )
        rounds.append(round_loc1)

    session.add_all(rounds)
    session.flush()

    # Create sample assignments (trays)
    assignments = []
    for i, round_obj in enumerate(rounds):
        for size_frac in ['1', '2', '7']:
            assignment = SampleAssignment(
                sample_id=f'S{i+1:03d}_{size_frac}mm',
                sampling_round_id=round_obj.id,
                size_fraction=size_frac,
                tray_size='large' if size_frac in ['7'] else 'small',
                weight=None  # Will be filled after weighing
            )
            assignments.append(assignment)

    session.add_all(assignments)
    session.commit()

    logger.info(f"✓ Created example data:")
    logger.info(f"  - 1 project")
    logger.info(f"  - 2 locations")
    logger.info(f"  - {len(rounds)} sampling rounds")
    logger.info(f"  - {len(assignments)} sample assignments")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Initialize SQLite database for arthropod classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--db-path',
        type=Path,
        default=Path('data/arthropod_pipeline.db'),
        help='Path to SQLite database file (default: data/arthropod_pipeline.db)'
    )

    parser.add_argument(
        '--taxonomy-file',
        type=Path,
        default=Path('data/taxonomy/col_arthropoda.json'),
        help='Path to JSON file with taxonomic hierarchy (default: data/taxonomy/col_arthropoda.json)'
    )

    parser.add_argument(
        '--example',
        action='store_true',
        help='Load example data (projects, locations, sampling rounds)'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing database if it exists'
    )

    args = parser.parse_args()

    # Check if database exists
    if args.db_path.exists() and not args.overwrite:
        logger.error(f"Database already exists: {args.db_path}")
        logger.error("Use --overwrite to replace it, or specify a different --db-path")
        return 1

    # Create parent directory
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize database
    logger.info("="*70)
    logger.info("INITIALIZING ARTHROPOD CLASSIFICATION DATABASE")
    logger.info("="*70)
    logger.info(f"Database: {args.db_path.absolute()}")
    logger.info(f"Taxonomy file: {args.taxonomy_file}")
    logger.info(f"Load example data: {args.example}")
    logger.info("")

    # Create database and schema
    logger.info("Step 1: Creating database schema...")
    db_url = f"sqlite:///{args.db_path}"
    engine = init_database(db_url)
    session = get_session(db_url)
    logger.info("✓ Database schema created")
    logger.info("")

    # Setup size fractions
    logger.info("Step 2: Creating size fraction definitions...")
    setup_size_fractions(session)
    logger.info("")

    # Load taxonomy
    logger.info("Step 3: Loading taxonomic hierarchy...")
    taxa_count = load_taxonomy(session, args.taxonomy_file)
    logger.info("")

    # Load example data if requested
    if args.example:
        logger.info("Step 4: Loading example data...")
        load_example_data(session)
        logger.info("")

    # Close session
    session.close()

    # Summary
    logger.info("="*70)
    logger.info("DATABASE INITIALIZATION COMPLETE")
    logger.info("="*70)
    logger.info(f"✓ Database created: {args.db_path.absolute()}")
    logger.info(f"✓ Size fractions: 8")
    logger.info(f"✓ Taxa loaded: {taxa_count}")
    if args.example:
        logger.info(f"✓ Example data loaded")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Run scripts/02_process_images.py to extract specimens")
    logger.info("  2. Manually classify specimens (or load pre-classified data)")
    logger.info("  3. Run scripts/03_train_models.py to train classification models")
    logger.info("")

    return 0


if __name__ == '__main__':
    sys.exit(main())
