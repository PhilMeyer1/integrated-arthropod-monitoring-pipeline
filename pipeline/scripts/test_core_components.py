"""
Quick test of core pipeline components with mock data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.utils import get_data_manager
from src.classification.taxonomy import TaxonomyHierarchy
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def test_database_connection():
    """Test 1: Can we connect to SQLite?"""
    logger.info("="*70)
    logger.info("TEST 1: Database Connection")
    logger.info("="*70)

    try:
        db_path = Path('data/arthropod_pipeline.db')
        db_url = f"sqlite:///{db_path}"

        data_manager = get_data_manager(use_database=True)
        data_manager.db_url = db_url
        data_manager.__init__(use_database=True, database_url=db_url)

        logger.info("✓ Database connection successful")
        return True, data_manager
    except Exception as e:
        logger.error(f"✗ Database connection failed: {e}")
        return False, None


def test_get_specimens(data_manager):
    """Test 2: Can we query specimens?"""
    logger.info("")
    logger.info("="*70)
    logger.info("TEST 2: Query Specimens")
    logger.info("="*70)

    try:
        # Get all specimens
        all_specimens = data_manager.get_specimens()
        logger.info(f"✓ Found {len(all_specimens)} total specimens")

        # Get specimens by taxon
        coleoptera = data_manager.get_specimens(taxon_id='COLEOPTERA')
        logger.info(f"✓ Found {len(coleoptera)} Coleoptera specimens")

        return True
    except Exception as e:
        logger.error(f"✗ Query failed: {e}")
        logger.exception(e)
        return False


def test_taxonomy_hierarchy(data_manager):
    """Test 3: Can we build taxonomy hierarchy?"""
    logger.info("")
    logger.info("="*70)
    logger.info("TEST 3: Taxonomy Hierarchy")
    logger.info("="*70)

    try:
        taxonomy = TaxonomyHierarchy(data_manager=data_manager)
        logger.info("✓ TaxonomyHierarchy initialized")

        # Try to get relevant taxa
        relevant_taxa = taxonomy.get_relevant_taxa()
        logger.info(f"✓ Found {len(relevant_taxa)} relevant taxa")

        if relevant_taxa:
            # Build hierarchy
            hierarchy = taxonomy.build_hierarchy(relevant_taxa)
            logger.info(f"✓ Built hierarchy with {len(hierarchy)} taxa")

            # Show structure
            logger.info("\nHierarchy structure:")
            for taxon_id, info in list(hierarchy.items())[:5]:
                logger.info(f"  {taxon_id}: {len(info['children'])} children, {len(info['images'])} images")

        return True
    except Exception as e:
        logger.error(f"✗ Taxonomy hierarchy failed: {e}")
        logger.exception(e)
        return False


def test_get_taxon_info(data_manager):
    """Test 4: Can we get taxon metadata?"""
    logger.info("")
    logger.info("="*70)
    logger.info("TEST 4: Taxon Metadata")
    logger.info("="*70)

    try:
        # Test getting parent
        parent = data_manager.get_parent_taxon('COLEOPTERA')
        logger.info(f"✓ COLEOPTERA parent: {parent}")

        # Test getting taxon info
        taxon = data_manager.get_taxon('COLEOPTERA')
        if taxon:
            logger.info(f"✓ COLEOPTERA info: {taxon.taxon_id} - {taxon.name}")
        else:
            logger.warning("Could not get taxon info")

        return True
    except Exception as e:
        logger.error(f"✗ Taxon metadata failed: {e}")
        logger.exception(e)
        return False


def test_model_operations(data_manager):
    """Test 5: Can we save/load model metadata?"""
    logger.info("")
    logger.info("="*70)
    logger.info("TEST 5: Model Operations")
    logger.info("="*70)

    try:
        # Test save model
        model_id = data_manager.save_model(
            taxon_id='INSECTA',
            set_number=1,
            train_count=100,
            val_count=25
        )
        logger.info(f"✓ Saved model with ID: {model_id}")

        # Test update path
        data_manager.update_model_path(model_id, '/fake/path/model.pt')
        logger.info("✓ Updated model path")

        # Test get models by set
        models = data_manager.get_models_by_set(1)
        logger.info(f"✓ Found {len(models)} models for set 1")

        # Test max set number
        max_set = data_manager.get_max_set_number()
        logger.info(f"✓ Max set number: {max_set}")

        return True
    except Exception as e:
        logger.error(f"✗ Model operations failed: {e}")
        logger.exception(e)
        return False


def main():
    logger.info("\n" + "="*70)
    logger.info("CORE COMPONENTS TEST SUITE")
    logger.info("="*70)
    logger.info("")

    results = {}

    # Test 1: Database connection
    success, data_manager = test_database_connection()
    results['database_connection'] = success

    if not success:
        logger.error("\n✗ Cannot proceed without database connection")
        return

    # Test 2: Query specimens
    results['query_specimens'] = test_get_specimens(data_manager)

    # Test 3: Taxonomy hierarchy
    results['taxonomy_hierarchy'] = test_taxonomy_hierarchy(data_manager)

    # Test 4: Taxon metadata
    results['taxon_metadata'] = test_get_taxon_info(data_manager)

    # Test 5: Model operations
    results['model_operations'] = test_model_operations(data_manager)

    # Summary
    logger.info("")
    logger.info("="*70)
    logger.info("TEST SUMMARY")
    logger.info("="*70)

    total = len(results)
    passed = sum(1 for v in results.values() if v)

    for test_name, passed_test in results.items():
        status = "✓ PASS" if passed_test else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("")
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("="*70)

    # Close database
    data_manager.close()

    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)


if __name__ == '__main__':
    main()
