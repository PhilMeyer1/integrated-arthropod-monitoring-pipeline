# Database Module

This module provides both database and file-based data storage options.

## Usage

### File-Based Mode (Default - Recommended for Reproducibility)

```python
from src.database.utils import get_data_manager

# Automatically uses file-based storage (default in config)
dm = get_data_manager()

# Save sample metadata
dm.save_sample_metadata([{
    'sample_id': 'S001',
    'location': 'Site_A',
    'size_fraction': '1',
    'tray_size': 'large'
}])

# Get image paths
images = dm.get_image_paths(sample_id='S001')
```

### Database Mode

Edit `config/default_config.yaml`:

```yaml
database:
  use_database: true
  url: "postgresql://user:pass@localhost:5432/arthropod_db"
```

Then use the same API:

```python
dm = get_data_manager()
# Uses database automatically
```

## Models

All database models are defined in `models.py` with English names.

### Translation from German

| German (Old) | English (New) |
|--------------|---------------|
| Zuordnung | SampleAssignment |
| Sammeldurchgang | SamplingRound |
| Standort | Location |
| Groessenfraktion | size_fraction |
| stitchedImage | CompositeImage |
| singleImage | SingleImage |

See `GLOSSARY.md` for complete translation table.

## File Structure (File-Based Mode)

```
data/
├── metadata.csv              # Sample metadata
├── composite_images.csv      # Composite image info
├── single_images.csv         # Extracted specimens
├── classifications.csv       # Classification results
└── models/
    └── model_registry.json   # Trained models
```

## Database Schema (Database Mode)

See `models.py` for full schema with relationships.

Key tables:
- `projects` - Research projects
- `locations` - Sampling sites
- `sampling_rounds` - Collection events
- `sample_assignments` - Trays/samples
- `composite_images` - Stitched images
- `single_images` - Extracted specimens
- `taxa` - Taxonomic hierarchy
- `models` - Trained models
- `inference_results` - Classifications
