# Publication Data

This directory contains the data that `analysis/scripts/Data_Analysis.Rmd` reads to reproduce the results of the manuscript (version 1.1.0). The files are reduced to what the analysis needs: sites carry neutral codes, sampling dates are given as year and day of the year, and coordinates, object IDs and taxonomic ranks the analysis does not use are not included.

---

## Files Overview

| File | Sheet(s) | Rows | Description |
|------|----------|------|-------------|
| `specimen_data.xlsx` | `Object List` | 40,090 specimens | Order and family of each specimen with an order in the 54 valid samples |
| `biomass_data.xlsx` | `Sheet 1` | 371 size fractions | Biomass per sample and size fraction |
| `model_performance.xlsx` | `TestObjects Model Performance`, `TestObjects Taxon Performance` | 57, 126 | Test-set performance per classification model and per taxon |
| `climate_data.xlsx` | `Sheet 1` | 54 site-days | Daily precipitation and temperature per site and sampling day |

---

## Coding

- **Sites** (`location`): neutral codes `site_01`, `site_01 b`, `site_02` to `site_10` (eleven sampling locations). The codes are chosen so that sorting `site_date` (site code, `#`, date) in the analysis gives the same row order as before the recoding; the seeded permutation test of the indicator species analysis (Table S12) depends on that order.
- **Samples** (`zuordnung_id`): the sample numbers used in the manuscript, unchanged (sample 180 is `180`). The number links `specimen_data.xlsx` and `biomass_data.xlsx`.
- **Dates**: `year` and `day_of_year` (1 = 1 January). The analysis rebuilds the sampling date as `make_date(year) + day_of_year - 1` (applied by `analysis/reproduction/reproduction.patch`).

---

## File Descriptions

### specimen_data.xlsx

**Description:** Specimen classifications from the automated hierarchical classification pipeline, one row per specimen; only the 54 valid samples and only specimens with an order (the analysis drops the rest). Row order as in the full data file.

**Columns:**
- `zuordnung_id` - Sample number (links to `biomass_data.xlsx`)
- `location` - Site code (see Coding)
- `year` - Sampling year (2023 or 2024)
- `day_of_year` - Sampling day as day of the year
- `Groessenfraktion` - Size fraction
- `order` - Order-level classification
- `family` - Family-level classification (empty if the specimen was not classified to family)
- `Hecken` - Treatment: hedgerow present (`True`/`False`)
- `Wildblumensaat` - Wildflower sowing (`True`/`False`)
- `is_valid` - Sample valid (`True` in all rows)

---

### biomass_data.xlsx

**Description:** Biomass per sample and size fraction (size fractions with a weight only)

**Columns:**
- `zuordnung_id` - Sample number (links to `specimen_data.xlsx`)
- `size` - Size fraction
- `biomass_size` - Biomass of the size fraction (mass after drying for 6 h at 56 °C)

---

### model_performance.xlsx

**Description:** Classification performance on the test set, per classification model (sheet `TestObjects Model Performance`) and per taxon (sheet `TestObjects Taxon Performance`); unchanged from version 1.0.0

**Columns (both sheets, 15 each):** taxon or model ID, scientific name and rank, validation, training and test counts, average confidence, correct and incorrect classifications without and with the confidence threshold, `accuracy_without_threshold`, `threshold_for_accuracy`, `accuracy_with_threshold`, `objects_without_determination`

---

### climate_data.xlsx

**Description:** Daily climate values for each site and sampling day from the gridded data set GeoSphere Austria SPARTACUS v2.1 (1 km), as queried for the reference run of version 1.1.0. The analysis reads this file instead of querying the API (the query needs the site coordinates, which are not archived).

**Columns:**
- `location` - Site code
- `year`, `day_of_year` - Sampling day
- `RR` - Daily precipitation sum (mm)
- `TN` - Daily minimum air temperature (°C)
- `TX` - Daily maximum air temperature (°C)

---

## Data Provenance

**Study:** Hedgerow effects on arthropod biodiversity (manuscript title)

**Methods:**
- Image acquisition: High-resolution digital photography of arthropod specimens
- Detection: YOLOv8 object detection with adaptive tiling
- Segmentation: YOLOv8 instance segmentation
- Classification: YOLOv11 hierarchical classification following Catalogue of Life taxonomy
- Threshold calibration: 95% accuracy criterion (see Methods and THRESHOLD_CALIBRATION.md)

**Data Collection Period:** 2023-2024 (field sampling)

**Processing:** Automated pipeline (see main README.md)

---

## Data Quality

### Specimen Data
- **Validation:** All specimens manually verified during model training (10% test set)
- **Coverage:** All arthropod orders present in samples
- **Size Fractions:** 7 fractions from 1.5 mm to >10 mm

### Biomass Data
- **Measurement:** Laboratory-grade scales (±0.001 mg precision)
- **Quality Control:** Replicate measurements for samples >100 mg

### Model Performance
- **Test Set:** Independent 10% holdout (never seen during training)
- **Evaluation:** Per-taxon accuracy, confidence, and threshold calibration
- **Criteria:** ≥95% accuracy with calibrated confidence thresholds

---

## Citation

If you use this data, please cite:

> Meyer, P., Scharnhorst, V., Lechner, M., Haslinger, H., Gierus, M., & Meimberg, H. Integrated arthropod monitoring pipeline detects size-dependent responses. Manuscript under revision at Scientific Reports (2026).

**Data Repository:** Zenodo DOI (all versions): 10.5281/zenodo.17661709; version 1.0.0: 10.5281/zenodo.17661710

---

## License

This data is released under [CC-BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

You are free to:
- Share: Copy and redistribute the material
- Adapt: Remix, transform, and build upon the material

Under the following terms:
- Attribution: You must give appropriate credit and indicate if changes were made

---

## Contact

For questions about the data:

**Corresponding Author:** Philipp Meyer
**Email:** philipp.meyer@boku.ac.at
**Institution:** University of Natural Resources and Life Sciences, Vienna (BOKU)

---

## Related Files

- **Code Repository:** See parent directory for complete classification pipeline
- **Models:** Trained YOLO models available upon request (large files)
- **Documentation:**
  - `../../README.md` - Pipeline overview and usage
  - `../../THRESHOLD_CALIBRATION.md` - Threshold methodology

---

**Last Updated:** 2026-10-11
**Version:** 1.1.0
