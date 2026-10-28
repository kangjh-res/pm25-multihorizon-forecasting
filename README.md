# PM2.5 Multi-Horizon Forecasting Using XGBoost and MLP

Research code accompanying the manuscript *Horizon-dependent atmospheric information and rapid-transition limitations in machine-learning PM2.5 forecasting*.

This repository contains the archived model-training sources, evaluation utilities, regression tests, and sensitivity-analysis scripts used to investigate 1–24 h PM2.5 forecasting.

## Repository Structure

- `locked_sources/` — Archived XGBoost and MLP training scripts.
- `tools/` — Model execution, evaluation, and validation utilities.
- `references/` — Reference metrics for selected Supporting Information tables.
- `sensitivity/v66/` — Retrospective CAMS forecast-cycle and spatial interpolation sensitivity analysis.
- `sensitivity/v67/` — Paired bootstrap uncertainty analysis.
- `tests/` — Automated regression tests.
- `docs/VALIDATION_REPORT.md` — Archived-results validation report.

## Software Environment

The code was evaluated on Windows using Python 3.13.9.

Package versions are documented in `requirements-locked.txt`, including XGBoost 3.2.0, PyTorch 2.14.0+cu126, scikit-learn 1.7.2, and SHAP 0.52.0.

Install the required packages using:

```bash
pip install -r requirements-locked.txt
```

Installation and execution on a clean environment have not been independently verified.

## Model Execution

### Prepared-feature preflight

```bash
python tools/run_locked.py --data-root /path/to/source_project --registry /path/to/final_feature_registry.json
```

The workflow requires prepared feature datasets for 2022–2024, corresponding evaluation keys, and the final feature registry.

### Model training

```bash
python tools/run_locked.py --data-root /path/to/source_project --registry /path/to/final_feature_registry.json --output-root /path/to/output --mode train --family both
```

### CAMS bootstrap sensitivity analysis

```bash
python sensitivity/v67/run_v67_bootstrap.py --v66-dir /path/to/v66_results --out-dir /path/to/output --bootstrap 2000
```

The bootstrap analysis requires the corresponding v66 prediction outputs.

## Validation

The following checks were completed in October 2026:

- Python source syntax checks: passed.
- Automated regression tests: 11/11 passed.
- Archived-results validation: 6/6 passed.
- Retrospective CAMS sensitivity results (v66): checked against archived results.
- Paired bootstrap uncertainty results (v67): checked against archived results.

To execute the automated tests:

```bash
python -m pytest tests -q
```

The archived-results validation uses previously generated predictions and prepared features. It does not establish complete raw-data-to-model reproducibility.

## Data Availability and Limitations

Raw air-quality observations, CAMS GRIB archives, and large prepared datasets are not distributed with this repository.

The CAMS forecast-cycle sensitivity analysis is retrospective. Operational product availability at the forecast issuance time and the original current-cycle product provenance have not been independently established.

Full reproduction requires access to the underlying datasets and the corresponding preprocessing workflow.

## Citation

Please cite the associated manuscript when using this research code. The final publication citation and archived software DOI will be added when available.

## Copyright and Licensing

Copyright and reuse permissions are subject to the applicable institutional requirements. No open-source license has been assigned to this repository.
