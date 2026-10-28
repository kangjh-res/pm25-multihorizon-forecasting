# PM2.5 1–24 h forecasting — v69 publication staging candidate

**NOT APPROVED FOR PUBLIC RELEASE.** This is a curated, static-checked code candidate derived from v68. It is not an independently reproduced end-to-end workflow.

## Contents
- `locked_sources/`: archived locked XGBoost and MLP training sources.
- `tools/run_locked.py`: portable prepared-feature preflight and model-training launcher.
- `tools/`: selected evaluation and manuscript validation utilities.
- `sensitivity/v66/`: retrospective prior-cycle CAMS reconstruction and retraining; requires original inputs and GRIB archive.
- `sensitivity/v67/`: paired bootstrap from final v66 prediction parquet files.
- `tests/`: source-level regression checks, not a complete clean-room test suite.

## Prepared-feature preflight (does not retrain)
```bash
python tools/run_locked.py --data-root /path/to/source_project --registry /path/to/final_feature_registry.json
```
Expected prepared inputs: `data/processed/model_features_{2022,2023,2024}.parquet`, `common_eval_keys_{2022,2023,2024}.parquet`, and the final feature registry.

## Training (only after preflight and environment validation)
```bash
python tools/run_locked.py --data-root /path/to/source_project --registry /path/to/final_feature_registry.json --output-root /path/to/separate/output --mode train --family both
```

## v67 uncertainty analysis (requires v66 predictions)
```bash
python sensitivity/v67/run_v67_bootstrap.py --v66-dir /path/to/v66_results --out-dir /path/to/new_v67_results --bootstrap 2000
```

## Release limitations
1. Data acquisition and complete raw-to-feature processing are not yet independently reproducible from this curated subset.
2. Exact Python/CUDA dependency versions must be exported from the validated Windows environment; `requirements.txt` is deliberately **not pinned**.
3. CAMS real-time dissemination availability and current-cycle product lineage remain unverified; prior-cycle results are retrospective.
4. Research notebooks and legacy exploratory scripts are intentionally excluded from this public candidate, but retained in private v68 provenance.
5. Check all data-provider licensing and obtain institutional approval for code ownership/license before GitHub publication. **No LICENSE file is assigned without authorization.**
6. Run a clean-environment execution test and compare 24-horizon outputs against archived reference metrics.

See `RELEASE_GATE.md` and `audit_release.py`.


## Validation status (October 2026)

- Python source syntax checks: passed.
- Automated regression tests: 11/11 passed.
- Archived-results validation: 6/6 passed.
- CAMS prior-cycle sensitivity (v66): archived results checked.
- Paired bootstrap uncertainty (v67): archived results checked.
- Evaluation: 2024, 133 monitoring stations, 1–24 h horizons.
- Environment: Python 3.13.9 on Windows.
- Exact direct dependency versions: `requirements-locked.txt`.

The archived-results validation uses saved predictions and prepared
features. It does not establish full raw-data-to-model reproducibility.
CAMS operational dissemination timing and original product provenance
have not been independently verified.

## Publication and licensing

This repository is prepared for scientific reproducibility.
Public release requires institutional authorization and verification
of applicable data-provider terms. No open-source license is granted
until authorized.
