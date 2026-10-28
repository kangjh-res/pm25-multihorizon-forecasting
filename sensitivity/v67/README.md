# v67 paired bootstrap of v66 final CAMS predictions

Run `run_v67_bootstrap.py --v66-dir PATH --out-dir PATH [--bootstrap 2000]`.

Requires `predictions_h01.parquet` through `predictions_h24.parquet` in the final v66 output folder. Reads existing predictions; does not extract GRIB or retrain models. Compares M5, current-cycle CAMS, prior-cycle Delaunay, and prior-cycle bilinear with paired issue-time and KST calendar-day cluster bootstraps. Checks that the input reproduces the known v66 pooled RMSE and 2,117,887 matched evaluation rows before proceeding. Results: `v67_report.json`, `v67_bootstrap_95CI.csv`, `v67_monthly_audit.csv`.
