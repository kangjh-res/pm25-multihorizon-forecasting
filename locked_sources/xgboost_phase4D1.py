# Extracted without changes from verified v16 embedded notebook source.
# ============================================================
# Phase 4D-1
# High-PM2.5 Episode Analysis
#
# LOCKED model:
#   Train = 2022 + 2023
#   Test  = 2024
#
# NO tuning / NO feature modification
#
# Models:
#   M0 Persistence
#   M1 Local
#   M2 Observed meteorology
#   M3 Forecast meteorology
#   M4 Transport
#   M5 Analogue
#   M5_CAMS
#
# Main high-PM2.5 definition:
#   Observed target PM2.5 >= 35 ug/m3
#
# Outputs:
#   1. Row-level predictions
#   2. High-PM2.5 performance
#   3. Incremental RMSE information gain
#   4. High vs non-high comparison
# ============================================================

from pathlib import Path
import json
import time
import warnings

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ------------------------------------------------------------
# 0. Paths
# ------------------------------------------------------------

ROOT = get_project_paths()["source_root"]

DATA = ROOT / "data" / "processed"
OUT  = PILOT_OUTPUT / "xgboost"

OUT.mkdir(parents=True, exist_ok=True)

FILE_2022 = DATA / "model_features_2022.parquet"
FILE_2023 = DATA / "model_features_2023.parquet"
FILE_2024 = DATA / "model_features_2024.parquet"

KEYS_2022 = DATA / "common_eval_keys_2022.parquet"
KEYS_2023 = DATA / "common_eval_keys_2023.parquet"
KEYS_2024 = DATA / "common_eval_keys_2024.parquet"

REGISTRY_FILE = PILOT_REGISTRY


# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------

HORIZONS = list(range(1, 25))

MODEL_NAMES = [
    "M1_local",
    "M2_obsmet",
    "M3_fcstmet",
    "M4_transport",
    "M5_analogue",
    "M5_CAMS",
]

TARGET = "PM25_target"
PERSISTENCE = "PM25_t"

HIGH_THRESHOLD = 35.0

ID_COLS = [
    "측정소코드",
    "issue_time_kst",
    "lead_h",
]

FINAL_XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.03,
    "max_depth": 5,
    "min_child_weight": 5,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 20.0,
    "tree_method": "hist",
    "device": "cuda",
    "random_state": 42,
}


print("=" * 90)
print("PHASE 4D-1 — HIGH-PM2.5 EPISODE ANALYSIS")
print("=" * 90)

print("Training        : 2022 + 2023")
print("Independent test: 2024")
print("High threshold  :", HIGH_THRESHOLD, "ug/m3")
print("Parameters      : LOCKED")
print("Device          :", FINAL_XGB_PARAMS["device"])


# ------------------------------------------------------------
# 2. Feature registry
# ------------------------------------------------------------

with open(
    safe_read_path(REGISTRY_FILE),
    "r",
    encoding="utf-8"
) as f:
    registry = json.load(f)


feature_sets = {
    name: registry["models"][name]
    for name in MODEL_NAMES
}

FULL_FEATURES = feature_sets["M5_CAMS"]


all_features = sorted(
    set(
        feature
        for model_name in MODEL_NAMES
        for feature in feature_sets[model_name]
    )
)


needed_cols = sorted(
    set(
        ID_COLS
        + [TARGET, PERSISTENCE]
        + all_features
    )
)


# ------------------------------------------------------------
# 3. Load data
# ------------------------------------------------------------

print("\nLoading data...")

d22 = pd.read_parquet(
    safe_read_path(FILE_2022),
    columns=needed_cols
)

d23 = pd.read_parquet(
    safe_read_path(FILE_2023),
    columns=needed_cols
)

d24 = pd.read_parquet(
    safe_read_path(FILE_2024),
    columns=needed_cols
)


def load_keys(path):
    return (
        pd.read_parquet(
            safe_read_path(path),
            columns=ID_COLS
        )
        .drop_duplicates()
    )


d22 = d22.merge(
    load_keys(KEYS_2022),
    on=ID_COLS,
    how="inner"
)

d23 = d23.merge(
    load_keys(KEYS_2023),
    on=ID_COLS,
    how="inner"
)

d24 = d24.merge(
    load_keys(KEYS_2024),
    on=ID_COLS,
    how="inner"
)


# ------------------------------------------------------------
# 4. Strict common population
# ------------------------------------------------------------

def strict_mask(df):

    return (
        df[FULL_FEATURES]
        .notna()
        .all(axis=1)
        &
        df[TARGET].notna()
        &
        df[PERSISTENCE].notna()
    )


d22 = d22.loc[
    strict_mask(d22)
].copy()

d23 = d23.loc[
    strict_mask(d23)
].copy()

d24 = d24.loc[
    strict_mask(d24)
].copy()


train = pd.concat(
    [d22, d23],
    ignore_index=True
)

test = d24.copy()


print("\nStrict population:")
print("Train:", train.shape)
print("Test :", test.shape)


# ------------------------------------------------------------
# 5. Prediction container
# ------------------------------------------------------------

prediction_frames = []

total_start = time.time()


# ------------------------------------------------------------
# 6. Train locked models and export predictions
# ------------------------------------------------------------

for h in HORIZONS:

    print("\n" + "=" * 70)
    print(f"LEAD {h:2d} h")
    print("=" * 70)


    tr_h = (
        train[
            train["lead_h"] == h
        ]
        .copy()
    )

    te_h = (
        test[
            test["lead_h"] == h
        ]
        .copy()
    )


    out_h = te_h[
        ID_COLS
        + [TARGET, PERSISTENCE]
    ].copy()


    # Persistence prediction
    out_h["pred_M0_persistence"] = (
        te_h[PERSISTENCE]
        .to_numpy(dtype=np.float32)
    )


    # Locked ML models
    for model_name in MODEL_NAMES:

        features = feature_sets[
            model_name
        ]

        X_train = (
            tr_h[features]
            .astype(np.float32)
        )

        y_train = (
            tr_h[TARGET]
            .astype(np.float32)
        )

        X_test = (
            te_h[features]
            .astype(np.float32)
        )


        model = xgb.XGBRegressor(
            **FINAL_XGB_PARAMS
        )


        t0 = time.time()


        model.fit(
            X_train,
            y_train,
            verbose=False
        )


        booster = model.get_booster()

        dtest = xgb.DMatrix(
            X_test,
            feature_names=features
        )

        pred = booster.predict(
            dtest
        )


        out_h[
            f"pred_{model_name}"
        ] = pred


        print(
            f"{model_name:15s} "
            f"{time.time()-t0:.1f}s"
        )


    prediction_frames.append(
        out_h
    )


predictions = pd.concat(
    prediction_frames,
    ignore_index=True
)


# ------------------------------------------------------------
# 7. Define concentration regimes
# ------------------------------------------------------------

predictions["PM25_regime"] = np.where(
    predictions[TARGET]
    >= HIGH_THRESHOLD,
    "high",
    "non_high"
)


print("\n2024 evaluation population:")

print(
    predictions[
        "PM25_regime"
    ].value_counts()
)

print("\nHigh-PM2.5 fraction:")

print(
    (
        predictions["PM25_regime"]
        == "high"
    ).mean()
)


# ------------------------------------------------------------
# 8. Metrics
# ------------------------------------------------------------

def calc_metrics(
    y,
    pred
):

    if len(y) < 2:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "Bias": np.nan,
            "R2": np.nan,
        }

    return {
        "RMSE":
            np.sqrt(
                mean_squared_error(
                    y,
                    pred
                )
            ),

        "MAE":
            mean_absolute_error(
                y,
                pred
            ),

        "Bias":
            np.mean(
                pred - y
            ),

        "R2":
            r2_score(
                y,
                pred
            ),
    }


PRED_COLS = {
    "M0_persistence":
        "pred_M0_persistence",

    "M1_local":
        "pred_M1_local",

    "M2_obsmet":
        "pred_M2_obsmet",

    "M3_fcstmet":
        "pred_M3_fcstmet",

    "M4_transport":
        "pred_M4_transport",

    "M5_analogue":
        "pred_M5_analogue",

    "M5_CAMS":
        "pred_M5_CAMS",
}


# ------------------------------------------------------------
# 9. Performance by horizon and PM regime
# ------------------------------------------------------------

metric_rows = []


for regime in [
    "high",
    "non_high",
]:

    for h in HORIZONS:

        tmp = predictions[
            (
                predictions["lead_h"]
                == h
            )
            &
            (
                predictions["PM25_regime"]
                == regime
            )
        ]


        y = tmp[
            TARGET
        ].to_numpy()


        for model_name, pred_col in PRED_COLS.items():

            pred = tmp[
                pred_col
            ].to_numpy()


            m = calc_metrics(
                y,
                pred
            )


            metric_rows.append({
                "regime":
                    regime,

                "lead_h":
                    h,

                "model":
                    model_name,

                "N":
                    len(tmp),

                **m,
            })


metrics = pd.DataFrame(
    metric_rows
)


# ------------------------------------------------------------
# 10. Incremental RMSE effects
# ------------------------------------------------------------

MODEL_SEQUENCE = [
    "M1_local",
    "M2_obsmet",
    "M3_fcstmet",
    "M4_transport",
    "M5_analogue",
    "M5_CAMS",
]


BLOCK_NAMES = {
    "M2_obsmet":
        "Observed_met_pct",

    "M3_fcstmet":
        "Forecast_met_pct",

    "M4_transport":
        "Transport_pct",

    "M5_analogue":
        "Analogue_pct",

    "M5_CAMS":
        "CAMS_PM25_pct",
}


increment_rows = []


for regime in [
    "high",
    "non_high",
]:

    for h in HORIZONS:

        temp = (
            metrics[
                (metrics["regime"] == regime)
                &
                (metrics["lead_h"] == h)
            ]
            .set_index("model")
        )


        for i in range(
            1,
            len(MODEL_SEQUENCE)
        ):

            prev_model = MODEL_SEQUENCE[
                i - 1
            ]

            curr_model = MODEL_SEQUENCE[
                i
            ]


            r0 = temp.loc[
                prev_model,
                "RMSE"
            ]

            r1 = temp.loc[
                curr_model,
                "RMSE"
            ]


            change = (
                (r1 - r0)
                / r0
                * 100
            )


            increment_rows.append({
                "regime":
                    regime,

                "lead_h":
                    h,

                "block":
                    BLOCK_NAMES[
                        curr_model
                    ],

                "RMSE_change_pct":
                    change,
            })


incremental = pd.DataFrame(
    increment_rows
)


# ------------------------------------------------------------
# 11. Wide table
# ------------------------------------------------------------

wide = (
    incremental
    .pivot_table(
        index=[
            "regime",
            "lead_h"
        ],
        columns="block",
        values="RMSE_change_pct"
    )
    .reset_index()
)


wide.columns.name = None


# ------------------------------------------------------------
# 12. Broad forecast-horizon regimes
# ------------------------------------------------------------

def horizon_group(h):

    if h <= 3:
        return "short_1_3h"

    elif h <= 12:
        return "medium_4_12h"

    else:
        return "long_13_24h"


wide["horizon_group"] = (
    wide["lead_h"]
    .apply(horizon_group)
)


block_cols = [
    "Observed_met_pct",
    "Forecast_met_pct",
    "Transport_pct",
    "Analogue_pct",
    "CAMS_PM25_pct",
]


regime_summary = (
    wide
    .groupby(
        [
            "regime",
            "horizon_group"
        ],
        as_index=False
    )[block_cols]
    .mean()
)


# ------------------------------------------------------------
# 13. Important horizons
# ------------------------------------------------------------

IMPORTANT = [
    1,
    3,
    6,
    12,
    24,
]


print("\n\n" + "=" * 100)
print("HIGH PM2.5 — IMPORTANT HORIZONS")
print("Negative = RMSE improvement")
print("=" * 100)


high_important = wide[
    (wide["regime"] == "high")
    &
    (wide["lead_h"].isin(IMPORTANT))
]


print(
    high_important[
        [
            "lead_h"
        ]
        + block_cols
    ]
    .round(3)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 14. High vs non-high regime summary
# ------------------------------------------------------------

print("\n\n" + "=" * 100)
print("HIGH vs NON-HIGH PM2.5 — INFORMATION GAIN")
print("=" * 100)


print(
    regime_summary
    .round(3)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 15. Final model high-PM performance
# ------------------------------------------------------------

print("\n\n" + "=" * 100)
print("M5_CAMS PERFORMANCE DURING HIGH PM2.5")
print("=" * 100)


final_high = metrics[
    (metrics["regime"] == "high")
    &
    (metrics["model"] == "M5_CAMS")
    &
    (metrics["lead_h"].isin(IMPORTANT))
][
    [
        "lead_h",
        "N",
        "RMSE",
        "MAE",
        "Bias",
        "R2",
    ]
]


print(
    final_high
    .round(3)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 16. High-event detection metrics
#
# Treat continuous prediction >=35 as predicted high event.
# ------------------------------------------------------------

event_rows = []


for h in HORIZONS:

    tmp = predictions[
        predictions["lead_h"] == h
    ]


    actual = (
        tmp[TARGET]
        >= HIGH_THRESHOLD
    ).to_numpy()


    for model_name, pred_col in PRED_COLS.items():

        predicted = (
            tmp[pred_col]
            >= HIGH_THRESHOLD
        ).to_numpy()


        TP = np.sum(
            actual & predicted
        )

        FP = np.sum(
            (~actual) & predicted
        )

        FN = np.sum(
            actual & (~predicted)
        )

        TN = np.sum(
            (~actual) & (~predicted)
        )


        precision = (
            TP / (TP + FP)
            if (TP + FP) > 0
            else np.nan
        )

        recall = (
            TP / (TP + FN)
            if (TP + FN) > 0
            else np.nan
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if (
                np.isfinite(precision)
                and np.isfinite(recall)
                and
                (precision + recall) > 0
            )
            else np.nan
        )

        far = (
            FP / (TP + FP)
            if (TP + FP) > 0
            else np.nan
        )


        event_rows.append({
            "lead_h": h,
            "model": model_name,
            "N": len(tmp),
            "event_N": int(actual.sum()),
            "TP": int(TP),
            "FP": int(FP),
            "FN": int(FN),
            "TN": int(TN),
            "Precision": precision,
            "Recall_POD": recall,
            "F1": f1,
            "FAR": far,
        })


events = pd.DataFrame(
    event_rows
)


print("\n\n" + "=" * 100)
print("M5_CAMS HIGH-PM2.5 EVENT DETECTION")
print("=" * 100)


print(
    events[
        (events["model"] == "M5_CAMS")
        &
        (events["lead_h"].isin(IMPORTANT))
    ]
    .round(3)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 17. Save
# ------------------------------------------------------------

PRED_FILE = (
    OUT /
    "phase4D1_2024_row_predictions.parquet"
)

METRIC_FILE = (
    OUT /
    "phase4D1_PM25_regime_performance.csv"
)

INCREMENT_FILE = (
    OUT /
    "phase4D1_PM25_regime_incremental_effect.csv"
)

SUMMARY_FILE = (
    OUT /
    "phase4D1_PM25_regime_information_gain.csv"
)

EVENT_FILE = (
    OUT /
    "phase4D1_high_PM25_event_detection.csv"
)


predictions.to_parquet(
    safe_write_path(PRED_FILE),
    index=False
)

metrics.to_csv(
    safe_write_path(METRIC_FILE),
    index=False,
    encoding="utf-8-sig"
)

incremental.to_csv(
    safe_write_path(INCREMENT_FILE),
    index=False,
    encoding="utf-8-sig"
)

regime_summary.to_csv(
    safe_write_path(SUMMARY_FILE),
    index=False,
    encoding="utf-8-sig"
)

events.to_csv(
    safe_write_path(EVENT_FILE),
    index=False,
    encoding="utf-8-sig"
)


print("\nSaved:")
print(PRED_FILE)
print(METRIC_FILE)
print(INCREMENT_FILE)
print(SUMMARY_FILE)
print(EVENT_FILE)

print(
    f"\nTotal elapsed: "
    f"{(time.time()-total_start)/60:.2f} min"
)

print("\nPHASE 4D-1 COMPLETED.")
