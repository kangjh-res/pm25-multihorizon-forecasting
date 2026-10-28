# Extracted without changes from verified v16 embedded notebook source.
# ============================================================
# PHASE 5C
# LOCKED 2024 INDEPENDENT TEST
#
# Models
#   ANN_M4      = M4_transport (78 features)
#   KNN_ANN_M5  = M5_analogue  (85 features)
#
# FIXED before opening 2024:
#   Architecture : 256 -> 128 -> 64 -> 1
#   Activation   : ReLU
#   Dropout      : 0.1
#   Optimizer    : Adam
#   LR           : 3e-4
#   Loss         : MSE
#   Batch size   : 1024
#   Max epochs   : 200
#   Patience     : 15
#   Seed         : 42
#
# Train      : 2022
# Validation : 2023 (early stopping only)
# Test       : 2024 (evaluation only)
#
# IMPORTANT:
#   - No tuning using 2024
#   - Imputer fitted ONLY on 2022
#   - Scaler fitted ONLY on 2022
#   - Same target-valid test rows for M4 and M5
# ============================================================

from pathlib import Path
import json
import random
import time
import copy
import gc

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 0. Configuration
# ============================================================

ROOT = get_project_paths()["source_root"]

DATA_DIR = ROOT / "data" / "processed"
OUT_DIR = PILOT_OUTPUT / "mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = {
    2022: DATA_DIR / "model_features_2022.parquet",
    2023: DATA_DIR / "model_features_2023.parquet",
    2024: DATA_DIR / "model_features_2024.parquet",
}

REGISTRY_FILE = get_project_paths()["registry_file"]

HORIZONS = list(range(1, 25))

SEED = 42

DROPOUT = 0.1
LEARNING_RATE = 3e-4

BATCH_SIZE = 1024
PRED_BATCH_SIZE = 4096

MAX_EPOCHS = 200
PATIENCE = 15

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 88)
print("PHASE 5C — LOCKED 2024 INDEPENDENT TEST")
print("=" * 88)
print("Device    :", device)

if device.type == "cuda":
    print("GPU       :", torch.cuda.get_device_name(0))

print("Seed      :", SEED)
print("Horizons  :", HORIZONS)
print("2024 use  : TEST ONLY")


# ============================================================
# 1. Registry
# ============================================================

with open(safe_read_path(REGISTRY_FILE), "r", encoding="utf-8") as f:
    registry = json.load(f)

FEATURE_SETS = {
    "ANN_M4": registry["models"]["M4_transport"],
    "KNN_ANN_M5": registry["models"]["M5_analogue"],
}

assert len(FEATURE_SETS["ANN_M4"]) == 78
assert len(FEATURE_SETS["KNN_ANN_M5"]) == 85

print("\nFeature counts:")
for name, features in FEATURE_SETS.items():
    print(f"  {name}: {len(features)}")


# ============================================================
# 2. Reproducibility
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 3. Fixed A6 network
# ============================================================

class PM25MLP(nn.Module):

    def __init__(self, n_features):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(1)


# ============================================================
# 4. Prediction
# ============================================================

@torch.no_grad()
def predict_batched(model, X):

    model.eval()

    preds = []

    for start in range(
        0,
        len(X),
        PRED_BATCH_SIZE
    ):

        xb = torch.from_numpy(
            X[start:start + PRED_BATCH_SIZE]
        ).to(
            device,
            non_blocking=True
        )

        pred = model(xb)

        preds.append(
            pred.detach().cpu().numpy()
        )

    return np.concatenate(preds)


# ============================================================
# 5. Load one horizon
#
# We load M5 columns for all years so M4/M5 use exactly the
# same target-valid rows.
# ============================================================

def load_horizon(horizon):

    all_features = FEATURE_SETS["KNN_ANN_M5"]

    id_columns = [
        "측정소코드",
        "issue_time_kst",
        "valid_time_kst",
        "lead_h",
        "PM25_target",
    ]

    columns = list(dict.fromkeys(
        id_columns + all_features
    ))

    data = {}

    for year in [2022, 2023, 2024]:

        df = pd.read_parquet(
            safe_read_path(FILES[year]),
            columns=columns,
            filters=[("lead_h", "==", horizon)]
        )

        y = pd.to_numeric(
            df["PM25_target"],
            errors="coerce"
        ).to_numpy(dtype=np.float64)

        valid = np.isfinite(y)

        df = (
            df.loc[valid]
            .reset_index(drop=True)
        )

        data[year] = df

    return data


# ============================================================
# 6. Preprocessing
#
# Fit imputer/scaler ONLY on 2022.
# Transform 2023 and 2024.
# ============================================================

def preprocess(
    train_df,
    val_df,
    test_df,
    features
):

    def numeric_matrix(df):

        return (
            df[features]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

    X_train_raw = numeric_matrix(train_df)
    X_val_raw = numeric_matrix(val_df)
    X_test_raw = numeric_matrix(test_df)

    y_train = (
        train_df["PM25_target"]
        .to_numpy(dtype=np.float32)
    )

    y_val = (
        val_df["PM25_target"]
        .to_numpy(dtype=np.float32)
    )

    y_test = (
        test_df["PM25_target"]
        .to_numpy(dtype=np.float32)
    )

    # --------------------------------------------------------
    # Imputation: fit ONLY on 2022
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    X_train = imputer.fit_transform(
        X_train_raw
    )

    X_val = imputer.transform(
        X_val_raw
    )

    X_test = imputer.transform(
        X_test_raw
    )

    # --------------------------------------------------------
    # Scaling: fit ONLY on 2022
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        X_train
    )

    X_val = scaler.transform(
        X_val
    )

    X_test = scaler.transform(
        X_test
    )

    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)

    assert np.isfinite(X_train).all()
    assert np.isfinite(X_val).all()
    assert np.isfinite(X_test).all()

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# 7. Locked training
#
# 2023 determines best epoch.
# 2024 is NEVER examined during training.
# ============================================================

def train_locked(
    X_train,
    y_train,
    X_val,
    y_val
):

    set_seed(SEED)

    dataset = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train)
    )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        generator=generator
    )

    model = PM25MLP(
        X_train.shape[1]
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.MSELoss()

    best_val_loss = np.inf
    best_epoch = 0
    best_state = None

    patience_count = 0

    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.train()

        for xb, yb in loader:

            xb = xb.to(
                device,
                non_blocking=True
            )

            yb = yb.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            pred = model(xb)

            loss = criterion(
                pred,
                yb
            )

            loss.backward()

            optimizer.step()

        # ----------------------------------------------------
        # Validation = 2023 only
        # ----------------------------------------------------

        val_pred = predict_batched(
            model,
            X_val
        )

        val_loss = float(
            np.mean(
                (val_pred - y_val) ** 2
            )
        )

        if val_loss < best_val_loss - 1e-6:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_count = 0

        else:

            patience_count += 1

        if patience_count >= PATIENCE:
            break

    # Restore 2023-selected best state

    model.load_state_dict(
        best_state
    )

    elapsed = time.time() - start_time

    return (
        model,
        best_epoch,
        elapsed
    )


# ============================================================
# 8. Metric function
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    return {

        "N": len(y_true),

        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),

        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        ),

        "Bias": float(
            np.mean(
                y_pred - y_true
            )
        ),
    }


# ============================================================
# 9. Locked 2024 test
# ============================================================

metric_records = []
prediction_records = []

total_start = time.time()

for horizon in HORIZONS:

    print("\n" + "=" * 88)
    print(f"HORIZON = {horizon:02d} h")
    print("=" * 88)

    raw = load_horizon(
        horizon
    )

    train_df = raw[2022]
    val_df = raw[2023]
    test_df = raw[2024]

    print(
        f"Train={len(train_df):,} | "
        f"Validation={len(val_df):,} | "
        f"Test={len(test_df):,}"
    )

    # ========================================================
    # Same test observations for M4 and M5
    # ========================================================

    base_prediction = test_df[
        [
            "측정소코드",
            "issue_time_kst",
            "valid_time_kst",
            "lead_h",
            "PM25_target",
        ]
    ].copy()

    horizon_predictions = base_prediction.copy()

    # ========================================================
    # M4 and M5
    # ========================================================

    for model_name in [
        "ANN_M4",
        "KNN_ANN_M5"
    ]:

        features = FEATURE_SETS[
            model_name
        ]

        (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test
        ) = preprocess(
            train_df,
            val_df,
            test_df,
            features
        )

        print(
            f"\n{model_name} | "
            f"features={len(features)}"
        )

        # ----------------------------------------------------
        # Train using 2022;
        # early stopping from 2023
        # ----------------------------------------------------

        (
            model,
            best_epoch,
            train_time
        ) = train_locked(
            X_train,
            y_train,
            X_val,
            y_val
        )

        # ----------------------------------------------------
        # 2023 validation prediction
        # purely for audit/reporting
        # ----------------------------------------------------

        val_pred = predict_batched(
            model,
            X_val
        )

        val_metrics = calculate_metrics(
            y_val,
            val_pred
        )

        # ----------------------------------------------------
        # 2024 independent test — first and only evaluation
        # ----------------------------------------------------

        test_pred = predict_batched(
            model,
            X_test
        )

        test_metrics = calculate_metrics(
            y_test,
            test_pred
        )

        metric_records.append({

            "horizon": horizon,
            "model": model_name,
            "seed": SEED,
            "n_features": len(features),
            "best_epoch": best_epoch,

            "val_N": val_metrics["N"],
            "val_RMSE": val_metrics["RMSE"],
            "val_MAE": val_metrics["MAE"],
            "val_R2": val_metrics["R2"],
            "val_Bias": val_metrics["Bias"],

            "test_N": test_metrics["N"],
            "test_RMSE": test_metrics["RMSE"],
            "test_MAE": test_metrics["MAE"],
            "test_R2": test_metrics["R2"],
            "test_Bias": test_metrics["Bias"],

            "training_time_sec": train_time,
        })

        horizon_predictions[
            f"pred_{model_name}"
        ] = test_pred

        print(
            f"  best epoch = {best_epoch}"
        )

        print(
            f"  2023 VAL  | "
            f"RMSE={val_metrics['RMSE']:.4f} | "
            f"MAE={val_metrics['MAE']:.4f} | "
            f"R2={val_metrics['R2']:.4f} | "
            f"Bias={val_metrics['Bias']:+.4f}"
        )

        print(
            f"  2024 TEST | "
            f"RMSE={test_metrics['RMSE']:.4f} | "
            f"MAE={test_metrics['MAE']:.4f} | "
            f"R2={test_metrics['R2']:.4f} | "
            f"Bias={test_metrics['Bias']:+.4f}"
        )

        del (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            model
        )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_records.append(
        horizon_predictions
    )

    del (
        raw,
        train_df,
        val_df,
        test_df,
        horizon_predictions
    )

    gc.collect()


# ============================================================
# 10. Combine results
# ============================================================

metrics = pd.DataFrame(
    metric_records
)

predictions = pd.concat(
    prediction_records,
    ignore_index=True
)

print("\n" + "=" * 88)
print("2024 LOCKED TEST METRICS")
print("=" * 88)

print(
    metrics[
        [
            "horizon",
            "model",
            "best_epoch",
            "test_N",
            "test_RMSE",
            "test_MAE",
            "test_R2",
            "test_Bias",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 11. Paired M5 - M4 analogue effect
#
# Negative RMSE_change_pct = improvement from analogue
# ============================================================

m4 = (
    metrics[
        metrics["model"] == "ANN_M4"
    ]
    .set_index("horizon")
)

m5 = (
    metrics[
        metrics["model"] == "KNN_ANN_M5"
    ]
    .set_index("horizon")
)

effect_records = []

for horizon in HORIZONS:

    m4_row = m4.loc[horizon]
    m5_row = m5.loc[horizon]

    effect_records.append({

        "horizon": horizon,

        "M4_RMSE":
            m4_row["test_RMSE"],

        "M5_RMSE":
            m5_row["test_RMSE"],

        "RMSE_change_pct":
            (
                (
                    m5_row["test_RMSE"]
                    - m4_row["test_RMSE"]
                )
                / m4_row["test_RMSE"]
                * 100
            ),

        "M4_MAE":
            m4_row["test_MAE"],

        "M5_MAE":
            m5_row["test_MAE"],

        "MAE_change_pct":
            (
                (
                    m5_row["test_MAE"]
                    - m4_row["test_MAE"]
                )
                / m4_row["test_MAE"]
                * 100
            ),

        "R2_delta":
            (
                m5_row["test_R2"]
                - m4_row["test_R2"]
            ),

        "Bias_delta":
            (
                m5_row["test_Bias"]
                - m4_row["test_Bias"]
            ),
    })

analogue_effect = pd.DataFrame(
    effect_records
)

print("\n" + "=" * 88)
print("2024 ANN ANALOGUE EFFECT")
print("Negative RMSE change = KNN+ANN improvement")
print("=" * 88)

print(
    analogue_effect
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 12. Horizon-regime summary
# ============================================================

def horizon_regime(h):

    if h <= 3:
        return "short_1_3"

    elif h <= 12:
        return "medium_4_12"

    else:
        return "long_13_24"


analogue_effect["regime"] = (
    analogue_effect["horizon"]
    .apply(horizon_regime)
)

regime_summary = (
    analogue_effect
    .groupby(
        "regime",
        sort=False
    )
    .agg(

        RMSE_change_mean=(
            "RMSE_change_pct",
            "mean"
        ),

        RMSE_change_sd=(
            "RMSE_change_pct",
            "std"
        ),

        MAE_change_mean=(
            "MAE_change_pct",
            "mean"
        ),

        MAE_change_sd=(
            "MAE_change_pct",
            "std"
        ),

        R2_delta_mean=(
            "R2_delta",
            "mean"
        ),

        horizons=(
            "horizon",
            "count"
        )
    )
    .reset_index()
)

print("\n" + "=" * 88)
print("ANALOGUE EFFECT BY FORECAST REGIME")
print("=" * 88)

print(
    regime_summary
    .round(4)
    .to_string(index=False)
)


# ============================================================
# 13. Horizon trend
# ============================================================

from scipy.stats import spearmanr

rho_rmse, p_rmse = spearmanr(
    analogue_effect["horizon"],
    analogue_effect["RMSE_change_pct"]
)

rho_mae, p_mae = spearmanr(
    analogue_effect["horizon"],
    analogue_effect["MAE_change_pct"]
)

print("\n" + "=" * 88)
print("HORIZON TREND")
print("=" * 88)

print(
    f"RMSE change vs horizon: "
    f"rho={rho_rmse:.4f}, "
    f"p={p_rmse:.4g}"
)

print(
    f"MAE change vs horizon : "
    f"rho={rho_mae:.4f}, "
    f"p={p_mae:.4g}"
)


# ============================================================
# 14. Prediction QC
# ============================================================

print("\n" + "=" * 88)
print("PREDICTION QC")
print("=" * 88)

print(
    "Rows:",
    f"{len(predictions):,}"
)

print(
    "Horizons:",
    predictions["lead_h"].nunique()
)

print(
    "ANN_M4 NaN:",
    predictions["pred_ANN_M4"].isna().sum()
)

print(
    "KNN_ANN_M5 NaN:",
    predictions["pred_KNN_ANN_M5"].isna().sum()
)

assert predictions["lead_h"].nunique() == 24

assert (
    predictions["pred_ANN_M4"]
    .notna()
    .all()
)

assert (
    predictions["pred_KNN_ANN_M5"]
    .notna()
    .all()
)

# Verify exactly identical observations
# are used for both models by construction.

print("Paired test observations: PASS")


# ============================================================
# 15. Save
# ============================================================

metrics_file = (
    OUT_DIR /
    "phase5C_locked_2024_metrics.csv"
)

effect_file = (
    OUT_DIR /
    "phase5C_locked_2024_analogue_effect.csv"
)

regime_file = (
    OUT_DIR /
    "phase5C_locked_2024_regime_summary.csv"
)

prediction_file = (
    OUT_DIR /
    "phase5C_locked_2024_predictions.parquet"
)

metrics.to_csv(
    safe_write_path(metrics_file),
    index=False,
    encoding="utf-8-sig"
)

analogue_effect.to_csv(
    safe_write_path(effect_file),
    index=False,
    encoding="utf-8-sig"
)

regime_summary.to_csv(
    safe_write_path(regime_file),
    index=False,
    encoding="utf-8-sig"
)

predictions.to_parquet(
    safe_write_path(prediction_file),
    index=False
)


print("\nSaved:")
print(metrics_file)
print(effect_file)
print(regime_file)
print(prediction_file)

print(
    "\nTotal runtime:",
    round(
        (time.time() - total_start) / 60,
        2
    ),
    "min"
)

print("\nPHASE 5C COMPLETE")
