"""Shared utilities for the turbine-output MLOps workflow."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

TRAINING_DATA_PATH = DATA_DIR / "training_data.csv"
NEW_DATA_PATH = DATA_DIR / "new_data.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"

FEATURE_COLUMNS = [
    "wind_speed_kmph",
    "blade_length_m",
    "altitude_m",
    "humidity_pct",
]
TARGET_COLUMN = "turbine_output_mwh"


def utc_now_iso() -> str:
    """Return an ISO 8601 UTC timestamp for reproducible machine logs."""
    return datetime.now(timezone.utc).isoformat()


def ensure_project_dirs() -> None:
    """Create output directories without changing any dataset files."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_training_frame(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load and validate the immutable training dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected training dataset at {path}. Place training_data.csv in data/."
        )

    frame = pd.read_csv(path)
    required_columns = [*FEATURE_COLUMNS, TARGET_COLUMN]
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    return frame


def split_features_target(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the regression target."""
    return frame[FEATURE_COLUMNS], frame[TARGET_COLUMN]


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate regression metrics as plain floats for JSON and MLflow."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Persist machine-readable output with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)


def save_model_bundle(path: Path, model: Any, model_type: str) -> None:
    """Save model plus feature contract for reproducible inference."""
    bundle = {
        "model": model,
        "model_type": model_type,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "version": "1.0",
    }
    joblib.dump(bundle, path)


def load_model_bundle(path: Path = BEST_MODEL_PATH) -> dict[str, Any]:
    """Load the promoted model bundle used by CLI and API inference."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found at {path}. Run src/train.py first.")
    return joblib.load(path)


def build_prediction_frame(values: dict[str, float]) -> pd.DataFrame:
    """Build a one-row inference frame using the training feature order."""
    return pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
