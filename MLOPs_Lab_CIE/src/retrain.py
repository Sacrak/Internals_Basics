"""Retrain the promoted turbine model when new data is available."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from common import (
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    NEW_DATA_PATH,
    RESULTS_DIR,
    TARGET_COLUMN,
    TRAINING_DATA_PATH,
    ensure_project_dirs,
    load_model_bundle,
    load_training_frame,
    regression_metrics,
    save_model_bundle,
    split_features_target,
    utc_now_iso,
    write_json,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
PROMOTION_THRESHOLD_MAE = 0.5


def build_same_model(model_type: str) -> object:
    """Create a fresh model matching the currently promoted model type."""
    if model_type == "Lasso":
        return Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE)
    if model_type == "GradientBoosting":
        return GradientBoostingRegressor(random_state=RANDOM_STATE)
    raise ValueError(f"Unsupported promoted model type: {model_type}")


def main() -> None:
    """Compare retrained model MAE against the existing promoted model."""
    ensure_project_dirs()

    if not NEW_DATA_PATH.exists():
        raise FileNotFoundError(f"Expected new data at {NEW_DATA_PATH}.")

    original_data = load_training_frame(TRAINING_DATA_PATH)
    new_data = load_training_frame(NEW_DATA_PATH)
    combined_data = pd.concat([original_data, new_data], ignore_index=True)

    original_features, original_target = split_features_target(original_data)
    x_train, x_test, y_train, y_test = train_test_split(
        original_features,
        original_target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    bundle = load_model_bundle()
    promoted_model_type = str(bundle["model_type"])
    original_model = bundle["model"]
    original_predictions = original_model.predict(x_test)
    original_metrics = regression_metrics(y_test, original_predictions)

    combined_features, combined_target = split_features_target(combined_data)
    retrained_model = build_same_model(promoted_model_type)
    retrained_model.fit(combined_features, combined_target)
    retrained_predictions = retrained_model.predict(x_test)
    retrained_metrics = regression_metrics(y_test, retrained_predictions)

    mae_improvement = original_metrics["mae"] - retrained_metrics["mae"]
    promoted = mae_improvement >= PROMOTION_THRESHOLD_MAE

    # Promote only when the new model clears the required MAE improvement.
    if promoted:
        save_model_bundle(BEST_MODEL_PATH, retrained_model, promoted_model_type)

    output = {
        "original_data_rows": int(len(original_data)),
        "new_data_rows": int(len(new_data)),
        "combined_data_rows": int(len(combined_data)),
        "champion_mae": original_metrics["mae"],
        "retrained_mae": retrained_metrics["mae"],
        "improvement": float(mae_improvement),
        "min_improvement_threshold": PROMOTION_THRESHOLD_MAE,
        "action": "promoted" if promoted else "kept_champion",
        "comparison_metric": "mae",
    }
    write_json(RESULTS_DIR / "step4_s8.json", output)


if __name__ == "__main__":
    main()
