"""Train and promote the best turbine-output regression model."""

from __future__ import annotations

import mlflow
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

from common import (
    BEST_MODEL_METADATA_PATH,
    BEST_MODEL_PATH,
    FEATURE_COLUMNS,
    MLRUNS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    TARGET_COLUMN,
    ensure_project_dirs,
    load_training_frame,
    regression_metrics,
    save_model_bundle,
    split_features_target,
    utc_now_iso,
    write_json,
)


RANDOM_STATE = 42
TEST_SIZE = 0.2
EXPERIMENT_NAME = "windcast-turbine-output-mwh"


def build_models() -> dict[str, object]:
    """Create candidate models with reproducible configuration."""
    return {
        "Lasso": Lasso(alpha=0.1, max_iter=10000, random_state=RANDOM_STATE),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def main() -> None:
    """Run training, MLflow tracking, model selection, and JSON reporting."""
    ensure_project_dirs()
    mlflow.set_tracking_uri((PROJECT_ROOT / "mlruns").as_uri())
    mlflow.set_experiment(EXPERIMENT_NAME)

    data = load_training_frame()
    features, target = split_features_target(data)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model_results: list[dict[str, object]] = []

    # Train each candidate under its own MLflow run for traceability.
    for model_name, model in build_models().items():
        with mlflow.start_run(run_name=model_name) as run:
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            metrics = regression_metrics(y_test, predictions)

            params = model.get_params()
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("priority", "high")
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                input_example=x_train.head(1),
            )

            model_results.append(
                {
                    "model_name": model_name,
                    "run_id": run.info.run_id,
                    "params": params,
                    "metrics": metrics,
                }
            )

    best_result = min(model_results, key=lambda item: item["metrics"]["rmse"])
    best_model_name = str(best_result["model_name"])
    best_model = build_models()[best_model_name]
    best_model.fit(x_train, y_train)

    save_model_bundle(BEST_MODEL_PATH, best_model, best_model_name)

    metadata = {
        "model_name": best_model_name,
        "model_path": str(BEST_MODEL_PATH),
        "selected_metric": "rmse",
        "version": "1.0",
        "created_at": utc_now_iso(),
    }
    write_json(BEST_MODEL_METADATA_PATH, metadata)

    output = {
        "experiment_name": EXPERIMENT_NAME,
        "models": [
            {
                "name": str(result["model_name"]),
                "mae": result["metrics"]["mae"],
                "rmse": result["metrics"]["rmse"],
                "r2": result["metrics"]["r2"],
            }
            for result in model_results
        ],
        "best_model": best_model_name,
        "best_metric_name": "rmse",
        "best_metric_value": best_result["metrics"]["rmse"],
    }
    write_json(RESULTS_DIR / "step1_s1.json", output)


if __name__ == "__main__":
    main()
