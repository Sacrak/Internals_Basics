"""Command-line predictor for turbine output."""

from __future__ import annotations

import argparse
import json
import sys

from common import RESULTS_DIR, build_prediction_frame, load_model_bundle, write_json


def parse_args() -> argparse.Namespace:
    """Parse feature inputs from the command line."""
    parser = argparse.ArgumentParser(description="Predict turbine_output_mwh.")
    parser.add_argument("--wind_speed_kmph", type=float, required=True)
    parser.add_argument("--blade_length_m", type=float, required=True)
    parser.add_argument("--altitude_m", type=float, required=True)
    parser.add_argument("--humidity_pct", type=float, required=True)
    return parser.parse_args()


def main() -> None:
    """Load the promoted model and emit a JSON prediction."""
    args = parse_args()
    values = vars(args)
    bundle = load_model_bundle()
    frame = build_prediction_frame(values)
    prediction = float(bundle["model"].predict(frame)[0])

    payload = {
        "image_name": "windcast-predictor",
        "image_tag": "v1",
        "base_image": "python:3.12-slim",
        "test_input": values,
        "features": values,
        "prediction": prediction,
    }
    step2_payload = {
        "image_name": "windcast-predictor",
        "image_tag": "v1",
        "base_image": "python:3.12-slim",
        "test_input": values,
        "prediction": prediction,
    }
    write_json(RESULTS_DIR / "step2_s3.json", step2_payload)
    sys.stdout.write(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
