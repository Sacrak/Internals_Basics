"""FastAPI service for turbine-output estimation."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

from common import build_prediction_frame, load_model_bundle
from common import RESULTS_DIR, write_json


class TurbineFeatures(BaseModel):
    """Validated feature payload with physically reasonable ranges."""

    wind_speed_kmph: float = Field(..., ge=5, le=60)
    blade_length_m: float = Field(..., ge=20, le=80)
    altitude_m: float = Field(..., ge=50, le=200)
    humidity_pct: float = Field(..., ge=20, le=90)


app = FastAPI(title="Turbine Output Estimator", version="1.0")
MODEL_BUNDLE = load_model_bundle()


@app.get("/health")
def health() -> dict[str, str]:
    """Return service and model health metadata."""
    return {
        "status": "running",
        "model": str(MODEL_BUNDLE["model_type"]),
        "version": "1.0",
    }


@app.post("/estimate")
def estimate(features: TurbineFeatures) -> dict[str, object]:
    """Estimate turbine output for a validated feature payload."""
    values = features.model_dump()
    frame = build_prediction_frame(values)
    prediction = float(MODEL_BUNDLE["model"].predict(frame)[0])
    response = {
        "prediction": prediction,
    }
    step3_payload = {
        "health_endpoint": "/health",
        "predict_endpoint": "/estimate",
        "port": 9000,
        "health_response": health(),
        "test_input": values,
        "prediction": prediction,
    }
    write_json(RESULTS_DIR / "step3_s4.json", step3_payload)
    return response


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=9000)
