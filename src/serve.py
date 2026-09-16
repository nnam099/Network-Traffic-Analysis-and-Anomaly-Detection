from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluator import IDSArtifacts, load_ids_artifacts, preprocess_raw_df, run_batch_scores
from ids.evaluator import predict_with_uncertainty
from inference_runtime import assess_normalization_quality, risk_score

MODEL_VERSION = os.getenv("IDS_MODEL_VERSION", "v14").strip().lower()
if MODEL_VERSION != "v14":
    raise RuntimeError(
        "v15 disabled: runtime does not implement KNN/ood_ensemble scoring used at evaluation time"
    )
LOW_CONFIDENCE_ENTROPY = 1.5


class PredictRequest(BaseModel):
    features: list[float]


class FlowPredictRequest(BaseModel):
    event: dict[str, Any]


def _required_env_path(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set")
    if not os.path.exists(value):
        raise RuntimeError(f"{name} does not exist: {value}")
    return value


def load_artifacts_from_env() -> IDSArtifacts:
    model_path = _required_env_path("IDS_MODEL_PATH")
    pipeline_path = _required_env_path("IDS_PIPELINE_PATH")
    return load_ids_artifacts(model_path, pipeline_path, MODEL_VERSION)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts_from_env()
    yield


app = FastAPI(title="IDS v14 Inference API", version=MODEL_VERSION, lifespan=lifespan)


def _artifacts() -> IDSArtifacts:
    artifacts = getattr(app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="IDS artifacts are not loaded")
    return artifacts


def _prediction_row(features: list[float], artifacts: IDSArtifacts) -> dict[str, Any]:
    if not features:
        raise HTTPException(status_code=400, detail="features must contain at least one value")
    if any(not isfinite(item) for item in features):
        raise HTTPException(status_code=400, detail="features must contain only finite numeric values")

    expected = int(artifacts.checkpoint.get("n_features", len(artifacts.feature_names)))
    if len(features) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"expected {expected} features, received {len(features)}",
        )

    raw = np.asarray(features, dtype=np.float32).reshape(1, -1)
    scores = run_batch_scores(raw, artifacts, batch_size=1)
    if scores.empty:
        raise HTTPException(status_code=500, detail="model produced no prediction")

    scaled = artifacts.pipeline["scaler"].transform(raw)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    x = torch.as_tensor(scaled, dtype=torch.float32)
    mean_probs, std_probs, entropy = predict_with_uncertainty(artifacts.model, x)
    row = scores.iloc[0].to_dict()
    pred_idx = int(np.asarray(mean_probs).argmax())
    row["uncertainty"] = {
        "entropy": float(entropy),
        "std_max_class": float(std_probs[pred_idx].item()),
    }
    return row


def _flow_prediction(event: dict[str, Any], artifacts: IDSArtifacts) -> dict[str, Any]:
    if not event:
        raise HTTPException(status_code=400, detail="event must contain at least one field")

    try:
        raw_features, normalization_report = preprocess_raw_df(pd.DataFrame([event]), artifacts)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not normalize event: {exc}") from exc

    scores = run_batch_scores(raw_features, artifacts, batch_size=1)
    if scores.empty:
        raise HTTPException(status_code=500, detail="model produced no prediction")

    row = scores.iloc[0].to_dict()
    quality = assess_normalization_quality(normalization_report)
    severity = "HIGH" if bool(row.get("is_zeroday", False)) else "LOW"
    risk = risk_score(row, severity=severity)
    return {
        "label": str(row.get("predicted_class", "Unknown")),
        "classifier_class": str(row.get("classifier_class", "Unknown")),
        "confidence": float(row.get("max_prob", 0.0)),
        "ae_re": float(row.get("ae_re", row.get("ae_score", 0.0))),
        "hybrid_score": float(row.get("hybrid_score", row.get("hybrid", 0.0))),
        "is_anomaly": bool(row.get("is_zeroday", False)),
        "zero_day_rule": str(row.get("zero_day_rule", "")),
        "risk": risk,
        "normalization": {
            "schema": str(normalization_report.get("schema", "unknown")),
            "quality": quality["level"],
            "feature_coverage": normalization_report.get("feature_coverage"),
            "mapped_columns": normalization_report.get("mapped_columns", {}),
            "warnings": quality["warnings"],
        },
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, Any]:
    row = _prediction_row(payload.features, _artifacts())
    uncertainty = row.get("uncertainty", {})
    entropy = float(uncertainty.get("entropy", 0.0))
    label = "LOW_CONFIDENCE" if entropy > LOW_CONFIDENCE_ENTROPY else str(row.get("predicted_class", "Unknown"))
    return {
        "label": label,
        "confidence": float(row.get("max_prob", 0.0)),
        "ae_re": float(row.get("ae_re", row.get("ae_score", 0.0))),
        "hybrid_score": float(row.get("hybrid_score", row.get("hybrid", 0.0))),
        "is_anomaly": bool(row.get("is_zeroday", False)),
        "uncertainty": {
            "entropy": entropy,
            "std_max_class": float(uncertainty.get("std_max_class", 0.0)),
        },
    }


@app.post("/predict/flow")
async def predict_flow(payload: FlowPredictRequest) -> dict[str, Any]:
    return _flow_prediction(payload.event, _artifacts())
