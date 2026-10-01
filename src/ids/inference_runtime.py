from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def _sigmoid(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def hybrid_score_from_meta(ae_score, max_prob, thresholds: dict | None = None, hybrid_meta: dict | None = None):
    meta = hybrid_meta or (thresholds or {}).get("hybrid_meta")
    ae_score = np.asarray(ae_score, dtype=np.float32)
    softmax_score = 1.0 - np.asarray(max_prob, dtype=np.float32)
    if isinstance(meta, dict):
        coef = np.asarray(meta.get("coef", []), dtype=np.float64).reshape(-1)
        if coef.size >= 2:
            intercept = float(meta.get("intercept", 0.0))
            return np.asarray(
                _sigmoid(intercept + coef[0] * ae_score + coef[1] * softmax_score),
                dtype=np.float32,
            )

    # Legacy artifacts produced before the meta-learner still need a deterministic score.
    return np.asarray(0.5 * ae_score + 0.5 * softmax_score, dtype=np.float32)


def zero_day_decision(
    ae_score,
    max_prob,
    hybrid_score,
    thresholds: dict | None = None,
    ae_threshold: float = 0.5,
    hybrid_threshold: float = 0.5,
):
    if thresholds and thresholds.get("decision_mode") == "vote":
        min_votes = int(thresholds.get("min_votes", 2))
        if min_votes <= 0:
            raise ValueError("vote threshold min_votes must be positive")
        votes = []
        if "hybrid" in thresholds:
            votes.append(np.asarray(hybrid_score) > float(thresholds["hybrid"]))
        if "ae_re" in thresholds:
            votes.append(np.asarray(ae_score) > float(thresholds["ae_re"]))
        if "softmax" in thresholds:
            votes.append((1.0 - np.asarray(max_prob)) > float(thresholds["softmax"]))
        if votes:
            vote_count = np.sum(np.stack(votes, axis=0), axis=0)
            return vote_count >= min_votes, f"vote_{min_votes}_of_{len(votes)}"

    if thresholds and "hybrid" in thresholds:
        calibrated_threshold = float(thresholds.get("hybrid", hybrid_threshold))
        return np.asarray(hybrid_score) > calibrated_threshold, "hybrid_calibrated"
    return (np.asarray(ae_score) > ae_threshold) & (np.asarray(max_prob) < 0.6), "ae_plus_confidence_fallback"


def _autoencoder_reconstruction_error(model, x, probs):
    if hasattr(model, "ae"):
        ae_score = model.ae.recon_error(x)
    elif hasattr(model, "vae"):
        ae_score = model.vae.recon_error(x)
    elif hasattr(model, "autoencoder"):
        recon = model.autoencoder(x)
        ae_score = torch.mean((x - recon) ** 2, dim=-1)
    else:
        ae_score = 1.0 - probs.max(axis=1)

    if isinstance(ae_score, torch.Tensor):
        ae_score = ae_score.detach().cpu().numpy()
    ae_score = np.atleast_1d(ae_score).astype(np.float32).reshape(-1)
    if ae_score.shape[0] != len(x):
        ae_score = np.full(len(x), float(np.mean(ae_score)), dtype=np.float32)
    return ae_score


def run_batch_inference(
    model,
    scaler,
    raw_features,
    class_names: list[str],
    thresholds: dict | None = None,
    ae_threshold: float = 0.5,
    hybrid_threshold: float = 0.5,
    batch_size: int = 512,
) -> pd.DataFrame:
    """Run classifier and reconstruction scoring for a feature matrix."""
    if model is None or scaler is None:
        return pd.DataFrame()
    if raw_features is None or len(raw_features) == 0:
        return pd.DataFrame()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    scaled = scaler.transform(raw_features)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    tensor = torch.as_tensor(scaled, dtype=torch.float32)
    probs_all = []
    ae_all = []

    if hasattr(model, "eval"):
        model.eval()

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            x = tensor[i:i + batch_size]
            outputs = model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs = np.atleast_2d(probs)
            probs_all.append(probs)
            ae_all.append(_autoencoder_reconstruction_error(model, x, probs))

    if not probs_all or not ae_all:
        return pd.DataFrame()

    probs_arr = np.concatenate(probs_all, axis=0)
    ae_arr = np.concatenate(ae_all, axis=0)
    max_prob = probs_arr.max(axis=1)
    pred_idx = probs_arr.argmax(axis=1)
    pred_class = [class_names[i] if i < len(class_names) else "Unknown" for i in pred_idx]
    hybrid_score = hybrid_score_from_meta(ae_arr, max_prob, thresholds=thresholds)
    is_zeroday, decision_rule = zero_day_decision(
        ae_arr,
        max_prob,
        hybrid_score,
        thresholds=thresholds,
        ae_threshold=ae_threshold,
        hybrid_threshold=hybrid_threshold,
    )
    is_zeroday = np.asarray(is_zeroday).astype(bool)
    verdict = [traffic_verdict(zd, cls) for zd, cls in zip(is_zeroday, pred_class, strict=True)]

    return pd.DataFrame({
        "predicted_class": verdict,
        "classifier_class": pred_class,
        "max_prob": max_prob,
        "ae_score": ae_arr,
        "hybrid_score": hybrid_score,
        "is_zeroday": is_zeroday,
        "zero_day_rule": decision_rule,
    })


def traffic_verdict(is_zeroday, classifier_class) -> str:
    if bool(is_zeroday):
        return "Zero-Day Candidate"
    if str(classifier_class).strip().lower() == "normal":
        return "Normal"
    return "Known-Attack"


def ground_truth_verdict(label) -> str:
    value = str(label).strip().lower()
    if value in {"", "nan", "none", "-", "unknown"}:
        return ""
    if value in {"0", "benign", "normal"}:
        return "Normal"
    return "Known-Attack"


def severity_rank(severity: str) -> int:
    return {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}.get(str(severity).upper(), 0)


def severity_class(severity: str) -> str:
    return {
        "CRITICAL": "sev-critical",
        "HIGH": "sev-high",
        "MEDIUM": "sev-medium",
        "LOW": "sev-low",
    }.get(str(severity).upper(), "sev-medium")


def risk_score(result: dict, severity: str | None = None) -> int:
    sev_weight = {"CRITICAL": 35, "HIGH": 25, "MEDIUM": 15, "LOW": 5}.get(str(severity or "").upper(), 10)
    hybrid = float(result.get("hybrid_score", 0)) * 45
    ae = min(float(result.get("ae_score", 0)), 1.0) * 15
    zd = 15 if result.get("is_zeroday") else 0
    return int(min(100, round(sev_weight + hybrid + ae + zd)))


def assess_normalization_quality(report: dict | None) -> dict:
    if not isinstance(report, dict):
        return {
            "level": "UNKNOWN",
            "message": "No CSV normalization report is available.",
            "warnings": ["Normalization did not run or did not return metadata."],
        }

    coverage = report.get("feature_coverage")
    try:
        coverage = float(coverage) if coverage is not None else None
    except (TypeError, ValueError):
        coverage = None

    missing = list(report.get("missing_core_features") or [])
    mapped_count = len(report.get("mapped_columns") or {})
    warnings = []

    if coverage is None:
        level = "UNKNOWN"
        warnings.append("Feature coverage is unavailable.")
    elif coverage >= 0.80:
        level = "GOOD"
    elif coverage >= 0.60:
        level = "MEDIUM"
        warnings.append("Feature coverage is moderate; treat scores as triage signals.")
    else:
        level = "LOW"
        warnings.append("Feature coverage is low; inference confidence is reduced.")

    directional = {"sbytes", "dbytes", "spkts", "dpkts"}
    missing_directional = sorted(directional & set(missing))
    if missing_directional:
        warnings.append("Missing directional counters: " + ", ".join(missing_directional))

    timing = {"dur", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat"}
    missing_timing = sorted(timing & set(missing))
    if missing_timing:
        warnings.append("Missing timing/TCP fields: " + ", ".join(missing_timing[:6]))

    if mapped_count < 5:
        warnings.append("Few source columns were mapped; verify CSV schema manually.")

    if not warnings:
        warnings.append("CSV schema has enough core fields for prototype inference.")

    cov_text = "unknown" if coverage is None else f"{coverage * 100:.1f}%"
    return {
        "level": level,
        "coverage": coverage,
        "mapped_columns": mapped_count,
        "message": f"Normalization quality: {level} (coverage {cov_text}, mapped columns {mapped_count}).",
        "warnings": warnings,
    }
