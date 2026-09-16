from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch

from artifact_validator import validate_artifact_contract
from ids.dataset import normalize_labels
from inference_runtime import (
    ground_truth_verdict,
    hybrid_score_from_meta,
    traffic_verdict,
    zero_day_decision,
)
from log_normalizer import normalize_real_world_logs


@dataclass
class IDSArtifacts:
    model: torch.nn.Module
    pipeline: dict[str, Any]
    checkpoint: dict[str, Any]
    feature_names: list[str]
    class_names: list[str]
    thresholds: dict[str, Any]
    centroids: torch.Tensor | None
    model_version: str


def load_ids_artifacts(model_path: str, pipeline_path: str, model_version: str = "v14") -> IDSArtifacts:
    model_version = model_version.strip().lower()
    if model_version == "v15":
        raise ValueError(
            "v15 disabled: runtime does not implement KNN/ood_ensemble scoring used at evaluation time"
        )
    if model_version != "v14":
        raise ValueError(f"unsupported runtime model version: {model_version}")
    train_mod = __import__(f"ids_{model_version}_unswnb15", fromlist=["IDSModel"])
    IDSModel = train_mod.IDSModel

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    with open(pipeline_path, "rb") as handle:
        pipeline = pickle.load(handle)
    validate_artifact_contract(checkpoint, pipeline).raise_for_errors()

    kwargs = {
        "n_features": checkpoint["n_features"],
        "n_classes": checkpoint["n_classes"],
        "hidden": checkpoint.get("hidden", 256),
        "ae_hidden": checkpoint.get("ae_hidden", 128),
    }
    if model_version == "v15":
        kwargs["latent_dim"] = checkpoint.get("latent_dim", 32)

    model = IDSModel(**kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_encoder = pipeline.get("label_encoder")
    label_classes = getattr(label_encoder, "classes_", None)
    class_names = [str(item) for item in label_classes] if label_classes is not None else []
    feature_names = list(pipeline.get("feature_names", pipeline.get("feat_cols", [])))
    thresholds = dict(pipeline.get("thresholds") or checkpoint.get("thresholds") or {})
    hybrid_meta = pipeline.get("hybrid_meta") or checkpoint.get("hybrid_meta") or thresholds.get("hybrid_meta")
    if hybrid_meta:
        thresholds["hybrid_meta"] = hybrid_meta

    centroids_np = pipeline.get("centroids_np")
    centroids = torch.FloatTensor(centroids_np) if centroids_np is not None else None

    return IDSArtifacts(
        model=model,
        pipeline=pipeline,
        checkpoint=checkpoint,
        feature_names=feature_names,
        class_names=class_names,
        thresholds=thresholds,
        centroids=centroids,
        model_version=model_version,
    )


def preprocess_raw_df(df_raw: pd.DataFrame, artifacts: IDSArtifacts) -> tuple[np.ndarray, dict[str, Any]]:
    train_mod = __import__(f"ids_{artifacts.model_version}_unswnb15", fromlist=["engineer_features"])
    engineer_features = train_mod.engineer_features

    df, report = normalize_real_world_logs(df_raw, expected_features=artifacts.feature_names)
    categorical_maps = artifacts.pipeline.get("categorical_maps", {}) if artifacts.pipeline else {}

    for cat in ["proto", "service", "state"]:
        if cat in df.columns:
            df[f"{cat}_num"] = _encode_categorical_column(df[cat], categorical_maps.get(cat))

    existing_numeric = [
        c for c in df.columns
        if c not in {
            "attack_cat", "label", "label_binary", "srcip", "dstip",
            "sport", "dsport", "stime", "ltime", "id", "proto", "service", "state",
        }
    ]
    try:
        df, _ = engineer_features(df, existing_numeric)
    except Exception:
        pass

    aligned = pd.DataFrame(index=df.index)
    for col in artifacts.feature_names:
        if col in df.columns:
            aligned[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        else:
            aligned[col] = 0.0

    raw = aligned.values.astype(np.float32)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    return raw, report.as_dict()


def run_batch_scores(raw_features: np.ndarray, artifacts: IDSArtifacts, batch_size: int = 512) -> pd.DataFrame:
    if raw_features is None or len(raw_features) == 0:
        return pd.DataFrame()
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    scaled = artifacts.pipeline["scaler"].transform(raw_features)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)
    tensor = torch.FloatTensor(scaled)

    probs_all: list[np.ndarray] = []
    ae_all: list[np.ndarray] = []
    fv_all: list[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(tensor), batch_size):
            x = tensor[i:i + batch_size]
            outputs = artifacts.model(x)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_all.append(np.atleast_2d(probs))

            ae_score = _reconstruction_error(artifacts.model, x, probs)
            ae_all.append(np.atleast_1d(ae_score))

            if artifacts.centroids is not None and hasattr(artifacts.model, "fv_cluster_score"):
                fv_all.append(artifacts.model.fv_cluster_score(x, artifacts.centroids).cpu().numpy())  # type: ignore

    probs_all_np = np.concatenate(probs_all, axis=0)
    ae_all_np = np.concatenate(ae_all, axis=0)
    max_prob = probs_all_np.max(axis=1)
    softmax_score = 1.0 - max_prob
    pred_idx = probs_all_np.argmax(axis=1)
    pred_class = [
        artifacts.class_names[i] if i < len(artifacts.class_names) else "Unknown"
        for i in pred_idx
    ]
    hybrid_score = hybrid_score_from_meta(ae_all_np, max_prob, thresholds=artifacts.thresholds)
    is_zeroday, decision_rule = zero_day_decision(
        ae_all_np,
        max_prob,
        hybrid_score,
        thresholds=artifacts.thresholds,
        ae_threshold=float(artifacts.thresholds.get("ae_re", 0.5)),
        hybrid_threshold=float(artifacts.thresholds.get("hybrid", 0.5)),
    )
    verdict = [_traffic_verdict_bool(zd, cls) for zd, cls in zip(is_zeroday, pred_class, strict=True)]

    out = pd.DataFrame({
        "predicted_class": verdict,
        "classifier_class": pred_class,
        "max_prob": max_prob,
        "softmax": softmax_score,
        "ae_re": ae_all_np,
        "ae_score": ae_all_np,
        "hybrid": hybrid_score,
        "hybrid_score": hybrid_score,
        "is_zeroday": np.asarray(is_zeroday).astype(bool),
        "zero_day_rule": decision_rule,
    })
    if fv_all:
        out["fv_cluster"] = np.concatenate(fv_all, axis=0)
    return out


def summarize_scores(
    scores: pd.DataFrame,
    raw_df: pd.DataFrame | None = None,
    label_col: str | None = None,
    class_names: list[str] | None = None,
    zero_day_labels: list[str] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if scores.empty:
        return {"rows": 0, "error": "no scores produced"}

    summary: dict[str, Any] = {
        "rows": int(len(scores)),
        "zero_day_count": int(scores["is_zeroday"].sum()),
        "zero_day_rate": round(float(scores["is_zeroday"].mean()), 6),
        "classifier_distribution": _value_counts(scores["classifier_class"]),
        "verdict_distribution": _value_counts(scores["predicted_class"]),
        "score_distribution": {},
    }
    for col in ["hybrid", "ae_re", "softmax", "max_prob", "fv_cluster"]:
        if col in scores.columns:
            summary["score_distribution"][col] = _distribution(scores[col])

    if raw_df is not None:
        label_col = label_col or detect_label_column(raw_df)
    if raw_df is not None and label_col and label_col in raw_df.columns:
        truth = raw_df[label_col].map(ground_truth_verdict)
        normal_mask = truth == "Normal"
        attack_mask = truth == "Known-Attack"
        summary["label_column"] = label_col
        summary["ground_truth_distribution"] = _value_counts(truth)
        evaluation = _labeled_evaluation(
            scores,
            raw_df[label_col],
            truth,
            class_names=class_names,
            zero_day_labels=zero_day_labels,
        )
        summary["evaluation"] = evaluation
        summary["accuracy"] = evaluation.get("detection_accuracy")
        summary["recall_per_class"] = evaluation.get("recall_per_class", {})
        summary["ood_detection_rate"] = evaluation.get("ood_detection_rate")
        if bool(normal_mask.any()):
            normal_scores = scores.loc[normal_mask.values]
            normal_ood = normal_scores["is_zeroday"].astype(bool)
            normal_classifier_alert = normal_scores["classifier_class"].astype(str).str.strip().str.lower() != "normal"
            normal_alert = normal_ood | normal_classifier_alert
            normal_overlap = normal_ood & normal_classifier_alert
            normal_verdict_alert = normal_scores["predicted_class"].isin(["Known-Attack", "Zero-Day Candidate"])
            if not normal_alert.equals(normal_verdict_alert):
                raise ValueError("predicted_class is inconsistent with the OOD/classifier alert union")
            summary["normal_ood_fpr"] = round(float(normal_ood.mean()), 6)
            summary["normal_alert_fpr"] = round(float(normal_alert.mean()), 6)
            summary["known_classifier_fpr"] = round(float(normal_classifier_alert.mean()), 6)
            summary["normal_ood_classifier_overlap_rate"] = round(float(normal_overlap.mean()), 6)
            summary["normal_ood_false_positive_count"] = int(normal_ood.sum())
            summary["normal_alert_false_positive_count"] = int(normal_alert.sum())
            summary["known_classifier_false_positive_count"] = int(normal_classifier_alert.sum())
            summary["normal_ood_classifier_overlap_count"] = int(normal_overlap.sum())
            summary["normal_count"] = int(normal_mask.sum())
        if bool(attack_mask.any()):
            attack_known = scores.loc[attack_mask.values, "predicted_class"].isin(["Known-Attack", "Zero-Day Candidate"])
            summary["attack_detection_rate"] = round(float(attack_known.mean()), 6)
            summary["attack_count"] = int(attack_mask.sum())

    if thresholds is not None:
        summary["threshold_profile"] = _threshold_profile(thresholds)

    return summary


def calibrate_thresholds(
    scores: pd.DataFrame,
    target_fpr: float = 0.01,
    raw_df: pd.DataFrame | None = None,
    label_col: str | None = None,
    normal_only: bool = True,
    decision_mode: str = "vote",
    min_votes: int = 2,
) -> dict[str, Any]:
    reference = scores
    detected_label = None
    if raw_df is not None:
        detected_label = label_col or detect_label_column(raw_df)
    if normal_only and raw_df is not None and detected_label and detected_label in raw_df.columns:
        truth = raw_df[detected_label].map(ground_truth_verdict)
        normal_mask = truth == "Normal"
        if bool(normal_mask.any()):
            reference = scores.loc[normal_mask.values].copy()

    if reference.empty:
        raise ValueError("No reference rows available for threshold calibration")

    threshold_cols = [c for c in ["hybrid", "ae_re", "softmax", "fv_cluster"] if c in reference.columns]
    thresholds: dict[str, Any] = {
        "decision_mode": decision_mode,
        "min_votes": int(min_votes),
    }
    for col in threshold_cols:
        key = "hybrid" if col == "hybrid" else col
        thresholds[key] = float(reference[col].quantile(1.0 - target_fpr))

    return {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "target_fpr": float(target_fpr),
        "reference_rows": int(len(reference)),
        "label_column": detected_label,
        "normal_only": bool(normal_only),
        "thresholds": thresholds,
    }


def save_threshold_profile(profile: dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(profile, handle, indent=2, sort_keys=True)


def load_threshold_profile(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_label_column(df: pd.DataFrame) -> str | None:
    candidates = ["label", "label_binary", "attack_cat", "attack_category", "class", "category"]
    lower_to_original = {str(col).strip().lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def _encode_categorical_column(series: pd.Series, mapping=None) -> np.ndarray:
    values = series.astype(str).fillna("unk")
    if mapping:
        return values.map(lambda x: mapping.get(x, mapping.get("unk", -1))).astype(np.float32).values
    codes, _ = pd.factorize(values, sort=True)
    return codes.astype(np.float32)


def _reconstruction_error(model: torch.nn.Module, x: torch.Tensor, probs: np.ndarray) -> np.ndarray:
    if hasattr(model, "ae"):
        ae_score = model.ae.recon_error(x)  # type: ignore
    elif hasattr(model, "vae"):
        ae_score = model.vae.recon_error(x)  # type: ignore
    elif hasattr(model, "autoencoder"):
        recon = model.autoencoder(x)  # type: ignore
        ae_score = torch.mean((x - recon) ** 2, dim=-1)
    else:
        return 1.0 - probs.max(axis=1)

    if isinstance(ae_score, torch.Tensor):
        ae_score = ae_score.cpu().numpy()
    ae_score = np.atleast_1d(ae_score).reshape(-1)
    if ae_score.shape[0] != len(x):
        ae_score = np.full(len(x), float(np.mean(ae_score)), dtype=np.float32)
    return ae_score


def _distribution(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {}
    return {
        "min": float(clean.min()),
        "p50": float(clean.quantile(0.50)),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def _value_counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts(dropna=False).to_dict().items()}


def _labeled_evaluation(
    scores: pd.DataFrame,
    labels: pd.Series,
    truth_verdict: pd.Series,
    class_names: list[str] | None = None,
    zero_day_labels: list[str] | None = None,
) -> dict[str, Any]:
    labels_text = _canonical_label_series(labels)
    truth_verdict = truth_verdict.astype(str)
    known_classes = set(_canonical_label_series(pd.Series(class_names or [], dtype=str)))
    zero_day_set = set(_canonical_label_series(pd.Series(zero_day_labels or [], dtype=str)))
    comparable_mask = truth_verdict.isin(["Normal", "Known-Attack"])
    predicted_attack = scores["predicted_class"].isin(["Known-Attack", "Zero-Day Candidate"])
    predicted_verdict = pd.Series(np.where(predicted_attack, "Known-Attack", "Normal"), index=scores.index)

    out: dict[str, Any] = {
        "rows_with_labels": int(comparable_mask.sum()),
        "detection_accuracy": None,
        "classifier_accuracy_known": None,
        "recall_per_class": {},
        "ood_detection_rate": None,
        "zero_day_recall_per_family": {},
    }

    if bool(comparable_mask.any()):
        out["detection_accuracy"] = round(
            float((predicted_verdict.loc[comparable_mask.values].values == truth_verdict.loc[comparable_mask].values).mean()),
            6,
        )

    if known_classes:
        known_mask = labels_text.isin(known_classes)
        if bool(known_mask.any()):
            known_scores = scores.loc[known_mask.values]
            known_labels = labels_text.loc[known_mask].reset_index(drop=True)
            known_pred = known_scores["classifier_class"].astype(str).reset_index(drop=True)
            out["classifier_accuracy_known"] = round(float((known_pred == known_labels).mean()), 6)
            recall: dict[str, dict[str, Any]] = {}
            for cls in sorted(known_classes):
                cls_mask = known_labels == cls
                if bool(cls_mask.any()):
                    correct = int((known_pred.loc[cls_mask] == cls).sum())
                    total = int(cls_mask.sum())
                    recall[cls] = {"recall": round(correct / total, 6), "support": total}
            out["recall_per_class"] = recall

    if zero_day_set:
        zd_mask = labels_text.isin(zero_day_set)
        if bool(zd_mask.any()):
            zd_scores = scores.loc[zd_mask.values]
            out["ood_detection_rate"] = round(float(zd_scores["is_zeroday"].mean()), 6)
            family_recall: dict[str, dict[str, Any]] = {}
            for family in sorted(zero_day_set):
                family_mask = labels_text == family
                if bool(family_mask.any()):
                    detected = scores.loc[family_mask.values, "is_zeroday"]
                    family_recall[family] = {
                        "recall": round(float(detected.mean()), 6),
                        "support": int(family_mask.sum()),
                    }
            out["zero_day_recall_per_family"] = family_recall

    return out


def _canonical_label_series(labels: pd.Series) -> pd.Series:
    frame = labels.rename("attack_cat").to_frame()
    return normalize_labels(frame)["attack_cat"]


def _threshold_profile(thresholds: dict[str, Any]) -> dict[str, Any]:
    profile = {}
    for key, value in thresholds.items():
        if key == "hybrid_meta":
            profile[key] = {
                "type": value.get("type"),
                "features": value.get("features"),
                "coef": value.get("coef"),
                "intercept": value.get("intercept"),
            } if isinstance(value, dict) else value
            continue
        try:
            profile[key] = float(value)
        except (TypeError, ValueError):
            profile[key] = value
    return profile


def _traffic_verdict_bool(is_zeroday: Any, classifier_class: str) -> str:
    return traffic_verdict(bool(is_zeroday), classifier_class)
