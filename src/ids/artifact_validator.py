from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ArtifactValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_for_errors(self) -> None:
        if self.errors:
            raise ValueError("Invalid IDS artifacts: " + "; ".join(self.errors))


def validate_artifact_contract(checkpoint: dict[str, Any], pipeline: dict[str, Any]) -> ArtifactValidationResult:
    """
    Validate the model checkpoint and preprocessing pipeline contract before inference.

    This intentionally checks only metadata and object shape contracts. It does not
    unpickle, load weights, or execute model code.
    """
    result = ArtifactValidationResult()

    if not isinstance(checkpoint, dict):
        result.errors.append("checkpoint must be a dict")
        return result
    if not isinstance(pipeline, dict):
        result.errors.append("pipeline must be a dict")
        return result

    _require_keys(checkpoint, ["model_state_dict", "n_features", "n_classes"], "checkpoint", result)
    _require_keys(pipeline, ["scaler", "label_encoder"], "pipeline", result)

    n_features = _positive_int(checkpoint.get("n_features"), "checkpoint.n_features", result)
    n_classes = _positive_int(checkpoint.get("n_classes"), "checkpoint.n_classes", result)

    feature_names = pipeline.get("feature_names", pipeline.get("feat_cols"))
    if feature_names is None:
        result.errors.append("pipeline must include feature_names or feat_cols")
    elif not isinstance(feature_names, (list, tuple)):
        result.errors.append("pipeline feature_names/feat_cols must be a list or tuple")
    elif n_features is not None and len(feature_names) != n_features:
        result.errors.append(
            f"feature count mismatch: checkpoint.n_features={n_features}, pipeline features={len(feature_names)}"
        )
    elif feature_names:
        normalized_features = [str(name).strip() for name in feature_names]
        if any(not name for name in normalized_features):
            result.errors.append("pipeline feature_names/feat_cols must not contain empty names")
        duplicates = sorted(name for name, count in Counter(normalized_features).items() if count > 1)
        if duplicates:
            result.errors.append("pipeline feature_names/feat_cols contains duplicates: " + ", ".join(duplicates[:5]))
        metadata_features = [name for name in normalized_features if name.startswith("__")]
        if metadata_features:
            result.errors.append(
                "pipeline feature_names/feat_cols contains metadata columns: "
                + ", ".join(metadata_features[:5])
            )

    checkpoint_features = checkpoint.get("feat_cols")
    if isinstance(checkpoint_features, (list, tuple)):
        checkpoint_metadata = [
            str(name).strip() for name in checkpoint_features
            if str(name).strip().startswith("__")
        ]
        if checkpoint_metadata:
            result.errors.append(
                "checkpoint feat_cols contains metadata columns: "
                + ", ".join(checkpoint_metadata[:5])
            )

    label_encoder = pipeline.get("label_encoder")
    label_classes = getattr(label_encoder, "classes_", None)
    if label_encoder is None:
        pass
    elif label_classes is None:
        result.errors.append("pipeline.label_encoder must expose classes_")
    elif n_classes is not None and len(label_classes) != n_classes:
        result.errors.append(
            f"class count mismatch: checkpoint.n_classes={n_classes}, label_encoder.classes_={len(label_classes)}"
        )
    elif label_classes is not None and "Normal" not in {str(item) for item in label_classes}:
        result.warnings.append("label_encoder.classes_ does not include Normal")

    scaler = pipeline.get("scaler")
    scaler_n_features = getattr(scaler, "n_features_in_", None)
    if scaler is None:
        pass
    elif scaler_n_features is not None and n_features is not None and int(scaler_n_features) != n_features:
        result.errors.append(
            f"scaler feature mismatch: scaler.n_features_in_={scaler_n_features}, checkpoint.n_features={n_features}"
        )

    thresholds = pipeline.get("thresholds", checkpoint.get("thresholds"))
    if thresholds is None:
        result.warnings.append("thresholds are missing; dashboard will use the fallback anomaly rule")
    elif not isinstance(thresholds, dict):
        result.errors.append("thresholds must be a dict when present")
    else:
        non_negative_thresholds = {
            "hybrid",
            "ae_re",
            "vae_recon",
            "ood_ensemble",
            "gradbp_l2",
            "softmax",
            "fv_cluster",
            "knn_dist",
        }
        control_keys = {"decision_mode", "hybrid_meta", "min_votes"}
        for key, value in thresholds.items():
            if key in control_keys:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                result.errors.append(f"thresholds.{key} must be numeric")
                continue
            if numeric_value != numeric_value or numeric_value in {float("inf"), float("-inf")}:
                result.errors.append(f"thresholds.{key} must be finite")
            if key in non_negative_thresholds and numeric_value < 0:
                result.errors.append(f"thresholds.{key} must be non-negative")
        if thresholds.get("decision_mode") == "vote":
            _positive_int_control(thresholds.get("min_votes", 2), "thresholds.min_votes", result)
            if not any(k in thresholds for k in ("hybrid", "ae_re", "softmax")):
                result.errors.append("vote thresholds require at least one of: hybrid, ae_re, softmax")

        if not any(k in thresholds for k in ("hybrid", "ae_re", "vae_recon", "ood_ensemble", "softmax")):
            result.warnings.append("thresholds do not include a recognized anomaly score key")

    checkpoint_version = checkpoint.get("version")
    pipeline_version = pipeline.get("version")
    if checkpoint_version and pipeline_version:
        ckpt_family = _version_family(str(checkpoint_version))
        pipe_family = _version_family(str(pipeline_version))
        if ckpt_family and pipe_family and ckpt_family != pipe_family:
            result.warnings.append(
                f"artifact version family differs: checkpoint={checkpoint_version}, pipeline={pipeline_version}"
            )

    return result


def _require_keys(obj: dict[str, Any], keys: list[str], name: str, result: ArtifactValidationResult) -> None:
    for key in keys:
        if key not in obj:
            result.errors.append(f"{name} missing required key: {key}")


def _positive_int(value: Any, name: str, result: ArtifactValidationResult) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        result.errors.append(f"{name} must be an integer")
        return None
    if parsed <= 0:
        result.errors.append(f"{name} must be positive")
        return None
    return parsed


def _positive_int_control(value: Any, name: str, result: ArtifactValidationResult) -> int | None:
    if isinstance(value, bool):
        result.errors.append(f"{name} must be an integer")
        return None
    try:
        parsed = int(value)
        numeric = float(value)
    except (TypeError, ValueError):
        result.errors.append(f"{name} must be an integer")
        return None
    if numeric != parsed:
        result.errors.append(f"{name} must be an integer")
        return None
    if parsed <= 0:
        result.errors.append(f"{name} must be positive")
        return None
    return parsed


def _version_family(version: str) -> str:
    version = version.lower()
    if "v14" in version:
        return "v14"
    if "v15" in version:
        return "v15"
    return ""
