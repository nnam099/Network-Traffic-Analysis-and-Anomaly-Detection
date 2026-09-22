"""P6 development-only scorer selection on frozen P3/P4/P5 contracts.

No official-test or true-target loader exists in this module. Role materialization
is explicit and restricted to the frozen official-training CSV.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F

from .dataset import (DEFAULT_TRAIN_FILE, KNOWN_ATTACK_CATS, ROW_ID_COLUMN,
                      SOURCE_ROLE_COLUMN, ZERO_DAY_ATTACK_CATS, load_unsw_csvs)
from .p2_protocol import continuous_matrix
from .p3_protocol import train_only_context
from .p4_model_interface import (CATEGORICAL_FIELDS, StructuredBatchAdapter,
                                 StructuredModelInput, canonical_json,
                                 vocabularies_from_manifest)
from .p5_training import (P3_HASH, SPLIT_SEEDS, TARGETS, TRAINING_CONFIG,
                          TrainingOnlyModel, _environment, _state_digest,
                          assert_checkpoint_provenance, batch_order, file_hash,
                          json_bytes, require_role, select_batch,
                          set_determinism, source_hash as p5_source_hash,
                          stable_hash, validate_cell, write_immutable)
from .target_isolation import CANONICAL_FINGERPRINT_COLUMN


P6_VERSION = "p6-identity-scorer-selection-v1"
SCORERS = ("negative_max_softmax", "nearest_centroid_l2")
SELECTION_UNIT = "canonical_identity_mean"
TIE_TOLERANCE = 1e-12
TIE_PREFERENCE = "nearest_centroid_l2"  # frozen P5 exact-tie preference
QUANTILE = 0.99
QUANTILE_METHOD = "higher"
DEVELOPMENT_ROLES = ("train", "val", "meta_known", "surrogate_ood", "calibration")


def output_path(root: Path, relative: str) -> Path:
    base = (root / "results/model_selection/p6").resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base) or any(
        part.lower() in {"official_test", "official-test-clean", "target_held_out"}
        for part in path.parts
    ):
        raise PermissionError("P6 output path outside development namespace")
    return path


def p6_source_hash(root: Path) -> tuple[dict, str]:
    names = ("src/ids/p6_selection.py", "scripts/run_p6_selection.py",
             "src/ids/p5_training.py", "scripts/train_p5_development.py",
             "src/ids/p4_model_interface.py")
    hashes = {name: file_hash(root / name) for name in names}
    return hashes, stable_hash(hashes)


def require_p6_role(operation: str, role: str) -> None:
    mapping = {
        "centroid_fit": ("train",),
        "scorer_selection": ("meta_known", "surrogate_ood"),
        "threshold_fit": ("calibration",),
        "development_diagnostic": ("meta_known", "surrogate_ood", "calibration"),
    }
    if operation not in mapping or role not in mapping[operation]:
        raise PermissionError(f"P6 forbids {operation} consuming {role}")


def verify_upstream(root: Path) -> dict:
    p5_dir = root / "results/training_protocol/p5"
    matrix = json.loads((p5_dir / "p5_matrix.json").read_text())
    if matrix["status"] != "P5_SMOKE_PASS" or len(matrix["cells"]) != 15:
        raise ValueError("P5_SMOKE_PASS with 15 cells is required")
    if matrix["p3_collection_sha256"] != P3_HASH:
        raise ValueError("P5/P3 collection mismatch")
    if matrix["training_config_sha256"] != stable_hash(TRAINING_CONFIG):
        raise ValueError("P5 training config hash mismatch")
    hashes, digest = p5_source_hash(root)
    if matrix["source_hash"] != digest or matrix["source_hashes"] != hashes:
        raise ValueError("frozen P5 source hash mismatch")
    config = json.loads((p5_dir / "training_config.json").read_text())
    scorer = json.loads((p5_dir / "scorer_contract.json").read_text())
    threshold = json.loads((p5_dir / "threshold_contract.json").read_text())
    if config != TRAINING_CONFIG:
        raise ValueError("frozen P5 optimizer contract mismatch")
    if tuple(item["id"] for item in scorer["candidates"]) != SCORERS:
        raise ValueError("frozen P5 scorer candidate set mismatch")
    if scorer["selection_tie_break"] != "nearest_centroid_l2 on exact tie":
        raise ValueError("frozen P5 tie preference mismatch")
    if scorer["selection_roles"] != ["meta_known", "surrogate_ood"]:
        raise ValueError("frozen P5 scorer roles mismatch")
    if (threshold["fit_role"] != "calibration" or threshold["quantile"] != QUANTILE
            or threshold["numpy_method"] != QUANTILE_METHOD
            or threshold["known_only"] is not True):
        raise ValueError("frozen P5 calibration contract mismatch")
    if matrix["official_test_access"] is not False:
        raise ValueError("P5 matrix indicates official-test access")
    return matrix


def validate_p6_cell(root: Path, target: str, split_seed: int) -> dict:
    verify_upstream(root)
    dry = validate_cell(root, target, split_seed, 10000 + split_seed)
    source_files, source_digest = p6_source_hash(root)
    return {"target": target, "split_seed": split_seed,
            "model_seed": 10000 + split_seed,
            "p3_manifest_sha256": dry["p3_manifest_sha256"],
            "p4_model_config_sha256": dry["p4_model_config_sha256"],
            "p5_training_config_sha256": dry["training_config_sha256"],
            "p5_source_hash": dry["source_hash"],
            "p6_source_hash": source_digest, "p6_source_files": source_files,
            "role_counts": dry["roles"], "selection_unit": SELECTION_UNIT,
            "scorer_candidates": list(SCORERS), "official_test_access": False,
            "target_held_out_access": False, "optimizer_created": False,
            "status": "DRY_RUN_PASS"}


def load_manifest(root: Path, target: str, split_seed: int) -> dict:
    dry = validate_p6_cell(root, target, split_seed)
    name = f"{target.lower()}_seed{split_seed}.json.gz"
    path = root / "results/data_quality/p3/manifests" / name
    raw = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw).hexdigest() != dry["p3_manifest_sha256"]:
        raise ValueError("P3 manifest hash mismatch")
    return json.loads(raw)


def checked_train_source(root: Path, manifest: dict, path: Path | None = None) -> Path:
    expected = root / "data" / DEFAULT_TRAIN_FILE
    source = expected if path is None else path
    if source.resolve() != expected.resolve() or source.name != DEFAULT_TRAIN_FILE:
        raise PermissionError("P6 may open only the frozen official-training CSV")
    if source.is_symlink() or not source.is_file():
        raise PermissionError("P6 source must be a regular official-training CSV")
    if file_hash(source) != manifest["source_dataset_hashes"]["official_train_sha256"]:
        raise ValueError("official-training CSV hash mismatch")
    return source


def prepare_context(root: Path, manifest: dict) -> tuple[dict, dict]:
    source = checked_train_source(root, manifest)
    frame = load_unsw_csvs(source.parent, files=[source], source_role="official_train")
    context = train_only_context(frame)
    if context["feature_schema_sha256"] != manifest["source_feature_schema_sha256"]:
        raise ValueError("P3 source schema mismatch")
    schema = json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())
    if schema["sha256"] != manifest["feature_schema_sha256"]:
        raise ValueError("P3 schema mismatch")
    return context, schema


def materialize_role(context: dict, schema: dict, manifest: dict,
                     role: str) -> dict:
    if role not in DEVELOPMENT_ROLES:
        raise PermissionError(f"P6 cannot materialize role {role}")
    allowed_operation = {"train": "gradient_fit", "val": "checkpoint_selection",
                         "meta_known": "scorer_family_selection",
                         "surrogate_ood": "scorer_family_selection",
                         "calibration": "threshold_fit"}[role]
    require_role(allowed_operation, role)
    frozen = manifest["roles"][role]
    row_ids = frozen["row_ids"]
    selected = context["train"].iloc[row_ids]
    if selected[ROW_ID_COLUMN].tolist() != row_ids or set(selected[SOURCE_ROLE_COLUMN]) != {"official_train"}:
        raise ValueError(f"{role} row/source mismatch")
    identities = [str(x) for x in selected[CANONICAL_FINGERPRINT_COLUMN]]
    if identities != frozen["canonical_identity_hashes"]:
        raise ValueError(f"{role} canonical identity mismatch")
    labels = selected["attack_cat"].tolist()
    if dict(selected["attack_cat"].value_counts()) != frozen["label_counts"]:
        raise ValueError(f"{role} label counts mismatch")
    if role == "surrogate_ood":
        if any(label not in ZERO_DAY_ATTACK_CATS or label == manifest["target"] for label in labels):
            raise ValueError("surrogate composition or target isolation mismatch")
    elif any(label not in KNOWN_ATTACK_CATS for label in labels):
        raise ValueError(f"{role} contains non-known label")
    raw = continuous_matrix(selected, context, schema)
    center = np.asarray(manifest["scaler_provenance"]["scaler_center"], dtype=np.float64)
    scale = np.asarray(manifest["scaler_provenance"]["scaler_scale"], dtype=np.float64)
    scaled = np.asarray((raw - center) / scale, dtype=np.float64)
    adapter = StructuredBatchAdapter(vocabularies_from_manifest(manifest))
    batch = adapter.transform(scaled, {name: selected[name].tolist() for name in CATEGORICAL_FIELDS})
    targets = (torch.tensor([KNOWN_ATTACK_CATS.index(x) for x in labels], dtype=torch.long)
               if role != "surrogate_ood" else None)
    return {"role": role, "batch": batch, "targets": targets,
            "labels": labels, "row_ids": row_ids, "identities": identities,
            "rows": len(row_ids), "identity_count": len(set(identities))}


def train_frozen_p5(model: TrainingOnlyModel, train: dict, val: dict,
                    model_seed: int) -> dict:
    require_role("gradient_fit", train["role"])
    require_role("checkpoint_selection", val["role"])
    if any(parameter.dtype != torch.float64 for parameter in model.parameters()):
        raise TypeError("P6 must preserve P5 float64 parameters")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_CONFIG["learning_rate"],
                                  weight_decay=TRAINING_CONFIG["weight_decay"])
    size = TRAINING_CONFIG["batch_size"]
    history, best_loss, best_epoch, best_state, stale = [], float("inf"), 0, None, 0
    for epoch in range(1, TRAINING_CONFIG["max_epochs"] + 1):
        model.train()
        for indices in np.array_split(batch_order(train["rows"], model_seed, epoch),
                                      range(size, train["rows"], size)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(select_batch(train["batch"], indices))
            loss = F.cross_entropy(logits, train["targets"][indices])
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite P5 training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG["gradient_clip_norm"])
            optimizer.step()
        model.eval()
        total = 0.0
        with torch.no_grad():
            for start in range(0, val["rows"], size):
                indices = np.arange(start, min(start + size, val["rows"]))
                logits = model(select_batch(val["batch"], indices))
                total += F.cross_entropy(logits, val["targets"][indices], reduction="sum").item()
        value = total / val["rows"]
        if not np.isfinite(value):
            raise ValueError("nonfinite P5 validation loss")
        history.append({"epoch": epoch, "development_val_cross_entropy": value})
        if value < best_loss - TRAINING_CONFIG["min_delta"]:
            best_loss, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= TRAINING_CONFIG["patience"]:
            break
    if best_state is None:
        raise AssertionError("no P5 checkpoint selected")
    model.load_state_dict(best_state)
    return {"state_dict": best_state, "selected_epoch": best_epoch,
            "validation_selection_value": best_loss, "epoch_history": history,
            "optimizer_steps": len(history) * ((train["rows"] + size - 1) // size),
            "checkpoint_state_sha256": _state_digest(best_state)}


def load_p5_pilot_checkpoint(root: Path, tag: str, dry: dict,
                             model: TrainingOnlyModel) -> dict:
    if tag not in {"smoke-a", "smoke-b"}:
        raise ValueError("pilot tag must be one frozen P5 smoke run")
    name = f"p5-fuzzers-s42-m10042-{tag}"
    path = root / "results/training_protocol/p5/training_runs" / f"{name}.json"
    run = json.loads(path.read_text())
    checkpoint = root / run["checkpoint_path"]
    if file_hash(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError("P5 pilot checkpoint file hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert_checkpoint_provenance(payload, run, {
        "p3_manifest_sha256": dry["p3_manifest_sha256"],
        "p4_model_config_sha256": dry["p4_model_config_sha256"],
        "training_config_sha256": dry["p5_training_config_sha256"],
        "vocabulary_sha256": validate_cell(root, "Fuzzers", 42, 10042)["vocabulary_sha256"],
        "scaler_sha256": validate_cell(root, "Fuzzers", 42, 10042)["scaler_sha256"],
        "source_hash": dry["p5_source_hash"],
    })
    if _state_digest(payload["state_dict"]) != run["checkpoint_state_sha256"]:
        raise ValueError("P5 pilot checkpoint state hash mismatch")
    model.load_state_dict(payload["state_dict"])
    return {"state_dict": payload["state_dict"], "selected_epoch": run["selected_epoch"],
            "validation_selection_value": run["validation_selection_value"],
            "epoch_history": run["epoch_history"], "optimizer_steps": run["optimizer_steps"],
            "checkpoint_state_sha256": run["checkpoint_state_sha256"],
            "checkpoint_file_sha256": run["checkpoint_sha256"],
            "checkpoint_path": run["checkpoint_path"], "p5_run_id": name}


def representations_and_logits(model: TrainingOnlyModel, role: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    reps, logits = [], []
    with torch.no_grad():
        for start in range(0, role["rows"], TRAINING_CONFIG["batch_size"]):
            indices = np.arange(start, min(start + TRAINING_CONFIG["batch_size"], role["rows"]))
            batch = select_batch(role["batch"], indices)
            rep = model.backbone(batch)
            output = model.training_head(rep)
            if rep.dtype != torch.float64 or output.dtype != torch.float64:
                raise TypeError("P6 representation/logit dtype must be float64")
            reps.append(rep.numpy())
            logits.append(output.numpy())
    return np.concatenate(reps), np.concatenate(logits)


def softmax_uncertainty(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != len(KNOWN_ATTACK_CATS):
        raise TypeError("softmax scorer requires float64[rows, known_classes]")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite softmax logits")
    stable = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(stable)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return np.asarray(1.0 - probabilities.max(axis=1), dtype=np.float64)


def fit_centroids(representations: np.ndarray, labels: list[str], *, role: str,
                  checkpoint_state_hash: str) -> tuple[np.ndarray, str]:
    require_p6_role("centroid_fit", role)
    values = np.asarray(representations)
    if values.dtype != np.float64 or values.ndim != 2 or values.shape[1] != 32:
        raise TypeError("centroids require float64[rows,32]")
    if len(labels) != len(values) or not np.isfinite(values).all():
        raise ValueError("invalid centroid rows")
    centroids = []
    for label in KNOWN_ATTACK_CATS:
        selected = values[np.asarray(labels) == label]
        if len(selected) == 0:
            raise ValueError(f"missing train known class: {label}")
        centroids.append(selected.mean(axis=0, dtype=np.float64))
    state = np.ascontiguousarray(np.stack(centroids).astype("<f8"))
    digest = centroid_state_hash(state, checkpoint_state_hash)
    return state, digest


def centroid_state_hash(centroids: np.ndarray, checkpoint_state_hash: str) -> str:
    state = np.asarray(centroids)
    if state.dtype != np.float64 or state.shape != (len(KNOWN_ATTACK_CATS), 32):
        raise TypeError("centroid state must be float64[known_classes,32]")
    state = np.ascontiguousarray(state.astype("<f8"))
    header = {"scorer_type": "nearest_centroid_l2", "class_order": list(KNOWN_ATTACK_CATS),
              "representation_dim": 32, "checkpoint_state_sha256": checkpoint_state_hash,
              "formula_version": "minimum-euclidean-distance-v1"}
    return hashlib.sha256(canonical_json(header) + state.tobytes()).hexdigest()


def centroid_distance(representations: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    values = np.asarray(representations)
    centers = np.asarray(centroids)
    if (values.dtype != np.float64 or centers.dtype != np.float64
            or values.ndim != 2 or values.shape[1] != 32
            or centers.shape != (len(KNOWN_ATTACK_CATS), 32)):
        raise TypeError("centroid score requires float64 representation/centroid shapes")
    if not np.isfinite(values).all() or not np.isfinite(centers).all():
        raise ValueError("nonfinite centroid scorer input")
    return np.asarray(np.linalg.norm(values[:, None, :] - centers[None, :, :], axis=2).min(axis=1),
                      dtype=np.float64)


def softmax_state_hash(checkpoint_state_hash: str) -> str:
    return stable_hash({"scorer_type": "negative_max_softmax",
                        "checkpoint_state_sha256": checkpoint_state_hash,
                        "class_order": list(KNOWN_ATTACK_CATS),
                        "formula_version": "one-minus-max-softmax-v1"})


def verify_scorer_states(states: dict, checkpoint_state_hash: str) -> np.ndarray:
    if states["checkpoint_state_sha256"] != checkpoint_state_hash:
        raise ValueError("scorer/checkpoint state mismatch")
    items = states["states"]
    if set(items) != set(SCORERS) or items["negative_max_softmax"]["scorer_state_sha256"] != softmax_state_hash(checkpoint_state_hash):
        raise ValueError("softmax scorer state mismatch")
    centroids = np.asarray(items["nearest_centroid_l2"]["centroids_float64"], dtype=np.float64)
    if items["nearest_centroid_l2"]["scorer_state_sha256"] != centroid_state_hash(centroids, checkpoint_state_hash):
        raise ValueError("centroid scorer state mismatch")
    return centroids


def identity_means(scores: np.ndarray, identities: list[str]) -> tuple[list[str], np.ndarray]:
    values = np.asarray(scores)
    if values.dtype != np.float64 or values.ndim != 1 or len(values) != len(identities):
        raise TypeError("identity aggregation requires aligned float64 row scores")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite identity scores")
    grouped: dict[str, list[float]] = {}
    for identity, score in zip(identities, values, strict=True):
        grouped.setdefault(str(identity), []).append(float(score))
    keys = sorted(grouped)
    means = np.asarray([math.fsum(sorted(grouped[key])) / len(grouped[key]) for key in keys],
                       dtype=np.float64)
    return keys, means


def development_auroc(known: np.ndarray, surrogate: np.ndarray) -> float:
    left, right = np.asarray(known), np.asarray(surrogate)
    if left.dtype != np.float64 or right.dtype != np.float64 or not len(left) or not len(right):
        raise TypeError("development AUROC requires nonempty float64 known/surrogate scores")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("nonfinite development AUROC score")
    labels = np.concatenate((np.zeros(len(left), dtype=np.int8), np.ones(len(right), dtype=np.int8)))
    scores = np.concatenate((left, right))
    return float(roc_auc_score(labels, scores))


def choose_scorer(softmax_auroc: float, centroid_auroc: float) -> dict:
    if not np.isfinite(softmax_auroc) or not np.isfinite(centroid_auroc):
        raise ValueError("nonfinite scorer-selection statistic")
    margin = centroid_auroc - softmax_auroc
    tie = abs(margin) <= TIE_TOLERANCE
    selected = TIE_PREFERENCE if tie or margin > 0 else "negative_max_softmax"
    return {"selected_scorer": selected, "selection_margin_centroid_minus_softmax": margin,
            "tie_break_used": tie, "tie_tolerance": TIE_TOLERANCE,
            "tie_preference": TIE_PREFERENCE}


def score_summary(scores: np.ndarray) -> dict:
    values = np.asarray(scores, dtype=np.float64)
    return {"count": len(values), "min": float(values.min()), "max": float(values.max()),
            "mean": float(values.mean())}


def select_scorer(model: TrainingOnlyModel, roles: dict, checkpoint_state_hash: str) -> tuple[dict, dict, np.ndarray]:
    train, meta, surrogate = (roles[name] for name in ("train", "meta_known", "surrogate_ood"))
    require_p6_role("centroid_fit", train["role"])
    require_p6_role("scorer_selection", meta["role"])
    require_p6_role("scorer_selection", surrogate["role"])
    train_reps, _ = representations_and_logits(model, train)
    centroids, centroid_hash = fit_centroids(train_reps, train["labels"], role="train",
                                            checkpoint_state_hash=checkpoint_state_hash)
    candidate_scores = {}
    summary = {}
    for role in (meta, surrogate):
        reps, logits = representations_and_logits(model, role)
        candidate_scores[role["role"]] = {
            "negative_max_softmax": softmax_uncertainty(logits),
            "nearest_centroid_l2": centroid_distance(reps, centroids),
        }
    for scorer in SCORERS:
        meta_rows = candidate_scores["meta_known"][scorer]
        surrogate_rows = candidate_scores["surrogate_ood"][scorer]
        meta_ids, meta_identity = identity_means(meta_rows, meta["identities"])
        surrogate_ids, surrogate_identity = identity_means(surrogate_rows, surrogate["identities"])
        if len(meta_ids) != meta["identity_count"] or len(surrogate_ids) != surrogate["identity_count"]:
            raise ValueError("identity aggregation count mismatch")
        summary[scorer] = {
            "surrogate_development_identity_auroc": development_auroc(meta_identity, surrogate_identity),
            "surrogate_development_row_auroc_diagnostic": development_auroc(meta_rows, surrogate_rows),
            "meta_known_rows": score_summary(meta_rows),
            "meta_known_identities": score_summary(meta_identity),
            "surrogate_ood_rows": score_summary(surrogate_rows),
            "surrogate_ood_identities": score_summary(surrogate_identity),
        }
    choice = choose_scorer(summary["negative_max_softmax"]["surrogate_development_identity_auroc"],
                           summary["nearest_centroid_l2"]["surrogate_development_identity_auroc"])
    states = {
        "version": "p6-scorer-states-v1", "checkpoint_state_sha256": checkpoint_state_hash,
        "class_order": list(KNOWN_ATTACK_CATS), "representation_dim": 32,
        "states": {
            "negative_max_softmax": {"scorer_state_sha256": softmax_state_hash(checkpoint_state_hash),
                                     "fitted_parameters": False},
            "nearest_centroid_l2": {"scorer_state_sha256": centroid_hash,
                                    "fit_role": "train", "centroids_float64": centroids.tolist()},
        },
    }
    selection = {"version": "p6-scorer-selection-v1", "selection_unit": SELECTION_UNIT,
                 "selection_statistic": "SURROGATE DEVELOPMENT AUROC",
                 "known_label": 0, "surrogate_label": 1,
                 "score_orientation": "higher means more OOD-like",
                 "meta_known_identity_count": meta["identity_count"],
                 "surrogate_identity_count": surrogate["identity_count"],
                 "candidate_statistics": summary, **choice,
                 "interpretation": ("Surrogate-based scorer selection estimates development discrimination "
                                    "against the predefined surrogate unknown families only. "
                                    "It does not establish performance on the held-out target family."),
                 "development_diagnostic_only": True}
    return states, selection, centroids


def fit_known_threshold(scores: np.ndarray, identities: list[str], *, role: str) -> dict:
    require_p6_role("threshold_fit", role)
    values = np.asarray(scores)
    if values.dtype != np.float64 or values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("known calibration scores must be nonempty finite float64")
    if len(identities) != len(values):
        raise ValueError("calibration identity alignment mismatch")
    threshold = np.quantile(values, QUANTILE, method=QUANTILE_METHOD)
    return {"calibration_sample_count": len(values),
            "calibration_identity_count": len(set(identities)),
            "minimum_score": float(values.min()), "maximum_score": float(values.max()),
            "quantile": QUANTILE, "quantile_method": QUANTILE_METHOD,
            "threshold": float(threshold), "comparison": "score > threshold",
            "known_only_operating_point": True}


def alert_mask(scores: np.ndarray, threshold: float) -> np.ndarray:
    values = np.asarray(scores)
    if values.dtype != np.float64 or not np.isfinite(values).all() or not np.isfinite(threshold):
        raise ValueError("alert decision requires finite float64 scores/threshold")
    return values > threshold


def scorer_scores(model: TrainingOnlyModel, role: dict, selected: str,
                  centroids: np.ndarray) -> np.ndarray:
    if selected not in SCORERS:
        raise ValueError("unapproved scorer")
    reps, logits = representations_and_logits(model, role)
    return softmax_uncertainty(logits) if selected == "negative_max_softmax" else centroid_distance(reps, centroids)


def calibration_artifact(root: Path, dry: dict, role: dict, selected: str,
                         scorer_state_hash: str, checkpoint_state_hash: str,
                         scores: np.ndarray) -> dict:
    details = fit_known_threshold(scores, role["identities"], role=role["role"])
    return {"version": "p6-known-only-calibration-v1",
            "target": dry["target"], "split_seed": dry["split_seed"],
            "model_seed": dry["model_seed"],
            "p3_manifest_sha256": dry["p3_manifest_sha256"],
            "p4_model_config_sha256": dry["p4_model_config_sha256"],
            "p5_training_config_sha256": dry["p5_training_config_sha256"],
            "checkpoint_state_sha256": checkpoint_state_hash,
            "selected_scorer": selected, "scorer_state_sha256": scorer_state_hash,
            "calibration_row_ids_sha256": stable_hash(role["row_ids"]),
            "calibration_identity_sha256": stable_hash(role["identities"]),
            "source_hash": dry["p6_source_hash"], **details,
            "official_test_access": False, "target_held_out_access": False}


def verify_calibration_artifact(artifact: dict, dry: dict, role: dict,
                                checkpoint_state_hash: str, scorer_state_hash: str) -> None:
    expected = {
        "target": dry["target"], "split_seed": dry["split_seed"],
        "model_seed": dry["model_seed"],
        "p3_manifest_sha256": dry["p3_manifest_sha256"],
        "p4_model_config_sha256": dry["p4_model_config_sha256"],
        "p5_training_config_sha256": dry["p5_training_config_sha256"],
        "checkpoint_state_sha256": checkpoint_state_hash,
        "scorer_state_sha256": scorer_state_hash,
        "calibration_row_ids_sha256": stable_hash(role["row_ids"]),
        "calibration_identity_sha256": stable_hash(role["identities"]),
        "source_hash": dry["p6_source_hash"],
        "quantile": QUANTILE, "quantile_method": QUANTILE_METHOD,
        "comparison": "score > threshold",
    }
    if any(artifact.get(key) != value for key, value in expected.items()):
        raise ValueError("P6 calibration provenance mismatch")


def development_diagnostics(model: TrainingOnlyModel, roles: dict, selected: str,
                            centroids: np.ndarray, threshold: float) -> dict:
    output = {"label": "DEVELOPMENT_DIAGNOSTIC_ONLY", "used_for_fitting_or_selection": False,
              "selected_scorer": selected, "threshold": threshold, "roles": {}}
    for name in ("meta_known", "surrogate_ood", "calibration"):
        role = roles[name]
        require_p6_role("development_diagnostic", role["role"])
        scores = scorer_scores(model, role, selected, centroids)
        _, identity_scores = identity_means(scores, role["identities"])
        output["roles"][name] = {
            "row_score_summary": score_summary(scores),
            "identity_score_summary": score_summary(identity_scores),
            "row_alert_fraction": float(alert_mask(scores, threshold).mean()),
            "identity_alert_fraction": float(alert_mask(identity_scores, threshold).mean()),
            "row_count": role["rows"], "identity_count": role["identity_count"],
        }
    return output
