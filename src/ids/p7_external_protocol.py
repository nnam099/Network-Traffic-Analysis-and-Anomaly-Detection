"""Frozen P7 preregistration, synthetic metric logic and one-way gate helpers.

Importing this module never loads official test. The only test-file reader is
in the separately gated Stage-B command.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from .dataset import KNOWN_ATTACK_CATS
from .p5_training import (P3_HASH, SPLIT_SEEDS, TARGETS, _state_digest, file_hash, json_bytes,
                          stable_hash, validate_cell, write_immutable)
from .p6_selection import (identity_means, validate_p6_cell, verify_scorer_states,
                           verify_calibration_artifact)


P7_VERSION = "p7-external-preregistration-v1"
SMALL_N = ("Analysis", "Backdoors", "Worms")
SURFACES = ("CLEAN", "DEVELOPMENT-OVERLAP", "TEST-INTERNAL-AMBIGUOUS",
            "TRAIN-CONFLICT-RELATED")
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CONFIDENCE = 0.95
BOOTSTRAP_MIN_VALID = 1900
BOOTSTRAP_SEED_BASE = 700000


def p7_path(root: Path, relative: str) -> Path:
    base = (root / "results/external_evaluation/p7").resolve()
    path = (base / relative).resolve()
    if not path.is_relative_to(base):
        raise PermissionError("P7 may write only its external-evaluation namespace")
    return path


def source_hashes(root: Path) -> tuple[dict[str, str], str]:
    names = ("src/ids/p7_external_protocol.py",
             "scripts/freeze_p7_external_protocol.py",
             "scripts/run_p7_external_evaluation.py",
             "tests/test_p7_external_protocol.py")
    hashes = {name: file_hash(root / name) for name in names}
    return hashes, stable_hash(hashes)


def _tree_hashes(root: Path, directory: str) -> dict[str, str]:
    base = root / directory
    return {str(path.relative_to(root)): file_hash(path)
            for path in sorted(base.rglob("*")) if path.is_file()}


def upstream_snapshot(root: Path) -> dict:
    """Hash only P3 development artifacts; never read P3 test audit artifacts."""
    p3 = root / "results/data_quality/p3"
    freeze = json.loads((p3 / "development_freeze.json").read_text())
    if stable_hash(freeze) != P3_HASH or freeze["official_test_consulted"] is not False:
        raise ValueError("P3 frozen collection mismatch")
    p3_names = ["development_freeze.json", "schema_v2.json", "train_only_ambiguity.json"]
    p3_names.extend(entry["manifest_filename"] for entry in freeze["cells"])
    p3_hashes = {}
    for name in p3_names:
        path = p3 / name
        if not path.resolve().is_relative_to(p3.resolve()):
            raise PermissionError("P3 development artifact path escapes namespace")
        p3_hashes[str(path.relative_to(root))] = file_hash(path)
    for entry in freeze["cells"]:
        raw = gzip.decompress((p3 / entry["manifest_filename"]).read_bytes())
        if hashlib.sha256(raw).hexdigest() != entry["manifest_sha256"]:
            raise ValueError("P3 cell manifest hash mismatch")
    p4 = json.loads((root / "results/model_design/p4/p4_matrix.json").read_text())
    p5 = json.loads((root / "results/training_protocol/p5/p5_matrix.json").read_text())
    p6 = json.loads((root / "results/model_selection/p6/p6_matrix.json").read_text())
    if (p4["status"] != "MODEL_INTERFACE_PASS" or p5["status"] != "P5_SMOKE_PASS"
            or p6["status"] != "P6_DEVELOPMENT_PASS" or len(p6["cells"]) != 15):
        raise ValueError("P4/P5/P6 gate not complete")
    for target in TARGETS:
        for seed in SPLIT_SEEDS:
            validate_cell(root, target, seed, 10000 + seed)
            verify_frozen_cell(root, target, seed)
    # Bind the previously frozen P3 external audit by file hash only. We do
    # not parse it or load official test during Stage A.
    prior_audit_hash = file_hash(p3 / "official_test_external_audit.json")
    groups = {
        "p3_development": p3_hashes,
        "p3_prior_external_audit_file_sha256": prior_audit_hash,
        "p4": _tree_hashes(root, "results/model_design/p4"),
        "p5": _tree_hashes(root, "results/training_protocol/p5"),
        "p6": _tree_hashes(root, "results/model_selection/p6"),
    }
    if any("official_test_external_audit" in name or "historical_738" in name
           or "p3_matrix.json" in name for name in groups["p3_development"]):
        raise AssertionError("Stage A attempted to bind a post-freeze test artifact")
    return {"groups": groups, "aggregate_sha256": stable_hash(groups),
            "p3_collection_sha256": P3_HASH,
            "p4_status": p4["status"], "p5_status": p5["status"], "p6_status": p6["status"]}


def verify_frozen_cell(root: Path, target: str, seed: int) -> dict:
    """Verify the exact P6 detector without reading any external-test artifact."""
    dry = validate_p6_cell(root, target, seed)
    prefix = root / f"results/model_selection/p6/runs/{target.lower()}/seed_{seed}"
    run = json.loads((prefix / "training_manifest.json").read_text())
    reference = json.loads((prefix / "checkpoint_reference.json").read_text())
    states = json.loads((prefix / "scorer_states.json").read_text())
    selection = json.loads((prefix / "scorer_selection.json").read_text())
    calibration_path = prefix / "calibration.json"
    calibration = json.loads(calibration_path.read_text())
    checkpoint = root / reference["path"]
    if not checkpoint.resolve().is_relative_to((root / "results/model_selection/p6").resolve()):
        raise PermissionError("P6 checkpoint reference escapes frozen namespace")
    if file_hash(checkpoint) != reference["checkpoint_file_sha256"]:
        raise ValueError("P6 checkpoint file hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if _state_digest(payload["state_dict"]) != run["checkpoint_state_sha256"]:
        raise ValueError("P6 checkpoint state hash mismatch")
    verify_scorer_states(states, run["checkpoint_state_sha256"])
    if selection["selected_scorer"] not in states["states"]:
        raise ValueError("P6 selected scorer missing state")
    if (run["p3_manifest_sha256"] != dry["p3_manifest_sha256"]
            or run["p4_model_config_sha256"] != dry["p4_model_config_sha256"]
            or run["p5_training_config_sha256"] != dry["p5_training_config_sha256"]
            or selection["checkpoint_state_sha256"] != run["checkpoint_state_sha256"]):
        raise ValueError("P6 upstream binding mismatch")
    manifest_path = root / f"results/data_quality/p3/manifests/{target.lower()}_seed{seed}.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    role = {"row_ids": manifest["roles"]["calibration"]["row_ids"],
            "identities": manifest["roles"]["calibration"]["canonical_identity_hashes"]}
    matrix = json.loads((root / "results/model_selection/p6/p6_matrix.json").read_text())
    cell = next((item for item in matrix["cells"] if item["target"] == target
                 and item["split_seed"] == seed), None)
    assert_detector_binding(dry, run, reference, states, selection, calibration,
                            role, cell, actual_file_hash=file_hash(checkpoint),
                            actual_state_hash=_state_digest(payload["state_dict"]))
    return {"dry": dry, "run": run, "reference": reference, "states": states,
            "selection": selection, "calibration": calibration,
            "calibration_artifact_sha256": file_hash(calibration_path)}


def assert_detector_binding(dry: dict, run: dict, reference: dict, states: dict,
                            selection: dict, calibration: dict, role: dict,
                            matrix_cell: dict | None, *, actual_file_hash: str,
                            actual_state_hash: str) -> None:
    if (actual_file_hash != reference["checkpoint_file_sha256"]
            or actual_state_hash != reference["checkpoint_state_sha256"]
            or actual_state_hash != run["checkpoint_state_sha256"]):
        raise ValueError("checkpoint hash/binding mismatch")
    verify_scorer_states(states, actual_state_hash)
    selected = selection["selected_scorer"]
    if selected not in states["states"] or calibration["selected_scorer"] != selected:
        raise ValueError("selected scorer/calibration mismatch")
    scorer_hash = states["states"][selected]["scorer_state_sha256"]
    verify_calibration_artifact(calibration, dry, role, actual_state_hash, scorer_hash)
    if (matrix_cell is None or matrix_cell["status"] != "DEVELOPMENT_CELL_PASS"
            or matrix_cell["checkpoint_state_hash"] != actual_state_hash
            or matrix_cell["selected_scorer"] != selected
            or matrix_cell["threshold"] != calibration["threshold"]):
        raise ValueError("P6 matrix does not bind cell artifacts")


def metric_contract() -> dict:
    return {
        "version": "p7-identity-metrics-v1", "primary_scientific_question": (
            "For each frozen target x seed detector, distinguish CLEAN official-test "
            "canonical identities of that target from CLEAN known/background identities."),
        "positive": "CLEAN identities of the cell's true target family only",
        "negative": "CLEAN identities of the five frozen known/background families only",
        "other_ood_families_in_primary": False,
        "primary_unit": "canonical_identity", "identity_score": "arithmetic mean of row scores",
        "identity_label": "unique already-audited CLEAN attack_cat; conflict fails",
        "main_threshold_independent_metric": "identity AUROC",
        "other_threshold_independent_metrics": ["identity Average Precision"],
        "frozen_operating_point_metrics": ["TPR", "FPR", "specificity", "precision", "F1",
                                           "balanced_accuracy", "TP", "FP", "TN", "FN"],
        "row_level_diagnostics": ["AUROC", "Average Precision", "TPR", "FPR"],
        "row_level_label": "ROW-LEVEL DIAGNOSTIC",
        "score_orientation": "higher is more OOD-like",
        "threshold_decision": "score > frozen P6 threshold",
        "precision_zero_division": 0.0, "f1_zero_division": 0.0,
        "seed_handling": "three distinct cells; descriptive mean, sample SD, min, max; no raw pooling",
        "no_pooled_cross_target_metric": True,
    }


def uncertainty_contract() -> dict:
    return {"version": "p7-stratified-identity-bootstrap-v1",
            "method": "stratified bootstrap separately within positive/negative canonical identities",
            "replicates": BOOTSTRAP_REPLICATES, "confidence_level": BOOTSTRAP_CONFIDENCE,
            "rng": "NumPy PCG64", "seed_rule": "700000 + 100*target_index + split_seed",
            "target_order": list(TARGETS), "seed_order": list(SPLIT_SEEDS),
            "interval_quantile_method": "linear", "interval_tails": [0.025, 0.975],
            "metrics": ["AUROC", "Average Precision", "TPR", "FPR"],
            "invalid_replicate": "discard only mathematically undefined metric; count and report",
            "minimum_valid_replicates": BOOTSTRAP_MIN_VALID,
            "failure_if_fewer_valid": True,
            "small_n_caution": (
                "A finite confidence interval does not imply adequate sample size or "
                "broad population representativeness."),
            }


def surface_contract() -> dict:
    return {"version": "p7-test-surfaces-v1", "primary": "CLEAN",
            "primary_presentation_name": "official-test-clean",
            "diagnostic_only": {
                "DEVELOPMENT-OVERLAP": "CONTAMINATED DIAGNOSTIC — NOT EXTERNAL",
                "TEST-INTERNAL-AMBIGUOUS": "AMBIGUOUS-LABEL DIAGNOSTIC; descriptive scores only",
                "TRAIN-CONFLICT-RELATED": "TRAIN-CONFLICT DIAGNOSTIC",
            },
            "diagnostic_order": ["DEVELOPMENT-OVERLAP", "TEST-INTERNAL-AMBIGUOUS",
                                 "TRAIN-CONFLICT-RELATED"],
            "diagnostics_allowed_only_after_primary_hash_freeze": True,
            "merge_diagnostics_into_primary": False,
            "small_n_targets": list(SMALL_N),
            "small_n_label": "SMALL-N EXTERNAL TARGET",
            "small_n_support_from_prior_audit": {"Fuzzers": 4317, "Analysis": 58,
                                                 "Backdoors": 57, "Shellcode": 364,
                                                 "Worms": 43},
            "small_n_interpretation": (
                "Uncertainty summaries are not proof of statistical power or "
                "broad population representativeness."),
            }


def result_schema() -> dict:
    return {"version": "p7-external-result-schema-v1",
            "primary_per_cell_files": ["external_counts.json", "primary_identity_metrics.json",
                                       "row_diagnostics.json", "confusion.json",
                                       "oov_audit.json", "provenance.json"],
            "primary_matrix": "primary/p7_primary_matrix.json",
            "primary_hash": "primary/primary_hash.txt",
            "post_primary_clean_score_summaries": "diagnostics/clean_score_summary/",
            "diagnostic_directories": ["development_overlap", "test_internal_ambiguous",
                                       "train_conflict_related"],
            "provenance_required": ["p3_collection_sha256", "p3_manifest_sha256",
                                    "p4_model_config_sha256", "p5_training_config_sha256",
                                    "checkpoint_file_sha256", "checkpoint_state_sha256",
                                    "p6_scorer_state_sha256", "p6_calibration_artifact_sha256",
                                    "p7_protocol_sha256", "official_test_source_sha256",
                                    "official_test_clean_surface_sha256", "source_code_sha256"],
            "status_on_failure": "P7_EXTERNAL_EVAL_BLOCKED",
            "status_on_complete": "P7_EXTERNAL_EVALUATION_COMPLETE"}


def frozen_contract(root: Path) -> dict:
    upstream = upstream_snapshot(root)
    source_files, source_digest = source_hashes(root)
    return {"version": P7_VERSION, "status": "P7_PREREGISTERED",
            "metric_contract": metric_contract(),
            "uncertainty_contract": uncertainty_contract(),
            "surface_contract": surface_contract(), "result_schema": result_schema(),
            "upstream_snapshot": upstream,
            "source_files_sha256": source_files, "source_code_sha256": source_digest,
            "official_test_loaded": False, "official_test_scores_computed": False,
            "stage_b_requires": ["--protocol-hash <exact frozen hash>",
                                 "--enable-official-test"],
            "one_way_policy": "atomic intent lock before official-test read; no rerun or upstream mutation",
            "interpretation_limit": (
                "Benchmark-specific external evaluation cannot alone establish universal "
                "zero-day effectiveness or deployment readiness."),
            }


def validate_preregistration(root: Path, supplied_hash: str) -> dict:
    if len(supplied_hash) != 64 or any(ch not in "0123456789abcdef" for ch in supplied_hash):
        raise PermissionError("exact lowercase 64-character protocol hash required")
    directory = p7_path(root, "preregistration")
    frozen = json.loads((directory / "external_protocol.json").read_text())
    actual = stable_hash(frozen)
    if actual != supplied_hash or (directory / "protocol_hash.txt").read_text().strip() != actual:
        raise PermissionError("P7 protocol hash mismatch")
    gate = json.loads((directory / "preregistration_gate.json").read_text())
    if gate["status"] != "P7_PREREGISTERED" or gate["protocol_sha256"] != actual:
        raise PermissionError("P7 Stage A gate not frozen")
    if frozen != frozen_contract(root):
        raise ValueError("P7 source/upstream/contract changed after preregistration")
    return frozen


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    values = np.asarray(scores)
    if values.dtype != np.float64 or values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("P7 requires finite float64 anomaly scores")
    return values


def binary_metrics(known_scores: np.ndarray, target_scores: np.ndarray,
                   threshold: float) -> dict:
    known, target = _finite_scores(known_scores), _finite_scores(target_scores)
    if not len(known) or not len(target) or not np.isfinite(threshold):
        raise ValueError("binary external task requires both classes and finite threshold")
    labels = np.concatenate((np.zeros(len(known), dtype=np.int8),
                             np.ones(len(target), dtype=np.int8)))
    scores = np.concatenate((known, target))
    auroc = float(roc_auc_score(labels, scores))
    ap = float(average_precision_score(labels, scores))
    known_alert, target_alert = known > threshold, target > threshold
    fp, tn = int(known_alert.sum()), int((~known_alert).sum())
    tp, fn = int(target_alert.sum()), int((~target_alert).sum())
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if precision + tpr else 0.0
    return {"AUROC": auroc, "Average_Precision": ap,
            "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "TPR": tpr, "FPR": fpr, "specificity": 1 - fpr,
            "precision": precision, "F1": f1,
            "balanced_accuracy": (tpr + 1 - fpr) / 2}


def _primary_frame(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in TARGETS:
        raise ValueError("undeclared target")
    required = {"surface", "canonical_identity", "family", "score"}
    if not required <= set(frame):
        raise ValueError("primary frame missing required columns")
    if len(frame) == 0 or set(frame["surface"].astype(str)) != {"CLEAN"}:
        raise PermissionError("primary evaluator rejects non-clean surface rows")
    families = set(frame["family"].astype(str))
    if not families <= (set(KNOWN_ATTACK_CATS) | {target}):
        raise PermissionError("primary evaluator rejects other OOD or unexpected labels")
    if any(frame.groupby("canonical_identity")["family"].nunique() != 1):
        raise ValueError("CLEAN canonical identity has conflicting labels")
    if frame["canonical_identity"].isna().any():
        raise ValueError("missing canonical identity")
    _finite_scores(frame["score"].to_numpy())
    return frame.copy()


def primary_metrics(frame: pd.DataFrame, target: str, threshold: float) -> dict:
    clean = _primary_frame(frame, target)
    positive = clean[clean.family == target]
    negative = clean[clean.family.isin(KNOWN_ATTACK_CATS)]
    if positive.empty or negative.empty:
        raise ValueError("primary target/known support is zero")
    pos_ids, pos_scores = identity_means(
        positive.score.to_numpy(dtype=np.float64), positive.canonical_identity.astype(str).tolist())
    neg_ids, neg_scores = identity_means(
        negative.score.to_numpy(dtype=np.float64), negative.canonical_identity.astype(str).tolist())
    if set(pos_ids) & set(neg_ids):
        raise ValueError("target/known identity intersection")
    counts = {"positive_rows": len(positive), "positive_identities": len(pos_ids),
              "known_rows": len(negative), "known_identities": len(neg_ids),
              "family_rows": clean.family.value_counts().sort_index().to_dict(),
              "family_identities": clean.groupby("family").canonical_identity.nunique().sort_index().to_dict()}
    identity = binary_metrics(neg_scores, pos_scores, threshold)
    row = binary_metrics(negative.score.to_numpy(dtype=np.float64),
                         positive.score.to_numpy(dtype=np.float64), threshold)
    row = {"label": "ROW-LEVEL DIAGNOSTIC", "metrics": row,
           "difference_from_identity": {
               key: row[key] - identity[key] for key in ("AUROC", "Average_Precision", "TPR", "FPR")}}
    return {"counts": counts, "identity_metrics": identity, "row_diagnostics": row,
            "identity_known_scores": neg_scores, "identity_target_scores": pos_scores}


def bootstrap_seed(target: str, split_seed: int) -> int:
    if target not in TARGETS or split_seed not in SPLIT_SEEDS:
        raise ValueError("unsupported bootstrap cell")
    return BOOTSTRAP_SEED_BASE + 100 * TARGETS.index(target) + split_seed


def bootstrap_intervals(known_scores: np.ndarray, target_scores: np.ndarray,
                        threshold: float, *, seed: int,
                        replicates: int = BOOTSTRAP_REPLICATES,
                        min_valid: int = BOOTSTRAP_MIN_VALID) -> dict:
    known, target = _finite_scores(known_scores), _finite_scores(target_scores)
    if not len(known) or not len(target) or replicates <= 0 or min_valid > replicates:
        raise ValueError("invalid bootstrap support/configuration")
    rng = np.random.default_rng(seed)
    collected = {key: [] for key in ("AUROC", "Average_Precision", "TPR", "FPR")}
    discarded = 0
    for _ in range(replicates):
        sampled_known = known[rng.integers(0, len(known), size=len(known))]
        sampled_target = target[rng.integers(0, len(target), size=len(target))]
        try:
            values = binary_metrics(sampled_known, sampled_target, threshold)
        except ValueError:
            discarded += 1
            continue
        for key in collected:
            collected[key].append(values[key])
    valid = replicates - discarded
    if valid < min_valid:
        raise ValueError("too few valid stratified identity-bootstrap replicates")
    return {"method": "stratified canonical-identity bootstrap",
            "rng": "NumPy PCG64", "seed": seed, "replicates": replicates,
            "valid_replicates": valid, "discarded_replicates": discarded,
            "confidence_level": BOOTSTRAP_CONFIDENCE,
            "intervals": {key: [float(x) for x in np.quantile(values, [0.025, 0.975], method="linear")]
                          for key, values in collected.items()}}


def score_summary(scores: np.ndarray) -> dict:
    values = _finite_scores(scores)
    if not len(values):
        raise ValueError("empty score summary")
    quantiles = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95], method="linear")
    return {"n": len(values), "min": float(values.min()), "median": float(quantiles[2]),
            "mean": float(values.mean()), "max": float(values.max()),
            "quantiles": {name: float(value) for name, value in zip(
                ("0.05", "0.25", "0.50", "0.75", "0.95"), quantiles, strict=True)}}


def across_seed_summary(cells: list[dict], target: str) -> dict:
    selected = [cell for cell in cells if cell["target"] == target]
    if sorted(cell["split_seed"] for cell in selected) != list(SPLIT_SEEDS):
        raise ValueError("target summary requires each frozen seed exactly once")
    metrics = ("AUROC", "Average_Precision", "TPR", "FPR", "specificity",
               "precision", "F1", "balanced_accuracy")
    return {"target": target, "seed_count": 3, "interpretation": "descriptive only; seeds not independent deployment samples",
            "metrics": {name: {"mean": float(np.mean(values)),
                                "sample_sd": float(np.std(values, ddof=1)),
                                "min": float(np.min(values)), "max": float(np.max(values))}
                        for name in metrics
                        for values in ([cell["identity_metrics"][name] for cell in selected],)}}
