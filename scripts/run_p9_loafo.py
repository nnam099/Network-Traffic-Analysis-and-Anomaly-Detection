#!/usr/bin/env python3
"""Execute frozen P9 dry, pilot, rerun, and development-only LOAFO stages."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p4_model_interface import CATEGORICAL_FIELDS, FrozenVocabulary  # noqa: E402
from ids.p8_loafo_protocol import P8KnownClassModel, file_sha256, json_bytes, write_immutable  # noqa: E402
from ids.p9_loafo import (  # noqa: E402
    P8_COMMIT, P8_PROTOCOL_HASH, P9_VERSION, SCORERS, SEEDS, TARGETS,
    TRAINING_CONFIG, centroid_distance, checked_input,
    choose_scorer, development_auroc, fit_centroids, fit_identity_threshold,
    identity_means, load_manifest, load_train_context, materialize_role,
    read_json, representations_logits, require_role, score_summary,
    set_determinism, softmax_uncertainty, source_hashes,
    state_hash, train_cell, verify_fold_fit, verify_manifest, verify_p8,
    selected_role_frame,
)
from ids.p3_protocol import sha256_json  # noqa: E402


OUT = ROOT / "results/loafo/p9"
PILOT_TARGET, PILOT_SEED = "Fuzzers", 42
RUN_FILES = ("training_manifest.json", "checkpoint.pt", "checkpoint_metadata.json",
             "scorer_state.json", "scorer_selection.json", "calibration.json",
             "development_diagnostics.json")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _p8_cell(p8: dict, target: str, seed: int) -> tuple[dict, dict, dict]:
    cell = next((item for item in p8["matrix"]["cells"]
                 if item["target"] == target and item["split_seed"] == seed), None)
    if cell is None:
        raise ValueError("cell absent from frozen P8 matrix")
    manifest = load_manifest(ROOT, cell)
    binding = verify_manifest(cell, manifest, p8)
    return cell, manifest, binding


def _source_binding() -> dict:
    files = source_hashes(ROOT)
    return {"source_code_sha256": files, "source_code_aggregate_sha256": sha256_json(files)}


def _protocol(p8: dict) -> dict:
    return {
        "version": P9_VERSION, "p8_repository_commit": P8_COMMIT,
        "p8_protocol_sha256": P8_PROTOCOL_HASH,
        "p8_model_contract_sha256": sha256_json(p8["model_contract"]),
        "p8_matrix_artifact_sha256": file_sha256(ROOT / "results/loafo/p8/loafo_matrix.json"),
        "training_config_sha256": sha256_json(TRAINING_CONFIG),
        **_source_binding(),
        "authorized_targets": list(TARGETS), "split_seeds": list(SEEDS),
        "role_permissions": {
            "train": ["gradient_fit", "centroid_fit", "scaler_fit", "vocabulary_fit"],
            "val": ["checkpoint_selection"],
            "meta_known": ["scorer_selection"],
            "surrogate_ood": ["scorer_selection"],
            "calibration": ["threshold_fit"],
            "target_held_out": [], "consumed_official_test": [],
        },
        "scorers": list(SCORERS), "selection_unit": "canonical_identity_mean",
        "selection_statistic": "surrogate_development_identity_AUROC",
        "threshold_unit": "known_calibration_canonical_identity_mean",
        "threshold_quantile": 0.99, "threshold_numpy_method": "higher",
        "threshold_comparison": "score > threshold",
        "development_only": True, "official_test_access": False,
        "target_held_out_scoring": False,
    }


def _read_p9(name: str) -> dict:
    return read_json(ROOT, "results/loafo/p9/" + name, "p9")


def dry_stage() -> dict:
    p8 = verify_p8(ROOT)
    context = load_train_context(ROOT, p8)
    entries = []
    for target in TARGETS:
        for seed in SEEDS:
            print(f"[P9 DRY] {target}/{seed}", flush=True)
            _, manifest, binding = _p8_cell(p8, target, seed)
            for role in ("train", "val", "meta_known", "surrogate_ood", "calibration"):
                selected_role_frame(context, manifest, role)
            verify_fold_fit(context, p8["schema"], manifest)
            entries.append(binding)
    if len(entries) != 27:
        raise ValueError("P9 dry matrix incomplete")
    protocol = _protocol(p8)
    dry = {"version": P9_VERSION, "status": "P9_PROTOCOL_PASS", "cells": entries,
           "cell_count": 27, "p8_protocol_sha256": P8_PROTOCOL_HASH,
           "protocol_sha256": sha256_json(protocol),
           "training_config_sha256": sha256_json(TRAINING_CONFIG),
           "optimizer_created": False, "optimizer_steps": 0,
           "target_held_out_access": False, "official_test_access": False}
    write_immutable(OUT / "protocol.json", json_bytes(protocol))
    write_immutable(OUT / "training_config.json", json_bytes(TRAINING_CONFIG))
    write_immutable(OUT / "dry_run_matrix.json", json_bytes(dry))
    print("P9_PROTOCOL_PASS: 27/27 cells; optimizer_steps=0", flush=True)
    return dry


def require_dry(p8: dict) -> dict:
    protocol = _read_p9("protocol.json")
    config = _read_p9("training_config.json")
    dry = _read_p9("dry_run_matrix.json")
    if protocol != _protocol(p8) or config != TRAINING_CONFIG:
        raise ValueError("P9 frozen protocol/training config mismatch")
    if (dry["status"] != "P9_PROTOCOL_PASS" or dry["cell_count"] != 27 or
            len(dry["cells"]) != 27 or dry["optimizer_steps"] != 0 or
            dry["protocol_sha256"] != sha256_json(protocol)):
        raise ValueError("P9 dry gate mismatch")
    return dry


def run_directory(target: str, seed: int, *, rerun: bool = False) -> Path:
    if target not in TARGETS or seed not in SEEDS:
        raise ValueError("unapproved P9 cell")
    if rerun:
        if (target, seed) != (PILOT_TARGET, PILOT_SEED):
            raise ValueError("only the pilot may be rerun")
        return OUT / "pilot_rerun/fuzzers_seed42"
    return OUT / "runs" / target.lower() / f"seed_{seed}"


def _model(manifest: dict) -> P8KnownClassModel:
    maps = manifest["vocabulary_provenance"]["vocabulary_maps"]
    vocab = {name: FrozenVocabulary.from_p3_mapping(name, maps[name])
             for name in CATEGORICAL_FIELDS}
    model = P8KnownClassModel(vocab, manifest["known_class_order"])
    if model.head.out_features != 7 or any(p.dtype != torch.float64 for p in model.parameters()):
        raise TypeError("P9 model violated frozen seven-class float64 contract")
    return model


def _score_vectors(model: P8KnownClassModel, role: dict,
                   centroids: np.ndarray, class_count: int) -> dict[str, np.ndarray]:
    representations, logits = representations_logits(model, role)
    return {
        "negative_max_softmax": softmax_uncertainty(logits, class_count),
        "nearest_centroid_l2": centroid_distance(representations, centroids, class_count),
    }


def _selection(model: P8KnownClassModel, roles: dict, centroids: np.ndarray,
               binding: dict) -> tuple[dict, dict]:
    require_role("scorer_selection", roles["meta_known"]["role"])
    require_role("scorer_selection", roles["surrogate_ood"]["role"])
    vectors = {name: _score_vectors(model, roles[name], centroids, len(binding["class_order"]))
               for name in ("meta_known", "surrogate_ood")}
    stats = {}
    for scorer in SCORERS:
        known_rows = vectors["meta_known"][scorer]
        surrogate_rows = vectors["surrogate_ood"][scorer]
        known_ids, known_identity = identity_means(known_rows, roles["meta_known"]["identities"])
        surrogate_ids, surrogate_identity = identity_means(
            surrogate_rows, roles["surrogate_ood"]["identities"])
        if (len(known_ids) != roles["meta_known"]["identity_count"] or
                len(surrogate_ids) != roles["surrogate_ood"]["identity_count"]):
            raise ValueError("P9 identity aggregation count mismatch")
        identity_auc = development_auroc(known_identity, surrogate_identity)
        row_auc = development_auroc(known_rows, surrogate_rows)
        stats[scorer] = {
            "surrogate_development_identity_auroc": identity_auc,
            "surrogate_development_row_auroc_diagnostic": row_auc,
            "row_minus_identity_auroc": row_auc - identity_auc,
            "meta_known_rows": score_summary(known_rows),
            "meta_known_identities": score_summary(known_identity),
            "surrogate_ood_rows": score_summary(surrogate_rows),
            "surrogate_ood_identities": score_summary(surrogate_identity),
        }
    choice = choose_scorer(
        stats["negative_max_softmax"]["surrogate_development_identity_auroc"],
        stats["nearest_centroid_l2"]["surrogate_development_identity_auroc"])
    return {"version": P9_VERSION, "label": "DEVELOPMENT_ONLY",
            "selection_unit": "canonical_identity_mean",
            "selection_roles": ["meta_known", "surrogate_ood"],
            "candidate_statistics": stats, **choice,
            "target_held_out_access": False, "official_test_access": False}, vectors


def _scorer_state(centroids: np.ndarray, centroid_digest: str,
                  checkpoint_hash: str, binding: dict) -> dict:
    softmax_digest = sha256_json({"scorer_type": "negative_max_softmax",
                                  "checkpoint_state_sha256": checkpoint_hash,
                                  "class_order": binding["class_order"],
                                  "formula_version": "one-minus-max-softmax-v1"})
    return {"version": P9_VERSION, "checkpoint_state_sha256": checkpoint_hash,
            "class_order": binding["class_order"], "representation_dim": 32,
            "states": {
                "negative_max_softmax": {"scorer_state_sha256": softmax_digest,
                                         "fitted_parameters": False},
                "nearest_centroid_l2": {"scorer_state_sha256": centroid_digest,
                                        "fit_role": "train", "class_order": binding["class_order"],
                                        "centroids_float64": centroids.tolist()},
            }, "official_test_access": False, "target_held_out_access": False}


def _diagnostics(selection: dict, vectors: dict, calibration_rows: np.ndarray,
                 calibration_identity: np.ndarray, threshold: float) -> dict:
    selected = selection["selected_scorer"]
    known_rows = vectors["meta_known"][selected]
    surrogate_rows = vectors["surrogate_ood"][selected]
    return {"version": P9_VERSION, "label": "DEVELOPMENT_ONLY",
            "used_for_fitting_or_selection": False,
            "selected_scorer": selected, "threshold": threshold,
            "selected_row_vs_identity_auroc_difference": selection[
                "candidate_statistics"][selected]["row_minus_identity_auroc"],
            "calibration_known_row_alert_fraction": float((calibration_rows > threshold).mean()),
            "calibration_known_identity_alert_fraction": float((calibration_identity > threshold).mean()),
            "meta_known_row_score_summary": score_summary(known_rows),
            "surrogate_ood_row_score_summary": score_summary(surrogate_rows),
            "calibration_row_score_summary": score_summary(calibration_rows),
            "calibration_identity_score_summary": score_summary(calibration_identity),
            "target_held_out_access": False, "official_test_access": False}


def execute_cell(p8: dict, context: dict, target: str, seed: int,
                 *, rerun: bool = False) -> dict:
    cell, manifest, binding = _p8_cell(p8, target, seed)
    directory = run_directory(target, seed, rerun=rerun)
    if directory.exists():
        return verify_run(p8, target, seed, rerun=rerun)
    # Revalidate source roles and fold-local preprocessing before optimizer creation.
    verify_fold_fit(context, p8["schema"], manifest)
    train = materialize_role(context, p8["schema"], manifest, "train")
    val = materialize_role(context, p8["schema"], manifest, "val")
    set_determinism(binding["model_seed"])
    model = _model(manifest)
    trained = train_cell(model, train, val, binding["model_seed"])
    del val
    train_representations, _ = representations_logits(model, train)
    centroids, centroid_digest = fit_centroids(
        train_representations, train["labels"], role="train",
        class_order=binding["class_order"], checkpoint_hash=trained["model_state_sha256"])
    del train_representations
    meta = materialize_role(context, p8["schema"], manifest, "meta_known")
    surrogate = materialize_role(context, p8["schema"], manifest, "surrogate_ood")
    roles = {"meta_known": meta, "surrogate_ood": surrogate}
    selection, vectors = _selection(model, roles, centroids, binding)
    del meta, surrogate, roles
    calibration = materialize_role(context, p8["schema"], manifest, "calibration")
    selected = selection["selected_scorer"]
    calibration_rows = _score_vectors(model, calibration, centroids,
                                      len(binding["class_order"]))[selected]
    threshold = fit_identity_threshold(calibration_rows, calibration["identities"],
                                       role="calibration")
    _, calibration_identity = identity_means(calibration_rows, calibration["identities"])
    diagnostics = _diagnostics(selection, vectors, calibration_rows,
                               calibration_identity, threshold["threshold"])
    states = _scorer_state(centroids, centroid_digest, trained["model_state_sha256"], binding)
    protocol = _read_p9("protocol.json")
    full_binding = {**binding, "p8_protocol_sha256": P8_PROTOCOL_HASH,
                    "p8_repository_commit": P8_COMMIT,
                    "p8_model_contract_sha256": protocol["p8_model_contract_sha256"],
                    "p9_training_config_sha256": protocol["training_config_sha256"],
                    "p9_source_code_aggregate_sha256": protocol["source_code_aggregate_sha256"],
                    "selected_epoch": trained["selected_epoch"],
                    "validation_value": trained["validation_value"],
                    "model_state_sha256": trained["model_state_sha256"],
                    "centroid_state_sha256": centroid_digest,
                    "selected_scorer": selected,
                    "selected_scorer_state_sha256": states["states"][selected]["scorer_state_sha256"],
                    "threshold": threshold["threshold"]}
    common = {"version": P9_VERSION, "provenance": full_binding,
              "label": "DEVELOPMENT_ONLY", "official_test_access": False,
              "target_held_out_scoring": False}
    training_manifest = {**common, "role": "train_and_val_only",
                         "optimizer": TRAINING_CONFIG,
                         "optimizer_steps": trained["optimizer_steps"],
                         "epoch_history": trained["epoch_history"],
                         "selected_epoch": trained["selected_epoch"],
                         "validation_value": trained["validation_value"]}
    checkpoint_payload = {"state_dict": trained["state_dict"], "provenance": full_binding}
    selection = {**common, **selection}
    states = {**common, **states}
    calibration_artifact = {**common, **threshold,
                            "fit_role": "calibration", "selected_scorer": selected,
                            "selected_scorer_state_sha256": full_binding[
                                "selected_scorer_state_sha256"]}
    diagnostics = {**common, **diagnostics}
    directory.parent.mkdir(parents=True, exist_ok=True)
    pending = Path(tempfile.mkdtemp(prefix=".pending-p9-", dir=directory.parent))
    with (pending / "checkpoint.pt").open("xb") as output:
        torch.save(checkpoint_payload, output)
    checkpoint_file_hash = file_sha256(pending / "checkpoint.pt")
    checkpoint_metadata = {**common,
                           "checkpoint_file_sha256": checkpoint_file_hash,
                           "checkpoint_state_sha256": trained["model_state_sha256"],
                           "checkpoint_path": str((directory / "checkpoint.pt").relative_to(ROOT))}
    payloads = {"training_manifest.json": training_manifest,
                "checkpoint_metadata.json": checkpoint_metadata,
                "scorer_state.json": states, "scorer_selection.json": selection,
                "calibration.json": calibration_artifact,
                "development_diagnostics.json": diagnostics}
    for name, payload in payloads.items():
        write_immutable(pending / name, json_bytes(payload))
    summary = {"version": P9_VERSION, "target": target, "split_seed": seed,
               "model_seed": binding["model_seed"], "known_class_order": binding["class_order"],
               "surrogate_families": binding["surrogate_families"],
               "train_rows": binding["role_rows"]["train"],
               "train_identities": binding["role_identities"]["train"],
               "selected_epoch": trained["selected_epoch"],
               "validation_value": trained["validation_value"],
               "model_state_sha256": trained["model_state_sha256"],
               "checkpoint_file_sha256": checkpoint_file_hash,
               "centroid_state_sha256": centroid_digest,
               "softmax_surrogate_development_identity_auroc": selection[
                   "candidate_statistics"]["negative_max_softmax"]["surrogate_development_identity_auroc"],
               "centroid_surrogate_development_identity_auroc": selection[
                   "candidate_statistics"]["nearest_centroid_l2"]["surrogate_development_identity_auroc"],
               "selected_scorer": selected,
               "selection_margin_centroid_minus_softmax": selection[
                   "selection_margin_centroid_minus_softmax"],
               "tie_break_used": selection["tie_break_used"],
               "calibration_row_count": threshold["calibration_row_count"],
               "calibration_identity_count": threshold["calibration_identity_count"],
               "threshold": threshold["threshold"],
               "optimizer_steps": trained["optimizer_steps"],
               "p8_manifest_sha256": binding["p8_manifest_sha256"],
               "p8_protocol_sha256": P8_PROTOCOL_HASH,
               "p9_training_config_sha256": protocol["training_config_sha256"],
               "p9_source_code_aggregate_sha256": protocol["source_code_aggregate_sha256"],
               "artifact_sha256": {name: file_sha256(pending / name)
                                   for name in RUN_FILES},
               "label": "DEVELOPMENT_ONLY", "target_held_out_scoring": False,
               "official_test_access": False, "status": "P9_CELL_PASS"}
    write_immutable(pending / "run_summary.json", json_bytes(summary))
    if directory.exists():
        raise FileExistsError("P9 run directory appeared before immutable publish")
    pending.rename(directory)
    return verify_run(p8, target, seed, rerun=rerun)


def verify_run(p8: dict, target: str, seed: int, *, rerun: bool = False) -> dict:
    directory = run_directory(target, seed, rerun=rerun)
    summary = read_json(ROOT, str((directory / "run_summary.json").relative_to(ROOT)), "p9")
    _, _, binding = _p8_cell(p8, target, seed)
    protocol = _read_p9("protocol.json")
    if (summary["status"] != "P9_CELL_PASS" or summary["target"] != target or
            summary["split_seed"] != seed or summary["model_seed"] != binding["model_seed"] or
            summary["p8_protocol_sha256"] != P8_PROTOCOL_HASH or
            summary["p8_manifest_sha256"] != binding["p8_manifest_sha256"] or
            summary["p9_training_config_sha256"] != protocol["training_config_sha256"] or
            summary["p9_source_code_aggregate_sha256"] != protocol["source_code_aggregate_sha256"] or
            summary["target_held_out_scoring"] is not False or summary["official_test_access"] is not False):
        raise ValueError("P9 run summary provenance mismatch")
    for name, expected in summary["artifact_sha256"].items():
        path = checked_input(ROOT, directory / name, "p9")
        if file_sha256(path) != expected:
            raise ValueError(f"P9 run artifact hash mismatch: {path}")
    meta = read_json(ROOT, str((directory / "checkpoint_metadata.json").relative_to(ROOT)), "p9")
    checkpoint = checked_input(ROOT, directory / "checkpoint.pt", "p9")
    if file_sha256(checkpoint) != meta["checkpoint_file_sha256"]:
        raise ValueError("P9 checkpoint file hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (state_hash(payload["state_dict"]) != summary["model_state_sha256"] or
            payload["provenance"] != meta["provenance"] or
            payload["provenance"]["threshold"] != summary["threshold"]):
        raise ValueError("P9 checkpoint state/provenance mismatch")
    selection = read_json(ROOT, str((directory / "scorer_selection.json").relative_to(ROOT)), "p9")
    calibration = read_json(ROOT, str((directory / "calibration.json").relative_to(ROOT)), "p9")
    states = read_json(ROOT, str((directory / "scorer_state.json").relative_to(ROOT)), "p9")
    if (selection["selected_scorer"] != summary["selected_scorer"] or
            calibration["threshold"] != summary["threshold"] or
            calibration["calibration_identity_count"] != summary["calibration_identity_count"] or
            states["states"]["nearest_centroid_l2"]["scorer_state_sha256"] != summary[
                "centroid_state_sha256"]):
        raise ValueError("P9 scorer/calibration provenance mismatch")
    for artifact in (meta, selection, calibration, states):
        if artifact["provenance"] != payload["provenance"]:
            raise ValueError("P9 artifact provenance chain incomplete")
    return summary


def _pilot_comparison(left: dict, right: dict) -> dict:
    keys = ("selected_epoch", "model_state_sha256", "centroid_state_sha256",
            "softmax_surrogate_development_identity_auroc",
            "centroid_surrogate_development_identity_auroc", "selected_scorer",
            "selection_margin_centroid_minus_softmax", "tie_break_used", "threshold")
    differences = [key for key in keys if left[key] != right[key]]
    return {"version": P9_VERSION, "target": PILOT_TARGET, "split_seed": PILOT_SEED,
            "model_seed": 10042, "exact_compared_fields": list(keys),
            "different_fields": differences,
            "checkpoint_file_hash_equal": left["checkpoint_file_sha256"] == right[
                "checkpoint_file_sha256"],
            "status": "P9_PILOT_PASS" if not differences else "P9_PILOT_BLOCKED",
            "development_only": True, "target_held_out_scoring": False,
            "official_test_access": False}


def pilot_stage() -> dict:
    p8 = verify_p8(ROOT)
    require_dry(p8)
    context = load_train_context(ROOT, p8)
    left = execute_cell(p8, context, PILOT_TARGET, PILOT_SEED)
    print("[P9 PILOT] first run complete; reproducibility rerun starting", flush=True)
    right = execute_cell(p8, context, PILOT_TARGET, PILOT_SEED, rerun=True)
    comparison = _pilot_comparison(left, right)
    if comparison["status"] != "P9_PILOT_PASS":
        raise ValueError(f"P9 pilot reproducibility mismatch: {comparison['different_fields']}")
    write_immutable(OUT / "pilot_reproducibility.json", json_bytes(comparison))
    print("P9_PILOT_PASS: exact selected epoch/state/scorer/threshold reproducibility", flush=True)
    return comparison


def require_pilot(p8: dict) -> dict:
    comparison = _read_p9("pilot_reproducibility.json")
    left = verify_run(p8, PILOT_TARGET, PILOT_SEED)
    right = verify_run(p8, PILOT_TARGET, PILOT_SEED, rerun=True)
    if comparison != _pilot_comparison(left, right) or comparison["status"] != "P9_PILOT_PASS":
        raise ValueError("P9 pilot reproducibility gate mismatch")
    return comparison


def _development_summary(cells: list[dict]) -> dict:
    counts = {scorer: sum(c["selected_scorer"] == scorer for c in cells) for scorer in SCORERS}
    by_target = {}
    for target in TARGETS:
        entries = sorted((c for c in cells if c["target"] == target),
                         key=lambda c: c["split_seed"])
        by_target[target] = {
            "selected_scorers_by_seed": {str(c["split_seed"]): c["selected_scorer"] for c in entries},
            "seed_agreement": len({c["selected_scorer"] for c in entries}) == 1,
            "selection_margins": [c["selection_margin_centroid_minus_softmax"] for c in entries],
            "selected_epochs": [c["selected_epoch"] for c in entries],
            "calibration_thresholds": [c["threshold"] for c in entries],
            "p8_support_flag": ("SMALL-N HELD-OUT TARGET" if target == "Worms" else
                                "retain P8 support context"),
        }
    return {"version": P9_VERSION, "label": "DEVELOPMENT_ONLY",
            "scorer_selection_counts": counts, "per_target": by_target,
            "selected_epoch_distribution": {str(epoch): sum(c["selected_epoch"] == epoch for c in cells)
                                            for epoch in range(1, 9)},
            "descriptive_only": True, "protocol_changed_after_pilot": False,
            "target_held_out_scoring": False, "official_test_access": False}


def matrix_stage() -> dict:
    p8 = verify_p8(ROOT)
    require_dry(p8)
    require_pilot(p8)
    context = load_train_context(ROOT, p8)
    cells = []
    for target in TARGETS:
        for seed in SEEDS:
            print(f"[P9 DEVELOPMENT] {target}/{seed}", flush=True)
            cells.append(execute_cell(p8, context, target, seed))
    if len(cells) != 27:
        raise ValueError("P9 development matrix incomplete")
    summary = _development_summary(cells)
    matrix = {"version": P9_VERSION, "status": "P9_DEVELOPMENT_PASS",
              "p8_protocol_sha256": P8_PROTOCOL_HASH,
              "p8_repository_commit": P8_COMMIT,
              "protocol_sha256": sha256_json(_read_p9("protocol.json")),
              "training_config_sha256": sha256_json(TRAINING_CONFIG),
              "cell_count": 27, "cells": cells,
              "development_summary_sha256": sha256_json(summary),
              "target_held_out_scoring": False, "official_test_access": False,
              "external_zero_day_claim": False}
    write_immutable(OUT / "development_summary.json", json_bytes(summary))
    write_immutable(OUT / "p9_matrix.json", json_bytes(matrix))
    print("P9_DEVELOPMENT_PASS: 27/27 cells; DEVELOPMENT_ONLY", flush=True)
    return matrix


def verify_stage() -> dict:
    p8 = verify_p8(ROOT)
    require_dry(p8)
    require_pilot(p8)
    matrix = _read_p9("p9_matrix.json")
    summary = _read_p9("development_summary.json")
    if (matrix["status"] != "P9_DEVELOPMENT_PASS" or matrix["cell_count"] != 27 or
            len(matrix["cells"]) != 27 or matrix["p8_protocol_sha256"] != P8_PROTOCOL_HASH or
            matrix["development_summary_sha256"] != sha256_json(summary)):
        raise ValueError("P9 final matrix binding mismatch")
    actual = [verify_run(p8, target, seed) for target in TARGETS for seed in SEEDS]
    if actual != matrix["cells"] or _development_summary(actual) != summary:
        raise ValueError("P9 development matrix/run mismatch")
    if (ROOT / "results/loafo/p9/target_held_out").exists():
        raise PermissionError("P9 target-held-out output forbidden")
    print("P9_DEVELOPMENT_PASS: read-only verification; 27/27 bound runs", flush=True)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("dry", "pilot", "matrix", "verify"))
    args = parser.parse_args()
    if args.stage == "dry":
        dry_stage()
    elif args.stage == "pilot":
        pilot_stage()
    elif args.stage == "matrix":
        matrix_stage()
    else:
        verify_stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
