#!/usr/bin/env python3
"""Execute frozen P6 development-only scorer selection in controlled stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p5_training import (  # noqa: E402
    SPLIT_SEEDS, TARGETS, TRAINING_CONFIG, TrainingOnlyModel,
    _environment, _state_digest, file_hash, json_bytes, set_determinism,
    stable_hash, write_immutable,
)
from ids.p4_model_interface import vocabularies_from_manifest  # noqa: E402
from ids.p6_selection import (  # noqa: E402
    P6_VERSION, QUANTILE, QUANTILE_METHOD, SCORERS, SELECTION_UNIT,
    TIE_PREFERENCE, TIE_TOLERANCE, calibration_artifact,
    development_diagnostics, load_manifest,
    materialize_role, output_path, p6_source_hash, prepare_context,
    scorer_scores, select_scorer, train_frozen_p5, validate_p6_cell,
    verify_calibration_artifact, verify_scorer_states, verify_upstream,
)


def contracts() -> dict[str, dict]:
    protocol = {
        "version": P6_VERSION, "upstream_status": "P5_SMOKE_PASS",
        "stages": ["A: 15 dry validations, no training", "B: Fuzzers/42/10042 pilot scorer selection",
                   "C: identical pilot rerun, then known-only calibration twice",
                   "D: all 15 independent cells with no adaptive changes"],
        "information_flow": ["train -> representation and centroids", "val -> checkpoint",
                             "meta_known + surrogate_ood -> scorer family",
                             "calibration -> known-only threshold", "official test/target -> forbidden"],
        "training_contract": "exact frozen P5 settings and architecture, no search",
        "true_target_access": False, "official_test_access": False,
        "scientific_effectiveness_claim": False,
    }
    selection = {
        "version": "p6-selection-contract-v1", "candidates": list(SCORERS),
        "unit": SELECTION_UNIT, "aggregation": "arithmetic mean of row scores per canonical identity; order-independent sorted math.fsum",
        "primary_metric": "SURROGATE DEVELOPMENT AUROC at canonical-identity level",
        "negative_role": "meta_known", "positive_role": "surrogate_ood",
        "score_orientation": "higher means more OOD-like",
        "row_metric": "diagnostic only; never selects scorer",
        "tie_tolerance": TIE_TOLERANCE,
        "tie_preference": TIE_PREFERENCE,
        "tie_rationale": ("P5 froze nearest_centroid_l2 on exact tie; P6 extends numerical equality "
                          "to 1e-12 while retaining P5's centroid preference."),
        "selection_scope": "independent target x split-seed cell; no cross-seed pooling",
        "surrogate_interpretation": ("Surrogate-based scorer selection estimates development discrimination "
                                     "against the predefined surrogate unknown families only. "
                                     "It does not establish performance on the held-out target family."),
    }
    calibration = {
        "version": "p6-calibration-contract-v1", "fit_role": "calibration",
        "known_only": True, "surrogate_allowed": False, "target_allowed": False,
        "quantile": QUANTILE, "numpy_method": QUANTILE_METHOD,
        "decision": "score > threshold", "score_equals_threshold": "known-side",
        "interpretation": ("known-only calibration operating point, not target-optimized; "
                           "nominal 1% upper tail does not guarantee future 1% FPR"),
    }
    return {"protocol.json": protocol, "selection_contract.json": selection,
            "calibration_contract.json": calibration}


def matrix_path() -> Path:
    return output_path(ROOT, "p6_matrix.json")


def load_matrix(required_status: str | None = None) -> dict:
    matrix = json.loads(matrix_path().read_text())
    if required_status is not None and matrix["status"] != required_status:
        raise ValueError(f"P6 gate requires {required_status}; observed {matrix['status']}")
    _, source_digest = p6_source_hash(ROOT)
    if matrix["source_hash"] != source_digest:
        raise ValueError("P6 source hash changed after protocol freeze")
    return matrix


def save_matrix(matrix: dict) -> None:
    matrix_path().write_bytes(json_bytes(matrix))


def stage_a() -> dict:
    verify_upstream(ROOT)
    if matrix_path().exists():
        raise FileExistsError("P6 protocol namespace already frozen")
    cells = [validate_p6_cell(ROOT, target, seed)
             for target in TARGETS for seed in SPLIT_SEEDS]
    for name, payload in contracts().items():
        write_immutable(output_path(ROOT, name), json_bytes(payload))
    for cell in cells:
        write_immutable(output_path(ROOT, f"dry_runs/{cell['target'].lower()}_seed{cell['split_seed']}.json"),
                        json_bytes(cell))
    files, digest = p6_source_hash(ROOT)
    matrix = {"version": "p6-development-matrix-v1", "status": "P6_PROTOCOL_PASS",
              "source_hashes": files, "source_hash": digest,
              "selection_unit": SELECTION_UNIT, "selection_contract_sha256": stable_hash(contracts()["selection_contract.json"]),
              "calibration_contract_sha256": stable_hash(contracts()["calibration_contract.json"]),
              "cells": [{"target": cell["target"], "split_seed": cell["split_seed"],
                         "model_seed": cell["model_seed"], "status": cell["status"]} for cell in cells],
              "pilot": {}, "development_runs": {}, "stability": None,
              "official_test_access": False, "true_target_access": False,
              "scientific_effectiveness_claim": False}
    save_matrix(matrix)
    return matrix


def cell_prefix(target: str, seed: int, pilot_tag: str | None = None) -> str:
    base = f"runs/{target.lower()}/seed_{seed}"
    return f"{base}/pilot_{pilot_tag}" if pilot_tag else base


def checkpoint_artifacts(prefix: str, dry: dict, training: dict,
                         *, reference: str | None = None) -> tuple[dict, dict]:
    run = {
        "version": "p6-frozen-p5-training-manifest-v1",
        "target": dry["target"], "split_seed": dry["split_seed"],
        "model_seed": dry["model_seed"],
        "p3_manifest_sha256": dry["p3_manifest_sha256"],
        "p4_model_config_sha256": dry["p4_model_config_sha256"],
        "p5_training_config_sha256": dry["p5_training_config_sha256"],
        "p5_source_hash": dry["p5_source_hash"],
        "p6_source_hash": dry["p6_source_hash"],
        "selected_epoch": training["selected_epoch"],
        "validation_selection_value": training["validation_selection_value"],
        "epoch_history": training["epoch_history"],
        "optimizer_steps": training["optimizer_steps"],
        "checkpoint_state_sha256": training["checkpoint_state_sha256"],
        "optimizer_config": {key: TRAINING_CONFIG[key] for key in (
            "optimizer", "learning_rate", "weight_decay", "batch_size", "max_epochs",
            "gradient_clip_norm", "patience", "min_delta", "class_weighting", "scheduler")},
        "role_counts": {name: item["rows"] for name, item in dry["role_counts"].items()},
        "identity_counts": {name: item["canonical_identities"] for name, item in dry["role_counts"].items()},
        "environment": _environment(ROOT), "official_test_access": False,
        "true_target_access": False, "development_only": True,
    }
    if reference is None:
        checkpoint_path = output_path(ROOT, f"{prefix}/checkpoint.pt")
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint_path.exists():
            raise FileExistsError("immutable P6 checkpoint already exists")
        torch.save({"state_dict": training["state_dict"], "provenance": {
            "target": dry["target"], "split_seed": dry["split_seed"],
            "model_seed": dry["model_seed"], "p3_manifest_sha256": dry["p3_manifest_sha256"],
            "p4_model_config_sha256": dry["p4_model_config_sha256"],
            "p5_training_config_sha256": dry["p5_training_config_sha256"],
            "p6_source_hash": dry["p6_source_hash"],
            "selected_epoch": training["selected_epoch"],
            "validation_selection_value": training["validation_selection_value"]}}, checkpoint_path)
        checkpoint_file_sha256 = file_hash(checkpoint_path)
        path_name = str(checkpoint_path.relative_to(ROOT))
    else:
        path_name = reference
        checkpoint_file_sha256 = file_hash(ROOT / reference)
    ref = {"version": "p6-checkpoint-reference-v1", "target": dry["target"],
           "split_seed": dry["split_seed"], "model_seed": dry["model_seed"],
           "path": path_name, "checkpoint_file_sha256": checkpoint_file_sha256,
           "checkpoint_state_sha256": training["checkpoint_state_sha256"],
           "p3_manifest_sha256": dry["p3_manifest_sha256"],
           "p4_model_config_sha256": dry["p4_model_config_sha256"],
           "p5_training_config_sha256": dry["p5_training_config_sha256"],
           "p6_source_hash": dry["p6_source_hash"]}
    run["checkpoint_reference_sha256"] = stable_hash(ref)
    write_immutable(output_path(ROOT, f"{prefix}/training_manifest.json"), json_bytes(run))
    write_immutable(output_path(ROOT, f"{prefix}/checkpoint_reference.json"), json_bytes(ref))
    return run, ref


def execute_cell(target: str, seed: int, *, pilot_tag: str | None = None,
                 reuse_pilot: bool = False, calibrate: bool = False) -> dict:
    dry = validate_p6_cell(ROOT, target, seed)
    prefix = cell_prefix(target, seed, pilot_tag)
    manifest = load_manifest(ROOT, target, seed)
    set_determinism(dry["model_seed"])
    context, schema = prepare_context(ROOT, manifest)
    roles = {name: materialize_role(context, schema, manifest, name)
             for name in ("train", "val", "meta_known", "surrogate_ood")}
    model = TrainingOnlyModel(vocabularies_from_manifest(manifest))
    if reuse_pilot:
        pilot_prefix = cell_prefix("Fuzzers", 42, "smoke-a")
        pilot_run = json.loads(output_path(ROOT, f"{pilot_prefix}/training_manifest.json").read_text())
        pilot_ref = json.loads(output_path(ROOT, f"{pilot_prefix}/checkpoint_reference.json").read_text())
        checkpoint = ROOT / pilot_ref["path"]
        if file_hash(checkpoint) != pilot_ref["checkpoint_file_sha256"]:
            raise ValueError("pilot checkpoint file changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        training = {"state_dict": payload["state_dict"],
                    "selected_epoch": pilot_run["selected_epoch"],
                    "validation_selection_value": pilot_run["validation_selection_value"],
                    "epoch_history": pilot_run["epoch_history"],
                    "optimizer_steps": pilot_run["optimizer_steps"],
                    "checkpoint_state_sha256": pilot_run["checkpoint_state_sha256"]}
        if _state_digest(training["state_dict"]) != training["checkpoint_state_sha256"]:
            raise ValueError("pilot checkpoint state changed")
        model.load_state_dict(training["state_dict"])
        run, ref = checkpoint_artifacts(prefix, dry, training, reference=pilot_ref["path"])
    else:
        training = train_frozen_p5(model, roles["train"], roles["val"], dry["model_seed"])
        run, ref = checkpoint_artifacts(prefix, dry, training)
    if pilot_tag is not None:
        p5_run = json.loads((ROOT / "results/training_protocol/p5/training_runs" /
                             f"p5-fuzzers-s42-m10042-{pilot_tag}.json").read_text())
        if training["checkpoint_state_sha256"] != p5_run["checkpoint_state_sha256"]:
            raise ValueError("P6 pilot training deviates from frozen P5 smoke model state")
    states, selection, centroids = select_scorer(model, roles, training["checkpoint_state_sha256"])
    verify_scorer_states(states, training["checkpoint_state_sha256"])
    selection.update({"target": target, "split_seed": seed, "model_seed": dry["model_seed"],
                      "selected_checkpoint": ref["path"],
                      "checkpoint_state_sha256": training["checkpoint_state_sha256"],
                      "p3_manifest_sha256": dry["p3_manifest_sha256"],
                      "p4_model_config_sha256": dry["p4_model_config_sha256"],
                      "p5_training_config_sha256": dry["p5_training_config_sha256"],
                      "source_hash": dry["p6_source_hash"],
                      "selected_epoch": training["selected_epoch"]})
    states.update({"target": target, "split_seed": seed, "model_seed": dry["model_seed"],
                   "p3_manifest_sha256": dry["p3_manifest_sha256"],
                   "p4_model_config_sha256": dry["p4_model_config_sha256"],
                   "p5_training_config_sha256": dry["p5_training_config_sha256"],
                   "source_hash": dry["p6_source_hash"]})
    write_immutable(output_path(ROOT, f"{prefix}/scorer_states.json"), json_bytes(states))
    write_immutable(output_path(ROOT, f"{prefix}/scorer_selection.json"), json_bytes(selection))
    result = {"training_manifest": run, "checkpoint_reference": ref,
              "scorer_states": states, "scorer_selection": selection}
    if calibrate:
        result.update(complete_calibration(ROOT, prefix, dry, manifest, context, schema,
                                           model, roles, states, selection, centroids))
    return result


def complete_calibration(root: Path, prefix: str, dry: dict, manifest: dict,
                         context: dict, schema: dict, model: TrainingOnlyModel,
                         roles: dict, states: dict, selection: dict,
                         centroids: np.ndarray) -> dict:
    # This function is called only after selection is persisted and frozen.
    stored = json.loads(output_path(root, f"{prefix}/scorer_selection.json").read_text())
    if stored != selection:
        raise ValueError("scorer selection changed before calibration")
    calibration = materialize_role(context, schema, manifest, "calibration")
    roles["calibration"] = calibration
    selected = selection["selected_scorer"]
    scores = scorer_scores(model, calibration, selected, centroids)
    state_hash = states["states"][selected]["scorer_state_sha256"]
    artifact = calibration_artifact(root, dry, calibration, selected, state_hash,
                                    selection["checkpoint_state_sha256"], scores)
    verify_calibration_artifact(artifact, dry, calibration,
                                selection["checkpoint_state_sha256"], state_hash)
    write_immutable(output_path(root, f"{prefix}/calibration.json"), json_bytes(artifact))
    diagnostics = development_diagnostics(model, roles, selected, centroids, artifact["threshold"])
    write_immutable(output_path(root, f"{prefix}/development_diagnostics.json"),
                    json_bytes(diagnostics))
    return {"calibration": artifact, "development_diagnostics": diagnostics}


def calibrate_existing_pilot(tag: str) -> dict:
    prefix = cell_prefix("Fuzzers", 42, tag)
    dry = validate_p6_cell(ROOT, "Fuzzers", 42)
    manifest = load_manifest(ROOT, "Fuzzers", 42)
    set_determinism(10042)
    context, schema = prepare_context(ROOT, manifest)
    model = TrainingOnlyModel(vocabularies_from_manifest(manifest))
    ref = json.loads(output_path(ROOT, f"{prefix}/checkpoint_reference.json").read_text())
    if file_hash(ROOT / ref["path"]) != ref["checkpoint_file_sha256"]:
        raise ValueError("pilot checkpoint changed before calibration")
    payload = torch.load(ROOT / ref["path"], map_location="cpu", weights_only=True)
    if _state_digest(payload["state_dict"]) != ref["checkpoint_state_sha256"]:
        raise ValueError("pilot checkpoint state changed before calibration")
    model.load_state_dict(payload["state_dict"])
    states = json.loads(output_path(ROOT, f"{prefix}/scorer_states.json").read_text())
    selection = json.loads(output_path(ROOT, f"{prefix}/scorer_selection.json").read_text())
    roles = {name: materialize_role(context, schema, manifest, name)
             for name in ("meta_known", "surrogate_ood")}
    centroids = verify_scorer_states(states, ref["checkpoint_state_sha256"])
    return complete_calibration(ROOT, prefix, dry, manifest, context, schema,
                                model, roles, states, selection, centroids)


def verify_pilot_selection() -> dict:
    matrix = load_matrix("P6_PROTOCOL_PASS")
    if not all(tag in matrix["pilot"] for tag in ("smoke-a", "smoke-b")):
        raise ValueError("two pilot selections are required")
    records = []
    for tag in ("smoke-a", "smoke-b"):
        prefix = cell_prefix("Fuzzers", 42, tag)
        run = json.loads(output_path(ROOT, f"{prefix}/training_manifest.json").read_text())
        states = json.loads(output_path(ROOT, f"{prefix}/scorer_states.json").read_text())
        selection = json.loads(output_path(ROOT, f"{prefix}/scorer_selection.json").read_text())
        verify_scorer_states(states, run["checkpoint_state_sha256"])
        records.append((run, states, selection))
    left, right = records
    for a, b in ((left[0]["selected_epoch"], right[0]["selected_epoch"]),
                 (left[0]["checkpoint_state_sha256"], right[0]["checkpoint_state_sha256"]),
                 (left[1]["states"], right[1]["states"]),
                 (left[2]["candidate_statistics"], right[2]["candidate_statistics"]),
                 (left[2]["selected_scorer"], right[2]["selected_scorer"])):
        if a != b:
            raise ValueError("pilot scorer-selection rerun mismatch")
    result = {"status": "PILOT_SELECTION_REPRODUCED", "selected_epoch": left[0]["selected_epoch"],
              "checkpoint_state_sha256": left[0]["checkpoint_state_sha256"],
              "centroid_state_sha256": left[1]["states"]["nearest_centroid_l2"]["scorer_state_sha256"],
              "selected_scorer": left[2]["selected_scorer"],
              "statistics_identical": True, "calibration_accessed": False}
    matrix["pilot"]["selection_reproducibility"] = result
    save_matrix(matrix)
    return result


def verify_pilot() -> dict:
    matrix = load_matrix("P6_PROTOCOL_PASS")
    base = cell_prefix("Fuzzers", 42)
    runs = []
    for tag in ("smoke-a", "smoke-b"):
        prefix = f"{base}/pilot_{tag}"
        runs.append({name: json.loads(output_path(ROOT, f"{prefix}/{name}.json").read_text())
                     for name in ("training_manifest", "checkpoint_reference",
                                  "scorer_states", "scorer_selection", "calibration")})
    left, right = runs
    for name, keys in {
        "training_manifest": ("selected_epoch", "checkpoint_state_sha256", "validation_selection_value"),
        "scorer_states": ("states",),
        "scorer_selection": ("candidate_statistics", "selected_scorer", "selection_margin_centroid_minus_softmax"),
        "calibration": ("selected_scorer", "scorer_state_sha256", "threshold", "quantile", "quantile_method"),
    }.items():
        for key in keys:
            if left[name][key] != right[name][key]:
                raise ValueError(f"pilot reproducibility failed: {name}.{key}")
    result = {"status": "P6_PILOT_PASS", "selected_epoch": left["training_manifest"]["selected_epoch"],
              "checkpoint_state_sha256": left["training_manifest"]["checkpoint_state_sha256"],
              "centroid_state_sha256": left["scorer_states"]["states"]["nearest_centroid_l2"]["scorer_state_sha256"],
              "selected_scorer": left["scorer_selection"]["selected_scorer"],
              "selected_threshold": left["calibration"]["threshold"],
              "identical_selection_statistics": True, "identical_threshold": True}
    matrix["pilot"]["reproducibility"] = result
    matrix["status"] = "P6_PILOT_PASS"
    save_matrix(matrix)
    return result


def summary_cell(result: dict) -> dict:
    run, selection, calibration = (result[name] for name in (
        "training_manifest", "scorer_selection", "calibration"))
    stats = selection["candidate_statistics"]
    return {"target": run["target"], "split_seed": run["split_seed"],
            "model_seed": run["model_seed"], "selected_epoch": run["selected_epoch"],
            "checkpoint_state_hash": run["checkpoint_state_sha256"],
            "selected_scorer": selection["selected_scorer"],
            "softmax_SURROGATE_DEVELOPMENT_AUROC": stats["negative_max_softmax"]["surrogate_development_identity_auroc"],
            "centroid_SURROGATE_DEVELOPMENT_AUROC": stats["nearest_centroid_l2"]["surrogate_development_identity_auroc"],
            "selection_margin_centroid_minus_softmax": selection["selection_margin_centroid_minus_softmax"],
            "threshold": calibration["threshold"],
            "meta_known_identity_count": selection["meta_known_identity_count"],
            "surrogate_identity_count": selection["surrogate_identity_count"],
            "calibration_identity_count": calibration["calibration_identity_count"],
            "status": "DEVELOPMENT_CELL_PASS"}


def full_matrix() -> dict:
    matrix = load_matrix("P6_PILOT_PASS")
    for target in TARGETS:
        for seed in SPLIT_SEEDS:
            print(f"[P6 DEVELOPMENT] {target}/{seed}", flush=True)
            result = execute_cell(target, seed, reuse_pilot=(target == "Fuzzers" and seed == 42),
                                  calibrate=True)
            key = f"{target}/{seed}"
            matrix["development_runs"][key] = summary_cell(result)
            save_matrix(matrix)
    cells = list(matrix["development_runs"].values())
    if len(cells) != 15:
        raise ValueError("incomplete P6 development matrix")
    stability = {"softmax_cells": sum(x["selected_scorer"] == SCORERS[0] for x in cells),
                 "centroid_cells": sum(x["selected_scorer"] == SCORERS[1] for x in cells),
                 "per_target": {}}
    for target in TARGETS:
        group = [x for x in cells if x["target"] == target]
        stability["per_target"][target] = {
            "seed_agreement": len({x["selected_scorer"] for x in group}) == 1,
            "selected_by_seed": {str(x["split_seed"]): x["selected_scorer"] for x in group},
            "selection_margins": {str(x["split_seed"]): x["selection_margin_centroid_minus_softmax"]
                                  for x in group},
        }
    matrix["stability"] = stability
    matrix["cells"] = cells
    matrix["status"] = "P6_DEVELOPMENT_PASS"
    save_matrix(matrix)
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_mutually_exclusive_group(required=True)
    stages.add_argument("--dry-all", action="store_true")
    stages.add_argument("--pilot-select", choices=("smoke-a", "smoke-b"))
    stages.add_argument("--pilot-calibrate", action="store_true")
    stages.add_argument("--verify-pilot-selection", action="store_true")
    stages.add_argument("--verify-pilot", action="store_true")
    stages.add_argument("--full-matrix", action="store_true")
    args = parser.parse_args()
    if args.dry_all:
        matrix = stage_a()
        print(f"[P6] {matrix['status']}: 15/15 dry validations, no optimizer")
    elif args.pilot_select:
        matrix = load_matrix("P6_PROTOCOL_PASS")
        if args.pilot_select == "smoke-b" and "smoke-a" not in matrix["pilot"]:
            raise ValueError("smoke-a must precede smoke-b")
        result = execute_cell("Fuzzers", 42, pilot_tag=args.pilot_select, calibrate=False)
        matrix["pilot"][args.pilot_select] = summary = {
            "selected_epoch": result["training_manifest"]["selected_epoch"],
            "checkpoint_state_sha256": result["training_manifest"]["checkpoint_state_sha256"],
            "selected_scorer": result["scorer_selection"]["selected_scorer"]}
        save_matrix(matrix)
        print(f"[P6 PILOT SELECT] {args.pilot_select}: {summary['selected_scorer']}; no calibration read")
    elif args.pilot_calibrate:
        matrix = load_matrix("P6_PROTOCOL_PASS")
        if matrix["pilot"].get("selection_reproducibility", {}).get("status") != "PILOT_SELECTION_REPRODUCED":
            raise ValueError("pilot scorer selection must reproduce before calibration")
        for tag in ("smoke-a", "smoke-b"):
            result = calibrate_existing_pilot(tag)
            matrix["pilot"][tag]["threshold"] = result["calibration"]["threshold"]
        save_matrix(matrix)
        print("[P6 PILOT CALIBRATE] two known-only thresholds fitted after scorer freeze")
    elif args.verify_pilot_selection:
        result = verify_pilot_selection()
        print(f"[P6] {result['status']}: {result['selected_scorer']}; no calibration read")
    elif args.verify_pilot:
        result = verify_pilot()
        print(f"[P6] {result['status']}: {result['selected_scorer']}")
    else:
        matrix = full_matrix()
        print(f"[P6] {matrix['status']}: {len(matrix['cells'])}/15 cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
