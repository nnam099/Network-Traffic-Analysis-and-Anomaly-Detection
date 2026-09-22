#!/usr/bin/env python3
"""P5 schema-V2-only development protocol; training requires explicit opt-in."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p5_training import (  # noqa: E402
    P3_HASH, PERMISSIONS, ROLES, SPLIT_SEEDS, TARGETS, TRAINING_CONFIG,
    checked_output, json_bytes, source_hash, stable_hash, train_smoke,
    validate_cell, verify_smoke, write_immutable,
)


def protocol_artifacts() -> dict[str, dict]:
    role_matrix = {
        "version": "p5-role-permissions-v1", "roles": list(ROLES),
        "operations": {operation: {role: role in allowed for role in ROLES}
                       for operation, allowed in PERMISSIONS.items()},
        "source_file_policy": "only exact official training CSV; hash checked; no test file opened",
        "target_policy": "held-out target and its canonical identities forbidden for all fitting",
    }
    scorer = {
        "version": "p5-scorer-candidates-v1", "selected_family": None,
        "candidates": [
            {"id": "negative_max_softmax", "requires": "temporary known-family head",
             "fit_role": None, "score": "1 - max softmax(logits)",
             "caveat": "head confidence is not a validated zero-day detector"},
            {"id": "nearest_centroid_l2", "requires": "frozen 32-D representation",
             "fit_role": "train", "score": "minimum Euclidean distance to known-family means",
             "centroid_policy": "one float64 mean per known family, sorted class order"},
        ],
        "selection_roles": ["meta_known", "surrogate_ood"],
        "selection_metric": "development AUROC: surrogate_ood positive, meta_known negative",
        "selection_tie_break": "nearest_centroid_l2 on exact tie",
        "selection_time": "after checkpoint and scorer parameters freeze, before calibration",
        "surrogate_composition": "all four non-target OOD families from frozen P3 manifest; no resampling",
        "model_or_architecture_selection": "not authorized in initial P5",
        "selection_executed_in_initial_p5": False,
    }
    threshold = {
        "version": "p5-known-only-quantile-v1", "fitted": False,
        "prerequisites": ["frozen checkpoint", "frozen scorer family", "frozen scorer parameters"],
        "fit_role": "calibration", "known_only": True, "surrogate_allowed": False,
        "target_allowed": False, "quantile": 0.99, "numpy_method": "higher",
        "rule": "anomaly score > 0.99 quantile of known calibration scores is flagged",
        "interpretation": "target-independent known-only calibration; no guaranteed population FPR",
        "calibration_design_selected_by_meta_or_test": False,
    }
    protocol = {
        "version": "p5-development-training-protocol-v1", "p3_collection_sha256": P3_HASH,
        "p4_required_status": "MODEL_INTERFACE_PASS", "first_objective": "known_family_cross_entropy",
        "objective_comparison": {
            "supervised_known_family": "selected: one CE loss, temporary head, auditable labels; no OOD claim",
            "continuous_reconstruction": "deferred: presupposes reconstruction error as anomaly signal",
            "metric_representation": "deferred: deterministic pair mining and extra choices not yet justified",
        },
        "stages": ["A: 15 dry runs, no optimizer", "B: one Fuzzers/42/10042 smoke run",
                   "C: identical Fuzzers/42/10042 rerun and parameter comparison"],
        "stage_a_status": "P5_PROTOCOL_PASS only after all 15 dry runs",
        "stage_c_status": "P5_SMOKE_PASS only after reproducibility verification",
        "future_model_selection": "checkpoint by val loss; scorer by meta_known versus fixed surrogate; threshold by calibration known-only; official test later only",
        "failure_criteria": ["P3/P4 hash mismatch", "role or identity overlap", "non-train source",
                             "target/surrogate in fit", "float32 or nonfinite tensor",
                             "non-deterministic rerun beyond 1e-12", "official-test access",
                             "checkpoint provenance mismatch"],
        "official_test_access": False, "test_effectiveness_claim": False,
    }
    return {"protocol.json": protocol, "training_config.json": TRAINING_CONFIG,
            "role_permission_matrix.json": role_matrix, "scorer_contract.json": scorer,
            "threshold_contract.json": threshold}


def all_dry_runs(root: Path) -> dict:
    existing_matrix = checked_output(root, "p5_matrix.json")
    if existing_matrix.exists() and json.loads(existing_matrix.read_text()).get("smoke_runs"):
        raise FileExistsError("Stage A cannot reset an existing smoke gate")
    cells = [validate_cell(root, target, seed, 10000 + seed)
             for target in TARGETS for seed in SPLIT_SEEDS]
    for filename, payload in protocol_artifacts().items():
        write_immutable(checked_output(root, filename), json_bytes(payload))
    for cell in cells:
        path = checked_output(root, f"dry_runs/{cell['target'].lower()}_seed{cell['split_seed']}.json")
        write_immutable(path, json_bytes(cell))
    source_files, source_digest = source_hash(root)
    matrix = {"version": "p5-training-protocol-matrix-v1", "status": "P5_PROTOCOL_PASS",
              "p3_collection_sha256": P3_HASH, "training_config_sha256": stable_hash(TRAINING_CONFIG),
              "source_hashes": source_files, "source_hash": source_digest,
              "cells": [{"target": cell["target"], "split_seed": cell["split_seed"],
                         "model_seed": cell["model_seed"], "p3_manifest_sha256": cell["p3_manifest_sha256"],
                         "p4_model_config_sha256": cell["p4_model_config_sha256"],
                         "status": cell["status"]} for cell in cells],
              "dry_run_optimizer_steps": 0, "smoke_optimizer_steps": 0,
              "official_test_access": False,
              "smoke_run": None, "reproducibility_check": None,
              "scientific_effectiveness_claim": False}
    # The gate record is phase-evolving; per-cell dry runs and protocol contracts are immutable.
    checked_output(root, "p5_matrix.json").write_bytes(json_bytes(matrix))
    return matrix


def update_matrix(root: Path, *, run: dict | None = None, reproducibility: dict | None = None) -> dict:
    path = checked_output(root, "p5_matrix.json")
    matrix = json.loads(path.read_text())
    if matrix["status"] not in {"P5_PROTOCOL_PASS", "P5_SMOKE_PASS"} or len(matrix["cells"]) != 15:
        raise ValueError("P5 Stage A gate missing")
    if run is not None:
        matrix.setdefault("smoke_runs", {})[run["run_id"]] = {
            "selected_epoch": run["selected_epoch"], "checkpoint_sha256": run["checkpoint_sha256"],
            "checkpoint_state_sha256": run["checkpoint_state_sha256"],
            "optimizer_steps": run["optimizer_steps"]}
        matrix["smoke_run"] = "completed; development-only, no effectiveness claim"
        matrix["smoke_optimizer_steps"] = sum(
            item["optimizer_steps"] for item in matrix["smoke_runs"].values()
        )
    if reproducibility is not None:
        matrix["reproducibility_check"] = reproducibility
        matrix["status"] = reproducibility["status"]
    path.write_bytes(json_bytes(matrix))
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=TARGETS)
    parser.add_argument("--split-seed", type=int, choices=SPLIT_SEEDS)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--all-dry-runs", action="store_true")
    parser.add_argument("--enable-training", action="store_true")
    parser.add_argument("--run-tag", choices=("smoke-a", "smoke-b"))
    parser.add_argument("--verify-smoke", action="store_true")
    args = parser.parse_args()
    if args.all_dry_runs:
        if args.enable_training or args.target or args.run_tag or args.verify_smoke:
            parser.error("all-dry-runs cannot be combined with training/individual cell")
        matrix = all_dry_runs(ROOT)
        print(f"[P5] {matrix['status']}: {len(matrix['cells'])}/15 dry runs, no optimizer")
        return 0
    if args.verify_smoke:
        if args.enable_training or args.target or args.run_tag:
            parser.error("verify-smoke cannot be combined with training/individual cell")
        result = verify_smoke(ROOT)
        update_matrix(ROOT, reproducibility=result)
        print(f"[P5] {result['status']}: max parameter difference {result['max_absolute_parameter_difference']}")
        return 0
    if args.target is None or args.split_seed is None or args.model_seed is None:
        parser.error("target, split-seed and model-seed are required for a cell")
    dry = validate_cell(ROOT, args.target, args.split_seed, args.model_seed)
    if not args.enable_training:
        if args.run_tag:
            parser.error("run-tag is only valid with --enable-training")
        print(json.dumps(dry, sort_keys=True))
        return 0
    if (args.target, args.split_seed, args.model_seed) != ("Fuzzers", 42, 10042):
        parser.error("initial P5 training is restricted to Fuzzers/42/10042 smoke")
    if args.run_tag is None:
        parser.error("explicit --run-tag smoke-a or smoke-b required")
    matrix = json.loads(checked_output(ROOT, "p5_matrix.json").read_text())
    if matrix["status"] != "P5_PROTOCOL_PASS" or matrix["source_hash"] != dry["source_hash"]:
        raise ValueError("complete Stage A gate with matching source required before training")
    if args.run_tag == "smoke-b" and "p5-fuzzers-s42-m10042-smoke-a" not in matrix.get("smoke_runs", {}):
        raise ValueError("smoke-a must complete before smoke-b")
    run = train_smoke(ROOT, args.run_tag)
    update_matrix(ROOT, run=run)
    print(f"[P5] {run['run_id']} selected epoch {run['selected_epoch']}; DEVELOPMENT SMOKE ONLY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
