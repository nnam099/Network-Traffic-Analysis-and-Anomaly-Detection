#!/usr/bin/env python3
"""Run the 5x3 target-isolated LOFO data gate without neural training."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ids.dataset import (
    ZERO_DAY_ATTACK_CATS,
    assert_lofo_retraining_gate,
    load_official_unsw_splits,
)
from ids.lineage import _git_metadata, _package_versions
from ids.preprocessing_p1 import (
    CATEGORY_ENCODING_VERSION, MODEL_INPUT_IDENTITY_VERSION,
    REQUIRED_UNICODE_DATA_VERSION,
)
from ids.target_isolation import (
    P0_SCHEMA_SHA256,
    audit_previous_fuzzers_normal_calibration,
    build_data_quality_report,
    build_target_isolation_context,
    prepare_target_isolated_fold,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT_DIR / "data")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "results/data_quality/p1/target_isolated_lofo_dry_run.json",
    )
    parser.add_argument(
        "--collision-report",
        type=Path,
        default=ROOT_DIR / "results/data_quality/p1/label_collision_report.json",
    )
    parser.add_argument(
        "--fail-on-gate", action="store_true",
        help="Compatibility flag: failed or incomplete gates always return exit code 2.",
    )
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("at least one seed is required; seeds must be unique")

    roles = load_official_unsw_splits(args.data_dir)
    context = build_target_isolation_context(roles["train"], roles["test"])
    data_quality = build_data_quality_report(context)
    data_quality["previous_fuzzers_normal_calibration"] = (
        audit_previous_fuzzers_normal_calibration(context)
    )
    data_quality["population"] = (
        "Official train and official test combined, before target purging; "
        "diagnostics only, never used to select or tune a model."
    )
    lineage = {
        "git": _git_metadata(ROOT_DIR),
        "package_versions": _package_versions(),
        "dataset_sha256": {
            name: _sha256_file(args.data_dir / name)
            for name in ("UNSW_NB15_training-set.csv", "UNSW_NB15_testing-set.csv")
        },
        "source_sha256": {
            name: _sha256_file(ROOT_DIR / name)
            for name in (
                "src/ids/dataset.py", "src/ids/target_isolation.py",
                "src/ids/preprocessing_p1.py", "src/ids/collision_audit.py",
                "src/ids/backdoors_forensics.py",
                "scripts/audit_target_isolated_lofo.py",
            )
        },
    }
    data_quality["lineage"] = lineage
    args.collision_report.parent.mkdir(parents=True, exist_ok=True)
    args.collision_report.write_text(
        json.dumps(data_quality, indent=2) + "\n", encoding="utf-8"
    )

    cells = []
    evidence_dir = args.output.parent / "target_isolated_evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    expected_schema = P0_SCHEMA_SHA256
    for target in ZERO_DAY_ATTACK_CATS:
        for seed in seeds:
            print(f"[DRY-RUN] target={target} seed={seed}", flush=True)
            try:
                fold = prepare_target_isolated_fold(
                    context,
                    target,
                    seed,
                    expected_schema_sha256=expected_schema,
                    include_post_transform_label_diagnostic=True,
                    include_backdoors_forensics=True,
                )
            except (AssertionError, ValueError) as exc:
                cells.append({
                    "target_family": target, "seed": seed, "gate": "FAIL",
                    "feature_schema_sha256": context["feature_schema_sha256"],
                    "error": str(exc), "invariants": {"pipeline_completed": False},
                })
                print(f"  FAIL: {exc}", flush=True)
                continue
            try:
                assert_lofo_retraining_gate(fold)
            except AssertionError as exc:
                fold["report"]["gate"] = "FAIL"
                fold["report"]["gate_error"] = str(exc)
            else:
                fold["report"]["gate"] = "PASS"
            if "backdoors_forensics" in fold["report"]:
                forensic = fold["report"].pop("backdoors_forensics")
                forensic_path = args.output.parent / f"backdoors_seed{seed}_forensics.json"
                forensic_path.write_text(json.dumps(forensic, indent=2) + "\n", encoding="utf-8")
                fold["report"]["backdoors_forensics"] = {
                    key: value for key, value in forensic.items() if key != "collisions"
                }
                fold["report"]["backdoors_forensics"].update({
                    "artifact_path": _display_path(forensic_path),
                    "artifact_sha256": _sha256_file(forensic_path),
                })
                # The archived Phase 3 split is reconstructed separately from
                # its saved backbone IDs and shared-fingerprint lists. Its
                # current-fold counterfactual is diagnostic, not a data gate.
            evidence_path = evidence_dir / f"{target.lower()}_seed{seed}.npz"
            role_evidence = {
                f"{role}_{field}": values
                for role, evidence in fold["fit_role_evidence"].items()
                for field, values in evidence.items()
                if field in ("row_ids", "canonical_fingerprints", "model_input_fingerprints")
            }
            np.savez_compressed(
                evidence_path,
                scaler_fit_row_ids=fold["scaler_fit_row_ids"],
                scaler_fit_canonical_fingerprints=(
                    fold["scaler_fit_canonical_fingerprints"]
                ),
                target_canonical_fingerprints=fold["target_canonical_fingerprints"],
                target_model_input_fingerprints=fold["target_model_input_fingerprints"],
                scaler_center=fold["scaler"].center_,
                scaler_scale=fold["scaler"].scale_,
                **role_evidence,
            )
            fold["report"]["categorical_maps"] = fold["categorical_maps"]
            fold["report"]["scaler_fit_evidence"].update({
                "artifact_path": _display_path(evidence_path),
                "artifact_sha256": _sha256_file(evidence_path),
            })
            cells.append(fold["report"])
            print(
                f"  {fold['report']['gate']} "
                f"counts={fold['report']['split_counts']} "
                f"post_target={fold['report']['post_transform_target_intersections']}",
                flush=True,
            )

    schema_hashes = sorted(set(cell["feature_schema_sha256"] for cell in cells))
    complete_matrix = set(seeds) == {42, 43, 44} and len(cells) == 15
    report = {
        "version": 2,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "target_isolated_outer_lofo_v1",
        "neural_training_performed": False,
        "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
        "categorical_encoding_version": CATEGORY_ENCODING_VERSION,
        "unicode_data_version": REQUIRED_UNICODE_DATA_VERSION,
        "ambiguity_exclusion_active": False,
        "targets": list(ZERO_DAY_ATTACK_CATS),
        "seeds": seeds,
        "feature_count": len(context["feat_cols"]),
        "feature_schema": context["feat_cols"],
        "feature_schema_sha256": context["feature_schema_sha256"],
        "required_p0_schema_sha256": expected_schema,
        "schema_hashes_observed": schema_hashes,
        "schema_consistent": schema_hashes == [expected_schema],
        "complete_required_5x3_matrix": complete_matrix,
        "all_cells_passed": (
            complete_matrix and schema_hashes == [expected_schema]
            and all(cell["gate"] == "PASS" for cell in cells)
        ),
        "lineage": lineage,
        "cells": cells,
        "data_quality_report": _display_path(args.collision_report),
    }
    data_quality["post_transform_by_fold"] = {
        f"{cell['target_family']}/seed{cell['seed']}": cell["post_transform_label_collisions"]
        for cell in cells if "post_transform_label_collisions" in cell
    }
    args.collision_report.write_text(
        json.dumps(data_quality, indent=2) + "\n", encoding="utf-8"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Dry-run report: {args.output}")
    print(f"Collision report: {args.collision_report}")
    print(f"ALL CELLS: {'PASS' if report['all_cells_passed'] else 'FAIL'}")
    if not report["all_cells_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
