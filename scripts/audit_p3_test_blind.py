#!/usr/bin/env python3
"""Freeze 15 official-train-only data states, THEN inspect official test."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import DEFAULT_TEST_FILE, DEFAULT_TRAIN_FILE, load_unsw_csvs  # noqa: E402
from ids.p2_protocol import build_schema_v2  # noqa: E402
from ids.p3_protocol import (  # noqa: E402
    PROTOCOL_VERSION, canonical_json, classify_official_test,
    construct_development_roles, development_manifest, development_model_audit,
    investigate_historical_738, post_freeze_model_input_audit,
    prepare_official_test, sha256_json,
    test_support, train_ambiguity_report, train_only_context,
)


TARGETS = ("Fuzzers", "Analysis", "Backdoors", "Shellcode", "Worms")
SEEDS = (42, 43, 44)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def commit_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value) -> None:
    path.write_bytes(json.dumps(value, indent=2, ensure_ascii=False,
                                allow_nan=False).encode("utf-8") + b"\n")


def write_manifest(path: Path, value: dict) -> str:
    raw = canonical_json(value)
    digest = hashlib.sha256(raw).hexdigest()
    compressed = gzip.compress(raw, compresslevel=9, mtime=0)
    if path.exists() and path.read_bytes() != compressed:
        raise AssertionError(f"frozen manifest differs from existing file: {path}")
    path.write_bytes(compressed)
    return digest


def verify_freeze(output_dir: Path, freeze: dict) -> None:
    for cell in freeze["cells"]:
        artifact = output_dir / cell["manifest_filename"]
        if hashlib.sha256(gzip.decompress(artifact.read_bytes())).hexdigest() != cell[
            "manifest_sha256"
        ]:
            raise AssertionError(f"development manifest changed: {artifact}")
        manifest = json.loads(gzip.decompress(artifact.read_bytes()))
        if manifest["official_test_consulted"] is not False:
            raise AssertionError("manifest claims test use")
    frozen = output_dir / "development_freeze.json"
    if json.loads(frozen.read_text()) != freeze:
        raise AssertionError("development freeze changed")


def development_stage(train_frame, *, output_dir: Path, train_sha: str,
                      code_hashes: dict, commit: str | None) -> tuple[dict, dict, dict, list]:
    """This callable has no test frame, path, label, or identity parameter."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    context = train_only_context(train_frame)
    schema = build_schema_v2(context)
    ambiguous, ambiguity = train_ambiguity_report(context)
    write_json(output_dir / "schema_v2.json", schema)
    write_json(output_dir / "train_only_ambiguity.json", ambiguity)
    cells = []
    frozen_entries = []
    for target in TARGETS:
        for seed in SEEDS:
            print(f"[P3 TRAIN ONLY] {target}/{seed}", flush=True)
            population, roles = construct_development_roles(
                context, target, seed, ambiguous
            )
            audit, fitting, _ = development_model_audit(
                context, schema, population, roles
            )
            manifest = development_manifest(
                context, schema, ambiguity, population, roles, audit, fitting,
                target=target, seed=seed, train_source_sha256=train_sha,
                commit_sha=commit, code_sha256=code_hashes,
            )
            filename = f"manifests/{target.lower()}_seed{seed}.json.gz"
            digest = write_manifest(output_dir / filename, manifest)
            frozen_entries.append({
                "target": target, "seed": seed, "manifest_filename": filename,
                "manifest_sha256": digest,
            })
            cells.append({
                "target": target, "seed": seed,
                "development_rows_before_ambiguity_exclusion": len(context["train"]),
                "train_only_ambiguous_groups_excluded": ambiguity["excluded_groups"],
                "train_only_ambiguous_rows_excluded": ambiguity["excluded_rows"],
                "development_rows_after_ambiguity_exclusion": len(population["retained"]),
                "target_rows_held_out_from_fit": len(population["target_train"]),
                "target_identity_groups_purged": len(population["target_identity_hashes"]),
                "known_rows_removed_by_target_identity_purge": population["known_rows_purged"],
                "surrogate_rows_removed_by_target_identity_purge": population[
                    "surrogate_rows_purged"
                ],
                "role_rows": {name: len(frame) for name, frame in roles.items()},
                "role_canonical_intersections": audit["canonical_role_intersections"],
                "model_input_audit": audit,
                "vocabulary_provenance": {
                    "source": fitting["vocabulary_source_role"],
                    "sha256": fitting["vocabulary_sha256"],
                    "sizes": fitting["vocabulary_sizes"],
                },
                "scaler_provenance": {
                    "source": fitting["scaler_source_role"],
                    "center_scale_sha256": fitting["scaler_center_scale_sha256"],
                    "fit_row_count": fitting["scaler_fit_row_count"],
                },
                "development_manifest_sha256": digest,
                "development_gate": audit["gate"]["gate"],
                "development_fail_reasons": audit["gate"]["fail_reasons"],
            })
    freeze = {
        "version": PROTOCOL_VERSION, "phase": "development frozen before official test open",
        "official_test_consulted": False,
        "source_dataset_hashes": {"official_train_sha256": train_sha},
        "code_commit_sha": commit, "code_source_sha256": code_hashes,
        "canonical_contract_version": context["canonical_identity_version"],
        "schema_version": schema["contract"]["version"],
        "schema_sha256": schema["sha256"],
        "train_ambiguity_report_sha256": sha256_json(ambiguity),
        "cells": frozen_entries,
    }
    write_json(output_dir / "development_freeze.json", freeze)
    verify_freeze(output_dir, freeze)
    return context, ambiguity, freeze, cells


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results/data_quality/p3")
    parser.add_argument("--development-only", action="store_true",
                        help="freeze from official train without even opening test")
    args = parser.parse_args()
    train_path = args.data_dir / DEFAULT_TRAIN_FILE
    train_hash = file_sha256(train_path)
    train = load_unsw_csvs(args.data_dir, files=[train_path],
                           source_role="official_train")
    code_files = (
        "src/ids/dataset.py", "src/ids/target_isolation.py",
        "src/ids/preprocessing_p1.py", "src/ids/collision_audit.py",
        "src/ids/p2_protocol.py",
        "src/ids/p3_protocol.py", "scripts/audit_p3_test_blind.py",
    )
    code_hashes = {name: file_sha256(ROOT / name) for name in code_files}
    context, ambiguity, freeze, cells = development_stage(
        train, output_dir=args.output_dir, train_sha=train_hash,
        code_hashes=code_hashes, commit=commit_sha(),
    )
    freeze_hash = sha256_json(freeze)
    print(f"[P3 FROZEN] {len(cells)} manifests; collection SHA-256 {freeze_hash}", flush=True)
    if args.development_only:
        print("[P3 STOP] Official test was not opened.", flush=True)
        return 2 if any(cell["development_gate"] == "FAIL" for cell in cells) else 0

    # FIRST official-test read occurs below, after all 15 manifests and their
    # hash-index freeze have been persisted and re-verified.
    test_path = args.data_dir / DEFAULT_TEST_FILE
    test = load_unsw_csvs(args.data_dir, files=[test_path],
                          source_role="official_test")
    test = prepare_official_test(test, context)
    classified, external = classify_official_test(
        test, context, set(int(x) for x in ambiguity["excluded_identity_hashes"])
    )
    external["source_dataset_hashes"] = {"official_test_sha256": file_sha256(test_path)}
    external["development_freeze_sha256"] = freeze_hash
    external["primary_evaluation_surface"] = "official-test-clean"
    external["diagnostic_surfaces"] = [
        "official-test-development-overlap", "official-test-internal-ambiguous",
        "official-test-train-conflict-related",
    ]
    write_json(args.output_dir / "official_test_external_audit.json", external)
    historical = investigate_historical_738(context, test, classified)
    historical["development_freeze_sha256"] = freeze_hash
    write_json(args.output_dir / "historical_738_investigation.json", historical)
    schema = build_schema_v2(context)
    if schema["sha256"] != freeze["schema_sha256"]:
        raise AssertionError("post-freeze schema does not match frozen schema")
    frozen_entries = {(entry["target"], entry["seed"]): entry
                      for entry in freeze["cells"]}
    for cell in cells:
        support = test_support(classified, cell["target"])
        frozen_entry = frozen_entries[(cell["target"], cell["seed"])]
        manifest = json.loads(gzip.decompress(
            (args.output_dir / frozen_entry["manifest_filename"]).read_bytes()
        ))
        external_model = post_freeze_model_input_audit(
            classified, context, schema, manifest
        )
        cell["official_test_support"] = support
        cell["post_freeze_official_test_model_input_audit"] = external_model
        cell["official_test_clean_target_rows"] = support["target"]["clean_rows"]
        cell["official_test_clean_known_background_rows"] = support[
            "known_background"
        ]["clean_rows"]
        cell["support_review_required"] = cell["target"] in (
            "Analysis", "Backdoors", "Worms"
        )
        cell["support_note"] = (
            "Small-sample regime flagged descriptively; no statistical-power "
            "threshold or model metric was inferred."
            if cell["support_review_required"] else
            "Structural support count only; statistical power not inferred."
        )
        cell["data_protocol_status"] = (
            "DATA_PROTOCOL_BLOCKED"
            if cell["development_gate"] == "FAIL" or not support["target"]["clean_rows"]
            or not support["known_background"]["clean_rows"]
            or external_model["cross_identity_model_input_collision_rows_clean_test"]
            else "DATA_PROTOCOL_PASS"
        )
    verify_freeze(args.output_dir, freeze)
    if sha256_json(json.loads((args.output_dir / "development_freeze.json").read_text())) != freeze_hash:
        raise AssertionError("official test audit mutated development freeze")
    overall = (
        "BLOCKED" if any(cell["data_protocol_status"] == "DATA_PROTOCOL_BLOCKED"
                         for cell in cells)
        else "DATA PROTOCOL APPROVED FOR MODEL-DESIGN PHASE"
    )
    matrix = {
        "version": PROTOCOL_VERSION, "status": overall,
        "neural_training_performed": False,
        "threshold_tuning_performed": False,
        "scientific_performance_metrics_generated": False,
        "official_test_opened_only_after_development_freeze": True,
        "development_freeze_sha256": freeze_hash,
        "development_freeze_artifact": "development_freeze.json",
        "official_test_external_audit_artifact": "official_test_external_audit.json",
        "historical_738_artifact": "historical_738_investigation.json",
        "schema_sha256": freeze["schema_sha256"], "cells": cells,
        "training_authorized": False,
        "model_design_required_before_training": True,
    }
    write_json(args.output_dir / "p3_matrix.json", matrix)
    print(args.output_dir / "p3_matrix.json")
    print(f"[P3 STATUS] {overall}; NO TRAINING", flush=True)
    return 2 if overall == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
