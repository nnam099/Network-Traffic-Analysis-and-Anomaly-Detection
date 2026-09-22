#!/usr/bin/env python3
"""Build immutable P8A/P8B train-only LOAFO manifests; never train or evaluate."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import DEFAULT_TRAIN_FILE, load_unsw_csvs  # noqa: E402
from ids.p3_protocol import sha256_json, train_ambiguity_report, train_only_context  # noqa: E402
from ids.p8_loafo_protocol import (  # noqa: E402
    CONSUMED_TEST_LABEL, P3_FREEZE_SHA256, SEEDS, SURROGATE_POLICY, VERSION,
    cell_manifest, construct_roles, file_sha256, json_bytes, support_tier,
    validate_cell, verified_family_universe, verify_frozen_p3, write_immutable,
)


def git_head() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()


def protocol_artifacts(universe: dict, schema: dict, provenance: dict) -> dict[str, dict]:
    support = {
        target: {
            "original_official_train_rows": universe["families"][target]["official_train_source_rows"],
            "post_ambiguity_official_train_rows": universe["families"][target]["official_train_retained_rows"],
            "original_official_train_canonical_identities": universe["families"][target]["official_train_source_canonical_identities"],
            "development_held_out_canonical_identities": universe["families"][target]["official_train_retained_canonical_identities"],
            "potential_new_external_identities": None,
            "historical_consumed_test_clean_identities_not_available_for_new_independent_validation": universe["families"][target]["historical_consumed_official_test_clean_canonical_identities"],
            "support_category": support_tier(universe["families"][target]["official_train_retained_canonical_identities"]),
        }
        for target in universe["attack_families"]
    }
    roles = {
        "version": VERSION,
        "scientific_unit": "canonical-model-features-v1 identity, not row",
        "source": "official_train only; whole conflicting canonical groups excluded from all roles",
        "roles": ["train", "val", "meta_known", "calibration", "surrogate_ood", "target_held_out"],
        "split": "StratifiedGroupKFold 10 folds, 0-6 train, 7 val, 8 meta_known, 9 calibration",
        "split_seeds": list(SEEDS), "model_seed_rule": "10000 + split_seed",
        "train_only_surrogate": SURROGATE_POLICY,
        "fit_permissions": {
            "scaler_and_vocabulary": ["train"], "gradient_and_representation": ["train"],
            "checkpoint_selection": ["val"], "centroid_fit": ["train"],
            "scorer_family_selection": ["meta_known", "surrogate_ood"],
            "threshold_fit": ["calibration"], "held_out_target_fit": [],
            "consumed_official_test_selection": [],
        },
        "target_isolation": "target label and canonical identities absent from every fitting/selection role",
        "structured_input_collision": "fail if target and fit canonical identities collapse to one final model input",
        "historical_test_policy": CONSUMED_TEST_LABEL,
        "no_training_in_p8": True,
    }
    surrogate = {
        "version": VERSION, "primary_policy": SURROGATE_POLICY,
        "ranking_data": "official_train post-ambiguity canonical identity counts only",
        "ranking_rule": "descending remaining-family identity count, exact label ascending tie-break; exclude target; top two",
        "candidate_S1": {
            "rule": "top one remaining non-target attack family by same train-only ranking",
            "advantage": "more known attack classes and a simple single OOD contrast",
            "risk": "single surrogate's behavior may not transfer to held-out target",
            "leakage": "safe only if family is absent from train/val/calibration and no target/test signals rank it",
            "selected": False,
        },
        "candidate_S2": {
            "rule": "top two remaining non-target attack families by same train-only ranking",
            "advantage": "multiple distinct OOD families for scorer selection",
            "risk": "removes two classes from representation fit; still may not represent target",
            "leakage": "surrogate identities forbidden from backbone, scaler, vocab, checkpoint, centroid, calibration",
            "selected": True,
        },
        "frozen_before_P9": True, "P7_results_used_for_choice": False,
    }
    model = {
        "version": VERSION, "schema_sha256": schema["sha256"],
        "structured_input": {"continuous": "torch.float64[B,58]", "proto": "torch.int64[B]",
                             "service": "torch.int64[B]", "state": "torch.int64[B]"},
        "vocabulary": "fold-local train only; immutable per-feature PAD=0, OOV=1; no modulo hashing",
        "reference_architecture": {"continuous_dims": [58, 64, 32],
                                   "categorical_embedding_dims": {"proto": 4, "service": 4, "state": 4},
                                   "fusion_dims": [44, 64, 32]},
        "reference_architecture_implementation": "P8ReferenceBackbone-v1 fixed-4 adapter; P4 generic vocabulary-size heuristic is not applied because LOAFO class mix can enlarge proto vocabulary",
        "architecture_change_disclosure": "This preserves the concrete P4/P5 44-D baseline; any other embedding width is a separate architecture ablation and protocol amendment",
        "normal_policy": "Normal is an in-distribution known classifier class, never a zero-day target",
        "class_order": "Normal first, then remaining fitted attack families in exact lexical order",
        "temporary_head": "Linear(32, len(known_class_order), dtype=torch.float64)",
        "objective": "known-class cross entropy on train only; P8 does not instantiate optimizer or train",
        "training_reference": {
            "optimizer": "AdamW", "lr": 0.001, "weight_decay": 0.01,
            "batch_size": 512, "max_epochs": 8, "gradient_clip_norm": 1.0,
            "dtype": "torch.float64", "amp": False,
            "checkpoint_selection": "minimum sample-mean val cross entropy; patience=2; min_delta=0; lowest epoch on exact tie",
            "model_seed_rule": "10000 + split_seed",
        },
        "scorers": ["1 - max softmax", "min_k Euclidean_distance(z, train_known_centroid_k)"],
        "centroid_fit_role": "train",
        "scorer_selection_statistic": "canonical identity arithmetic mean of row scores, then identity-level AUROC on meta_known vs surrogate_ood",
        "scorer_selection_tie_tolerance": 1e-12,
        "scorer_selection_tie_preference": "nearest_centroid_l2",
        "threshold": {"role": "known-only calibration", "quantile": 0.99,
                      "method": "higher", "alert_rule": "score > threshold"},
        "development_metrics_label": "DEVELOPMENT ONLY",
        "architecture_ablation_required_for_change": True,
        "neural_training_performed": False,
    }
    external = {
        "version": VERSION, "official_test_status": "consumed by P7; not a new validation source",
        "reused_test_required_label": CONSUMED_TEST_LABEL,
        "categories": {
            "DEVELOPMENT": "P8/P9 official-train-derived roles only; may guide development under protocol",
            "REUSED_TEST_DIAGNOSTIC": "old official test only if separately authorized later, never independent external validation",
            "NEW_EXTERNAL_VALIDATION": "new untouched preregistered partition, genuinely external dataset, or newly collected traffic; P10 only",
        },
        "future_candidate_assessment": {
            "CIC-IDS2017": "Different flow extraction, feature and attack taxonomy; 58-feature mapping unverified; do not ingest as-is",
            "CSE-CIC-IDS2018": "Different schema and temporal distribution; semantic labels and categorical domains require mapping",
            "CIC-DDoS2019": "DDoS-focused label coverage; not a complete LOAFO-family external universe; feature map unverified",
            "TON_IoT": "Different telemetry domains and background definition; likely incompatible without new interface study",
            "Bot-IoT": "Different traffic generation and severe shift; label and feature mappings unverified",
            "new_reserved_UNSW_partition": "Potential schema match, only independent if cryptographically reserved before any P8/P9 decision",
            "newly_collected_traffic": "Potentially strongest temporal validation; requires frozen feature extractor and independent provenance",
        },
        "pre_P10_requirements": ["semantic attack mapping", "feature mapping for all 58 continuous fields",
                                 "missing feature behavior", "proto/service/state OOV behavior",
                                 "normal/background definition", "source hash and reservation evidence",
                                 "one-way evaluation intent and immutable protocol"],
        "download_or_external_access_performed": False,
    }
    return {
        "attack_family_universe.json": universe,
        "target_support_audit.json": {"version": VERSION, "tiers_train_only": {
            "HIGH SUPPORT": ">=1000 identities", "MODERATE SUPPORT": "200-999 identities",
            "SMALL-N": "50-199 identities", "INSUFFICIENT": "<50 identities"},
            "descriptive_only": True, "future_external_support_unknown": True, "targets": support},
        "loafo_role_contract.json": roles,
        "surrogate_policy_design.json": surrogate,
        "model_contract.json": model,
        "external_validation_plan.json": external,
        "provenance.json": provenance,
    }


def run(output_dir: Path) -> dict:
    freeze, frozen_ambiguity, schema, historical = verify_frozen_p3(ROOT)
    train_path = ROOT / "data" / DEFAULT_TRAIN_FILE
    train_hash = file_sha256(train_path)
    if train_hash != freeze["source_dataset_hashes"]["official_train_sha256"]:
        raise ValueError("official train source hash differs from P3 freeze")
    train = load_unsw_csvs(train_path.parent, files=[train_path], source_role="official_train")
    context = train_only_context(train)
    ambiguous, ambiguity = train_ambiguity_report(context)
    if sha256_json(ambiguity) != sha256_json(frozen_ambiguity):
        raise ValueError("recomputed train-only ambiguity differs from P3 freeze")
    universe = verified_family_universe(context, ambiguity, historical)
    historical_targets = {entry["target"] for entry in freeze["cells"]}
    historical_surrogates = set()
    for entry in freeze["cells"]:
        prior = json.loads(gzip.decompress((ROOT / "results/data_quality/p3" /
                                            entry["manifest_filename"]).read_bytes()))
        historical_surrogates.update(prior["roles"]["surrogate_ood"]["label_counts"])
    for label, family in universe["families"].items():
        family["historically_p3_target"] = label in historical_targets
        family["historically_p3_surrogate"] = label in historical_surrogates
    provenance = {
        "repository_head_before_p8": git_head(),
        "p3_development_freeze_sha256": P3_FREEZE_SHA256,
        "official_train_sha256": train_hash,
        "p3_schema_sha256": schema["sha256"],
        "p3_frozen_historical_test_audit_artifact_sha256": file_sha256(
            ROOT / "results/data_quality/p3/official_test_external_audit.json"),
        "p8_source_sha256": {
            name: file_sha256(ROOT / name) for name in (
                "src/ids/p8_loafo_protocol.py", "scripts/audit_p8_loafo.py",
                "tests/test_p8_loafo_protocol.py", "docs/p8_loafo_protocol.md")
        },
        "historical_test_used_for": "descriptive support counts only; never selection",
        "official_test_csv_accessed": False, "p7_primary_results_accessed": False,
    }
    artifacts = protocol_artifacts(universe, schema, provenance)
    design_hash = sha256_json({name: artifacts[name] for name in sorted(artifacts)})
    cells = []
    with tempfile.TemporaryDirectory(prefix="p8_loafo_candidate_") as candidate:
        staging = Path(candidate)
        for target in universe["attack_families"]:
            for seed in SEEDS:
                print(f"[P8 TRAIN ONLY] {target}/{seed}", flush=True)
                population, roles, class_order, surrogates = construct_roles(
                    context, ambiguous, target, seed, universe)
                audit, fitting = validate_cell(context, schema, population, roles,
                                               target, class_order, surrogates)
                manifest = cell_manifest(context, schema, universe, population, roles,
                                         target, seed, class_order, surrogates,
                                         audit, fitting, provenance)
                raw = json_bytes(manifest)
                compressed = gzip.compress(raw, compresslevel=9, mtime=0)
                filename = f"manifests/{target.lower()}_seed{seed}.json.gz"
                path = staging / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(compressed)
                cells.append({
                    "target": target, "split_seed": seed, "model_seed": 10000 + seed,
                    "known_class_order": class_order, "num_known_classes": len(class_order),
                    "surrogate_families": surrogates,
                    "target_held_out_rows": len(population["target_train"]),
                    "target_held_out_identities": len(population["target_identity_hashes"]),
                    "role_rows": {name: len(frame) for name, frame in roles.items()},
                    "role_canonical_identities": {name: int(frame["__canonical_feature_fingerprint"].nunique())
                                                  for name, frame in roles.items()},
                    "known_train_class_support": {
                        label: {"rows": int((roles["train"].attack_cat == label).sum()),
                                "canonical_identities": int(roles["train"].loc[
                                    roles["train"].attack_cat == label,
                                    "__canonical_feature_fingerprint"].nunique())}
                        for label in class_order
                    },
                    "surrogate_class_support": {
                        label: {"rows": int((roles["surrogate_ood"].attack_cat == label).sum()),
                                "canonical_identities": int(roles["surrogate_ood"].loc[
                                    roles["surrogate_ood"].attack_cat == label,
                                    "__canonical_feature_fingerprint"].nunique())}
                        for label in surrogates
                    },
                    "canonical_role_intersections": audit["canonical_role_intersections"],
                    "structured_model_input_cross_identity_collisions": audit[
                        "model_input_cross_identity_final"],
                    "scaler_fit_row_ids_sha256": fitting["scaler_fit_row_ids_sha256"],
                    "vocabulary_sha256": fitting["vocabulary_sha256"],
                    "manifest_filename": filename,
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "manifest_artifact_sha256": hashlib.sha256(compressed).hexdigest(),
                    "gate": "LOAFO_PROTOCOL_PASS",
                })
        matrix = {
            "version": VERSION, "status": "LOAFO_PROTOCOL_PASS",
            "verified_attack_family_count": len(universe["attack_families"]),
            "split_seeds": list(SEEDS), "expected_cells": len(universe["attack_families"]) * len(SEEDS),
            "passed_cells": len(cells), "protocol_design_sha256": design_hash,
            "source_provenance": provenance, "cells": cells,
            "official_test_csv_accessed": False, "p7_primary_results_accessed": False,
            "neural_training_performed": False, "optimizer_steps": 0,
            "checkpoints_created": 0, "target_performance_computed": False,
            "external_evaluation_performed": False,
        }
        if len(cells) != matrix["expected_cells"]:
            raise ValueError("incomplete LOAFO target x seed matrix")
        artifacts["loafo_matrix.json"] = matrix
        for name, payload in artifacts.items():
            (staging / name).write_bytes(json_bytes(payload))
        # Verify every candidate manifest before publishing any P8 artifact.
        for cell in cells:
            raw = gzip.decompress((staging / cell["manifest_filename"]).read_bytes())
            if hashlib.sha256(raw).hexdigest() != cell["manifest_sha256"]:
                raise ValueError("candidate P8 manifest hash mismatch")
        # Fail before the first write if a prior immutable artifact differs.
        for path in staging.rglob("*"):
            if path.is_file():
                destination = output_dir / path.relative_to(staging)
                if destination.exists() and destination.read_bytes() != path.read_bytes():
                    raise FileExistsError(f"existing P8 freeze differs: {destination}")
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                write_immutable(output_dir / path.relative_to(staging), path.read_bytes())
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/loafo/p8")
    args = parser.parse_args()
    matrix = run(args.output_dir)
    print(f"{matrix['status']}: {matrix['passed_cells']}/{matrix['expected_cells']} cells")
    print(f"protocol_design_sha256={matrix['protocol_design_sha256']}")
    print("Official-test evaluation has NOT been executed in P8.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
