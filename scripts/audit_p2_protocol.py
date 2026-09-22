#!/usr/bin/env python3
"""Data-only P2 comparison: A=P1, B=float64, C=structured, D=structured+exclusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import load_official_unsw_splits  # noqa: E402
from ids.lineage import _git_metadata, _package_versions  # noqa: E402
from ids.preprocessing_p1 import MODEL_INPUT_IDENTITY_VERSION as P1_CONTRACT  # noqa: E402
from ids.preprocessing_p1 import matrix_fingerprints, transform_stages  # noqa: E402
from ids.p2_protocol import (  # noqa: E402
    AMBIGUITY_POLICY_VERSION, FLOAT64_LEGACY_SCHEMA_VERSION, MODEL_INPUT_VERSION,
    ROLE_NAMES, _candidate_frames, ambiguity_impact,
    ambiguous_canonical_set, audit_model_inputs, build_schema_v2,
    candidate_fingerprints, continuous_matrix,
    group_split_roles, opaque_category_ids,
    role_canonical_intersections, row_split_comparator, structured_model_fingerprints,
)
from ids.target_isolation import (  # noqa: E402
    P0_SCHEMA_SHA256,
    _prepare_model_frame, build_target_isolation_context,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def saved_p1_roles(context, cell, evidence):
    train = context["train"]
    roles = {name: train.iloc[evidence[name + "_row_ids"]].copy()
             for name in ROLE_NAMES}
    population = _candidate_frames(context, cell["target_family"], set())
    expected_known = set(population["known"]["__official_train_row_id"])
    observed_known = set().union(*(set(roles[name]["__official_train_row_id"])
                                   for name in ROLE_NAMES[:4]))
    if expected_known != observed_known:
        raise AssertionError("P1 known role IDs do not reconstruct target-purged population")
    if set(population["surrogate"]["__official_train_row_id"]) != set(
        roles["surrogate_ood"]["__official_train_row_id"]
    ):
        raise AssertionError("P1 surrogate evidence differs from population")
    return population, roles


def p1_scalar_inputs(context, cell, evidence, roles, population):
    frames = {**roles, "target": population["target"],
              "official_test_known": population["test_known"]}
    scaler = RobustScaler()
    scaler.center_, scaler.scale_ = evidence["scaler_center"], evidence["scaler_scale"]
    scaler.n_features_in_ = len(context["feat_cols"])
    scaled, final32 = {}, {}
    for name, frame in frames.items():
        prepared = _prepare_model_frame(frame, cell["categorical_maps"],
                                        context["base_feat_cols"], context["feat_cols"])
        scaled[name], final32[name] = transform_stages(
            prepared[context["feat_cols"]].to_numpy(np.float64), scaler
        )
        if name in roles:
            expected = evidence[name + "_model_input_fingerprints"]
            if not np.array_equal(matrix_fingerprints(final32[name]), expected):
                raise AssertionError(f"P1 evidence does not reconstruct: {name}")
    if not np.array_equal(matrix_fingerprints(final32["target"]),
                          evidence["target_model_input_fingerprints"]):
        raise AssertionError("P1 target evidence does not reconstruct")
    return scaled, final32


def model_report(context, config, cell, roles, population, fingerprints,
                 schema_hash, contract, split_policy, ambiguity_policy,
                 vocabulary, scaler_row_ids, transform_merges, risk=None):
    result = audit_model_inputs(roles, population, fingerprints)
    target = cell["target_family"]
    if any(target in set(frame.attack_cat) for frame in roles.values()):
        result["gate"]["fail_reasons"].append({
            "severity": "CRITICAL FAIL", "reason": "target_label_in_fit_role"
        })
        result["gate"]["gate"] = "FAIL"
        result["gate"]["severity"] = "CRITICAL FAIL"
    fit_ids = np.asarray(scaler_row_ids, dtype=np.int64)
    backbone_ids = roles["train"]["__official_train_row_id"].to_numpy(np.int64)
    if not np.array_equal(np.sort(fit_ids), np.sort(backbone_ids)):
        raise AssertionError("scaler fit did not use the backbone exclusively")
    if any(name not in vocabulary for name in ("proto", "service", "state")):
        raise AssertionError("categorical vocabulary incomplete")
    source_roles = {name: sorted(frame["__source_role"].astype(str).unique())
                    for name, frame in roles.items()}
    return {
        "configuration": config, "target_family": target, "seed": cell["seed"],
        "feature_schema_contract": (
            "P0-61" if config in ("A", "B") else "scientific-feature-schema-v2"
        ),
        "schema_sha256": schema_hash, "model_input_identity_contract": contract,
        "ambiguity_policy": ambiguity_policy, "split_policy": split_policy,
        "role_rows": {name: len(frame) for name, frame in roles.items()},
        "role_labels": {name: frame.attack_cat.value_counts().sort_index().to_dict()
                        for name, frame in roles.items()},
        "official_test_known_rows": len(population["test_known"]),
        "target_train_rows": len(population["target_train"]),
        "target_official_test_rows": len(population["target_test"]),
        "fitting_provenance": {
            "vocabulary_source": "target-purged backbone train only",
            "scaler_source": "target-purged backbone train only",
            "vocabulary_sizes": {name: len(mapping) for name, mapping in vocabulary.items()},
            "scaler_fit_rows": len(fit_ids),
            "scaler_fit_row_ids_sha256": hashlib.sha256(
                np.sort(fit_ids).astype("<i8").tobytes()
            ).hexdigest(),
            "role_source_roles": source_roles,
            "official_test_used_for_parameter_fit": False,
            "official_test_labels_used_for_training_population_curation": config == "D",
        },
        "new_transform_merges": transform_merges,
        "oov_hash_risk": risk,
        "model_input_collision_audit": result,
        "data_integrity_gate": result["gate"]["gate"],
        "exact_fail_reasons": result["gate"]["fail_reasons"],
        "scientific_protocol_status": "NOT APPROVED; NO TRAINING",
    }


def structured_inputs(context, schema, roles, population):
    identities, scaler_info, vocabulary, risk = candidate_fingerprints(
        context, schema, roles, population
    )
    frames = {**roles, "target": population["target"],
              "official_test_known": population["test_known"]}
    # A conversion from raw continuous float64 to scaled float64 is audited
    # independently of the schema hash and of the final structured identity.
    raw_identifiers = {}
    for name, frame in frames.items():
        raw = continuous_matrix(frame, context, schema)
        raw_identifiers[name] = structured_model_fingerprints(
            raw, opaque_category_ids(frame, vocabulary)
        )
    before = np.concatenate([raw_identifiers[name] for name in frames])
    after = np.concatenate([identities[name] for name in frames])
    merged = pd.DataFrame({"before": before, "after": after}).groupby("after")[
        "before"
    ].nunique()
    merges = int((merged > 1).sum())
    return identities, scaler_info, vocabulary, risk, {
        "new_scaling_merges": merges,
        "new_dtype_conversion_merges": 0,
        "raw_structured_unique_identities": len(set(before)),
    }


def main() -> int:
    p1_path = ROOT / "results/data_quality/p1/target_isolated_lofo_dry_run.json"
    p1 = json.loads(p1_path.read_text())
    if not p1["complete_required_5x3_matrix"] or len(p1["cells"]) != 15:
        raise AssertionError("P1 baseline matrix is incomplete")
    loaded = load_official_unsw_splits(ROOT / "data")
    context = build_target_isolation_context(loaded["train"], loaded["test"])
    schema = build_schema_v2(context)
    ambiguous = ambiguous_canonical_set(context)
    impact = ambiguity_impact(context, ambiguous)
    destination = ROOT / "results/data_quality/p2"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "schema_v2.json").write_text(json.dumps(schema, indent=2) + "\n")
    (destination / "ambiguity_impact.json").write_text(json.dumps(impact, indent=2) + "\n")
    records, split_analysis = [], {}
    for cell in p1["cells"]:
        target, seed = cell["target_family"], cell["seed"]
        print(f"[P2 DATA ONLY] {target}/{seed}", flush=True)
        path = ROOT / cell["scaler_fit_evidence"]["artifact_path"]
        if sha(path) != cell["scaler_fit_evidence"]["artifact_sha256"]:
            raise AssertionError("P1 evidence checksum mismatch")
        with np.load(path) as evidence:
            population, roles = saved_p1_roles(context, cell, evidence)
            scaled, final32 = p1_scalar_inputs(context, cell, evidence, roles, population)
            a_fps = {name: matrix_fingerprints(values) for name, values in final32.items()}
            b_fps = {name: matrix_fingerprints(values) for name, values in scaled.items()}
            expected = cell["stage_collision_audit"]["C_scaled_float64"]["taxonomy"]
            a = model_report(
                context, "A", cell, roles, population, a_fps, P0_SCHEMA_SHA256,
                P1_CONTRACT, "P1 canonical-grouped known roles; separate surrogate",
                "inactive", cell["categorical_maps"], evidence["scaler_fit_row_ids"],
                {"encoded_new_merges": 0, "scaled_float64_new_merges": 0,
                 "float32_new_merges": cell["stage_collision_audit"]["D_model_float32"][
                     "newly_merged_fingerprints_from_previous_stage"]},
            )
            a["historical_P1_gate"] = cell["gate"]
            b = model_report(
                context, "B", cell, roles, population, b_fps, P0_SCHEMA_SHA256,
                FLOAT64_LEGACY_SCHEMA_VERSION,
                "same P1 target-purged canonical split", "inactive",
                cell["categorical_maps"], evidence["scaler_fit_row_ids"],
                {"encoded_new_merges": 0,
                 "scaled_float64_new_merges": cell["stage_collision_audit"][
                     "C_scaled_float64"]["newly_merged_fingerprints_from_previous_stage"],
                 "dtype_conversion_new_merges": 0},
            )
            if b["model_input_collision_audit"]["collision_taxonomy"] != expected:
                raise AssertionError("float64 candidate does not reproduce P1 stage C")
            c_fps, c_scaler, c_vocab, c_risk, c_merges = structured_inputs(
                context, schema, roles, population
            )
            c = model_report(
                context, "C", cell, roles, population, c_fps, schema["sha256"],
                MODEL_INPUT_VERSION, "same P1 target-purged canonical split",
                "inactive", c_vocab, c_scaler["train_row_ids"], c_merges, c_risk,
            )
        excluded, d_roles = group_split_roles(
            context, target, seed, ambiguous, exclude_ambiguous=True
        )
        d_fps, d_scaler, d_vocab, d_risk, d_merges = structured_inputs(
            context, schema, d_roles, excluded
        )
        d = model_report(
            context, "D", cell, d_roles, excluded, d_fps, schema["sha256"],
            MODEL_INPUT_VERSION,
            "whole-group ambiguity exclusion before target purge; canonical-group split",
            AMBIGUITY_POLICY_VERSION + " (dry-run only)",
            d_vocab, d_scaler["train_row_ids"], d_merges, d_risk,
        )
        records.extend((a, b, c, d))
        split_analysis[f"{target}/{seed}"] = {
            "row_level_without_exclusion": row_split_comparator(
                population["known"], population["surrogate"], seed),
            "P1_grouped_known_roles": role_canonical_intersections(roles),
            "row_level_after_exclusion": row_split_comparator(
                excluded["known"], excluded["surrogate"], seed),
            "whole_group_after_exclusion": role_canonical_intersections(d_roles),
            "conclusion": (
                "Known-only grouping leaves known/surrogate conflicts until whole "
                "ambiguous equivalence groups are excluded. Row splitting additionally "
                "splits canonical groups among known roles."
            ),
        }
    summary = {
        "version": "p2-data-only-protocol-v1",
        "neural_training_performed": False, "threshold_tuning_performed": False,
        "scientific_performance_metrics_generated": False,
        "p1_report_sha256": sha(p1_path),
        "lineage": {
            "git": _git_metadata(ROOT), "package_versions": _package_versions(),
            "dataset_sha256": p1["lineage"]["dataset_sha256"],
            "source_sha256": {name: sha(ROOT / name) for name in (
                "src/ids/dataset.py", "src/ids/target_isolation.py",
                "src/ids/preprocessing_p1.py", "src/ids/collision_audit.py",
                "src/ids/p2_protocol.py", "scripts/audit_p2_protocol.py",
                "scripts/p2_float32_forensics.py")},
        },
        "schema_v2_hash": schema["sha256"],
        "historical_contracts_immutable": [
            "canonical-model-features-v1", "post-scaling-clipping-float32-v1",
            P1_CONTRACT,
        ],
        "configurations": {
            "A": "P1 61 scalar, float32",
            "B": "P1 61 scalar, float64 boundary",
            "C": "58 continuous float64 + 3 categorical opaque IDs",
            "D": "C + candidate whole ambiguous-group exclusion before splitting",
        },
        "all_cells": records, "group_split_analysis": split_analysis,
        "ambiguity_impact_artifact": "results/data_quality/p2/ambiguity_impact.json",
        "schema_v2_artifact": "results/data_quality/p2/schema_v2.json",
        "training_authorized": False,
        "scientific_protocol_approval": "AWAITING PROTOCOL APPROVAL",
    }
    output = destination / "p2_configuration_matrix.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(output)
    return 2 if any(x["data_integrity_gate"] == "FAIL" for x in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
