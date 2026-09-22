from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from test_step2_data_isolation import _official_frames
from ids.collision_audit import (
    ambiguity_exclusion_mask, ambiguity_policy_report, collision_gate,
    collision_taxonomy, stage_audit,
)
from ids.dataset import KNOWN_ATTACK_CATS, model_input_fingerprints
from ids.preprocessing_p1 import (
    MODEL_INPUT_IDENTITY_VERSION, encode_categories, fit_categorical_maps,
    REQUIRED_UNICODE_DATA_VERSION, matrix_fingerprints, normalized_category,
    transform_stages, unknown_category_code,
)
from ids.target_isolation import (
    CANONICAL_IDENTITY_VERSION, FIT_ROLE_NAMES, P0_FEATURE_SCHEMA, REQUIRED_INVARIANTS,
    assert_target_fold_retraining_gate, build_target_isolation_context,
    prepare_target_isolated_fold,
)


def test_unknown_encoding_golden_and_independent_processes():
    import unicodedata

    assert unicodedata.unidata_version == REQUIRED_UNICODE_DATA_VERSION
    expected = -1.6068730354309082
    assert unknown_category_code("proto", "value:new-protocol") == expected
    code = (
        "from ids.preprocessing_p1 import unknown_category_code; "
        "print(unknown_category_code('proto', 'value:new-protocol'))"
    )
    for seed in ("1", "98271"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1",
               "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
        result = subprocess.check_output([sys.executable, "-c", code], env=env, text=True)
        assert float(result) == expected


def test_unknown_codes_distinguish_tokens_namespace_and_known_categories():
    maps = fit_categorical_maps(pd.DataFrame({"proto": ["tcp", "udp"]}))
    frame = pd.DataFrame({"proto": ["tcp", "udp", "new-a", "new-b", " ＴＣＰ "]})
    encode_categories(frame, json.loads(json.dumps(maps)))
    codes = frame["proto_num"].to_numpy()
    assert codes[0] == codes[4]
    assert len(set(codes[:4])) == 4
    assert (codes[:2] >= 0).all() and (codes[2:4] < -1).all()
    np.testing.assert_array_equal(codes, codes.astype(np.float32))
    assert unknown_category_code("proto", "value:new-a") != unknown_category_code("service", "value:new-a")
    assert normalized_category(None) != normalized_category("missing:")
    scaler = RobustScaler().fit(codes[:2].reshape(-1, 1))
    _, final = transform_stages(codes[:4].reshape(-1, 1), scaler)
    assert len(set(matrix_fingerprints(final))) == 4
    with pytest.raises(ValueError, match="missing backbone vocabulary"):
        encode_categories(frame, {})


def test_no_target_or_test_contribution_to_vocabulary_and_scaler():
    train, test = _official_frames()
    baseline_context = build_target_isolation_context(train, test)
    baseline = prepare_target_isolated_fold(
        baseline_context, "Fuzzers", 42, include_post_transform_label_diagnostic=False,
    )
    changed_train, changed_test = train.copy(), test.copy()
    mask = changed_train.attack_cat == "Fuzzers"
    changed_train.loc[mask, "proto"] = "target-train-only"
    changed_train.loc[mask, "dur"] = 1e20
    changed_test["proto"] = "official-test-only"
    changed_test["dur"] = 1e25
    changed = prepare_target_isolated_fold(
        build_target_isolation_context(changed_train, changed_test), "Fuzzers", 42,
        include_post_transform_label_diagnostic=False,
    )
    assert changed["categorical_maps"] == baseline["categorical_maps"]
    for name in ("center_", "scale_"):
        np.testing.assert_array_equal(getattr(changed["scaler"], name), getattr(baseline["scaler"], name))
    np.testing.assert_array_equal(changed["scaler_fit_row_ids"], baseline["scaler_fit_row_ids"])
    for mapping in changed["categorical_maps"].values():
        assert "value:target-train-only" not in mapping
        assert "value:official-test-only" not in mapping
    fitted_rows = baseline_context["train"].iloc[baseline["scaler_fit_row_ids"]]
    assert set(fitted_rows.attack_cat) <= set(KNOWN_ATTACK_CATS)
    assert baseline["categorical_maps"] == fit_categorical_maps(fitted_rows)
    assert changed["report"]["model_input_identity_version"] == MODEL_INPUT_IDENTITY_VERSION


def test_removed_clipping_preserves_extreme_numeric_vectors():
    scaler = RobustScaler().fit(np.array([[0.], [1.], [2.]], dtype=np.float64))
    scaled, final = transform_stages(np.array([[100.], [1000.]]), scaler)
    assert scaled.dtype == np.float64 and final.dtype == np.float32
    assert (final > 10).all()
    assert len(set(matrix_fingerprints(final))) == 2
    assert len(set(matrix_fingerprints(np.clip(final, -10, 10)))) == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1e100])
def test_nonfinite_and_float32_overflow_fail_without_saturation(bad):
    scaler = RobustScaler().fit(np.array([[0.], [1.], [2.]]))
    with pytest.raises(ValueError):
        transform_stages(np.array([[bad]]), scaler)


def test_stage_audit_detects_float32_rounding_at_the_correct_transition():
    encoded = np.array([[1.], [1. + 2**-25]], dtype=np.float64)
    stages = {
        "A": {"train": np.array([101]), "val": np.array([102])},
        "B": {"train": matrix_fingerprints(encoded[:1]), "val": matrix_fingerprints(encoded[1:])},
        "C": {"train": matrix_fingerprints(encoded[:1]), "val": matrix_fingerprints(encoded[1:])},
        "D": {"train": matrix_fingerprints(encoded[:1].astype(np.float32)),
              "val": matrix_fingerprints(encoded[1:].astype(np.float32))},
    }
    audit = stage_audit(stages, {"train": ["Normal"], "val": ["Normal"]})
    assert audit["B"]["newly_merged_fingerprints_from_previous_stage"] == 0
    assert audit["C"]["newly_merged_fingerprints_from_previous_stage"] == 0
    assert audit["D"]["newly_merged_fingerprints_from_previous_stage"] == 1
    gate = collision_gate(audit["D"]["taxonomy"], {})
    assert gate["severity"] == "FAIL"
    assert gate["fail_reasons"][0]["reason"] == "cross_role_same_label"


def test_taxonomy_and_critical_severity_are_not_just_total_counts():
    fps = {"train": [1, 1, 2, 2], "val": [1, 2], "target": [2],
           "meta_known": [3], "calibration": [2], "surrogate_ood": [2]}
    labels = {"train": ["Normal", "Normal", "DoS", "Exploits"],
              "val": ["Normal", "Exploits"], "target": ["Backdoors"],
              "meta_known": ["Normal"], "calibration": ["Normal"], "surrogate_ood": ["Analysis"]}
    result = collision_taxonomy(fps, labels)
    assert result["same_role"]["train"]["same_label_fingerprints"] == 1
    assert result["same_role"]["train"]["same_label_duplicate_rows_beyond_first"] == 1
    assert result["same_role"]["train"]["cross_label_fingerprints"] == 1
    assert result["cross_role_pairs"]["train__val"]["same_label_fingerprints"] == 2
    assert result["cross_role_pairs"]["train__val"]["cross_label_fingerprints"] == 1
    assert result["target_vs_roles"] == {"train": 1, "val": 1, "meta_known": 0,
                                          "calibration": 1, "surrogate_ood": 1}
    gate = collision_gate(result, {"val": 1})
    assert gate["severity"] == "CRITICAL FAIL"
    assert {r["reason"] for r in gate["fail_reasons"]} == {
        "target_vs_fit", "canonical_target_contamination", "cross_role_same_label",
        "cross_role_cross_label", "same_role_cross_label",
    }


def test_same_role_same_label_duplicates_are_report_only():
    tax = collision_taxonomy({"train": [1, 1, 1], "val": [2]},
                             {"train": ["Normal"] * 3, "val": ["Normal"]})
    assert tax["same_role"]["train"]["same_label_duplicate_rows_beyond_first"] == 2
    assert collision_gate(tax, {})["gate"] == "PASS"


def test_same_role_cross_label_collision_fails_without_selecting_a_label():
    tax = collision_taxonomy({"train": [1, 1], "val": [2]},
                             {"train": ["DoS", "Exploits"], "val": ["Normal"]})
    result = collision_gate(tax, {})
    assert result["severity"] == "CRITICAL FAIL"
    assert result["fail_reasons"] == [{
        "severity": "CRITICAL FAIL", "reason": "same_role_cross_label",
        "role": "train", "fingerprints": 1,
    }]


def _valid_gate_fold():
    rng = np.random.default_rng(901)
    labels = np.array(["Normal"] * 20 + [label for label in KNOWN_ATTACK_CATS
                                         if label != "Normal" for _ in range(2)])
    fold = {
        "feat_cols": list(P0_FEATURE_SCHEMA), "X_target_identity": rng.normal(size=(1, 61)).astype(np.float32),
        "target_canonical_fingerprints": np.array([9999], dtype=np.uint64), "fit_role_evidence": {},
        "report": {"target_family": "Backdoors", "seed": 42,
                   "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
                   "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
                   "invariants": dict.fromkeys(REQUIRED_INVARIANTS, True)},
    }
    for i, role in enumerate(FIT_ROLE_NAMES):
        values = rng.normal(size=(len(labels), 61)).astype(np.float32)
        ids = np.arange(len(labels)) + i * 100
        fold["X_" + ("ood_train" if role == "surrogate_ood" else role)] = values
        fold["fit_role_evidence"][role] = {
            "row_ids": ids, "source_roles": np.array(["official_train"] * len(ids)),
            "labels": labels if role != "surrogate_ood" else np.array(["Analysis"] * len(ids)),
            "canonical_fingerprints": ids.astype(np.uint64),
            "model_input_fingerprints": model_input_fingerprints(values).to_numpy(),
        }
    fold["scaler_fit_row_ids"] = fold["fit_role_evidence"]["train"]["row_ids"]
    fold["scaler_fit_canonical_fingerprints"] = fold["fit_role_evidence"]["train"]["canonical_fingerprints"]
    return fold


@pytest.mark.parametrize("kind", ["target", "cross_role"])
def test_live_gate_rejects_injected_collisions_even_with_pass_flags(kind):
    fold = _valid_gate_fold()
    assert_target_fold_retraining_gate(fold)
    if kind == "target":
        fold["X_val"][0] = fold["X_target_identity"][0]
        expected = "target_model_collision/val"
    else:
        fold["X_val"][0] = fold["X_train"][0]
        expected = "all_fit_roles_model_collision"
    with pytest.raises(AssertionError, match=expected):
        assert_target_fold_retraining_gate(fold)


def test_live_gate_rejects_same_role_cross_label_with_current_vectors():
    fold = _valid_gate_fold()
    fold["X_train"][20] = fold["X_train"][0]
    fold["fit_role_evidence"]["train"]["model_input_fingerprints"] = (
        model_input_fingerprints(fold["X_train"]).to_numpy()
    )
    with pytest.raises(AssertionError, match="scientific_collision/same_role_cross_label/train"):
        assert_target_fold_retraining_gate(fold)


def test_ambiguity_exclusion_is_whole_group_and_inactive_by_default():
    context = {"train": pd.DataFrame({"attack_cat": ["Normal", "Fuzzers", "DoS"],
                 "__source_role": ["official_train"] * 3,
                 "__canonical_feature_fingerprint": np.array([1, 1, 2], dtype=np.uint64)}),
               "test": pd.DataFrame({"attack_cat": ["Normal", "Analysis"],
                 "__source_role": ["official_test"] * 2,
                 "__canonical_feature_fingerprint": np.array([1, 3], dtype=np.uint64)})}
    original = copy.deepcopy(context)
    report = ambiguity_policy_report(context)
    assert report["active"] is False
    assert report["rows_lost"] == 3 and report["ambiguous_fingerprints"] == 1
    assert report["per_family_rows_lost"] == {"Fuzzers": 1, "Normal": 2}
    for role, expected in (("train", [False, False, True]), ("test", [False, True])):
        assert ambiguity_exclusion_mask(context[role], [1]).all()
        assert ambiguity_exclusion_mask(context[role], [1], activate=True).tolist() == expected
        pd.testing.assert_frame_equal(context[role], original[role])


def test_schema_gate_requires_exactly_the_p0_61_features():
    fold = _valid_gate_fold()
    assert len(fold["feat_cols"]) == 61
    assert_target_fold_retraining_gate(fold)
    fold["feat_cols"] = fold["feat_cols"][:-1]
    with pytest.raises(AssertionError, match="p0_schema_mismatch"):
        assert_target_fold_retraining_gate(fold)


def test_live_gate_rejects_stale_identity_contract_even_with_pass_flags():
    fold = _valid_gate_fold()
    fold["report"]["model_input_identity_version"] = "post-scaling-clipping-float32-v1"
    with pytest.raises(AssertionError, match="model_input_contract_mismatch"):
        assert_target_fold_retraining_gate(fold)
