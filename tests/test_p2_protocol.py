"""Synthetic safeguards for P2 data-only candidate contracts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_step2_data_isolation import _official_frames  # noqa: E402
from ids.preprocessing_p1 import (  # noqa: E402
    LEGACY_MODEL_INPUT_IDENTITY_VERSION, MODEL_INPUT_IDENTITY_VERSION,
    matrix_fingerprints, normalized_category,
)
from ids.p2_protocol import (  # noqa: E402
    MODEL_INPUT_VERSION, SCHEMA_VERSION, _candidate_frames, ambiguous_canonical_set,
    audit_model_inputs, build_schema_v2, candidate_fingerprints,
    fit_opaque_vocabulary, group_split_roles, opaque_category_ids,
    oov_digest, role_canonical_intersections, row_split_comparator,
    structured_model_fingerprints,
)
from ids.target_isolation import (  # noqa: E402
    CANONICAL_FINGERPRINT_COLUMN as CANONICAL, CANONICAL_IDENTITY_VERSION,
    P0_FEATURE_SCHEMA,
    build_target_isolation_context,
)


def context_with_conflict():
    train, test = _official_frames()
    for frame in (train, test):
        for feature in P0_FEATURE_SCHEMA[:42]:
            if feature.endswith("_num") or feature in frame:
                continue
            frame[feature] = 0.0
    # The inserted train row and official test row are one full canonical group
    # with opposing semantic labels.
    duplicate = train.iloc[0].copy()
    duplicate["attack_cat"], duplicate["label"] = "Analysis", 1
    train = pd.concat([train, duplicate.to_frame().T], ignore_index=True)
    for column in ("label", *P0_FEATURE_SCHEMA[:42]):
        if column not in train:
            continue
        train[column] = pd.to_numeric(train[column])
    test = pd.concat([test, train.iloc[[0]].assign(attack_cat="Fuzzers", label=1)],
                     ignore_index=True)
    return build_target_isolation_context(train, test)


def test_immutable_versions_and_deterministic_schema():
    context = context_with_conflict()
    assert CANONICAL_IDENTITY_VERSION == "canonical-model-features-v1"
    assert LEGACY_MODEL_INPUT_IDENTITY_VERSION == "post-scaling-clipping-float32-v1"
    assert MODEL_INPUT_IDENTITY_VERSION == "post-robustscale-float32-sha256fallback-v2"
    first = build_schema_v2(context)
    assert first == build_schema_v2(context)
    assert first["contract"]["version"] == SCHEMA_VERSION
    assert MODEL_INPUT_VERSION != MODEL_INPUT_IDENTITY_VERSION
    assert len(first["contract"]["continuous_features"]) == 58
    assert set(first["contract"]["categorical_features"]) == {"proto", "service", "state"}
    assert not set(("proto_num", "service_num", "state_num")) & set(
        first["contract"]["continuous_features"]
    )


def test_float32_merges_intentionally_close_float64_vectors():
    close = np.array([[1.0], [1.0 + 2**-26]], dtype=np.float64)
    assert len(set(matrix_fingerprints(close))) == 2
    assert len(set(matrix_fingerprints(close.astype(np.float32)))) == 1
    ids = {feature: np.asarray(["known:0", "known:0"])
           for feature in ("proto", "service", "state")}
    assert len(set(structured_model_fingerprints(close, ids))) == 2


def test_feature_namespaced_deterministic_oov_without_test_vocabulary_growth():
    context = context_with_conflict()
    ambiguous = ambiguous_canonical_set(context)
    population, roles = group_split_roles(
        context, "Fuzzers", 42, ambiguous, exclude_ambiguous=True
    )
    vocabulary = fit_opaque_vocabulary(roles["train"])
    altered_test = population["target"].copy()
    altered_test.loc[:, "proto"] = "never-in-backbone"
    assert vocabulary == fit_opaque_vocabulary(roles["train"])
    token = normalized_category("never-in-backbone")
    ids = opaque_category_ids(altered_test, vocabulary)
    assert len(set(ids["proto"])) == 1
    assert ids["proto"][0] == "oov:" + oov_digest("proto", token)
    assert oov_digest("proto", token) != oov_digest("service", token)
    assert token not in vocabulary["proto"]
    assert not any("Fuzzers" in set(frame.attack_cat) for frame in roles.values())


def test_conflict_excludes_complete_group_before_purge_and_group_split():
    context = context_with_conflict()
    ambiguous = ambiguous_canonical_set(context)
    conflict = int(context["train"].iloc[0][CANONICAL])
    assert conflict in ambiguous
    before = _candidate_frames(context, "Fuzzers", set())
    assert conflict in set(before["target"][CANONICAL])
    after, roles = group_split_roles(
        context, "Fuzzers", 42, ambiguous, exclude_ambiguous=True
    )
    for frame in (*roles.values(), after["target"], after["test_known"]):
        assert conflict not in set(frame[CANONICAL])
    assert not any(role_canonical_intersections(roles).values())
    assert not set(after["target"][CANONICAL]) & set(after["known"][CANONICAL])
    assert all(set(frame["__source_role"]) == {"official_train"}
               for frame in roles.values())


def test_fold_local_fits_and_official_test_not_parameter_fit():
    context = context_with_conflict()
    population, roles = group_split_roles(
        context, "Fuzzers", 43, ambiguous_canonical_set(context),
        exclude_ambiguous=True,
    )
    schema = build_schema_v2(context)
    original = candidate_fingerprints(context, schema, roles, population)
    changed = dict(population)
    changed["target"] = population["target"].copy()
    changed["target"].loc[:, "proto"] = "test-only-token"
    changed["test_known"] = population["test_known"].copy()
    changed["test_known"].loc[:, "service"] = "test-only-service"
    mutated = candidate_fingerprints(context, schema, roles, changed)
    assert original[2] == mutated[2]
    np.testing.assert_array_equal(original[1]["center"], mutated[1]["center"])
    np.testing.assert_array_equal(original[1]["scale"], mutated[1]["scale"])
    np.testing.assert_array_equal(original[0]["train"], mutated[0]["train"])


def test_injected_leakage_is_fail_closed():
    context = context_with_conflict()
    population, roles = group_split_roles(
        context, "Fuzzers", 44, ambiguous_canonical_set(context),
        exclude_ambiguous=True,
    )
    schema = build_schema_v2(context)
    fingerprints, _, _, _ = candidate_fingerprints(context, schema, roles, population)
    leaked_roles = dict(roles)
    leaked_roles["calibration"] = pd.concat(
        [roles["calibration"], population["target"].iloc[[0]]], ignore_index=True
    )
    leaked_fps = dict(fingerprints)
    leaked_fps["calibration"] = np.concatenate([
        fingerprints["calibration"], fingerprints["target"][:1]
    ])
    audited = audit_model_inputs(leaked_roles, population, leaked_fps)
    assert audited["gate"]["gate"] == "FAIL"
    assert audited["canonical_target_intersections"]["calibration"] > 0


def test_row_split_is_measured_and_group_split_is_disjoint():
    context = context_with_conflict()
    ambiguous = ambiguous_canonical_set(context)
    population, roles = group_split_roles(
        context, "Fuzzers", 42, ambiguous, exclude_ambiguous=True
    )
    comparator = row_split_comparator(population["known"], population["surrogate"], 42)
    assert set(comparator["canonical_role_intersections"]) == set(
        role_canonical_intersections(roles)
    )
    assert not any(role_canonical_intersections(roles).values())
