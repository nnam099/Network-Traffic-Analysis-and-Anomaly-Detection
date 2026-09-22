"""Test-blind P3 isolation tests with small 61-feature synthetic partitions."""

from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_p2_protocol import context_with_conflict  # noqa: E402
from ids.dataset import (  # noqa: E402
    DEFAULT_TRAIN_FILE, load_unsw_csvs,
)
from ids.p2_protocol import build_schema_v2, role_canonical_intersections  # noqa: E402
from ids.p3_protocol import (  # noqa: E402
    CANONICAL, canonical_json, classify_official_test, construct_development_roles,
    development_manifest, development_model_audit, post_freeze_model_input_audit,
    prepare_official_test,
    sha256_json, test_support as summarize_test_support,
    train_ambiguity_report, train_only_context,
)


def prepared_case():
    prior = context_with_conflict()
    context = train_only_context(prior["train"])
    ambiguous, ambiguity = train_ambiguity_report(context)
    schema = build_schema_v2(context)
    population, roles = construct_development_roles(context, "Fuzzers", 42, ambiguous)
    return prior, context, ambiguous, ambiguity, schema, population, roles


def _manifest(context, ambiguity, schema, population, roles):
    audit, fitting, vocabulary = development_model_audit(
        context, schema, population, roles
    )
    manifest = development_manifest(
        context, schema, ambiguity, population, roles, audit, fitting,
        target="Fuzzers", seed=42, train_source_sha256="synthetic-train-sha",
        commit_sha=None, code_sha256={"p3": "synthetic-code-sha"},
    )
    return manifest, audit, fitting, vocabulary


def test_train_only_ambiguity_whole_group_and_target_purge():
    prior, context, ambiguous, ambiguity, _, population, roles = prepared_case()
    conflicted = int(context["train"].iloc[0][CANONICAL])
    assert conflicted in ambiguous
    assert ambiguity["official_test_consulted"] is False
    assert ambiguity["excluded_groups"] >= 1
    assert not any(conflicted in set(frame[CANONICAL]) for frame in (
        population["retained"], population["target_train"], *roles.values()
    ))
    assert sum(ambiguity["per_family"][name]["excluded_rows"] for name in
               ambiguity["per_family"]) == ambiguity["excluded_rows"]
    assert not any(role_canonical_intersections(roles).values())
    assert not any(set(frame[CANONICAL]) & population["target_identity_hashes"]
                   for frame in roles.values())
    assert len(prior["test"]) > 0  # the test exists but had no input path into P3 dev


def test_test_labels_or_identities_cannot_change_dev_manifest_or_target_purge():
    prior, context, _, ambiguity, schema, population, roles = prepared_case()
    manifest, audit, fitting, vocabulary = _manifest(
        context, ambiguity, schema, population, roles
    )
    baseline_hash = sha256_json(manifest)
    changed_test = prior["test"].copy()
    changed_test.loc[0, "attack_cat"] = "Backdoors"
    changed_test.loc[:, "proto"] = "never-seen-test-only-token"
    changed_test.iloc[0] = prior["train"].iloc[1]
    changed_test.loc[:, "__source_role"] = "official_test"
    # There is no test parameter in any development-construction callable.
    second = train_only_context(prior["train"])
    second_ambiguous, second_ambiguity = train_ambiguity_report(second)
    second_population, second_roles = construct_development_roles(
        second, "Fuzzers", 42, second_ambiguous
    )
    second_manifest, _, second_fitting, second_vocabulary = _manifest(
        second, second_ambiguity, build_schema_v2(second),
        second_population, second_roles,
    )
    assert baseline_hash == sha256_json(second_manifest)
    assert population["target_identity_hashes"] == second_population[
        "target_identity_hashes"
    ]
    assert fitting["scaler_center_scale_sha256"] == second_fitting[
        "scaler_center_scale_sha256"
    ]
    assert vocabulary == second_vocabulary
    external = prepare_official_test(changed_test, context)
    _, external_report = classify_official_test(external, context, set(
        int(x) for x in ambiguity["excluded_identity_hashes"]
    ))
    assert sum(external_report["category_rows"].values()) == len(changed_test)
    assert audit["gate"]["gate"] == "PASS"
    assert sha256_json(manifest) == baseline_hash
    assert len(changed_test) == len(prior["test"])


def test_missing_official_test_file_preserves_development_split(tmp_path):
    prior = context_with_conflict()
    path = tmp_path / DEFAULT_TRAIN_FILE
    test_path = tmp_path / "UNSW_NB15_testing-set.csv"
    prior["train"].drop(columns=[column for column in prior["train"]
                                 if column.startswith("__")]).to_csv(path, index=False)
    prior["test"].drop(columns=[column for column in prior["test"]
                                if column.startswith("__")]).to_csv(test_path, index=False)
    first = train_only_context(load_unsw_csvs(
        tmp_path, files=[path], source_role="official_train"
    ))
    ambiguous, _ = train_ambiguity_report(first)
    before, roles_before = construct_development_roles(first, "Fuzzers", 43, ambiguous)
    test_path.unlink()
    assert not test_path.exists()
    second = train_only_context(load_unsw_csvs(
        tmp_path, files=[path], source_role="official_train"
    ))
    ambiguous_second, _ = train_ambiguity_report(second)
    after, roles_after = construct_development_roles(
        second, "Fuzzers", 43, ambiguous_second
    )
    assert before["target_identity_hashes"] == after["target_identity_hashes"]
    for name in roles_before:
        np.testing.assert_array_equal(
            roles_before[name]["__official_train_row_id"],
            roles_after[name]["__official_train_row_id"],
        )


def test_manifest_is_deterministic_and_contains_only_train_role_references():
    _, context, _, ambiguity, schema, population, roles = prepared_case()
    manifest, audit, fitting, vocabulary = _manifest(
        context, ambiguity, schema, population, roles
    )
    assert sha256_json(manifest) == hashlib.sha256(canonical_json(manifest)).hexdigest()
    assert manifest["official_test_consulted"] is False
    assert manifest["feature_schema_sha256"] == schema["sha256"]
    assert manifest["vocabulary_provenance"]["vocabulary_sha256"] == fitting[
        "vocabulary_sha256"
    ]
    assert set(vocabulary) == {"proto", "service", "state"}
    assert all(int(identity) not in population["target_identity_hashes"]
               for role in manifest["roles"].values()
               for identity in role["canonical_identity_hashes"])
    assert all(set(frame["__source_role"]) == {"official_train"}
               for frame in roles.values())
    assert audit["gate"]["gate"] == "PASS"


def test_external_classification_is_exclusive_and_never_mutates_frozen_manifest():
    prior, context, ambiguous, ambiguity, schema, population, roles = prepared_case()
    manifest, _, _, _ = _manifest(context, ambiguity, schema, population, roles)
    frozen_hash = sha256_json(manifest)
    test = prior["test"].copy()
    overlap = prior["train"].iloc[[1]].copy()
    overlap["__source_role"] = "official_test"
    overlap["attack_cat"] = "Normal"
    overlap["label"] = 0
    internal = test.iloc[[0]].copy()
    internal["attack_cat"] = "Generic"
    internal["label"] = 1
    test = pd.concat([test, overlap, internal], ignore_index=True)
    external = prepare_official_test(test, context)
    classified, report = classify_official_test(external, context, ambiguous)
    assert len(classified) == sum(report["category_rows"].values())
    assert set(classified.test_category) <= set(report["category_rows"])
    assert report["category_rows"]["TRAIN-CONFLICT-RELATED"] > 0
    assert report["category_rows"]["TEST-INTERNAL-AMBIGUOUS"] > 0
    assert report["category_rows"]["DEVELOPMENT-OVERLAP"] > 0
    assert report["category_rows"]["CLEAN"] > 0
    external_input = post_freeze_model_input_audit(classified, context, schema, manifest)
    assert external_input["official_test_used_for_parameter_fit"] is False
    assert external_input["clean_test_rows"] == report["category_rows"]["CLEAN"]
    assert sha256_json(manifest) == frozen_hash
    assert summarize_test_support(classified, "Fuzzers")["target"]["original_rows"] > 0


def test_injected_target_leakage_still_fails():
    _, context, _, _, schema, population, roles = prepared_case()
    leaked = copy.copy(roles)
    leaked["calibration"] = pd.concat([
        roles["calibration"], population["target_train"].iloc[[0]]
    ], ignore_index=True)
    audited, _, _ = development_model_audit(context, schema, population, leaked)
    assert audited["gate"]["gate"] == "FAIL"
    assert audited["canonical_target_intersections"]["calibration"] > 0
