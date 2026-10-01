from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p10_external_source import (
    CLASSIFICATIONS, OFFICIAL_SOURCES, TARGETS, build_artifacts, validate_artifacts,
)


STAMP = "2026-09-23T00:00:00Z"


def test_every_candidate_classifies_all_frozen_features_once():
    artifacts = build_artifacts(ROOT, STAMP)
    mappings = artifacts["feature_mapping.json"]
    assert mappings["required_continuous_count"] == 58
    assert mappings["required_categorical_count"] == 3
    assert set(mappings["candidate_mappings"]) == set(OFFICIAL_SOURCES)
    for candidate in mappings["candidate_mappings"].values():
        rows = candidate["mapping"]
        assert len(rows) == 61
        assert len({row["p9_feature"] for row in rows}) == 61
        assert all(row["classification"] in CLASSIFICATIONS for row in rows)
        assert all(row["rationale"] for row in rows)


def test_gate_fails_closed_without_structural_imputation():
    artifacts = build_artifacts(ROOT, STAMP)
    for candidate in artifacts["feature_mapping.json"]["candidate_mappings"].values():
        assert candidate["gate"] == "EXTERNAL_SCHEMA_INCOMPATIBLE"
        assert candidate["blocking_features"]
    reservation = artifacts["external_source_reservation.json"]
    assert reservation["reservation_status"] == "NOT_RESERVED"
    assert reservation["raw_files"] == []
    assert reservation["aggregate_dataset_sha256"] is None


def test_label_and_identity_contracts_cannot_activate_after_schema_failure():
    artifacts = build_artifacts(ROOT, STAMP)
    labels = artifacts["label_mapping.json"]
    for candidate in labels["candidate_target_mappings"].values():
        assert [row["p9_target"] for row in candidate["targets"]] == list(TARGETS)
        assert all(row["eligible_for_evaluation"] is False for row in candidate["targets"])
    identity = artifacts["identity_contract.json"]
    assert identity["frozen_for_execution"] is False
    assert "model score" in identity["forbidden_inputs"]
    assert "row index" in identity["forbidden_inputs"]


def test_no_model_or_metric_access_is_claimed():
    artifacts = build_artifacts(ROOT, STAMP)
    validate_artifacts(ROOT, artifacts)
    prereg = artifacts["p10_preregistration.json"]
    assert prereg["status"] == "P10_EXTERNAL_SOURCE_BLOCKED"
    assert prereg["p9_checkpoint_opened"] is False
    assert prereg["p9_model_evaluated"] is False
    assert prereg["anomaly_scores_computed"] is False
    assert prereg["external_metrics_computed"] is False
    assert prereg["readiness"] == "P10B_NOT_AUTHORIZED"
