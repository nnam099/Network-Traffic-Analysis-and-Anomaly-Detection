"""P7 Stage-A synthetic tests: no official-test CSV or scores are opened."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p7_external_protocol import (  # noqa: E402
    BOOTSTRAP_REPLICATES, SMALL_N, assert_detector_binding,
    across_seed_summary, binary_metrics, bootstrap_intervals, bootstrap_seed,
    metric_contract, p7_path, primary_metrics, result_schema, surface_contract,
    uncertainty_contract, upstream_snapshot,
)
from ids.p6_selection import validate_p6_cell  # noqa: E402


def synthetic_frame():
    return pd.DataFrame({
        "surface": ["CLEAN"] * 5,
        "canonical_identity": ["known-a", "known-a", "known-b", "target-a", "target-b"],
        "family": ["Normal", "Normal", "DoS", "Fuzzers", "Fuzzers"],
        "score": np.array([0.0, 0.0, 0.1, 0.9, 1.0], dtype=np.float64),
    })


def test_metric_contract_orientation_ap_and_confusion_strict_threshold():
    assert metric_contract()["main_threshold_independent_metric"] == "identity AUROC"
    good = binary_metrics(np.array([0.0, 0.1], dtype=np.float64),
                          np.array([0.9, 1.0], dtype=np.float64), 0.9)
    assert good["AUROC"] == 1.0 and good["Average_Precision"] == 1.0
    assert (good["TP"], good["FP"], good["TN"], good["FN"]) == (1, 0, 2, 1)
    assert good["TPR"] == 0.5 and good["FPR"] == 0.0
    assert good["specificity"] == 1.0 and good["balanced_accuracy"] == 0.75
    assert good["precision"] == 1.0 and good["F1"] == 2 / 3
    reversed_scores = binary_metrics(np.array([0.9, 1.0], dtype=np.float64),
                                     np.array([0.0, 0.1], dtype=np.float64), 0.9)
    assert reversed_scores["AUROC"] == 0.0
    assert reversed_scores["Average_Precision"] < good["Average_Precision"]
    with pytest.raises(ValueError):
        binary_metrics(np.array([np.nan], dtype=np.float64),
                       np.array([1.0], dtype=np.float64), 0.5)


def test_identity_aggregation_duplicate_invariance_and_row_diagnostic_separation():
    first = primary_metrics(synthetic_frame(), "Fuzzers", 0.5)
    duplicated = pd.concat([synthetic_frame(), synthetic_frame().iloc[[0]]], ignore_index=True)
    second = primary_metrics(duplicated, "Fuzzers", 0.5)
    assert first["counts"]["known_identities"] == 2
    assert second["counts"]["known_identities"] == 2
    assert first["identity_metrics"] == second["identity_metrics"]
    assert second["counts"]["known_rows"] == first["counts"]["known_rows"] + 1
    assert first["row_diagnostics"]["label"] == "ROW-LEVEL DIAGNOSTIC"
    reordered = primary_metrics(synthetic_frame().iloc[::-1].reset_index(drop=True), "Fuzzers", 0.5)
    assert first["identity_metrics"] == reordered["identity_metrics"]


@pytest.mark.parametrize("surface", ["DEVELOPMENT-OVERLAP", "TEST-INTERNAL-AMBIGUOUS",
                                      "TRAIN-CONFLICT-RELATED"])
def test_primary_rejects_every_diagnostic_surface(surface):
    frame = synthetic_frame()
    frame.loc[0, "surface"] = surface
    with pytest.raises(PermissionError):
        primary_metrics(frame, "Fuzzers", 0.5)


def test_primary_rejects_conflicting_identity_and_other_ood_family():
    frame = synthetic_frame()
    frame.loc[3, "canonical_identity"] = "known-a"
    with pytest.raises(ValueError, match="conflicting"):
        primary_metrics(frame, "Fuzzers", 0.5)
    frame = synthetic_frame()
    frame.loc[3, "family"] = "Worms"
    with pytest.raises(PermissionError, match="other OOD"):
        primary_metrics(frame, "Fuzzers", 0.5)


def test_bootstrap_reproducibility_seed_separation_and_invalid_handling(monkeypatch):
    known = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    target = np.array([0.6, 0.9], dtype=np.float64)
    seed = bootstrap_seed("Fuzzers", 42)
    assert seed != bootstrap_seed("Fuzzers", 43)
    assert seed != bootstrap_seed("Analysis", 42)
    first = bootstrap_intervals(known, target, 0.5, seed=seed,
                                replicates=40, min_valid=40)
    second = bootstrap_intervals(known, target, 0.5, seed=seed,
                                 replicates=40, min_valid=40)
    assert first == second
    assert first["discarded_replicates"] == 0
    assert uncertainty_contract()["replicates"] == BOOTSTRAP_REPLICATES == 2000
    with pytest.raises(ValueError):
        bootstrap_intervals(np.empty(0, dtype=np.float64), target, 0.5,
                            seed=seed, replicates=40, min_valid=40)
    import ids.p7_external_protocol as p7
    def invalid(*_args, **_kwargs):
        raise ValueError("mathematically undefined")
    monkeypatch.setattr(p7, "binary_metrics", invalid)
    with pytest.raises(ValueError, match="too few valid"):
        bootstrap_intervals(known, target, 0.5, seed=seed,
                            replicates=4, min_valid=3)


def test_seed_summary_requires_exact_seeds_and_no_pooling():
    cells = [{"target": "Fuzzers", "split_seed": seed,
              "identity_metrics": {name: 0.5 for name in (
                  "AUROC", "Average_Precision", "TPR", "FPR", "specificity",
                  "precision", "F1", "balanced_accuracy")}}
             for seed in (42, 43, 44)]
    summary = across_seed_summary(cells, "Fuzzers")
    assert summary["seed_count"] == 3
    assert summary["metrics"]["AUROC"]["sample_sd"] == 0.0
    with pytest.raises(ValueError, match="exactly once"):
        across_seed_summary(cells[:2], "Fuzzers")


def test_frozen_reporting_contract_and_output_namespace():
    surfaces = surface_contract()
    assert surfaces["primary"] == "CLEAN"
    assert set(SMALL_N) == {"Analysis", "Backdoors", "Worms"}
    assert surfaces["diagnostics_allowed_only_after_primary_hash_freeze"] is True
    assert result_schema()["provenance_required"]
    with pytest.raises(PermissionError):
        p7_path(ROOT, "../../training_protocol/p5/p5_matrix.json")


def test_upstream_snapshot_does_not_read_test_artifacts(monkeypatch):
    original = Path.open
    def checked_open(path, *args, **kwargs):
        name = str(path)
        if any(value in name for value in (
            "UNSW_NB15_testing-set.csv", "historical_738_investigation.json",
            "p3_matrix.json")):
            raise AssertionError(f"Stage A attempted test-artifact access: {name}")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", checked_open)
    snapshot = upstream_snapshot(ROOT)
    assert snapshot["p6_status"] == "P6_DEVELOPMENT_PASS"
    assert len(snapshot["groups"]["p3_development"]) == 18
    assert len(snapshot["groups"]["p3_prior_external_audit_file_sha256"]) == 64


def test_changed_checkpoint_scorer_threshold_or_calibration_identity_is_rejected():
    target, seed = "Fuzzers", 42
    prefix = ROOT / "results/model_selection/p6/runs/fuzzers/seed_42"
    def read(name):
        return json.loads((prefix / name).read_text())
    dry = validate_p6_cell(ROOT, target, seed)
    run, ref, states, selection, calibration = (
        read(name) for name in ("training_manifest.json", "checkpoint_reference.json",
                               "scorer_states.json", "scorer_selection.json", "calibration.json"))
    with gzip.open(ROOT / "results/data_quality/p3/manifests/fuzzers_seed42.json.gz", "rt") as file:
        manifest = json.load(file)
    role = {"row_ids": manifest["roles"]["calibration"]["row_ids"],
            "identities": manifest["roles"]["calibration"]["canonical_identity_hashes"]}
    matrix = json.loads((ROOT / "results/model_selection/p6/p6_matrix.json").read_text())
    cell = next(item for item in matrix["cells"] if item["target"] == target and item["split_seed"] == seed)
    file_hash = ref["checkpoint_file_sha256"]
    state_hash = ref["checkpoint_state_sha256"]
    assert_detector_binding(dry, run, ref, states, selection, calibration,
                            role, cell, actual_file_hash=file_hash,
                            actual_state_hash=state_hash)
    with pytest.raises(ValueError, match="checkpoint"):
        assert_detector_binding(dry, run, ref, states, selection, calibration,
                                role, cell, actual_file_hash="altered",
                                actual_state_hash=state_hash)
    altered = copy.deepcopy(states)
    altered["states"]["nearest_centroid_l2"]["centroids_float64"][0][0] += 1
    with pytest.raises(ValueError, match="centroid scorer"):
        assert_detector_binding(dry, run, ref, altered, selection, calibration,
                                role, cell, actual_file_hash=file_hash,
                                actual_state_hash=state_hash)
    altered = copy.deepcopy(calibration)
    altered["threshold"] += 1
    with pytest.raises(ValueError, match="matrix"):
        assert_detector_binding(dry, run, ref, states, selection, altered,
                                role, cell, actual_file_hash=file_hash,
                                actual_state_hash=state_hash)
    altered_role = copy.deepcopy(role)
    altered_role["identities"][0] = "tampered-identity"
    with pytest.raises(ValueError, match="provenance"):
        assert_detector_binding(dry, run, ref, states, selection, calibration,
                                altered_role, cell, actual_file_hash=file_hash,
                                actual_state_hash=state_hash)


@pytest.mark.parametrize("flags,missing", [
    ([], "--protocol-hash"),
    (["--protocol-hash", "0" * 64], "--enable-official-test"),
    (["--enable-official-test"], "--protocol-hash"),
])
def test_external_cli_without_two_keys_fails_before_any_test_access(flags, missing):
    intent = ROOT / "results/external_evaluation/p7/evaluation_intent.json"
    before = intent.exists()
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_p7_external_evaluation.py"),
                             *flags],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert missing in result.stderr
    assert intent.exists() == before
