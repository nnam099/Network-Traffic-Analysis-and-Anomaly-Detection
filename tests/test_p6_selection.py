"""Synthetic and frozen-artifact tests for P6; no official-test read or training."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p6_selection import (  # noqa: E402
    SCORERS, TIE_TOLERANCE, alert_mask, calibration_artifact, centroid_distance,
    checked_train_source, choose_scorer, development_auroc, fit_centroids,
    fit_known_threshold, identity_means, output_path, require_p6_role,
    softmax_state_hash, softmax_uncertainty, validate_p6_cell,
    verify_calibration_artifact, verify_scorer_states,
)
from ids.p5_training import _state_digest, file_hash  # noqa: E402


def frozen_case(target="Fuzzers", seed=42):
    with gzip.open(ROOT / f"results/data_quality/p3/manifests/{target.lower()}_seed{seed}.json.gz", "rt") as file:
        return json.load(file)


def test_role_safety_and_official_test_path_rejection():
    for role in ("val", "meta_known", "surrogate_ood", "calibration", "target_development_held_out", "official_test"):
        with pytest.raises(PermissionError):
            require_p6_role("centroid_fit", role)
    for role in ("train", "val", "calibration", "target_development_held_out", "official_test"):
        with pytest.raises(PermissionError):
            require_p6_role("scorer_selection", role)
    for role in ("train", "val", "meta_known", "surrogate_ood", "target_development_held_out", "official_test"):
        with pytest.raises(PermissionError):
            require_p6_role("threshold_fit", role)
    with pytest.raises(PermissionError):
        checked_train_source(ROOT, frozen_case(), ROOT / "data/UNSW_NB15_testing-set.csv")
    with pytest.raises(PermissionError):
        output_path(ROOT, "../../official_test/results.json")
    sources = (ROOT / "src/ids/p6_selection.py").read_text()
    assert "DEFAULT_TEST_FILE" not in sources
    assert "official_test_external_audit.json" not in sources


def test_softmax_orientation_dtype_and_logical_state_hash():
    known = np.array([[8.0, 0, 0, 0, 0]], dtype=np.float64)
    uncertain = np.zeros((1, 5), dtype=np.float64)
    assert softmax_uncertainty(known)[0] < softmax_uncertainty(uncertain)[0]
    assert softmax_uncertainty(known).dtype == np.float64
    with pytest.raises(TypeError):
        softmax_uncertainty(known.astype(np.float32))
    assert softmax_state_hash("a") == softmax_state_hash("a")
    assert softmax_state_hash("a") != softmax_state_hash("b")


def test_centroids_train_only_order_float64_hash_and_orientation():
    labels = ["Generic", "Reconnaissance", "Exploits", "DoS", "Normal"]
    rows = np.stack([np.full(32, value, dtype=np.float64) for value in (4, 3, 2, 1, 0)])
    centers, digest = fit_centroids(rows, labels, role="train", checkpoint_state_hash="checkpoint")
    assert centers.shape == (5, 32)
    assert centers.dtype == np.float64
    assert np.all(centers[:, 0] == np.array([0, 1, 2, 3, 4]))
    again, repeated_digest = fit_centroids(rows, labels, role="train", checkpoint_state_hash="checkpoint")
    np.testing.assert_array_equal(centers, again)
    assert digest == repeated_digest
    assert digest != fit_centroids(rows, labels, role="train", checkpoint_state_hash="changed")[1]
    known = np.zeros((1, 32), dtype=np.float64)
    distant = np.full((1, 32), 100.0, dtype=np.float64)
    assert centroid_distance(known, centers)[0] < centroid_distance(distant, centers)[0]
    with pytest.raises(PermissionError):
        fit_centroids(rows, labels, role="val", checkpoint_state_hash="checkpoint")
    with pytest.raises(TypeError):
        fit_centroids(rows[:, :31], labels, role="train", checkpoint_state_hash="checkpoint")
    states = {"checkpoint_state_sha256": "checkpoint", "states": {
        "negative_max_softmax": {"scorer_state_sha256": softmax_state_hash("checkpoint")},
        "nearest_centroid_l2": {"scorer_state_sha256": digest,
                                "centroids_float64": centers.tolist()}}}
    np.testing.assert_array_equal(verify_scorer_states(states, "checkpoint"), centers)
    with pytest.raises(ValueError, match="checkpoint"):
        verify_scorer_states(states, "altered")
    altered = copy.deepcopy(states)
    altered["states"]["nearest_centroid_l2"]["centroids_float64"][0][0] += 1.0
    with pytest.raises(ValueError, match="centroid scorer state"):
        verify_scorer_states(altered, "checkpoint")


def test_identity_mean_order_and_duplicate_invariance():
    scores = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    identities = ["b", "a", "b", "a"]
    keys, means = identity_means(scores, identities)
    assert keys == ["a", "b"]
    np.testing.assert_array_equal(means, [5.0, 3.0])
    order = [3, 2, 1, 0]
    keys2, means2 = identity_means(scores[order], [identities[i] for i in order])
    assert keys2 == keys
    np.testing.assert_array_equal(means2, means)
    keys3, means3 = identity_means(np.array([1, 3, 5, 7, 1, 5], dtype=np.float64),
                                  identities + ["b", "b"])
    assert len(keys3) == len(keys)
    assert means3[0] == means[0]


def test_canonical_identity_auroc_and_tie_rule():
    assert development_auroc(np.array([0.0, 0.1], dtype=np.float64),
                             np.array([0.9, 1.0], dtype=np.float64)) == 1.0
    assert development_auroc(np.array([0.9, 1.0], dtype=np.float64),
                             np.array([0.0, 0.1], dtype=np.float64)) == 0.0
    with pytest.raises(ValueError):
        development_auroc(np.array([np.nan], dtype=np.float64), np.array([1.0], dtype=np.float64))
    assert choose_scorer(0.7, 0.8)["selected_scorer"] == SCORERS[1]
    assert choose_scorer(0.8, 0.7)["selected_scorer"] == SCORERS[0]
    assert choose_scorer(0.8, 0.8)["tie_break_used"] is True
    assert choose_scorer(0.8, 0.8)["selected_scorer"] == SCORERS[1]
    assert choose_scorer(0.8, 0.8 + TIE_TOLERANCE / 2)["selected_scorer"] == SCORERS[1]


def test_known_only_quantile_strict_comparison_and_provenance():
    scores = np.arange(100, dtype=np.float64)
    identities = [str(i) for i in range(100)]
    fitted = fit_known_threshold(scores, identities, role="calibration")
    assert fitted["quantile"] == 0.99
    assert fitted["quantile_method"] == "higher"
    assert fitted["threshold"] == 99.0
    assert not alert_mask(np.array([99.0], dtype=np.float64), fitted["threshold"])[0]
    assert alert_mask(np.array([100.0], dtype=np.float64), fitted["threshold"])[0]
    assert fit_known_threshold(scores, identities, role="calibration") == fitted
    with pytest.raises(PermissionError):
        fit_known_threshold(scores, identities, role="surrogate_ood")
    dry = validate_p6_cell(ROOT, "Fuzzers", 42)
    role = {"role": "calibration", "row_ids": list(range(100)), "identities": identities}
    artifact = calibration_artifact(ROOT, dry, role, "negative_max_softmax", "scorer",
                                    "checkpoint", scores)
    verify_calibration_artifact(artifact, dry, role, "checkpoint", "scorer")
    for field in ("checkpoint_state_sha256", "scorer_state_sha256",
                  "calibration_row_ids_sha256", "calibration_identity_sha256"):
        altered = copy.deepcopy(artifact)
        altered[field] = "altered"
        with pytest.raises(ValueError, match="provenance"):
            verify_calibration_artifact(altered, dry, role, "checkpoint", "scorer")


def test_all_fifteen_dry_cells_and_pilot_artifacts_if_present():
    for target in ("Fuzzers", "Analysis", "Backdoors", "Shellcode", "Worms"):
        for seed in (42, 43, 44):
            assert validate_p6_cell(ROOT, target, seed)["status"] == "DRY_RUN_PASS"
    for tag in ("smoke-a", "smoke-b"):
        prefix = ROOT / f"results/model_selection/p6/runs/fuzzers/seed_42/pilot_{tag}"
        if not prefix.exists():
            continue
        run = json.loads((prefix / "training_manifest.json").read_text())
        selection = json.loads((prefix / "scorer_selection.json").read_text())
        assert run["checkpoint_state_sha256"] == selection["checkpoint_state_sha256"]
        assert selection["selection_unit"] == "canonical_identity_mean"
        assert selection["selected_scorer"] in SCORERS
        assert selection["development_diagnostic_only"] is True


def test_complete_matrix_artifacts_and_provenance_if_present():
    matrix_path = ROOT / "results/model_selection/p6/p6_matrix.json"
    if not matrix_path.exists():
        return
    matrix = json.loads(matrix_path.read_text())
    if matrix["status"] != "P6_DEVELOPMENT_PASS":
        return
    assert len(matrix["cells"]) == 15
    assert len({(cell["target"], cell["split_seed"]) for cell in matrix["cells"]}) == 15
    for cell in matrix["cells"]:
        target, seed = cell["target"], cell["split_seed"]
        prefix = ROOT / f"results/model_selection/p6/runs/{target.lower()}/seed_{seed}"
        run = json.loads((prefix / "training_manifest.json").read_text())
        reference = json.loads((prefix / "checkpoint_reference.json").read_text())
        states = json.loads((prefix / "scorer_states.json").read_text())
        selection = json.loads((prefix / "scorer_selection.json").read_text())
        calibration = json.loads((prefix / "calibration.json").read_text())
        diagnostics = json.loads((prefix / "development_diagnostics.json").read_text())
        checkpoint = ROOT / reference["path"]
        assert file_hash(checkpoint) == reference["checkpoint_file_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert _state_digest(payload["state_dict"]) == run["checkpoint_state_sha256"]
        verify_scorer_states(states, run["checkpoint_state_sha256"])
        assert selection["selected_scorer"] in SCORERS
        assert calibration["selected_scorer"] == selection["selected_scorer"]
        assert diagnostics["label"] == "DEVELOPMENT_DIAGNOSTIC_ONLY"
        dry = validate_p6_cell(ROOT, target, seed)
        manifest = frozen_case(target, seed)
        role = {"row_ids": manifest["roles"]["calibration"]["row_ids"],
                "identities": manifest["roles"]["calibration"]["canonical_identity_hashes"]}
        scorer_hash = states["states"][selection["selected_scorer"]]["scorer_state_sha256"]
        verify_calibration_artifact(calibration, dry, role,
                                    run["checkpoint_state_sha256"], scorer_hash)
        assert calibration["calibration_identity_count"] == cell["calibration_identity_count"]
        assert calibration["threshold"] == cell["threshold"]
