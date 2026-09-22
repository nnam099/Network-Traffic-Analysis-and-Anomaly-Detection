"""P5 contract tests never open official test or run optimizer steps."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p4_model_interface import (  # noqa: E402
    StructuredModelInput, bound_model_config, vocabularies_from_manifest,
)
from ids.p5_training import (  # noqa: E402
    PERMISSIONS, TRAINING_CONFIG, TrainingOnlyModel, batch_order, checked_output,
    assert_checkpoint_provenance, require_role, set_determinism, source_hash,
    stable_hash, validate_cell,
    validate_roles,
)


def case():
    with gzip.open(ROOT / "results/data_quality/p3/manifests/fuzzers_seed42.json.gz", "rt") as file:
        return json.load(file)


def test_role_matrix_is_fail_closed():
    assert PERMISSIONS["gradient_fit"] == ["train"]
    assert PERMISSIONS["checkpoint_selection"] == ["val"]
    assert PERMISSIONS["threshold_fit"] == ["calibration"]
    assert PERMISSIONS["final_external_evaluation"] == []
    for operation in PERMISSIONS:
        for role in ("official_test", "target_development_held_out"):
            with pytest.raises(PermissionError):
                require_role(operation, role)
    for role in ("val", "meta_known", "calibration", "surrogate_ood"):
        with pytest.raises(PermissionError):
            require_role("gradient_fit", role)
    for role in ("meta_known", "calibration", "surrogate_ood", "train"):
        with pytest.raises(PermissionError):
            require_role("checkpoint_selection", role)
    for role in ("train", "val", "meta_known", "surrogate_ood"):
        with pytest.raises(PermissionError):
            require_role("threshold_fit", role)


def test_role_manifest_rejects_overlap_and_test_consultation():
    manifest = case()
    validate_roles(manifest)
    contaminated = copy.deepcopy(manifest)
    contaminated["roles"]["val"]["canonical_identity_hashes"][0] = contaminated["roles"]["train"]["canonical_identity_hashes"][0]
    with pytest.raises(ValueError, match="overlap"):
        validate_roles(contaminated)
    contaminated = copy.deepcopy(manifest)
    contaminated["official_test_consulted"] = True
    with pytest.raises(ValueError, match="test-informed"):
        validate_roles(contaminated)


def test_all_frozen_cells_and_provenance_mismatch_detection():
    p4 = json.loads((ROOT / "results/model_design/p4/p4_matrix.json").read_text())
    for item in p4["cells"]:
        dry = validate_cell(ROOT, item["target"], item["seed"], 10000 + item["seed"])
        assert dry["status"] == "DRY_RUN_PASS"
        assert dry["official_test_access"] is False
        assert dry["p4_model_config_sha256"] == item["bound_model_config"]["model_config_sha256"]
    with pytest.raises(ValueError, match="unsupported"):
        validate_cell(ROOT, "NotATarget", 42, 10042)
    with pytest.raises(ValueError, match="model seed"):
        validate_cell(ROOT, "Fuzzers", 42, 42)
    manifest = case()
    broken = copy.deepcopy(manifest)
    broken["vocabulary_provenance"]["vocabulary_maps"]["proto"]["value:tcp"] = 999
    from ids.p4_model_interface import vocabularies_from_manifest
    with pytest.raises(ValueError, match="vocabulary hash"):
        vocabularies_from_manifest(broken)
    schema = json.loads((ROOT / "results/data_quality/p3/schema_v2.json").read_text())
    broken = copy.deepcopy(manifest)
    broken["scaler_provenance"]["scaler_center"][0] += 1.0
    with pytest.raises(ValueError, match="scaler hash"):
        bound_model_config(broken, manifest_sha256="ignored", p3_collection_sha256="ignored",
                           schema=schema, source_code_sha256="ignored")


def test_checkpoint_provenance_rejects_wrong_manifest_config_vocab_scaler():
    dry = validate_cell(ROOT, "Fuzzers", 42, 10042)
    run = {**{name: dry[name] for name in (
        "p3_manifest_sha256", "p4_model_config_sha256", "training_config_sha256",
        "vocabulary_sha256", "scaler_sha256", "source_hash")},
        "run_id": "p5-fuzzers-s42-m10042-smoke-a", "selected_epoch": 3,
        "validation_selection_value": 0.5}
    payload = {"provenance": {"target": "Fuzzers", "split_seed": 42, "model_seed": 10042,
                              "run_id": run["run_id"], "epoch": 3,
                              "validation_selection_value": 0.5,
                              **{name: dry[name] for name in (
                                  "p3_manifest_sha256", "p4_model_config_sha256",
                                  "training_config_sha256", "vocabulary_sha256",
                                  "scaler_sha256", "source_hash")}}}
    assert_checkpoint_provenance(payload, run, dry)
    for name in ("p3_manifest_sha256", "p4_model_config_sha256", "vocabulary_sha256", "scaler_sha256"):
        altered = copy.deepcopy(payload)
        altered["provenance"][name] = "altered"
        with pytest.raises(ValueError, match="checkpoint provenance"):
            assert_checkpoint_provenance(altered, run, dry)


def test_float64_initialization_and_deterministic_batches():
    vocab = vocabularies_from_manifest(case())
    set_determinism(10042)
    first = TrainingOnlyModel(vocab)
    state = {name: value.clone() for name, value in first.state_dict().items()}
    set_determinism(10042)
    second = TrainingOnlyModel(vocab)
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in state.items())
    assert all(parameter.dtype == torch.float64 for parameter in first.parameters())
    assert np.array_equal(batch_order(99, 10042, 1), batch_order(99, 10042, 1))
    assert not np.array_equal(batch_order(99, 10042, 1), batch_order(99, 10042, 2))
    assert stable_hash(TRAINING_CONFIG) == stable_hash(json.loads(json.dumps(TRAINING_CONFIG)))
    with pytest.raises(TypeError, match="torch.float64"):
        StructuredModelInput(torch.zeros(2, 58, dtype=torch.float32),
                             torch.zeros(2, dtype=torch.long),
                             torch.zeros(2, dtype=torch.long),
                             torch.zeros(2, dtype=torch.long))


def test_output_namespace_and_cli_default_do_not_train():
    with pytest.raises(PermissionError):
        checked_output(ROOT, "../../data/UNSW_NB15_testing-set.csv")
    with pytest.raises(PermissionError):
        checked_output(ROOT, "official_test/results.json")
    before = set((ROOT / "results/training_protocol/p5/checkpoints").glob("*.pt"))
    result = subprocess.run([sys.executable, str(ROOT / "scripts/train_p5_development.py"),
                             "--target", "Fuzzers", "--split-seed", "42",
                             "--model-seed", "10042"], cwd=ROOT,
                            capture_output=True, text=True, check=True)
    assert '"optimizer_created": false' in result.stdout
    assert set((ROOT / "results/training_protocol/p5/checkpoints").glob("*.pt")) == before
    source_files, _ = source_hash(ROOT)
    assert "src/train.py" not in source_files
    p5_source = (ROOT / "src/ids/p5_training.py").read_text()
    assert "from .trainer" not in p5_source and "from .models" not in p5_source


def test_legacy_training_cli_rejects_schema_v2_request():
    result = subprocess.run([sys.executable, str(ROOT / "src/train.py"),
                             "--schema_version", "scientific-feature-schema-v2"],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_smoke_artifacts_if_present():
    base = ROOT / "results/training_protocol/p5"
    matrix_path = base / "p5_matrix.json"
    if matrix_path.exists():
        matrix = json.loads(matrix_path.read_text())
        assert matrix["dry_run_optimizer_steps"] == 0
        assert matrix["smoke_optimizer_steps"] == sum(
            item["optimizer_steps"] for item in matrix.get("smoke_runs", {}).values()
        )
    for tag in ("smoke-a", "smoke-b"):
        path = base / f"training_runs/p5-fuzzers-s42-m10042-{tag}.json"
        if not path.exists():
            continue
        run = json.loads(path.read_text())
        for name in ("p3_manifest_sha256", "p4_model_config_sha256",
                     "training_config_sha256", "source_hash", "vocabulary_sha256",
                     "scaler_sha256", "environment", "role_counts", "identity_counts",
                     "epoch_history", "checkpoint_sha256", "selection_rule"):
            assert name in run
        assert run["official_test_access"] is False
        assert run["optimizer_steps"] > 0
        assert run["scientific_effectiveness_claim"] is False
        assert all("final_performance" not in key for key in run)
