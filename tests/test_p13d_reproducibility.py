"""P13D provenance, independence and identity-comparison tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p13d_reproducibility import compare_identity_label_maps  # noqa: E402


OUTPUT = ROOT / "results/portable_schema/p13d"
SESSION_A = ROOT / "results/portable_schema/p13c/attempt_002"
P13C_GATE_SHA256 = "6b8aef91964a18a606d3b992c42db18fe261b7c77b10d3abc3e0a51ed17fcb3f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_session_a_reference_is_immutable():
    assert sha256(SESSION_A / "p13c_gate.json") == P13C_GATE_SHA256
    reference = json.loads((OUTPUT / "session_a_reference.json").read_text())
    assert reference["gate_sha256"] == P13C_GATE_SHA256


def test_session_b_source_and_schedule_are_distinct_and_reserved():
    reference = json.loads((OUTPUT / "session_a_reference.json").read_text())
    reservation = json.loads((OUTPUT / "session_b_raw_reservation.json").read_text())
    assert reservation["reservation_status"] == "P13D_SESSION_B_RAW_RESERVED"
    assert reservation["raw_files"][0]["sha256"] != reference["raw_sha256"]
    assert reservation["schedule_sha256"] != reference["schedule_sha256"]


def test_session_b_double_extraction_and_implementation_binding():
    first = json.loads((OUTPUT / "session_b_extraction_run_1.json").read_text())
    second = json.loads((OUTPUT / "session_b_extraction_run_2.json").read_text())
    audit = json.loads((OUTPUT / "session_b_determinism.json").read_text())
    binding = json.loads((OUTPUT / "implementation_binding.json").read_text())
    assert first["semantic_content_sha256"] == second["semantic_content_sha256"]
    assert audit["status"] == "P13D_SESSION_B_DETERMINISM_PASS"
    assert binding["status"] == "PASS"
    assert all(
        item["matches_frozen_commit"]
        for item in binding["scientific_source_bindings"].values()
    )


def test_cross_session_label_conflict_is_surfaced_without_relabeling():
    comparison = compare_identity_label_maps(
        {"same": {"NORMAL"}, "conflict": {"NORMAL"}},
        {"same": {"NORMAL"}, "conflict": {"ATTACK"}},
    )
    assert comparison["intersection_count"] == 2
    assert comparison["cross_label_intersection_count"] == 1
    assert comparison["group_identity_review_required"] is True
    assert comparison["cross_session_label_relationship_counts"] == {
        "A_NORMAL/B_ATTACK": 1,
        "A_NORMAL/B_NORMAL": 1,
    }


def test_p13d_has_no_model_access_or_ml_split():
    model = json.loads((OUTPUT / "model_access_audit.json").read_text())
    combined = json.loads((OUTPUT / "combined_population_audit.json").read_text())
    assert model["optimizer_steps"] == 0
    assert model["model_forward_passes"] == 0
    assert model["anomaly_scores"] == 0
    assert combined["ml_splits_created"] is False
    sources = "\n".join(
        (ROOT / name).read_text()
        for name in (
            "scripts/run_p13d_session_b.py",
            "scripts/audit_p13d_reproducibility.py",
        )
    )
    for forbidden in (
        "import torch",
        "ids.p9",
        "load_state_dict",
        "optimizer.step",
        "RobustScaler",
        "StandardScaler",
        "roc_auc",
        "average_precision_score",
    ):
        assert forbidden not in sources
