"""Repository and provenance checks for frozen P13C attempt 002."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/portable_schema/p13c/attempt_002"
RAW = ROOT / "data/controlled_capture/p13c/attempt_002/session_a_raw.pcap"
P13C_GATE_SHA256 = "6b8aef91964a18a606d3b992c42db18fe261b7c77b10d3abc3e0a51ed17fcb3f"
RAW_SHA256 = "570fad15116845545d92914aa1caf1848a1d979030352f15c1bd4c6caa84b986"
BLOCKED_SCHEDULE_SHA256 = "625374f24e04caf809a3021ac025e9fe549fa26bb3ebc7b097e963db942329de"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_gate_and_all_bound_artifact_hashes():
    gate_path = OUTPUT / "p13c_gate.json"
    assert sha256(gate_path) == P13C_GATE_SHA256
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "P13C_CONTROLLED_SOURCE_PASS"
    for name, expected in gate["artifact_sha256"].items():
        assert sha256(OUTPUT / name) == expected


def test_raw_reservation_is_distinct_and_optional_external_file_is_immutable():
    reservation = json.loads((OUTPUT / "raw_source_reservation.json").read_text())
    schedule_hash = sha256(OUTPUT / "scenario_schedule.json")
    assert reservation["reservation_status"] == "P13C_RAW_SOURCE_RESERVED"
    assert reservation["scenario_schedule_sha256"] == schedule_hash
    assert schedule_hash != BLOCKED_SCHEDULE_SHA256
    assert reservation["raw_files"][0]["sha256"] == RAW_SHA256
    if RAW.exists():
        assert RAW.stat().st_size == 77_284
        assert sha256(RAW) == RAW_SHA256
        assert RAW.stat().st_mode & 0o222 == 0


def test_p13c_is_deterministic_and_model_free():
    first = json.loads((OUTPUT / "extraction_run_1.json").read_text())
    second = json.loads((OUTPUT / "extraction_run_2.json").read_text())
    determinism = json.loads((OUTPUT / "determinism_audit.json").read_text())
    assert first["semantic_content_sha256"] == second["semantic_content_sha256"]
    assert determinism["status"] == "P13C_EXTRACTION_DETERMINISM_PASS"
    assert first["flow_count"] == second["flow_count"] == 80
    model = json.loads((OUTPUT / "model_access_audit.json").read_text())
    assert model["optimizer_steps"] == 0
    assert model["anomaly_scores"] == 0
    assert model["p7_model_evaluated"] is False
    assert model["p9_model_evaluated"] is False


def test_p13c_source_has_no_model_or_scaler_access():
    sources = [
        ROOT / "scripts/run_p13c_attempt_002.py",
        ROOT / "scripts/audit_p13c_attempt_002.py",
    ]
    text = "\n".join(path.read_text() for path in sources)
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
        assert forbidden not in text
