"""P13B controlled-corpus contract tests using synthetic observations only."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import PacketObservation, build_portable_flows  # noqa: E402
from ids.p13.labels import LabelJoinResult  # noqa: E402
from ids.p13.portable_extractor import extract_rows, semantic_content_hash  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    CORPUS_NAME, aggregate_corpus_sha256, file_sha256, identity_label_audit,
    join_controlled_labels, pretty_json_bytes, verify_raw_reservation,
)


def packet(index, timestamp, src="127.0.0.2", dst="127.0.0.3", sport=50000, dport=18080,
           protocol=6, length=60, flags=0x10):
    return PacketObservation(
        index, timestamp, 4, src, dst, protocol, sport, dport, length,
        flags if protocol == 6 else None,
    )


def schedule(start=100, end=200):
    return {
        "scenarios": [{
            "scenario_id": "normal_http",
            "scenario_type": "NORMAL_HTTP",
            "portable_label": "NORMAL",
            "generator_host": "127.0.0.2",
            "target_host": "127.0.0.3",
            "ip_protocol": 6,
            "destination_port": 18080,
            "expected_start_ns": start,
            "expected_end_ns": end,
        }]
    }


def test_schedule_serialization_is_deterministic():
    left = pretty_json_bytes({"b": 2, "a": [1, 2]})
    right = pretty_json_bytes({"a": [1, 2], "b": 2})
    assert left == right


def test_raw_source_hash_and_read_only_reservation(tmp_path):
    raw = tmp_path / "raw.pcap"
    raw.write_bytes(b"controlled-capture")
    raw.chmod(0o444)
    schedule_path = tmp_path / "schedule.json"
    schedule_path.write_bytes(pretty_json_bytes(schedule()))
    entry = {
        "filename": raw.name,
        "path": raw.name,
        "size_bytes": raw.stat().st_size,
        "sha256": file_sha256(raw),
    }
    reservation = {
        "corpus_name": CORPUS_NAME,
        "reservation_status": "RAW_RESERVED",
        "scenario_schedule_sha256": file_sha256(schedule_path),
        "raw_files": [entry],
        "aggregate_corpus_sha256": aggregate_corpus_sha256([entry]),
    }
    verify_raw_reservation(tmp_path, reservation, schedule_path)
    raw.chmod(0o644)
    raw.write_bytes(b"mutated")
    raw.chmod(0o444)
    with pytest.raises(ValueError):
        verify_raw_reservation(tmp_path, reservation, schedule_path)


def test_label_join_inclusive_boundaries_unmatched_and_overlap_ambiguity():
    boundary_flows = build_portable_flows([packet(0, 100), packet(1, 200, sport=50001)])
    joined = join_controlled_labels(boundary_flows, schedule())
    assert [item.status for item in joined] == ["MATCHED", "MATCHED"]
    unmatched = build_portable_flows([packet(2, 201)])
    assert join_controlled_labels(unmatched, schedule())[0].status == "UNMATCHED"
    overlapping = schedule()
    overlapping["scenarios"].append({
        **overlapping["scenarios"][0],
        "scenario_id": "attack_overlap",
        "scenario_type": "ATTACK_HTTP",
        "portable_label": "ATTACK",
    })
    assert join_controlled_labels([boundary_flows[0]], overlapping)[0].status == "AMBIGUOUS"


def test_identity_label_conflict_is_surfaced_without_majority_vote():
    first = build_portable_flows([packet(0, 100)])[0]
    second = build_portable_flows([packet(1, 1_000, src="127.0.0.4", dst="127.0.0.5")])[0]
    rows = [extract_rows([first])[0], extract_rows([second])[0]]
    assert rows[0].canonical_identity_v3 == rows[1].canonical_identity_v3
    labels = [
        LabelJoinResult("MATCHED", "Normal", "NORMAL", ("n",)),
        LabelJoinResult("MATCHED", "Attack", "ATTACK", ("a",)),
    ]
    audit = identity_label_audit(rows, labels)
    assert audit["mixed_identity_count"] == 1
    assert audit["mixed_flow_count"] == 2
    assert audit["majority_vote_used"] is False


def test_content_hash_equality_and_label_independent_extraction():
    flow = build_portable_flows([packet(0, 100), packet(1, 150, length=80)])[0]
    before = extract_rows([flow])
    labels = join_controlled_labels([flow], schedule())
    after = extract_rows([flow])
    assert labels[0].portable_label == "NORMAL"
    assert before == after
    assert semantic_content_hash(before) == semantic_content_hash(after)


def test_no_model_access_or_scaling_code():
    paths = [
        ROOT / "src/ids/p13b_controlled.py",
        ROOT / "scripts/run_p13b_controlled_capture.py",
        ROOT / "scripts/audit_p13b_controlled_capture.py",
    ]
    source = "\n".join(path.read_text() for path in paths)
    for forbidden in (
        "import torch", "ids.p9", "load_state_dict", "optimizer.step",
        "RobustScaler", "StandardScaler", "roc_auc", "average_precision_score",
    ):
        assert forbidden not in source
    assert json.loads(json.dumps({"optimizer_steps": 0}))["optimizer_steps"] == 0
