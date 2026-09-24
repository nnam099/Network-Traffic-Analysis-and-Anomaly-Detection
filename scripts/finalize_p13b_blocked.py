#!/usr/bin/env python3
"""Record a fail-closed P13B gate when raw capture cannot be authorized."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/portable_schema/p13b"
RAW_RELATIVE = Path("data/controlled_capture/p13b/session_a_raw.pcap")
P12_FREEZE_COMMIT = "720789fb90f36e1413cef47cd96f3126d2b8f3a4"
P12_PROTOCOL_SHA256 = "db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3"
P12_SCHEMA_SHA256 = "db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a"
STATUS = "P13B_EXTRACTION_BLOCKED"
BLOCKER = (
    "Raw capture precondition failed before traffic generation: tcpdump could not open "
    "the loopback capture device because CAP_NET_RAW was unavailable, and passwordless "
    "sudo was not authorized. No raw PCAP exists, so extraction and downstream audits "
    "cannot run."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(json_bytes(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def not_run(artifact_type: str, *, extra: dict[str, object] | None = None) -> dict[str, object]:
    artifact: dict[str, object] = {
        "artifact_type": artifact_type,
        "status": "NOT_RUN_NO_RAW_PCAP",
        "reason": BLOCKER,
        "raw_pcap_exists": False,
    }
    if extra:
        artifact.update(extra)
    return artifact


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw_path = ROOT / RAW_RELATIVE
    if raw_path.exists():
        raise SystemExit("Refusing blocked finalization because a raw P13B PCAP exists")
    if git("rev-parse", "HEAD") != P12_FREEZE_COMMIT:
        raise SystemExit("Refusing blocked finalization because HEAD changed")
    if git("status", "--porcelain", "--untracked-files=no"):
        raise SystemExit("Refusing blocked finalization because tracked files changed")

    schedule_path = OUTPUT / "scenario_schedule.json"
    topology_path = OUTPUT / "lab_topology.json"
    environment_path = OUTPUT / "capture_environment.json"
    for required in (schedule_path, topology_path, environment_path):
        if not required.is_file():
            raise SystemExit(f"Missing pre-capture artifact: {required}")

    created_at = utc_now()
    schedule = json.loads(schedule_path.read_text())
    scenario_ids = [item["scenario_id"] for item in schedule["scenarios"]]
    source_paths = [
        ROOT / "src/ids/p12_portable_schema.py",
        ROOT / "src/ids/p13/labels.py",
        ROOT / "src/ids/p13/portable_extractor.py",
        ROOT / "src/ids/p13/raw_reader/__init__.py",
        ROOT / "src/ids/p13/raw_reader/classic_pcap.py",
        ROOT / "src/ids/p13/validation.py",
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path) for path in source_paths
    }

    write_json(
        "capture_attempt_audit.json",
        {
            "artifact_type": "P13B_CAPTURE_ATTEMPT_AUDIT",
            "status": "CAPTURE_PRIVILEGE_BLOCKED",
            "created_at_utc": created_at,
            "attempts": [
                {
                    "attempt": 1,
                    "tool": "tcpdump",
                    "result": "FAILED_BEFORE_CAPTURE",
                    "error": (
                        "tcpdump: lo: You don't have permission to perform this capture "
                        "on that device (socket: Operation not permitted)"
                    ),
                },
                {
                    "attempt": 2,
                    "tool": "sudo -n tcpdump -D",
                    "result": "FAILED_BEFORE_CAPTURE",
                    "error": "sudo: a password is required",
                },
            ],
            "traffic_generation_started": False,
            "raw_pcap_created": False,
            "packets_captured": 0,
            "external_destinations_contacted": False,
            "blocker": BLOCKER,
        },
    )
    write_json(
        "scenario_execution_log.json",
        {
            "artifact_type": "P13B_SCENARIO_EXECUTION_LOG",
            "status": "NOT_EXECUTED_CAPTURE_PRIVILEGE_BLOCKED",
            "scenario_schedule_sha256": file_sha256(schedule_path),
            "scheduled_scenario_ids": scenario_ids,
            "executed_scenario_ids": [],
            "traffic_generation_started": False,
            "reason": BLOCKER,
        },
    )
    write_json(
        "raw_source_reservation.json",
        {
            "artifact_type": "P13B_RAW_SOURCE_RESERVATION",
            "corpus_name": "P13B_CONTROLLED_CAPTURE",
            "reservation_status": "NOT_RESERVED_CAPTURE_PRIVILEGE_BLOCKED",
            "created_at_utc": created_at,
            "raw_files": [],
            "aggregate_corpus_sha256": None,
            "scenario_schedule_sha256": file_sha256(schedule_path),
            "raw_pcap_created": False,
            "reason": BLOCKER,
        },
    )
    write_json(
        "capture_format_audit.json",
        not_run(
            "P13B_CAPTURE_FORMAT_AUDIT",
            extra={
                "link_layer_type": None,
                "timestamp_precision": None,
                "packet_count": 0,
            },
        ),
    )
    write_json(
        "extraction_run_1.json",
        not_run(
            "P13B_EXTRACTION_RUN",
            extra={"run": 1, "packet_count": 0, "flow_count": 0, "semantic_content_sha256": None},
        ),
    )
    write_json(
        "extraction_run_2.json",
        not_run(
            "P13B_EXTRACTION_RUN",
            extra={"run": 2, "packet_count": 0, "flow_count": 0, "semantic_content_sha256": None},
        ),
    )
    write_json(
        "determinism_audit.json",
        not_run(
            "P13B_DETERMINISM_AUDIT",
            extra={"deterministic_double_run": False, "comparison_performed": False},
        ),
    )
    write_json(
        "label_join_audit.json",
        not_run(
            "P13B_LABEL_JOIN_AUDIT",
            extra={
                "outcome_counts": {
                    "MATCHED_NORMAL": 0,
                    "MATCHED_ATTACK": 0,
                    "UNMATCHED": 0,
                    "AMBIGUOUS": 0,
                },
                "majority_vote_used": False,
            },
        ),
    )
    empty_populations = {
        name: {"rows": 0, "canonical_identities": 0}
        for name in ("NORMAL", "ATTACK", "UNMATCHED", "AMBIGUOUS")
    }
    write_json(
        "population_audit.json",
        not_run(
            "P13B_POPULATION_AUDIT",
            extra={"populations": empty_populations, "per_scenario": {}},
        ),
    )
    write_json(
        "identity_multiplicity_audit.json",
        not_run(
            "P13B_IDENTITY_MULTIPLICITY_AUDIT",
            extra={
                "total_flow_instances": 0,
                "unique_canonical_identities": 0,
                "largest_identity_groups": [],
            },
        ),
    )
    write_json(
        "cross_label_identity_audit.json",
        not_run(
            "P13B_CROSS_LABEL_IDENTITY_AUDIT",
            extra={
                "normal_only_identity_count": 0,
                "attack_only_identity_count": 0,
                "mixed_identity_count": 0,
                "mixed_flow_count": 0,
                "identity_review_required": False,
                "identity_review_evaluated": False,
            },
        ),
    )
    write_json(
        "feature_sanity_audit.json",
        not_run(
            "P13B_FEATURE_SANITY_AUDIT",
            extra={"groups": {}, "feature_selection_performed": False},
        ),
    )
    write_json(
        "protocol_audit.json",
        not_run(
            "P13B_PROTOCOL_AUDIT",
            extra={"counts": {}, "rare_protocols_dropped": False},
        ),
    )
    write_json(
        "scenario_coverage.json",
        not_run(
            "P13B_SCENARIO_COVERAGE",
            extra={
                "scheduled_scenarios": scenario_ids,
                "executed_scenarios": [],
                "scenarios": {},
            },
        ),
    )

    artifact_names = sorted(
        path.name
        for path in OUTPUT.glob("*.json")
        if path.name != "p13b_gate.json"
    )
    artifact_hashes = {
        name: file_sha256(OUTPUT / name) for name in artifact_names
    }
    write_json(
        "p13b_gate.json",
        {
            "artifact_type": "P13B_GATE",
            "corpus_name": "P13B_CONTROLLED_CAPTURE",
            "version": "p13b-controlled-capture-v1",
            "status": STATUS,
            "gate_phase": "CAPTURE_PRECONDITION",
            "created_at_utc": created_at,
            "blocker": BLOCKER,
            "p12_freeze_commit": P12_FREEZE_COMMIT,
            "p12_protocol_sha256": P12_PROTOCOL_SHA256,
            "p12_schema_sha256": P12_SCHEMA_SHA256,
            "repository_head": git("rev-parse", "HEAD"),
            "tracked_worktree_clean": True,
            "p13_source_sha256": source_hashes,
            "artifact_sha256": artifact_hashes,
            "raw_pcap_created": False,
            "raw_aggregate_sha256": None,
            "traffic_generation_started": False,
            "complete_19_field_extraction": False,
            "deterministic_double_run": False,
            "label_join_pass": False,
            "population_audit_completed": False,
            "identity_multiplicity_audited": False,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "anomaly_scores": 0,
            "model_metrics": 0,
            "historical_model_opened": False,
            "scaler_fitted": False,
            "readiness": "CAPTURE_PRIVILEGE_REQUIRED",
        },
    )
    print(STATUS)
    print(f"scenario_schedule_sha256={file_sha256(schedule_path)}")
    print(f"p13b_gate_sha256={file_sha256(OUTPUT / 'p13b_gate.json')}")
    print("No neural network was trained and no historical detector was evaluated during P13B.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
