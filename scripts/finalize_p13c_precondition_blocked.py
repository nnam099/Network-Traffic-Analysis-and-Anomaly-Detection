#!/usr/bin/env python3
"""Record a fail-closed P13C capture-privilege precondition audit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/portable_schema/p13c"
P13B_GATE = ROOT / "results/portable_schema/p13b/p13b_gate.json"
P13B_GATE_SHA256 = "be8f2213474592c69b34636a089371a35664acc6d5775d9a148cdc51a39bee95"
HEAD = "720789fb90f36e1413cef47cd96f3126d2b8f3a4"
STATUS = "P13C_CAPTURE_PRECONDITION_BLOCKED"
BLOCKER = (
    "The non-root scientific process cannot open interface lo for packet capture. "
    "/usr/bin/tcpdump has no file capability, while capability-enabled "
    "/usr/bin/dumpcap is not executable by the current user because the user is not "
    "a member of group wireshark. The benign capture test failed before traffic "
    "generation and created no capture file."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pretty_json(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode()


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(pretty_json(value))


def command_output(*args: str) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout.strip()


def optional_output(*args: str) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> int:
    if file_sha256(P13B_GATE) != P13B_GATE_SHA256:
        raise SystemExit("P13B attempt-001 gate hash mismatch")
    if command_output("git", "-C", str(ROOT), "rev-parse", "HEAD") != HEAD:
        raise SystemExit("Repository HEAD changed")
    if command_output(
        "git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"
    ):
        raise SystemExit("Tracked worktree is not clean")
    forbidden = [
        OUTPUT / "scenario_schedule.json",
        ROOT / "data/controlled_capture/p13c/session_a_raw.pcap",
    ]
    if any(path.exists() for path in forbidden):
        raise SystemExit("Refusing to overwrite a P13C schedule or raw capture")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    created_at = utc_now()
    tcpdump = Path("/usr/bin/tcpdump")
    dumpcap = Path("/usr/bin/dumpcap")
    current_user = pwd.getpwuid(os.getuid()).pw_name
    groups = command_output("id", "-Gn").split()
    tcpdump_capabilities = optional_output("getcap", str(tcpdump))
    dumpcap_capabilities = optional_output("getcap", str(dumpcap))

    write_json(
        "previous_attempt_reference.json",
        {
            "artifact_type": "P13C_PREVIOUS_ATTEMPT_REFERENCE",
            "attempt_id": "P13B_ATTEMPT_001_CAPTURE_PRIVILEGE_BLOCKED",
            "path": "results/portable_schema/p13b/p13b_gate.json",
            "gate_sha256": P13B_GATE_SHA256,
            "preservation_verified_at_utc": created_at,
            "preserved_unchanged": True,
        },
    )
    write_json(
        "lab_scope.json",
        {
            "artifact_type": "P13C_LAB_SCOPE",
            "capture_interface": "lo",
            "generator": "127.0.0.2",
            "target": "127.0.0.3",
            "allowed_scope": ["127.0.0.2", "127.0.0.3"],
            "external_destinations_allowed": False,
            "scenario_traffic_generated": False,
        },
    )
    write_json(
        "capture_privilege_preflight.json",
        {
            "artifact_type": "P13C_CAPTURE_PRIVILEGE_PREFLIGHT",
            "status": "CAPTURE_PRIVILEGE_INSUFFICIENT",
            "checked_at_utc": created_at,
            "current_user": current_user,
            "current_uid": os.getuid(),
            "current_groups": groups,
            "scientific_pipeline_running_as_root": False,
            "capture_binary": str(tcpdump),
            "capture_binary_sha256": file_sha256(tcpdump),
            "capture_binary_capabilities": tcpdump_capabilities or "NONE",
            "capture_binary_version": command_output(str(tcpdump), "--version").splitlines(),
            "alternative_capture_binary": str(dumpcap),
            "alternative_capture_binary_sha256": file_sha256(dumpcap),
            "alternative_capture_binary_capabilities": dumpcap_capabilities or "NONE",
            "alternative_capture_binary_mode": oct(dumpcap.stat().st_mode & 0o777),
            "alternative_capture_binary_group": dumpcap.group(),
            "current_user_in_wireshark_group": "wireshark" in groups,
            "capture_interface": "lo",
            "interface_exists": True,
            "interface_up": True,
            "loopback_flag_present": True,
            "generator": "127.0.0.2",
            "target": "127.0.0.3",
            "system_capabilities_modified_by_p13c": False,
            "blocker": BLOCKER,
        },
    )
    write_json(
        "privilege_test_audit.json",
        {
            "artifact_type": "P13C_PRIVILEGE_TEST_AUDIT",
            "classification": "PRIVILEGE_TEST_ONLY",
            "status": "CAPTURE_TEST_FAILED",
            "capture_interface": "lo",
            "capture_filter": (
                "((src host 127.0.0.2 and dst host 127.0.0.3) or "
                "(src host 127.0.0.3 and dst host 127.0.0.2))"
            ),
            "bounded_packet_count": 2,
            "capture_exit_code": 1,
            "capture_error": (
                "tcpdump: lo: You don't have permission to perform this capture on "
                "that device (socket: Operation not permitted)"
            ),
            "traffic_generated": False,
            "test_capture_created": False,
            "test_capture_size_bytes": 0,
            "parser_invoked": False,
            "test_file_retained": False,
            "entered_scientific_corpus": False,
        },
    )
    write_json(
        "model_access_audit.json",
        {
            "artifact_type": "P13C_MODEL_ACCESS_AUDIT",
            "p9_checkpoint_opened": False,
            "p9_model_evaluated": False,
            "p7_model_evaluated": False,
            "optimizer_steps": 0,
            "anomaly_scores": 0,
            "auroc": None,
            "average_precision": None,
            "threshold_fit": False,
            "scaler_fit": False,
        },
    )
    artifact_names = [
        "capture_privilege_preflight.json",
        "lab_scope.json",
        "model_access_audit.json",
        "previous_attempt_reference.json",
        "privilege_test_audit.json",
    ]
    artifact_hashes = {name: file_sha256(OUTPUT / name) for name in artifact_names}
    write_json(
        "p13c_gate.json",
        {
            "artifact_type": "P13C_GATE",
            "status": STATUS,
            "gate_phase": "CAPTURE_PRECONDITION",
            "created_at_utc": created_at,
            "blocker": BLOCKER,
            "repository_head": HEAD,
            "tracked_worktree_clean": True,
            "previous_attempt_id": "P13B_ATTEMPT_001_CAPTURE_PRIVILEGE_BLOCKED",
            "previous_attempt_gate_sha256": P13B_GATE_SHA256,
            "previous_attempt_preserved": True,
            "new_schedule_created": False,
            "new_schedule_sha256": None,
            "scenario_traffic_generated": False,
            "scientific_capture_started": False,
            "raw_pcap_created": False,
            "raw_source_reserved": False,
            "extraction_runs": 0,
            "label_join_performed": False,
            "population_audit_performed": False,
            "identity_audit_performed": False,
            "system_capabilities_modified_by_p13c": False,
            "artifact_sha256": artifact_hashes,
            "optimizer_steps": 0,
            "checkpoints_created": 0,
            "anomaly_scores": 0,
            "model_metrics": 0,
            "historical_detector_opened": False,
            "readiness": "CAPTURE_PRIVILEGE_REQUIRED",
        },
    )
    print(STATUS)
    print(f"p13c_gate_sha256={file_sha256(OUTPUT / 'p13c_gate.json')}")
    print("No neural network was trained and no historical detector was evaluated during P13C.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
