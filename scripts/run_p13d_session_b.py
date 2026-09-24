#!/usr/bin/env python3
"""Capture and reserve independent controlled P13D Session B."""

from __future__ import annotations

from datetime import datetime, timezone
import grp
import hashlib
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ids.p12_portable_schema import IDLE_TIMEOUT_NS  # noqa: E402
from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    aggregate_corpus_sha256,
    file_sha256,
    pretty_json_bytes,
)
import run_p13b_controlled_capture as traffic  # noqa: E402


OUTPUT = ROOT / "results/portable_schema/p13d"
RAW_RELATIVE = Path("data/controlled_capture/p13d/session_b_raw.pcap")
DUMPCAP = Path("/usr/bin/dumpcap")
P13C_FREEZE_COMMIT = "ce0b6be7604acd748015834df3f024e62afb5d07"
P13C_GATE_SHA256 = "6b8aef91964a18a606d3b992c42db18fe261b7c77b10d3abc3e0a51ed17fcb3f"
SESSION_A_RAW_SHA256 = "570fad15116845545d92914aa1caf1848a1d979030352f15c1bd4c6caa84b986"
SESSION_A_SCHEDULE_SHA256 = "4920c3efcba623d1838a9f3f3a1fd4ff008925288a3b0e5b7ca7538f2a886761"
SESSION_A_SEMANTIC_SHA256 = "f65f0a71d35f432011ab4abe7159319499ed6385809d3829b0066ce6d222a526"
P12_PROTOCOL_SHA256 = "db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3"
P12_SCHEMA_SHA256 = "db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a"
SCIENTIFIC_SOURCES = [
    "src/ids/p12_portable_schema.py",
    "src/ids/p13/labels.py",
    "src/ids/p13/portable_extractor.py",
    "src/ids/p13/raw_reader/__init__.py",
    "src/ids/p13/raw_reader/classic_pcap.py",
    "src/ids/p13/validation.py",
    "src/ids/p13b_controlled.py",
]


def utc_iso(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_output(*args: str, text: bool = True):
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=text,
    ).stdout


def sleep_until(target_monotonic_ns: int) -> None:
    while True:
        remaining = (target_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def udp_independent_flows(count: int) -> None:
    for index in range(count):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
            client.settimeout(2)
            client.bind((traffic.GENERATOR, 0))
            payload = f"p13d-session-b-udp-{index}".encode()
            client.sendto(payload, (traffic.TARGET, traffic.PORTS["udp"]))
            response, _ = client.recvfrom(4096)
            if response != payload:
                raise RuntimeError("controlled UDP echo mismatch")


def verify_implementation_binding() -> dict[str, object]:
    if git_output("rev-parse", "HEAD").strip() != P13C_FREEZE_COMMIT:
        raise RuntimeError("HEAD is not the frozen P13C commit")
    if git_output("status", "--porcelain", "--untracked-files=no").strip():
        raise RuntimeError("tracked worktree changed after P13C freeze")
    bindings = {}
    for relative in SCIENTIFIC_SOURCES:
        current = file_sha256(ROOT / relative)
        frozen_bytes = git_output("show", f"{P13C_FREEZE_COMMIT}:{relative}", text=False)
        frozen = sha256_bytes(frozen_bytes)
        if current != frozen:
            raise RuntimeError(f"P13D_IMPLEMENTATION_DRIFT_BLOCKED: {relative}")
        bindings[relative] = {
            "sha256": current,
            "matches_frozen_commit": True,
        }
    return {
        "artifact_type": "P13D_IMPLEMENTATION_BINDING",
        "status": "PASS",
        "p13c_freeze_commit": P13C_FREEZE_COMMIT,
        "p12_protocol_sha256": P12_PROTOCOL_SHA256,
        "p12_schema_sha256": P12_SCHEMA_SHA256,
        "scientific_source_bindings": bindings,
        "idle_timeout_ns": IDLE_TIMEOUT_NS,
        "direction_rule": "first observed packet defines forward direction",
        "protocol_mapping": "IPPROTO_n from the observed IP protocol number",
        "canonical_identity_contract": "canonical-flow-identity-v3",
        "label_join_contract": (
            "inclusive flow start + endpoints + protocol + optional destination port"
        ),
        "scientific_implementation_changed_after_p13c_freeze": False,
    }


def verify_capture_privilege() -> dict[str, object]:
    wireshark_gid = grp.getgrnam("wireshark").gr_gid
    if os.geteuid() == 0:
        raise RuntimeError("scientific capture must not run as root")
    if wireshark_gid not in {os.getgid(), *os.getgroups()}:
        raise RuntimeError("wireshark group is not active")
    capabilities = subprocess.run(
        ["getcap", str(DUMPCAP)], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "cap_net_raw" not in capabilities or "cap_net_admin" not in capabilities:
        raise RuntimeError("dumpcap capabilities are incomplete")
    interfaces = subprocess.run(
        [str(DUMPCAP), "-D"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    if not any(line.strip().endswith("lo (Loopback)") for line in interfaces):
        raise RuntimeError("loopback interface unavailable")
    version = subprocess.run(
        [str(DUMPCAP), "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "artifact_type": "P13D_CAPTURE_PRIVILEGE_PREFLIGHT",
        "status": "PASS",
        "verified_at_utc": utc_iso(time.time_ns()),
        "effective_uid": os.geteuid(),
        "active_group_names": subprocess.run(
            ["id", "-Gn"], check=True, capture_output=True, text=True
        ).stdout.strip().split(),
        "capture_binary": str(DUMPCAP),
        "capture_binary_sha256": file_sha256(DUMPCAP),
        "capture_binary_version": version,
        "capabilities": capabilities,
        "interfaces": interfaces,
        "capture_interface": "lo",
        "sudo_used": False,
        "system_permissions_modified": False,
    }


def main() -> int:
    raw_path = ROOT / RAW_RELATIVE
    if OUTPUT.exists() or raw_path.exists():
        raise SystemExit("Refusing to overwrite P13D Session B")
    if file_sha256(
        ROOT / "results/portable_schema/p13c/attempt_002/p13c_gate.json"
    ) != P13C_GATE_SHA256:
        raise RuntimeError("frozen P13C gate hash mismatch")
    implementation = verify_implementation_binding()
    preflight = verify_capture_privilege()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT / "implementation_binding.json", implementation)
    write_json(OUTPUT / "capture_privilege_preflight.json", preflight)
    write_json(
        OUTPUT / "session_a_reference.json",
        {
            "artifact_type": "P13D_SESSION_A_REFERENCE",
            "p13c_freeze_commit": P13C_FREEZE_COMMIT,
            "gate_path": "results/portable_schema/p13c/attempt_002/p13c_gate.json",
            "gate_sha256": P13C_GATE_SHA256,
            "raw_path": "data/controlled_capture/p13c/attempt_002/session_a_raw.pcap",
            "raw_sha256": SESSION_A_RAW_SHA256,
            "raw_size_bytes": 77_284,
            "raw_read_only": True,
            "schedule_sha256": SESSION_A_SCHEDULE_SHA256,
            "semantic_content_sha256": SESSION_A_SEMANTIC_SHA256,
            "packet_count": 793,
            "flow_count": 80,
            "canonical_identity_count": 80,
            "normal_rows": 12,
            "attack_rows": 68,
            "unmatched_rows": 0,
            "ambiguous_rows": 0,
            "mixed_identity_count": 0,
        },
    )

    base_wall_ns = time.time_ns() + 6_000_000_000
    base_mono_ns = time.monotonic_ns() + 6_000_000_000
    definitions = [
        ("session_b_normal_icmp", "NORMAL_ICMP", "NORMAL", 0.0, 2.4, 1, None,
         "eight loopback ICMP echoes for packet-level protocol coverage"),
        ("session_b_normal_http", "NORMAL_HTTP", "NORMAL", 3.0, 1.8, 6,
         traffic.PORTS["http"], "six ordinary local HTTP requests"),
        ("session_b_normal_udp", "NORMAL_UDP_ECHO", "NORMAL", 5.5, 2.0, 17,
         traffic.PORTS["udp"], "twelve independent UDP five-tuples"),
        ("session_b_normal_tcp_short", "NORMAL_TCP_SHORT", "NORMAL", 8.2, 1.8, 6,
         traffic.PORTS["echo_short"], "eight short local TCP echo connections"),
        ("session_b_normal_tcp_long", "NORMAL_TCP_LONG", "NORMAL", 10.7, 3.0, 6,
         traffic.PORTS["echo_long"], "one bounded five-message TCP session"),
        ("session_b_attack_scan", "ATTACK_PORT_SCAN", "ATTACK", 14.5, 2.0, 6, None,
         "one connect attempt to each local port 18080-18095"),
        ("session_b_attack_http_burst", "ATTACK_HTTP_BURST", "ATTACK", 17.3, 2.8, 6,
         traffic.PORTS["http"], "thirty bounded local HTTP requests"),
        ("session_b_attack_auth_failure", "ATTACK_AUTH_FAILURE", "ATTACK", 20.9, 2.2, 6,
         traffic.PORTS["auth"], "ten failed attempts against disposable auth service"),
    ]
    scenarios = []
    for scenario_id, scenario_type, label, offset, duration, protocol, port, notes in definitions:
        start_ns = base_wall_ns + int(offset * 1_000_000_000)
        end_ns = start_ns + int(duration * 1_000_000_000)
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_type": scenario_type,
                "portable_label": label,
                "generator_host": traffic.GENERATOR,
                "target_host": traffic.TARGET,
                "ip_protocol": protocol,
                "destination_port": port,
                "expected_start_ns": start_ns,
                "expected_start_utc": utc_iso(start_ns),
                "expected_end_ns": end_ns,
                "expected_end_utc": utc_iso(end_ns),
                "notes": notes,
            }
        )
    schedule = {
        "artifact_type": "P13D_SESSION_B_SCHEDULE",
        "session": "SESSION_B",
        "frozen_before_capture": True,
        "created_at_utc": utc_iso(time.time_ns()),
        "timezone": "UTC",
        "join_basis": (
            "inclusive flow start time + generator/target + protocol + optional "
            "destination port"
        ),
        "protocol_coverage_intent": (
            "more ICMP packets and independent UDP five-tuples; no balancing or model feedback"
        ),
        "scenarios": scenarios,
    }
    schedule_path = OUTPUT / "session_b_schedule.json"
    write_json(schedule_path, schedule)
    schedule_hash = file_sha256(schedule_path)
    if schedule_hash == SESSION_A_SCHEDULE_SHA256:
        raise RuntimeError("Session B schedule is not independent")

    capture_filter = (
        f"((src host {traffic.GENERATOR} and dst host {traffic.TARGET}) or "
        f"(src host {traffic.TARGET} and dst host {traffic.GENERATOR}))"
    )
    command = [
        str(DUMPCAP), "-i", "lo", "-f", capture_filter, "-s", "0", "-p",
        "-F", "pcap", "--time-stamp-type", "host", "-w", str(raw_path),
    ]
    servers, threads, udp_server = traffic.start_servers()
    capture_start_ns = time.time_ns()
    capture = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    execution: list[dict[str, object]] = []
    diagnostics = ""
    try:
        time.sleep(1.0)
        if capture.poll() is not None:
            raise RuntimeError(f"dumpcap exited before traffic: {capture.stderr.read()}")
        actions = {
            "session_b_normal_icmp": lambda: subprocess.run(
                ["ping", "-I", traffic.GENERATOR, "-c", "8", "-i", "0.2", traffic.TARGET],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            ),
            "session_b_normal_http": lambda: traffic.http_requests(6),
            "session_b_normal_udp": lambda: udp_independent_flows(12),
            "session_b_normal_tcp_short": lambda: [
                traffic.tcp_echo(
                    traffic.PORTS["echo_short"], [f"session-b-short-{index}".encode()]
                )
                for index in range(8)
            ],
            "session_b_normal_tcp_long": lambda: traffic.tcp_echo(
                traffic.PORTS["echo_long"],
                [b"session-b-a", b"session-b-b", b"session-b-c", b"session-b-d", b"session-b-e"],
                0.35,
            ),
            "session_b_attack_scan": traffic.port_scan,
            "session_b_attack_http_burst": lambda: traffic.http_requests(30),
            "session_b_attack_auth_failure": lambda: traffic.auth_failures(10),
        }
        for scenario, definition in zip(scenarios, definitions, strict=True):
            sleep_until(base_mono_ns + int(definition[3] * 1_000_000_000))
            observed_start = time.time_ns()
            error = None
            try:
                actions[scenario["scenario_id"]]()
                result = "SUCCESS"
            except Exception as exc:  # noqa: BLE001 - record without substitution
                result = "FAILED"
                error = f"{type(exc).__name__}: {exc}"
            observed_end = time.time_ns()
            execution.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario_type": scenario["scenario_type"],
                    "portable_label": scenario["portable_label"],
                    "generator": scenario["generator_host"],
                    "target": scenario["target_host"],
                    "ip_protocol": scenario["ip_protocol"],
                    "destination_port": scenario["destination_port"],
                    "execution_count": 1,
                    "status": result,
                    "error": error,
                    "observed_start_ns": observed_start,
                    "observed_start_utc": utc_iso(observed_start),
                    "observed_end_ns": observed_end,
                    "observed_end_utc": utc_iso(observed_end),
                    "within_predeclared_interval": (
                        observed_start >= scenario["expected_start_ns"]
                        and observed_end <= scenario["expected_end_ns"]
                    ),
                }
            )
        time.sleep(1.0)
    finally:
        if capture.poll() is None:
            capture.send_signal(signal.SIGINT)
        diagnostics = capture.communicate(timeout=10)[1]
        traffic.stop_servers(servers, threads, udp_server)
    capture_end_ns = time.time_ns()
    if capture.returncode != 0:
        raise RuntimeError(f"dumpcap failed with {capture.returncode}: {diagnostics}")

    write_json(
        OUTPUT / "session_b_execution_log.json",
        {
            "artifact_type": "P13D_SESSION_B_EXECUTION_LOG",
            "schedule_sha256": schedule_hash,
            "capture_started_before_scenario_traffic": bool(
                execution and capture_start_ns < int(execution[0]["observed_start_ns"])
            ),
            "all_scenarios_executed_once": all(item["execution_count"] == 1 for item in execution),
            "all_scenarios_succeeded": all(item["status"] == "SUCCESS" for item in execution),
            "all_scenarios_within_predeclared_interval": all(
                item["within_predeclared_interval"] for item in execution
            ),
            "events": execution,
        },
    )
    write_json(
        OUTPUT / "session_b_capture_environment.json",
        {
            "artifact_type": "P13D_SESSION_B_CAPTURE_ENVIRONMENT",
            "capture_binary": str(DUMPCAP),
            "capture_binary_sha256": file_sha256(DUMPCAP),
            "capture_binary_version": preflight["capture_binary_version"],
            "command": command,
            "interface": "lo",
            "capture_format_requested": "classic_pcap",
            "snaplen_requested": 0,
            "timestamp_type": "host",
            "capture_filter": capture_filter,
            "capture_start_ns": capture_start_ns,
            "capture_start_utc": utc_iso(capture_start_ns),
            "capture_end_ns": capture_end_ns,
            "capture_end_utc": utc_iso(capture_end_ns),
            "diagnostics": diagnostics.splitlines(),
            "sudo_used": False,
        },
    )

    if not raw_path.is_file() or raw_path.stat().st_size <= 0:
        raise RuntimeError("Session B raw capture is missing or empty")
    parsed = read_classic_pcap(raw_path)
    raw_hash = file_sha256(raw_path)
    if raw_hash == SESSION_A_RAW_SHA256:
        raise RuntimeError("Session B raw hash equals Session A")
    captured_match = re.search(r"Packets captured: (\d+)", diagnostics)
    dropped_match = re.search(r"received/dropped.*: (\d+)/(\d+)", diagnostics)
    captured_count = int(captured_match.group(1)) if captured_match else None
    drop_count = int(dropped_match.group(2)) if dropped_match else None
    raw_size = raw_path.stat().st_size
    raw_path.chmod(0o444)
    raw_entry = {
        "filename": raw_path.name,
        "path": str(RAW_RELATIVE),
        "size_bytes": raw_size,
        "sha256": raw_hash,
        "packet_count": parsed.audit["packet_records"],
        "dumpcap_reported_packet_count": captured_count,
        "drop_count": drop_count,
        "drop_count_source": "dumpcap final received/dropped diagnostic",
        "packet_capture_start_ns": parsed.audit["first_timestamp_ns"],
        "packet_capture_end_ns": parsed.audit["last_timestamp_ns"],
        "capture_start_utc": utc_iso(capture_start_ns),
        "capture_end_utc": utc_iso(capture_end_ns),
        "dumpcap_version": preflight["capture_binary_version"][0],
        "dumpcap_binary_sha256": file_sha256(DUMPCAP),
        "capture_interface": "lo",
        "capture_format": parsed.audit["capture_format"],
        "link_type": parsed.audit["link_type_name"],
        "timestamp_precision": parsed.audit["timestamp_precision"],
        "schedule_sha256": schedule_hash,
    }
    reservation = {
        "artifact_type": "P13D_SESSION_B_RAW_RESERVATION",
        "reservation_status": "P13D_SESSION_B_RAW_RESERVED",
        "reserved_at_utc": utc_iso(time.time_ns()),
        "raw_files": [raw_entry],
        "aggregate_session_sha256": aggregate_corpus_sha256([raw_entry]),
        "aggregate_method": "SHA-256 of sorted filename\\0size_bytes\\0file_sha256\\n records",
        "schedule_sha256": schedule_hash,
        "raw_files_read_only": True,
        "post_reservation_editing_forbidden": True,
        "session_a_raw_sha256": SESSION_A_RAW_SHA256,
        "raw_hashes_distinct": True,
        "session_a_schedule_sha256": SESSION_A_SCHEDULE_SHA256,
        "schedule_hashes_distinct": True,
    }
    write_json(OUTPUT / "session_b_raw_reservation.json", reservation)
    print("P13D_SESSION_B_RAW_RESERVED")
    print(f"p13c_freeze_commit={P13C_FREEZE_COMMIT}")
    print(f"schedule_sha256={schedule_hash}")
    print(f"raw_sha256={raw_hash}")
    print(f"aggregate_sha256={reservation['aggregate_session_sha256']}")
    print(f"packet_records={parsed.audit['packet_records']}")
    print(f"drop_count={drop_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
