#!/usr/bin/env python3
"""Run one isolated P13C attempt-002 capture and reserve it before extraction."""

from __future__ import annotations

from datetime import datetime, timezone
import grp
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    aggregate_corpus_sha256,
    file_sha256,
    pretty_json_bytes,
)
import run_p13b_controlled_capture as traffic  # noqa: E402


CORPUS_NAME = "P13C_CONTROLLED_CAPTURE"
VERSION = "p13c-controlled-capture-attempt-002-v1"
ATTEMPT_ID = "P13C_ATTEMPT_002"
OUTPUT_RELATIVE = Path("results/portable_schema/p13c/attempt_002")
RAW_RELATIVE = Path("data/controlled_capture/p13c/attempt_002/session_a_raw.pcap")
DUMPCAP = Path("/usr/bin/dumpcap")
P13B_BLOCKED_GATE_SHA256 = "be8f2213474592c69b34636a089371a35664acc6d5775d9a148cdc51a39bee95"
P13C_BLOCKED_GATE_SHA256 = "f70e98af206f7c037975b2f97d3215e891645ee97fff5439022ff16c4d179a69"


def utc_iso(timestamp_ns: int) -> str:
    return (
        datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def sleep_until(target_monotonic_ns: int) -> None:
    while True:
        remaining = (target_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def verify_previous_attempts() -> None:
    expected = {
        ROOT / "results/portable_schema/p13b/p13b_gate.json": P13B_BLOCKED_GATE_SHA256,
        ROOT / "results/portable_schema/p13c/p13c_gate.json": P13C_BLOCKED_GATE_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise RuntimeError(f"previous blocked-attempt gate changed: {path}")


def verify_capture_privilege() -> dict[str, object]:
    wireshark_gid = grp.getgrnam("wireshark").gr_gid
    active_groups = {os.getgid(), *os.getgroups()}
    if os.geteuid() == 0:
        raise RuntimeError("scientific capture must not run as root")
    if wireshark_gid not in active_groups:
        raise RuntimeError("wireshark group is not active in this process")
    capabilities = subprocess.run(
        ["getcap", str(DUMPCAP)], check=True, capture_output=True, text=True
    ).stdout.strip()
    if "cap_net_raw" not in capabilities or "cap_net_admin" not in capabilities:
        raise RuntimeError("dumpcap capture capabilities are incomplete")
    interfaces = subprocess.run(
        [str(DUMPCAP), "-D"], check=True, capture_output=True, text=True
    ).stdout
    if not any(line.strip().endswith("lo (Loopback)") for line in interfaces.splitlines()):
        raise RuntimeError("loopback interface is not available to dumpcap")
    version = subprocess.run(
        [str(DUMPCAP), "--version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return {
        "artifact_type": "P13C_CAPTURE_PRIVILEGE_PREFLIGHT",
        "attempt_id": ATTEMPT_ID,
        "status": "PASS",
        "verified_at_utc": utc_iso(time.time_ns()),
        "current_user": os.environ.get("USER", "nnam09"),
        "effective_uid": os.geteuid(),
        "active_primary_gid": os.getgid(),
        "active_group_names": subprocess.run(
            ["id", "-Gn"], check=True, capture_output=True, text=True
        ).stdout.strip().split(),
        "capture_binary": str(DUMPCAP),
        "capture_binary_sha256": file_sha256(DUMPCAP),
        "capabilities": capabilities,
        "capture_binary_version": version,
        "interface_inventory": interfaces.splitlines(),
        "capture_interface": "lo",
        "no_sudo_used": True,
        "system_permissions_modified": False,
    }


def main() -> int:
    verify_previous_attempts()
    output = ROOT / OUTPUT_RELATIVE
    raw_path = ROOT / RAW_RELATIVE
    reservation_path = output / "raw_source_reservation.json"
    if output.exists() or raw_path.exists() or reservation_path.exists():
        raise SystemExit("Refusing to overwrite P13C attempt 002")

    preflight = verify_capture_privilege()
    output.mkdir(parents=True, exist_ok=False)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output / "capture_privilege_preflight.json", preflight)
    write_json(
        output / "previous_attempt_reference.json",
        {
            "artifact_type": "P13C_PREVIOUS_ATTEMPT_REFERENCE",
            "attempt_id": ATTEMPT_ID,
            "references": [
                {
                    "attempt": "P13B_ATTEMPT_001_CAPTURE_PRIVILEGE_BLOCKED",
                    "gate_path": "results/portable_schema/p13b/p13b_gate.json",
                    "gate_sha256": P13B_BLOCKED_GATE_SHA256,
                },
                {
                    "attempt": "P13C_CAPTURE_PRECONDITION_BLOCKED",
                    "gate_path": "results/portable_schema/p13c/p13c_gate.json",
                    "gate_sha256": P13C_BLOCKED_GATE_SHA256,
                },
            ],
            "preserved_unchanged": True,
        },
    )

    base_wall_ns = time.time_ns() + 6_000_000_000
    base_mono_ns = time.monotonic_ns() + 6_000_000_000
    definitions = [
        ("a002_normal_icmp", "NORMAL_ICMP", "NORMAL", 0.0, 1.5, 1, None,
         "three local ICMP echoes"),
        ("a002_normal_http", "NORMAL_HTTP", "NORMAL", 2.2, 1.5, 6,
         traffic.PORTS["http"], "four ordinary local HTTP requests"),
        ("a002_normal_udp", "NORMAL_UDP_ECHO", "NORMAL", 4.4, 1.5, 17,
         traffic.PORTS["udp"], "five local UDP request/response exchanges"),
        ("a002_normal_tcp_short", "NORMAL_TCP_SHORT", "NORMAL", 6.6, 1.5, 6,
         traffic.PORTS["echo_short"], "five short local TCP echo connections"),
        ("a002_normal_tcp_long", "NORMAL_TCP_LONG", "NORMAL", 8.8, 2.5, 6,
         traffic.PORTS["echo_long"], "one bounded multi-message TCP session"),
        ("a002_attack_scan", "ATTACK_PORT_SCAN", "ATTACK", 12.0, 1.8, 6, None,
         "one connect attempt to each local port 18080-18095"),
        ("a002_attack_http_burst", "ATTACK_HTTP_BURST", "ATTACK", 14.8, 2.8, 6,
         traffic.PORTS["http"], "forty bounded local HTTP requests"),
        ("a002_attack_auth_failure", "ATTACK_AUTH_FAILURE", "ATTACK", 18.4, 2.2, 6,
         traffic.PORTS["auth"], "twelve failed attempts against disposable auth service"),
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
        "artifact_type": "P13C_SCENARIO_SCHEDULE",
        "attempt_id": ATTEMPT_ID,
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "frozen_before_capture": True,
        "created_at_utc": utc_iso(time.time_ns()),
        "timezone": "UTC",
        "join_basis": (
            "inclusive flow start time + generator/target + protocol + optional "
            "destination port"
        ),
        "scenarios": scenarios,
    }
    schedule_path = output / "scenario_schedule.json"
    write_json(schedule_path, schedule)
    schedule_hash = file_sha256(schedule_path)
    if schedule_hash == "625374f24e04caf809a3021ac025e9fe549fa26bb3ebc7b097e963db942329de":
        raise RuntimeError("new schedule unexpectedly reused the blocked P13B hash")

    write_json(
        output / "lab_topology.json",
        {
            "artifact_type": "P13C_LAB_TOPOLOGY",
            "attempt_id": ATTEMPT_ID,
            "isolation": "Linux loopback only",
            "generator": {"address": traffic.GENERATOR, "role": "traffic generator"},
            "target": {
                "address": traffic.TARGET,
                "role": "disposable services",
                "ports": traffic.PORTS,
            },
            "capture_point": {"interface": "lo", "link_type": "EN10MB"},
            "allowed_endpoints": [traffic.GENERATOR, traffic.TARGET],
            "external_destinations": [],
        },
    )

    capture_filter = (
        f"((src host {traffic.GENERATOR} and dst host {traffic.TARGET}) or "
        f"(src host {traffic.TARGET} and dst host {traffic.GENERATOR}))"
    )
    command = [
        str(DUMPCAP), "-i", "lo", "-f", capture_filter, "-s", "0", "-p", "-P",
        "--time-stamp-type", "host", "-w", str(raw_path),
    ]
    capture_start_ns = time.time_ns()
    servers, threads, udp_server = traffic.start_servers()
    capture = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    execution: list[dict[str, object]] = []
    capture_diagnostics = ""
    try:
        time.sleep(1.0)
        if capture.poll() is not None:
            raise RuntimeError(
                f"dumpcap exited before traffic generation: {capture.stderr.read()}"
            )
        actions = {
            "a002_normal_icmp": lambda: subprocess.run(
                [
                    "ping", "-I", traffic.GENERATOR, "-c", "3", "-i", "0.2",
                    traffic.TARGET,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            ),
            "a002_normal_http": lambda: traffic.http_requests(4),
            "a002_normal_udp": lambda: traffic.udp_echo(5),
            "a002_normal_tcp_short": lambda: [
                traffic.tcp_echo(
                    traffic.PORTS["echo_short"], [f"short-{index}".encode()]
                )
                for index in range(5)
            ],
            "a002_normal_tcp_long": lambda: traffic.tcp_echo(
                traffic.PORTS["echo_long"], [b"long-a", b"long-b", b"long-c"], 0.35
            ),
            "a002_attack_scan": traffic.port_scan,
            "a002_attack_http_burst": lambda: traffic.http_requests(40),
            "a002_attack_auth_failure": lambda: traffic.auth_failures(12),
        }
        for scenario, definition in zip(scenarios, definitions, strict=True):
            sleep_until(base_mono_ns + int(definition[3] * 1_000_000_000))
            observed_start = time.time_ns()
            error = None
            try:
                actions[scenario["scenario_id"]]()
                scenario_status = "SUCCESS"
            except Exception as exc:  # noqa: BLE001 - record; never substitute
                scenario_status = "FAILED"
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
                    "status": scenario_status,
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
        capture_diagnostics = capture.communicate(timeout=10)[1]
        traffic.stop_servers(servers, threads, udp_server)
    capture_end_ns = time.time_ns()
    if capture.returncode != 0:
        raise RuntimeError(
            f"dumpcap failed with {capture.returncode}: {capture_diagnostics}"
        )
    write_json(
        output / "scenario_execution_log.json",
        {
            "artifact_type": "P13C_SCENARIO_EXECUTION_LOG",
            "attempt_id": ATTEMPT_ID,
            "schedule_sha256": schedule_hash,
            "capture_started_before_scenario_traffic": bool(
                execution and capture_start_ns < int(execution[0]["observed_start_ns"])
            ),
            "all_scenarios_executed_once": all(
                item["execution_count"] == 1 for item in execution
            ),
            "all_scenarios_succeeded": all(
                item["status"] == "SUCCESS" for item in execution
            ),
            "all_scenarios_within_predeclared_interval": all(
                item["within_predeclared_interval"] for item in execution
            ),
            "events": execution,
        },
    )
    write_json(
        output / "capture_environment.json",
        {
            "artifact_type": "P13C_CAPTURE_ENVIRONMENT",
            "attempt_id": ATTEMPT_ID,
            "capture_binary": str(DUMPCAP),
            "capture_binary_sha256": file_sha256(DUMPCAP),
            "capture_binary_version": preflight["capture_binary_version"],
            "command": command,
            "interface": "lo",
            "capture_format_requested": "classic_pcap",
            "link_type_requested": "EN10MB",
            "snaplen": 0,
            "timestamp_type": "host",
            "capture_filter": capture_filter,
            "capture_start_ns": capture_start_ns,
            "capture_start_utc": utc_iso(capture_start_ns),
            "capture_end_ns": capture_end_ns,
            "capture_end_utc": utc_iso(capture_end_ns),
            "diagnostics": capture_diagnostics.splitlines(),
            "scientific_pipeline_running_as_root": False,
            "sudo_used": False,
        },
    )

    if not raw_path.is_file() or raw_path.stat().st_size <= 0:
        raise RuntimeError("dumpcap produced no non-empty raw PCAP")
    parsed = read_classic_pcap(raw_path)
    if parsed.audit["packet_records"] <= 0:
        raise RuntimeError("raw PCAP contains no packet records")
    raw_hash = file_sha256(raw_path)
    raw_size = raw_path.stat().st_size
    raw_path.chmod(0o444)
    raw_entry = {
        "filename": raw_path.name,
        "path": str(RAW_RELATIVE),
        "size_bytes": raw_size,
        "sha256": raw_hash,
        "packet_count": parsed.audit["packet_records"],
        "packet_capture_start_ns": parsed.audit["first_timestamp_ns"],
        "packet_capture_end_ns": parsed.audit["last_timestamp_ns"],
        "capture_tool": preflight["capture_binary_version"][0],
        "capture_interface": "lo",
        "capture_format": parsed.audit["capture_format"],
        "link_type": parsed.audit["link_type_name"],
        "timestamp_precision": parsed.audit["timestamp_precision"],
        "scenario_schedule_sha256": schedule_hash,
    }
    reservation = {
        "artifact_type": "P13C_RAW_SOURCE_RESERVATION",
        "attempt_id": ATTEMPT_ID,
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "reservation_status": "P13C_RAW_SOURCE_RESERVED",
        "reserved_at_utc": utc_iso(time.time_ns()),
        "raw_files": [raw_entry],
        "aggregate_corpus_sha256": aggregate_corpus_sha256([raw_entry]),
        "aggregate_method": "SHA-256 of sorted filename\\0size_bytes\\0file_sha256\\n records",
        "scenario_schedule_sha256": schedule_hash,
        "raw_files_read_only": True,
        "post_reservation_editing_forbidden": True,
    }
    write_json(reservation_path, reservation)
    print("P13C_RAW_SOURCE_RESERVED")
    print(f"schedule_sha256={schedule_hash}")
    print(f"raw_sha256={raw_hash}")
    print(f"aggregate_sha256={reservation['aggregate_corpus_sha256']}")
    print(f"packet_records={parsed.audit['packet_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
