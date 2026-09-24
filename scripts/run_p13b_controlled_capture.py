#!/usr/bin/env python3
"""Create one small isolated P13B loopback capture and reserve it immediately."""

from __future__ import annotations

from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    CORPUS_NAME, VERSION, aggregate_corpus_sha256, file_sha256, pretty_json_bytes,
)


GENERATOR = "127.0.0.2"
TARGET = "127.0.0.3"
RAW_RELATIVE = Path("data/controlled_capture/p13b/session_a_raw.pcap")
OUTPUT_RELATIVE = Path("results/portable_schema/p13b")
PORTS = {"http": 18080, "echo_short": 18081, "echo_long": 18082, "auth": 18083, "udp": 18053}


class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


class QuietHTTPHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):  # noqa: N802
        body = b"p13b-controlled-response\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


class EchoHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(3)
        while True:
            try:
                data = self.request.recv(4096)
            except TimeoutError:
                return
            if not data:
                return
            self.request.sendall(data)


class AuthHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(1)
        try:
            self.request.recv(256)
            self.request.sendall(b"AUTH_DENIED\n")
        except (TimeoutError, ConnectionError):
            return


class UDPEchoThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((TARGET, PORTS["udp"]))
        self.sock.settimeout(0.2)

    def run(self):
        while not self.stop_event.is_set():
            try:
                data, address = self.sock.recvfrom(4096)
            except TimeoutError:
                continue
            self.sock.sendto(data, address)

    def close(self):
        self.stop_event.set()
        self.join(timeout=2)
        self.sock.close()


def utc_iso(timestamp_ns: int) -> str:
    return datetime.fromtimestamp(timestamp_ns / 1_000_000_000, timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pretty_json_bytes(value))


def sleep_until(target_monotonic_ns: int) -> None:
    while True:
        remaining = (target_monotonic_ns - time.monotonic_ns()) / 1_000_000_000
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def http_requests(count: int) -> None:
    for index in range(count):
        connection = http.client.HTTPConnection(
            TARGET, PORTS["http"], timeout=2, source_address=(GENERATOR, 0)
        )
        connection.request("GET", f"/scenario/{index}")
        response = connection.getresponse()
        response.read()
        if response.status != 200:
            raise RuntimeError("controlled HTTP server returned non-200")
        connection.close()


def tcp_echo(port: int, messages: list[bytes], pause: float = 0.0) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.bind((GENERATOR, 0))
        client.connect((TARGET, port))
        for message in messages:
            client.sendall(message)
            received = client.recv(len(message))
            if received != message:
                raise RuntimeError("controlled echo mismatch")
            if pause:
                time.sleep(pause)


def udp_echo(count: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client:
        client.settimeout(2)
        client.bind((GENERATOR, 0))
        for index in range(count):
            payload = f"p13b-udp-{index}".encode()
            client.sendto(payload, (TARGET, PORTS["udp"]))
            response, _ = client.recvfrom(4096)
            if response != payload:
                raise RuntimeError("controlled UDP echo mismatch")


def port_scan() -> None:
    for port in range(18080, 18096):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(0.15)
            client.bind((GENERATOR, 0))
            try:
                client.connect((TARGET, port))
            except (ConnectionRefusedError, TimeoutError, OSError):
                pass


def auth_failures(count: int) -> None:
    for index in range(count):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
            client.settimeout(2)
            client.bind((GENERATOR, 0))
            client.connect((TARGET, PORTS["auth"]))
            client.sendall(f"user lab bad-password-{index}\n".encode())
            client.recv(256)


def start_servers():
    servers = [
        ThreadingHTTPServer((TARGET, PORTS["http"]), QuietHTTPHandler),
        ReusableTCPServer((TARGET, PORTS["echo_short"]), EchoHandler),
        ReusableTCPServer((TARGET, PORTS["echo_long"]), EchoHandler),
        ReusableTCPServer((TARGET, PORTS["auth"]), AuthHandler),
    ]
    threads = []
    for server in servers:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    udp_server = UDPEchoThread()
    udp_server.start()
    return servers, threads, udp_server


def stop_servers(servers, threads, udp_server):
    for server in servers:
        server.shutdown()
        server.server_close()
    for thread in threads:
        thread.join(timeout=2)
    udp_server.close()


def main() -> int:
    raw_path = ROOT / RAW_RELATIVE
    output = ROOT / OUTPUT_RELATIVE
    reservation_path = output / "raw_source_reservation.json"
    if raw_path.exists() or reservation_path.exists():
        raise SystemExit("Refusing to overwrite an existing P13B raw capture or reservation")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)

    base_wall_ns = time.time_ns() + 3_000_000_000
    base_mono_ns = time.monotonic_ns() + 3_000_000_000
    definitions = [
        ("normal_icmp", "NORMAL_ICMP", "NORMAL", 0.0, 1.2, 1, None, "three local ICMP echoes"),
        ("normal_http", "NORMAL_HTTP", "NORMAL", 2.0, 1.2, 6, PORTS["http"], "four ordinary local HTTP requests"),
        ("normal_udp", "NORMAL_UDP_ECHO", "NORMAL", 4.0, 1.2, 17, PORTS["udp"], "five local UDP request/response exchanges"),
        ("normal_tcp_short", "NORMAL_TCP_SHORT", "NORMAL", 6.0, 1.2, 6, PORTS["echo_short"], "five short local TCP echo connections"),
        ("normal_tcp_long", "NORMAL_TCP_LONG", "NORMAL", 8.0, 2.2, 6, PORTS["echo_long"], "one bounded multi-message TCP session"),
        ("attack_scan", "ATTACK_PORT_SCAN", "ATTACK", 11.0, 1.5, 6, None, "one connect attempt to each local port 18080-18095"),
        ("attack_http_burst", "ATTACK_HTTP_BURST", "ATTACK", 13.5, 2.2, 6, PORTS["http"], "forty bounded local HTTP requests"),
        ("attack_auth_failure", "ATTACK_AUTH_FAILURE", "ATTACK", 16.5, 1.8, 6, PORTS["auth"], "twelve failed attempts against disposable local auth service"),
    ]
    scenarios = []
    for scenario_id, scenario_type, label, offset, duration, protocol, port, notes in definitions:
        start_ns = base_wall_ns + int(offset * 1_000_000_000)
        end_ns = start_ns + int(duration * 1_000_000_000)
        scenarios.append({
            "scenario_id": scenario_id,
            "scenario_type": scenario_type,
            "portable_label": label,
            "generator_host": GENERATOR,
            "target_host": TARGET,
            "ip_protocol": protocol,
            "destination_port": port,
            "expected_start_ns": start_ns,
            "expected_start_utc": utc_iso(start_ns),
            "expected_end_ns": end_ns,
            "expected_end_utc": utc_iso(end_ns),
            "notes": notes,
        })
    schedule = {
        "artifact_type": "P13B_SCENARIO_SCHEDULE",
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "frozen_before_capture": True,
        "created_at_utc": utc_iso(time.time_ns()),
        "timezone": "UTC",
        "join_basis": "inclusive flow start time + generator/target + protocol + optional destination port",
        "scenarios": scenarios,
    }
    schedule_path = output / "scenario_schedule.json"
    write_json(schedule_path, schedule)
    schedule_hash = file_sha256(schedule_path)

    topology = {
        "artifact_type": "P13B_LAB_TOPOLOGY",
        "corpus_name": CORPUS_NAME,
        "isolation": "Linux loopback only; no route to a public target is used by any scenario",
        "ownership": "single local researcher-controlled host and disposable local services",
        "generator": {"address": GENERATOR, "role": "controlled traffic generator"},
        "target": {"address": TARGET, "role": "controlled disposable services", "ports": PORTS},
        "capture_point": {"interface": "lo", "filter_endpoints": [GENERATOR, TARGET]},
        "external_destinations": [],
    }
    write_json(output / "lab_topology.json", topology)

    tcpdump_version = subprocess.run(
        ["tcpdump", "--version"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    capture_filter = (
        f"((src host {GENERATOR} and dst host {TARGET}) or "
        f"(src host {TARGET} and dst host {GENERATOR}))"
    )
    user = os.environ.get("USER", "nobody")
    command = [
        "tcpdump", "-Z", user, "-i", "lo", "-nn", "-s", "0",
        "--time-stamp-precision=nano", "-U", "-w", str(raw_path), capture_filter,
    ]
    environment = {
        "artifact_type": "P13B_CAPTURE_ENVIRONMENT",
        "corpus_name": CORPUS_NAME,
        "capture_tool": tcpdump_version[0],
        "libpcap": tcpdump_version[1] if len(tcpdump_version) > 1 else None,
        "command": command,
        "interface": "lo",
        "snaplen": 0,
        "requested_timestamp_precision": "nanoseconds",
        "capture_filter": capture_filter,
        "host_timezone": time.tzname,
        "clock_source": "single-host CLOCK_REALTIME/CLOCK_MONOTONIC; no cross-host offset",
        "generator_target_clock_offset_ns": 0,
        "privacy_boundary": "only packets between 127.0.0.2 and 127.0.0.3",
    }
    write_json(output / "capture_environment.json", environment)

    servers, threads, udp_server = start_servers()
    capture = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    execution = []
    try:
        time.sleep(0.8)
        if capture.poll() is not None:
            raise RuntimeError(f"tcpdump exited before traffic generation: {capture.stderr.read()}")
        actions = {
            "normal_icmp": lambda: subprocess.run(
                ["ping", "-I", GENERATOR, "-c", "3", "-i", "0.2", TARGET],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            ),
            "normal_http": lambda: http_requests(4),
            "normal_udp": lambda: udp_echo(5),
            "normal_tcp_short": lambda: [tcp_echo(PORTS["echo_short"], [f"short-{i}".encode()]) for i in range(5)],
            "normal_tcp_long": lambda: tcp_echo(PORTS["echo_long"], [b"long-a", b"long-b", b"long-c"], 0.35),
            "attack_scan": port_scan,
            "attack_http_burst": lambda: http_requests(40),
            "attack_auth_failure": lambda: auth_failures(12),
        }
        for scenario, definition in zip(scenarios, definitions, strict=True):
            offset = definition[3]
            sleep_until(base_mono_ns + int(offset * 1_000_000_000))
            observed_start = time.time_ns()
            actions[scenario["scenario_id"]]()
            observed_end = time.time_ns()
            execution.append({
                "scenario_id": scenario["scenario_id"],
                "observed_start_ns": observed_start,
                "observed_start_utc": utc_iso(observed_start),
                "observed_end_ns": observed_end,
                "observed_end_utc": utc_iso(observed_end),
                "within_predeclared_interval": (
                    observed_start >= scenario["expected_start_ns"] and observed_end <= scenario["expected_end_ns"]
                ),
            })
        time.sleep(1.0)
    finally:
        if capture.poll() is None:
            capture.send_signal(signal.SIGINT)
            capture.wait(timeout=10)
        stop_servers(servers, threads, udp_server)
    if capture.returncode not in {0, 130}:
        raise RuntimeError(f"tcpdump failed with {capture.returncode}: {capture.stderr.read()}")
    if not all(item["within_predeclared_interval"] for item in execution):
        raise RuntimeError("one or more scenarios exceeded its predeclared schedule")
    write_json(output / "scenario_execution_log.json", {
        "artifact_type": "P13B_SCENARIO_EXECUTION_LOG",
        "schedule_sha256": schedule_hash,
        "events": execution,
    })

    parsed = read_classic_pcap(raw_path)
    raw_hash = file_sha256(raw_path)
    raw_size = raw_path.stat().st_size
    raw_path.chmod(0o444)
    raw_entry = {
        "filename": raw_path.name,
        "path": str(RAW_RELATIVE),
        "size_bytes": raw_size,
        "sha256": raw_hash,
        "packet_capture_start_ns": parsed.audit["first_timestamp_ns"],
        "packet_capture_end_ns": parsed.audit["last_timestamp_ns"],
        "capture_tool": tcpdump_version[0],
        "capture_interface": "lo",
        "scenario_schedule_sha256": schedule_hash,
    }
    reservation = {
        "artifact_type": "P13B_RAW_SOURCE_RESERVATION",
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "reservation_status": "RAW_RESERVED",
        "reserved_at_utc": utc_iso(time.time_ns()),
        "raw_files": [raw_entry],
        "aggregate_corpus_sha256": aggregate_corpus_sha256([raw_entry]),
        "aggregate_method": "SHA-256 of sorted filename\\0size_bytes\\0file_sha256\\n records",
        "scenario_schedule_sha256": schedule_hash,
        "raw_files_read_only": True,
        "post_reservation_editing_forbidden": True,
    }
    write_json(reservation_path, reservation)
    print("P13B_RAW_RESERVED")
    print(f"raw_sha256={raw_hash}")
    print(f"aggregate_sha256={reservation['aggregate_corpus_sha256']}")
    print(f"packet_records={parsed.audit['packet_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
