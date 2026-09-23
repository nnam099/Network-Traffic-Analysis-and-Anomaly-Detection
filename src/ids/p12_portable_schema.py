"""Portable feature-schema v3 design and deterministic reference formulas.

P12 is a new scientific branch.  It defines packet-observable semantics for a
future extractor and model; it does not adapt, open, or evaluate a P9 model.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import ipaddress
import json
import math
from pathlib import Path
import statistics
import struct
from typing import Iterable


VERSION = "portable-feature-schema-v3"
IDENTITY_VERSION = "canonical-flow-identity-v3"
STATE_VERSION = "portable-connection-state-v1"
TRANSPORT_EXTENSION_VERSION = "portable-transport-extension-v1"
IDLE_TIMEOUT_NS = 60_000_000_000
P8_PROTOCOL_SHA256 = "62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216"
P9_PROTOCOL_SHA256 = "c1c7352a9ca546a3e8e456b112d09fd42f29243e205a8760703b27f68b5baf87"

MISSING_STATES = (
    "ZERO_IS_VALID_ABSENCE",
    "NULL_NOT_APPLICABLE",
    "NULL_INSUFFICIENT_PACKETS",
    "NULL_EXTRACTION_FAILURE",
)

CONTINUOUS_FEATURES = (
    "duration_seconds",
    "forward_packet_count",
    "backward_packet_count",
    "total_packet_count",
    "forward_byte_count",
    "backward_byte_count",
    "total_byte_count",
    "forward_packet_fraction",
    "forward_byte_fraction",
    "mean_packet_size_bytes",
    "forward_mean_packet_size_bytes",
    "backward_mean_packet_size_bytes",
    "packets_per_second",
    "bytes_per_second",
    "forward_packets_per_second",
    "backward_packets_per_second",
    "forward_bytes_per_second",
    "backward_bytes_per_second",
)
CATEGORICAL_FEATURES = ("ip_protocol",)

TRANSPORT_EXTENSION_FIELDS = (
    "flow_iat_mean_seconds",
    "flow_iat_std_seconds",
    "forward_iat_mean_seconds",
    "forward_iat_std_seconds",
    "backward_iat_mean_seconds",
    "backward_iat_std_seconds",
    "tcp_fin_count",
    "tcp_syn_count",
    "tcp_rst_count",
    "tcp_psh_count",
    "tcp_ack_count",
    "tcp_urg_count",
    "tcp_ece_count",
    "tcp_cwr_count",
    "tcp_synack_seconds",
    "tcp_ack_after_synack_seconds",
    "tcp_handshake_seconds",
    "portable_connection_state",
)


def canonical_json_bytes(value: object, *, pretty: bool = False) -> bytes:
    """Serialize without NaN/Infinity and with stable key ordering."""

    def clean(item: object) -> object:
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("non-finite values are forbidden; use null with a reason")
            return 0.0 if item == 0.0 else item
        if isinstance(item, dict):
            return {str(key): clean(value) for key, value in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(value) for value in item]
        return item

    options = {"ensure_ascii": False, "sort_keys": True, "allow_nan": False}
    if pretty:
        return (json.dumps(clean(value), indent=2, **options) + "\n").encode()
    return (json.dumps(clean(value), separators=(",", ":"), **options) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PacketObservation:
    """P12 normalized packet observation; labels and payload are absent by design."""

    capture_index: int
    timestamp_ns: int
    ip_version: int
    src_address: str
    dst_address: str
    ip_protocol: int
    src_port: int
    dst_port: int
    ip_total_length: int
    tcp_flags: int | None = None

    def __post_init__(self) -> None:
        if self.capture_index < 0 or self.timestamp_ns < 0:
            raise ValueError("capture_index and timestamp_ns must be non-negative")
        source = ipaddress.ip_address(self.src_address)
        destination = ipaddress.ip_address(self.dst_address)
        if source.version != self.ip_version or destination.version != self.ip_version:
            raise ValueError("IP version/address mismatch")
        if self.ip_version not in {4, 6}:
            raise ValueError("only normalized IPv4 and IPv6 observations are allowed")
        if not 0 <= self.ip_protocol <= 255:
            raise ValueError("ip_protocol must be an unsigned 8-bit IANA number")
        if not 0 <= self.src_port <= 65535 or not 0 <= self.dst_port <= 65535:
            raise ValueError("ports must be unsigned 16-bit values")
        minimum = 20 if self.ip_version == 4 else 40
        if self.ip_total_length < minimum:
            raise ValueError("ip_total_length is smaller than the base IP header")
        if self.ip_protocol == 6:
            if self.tcp_flags is None or not 0 <= self.tcp_flags <= 255:
                raise ValueError("TCP observations require an unsigned 8-bit flag field")
        elif self.tcp_flags is not None:
            raise ValueError("tcp_flags must be null for non-TCP observations")


def _endpoint(packet: PacketObservation, *, source: bool) -> tuple[int, bytes, int]:
    address = packet.src_address if source else packet.dst_address
    port = packet.src_port if source else packet.dst_port
    return packet.ip_version, ipaddress.ip_address(address).packed, port


def _flow_key(packet: PacketObservation) -> tuple[int, tuple, tuple]:
    left, right = _endpoint(packet, source=True), _endpoint(packet, source=False)
    a, b = sorted((left, right))
    return packet.ip_protocol, a, b


@dataclass(frozen=True, slots=True)
class PortableFlow:
    flow_sequence: int
    originator: tuple[int, bytes, int]
    responder: tuple[int, bytes, int]
    packets: tuple[PacketObservation, ...]

    def __post_init__(self) -> None:
        if not self.packets:
            raise ValueError("a portable flow must contain at least one packet")
        if tuple(sorted(self.packets, key=lambda packet: (packet.timestamp_ns, packet.capture_index))) != self.packets:
            raise ValueError("flow packets must be in canonical timestamp/capture order")
        expected = _flow_key(self.packets[0])
        if any(_flow_key(packet) != expected for packet in self.packets):
            raise ValueError("all packets in a flow must share a bidirectional five-tuple")

    @property
    def protocol_number(self) -> int:
        return self.packets[0].ip_protocol

    @property
    def start_ns(self) -> int:
        return self.packets[0].timestamp_ns

    @property
    def end_ns(self) -> int:
        return self.packets[-1].timestamp_ns

    def is_forward(self, packet: PacketObservation) -> bool:
        return _endpoint(packet, source=True) == self.originator


def build_portable_flows(
    packets: Iterable[PacketObservation], idle_timeout_ns: int = IDLE_TIMEOUT_NS
) -> list[PortableFlow]:
    """Build flows with first-observed direction and a fixed strict-greater idle split."""
    if idle_timeout_ns <= 0:
        raise ValueError("idle_timeout_ns must be positive")
    ordered = sorted(packets, key=lambda packet: (packet.timestamp_ns, packet.capture_index))
    indices = [packet.capture_index for packet in ordered]
    if len(indices) != len(set(indices)):
        raise ValueError("capture_index must be unique to break timestamp ties deterministically")
    active: dict[tuple, tuple[int, tuple, tuple, list[PacketObservation]]] = {}
    completed: list[PortableFlow] = []
    sequence = 0
    for packet in ordered:
        key = _flow_key(packet)
        current = active.get(key)
        if current is not None and packet.timestamp_ns - current[3][-1].timestamp_ns > idle_timeout_ns:
            completed.append(PortableFlow(current[0], current[1], current[2], tuple(current[3])))
            current = None
        if current is None:
            current = (sequence, _endpoint(packet, source=True), _endpoint(packet, source=False), [])
            active[key] = current
            sequence += 1
        current[3].append(packet)
    completed.extend(PortableFlow(item[0], item[1], item[2], tuple(item[3])) for item in active.values())
    return sorted(completed, key=lambda flow: (flow.start_ns, flow.flow_sequence, _flow_key(flow.packets[0])))


def protocol_token(number: int) -> str:
    if not 0 <= number <= 255:
        return "IPPROTO_OOV"
    return f"IPPROTO_{number}"


def encode_protocol_token(token: str) -> int:
    if token.startswith("IPPROTO_"):
        suffix = token.removeprefix("IPPROTO_")
        if suffix.isdigit() and 0 <= int(suffix) <= 255 and suffix == str(int(suffix)):
            return int(suffix) + 2
    return 1


def _directions(flow: PortableFlow) -> tuple[list[PacketObservation], list[PacketObservation]]:
    forward = [packet for packet in flow.packets if flow.is_forward(packet)]
    backward = [packet for packet in flow.packets if not flow.is_forward(packet)]
    return forward, backward


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0 else None


def extract_core_features(flow: PortableFlow) -> dict[str, float | str | None]:
    """Compute the 19 semantic P12 baseline fields in IEEE-754 binary64."""
    forward, backward = _directions(flow)
    duration = (flow.end_ns - flow.start_ns) / 1_000_000_000.0
    forward_packets = float(len(forward))
    backward_packets = float(len(backward))
    total_packets = forward_packets + backward_packets
    forward_bytes = float(sum(packet.ip_total_length for packet in forward))
    backward_bytes = float(sum(packet.ip_total_length for packet in backward))
    total_bytes = forward_bytes + backward_bytes
    result: dict[str, float | str | None] = {
        "duration_seconds": float(duration),
        "forward_packet_count": forward_packets,
        "backward_packet_count": backward_packets,
        "total_packet_count": total_packets,
        "forward_byte_count": forward_bytes,
        "backward_byte_count": backward_bytes,
        "total_byte_count": total_bytes,
        "forward_packet_fraction": _safe_divide(forward_packets, total_packets),
        "forward_byte_fraction": _safe_divide(forward_bytes, total_bytes),
        "mean_packet_size_bytes": _safe_divide(total_bytes, total_packets),
        "forward_mean_packet_size_bytes": _safe_divide(forward_bytes, forward_packets),
        "backward_mean_packet_size_bytes": _safe_divide(backward_bytes, backward_packets),
        "packets_per_second": _safe_divide(total_packets, duration),
        "bytes_per_second": _safe_divide(total_bytes, duration),
        "forward_packets_per_second": _safe_divide(forward_packets, duration),
        "backward_packets_per_second": _safe_divide(backward_packets, duration),
        "forward_bytes_per_second": _safe_divide(forward_bytes, duration),
        "backward_bytes_per_second": _safe_divide(backward_bytes, duration),
        "ip_protocol": protocol_token(flow.protocol_number),
    }
    if tuple(result) != (*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES):
        raise AssertionError("implementation order diverged from schema")
    return result


def _iat_seconds(packets: list[PacketObservation]) -> list[float]:
    return [
        (right.timestamp_ns - left.timestamp_ns) / 1_000_000_000.0
        for left, right in zip(packets, packets[1:])
    ]


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _population_std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def _handshake(flow: PortableFlow) -> tuple[float | None, float | None, float | None]:
    if flow.protocol_number != 6:
        return None, None, None
    syn_ns = synack_ns = ack_ns = None
    for packet in flow.packets:
        flags = packet.tcp_flags or 0
        forward = flow.is_forward(packet)
        if syn_ns is None and forward and flags & 0x02 and not flags & 0x10:
            syn_ns = packet.timestamp_ns
        elif syn_ns is not None and synack_ns is None and not forward and flags & 0x12 == 0x12:
            synack_ns = packet.timestamp_ns
        elif synack_ns is not None and ack_ns is None and forward and flags & 0x10 and not flags & 0x02:
            ack_ns = packet.timestamp_ns
            break
    if syn_ns is None or synack_ns is None:
        return None, None, None
    synack = (synack_ns - syn_ns) / 1_000_000_000.0
    if ack_ns is None:
        return synack, None, None
    ack = (ack_ns - synack_ns) / 1_000_000_000.0
    return synack, ack, synack + ack


def portable_connection_state(flow: PortableFlow) -> str:
    """Return a local observable state, never an Argus/Zeek compatibility token."""
    if flow.protocol_number != 6:
        return "NOT_TCP"
    flags = [(packet.tcp_flags or 0, flow.is_forward(packet)) for packet in flow.packets]
    if any(value & 0x04 for value, _ in flags):
        return "TCP_RESET"
    if any(value & 0x01 and forward for value, forward in flags) and any(
        value & 0x01 and not forward for value, forward in flags
    ):
        return "TCP_CLOSED"
    _, _, complete = _handshake(flow)
    if complete is not None:
        return "TCP_ESTABLISHED"
    if any(value & 0x02 for value, _ in flags):
        return "TCP_PARTIAL_HANDSHAKE"
    return "TCP_OBSERVED"


def extract_transport_extension(flow: PortableFlow) -> dict[str, float | str | None]:
    """Compute optional observable transport diagnostics excluded from the baseline model."""
    forward, backward = _directions(flow)
    flow_iat = _iat_seconds(list(flow.packets))
    forward_iat = _iat_seconds(forward)
    backward_iat = _iat_seconds(backward)
    flag_bits = {
        "tcp_fin_count": 0x01,
        "tcp_syn_count": 0x02,
        "tcp_rst_count": 0x04,
        "tcp_psh_count": 0x08,
        "tcp_ack_count": 0x10,
        "tcp_urg_count": 0x20,
        "tcp_ece_count": 0x40,
        "tcp_cwr_count": 0x80,
    }
    flags = [packet.tcp_flags or 0 for packet in flow.packets] if flow.protocol_number == 6 else []
    synack, ack, complete = _handshake(flow)
    result: dict[str, float | str | None] = {
        "flow_iat_mean_seconds": _mean(flow_iat),
        "flow_iat_std_seconds": _population_std(flow_iat),
        "forward_iat_mean_seconds": _mean(forward_iat),
        "forward_iat_std_seconds": _population_std(forward_iat),
        "backward_iat_mean_seconds": _mean(backward_iat),
        "backward_iat_std_seconds": _population_std(backward_iat),
    }
    result.update({name: float(sum(bool(value & bit) for value in flags)) for name, bit in flag_bits.items()})
    result.update({
        "tcp_synack_seconds": synack,
        "tcp_ack_after_synack_seconds": ack,
        "tcp_handshake_seconds": complete,
        "portable_connection_state": portable_connection_state(flow),
    })
    if tuple(result) != TRANSPORT_EXTENSION_FIELDS:
        raise AssertionError("transport-extension order diverged from contract")
    return result


def _float64_identity_token(value: float | None) -> str | None:
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("identity cannot include non-finite values")
    if number == 0.0:
        number = 0.0
    return struct.pack(">d", number).hex()


def canonical_flow_identity_v3(features: dict[str, object]) -> str:
    """Hash only semantic model inputs; provenance, labels and scores are excluded."""
    missing = [name for name in (*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES) if name not in features]
    if missing:
        raise ValueError(f"missing identity fields: {missing}")
    protocol = str(features["ip_protocol"])
    if encode_protocol_token(protocol) == 1:
        protocol = "IPPROTO_OOV"
    payload = {
        "identity_version": IDENTITY_VERSION,
        "schema_version": VERSION,
        "continuous_float64_be": [
            [name, _float64_identity_token(features[name])] for name in CONTINUOUS_FEATURES
        ],
        "categorical": [["ip_protocol", protocol]],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def _feature_specs() -> list[dict]:
    formulas = {
        "duration_seconds": "(last_packet_timestamp_ns - first_packet_timestamp_ns) / 1e9",
        "forward_packet_count": "count(packet where packet.source_endpoint == first_observed_source_endpoint)",
        "backward_packet_count": "count(packet where packet.source_endpoint != first_observed_source_endpoint)",
        "total_packet_count": "forward_packet_count + backward_packet_count",
        "forward_byte_count": "sum(ip_total_length for forward packets)",
        "backward_byte_count": "sum(ip_total_length for backward packets)",
        "total_byte_count": "forward_byte_count + backward_byte_count",
        "forward_packet_fraction": "forward_packet_count / total_packet_count",
        "forward_byte_fraction": "forward_byte_count / total_byte_count",
        "mean_packet_size_bytes": "total_byte_count / total_packet_count",
        "forward_mean_packet_size_bytes": "forward_byte_count / forward_packet_count",
        "backward_mean_packet_size_bytes": "backward_byte_count / backward_packet_count",
        "packets_per_second": "total_packet_count / duration_seconds",
        "bytes_per_second": "total_byte_count / duration_seconds",
        "forward_packets_per_second": "forward_packet_count / duration_seconds",
        "backward_packets_per_second": "backward_packet_count / duration_seconds",
        "forward_bytes_per_second": "forward_byte_count / duration_seconds",
        "backward_bytes_per_second": "backward_byte_count / duration_seconds",
        "ip_protocol": "'IPPROTO_' + unsigned decimal IANA IP protocol number",
    }
    count_fields = {name for name in CONTINUOUS_FEATURES if name.endswith("_count")}
    specs = []
    for position, name in enumerate((*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES)):
        categorical = name in CATEGORICAL_FEATURES
        if name == "duration_seconds":
            unit = "seconds"
        elif "byte" in name and "fraction" not in name and "per_second" not in name:
            unit = "IP-layer bytes" if "size" not in name else "IP-layer bytes/packet"
        elif name.endswith("bytes_per_second") or name == "bytes_per_second":
            unit = "IP-layer bytes/second"
        elif name.endswith("packets_per_second") or name == "packets_per_second":
            unit = "packets/second"
        elif "fraction" in name:
            unit = "unitless ratio"
        elif "packet_count" in name:
            unit = "packets"
        else:
            unit = "categorical token"
        if categorical:
            missing = "NULL_EXTRACTION_FAILURE only; malformed adapter values map to IPPROTO_OOV"
        elif name in count_fields:
            missing = "ZERO_IS_VALID_ABSENCE; null only on extraction failure"
        elif name.endswith("per_second") or name in {"packets_per_second", "bytes_per_second"}:
            missing = "NULL_NOT_APPLICABLE when duration_seconds == 0; NULL_EXTRACTION_FAILURE otherwise"
        elif name == "backward_mean_packet_size_bytes":
            missing = "NULL_INSUFFICIENT_PACKETS when backward_packet_count == 0"
        else:
            missing = "NULL_EXTRACTION_FAILURE only for an invalid/failed flow observation"
        specs.append({
            "position": position,
            "name": name,
            "kind": "categorical" if categorical else "continuous",
            "dtype": "UTF-8 controlled token" if categorical else "IEEE-754 binary64",
            "unit": unit,
            "direction": (
                "first-observed source to destination" if name.startswith("forward_") else
                "reverse of first-observed direction" if name.startswith("backward_") else
                "bidirectional/not direction-specific"
            ),
            "aggregation_window": "one deterministic bidirectional flow",
            "formula": formulas[name],
            "source_dependency": ["normalized IP packet header", "capture timestamp"],
            "missing_value_semantics": missing,
            "label_dependency": False,
        })
    return specs


def _mapping(original_fields: list[str], p11_gap_sha256: str) -> dict:
    redefine = {
        "dur": ("duration_seconds", "fixed nanosecond timestamps and seconds; no historical Argus timeout claim"),
        "sbytes": ("forward_byte_count", "IP total-length bytes replace historical source-byte semantics"),
        "dbytes": ("backward_byte_count", "IP total-length bytes replace historical destination-byte semantics"),
        "spkts": ("forward_packet_count", "first-observed direction replaces historical Argus direction"),
        "dpkts": ("backward_packet_count", "reverse first-observed direction replaces historical Argus direction"),
        "smean": ("forward_mean_packet_size_bytes", "explicit IP-layer bytes/count with null on zero denominator"),
        "dmean": ("backward_mean_packet_size_bytes", "explicit IP-layer bytes/count with null on zero denominator"),
        "rate": ("packets_per_second", "total observed packets/duration; zero-duration is null, no epsilon"),
        "sload": ("forward_bytes_per_second", "bytes/second replaces historical bits/second load"),
        "dload": ("backward_bytes_per_second", "bytes/second replaces historical bits/second load"),
        "bytes_ratio": ("forward_byte_fraction", "no epsilon; denominator-zero is explicit null"),
        "pkts_ratio": ("forward_packet_fraction", "no epsilon; denominator-zero is explicit null"),
        "src_bps": ("forward_bytes_per_second", "same high-level rate but new byte and duration semantics"),
        "pkt_rate": ("forward_packets_per_second", "same high-level rate but no epsilon/min-duration clamp"),
        "sinpkt": ("forward_iat_mean_seconds", "optional extension uses seconds and observed adjacent intervals"),
        "dinpkt": ("backward_iat_mean_seconds", "optional extension uses seconds and observed adjacent intervals"),
        "sjit": ("forward_iat_std_seconds", "optional population standard deviation; not Argus jitter"),
        "djit": ("backward_iat_std_seconds", "optional population standard deviation; not Argus jitter"),
        "synack": ("tcp_synack_seconds", "optional ordered packet-observed SYN to SYN-ACK interval"),
        "ackdat": ("tcp_ack_after_synack_seconds", "optional ordered SYN-ACK to ACK interval"),
        "tcprtt": ("tcp_handshake_seconds", "optional handshake completion interval; not Argus RTT"),
        "incomplete_tcp": ("portable_connection_state", "optional categorical state replaces a historical binary heuristic"),
        "proto": ("ip_protocol", "IANA numeric protocol token replaces dataset vocabulary"),
        "state": ("portable_connection_state", "optional local state machine; explicitly not Argus/Zeek-equivalent"),
    }
    drop_reasons = {
        "service": "application-service inference is removed; port guesses and payload parsing are forbidden",
        "ct_flw_http_mthd": "payload/application parsing is outside the portable baseline",
        "ct_ftp_cmd": "payload/application parsing is outside the portable baseline",
        "is_ftp_login": "payload/application parsing is outside the portable baseline",
        "trans_depth": "payload/application parsing is outside the portable baseline",
        "response_body_len": "payload/application parsing is outside the portable baseline",
        "sloss": "packet loss is not observable consistently from a single passive capture",
        "dloss": "packet loss is not observable consistently from a single passive capture",
        "loss_rate_src": "depends on a non-portable packet-loss estimate",
        "loss_rate_dst": "depends on a non-portable packet-loss estimate",
        "swin": "TCP window semantics depend on packet selection and protocol state",
        "dwin": "TCP window semantics depend on packet selection and protocol state",
        "stcpb": "raw TCP sequence number is endpoint/session-specific and unsuitable as a portable model feature",
        "dtcpb": "raw TCP sequence number is endpoint/session-specific and unsuitable as a portable model feature",
        "sttl": "TTL is capture-point and route dependent",
        "dttl": "TTL is capture-point and route dependent",
        "ttl_diff": "derived from capture-point and route-dependent TTL values",
        "ttl_sum": "derived from capture-point and route-dependent TTL values",
        "is_sm_ips_ports": "endpoint-identity coincidence is brittle and not retained as behavior",
    }
    rows = []
    for name in original_fields:
        if name in redefine:
            target, difference = redefine[name]
            rows.append({
                "original_feature": name,
                "decision": "REDEFINE_PORTABLE",
                "portable_feature": target,
                "inclusion_scope": "optional_extension" if target in TRANSPORT_EXTENSION_FIELDS else "baseline",
                "reason": "retain a network-behavior concept only under a fully explicit packet-observable formula",
                "semantic_difference": difference,
            })
        else:
            if name.startswith("ct_"):
                reason = "historical last-N/context semantics are extractor- and ordering-dependent"
            elif name.startswith("log_"):
                reason = "redundant nonlinear transform belongs in a future fold-local model pipeline, not the schema"
            else:
                reason = drop_reasons.get(name, "historical derived semantic is redundant, unstable, or not portable")
            rows.append({
                "original_feature": name,
                "decision": "DROP_NONPORTABLE",
                "portable_feature": None,
                "inclusion_scope": "excluded",
                "reason": reason,
                "semantic_difference": "feature is absent from schema v3 and must not be imputed",
            })
    counts = dict(sorted(Counter(row["decision"] for row in rows).items()))
    return {
        "artifact_type": "P12_ORIGINAL_TO_V3_MAPPING",
        "schema_version": VERSION,
        "p11_gap_analysis_sha256": p11_gap_sha256,
        "original_semantic_feature_count": len(original_fields),
        "decision_vocabulary": ["KEEP_EXACT", "REDEFINE_PORTABLE", "DROP_NONPORTABLE"],
        "decision_counts": counts,
        "keep_exact_count": counts.get("KEEP_EXACT", 0),
        "note": "No P9 field is asserted exact: even retained concepts receive new observable semantics.",
        "rows": rows,
    }


DATASETS = {
    "unsw_nb15": ("UNSW-NB15", "unsw-nb15-capture", "https://research.unsw.edu.au/projects/unsw-nb15-dataset"),
    "cic_ids_2017": ("CIC-IDS2017", "cic-ids-2017-capture", "https://www.unb.ca/cic/datasets/ids-2017.html"),
    "cse_cic_ids_2018": ("CSE-CIC-IDS2018", "cse-cic-ids-2018-capture", "https://www.unb.ca/cic/datasets/ids-2018.html"),
    "ton_iot": ("TON_IoT network", "ton-iot-network-capture", "https://research.unsw.edu.au/projects/toniot-datasets"),
    "bot_iot": ("Bot-IoT", "bot-iot-capture", "https://research.unsw.edu.au/projects/bot-iot-dataset"),
    "cic_ddos_2019": ("CIC-DDoS2019", "cic-ddos-2019-capture", "https://www.unb.ca/cic/datasets/ddos-2019.html"),
}


def _compatibility_matrix() -> dict:
    features = [*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES]
    rows = []
    for dataset_id, (dataset, family, url) in DATASETS.items():
        for feature in features:
            classification = "DIRECT" if feature == "ip_protocol" else "DERIVABLE"
            rows.append({
                "dataset_id": dataset_id,
                "dataset": dataset,
                "capture_family": family,
                "feature": feature,
                "classification": classification,
                "basis": (
                    "read unsigned protocol number from normalized IP header"
                    if feature == "ip_protocol"
                    else "apply the frozen P12 formula to timestamped IP packets from the documented raw PCAP"
                ),
                "source_url": url,
                "structural_imputation_required": False,
            })
    coverage = {}
    for dataset_id, (dataset, family, url) in DATASETS.items():
        selected = [row for row in rows if row["dataset_id"] == dataset_id]
        coverage[dataset_id] = {
            "dataset": dataset,
            "capture_family": family,
            "official_source": url,
            "raw_pcap_documented": True,
            "core_supported": sum(row["classification"] != "UNAVAILABLE" for row in selected),
            "core_total": len(features),
            "complete_core": all(row["classification"] != "UNAVAILABLE" for row in selected),
        }
    return {
        "artifact_type": "P12_DATASET_COMPATIBILITY_MATRIX",
        "schema_version": VERSION,
        "classification_vocabulary": ["DIRECT", "DERIVABLE", "UNAVAILABLE"],
        "assessment_basis": "official dataset landing-page metadata only; no dataset file was opened",
        "baseline_feature_count": len(features),
        "datasets": coverage,
        "rows": rows,
        "minimum_complete_capture_families": 3,
        "complete_capture_family_count": len({item["capture_family"] for item in coverage.values() if item["complete_core"]}),
        "full_core_schema_implementable": True,
        "structural_imputation_required": False,
        "extension_scope": "transport extension is not required for baseline compatibility and is not claimed validated",
        "raw_data_accessed": False,
    }


def build_artifacts(root: Path, created_at_utc: str) -> dict[str, dict]:
    frozen_schema_path = root / "results/data_quality/p3/schema_v2.json"
    p11_gap_path = root / "results/extractor/p11/portable_schema_gap_analysis.json"
    frozen_schema = json.loads(frozen_schema_path.read_text())
    original_fields = [
        *frozen_schema["contract"]["continuous_features"],
        *frozen_schema["contract"]["categorical_features"],
    ]
    p11_gap_sha = file_sha256(p11_gap_path)
    fields = _feature_specs()
    schema_payload = {
        "artifact_type": "P12_PORTABLE_FEATURE_SCHEMA",
        "schema_version": VERSION,
        "scientific_branch": "new-model-required",
        "p9_input_compatible": False,
        "p9_checkpoint_migration_forbidden": True,
        "feature_selection_used_performance": False,
        "baseline_continuous_count": len(CONTINUOUS_FEATURES),
        "baseline_categorical_count": len(CATEGORICAL_FEATURES),
        "baseline_semantic_dimension": len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES),
        "ordered_baseline_fields": [*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES],
        "fields": fields,
        "byte_definition": "IPv4 total_length or IPv6 fixed 40-byte header plus payload_length; includes IP header, excludes L2",
        "direction_definition": "first observed packet after canonical time/capture-index ordering defines forward direction",
        "flow_key": "unordered normalized endpoint pair plus IANA IP protocol number",
        "flow_segmentation": "new flow only when inter-packet idle gap is strictly greater than 60 seconds; capture end terminates",
        "packet_tie_break": "unique capture_index ascending",
        "fragment_policy": "P13 must explicitly reassemble or reject fragmented traffic; silent partial accounting is forbidden",
        "payload_parsing": False,
        "context_extension_in_baseline": False,
        "service_policy": "REMOVED_FROM_BASELINE; no port-to-service inference and no Bro/Zeek equivalence claim",
        "transport_extension": {
            "version": TRANSPORT_EXTENSION_VERSION,
            "baseline_model_input": False,
            "fields": list(TRANSPORT_EXTENSION_FIELDS),
        },
        "canonical_dtype": "IEEE-754 binary64",
        "source_bindings": {
            "p3_schema_path": str(frozen_schema_path.relative_to(root)),
            "p3_schema_file_sha256": file_sha256(frozen_schema_path),
            "p3_schema_contract_sha256": frozen_schema["sha256"],
            "p11_gap_analysis_path": str(p11_gap_path.relative_to(root)),
            "p11_gap_analysis_file_sha256": p11_gap_sha,
            "p8_protocol_sha256": P8_PROTOCOL_SHA256,
            "p9_protocol_sha256": P9_PROTOCOL_SHA256,
        },
    }
    schema_payload["schema_sha256"] = sha256_bytes(canonical_json_bytes(schema_payload))
    mapping = _mapping(original_fields, p11_gap_sha)
    semantics = {
        "artifact_type": "P12_FEATURE_SEMANTIC_CONTRACT",
        "schema_version": VERSION,
        "canonical_dtype": "IEEE-754 binary64 before and after fold-local preprocessing",
        "flow_observation_contract": {
            "packet_source": "timestamped IPv4/IPv6 headers normalized by a P13 dataset adapter",
            "minimum_flow_packets": 1,
            "idle_timeout_ns": IDLE_TIMEOUT_NS,
            "direction": "originator is source endpoint of first observed packet in the segmented flow",
            "non_ip_frames": "excluded with audited counts",
            "fragments": "fail closed unless P13 freezes deterministic reassembly",
            "out_of_order_input": "sort by timestamp_ns then unique capture_index",
        },
        "fields": fields,
        "label_independence": {
            "labels_used_in_extraction": False,
            "labels_used_in_feature_selection": False,
            "target_family_used_in_features": False,
            "per_dataset_feature_formulas": False,
        },
    }
    state_contract = {
        "artifact_type": "P12_PORTABLE_STATE_CONTRACT",
        "version": STATE_VERSION,
        "baseline_model_input": False,
        "historical_equivalence": "NONE; not Argus state and not Zeek conn_state",
        "categories": [
            "NOT_TCP", "TCP_RESET", "TCP_CLOSED", "TCP_ESTABLISHED",
            "TCP_PARTIAL_HANDSHAKE", "TCP_OBSERVED",
        ],
        "precedence": [
            "NOT_TCP when IP protocol is not 6",
            "TCP_RESET when any observed TCP packet has RST",
            "TCP_CLOSED when both directions contain FIN",
            "TCP_ESTABLISHED when ordered forward SYN, backward SYN+ACK, forward ACK completes",
            "TCP_PARTIAL_HANDSHAKE when any SYN is observed without a complete handshake",
            "TCP_OBSERVED otherwise",
        ],
        "handshake_formulas": {
            "tcp_synack_seconds": "first valid backward SYN+ACK timestamp minus first forward SYN-without-ACK timestamp",
            "tcp_ack_after_synack_seconds": "first later forward ACK-without-SYN minus valid SYN+ACK timestamp",
            "tcp_handshake_seconds": "sum of the preceding two intervals",
        },
        "tcp_flag_counts": "count each TCP packet on which the corresponding FIN/SYN/RST/PSH/ACK/URG/ECE/CWR bit is set",
        "missing_semantics": "handshake times are NULL_NOT_APPLICABLE for non-TCP and NULL_INSUFFICIENT_PACKETS for incomplete TCP",
    }
    categorical = {
        "artifact_type": "P12_CATEGORICAL_CONTRACT",
        "schema_version": VERSION,
        "baseline": {
            "ip_protocol": {
                "semantic_source": "unsigned 8-bit IANA Protocol/Next Header value",
                "normalization": "canonical ASCII IPPROTO_<minimal unsigned decimal>",
                "vocabulary": "fixed IPPROTO_0 through IPPROTO_255; independent of labels and dataset",
                "integer_encoding": {"PAD": 0, "OOV": 1, "IPPROTO_n": "n + 2"},
                "oov_behavior": "malformed or adapter-unknown tokens map to OOV; valid 0..255 never map to OOV",
            }
        },
        "service": {"decision": "REMOVED", "reason": "portable baseline performs no payload or port service inference"},
        "state": {"decision": "OPTIONAL_EXTENSION", "contract": STATE_VERSION},
        "label_tokens_forbidden": True,
    }
    missing = {
        "artifact_type": "P12_MISSING_VALUE_CONTRACT",
        "schema_version": VERSION,
        "semantic_states": list(MISSING_STATES),
        "rules": {
            "counts": "zero is a measured valid absence; never convert zero to null",
            "directional_means": "null with NULL_INSUFFICIENT_PACKETS when the direction has zero packets",
            "rates": "null with NULL_NOT_APPLICABLE when duration is zero; no epsilon or duration clamp",
            "IAT": "mean requires at least one interval; population std requires at least two intervals",
            "TCP_handshake": "non-TCP is NULL_NOT_APPLICABLE; incomplete observation is NULL_INSUFFICIENT_PACKETS",
            "failure": "parse/reassembly/truncation failures are NULL_EXTRACTION_FAILURE and audited; never structural zero-fill",
        },
        "model_boundary": {
            "strategy": "continuous values plus an equal-width boolean missing mask",
            "semantic_null_to_numeric": "after fold-local scaling only, place numeric 0.0 where mask=true; mask preserves missingness",
            "scaler_fit": "development-training fold only, observed values only",
            "scaler_reuse_from_p9": False,
            "structural_imputation": False,
        },
    }
    identity = {
        "artifact_type": "P12_CANONICAL_IDENTITY_CONTRACT",
        "identity_version": IDENTITY_VERSION,
        "definition": "SHA-256 of schema/version-tagged ordered baseline semantics; continuous values are big-endian float64 hex or null",
        "categorical_representation": "canonical IPPROTO_n token or IPPROTO_OOV",
        "included": [*CONTINUOUS_FEATURES, *CATEGORICAL_FEATURES],
        "excluded": [
            "row index", "dataset name", "file path", "capture timestamp", "IP address", "port",
            "label", "attack family", "split role", "model score", "threshold", "prediction",
        ],
        "uses": ["duplicate audit", "group-aware split", "cross-dataset semantic-overlap audit"],
        "reconstruction_key_is_distinct": True,
        "reconstruction_key": "dataset/file provenance plus bidirectional five-tuple, flow start/end and segment ordinal",
    }
    compatibility = _compatibility_matrix()
    future_interface = {
        "artifact_type": "P12_FUTURE_MODEL_INTERFACE",
        "schema_version": VERSION,
        "new_model_required": True,
        "p9_checkpoint_reuse": False,
        "p9_scaler_reuse": False,
        "p9_threshold_reuse": False,
        "inputs": {
            "continuous_values": f"float64[batch,{len(CONTINUOUS_FEATURES)}] in frozen field order",
            "continuous_missing_mask": f"bool[batch,{len(CONTINUOUS_FEATURES)}], true exactly where semantic value is null",
            "ip_protocol_id": "int64[batch], fixed categorical encoding",
        },
        "fold_local_preprocessing": {
            "continuous": "RobustScaler-like median/IQR transform fit on development-training identities only and observed values only",
            "categorical": "fixed protocol vocabulary; no labels, target rows or external rows used",
            "masked_numeric_fill": "0.0 after transform with mask retained as a separate input",
            "external_fit_forbidden": True,
        },
        "conservative_future_architecture": {
            "status": "DESIGN_ONLY_UNTRAINED",
            "continuous_branch": "small fully connected encoder over values concatenated with missing mask",
            "categorical_branch": "protocol embedding with dimension 4",
            "fusion": "concatenate branches before compact bottleneck and symmetric reconstruction heads",
            "transport_extension": "excluded from baseline; requires a later preregistered amendment to enter a model",
        },
        "evaluation_tracks": {
            "within_dataset_loafo": "future P14/P15 only",
            "cross_dataset_ood": "future P16 only; never mixed with LOAFO claims",
        },
        "training_performed": False,
    }
    artifacts = {
        "schema_v3.json": schema_payload,
        "original_to_v3_mapping.json": mapping,
        "feature_semantic_contract.json": semantics,
        "portable_state_contract.json": state_contract,
        "categorical_contract.json": categorical,
        "missing_value_contract.json": missing,
        "canonical_identity_contract.json": identity,
        "dataset_compatibility_matrix.json": compatibility,
        "future_model_interface.json": future_interface,
    }
    artifact_hashes = {name: sha256_bytes(canonical_json_bytes(value, pretty=True)) for name, value in artifacts.items()}
    source_hash = file_sha256(root / "src/ids/p12_portable_schema.py")
    protocol_binding = {"artifact_sha256": artifact_hashes, "source_sha256": source_hash}
    protocol_hash = sha256_bytes(canonical_json_bytes(protocol_binding))
    artifacts["p12_gate.json"] = {
        "artifact_type": "P12_PORTABLE_SCHEMA_GATE",
        "status": "P12_PORTABLE_SCHEMA_PASS",
        "created_at_utc": created_at_utc,
        "schema_version": VERSION,
        "p12_protocol_sha256": protocol_hash,
        "source_sha256": source_hash,
        "artifact_sha256": artifact_hashes,
        "baseline_semantic_dimension": len(CONTINUOUS_FEATURES) + len(CATEGORICAL_FEATURES),
        "complete_capture_family_count": compatibility["complete_capture_family_count"],
        "minimum_complete_capture_families": compatibility["minimum_complete_capture_families"],
        "structural_imputation_required": False,
        "p9_compatibility_claimed": False,
        "p9_checkpoint_opened": False,
        "model_training_performed": False,
        "p7_or_p9_detector_evaluated": False,
        "external_dataset_file_opened": False,
        "performance_guided_feature_selection": False,
        "readiness": "P13_EXTRACTION_AND_DATA_PROTOCOL_DESIGN_ONLY",
    }
    return artifacts


def validate_artifacts(root: Path, artifacts: dict[str, dict]) -> None:
    required = {
        "schema_v3.json", "original_to_v3_mapping.json", "feature_semantic_contract.json",
        "portable_state_contract.json", "categorical_contract.json", "missing_value_contract.json",
        "canonical_identity_contract.json", "dataset_compatibility_matrix.json",
        "future_model_interface.json", "p12_gate.json",
    }
    if set(artifacts) != required:
        raise ValueError("P12 artifact inventory mismatch")
    schema = artifacts["schema_v3.json"]
    if schema["baseline_semantic_dimension"] != 19 or len(schema["fields"]) != 19:
        raise ValueError("P12 baseline dimensionality mismatch")
    if schema["p9_input_compatible"] or not schema["p9_checkpoint_migration_forbidden"]:
        raise ValueError("P12/P9 scientific branch separation failed")
    expected_schema_hash = schema.pop("schema_sha256")
    actual_schema_hash = sha256_bytes(canonical_json_bytes(schema))
    schema["schema_sha256"] = expected_schema_hash
    if actual_schema_hash != expected_schema_hash:
        raise ValueError("P12 schema self-hash mismatch")
    mapping = artifacts["original_to_v3_mapping.json"]
    original = [
        *json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())["contract"]["continuous_features"],
        *json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())["contract"]["categorical_features"],
    ]
    if len(mapping["rows"]) != 61 or [row["original_feature"] for row in mapping["rows"]] != original:
        raise ValueError("P12 original-61 mapping is incomplete or reordered")
    if any(row["decision"] == "KEEP_EXACT" for row in mapping["rows"]):
        raise ValueError("P12 must not claim exact P9 semantics")
    compatibility = artifacts["dataset_compatibility_matrix.json"]
    if compatibility["complete_capture_family_count"] < compatibility["minimum_complete_capture_families"]:
        raise ValueError("P12 portability minimum is not met")
    if compatibility["structural_imputation_required"] or not compatibility["full_core_schema_implementable"]:
        raise ValueError("P12 core requires forbidden structural imputation")
    if len(compatibility["rows"]) != len(DATASETS) * 19:
        raise ValueError("P12 compatibility matrix is incomplete")
    gate = artifacts["p12_gate.json"]
    recalculated = {
        name: sha256_bytes(canonical_json_bytes(value, pretty=True))
        for name, value in artifacts.items() if name != "p12_gate.json"
    }
    if gate["artifact_sha256"] != recalculated:
        raise ValueError("P12 gate artifact binding mismatch")
    binding = {"artifact_sha256": recalculated, "source_sha256": gate["source_sha256"]}
    if gate["p12_protocol_sha256"] != sha256_bytes(canonical_json_bytes(binding)):
        raise ValueError("P12 protocol hash mismatch")
    if gate["status"] != "P12_PORTABLE_SCHEMA_PASS":
        raise ValueError("P12 gate status mismatch")
    forbidden_true = (
        gate["p9_checkpoint_opened"], gate["model_training_performed"],
        gate["p7_or_p9_detector_evaluated"], gate["external_dataset_file_opened"],
        gate["performance_guided_feature_selection"],
    )
    if any(forbidden_true):
        raise ValueError("P12 no-training/no-evaluation boundary failed")
