"""Synthetic P13 protocol tests; no external dataset or historical model access."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import PacketObservation, build_portable_flows  # noqa: E402
from ids.p13.labels import GroundTruthRule, join_labels  # noqa: E402
from ids.p13.portable_extractor import extract_rows, semantic_content_hash  # noqa: E402
from ids.p13.raw_reader import CaptureFormatError, read_classic_pcap  # noqa: E402
from ids.p13.validation import feature_sanity, identity_audit, population_audit  # noqa: E402


def ethernet(payload: bytes, ether_type: int = 0x0800, vlan: int | None = None) -> bytes:
    header = b"\x00" * 12
    if vlan is None:
        return header + struct.pack("!H", ether_type) + payload
    return header + b"\x81\x00" + struct.pack("!HH", vlan & 0x0FFF, ether_type) + payload


def tcp(sport=1234, dport=80, flags=0x10) -> bytes:
    return struct.pack("!HHIIBBHHH", sport, dport, 1, 1, 0x50, flags, 4096, 0, 0)


def udp(sport=5353, dport=53, payload=b"") -> bytes:
    return struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload


def ipv4(segment: bytes, protocol: int, src="10.0.0.1", dst="10.0.0.2", fragment=0) -> bytes:
    source = ipaddress.IPv4Address(src).packed
    destination = ipaddress.IPv4Address(dst).packed
    return struct.pack(
        "!BBHHHBBH4s4s", 0x45, 0, 20 + len(segment), 1, fragment, 64, protocol, 0, source, destination
    ) + segment


def ipv6(segment: bytes, next_header: int, src="2001:db8::1", dst="2001:db8::2") -> bytes:
    source = ipaddress.IPv6Address(src).packed
    destination = ipaddress.IPv6Address(dst).packed
    return struct.pack("!IHBB16s16s", 6 << 28, len(segment), next_header, 64, source, destination) + segment


def write_pcap(path: Path, records: list[tuple[int, int, bytes, int | None]], nanoseconds=False) -> None:
    magic = b"\x4d\x3c\xb2\xa1" if nanoseconds else b"\xd4\xc3\xb2\xa1"
    content = bytearray(magic + struct.pack("<HHIIII", 2, 4, 0, 0, 65535, 1))
    for seconds, fraction, frame, original in records:
        content.extend(struct.pack("<IIII", seconds, fraction, len(frame), original or len(frame)))
        content.extend(frame)
    path.write_bytes(content)


def test_parser_timestamps_ethernet_ipv4_tcp_udp_icmp_vlan_and_ipv6(tmp_path):
    frames = [
        ethernet(ipv4(tcp(flags=0x02), 6)),
        ethernet(ipv4(udp(), 17), vlan=7),
        ethernet(ipv4(b"\x08\x00" + b"\x00" * 6, 1)),
        ethernet(ipv6(udp(1111, 2222), 17), ether_type=0x86DD),
    ]
    path = tmp_path / "supported.pcap"
    write_pcap(path, [(1, 250_000 + index, frame, None) for index, frame in enumerate(frames)])
    result = read_classic_pcap(path)
    assert len(result.packets) == 4
    assert result.packets[0].timestamp_ns == 1_250_000_000
    assert result.packets[0].tcp_flags == 0x02
    assert (result.packets[1].src_port, result.packets[1].dst_port) == (5353, 53)
    assert result.packets[2].ip_protocol == 1 and result.packets[2].src_port == 0
    assert result.packets[3].ip_version == 6 and result.packets[3].ip_protocol == 17
    assert result.audit["vlan_packets"] == 1
    assert result.audit["ipv4_packets_emitted"] == 3
    assert result.audit["ipv6_packets_emitted"] == 1


def test_nanosecond_timestamp_precision(tmp_path):
    frame = ethernet(ipv4(udp(), 17))
    path = tmp_path / "nano.pcap"
    write_pcap(path, [(2, 123, frame, None)], nanoseconds=True)
    result = read_classic_pcap(path)
    assert result.packets[0].timestamp_ns == 2_000_000_123
    assert result.audit["timestamp_precision"] == "nanoseconds"


def test_malformed_truncated_non_ip_and_fragments_are_counted(tmp_path):
    good = ethernet(ipv4(udp(), 17))
    malformed = b"\x00" * 12 + b"\x08\x00"
    non_ip = ethernet(b"\x00" * 20, ether_type=0x0806)
    ipv4_fragment = ethernet(ipv4(udp(), 17, fragment=0x2000))
    ipv6_fragment = ethernet(ipv6(b"\x00" * 8, 44), ether_type=0x86DD)
    path = tmp_path / "audit.pcap"
    records = [
        (1, 0, good, None),
        (2, 0, malformed, None),
        (3, 0, non_ip, None),
        (4, 0, ipv4_fragment, None),
        (5, 0, ipv6_fragment, None),
        (6, 0, good, len(good) + 10),
    ]
    write_pcap(path, records)
    result = read_classic_pcap(path)
    assert len(result.packets) == 1
    assert result.audit["malformed_packets_excluded"] == 1
    assert result.audit["truncated_capture_packets_excluded"] == 1
    assert result.audit["non_ip_packets_excluded"] == 1
    assert result.audit["ipv4_fragment_packets_excluded"] == 1
    assert result.audit["ipv6_fragment_packets_excluded"] == 1


def test_pcapng_and_non_ethernet_fail_closed(tmp_path):
    bad = tmp_path / "bad.pcapng"
    bad.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 20)
    with pytest.raises(CaptureFormatError):
        read_classic_pcap(bad)
    non_ethernet = tmp_path / "raw-ip.pcap"
    non_ethernet.write_bytes(
        b"\xd4\xc3\xb2\xa1" + struct.pack("<HHIIII", 2, 4, 0, 0, 65535, 101)
    )
    with pytest.raises(CaptureFormatError):
        read_classic_pcap(non_ethernet)


def observation(index, timestamp_ns, src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80,
                protocol=6, length=60, flags=0x10):
    return PacketObservation(
        index, timestamp_ns, 4, src, dst, protocol, sport, dport, length,
        flags if protocol == 6 else None,
    )


def test_flow_direction_five_tuple_timeout_boundary_and_repeated_session():
    packets = [
        observation(0, 0, flags=0x02),
        observation(1, 1, src="10.0.0.2", dst="10.0.0.1", sport=80, dport=1234, flags=0x12),
        observation(2, 60_000_000_002),
        observation(3, 120_000_000_002),
        observation(4, 180_000_000_003),
    ]
    flows = build_portable_flows(list(reversed(packets)))
    assert [len(flow.packets) for flow in flows] == [2, 2, 1]
    assert flows[0].is_forward(flows[0].packets[0])
    assert not flows[0].is_forward(flows[0].packets[1])


def test_all_18_features_missing_mask_and_protocol():
    flow = build_portable_flows([observation(0, 10)])[0]
    row = extract_rows([flow])[0]
    assert len(row.continuous) == len(row.missing_mask) == 18
    assert sum(row.missing_mask) == 7
    assert row.ip_protocol == "IPPROTO_6"
    assert row.continuous[0] == 0.0
    assert row.continuous[2] == 0.0
    assert row.continuous[11] is None


def test_identity_excludes_addresses_ports_absolute_time_and_labels():
    left = build_portable_flows([
        observation(0, 1_000, length=60), observation(1, 2_000, length=80),
    ])[0]
    right = build_portable_flows([
        observation(10, 99_001_000, src="192.0.2.1", dst="192.0.2.2", sport=9, dport=10, length=60),
        observation(11, 99_002_000, src="192.0.2.1", dst="192.0.2.2", sport=9, dport=10, length=80),
    ])[0]
    left_row, right_row = extract_rows([left])[0], extract_rows([right])[0]
    assert left_row.canonical_identity_v3 == right_row.canonical_identity_v3
    rules = [GroundTruthRule("label", 0, 10_000, "attack-x", "ATTACK")]
    assert join_labels([left], rules)[0].portable_label == "ATTACK"
    assert extract_rows([left])[0].canonical_identity_v3 == left_row.canonical_identity_v3


def test_label_join_matched_unmatched_and_conflict():
    flows = build_portable_flows([
        observation(0, 100),
        observation(1, 70_000_000_100, src="10.0.0.3", dst="10.0.0.4"),
    ])
    rules = [
        GroundTruthRule("a", 0, 1_000, "Benign", "NORMAL"),
        GroundTruthRule("b", 0, 1_000, "Attack", "ATTACK"),
    ]
    joined = join_labels(flows, rules)
    assert joined[0].status == "AMBIGUOUS"
    assert joined[1].status == "UNMATCHED"
    matched = join_labels(flows, [rules[0]])
    assert matched[0].status == "MATCHED" and matched[0].source_label == "Benign"


def test_deterministic_two_run_content_hash_and_audits(tmp_path):
    frames = [ethernet(ipv4(tcp(flags=0x02), 6)), ethernet(ipv4(tcp(flags=0x10), 6))]
    path = tmp_path / "deterministic.pcap"
    write_pcap(path, [(1, 0, frames[0], None), (1, 100_000, frames[1], None)])

    def run_once():
        parsed = read_classic_pcap(path)
        return extract_rows(build_portable_flows(parsed.packets))

    first, second = run_once(), run_once()
    assert semantic_content_hash(first) == semantic_content_hash(second)
    sanity = feature_sanity(first)
    assert sanity["row_count"] == 1 and sanity["impossible_value_violations"] == 0
    labels = join_labels(build_portable_flows(read_classic_pcap(path).packets), [
        GroundTruthRule("normal", 0, 2_000_000_000, "Benign", "NORMAL")
    ])
    identities = identity_audit(first, labels)
    populations = population_audit(first, labels)
    assert identities["unique_canonical_identities"] == 1
    assert populations["row_counts"] == {"NORMAL": 1}


def test_source_has_no_model_training_or_scaler_dependency():
    sources = "\n".join(path.read_text() for path in (ROOT / "src/ids/p13").rglob("*.py"))
    for forbidden in (
        "import torch", "ids.p9", "load_state_dict", "optimizer.step", "RobustScaler",
        "StandardScaler", "roc_auc", "anomaly_score",
    ):
        assert forbidden not in sources
