"""Synthetic P11 extractor tests; no P9 model/checkpoint or external corpus access."""

from __future__ import annotations

import ipaddress
from pathlib import Path
import struct
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p11_extractor.audit import build_artifacts, validate_artifacts  # noqa: E402
from ids.p11_extractor.categorical_features import (  # noqa: E402
    application_candidates, protocol_token, state_candidate,
)
from ids.p11_extractor.context_features import (  # noqa: E402
    UnverifiedContextSemantics, historical_unsw_context_features,
)
from ids.p11_extractor.continuous_features import (  # noqa: E402
    add_frozen_engineered_features, extract_candidate_features,
)
from ids.p11_extractor.flow_builder import FlowBuilder  # noqa: E402
from ids.p11_extractor.packet_reader import PacketRecord, PcapFormatError, read_pcap  # noqa: E402
from ids.p11_extractor.serialization import deterministic_json_bytes, serialized_sha256  # noqa: E402
from ids.p11_extractor.validation import Tolerance, compare_continuous  # noqa: E402


def packet(index, timestamp_ns, src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80,
           protocol=6, length=60, ttl=64, flags=0x10, seq=1, ack=1, window=4096, payload=b""):
    return PacketRecord(index, timestamp_ns, 4, src, dst, protocol, sport, dport,
                        length, ttl, flags if protocol == 6 else None,
                        seq if protocol == 6 else None, ack if protocol == 6 else None,
                        window if protocol == 6 else None, payload)


def handshake_packets():
    return [
        packet(0, 1_000_000_000, flags=0x02, seq=100, ack=0),
        packet(1, 1_200_000_000, src="10.0.0.2", dst="10.0.0.1", sport=80, dport=1234,
               flags=0x12, seq=900, ack=101, ttl=63, window=2048),
        packet(2, 1_300_000_000, flags=0x10, seq=101, ack=901),
    ]


def test_packet_ordering_and_bidirectional_grouping_are_deterministic():
    packets = handshake_packets()
    forward = FlowBuilder().build(packets)
    reverse = FlowBuilder().build(list(reversed(packets)))
    assert len(forward) == len(reverse) == 1
    assert [item.capture_index for item in forward[0].packets] == [0, 1, 2]
    assert extract_candidate_features(forward[0]) == extract_candidate_features(reverse[0])


def test_idle_timeout_splits_reused_five_tuple():
    packets = [packet(0, 0), packet(1, 10_000_000_000), packet(2, 80_000_000_000)]
    flows = FlowBuilder(idle_timeout_ns=60_000_000_000).build(packets)
    assert [len(flow.packets) for flow in flows] == [2, 1]


def test_duration_directional_counts_bytes_and_ttl():
    flow = FlowBuilder().build(handshake_packets())[0]
    values = extract_candidate_features(flow)
    assert values["dur"] == pytest.approx(0.3)
    assert (values["spkts"], values["dpkts"]) == (2, 1)
    assert (values["sbytes"], values["dbytes"]) == (120, 60)
    assert (values["sttl"], values["dttl"]) == (64, 63)


def test_packet_loss_fails_closed_instead_of_zero_imputation():
    values = extract_candidate_features(FlowBuilder().build(handshake_packets())[0])
    assert values["sloss"] is None and values["dloss"] is None
    engineered = add_frozen_engineered_features(values)
    assert engineered["loss_rate_src"] is None
    assert engineered["loss_rate_dst"] is None


def test_interpacket_timing_jitter_rate_and_load_candidates():
    packets = [packet(0, 0), packet(1, 100_000_000), packet(2, 300_000_000)]
    values = extract_candidate_features(FlowBuilder().build(packets)[0])
    assert values["sinpkt"] == pytest.approx(150.0)
    assert values["sjit"] == pytest.approx(50.0)
    assert values["rate"] == pytest.approx(10.0)
    assert values["sload"] == pytest.approx(4800.0)


def test_zero_duration_has_no_silent_raw_rate_epsilon():
    packets = [packet(0, 0), packet(1, 0)]
    values = extract_candidate_features(FlowBuilder().build(packets)[0])
    assert values["dur"] == 0.0
    assert values["rate"] is None and values["sload"] is None


def test_tcp_handshake_timing_and_local_state():
    flow = FlowBuilder().build(handshake_packets())[0]
    values = extract_candidate_features(flow)
    assert values["synack"] == pytest.approx(0.2)
    assert values["ackdat"] == pytest.approx(0.1)
    assert values["tcprtt"] == pytest.approx(0.3)
    assert state_candidate(flow) == "LOCAL_ESTABLISHED"


def test_protocol_and_payload_evidenced_service_without_port_guessing():
    http = FlowBuilder().build([packet(0, 0, dport=4444, payload=b"GET / HTTP/1.1\r\n\r\n")])[0]
    opaque = FlowBuilder().build([packet(0, 0, dport=80, payload=b"\x01\x02")])[0]
    assert protocol_token(http) == "tcp"
    assert application_candidates(http)["service_candidate"] == "http"
    assert application_candidates(opaque)["service_candidate"] is None


def test_missing_application_data_is_explicit():
    flow = FlowBuilder().build([packet(0, 0, payload=b"")])[0]
    values = extract_candidate_features(flow)
    assert values["service"] is None
    assert values["is_ftp_login"] is None


def test_context_features_fail_closed():
    with pytest.raises(UnverifiedContextSemantics):
        historical_unsw_context_features([])


def test_deterministic_serialization_and_nonfinite_rejection():
    left = deterministic_json_bytes({"b": 2, "a": [1, None]})
    right = deterministic_json_bytes({"a": [1, None], "b": 2})
    assert left == right
    assert serialized_sha256({"b": 2, "a": [1, None]}) == serialized_sha256({"a": [1, None], "b": 2})
    with pytest.raises(ValueError):
        deterministic_json_bytes({"bad": float("nan")})


def _pcap_frame() -> bytes:
    ethernet = b"\x00" * 12 + b"\x08\x00"
    src = ipaddress.IPv4Address("10.0.0.1").packed
    dst = ipaddress.IPv4Address("10.0.0.2").packed
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 40, 1, 0, 64, 6, 0, src, dst)
    tcp = struct.pack("!HHIIBBHHH", 1234, 80, 100, 0, 0x50, 0x02, 4096, 0, 0)
    return ethernet + ip + tcp


def test_classic_pcap_reader(tmp_path):
    frame = _pcap_frame()
    raw = (b"\xd4\xc3\xb2\xa1" + struct.pack("<HHIIII", 2, 4, 0, 0, 65535, 1) +
           struct.pack("<IIII", 1, 250_000, len(frame), len(frame)) + frame)
    path = tmp_path / "one.pcap"
    path.write_bytes(raw)
    packets = list(read_pcap(path))
    assert len(packets) == 1
    assert packets[0].timestamp_ns == 1_250_000_000
    assert (packets[0].src_port, packets[0].dst_port, packets[0].tcp_flags) == (1234, 80, 0x02)


def test_pcapng_is_not_silently_accepted(tmp_path):
    path = tmp_path / "bad.pcapng"
    path.write_bytes(b"\x0a\x0d\x0d\x0a" + b"\x00" * 20)
    with pytest.raises(PcapFormatError):
        list(read_pcap(path))


def test_feature_specific_comparison_metrics():
    result = compare_continuous([1.0, 2.0, None], [1.0, 2.0001, 3.0],
                                Tolerance(0.001, 0.0, "test"))
    assert result["exact_match_count"] == 1
    assert result["within_tolerance_count"] == 2
    assert result["nan_mismatch_count"] == 1
    assert result["absolute_error"]["max"] == pytest.approx(0.0001)


def test_61_field_contract_and_reference_gate():
    artifacts = build_artifacts(ROOT, "2026-09-23T00:00:00Z")
    validate_artifacts(ROOT, artifacts)
    contract = artifacts["schema_contract.json"]
    assert len(contract["fields"]) == 61
    assert contract["continuous_count"] == 58 and contract["categorical_count"] == 3
    assert all(field["type"] != "float32" for field in contract["fields"])
    gate = artifacts["p11_gate.json"]
    assert gate["status"] == "P11_REFERENCE_CORPUS_BLOCKED"
    assert gate["p9_checkpoint_opened"] is False
    assert gate["p9_model_evaluated"] is False
    assert gate["new_external_pcap_reserved"] is False


def test_extractor_package_has_no_torch_or_p9_imports():
    sources = "\n".join(path.read_text() for path in
                        (ROOT / "src/ids/p11_extractor").glob("*.py"))
    assert "import torch" not in sources
    assert "ids.p9" not in sources
    assert "roc_auc" not in sources
