"""P12 design/formula tests; no model, checkpoint, score, or external dataset access."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import (  # noqa: E402
    CONTINUOUS_FEATURES,
    PacketObservation,
    build_artifacts,
    build_portable_flows,
    canonical_flow_identity_v3,
    canonical_json_bytes,
    encode_protocol_token,
    extract_core_features,
    extract_transport_extension,
    portable_connection_state,
    protocol_token,
    validate_artifacts,
)


def packet(
    index: int,
    timestamp_ns: int,
    *,
    src: str = "10.0.0.1",
    dst: str = "10.0.0.2",
    sport: int = 12345,
    dport: int = 443,
    protocol: int = 6,
    length: int = 60,
    flags: int | None = 0x10,
) -> PacketObservation:
    return PacketObservation(
        index, timestamp_ns, 4, src, dst, protocol, sport, dport, length,
        flags if protocol == 6 else None,
    )


def reverse_packet(index: int, timestamp_ns: int, **kwargs) -> PacketObservation:
    return packet(
        index,
        timestamp_ns,
        src="10.0.0.2",
        dst="10.0.0.1",
        sport=443,
        dport=12345,
        **kwargs,
    )


def handshake_flow():
    packets = [
        packet(0, 1_000_000_000, flags=0x02, length=60),
        reverse_packet(1, 1_200_000_000, flags=0x12, length=60),
        packet(2, 1_300_000_000, flags=0x10, length=52),
        packet(3, 1_500_000_000, flags=0x18, length=100),
        reverse_packet(4, 1_800_000_000, flags=0x10, length=60),
    ]
    return build_portable_flows(packets)[0]


def test_core_formula_units_direction_and_dimension():
    values = extract_core_features(handshake_flow())
    assert list(values) == [*CONTINUOUS_FEATURES, "ip_protocol"]
    assert len(values) == 19
    assert values["duration_seconds"] == pytest.approx(0.8)
    assert values["forward_packet_count"] == 3.0
    assert values["backward_packet_count"] == 2.0
    assert values["total_packet_count"] == 5.0
    assert values["forward_byte_count"] == 212.0
    assert values["backward_byte_count"] == 120.0
    assert values["total_byte_count"] == 332.0
    assert values["forward_packet_fraction"] == pytest.approx(0.6)
    assert values["forward_byte_fraction"] == pytest.approx(212 / 332)
    assert values["mean_packet_size_bytes"] == pytest.approx(332 / 5)
    assert values["forward_mean_packet_size_bytes"] == pytest.approx(212 / 3)
    assert values["backward_mean_packet_size_bytes"] == pytest.approx(60.0)
    assert values["packets_per_second"] == pytest.approx(6.25)
    assert values["bytes_per_second"] == pytest.approx(415.0)
    assert values["ip_protocol"] == "IPPROTO_6"


def test_bidirectional_direction_normalization_is_input_order_independent():
    packets = list(handshake_flow().packets)
    left = build_portable_flows(packets)[0]
    right = build_portable_flows(list(reversed(packets)))[0]
    assert left.originator == right.originator
    assert extract_core_features(left) == extract_core_features(right)
    assert extract_transport_extension(left) == extract_transport_extension(right)


def test_idle_timeout_boundary_is_explicit():
    packets = [packet(0, 0), packet(1, 60_000_000_000), packet(2, 120_000_000_001)]
    flows = build_portable_flows(packets)
    assert [len(flow.packets) for flow in flows] == [2, 1]


def test_zero_and_one_packet_edges_and_missing_semantics():
    assert build_portable_flows([]) == []
    values = extract_core_features(build_portable_flows([packet(0, 5)])[0])
    assert values["duration_seconds"] == 0.0
    assert values["forward_packet_count"] == 1.0
    assert values["backward_packet_count"] == 0.0
    assert values["backward_mean_packet_size_bytes"] is None
    for name in (
        "packets_per_second", "bytes_per_second", "forward_packets_per_second",
        "backward_packets_per_second", "forward_bytes_per_second", "backward_bytes_per_second",
    ):
        assert values[name] is None


def test_zero_duration_multi_packet_flow_does_not_use_epsilon():
    flow = build_portable_flows([packet(0, 10), packet(1, 10, length=100)])[0]
    values = extract_core_features(flow)
    assert values["total_packet_count"] == 2.0
    assert values["duration_seconds"] == 0.0
    assert values["packets_per_second"] is None
    assert values["bytes_per_second"] is None


def test_iat_flag_and_handshake_calculations():
    extension = extract_transport_extension(handshake_flow())
    assert extension["flow_iat_mean_seconds"] == pytest.approx(0.2)
    assert extension["flow_iat_std_seconds"] == pytest.approx(0.07071067811865475)
    assert extension["forward_iat_mean_seconds"] == pytest.approx(0.25)
    assert extension["forward_iat_std_seconds"] == pytest.approx(0.05)
    assert extension["backward_iat_mean_seconds"] == pytest.approx(0.6)
    assert extension["backward_iat_std_seconds"] is None
    assert extension["tcp_syn_count"] == 2.0
    assert extension["tcp_ack_count"] == 4.0
    assert extension["tcp_rst_count"] == 0.0
    assert extension["tcp_synack_seconds"] == pytest.approx(0.2)
    assert extension["tcp_ack_after_synack_seconds"] == pytest.approx(0.1)
    assert extension["tcp_handshake_seconds"] == pytest.approx(0.3)


def test_state_machine_transitions_and_precedence():
    assert portable_connection_state(handshake_flow()) == "TCP_ESTABLISHED"
    partial = build_portable_flows([packet(0, 0, flags=0x02)])[0]
    assert portable_connection_state(partial) == "TCP_PARTIAL_HANDSHAKE"
    observed = build_portable_flows([packet(0, 0, flags=0x10)])[0]
    assert portable_connection_state(observed) == "TCP_OBSERVED"
    closed = build_portable_flows([
        packet(0, 0, flags=0x11), reverse_packet(1, 1, flags=0x11),
    ])[0]
    assert portable_connection_state(closed) == "TCP_CLOSED"
    reset = build_portable_flows([
        packet(0, 0, flags=0x11), reverse_packet(1, 1, flags=0x15),
    ])[0]
    assert portable_connection_state(reset) == "TCP_RESET"
    udp = build_portable_flows([packet(0, 0, protocol=17, flags=None)])[0]
    assert portable_connection_state(udp) == "NOT_TCP"
    udp_extension = extract_transport_extension(udp)
    assert udp_extension["tcp_syn_count"] == 0.0
    assert udp_extension["tcp_handshake_seconds"] is None


def test_protocol_vocabulary_and_oov_behavior():
    assert protocol_token(0) == "IPPROTO_0"
    assert protocol_token(255) == "IPPROTO_255"
    assert protocol_token(999) == "IPPROTO_OOV"
    assert encode_protocol_token("IPPROTO_0") == 2
    assert encode_protocol_token("IPPROTO_255") == 257
    assert encode_protocol_token("IPPROTO_06") == 1
    assert encode_protocol_token("tcp") == 1


def test_canonical_identity_is_stable_and_label_score_independent():
    values = extract_core_features(handshake_flow())
    identity = canonical_flow_identity_v3(values)
    decorated = dict(values, label="attack", model_score=99.0, row_index=123)
    assert canonical_flow_identity_v3(decorated) == identity
    reordered = dict(reversed(list(values.items())))
    assert canonical_flow_identity_v3(reordered) == identity
    changed = dict(values, total_byte_count=float(values["total_byte_count"]) + 1.0)
    assert canonical_flow_identity_v3(changed) != identity


def test_serialization_is_reproducible_and_rejects_nonfinite():
    assert canonical_json_bytes({"b": 2, "a": -0.0}) == canonical_json_bytes({"a": 0.0, "b": 2})
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_packet_contract_rejects_ambiguous_observations():
    with pytest.raises(ValueError):
        PacketObservation(0, 0, 4, "10.0.0.1", "10.0.0.2", 17, 1, 2, 60, 0x10)
    with pytest.raises(ValueError):
        PacketObservation(0, 0, 4, "10.0.0.1", "10.0.0.2", 6, 1, 2, 10, 0x02)
    with pytest.raises(ValueError):
        build_portable_flows([packet(0, 0), packet(0, 1)])


def test_complete_61_mapping_compatibility_gate_and_no_training():
    artifacts = build_artifacts(ROOT, "2026-09-23T00:00:00Z")
    validate_artifacts(ROOT, artifacts)
    mapping = artifacts["original_to_v3_mapping.json"]
    assert len(mapping["rows"]) == 61
    assert mapping["keep_exact_count"] == 0
    assert {row["decision"] for row in mapping["rows"]} == {
        "REDEFINE_PORTABLE", "DROP_NONPORTABLE",
    }
    matrix = artifacts["dataset_compatibility_matrix.json"]
    assert matrix["complete_capture_family_count"] == 6
    assert matrix["full_core_schema_implementable"] is True
    assert matrix["structural_imputation_required"] is False
    gate = artifacts["p12_gate.json"]
    assert gate["status"] == "P12_PORTABLE_SCHEMA_PASS"
    assert gate["p9_checkpoint_opened"] is False
    assert gate["model_training_performed"] is False
    assert gate["p7_or_p9_detector_evaluated"] is False
    assert gate["external_dataset_file_opened"] is False


def test_p12_source_has_no_training_or_model_dependency():
    source = (ROOT / "src/ids/p12_portable_schema.py").read_text()
    assert "import torch" not in source
    assert "ids.p9" not in source
    assert "load_state_dict" not in source
    assert "optimizer.step" not in source
    assert "roc_auc" not in source
