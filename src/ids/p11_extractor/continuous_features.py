"""Candidate packet/flow computations for P11 reference validation.

Only formulas explicitly frozen in repository P0 feature engineering are named
as frozen. Raw Argus/Bro candidates remain unverified until compared row-wise to
an authoritative reference corpus.
"""

from __future__ import annotations

import math
import statistics

from .categorical_features import application_candidates, protocol_token, state_candidate
from .flow_builder import Flow


P0_EPSILON = 1e-8
P0_MIN_DURATION_SECONDS = 1e-6


def _directions(flow: Flow):
    source = [packet for packet in flow.packets if flow.is_originator_packet(packet)]
    destination = [packet for packet in flow.packets if not flow.is_originator_packet(packet)]
    return source, destination


def _iat_ms(packets) -> list[float]:
    return [(right.timestamp_ns - left.timestamp_ns) / 1_000_000.0
            for left, right in zip(packets, packets[1:])]


def _mean_or_zero(values: list[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _jitter_candidate(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _tcp_handshake(flow: Flow) -> tuple[float | None, float | None, float | None]:
    if flow.key.protocol_number != 6:
        return None, None, None
    syn_time = syn_ack_time = ack_time = None
    for packet in flow.packets:
        flags = packet.tcp_flags or 0
        from_source = flow.is_originator_packet(packet)
        if syn_time is None and from_source and flags & 0x02 and not flags & 0x10:
            syn_time = packet.timestamp_ns
        elif syn_time is not None and syn_ack_time is None and not from_source and flags & 0x12 == 0x12:
            syn_ack_time = packet.timestamp_ns
        elif syn_ack_time is not None and ack_time is None and from_source and flags & 0x10 and not flags & 0x02:
            ack_time = packet.timestamp_ns
            break
    if syn_time is None or syn_ack_time is None:
        return None, None, None
    synack = (syn_ack_time - syn_time) / 1_000_000_000.0
    if ack_time is None:
        return synack, None, None
    ackdat = (ack_time - syn_ack_time) / 1_000_000_000.0
    return synack, ackdat, synack + ackdat


def _first(values):
    return values[0] if values else None


def extract_candidate_features(flow: Flow) -> dict[str, object]:
    """Produce candidates and explicit ``None`` for unverified loss/context fields."""
    source, destination = _directions(flow)
    duration = (flow.end_ns - flow.start_ns) / 1_000_000_000.0
    spkts, dpkts = len(source), len(destination)
    sbytes = sum(packet.ip_total_length for packet in source)
    dbytes = sum(packet.ip_total_length for packet in destination)
    siat, diat = _iat_ms(source), _iat_ms(destination)
    synack, ackdat, tcprtt = _tcp_handshake(flow)
    app = application_candidates(flow)
    raw: dict[str, object] = {
        "dur": duration,
        "proto": protocol_token(flow),
        "service": app["service_candidate"],
        "state": state_candidate(flow),
        "spkts": spkts, "dpkts": dpkts,
        "sbytes": sbytes, "dbytes": dbytes,
        "rate": (spkts + dpkts) / duration if duration > 0 else None,
        "sttl": _first([packet.ttl for packet in source]),
        "dttl": _first([packet.ttl for packet in destination]),
        "sload": 8.0 * sbytes / duration if duration > 0 else None,
        "dload": 8.0 * dbytes / duration if duration > 0 else None,
        "sloss": None, "dloss": None,
        "sinpkt": _mean_or_zero(siat), "dinpkt": _mean_or_zero(diat),
        "sjit": _jitter_candidate(siat), "djit": _jitter_candidate(diat),
        "swin": _first([packet.tcp_window for packet in source if packet.tcp_window is not None]),
        "dwin": _first([packet.tcp_window for packet in destination if packet.tcp_window is not None]),
        "stcpb": _first([packet.tcp_seq for packet in source if packet.tcp_seq is not None]),
        "dtcpb": _first([packet.tcp_seq for packet in destination if packet.tcp_seq is not None]),
        "tcprtt": tcprtt, "synack": synack, "ackdat": ackdat,
        "smean": sbytes / spkts if spkts else None,
        "dmean": dbytes / dpkts if dpkts else None,
        "trans_depth": app["http_transaction_depth_candidate"],
        "response_body_len": app["http_response_body_len_candidate"],
        "is_ftp_login": app["ftp_login_success_candidate"],
        "ct_ftp_cmd": app["ftp_command_count_candidate"],
        "ct_flw_http_mthd": app["http_method_count"],
        "is_sm_ips_ports": int(flow.originator.address == flow.responder.address and
                               flow.originator.port == flow.responder.port),
    }
    for name in ("ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
                 "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"):
        raw[name] = None
    return raw


def add_frozen_engineered_features(row: dict[str, object]) -> dict[str, object]:
    """Apply documented P0 formulas in float64, without filling missing inputs."""
    result = dict(row)

    def available(*names: str) -> bool:
        return all(result.get(name) is not None for name in names)

    if available("sbytes", "dbytes"):
        total = float(result["sbytes"]) + float(result["dbytes"]) + P0_EPSILON
        result.update(bytes_ratio=float(result["sbytes"]) / total,
                      log_total_bytes=math.log1p(total),
                      log_sbytes=math.log1p(max(float(result["sbytes"]), 0.0)),
                      log_dbytes=math.log1p(max(float(result["dbytes"]), 0.0)))
    if available("spkts", "dpkts"):
        total = float(result["spkts"]) + float(result["dpkts"]) + P0_EPSILON
        result.update(pkts_ratio=float(result["spkts"]) / total,
                      log_total_pkts=math.log1p(total))
    if available("sbytes", "dur"):
        duration = max(float(result["dur"]), P0_MIN_DURATION_SECONDS)
        source_bps = float(result["sbytes"]) / duration
        result.update(src_bps=source_bps, log_src_bps=math.log1p(source_bps))
        result["pkt_rate"] = ((float(result["spkts"]) + P0_EPSILON) / duration
                              if result.get("spkts") is not None else None)
    if available("sload", "dload"):
        total = float(result["sload"]) + float(result["dload"]) + P0_EPSILON
        result.update(load_asym=abs(float(result["sload"]) - float(result["dload"])) / total,
                      log_sload=math.log1p(max(float(result["sload"]), 0.0)))
    if available("sttl", "dttl"):
        result.update(ttl_diff=abs(float(result["sttl"]) - float(result["dttl"])),
                      ttl_sum=float(result["sttl"]) + float(result["dttl"]))
    result["loss_rate_src"] = (float(result["sloss"]) /
                               (float(result["spkts"]) + P0_EPSILON)
                               if available("sloss", "spkts") else None)
    result["loss_rate_dst"] = (float(result["dloss"]) /
                               (float(result["dpkts"]) + P0_EPSILON)
                               if available("dloss", "dpkts") else None)
    if available("sjit", "djit"):
        result.update(jit_ratio=float(result["sjit"]) / (float(result["djit"]) + P0_EPSILON),
                      log_sjit=math.log1p(max(float(result["sjit"]), 0.0)))
    if available("synack", "ackdat"):
        result.update(handshake_ratio=float(result["synack"]) /
                      (float(result["ackdat"]) + P0_EPSILON),
                      incomplete_tcp=float(result["synack"] > 0 and result["ackdat"] == 0))
    else:
        result.update(handshake_ratio=None, incomplete_tcp=None)
    return result
