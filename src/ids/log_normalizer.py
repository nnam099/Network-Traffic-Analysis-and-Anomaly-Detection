from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class LogNormalizationReport:
    schema: str
    input_columns: int
    output_columns: int
    rows: int
    mapped_columns: dict[str, str]
    derived_columns: list[str]
    missing_core_features: list[str]
    feature_coverage: float

    def as_dict(self) -> dict:
        return {
            "schema": self.schema,
            "rows": self.rows,
            "input_columns": self.input_columns,
            "output_columns": self.output_columns,
            "mapped_columns": self.mapped_columns,
            "derived_columns": self.derived_columns,
            "missing_core_features": self.missing_core_features,
            "feature_coverage": self.feature_coverage,
        }


CORE_FEATURES = [
    "proto", "service", "state", "dur", "sbytes", "dbytes", "spkts", "dpkts",
    "sload", "dload", "sttl", "dttl", "sloss", "dloss", "sintpkt", "dintpkt",
    "tcprtt", "synack", "ackdat", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]


ALIASES = {
    "srcip": [
        "srcip", "src_ip", "source_ip", "source.address", "sourceip", "id.orig_h",
        "srcaddr", "src_addr", "client_ip", "source",
    ],
    "dstip": [
        "dstip", "dst_ip", "destination_ip", "destination.address", "dest_ip",
        "dstaddr", "dst_addr", "id.resp_h", "server_ip", "destination",
    ],
    "sport": [
        "sport", "src_port", "source_port", "source.port", "id.orig_p",
        "srcport", "client_port", "spt",
    ],
    "dsport": [
        "dsport", "dst_port", "destination_port", "destination.port", "dest_port",
        "id.resp_p", "dstport", "server_port", "dpt",
    ],
    "proto": ["proto", "protocol", "transport", "network.transport", "ip_proto"],
    "service": ["service", "app_proto", "application", "app", "protocol_name", "event_type"],
    "state": ["state", "conn_state", "tcp_state", "session_state", "action", "flow_state"],
    "dur": ["dur", "duration", "duration_sec", "elapsed", "flow_duration", "flow_duration_sec"],
    "sbytes": [
        "sbytes", "src_bytes", "bytes_src", "source.bytes", "orig_bytes",
        "flow.bytes_toserver", "bytes_toserver", "sent_bytes",
        "total_length_of_fwd_packets", "subflow_fwd_bytes",
    ],
    "dbytes": [
        "dbytes", "dst_bytes", "bytes_dst", "destination.bytes", "resp_bytes",
        "flow.bytes_toclient", "bytes_toclient", "received_bytes",
        "total_length_of_bwd_packets", "subflow_bwd_bytes",
    ],
    "spkts": [
        "spkts", "src_pkts", "source.packets", "orig_pkts", "flow.pkts_toserver",
        "pkts_toserver", "sent_pkts",
        "total_fwd_packets", "subflow_fwd_packets",
    ],
    "dpkts": [
        "dpkts", "dst_pkts", "destination.packets", "resp_pkts", "flow.pkts_toclient",
        "pkts_toclient", "received_pkts",
        "total_backward_packets", "total_bwd_packets", "subflow_bwd_packets",
    ],
    "sttl": ["sttl", "src_ttl", "ttl", "ip_ttl", "source.ttl"],
    "dttl": ["dttl", "dst_ttl", "destination.ttl"],
    "sloss": ["sloss", "src_loss", "source.loss", "lost_src"],
    "dloss": ["dloss", "dst_loss", "destination.loss", "lost_dst"],
    "sload": ["sload", "src_load", "source.load"],
    "dload": ["dload", "dst_load", "destination.load"],
    "smeansz": ["smeansz", "src_mean_size", "source.mean_packet_size", "fwd_packet_length_mean", "avg_fwd_segment_size"],
    "dmeansz": ["dmeansz", "dst_mean_size", "destination.mean_packet_size", "bwd_packet_length_mean", "avg_bwd_segment_size"],
    "sjit": ["sjit", "src_jitter", "jitter_src"],
    "djit": ["djit", "dst_jitter", "jitter_dst"],
    "sintpkt": ["sintpkt", "src_inter_packet_time", "source.inter_packet_time", "fwd_iat_mean"],
    "dintpkt": ["dintpkt", "dst_inter_packet_time", "destination.inter_packet_time", "bwd_iat_mean"],
    "tcprtt": ["tcprtt", "tcp_rtt", "rtt", "round_trip_time"],
    "synack": ["synack", "syn_ack", "tcp_synack"],
    "ackdat": ["ackdat", "ack_data", "tcp_ackdat"],
    "ct_srv_src": ["ct_srv_src"],
    "ct_srv_dst": ["ct_srv_dst"],
    "ct_dst_ltm": ["ct_dst_ltm"],
    "ct_src_ltm": ["ct_src_ltm"],
    "ct_src_dport_ltm": ["ct_src_dport_ltm"],
    "ct_dst_sport_ltm": ["ct_dst_sport_ltm"],
    "ct_dst_src_ltm": ["ct_dst_src_ltm"],
    "stime": ["stime", "start_time", "start", "ts", "timestamp", "@timestamp", "time"],
    "ltime": ["ltime", "end_time", "end", "endtime"],
    "label": ["label", "is_attack", "event.severity"],
    "attack_cat": ["attack_cat", "attack_category", "category", "alert.signature", "signature"],
}


TOTAL_BYTES_ALIASES = [
    "bytes", "total_bytes", "flow.bytes", "network.bytes", "bytes_total",
]
TOTAL_PKTS_ALIASES = [
    "pkts", "packets", "total_pkts", "flow.pkts", "network.packets", "packets_total",
]
BYTES_RATE_ALIASES = ["bps", "bytes_per_second", "throughput_bps"]
PKT_RATE_ALIASES = ["pps", "packets_per_second", "pkt_rate"]


def normalize_real_world_logs(df_raw: pd.DataFrame, expected_features: Iterable[str] | None = None):
    """
    Normalize common market CSV logs into the UNSW-like flow schema used by the IDS.

    Supported inputs include NetFlow/IPFIX, Zeek conn.log CSV exports, Suricata EVE
    flattened CSV, firewall traffic logs and already-normalized UNSW-NB15 rows.
    """
    original_columns = list(df_raw.columns)
    df = df_raw.copy()
    df.columns = [_clean_col(c) for c in df.columns]

    out = pd.DataFrame(index=df.index)
    mapped: dict[str, str] = {}
    derived: list[str] = []

    for target, aliases in ALIASES.items():
        src = _first_present(df, aliases)
        if src is not None:
            out[target] = df[src]
            mapped[target] = src

    _derive_duration(df, out, mapped, derived)
    _derive_directional_totals(df, out, derived)
    _normalize_cic_units(out, mapped, derived)
    _derive_market_flow_features(out, derived)
    _derive_cic_tcp_state(df, out, mapped, derived)
    _derive_context_counts(out, derived)
    _coerce_types(out)

    for col, default in {
        "proto": "tcp",
        "service": "unknown",
        "state": "UNK",
        "sbytes": 0.0,
        "dbytes": 0.0,
        "spkts": 0.0,
        "dpkts": 0.0,
        "dur": 0.0,
    }.items():
        if col not in out.columns:
            out[col] = default
            derived.append(col)

    expected = [c for c in (expected_features or CORE_FEATURES) if c in CORE_FEATURES]
    if not expected:
        expected = CORE_FEATURES
    missing = [c for c in expected if c not in out.columns]
    covered = sum(1 for c in expected if c in out.columns)
    coverage = covered / len(expected)

    report = LogNormalizationReport(
        schema=_detect_schema(set(df.columns)),
        input_columns=len(original_columns),
        output_columns=len(out.columns),
        rows=len(out),
        mapped_columns=mapped,
        derived_columns=sorted(set(derived)),
        missing_core_features=missing,
        feature_coverage=round(float(coverage), 4),
    )
    return out, report


def _clean_col(col) -> str:
    return str(col).strip().lower().replace(" ", "_").replace("-", "_")


def _first_present(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        clean = _clean_col(name)
        if clean in df.columns:
            return clean
    return None


def _num(series, default=0.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(default).astype(float)


def _derive_duration(df: pd.DataFrame, out: pd.DataFrame, mapped: dict, derived: list[str]) -> None:
    if "dur" in out.columns:
        out["dur"] = _num(out["dur"])
        if mapped.get("dur") == "flow_duration":
            out["dur"] = out["dur"] / 1_000_000.0
            derived.append("dur")
            return
        # Some exporters use milliseconds or microseconds for flow duration.
        med = float(out["dur"].median()) if len(out) else 0.0
        if med > 100000:
            out["dur"] = out["dur"] / 1_000_000.0
            derived.append("dur")
        elif med > 1000:
            out["dur"] = out["dur"] / 1000.0
            derived.append("dur")
        return

    start = _first_present(df, ALIASES["stime"])
    end = _first_present(df, ALIASES["ltime"])
    if start and end:
        start_ts = pd.to_datetime(df[start], errors="coerce", utc=True)
        end_ts = pd.to_datetime(df[end], errors="coerce", utc=True)
        out["dur"] = (end_ts - start_ts).dt.total_seconds().fillna(0).clip(lower=0)
        mapped["stime"] = start
        mapped["ltime"] = end
        derived.append("dur")


def _derive_directional_totals(df: pd.DataFrame, out: pd.DataFrame, derived: list[str]) -> None:
    if "sbytes" not in out.columns or "dbytes" not in out.columns:
        total_col = _first_present(df, TOTAL_BYTES_ALIASES)
        if total_col:
            total = _num(df[total_col])
            if "sbytes" not in out.columns and "dbytes" in out.columns:
                out["sbytes"] = (total - _num(out["dbytes"])).clip(lower=0)
            elif "dbytes" not in out.columns and "sbytes" in out.columns:
                out["dbytes"] = (total - _num(out["sbytes"])).clip(lower=0)
            else:
                out["sbytes"] = total * 0.5
                out["dbytes"] = total * 0.5
            derived.extend(["sbytes", "dbytes"])

    if "spkts" not in out.columns or "dpkts" not in out.columns:
        total_col = _first_present(df, TOTAL_PKTS_ALIASES)
        if total_col:
            total = _num(df[total_col])
            if "spkts" not in out.columns and "dpkts" in out.columns:
                out["spkts"] = (total - _num(out["dpkts"])).clip(lower=0)
            elif "dpkts" not in out.columns and "spkts" in out.columns:
                out["dpkts"] = (total - _num(out["spkts"])).clip(lower=0)
            else:
                out["spkts"] = np.ceil(total * 0.5)
                out["dpkts"] = np.floor(total * 0.5)
            derived.extend(["spkts", "dpkts"])


def _normalize_cic_units(out: pd.DataFrame, mapped: dict, derived: list[str]) -> None:
    # CIC-IDS2017 IAT fields are in microseconds; UNSW-style sintpkt/dintpkt are seconds-like.
    for col in ["sintpkt", "dintpkt", "sjit", "djit"]:
        src = mapped.get(col, "")
        if col in out.columns and ("iat" in src or src.endswith("_std")):
            out[col] = _num(out[col]) / 1_000_000.0
            derived.append(col)


def _derive_market_flow_features(out: pd.DataFrame, derived: list[str]) -> None:
    eps = 1e-8
    dur = _num(out["dur"]).clip(lower=1e-6) if "dur" in out.columns else pd.Series(1.0, index=out.index)

    for col in ["sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "sttl", "dttl", "sloss", "dloss", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat"]:
        if col in out.columns:
            out[col] = _num(out[col])

    if "sbytes" in out.columns and "sload" not in out.columns:
        out["sload"] = _num(out["sbytes"]) * 8.0 / dur
        derived.append("sload")
    if "dbytes" in out.columns and "dload" not in out.columns:
        out["dload"] = _num(out["dbytes"]) * 8.0 / dur
        derived.append("dload")

    if "sbytes" in out.columns and "spkts" in out.columns and "smeansz" not in out.columns:
        out["smeansz"] = _num(out["sbytes"]) / (_num(out["spkts"]) + eps)
        derived.append("smeansz")
    if "dbytes" in out.columns and "dpkts" in out.columns and "dmeansz" not in out.columns:
        out["dmeansz"] = _num(out["dbytes"]) / (_num(out["dpkts"]) + eps)
        derived.append("dmeansz")

    if "spkts" in out.columns and "sintpkt" not in out.columns:
        out["sintpkt"] = dur / (_num(out["spkts"]).clip(lower=1.0))
        derived.append("sintpkt")
    if "dpkts" in out.columns and "dintpkt" not in out.columns:
        out["dintpkt"] = dur / (_num(out["dpkts"]).clip(lower=1.0))
        derived.append("dintpkt")

    for col in ["sttl", "dttl", "sloss", "dloss", "tcprtt", "synack", "ackdat"]:
        if col not in out.columns:
            out[col] = 0.0
            derived.append(col)


def _derive_cic_tcp_state(df: pd.DataFrame, out: pd.DataFrame, mapped: dict, derived: list[str]) -> None:
    if "state" in out.columns and mapped.get("state"):
        return
    syn = _num(df["syn_flag_count"]) if "syn_flag_count" in df.columns else pd.Series(0, index=df.index)
    fin = _num(df["fin_flag_count"]) if "fin_flag_count" in df.columns else pd.Series(0, index=df.index)
    rst = _num(df["rst_flag_count"]) if "rst_flag_count" in df.columns else pd.Series(0, index=df.index)
    ack = _num(df["ack_flag_count"]) if "ack_flag_count" in df.columns else pd.Series(0, index=df.index)
    if any(c in df.columns for c in ["syn_flag_count", "fin_flag_count", "rst_flag_count", "ack_flag_count"]):
        out["state"] = np.select(
            [rst > 0, fin > 0, (syn > 0) & (ack > 0), syn > 0],
            ["RST", "FIN", "CON", "SYN"],
            default="CON",
        )
        derived.append("state")


def _derive_context_counts(out: pd.DataFrame, derived: list[str]) -> None:
    rows = len(out)
    if rows == 0:
        return

    def count_by(cols: list[str], name: str) -> None:
        if name in out.columns:
            return
        if all(c in out.columns for c in cols):
            out[name] = out.groupby(cols, dropna=False)[cols[0]].transform("size").astype(float)
        else:
            out[name] = 1.0
        derived.append(name)

    count_by(["srcip", "service"], "ct_srv_src")
    count_by(["dstip", "service"], "ct_srv_dst")
    count_by(["dstip"], "ct_dst_ltm")
    count_by(["srcip"], "ct_src_ltm")
    count_by(["srcip", "dsport"], "ct_src_dport_ltm")
    count_by(["dstip", "sport"], "ct_dst_sport_ltm")
    count_by(["srcip", "dstip"], "ct_dst_src_ltm")


def _coerce_types(out: pd.DataFrame) -> None:
    for col in out.columns:
        if col in {"srcip", "dstip", "proto", "service", "state", "attack_cat"}:
            out[col] = out[col].astype(str).fillna("unknown").str.strip().replace({"": "unknown"})
        elif col == "label":
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)


def _detect_schema(columns: set[str]) -> str:
    if {"flow_duration", "total_fwd_packets", "total_backward_packets"} <= columns:
        return "cic_ids2017"
    if {"id.orig_h", "id.resp_h"} & columns or {"conn_state", "orig_bytes", "resp_bytes"} <= columns:
        return "zeek_conn"
    if any(c.startswith("flow.") for c in columns) or "alert.signature" in columns:
        return "suricata_eve_flat"
    if {"srcaddr", "dstaddr"} & columns:
        return "netflow_or_cloud_flow"
    if {"src_ip", "dst_ip"} <= columns or {"source_ip", "destination_ip"} <= columns:
        return "firewall_or_flow_csv"
    if {"attack_cat", "sbytes", "dbytes"} <= columns:
        return "unsw_nb15"
    return "generic_csv"
