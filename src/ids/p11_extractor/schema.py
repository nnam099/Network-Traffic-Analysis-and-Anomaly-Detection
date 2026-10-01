"""Machine-readable P11 contract derived from frozen repository semantics."""

from __future__ import annotations

import hashlib
import json


VERSION = "p11-semantic-extractor-v1"
OFFICIAL_DATASET_URL = "https://research.unsw.edu.au/projects/unsw-nb15-dataset"
PAPER_URL = "https://doi.org/10.1109/MilCIS.2015.7348942"

CLASSIFICATIONS = (
    "PACKET_DIRECT", "FLOW_DERIVED", "TCP_STATE_DERIVED", "SERVICE_DERIVED",
    "TEMPORAL_CONTEXT", "LAST_N_CONNECTION_CONTEXT", "APPLICATION_PROTOCOL_DERIVED",
    "UNKNOWN_OR_UNVERIFIED",
)

LAST_N = {
    "ct_state_ttl", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
}
APPLICATION = {
    "ct_flw_http_mthd", "ct_ftp_cmd", "is_ftp_login", "trans_depth", "response_body_len",
}
TCP_STATE = {
    "swin", "dwin", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
    "handshake_ratio", "incomplete_tcp", "state",
}
TEMPORAL = {"sinpkt", "dinpkt", "sjit", "djit", "jit_ratio", "log_sjit"}
PACKET = {"sttl", "dttl", "proto"}
FLOW = {
    "dur", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload", "sloss", "dloss",
    "smean", "dmean", "rate", "is_sm_ips_ports", "bytes_ratio", "log_total_bytes",
    "log_sbytes", "log_dbytes", "pkts_ratio", "log_total_pkts", "src_bps",
    "log_src_bps", "pkt_rate", "load_asym", "log_sload", "ttl_diff", "ttl_sum",
    "loss_rate_src", "loss_rate_dst",
}

UNITS = {
    "dur": "seconds", "sbytes": "bytes", "dbytes": "bytes", "spkts": "packets",
    "dpkts": "packets", "sttl": "IP hops", "dttl": "IP hops", "sloss": "packets",
    "dloss": "packets", "sload": "bits/second", "dload": "bits/second",
    "rate": "packets/second", "sinpkt": "milliseconds", "dinpkt": "milliseconds",
    "sjit": "milliseconds", "djit": "milliseconds", "swin": "bytes",
    "dwin": "bytes", "stcpb": "TCP sequence number", "dtcpb": "TCP sequence number",
    "tcprtt": "seconds", "synack": "seconds", "ackdat": "seconds",
    "smean": "bytes/packet", "dmean": "bytes/packet", "response_body_len": "bytes",
    "trans_depth": "transactions", "ct_ftp_cmd": "commands", "is_ftp_login": "binary",
    "is_sm_ips_ports": "binary", "ct_flw_http_mthd": "count",
    "bytes_ratio": "ratio", "log_total_bytes": "log(bytes)", "log_sbytes": "log(bytes)",
    "log_dbytes": "log(bytes)", "pkts_ratio": "ratio", "log_total_pkts": "log(packets)",
    "src_bps": "bytes/second", "log_src_bps": "log(bytes/second)",
    "pkt_rate": "packets/second", "load_asym": "ratio", "log_sload": "log(bits/second)",
    "ttl_diff": "IP hops", "ttl_sum": "IP hops", "loss_rate_src": "ratio",
    "loss_rate_dst": "ratio", "jit_ratio": "ratio", "log_sjit": "log(milliseconds)",
    "handshake_ratio": "ratio", "incomplete_tcp": "binary",
}

DIRECTION = {
    **{name: "source-to-destination" for name in (
        "sbytes", "spkts", "sttl", "sloss", "sload", "sinpkt", "sjit", "swin", "stcpb", "smean")},
    **{name: "destination-to-source" for name in (
        "dbytes", "dpkts", "dttl", "dloss", "dload", "dinpkt", "djit", "dwin", "dtcpb", "dmean")},
    "synack": "source SYN to destination SYN-ACK",
    "ackdat": "destination SYN-ACK to source ACK",
}

P0_FORMULAS = {
    "bytes_ratio": "sbytes / (sbytes + dbytes + 1e-8)",
    "log_total_bytes": "log1p(sbytes + dbytes + 1e-8)",
    "log_sbytes": "log1p(max(sbytes, 0))", "log_dbytes": "log1p(max(dbytes, 0))",
    "pkts_ratio": "spkts / (spkts + dpkts + 1e-8)",
    "log_total_pkts": "log1p(spkts + dpkts + 1e-8)",
    "src_bps": "sbytes / max(dur, 1e-6)",
    "log_src_bps": "log1p(sbytes / max(dur, 1e-6))",
    "pkt_rate": "(spkts + 1e-8) / max(dur, 1e-6)",
    "load_asym": "abs(sload - dload) / (sload + dload + 1e-8)",
    "log_sload": "log1p(max(sload, 0))", "ttl_diff": "abs(sttl - dttl)",
    "ttl_sum": "sttl + dttl", "loss_rate_src": "sloss / (spkts + 1e-8)",
    "loss_rate_dst": "dloss / (dpkts + 1e-8)",
    "jit_ratio": "sjit / (djit + 1e-8)", "log_sjit": "log1p(max(sjit, 0))",
    "handshake_ratio": "synack / (ackdat + 1e-8)",
    "incomplete_tcp": "float((synack > 0) and (ackdat == 0))",
}

DEPENDENCIES = {
    "bytes_ratio": ["sbytes", "dbytes"], "log_total_bytes": ["sbytes", "dbytes"],
    "log_sbytes": ["sbytes"], "log_dbytes": ["dbytes"],
    "pkts_ratio": ["spkts", "dpkts"], "log_total_pkts": ["spkts", "dpkts"],
    "src_bps": ["sbytes", "dur"], "log_src_bps": ["sbytes", "dur"],
    "pkt_rate": ["spkts", "dur"], "load_asym": ["sload", "dload"],
    "log_sload": ["sload"], "ttl_diff": ["sttl", "dttl"], "ttl_sum": ["sttl", "dttl"],
    "loss_rate_src": ["sloss", "spkts"], "loss_rate_dst": ["dloss", "dpkts"],
    "jit_ratio": ["sjit", "djit"], "log_sjit": ["sjit"],
    "handshake_ratio": ["synack", "ackdat"], "incomplete_tcp": ["synack", "ackdat"],
}


def classification(name: str) -> str:
    if name in LAST_N:
        return "LAST_N_CONNECTION_CONTEXT"
    if name in APPLICATION:
        return "APPLICATION_PROTOCOL_DERIVED"
    if name == "service":
        return "SERVICE_DERIVED"
    if name in TCP_STATE:
        return "TCP_STATE_DERIVED"
    if name in TEMPORAL:
        return "TEMPORAL_CONTEXT"
    if name in PACKET:
        return "PACKET_DIRECT"
    if name in FLOW:
        return "FLOW_DERIVED"
    return "UNKNOWN_OR_UNVERIFIED"


def _flow_semantics(name: str) -> str:
    if name in P0_FORMULAS:
        return "Repository P0 post-flow engineered feature; depends on upstream raw semantics."
    if name in LAST_N:
        return "Historical recent-connection statistic; exact C# ordering/window behavior is unavailable."
    if name in APPLICATION:
        return "Merged from historical Bro HTTP/FTP logs or application-aware post-processing."
    if name == "state":
        return "Historical Argus connection state token; not interchangeable with Zeek conn_state."
    if name == "service":
        return "Historical application service classification associated with Bro/Argus merge."
    return "Bidirectional flow semantic produced by the historical Argus/Bro/post-processing pipeline."


def _window(name: str) -> str:
    if name in LAST_N:
        return "nominal last 100 connections; exact inclusion, ordering and boundary rules unverified"
    if name in TEMPORAL or name in FLOW or name in TCP_STATE or name in APPLICATION:
        return "current reconstructed bidirectional flow"
    return "single packet or flow representative, historical aggregation rule unverified"


def feature_spec(name: str, kind: str) -> dict:
    categorical = kind == "categorical"
    provenance = [
        {"type": "frozen_repository_schema", "path": "results/data_quality/p3/schema_v2.json"},
        {"type": "official_dataset_page", "url": OFFICIAL_DATASET_URL},
        {"type": "dataset_generation_paper", "url": PAPER_URL},
    ]
    if name in P0_FORMULAS:
        provenance.append({"type": "frozen_repository_formula", "path": "src/ids/dataset.py",
                           "symbol": "engineer_features"})
    formula = P0_FORMULAS.get(name)
    if formula is None:
        formula = "UNVERIFIED_HISTORICAL_TOOL_OR_POSTPROCESSING_SEMANTIC"
    return {
        "name": name, "type": "UTF-8 categorical token" if categorical else "finite float64",
        "units": "categorical token" if categorical else UNITS.get(name, "count"),
        "directionality": DIRECTION.get(name, "bidirectional/derived or not direction-specific"),
        "aggregation_window": _window(name), "flow_semantics": _flow_semantics(name),
        "source_dependency": DEPENDENCIES.get(name, [
            "raw packets", "historical Argus/Bro output or unavailable C# post-processing"
        ]),
        "formula": formula,
        "missing_value_behavior": "fail closed; use explicit null in intermediate output; never zero-impute",
        "categorical_semantics": (
            "normalize only at the later frozen P9 vocabulary boundary; extractor emits semantic token"
            if categorical else None
        ),
        "reference_provenance": provenance,
        "semantic_class": classification(name),
        "exactness_claim": "none until row-wise reference validation",
    }


def build_schema_contract(frozen_schema: dict, source_hashes: dict[str, str]) -> dict:
    contract = frozen_schema["contract"]
    continuous = contract["continuous_features"]
    categorical = contract["categorical_features"]
    fields = [feature_spec(name, "continuous") for name in continuous]
    fields.extend(feature_spec(name, "categorical") for name in categorical)
    payload = {
        "version": VERSION, "frozen_p3_schema_sha256": frozen_schema["sha256"],
        "semantic_field_count": len(fields), "continuous_count": len(continuous),
        "categorical_count": len(categorical), "continuous_dtype": "finite IEEE-754 binary64",
        "comparison_stage": "before P9 RobustScaler", "float32_forbidden_in_p11": True,
        "field_order": [*continuous, *categorical], "fields": fields,
        "source_code_sha256": source_hashes,
    }
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode()
    payload["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def semantic_inventory(contract: dict) -> dict:
    items = [{"name": field["name"], "type": field["type"],
              "semantic_class": field["semantic_class"],
              "verification_state": "UNKNOWN_OR_UNVERIFIED",
              "sensitive": field["semantic_class"] in {
                  "TCP_STATE_DERIVED", "SERVICE_DERIVED", "TEMPORAL_CONTEXT",
                  "LAST_N_CONNECTION_CONTEXT", "APPLICATION_PROTOCOL_DERIVED",
              }} for field in contract["fields"]]
    counts = {name: sum(item["semantic_class"] == name for item in items)
              for name in CLASSIFICATIONS}
    return {"version": VERSION, "schema_contract_sha256": contract["contract_sha256"],
            "classification_vocabulary": list(CLASSIFICATIONS), "classification_counts": counts,
            "features": items}
