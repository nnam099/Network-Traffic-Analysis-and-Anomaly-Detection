"""P10A metadata-only external-source compatibility gate.

This module never opens an external dataset or a P9 checkpoint.  It records a
conservative, source-documentation-based schema audit and fails closed when a
candidate cannot reproduce every frozen P9 semantic input.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path


P8_PROTOCOL_SHA256 = "62916073ef6048503415d45e2605bbf475b729e032c838a60cc7e94c43768216"
P9_PROTOCOL_SHA256 = "c1c7352a9ca546a3e8e456b112d09fd42f29243e205a8760703b27f68b5baf87"
P9_FREEZE_COMMIT = "3dd5f12a6bf3948efff817654111347d1fedc7ce"
P9_ARTIFACT_AGGREGATE_SHA256 = "0f11f518391a66a0ea320e0b3558ea5de9b7440910014f57457f0f0ae9908ed7"
P10_VERSION = "p10a-external-source-compatibility-v1"

CLASSIFICATIONS = {
    "EXACT", "SEMANTICALLY_EQUIVALENT", "DERIVABLE", "MISSING", "INCOMPATIBLE"
}
ADMISSIBLE = {"EXACT", "SEMANTICALLY_EQUIVALENT", "DERIVABLE"}
TARGETS = (
    "Analysis", "Backdoors", "DoS", "Exploits", "Fuzzers", "Generic",
    "Reconnaissance", "Shellcode", "Worms",
)

OFFICIAL_SOURCES = {
    "cic_ids_2017": {
        "dataset": "CIC-IDS2017",
        "official_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
        "schema_reference": "https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt",
        "extractor": "CICFlowMeter; more than 80 bidirectional-flow fields",
        "raw_formats": ["PCAP", "CICFlowMeter CSV"],
        "capture": "2017-07-03 through 2017-07-07; independent UNB/CIC testbed",
        "license": "Publicly available for research; citation requested by the official page",
        "labels": ["BENIGN", "FTP-Patator", "SSH-Patator", "DoS", "Heartbleed",
                   "Web Attack", "Infiltration", "Bot", "PortScan", "DDoS"],
        "profile": "cicflowmeter",
        "family_mapping_quality": "partial",
        "identity_quality": "flow identifiers/timestamps exist in labeled-flow products; raw artifact not reserved",
        "support": "five capture days with benign and attack traffic; row counts not audited",
    },
    "cse_cic_ids_2018": {
        "dataset": "CSE-CIC-IDS2018",
        "official_url": "https://www.unb.ca/cic/datasets/ids-2018.html",
        "schema_reference": "https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt",
        "extractor": "CICFlowMeter-V3; 80 extracted fields",
        "raw_formats": ["PCAP/system logs", "CICFlowMeter CSV on AWS"],
        "capture": "independent CSE/CIC testbed with 420 machines and 30 servers",
        "license": "Official public dataset page; exact redistribution terms not frozen here",
        "labels": ["Benign", "Brute Force", "Heartbleed", "Botnet", "DoS",
                   "DDoS", "Web attacks", "Infiltration"],
        "profile": "cicflowmeter",
        "family_mapping_quality": "partial",
        "identity_quality": "flow records available; exact raw file inventory not reserved",
        "support": "normal traffic and seven attack scenarios; row counts not audited",
    },
    "ton_iot_network": {
        "dataset": "TON_IoT network",
        "official_url": "https://research.unsw.edu.au/projects/toniot-datasets",
        "schema_reference": "https://research.unsw.edu.au/projects/toniot-datasets",
        "extractor": "Zeek/Bro network logs; 44 network fields plus label/type",
        "raw_formats": ["PCAP", "Zeek log", "Zeek-derived CSV"],
        "capture": "new Industry 4.0/IoT testbed, distinct from UNSW-NB15",
        "license": "Academic use granted; commercial use requires author permission; citation required",
        "labels": ["normal", "backdoor", "ddos", "dos", "injection", "mitm",
                   "password", "ransomware", "scanning", "xss"],
        "profile": "ton_zeek",
        "family_mapping_quality": "partial",
        "identity_quality": "source/destination tuple and timestamp fields documented; raw artifact not reserved",
        "support": "normal plus nine attack types; row/identity counts not audited",
    },
    "bot_iot": {
        "dataset": "Bot-IoT",
        "official_url": "https://research.unsw.edu.au/projects/bot-iot-dataset",
        "schema_reference": "https://arxiv.org/abs/1811.00701",
        "extractor": "Argus CSV plus engineered flow statistics",
        "raw_formats": ["PCAP", "Argus", "CSV"],
        "capture": "separate UNSW Canberra Cyber Range botnet/IoT environment",
        "license": "Academic use granted; commercial use requires author agreement; citation required",
        "labels": ["Normal", "DDoS", "DoS", "OS Scan", "Service Scan",
                   "Keylogging", "Data exfiltration"],
        "profile": "bot_argus",
        "family_mapping_quality": "partial",
        "identity_quality": "flow tuple/time fields documented; raw artifact not reserved",
        "support": "72M+ records; official 5% subset about 3M records; exact class counts not audited",
    },
    "cic_ddos_2019": {
        "dataset": "CIC-DDoS2019",
        "official_url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
        "schema_reference": "https://github.com/ahlashkari/CICFlowMeter/blob/master/ReadMe.txt",
        "extractor": "CICFlowMeter-V3; more than 80 bidirectional-flow fields",
        "raw_formats": ["PCAP/event logs", "CICFlowMeter CSV"],
        "capture": "independent two-day UNB/CIC DDoS testbed",
        "license": "Official public dataset page; citation terms apply",
        "labels": ["BENIGN", "NTP", "DNS", "LDAP", "MSSQL", "NetBIOS", "SNMP",
                   "SSDP", "UDP", "UDP-Lag", "WebDDoS", "SYN", "TFTP", "PortScan"],
        "profile": "cicflowmeter",
        "family_mapping_quality": "DDoS-specific; not equivalent to UNSW-NB15 DoS",
        "identity_quality": "flow identifiers/timestamps documented; raw artifact not reserved",
        "support": "benign plus twelve DDoS types across two days; row counts not audited",
    },
}

DERIVED = {
    "bytes_ratio": ("float32(sbytes / (sbytes + dbytes + 1e-8))", ["sbytes", "dbytes"]),
    "log_total_bytes": ("float32(log1p(sbytes + dbytes + 1e-8))", ["sbytes", "dbytes"]),
    "log_sbytes": ("float32(log1p(max(sbytes, 0)))", ["sbytes"]),
    "log_dbytes": ("float32(log1p(max(dbytes, 0)))", ["dbytes"]),
    "pkts_ratio": ("float32(spkts / (spkts + dpkts + 1e-8))", ["spkts", "dpkts"]),
    "log_total_pkts": ("float32(log1p(spkts + dpkts + 1e-8))", ["spkts", "dpkts"]),
    "src_bps": ("float32(sbytes / max(dur_seconds, 1e-6))", ["sbytes", "dur"]),
    "log_src_bps": ("float32(log1p(sbytes / max(dur_seconds, 1e-6)))", ["sbytes", "dur"]),
    "pkt_rate": ("float32((spkts + 1e-8) / max(dur_seconds, 1e-6))", ["spkts", "dur"]),
}

PROFILE_FIELDS = {
    "cicflowmeter": {
        "dbytes": ("Total Length of Bwd Packets", "bytes", None),
        "dinpkt": ("Bwd IAT Mean", "microseconds", "float32(Bwd IAT Mean / 1e6)"),
        "dmean": ("Bwd Packet Length Mean", "bytes", None),
        "dpkts": ("Total Backward Packets", "packets", None),
        "dur": ("Flow Duration", "microseconds", "float32(Flow Duration / 1e6)"),
        "rate": ("Flow Packets/s", "packets/second", None),
        "sbytes": ("Total Length of Fwd Packets", "bytes", None),
        "sinpkt": ("Fwd IAT Mean", "microseconds", "float32(Fwd IAT Mean / 1e6)"),
        "smean": ("Fwd Packet Length Mean", "bytes", None),
        "spkts": ("Total Fwd Packets", "packets", None),
        "proto": ("Protocol", "token", "normalize then frozen per-cell OOV lookup"),
    },
    "ton_zeek": {
        "dbytes": ("dst_bytes", "payload bytes", None),
        "dpkts": ("dst_pkts", "packets", None),
        "dur": ("duration", "seconds", None),
        "response_body_len": ("http_response_body_len", "bytes", None),
        "sbytes": ("src_bytes", "payload bytes", None),
        "spkts": ("src_pkts", "packets", None),
        "trans_depth": ("http_trans_depth", "transactions", None),
        "proto": ("proto", "token", "normalize then frozen per-cell OOV lookup"),
        "service": ("service", "Zeek-detected service token", "normalize then frozen per-cell OOV lookup"),
    },
    "bot_argus": {
        "dbytes": ("dbytes", "bytes", None),
        "dpkts": ("dpkts", "packets", None),
        "dur": ("dur", "seconds", None),
        "rate": ("rate", "packets/second", None),
        "sbytes": ("sbytes", "bytes", None),
        "spkts": ("spkts", "packets", None),
        "proto": ("proto", "Argus token", "normalize then frozen per-cell OOV lookup"),
        "state": ("state", "Argus state token", "normalize then frozen per-cell OOV lookup"),
    },
}

PROFILE_INCOMPATIBLE = {
    "cicflowmeter": {
        "swin": "CIC initial forward-window bytes is not UNSW/Argus source TCP window semantics",
        "dwin": "CIC initial backward-window bytes is not UNSW/Argus destination TCP window semantics",
        "service": "CIC ML CSV has no authoritative application-service field; port inference is forbidden",
        "state": "CIC TCP flag counters do not reproduce the UNSW/Argus connection-state token",
    },
    "ton_zeek": {
        "state": "Zeek conn_state is a different state machine from the frozen UNSW/Argus state field",
        "ct_flw_http_mthd": "http_method presence is not the UNSW rolling flow-method count",
    },
    "bot_argus": {
        "service": "Bot-IoT CSV does not provide the frozen UNSW service semantic",
    },
}

SPECIAL_DERIVED = {
    "ton_zeek": {
        "is_sm_ips_ports": {
            "source_columns": ["src_ip", "dst_ip", "src_port", "dst_port"],
            "formula": "float32((src_ip == dst_ip) and (src_port == dst_port))",
            "units": "binary indicator",
        }
    }
}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_reason(feature: str, profile: str) -> str:
    if feature.startswith("ct_"):
        return "candidate schema lacks the documented UNSW last-100-connection context statistic"
    if feature in {"sttl", "dttl", "ttl_diff", "ttl_sum", "ct_state_ttl"}:
        return "candidate flow product lacks direction-specific TTL semantics required by P9"
    if feature in {"stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
                   "handshake_ratio", "incomplete_tcp"}:
        return "candidate schema lacks the frozen Argus TCP sequence/handshake timing semantic"
    if feature in {"sloss", "dloss", "loss_rate_src", "loss_rate_dst"}:
        return "candidate schema lacks direction-specific packet-loss counts"
    if feature in {"sjit", "djit", "jit_ratio", "log_sjit"}:
        return "candidate timing dispersion is absent or not equivalent to Argus jitter"
    if feature in {"sload", "dload", "load_asym", "log_sload"}:
        return "candidate schema does not expose equivalent direction-specific Argus load values"
    if feature in {"is_ftp_login", "ct_ftp_cmd"}:
        return "candidate schema lacks equivalent FTP login/command semantics"
    return f"no documented {profile} field reproduces this frozen P9 semantic"


def feature_mapping(schema: dict) -> dict:
    continuous = schema["contract"]["continuous_features"]
    categorical = schema["contract"]["categorical_features"]
    expected = [*continuous, *categorical]
    candidates = {}
    for candidate_id, candidate in OFFICIAL_SOURCES.items():
        profile = candidate["profile"]
        rows = []
        direct = PROFILE_FIELDS[profile]
        incompatible = PROFILE_INCOMPATIBLE[profile]
        special = SPECIAL_DERIVED.get(profile, {})
        for feature in expected:
            if feature in direct:
                source, units, conversion = direct[feature]
                row = {
                    "p9_feature": feature,
                    "kind": "categorical" if feature in categorical else "continuous",
                    "classification": "SEMANTICALLY_EQUIVALENT",
                    "source_columns": [source],
                    "units": units,
                    "conversion": conversion,
                    "rationale": (
                        "Same high-level quantity, but extractor/version/direction conventions are not "
                        "proven byte-for-byte identical to the frozen UNSW/Argus/Bro pipeline."
                    ),
                }
            elif feature in special:
                spec = special[feature]
                row = {
                    "p9_feature": feature, "kind": "continuous",
                    "classification": "DERIVABLE", **spec,
                    "dtype": "float32 before frozen per-cell RobustScaler; float64 at model boundary",
                    "clipping": "none", "missing_policy": "fail closed",
                    "rationale": "Exact P9 formula can be applied to documented source metadata.",
                }
            elif feature in DERIVED and all(dep in direct for dep in DERIVED[feature][1]):
                formula, dependencies = DERIVED[feature]
                row = {
                    "p9_feature": feature, "kind": "continuous",
                    "classification": "DERIVABLE", "source_columns": dependencies,
                    "formula": formula, "units": "dimensionless or formula-defined",
                    "dtype": "float32 before frozen per-cell RobustScaler; float64 at model boundary",
                    "clipping": "only max/clip explicitly present in the frozen formula",
                    "missing_policy": "fail closed",
                    "rationale": "Formula copied from the frozen repository feature-engineering contract.",
                }
            elif feature in incompatible:
                row = {
                    "p9_feature": feature,
                    "kind": "categorical" if feature in categorical else "continuous",
                    "classification": "INCOMPATIBLE", "source_columns": [],
                    "rationale": incompatible[feature],
                }
            else:
                row = {
                    "p9_feature": feature,
                    "kind": "categorical" if feature in categorical else "continuous",
                    "classification": "MISSING", "source_columns": [],
                    "rationale": _missing_reason(feature, profile),
                }
            rows.append(row)
        counts = dict(sorted(Counter(item["classification"] for item in rows).items()))
        blockers = [item["p9_feature"] for item in rows if item["classification"] not in ADMISSIBLE]
        candidates[candidate_id] = {
            "dataset": candidate["dataset"], "mapping": rows,
            "classification_counts": counts, "blocking_features": blockers,
            "gate": "EXTERNAL_SCHEMA_COMPATIBLE" if not blockers else "EXTERNAL_SCHEMA_INCOMPATIBLE",
        }
    return {
        "version": P10_VERSION,
        "frozen_schema_sha256": schema["sha256"],
        "required_continuous_count": len(continuous),
        "required_categorical_count": len(categorical),
        "classification_vocabulary": sorted(CLASSIFICATIONS),
        "exact_name_is_not_semantic_equivalence": True,
        "candidate_mappings": candidates,
    }


def _label_rows(candidate_id: str) -> list[dict]:
    maps = {
        "cic_ids_2017": {
            "Backdoors": (["Infiltration"], "BROADER_THAN_P9_TARGET"),
            "DoS": (["DoS", "DDoS"], "BROADER_THAN_P9_TARGET"),
            "Exploits": (["Heartbleed", "Web Attack"], "NARROWER_THAN_P9_TARGET"),
            "Reconnaissance": (["PortScan"], "NARROWER_THAN_P9_TARGET"),
        },
        "cse_cic_ids_2018": {
            "Backdoors": (["Infiltration"], "BROADER_THAN_P9_TARGET"),
            "DoS": (["DoS", "DDoS"], "BROADER_THAN_P9_TARGET"),
            "Exploits": (["Heartbleed", "Web attacks"], "NARROWER_THAN_P9_TARGET"),
        },
        "ton_iot_network": {
            "Backdoors": (["backdoor"], "DIRECTLY_COMPARABLE"),
            "DoS": (["dos"], "DIRECTLY_COMPARABLE"),
            "Exploits": (["injection", "xss"], "NARROWER_THAN_P9_TARGET"),
            "Reconnaissance": (["scanning"], "NARROWER_THAN_P9_TARGET"),
        },
        "bot_iot": {
            "DoS": (["DoS"], "DIRECTLY_COMPARABLE"),
            "Reconnaissance": (["OS Scan", "Service Scan"], "NARROWER_THAN_P9_TARGET"),
        },
        "cic_ddos_2019": {
            "DoS": (["NTP", "DNS", "LDAP", "MSSQL", "NetBIOS", "SNMP", "SSDP",
                     "UDP", "UDP-Lag", "WebDDoS", "SYN", "TFTP"], "SEMANTICALLY_DIFFERENT"),
            "Reconnaissance": (["PortScan"], "NARROWER_THAN_P9_TARGET"),
        },
    }
    selected = maps[candidate_id]
    rows = []
    for target in TARGETS:
        labels, relation = selected.get(target, ([], "UNMAPPABLE"))
        rows.append({
            "p9_target": target, "external_labels": labels, "relationship": relation,
            "target_status": "NOT_COMPARABLE",
            "eligible_for_evaluation": False,
            "rationale": (
                "Candidate failed the frozen 61-input schema gate; label similarity cannot override it."
                if labels else "No documented external category supports this P9 family."
            ),
        })
    return rows


def label_mapping() -> dict:
    return {
        "version": P10_VERSION, "contract_state": "PROVISIONAL_INACTIVE",
        "normal_mapping": "candidate benign/normal only; background semantics remain dataset-specific",
        "ddos_is_not_automatically_unsw_dos": True,
        "evaluation_mode_if_a_source_later_passes": "A_GENERIC_UNSEEN_ATTACK_VS_EXTERNAL_NORMAL",
        "candidate_target_mappings": {
            candidate_id: {"dataset": spec["dataset"], "targets": _label_rows(candidate_id)}
            for candidate_id, spec in OFFICIAL_SOURCES.items()
        },
    }


def candidate_audit(mapping: dict, audited_at_utc: str) -> dict:
    candidates = []
    for candidate_id, spec in OFFICIAL_SOURCES.items():
        mapped = mapping["candidate_mappings"][candidate_id]
        candidates.append({
            "candidate_id": candidate_id,
            **{key: value for key, value in spec.items() if key != "profile"},
            "schema_coverage": mapped["classification_counts"],
            "schema_gate": mapped["gate"],
            "independence": "metadata supports a distinct capture/testbed; no UNSW-NB15 incorporation found",
            "categorical_compatibility": "frozen vocabularies only; unseen tokens would map to OOV; no extension",
            "reproducibility": "official source is documented, but no raw file inventory/hash was acquired",
            "decision": "REJECT_SCHEMA_INCOMPATIBLE",
        })
    return {
        "version": P10_VERSION, "audited_at_utc": audited_at_utc,
        "audit_basis": "official landing pages and published extractor/schema documentation only",
        "external_raw_data_opened": False, "model_scores_computed": False,
        "selection_criterion": "schema compatibility, independence, reproducibility, label clarity, support",
        "performance_guided_selection": False, "candidates": candidates,
        "selected_external_source": None,
        "status": "P10_EXTERNAL_SOURCE_BLOCKED",
        "blocking_reason": "Every candidate has mandatory frozen P9 inputs marked MISSING or INCOMPATIBLE.",
    }


def identity_contract() -> dict:
    return {
        "version": P10_VERSION, "contract_state": "INACTIVE_NO_COMPATIBLE_SOURCE",
        "identity_version": "p10-semantic-model-input-identity-v1",
        "definition": (
            "SHA-256 over canonical finite pre-scaled values for the 58 frozen continuous semantics "
            "plus normalized proto/service/state tokens, with field names and length prefixes"
        ),
        "forbidden_inputs": ["row index", "model score", "threshold result", "label"],
        "duplicate_audit": "within and across raw files before any scoring",
        "timestamp_policy": "timestamp excluded from identity; retained only as provenance",
        "activation_condition": "all 61 feature semantics pass and raw source is cryptographically reserved",
        "frozen_for_execution": False,
    }


def reservation(source_sha256: str) -> dict:
    return {
        "version": P10_VERSION, "reservation_status": "NOT_RESERVED",
        "selected_external_source": None, "source_url": None, "acquisition_date_utc": None,
        "raw_files": [], "aggregate_dataset_sha256": None, "license_version": None,
        "preprocessing_script": "src/ids/p10_external_source.py",
        "preprocessing_script_sha256": source_sha256,
        "external_raw_data_opened": False,
        "reason": "No candidate passed schema compatibility; downloading or hashing raw data was not justified.",
    }


def population_audit() -> dict:
    return {
        "version": P10_VERSION, "audit_status": "NOT_RUN_NO_RESERVED_SOURCE",
        "total_rows": None, "normal_rows": None, "attack_rows": None,
        "candidate_identity_count": None, "per_attack_category": {},
        "mapped_categories": [], "unmapped_categories": [],
        "categorical_oov_counts": {"proto": None, "service": None, "state": None},
        "anomaly_scores_computed": False,
        "reason": "Population/OOV counts require a reserved compatible raw source; none exists.",
    }


def preregistration(audited_at_utc: str, artifact_hashes: dict) -> dict:
    return {
        "version": P10_VERSION, "created_at_utc": audited_at_utc,
        "status": "P10_EXTERNAL_SOURCE_BLOCKED", "selected_external_source": None,
        "p8_protocol_sha256": P8_PROTOCOL_SHA256,
        "p9_protocol_sha256": P9_PROTOCOL_SHA256,
        "p9_freeze_commit": P9_FREEZE_COMMIT,
        "p9_artifact_aggregate_sha256": P9_ARTIFACT_AGGREGATE_SHA256,
        "artifact_sha256": artifact_hashes,
        "external_schema_gate": "EXTERNAL_SCHEMA_INCOMPATIBLE",
        "identity_contract_frozen": False, "label_mapping_frozen": False,
        "external_source_reserved": False, "external_raw_data_opened": False,
        "p9_checkpoint_opened": False, "p9_model_evaluated": False,
        "anomaly_scores_computed": False, "external_metrics_computed": False,
        "model_changed": False, "scorer_changed": False, "threshold_changed": False,
        "readiness": "P10B_NOT_AUTHORIZED",
    }


def verify_upstream(root: Path) -> dict:
    schema = json.loads((root / "results/data_quality/p3/schema_v2.json").read_text())
    p9 = json.loads((root / "results/loafo/p9/p9_matrix.json").read_text())
    if schema["sha256"] != "f3ecf12891e5b0d9b5e1fe38a22c23720a71eaf1d5a47b763911df363b3b36d7":
        raise ValueError("frozen P3/P9 semantic schema hash mismatch")
    if p9.get("status") != "P9_DEVELOPMENT_PASS" or p9.get("protocol_sha256") != P9_PROTOCOL_SHA256:
        raise ValueError("frozen P9 development gate mismatch")
    if p9.get("p8_protocol_sha256") != P8_PROTOCOL_SHA256 or p9.get("cell_count") != 27:
        raise ValueError("frozen P8 binding or P9 cell count mismatch")
    if p9.get("official_test_access") is not False or p9.get("target_held_out_scoring") is not False:
        raise ValueError("P9 provenance boundary mismatch")
    return schema


def build_artifacts(root: Path, audited_at_utc: str) -> dict[str, dict]:
    schema = verify_upstream(root)
    source_hash = file_sha256(root / "src/ids/p10_external_source.py")
    mappings = feature_mapping(schema)
    artifacts = {
        "feature_mapping.json": mappings,
        "candidate_dataset_audit.json": candidate_audit(mappings, audited_at_utc),
        "label_mapping.json": label_mapping(),
        "identity_contract.json": identity_contract(),
        "external_source_reservation.json": reservation(source_hash),
        "external_population_audit.json": population_audit(),
    }
    hashes = {name: hashlib.sha256(json_bytes(value)).hexdigest()
              for name, value in artifacts.items()}
    artifacts["p10_preregistration.json"] = preregistration(audited_at_utc, hashes)
    return artifacts


def validate_artifacts(root: Path, artifacts: dict[str, dict]) -> None:
    schema = verify_upstream(root)
    expected = set(schema["contract"]["continuous_features"] +
                   schema["contract"]["categorical_features"])
    mappings = artifacts["feature_mapping.json"]
    if mappings["required_continuous_count"] != 58 or mappings["required_categorical_count"] != 3:
        raise ValueError("P10A feature cardinality mismatch")
    for candidate in mappings["candidate_mappings"].values():
        rows = candidate["mapping"]
        if {row["p9_feature"] for row in rows} != expected or len(rows) != 61:
            raise ValueError("candidate does not classify each frozen P9 feature exactly once")
        if any(row["classification"] not in CLASSIFICATIONS or not row.get("rationale") for row in rows):
            raise ValueError("invalid or unexplained feature classification")
        if not candidate["blocking_features"] or candidate["gate"] != "EXTERNAL_SCHEMA_INCOMPATIBLE":
            raise ValueError("P10A must fail closed for documented mandatory schema gaps")
    labels = artifacts["label_mapping.json"]["candidate_target_mappings"]
    if any([row["p9_target"] for row in item["targets"]] != list(TARGETS)
           for item in labels.values()):
        raise ValueError("target coverage matrix mismatch")
    prereg = artifacts["p10_preregistration.json"]
    calculated_hashes = {
        name: hashlib.sha256(json_bytes(value)).hexdigest()
        for name, value in artifacts.items() if name != "p10_preregistration.json"
    }
    if prereg.get("artifact_sha256") != calculated_hashes:
        raise ValueError("P10A artifact hash binding mismatch")
    if artifacts["external_source_reservation.json"].get(
            "preprocessing_script_sha256") != file_sha256(
                root / "src/ids/p10_external_source.py"):
        raise ValueError("P10A preprocessing source binding mismatch")
    if (prereg["status"] != "P10_EXTERNAL_SOURCE_BLOCKED" or
            prereg["external_source_reserved"] is not False or
            prereg["p9_model_evaluated"] is not False or
            prereg["external_metrics_computed"] is not False):
        raise ValueError("P10A fail-closed provenance mismatch")
