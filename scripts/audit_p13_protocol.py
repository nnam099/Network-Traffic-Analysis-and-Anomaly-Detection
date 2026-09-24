#!/usr/bin/env python3
"""Emit the fail-closed P13 source-triage/protocol artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


P12_FREEZE_COMMIT = "720789fb90f36e1413cef47cd96f3126d2b8f3a4"
P12_PROTOCOL_SHA256 = "db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3"
P12_SCHEMA_SHA256 = "db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a"
VERSION = "p13-source-reservation-extraction-v1"


CSE_OBJECTS = [
    ("Wednesday-14-02-2018/pcap.zip", 39_913_353_098, "845cbc33e555f5906ea5c9bc520113ab-2380"),
    ("Thursday-15-02-2018/pcap.zip", 41_283_382_768, "e281e31b104af08f1e670e9bd703ff77-2461"),
    ("Friday-16-02-2018/pcap.zip", 38_535_667_707, "7ef6f23385e8cafe3427e5c612e41eeb-2297"),
    ("Tuesday-20-02-2018/pcap.rar", 44_391_980_380, "8aa242e62fde4eaa43d1ad66d545a213-2646"),
    ("Wednesday-21-02-2018/pcap.zip", 53_462_820_707, "d8e019df7ae51d9fefb558f57cdfb5f9-3187"),
    ("Thursday-22-02-2018/pcap.zip", 50_240_938_251, "23f6eb47cd517efd60aa86fea92c5264-2995"),
    ("Friday-23-02-2018/pcap.zip", 59_093_922_810, "a39a7470586d7fd0930431980a5b0c31-3523"),
    ("Wednesday-28-02-2018/pcap.zip", 53_251_694_487, "b688b1c7c529c8754fe11aec1a963270-3175"),
    ("Thursday-01-03-2018/pcap.zip", 52_358_069_106, "652d07144e22ca627d2362fa9e77c75e-3121"),
    ("Friday-02-03-2018/pcap.zip", 44_789_835_888, "290fa7c40359b9fb8cffe29917bbde36-2670"),
]


def pretty_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifacts(root: Path, timestamp: str) -> dict[str, dict]:
    candidates = [
        {
            "priority": 1, "dataset": "UNSW-NB15", "decision": "BLOCKED_ACCESS_AND_SIZE",
            "official_url": "https://research.unsw.edu.au/projects/unsw-nb15-dataset",
            "raw_pcap_documented": True, "documented_raw_size": "100 GB",
            "observation": "Official download redirects to UNSW SharePoint authentication; no immutable object URL acquired.",
        },
        {
            "priority": 2, "dataset": "CIC-IDS2017", "decision": "BLOCKED_DOWNLOAD_FORM",
            "official_url": "https://www.unb.ca/cic/datasets/ids-2017.html",
            "raw_pcap_documented": True, "documented_raw_size": "7.8-13 GB per capture day",
            "observation": "Official download requires a personal-information form which returned a server error; no raw object URL acquired.",
        },
        {
            "priority": 3, "dataset": "CSE-CIC-IDS2018", "decision": "BLOCKED_UNREASONABLE_AUTOMATIC_ARCHIVE",
            "official_url": "https://www.unb.ca/cic/datasets/ids-2018.html",
            "raw_pcap_documented": True,
            "public_bucket": "s3://cse-cic-ids2018/Original Network Traffic and Log data/",
            "raw_objects": [
                {"key": key, "size_bytes": size, "s3_multipart_etag_not_sha256": etag}
                for key, size, etag in CSE_OBJECTS
            ],
            "minimum_archive_size_bytes": min(item[1] for item in CSE_OBJECTS),
            "observation": "Bucket metadata is public, but each raw archive is 38.5-59.1 GB and exposes no SHA-256. P13 did not download one merely to probe suitability.",
        },
        {
            "priority": 4, "dataset": "TON_IoT network", "decision": "BLOCKED_ACCESS",
            "official_url": "https://research.unsw.edu.au/projects/toniot-datasets",
            "raw_pcap_documented": True,
            "observation": "Official download redirects to UNSW SharePoint authentication; no immutable object URL acquired.",
        },
        {
            "priority": 5, "dataset": "Bot-IoT", "decision": "BLOCKED_ACCESS_AND_SIZE",
            "official_url": "https://research.unsw.edu.au/projects/bot-iot-dataset",
            "raw_pcap_documented": True, "documented_raw_size": "69.3 GB",
            "observation": "Official download redirects to UNSW SharePoint authentication; the documented 5% subset is CSV, not raw PCAP.",
        },
        {
            "priority": 6, "dataset": "CIC-DDoS2019", "decision": "BLOCKED_DOWNLOAD_FORM",
            "official_url": "https://www.unb.ca/cic/datasets/ddos-2019.html",
            "raw_pcap_documented": True,
            "observation": "No direct immutable raw-object URL and SHA-256 were available without the CIC download workflow.",
        },
    ]
    source_candidate_audit = {
        "artifact_type": "P13_SOURCE_CANDIDATE_AUDIT",
        "version": VERSION,
        "audited_at_utc": timestamp,
        "selection_order": [item["dataset"] for item in candidates],
        "selection_criteria": [
            "accessible raw PCAP", "documented labels", "reproducible acquisition",
            "usable capture files", "reasonable size", "clear normal/attack mapping",
            "documented label-join metadata",
        ],
        "performance_guided_selection": False,
        "network_metadata_only": True,
        "external_dataset_file_opened": False,
        "candidates": candidates,
        "selected_source": None,
        "conclusion": "No candidate presently supports a complete immutable reservation without credentials/personal form input or an unapproved 38.5+ GB archive transfer.",
    }
    source_reservation = {
        "artifact_type": "P13_RAW_SOURCE_RESERVATION",
        "version": VERSION,
        "reservation_status": "NOT_RESERVED",
        "dataset_name": None,
        "dataset_version": None,
        "source_url": None,
        "download_timestamp_utc": None,
        "license": None,
        "raw_file_names": [],
        "raw_file_sizes": [],
        "sha256_per_file": {},
        "aggregate_sha256": None,
        "archive_sha256_if_applicable": None,
        "source_documentation_hashes": {},
        "raw_directory_read_only": False,
        "reason": "No candidate passed accessibility, size, and SHA-256 reservation requirements.",
    }
    capture_audit = {
        "artifact_type": "P13_CAPTURE_FORMAT_AUDIT",
        "version": VERSION,
        "status": "NOT_RUN_NO_RESERVED_SOURCE",
        "pcap_format": None,
        "link_layer": None,
        "packet_count": None,
        "capture_duration": None,
        "reason": "Technical capture metadata requires the reserved bytes; S3 object listing metadata is not a capture audit.",
    }
    source_hashes = {
        path: file_sha256(root / path) for path in (
            "src/ids/p13/__init__.py",
            "src/ids/p13/raw_reader/__init__.py",
            "src/ids/p13/raw_reader/classic_pcap.py",
            "src/ids/p13/portable_extractor.py",
            "src/ids/p13/labels.py",
            "src/ids/p13/validation.py",
        )
    }
    extractor_contract = {
        "artifact_type": "P13_EXTRACTOR_CONTRACT",
        "version": VERSION,
        "activation_state": "CANDIDATE_UNACTIVATED_NO_RESERVED_SOURCE",
        "p12_protocol_sha256": P12_PROTOCOL_SHA256,
        "p12_schema_sha256": P12_SCHEMA_SHA256,
        "parser_version": "p13-classic-pcap-ethernet-v1",
        "parser_scope": "classic PCAP 2.4, DLT_EN10MB, IPv4/IPv6 without extension headers, TCP/UDP/other IP",
        "dependency_versions": {"python": ">=3.11", "third_party_packet_parser": None},
        "timestamp_normalization": "integer nanoseconds",
        "packet_order": "timestamp_ns ascending, then unique capture_index ascending",
        "direction": "first observed packet after deterministic ordering is forward",
        "flow_key": "unordered endpoint pair plus IP protocol; labels and dataset identifiers excluded",
        "idle_timeout": "strictly greater than 60 seconds splits; exactly 60 seconds stays in the flow",
        "fragment_policy": "exclude all IPv4 and IPv6 fragments with explicit counters; no implicit reassembly",
        "malformed_policy": "exclude/count packet-level malformation; fail on file-level corruption",
        "unsupported_policy": "exclude/count unsupported packets; fail on unsupported file/link format",
        "output": "18 nullable float64 values + 18 booleans + canonical IPPROTO token + canonical identity v3",
        "structural_imputation": False,
        "scaling": False,
        "source_sha256": source_hashes,
    }
    label_join = {
        "artifact_type": "P13_LABEL_JOIN_CONTRACT",
        "version": VERSION,
        "activation_state": "INACTIVE_NO_RESERVED_SOURCE",
        "pipeline_order": ["raw capture", "flow extraction", "portable features", "canonical identity", "label join"],
        "reference_join": "inclusive flow-start interval plus optional protocol/endpoints/ports",
        "time_tolerance": None,
        "ambiguity_policy": "AMBIGUOUS; never majority vote",
        "unmatched_policy": "UNMATCHED; never silently drop",
        "multi_label_conflict_policy": "AMBIGUOUS",
        "warning": "Exact fields and tolerance must be frozen from selected source documentation after reservation.",
    }
    not_run = {
        "label_taxonomy_audit.json": {
            "artifact_type": "P13_LABEL_TAXONOMY_AUDIT", "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "source_categories": [], "portable_primary_taxonomy": ["NORMAL", "ATTACK"],
            "forced_unsw_family_mapping": False,
        },
        "extraction_run_1.json": {
            "artifact_type": "P13_EXTRACTION_RUN", "run": 1, "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "semantic_content_sha256": None, "rows": None,
        },
        "extraction_run_2.json": {
            "artifact_type": "P13_EXTRACTION_RUN", "run": 2, "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "semantic_content_sha256": None, "rows": None,
        },
        "determinism_audit.json": {
            "artifact_type": "P13_DETERMINISM_AUDIT", "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "expected_status_after_source": "P13_EXTRACTION_DETERMINISM_PASS",
        },
        "feature_sanity_audit.json": {
            "artifact_type": "P13_FEATURE_SANITY_AUDIT", "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "features": {}, "impossible_value_violations": None,
        },
        "identity_audit.json": {
            "artifact_type": "P13_CANONICAL_IDENTITY_AUDIT", "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "rows": None, "unique_canonical_identities": None, "ambiguous_canonical_identity_count": None,
        },
        "population_audit.json": {
            "artifact_type": "P13_POPULATION_AUDIT", "status": "NOT_RUN_NO_RESERVED_SOURCE",
            "normal_rows": None, "attack_rows": None, "unmatched_rows": None, "ambiguous_rows": None,
            "rows_dropped": 0,
        },
    }
    artifacts = {
        "source_candidate_audit.json": source_candidate_audit,
        "source_reservation.json": source_reservation,
        "capture_format_audit.json": capture_audit,
        "extractor_contract.json": extractor_contract,
        "label_join_contract.json": label_join,
        **not_run,
    }
    hashes = {name: hashlib.sha256(pretty_bytes(value)).hexdigest() for name, value in artifacts.items()}
    artifacts["p13_gate.json"] = {
        "artifact_type": "P13_GATE",
        "version": VERSION,
        "status": "P13_SOURCE_BLOCKED",
        "created_at_utc": timestamp,
        "p12_freeze_commit": P12_FREEZE_COMMIT,
        "p12_protocol_sha256": P12_PROTOCOL_SHA256,
        "p12_schema_sha256": P12_SCHEMA_SHA256,
        "artifact_sha256": hashes,
        "selected_source": None,
        "raw_source_reserved": False,
        "raw_source_opened": False,
        "extraction_executed": False,
        "label_join_executed": False,
        "population_constructed": False,
        "optimizer_steps": 0,
        "checkpoints_created": 0,
        "anomaly_scores": 0,
        "auroc": None,
        "threshold": None,
        "historical_model_opened": False,
        "readiness": "RAW_SOURCE_ACCESS_OR_EXPLICIT_LARGE_TRANSFER_DECISION_REQUIRED",
        "blocking_reason": "No raw source could be immutably SHA-256-reserved under the accessible and reasonable-size gate.",
    }
    return artifacts


def validate(artifacts: dict[str, dict]) -> None:
    gate = artifacts["p13_gate.json"]
    calculated = {
        name: hashlib.sha256(pretty_bytes(value)).hexdigest()
        for name, value in artifacts.items() if name != "p13_gate.json"
    }
    if calculated != gate["artifact_sha256"]:
        raise ValueError("P13 artifact binding mismatch")
    if gate["status"] != "P13_SOURCE_BLOCKED":
        raise ValueError("P13 must fail closed without a reservation")
    if artifacts["source_reservation.json"]["reservation_status"] != "NOT_RESERVED":
        raise ValueError("P13 source reservation state mismatch")
    if any((gate["raw_source_opened"], gate["extraction_executed"], gate["label_join_executed"],
            gate["population_constructed"], gate["optimizer_steps"], gate["checkpoints_created"],
            gate["anomaly_scores"], gate["historical_model_opened"])):
        raise ValueError("P13 no-source/no-model boundary failed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = root / "results/portable_schema/p13"
    if args.check:
        artifacts = {path.name: json.loads(path.read_text()) for path in output.glob("*.json")}
    else:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        artifacts = build_artifacts(root, timestamp)
        output.mkdir(parents=True, exist_ok=True)
        for name, value in artifacts.items():
            (output / name).write_bytes(pretty_bytes(value))
        artifacts = {path.name: json.loads(path.read_text()) for path in output.glob("*.json")}
    validate(artifacts)
    print(artifacts["p13_gate.json"]["status"])
    print(artifacts["p13_gate.json"]["blocking_reason"])
    print("No neural network was trained and no historical detector was evaluated during P13.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
