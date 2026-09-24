#!/usr/bin/env python3
"""Run frozen P13 extraction twice and audit P13C attempt 002."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import CONTINUOUS_FEATURES, build_portable_flows  # noqa: E402
from ids.p13.portable_extractor import extract_rows, semantic_content_hash  # noqa: E402
from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13.validation import validate_row  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    aggregate_corpus_sha256,
    feature_summary,
    file_sha256,
    identity_label_audit,
    join_controlled_labels,
    outcome_name,
    pretty_json_bytes,
)


OUTPUT = ROOT / "results/portable_schema/p13c/attempt_002"
ATTEMPT_ID = "P13C_ATTEMPT_002"
CORPUS_NAME = "P13C_CONTROLLED_CAPTURE"
VERSION = "p13c-controlled-capture-attempt-002-v1"
P12_FREEZE_COMMIT = "720789fb90f36e1413cef47cd96f3126d2b8f3a4"
P12_PROTOCOL_SHA256 = "db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3"
P12_SCHEMA_SHA256 = "db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a"
P13B_BLOCKED_GATE_SHA256 = "be8f2213474592c69b34636a089371a35664acc6d5775d9a148cdc51a39bee95"
P13C_BLOCKED_GATE_SHA256 = "f70e98af206f7c037975b2f97d3215e891645ee97fff5439022ff16c4d179a69"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(pretty_json_bytes(value))


def write_jsonl(name: str, rows: list[dict[str, object]]) -> str:
    path = OUTPUT / name
    payload = b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode()
        for row in rows
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_reservation(reservation: dict, schedule_path: Path) -> Path:
    if reservation["attempt_id"] != ATTEMPT_ID:
        raise ValueError("attempt id mismatch")
    if reservation["corpus_name"] != CORPUS_NAME:
        raise ValueError("corpus name mismatch")
    if reservation["reservation_status"] != "P13C_RAW_SOURCE_RESERVED":
        raise ValueError("raw source has not passed reservation")
    if file_sha256(schedule_path) != reservation["scenario_schedule_sha256"]:
        raise ValueError("scenario schedule hash changed after reservation")
    if len(reservation["raw_files"]) != 1:
        raise ValueError("attempt 002 requires exactly one authoritative raw PCAP")
    for item in reservation["raw_files"]:
        path = ROOT / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError("reserved raw size mismatch")
        if file_sha256(path) != item["sha256"]:
            raise ValueError("reserved raw hash mismatch")
        if path.stat().st_mode & 0o222:
            raise ValueError("reserved raw PCAP is not read-only")
    if aggregate_corpus_sha256(reservation["raw_files"]) != reservation["aggregate_corpus_sha256"]:
        raise ValueError("reserved aggregate corpus hash mismatch")
    return ROOT / reservation["raw_files"][0]["path"]


def main() -> int:
    reservation_path = OUTPUT / "raw_source_reservation.json"
    schedule_path = OUTPUT / "scenario_schedule.json"
    reservation = json.loads(reservation_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    raw_path = verify_reservation(reservation, schedule_path)

    parse_one = read_classic_pcap(raw_path)
    flows_one = build_portable_flows(parse_one.packets)
    rows_one = extract_rows(flows_one)
    parse_two = read_classic_pcap(raw_path)
    flows_two = build_portable_flows(parse_two.packets)
    rows_two = extract_rows(flows_two)
    for row in rows_one + rows_two:
        validate_row(row)
    hash_one = semantic_content_hash(rows_one)
    hash_two = semantic_content_hash(rows_two)
    deterministic = (
        parse_one.packets == parse_two.packets
        and flows_one == flows_two
        and rows_one == rows_two
        and hash_one == hash_two
    )
    write_json(
        "capture_format_audit.json",
        {
            "artifact_type": "P13C_CAPTURE_FORMAT_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "raw_sha256": reservation["raw_files"][0]["sha256"],
            **parse_one.audit,
        },
    )
    for run, parse_result, flows, _rows, content_hash in (
        (1, parse_one, flows_one, rows_one, hash_one),
        (2, parse_two, flows_two, rows_two, hash_two),
    ):
        write_json(
            f"extraction_run_{run}.json",
            {
                "artifact_type": "P13C_EXTRACTION_RUN",
                "attempt_id": ATTEMPT_ID,
                "run": run,
                "source": str(raw_path.relative_to(ROOT)),
                "source_sha256": reservation["raw_files"][0]["sha256"],
                "fresh_raw_parse": True,
                "packet_count": len(parse_result.packets),
                "flow_count": len(flows),
                "semantic_content_sha256": content_hash,
                "continuous_field_count": 18,
                "missing_mask_field_count": 18,
                "categorical_field_count": 1,
                "complete_19_fields": True,
                "scaling_performed": False,
                "structural_imputation_performed": False,
                "labels_accessed_during_extraction": False,
            },
        )
    write_json(
        "determinism_audit.json",
        {
            "artifact_type": "P13C_DETERMINISM_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "status": (
                "P13C_EXTRACTION_DETERMINISM_PASS"
                if deterministic
                else "P13C_EXTRACTION_DETERMINISM_FAIL"
            ),
            "packet_equality": parse_one.packets == parse_two.packets,
            "flow_count_equality": len(flows_one) == len(flows_two),
            "flow_order_and_content_equality": flows_one == flows_two,
            "row_order_and_binary64_content_equality": rows_one == rows_two,
            "run_1_semantic_content_sha256": hash_one,
            "run_2_semantic_content_sha256": hash_two,
        },
    )
    if not deterministic:
        raise SystemExit("P13C_EXTRACTION_BLOCKED: double-run semantic mismatch")

    labels = join_controlled_labels(flows_one, schedule)
    outcomes = [outcome_name(label) for label in labels]
    outcome_counts = Counter(outcomes)
    label_details = [
        {
            "flow_id": row.flow_id,
            "outcome": outcome_name(label),
            "source_label": label.source_label,
            "portable_label": label.portable_label,
            "matched_scenario_ids": list(label.matched_rule_ids),
        }
        for row, label in zip(rows_one, labels, strict=True)
    ]
    write_json(
        "label_join_audit.json",
        {
            "artifact_type": "P13C_LABEL_JOIN_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "join_basis": schedule["join_basis"],
            "feature_values_used_for_join": False,
            "model_outputs_used_for_join": False,
            "majority_vote_used": False,
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "details": label_details,
        },
    )

    buckets = {
        name: [] for name in ("NORMAL", "ATTACK", "UNMATCHED", "AMBIGUOUS")
    }
    for row, label in zip(rows_one, labels, strict=True):
        if label.status == "MATCHED":
            buckets[str(label.portable_label)].append(row)
        else:
            buckets[label.status].append(row)
    populations = {
        name: {
            "rows": len(items),
            "canonical_identities": len(
                {row.canonical_identity_v3 for row in items}
            ),
        }
        for name, items in buckets.items()
    }
    scenario_population = {}
    for scenario in schedule["scenarios"]:
        scenario_id = scenario["scenario_id"]
        selected = [
            (row, label)
            for row, label in zip(rows_one, labels, strict=True)
            if scenario_id in label.matched_rule_ids
        ]
        scenario_population[scenario_id] = {
            "portable_label": scenario["portable_label"],
            "flows": len(selected),
            "canonical_identities": len(
                {row.canonical_identity_v3 for row, _ in selected}
            ),
            "matched": sum(label.status == "MATCHED" for _, label in selected),
            "ambiguous": sum(label.status == "AMBIGUOUS" for _, label in selected),
            "protocol_distribution": dict(
                sorted(Counter(row.ip_protocol for row, _ in selected).items())
            ),
        }
    write_json(
        "population_audit.json",
        {
            "artifact_type": "P13C_POPULATION_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "populations": populations,
            "per_scenario": scenario_population,
            "rows_silently_dropped": 0,
        },
    )

    identity = identity_label_audit(rows_one, labels)
    write_json(
        "identity_multiplicity_audit.json",
        {
            "artifact_type": "P13C_IDENTITY_MULTIPLICITY_AUDIT",
            "attempt_id": ATTEMPT_ID,
            **identity,
        },
    )
    write_json(
        "cross_label_identity_audit.json",
        {
            "artifact_type": "P13C_CROSS_LABEL_IDENTITY_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "normal_only_identity_count": identity["normal_only_identity_count"],
            "attack_only_identity_count": identity["attack_only_identity_count"],
            "mixed_identity_count": identity["mixed_identity_count"],
            "mixed_flow_instance_count": identity["mixed_flow_count"],
            "mixed_matched_flow_share": identity["mixed_matched_flow_share"],
            "identity_review_required": identity["identity_review_required"],
            "review_rule": identity["review_rule"],
        },
    )

    feature_groups = {
        "ALL": feature_summary(rows_one),
        "NORMAL": feature_summary(buckets["NORMAL"]),
        "ATTACK": feature_summary(buckets["ATTACK"]),
    }
    for group in feature_groups.values():
        for index, name in enumerate(CONTINUOUS_FEATURES):
            group[name]["finite_value_violations"] = sum(
                row.continuous[index] is not None
                and not math.isfinite(float(row.continuous[index]))
                for row in rows_one
            )
    write_json(
        "feature_sanity_audit.json",
        {
            "artifact_type": "P13C_FEATURE_SANITY_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "groups": feature_groups,
            "mathematical_constraint_violations": 0,
            "feature_selection_performed": False,
        },
    )
    protocol_counts = dict(sorted(Counter(row.ip_protocol for row in rows_one).items()))
    write_json(
        "protocol_audit.json",
        {
            "artifact_type": "P13C_PROTOCOL_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "canonical_token_counts": protocol_counts,
            "token_inferred_from_ports": False,
            "vocabulary_learning_performed": False,
            "rare_protocols_dropped": False,
        },
    )
    write_json(
        "scenario_coverage.json",
        {
            "artifact_type": "P13C_SCENARIO_COVERAGE",
            "attempt_id": ATTEMPT_ID,
            "scenarios": scenario_population,
        },
    )

    allowed = {"127.0.0.2", "127.0.0.3"}
    endpoint_violations = [
        {
            "capture_index": packet.capture_index,
            "source": packet.src_address,
            "destination": packet.dst_address,
        }
        for packet in parse_one.packets
        if {packet.src_address, packet.dst_address} != allowed
    ]
    write_json(
        "safety_scope_audit.json",
        {
            "artifact_type": "P13C_SAFETY_SCOPE_AUDIT",
            "attempt_id": ATTEMPT_ID,
            "capture_interface": "lo",
            "declared_endpoints": sorted(allowed),
            "packet_records_checked": len(parse_one.packets),
            "endpoint_scope_violations": endpoint_violations,
            "external_destination_detected": bool(endpoint_violations),
            "status": "PASS" if not endpoint_violations else "FAIL",
        },
    )

    portable_output = []
    provenance_output = []
    for row, flow, label in zip(rows_one, flows_one, labels, strict=True):
        portable_output.append(
            {
                "flow_id": row.flow_id,
                "continuous": list(row.continuous),
                "missing_mask": list(row.missing_mask),
                "ip_protocol": row.ip_protocol,
                "canonical_identity_v3": row.canonical_identity_v3,
                "source_label": label.source_label,
                "portable_label": label.portable_label,
                "join_outcome": outcome_name(label),
            }
        )
        first = flow.packets[0]
        provenance_output.append(
            {
                "flow_id": row.flow_id,
                "capture_filename": raw_path.name,
                "start_ns": flow.start_ns,
                "end_ns": flow.end_ns,
                "source_address": first.src_address,
                "destination_address": first.dst_address,
                "source_port": first.src_port,
                "destination_port": first.dst_port,
                "ip_protocol_number": first.ip_protocol,
                "matched_scenario_ids": list(label.matched_rule_ids),
            }
        )
    portable_hash = write_jsonl("portable_rows.jsonl", portable_output)
    provenance_hash = write_jsonl(
        "private_flow_provenance.jsonl", provenance_output
    )

    execution = json.loads((OUTPUT / "scenario_execution_log.json").read_text())
    execution_pass = (
        execution["capture_started_before_scenario_traffic"]
        and execution["all_scenarios_executed_once"]
        and execution["all_scenarios_succeeded"]
        and execution["all_scenarios_within_predeclared_interval"]
    )
    label_join_pass = (
        execution_pass
        and populations["NORMAL"]["rows"] > 0
        and populations["ATTACK"]["rows"] > 0
        and populations["UNMATCHED"]["rows"] == 0
        and populations["AMBIGUOUS"]["rows"] == 0
    )
    if not label_join_pass:
        status = "P13C_LABEL_JOIN_BLOCKED"
    elif identity["identity_review_required"]:
        status = "P13C_IDENTITY_REVIEW_REQUIRED"
    else:
        status = "P13C_CONTROLLED_SOURCE_PASS"

    model_access = {
        "artifact_type": "P13C_MODEL_ACCESS_AUDIT",
        "attempt_id": ATTEMPT_ID,
        "p9_checkpoint_opened": False,
        "p9_model_evaluated": False,
        "p7_model_evaluated": False,
        "optimizer_steps": 0,
        "checkpoints_created": 0,
        "anomaly_scores": 0,
        "auroc": None,
        "average_precision": None,
        "threshold_fit": False,
        "scaler_fit": False,
    }
    write_json("model_access_audit.json", model_access)
    artifact_names = [
        path.name
        for path in OUTPUT.iterdir()
        if path.is_file()
        and path.name not in {
            "p13c_gate.json",
            "portable_rows.jsonl",
            "private_flow_provenance.jsonl",
        }
    ]
    artifact_hashes = {
        name: file_sha256(OUTPUT / name) for name in sorted(artifact_names)
    }
    artifact_hashes.update(
        {
            "portable_rows.jsonl": portable_hash,
            "private_flow_provenance.jsonl": provenance_hash,
        }
    )
    gate = {
        "artifact_type": "P13C_GATE",
        "attempt_id": ATTEMPT_ID,
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "status": status,
        "created_at_utc": utc_now(),
        "p12_freeze_commit": P12_FREEZE_COMMIT,
        "p12_protocol_sha256": P12_PROTOCOL_SHA256,
        "p12_schema_sha256": P12_SCHEMA_SHA256,
        "previous_p13b_blocked_gate_sha256": P13B_BLOCKED_GATE_SHA256,
        "previous_p13c_blocked_gate_sha256": P13C_BLOCKED_GATE_SHA256,
        "schedule_sha256": file_sha256(schedule_path),
        "raw_sha256": reservation["raw_files"][0]["sha256"],
        "raw_aggregate_sha256": reservation["aggregate_corpus_sha256"],
        "semantic_content_sha256": hash_one,
        "artifact_sha256": artifact_hashes,
        "complete_19_field_extraction": True,
        "deterministic_double_run": deterministic,
        "label_join_pass": label_join_pass,
        "identity_review_required": identity["identity_review_required"],
        "safety_scope_pass": not endpoint_violations,
        **{
            key: model_access[key]
            for key in (
                "optimizer_steps",
                "checkpoints_created",
                "anomaly_scores",
                "auroc",
                "average_precision",
                "threshold_fit",
                "scaler_fit",
            )
        },
        "historical_detector_opened": False,
        "readiness": (
            "CONTROLLED_EXTRACTION_VALIDATION_COMPLETE"
            if status == "P13C_CONTROLLED_SOURCE_PASS"
            else "REVIEW_OR_REMEDIATION_REQUIRED"
        ),
    }
    write_json("p13c_gate.json", gate)
    print(status)
    print(f"flows={len(rows_one)}")
    print(f"semantic_content_sha256={hash_one}")
    print(f"population={populations}")
    print(f"mixed_identity_count={identity['mixed_identity_count']}")
    print(
        "No neural network was trained and no historical detector was evaluated "
        "during this P13C attempt."
    )
    return 0 if status in {
        "P13C_CONTROLLED_SOURCE_PASS",
        "P13C_IDENTITY_REVIEW_REQUIRED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
