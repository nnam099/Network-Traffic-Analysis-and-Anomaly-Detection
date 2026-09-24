#!/usr/bin/env python3
"""Verify, extract twice, label, and audit the reserved P13B corpus."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import build_portable_flows  # noqa: E402
from ids.p13.portable_extractor import extract_rows, semantic_content_hash  # noqa: E402
from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13.validation import validate_row  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    CORPUS_NAME, VERSION, feature_summary, file_sha256, identity_label_audit,
    join_controlled_labels, outcome_name, pretty_json_bytes, verify_raw_reservation,
)


OUTPUT = ROOT / "results/portable_schema/p13b"
P12_FREEZE_COMMIT = "720789fb90f36e1413cef47cd96f3126d2b8f3a4"
P12_PROTOCOL_SHA256 = "db6405fcac5d8fd94e406660ea60c22853a9c0073762df0e3f21881fe806fec3"
P12_SCHEMA_SHA256 = "db430498147a0601f0d1db9ef5579e210ec9795fa871043d0ab1bf5d328adb7a"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(pretty_json_bytes(value))


def write_jsonl(name: str, rows: list[dict[str, object]]) -> str:
    path = OUTPUT / name
    payload = b"".join(
        (json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
        for row in rows
    )
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def main() -> int:
    reservation_path = OUTPUT / "raw_source_reservation.json"
    schedule_path = OUTPUT / "scenario_schedule.json"
    reservation = json.loads(reservation_path.read_text())
    schedule = json.loads(schedule_path.read_text())
    verify_raw_reservation(ROOT, reservation, schedule_path)
    raw_path = ROOT / reservation["raw_files"][0]["path"]

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
        parse_one.packets == parse_two.packets and flows_one == flows_two and
        rows_one == rows_two and hash_one == hash_two
    )
    write_json("capture_format_audit.json", {
        "artifact_type": "P13B_CAPTURE_FORMAT_AUDIT",
        "corpus_name": CORPUS_NAME,
        "raw_sha256": reservation["raw_files"][0]["sha256"],
        **parse_one.audit,
    })
    write_json("extraction_run_1.json", {
        "artifact_type": "P13B_EXTRACTION_RUN", "run": 1,
        "packet_count": len(parse_one.packets), "flow_count": len(flows_one),
        "semantic_content_sha256": hash_one, "complete_19_fields": True,
    })
    write_json("extraction_run_2.json", {
        "artifact_type": "P13B_EXTRACTION_RUN", "run": 2,
        "packet_count": len(parse_two.packets), "flow_count": len(flows_two),
        "semantic_content_sha256": hash_two, "complete_19_fields": True,
    })
    write_json("determinism_audit.json", {
        "artifact_type": "P13B_DETERMINISM_AUDIT",
        "status": "P13B_EXTRACTION_DETERMINISM_PASS" if deterministic else "P13B_EXTRACTION_DETERMINISM_FAIL",
        "packet_equality": parse_one.packets == parse_two.packets,
        "flow_equality": flows_one == flows_two,
        "row_equality": rows_one == rows_two,
        "run_1_semantic_content_sha256": hash_one,
        "run_2_semantic_content_sha256": hash_two,
    })
    if not deterministic:
        raise SystemExit("P13B_EXTRACTION_BLOCKED: double-run semantic mismatch")

    labels = join_controlled_labels(flows_one, schedule)
    outcomes = [outcome_name(label) for label in labels]
    outcome_counts = Counter(outcomes)
    label_details = []
    for row, label in zip(rows_one, labels, strict=True):
        label_details.append({
            "flow_id": row.flow_id,
            "outcome": outcome_name(label),
            "source_label": label.source_label,
            "portable_label": label.portable_label,
            "matched_scenario_ids": list(label.matched_rule_ids),
        })
    write_json("label_join_audit.json", {
        "artifact_type": "P13B_LABEL_JOIN_AUDIT",
        "join_basis": schedule["join_basis"],
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "majority_vote_used": False,
        "details": label_details,
    })

    buckets = {name: [] for name in ("NORMAL", "ATTACK", "UNMATCHED", "AMBIGUOUS")}
    for row, label in zip(rows_one, labels, strict=True):
        if label.status == "MATCHED":
            buckets[str(label.portable_label)].append(row)
        else:
            buckets[label.status].append(row)
    population = {
        name: {"rows": len(items), "canonical_identities": len({row.canonical_identity_v3 for row in items})}
        for name, items in buckets.items()
    }
    scenario_population = {}
    for scenario in schedule["scenarios"]:
        scenario_id = scenario["scenario_id"]
        selected = [
            (row, label) for row, label in zip(rows_one, labels, strict=True)
            if scenario_id in label.matched_rule_ids
        ]
        scenario_population[scenario_id] = {
            "portable_label": scenario["portable_label"],
            "flows": len(selected),
            "canonical_identities": len({row.canonical_identity_v3 for row, _ in selected}),
            "matched": sum(label.status == "MATCHED" for _, label in selected),
            "unmatched": 0,
            "ambiguous": sum(label.status == "AMBIGUOUS" for _, label in selected),
            "protocol_distribution": dict(sorted(Counter(row.ip_protocol for row, _ in selected).items())),
        }
    write_json("population_audit.json", {
        "artifact_type": "P13B_POPULATION_AUDIT",
        "populations": population,
        "per_scenario": scenario_population,
        "rows_silently_dropped": 0,
    })

    identity = identity_label_audit(rows_one, labels)
    write_json("identity_multiplicity_audit.json", {
        "artifact_type": "P13B_IDENTITY_MULTIPLICITY_AUDIT", **identity,
    })
    write_json("cross_label_identity_audit.json", {
        "artifact_type": "P13B_CROSS_LABEL_IDENTITY_AUDIT",
        "normal_only_identity_count": identity["normal_only_identity_count"],
        "attack_only_identity_count": identity["attack_only_identity_count"],
        "mixed_identity_count": identity["mixed_identity_count"],
        "mixed_flow_count": identity["mixed_flow_count"],
        "identity_review_required": identity["identity_review_required"],
        "review_rule": identity["review_rule"],
    })

    feature_audit = {
        "artifact_type": "P13B_FEATURE_SANITY_AUDIT",
        "groups": {
            "ALL": feature_summary(rows_one),
            "NORMAL": feature_summary(buckets["NORMAL"]),
            "ATTACK": feature_summary(buckets["ATTACK"]),
        },
        "feature_selection_performed": False,
        "impossible_value_violations": 0,
    }
    write_json("feature_sanity_audit.json", feature_audit)
    protocol_counts = dict(sorted(Counter(row.ip_protocol for row in rows_one).items()))
    write_json("protocol_audit.json", {
        "artifact_type": "P13B_PROTOCOL_AUDIT",
        "counts": protocol_counts,
        "rare_protocols_dropped": False,
        "category_transformation_from_counts": False,
    })
    write_json("scenario_coverage.json", {
        "artifact_type": "P13B_SCENARIO_COVERAGE",
        "scenarios": scenario_population,
    })

    portable_output = []
    provenance_output = []
    for row, flow, label in zip(rows_one, flows_one, labels, strict=True):
        portable_output.append({
            "flow_id": row.flow_id,
            "continuous": list(row.continuous),
            "missing_mask": list(row.missing_mask),
            "ip_protocol": row.ip_protocol,
            "canonical_identity_v3": row.canonical_identity_v3,
            "source_label": label.source_label,
            "portable_label": label.portable_label,
            "portable_family_optional": None,
            "join_outcome": outcome_name(label),
        })
        first = flow.packets[0]
        provenance_output.append({
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
        })
    portable_hash = write_jsonl("portable_rows.jsonl", portable_output)
    provenance_hash = write_jsonl("private_flow_provenance.jsonl", provenance_output)

    label_join_pass = (
        population["NORMAL"]["rows"] > 0 and population["ATTACK"]["rows"] > 0 and
        population["UNMATCHED"]["rows"] == 0 and population["AMBIGUOUS"]["rows"] == 0
    )
    if not label_join_pass:
        status = "P13B_LABEL_JOIN_BLOCKED"
    elif identity["identity_review_required"]:
        status = "P13B_IDENTITY_REVIEW_REQUIRED"
    else:
        status = "P13B_CONTROLLED_SOURCE_PASS"
    artifact_names = [
        "lab_topology.json", "scenario_schedule.json", "scenario_execution_log.json",
        "raw_source_reservation.json", "capture_environment.json", "capture_format_audit.json",
        "extraction_run_1.json", "extraction_run_2.json", "determinism_audit.json",
        "label_join_audit.json", "population_audit.json", "identity_multiplicity_audit.json",
        "cross_label_identity_audit.json", "feature_sanity_audit.json", "protocol_audit.json",
        "scenario_coverage.json",
    ]
    artifact_hashes = {name: file_sha256(OUTPUT / name) for name in artifact_names}
    artifact_hashes.update({
        "portable_rows.jsonl": portable_hash,
        "private_flow_provenance.jsonl": provenance_hash,
    })
    gate = {
        "artifact_type": "P13B_GATE",
        "corpus_name": CORPUS_NAME,
        "version": VERSION,
        "status": status,
        "created_at_utc": utc_now(),
        "p12_freeze_commit": P12_FREEZE_COMMIT,
        "p12_protocol_sha256": P12_PROTOCOL_SHA256,
        "p12_schema_sha256": P12_SCHEMA_SHA256,
        "raw_aggregate_sha256": reservation["aggregate_corpus_sha256"],
        "semantic_content_sha256": hash_one,
        "artifact_sha256": artifact_hashes,
        "complete_19_field_extraction": True,
        "deterministic_double_run": deterministic,
        "label_join_pass": label_join_pass,
        "identity_review_required": identity["identity_review_required"],
        "optimizer_steps": 0,
        "checkpoints_created": 0,
        "anomaly_scores": 0,
        "model_metrics": 0,
        "historical_model_opened": False,
        "scaler_fitted": False,
        "readiness": "EXTRACTION_VALIDATION_COMPLETE" if status == "P13B_CONTROLLED_SOURCE_PASS" else "REVIEW_REQUIRED",
    }
    write_json("p13b_gate.json", gate)
    print(status)
    print(f"flows={len(rows_one)}")
    print(f"semantic_content_sha256={hash_one}")
    print(f"population={population}")
    print(f"mixed_identity_count={identity['mixed_identity_count']}")
    print("No neural network was trained and no historical detector was evaluated during P13B.")
    return 0 if status in {"P13B_CONTROLLED_SOURCE_PASS", "P13B_IDENTITY_REVIEW_REQUIRED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
