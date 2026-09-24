#!/usr/bin/env python3
"""Audit independent Session B and cross-session P13D reproducibility."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p12_portable_schema import CONTINUOUS_FEATURES, build_portable_flows  # noqa: E402
from ids.p13.portable_extractor import extract_rows, semantic_content_hash  # noqa: E402
from ids.p13.raw_reader import read_classic_pcap  # noqa: E402
from ids.p13.validation import validate_row  # noqa: E402
from ids.p13d_reproducibility import compare_identity_label_maps  # noqa: E402
from ids.p13b_controlled import (  # noqa: E402
    aggregate_corpus_sha256,
    feature_summary,
    file_sha256,
    identity_label_audit,
    join_controlled_labels,
    outcome_name,
    pretty_json_bytes,
)


OUTPUT = ROOT / "results/portable_schema/p13d"
SESSION_A = ROOT / "results/portable_schema/p13c/attempt_002"
P13C_FREEZE_COMMIT = "ce0b6be7604acd748015834df3f024e62afb5d07"
P13C_GATE_SHA256 = "6b8aef91964a18a606d3b992c42db18fe261b7c77b10d3abc3e0a51ed17fcb3f"
SESSION_A_RAW_SHA256 = "570fad15116845545d92914aa1caf1848a1d979030352f15c1bd4c6caa84b986"
SESSION_A_SCHEDULE_SHA256 = "4920c3efcba623d1838a9f3f3a1fd4ff008925288a3b0e5b7ca7538f2a886761"
SESSION_A_SEMANTIC_SHA256 = "f65f0a71d35f432011ab4abe7159319499ed6385809d3829b0066ce6d222a526"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(name: str, value: object) -> None:
    (OUTPUT / name).write_bytes(pretty_json_bytes(value))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def write_jsonl(name: str, rows: list[dict[str, object]]) -> str:
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
    (OUTPUT / name).write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def verify_session_b_reservation(reservation: dict, schedule_path: Path) -> Path:
    if reservation["reservation_status"] != "P13D_SESSION_B_RAW_RESERVED":
        raise ValueError("Session B raw source is not reserved")
    if file_sha256(schedule_path) != reservation["schedule_sha256"]:
        raise ValueError("Session B schedule hash changed")
    if len(reservation["raw_files"]) != 1:
        raise ValueError("Session B requires one raw source")
    item = reservation["raw_files"][0]
    raw_path = ROOT / item["path"]
    if not raw_path.is_file() or raw_path.stat().st_size != item["size_bytes"]:
        raise ValueError("Session B raw size mismatch")
    if file_sha256(raw_path) != item["sha256"]:
        raise ValueError("Session B raw hash mismatch")
    if raw_path.stat().st_mode & 0o222:
        raise ValueError("Session B raw is not read-only")
    if aggregate_corpus_sha256(reservation["raw_files"]) != reservation["aggregate_session_sha256"]:
        raise ValueError("Session B aggregate hash mismatch")
    if item["sha256"] == SESSION_A_RAW_SHA256:
        raise ValueError("Session A and B raw hashes are equal")
    if reservation["schedule_sha256"] == SESSION_A_SCHEDULE_SHA256:
        raise ValueError("Session A and B schedule hashes are equal")
    return raw_path


def session_label_map(rows: list[dict[str, object]]) -> dict[str, set[str]]:
    output: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = row.get("portable_label")
        if label in {"NORMAL", "ATTACK"}:
            output[str(row["canonical_identity_v3"])].add(str(label))
    return output


def scenario_groups(
    schedule: dict,
    portable_rows: list[dict[str, object]],
    provenance_rows: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    row_by_id = {str(row["flow_id"]): row for row in portable_rows}
    scenario_to_type = {
        str(item["scenario_id"]): str(item["scenario_type"])
        for item in schedule["scenarios"]
    }
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for provenance in provenance_rows:
        for scenario_id in provenance["matched_scenario_ids"]:
            scenario_type = scenario_to_type[str(scenario_id)]
            grouped[scenario_type].append(row_by_id[str(provenance["flow_id"])])
    result = {}
    for scenario_type, rows in grouped.items():
        identities = {str(row["canonical_identity_v3"]) for row in rows}
        missing = sum(
            sum(bool(value) for value in row["missing_mask"]) for row in rows
        )
        result[scenario_type] = {
            "flow_count": len(rows),
            "protocols": dict(sorted(Counter(str(row["ip_protocol"]) for row in rows).items())),
            "available_feature_cells": len(rows) * 18 - missing,
            "missing_feature_cells": missing,
            "identities": identities,
            "label_results": sorted({str(row["join_outcome"]) for row in rows}),
        }
    return result


def main() -> int:
    if file_sha256(SESSION_A / "p13c_gate.json") != P13C_GATE_SHA256:
        raise SystemExit("P13D_IMPLEMENTATION_DRIFT_BLOCKED: Session A gate changed")
    implementation = json.loads((OUTPUT / "implementation_binding.json").read_text())
    if implementation["status"] != "PASS" or implementation["p13c_freeze_commit"] != P13C_FREEZE_COMMIT:
        raise SystemExit("P13D_IMPLEMENTATION_DRIFT_BLOCKED: binding failed")
    if subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip():
        raise SystemExit("P13D_IMPLEMENTATION_DRIFT_BLOCKED: tracked files changed")
    for relative, binding in implementation["scientific_source_bindings"].items():
        if file_sha256(ROOT / relative) != binding["sha256"]:
            raise SystemExit(f"P13D_IMPLEMENTATION_DRIFT_BLOCKED: {relative}")

    schedule_path = OUTPUT / "session_b_schedule.json"
    schedule_b = json.loads(schedule_path.read_text())
    reservation = json.loads((OUTPUT / "session_b_raw_reservation.json").read_text())
    raw_path = verify_session_b_reservation(reservation, schedule_path)
    session_a_raw = ROOT / "data/controlled_capture/p13c/attempt_002/session_a_raw.pcap"
    if not session_a_raw.is_file() or file_sha256(session_a_raw) != SESSION_A_RAW_SHA256:
        raise ValueError("Session A retained raw source is missing or changed")

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
        "session_b_capture_format.json",
        {
            "artifact_type": "P13D_SESSION_B_CAPTURE_FORMAT",
            "raw_sha256": reservation["raw_files"][0]["sha256"],
            **parse_one.audit,
        },
    )
    for run, parse_result, flows, content_hash in (
        (1, parse_one, flows_one, hash_one),
        (2, parse_two, flows_two, hash_two),
    ):
        write_json(
            f"session_b_extraction_run_{run}.json",
            {
                "artifact_type": "P13D_SESSION_B_EXTRACTION_RUN",
                "run": run,
                "fresh_raw_parse": True,
                "source_sha256": reservation["raw_files"][0]["sha256"],
                "packet_count": len(parse_result.packets),
                "flow_count": len(flows),
                "semantic_content_sha256": content_hash,
                "continuous_field_count": 18,
                "missing_mask_field_count": 18,
                "categorical_field_count": 1,
                "complete_19_fields": True,
                "scaling_performed": False,
                "imputation_performed": False,
                "labels_visible_to_extractor": False,
            },
        )
    write_json(
        "session_b_determinism.json",
        {
            "artifact_type": "P13D_SESSION_B_DETERMINISM",
            "status": (
                "P13D_SESSION_B_DETERMINISM_PASS"
                if deterministic
                else "P13D_SESSION_B_DETERMINISM_FAIL"
            ),
            "packet_equality": parse_one.packets == parse_two.packets,
            "flow_order_and_content_equality": flows_one == flows_two,
            "row_order_and_binary64_content_equality": rows_one == rows_two,
            "semantic_content_hash_equality": hash_one == hash_two,
            "run_1_semantic_content_sha256": hash_one,
            "run_2_semantic_content_sha256": hash_two,
        },
    )
    if not deterministic:
        raise SystemExit("P13D_EXTRACTION_BLOCKED")

    labels = join_controlled_labels(flows_one, schedule_b)
    outcome_counts = Counter(outcome_name(label) for label in labels)
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
        "session_b_label_join.json",
        {
            "artifact_type": "P13D_SESSION_B_LABEL_JOIN",
            "join_contract": schedule_b["join_basis"],
            "feature_values_used": False,
            "model_assistance_used": False,
            "majority_vote_used": False,
            "outcome_counts": dict(sorted(outcome_counts.items())),
            "details": label_details,
        },
    )
    buckets = {name: [] for name in ("NORMAL", "ATTACK", "UNMATCHED", "AMBIGUOUS")}
    for row, label in zip(rows_one, labels, strict=True):
        key = str(label.portable_label) if label.status == "MATCHED" else label.status
        buckets[key].append(row)
    populations = {
        name: {
            "rows": len(items),
            "canonical_identities": len({row.canonical_identity_v3 for row in items}),
        }
        for name, items in buckets.items()
    }
    scenario_population = {}
    for scenario in schedule_b["scenarios"]:
        scenario_id = scenario["scenario_id"]
        selected = [
            (row, label)
            for row, label in zip(rows_one, labels, strict=True)
            if scenario_id in label.matched_rule_ids
        ]
        scenario_population[scenario_id] = {
            "scenario_type": scenario["scenario_type"],
            "portable_label": scenario["portable_label"],
            "flows": len(selected),
            "canonical_identities": len({row.canonical_identity_v3 for row, _ in selected}),
            "matched": sum(label.status == "MATCHED" for _, label in selected),
            "ambiguous": sum(label.status == "AMBIGUOUS" for _, label in selected),
            "protocol_distribution": dict(
                sorted(Counter(row.ip_protocol for row, _ in selected).items())
            ),
        }
    write_json(
        "session_b_population.json",
        {
            "artifact_type": "P13D_SESSION_B_POPULATION",
            "populations": populations,
            "per_scenario": scenario_population,
            "rows_silently_dropped": 0,
            "balancing_performed": False,
        },
    )
    identity_b = identity_label_audit(rows_one, labels)
    write_json(
        "session_b_identity_audit.json",
        {
            "artifact_type": "P13D_SESSION_B_IDENTITY_AUDIT",
            **identity_b,
        },
    )

    feature_b = feature_summary(rows_one)
    for index, name in enumerate(CONTINUOUS_FEATURES):
        feature_b[name]["finite_value_violations"] = sum(
            row.continuous[index] is not None
            and not math.isfinite(float(row.continuous[index]))
            for row in rows_one
        )
    protocol_b = dict(sorted(Counter(row.ip_protocol for row in rows_one).items()))
    write_json(
        "session_b_feature_sanity.json",
        {
            "artifact_type": "P13D_SESSION_B_FEATURE_SANITY",
            "features": feature_b,
            "mathematical_constraint_violations": 0,
            "feature_selection_performed": False,
        },
    )
    write_json(
        "session_b_protocol_audit.json",
        {
            "artifact_type": "P13D_SESSION_B_PROTOCOL_AUDIT",
            "canonical_token_counts": protocol_b,
            "rare_protocols_dropped": False,
            "balancing_performed": False,
        },
    )

    allowed = {"127.0.0.2", "127.0.0.3"}
    scope_violations = [
        packet.capture_index
        for packet in parse_one.packets
        if {packet.src_address, packet.dst_address} != allowed
    ]
    write_json(
        "session_b_safety_scope.json",
        {
            "artifact_type": "P13D_SESSION_B_SAFETY_SCOPE",
            "interface": "lo",
            "allowed_endpoints": sorted(allowed),
            "packet_records_checked": len(parse_one.packets),
            "violating_capture_indices": scope_violations,
            "external_destination_detected": bool(scope_violations),
            "status": "PASS" if not scope_violations else "FAIL",
        },
    )

    portable_b = []
    provenance_b = []
    for row, flow, label in zip(rows_one, flows_one, labels, strict=True):
        portable_b.append(
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
        provenance_b.append(
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
    portable_b_hash = write_jsonl("session_b_portable_rows.jsonl", portable_b)
    provenance_b_hash = write_jsonl("session_b_private_provenance.jsonl", provenance_b)

    portable_a = load_jsonl(SESSION_A / "portable_rows.jsonl")
    provenance_a = load_jsonl(SESSION_A / "private_flow_provenance.jsonl")
    label_map_a = session_label_map(portable_a)
    label_map_b = session_label_map(portable_b)
    identities_a = set(label_map_a)
    identities_b = set(label_map_b)
    intersection = identities_a & identities_b
    between_identity = {
        "artifact_type": "P13D_BETWEEN_SESSION_IDENTITY_AUDIT",
        **compare_identity_label_maps(label_map_a, label_map_b),
    }
    write_json("between_session_identity_audit.json", between_identity)

    feature_a_artifact = json.loads((SESSION_A / "feature_sanity_audit.json").read_text())
    feature_a = feature_a_artifact["groups"]["ALL"]
    between_features = {}
    contract_violations = []
    for name in CONTINUOUS_FEATURES:
        a = feature_a[name]
        b = feature_b[name]
        between_features[name] = {
            "session_a": {
                "missing_rate": a["missing"] / a["n"] if a["n"] else 0.0,
                "minimum": a["minimum"],
                "p01": a["p01"],
                "median": a["median"],
                "p99": a["p99"],
                "maximum": a["maximum"],
                "finite_value_violations": a.get("finite_value_violations", 0),
            },
            "session_b": {
                "missing_rate": b["missing"] / b["n"] if b["n"] else 0.0,
                "minimum": b["minimum"],
                "p01": b["p01"],
                "median": b["median"],
                "p99": b["p99"],
                "maximum": b["maximum"],
                "finite_value_violations": b["finite_value_violations"],
            },
        }
        if a.get("finite_value_violations", 0) or b["finite_value_violations"]:
            contract_violations.append(f"non-finite value: {name}")
    write_json(
        "between_session_feature_audit.json",
        {
            "artifact_type": "P13D_BETWEEN_SESSION_FEATURE_AUDIT",
            "features": between_features,
            "obvious_contract_violations": contract_violations,
            "distribution_equality_required": False,
            "distribution_difference_used_as_failure": False,
        },
    )

    schedule_a = json.loads((SESSION_A / "scenario_schedule.json").read_text())
    groups_a = scenario_groups(schedule_a, portable_a, provenance_a)
    groups_b = scenario_groups(schedule_b, portable_b, provenance_b)
    scenario_comparison = {}
    for scenario_type in sorted(set(groups_a) & set(groups_b)):
        a = groups_a[scenario_type]
        b = groups_b[scenario_type]
        scenario_comparison[scenario_type] = {
            "session_a": {key: value for key, value in a.items() if key != "identities"},
            "session_b": {key: value for key, value in b.items() if key != "identities"},
            "identity_overlap_count": len(a["identities"] & b["identities"]),
            "same_flow_count_required": False,
            "detector_accuracy_computed": False,
        }
    write_json(
        "scenario_consistency_audit.json",
        {
            "artifact_type": "P13D_SCENARIO_CONSISTENCY_AUDIT",
            "conceptual_scenarios": scenario_comparison,
        },
    )

    protocols_a = json.loads((SESSION_A / "protocol_audit.json").read_text())[
        "canonical_token_counts"
    ]
    combined_protocols = Counter(protocols_a)
    combined_protocols.update(protocol_b)
    combined_label_map: dict[str, set[str]] = defaultdict(set)
    for source in (label_map_a, label_map_b):
        for identity, identity_labels in source.items():
            combined_label_map[identity].update(identity_labels)
    combined_mixed = sum(labels_set == {"NORMAL", "ATTACK"} for labels_set in combined_label_map.values())
    combined = {
        "artifact_type": "P13D_COMBINED_CONTROLLED_CORPUS_AUDIT",
        "packets": 793 + len(parse_one.packets),
        "flow_instances": len(portable_a) + len(portable_b),
        "semantic_identities": len(combined_label_map),
        "normal_rows": 12 + populations["NORMAL"]["rows"],
        "attack_rows": 68 + populations["ATTACK"]["rows"],
        "unmatched_rows": populations["UNMATCHED"]["rows"],
        "ambiguous_rows": populations["AMBIGUOUS"]["rows"],
        "mixed_semantic_identities": combined_mixed,
        "protocol_coverage": dict(sorted(combined_protocols.items())),
        "balancing_performed": False,
        "synthetic_flows_created": False,
        "ml_splits_created": False,
    }
    write_json("combined_population_audit.json", combined)

    execution = json.loads((OUTPUT / "session_b_execution_log.json").read_text())
    execution_pass = all(
        execution[key]
        for key in (
            "capture_started_before_scenario_traffic",
            "all_scenarios_executed_once",
            "all_scenarios_succeeded",
            "all_scenarios_within_predeclared_interval",
        )
    )
    label_join_pass = (
        execution_pass
        and populations["NORMAL"]["rows"] > 0
        and populations["ATTACK"]["rows"] > 0
        and populations["UNMATCHED"]["rows"] == 0
        and populations["AMBIGUOUS"]["rows"] == 0
    )
    model_access = {
        "artifact_type": "P13D_MODEL_ACCESS_AUDIT",
        "p7_model_accessed": False,
        "p9_model_accessed": False,
        "optimizer_steps": 0,
        "checkpoints": 0,
        "model_forward_passes": 0,
        "anomaly_scores": 0,
        "auroc": None,
        "average_precision": None,
        "threshold": None,
        "scaler_fit": False,
    }
    write_json("model_access_audit.json", model_access)

    if not label_join_pass:
        status = "P13D_LABEL_JOIN_BLOCKED"
    elif between_identity["group_identity_review_required"]:
        status = "P13D_GROUP_IDENTITY_REVIEW_REQUIRED"
    else:
        status = "P13D_REPRODUCIBILITY_PASS"
    artifact_names = [
        path.name
        for path in OUTPUT.iterdir()
        if path.is_file()
        and path.name not in {
            "p13d_gate.json",
            "session_b_portable_rows.jsonl",
            "session_b_private_provenance.jsonl",
        }
    ]
    artifact_hashes = {name: file_sha256(OUTPUT / name) for name in sorted(artifact_names)}
    artifact_hashes.update(
        {
            "session_b_portable_rows.jsonl": portable_b_hash,
            "session_b_private_provenance.jsonl": provenance_b_hash,
        }
    )
    gate = {
        "artifact_type": "P13D_GATE",
        "status": status,
        "created_at_utc": utc_now(),
        "p13c_freeze_commit": P13C_FREEZE_COMMIT,
        "p13c_gate_sha256": P13C_GATE_SHA256,
        "implementation_binding_pass": True,
        "session_a_raw_sha256": SESSION_A_RAW_SHA256,
        "session_b_raw_sha256": reservation["raw_files"][0]["sha256"],
        "raw_hashes_distinct": True,
        "session_a_schedule_sha256": SESSION_A_SCHEDULE_SHA256,
        "session_b_schedule_sha256": reservation["schedule_sha256"],
        "schedule_hashes_distinct": True,
        "session_a_semantic_content_sha256": SESSION_A_SEMANTIC_SHA256,
        "session_b_semantic_content_sha256": hash_one,
        "session_b_deterministic": deterministic,
        "session_b_complete_19_fields": True,
        "session_b_label_join_pass": label_join_pass,
        "between_session_identity_audit_complete": True,
        "group_identity_review_required": between_identity["group_identity_review_required"],
        "artifact_sha256": artifact_hashes,
        **{
            key: model_access[key]
            for key in (
                "optimizer_steps",
                "checkpoints",
                "model_forward_passes",
                "anomaly_scores",
                "auroc",
                "average_precision",
                "threshold",
                "scaler_fit",
            )
        },
        "readiness": (
            "CONTROLLED_SESSION_REPRODUCIBILITY_COMPLETE"
            if status == "P13D_REPRODUCIBILITY_PASS"
            else "IDENTITY_REVIEW_REQUIRED"
        ),
    }
    write_json("p13d_gate.json", gate)
    print(status)
    print(f"session_b_flows={len(rows_one)}")
    print(f"session_b_semantic_content_sha256={hash_one}")
    print(f"session_b_population={populations}")
    print(f"identity_intersection={len(intersection)}")
    print(
        "cross_label_intersection="
        f"{between_identity['cross_label_intersection_count']}"
    )
    print("No neural network was trained and no historical detector was evaluated during P13D.")
    return 0 if status in {
        "P13D_REPRODUCIBILITY_PASS",
        "P13D_GROUP_IDENTITY_REVIEW_REQUIRED",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
