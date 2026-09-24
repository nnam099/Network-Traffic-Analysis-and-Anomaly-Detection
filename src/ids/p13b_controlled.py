"""Contracts and audits for the P13B controlled extraction-validation corpus."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from ids.p13.labels import GroundTruthRule, LabelJoinResult, join_labels
from ids.p13.portable_extractor import ExtractedRow


CORPUS_NAME = "P13B_CONTROLLED_CAPTURE"
VERSION = "p13b-controlled-capture-v1"


def pretty_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_corpus_sha256(files: list[dict[str, object]]) -> str:
    lines = [f"{item['filename']}\0{item['size_bytes']}\0{item['sha256']}\n" for item in files]
    return hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()


def verify_raw_reservation(root: Path, reservation: dict, schedule_path: Path) -> None:
    if reservation["corpus_name"] != CORPUS_NAME or reservation["reservation_status"] != "RAW_RESERVED":
        raise ValueError("P13B reservation status/corpus mismatch")
    if file_sha256(schedule_path) != reservation["scenario_schedule_sha256"]:
        raise ValueError("scenario schedule hash changed after reservation")
    for item in reservation["raw_files"]:
        path = root / str(item["path"])
        if not path.is_file() or path.stat().st_size != item["size_bytes"]:
            raise ValueError(f"reserved file size mismatch: {path}")
        if file_sha256(path) != item["sha256"]:
            raise ValueError(f"reserved file hash mismatch: {path}")
        if path.stat().st_mode & 0o222:
            raise ValueError(f"reserved file is not read-only: {path}")
    if aggregate_corpus_sha256(reservation["raw_files"]) != reservation["aggregate_corpus_sha256"]:
        raise ValueError("aggregate corpus hash mismatch")


def schedule_rules(schedule: dict) -> list[GroundTruthRule]:
    rules = []
    for scenario in schedule["scenarios"]:
        rules.append(GroundTruthRule(
            rule_id=scenario["scenario_id"],
            start_ns=int(scenario["expected_start_ns"]),
            end_ns=int(scenario["expected_end_ns"]),
            source_label=scenario["scenario_type"],
            portable_label=scenario["portable_label"],
            ip_protocol=scenario.get("ip_protocol"),
            source_address=scenario.get("generator_host"),
            destination_address=scenario.get("target_host"),
            destination_port=scenario.get("destination_port"),
        ))
    return rules


def join_controlled_labels(flows, schedule: dict) -> list[LabelJoinResult]:
    return join_labels(flows, schedule_rules(schedule))


def outcome_name(result: LabelJoinResult) -> str:
    if result.status == "MATCHED":
        return f"MATCHED_{result.portable_label}"
    return result.status


def identity_label_audit(rows: list[ExtractedRow], labels: list[LabelJoinResult]) -> dict[str, object]:
    if len(rows) != len(labels):
        raise ValueError("row/label cardinality mismatch")
    identities: dict[str, dict[str, object]] = {}
    for row, label in zip(rows, labels):
        group = identities.setdefault(row.canonical_identity_v3, {"flow_ids": [], "labels": set()})
        group["flow_ids"].append(row.flow_id)
        if label.status == "MATCHED" and label.portable_label is not None:
            group["labels"].add(label.portable_label)
    multiplicities = sorted(len(group["flow_ids"]) for group in identities.values())
    matched_flows = sum(label.status == "MATCHED" for label in labels)
    mixed = {key: value for key, value in identities.items() if value["labels"] == {"NORMAL", "ATTACK"}}
    mixed_flows = sum(len(value["flow_ids"]) for value in mixed.values())

    def quantile(values: list[int], fraction: float) -> float:
        if not values:
            return 0.0
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    largest = sorted(
        ({"canonical_identity_v3": key, "flow_count": len(value["flow_ids"]),
          "label_class": "MIXED" if value["labels"] == {"NORMAL", "ATTACK"}
          else next(iter(value["labels"]), "UNLABELED")}
         for key, value in identities.items()),
        key=lambda item: (-item["flow_count"], item["canonical_identity_v3"]),
    )[:10]
    mixed_share = mixed_flows / matched_flows if matched_flows else 0.0
    review_required = len(mixed) >= 5 and mixed_share >= 0.05
    return {
        "total_flow_instances": len(rows),
        "unique_canonical_identities": len(identities),
        "flows_per_identity_mean": len(rows) / len(identities) if identities else 0.0,
        "flows_per_identity_median": statistics.median(multiplicities) if multiplicities else 0.0,
        "flows_per_identity_p95": quantile(multiplicities, 0.95),
        "flows_per_identity_maximum": max(multiplicities) if multiplicities else 0,
        "largest_identity_groups": largest,
        "normal_only_identity_count": sum(value["labels"] == {"NORMAL"} for value in identities.values()),
        "attack_only_identity_count": sum(value["labels"] == {"ATTACK"} for value in identities.values()),
        "mixed_identity_count": len(mixed),
        "mixed_flow_count": mixed_flows,
        "mixed_matched_flow_share": mixed_share,
        "review_rule": "mixed_identity_count >= 5 AND mixed_matched_flow_share >= 0.05",
        "identity_review_required": review_required,
        "majority_vote_used": False,
    }


def feature_summary(rows: list[ExtractedRow]) -> dict[str, dict[str, object]]:
    from ids.p12_portable_schema import CONTINUOUS_FEATURES

    output = {}
    for index, name in enumerate(CONTINUOUS_FEATURES):
        observed = sorted(float(row.continuous[index]) for row in rows if row.continuous[index] is not None)

        def quantile(fraction: float) -> float | None:
            if not observed:
                return None
            position = (len(observed) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(observed) - 1)
            weight = position - lower
            return observed[lower] * (1 - weight) + observed[upper] * weight

        common_fraction = 0.0
        if observed:
            common_fraction = max(Counter(observed).values()) / len(observed)
        output[name] = {
            "n": len(rows),
            "missing": len(rows) - len(observed),
            "minimum": min(observed) if observed else None,
            "p01": quantile(0.01),
            "median": statistics.median(observed) if observed else None,
            "mean": math.fsum(observed) / len(observed) if observed else None,
            "p99": quantile(0.99),
            "maximum": max(observed) if observed else None,
            "constant": len(set(observed)) <= 1 if observed else None,
            "near_constant": common_fraction >= 0.99 if observed else None,
            "most_common_fraction": common_fraction if observed else None,
        }
    return output
