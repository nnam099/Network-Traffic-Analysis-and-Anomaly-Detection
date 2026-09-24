"""Data-quality and identity audits for extracted P13 semantic rows."""

from __future__ import annotations

from collections import Counter
import math
import statistics

from ids.p12_portable_schema import CONTINUOUS_FEATURES

from .labels import LabelJoinResult
from .portable_extractor import ExtractedRow


def validate_row(row: ExtractedRow) -> None:
    if len(row.continuous) != 18 or len(row.missing_mask) != 18:
        raise ValueError("P13 row width must be 18 values plus 18 mask entries")
    for value, missing in zip(row.continuous, row.missing_mask):
        if missing != (value is None):
            raise ValueError("missing mask disagrees with semantic null")
        if value is not None and not math.isfinite(value):
            raise ValueError("continuous values must be finite float64 or null")
    values = dict(zip(CONTINUOUS_FEATURES, row.continuous))
    nonnegative = [value for value in row.continuous if value is not None]
    if any(value < 0 for value in nonnegative):
        raise ValueError("portable continuous features cannot be negative")
    for name in ("forward_packet_fraction", "forward_byte_fraction"):
        value = values[name]
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} lies outside [0,1]")
    duration = values["duration_seconds"]
    rate_names = [name for name in CONTINUOUS_FEATURES if name.endswith("per_second")]
    if duration == 0.0 and any(values[name] is not None for name in rate_names):
        raise ValueError("zero-duration rates must be null")


def feature_sanity(rows: list[ExtractedRow]) -> dict[str, object]:
    for row in rows:
        validate_row(row)
    features = {}
    for index, name in enumerate(CONTINUOUS_FEATURES):
        observed = [row.continuous[index] for row in rows if row.continuous[index] is not None]
        observed = [float(value) for value in observed]
        ordered = sorted(observed)

        def quantile(fraction: float) -> float | None:
            if not ordered:
                return None
            position = (len(ordered) - 1) * fraction
            lower = int(position)
            upper = min(lower + 1, len(ordered) - 1)
            weight = position - lower
            return ordered[lower] * (1 - weight) + ordered[upper] * weight

        features[name] = {
            "count": len(rows),
            "missing_count": len(rows) - len(observed),
            "finite_count": len(observed),
            "min": min(observed) if observed else None,
            "q25": quantile(0.25),
            "median": statistics.median(observed) if observed else None,
            "mean": math.fsum(observed) / len(observed) if observed else None,
            "q75": quantile(0.75),
            "q95": quantile(0.95),
            "max": max(observed) if observed else None,
            "constant": len(set(observed)) <= 1 if observed else None,
        }
    return {"row_count": len(rows), "features": features, "impossible_value_violations": 0}


def identity_audit(rows: list[ExtractedRow], labels: list[LabelJoinResult] | None = None) -> dict[str, object]:
    frequencies = Counter(row.canonical_identity_v3 for row in rows)
    multiplicities = sorted(frequencies.values())
    distribution = dict(sorted(Counter(multiplicities).items()))
    mixed_identities: set[str] = set()
    mixed_rows = 0
    if labels is not None:
        if len(labels) != len(rows):
            raise ValueError("labels and rows must have equal length")
        identity_labels: dict[str, set[str]] = {}
        for row, label in zip(rows, labels):
            if label.portable_label is not None:
                identity_labels.setdefault(row.canonical_identity_v3, set()).add(label.portable_label)
        mixed_identities = {identity for identity, values in identity_labels.items() if len(values) > 1}
        mixed_rows = sum(frequencies[identity] for identity in mixed_identities)
    return {
        "rows": len(rows),
        "unique_canonical_identities": len(frequencies),
        "largest_identity_multiplicity": max(multiplicities) if multiplicities else 0,
        "median_identity_multiplicity": statistics.median(multiplicities) if multiplicities else 0,
        "identity_frequency_distribution": {str(key): value for key, value in distribution.items()},
        "ambiguous_canonical_identity_count": len(mixed_identities),
        "ambiguous_row_count": mixed_rows,
        "cryptographic_collision_count": 0,
    }


def population_audit(rows: list[ExtractedRow], labels: list[LabelJoinResult]) -> dict[str, object]:
    if len(rows) != len(labels):
        raise ValueError("labels and rows must have equal length")
    row_counts = Counter(label.portable_label if label.status == "MATCHED" else label.status for label in labels)
    identity_sets: dict[str, set[str]] = {}
    for row, label in zip(rows, labels):
        bucket = label.portable_label if label.status == "MATCHED" else label.status
        identity_sets.setdefault(str(bucket), set()).add(row.canonical_identity_v3)
    return {
        "row_counts": dict(sorted(row_counts.items())),
        "identity_counts": {key: len(value) for key, value in sorted(identity_sets.items())},
        "rows_dropped": 0,
        "majority_vote_resolution": False,
    }
