"""Reference-row comparison utilities with feature-specific tolerances."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class Tolerance:
    absolute: float
    relative: float
    rationale: str


def declared_tolerance(units: str) -> Tolerance:
    if units in {"packets", "bytes", "count", "commands", "transactions", "binary",
                 "IP hops", "TCP sequence number"}:
        return Tolerance(0.0, 0.0, "integer semantic requires exact equality")
    if units == "seconds":
        return Tolerance(1e-9, 1e-12, "nanosecond parser resolution; declared before reference access")
    if units == "milliseconds":
        return Tolerance(1e-6, 1e-12, "nanosecond-to-millisecond conversion tolerance")
    if "/second" in units or units in {"ratio", "log(bytes)", "log(packets)",
                                        "log(milliseconds)", "log(bits/second)"}:
        return Tolerance(1e-9, 1e-9, "float64 arithmetic tolerance")
    return Tolerance(0.0, 0.0, "exact comparison unless amended before reference evaluation")


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def compare_continuous(reference: list[float | None], actual: list[float | None],
                       tolerance: Tolerance) -> dict:
    if len(reference) != len(actual):
        raise ValueError("reference/actual row count mismatch")
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    exact = nan_mismatch = inf_mismatch = sign_mismatch = within = 0
    for expected, observed in zip(reference, actual):
        if expected is None or observed is None:
            if expected is not observed:
                nan_mismatch += 1
            continue
        expected = float(expected)
        observed = float(observed)
        if math.isnan(expected) or math.isnan(observed):
            if not (math.isnan(expected) and math.isnan(observed)):
                nan_mismatch += 1
            continue
        if math.isinf(expected) or math.isinf(observed):
            if expected != observed:
                inf_mismatch += 1
            continue
        error = abs(observed - expected)
        relative = error / max(abs(expected), 1e-300)
        absolute_errors.append(error)
        relative_errors.append(relative)
        exact += int(observed == expected)
        sign_mismatch += int(expected != 0 and observed != 0 and
                             math.copysign(1.0, expected) != math.copysign(1.0, observed))
        within += int(error <= tolerance.absolute + tolerance.relative * abs(expected))
    return {
        "row_count": len(reference), "exact_match_count": exact,
        "within_tolerance_count": within,
        "absolute_error": {
            "mean": math.fsum(absolute_errors) / len(absolute_errors) if absolute_errors else None,
            "max": max(absolute_errors) if absolute_errors else None,
            "quantiles": {str(q): _quantile(absolute_errors, q) for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
        },
        "relative_error": {
            "mean": math.fsum(relative_errors) / len(relative_errors) if relative_errors else None,
            "max": max(relative_errors) if relative_errors else None,
            "quantiles": {str(q): _quantile(relative_errors, q) for q in (0.0, 0.5, 0.9, 0.99, 1.0)},
        },
        "nan_mismatch_count": nan_mismatch, "inf_mismatch_count": inf_mismatch,
        "sign_mismatch_count": sign_mismatch,
        "tolerance": {"absolute": tolerance.absolute, "relative": tolerance.relative,
                      "rationale": tolerance.rationale},
    }
