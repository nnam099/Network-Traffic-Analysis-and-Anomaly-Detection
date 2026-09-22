#!/usr/bin/env python3
"""Aggregate leakage-safe Step 2 LOFO reports across random seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ASSERTION_NAMES = (
    "official_test_excluded_from_fit",
    "official_test_excluded_from_preprocessing",
    "metadata_excluded_from_model_features",
    "known_split_fingerprints_disjoint",
    "target_family_excluded_from_surrogate",
    "target_fingerprints_excluded_from_surrogate",
)


def _sample_summary(values: list[float]) -> dict[str, float | list[float]]:
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        raise ValueError("at least two seeds are required to estimate sample std")
    return {
        "values_by_seed_order": [float(value) for value in array],
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)),
    }


def _load_reports(paths: list[Path]) -> list[dict]:
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    seeds = [int(report["seed"]) for report in reports]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed reports must be unique; got {seeds}")

    protocol = reports[0]["protocol"]
    families = list(reports[0]["ood_cats"])
    feature_schema = list(reports[0]["feature_schema"])
    feature_schema_sha256 = reports[0]["feature_schema_sha256"]
    for path, report in zip(paths, reports, strict=True):
        if report["protocol"] != protocol:
            raise ValueError(f"protocol mismatch in {path}")
        if list(report["ood_cats"]) != families:
            raise ValueError(f"OOD family mismatch in {path}")
        if set(report["folds"]) != set(families):
            raise ValueError(f"LOFO fold mismatch in {path}")
        if list(report.get("feature_schema", [])) != feature_schema:
            raise ValueError(f"ordered feature schema mismatch in {path}")
        if report.get("feature_schema_sha256") != feature_schema_sha256:
            raise ValueError(f"feature schema hash mismatch in {path}")
        if any(str(column).startswith("__") for column in feature_schema):
            raise AssertionError(f"metadata column found in model schema in {path}")
        for assertion in ASSERTION_NAMES:
            if report["assertions"].get(assertion) is not True:
                raise AssertionError(f"{assertion} did not pass in {path}")
        for family in families:
            fold = report["folds"][family]
            for assertion in ASSERTION_NAMES:
                if fold["assertions"].get(assertion) is not True:
                    raise AssertionError(
                        f"{assertion} did not pass for {family} in {path}"
                    )
            if family in fold["surrogate_families"]:
                raise AssertionError(
                    f"target family {family} appears in its surrogate set in {path}"
                )
    return reports


def aggregate(paths: list[Path]) -> dict:
    reports = _load_reports(paths)
    ordered = sorted(
        zip(paths, reports, strict=True),
        key=lambda item: int(item[1]["seed"]),
    )
    paths = [item[0] for item in ordered]
    reports = [item[1] for item in ordered]
    seeds = [int(report["seed"]) for report in reports]
    families = list(reports[0]["ood_cats"])
    feature_schema = list(reports[0]["feature_schema"])
    feature_schema_sha256 = reports[0]["feature_schema_sha256"]

    within_seed_family_variance = {}
    macro_recalls = []
    macro_aurocs = []
    for report in reports:
        seed = str(report["seed"])
        metrics = report["aggregate_metrics"]["official_test"]
        within_seed_family_variance[seed] = {
            "macro_recall_mean_across_families": float(metrics["macro_recall_mean"]),
            "recall_std_across_families": float(metrics["macro_recall_std"]),
            "macro_auroc_mean_across_families": float(metrics["macro_auroc_mean"]),
            "auroc_std_across_families": float(metrics["macro_auroc_std"]),
            "ddof": 1,
            "n_families": len(families),
        }
        macro_recalls.append(float(metrics["macro_recall_mean"]))
        macro_aurocs.append(float(metrics["macro_auroc_mean"]))

    per_family = {}
    for family in families:
        recalls = [
            float(report["folds"][family]["official_test"]["ood_target_recall"])
            for report in reports
        ]
        aurocs = [
            float(report["folds"][family]["official_test"]["ood_auroc"])
            for report in reports
        ]
        supports = [
            int(report["folds"][family]["official_test"]["support"])
            for report in reports
        ]
        if len(set(supports)) != 1:
            raise ValueError(f"official-test support changed across seeds for {family}")
        per_family[family] = {
            "support_per_seed": supports[0],
            "recall": _sample_summary(recalls),
            "auroc": _sample_summary(aurocs),
        }

    return {
        "version": "v14.0-step2-multiseed",
        "protocol": reports[0]["protocol"],
        "metric_scope": "official_test",
        "seeds": seeds,
        "source_reports": [str(path) for path in paths],
        "feature_schema": feature_schema,
        "feature_count": len(feature_schema),
        "feature_schema_sha256": feature_schema_sha256,
        "all_leakage_assertions_passed": True,
        "assertions_checked": list(ASSERTION_NAMES),
        "variance_layers": {
            "within_seed_across_families": {
                "description": (
                    "Family variance within each seed; copied from each seed's "
                    "five-fold official-test aggregate."
                ),
                "by_seed": within_seed_family_variance,
            },
            "between_seeds": {
                "description": (
                    "Seed variance across independent runs; never pooled with "
                    "within-seed family variance."
                ),
                "per_family": per_family,
                "global_macro": {
                    "recall": _sample_summary(macro_recalls),
                    "auroc": _sample_summary(macro_aurocs),
                },
                "ddof": 1,
                "n_seeds": len(seeds),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    combined = aggregate(args.reports)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(combined, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Combined report: {args.output}")
    print(f"Seeds: {combined['seeds']}")
    print("Leakage assertions: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
