from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ids.batch_evaluator import load_ids_artifacts, preprocess_raw_df, run_batch_scores, summarize_scores  # noqa: E402


SCORE_COLUMNS = ["hybrid", "ae_re", "softmax", "max_prob", "fv_cluster"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report score, verdict and normalization drift for an input CSV.")
    parser.add_argument("csv_path", help="UNSW/CICIDS/firewall/flow CSV to profile.")
    parser.add_argument("--model-version", default="v14", choices=["v14", "v15"])
    parser.add_argument("--model-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_model.pth"))
    parser.add_argument("--pipeline-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl"))
    parser.add_argument("--reference-report", default=str(ROOT_DIR / "results" / "ids_v14_results.json"))
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "results" / "drift"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--scores-csv", action="store_true")
    parser.add_argument("--anomaly-rate-delta", type=float, default=0.10)
    parser.add_argument("--zero-day-rate-delta", dest="anomaly_rate_delta", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--p95-ratio-threshold", type=float, default=1.50)
    return parser.parse_args()


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or Path(args.csv_path).stem

    raw_df = pd.read_csv(args.csv_path, nrows=args.max_rows)
    artifacts = load_ids_artifacts(args.model_path, args.pipeline_path, args.model_version)
    raw_features, normalization_report = preprocess_raw_df(raw_df, artifacts)
    scores = run_batch_scores(raw_features, artifacts, batch_size=args.batch_size)
    current = summarize_scores(
        scores,
        raw_df=raw_df,
        label_col=args.label_col,
        class_names=artifacts.class_names,
        zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
        thresholds=artifacts.thresholds,
    )

    reference = _load_reference(args.reference_report)
    drift = _compare_reports(current, normalization_report, reference, args)

    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_csv": display_path(args.csv_path),
        "model_path": display_path(args.model_path),
        "pipeline_path": display_path(args.pipeline_path),
        "reference_report": display_path(args.reference_report) if Path(args.reference_report).exists() else None,
        "rows": int(len(raw_df)),
        "normalization_report": normalization_report,
        "current_summary": current,
        "drift": drift,
        "notes": [
            "This is a lightweight deployment drift report based on score and normalization summaries.",
            "Investigate high drift before trusting anomaly rates on a new environment.",
            "Use scripts/evaluate_csv.py --calibrate-thresholds on representative benign traffic after drift.",
        ],
    }

    report_path = output_dir / f"{stem}_drift_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    if args.scores_csv:
        scores_path = output_dir / f"{stem}_scores.csv"
        scores.to_csv(scores_path, index=False)
        print(f"Scores CSV: {scores_path}")

    print(f"Report: {report_path}")
    print(f"Drift level: {drift['level']}")
    for warning in drift["warnings"]:
        print(f"- {warning}")
    return 0


def _load_reference(path: str) -> dict[str, Any] | None:
    report_path = Path(path)
    if not report_path.exists():
        return None
    with open(report_path, encoding="utf-8") as handle:
        data = json.load(handle)
    if "metrics" in data:
        return data["metrics"]
    return data


def _compare_reports(
    current: dict[str, Any],
    normalization_report: dict[str, Any],
    reference: dict[str, Any] | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    warnings: list[str] = []
    score_drift: dict[str, Any] = {}

    coverage = _to_float(normalization_report.get("feature_coverage"))
    if coverage is None:
        warnings.append("Feature coverage is unavailable.")
    elif coverage < 0.60:
        warnings.append(f"Feature coverage is low ({coverage:.1%}); scores are weak triage signals.")
    elif coverage < 0.80:
        warnings.append(f"Feature coverage is moderate ({coverage:.1%}); review mapped columns.")

    if reference:
        ref_zero_day_rate = _to_float(reference.get("zero_day_rate"))
        cur_zero_day_rate = _to_float(current.get("zero_day_rate"))
        if ref_zero_day_rate is not None and cur_zero_day_rate is not None:
            delta = cur_zero_day_rate - ref_zero_day_rate
            if abs(delta) >= args.anomaly_rate_delta:
                warnings.append(
                    f"Anomaly rate changed by {delta:+.1%} "
                    f"(reference {ref_zero_day_rate:.1%}, current {cur_zero_day_rate:.1%})."
                )

        ref_dist = reference.get("score_distribution") or {}
        cur_dist = current.get("score_distribution") or {}
        for col in SCORE_COLUMNS:
            if col not in ref_dist or col not in cur_dist:
                continue
            item = _score_delta(ref_dist[col], cur_dist[col], args.p95_ratio_threshold)
            score_drift[col] = item
            if item.get("warning"):
                warnings.append(item["warning"])
    else:
        warnings.append("No reference report was available; only absolute current distributions were reported.")

    if not warnings:
        level = "LOW"
        warnings.append("No major drift signals were detected against the available reference.")
    elif any("low" in item.lower() or "changed" in item.lower() for item in warnings):
        level = "HIGH" if len(warnings) >= 2 else "MEDIUM"
    else:
        level = "MEDIUM"

    return {
        "level": level,
        "warnings": warnings,
        "score_drift": score_drift,
        "reference_available": reference is not None,
        "recommended_action": _recommended_action(level),
    }


def _score_delta(reference: dict[str, Any], current: dict[str, Any], ratio_threshold: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in ["mean", "p50", "p90", "p95", "p99"]:
        ref_val = _to_float(reference.get(key))
        cur_val = _to_float(current.get(key))
        if ref_val is None or cur_val is None:
            continue
        delta = cur_val - ref_val
        ratio = None if abs(ref_val) < 1e-12 else cur_val / ref_val
        out[key] = {
            "reference": ref_val,
            "current": cur_val,
            "delta": delta,
            "ratio": ratio,
        }

    p95 = out.get("p95", {})
    ratio = p95.get("ratio")
    if ratio is not None and ratio > ratio_threshold:
        out["warning"] = (
            f"{p95.get('current'):.6f} current p95 is {ratio:.2f}x the reference p95 "
            f"for score distribution."
        )
    return out


def _recommended_action(level: str) -> str:
    if level == "HIGH":
        return "Review normalization quality, inspect top alerts, and recalibrate thresholds on representative benign rows."
    if level == "MEDIUM":
        return "Inspect score distribution changes and consider local threshold calibration before operational demos."
    return "Continue monitoring; recalibrate when traffic source or feature schema changes."


def _to_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(parsed):
        return None
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
