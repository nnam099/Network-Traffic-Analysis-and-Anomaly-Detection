from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ids.batch_evaluator import (  # noqa: E402
    calibrate_thresholds,
    load_ids_artifacts,
    preprocess_raw_df,
    run_batch_scores,
    save_threshold_profile,
    summarize_scores,
)


THRESHOLD_LEAKAGE_WARNING = (
    "THRESHOLD LEAKAGE WARNING: this CSV is being used for both threshold "
    "calibration and report generation. The report is not an independent "
    "evaluation of the calibrated profile; evaluate that profile on a separate "
    "held-out file before reporting performance."
)


def display_path(path: str) -> str:
    abs_path = os.path.abspath(path)
    try:
        return os.path.relpath(abs_path, ROOT_DIR).replace(os.sep, "/")
    except ValueError:
        return abs_path


def configure_console_encoding() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")


def add_threshold_leakage_warning(summary: dict, csv_path: str) -> None:
    """Record that calibration and reporting reuse the same input rows."""
    summary.setdefault("methodology_warnings", []).append(THRESHOLD_LEAKAGE_WARNING)
    summary["threshold_calibration_evaluation_independence"] = {
        "calibration_input_csv": display_path(csv_path),
        "report_input_csv": display_path(csv_path),
        "same_input_file": True,
        "independent_evaluation": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate IDS anomaly detection on a CSV file.")
    parser.add_argument("csv_path", help="Path to a UNSW/CICIDS/firewall/flow CSV file.")
    parser.add_argument("--model-version", default=os.getenv("IDS_MODEL_VERSION", "v14"), choices=["v14", "v15"])
    parser.add_argument("--model-path", default=os.path.join(ROOT_DIR, "checkpoints", "ids_v14_model.pth"))
    parser.add_argument("--pipeline-path", default=os.path.join(ROOT_DIR, "checkpoints", "ids_v14_pipeline.pkl"))
    parser.add_argument("--label-col", default=None, help="Optional label column for FPR/recall reporting.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional row cap for quick profiling.")
    parser.add_argument("--output-dir", default=os.path.join(ROOT_DIR, "results", "csv_eval"))
    parser.add_argument("--name", default=None, help="Output filename stem. Defaults to CSV basename.")
    parser.add_argument("--scores-csv", action="store_true", help="Also write row-level score CSV.")
    parser.add_argument("--calibrate-thresholds", action="store_true", help="Write a threshold profile from this CSV.")
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument(
        "--all-reference-rows", action="store_true", help="Calibrate from all rows instead of Normal rows only."
    )
    parser.add_argument("--threshold-output", default=os.path.join(ROOT_DIR, "checkpoints", "local_thresholds.json"))
    return parser.parse_args()


def main() -> int:
    configure_console_encoding()
    args = parse_args()
    if args.calibrate_thresholds:
        print(THRESHOLD_LEAKAGE_WARNING, file=sys.stderr)
    os.makedirs(args.output_dir, exist_ok=True)
    stem = args.name or os.path.splitext(os.path.basename(args.csv_path))[0]

    raw_df = pd.read_csv(args.csv_path, nrows=args.max_rows)
    artifacts = load_ids_artifacts(args.model_path, args.pipeline_path, args.model_version)
    raw_features, normalization_report = preprocess_raw_df(raw_df, artifacts)
    scores = run_batch_scores(raw_features, artifacts, batch_size=args.batch_size)
    summary = summarize_scores(
        scores,
        raw_df=raw_df,
        label_col=args.label_col,
        class_names=artifacts.class_names,
        zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
        thresholds=artifacts.thresholds,
    )
    summary["input_csv"] = display_path(args.csv_path)
    summary["model_path"] = display_path(args.model_path)
    summary["pipeline_path"] = display_path(args.pipeline_path)
    summary["normalization_report"] = normalization_report
    summary["active_thresholds"] = artifacts.thresholds

    report_path = os.path.join(args.output_dir, f"{stem}_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    if args.scores_csv:
        scores_path = os.path.join(args.output_dir, f"{stem}_scores.csv")
        scores.to_csv(scores_path, index=False)
        print(f"Scores CSV: {scores_path}")

    if args.calibrate_thresholds:
        add_threshold_leakage_warning(summary, args.csv_path)
        profile = calibrate_thresholds(
            scores,
            target_fpr=args.target_fpr,
            raw_df=raw_df,
            label_col=args.label_col,
            normal_only=not args.all_reference_rows,
        )
        save_threshold_profile(profile, args.threshold_output)
        summary["calibrated_threshold_profile"] = profile
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"Threshold profile: {args.threshold_output}")

    print(f"Report: {report_path}")
    print(f"Rows: {summary['rows']:,}")
    print(f"Anomaly rate: {summary['zero_day_rate']:.2%}")
    if "normal_ood_fpr" in summary:
        print(f"Normal OOD FPR: {summary['normal_ood_fpr']:.2%}")
        print(f"Normal alert FPR: {summary['normal_alert_fpr']:.2%}")
        print(f"Known-classifier FPR: {summary['known_classifier_fpr']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
