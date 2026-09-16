from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluator import load_ids_artifacts, preprocess_raw_df, run_batch_scores, summarize_scores  # noqa: E402
from ids.dataset import normalize_labels  # noqa: E402
from inference_runtime import ground_truth_verdict, zero_day_decision  # noqa: E402


def display_path(path: str | Path) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate v14 artifact evaluation metrics and plots.")
    parser.add_argument("--csv-path", default=str(ROOT_DIR / "data" / "UNSW_NB15_testing-set.csv"))
    parser.add_argument("--label-col", default="attack_cat")
    parser.add_argument("--model-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_model.pth"))
    parser.add_argument("--pipeline-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl"))
    parser.add_argument("--output-json", default=str(ROOT_DIR / "results" / "ids_v14_results.json"))
    parser.add_argument("--plots-dir", default=str(ROOT_DIR / "plots"))
    parser.add_argument("--scores-csv", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--calibrated-threshold-profile",
        default=str(ROOT_DIR / "checkpoints" / "local_thresholds.json"),
        help="Optional local threshold profile to evaluate as a what-if scenario.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plots_dir = Path(args.plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.csv_path, nrows=args.max_rows)
    artifacts = load_ids_artifacts(args.model_path, args.pipeline_path, "v14")
    raw_features, normalization_report = preprocess_raw_df(raw_df, artifacts)
    scores = run_batch_scores(raw_features, artifacts)
    summary = summarize_scores(
        scores,
        raw_df=raw_df,
        label_col=args.label_col,
        class_names=artifacts.class_names,
        zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
        thresholds=artifacts.thresholds,
    )
    calibrated = calibrated_threshold_summary(
        scores,
        raw_df,
        label_col=args.label_col,
        profile_path=args.calibrated_threshold_profile,
        zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
    )
    plots = write_evaluation_plots(scores, summary, plots_dir)

    if args.scores_csv:
        scores.to_csv(args.scores_csv, index=False)

    report = {
        "version": "v14.0",
        "status": "artifact_evaluation_regenerated",
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "regeneration_mode": "current_artifact_evaluation",
        "input_csv": display_path(args.csv_path),
        "model_path": display_path(args.model_path),
        "pipeline_path": display_path(args.pipeline_path),
        "n_features": len(artifacts.feature_names),
        "n_classes": len(artifacts.class_names),
        "known_cats": list(artifacts.pipeline.get("known_cats", [])),
        "zd_cats": list(artifacts.pipeline.get("zd_cats", [])),
        "metrics": summary,
        "calibrated_threshold_evaluation": calibrated,
        "normalization_report": normalization_report,
        "threshold_profile": summary.get("threshold_profile", {}),
        "metric_definitions": {
            "normal_ood_fpr": "Share of labeled Normal rows flagged as OOD/zero-day candidates.",
            "normal_alert_fpr": "Share of labeled Normal rows receiving any attack alert after classifier and OOD decisions.",
            "known_classifier_fpr": "Share of labeled Normal rows classified as a known attack class before OOD override.",
            "normal_ood_classifier_overlap_rate": "Share of labeled Normal rows counted in both OOD and known-classifier false positives.",
        },
        "plots": plots,
        "limitations_and_safety": {
            "zero_day_candidate_meaning": "OOD hypothesis for analyst review; not a confirmed novel attack.",
            "regeneration_note": "This report evaluates the current saved v14 artifacts. It does not retrain model weights.",
            "required_validation": "Confirm suspicious rows with SIEM, firewall, endpoint and packet evidence.",
        },
    }
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    print(f"Report: {args.output_json}")
    print(f"Rows: {summary.get('rows', 0):,}")
    print(f"Accuracy: {summary.get('accuracy')}")
    print(f"Normal OOD FPR: {summary.get('normal_ood_fpr')}")
    print(f"Normal alert FPR: {summary.get('normal_alert_fpr')}")
    print(f"Known-classifier FPR: {summary.get('known_classifier_fpr')}")
    print(
        "Normal OOD/classifier overlap: "
        f"{summary.get('normal_ood_classifier_overlap_count')} rows "
        f"({summary.get('normal_ood_classifier_overlap_rate')})"
    )
    print(f"OOD detection rate: {summary.get('ood_detection_rate')}")
    if calibrated:
        print(f"Calibrated normal OOD FPR: {calibrated.get('normal_ood_fpr')}")
        print(f"Calibrated normal alert FPR: {calibrated.get('normal_alert_fpr')}")
        print(f"Calibrated OOD detection rate: {calibrated.get('ood_detection_rate')}")
    return 0


def write_evaluation_plots(scores: pd.DataFrame, summary: dict, plots_dir: Path) -> list[str]:
    paths = []

    verdict_counts = scores["predicted_class"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    verdict_counts.plot(kind="bar", ax=ax, color="#39b7e8")
    ax.set_title("v14 Verdict Distribution")
    ax.set_xlabel("Verdict")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=25)
    path = plots_dir / "v14_eval_verdict_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(display_path(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    scores[["hybrid", "ae_re", "softmax"]].plot(kind="box", ax=ax)
    ax.set_title("v14 Score Distribution")
    ax.set_ylabel("Score")
    path = plots_dir / "v14_eval_score_distribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(display_path(path))

    recall = summary.get("recall_per_class") or {}
    if recall:
        recall_df = pd.DataFrame(
            [
                {"Class": key, "Recall": value.get("recall", 0.0), "Support": value.get("support", 0)}
                for key, value in recall.items()
            ]
        ).sort_values("Class")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(recall_df["Class"], recall_df["Recall"], color="#42d392")
        ax.set_ylim(0, 1.0)
        ax.set_title("v14 Known-Class Recall")
        ax.set_xlabel("Class")
        ax.set_ylabel("Recall")
        ax.tick_params(axis="x", rotation=25)
        path = plots_dir / "v14_eval_known_class_recall.png"
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(display_path(path))

    return paths


def calibrated_threshold_summary(
    scores: pd.DataFrame,
    raw_df: pd.DataFrame,
    label_col: str,
    profile_path: str | None,
    zero_day_labels: list[str],
) -> dict | None:
    if not profile_path or not Path(profile_path).exists():
        return None
    with open(profile_path, encoding="utf-8") as handle:
        profile = json.load(handle)
    thresholds = profile.get("thresholds", {})
    if not thresholds:
        return None
    decisions, rule = zero_day_decision(
        scores["ae_re"].values,
        scores["max_prob"].values,
        scores["hybrid"].values,
        thresholds=thresholds,
    )
    decisions = pd.Series(decisions.astype(bool), index=scores.index)
    if label_col in raw_df.columns:
        label_frame = raw_df[label_col].rename("attack_cat").to_frame()
        labels = normalize_labels(label_frame)["attack_cat"]
        truth = labels.map(ground_truth_verdict)
    else:
        truth = pd.Series([], dtype=str)
        labels = pd.Series([], dtype=str)
    normal_mask = truth == "Normal"
    zero_day_frame = pd.Series(zero_day_labels, name="attack_cat", dtype=str).to_frame()
    canonical_zero_day_labels = set(normalize_labels(zero_day_frame)["attack_cat"])
    zd_mask = labels.isin(canonical_zero_day_labels)
    normal_ood = decisions.loc[normal_mask.values]
    normal_classifier_alert = (
        scores.loc[normal_mask.values, "classifier_class"].astype(str).str.strip().str.lower() != "normal"
    )
    normal_alert = normal_ood | normal_classifier_alert
    normal_overlap = normal_ood & normal_classifier_alert
    return {
        "profile_path": display_path(profile_path),
        "target_fpr": profile.get("target_fpr"),
        "reference_rows": profile.get("reference_rows"),
        "decision_rule": rule,
        "zero_day_count": int(decisions.sum()),
        "zero_day_rate": round(float(decisions.mean()), 6),
        "normal_ood_fpr": round(float(normal_ood.mean()), 6) if bool(normal_mask.any()) else None,
        "normal_alert_fpr": round(float(normal_alert.mean()), 6) if bool(normal_mask.any()) else None,
        "known_classifier_fpr": round(float(normal_classifier_alert.mean()), 6) if bool(normal_mask.any()) else None,
        "normal_ood_classifier_overlap_rate": round(float(normal_overlap.mean()), 6)
        if bool(normal_mask.any())
        else None,
        "normal_ood_classifier_overlap_count": int(normal_overlap.sum()) if bool(normal_mask.any()) else None,
        "ood_detection_rate": round(float(decisions.loc[zd_mask.values].mean()), 6) if bool(zd_mask.any()) else None,
        "thresholds": thresholds,
        "tradeoff_note": "Lower FPR can reduce OOD recall; validate the chosen target_fpr against analyst capacity.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
