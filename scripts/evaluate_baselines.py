from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from batch_evaluator import load_ids_artifacts, preprocess_raw_df, run_batch_scores  # noqa: E402
from ids.dataset import normalize_labels  # noqa: E402
from inference_runtime import ground_truth_verdict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare IDS anomaly scores against classical OOD baselines.")
    parser.add_argument("csv_path", help="UNSW/CICIDS/firewall/flow CSV to evaluate.")
    parser.add_argument("--model-version", default="v14", choices=["v14", "v15"])
    parser.add_argument("--model-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_model.pth"))
    parser.add_argument("--pipeline-path", default=str(ROOT_DIR / "checkpoints" / "ids_v14_pipeline.pkl"))
    parser.add_argument("--label-col", default=None, help="Optional label column for supervised metrics.")
    parser.add_argument("--target-fpr", type=float, default=0.05)
    parser.add_argument("--train-ratio", type=float, default=0.50, help="Chronological prefix used as baseline reference.")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-fit-rows", type=int, default=10000, help="Cap expensive baseline fitting rows.")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--skip-ocsvm", action="store_true", help="Skip One-Class SVM for large quick runs.")
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "results" / "baseline_eval"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--scores-csv", action="store_true", help="Write row-level baseline scores.")
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
    scaled = artifacts.pipeline["scaler"].transform(raw_features)
    scaled = np.clip(np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0)

    ids_scores = run_batch_scores(raw_features, artifacts, batch_size=args.batch_size)
    labels = _labels(raw_df, args.label_col)
    train_mask, eval_mask = _split_masks(len(raw_df), labels, args.train_ratio)
    reference_mask = train_mask if labels is None else (train_mask & (labels["truth"] == "Normal").to_numpy())
    if not reference_mask.any():
        reference_mask = train_mask
    fit_X = _sample_fit_rows(scaled[reference_mask], args.max_fit_rows, seed=42)

    score_table = pd.DataFrame(index=np.arange(len(raw_df)))
    score_table["ids_active"] = ids_scores["is_zeroday"].astype(float)
    for col in ["hybrid", "ae_re", "softmax"]:
        if col in ids_scores:
            score_table[f"ids_{col}"] = ids_scores[col].astype(float)

    baseline_scores, skipped_baselines = _baseline_scores(fit_X, scaled, args)
    for name, values in baseline_scores.items():
        score_table[name] = values

    methods = {}
    for method in score_table.columns:
        values = score_table[method].to_numpy(dtype=float)
        if method == "ids_active":
            decisions = values.astype(bool)
            threshold = 0.5
        else:
            threshold = float(np.quantile(values[reference_mask], 1.0 - args.target_fpr))
            decisions = values > threshold
        methods[method] = _method_metrics(
            values=values,
            decisions=decisions,
            threshold=threshold,
            eval_mask=eval_mask,
            labels=labels,
            zero_day_labels=list(artifacts.pipeline.get("zd_cats", [])),
        )

    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "input_csv": display_path(args.csv_path),
        "model_path": display_path(args.model_path),
        "pipeline_path": display_path(args.pipeline_path),
        "rows": int(len(raw_df)),
        "reference_rows": int(reference_mask.sum()),
        "evaluation_rows": int(eval_mask.sum()),
        "target_fpr": float(args.target_fpr),
        "label_column": args.label_col,
        "normalization_report": normalization_report,
        "methods": methods,
        "ranking": _rank_methods(methods),
        "skipped_baselines": skipped_baselines,
        "notes": [
            "Classical baselines are fitted on a chronological reference prefix.",
            "When labels are available, Normal rows in the reference prefix are preferred for fitting.",
            "Use this report as a sanity baseline, not as a replacement for cross-dataset validation.",
        ],
    }

    report_path = output_dir / f"{stem}_baseline_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, default=str)

    if args.scores_csv:
        scores_path = output_dir / f"{stem}_baseline_scores.csv"
        out = score_table.copy()
        if labels is not None:
            out["label"] = labels["label"].values
            out["truth"] = labels["truth"].values
        out.to_csv(scores_path, index=False)
        print(f"Scores CSV: {scores_path}")

    print(f"Report: {report_path}")
    for item in report["ranking"][:5]:
        print(f"{item['method']}: AUROC={item.get('auroc')} OOD={item.get('ood_detection_rate')} FPR={item.get('normal_fpr')}")
    return 0


def _labels(raw_df: pd.DataFrame, label_col: str | None) -> pd.DataFrame | None:
    if not label_col or label_col not in raw_df.columns:
        return None
    label_frame = raw_df[label_col].rename("attack_cat").to_frame()
    label = normalize_labels(label_frame)["attack_cat"]
    truth = label.map(ground_truth_verdict)
    return pd.DataFrame({"label": label, "truth": truth})


def _split_masks(rows: int, labels: pd.DataFrame | None, train_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    split_at = int(max(1, min(rows - 1, round(rows * train_ratio)))) if rows > 1 else rows
    train_mask = np.zeros(rows, dtype=bool)
    train_mask[:split_at] = True
    eval_mask = ~train_mask
    if rows == 1:
        eval_mask[:] = True
    if labels is not None and not eval_mask.any():
        eval_mask[:] = True
    return train_mask, eval_mask


def _sample_fit_rows(values: np.ndarray, max_rows: int, seed: int) -> np.ndarray:
    if len(values) <= max_rows:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(values), size=max_rows, replace=False)
    return values[idx]


def _baseline_scores(fit_X: np.ndarray, all_X: np.ndarray, args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    out: dict[str, np.ndarray] = {}
    skipped: list[str] = []

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    iso.fit(fit_X)
    out["isolation_forest"] = -iso.decision_function(all_X)

    if len(fit_X) >= 3:
        n_neighbors = min(35, len(fit_X) - 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True, contamination="auto")
        lof.fit(fit_X)
        out["local_outlier_factor"] = -lof.decision_function(all_X)
    else:
        skipped.append("local_outlier_factor requires at least 3 reference rows")

    if not args.skip_ocsvm and len(fit_X) >= 10:
        ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=min(max(args.target_fpr, 0.001), 0.5))
        ocsvm.fit(fit_X)
        out["one_class_svm"] = -ocsvm.decision_function(all_X)
    elif args.skip_ocsvm:
        skipped.append("one_class_svm skipped by --skip-ocsvm")
    else:
        skipped.append("one_class_svm requires at least 10 reference rows")

    return out, skipped


def _method_metrics(
    values: np.ndarray,
    decisions: np.ndarray,
    threshold: float,
    eval_mask: np.ndarray,
    labels: pd.DataFrame | None,
    zero_day_labels: list[str],
) -> dict:
    out = {
        "threshold": float(threshold),
        "alert_rate": round(float(decisions[eval_mask].mean()), 6) if eval_mask.any() else None,
        "score_distribution": _distribution(values[eval_mask]),
    }
    if labels is None:
        return out

    truth = labels["truth"].astype(str)
    comparable = eval_mask & truth.isin(["Normal", "Known-Attack"]).to_numpy()
    normal = eval_mask & (truth == "Normal").to_numpy()
    attack = eval_mask & (truth == "Known-Attack").to_numpy()
    target = (truth == "Known-Attack").to_numpy().astype(int)

    if normal.any():
        out["normal_fpr"] = round(float(decisions[normal].mean()), 6)
    if attack.any():
        out["attack_detection_rate"] = round(float(decisions[attack].mean()), 6)
    if comparable.any():
        pred = np.where(decisions, "Known-Attack", "Normal")
        out["detection_accuracy"] = round(float((pred[comparable] == truth.to_numpy()[comparable]).mean()), 6)
    if comparable.any() and len(np.unique(target[comparable])) == 2:
        out["auroc"] = round(float(roc_auc_score(target[comparable], values[comparable])), 6)
        out["auprc"] = round(float(average_precision_score(target[comparable], values[comparable])), 6)

    if zero_day_labels:
        zero_day_frame = pd.Series(zero_day_labels, name="attack_cat", dtype=str).to_frame()
        canonical_zero_day_labels = set(normalize_labels(zero_day_frame)["attack_cat"])
        zd_mask = eval_mask & labels["label"].isin(canonical_zero_day_labels).to_numpy()
        if zd_mask.any():
            out["ood_detection_rate"] = round(float(decisions[zd_mask].mean()), 6)
    return out


def _distribution(values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {}
    return {
        "min": float(np.min(clean)),
        "p50": float(np.quantile(clean, 0.50)),
        "p90": float(np.quantile(clean, 0.90)),
        "p95": float(np.quantile(clean, 0.95)),
        "p99": float(np.quantile(clean, 0.99)),
        "max": float(np.max(clean)),
        "mean": float(np.mean(clean)),
    }


def _rank_methods(methods: dict[str, dict]) -> list[dict]:
    def key(item: tuple[str, dict]) -> tuple[float, float, float]:
        metrics = item[1]
        return (
            float(metrics.get("auroc") or -1.0),
            float(metrics.get("ood_detection_rate") or -1.0),
            -float(metrics.get("normal_fpr") or 1.0),
        )

    ranked = []
    for name, metrics in sorted(methods.items(), key=key, reverse=True):
        ranked.append({
            "method": name,
            "auroc": metrics.get("auroc"),
            "auprc": metrics.get("auprc"),
            "ood_detection_rate": metrics.get("ood_detection_rate"),
            "normal_fpr": metrics.get("normal_fpr"),
            "alert_rate": metrics.get("alert_rate"),
        })
    return ranked


if __name__ == "__main__":
    raise SystemExit(main())
