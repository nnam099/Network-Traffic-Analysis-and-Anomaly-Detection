from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
DASHBOARD_DIR = ROOT_DIR / "dashboard"
for path in [SRC_DIR, DASHBOARD_DIR, ROOT_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ids.batch_evaluator import _labeled_evaluation, summarize_scores  # noqa: E402
from ids.dataset import normalize_labels, prepare_splits  # noqa: E402
from scripts.evaluate_baselines import _labels, _method_metrics  # noqa: E402
from views_batch import compute_normal_fpr_metrics  # noqa: E402


class CanonicalLabelTests(unittest.TestCase):
    def test_training_pipeline_canonicalizes_backdoor(self):
        rows = []
        for label, count, offset in [("Normal", 10, 0), ("DoS", 10, 100), ("Backdoor", 3, 200)]:
            rows.extend(
                {"attack_cat": label, "feature": float(offset + index)}
                for index in range(count)
            )

        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_splits(pd.DataFrame(rows), seed=7)

        self.assertEqual(set(splits["y_zd"]), {"Backdoors"})
        self.assertIn("Backdoors", splits["zd_cats"])
        direct = normalize_labels(pd.DataFrame({"attack_cat": ["Backdoor"]}))
        self.assertEqual(direct.loc[0, "attack_cat"], "Backdoors")

    def test_batch_evaluation_canonicalizes_backdoor(self):
        scores = pd.DataFrame(
            {
                "predicted_class": ["Zero-Day Candidate"],
                "classifier_class": ["Normal"],
                "is_zeroday": [True],
            }
        )
        evaluation = _labeled_evaluation(
            scores,
            labels=pd.Series(["Backdoor"]),
            truth_verdict=pd.Series(["Known-Attack"]),
            class_names=["Normal", "DoS"],
            zero_day_labels=["Backdoors"],
        )

        self.assertEqual(evaluation["ood_detection_rate"], 1.0)
        self.assertEqual(evaluation["zero_day_recall_per_family"]["Backdoors"]["support"], 1)

    def test_baseline_evaluation_canonicalizes_backdoor(self):
        labels = _labels(pd.DataFrame({"attack_cat": ["Backdoor"]}), "attack_cat")

        self.assertIsNotNone(labels)
        self.assertEqual(labels.loc[0, "label"], "Backdoors")
        self.assertEqual(labels.loc[0, "truth"], "Known-Attack")
        metrics = _method_metrics(
            values=np.asarray([0.9]),
            decisions=np.asarray([True]),
            threshold=0.5,
            eval_mask=np.asarray([True]),
            labels=labels,
            zero_day_labels=["Backdoors"],
        )
        self.assertEqual(metrics["ood_detection_rate"], 1.0)


class FalsePositiveRateTests(unittest.TestCase):
    def test_normal_ood_alert_and_known_classifier_fpr_are_distinct(self):
        scores = pd.DataFrame(
            {
                "predicted_class": ["Normal", "Known-Attack", "Zero-Day Candidate"],
                "classifier_class": ["Normal", "DoS", "Normal"],
                "is_zeroday": [False, False, True],
                "hybrid": [0.1, 0.4, 0.9],
                "ae_re": [0.1, 0.2, 0.8],
                "softmax": [0.05, 0.2, 0.7],
                "max_prob": [0.95, 0.8, 0.3],
            }
        )
        raw = pd.DataFrame({"attack_cat": ["Normal", "Normal", "Normal"]})

        summary = summarize_scores(
            scores,
            raw_df=raw,
            label_col="attack_cat",
            class_names=["Normal", "DoS"],
            zero_day_labels=["Backdoors"],
        )

        self.assertEqual(summary["normal_ood_fpr"], 0.333333)
        self.assertEqual(summary["normal_alert_fpr"], 0.666667)
        self.assertEqual(summary["known_classifier_fpr"], 0.333333)
        self.assertEqual(summary["normal_ood_false_positive_count"], 1)
        self.assertEqual(summary["normal_alert_false_positive_count"], 2)
        self.assertEqual(summary["known_classifier_false_positive_count"], 1)
        self.assertEqual(summary["normal_ood_classifier_overlap_count"], 0)
        self.assertEqual(
            summary["normal_alert_false_positive_count"],
            summary["normal_ood_false_positive_count"]
            + summary["known_classifier_false_positive_count"]
            - summary["normal_ood_classifier_overlap_count"],
        )

        dashboard_rows = scores.assign(
            ground_truth="Normal",
            detection=scores["predicted_class"],
        )
        dashboard_metrics = compute_normal_fpr_metrics(dashboard_rows)
        self.assertIsNotNone(dashboard_metrics)
        self.assertAlmostEqual(dashboard_metrics["normal_ood_fpr"], 1 / 3)
        self.assertAlmostEqual(dashboard_metrics["normal_alert_fpr"], 2 / 3)
        self.assertAlmostEqual(dashboard_metrics["known_classifier_fpr"], 1 / 3)
        self.assertEqual(dashboard_metrics["normal_ood_classifier_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
