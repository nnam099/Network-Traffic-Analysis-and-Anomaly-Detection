"""Static and adaptive anomaly threshold helpers for reconstruction-error scoring."""

from collections import deque
import numpy as np


def static_threshold(scores, target_fpr=0.05):
    return float(np.quantile(np.asarray(scores), 1.0 - target_fpr))


class AdaptiveThreshold:
    def __init__(self, window_size=1000, target_fpr=0.05):
        self.window_size = int(window_size)
        self.target_fpr = float(target_fpr)
        self.buffer = deque(maxlen=self.window_size)
        self.threshold = float("inf")

    def update(self, re_scores: np.ndarray):
        scores = np.asarray(re_scores, dtype=np.float64).reshape(-1)
        scores = scores[np.isfinite(scores)]
        self.buffer.extend(float(score) for score in scores)
        if self.buffer:
            self.threshold = float(
                np.quantile(
                    np.asarray(self.buffer, dtype=np.float64),
                    1.0 - self.target_fpr,
                )
            )
        return self.threshold

    def __call__(self, re_score: float) -> bool:
        return bool(float(re_score) > self.threshold)
