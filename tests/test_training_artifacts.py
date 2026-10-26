from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.batch_evaluator import load_ids_artifacts  # noqa: E402
from ids.models import IDSModel  # noqa: E402
from ids.training import save_artifacts  # noqa: E402


def test_saved_artifacts_keep_calibrated_thresholds(tmp_path):
    n_features = 4
    model = IDSModel(n_features=n_features, n_classes=2, hidden=8, ae_hidden=4)
    scaler = RobustScaler().fit(np.zeros((8, n_features), dtype=np.float32))
    label_encoder = LabelEncoder().fit(["Normal", "DoS"])
    splits = {
        "feat_cols": [f"feature_{index}" for index in range(n_features)],
        "n_features": n_features,
        "n_classes": 2,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "known_cats": ["Normal", "DoS"],
        "zd_cats": ["Fuzzers"],
        "categorical_maps": {},
    }
    hybrid_meta = {
        "type": "logistic_regression",
        "features": ["ae_re", "softmax"],
        "coef": [1.5, 2.5],
        "intercept": -1.0,
    }
    thresholds = {
        "hybrid": 0.75,
        "ae_re": 0.25,
        "hybrid_meta": hybrid_meta,
    }

    model_path, pipeline_path = save_artifacts(
        model,
        splits,
        thresholds=thresholds,
        history=[],
        centroids=torch.zeros(2, 8),
        save_dir=str(tmp_path),
        hybrid_meta=hybrid_meta,
    )

    artifacts = load_ids_artifacts(model_path, pipeline_path)
    assert artifacts.thresholds["hybrid"] == 0.75
    assert artifacts.thresholds["ae_re"] == 0.25
    assert artifacts.thresholds["hybrid_meta"] == hybrid_meta
    assert artifacts.checkpoint["hybrid_meta"] == hybrid_meta
