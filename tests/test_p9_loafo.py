"""Synthetic P9 scientific-boundary tests; no target/test evaluation or training."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.p4_model_interface import FrozenVocabulary, StructuredModelInput  # noqa: E402
from ids.p8_loafo_protocol import P8KnownClassModel  # noqa: E402
from ids.p9_loafo import (  # noqa: E402
    TIE_PREFERENCE, centroid_distance, checked_input, choose_scorer,
    fit_centroids, fit_identity_threshold, identity_means, require_role,
    softmax_uncertainty,
)


def test_role_permissions_fail_closed():
    require_role("gradient_fit", "train")
    require_role("checkpoint_selection", "val")
    require_role("scorer_selection", "meta_known")
    require_role("scorer_selection", "surrogate_ood")
    require_role("threshold_fit", "calibration")
    for operation, role in (("gradient_fit", "surrogate_ood"),
                            ("gradient_fit", "target_held_out"),
                            ("scorer_selection", "calibration"),
                            ("scorer_selection", "val"),
                            ("threshold_fit", "surrogate_ood"),
                            ("checkpoint_selection", "meta_known")):
        with pytest.raises(PermissionError):
            require_role(operation, role)


def test_input_guard_rejects_consumed_test_and_p7_results(tmp_path):
    root = tmp_path
    for relative in ("data/UNSW_NB15_training-set.csv",
                     "data/UNSW_NB15_testing-set.csv",
                     "results/external_evaluation/p7/primary/metric.json",
                     "results/loafo/p8/loafo_matrix.json"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x")
    assert checked_input(root, Path("data/UNSW_NB15_training-set.csv"), "train").name == (
        "UNSW_NB15_training-set.csv")
    for relative, kind in (("data/UNSW_NB15_testing-set.csv", "train"),
                           ("results/external_evaluation/p7/primary/metric.json", "p8"),
                           ("results/external_evaluation/p7/primary/metric.json", "p9"),
                           ("data/UNSW_NB15_testing-set.csv", "p8")):
        with pytest.raises(PermissionError):
            checked_input(root, Path(relative), kind)


def test_fixed_four_embedding_variable_seven_class_float64_model():
    vocab = {name: FrozenVocabulary.from_p3_mapping(name, {"value:tcp": 0})
             for name in ("proto", "service", "state")}
    classes = ["Normal", "Analysis", "Backdoors", "DoS", "Generic", "Shellcode", "Worms"]
    model = P8KnownClassModel(vocab, classes)
    assert model.head.out_features == 7
    assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
    assert all(layer.embedding_dim == 4 for layer in model.backbone.embeddings.values())
    batch = StructuredModelInput(torch.zeros((2, 58), dtype=torch.float64),
                                 torch.tensor([2, 1], dtype=torch.int64),
                                 torch.tensor([2, 1], dtype=torch.int64),
                                 torch.tensor([2, 1], dtype=torch.int64))
    assert tuple(model(batch).shape) == (2, 7)
    with pytest.raises(TypeError):
        StructuredModelInput(torch.zeros((2, 58), dtype=torch.float32),
                             batch.proto, batch.service, batch.state)
    with pytest.raises(TypeError):
        StructuredModelInput(batch.continuous, batch.proto.float(), batch.service, batch.state)


def test_train_only_seven_centroids_and_deterministic_hash():
    classes = ["Normal", "Analysis", "Backdoors", "DoS", "Generic", "Shellcode", "Worms"]
    labels = list(reversed(classes)) + classes
    values = np.arange(14 * 32, dtype=np.float64).reshape(14, 32)
    state_a, digest_a = fit_centroids(values, labels, role="train",
                                      class_order=classes, checkpoint_hash="a" * 64)
    state_b, digest_b = fit_centroids(values, labels, role="train",
                                      class_order=classes, checkpoint_hash="a" * 64)
    assert state_a.shape == (7, 32) and np.array_equal(state_a, state_b)
    assert digest_a == digest_b
    assert np.array_equal(state_a[0], values[np.asarray(labels) == "Normal"].mean(axis=0))
    assert centroid_distance(values, state_a, 7).shape == (14,)
    with pytest.raises(PermissionError):
        fit_centroids(values, labels, role="val", class_order=classes,
                      checkpoint_hash="a" * 64)


def test_identity_mean_order_invariant_and_duplicates_do_not_increase_identity_count():
    scores = np.asarray([0.4, 0.2, 0.6, 0.4], dtype=np.float64)
    identities = ["b", "a", "a", "b"]
    keys, means = identity_means(scores, identities)
    keys2, means2 = identity_means(scores[::-1].copy(), list(reversed(identities)))
    assert keys == keys2 == ["a", "b"]
    assert np.array_equal(means, means2)
    assert np.array_equal(means, np.asarray([0.4, 0.4]))


def test_identity_calibration_q99_higher_and_strict_alert():
    row_scores = np.asarray([0.1, 0.3, 0.8, 0.9], dtype=np.float64)
    identities = ["a", "a", "b", "c"]
    artifact = fit_identity_threshold(row_scores, identities, role="calibration")
    _, identity_scores = identity_means(row_scores, identities)
    assert artifact["calibration_row_count"] == 4
    assert artifact["calibration_identity_count"] == 3
    assert artifact["threshold"] == float(np.quantile(identity_scores, 0.99, method="higher"))
    assert artifact["numpy_method"] == "higher" and artifact["comparison"] == "score > threshold"
    assert not bool(np.asarray([artifact["threshold"]]) > artifact["threshold"])
    with pytest.raises(PermissionError):
        fit_identity_threshold(row_scores, identities, role="surrogate_ood")


def test_two_scorers_and_tie_preference():
    logits = np.asarray([[1.0] * 7, [10.0] + [0.0] * 6], dtype=np.float64)
    scores = softmax_uncertainty(logits, 7)
    assert scores[0] > scores[1]
    assert choose_scorer(0.7, 0.7)["selected_scorer"] == TIE_PREFERENCE
    assert choose_scorer(0.8, 0.7)["selected_scorer"] == "negative_max_softmax"
    with pytest.raises(TypeError):
        softmax_uncertainty(logits.astype(np.float32), 7)
