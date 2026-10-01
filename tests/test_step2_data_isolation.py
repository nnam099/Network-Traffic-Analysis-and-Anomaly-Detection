from __future__ import annotations

import contextlib
import inspect
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ids.dataset import (  # noqa: E402
    KNOWN_ATTACK_CATS,
    ZERO_DAY_ATTACK_CATS,
    assert_lofo_retraining_gate,
    assert_training_isolation,
    build_lofo_surrogate_mask,
    feature_fingerprints,
    load_official_unsw_splits,
    model_input_fingerprints,
    prepare_official_splits,
    prepare_splits,
)
from ids.evaluator import (  # noqa: E402
    evaluate_zero_day_from_scores,
    roc_operating_point_at_fpr,
)
from ids.lineage import build_experiment_lineage  # noqa: E402
from ids.target_isolation import (  # noqa: E402
    CANONICAL_IDENTITY_VERSION,
    FIT_ROLE_NAMES,
    MODEL_INPUT_IDENTITY_VERSION,
    P0_FEATURE_SCHEMA,
    REQUIRED_INVARIANTS,
    assert_target_fold_retraining_gate,
    build_data_quality_report,
    build_target_isolation_context,
    canonical_feature_fingerprints,
    prepare_target_isolated_fold,
)
from ids.artifact_validator import validate_artifact_contract  # noqa: E402
from ids.training import run_full  # noqa: E402


def _row(label: str, value: float) -> dict:
    return {
        "attack_cat": label,
        "label": int(label != "Normal"),
        "dur": value + 0.01,
        "sbytes": value + 10.0,
        "dbytes": value + 20.0,
        "spkts": value + 2.0,
        "dpkts": value + 3.0,
        "proto": "tcp" if int(value) % 2 else "udp",
        "service": "http",
        "state": "FIN",
    }


def _official_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []
    for class_index, label in enumerate(KNOWN_ATTACK_CATS):
        for row_index in range(20):
            train_rows.append(_row(label, class_index * 1000 + row_index))
        for row_index in range(5):
            test_rows.append(_row(label, class_index * 1000 + 100 + row_index))
    raw_ood_names = ["Fuzzers", "Analysis", "Backdoor", "Shellcode", "Worms"]
    for class_index, label in enumerate(raw_ood_names):
        for row_index in range(4):
            train_rows.append(_row(label, 10000 + class_index * 100 + row_index))
        for row_index in range(2):
            test_rows.append(_row(label, 20000 + class_index * 100 + row_index))
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


class Step2DataIsolationTests(unittest.TestCase):
    def test_canonical_identity_is_vocabulary_independent_and_excludes_labels(self):
        train_df, test_df = _official_frames()
        context = build_target_isolation_context(train_df, test_df)
        probe = pd.DataFrame([
            {**_row("Normal", 1.0), "proto": " TCP ", "id": 1},
            {**_row("Fuzzers", 1.0), "proto": "tcp", "id": 999},
        ])

        fingerprints = canonical_feature_fingerprints(
            probe, context["base_feat_cols"], context["feat_cols"]
        )

        self.assertEqual(int(fingerprints.iloc[0]), int(fingerprints.iloc[1]))

    def test_canonical_missing_category_is_not_a_literal_category(self):
        train_df, test_df = _official_frames()
        context = build_target_isolation_context(train_df, test_df)
        probe = pd.DataFrame([
            {**_row("Normal", 1.0), "proto": None},
            {**_row("Normal", 1.0), "proto": "<missing>"},
        ], index=[5, 500])
        fingerprints = canonical_feature_fingerprints(
            probe, context["base_feat_cols"], context["feat_cols"]
        )
        self.assertNotEqual(int(fingerprints.iloc[0]), int(fingerprints.iloc[1]))

    def test_non_target_test_changes_do_not_change_fold_fitting(self):
        train_df, test_df = _official_frames()
        changed_test = test_df.copy()
        mask = changed_test["attack_cat"] == "Normal"
        changed_test.loc[mask, "proto"] = "never-fit-this-test-category"
        changed_test.loc[mask, "dur"] = 12345678.0
        folds = []
        for frame in (test_df, changed_test):
            context = build_target_isolation_context(train_df, frame)
            folds.append(prepare_target_isolated_fold(
                context, "Fuzzers", 42,
                include_post_transform_label_diagnostic=False,
            ))
        self.assertEqual(folds[0]["categorical_maps"], folds[1]["categorical_maps"])
        for attribute in ("center_", "scale_"):
            np.testing.assert_array_equal(
                getattr(folds[0]["scaler"], attribute),
                getattr(folds[1]["scaler"], attribute),
            )
        for role in ("train", "val", "meta_known", "calibration", "ood_train"):
            np.testing.assert_array_equal(folds[0][f"X_{role}"], folds[1][f"X_{role}"])

    def test_target_gate_checks_surrogate_against_every_known_role(self):
        train_df, test_df = _official_frames()
        duplicate = train_df.iloc[0].copy()
        duplicate["attack_cat"] = "Analysis"
        duplicate["label"] = 1
        train_df = pd.concat([train_df, duplicate.to_frame().T], ignore_index=True)
        for column in ("dur", "sbytes", "dbytes", "spkts", "dpkts", "label"):
            train_df[column] = pd.to_numeric(train_df[column])
        context = build_target_isolation_context(train_df, test_df)
        fold = prepare_target_isolated_fold(
            context, "Fuzzers", 42,
            include_post_transform_label_diagnostic=False,
        )
        self.assertFalse(fold["report"]["invariants"][
            "all_fit_roles_canonical_fingerprint_disjoint"
        ])
        # Serialized PASS flags must never override the live collision evidence.
        fold["report"]["invariants"] = {name: True for name in REQUIRED_INVARIANTS}
        with self.assertRaisesRegex(AssertionError, "all_fit_roles_canonical_collision"):
            assert_lofo_retraining_gate(fold)

    def test_target_gate_rejects_missing_invariants_even_if_report_says_pass(self):
        fold = {"report": {
            "protocol": "target_isolated_outer_lofo_v1",
            "target_family": "Fuzzers", "seed": 42,
            "gate": "PASS", "invariants": {},
        }}
        with self.assertRaisesRegex(AssertionError, "missing_fit_role_evidence"):
            assert_lofo_retraining_gate(fold)

    def test_target_gate_recomputes_model_inputs_at_the_training_boundary(self):
        rng = np.random.default_rng(42)
        target = rng.normal(size=(1, 61)).astype(np.float32)
        fold = {
            "feat_cols": list(P0_FEATURE_SCHEMA),
            "X_target_identity": target,
            "target_canonical_fingerprints": np.asarray([9999], dtype=np.uint64),
            "fit_role_evidence": {},
            "report": {
                "protocol": "target_isolated_outer_lofo_v1",
                "target_family": "Fuzzers", "seed": 42,
                "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
                "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
                "invariants": {name: True for name in REQUIRED_INVARIANTS},
            },
        }
        labels = np.asarray(["Normal"] * 20 + [
            label for label in KNOWN_ATTACK_CATS if label != "Normal"
            for _ in range(2)
        ])
        for offset, role in enumerate(FIT_ROLE_NAMES):
            values = rng.normal(size=(len(labels), 61)).astype(np.float32)
            key = "ood_train" if role == "surrogate_ood" else role
            fold[f"X_{key}"] = values
            row_ids = np.arange(len(labels)) + offset * 100
            fold["fit_role_evidence"][role] = {
                "row_ids": row_ids,
                "source_roles": np.asarray(["official_train"] * len(labels)),
                "labels": (
                    np.asarray(["Analysis"] * len(labels))
                    if role == "surrogate_ood" else labels.copy()
                ),
                "canonical_fingerprints": row_ids.astype(np.uint64),
                "model_input_fingerprints": model_input_fingerprints(values).to_numpy(),
            }
        fold["scaler_fit_row_ids"] = fold["fit_role_evidence"]["train"]["row_ids"]
        fold["scaler_fit_canonical_fingerprints"] = fold["fit_role_evidence"]["train"][
            "canonical_fingerprints"
        ]
        assert_lofo_retraining_gate(fold)
        fold["X_calibration"][0] = target[0]
        with self.assertRaisesRegex(AssertionError, "target_model_collision/calibration"):
            assert_lofo_retraining_gate(fold)

    def test_target_fold_uses_test_identity_only_to_purge_fit_roles(self):
        train_df, test_df = _official_frames()
        target_test_index = int(
            np.flatnonzero(test_df["attack_cat"].to_numpy() == "Fuzzers")[0]
        )
        normal_train_index = int(
            np.flatnonzero(train_df["attack_cat"].to_numpy() == "Normal")[0]
        )
        replacement = test_df.iloc[target_test_index].copy()
        replacement["attack_cat"] = "Normal"
        replacement["label"] = 0
        train_df.iloc[normal_train_index] = replacement
        context = build_target_isolation_context(train_df, test_df)

        fold = prepare_target_isolated_fold(
            context,
            "Fuzzers",
            42,
            expected_schema_sha256=context["feature_schema_sha256"],
            include_post_transform_label_diagnostic=False,
        )

        report = fold["report"]
        self.assertGreaterEqual(report["known_removed"]["rows"], 1)
        self.assertTrue(
            report["invariants"]["target_canonical_fingerprints_absent_from_fit"]
        )
        self.assertTrue(report["invariants"]["official_test_excluded_from_fit"])
        self.assertNotIn(
            "Fuzzers", report["surrogate_ood"]["families_remaining"]
        )

    def test_target_fold_schema_is_stable_across_targets_and_seeds(self):
        train_df, test_df = _official_frames()
        context = build_target_isolation_context(train_df, test_df)
        observed = set()
        for target, seed in (("Fuzzers", 42), ("Backdoors", 43), ("Worms", 44)):
            fold = prepare_target_isolated_fold(
                context,
                target,
                seed,
                expected_schema_sha256=context["feature_schema_sha256"],
                include_post_transform_label_diagnostic=False,
            )
            observed.add(fold["feature_schema_sha256"])
            self.assertEqual(fold["report"]["feature_count"], len(context["feat_cols"]))
        self.assertEqual(observed, {context["feature_schema_sha256"]})

    def test_target_gate_fails_closed_and_collision_report_counts_conflicts(self):
        train_df, test_df = _official_frames()
        target_row = test_df[test_df["attack_cat"] == "Fuzzers"].iloc[0].copy()
        target_row["attack_cat"] = "Normal"
        target_row["label"] = 0
        train_df.iloc[0] = target_row
        context = build_target_isolation_context(train_df, test_df)
        quality = build_data_quality_report(context)
        self.assertGreater(
            quality["canonical_label_collisions"][
                "known_ood_conflicting_fingerprints"
            ],
            0,
        )
        fake_fold = {
            "report": {
                "target_family": "Fuzzers",
                "seed": 42,
                "invariants": {"target_model_input_fingerprints_absent_from_fit": False},
            }
        }
        with self.assertRaisesRegex(AssertionError, "target-isolated retraining blocked"):
            assert_target_fold_retraining_gate(fake_fold)

    def test_fingerprints_match_model_identity_contract(self):
        left = _row("Normal", 1.0)
        right = dict(left)
        left.update({"id": 1, "srcip": "10.0.0.1", "__source_role": "left"})
        right.update({"id": 2, "srcip": "10.0.0.2", "__source_role": "right"})
        probe = pd.DataFrame([left, right])

        fingerprints = feature_fingerprints(probe)

        self.assertEqual(int(fingerprints.iloc[0]), int(fingerprints.iloc[1]))
        changed = probe.copy()
        changed.loc[1, "dur"] += 1.0
        changed_fingerprints = feature_fingerprints(changed)
        self.assertNotEqual(
            int(changed_fingerprints.iloc[0]), int(changed_fingerprints.iloc[1])
        )

        vectors = np.asarray([[0.0, 1.0], [-0.0, 1.0], [0.0, 2.0]], dtype=np.float32)
        vector_fingerprints = model_input_fingerprints(vectors)
        self.assertEqual(
            int(vector_fingerprints.iloc[0]), int(vector_fingerprints.iloc[1])
        )
        self.assertNotEqual(
            int(vector_fingerprints.iloc[0]), int(vector_fingerprints.iloc[2])
        )

    def test_metadata_is_not_a_model_feature_but_fingerprinting_still_works(self):
        train_df, test_df = _official_frames()
        train_df["__numeric_metadata"] = np.arange(len(train_df), dtype=float)
        test_df["__numeric_metadata"] = np.arange(len(test_df), dtype=float)
        fingerprint_probe = pd.DataFrame([
            {**_row("Normal", 1.0), "__source_role": "left"},
            {**_row("Normal", 1.0), "__source_role": "right"},
        ])

        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=42)

        self.assertFalse(any(name.startswith("__") for name in splits["feat_cols"]))
        self.assertNotIn("__feature_fingerprint", splits["feat_cols"])
        fingerprints = feature_fingerprints(fingerprint_probe)
        self.assertEqual(int(fingerprints.iloc[0]), int(fingerprints.iloc[1]))

    def test_inference_artifact_rejects_metadata_features(self):
        checkpoint = {
            "model_state_dict": {},
            "n_features": 2,
            "n_classes": 1,
            "feat_cols": ["dur", "__feature_fingerprint"],
        }
        pipeline = {
            "scaler": SimpleNamespace(n_features_in_=2),
            "label_encoder": SimpleNamespace(classes_=np.asarray(["Normal"])),
            "feature_names": ["dur", "__feature_fingerprint"],
            "thresholds": {"hybrid": 0.5},
        }

        result = validate_artifact_contract(checkpoint, pipeline)

        self.assertFalse(result.ok)
        self.assertTrue(any("metadata columns" in error for error in result.errors))

    def test_feature_schema_is_identical_across_required_seeds(self):
        train_df, test_df = _official_frames()
        schemas = []
        with contextlib.redirect_stdout(io.StringIO()):
            for seed in (42, 43, 44):
                splits = prepare_official_splits(train_df, test_df, seed=seed)
                schemas.append((
                    splits["feat_cols"],
                    splits["n_features"],
                    splits["feature_schema_sha256"],
                ))

        self.assertEqual(schemas[0], schemas[1])
        self.assertEqual(schemas[1], schemas[2])

    def test_official_test_cannot_change_preprocessing_fit(self):
        train_df, test_df = _official_frames()
        mutated_test = test_df.copy()
        mutated_test["dur"] = mutated_test["dur"] * 1000000
        mutated_test["proto"] = "test-only-protocol"
        mutated_test["__test_metadata"] = 999.0

        with contextlib.redirect_stdout(io.StringIO()):
            baseline = prepare_official_splits(train_df, test_df, seed=42)
            mutated = prepare_official_splits(train_df, mutated_test, seed=42)

        self.assertEqual(baseline["feat_cols"], mutated["feat_cols"])
        self.assertEqual(baseline["categorical_maps"], mutated["categorical_maps"])
        np.testing.assert_array_equal(baseline["scaler"].center_, mutated["scaler"].center_)
        np.testing.assert_array_equal(baseline["scaler"].scale_, mutated["scaler"].scale_)
        self.assertEqual(
            baseline["fit_role_contract"]["final_evaluation"], ["official_test"]
        )
        for key in (
            "source_roles_train", "source_roles_val", "source_roles_meta_known",
            "source_roles_calibration", "source_roles_ood_train",
        ):
            self.assertNotIn("official_test", set(baseline[key]))

    def test_label_assisted_adaptive_threshold_is_disabled_for_benchmark(self):
        with self.assertRaisesRegex(ValueError, "disabled for scientific benchmarks"):
            run_full(SimpleNamespace(seed=42, adaptive_threshold=True))

    def test_backdoors_is_quarantined_from_all_known_splits(self):
        train_df, test_df = _official_frames()
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=7)

        encoder = splits["label_encoder"]
        for key in ["y_train", "y_val", "y_meta_known", "y_calibration"]:
            decoded = set(encoder.inverse_transform(splits[key]))
            self.assertTrue(decoded.issubset(set(KNOWN_ATTACK_CATS)))
            self.assertTrue(decoded.isdisjoint(set(ZERO_DAY_ATTACK_CATS)))
        self.assertIn("Backdoors", set(splits["y_ood_train"]))
        self.assertIn("Backdoors", set(splits["y_ood_test"]))

    def test_runtime_assertions_cover_sources_fingerprints_and_lofo_target(self):
        train_df, test_df = _official_frames()
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=11)

        for target in ZERO_DAY_ATTACK_CATS:
            surrogate, _ = build_lofo_surrogate_mask(splits, target)
            result = assert_training_isolation(
                splits,
                target_family=target,
                surrogate_labels=splits["y_ood_train"][surrogate],
                surrogate_fingerprints=splits["fingerprints_ood_train"][surrogate],
                target_fingerprints=splits["fingerprints_ood_train"][
                    splits["y_ood_train"] == target
                ],
            )
            self.assertTrue(result["official_test_excluded_from_fit"])
            self.assertTrue(result["known_split_fingerprints_disjoint"])
            self.assertTrue(result["known_split_rows_disjoint"])
            self.assertTrue(result["target_family_excluded_from_surrogate"])

        corrupted = dict(splits)
        corrupted["source_roles_calibration"] = np.asarray(["official_test"])
        with self.assertRaisesRegex(AssertionError, "official testing set leaked"):
            assert_training_isolation(corrupted)

        with self.assertRaisesRegex(AssertionError, "target OOD family Backdoors leaked"):
            assert_training_isolation(
                splits,
                target_family="Backdoors",
                surrogate_labels=np.asarray(["Fuzzers", "Backdoors"]),
            )

    def test_known_partitions_are_row_and_exact_model_input_disjoint(self):
        train_df, test_df = _official_frames()
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=42)

        keys = ("train", "val", "meta_known", "calibration")
        for left_index, left in enumerate(keys):
            for right in keys[left_index + 1:]:
                self.assertTrue(
                    set(splits[f"row_ids_{left}"]).isdisjoint(
                        set(splits[f"row_ids_{right}"])
                    )
                )
                self.assertTrue(
                    set(splits[f"fingerprints_{left}"]).isdisjoint(
                        set(splits[f"fingerprints_{right}"])
                    )
                )
        self.assertEqual(
            set(splits["known_split_model_input_purge"]), set(keys)
        )

    def test_retraining_gate_rejects_target_fingerprint_in_known_fit_role(self):
        train_df, test_df = _official_frames()
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=42)
        contaminated = dict(splits)
        contaminated["fingerprints_train"] = splits["fingerprints_train"].copy()
        target_index = int(np.flatnonzero(splits["y_ood_train"] == "Backdoors")[0])
        contaminated["fingerprints_train"][0] = splits["fingerprints_ood_train"][
            target_index
        ]

        with self.assertRaisesRegex(
            AssertionError, "corrected retraining blocked.*Backdoors/train"
        ):
            assert_lofo_retraining_gate(contaminated)

    def test_official_test_scoring_occurs_after_all_fit_steps(self):
        source = inspect.getsource(run_full)
        retraining_gate = source.index("assert_lofo_retraining_gate(splits)")
        model_build = source.index("model = IDSModel(")
        first_test_score = source.index(
            "collect_ood_scores(model, splits['X_test']"
        )
        self.assertLess(retraining_gate, model_build)
        self.assertLess(source.index("fit_hybrid_meta_learner("), first_test_score)
        self.assertLess(source.index("thresholds = calibrate("), first_test_score)

    def test_lofo_purges_other_labels_that_share_target_fingerprints(self):
        train_df, test_df = _official_frames()
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=42)
        labels = splits["y_ood_train"]
        fingerprints = splits["fingerprints_ood_train"].copy()
        target_index = int(np.flatnonzero(labels == "Backdoors")[0])
        other_index = int(np.flatnonzero(labels == "Analysis")[0])
        fingerprints[other_index] = fingerprints[target_index]
        contaminated = dict(splits)
        contaminated["fingerprints_ood_train"] = fingerprints

        mask, audit = build_lofo_surrogate_mask(contaminated, "Backdoors")

        self.assertFalse(mask[target_index])
        self.assertFalse(mask[other_index])
        self.assertEqual(audit["surrogate_rows_removed_due_to_target_fingerprint"], 1)
        self.assertEqual(
            audit["target_surrogate_fingerprint_intersection_after_purge"], 0
        )

    def test_unseen_fingerprint_partition_excludes_shared_test_rows(self):
        train_df, test_df = _official_frames()
        test_df.iloc[0] = train_df.iloc[0]
        with contextlib.redirect_stdout(io.StringIO()):
            splits = prepare_official_splits(train_df, test_df, seed=42)

        self.assertFalse(bool(splits["test_unseen_fingerprint_mask"][0]))
        contamination = splits["fingerprint_contamination"]
        self.assertGreaterEqual(contamination["shared_fingerprints"], 1)
        self.assertGreaterEqual(contamination["shared_test_rows"], 1)
        self.assertEqual(
            contamination["shared_test_rows"] + contamination["unseen_test_rows"],
            len(test_df),
        )

    def test_undeclared_class_fails_fast_instead_of_using_row_count(self):
        rows = [_row("Normal", float(index)) for index in range(20)]
        rows += [_row("MysteryAttack", 100.0 + index) for index in range(3)]
        with self.assertRaisesRegex(ValueError, "Undeclared attack classes.*Mysteryattack"):
            prepare_splits(pd.DataFrame(rows), known_cats=["Normal"], zd_cats=[])

    def test_loader_keeps_official_file_roles_separate(self):
        train_df, test_df = _official_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            train_df.to_csv(data_dir / "UNSW_NB15_training-set.csv", index=False)
            test_df.to_csv(data_dir / "UNSW_NB15_testing-set.csv", index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                roles = load_official_unsw_splits(data_dir)

        self.assertEqual(set(roles["train"]["__source_role"]), {"official_train"})
        self.assertEqual(set(roles["test"]["__source_role"]), {"official_test"})
        self.assertIsNone(roles["calibration"])

    def test_target_metrics_use_preselected_method_not_target_auc_winner(self):
        known = {
            "hybrid": np.asarray([0.1, 0.2, 0.3]),
            "ae_re": np.asarray([0.1, 0.2, 0.3]),
        }
        target = {
            "hybrid": np.asarray([0.4, 0.5]),
            "ae_re": np.asarray([0.0, 0.1]),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            result = evaluate_zero_day_from_scores(
                known,
                target,
                np.asarray(["Backdoors", "Backdoors"]),
                {"hybrid": 0.25, "ae_re": 0.25},
                selected_method="hybrid",
                y_known=np.asarray([0, 1, 0]),
                known_predictions=np.asarray([0, 1, 1]),
                normal_idx=0,
            )

        self.assertEqual(result["_selected_method"], "hybrid")
        self.assertEqual(result["_per_class"]["Backdoors"]["recall"], 1.0)
        self.assertGreater(result["hybrid"]["auc"], result["ae_re"]["auc"])
        self.assertEqual(result["_deprecated_metric_aliases"]["auc"], "ood_auroc")
        self.assertIn("ood_auprc", result["ood_metrics"])
        self.assertIn("ood_tpr_at_5pct_fpr", result["ood_metrics"])
        self.assertEqual(result["ood_metrics"]["known_false_unknown_rate"], 1 / 3)
        self.assertEqual(result["ood_metrics"]["normal_ood_fpr"], 1 / 2)
        self.assertEqual(result["ood_metrics"]["total_alert_fpr"], 1 / 2)

    def test_tpr_operating_point_never_exceeds_fpr_budget(self):
        point = roc_operating_point_at_fpr(
            np.asarray([0.0, 0.005, 0.012, 0.03]),
            np.asarray([0.0, 0.4, 0.9, 1.0]),
            np.asarray([np.inf, 0.8, 0.5, 0.1]),
            0.01,
        )

        self.assertEqual(point["target_fpr"], 0.01)
        self.assertEqual(point["achieved_fpr"], 0.005)
        self.assertEqual(point["tpr"], 0.4)
        self.assertEqual(point["threshold"], 0.8)

        floating_edge = roc_operating_point_at_fpr(
            np.asarray([0.0, 0.01, 0.0100000000005]),
            np.asarray([0.0, 0.4, 0.9]),
            np.asarray([np.inf, 0.8, 0.5]),
            0.01,
        )
        self.assertLessEqual(floating_edge["achieved_fpr"], 0.01)
        self.assertEqual(floating_edge["tpr"], 0.4)

    def test_tpr_operating_point_handles_duplicate_fpr_deterministically(self):
        point = roc_operating_point_at_fpr(
            np.asarray([0.0, 0.01, 0.01, 0.02]),
            np.asarray([0.0, 0.4, 0.6, 0.8]),
            np.asarray([np.inf, 0.9, 0.8, 0.7]),
            0.01,
        )

        self.assertEqual(point["achieved_fpr"], 0.01)
        self.assertEqual(point["tpr"], 0.6)
        self.assertEqual(point["threshold"], 0.8)

    def test_tpr_operating_point_rejects_unreachable_or_invalid_budget(self):
        with self.assertRaisesRegex(ValueError, "no point within"):
            roc_operating_point_at_fpr(
                np.asarray([0.02, 0.03]),
                np.asarray([0.1, 0.2]),
                np.asarray([0.9, 0.8]),
                0.01,
            )
        with self.assertRaisesRegex(ValueError, r"within \[0, 1\]"):
            roc_operating_point_at_fpr(
                np.asarray([0.0]), np.asarray([0.0]), np.asarray([np.inf]), 1.1
            )

    def test_lineage_contains_required_reproducibility_fields(self):
        train_df, test_df = _official_frames()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            train_path = data_dir / "train.csv"
            test_path = data_dir / "test.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            with contextlib.redirect_stdout(io.StringIO()):
                roles = load_official_unsw_splits(
                    data_dir, train_files=[train_path], test_files=[test_path]
                )
                splits = prepare_official_splits(roles["train"], roles["test"], seed=42)
            lineage = build_experiment_lineage(
                SimpleNamespace(seed=42, target_fpr=0.01),
                roles,
                splits,
                repo_dir=ROOT_DIR,
            )

        for key in (
            "git", "invocation", "execution_environment", "seed",
            "dataset_files", "config", "feature_schema",
            "feature_count", "feature_schema_sha256", "split_counts",
            "train_test_role_contract", "threshold_calibration_population",
            "package_versions",
        ):
            self.assertIn(key, lineage)
        for key in (
            "commit_sha", "dirty", "working_tree_diff_sha256", "untracked_files"
        ):
            self.assertIn(key, lineage["git"])
        self.assertEqual(len(lineage["dataset_files"]["train"][0]["sha256"]), 64)
        self.assertEqual(
            set(lineage["package_versions"]),
            {"python", "numpy", "pandas", "scikit-learn", "torch"},
        )


if __name__ == "__main__":
    unittest.main()
