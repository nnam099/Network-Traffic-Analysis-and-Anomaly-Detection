import os
import json
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder, RobustScaler


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
DASHBOARD_DIR = os.path.join(ROOT_DIR, "dashboard")
if DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, DASHBOARD_DIR)


class CoreSmokeTests(unittest.TestCase):
    def test_artifact_validator_accepts_matching_metadata(self):
        from ids.artifact_validator import validate_artifact_contract

        checkpoint = {
            "model_state_dict": {},
            "n_features": 3,
            "n_classes": 2,
            "version": "v14.0",
            "thresholds": {"hybrid": 0.5},
        }
        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        pipeline = {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": ["dur", "sbytes", "dbytes"],
            "version": "v14.0",
        }

        result = validate_artifact_contract(checkpoint, pipeline)

        self.assertTrue(result.ok)
        self.assertEqual(result.errors, [])

    def test_artifact_validator_rejects_feature_mismatch(self):
        from ids.artifact_validator import validate_artifact_contract

        checkpoint = {"model_state_dict": {}, "n_features": 4, "n_classes": 2}
        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        pipeline = {
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_names": ["dur", "sbytes", "dbytes"],
        }

        result = validate_artifact_contract(checkpoint, pipeline)

        self.assertFalse(result.ok)
        self.assertTrue(any("feature count mismatch" in err for err in result.errors))

    def test_artifact_validator_rejects_duplicate_feature_names(self):
        from ids.artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {"model_state_dict": {}, "n_features": 3, "n_classes": 2},
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "dur", "dbytes"],
            },
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("duplicates" in err for err in result.errors))

    def test_artifact_validator_rejects_invalid_thresholds(self):
        from ids.artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {"model_state_dict": {}, "n_features": 3, "n_classes": 2, "thresholds": {"hybrid": -0.1}},
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
            },
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("non-negative" in err for err in result.errors))

    def test_artifact_validator_allows_negative_energy_threshold(self):
        from ids.artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {
                "model_state_dict": {},
                "n_features": 3,
                "n_classes": 2,
                "thresholds": {"hybrid": 0.5, "energy": -1.2},
            },
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
            },
        )

        self.assertTrue(result.ok)

    def test_artifact_validator_accepts_vote_threshold_controls(self):
        from ids.artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {"model_state_dict": {}, "n_features": 3, "n_classes": 2},
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
                "thresholds": {
                    "decision_mode": "vote",
                    "min_votes": 2,
                    "hybrid": 0.5,
                    "ae_re": 0.8,
                    "softmax": 0.4,
                },
            },
        )

        self.assertTrue(result.ok)

    def test_artifact_validator_rejects_invalid_vote_threshold_controls(self):
        from ids.artifact_validator import validate_artifact_contract

        scaler = RobustScaler().fit([[0, 1, 2], [3, 4, 5]])
        label_encoder = LabelEncoder().fit(["Normal", "DoS"])
        result = validate_artifact_contract(
            {
                "model_state_dict": {},
                "n_features": 3,
                "n_classes": 2,
                "thresholds": {
                    "decision_mode": "vote",
                    "min_votes": -1,
                    "hybrid": 0.5,
                    "ae_re": 0.8,
                },
            },
            {
                "scaler": scaler,
                "label_encoder": label_encoder,
                "feature_names": ["dur", "sbytes", "dbytes"],
            },
        )

        self.assertFalse(result.ok)
        self.assertTrue(any("thresholds.min_votes must be positive" in err for err in result.errors))

    def test_patch_checkpoint_infers_dims_from_state_dict(self):
        from scripts.checkpoint_metadata import infer_dims

        checkpoint = {
            "model_state_dict": {
                "ae.enc.4.weight": torch.zeros(64, 128),
                "backbone.input_proj.0.weight": torch.zeros(256, 61),
            }
        }

        self.assertEqual(infer_dims(checkpoint), (128, 256))

    def test_train_class_weight_overrides_target_weak_classes(self):
        from ids.dataset import make_loaders
        from ids.losses import IDSLoss
        from ids.training import parse_class_weight_overrides

        labels = ["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"]
        overrides = parse_class_weight_overrides("Exploits=3.0,Reconnaissance=4.0", labels)

        self.assertEqual(overrides, {2: 3.0, 3: 4.0})

        criterion = IDSLoss(
            n_classes=5,
            dos_class_idx=1,
            dos_weight=1.5,
            class_weight_overrides=overrides,
        )
        weights = criterion.focal.w.detach().cpu().numpy()
        self.assertGreater(weights[2], weights[1])
        self.assertGreater(weights[3], weights[2])

        splits = {
            "X_train": [[0.0], [1.0], [2.0], [3.0]],
            "y_train": [0, 1, 2, 3],
            "X_val": [[0.0]],
            "y_val": [0],
            "X_test": [[0.0]],
            "y_test": [0],
        }
        loaders = make_loaders(
            splits,
            batch_size=2,
            num_workers=0,
            dos_class_idx=1,
            dos_over=1.5,
            class_sample_weights=overrides,
            seed=7,
        )
        sampler_weights = loaders["train"].sampler.weights.detach().cpu().numpy()
        self.assertGreater(sampler_weights[2], sampler_weights[1])
        self.assertGreater(sampler_weights[3], sampler_weights[2])

    def test_normal_idx_matches_label_encoder(self):
        from torch.utils.data import DataLoader, TensorDataset

        from ids.trainer import eval_epoch, normal_class_index

        encoder = LabelEncoder().fit(["Normal", "DoS", "Exploits", "Reconnaissance", "Generic"])
        normal_idx = normal_class_index(encoder.classes_)

        self.assertEqual(encoder.inverse_transform([normal_idx])[0], "Normal")
        self.assertNotEqual(normal_idx, 0)

        labels = torch.tensor([normal_idx, normal_idx, 0, 1, 2, 4], dtype=torch.long)
        logits = torch.full((len(labels), len(encoder.classes_)), -4.0)
        logits[torch.arange(len(labels)), labels] = 4.0
        loader = DataLoader(TensorDataset(logits, labels), batch_size=3, shuffle=False)

        class FixedLogitModel(torch.nn.Module):
            def forward(self, values):
                return values, values

        metrics = eval_epoch(FixedLogitModel(), loader, "cpu", normal_idx=normal_idx)
        self.assertAlmostEqual(metrics["auc"], 1.0)

    def test_runtime_rejects_v15_until_scoring_parity(self):
        from ids.batch_evaluator import load_ids_artifacts

        with self.assertRaisesRegex(
            ValueError,
            "v15 disabled: runtime does not implement KNN/ood_ensemble scoring used at evaluation time",
        ):
            load_ids_artifacts("missing.pth", "missing.pkl", model_version="v15")

    def test_environment_check_does_not_expose_secret_values(self):
        from scripts.check_environment import assess_readiness, collect_environment, python_version_status

        old_provider = os.environ.get("LLM_PROVIDER")
        old_key = os.environ.get("GROQ_API_KEY")
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = "secret-value"
        try:
            env = collect_environment()
        finally:
            if old_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = old_provider
            if old_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_key

        self.assertTrue(env["llm"]["key_present"])
        self.assertNotIn("secret-value", json.dumps(env))

        readiness = assess_readiness(
            {
                "packages": {"torch": False, "numpy": True, "pandas": True, "sklearn": True, "matplotlib": True, "dotenv": True},
                "artifacts": {},
                "data": {"data_dir": True},
                "llm": {},
            }
        )
        self.assertEqual(readiness["status"], "BLOCKED")
        self.assertTrue(any("torch" in item for item in readiness["blockers"]))

        self.assertTrue(python_version_status((3, 11, 9))["supported"])
        self.assertFalse(python_version_status((3, 13, 0))["supported"])

    def test_resolve_paths_keeps_existing_kaggle_paths(self):
        from ids.config import resolve_paths

        cfg = SimpleNamespace(
            data_dir="/kaggle/input",
            save_dir="/kaggle/working/checkpoints_v14",
            plot_dir="/kaggle/working/plots_v14",
        )

        def exists(path):
            return path in {"/kaggle/input", "/kaggle/working"}

        with patch("ids.config.os.path.exists", side_effect=exists):
            resolved = resolve_paths(cfg)

        self.assertEqual(resolved.data_dir, "/kaggle/input")
        self.assertEqual(resolved.save_dir, "/kaggle/working/checkpoints_v14")
        self.assertEqual(resolved.plot_dir, "/kaggle/working/plots_v14")

    def test_resolve_paths_falls_back_to_local_outside_kaggle(self):
        from ids.config import resolve_paths

        cfg = SimpleNamespace(
            data_dir="/missing/kaggle/input",
            save_dir="/kaggle/working/checkpoints_v14",
            plot_dir="/kaggle/working/plots_v14",
        )

        def exists(path):
            return os.path.basename(path) == "data"

        with patch("ids.config.os.path.exists", side_effect=exists):
            resolved = resolve_paths(cfg)

        self.assertTrue(resolved.data_dir.endswith(os.path.join("ZeroDay-Detection-AutoEncoder-IDS", "data")))
        self.assertTrue(resolved.save_dir.endswith(os.path.join("ZeroDay-Detection-AutoEncoder-IDS", "checkpoints")))
        self.assertTrue(resolved.plot_dir.endswith(os.path.join("ZeroDay-Detection-AutoEncoder-IDS", "plots")))

    def test_environment_check_configures_utf8_console_output(self):
        from scripts.check_environment import configure_console_encoding

        class FakeStream:
            def __init__(self):
                self.calls = []

            def reconfigure(self, **kwargs):
                self.calls.append(kwargs)

        old_stdout = sys.stdout
        old_stderr = sys.stderr
        fake_stdout = FakeStream()
        fake_stderr = FakeStream()
        try:
            sys.stdout = fake_stdout
            sys.stderr = fake_stderr
            configure_console_encoding()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        self.assertEqual(fake_stdout.calls, [{"encoding": "utf-8", "errors": "backslashreplace"}])
        self.assertEqual(fake_stderr.calls, [{"encoding": "utf-8", "errors": "backslashreplace"}])

    def test_artifact_manifest_detects_hash_changes(self):
        from scripts.artifact_manifest import build_manifest, verify_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.abspath(tmp)
            os.makedirs(os.path.join(root, "checkpoints"))
            artifact = os.path.join(root, "checkpoints", "ids_v14_model.pth")
            with open(artifact, "wb") as f:
                f.write(b"model-v1")

            manifest = build_manifest(Path(root), ["checkpoints/ids_v14_model.pth"])
            self.assertTrue(verify_manifest(Path(root), manifest)["ok"])

            with open(artifact, "wb") as f:
                f.write(b"model-v2")

            result = verify_manifest(Path(root), manifest)
            self.assertFalse(result["ok"])
            self.assertTrue(any("sha256" in err for err in result["errors"]))

    def test_alert_store_persists_alert_history_and_status(self):
        from ids.alert_store import list_alerts, save_alert, update_alert_status

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alerts.sqlite3")
            alert = {
                "alert_id": "A-001",
                "timestamp": "2026-05-14 10:00:00",
                "hybrid_score": 0.73,
                "ae_score": 0.6,
                "max_prob": 0.4,
                "predicted_class": "Zero-Day Candidate",
                "classifier_class": "Normal",
                "is_zeroday": True,
            }
            llm = {"severity": "HIGH", "analyst_note": "review source IP"}

            save_alert(db_path, alert, llm=llm, source="single")
            rows = list_alerts(db_path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["alert_id"], "A-001")
            self.assertEqual(rows[0]["llm_severity"], "HIGH")
            self.assertEqual(rows[0]["source"], "single")
            self.assertTrue(rows[0]["is_zeroday"])
            self.assertEqual(rows[0]["llm_analysis"]["analyst_note"], "review source IP")

            update_alert_status(db_path, "A-001", "false_positive", "benign scan")
            rows = list_alerts(db_path)

            self.assertEqual(rows[0]["status"], "false_positive")
            self.assertEqual(rows[0]["analyst_note"], "benign scan")

    def test_alert_store_rejects_missing_alert_status_update(self):
        from ids.alert_store import list_alerts, update_alert_status

        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "alerts.sqlite3")

            with self.assertRaisesRegex(ValueError, "alert not found: A-404"):
                update_alert_status(db_path, "A-404", "closed", "stale selection")

            self.assertEqual(list_alerts(db_path), [])

    def test_log_normalizer_maps_common_firewall_csv(self):
        from ids.log_normalizer import normalize_real_world_logs

        df = pd.DataFrame(
            {
                "src_ip": ["10.0.0.5", "10.0.0.5"],
                "dst_ip": ["172.16.1.10", "172.16.1.11"],
                "src_port": [52512, 52513],
                "dst_port": [443, 443],
                "protocol": ["tcp", "tcp"],
                "duration": [1.5, 0.2],
                "sent_bytes": [1200, 80],
                "received_bytes": [3200, 40],
                "sent_pkts": [10, 2],
                "received_pkts": [14, 1],
                "action": ["CON", "RST"],
            }
        )

        normalized, report = normalize_real_world_logs(df)

        self.assertEqual(len(normalized), 2)
        for col in ["proto", "state", "dur", "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload"]:
            self.assertIn(col, normalized.columns)
        self.assertGreaterEqual(report.feature_coverage, 0.7)
        self.assertEqual(report.schema, "firewall_or_flow_csv")

    def test_production_schema_prepares_cicflowmeter_rows(self):
        from ids.production_schema import (
            PRODUCTION_FLOW_COLUMNS,
            apply_label_overrides,
            normalize_to_production_schema,
            split_by_event_time,
            summarize_production_flows,
        )

        sample_path = Path(ROOT_DIR) / "data" / "samples" / "cicflowmeter_sample.csv"
        raw = pd.read_csv(sample_path)

        result = normalize_to_production_schema(raw, source="cicflowmeter", source_file=sample_path.name)

        self.assertEqual(list(result.data.columns), PRODUCTION_FLOW_COLUMNS)
        self.assertEqual(result.report["normalization_report"]["schema"], "cic_ids2017")
        self.assertEqual(result.data.loc[0, "analyst_label"], "normal")
        self.assertEqual(result.data.loc[1, "analyst_label"], "known_attack")
        self.assertTrue(str(result.data.loc[0, "event_time"]).startswith("2026-05-19T08:00:00"))
        self.assertGreater(float(result.data.loc[0, "duration"]), 0)

        overrides = pd.DataFrame({
            "flow_id": [result.data.loc[1, "flow_id"]],
            "analyst_label": ["suspicious"],
            "attack_category": ["Needs review"],
        })
        reviewed = apply_label_overrides(result.data, overrides)
        self.assertEqual(reviewed.loc[1, "analyst_label"], "suspicious")
        self.assertEqual(reviewed.loc[1, "attack_category"], "Needs review")

        split = split_by_event_time(reviewed, train_ratio=0.34, validation_ratio=0.33, test_ratio=0.33)
        self.assertEqual(set(split["split"]), {"train", "validation", "test"})
        summary = summarize_production_flows(split)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["source_distribution"]["cicflowmeter"], 3)

    def test_prepare_production_flow_data_cli_writes_splits(self):
        import scripts.prepare_production_flow_data as prep

        sample_path = Path(ROOT_DIR) / "data" / "samples" / "cicflowmeter_sample.csv"
        with tempfile.TemporaryDirectory() as tmp:
            old_argv = sys.argv
            sys.argv = [
                "prepare_production_flow_data.py",
                str(sample_path),
                "--output-dir",
                tmp,
                "--train-ratio",
                "0.34",
                "--validation-ratio",
                "0.33",
                "--test-ratio",
                "0.33",
            ]
            try:
                exit_code = prep.main()
            finally:
                sys.argv = old_argv

            self.assertEqual(exit_code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "production_flows.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "train.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "validation.csv")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "test.csv")))
            with open(os.path.join(tmp, "manifest.json"), encoding="utf-8") as handle:
                manifest = json.load(handle)
            self.assertEqual(manifest["summary"]["rows"], 3)
            self.assertEqual(manifest["schema_version"], 1)

    def test_input_guard_rejects_bad_csv_shape(self):
        from ids.input_guard import CSVInputPolicy, validate_uploaded_csv

        empty_df = pd.DataFrame()
        result = validate_uploaded_csv(empty_df, size_bytes=10, policy=CSVInputPolicy(max_rows=10, min_columns=3))

        self.assertFalse(result.ok)
        self.assertTrue(any("no rows" in err for err in result.errors))
        self.assertTrue(any("too few columns" in err for err in result.errors))

        large_df = pd.DataFrame({"a": range(12), "b": range(12), "c": range(12)})
        result = validate_uploaded_csv(large_df, size_bytes=10, policy=CSVInputPolicy(max_rows=10))

        self.assertFalse(result.ok)
        self.assertTrue(any("too many rows" in err for err in result.errors))

        dup_df = pd.DataFrame([[1, 2, 3]], columns=["src_ip", "src_ip", "dst_ip"])
        result = validate_uploaded_csv(dup_df)

        self.assertFalse(result.ok)
        self.assertTrue(any("duplicate column" in err for err in result.errors))

    def test_mitre_mapper_known_attack(self):
        from ids.mitre_mapper import MITREMapper

        result = MITREMapper().map_known_attack(1, class_names=["Normal", "DoS"])

        self.assertEqual(result["attack_class"], "DoS")
        self.assertEqual(result["mapping_mode"], "known_attack")
        self.assertTrue(result["techniques"])

    def test_inference_runtime_verdict_and_risk_helpers(self):
        from ids.inference_runtime import (
            assess_normalization_quality,
            ground_truth_verdict,
            hybrid_score_from_meta,
            risk_score,
            traffic_verdict,
            zero_day_decision,
        )

        decision, rule = zero_day_decision(0.8, 0.4, 0.7)
        self.assertTrue(bool(decision))
        self.assertEqual(rule, "ae_plus_confidence_fallback")

        decision, rule = zero_day_decision(0.1, 0.9, 0.8, thresholds={"hybrid": 0.5}, hybrid_threshold=0.6)
        self.assertTrue(bool(decision))
        self.assertEqual(rule, "hybrid_calibrated")

        decision, rule = zero_day_decision(0.1, 0.9, 0.4, thresholds={"hybrid": 0.3})
        self.assertTrue(bool(decision))
        self.assertEqual(rule, "hybrid_calibrated")

        with self.assertRaisesRegex(ValueError, "min_votes must be positive"):
            zero_day_decision(0.8, 0.4, 0.7, thresholds={"decision_mode": "vote", "min_votes": 0, "hybrid": 0.5})

        self.assertEqual(traffic_verdict(True, "Normal"), "Zero-Day Candidate")
        self.assertEqual(traffic_verdict(False, "Normal"), "Normal")
        self.assertEqual(traffic_verdict(False, "DoS"), "Known-Attack")
        self.assertEqual(ground_truth_verdict("benign"), "Normal")
        self.assertEqual(ground_truth_verdict("Exploit"), "Known-Attack")
        self.assertGreater(risk_score({"hybrid_score": 0.9, "ae_score": 0.8, "is_zeroday": True}, "HIGH"), 80)
        learned = hybrid_score_from_meta(
            [0.1, 1.0],
            [0.95, 0.50],
            thresholds={"hybrid_meta": {"coef": [3.0, 2.0], "intercept": -2.0}},
        )
        self.assertLess(float(learned[0]), float(learned[1]))

        quality = assess_normalization_quality(
            {
                "feature_coverage": 0.52,
                "mapped_columns": {"src_ip": "src_ip", "dst_ip": "dst_ip"},
                "missing_core_features": ["sbytes", "dbytes", "dur"],
            }
        )
        self.assertEqual(quality["level"], "LOW")
        self.assertTrue(any("directional" in warning for warning in quality["warnings"]))

    def test_dashboard_runtime_preprocess_aligns_features(self):
        from ids.dashboard_runtime import preprocess_dashboard_df

        df = pd.DataFrame({
            "proto": ["tcp"],
            "service": ["http"],
            "state": ["CON"],
            "dur": [2.0],
            "sbytes": [100],
            "dbytes": [50],
        })
        result = preprocess_dashboard_df(
            df,
            ["dur", "proto_num", "service_num", "state_num", "bytes_ratio", "missing_feature"],
            "v14",
            pipeline_meta={
                "categorical_maps": {
                    "proto": {"tcp": 7, "unk": -1},
                    "service": {"http": 3, "unk": -1},
                    "state": {"CON": 2, "unk": -1},
                }
            },
        )

        self.assertEqual(result.features.shape, (1, 6))
        self.assertEqual(result.features[0, 0], 2.0)
        self.assertEqual(result.features[0, 1], 7.0)
        self.assertEqual(result.features[0, 2], 3.0)
        self.assertEqual(result.features[0, 3], 2.0)
        self.assertAlmostEqual(float(result.features[0, 4]), 100 / 150, places=5)
        self.assertEqual(result.features[0, 5], 0.0)
        self.assertIsNone(result.normalization_report)

    def test_dashboard_runtime_preprocess_reports_normalizer_failure(self):
        from ids.dashboard_runtime import preprocess_dashboard_df

        def broken_normalizer(_df):
            raise ValueError("bad schema")

        result = preprocess_dashboard_df(
            pd.DataFrame({"dur": [1.5]}),
            ["dur", "missing_feature"],
            "v14",
            normalizer=broken_normalizer,
        )

        self.assertEqual(result.features.tolist(), [[1.5, 0.0]])
        self.assertEqual(result.normalization_report["schema"], "normalization_failed")
        self.assertIn("bad schema", result.normalization_report["error"])

    def test_dashboard_runtime_builds_alert_context_contract(self):
        from ids.dashboard_runtime import build_alert_context_from_log

        context = build_alert_context_from_log(
            {
                "source_row": 42,
                "classifier_class": "Normal",
                "hybrid_score": 0.91,
                "ae_score": 0.8,
                "max_prob": 0.4,
                "is_zeroday": True,
                "zero_day_family": "Shellcode",
            },
            timestamp="2026-05-14 12:00:00",
        )

        self.assertEqual(context["alert_id"], "ZD-000042")
        self.assertEqual(context["timestamp"], "2026-05-14 12:00:00")
        self.assertEqual(context["predicted_class"], "Zero-Day Candidate")
        self.assertEqual(context["classifier_class"], "Normal")
        self.assertTrue(context["is_zeroday"])
        self.assertEqual(context["zero_day_family"], "Shellcode")
        self.assertIn("hybrid_score", context["raw_scores"])

    def test_dashboard_runtime_builds_ai_context_options(self):
        from ids.dashboard_runtime import build_ai_context_options, default_ai_context_index

        history = [
            {"alert_id": "A1", "predicted_class": "Known-Attack", "hybrid_score": 0.4},
            {"alert_id": "A2", "predicted_class": "Zero-Day Candidate", "hybrid_score": 0.8},
        ]
        bulk = pd.DataFrame({
            "source_row": [10, 11],
            "detection": ["Normal", "Zero-Day Candidate"],
            "classifier_class": ["Normal", "DoS"],
            "hybrid_score": [0.1, 0.9],
            "ae_score": [0.1, 0.8],
            "max_prob": [0.95, 0.4],
            "is_zeroday": [False, True],
        })

        options = build_ai_context_options(history, bulk, max_history=1, max_bulk_logs=1)

        self.assertEqual(len(options), 2)
        self.assertIn("A2", options[0].label)
        self.assertIn("row=11", options[1].label)
        self.assertEqual(options[1].context["alert_id"], "ZD-000011")
        self.assertEqual(default_ai_context_index(options, "ZD-000011"), 1)
        self.assertEqual(default_ai_context_index(options, "missing"), 0)

    def test_dashboard_runtime_llm_fallbacks_are_testable(self):
        from ids.dashboard_runtime import answer_analyst_question, triage_alert_with_fallback

        alert = {
            "hybrid_score": 0.7,
            "ae_score": 0.6,
            "predicted_class": "Zero-Day Candidate",
            "is_zeroday": True,
        }
        fallback = triage_alert_with_fallback(alert)
        self.assertEqual(fallback["severity"], "HIGH")
        self.assertIn("hybrid score: 0.700", fallback["verdict"])

        class BrokenAgent:
            def triage_alert(self, _result):
                raise RuntimeError("provider failed")

        error_result = triage_alert_with_fallback(alert, BrokenAgent())
        self.assertEqual(error_result["false_positive_risk"], "UNKNOWN")
        self.assertIn("provider failed", error_result["attack_summary"])

        missing_dep = answer_analyst_question(
            "why?",
            alert,
            has_llm=False,
            llm_provider="groq",
            llm_dependency="groq",
            has_llm_dependency=False,
        )
        self.assertIn("Chua cai thu vien", missing_dep)

        class ExplainAgent:
            def explain_to_analyst(self, question, context):
                return f"{question}:{context['predicted_class']}"

        answer = answer_analyst_question("why", alert, True, agent_factory=ExplainAgent)
        self.assertEqual(answer, "why:Zero-Day Candidate")

    def test_dashboard_runtime_filters_alert_history(self):
        from ids.dashboard_runtime import filter_alert_history

        alerts = [
            {
                "alert_id": "A1",
                "status": "new",
                "llm_severity": "HIGH",
                "is_zeroday": True,
                "predicted_class": "Zero-Day Candidate",
                "analyst_note": "source ip suspicious",
            },
            {
                "alert_id": "A2",
                "status": "closed",
                "llm_severity": "LOW",
                "is_zeroday": False,
                "predicted_class": "Normal",
                "analyst_note": "benign",
            },
        ]

        self.assertEqual([a["alert_id"] for a in filter_alert_history(alerts, status="new")], ["A1"])
        self.assertEqual([a["alert_id"] for a in filter_alert_history(alerts, severity="LOW")], ["A2"])
        self.assertEqual([a["alert_id"] for a in filter_alert_history(alerts, ood_filter="OOD only")], ["A1"])
        self.assertEqual([a["alert_id"] for a in filter_alert_history(alerts, query="benign")], ["A2"])

    def test_dashboard_runtime_builds_top_batch_alerts(self):
        from ids.dashboard_runtime import build_top_batch_alerts

        scores = pd.DataFrame({
            "source_row": [0, 1, 2],
            "detection": ["Normal", "Known-Attack", "Zero-Day Candidate"],
            "classifier_class": ["Normal", "DoS", "Normal"],
            "hybrid_score": [0.2, 0.7, 0.6],
            "ae_score": [0.1, 0.5, 0.9],
            "max_prob": [0.95, 0.5, 0.3],
            "is_zeroday": [False, False, True],
        })

        alerts = build_top_batch_alerts(scores, file_hash="abcdef123456", limit=2, timestamp="2026-05-14 10:00:00")

        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["alert_id"], "BATCH-abcdef12-000002")
        self.assertEqual(alerts[0]["predicted_class"], "Zero-Day Candidate")
        self.assertEqual(alerts[1]["alert_id"], "BATCH-abcdef12-000001")
        self.assertEqual(alerts[0]["source_file_hash"], "abcdef123456")

    def test_dashboard_runtime_enriches_batch_alert_entities(self):
        from ids.dashboard_runtime import build_top_batch_alerts

        scores = pd.DataFrame({
            "source_row": [0],
            "detection": ["Zero-Day Candidate"],
            "classifier_class": ["DoS"],
            "hybrid_score": [0.9],
            "ae_score": [0.8],
            "max_prob": [0.2],
            "is_zeroday": [True],
        })
        raw = pd.DataFrame({
            "src_ip": ["10.0.0.5"],
            "dst_ip": ["172.16.1.10"],
            "src_port": [52512],
            "dst_port": [443],
            "protocol": ["tcp"],
            "service": ["https"],
        })

        alert = build_top_batch_alerts(scores, file_hash="abcdef123456", raw_df=raw)[0]

        self.assertEqual(alert["src_ip"], "10.0.0.5")
        self.assertEqual(alert["dst_ip"], "172.16.1.10")
        self.assertEqual(alert["dst_port"], "443")
        self.assertEqual(alert["service"], "https")

    def test_dashboard_runtime_correlates_alerts(self):
        from ids.dashboard_runtime import correlate_alerts

        alerts = [
            {"alert_id": "A1", "src_ip": "10.0.0.5", "classifier_class": "DoS", "is_zeroday": True, "risk": 90, "timestamp": "2026-05-14 10:00:00"},
            {"alert_id": "A2", "src_ip": "10.0.0.5", "classifier_class": "DoS", "is_zeroday": False, "risk": 60, "timestamp": "2026-05-14 10:01:00"},
            {"alert_id": "A3", "src_ip": "10.0.0.8", "classifier_class": "Normal", "is_zeroday": False, "risk": 10, "timestamp": "2026-05-14 10:02:00"},
        ]

        groups = correlate_alerts(alerts, min_count=2)

        self.assertTrue(any(g["group_type"] == "Source IP" and g["key"] == "10.0.0.5" for g in groups))
        dos_group = next(g for g in groups if g["group_type"] == "Classifier Class" and g["key"] == "DoS")
        self.assertEqual(dos_group["alert_count"], 2)
        self.assertEqual(dos_group["ood_count"], 1)
        self.assertEqual(dos_group["max_risk"], 90)

    def test_dashboard_runtime_builds_time_window_incidents(self):
        from ids.dashboard_runtime import build_time_window_incidents

        alerts = [
            {
                "alert_id": "A1",
                "timestamp": "2026-05-14 10:00:00",
                "src_ip": "10.0.0.5",
                "service": "https",
                "classifier_class": "DoS",
                "zero_day_family": "Shellcode",
                "is_zeroday": True,
                "risk": 92,
                "llm_severity": "CRITICAL",
            },
            {
                "alert_id": "A2",
                "timestamp": "2026-05-14 10:08:00",
                "src_ip": "10.0.0.5",
                "service": "https",
                "classifier_class": "DoS",
                "zero_day_family": "Shellcode",
                "is_zeroday": False,
                "risk": 76,
                "llm_severity": "HIGH",
            },
            {
                "alert_id": "A3",
                "timestamp": "2026-05-14 10:40:00",
                "src_ip": "10.0.0.5",
                "service": "https",
                "classifier_class": "DoS",
                "is_zeroday": False,
                "risk": 55,
                "llm_severity": "MEDIUM",
            },
            {
                "alert_id": "A4",
                "timestamp": "2026-05-14 10:05:00",
                "src_ip": "10.0.0.8",
                "service": "dns",
                "classifier_class": "Normal",
                "is_zeroday": False,
                "risk": 10,
            },
        ]

        incidents = build_time_window_incidents(alerts, window_minutes=15, min_alerts=2)

        source_incident = next(item for item in incidents if item["group_type"] == "Source IP")
        self.assertEqual(source_incident["key"], "10.0.0.5")
        self.assertEqual(source_incident["alert_count"], 2)
        self.assertEqual(source_incident["ood_count"], 1)
        self.assertEqual(source_incident["high_count"], 2)
        self.assertEqual(source_incident["max_risk"], 92)
        self.assertEqual(source_incident["severity"], "CRITICAL")
        self.assertEqual(source_incident["primary_classes"], ["DoS"])
        self.assertEqual(source_incident["families"], ["Shellcode"])
        self.assertEqual(source_incident["alert_ids"], ["A1", "A2"])
        self.assertIn("Source IP: 10.0.0.5", source_incident["recommended_focus"])
        self.assertTrue(all(item["end_time"] <= "2026-05-14 10:15:00" for item in incidents))

    def test_dashboard_view_helpers_are_testable(self):
        from ui_safety import attach_report_safety_note
        from views_ood import build_feature_table, build_score_table, enrich_ood_row
        from views_queue import build_history_dataframe, queue_summary
        from views_report import build_export_report

        alerts = [
            {
                "alert_id": "A1",
                "timestamp": "2026-05-14 10:00:00",
                "status": "new",
                "source": "single",
                "llm_severity": "HIGH",
                "predicted_class": "Zero-Day Candidate",
                "classifier_class": "DoS",
                "hybrid_score": 0.9,
                "ae_score": 0.8,
                "is_zeroday": True,
            },
            {
                "alert_id": "A2",
                "timestamp": "2026-05-14 10:01:00",
                "status": "closed",
                "source": "single",
                "llm_severity": "LOW",
                "predicted_class": "Normal",
                "classifier_class": "Normal",
                "hybrid_score": 0.1,
                "ae_score": 0.1,
                "is_zeroday": False,
            },
        ]

        summary = queue_summary(alerts)
        self.assertEqual(summary["alerts"], 2)
        self.assertEqual(summary["critical_high"], 1)
        self.assertEqual(summary["ood"], 1)
        self.assertGreater(summary["average_risk"], 0)

        df = build_history_dataframe(alerts)
        self.assertEqual(df.iloc[0]["Alert ID"], "A1")
        self.assertEqual(df.iloc[0]["OOD Candidate"], "YES")

        report = attach_report_safety_note({"alert_id": "A1"})
        self.assertIn("limitations_and_safety", report)
        self.assertIn("zero_day_candidate_meaning", report["limitations_and_safety"])

        export = build_export_report({"alert_id": "A1", "shap_values": [1], "probs": [0.5]}, {"severity": "HIGH"})
        self.assertNotIn("shap_values", export)
        self.assertNotIn("probs", export)
        self.assertEqual(export["llm_analysis"]["severity"], "HIGH")

        ood_row = enrich_ood_row({"source_row": 5, "hybrid_score": 0.9, "ae_score": 0.8, "is_zeroday": True})
        self.assertEqual(ood_row["detection"], "Zero-Day Candidate")
        self.assertGreater(ood_row["risk"], 80)
        feature_table = build_feature_table(pd.Series({"dur": 1.2, "zzz": "tail"}), search="dur")
        self.assertEqual(feature_table["Feature"].tolist(), ["dur"])
        score_table = build_score_table({"source_row": 5, "hybrid_score": 0.9})
        self.assertEqual(score_table["Metric"].tolist(), ["hybrid_score"])

    def test_batch_evaluator_reports_labeled_metrics(self):
        from ids.batch_evaluator import summarize_scores

        scores = pd.DataFrame({
            "predicted_class": ["Normal", "Known-Attack", "Zero-Day Candidate", "Known-Attack"],
            "classifier_class": ["Normal", "DoS", "Normal", "Exploits"],
            "is_zeroday": [False, False, True, False],
            "hybrid": [0.1, 0.7, 0.8, 0.6],
            "ae_re": [0.1, 0.5, 0.9, 0.4],
            "softmax": [0.05, 0.3, 0.7, 0.2],
            "max_prob": [0.95, 0.7, 0.3, 0.8],
        })
        raw = pd.DataFrame({"attack_cat": ["Normal", "DoS", "Shellcode", "DoS"]})

        summary = summarize_scores(
            scores,
            raw_df=raw,
            label_col="attack_cat",
            class_names=["Normal", "DoS", "Exploits"],
            zero_day_labels=["Shellcode"],
            thresholds={"hybrid": 0.5, "ae_re": 0.4, "hybrid_meta": {"type": "logistic_regression"}},
        )

        self.assertEqual(summary["accuracy"], 1.0)
        self.assertEqual(summary["normal_ood_fpr"], 0.0)
        self.assertEqual(summary["normal_alert_fpr"], 0.0)
        self.assertEqual(summary["known_classifier_fpr"], 0.0)
        self.assertEqual(summary["ood_detection_rate"], 1.0)
        self.assertEqual(summary["recall_per_class"]["Normal"]["recall"], 1.0)
        self.assertEqual(summary["recall_per_class"]["DoS"]["recall"], 0.5)
        self.assertEqual(summary["threshold_profile"]["hybrid"], 0.5)
        self.assertEqual(summary["threshold_profile"]["hybrid_meta"]["type"], "logistic_regression")

    def test_zero_day_vote_decision_requires_multiple_signals(self):
        from ids.inference_runtime import zero_day_decision

        decision, rule = zero_day_decision(
            ae_score=[0.9, 0.2],
            max_prob=[0.4, 0.9],
            hybrid_score=[0.8, 0.3],
            thresholds={
                "decision_mode": "vote",
                "min_votes": 2,
                "hybrid": 0.5,
                "ae_re": 0.5,
                "softmax": 0.5,
            },
        )

        self.assertEqual(rule, "vote_2_of_3")
        self.assertEqual(decision.tolist(), [True, False])

    def test_runtime_batch_inference_returns_dashboard_contract(self):
        from ids.inference_runtime import run_batch_inference

        class IdentityScaler:
            def transform(self, values):
                return values

        class TinyAE:
            def recon_error(self, x):
                return x[:, 0]

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ae = TinyAE()

            def forward(self, x):
                return torch.stack([x[:, 1], x[:, 2]], dim=1)

        raw = torch.tensor(
            [[0.8, 0.1, 2.0], [0.1, 2.0, 0.1], [0.4, 1.8, 0.2]],
            dtype=torch.float32,
        ).numpy()
        result = run_batch_inference(
            TinyModel(),
            IdentityScaler(),
            raw,
            ["Normal", "DoS"],
            thresholds={"decision_mode": "vote", "min_votes": 2, "hybrid": 0.2, "ae_re": 0.5},
            batch_size=2,
        )

        self.assertEqual(len(result), 3)
        self.assertEqual(result["classifier_class"].tolist(), ["DoS", "Normal", "Normal"])
        self.assertEqual(result["predicted_class"].tolist(), ["Zero-Day Candidate", "Normal", "Normal"])
        self.assertEqual(result["is_zeroday"].tolist(), [True, False, False])
        self.assertEqual(result["zero_day_rule"].iloc[0], "vote_2_of_2")
        np.testing.assert_allclose(result["ae_score"].to_numpy(), [0.8, 0.1, 0.4], rtol=1e-6)
        self.assertTrue(all(np.isscalar(value) for value in result["ae_score"]))

    def test_predict_with_uncertainty_uses_mc_dropout_contract(self):
        from ids.evaluator import predict_with_uncertainty
        from ids.models import IDSModel

        model = IDSModel(n_features=4, n_classes=3, hidden=16, ae_hidden=8)
        model.eval()
        x = torch.ones(1, 4)

        mean_probs, std_probs, entropy = predict_with_uncertainty(model, x, n_samples=5)

        self.assertFalse(model.training)
        self.assertEqual(tuple(mean_probs.shape), (3,))
        self.assertEqual(tuple(std_probs.shape), (3,))
        self.assertAlmostEqual(float(mean_probs.sum()), 1.0, places=5)
        self.assertGreaterEqual(entropy, 0.0)

    def test_batch_evaluator_calibrates_threshold_profile(self):
        from ids.batch_evaluator import calibrate_thresholds

        scores = pd.DataFrame({
            "hybrid": [0.1, 0.2, 0.3, 0.4],
            "ae_re": [0.2, 0.3, 0.4, 0.5],
            "softmax": [0.05, 0.1, 0.15, 0.2],
        })
        raw_df = pd.DataFrame({"label": ["Benign", "Benign", "Attack", "Attack"]})

        profile = calibrate_thresholds(scores, target_fpr=0.5, raw_df=raw_df)

        self.assertEqual(profile["thresholds"]["decision_mode"], "vote")
        self.assertEqual(profile["reference_rows"], 2)
        self.assertIn("hybrid", profile["thresholds"])

    def test_llm_agent_import_has_no_provider_side_effect(self):
        sys.modules.pop("ids.llm_agent", None)
        buf = StringIO()

        with redirect_stdout(buf):
            import ids.llm_agent as llm_agent

        self.assertEqual(buf.getvalue(), "")
        status = llm_agent.get_llm_status()
        self.assertFalse(status["initialized"])
        self.assertIsNone(status["error"])

    def test_llm_agent_falls_back_without_api_key(self):
        old_provider = os.environ.get("LLM_PROVIDER")
        old_key = os.environ.get("GROQ_API_KEY")
        os.environ["LLM_PROVIDER"] = "groq"
        os.environ["GROQ_API_KEY"] = ""
        sys.modules.pop("llm_agent", None)
        try:
            from ids.llm_agent import SOCTriageAgent

            result = SOCTriageAgent().triage_alert(
                {
                    "alert_id": "A1",
                    "timestamp": "2026-05-13 00:00:00",
                    "hybrid_score": 0.7,
                    "ae_score": 0.6,
                    "max_prob": 0.4,
                    "predicted_class": "Zero-Day Candidate",
                    "is_zeroday": True,
                }
            )
        finally:
            sys.modules.pop("llm_agent", None)
            if old_provider is None:
                os.environ.pop("LLM_PROVIDER", None)
            else:
                os.environ["LLM_PROVIDER"] = old_provider
            if old_key is None:
                os.environ.pop("GROQ_API_KEY", None)
            else:
                os.environ["GROQ_API_KEY"] = old_key

        self.assertEqual(result["alert_id"], "A1")
        self.assertIn("severity", result)
        self.assertIn("LLM", result["verdict"])

    def test_v14_artifacts_load_when_present(self):
        model_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v14_model.pth")
        pipeline_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v14_pipeline.pkl")
        if not (os.path.exists(model_path) and os.path.exists(pipeline_path)):
            self.skipTest("v14 checkpoint/pipeline artifacts are not present")

        from ids_v14_unswnb15 import IDSModel

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        with open(pipeline_path, "rb") as f:
            pipeline = pickle.load(f)

        feature_names = pipeline.get("feature_names", pipeline.get("feat_cols", []))
        self.assertEqual(checkpoint["n_features"], len(feature_names))
        self.assertEqual(checkpoint["n_classes"], len(pipeline["label_encoder"].classes_))

        model = IDSModel(
            n_features=checkpoint["n_features"],
            n_classes=checkpoint["n_classes"],
            hidden=checkpoint.get("hidden", 256),
            ae_hidden=checkpoint.get("ae_hidden", 128),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        x = torch.zeros(2, checkpoint["n_features"])
        with torch.no_grad():
            logits, features = model(x)
            ae_score = model.ae.recon_error(x)

        self.assertEqual(tuple(logits.shape), (2, checkpoint["n_classes"]))
        self.assertEqual(features.shape[0], 2)
        self.assertEqual(tuple(ae_score.shape), (2,))

    def test_v15_artifacts_load_when_present(self):
        model_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v15_model.pth")
        pipeline_path = os.path.join(ROOT_DIR, "checkpoints", "ids_v15_pipeline.pkl")
        if not (os.path.exists(model_path) and os.path.exists(pipeline_path)):
            self.skipTest("v15 checkpoint/pipeline artifacts are not present")

        from ids.batch_evaluator import load_ids_artifacts

        artifacts = load_ids_artifacts(model_path, pipeline_path, "v15")

        self.assertEqual(artifacts.checkpoint["n_features"], len(artifacts.feature_names))
        self.assertEqual(artifacts.checkpoint["n_classes"], len(artifacts.class_names))
        self.assertTrue(artifacts.thresholds)

    def test_ids_model_forward_shapes(self):
        from ids.models import IDSModel

        model = IDSModel(n_features=6, n_classes=3, hidden=16, ae_hidden=8)
        x = torch.randn(4, 6)
        logits, features = model(x)
        embeddings = model.get_embed(x)
        recon = model.ae(x)

        self.assertEqual(tuple(logits.shape), (4, 3))
        self.assertEqual(tuple(features.shape), (4, 16))
        self.assertEqual(tuple(embeddings.shape), (4, 64))
        self.assertEqual(tuple(recon.shape), (4, 6))

    def test_focal_loss_output_range(self):
        from ids.losses import FocalLoss

        criterion = FocalLoss(n_classes=3, gamma=2.0)
        logits = torch.tensor([[2.0, 0.2, -1.0], [0.1, 1.5, -0.2]], dtype=torch.float32)
        labels = torch.tensor([0, 1], dtype=torch.long)
        loss = criterion(logits, labels)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(float(loss.item()), 0.0)

    def test_dataset_split_sizes(self):
        import numpy as np
        from ids.dataset import prepare_splits

        rows = []
        for attack_cat, label, offset, n in [
            ("Normal", 0, 0.0, 20),
            ("DoS", 1, 1.0, 20),
            ("Fuzzers", 1, 2.0, 10),
        ]:
            for i in range(n):
                rows.append({
                    "attack_cat": attack_cat,
                    "label": label,
                    "dur": 1.0 + i * 0.01 + offset,
                    "sbytes": 100 + i + offset,
                    "dbytes": 80 + i + offset,
                    "spkts": 10 + i % 3,
                    "dpkts": 8 + i % 4,
                })
        df = pd.DataFrame(rows)

        splits = prepare_splits(
            df,
            known_cats=["Normal", "DoS"],
            zd_cats=["Fuzzers"],
            test_ratio=0.20,
            val_ratio=0.10,
            seed=7,
        )

        self.assertEqual(len(splits["X_train"]), 28)
        self.assertEqual(len(splits["X_val"]), 4)
        self.assertEqual(len(splits["X_test"]), 8)
        self.assertEqual(len(splits["X_zd"]), 10)
        self.assertEqual(splits["n_classes"], 2)
        self.assertTrue(np.isfinite(splits["X_train"]).all())

    def test_recon_dos_separation(self):
        from ids_v14_unswnb15 import SupConLoss, class_prototype_cosine_similarity

        class TinyProbe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.eye(2))

            def get_embed(self, x):
                return torch.nn.functional.normalize(self.linear(x), dim=-1)

        x = torch.tensor([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]], dtype=torch.float32)
        y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        model = TinyProbe()
        criterion = SupConLoss(
            T=0.20,
            recon_class_idx=0,
            dos_class_idx=1,
            hard_negative_weight=1.0,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

        for _ in range(8):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model.get_embed(x), y)
            loss.backward()
            optimizer.step()

        sim = class_prototype_cosine_similarity(model, x.numpy(), y.numpy(), class_a=0, class_b=1)

        self.assertLess(sim, 0.5)

    def test_v14_adaptive_threshold_uses_recent_window(self):
        import numpy as np
        from ids_v14_unswnb15 import AdaptiveThreshold

        tracker = AdaptiveThreshold(window_size=3, target_fpr=1 / 3)
        first = tracker.update(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        self.assertAlmostEqual(first, float(np.quantile([1.0, 2.0, 3.0], 2 / 3)))
        self.assertTrue(tracker(3.5))
        self.assertFalse(tracker(1.5))

        second = tracker.update(np.asarray([9.0], dtype=np.float32))
        self.assertAlmostEqual(second, float(np.quantile([2.0, 3.0, 9.0], 2 / 3)))
        self.assertEqual(list(tracker.buffer), [2.0, 3.0, 9.0])

    def test_v14_hybrid_meta_learner_separates_validation_zero_day(self):
        from ids_v14_unswnb15 import compute_hybrid_meta_score, fit_hybrid_meta_learner

        class TinyAE:
            def recon_error(self, x):
                return x[:, 0]

        class TinyHybridModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.ae = TinyAE()

            def forward(self, x):
                return torch.stack([x[:, 1], x[:, 2]], dim=1), x

        known = torch.tensor(
            [[0.05, 4.0, 0.0], [0.10, 3.5, 0.0], [0.08, 4.5, 0.0]],
            dtype=torch.float32,
        ).numpy()
        zero_day = torch.tensor(
            [[1.00, 0.0, 0.0], [1.20, 0.2, 0.1], [0.90, 0.1, 0.2]],
            dtype=torch.float32,
        ).numpy()

        meta = fit_hybrid_meta_learner(TinyHybridModel(), known, zero_day, "cpu", seed=42)
        known_scores = compute_hybrid_meta_score([0.05, 0.10], [0.02, 0.03], meta)
        zd_scores = compute_hybrid_meta_score([1.00, 1.20], [0.50, 0.48], meta)

        self.assertIn("coef", meta)
        self.assertGreater(float(zd_scores.mean()), float(known_scores.mean()))


if __name__ == "__main__":
    unittest.main()
