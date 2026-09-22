#!/usr/bin/env python3
"""P7 Stage B one-way external evaluation. NEVER run during preregistration."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import (  # noqa: E402
    DEFAULT_TEST_FILE, DEFAULT_TRAIN_FILE, KNOWN_ATTACK_CATS,
    SOURCE_ROLE_COLUMN, load_unsw_csvs,
)
from ids.p2_protocol import continuous_matrix  # noqa: E402
from ids.p3_protocol import (  # noqa: E402
    classify_official_test, prepare_official_test, train_only_context,
)
from ids.p4_model_interface import (  # noqa: E402
    CATEGORICAL_FIELDS, OOV_INDEX, StructuredBatchAdapter, vocabularies_from_manifest,
)
from ids.p5_training import P3_HASH, TrainingOnlyModel, file_hash, json_bytes, stable_hash, write_immutable  # noqa: E402
from ids.p6_selection import (  # noqa: E402
    identity_means, scorer_scores, verify_scorer_states,
)
from ids.p7_external_protocol import (  # noqa: E402
    SMALL_N, SURFACES, TARGETS, SPLIT_SEEDS, across_seed_summary,
    bootstrap_intervals, bootstrap_seed, p7_path, primary_metrics,
    score_summary, validate_preregistration, verify_frozen_cell,
    upstream_snapshot,
)
from ids.target_isolation import CANONICAL_FINGERPRINT_COLUMN as CANONICAL  # noqa: E402


def _write(relative: str, payload: dict) -> None:
    write_immutable(p7_path(ROOT, relative), json_bytes(payload))


def _source_frames() -> tuple[dict, dict, pd.DataFrame, str]:
    # This function must only be called after the one-way gate is atomically opened.
    p3 = ROOT / "results/data_quality/p3"
    manifest = json.loads(gzip.decompress((p3 / "manifests/fuzzers_seed42.json.gz").read_bytes()))
    train_path = ROOT / "data" / DEFAULT_TRAIN_FILE
    if train_path.is_symlink() or file_hash(train_path) != manifest["source_dataset_hashes"]["official_train_sha256"]:
        raise ValueError("official-training source differs from frozen P3")
    train = load_unsw_csvs(train_path.parent, files=[train_path], source_role="official_train")
    context = train_only_context(train)
    schema = json.loads((p3 / "schema_v2.json").read_text())
    if schema["sha256"] != manifest["feature_schema_sha256"]:
        raise ValueError("frozen P3 schema mismatch")
    test_path = ROOT / "data" / DEFAULT_TEST_FILE
    if test_path.is_symlink() or not test_path.is_file():
        raise PermissionError("official test must be the exact regular file")
    source_sha256 = file_hash(test_path)
    prior_audit = json.loads((p3 / "official_test_external_audit.json").read_text())
    if source_sha256 != prior_audit["source_dataset_hashes"]["official_test_sha256"]:
        raise ValueError("official-test CSV differs from previously frozen audit")
    test = load_unsw_csvs(test_path.parent, files=[test_path], source_role="official_test")
    prepared = prepare_official_test(test, context)
    ambiguity = json.loads((p3 / "train_only_ambiguity.json").read_text())
    conflicts = set(int(value) for value in ambiguity["excluded_identity_hashes"])
    classified, report = classify_official_test(prepared, context, conflicts)
    if report["category_rows"] != prior_audit["category_rows"] or report["category_unique_identities"] != prior_audit["category_unique_identities"]:
        raise ValueError("official-test surface classification differs from frozen P3 audit")
    if set(classified.test_category.astype(str)) != set(SURFACES):
        raise ValueError("official-test surface classification incomplete")
    clean = classified[classified.test_category == "CLEAN"]
    retained_train_ids = set(context["train"].loc[
        ~context["train"][CANONICAL].isin(conflicts), CANONICAL].astype(str))
    if (set(clean[CANONICAL].astype(str)) & retained_train_ids
            or set(clean[CANONICAL].astype(str)) & set(map(str, conflicts))
            or (clean.groupby(CANONICAL).attack_cat.nunique() > 1).any()):
        raise ValueError("forbidden canonical identity intersection on clean surface")
    return context, schema, classified, source_sha256


def clean_surface_hash(classified: pd.DataFrame) -> str:
    clean = classified[classified.test_category == "CLEAN"]
    records = [[int(index), str(int(row[CANONICAL])), str(row.attack_cat)]
               for index, row in clean.iterrows()]
    return stable_hash(records)


def _detector(cell: dict, manifest: dict) -> tuple[TrainingOnlyModel, np.ndarray]:
    model = TrainingOnlyModel(vocabularies_from_manifest(manifest))
    checkpoint = ROOT / cell["reference"]["path"]
    if file_hash(checkpoint) != cell["reference"]["checkpoint_file_sha256"]:
        raise ValueError("P6 checkpoint changed after pre-registration")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    centers = verify_scorer_states(cell["states"], cell["run"]["checkpoint_state_sha256"])
    return model, centers


def _score_frame(frame: pd.DataFrame, context: dict, schema: dict, manifest: dict,
                 model: TrainingOnlyModel, selected: str, centers: np.ndarray) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype=np.float64)
    if set(frame[SOURCE_ROLE_COLUMN].astype(str)) != {"official_test"}:
        raise ValueError("external scorer accepts official-test source only")
    raw = continuous_matrix(frame, context, schema)
    scaler = manifest["scaler_provenance"]
    center = np.asarray(scaler["scaler_center"], dtype=np.float64)
    scale = np.asarray(scaler["scaler_scale"], dtype=np.float64)
    scaled = np.asarray((raw - center) / scale, dtype=np.float64)
    adapter = StructuredBatchAdapter(vocabularies_from_manifest(manifest))
    batch = adapter.transform(scaled, {name: frame[name].tolist() for name in CATEGORICAL_FIELDS})
    role = {"batch": batch, "rows": len(frame)}
    scores = scorer_scores(model, role, selected, centers)
    if scores.dtype != np.float64 or not np.isfinite(scores).all():
        raise ValueError("nonfinite or non-float64 official-test score")
    return scores


def _oov_audit(frame: pd.DataFrame, manifest: dict, target: str) -> dict:
    vocab = vocabularies_from_manifest(manifest)
    output = {"label": "TEST-TIME OOV DIAGNOSTIC ONLY", "population": "primary CLEAN target + known",
              "features": {}}
    target_mask = (frame.attack_cat == target).to_numpy()
    known_mask = frame.attack_cat.isin(KNOWN_ATTACK_CATS).to_numpy()
    for name in CATEGORICAL_FIELDS:
        mapped = np.asarray([vocab[name].encode(value) == OOV_INDEX for value in frame[name]], dtype=bool)
        identities = set(frame.loc[mapped, CANONICAL].astype(str))
        output["features"][name] = {
            "oov_rows": int(mapped.sum()), "oov_identities": len(identities),
            "oov_row_rate": float(mapped.mean()),
            "target_row_oov_rate": float(mapped[target_mask].mean()),
            "known_row_oov_rate": float(mapped[known_mask].mean()),
            "vocabulary_expanded": False,
        }
    return output


def _primary_cell(target: str, seed: int, context: dict, schema: dict,
                  classified: pd.DataFrame, test_sha: str,
                  surface_sha: str, protocol_hash: str, source_sha: str) -> dict:
    cell = verify_frozen_cell(ROOT, target, seed)
    manifest_path = ROOT / f"results/data_quality/p3/manifests/{target.lower()}_seed{seed}.json.gz"
    manifest = json.loads(gzip.decompress(manifest_path.read_bytes()))
    model, centers = _detector(cell, manifest)
    selected = cell["selection"]["selected_scorer"]
    threshold = float(cell["calibration"]["threshold"])
    primary = classified[(classified.test_category == "CLEAN") &
                         classified.attack_cat.isin([*KNOWN_ATTACK_CATS, target])].copy()
    scores = _score_frame(primary, context, schema, manifest, model, selected, centers)
    metric_frame = pd.DataFrame({
        "surface": primary.test_category.to_numpy(str),
        "canonical_identity": primary[CANONICAL].astype(str).to_numpy(),
        "family": primary.attack_cat.to_numpy(str),
        "score": scores,
    })
    measured = primary_metrics(metric_frame, target, threshold)
    intervals = bootstrap_intervals(
        measured["identity_known_scores"], measured["identity_target_scores"], threshold,
        seed=bootstrap_seed(target, seed))
    target_scores = measured.pop("identity_target_scores")
    known_scores = measured.pop("identity_known_scores")
    prefix = f"primary/{target.lower()}/seed_{seed}"
    counts = {"target": target, "split_seed": seed, **measured["counts"],
              "primary_surface": "official-test-clean",
              "small_n_warning": "SMALL-N EXTERNAL TARGET" if target in SMALL_N else None}
    identity = {"label": "PRIMARY EXTERNAL CANONICAL-IDENTITY RESULT",
                "target": target, "split_seed": seed,
                "metrics": measured["identity_metrics"], "confidence_intervals": intervals,
                "small_n_warning": counts["small_n_warning"],
                "uncertainty_caution": ("A finite confidence interval does not imply adequate "
                                        "sample size or broad population representativeness.")}
    row = measured["row_diagnostics"]
    confusion = {key: measured["identity_metrics"][key] for key in ("TP", "FP", "TN", "FN")}
    oov = _oov_audit(primary, manifest, target)
    provenance = {"p3_collection_sha256": P3_HASH,
                  "p3_manifest_sha256": cell["dry"]["p3_manifest_sha256"],
                  "p4_model_config_sha256": cell["dry"]["p4_model_config_sha256"],
                  "p5_training_config_sha256": cell["dry"]["p5_training_config_sha256"],
                  "checkpoint_file_sha256": cell["reference"]["checkpoint_file_sha256"],
                  "checkpoint_state_sha256": cell["run"]["checkpoint_state_sha256"],
                  "p6_scorer_state_sha256": cell["states"]["states"][selected]["scorer_state_sha256"],
                  "p6_calibration_artifact_sha256": cell["calibration_artifact_sha256"],
                  "p7_protocol_sha256": protocol_hash,
                  "official_test_source_sha256": test_sha,
                  "official_test_clean_surface_sha256": surface_sha,
                  "source_code_sha256": source_sha,
                  "selected_scorer": selected, "frozen_threshold": threshold,
                  "calibration_known_row_alert_fraction": json.loads((ROOT / f"results/model_selection/p6/runs/{target.lower()}/seed_{seed}/development_diagnostics.json").read_text())["roles"]["calibration"]["row_alert_fraction"],
                  "external_clean_known_row_alert_fraction": row["metrics"]["FPR"],
                  "threshold_changed": False, "scorer_refitted": False,
                  "official_test_used_for_fitting": False}
    for name, payload in (("external_counts.json", counts), ("primary_identity_metrics.json", identity),
                          ("row_diagnostics.json", row), ("confusion.json", confusion),
                          ("oov_audit.json", oov),
                          ("provenance.json", provenance)):
        _write(f"{prefix}/{name}", payload)
    return {"target": target, "split_seed": seed, "selected_scorer": selected,
            "threshold": threshold, "identity_metrics": measured["identity_metrics"],
            "confidence_intervals": intervals, "counts": counts,
            "small_n_warning": counts["small_n_warning"],
            "checkpoint_state_sha256": cell["run"]["checkpoint_state_sha256"],
            "p6_scorer_state_sha256": provenance["p6_scorer_state_sha256"],
            "p6_calibration_artifact_sha256": provenance["p6_calibration_artifact_sha256"],
            "_known_identity_scores": known_scores, "_target_identity_scores": target_scores}


def _primary_hash() -> str:
    base = p7_path(ROOT, "primary")
    hashes = {str(path.relative_to(base)): file_hash(path)
              for path in sorted(base.rglob("*")) if path.is_file() and path.name != "primary_hash.txt"}
    return stable_hash(hashes)


def _diagnostic_surfaces(context: dict, schema: dict, classified: pd.DataFrame) -> dict:
    frozen = p7_path(ROOT, "primary/primary_hash.txt").read_text().strip()
    if _primary_hash() != frozen:
        raise ValueError("primary clean results changed before diagnostic surfaces")
    names = {"DEVELOPMENT-OVERLAP": "development_overlap",
             "TEST-INTERNAL-AMBIGUOUS": "test_internal_ambiguous",
             "TRAIN-CONFLICT-RELATED": "train_conflict_related"}
    labels = {"DEVELOPMENT-OVERLAP": "CONTAMINATED DIAGNOSTIC — NOT EXTERNAL",
              "TEST-INTERNAL-AMBIGUOUS": "AMBIGUOUS-LABEL DIAGNOSTIC",
              "TRAIN-CONFLICT-RELATED": "TRAIN-CONFLICT DIAGNOSTIC"}
    output = {}
    for surface, directory in names.items():
        rows = classified[classified.test_category == surface]
        output[surface] = {"rows": len(rows), "identities": int(rows[CANONICAL].nunique()),
                           "label": labels[surface]}
        for target in TARGETS:
            for seed in SPLIT_SEEDS:
                cell = verify_frozen_cell(ROOT, target, seed)
                manifest = json.loads(gzip.decompress((ROOT / f"results/data_quality/p3/manifests/{target.lower()}_seed{seed}.json.gz").read_bytes()))
                model, centers = _detector(cell, manifest)
                scores = _score_frame(rows, context, schema, manifest, model,
                                      cell["selection"]["selected_scorer"], centers)
                _, identity_scores = identity_means(scores, rows[CANONICAL].astype(str).tolist())
                report = {"label": labels[surface], "surface": surface, "target": target,
                          "split_seed": seed, "primary_hash_frozen_before_diagnostic": frozen,
                          "rows": len(rows), "canonical_identities": len(identity_scores),
                          "family_rows": rows.attack_cat.value_counts().sort_index().to_dict(),
                          "row_score_summary": score_summary(scores),
                          "identity_score_summary": score_summary(identity_scores),
                          "performance_metric_computed": False,
                          "used_to_modify_detector": False}
                _write(f"diagnostics/{directory}/{target.lower()}_seed{seed}.json", report)
    return output


def run_external(protocol_hash: str) -> dict:
    frozen = validate_preregistration(ROOT, protocol_hash)
    intent_path = p7_path(ROOT, "evaluation_intent.json")
    intent_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create is the one-way gate: a failed or completed run cannot silently rerun.
    with intent_path.open("xb") as file:
        file.write(json_bytes({"status": "P7_TEST_GATE_OPENED", "protocol_sha256": protocol_hash,
                               "upstream_aggregate_sha256": frozen["upstream_snapshot"]["aggregate_sha256"],
                               "source_code_sha256": frozen["source_code_sha256"]}))
    try:
        context, schema, classified, test_sha = _source_frames()
        surface_sha = clean_surface_hash(classified)
        raw_cells = []
        for target in TARGETS:
            for seed in SPLIT_SEEDS:
                raw_cells.append(_primary_cell(target, seed, context, schema, classified,
                                               test_sha, surface_sha, protocol_hash,
                                               frozen["source_code_sha256"]))
        cells = [{key: value for key, value in cell.items() if not key.startswith("_")}
                 for cell in raw_cells]
        matrix = {"status": "PRIMARY_CLEAN_FROZEN", "protocol_sha256": protocol_hash,
                  "primary_unit": "canonical_identity", "cells": cells,
                  "per_target_descriptive": {target: across_seed_summary(cells, target)
                                             for target in TARGETS},
                  "no_pooled_cross_target_metric": True}
        _write("primary/p7_primary_matrix.json", matrix)
        primary_digest = _primary_hash()
        write_immutable(p7_path(ROOT, "primary/primary_hash.txt"), (primary_digest + "\n").encode())
        for cell in raw_cells:
            if _primary_hash() != primary_digest:
                raise ValueError("primary hash changed before clean score summary")
            summary = {"label": "POST-PRIMARY DESCRIPTIVE ONLY",
                       "primary_sha256": primary_digest,
                       "target": cell["target"], "split_seed": cell["split_seed"],
                       "known_clean_identities": score_summary(cell["_known_identity_scores"]),
                       "target_clean_identities": score_summary(cell["_target_identity_scores"]),
                       "used_to_modify_detector": False}
            _write(f"diagnostics/clean_score_summary/{cell['target'].lower()}_seed{cell['split_seed']}.json",
                   summary)
        diagnostics = _diagnostic_surfaces(context, schema, classified)
        if upstream_snapshot(ROOT)["aggregate_sha256"] != frozen["upstream_snapshot"]["aggregate_sha256"]:
            raise ValueError("upstream scientific artifacts changed during external evaluation")
        diag_base = p7_path(ROOT, "diagnostics")
        diagnostic_hashes = {str(path.relative_to(diag_base)): file_hash(path)
                             for path in sorted(diag_base.rglob("*")) if path.is_file()}
        gate = {"status": "P7_EXTERNAL_EVALUATION_COMPLETE",
                "protocol_sha256": protocol_hash, "primary_sha256": primary_digest,
                "diagnostic_files_sha256": diagnostic_hashes,
                "diagnostic_surface_counts": diagnostics,
                "upstream_aggregate_sha256": frozen["upstream_snapshot"]["aggregate_sha256"],
                "official_test_source_sha256": test_sha,
                "official_test_clean_surface_sha256": surface_sha,
                "all_fifteen_cells": len(cells) == 15,
                "upstream_mutated": False, "adaptive_protocol_change": False,
                "universal_zero_day_effectiveness_claim": False}
        _write("evaluation_gate.json", gate)
        return gate
    except Exception as error:
        _write("external_eval_blocked.json", {"status": "P7_EXTERNAL_EVAL_BLOCKED",
                                               "protocol_sha256": protocol_hash,
                                               "reason": f"{type(error).__name__}: {error}"})
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-hash", required=True)
    parser.add_argument("--enable-official-test", action="store_true", required=True)
    args = parser.parse_args()
    if not args.enable_official_test:
        parser.error("--enable-official-test is mandatory")
    gate = run_external(args.protocol_hash)
    print(f"[P7] {gate['status']}; primary hash {gate['primary_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
