#!/usr/bin/env python3
"""Reconstruct every P1 float64-to-float32 merge with full vector evidence."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ids.dataset import load_official_unsw_splits  # noqa: E402
from ids.preprocessing_p1 import matrix_fingerprints, transform_stages  # noqa: E402
from ids.target_isolation import (  # noqa: E402
    CANONICAL_FINGERPRINT_COLUMN as CANONICAL,
    _prepare_model_frame, build_target_isolation_context,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ulp32(value: float) -> dict:
    scalar = np.float32(value)
    up = np.nextafter(scalar, np.float32(np.inf))
    down = np.nextafter(scalar, np.float32(-np.inf))
    upper = float(up) - float(scalar) if np.isfinite(up) else None
    lower = float(scalar) - float(down) if np.isfinite(down) else None
    return {"float32_value": float(scalar), "next_up": float(up) if np.isfinite(up) else None,
            "next_down": float(down) if np.isfinite(down) else None,
            "upper_ulp": upper, "lower_ulp": lower}


def forensic_cell(context, cell: dict, evidence) -> dict:
    train, test = context["train"], context["test"]
    frames = {name: train.iloc[evidence[name + "_row_ids"]] for name in cell["split_counts"]}
    frames["target"] = pd.concat([
        part[part.attack_cat == cell["target_family"]] for part in (train, test)
    ], ignore_index=True)
    features = list(context["feat_cols"])
    scaler = RobustScaler()
    scaler.center_, scaler.scale_ = evidence["scaler_center"], evidence["scaler_scale"]
    scaler.n_features_in_ = len(features)
    scaled, final, fps = {}, {}, {}
    for name, frame in frames.items():
        prepared = _prepare_model_frame(frame, cell["categorical_maps"],
                                        context["base_feat_cols"], features)
        scaled[name], final[name] = transform_stages(prepared[features].to_numpy(np.float64), scaler)
        fps[name] = matrix_fingerprints(final[name])
        archived = (evidence[name + "_model_input_fingerprints"] if name != "target"
                    else evidence["target_model_input_fingerprints"])
        if not np.array_equal(fps[name], archived):
            raise AssertionError(f"P1 vector evidence mismatch: {name}")
    groups = []
    aggregate = Counter()
    entries = cell["stage_collision_audit"]["D_model_float32"]["newly_merged_groups"]
    for entry in entries:
        target_fp = int(entry["fingerprint"])
        locations = {name: np.flatnonzero(values == target_fp) for name, values in fps.items()}
        locations = {name: idx for name, idx in locations.items() if len(idx)}
        pre = np.concatenate([scaled[name][idx] for name, idx in locations.items()])
        post = np.concatenate([final[name][idx] for name, idx in locations.items()])
        if not np.all(post == post[0]):
            raise AssertionError("model fingerprint matches distinct float32 vectors")
        upstream = matrix_fingerprints(pre)
        if len(set(upstream)) < 2:
            raise AssertionError("claimed conversion merge already identical in float64")
        differing = np.flatnonzero(np.any(pre != pre[0], axis=0))
        members = []
        for name, idx in locations.items():
            pre_fp = matrix_fingerprints(scaled[name][idx])
            canonical = frames[name].iloc[idx][CANONICAL].to_numpy(np.uint64)
            labels = frames[name].iloc[idx].attack_cat.astype(str).to_numpy()
            table = pd.DataFrame({"pre_fp": pre_fp, "canonical": canonical, "label": labels})
            for (_, _, label), local in table.groupby(["pre_fp", "canonical", "label"]).indices.items():
                at = idx[local]
                i = int(at[0])
                members.append({
                    "role": name, "label": label, "rows": len(at),
                    "canonical_fingerprint": str(int(frames[name].iloc[i][CANONICAL])),
                    "scaled_float64_vector": scaled[name][i].tolist(),
                    "model_float32_vector": final[name][i].tolist(),
                })
        feature_differences = {}
        for j in differing:
            values = pre[:, j]
            lo, hi = float(values.min()), float(values.max())
            absolute = hi - lo
            relative = absolute / max(abs(lo), abs(hi), np.finfo(np.float64).tiny)
            ulp = _ulp32(post[0, j])
            spacing = ulp["upper_ulp"] or ulp["lower_ulp"]
            feature_differences[features[j]] = {
                "float64_min": lo, "float64_max": hi,
                "absolute_difference": absolute, "relative_difference": relative,
                "float32_ulp": ulp,
                "difference_in_float32_ulp": absolute / spacing if spacing else None,
            }
            aggregate[features[j]] += 1
        groups.append({
            "final_model_fingerprint": entry["fingerprint"],
            "roles": sorted(locations), "labels": sorted({m["label"] for m in members}),
            "rows": len(pre), "distinct_float64_vectors": len(set(upstream)),
            "float32_vector": post[0].tolist(),
            "differing_features": [features[j] for j in differing],
            "feature_differences": feature_differences,
            "members": members,
        })
    if len(groups) != cell["stage_collision_audit"]["D_model_float32"][
        "newly_merged_fingerprints_from_previous_stage"
    ]:
        raise AssertionError("incomplete float32 forensic enumeration")
    return {"target_family": cell["target_family"], "seed": cell["seed"],
            "new_float32_merge_groups": len(groups),
            "groups_with_sinpkt_difference": int(sum("sinpkt" in group["differing_features"] for group in groups)),
            "per_feature_group_counts": dict(sorted(aggregate.items())),
            "stage_B_new_merges": cell["stage_collision_audit"]["B_encoded_float64"]["newly_merged_fingerprints_from_previous_stage"],
            "stage_C_new_merges": cell["stage_collision_audit"]["C_scaled_float64"]["newly_merged_fingerprints_from_previous_stage"],
            "collision_groups": groups}


def main() -> None:
    p1_path = ROOT / "results/data_quality/p1/target_isolated_lofo_dry_run.json"
    p1 = json.loads(p1_path.read_text())
    loaded = load_official_unsw_splits(ROOT / "data")
    context = build_target_isolation_context(loaded["train"], loaded["test"])
    cells = []
    for cell in p1["cells"]:
        print(f"[FLOAT32 FORENSIC] {cell['target_family']}/{cell['seed']}", flush=True)
        evidence_file = ROOT / cell["scaler_fit_evidence"]["artifact_path"]
        if _sha(evidence_file) != cell["scaler_fit_evidence"]["artifact_sha256"]:
            raise AssertionError("P1 evidence checksum mismatch")
        with np.load(evidence_file) as evidence:
            cells.append(forensic_cell(context, cell, evidence))
    total = sum(cell["new_float32_merge_groups"] for cell in cells)
    sinpkt = sum(cell["groups_with_sinpkt_difference"] for cell in cells)
    features = Counter()
    for cell in cells:
        features.update(cell["per_feature_group_counts"])
    report = {
        "version": "float64-to-float32-forensics-v1",
        "p1_audit_sha256": _sha(p1_path), "script_sha256": _sha(Path(__file__)),
        "unit": "unique newly merged final model-input fingerprint per cell; cells are not independent populations",
        "total_cell_merge_incidents": total, "sinpkt_cell_merge_incidents": sinpkt,
        "sinpkt_incident_fraction": sinpkt / total if total else None,
        "feature_counts": dict(sorted(features.items())), "cells": cells,
        "neural_training_performed": False,
    }
    output = ROOT / "results/data_quality/p2/float32_forensics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
