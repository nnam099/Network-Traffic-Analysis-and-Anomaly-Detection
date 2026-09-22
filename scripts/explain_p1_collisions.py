#!/usr/bin/env python3
"""Explain saved P1 stage merges from saved row/scaler evidence, without fitting."""

from __future__ import annotations

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
from ids.preprocessing_p1 import (  # noqa: E402
    CATEGORY_COLUMNS, MODEL_INPUT_IDENTITY_VERSION, matrix_fingerprints,
    normalized_category, transform_stages, unknown_category_code,
)
from ids.target_isolation import (  # noqa: E402
    CANONICAL_FINGERPRINT_COLUMN, _prepare_model_frame, build_target_isolation_context,
)


def explain_cell(context, cell, evidence):
    train, test = context["train"], context["test"]
    roles = list(cell["split_counts"])
    frames = {role: train.iloc[evidence[role + "_row_ids"]] for role in roles}
    frames["target"] = pd.concat([
        frame[frame.attack_cat == cell["target_family"]] for frame in (train, test)
    ], ignore_index=True)
    columns = context["feat_cols"]
    scaler = RobustScaler()
    scaler.center_, scaler.scale_ = evidence["scaler_center"], evidence["scaler_scale"]
    scaler.n_features_in_ = len(columns)
    encoded, scaled, final = {}, {}, {}
    for role, frame in frames.items():
        prepared = _prepare_model_frame(frame, cell["categorical_maps"], context["base_feat_cols"], columns)
        encoded[role] = prepared[columns].to_numpy(np.float64)
        scaled[role], final[role] = transform_stages(encoded[role], scaler)
        expected = (evidence[role + "_model_input_fingerprints"] if role != "target"
                    else evidence["target_model_input_fingerprints"])
        if not np.array_equal(matrix_fingerprints(final[role]), expected):
            raise AssertionError(f"saved model-input evidence does not reproduce: {role}")
    matrices = {"B_encoded_float64": encoded, "C_scaled_float64": scaled, "D_model_float32": final}
    previous = {"C_scaled_float64": encoded, "D_model_float32": scaled}
    explanations = {}
    for stage, stage_report in cell["stage_collision_audit"].items():
        if stage == "A_canonical":
            continue
        entries = stage_report["newly_merged_groups"]
        explanations[stage] = []
        if not entries:
            continue
        fingerprints = {role: matrix_fingerprints(x) for role, x in matrices[stage].items()}
        for entry in entries:
            fp = int(entry["fingerprint"])
            positions = {role: np.flatnonzero(values == fp) for role, values in fingerprints.items()}
            positions = {role: idx for role, idx in positions.items() if len(idx)}
            upstream = previous.get(stage, encoded)
            before = np.concatenate([upstream[role][idx] for role, idx in positions.items()])
            after = np.concatenate([matrices[stage][role][idx] for role, idx in positions.items()])
            if not np.all(after == after[0]):
                raise AssertionError(f"hash collision is not vector equality: {stage}/{fp}")
            varying = np.flatnonzero(np.any(before != before[0], axis=0))
            members = []
            for role, idx in positions.items():
                # Collapse only repeated identical provenance/vector records.
                identities = frames[role].iloc[idx][CANONICAL_FINGERPRINT_COLUMN].to_numpy(np.uint64)
                up_fp = matrix_fingerprints(upstream[role][idx])
                unique_pairs = pd.DataFrame({"canonical": identities, "upstream": up_fp})
                for (_, _), positions_in_group in unique_pairs.groupby(["canonical", "upstream"]).indices.items():
                    at = idx[positions_in_group]
                    i = int(at[0])
                    members.append({
                        "role": role, "rows": len(at),
                        "canonical_fingerprint": str(int(frames[role].iloc[i][CANONICAL_FINGERPRINT_COLUMN])),
                        "labels": sorted(frames[role].iloc[at].attack_cat.unique()),
                        "upstream_differing_values": {columns[j]: float(upstream[role][i, j]) for j in varying},
                        "collapsed_output_values": {columns[j]: float(matrices[stage][role][i, j]) for j in varying},
                    })
            explanations[stage].append({
                **entry, "differing_features": [columns[j] for j in varying],
                "exact_vector_equality_verified": True, "members": members,
                "cause": {"B_encoded_float64": "category encoding or numeric preparation",
                          "C_scaled_float64": "float64 affine transform rounding",
                          "D_model_float32": "float64-to-float32 rounding"}[stage],
            })
    categorical_diagnostics = {}
    for category in CATEGORY_COLUMNS:
        mapping = cell["categorical_maps"][category]
        observed = set().union(*(set(f[category].map(normalized_category)) for f in frames.values()))
        unknown = sorted(observed - mapping.keys())
        codes = {token: unknown_category_code(category, token) for token in unknown}
        table = pd.DataFrame({"token": list(codes), "code": list(codes.values())})
        collapsed = {str(code): part.token.tolist() for code, part in table.groupby("code") if len(part) > 1}
        j = columns.index(category + "_num")
        all_codes = {**{token: mapping[token] for token in observed & mapping.keys()}, **codes}
        coordinates = {
            token: float(np.float32((code - scaler.center_[j]) / scaler.scale_[j]))
            for token, code in all_codes.items()
        }
        coord_table = pd.DataFrame({"token": list(coordinates), "coordinate": list(coordinates.values())})
        coordinate_collapses = {
            str(code): part.token.tolist() for code, part in coord_table.groupby("coordinate") if len(part) > 1
        }
        categorical_diagnostics[category] = {
            "known_count": len(mapping), "observed_unknown_count": len(unknown),
            "fallback_token_codes": codes, "fallback_code_collisions": collapsed,
            "final_coordinate_collisions": coordinate_collapses,
            "known_unknown_code_domains_disjoint": all(v < 0 for v in codes.values()) and all(v >= 0 for v in mapping.values()),
        }
    return {"target_family": cell["target_family"], "seed": cell["seed"],
            "saved_model_inputs_reproduced_exactly": True,
            "new_collision_provenance": explanations,
            "categorical_identity_diagnostics": categorical_diagnostics}


def main():
    path = ROOT / "results/data_quality/p1/target_isolated_lofo_dry_run.json"
    report = json.loads(path.read_text())
    if report["model_input_identity_version"] != MODEL_INPUT_IDENTITY_VERSION:
        raise ValueError("wrong input contract")
    roles = load_official_unsw_splits(ROOT / "data")
    context = build_target_isolation_context(roles["train"], roles["test"])
    cells = []
    for cell in report["cells"]:
        print(f"[EXPLAIN] {cell['target_family']}/{cell['seed']}", flush=True)
        evidence_path = ROOT / cell["scaler_fit_evidence"]["artifact_path"]
        if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != cell["scaler_fit_evidence"]["artifact_sha256"]:
            raise AssertionError("evidence checksum mismatch")
        with np.load(evidence_path) as evidence:
            cells.append(explain_cell(context, cell, evidence))
    output = {
        "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
        "neural_training_performed": False, "fitting_performed": False,
        "input_audit_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": cells,
    }
    dest = path.parent / "collision_provenance.json"
    dest.write_text(json.dumps(output, indent=2) + "\n")
    print(dest)


if __name__ == "__main__":
    main()
