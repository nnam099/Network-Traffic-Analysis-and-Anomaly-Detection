"""Reproduce all v1 Backdoors collisions and trace their P1 counterparts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .dataset import _encode_categorical_features, engineer_features, model_input_fingerprints
from .preprocessing_p1 import (
    CATEGORY_COLUMNS, LEGACY_MODEL_INPUT_IDENTITY_VERSION,
    MODEL_INPUT_IDENTITY_VERSION, matrix_fingerprints, normalized_category,
)


def backdoors_forensics(role_frames, target, model_frames, target_model, scaler,
                       final, target_final, base_cols, feat_cols, seed):
    frames = {**role_frames, "target": target}
    maps = _encode_categorical_features(role_frames["train"].copy())
    old_encoded = {}
    for role, frame in frames.items():
        encoded = frame.copy()
        _encode_categorical_features(encoded, maps)
        encoded, _ = engineer_features(encoded, base_cols.copy())
        old_encoded[role] = encoded[feat_cols].to_numpy(np.float32)
    legacy_scaler = RobustScaler().fit(old_encoded["train"])
    old_scaled = {role: legacy_scaler.transform(x) for role, x in old_encoded.items()}
    old_clipped = {
        role: np.clip(np.nan_to_num(x, nan=0., posinf=10., neginf=-10.), -10., 10.).astype(np.float32)
        for role, x in old_scaled.items()
    }
    old_fp = {role: model_input_fingerprints(x).to_numpy(np.uint64) for role, x in old_clipped.items()}
    new_encoded = {
        role: frame[feat_cols].to_numpy(np.float64)
        for role, frame in {**model_frames, "target": target_model}.items()
    }
    new_scaled = {role: scaler.transform(x) for role, x in new_encoded.items()}
    new_final = {**final, "target": target_final}
    new_fp = {role: matrix_fingerprints(x) for role, x in new_final.items()}
    records, counts = [], {}
    for role in role_frames:
        common = sorted(set(old_fp["target"]) & set(old_fp[role]))
        counts[role] = len(common)
        for fp in common:
            positions = {r: np.flatnonzero(old_fp[r] == fp) for r in ("target", role)}
            remaining = set(new_fp["target"][positions["target"]]) & set(new_fp[role][positions[role]])
            upstream = np.concatenate([old_encoded[r][idx] for r, idx in positions.items()])
            scaled = np.concatenate([old_scaled[r][idx] for r, idx in positions.items()])
            causes = []
            if len(set(matrix_fingerprints(scaled))) > 1:
                causes.append("hard_clipping")
            token_rows = pd.concat([frames[r].iloc[idx] for r, idx in positions.items()])
            for feature in CATEGORY_COLUMNS:
                if feature not in token_rows:
                    continue
                tokens = token_rows[feature].map(normalized_category)
                col = feat_cols.index(feature + "_num")
                if tokens.nunique() > len(np.unique(upstream[:, col])):
                    causes.append("unknown_category_collapse/" + feature)
            members = []
            for r, idx in positions.items():
                # One trace per distinct canonical identity per side, retaining
                # multiplicities; raw record identifiers are never exported.
                selected = frames[r].iloc[idx]
                canonical = selected["__canonical_feature_fingerprint"].to_numpy(np.uint64)
                for identity in sorted(set(canonical)):
                    at = idx[np.flatnonzero(canonical == identity)]
                    i = int(at[0])
                    canonical_vector = {}
                    for j, column in enumerate(feat_cols):
                        raw = column[:-4] if column.endswith("_num") else None
                        canonical_vector[column] = (
                            normalized_category(frames[r].iloc[i][raw])
                            if raw in CATEGORY_COLUMNS
                            else float(np.float32(new_encoded[r][i, j]))
                        )
                    members.append({
                        "role": r, "canonical_fingerprint": str(int(identity)),
                        "row_count": len(at),
                        "source_roles": sorted(frames[r].iloc[at]["__source_role"].unique()),
                        "canonical_engineered_vector": canonical_vector,
                        "normalized_categorical_values": {
                            c: normalized_category(frames[r].iloc[i][c]) for c in CATEGORY_COLUMNS
                            if c in frames[r]
                        },
                        "prior_encoded_float32": old_encoded[r][i].tolist(),
                        "prior_scaled_float32": old_scaled[r][i].tolist(),
                        "prior_clipped_float32": old_clipped[r][i].tolist(),
                        "new_encoded_float64": new_encoded[r][i].tolist(),
                        "new_scaled_float64": new_scaled[r][i].tolist(),
                        "new_final_float32": new_final[r][i].tolist(),
                        "new_model_fingerprint": str(int(new_fp[r][i])),
                        "all_new_model_fingerprints": sorted({str(int(v)) for v in new_fp[r][at]}),
                    })
            records.append({
                "fit_role": role, "prior_shared_fingerprint": str(int(fp)),
                "prior_causes": causes,
                "disappeared": not remaining,
                "remaining_new_shared_fingerprints": sorted(str(int(v)) for v in remaining),
                "explanation": (
                    "P1 keeps unsaturated numeric coordinates and distinguishes unknown categorical tokens; "
                    "all row-level new fingerprints in this old group were compared."
                    if not remaining else "Collision remains; inspect the P1 stage audit."
                ),
                "members": members,
            })
    expected = {
        42: [11, 2, 4, 1, 18], 43: [8, 3, 1, 2, 14], 44: [11, 2, 2, 2, 17],
    }
    return {
        "seed": seed, "prior_contract": LEGACY_MODEL_INPUT_IDENTITY_VERSION,
        "new_contract": MODEL_INPUT_IDENTITY_VERSION,
        "vector_feature_order": list(feat_cols),
        "prior_intersections": counts,
        "current_fold_matches_phase3_reference_counts": list(counts.values()) == expected.get(seed),
        "reference_comparison_note": (
            "Current fold allocations differ from the archived Phase 3 allocations. "
            "This comparison is diagnostic only; recheck_backdoors_phase3.py "
            "uses the saved Phase 3 backbone and fingerprint lists."
        ),
        "all_current_fold_v1_collisions_disappeared": all(r["disappeared"] for r in records),
        "role_fingerprint_incidents": len(records), "collisions": records,
        "prior_scaler_center": legacy_scaler.center_.tolist(),
        "prior_scaler_scale": legacy_scaler.scale_.tolist(),
    }
