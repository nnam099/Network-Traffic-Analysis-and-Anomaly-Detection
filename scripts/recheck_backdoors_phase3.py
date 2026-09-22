#!/usr/bin/env python3
"""Trace archived Phase 3 Backdoors incidents using its saved backbone row IDs.

The Phase 3 artifact did not save held-out role row IDs. Their complete shared
fingerprint lists (each <= 10) are preserved in the report. We trace every
matching held-out candidate, never infer a particular held-out row assignment.
"""

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

from ids.dataset import (  # noqa: E402
    _encode_categorical_features, engineer_features, load_official_unsw_splits,
    model_input_fingerprints,
)
from ids.preprocessing_p1 import (  # noqa: E402
    CATEGORY_COLUMNS, LEGACY_MODEL_INPUT_IDENTITY_VERSION,
    MODEL_INPUT_IDENTITY_VERSION, normalized_category, transform_stages,
)
from ids.target_isolation import (  # noqa: E402
    CANONICAL_FINGERPRINT_COLUMN as CANONICAL,
    _prepare_model_frame, build_target_isolation_context,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _previous_model(frame, categorical_maps, base_cols, feat_cols):
    prepared = frame.copy()
    _encode_categorical_features(prepared, categorical_maps)
    prepared, _ = engineer_features(prepared, base_cols.copy())
    return prepared[feat_cols].to_numpy(np.float32)


def _traces(frame, idx, old_encoded, old_scaled, old_clipped, new_encoded, new_scaled, new_final,
            new_fp, features, role):
    identities = frame.iloc[idx][CANONICAL].to_numpy(np.uint64)
    result = []
    # Preserve distinct new model vectors within each canonical group and
    # aggregate only truly identical rows to keep the record reviewable.
    pairs = pd.DataFrame({"canonical": identities, "new_fp": new_fp[idx]})
    for (_, _), local in pairs.groupby(["canonical", "new_fp"]).indices.items():
        at = idx[local]
        i = int(at[0])
        vector = {}
        for j, name in enumerate(features):
            category = name[:-4] if name.endswith("_num") else None
            vector[name] = (
                normalized_category(frame.iloc[i][category]) if category in CATEGORY_COLUMNS
                else float(np.float32(new_encoded[i, j]))
            )
        result.append({
            "candidate_role": role, "row_count": len(at),
            "canonical_fingerprint_under_current_contract": str(int(identities[local[0]])),
            "canonical_engineered_vector": vector,
            "normalized_categorical_values": {
                name: normalized_category(frame.iloc[i][name]) for name in CATEGORY_COLUMNS
                if name in frame
            },
            "prior_encoded_float32": old_encoded[i].tolist(),
            "prior_scaled_float32": old_scaled[i].tolist(),
            "prior_clipped_float32": old_clipped[i].tolist(),
            "new_encoded_float64": new_encoded[i].tolist(),
            "new_scaled_float64": new_scaled[i].tolist(),
            "new_final_float32": new_final[i].tolist(),
            "new_model_fingerprint": str(int(new_fp[i])),
        })
    return result


def recheck(context, legacy_cell, p1_cell, old_saved, new_saved):
    train, test = context["train"], context["test"]
    features, base = context["feat_cols"], context["base_feat_cols"]
    seed = legacy_cell["seed"]
    target = pd.concat([
        frame[frame.attack_cat == "Backdoors"] for frame in (train, test)
    ], ignore_index=True)
    target_set = set(target[CANONICAL])
    known = train[train.attack_cat.isin(context["known_cats"]) & ~train[CANONICAL].isin(target_set)]
    surrogate = train[train.attack_cat.isin(context["zd_cats"]) &
                      (train.attack_cat != "Backdoors") & ~train[CANONICAL].isin(target_set)]
    backbone_ids = old_saved["scaler_fit_row_ids"]
    backbone = train.iloc[backbone_ids]
    if len(backbone) != legacy_cell["scaler_fit_evidence"]["row_count"]:
        raise AssertionError("archived backbone count differs")
    held = known[~known["__official_train_row_id"].isin(backbone_ids)]
    frames = {"target": target, "train": backbone, "held_out_known": held,
              "surrogate_ood": surrogate}
    old_map = _encode_categorical_features(backbone.copy())
    old_encoded = {role: _previous_model(frame, old_map, base, features) for role, frame in frames.items()}
    old_scaler = RobustScaler().fit(old_encoded["train"])
    old_scaled = {role: old_scaler.transform(x) for role, x in old_encoded.items()}
    old_clipped = {role: np.clip(np.nan_to_num(x, nan=0., posinf=10., neginf=-10.),
                                 -10., 10.).astype(np.float32) for role, x in old_scaled.items()}
    old_fp = {role: model_input_fingerprints(x).to_numpy(np.uint64)
              for role, x in old_clipped.items()}

    new_scaler = RobustScaler()
    new_scaler.center_, new_scaler.scale_ = new_saved["scaler_center"], new_saved["scaler_scale"]
    new_scaler.n_features_in_ = len(features)
    new_encoded = {}
    new_scaled, new_final, new_fp = {}, {}, {}
    for role, frame in frames.items():
        prepared = _prepare_model_frame(frame, p1_cell["categorical_maps"], base, features)
        new_encoded[role] = prepared[features].to_numpy(np.float64)
        new_scaled[role], new_final[role] = transform_stages(new_encoded[role], new_scaler)
        new_fp[role] = model_input_fingerprints(new_final[role]).to_numpy(np.uint64)
    if not np.array_equal(new_fp["target"], new_saved["target_model_input_fingerprints"]):
        raise AssertionError("P1 target evidence does not reproduce")
    target_old = set(old_fp["target"])
    computed = {
        "train": target_old & set(old_fp["train"]),
        "surrogate_ood": target_old & set(old_fp["surrogate_ood"]),
    }
    old_roles = legacy_cell["transformation_collision_diagnostics"]["target_vs_fit_roles"]
    shared_by_role = {}
    for role in ("train", "val", "meta_known", "calibration", "surrogate_ood"):
        published = old_roles.get(role)
        if published is None:
            shared_by_role[role] = set()
            continue
        samples = {int(item["model_input_fingerprint"]) for item in published["safe_examples"]}
        if role in computed:
            shared = computed[role]
            if not samples.issubset(shared):
                raise AssertionError(f"archived {role} examples not reproduced")
        else:
            if published["shared_model_input_fingerprints"] != len(samples):
                raise AssertionError("held-out examples are incomplete in archive")
            shared = samples
            if not shared.issubset(target_old & set(old_fp["held_out_known"])):
                raise AssertionError(f"archived {role} fingerprint missing from reconstructed held pool")
        if len(shared) != legacy_cell["post_transform_target_intersections"][role]:
            raise AssertionError(f"Phase 3 reference count not reproduced: {role}")
        shared_by_role[role] = shared
    records = []
    for role, fingerprints in shared_by_role.items():
        fit = role if role in ("train", "surrogate_ood") else "held_out_known"
        for fp in sorted(fingerprints):
            target_idx = np.flatnonzero(old_fp["target"] == fp)
            fit_idx = np.flatnonzero(old_fp[fit] == fp)
            if not len(target_idx) or not len(fit_idx):
                raise AssertionError("archived fingerprint lacks both sides")
            overlap = set(new_fp["target"][target_idx]) & set(new_fp[fit][fit_idx])
            combined_old_scaled = np.concatenate([old_scaled["target"][target_idx], old_scaled[fit][fit_idx]])
            prior_causes = []
            if len(set(model_input_fingerprints(combined_old_scaled.astype(np.float32)))) > 1:
                prior_causes.append("hard_clipping")
            for feature in CATEGORY_COLUMNS:
                if feature not in frames[fit]:
                    continue
                j = features.index(feature + "_num")
                tokens = pd.concat([frames["target"].iloc[target_idx][feature],
                                    frames[fit].iloc[fit_idx][feature]]).map(normalized_category)
                codes = np.concatenate([old_encoded["target"][target_idx, j],
                                        old_encoded[fit][fit_idx, j]])
                if tokens.nunique() > len(np.unique(codes)):
                    prior_causes.append("unknown_category_collapse/" + feature)
            records.append({
                "published_role": role, "archived_fingerprint": str(fp),
                "prior_causes": prior_causes,
                "old_shared_fingerprint_reproduced": True,
                "new_collision_disappeared": not overlap,
                "new_shared_fingerprints": sorted(str(int(v)) for v in overlap),
                "held_out_row_assignment_recoverable": role in ("train", "surrogate_ood"),
                "target_traces": _traces(frames["target"], target_idx, old_encoded["target"],
                                          old_scaled["target"], old_clipped["target"],
                                          new_encoded["target"], new_scaled["target"], new_final["target"],
                                          new_fp["target"], features, "target"),
                "fit_candidate_traces": _traces(frames[fit], fit_idx, old_encoded[fit],
                                                 old_scaled[fit], old_clipped[fit],
                                                 new_encoded[fit], new_scaled[fit], new_final[fit],
                                                 new_fp[fit], features, fit),
            })
    return {
        "seed": seed, "prior_contract": LEGACY_MODEL_INPUT_IDENTITY_VERSION,
        "new_contract": MODEL_INPUT_IDENTITY_VERSION,
        "vector_feature_order": features,
        "published_intersections": legacy_cell["post_transform_target_intersections"],
        "reproduced_intersections": {name: len(group) for name, group in shared_by_role.items()},
        "all_published_fingerprints_reproduced": True,
        "all_previous_collisions_disappeared": all(x["new_collision_disappeared"] for x in records),
        "role_fingerprint_incidents": len(records), "collisions": records,
        "held_out_assignment_limit": (
            "Phase 3 retained held-out shared fingerprint lists but not row IDs. "
            "Each held-out trace includes every matching candidate in the exact "
            "non-backbone known pool, without claiming an unrecoverable row assignment."
        ),
    }


def main():
    old_path = ROOT / "results/data_quality/target_isolated_lofo_dry_run.json"
    new_path = ROOT / "results/data_quality/p1/target_isolated_lofo_dry_run.json"
    legacy = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    roles = load_official_unsw_splits(ROOT / "data")
    context = build_target_isolation_context(roles["train"], roles["test"])
    results = []
    for seed in (42, 43, 44):
        print(f"[BACKDOORS ARCHIVE] seed={seed}", flush=True)
        old = next(c for c in legacy["cells"] if c["target_family"] == "Backdoors" and c["seed"] == seed)
        p1 = next(c for c in new["cells"] if c["target_family"] == "Backdoors" and c["seed"] == seed)
        old_file = ROOT / old["scaler_fit_evidence"]["artifact_path"]
        new_file = ROOT / p1["scaler_fit_evidence"]["artifact_path"]
        for path, cell in ((old_file, old), (new_file, p1)):
            if _sha(path) != cell["scaler_fit_evidence"]["artifact_sha256"]:
                raise AssertionError("evidence artifact checksum mismatch")
        with np.load(old_file) as old_saved, np.load(new_file) as new_saved:
            result = recheck(context, old, p1, old_saved, new_saved)
        destination = new_path.parent / f"backdoors_seed{seed}_archived_recheck.json"
        destination.write_text(json.dumps(result, indent=2) + "\n")
        results.append({key: value for key, value in result.items() if key != "collisions"} | {
            "artifact_path": str(destination.relative_to(ROOT)), "artifact_sha256": _sha(destination)
        })
    summary = {"old_report_sha256": _sha(old_path), "new_report_sha256": _sha(new_path),
               "script_sha256": _sha(Path(__file__)),
               "dataset_sha256": new.get("lineage", {}).get("dataset_sha256"),
               "prior_contract": LEGACY_MODEL_INPUT_IDENTITY_VERSION,
               "new_contract": MODEL_INPUT_IDENTITY_VERSION,
               "all_reference_counts_reproduced": all(r["all_published_fingerprints_reproduced"] for r in results),
               "all_previous_collisions_disappeared": all(r["all_previous_collisions_disappeared"] for r in results),
               "seeds": results}
    destination = new_path.parent / "backdoors_archived_recheck_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
