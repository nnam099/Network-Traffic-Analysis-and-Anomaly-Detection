"""P2 data-only scientific representation candidates; no model is constructed."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler

from .collision_audit import collision_gate, collision_taxonomy
from .dataset import engineer_features
from .preprocessing_p1 import CATEGORY_COLUMNS, normalized_category
from .target_isolation import (
    CANONICAL_FINGERPRINT_COLUMN as CANONICAL,
    P0_SCHEMA_SHA256,
    _split_canonical_known,
)


SCHEMA_VERSION = "scientific-feature-schema-v2"
MODEL_INPUT_VERSION = "continuous-float64-categorical-opaque-id-v1"
FLOAT64_LEGACY_SCHEMA_VERSION = "post-robustscale-float64-sha256fallback-v3"
AMBIGUITY_POLICY_VERSION = "whole-canonical-conflict-group-exclusion-candidate-v1"
ROLE_NAMES = ("train", "val", "meta_known", "calibration", "surrogate_ood")
KNOWN_ROLE_NAMES = ROLE_NAMES[:4]
OOV_ID_VERSION = "feature-namespaced-full-sha256-opaque-v1"


def _json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def build_schema_v2(context: dict) -> dict:
    continuous = [name for name in context["feat_cols"]
                  if name not in {f"{category}_num" for category in CATEGORY_COLUMNS}]
    if len(continuous) != 58 or len(context["feat_cols"]) != 61:
        raise AssertionError("P2 schema requires the P0 semantic feature set")
    contract = {
        "version": SCHEMA_VERSION,
        "source_feature_names": list(context["feat_cols"]),
        "source_schema_sha256": P0_SCHEMA_SHA256,
        "continuous_features": continuous,
        "categorical_features": list(CATEGORY_COLUMNS),
        "semantic_feature_count": 61,
        "ordered_model_inputs": ["continuous_float64", *(
            f"{category}_opaque_id" for category in CATEGORY_COLUMNS
        )],
        "continuous_dtype": "IEEE-754 binary64, finite; little-endian serialization; signed zero canonicalized",
        "continuous_preprocessing": (
            "existing P0 engineered feature formulas (derived values remain float32 before scaling), "
            "RobustScaler fit on target-purged backbone only, no clipping, float64 model boundary"
        ),
        "categorical_dtype": "opaque tagged ID; never cast to a model numeric magnitude",
        "categorical_normalization": "Unicode 15.1.0 NFKC + trim + casefold; missing: distinct from literal",
        "categorical_vocabulary_provenance": "target-purged backbone train only; sorted normalized tokens",
        "oov_semantics": (
            "Full feature-namespaced SHA-256 digest as an opaque OOV identity, without a fixed bucket count. "
            "A finite trained embedding table for arbitrary unseen tokens is NOT specified."
        ),
        "model_input_structure": (
            "tuple(float64[58], proto ID, service ID, state ID); categorical IDs require "
            "categorical-specific components in a future untrained model"
        ),
        "model_input_identity_version": MODEL_INPUT_VERSION,
    }
    return {"contract": contract, "sha256": _json_hash(contract)}


def ambiguous_canonical_set(context: dict) -> set[int]:
    combined = pd.concat([context["train"], context["test"]], ignore_index=True)
    counts = combined.groupby(CANONICAL)["attack_cat"].nunique()
    return set(int(value) for value in counts[counts > 1].index)


def ambiguity_impact(context: dict, ambiguous: set[int]) -> dict:
    output = {"policy": AMBIGUITY_POLICY_VERSION, "active_in_repository": False,
              "test_label_aware_curation": True, "per_source_family": {}, "per_target": {}}
    for name, frame in (("official_train", context["train"]), ("official_test", context["test"])):
        removed = frame[CANONICAL].isin(ambiguous)
        output["per_source_family"][name] = {}
        for family in [*context["known_cats"], *context["zd_cats"]]:
            original = frame[frame.attack_cat == family]
            kept = original[~original[CANONICAL].isin(ambiguous)]
            output["per_source_family"][name][family] = {
                "original_rows": len(original),
                "ambiguous_rows_removed": len(original) - len(kept),
                "retained_rows": len(kept),
                "percent_retained": 100.0 * len(kept) / len(original) if len(original) else None,
                "unique_canonical_identities_retained": int(kept[CANONICAL].nunique()),
            }
        output[name + "_rows_removed"] = int(removed.sum())
    output["ambiguous_groups"] = len(ambiguous)
    output["rows_removed_total"] = sum(output[name + "_rows_removed"] for name in
                                        ("official_train", "official_test"))
    for target in context["zd_cats"]:
        versions = {}
        for activate in (False, True):
            frames = _candidate_frames(context, target, ambiguous if activate else set())
            key = "after_whole_group_exclusion" if activate else "before_exclusion"
            versions[key] = {
                "target_ood_train_rows": len(frames["target_train"]),
                "target_official_test_rows": len(frames["target_test"]),
                "known_rows_before_target_purge": len(frames["known_before"]),
                "known_rows_after_target_purge": len(frames["known"]),
                "known_class_support_after_target_purge": frames["known"].attack_cat.value_counts().to_dict(),
                "surrogate_rows_after_target_purge": len(frames["surrogate"]),
                "surrogate_family_support": frames["surrogate"].attack_cat.value_counts().to_dict(),
                "official_test_known_class_support": frames["test_known"].attack_cat.value_counts().to_dict(),
            }
        after = versions["after_whole_group_exclusion"]
        output["per_target"][target] = {
            **versions,
            "zero_support_flag": [name for name, value in (
                ("target_ood_train", after["target_ood_train_rows"]),
                ("target_official_test", after["target_official_test_rows"]),
                ("known", after["known_rows_after_target_purge"]),
                ("surrogate", after["surrogate_rows_after_target_purge"]),
            ) if value == 0],
            "statistical_power_note": "No minimum-sample threshold or power guarantee is inferred from nonzero support.",
        }
    return output


def _candidate_frames(context: dict, target: str, ambiguous: set[int]) -> dict:
    train = context["train"][~context["train"][CANONICAL].isin(ambiguous)]
    test = context["test"][~context["test"][CANONICAL].isin(ambiguous)]
    target_train = train[train.attack_cat == target]
    target_test = test[test.attack_cat == target]
    identities = set(target_train[CANONICAL]) | set(target_test[CANONICAL])
    known_before = train[train.attack_cat.isin(context["known_cats"])]
    known = known_before[~known_before[CANONICAL].isin(identities)]
    surrogate = train[train.attack_cat.isin(context["zd_cats"]) &
                      (train.attack_cat != target) & ~train[CANONICAL].isin(identities)]
    return {
        "target_train": target_train, "target_test": target_test,
        "target": pd.concat([target_train, target_test], ignore_index=True),
        "known_before": known_before, "known": known, "surrogate": surrogate,
        "test_known": test[test.attack_cat.isin(context["known_cats"])],
        "target_identity_count": len(identities),
    }


def group_split_roles(context: dict, target: str, seed: int,
                      ambiguous: set[int], *, exclude_ambiguous: bool) -> tuple[dict, dict]:
    population = _candidate_frames(context, target, ambiguous if exclude_ambiguous else set())
    known_parts = _split_canonical_known(population["known"], seed)
    roles = {
        "train": known_parts["backbone_train"],
        "val": known_parts["validation"],
        "meta_known": known_parts["meta_known"],
        "calibration": known_parts["calibration"],
        "surrogate_ood": population["surrogate"],
    }
    return population, roles


def role_canonical_intersections(roles: dict) -> dict:
    sets = {name: set(frame[CANONICAL]) for name, frame in roles.items()}
    return {f"{left}__{right}": len(sets[left] & sets[right])
            for left, right in itertools.combinations(roles, 2)}


def row_split_comparator(known: pd.DataFrame, surrogate: pd.DataFrame, seed: int) -> dict:
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    ids = np.empty(len(known), dtype=np.int8)
    labels = known.attack_cat.to_numpy()
    for fold_id, (_, held) in enumerate(splitter.split(np.zeros(len(known)), labels)):
        ids[held] = fold_id
    roles = {"train": known.iloc[np.flatnonzero(ids < 7)],
             "val": known.iloc[np.flatnonzero(ids == 7)],
             "meta_known": known.iloc[np.flatnonzero(ids == 8)],
             "calibration": known.iloc[np.flatnonzero(ids == 9)],
             "surrogate_ood": surrogate}
    return {"split_policy": "diagnostic_row_level_70_10_10_10",
            "role_rows": {k: len(v) for k, v in roles.items()},
            "canonical_role_intersections": role_canonical_intersections(roles)}


def continuous_matrix(frame: pd.DataFrame, context: dict, schema: dict) -> np.ndarray:
    prepared = frame.copy()
    base = [name for name in context["base_feat_cols"]
            if name not in {f"{category}_num" for category in CATEGORY_COLUMNS}]
    prepared, _ = engineer_features(prepared, base)
    values = prepared[schema["contract"]["continuous_features"]].to_numpy(np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("nonfinite P2 continuous features; no imputation or saturation authorized")
    return values


def fit_opaque_vocabulary(backbone: pd.DataFrame) -> dict:
    return {feature: {token: i for i, token in enumerate(sorted(
        backbone[feature].map(normalized_category).unique()
    ))} for feature in CATEGORY_COLUMNS}


def oov_digest(feature: str, token: str) -> str:
    payload = json.dumps([OOV_ID_VERSION, feature, token], ensure_ascii=False,
                         separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def opaque_category_ids(frame: pd.DataFrame, vocabulary: dict) -> dict[str, np.ndarray]:
    output = {}
    for feature in CATEGORY_COLUMNS:
        mapping = vocabulary[feature]
        tokens = frame[feature].map(normalized_category)
        lookup = {token: f"known:{mapping[token]}" if token in mapping
                  else "oov:" + oov_digest(feature, token)
                  for token in tokens.unique()}
        output[feature] = tokens.map(lookup).to_numpy(str)
    return output


def structured_model_fingerprints(numeric: np.ndarray, category_ids: dict) -> np.ndarray:
    """SHA-256 over binary64 coordinates and opaque categorical IDs; no magnitudes."""
    values = np.ascontiguousarray(numeric, dtype="<f8").copy()
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("P2 model input requires finite float64 continuous vectors")
    values[values == 0] = 0
    count = len(values)
    if any(len(category_ids[feature]) != count for feature in CATEGORY_COLUMNS):
        raise ValueError("misaligned categorical IDs")
    result = []
    for i, row in enumerate(values):
        digest = hashlib.sha256(MODEL_INPUT_VERSION.encode() + row.tobytes())
        for feature in CATEGORY_COLUMNS:
            identity = category_ids[feature][i].encode()
            digest.update(len(identity).to_bytes(4, "big"))
            digest.update(identity)
        result.append(digest.hexdigest())
    return np.asarray(result, dtype="U64")


def oov_risk_report(frames: dict, vocabulary: dict) -> dict:
    """Illustrative bucket sizes are compared, never selected as a contract."""
    output = {"selected_embedding_bucket_count": None, "per_feature": {}}
    for feature in CATEGORY_COLUMNS:
        observed = set().union(*(set(frame[feature].map(normalized_category))
                                 for frame in frames.values()))
        unseen = sorted(observed - vocabulary[feature].keys())
        digests = [oov_digest(feature, token) for token in unseen]
        n = len(digests)
        bucket_examples = {}
        for bits in (10, 16, 20, 24):
            buckets = 2**bits
            assigned = [int(digest, 16) % buckets for digest in digests]
            histogram = pd.Series(assigned).value_counts() if assigned else pd.Series(dtype=int)
            pair_probability = -math.expm1(-n * (n - 1) / (2 * buckets))
            bucket_examples[str(buckets)] = {
                "expected_colliding_pairs": n * (n - 1) / (2 * buckets),
                "birthday_approx_probability_at_least_one": pair_probability,
                "observed_colliding_buckets": int((histogram > 1).sum()),
                "observed_colliding_pairs": int(sum(x * (x - 1) // 2 for x in histogram)),
            }
        output["per_feature"][feature] = {
            "known_vocabulary_tokens": len(vocabulary[feature]),
            "observed_unseen_tokens": n,
            "full_sha256_observed_collisions": n - len(set(digests)),
            "full_sha256_pair_bound": n * (n - 1) / (2 * 2**256),
            "hypothetical_bounded_tables_not_selected": bucket_examples,
        }
    return output


def candidate_fingerprints(context: dict, schema: dict, roles: dict,
                           population: dict) -> tuple[dict, dict, dict, dict]:
    """Fit only on known backbone; return model IDs, scaler, vocab, OOV diagnostics."""
    frames = {**roles, "target": population["target"], "official_test_known": population["test_known"]}
    raw = {name: continuous_matrix(frame, context, schema) for name, frame in frames.items()}
    scaler = RobustScaler().fit(raw["train"])
    vocabulary = fit_opaque_vocabulary(roles["train"])
    transformed = {}
    identities = {}
    for name, frame in frames.items():
        transformed[name] = np.asarray(scaler.transform(raw[name]), dtype=np.float64)
        if not np.isfinite(transformed[name]).all():
            raise ValueError("nonfinite P2 scaled value")
        identities[name] = structured_model_fingerprints(
            transformed[name], opaque_category_ids(frame, vocabulary)
        )
    diagnostics = oov_risk_report(frames, vocabulary)
    return identities, {"center": scaler.center_, "scale": scaler.scale_,
                        "train_row_ids": roles["train"]["__official_train_row_id"].to_numpy()}, vocabulary, diagnostics


def audit_model_inputs(roles: dict, population: dict, fingerprints: dict,
                       *, extra_fail_reasons: list[dict] | None = None) -> dict:
    fit = {name: fingerprints[name] for name in ROLE_NAMES}
    fit["target"] = fingerprints["target"]
    labels = {name: roles[name].attack_cat.astype(str).to_numpy() for name in ROLE_NAMES}
    labels["target"] = population["target"].attack_cat.astype(str).to_numpy()
    taxonomy = collision_taxonomy(fit, labels)
    target_set = set(population["target"][CANONICAL])
    canonical_target = {role: len(target_set & set(frame[CANONICAL])) for role, frame in roles.items()}
    canonical_pairs = role_canonical_intersections(roles)
    gate = collision_gate(taxonomy, canonical_target)
    extra = list(extra_fail_reasons or [])
    for pair, count in canonical_pairs.items():
        if count:
            extra.append({"severity": "CRITICAL FAIL", "reason": "cross_role_canonical_identity",
                          "pair": pair, "fingerprints": count})
    train_ids = set().union(*(set(frame[CANONICAL]) for frame in roles.values()))
    test_known_intersection = len(train_ids & set(population["test_known"][CANONICAL]))
    if test_known_intersection:
        extra.append({"severity": "CRITICAL FAIL", "reason": "official_known_test_canonical_exposure",
                      "fingerprints": test_known_intersection})
    test_model_intersection = len(set(fingerprints["official_test_known"]) & set().union(*(
        set(fingerprints[role]) for role in ROLE_NAMES
    )))
    if test_model_intersection:
        extra.append({"severity": "CRITICAL FAIL", "reason": "official_known_test_model_input_exposure",
                      "fingerprints": test_model_intersection})
    if any(set(frame["__source_role"].astype(str)) != {"official_train"} for frame in roles.values()):
        extra.append({"severity": "CRITICAL FAIL", "reason": "official_test_in_fit_role"})
    all_reasons = gate["fail_reasons"] + extra
    gate["fail_reasons"] = all_reasons
    gate["gate"] = "FAIL" if all_reasons else "PASS"
    gate["severity"] = "CRITICAL FAIL" if any(x["severity"] == "CRITICAL FAIL" for x in all_reasons) else (
        "FAIL" if all_reasons else "PASS"
    )
    return {"gate": gate, "collision_taxonomy": taxonomy,
            "canonical_target_intersections": canonical_target,
            "canonical_fit_role_intersections": canonical_pairs,
            "official_test_known_canonical_exposure": test_known_intersection,
            "official_test_known_model_exposure": test_model_intersection,
            "unique_model_input_collision_fingerprints": int(sum(
                count > 1 for count in Counter(itertools.chain.from_iterable(
                    fingerprints[role] for role in (*ROLE_NAMES, "target")
                )).values()
            )),
    }
