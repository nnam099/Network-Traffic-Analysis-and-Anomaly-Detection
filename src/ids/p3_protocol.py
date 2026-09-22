"""P3 test-blind, data-only development and post-freeze test diagnostics.

The development functions accept official train only. External-test functions
are deliberately separate and must be called after the caller persists and
hashes the complete development manifest collection.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .collision_audit import collision_gate, collision_taxonomy
from .dataset import (
    KNOWN_ATTACK_CATS, MODEL_EXCLUDED_COLUMNS, ROW_ID_COLUMN, SOURCE_FILE_COLUMN,
    SOURCE_ROLE_COLUMN, ZERO_DAY_ATTACK_CATS, _assert_model_feature_schema,
    engineer_features, feature_schema_hash, normalize_labels,
    validate_declared_classes,
)
from .p2_protocol import (
    MODEL_INPUT_VERSION, ROLE_NAMES, continuous_matrix,
    fit_opaque_vocabulary, opaque_category_ids, role_canonical_intersections,
    structured_model_fingerprints,
)
from .target_isolation import (
    CANONICAL_CATEGORY_COLUMNS, CANONICAL_FINGERPRINT_COLUMN as CANONICAL,
    CANONICAL_IDENTITY_VERSION, P0_FEATURE_SCHEMA, _split_canonical_known,
    canonical_feature_fingerprints,
)


PROTOCOL_VERSION = "test-blind-scientific-data-protocol-v1"
TRAIN_AMBIGUITY_VERSION = "official-train-only-whole-conflict-group-exclusion-v1"
TEST_CLASSIFICATION_VERSION = "post-freeze-official-test-exclusive-categories-v1"
TEST_CATEGORIES = (
    "CLEAN", "DEVELOPMENT-OVERLAP", "TEST-INTERNAL-AMBIGUOUS",
    "TRAIN-CONFLICT-RELATED",
)


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def train_only_context(train_df: pd.DataFrame) -> dict:
    """Construct the P0 canonical feature contract without a test argument."""
    train = normalize_labels(train_df.copy())
    validate_declared_classes(train, KNOWN_ATTACK_CATS, ZERO_DAY_ATTACK_CATS)
    if SOURCE_ROLE_COLUMN not in train:
        train[SOURCE_ROLE_COLUMN] = "official_train"
    if set(train[SOURCE_ROLE_COLUMN].astype(str)) != {"official_train"}:
        raise ValueError("development context accepts official_train rows only")
    if SOURCE_FILE_COLUMN not in train:
        train[SOURCE_FILE_COLUMN] = "<in-memory-train>"
    train[ROW_ID_COLUMN] = np.arange(len(train), dtype=np.int64)
    schema_reference = train[train.attack_cat.isin(KNOWN_ATTACK_CATS)].copy()
    for category in CANONICAL_CATEGORY_COLUMNS:
        if category in schema_reference:
            schema_reference[f"{category}_num"] = np.float32(0)
    base = sorted(
        column for column in schema_reference
        if column not in MODEL_EXCLUDED_COLUMNS
        and not str(column).startswith("__")
        and pd.api.types.is_numeric_dtype(schema_reference[column])
    )
    schema_reference, features = engineer_features(schema_reference, base.copy())
    features = list(dict.fromkeys(features))
    _assert_model_feature_schema(features)
    if tuple(features) != P0_FEATURE_SCHEMA:
        raise AssertionError("P3 requires unchanged canonical-model-features-v1 schema")
    train[CANONICAL] = canonical_feature_fingerprints(
        train, base, features
    ).to_numpy(dtype=np.uint64)
    return {
        "train": train, "known_cats": list(KNOWN_ATTACK_CATS),
        "zd_cats": list(ZERO_DAY_ATTACK_CATS), "base_feat_cols": base,
        "feat_cols": features, "feature_schema_sha256": feature_schema_hash(features),
        "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
    }


def train_ambiguity_report(context: dict) -> tuple[set[int], dict]:
    """Classify and exclude conflicts using labels in official train only."""
    train = context["train"]
    grouped = train.groupby(CANONICAL, sort=True)["attack_cat"]
    labels = grouped.agg(lambda values: frozenset(values))
    ambiguous = set(int(x) for x in labels.index[labels.map(len) > 1])
    removed = train[train[CANONICAL].isin(ambiguous)]
    retained = train[~train[CANONICAL].isin(ambiguous)]
    known, ood = set(KNOWN_ATTACK_CATS), set(ZERO_DAY_ATTACK_CATS)
    group_counts = grouped.size()
    per_target = {}
    for target in ZERO_DAY_ATTACK_CATS:
        flags = {
            "unambiguous_same_label": labels.map(lambda group: len(group) == 1),
            "conflicting_labels": labels.map(lambda group: len(group) > 1),
            "target_family_containing": labels.map(lambda group, target=target: target in group),
            "known_family_only": labels.map(lambda group: bool(group) and group <= known),
            "mixed_known_ood": labels.map(lambda group: bool(group & known) and bool(group & ood)),
            "mixed_known_target": labels.map(
                lambda group, target=target: bool(group & known) and target in group
            ),
        }
        per_target[target] = {
            name: {"groups": int(mask.sum()),
                   "rows": int(group_counts[mask].sum())}
            for name, mask in flags.items()
        }
    per_family = {}
    for family in [*KNOWN_ATTACK_CATS, *ZERO_DAY_ATTACK_CATS]:
        original = train[train.attack_cat == family]
        excluded = removed[removed.attack_cat == family]
        remaining = retained[retained.attack_cat == family]
        per_family[family] = {
            "original_rows": len(original), "excluded_rows": len(excluded),
            "remaining_rows": len(remaining),
            "remaining_unique_canonical_identities": int(remaining[CANONICAL].nunique()),
        }
    return ambiguous, {
        "version": TRAIN_AMBIGUITY_VERSION,
        "source_role": "official_train ONLY", "official_test_consulted": False,
        "excluded_groups": len(ambiguous), "excluded_rows": len(removed),
        "remaining_rows": len(retained),
        "excluded_identity_hashes": [str(x) for x in sorted(ambiguous)],
        "per_family": per_family, "group_classification_by_target": per_target,
    }


def construct_development_roles(context: dict, target: str, seed: int,
                                ambiguous: set[int]) -> tuple[dict, dict]:
    if target not in ZERO_DAY_ATTACK_CATS:
        raise ValueError(f"undeclared target: {target}")
    retained = context["train"][~context["train"][CANONICAL].isin(ambiguous)]
    target_frame = retained[retained.attack_cat == target]
    target_ids = set(int(x) for x in target_frame[CANONICAL])
    known_before = retained[retained.attack_cat.isin(KNOWN_ATTACK_CATS)]
    surrogate_before = retained[retained.attack_cat.isin(ZERO_DAY_ATTACK_CATS) &
                                (retained.attack_cat != target)]
    known = known_before[~known_before[CANONICAL].isin(target_ids)]
    surrogate = surrogate_before[~surrogate_before[CANONICAL].isin(target_ids)]
    parts = _split_canonical_known(known, seed)
    roles = {
        "train": parts["backbone_train"], "val": parts["validation"],
        "meta_known": parts["meta_known"], "calibration": parts["calibration"],
        "surrogate_ood": surrogate,
    }
    population = {
        "retained": retained, "target_train": target_frame,
        "target_identity_hashes": target_ids,
        "known_rows_before_purge": len(known_before),
        "known_rows_purged": len(known_before) - len(known),
        "surrogate_rows_before_purge": len(surrogate_before),
        "surrogate_rows_purged": len(surrogate_before) - len(surrogate),
        "development_identity_hashes": set(int(x) for x in retained[CANONICAL]),
    }
    return population, roles


def development_model_audit(context: dict, schema: dict, population: dict,
                            roles: dict) -> tuple[dict, dict, dict]:
    """No external-test frame can enter this fitting or audit boundary."""
    frames = {**roles, "target_train": population["target_train"]}
    raw = {name: continuous_matrix(frame, context, schema)
           for name, frame in frames.items()}
    scaler = RobustScaler().fit(raw["train"])
    vocabulary = fit_opaque_vocabulary(roles["train"])
    raw_fps, scaled_fps = {}, {}
    for name, frame in frames.items():
        categories = opaque_category_ids(frame, vocabulary)
        raw_fps[name] = structured_model_fingerprints(raw[name], categories)
        scaled_fps[name] = structured_model_fingerprints(
            np.asarray(scaler.transform(raw[name]), dtype=np.float64), categories
        )
    role_fps = {name: scaled_fps[name] for name in ROLE_NAMES}
    role_fps["target"] = scaled_fps["target_train"]
    labels = {name: roles[name].attack_cat.to_numpy(str) for name in ROLE_NAMES}
    labels["target"] = population["target_train"].attack_cat.to_numpy(str)
    taxonomy = collision_taxonomy(role_fps, labels)
    target_set = population["target_identity_hashes"]
    canonical_target = {name: len(target_set & set(int(x) for x in frame[CANONICAL]))
                        for name, frame in roles.items()}
    role_intersections = role_canonical_intersections(roles)
    gate = collision_gate(taxonomy, canonical_target)
    reasons = gate["fail_reasons"]
    for pair, count in role_intersections.items():
        if count:
            reasons.append({"reason": "cross_role_canonical_identity", "pair": pair,
                            "fingerprints": count, "severity": "CRITICAL FAIL"})
    if any(target_set & set(int(x) for x in frame[CANONICAL]) for frame in roles.values()):
        reasons.append({"reason": "target_identity_in_fit", "severity": "CRITICAL FAIL"})
    if any(set(frame[SOURCE_ROLE_COLUMN]) != {"official_train"}
           for frame in frames.values()):
        reasons.append({"reason": "non_train_source_in_development", "severity": "CRITICAL FAIL"})
    if any(set(frame.attack_cat) & set(population["target_train"].attack_cat)
           for frame in roles.values()):
        reasons.append({"reason": "target_label_in_fit", "severity": "CRITICAL FAIL"})
    table = pd.concat([
        pd.DataFrame({"canonical": frame[CANONICAL].to_numpy(np.uint64),
                      "raw_fp": raw_fps[name], "scaled_fp": scaled_fps[name]})
        for name, frame in frames.items()
    ], ignore_index=True)
    cross_identity_raw = int((table.groupby("raw_fp")["canonical"].nunique() > 1).sum())
    cross_identity_final = int((table.groupby("scaled_fp")["canonical"].nunique() > 1).sum())
    new_scaling_merges = int((table.groupby("scaled_fp")["raw_fp"].nunique() > 1).sum())
    if cross_identity_final:
        reasons.append({"reason": "model_input_cross_identity_collision",
                        "fingerprints": cross_identity_final, "severity": "CRITICAL FAIL"})
    if new_scaling_merges:
        reasons.append({"reason": "scaling_introduced_merge",
                        "fingerprints": new_scaling_merges, "severity": "CRITICAL FAIL"})
    gate["gate"] = "FAIL" if reasons else "PASS"
    gate["severity"] = "CRITICAL FAIL" if any(
        item["severity"] == "CRITICAL FAIL" for item in reasons
    ) else ("FAIL" if reasons else "PASS")
    scaler_payload = np.concatenate([scaler.center_, scaler.scale_]).astype("<f8")
    scaler_row_ids = np.sort(roles["train"][ROW_ID_COLUMN].to_numpy(np.int64))
    fitting = {
        "vocabulary_source_role": "official_train target-purged backbone train",
        "vocabulary_sha256": sha256_json(vocabulary),
        "vocabulary_sizes": {name: len(mapping) for name, mapping in vocabulary.items()},
        "vocabulary_maps": vocabulary,
        "scaler_source_role": "official_train target-purged backbone train",
        "scaler_center_scale_sha256": hashlib.sha256(scaler_payload.tobytes()).hexdigest(),
        "scaler_center": scaler.center_.astype(np.float64).tolist(),
        "scaler_scale": scaler.scale_.astype(np.float64).tolist(),
        "scaler_fit_row_ids_sha256": hashlib.sha256(
            scaler_row_ids.astype("<i8").tobytes()
        ).hexdigest(),
        "scaler_fit_row_count": len(scaler_row_ids),
        "official_test_used": False, "target_used": False,
    }
    audit = {
        "gate": gate, "canonical_role_intersections": role_intersections,
        "canonical_target_intersections": canonical_target,
        "collision_taxonomy": taxonomy,
        "model_input_cross_identity_before_scaling": cross_identity_raw,
        "model_input_cross_identity_final": cross_identity_final,
        "new_scaling_merges": new_scaling_merges,
        "new_dtype_conversion_merges": 0,
    }
    return audit, fitting, vocabulary


def post_freeze_model_input_audit(test: pd.DataFrame, context: dict, schema: dict,
                                  manifest: dict) -> dict:
    """Inspect test only with already frozen preprocessing; never refit."""
    vocabulary = manifest["vocabulary_provenance"]["vocabulary_maps"]
    scaler_meta = manifest["scaler_provenance"]
    center = np.asarray(scaler_meta["scaler_center"], dtype=np.float64)
    scale = np.asarray(scaler_meta["scaler_scale"], dtype=np.float64)
    digest = hashlib.sha256(np.concatenate([center, scale]).astype("<f8").tobytes()).hexdigest()
    if digest != scaler_meta["scaler_center_scale_sha256"]:
        raise AssertionError("frozen scaler parameters differ from provenance hash")
    if sha256_json(vocabulary) != manifest["vocabulary_provenance"]["vocabulary_sha256"]:
        raise AssertionError("frozen vocabulary differs from provenance hash")
    scaler = RobustScaler()
    scaler.center_, scaler.scale_ = center, scale
    scaler.n_features_in_ = len(center)

    def fingerprints(frame):
        numeric = continuous_matrix(frame, context, schema)
        scaled = np.asarray(scaler.transform(numeric), dtype=np.float64)
        return structured_model_fingerprints(
            scaled, opaque_category_ids(frame, vocabulary)
        )

    development_ids = sorted(set(itertools.chain.from_iterable(
        entry["row_ids"] for entry in (
            *manifest["roles"].values(), manifest["target_development_held_out"]
        )
    )))
    development = context["train"].iloc[development_ids]
    development_fps = fingerprints(development)
    test_fps = fingerprints(test)
    development_by_fp = {}
    for fp, canonical in zip(development_fps, development[CANONICAL], strict=True):
        identity = int(canonical)
        previous = development_by_fp.setdefault(fp, identity)
        if previous != identity:
            raise AssertionError("frozen development input crosses canonical identities")
    cross_identity = np.asarray([
        fp in development_by_fp and int(canonical) != development_by_fp[fp]
        for fp, canonical in zip(test_fps, test[CANONICAL], strict=True)
    ], dtype=bool)
    clean = test.test_category.to_numpy(str) == "CLEAN"
    return {
        "frozen_model_input_version": manifest["model_input_identity_version"],
        "official_test_rows": len(test), "clean_test_rows": int(clean.sum()),
        "cross_identity_model_input_collision_rows_all_test": int(cross_identity.sum()),
        "cross_identity_model_input_collision_unique_fingerprints_all_test": len(set(
            test_fps[cross_identity]
        )),
        "cross_identity_model_input_collision_rows_clean_test": int((cross_identity & clean).sum()),
        "cross_identity_model_input_collision_unique_fingerprints_clean_test": len(set(
            test_fps[cross_identity & clean]
        )),
        "official_test_used_for_parameter_fit": False,
    }


def _role_lineage(frame: pd.DataFrame) -> dict:
    ordered = frame.sort_values(ROW_ID_COLUMN)
    return {
        "row_ids": [int(x) for x in ordered[ROW_ID_COLUMN]],
        "canonical_identity_hashes": [str(int(x)) for x in ordered[CANONICAL]],
        "label_counts": ordered.attack_cat.value_counts().sort_index().to_dict(),
    }


def development_manifest(context: dict, schema: dict, ambiguity: dict,
                         population: dict, roles: dict, audit: dict, fitting: dict,
                         *, target: str, seed: int, train_source_sha256: str,
                         commit_sha: str | None, code_sha256: dict) -> dict:
    """Self-contained frozen state; intentionally contains no test metadata."""
    return {
        "version": PROTOCOL_VERSION, "target": target, "seed": seed,
        "source_dataset_hashes": {"official_train_sha256": train_source_sha256},
        "code_commit_sha": commit_sha, "code_source_sha256": code_sha256,
        "canonical_contract_version": CANONICAL_IDENTITY_VERSION,
        "source_feature_schema_sha256": context["feature_schema_sha256"],
        "feature_schema_version": schema["contract"]["version"],
        "feature_schema_sha256": schema["sha256"],
        "model_input_identity_version": MODEL_INPUT_VERSION,
        "ambiguity_policy": TRAIN_AMBIGUITY_VERSION,
        "excluded_ambiguous_official_train_identity_hashes": ambiguity[
            "excluded_identity_hashes"
        ],
        "target_purged_identity_hashes": [str(x) for x in sorted(
            population["target_identity_hashes"]
        )],
        "roles": {name: _role_lineage(frame) for name, frame in roles.items()},
        "target_development_held_out": _role_lineage(population["target_train"]),
        "vocabulary_provenance": {
            key: value for key, value in fitting.items() if key.startswith("vocabulary_")
        },
        "scaler_provenance": {
            key: value for key, value in fitting.items() if key.startswith("scaler_")
        },
        "development_gate": audit["gate"],
        "development_role_intersections": audit["canonical_role_intersections"],
        "official_test_consulted": False,
    }


def prepare_official_test(test_df: pd.DataFrame, context: dict) -> pd.DataFrame:
    """Post-freeze only: apply an already frozen canonical feature contract."""
    test = normalize_labels(test_df.copy())
    validate_declared_classes(test, KNOWN_ATTACK_CATS, ZERO_DAY_ATTACK_CATS)
    if SOURCE_ROLE_COLUMN not in test:
        test[SOURCE_ROLE_COLUMN] = "official_test"
    if set(test[SOURCE_ROLE_COLUMN].astype(str)) != {"official_test"}:
        raise ValueError("external audit accepts official_test rows only")
    test[CANONICAL] = canonical_feature_fingerprints(
        test, context["base_feat_cols"], context["feat_cols"]
    ).to_numpy(dtype=np.uint64)
    return test


def classify_official_test(test: pd.DataFrame, context: dict,
                           train_conflict: set[int]) -> tuple[pd.DataFrame, dict]:
    """Exclusive categories with explicit priority D > C > B > A."""
    development = context["train"][~context["train"][CANONICAL].isin(train_conflict)]
    development_ids = set(int(x) for x in development[CANONICAL])
    internal_counts = test.groupby(CANONICAL).attack_cat.nunique()
    internal_ambiguous = set(int(x) for x in internal_counts[internal_counts > 1].index)
    classified = test.copy()
    classified["train_conflict_related"] = classified[CANONICAL].isin(train_conflict)
    classified["test_internal_ambiguous"] = classified[CANONICAL].isin(internal_ambiguous)
    classified["development_overlap"] = classified[CANONICAL].isin(development_ids)
    classified["test_category"] = np.select(
        [classified["train_conflict_related"], classified["test_internal_ambiguous"],
         classified["development_overlap"]],
        ["TRAIN-CONFLICT-RELATED", "TEST-INTERNAL-AMBIGUOUS", "DEVELOPMENT-OVERLAP"],
        default="CLEAN",
    )
    counts = classified.test_category.value_counts()
    by_family = {}
    for family in [*KNOWN_ATTACK_CATS, *ZERO_DAY_ATTACK_CATS]:
        frame = classified[classified.attack_cat == family]
        by_family[family] = {
            category: {"rows": int((frame.test_category == category).sum()),
                       "unique_canonical_identities": int(frame.loc[
                           frame.test_category == category, CANONICAL
                       ].nunique())}
            for category in TEST_CATEGORIES
        }
    report = {
        "version": TEST_CLASSIFICATION_VERSION,
        "precedence": list(reversed(TEST_CATEGORIES)),
        "all_original_test_rows": len(test),
        "category_rows": {category: int(counts.get(category, 0))
                          for category in TEST_CATEGORIES},
        "category_unique_identities": {
            category: int(classified.loc[
                classified.test_category == category, CANONICAL
            ].nunique()) for category in TEST_CATEGORIES
        },
        "test_internal_ambiguous_groups": len(internal_ambiguous),
        "per_family": by_family,
        "overlapping_condition_counts": {
            "train_conflict_and_test_internal_ambiguous": int((
                classified.train_conflict_related & classified.test_internal_ambiguous
            ).sum()),
            "development_overlap_and_test_internal_ambiguous": int((
                classified.development_overlap & classified.test_internal_ambiguous
            ).sum()),
        },
    }
    if sum(report["category_rows"].values()) != len(test):
        raise AssertionError("official test categorization is not exclusive and exhaustive")
    return classified, report


def test_support(classified: pd.DataFrame, target: str) -> dict:
    target_rows = classified[classified.attack_cat == target]
    known_rows = classified[classified.attack_cat.isin(KNOWN_ATTACK_CATS)]

    def side(frame):
        counts = frame.test_category.value_counts()
        clean = frame[frame.test_category == "CLEAN"]
        return {
            "original_rows": len(frame),
            "clean_rows": len(clean),
            "development_overlap_rows": int(counts.get("DEVELOPMENT-OVERLAP", 0)),
            "test_internal_ambiguous_rows": int(counts.get("TEST-INTERNAL-AMBIGUOUS", 0)),
            "train_conflict_related_rows": int(counts.get("TRAIN-CONFLICT-RELATED", 0)),
            "ambiguous_rows": int(counts.get("TEST-INTERNAL-AMBIGUOUS", 0) +
                                  counts.get("TRAIN-CONFLICT-RELATED", 0)),
            "clean_unique_canonical_identities": int(clean[CANONICAL].nunique()),
            "percentage_clean_retained": 100 * len(clean) / len(frame) if len(frame) else None,
        }

    return {"target_family": target, "target": side(target_rows),
            "known_background": side(known_rows)}


def investigate_historical_738(context: dict, test: pd.DataFrame,
                                classified: pd.DataFrame) -> dict:
    """After freeze, reconstruct P2 D's old test-aware overlap for attribution."""
    from .p2_protocol import ambiguous_canonical_set, group_split_roles

    old_context = {**context, "test": test}
    old_ambiguous = ambiguous_canonical_set(old_context)
    train_groups = context["train"].groupby(CANONICAL).attack_cat
    test_groups = test.groupby(CANONICAL).attack_cat
    train_labels = train_groups.agg(lambda labels: sorted(set(labels))).to_dict()
    test_labels = test_groups.agg(lambda labels: sorted(set(labels))).to_dict()
    train_counts = train_groups.size().to_dict()
    test_counts = test_groups.size().to_dict()
    by_cell, affected = {}, defaultdict(list)
    family_target_seed_role = Counter()
    for target, seed in itertools.product(ZERO_DAY_ATTACK_CATS, (42, 43, 44)):
        old_population, roles = group_split_roles(
            old_context, target, seed, old_ambiguous, exclude_ambiguous=True
        )
        known_test = old_population["test_known"]
        test_ids = set(int(x) for x in known_test[CANONICAL])
        fit_ids = set().union(*(set(int(x) for x in role[CANONICAL])
                                for role in roles.values()))
        overlap = test_ids & fit_ids
        by_cell[f"{target}/{seed}"] = {
            "unique_canonical_identities": len(overlap),
            "role_unique_identities": {},
        }
        for role, frame in roles.items():
            role_ids = overlap & set(int(x) for x in frame[CANONICAL])
            by_cell[f"{target}/{seed}"]["role_unique_identities"][role] = len(role_ids)
            for identity in role_ids:
                affected[identity].append({"target": target, "seed": seed, "role": role})
                for family in test_labels.get(identity, []):
                    family_target_seed_role[(family, target, seed, role)] += 1
    categories = classified.drop_duplicates(CANONICAL).set_index(CANONICAL).test_category.to_dict()
    train_ambiguous = set(int(x) for x in context["train"].groupby(CANONICAL)
                          .attack_cat.nunique().loc[lambda values: values > 1].index)
    identities = []
    for identity in sorted(affected):
        old_train_labels = train_labels.get(identity, [])
        old_test_labels = test_labels.get(identity, [])
        identities.append({
            "canonical_identity_hash": str(identity),
            "official_train_count": int(train_counts.get(identity, 0)),
            "official_test_count": int(test_counts.get(identity, 0)),
            "official_train_labels": old_train_labels,
            "official_test_labels": old_test_labels,
            "labels_agree": old_train_labels == old_test_labels,
            "family": old_test_labels,
            "development_role_assignments": affected[identity],
            "p3_official_test_category": categories.get(identity),
            "in_train_only_ambiguity_exclusion": identity in train_ambiguous,
        })
    return {
        "source": "reconstructed historical P2 configuration D; post-freeze diagnostic only",
        "global_unique_canonical_identities": len(identities),
        "per_cell": by_cell,
        "interpretation": "per-cell overlap counts; union and per-identity records are reported separately",
        "by_family_target_seed_role": [
            {"family": family, "target": target, "seed": seed, "role": role,
             "unique_identities": count}
            for (family, target, seed, role), count in sorted(family_target_seed_role.items())
        ],
        "identity_records": identities,
        "label_agreement_counts": dict(Counter(
            "agree" if row["labels_agree"] else "disagree" for row in identities
        )),
        "p3_test_category_counts": dict(Counter(
            row["p3_official_test_category"] for row in identities
        )),
        "train_only_conflict_involvement": sum(
            row["in_train_only_ambiguity_exclusion"] for row in identities
        ),
    }
