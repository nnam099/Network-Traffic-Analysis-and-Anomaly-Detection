"""Target-isolated outer-LOFO data protocol and collision diagnostics."""

from __future__ import annotations

import hashlib
import itertools
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, RobustScaler

from .collision_audit import (
    ambiguity_policy_report, collision_gate, collision_taxonomy, stage_audit,
)
from .preprocessing_p1 import (
    CATEGORY_ENCODING_VERSION,
    MODEL_INPUT_IDENTITY_VERSION,
    REQUIRED_UNICODE_DATA_VERSION,
    encode_categories,
    fit_categorical_maps,
    matrix_fingerprints,
    normalized_category as _normalized_category,
    transform_stages,
)

from .dataset import (
    FINGERPRINT_COLUMN as GROUP_FINGERPRINT_COLUMN,
    KNOWN_ATTACK_CATS,
    MODEL_EXCLUDED_COLUMNS,
    ROW_ID_COLUMN,
    SOURCE_FILE_COLUMN,
    SOURCE_ROLE_COLUMN,
    ZERO_DAY_ATTACK_CATS,
    _assert_model_feature_schema,
    _split_known_by_fingerprint,
    engineer_features,
    feature_schema_hash,
    model_input_fingerprints,
    normalize_labels,
    validate_declared_classes,
)


CANONICAL_FINGERPRINT_COLUMN = "__canonical_feature_fingerprint"
CANONICAL_CATEGORY_COLUMNS = ("proto", "service", "state")
CANONICAL_IDENTITY_VERSION = "canonical-model-features-v1"
P0_FEATURE_COUNT = 61
P0_SCHEMA_SHA256 = "56fdea438ba39694075c4fbfc4584340cf9bfa31e0f1e757a2b557c5473869da"
P0_FEATURE_SCHEMA = (
    "ackdat", "ct_dst_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_flw_http_mthd",
    "ct_ftp_cmd", "ct_src_dport_ltm", "ct_src_ltm", "ct_srv_dst", "ct_srv_src",
    "ct_state_ttl", "dbytes", "dinpkt", "djit", "dload", "dloss", "dmean", "dpkts",
    "dtcpb", "dttl", "dur", "dwin", "is_ftp_login", "is_sm_ips_ports", "proto_num",
    "rate", "response_body_len", "sbytes", "service_num", "sinpkt", "sjit", "sload",
    "sloss", "smean", "spkts", "state_num", "stcpb", "sttl", "swin", "synack",
    "tcprtt", "trans_depth", "bytes_ratio", "log_total_bytes", "log_sbytes",
    "log_dbytes", "pkts_ratio", "log_total_pkts", "src_bps", "log_src_bps",
    "pkt_rate", "load_asym", "log_sload", "ttl_diff", "ttl_sum", "loss_rate_src",
    "loss_rate_dst", "jit_ratio", "log_sjit", "handshake_ratio", "incomplete_tcp",
)
FIT_ROLE_NAMES = ("train", "val", "meta_known", "calibration", "surrogate_ood")
REQUIRED_INVARIANTS = (
    "target_label_absent_from_fit",
    "target_canonical_fingerprints_absent_from_fit",
    "target_model_input_fingerprints_absent_from_fit",
    "surrogate_target_canonical_fingerprints_absent",
    "surrogate_target_model_fingerprints_absent",
    "schema_frozen",
    "known_roles_row_disjoint",
    "known_roles_canonical_fingerprint_disjoint",
    "known_roles_model_input_fingerprint_disjoint",
    "all_fit_roles_row_disjoint",
    "all_fit_roles_canonical_fingerprint_disjoint",
    "all_fit_roles_model_input_fingerprint_disjoint",
    "official_test_excluded_from_fit",
    "calibration_has_no_target_equivalent_rows",
    "required_roles_usable",
)


def _digest_uint64(values) -> str:
    array = np.sort(np.unique(np.asarray(values, dtype=np.uint64)))
    return hashlib.sha256(array.astype("<u8", copy=False).tobytes()).hexdigest()


def _digest_int64(values) -> str:
    array = np.sort(np.unique(np.asarray(values, dtype=np.int64)))
    return hashlib.sha256(array.astype("<i8", copy=False).tobytes()).hexdigest()


def _pairwise_intersections(parts: dict[str, np.ndarray]) -> dict[str, int]:
    sets = {
        name: set(np.asarray(values).tolist()) for name, values in parts.items()
    }
    return {
        f"{left}__{right}": len(sets[left] & sets[right])
        for left, right in itertools.combinations(parts, 2)
    }


def _shared_fingerprint_union(parts: dict[str, np.ndarray]) -> set:
    sets = [set(np.asarray(values).tolist()) for values in parts.values()]
    return set().union(*(left & right for left, right in itertools.combinations(sets, 2)))


def _split_canonical_known(frame: pd.DataFrame, seed: int) -> dict:
    grouping = frame.copy()
    grouping[GROUP_FINGERPRINT_COLUMN] = grouping[CANONICAL_FINGERPRINT_COLUMN]
    parts = _split_known_by_fingerprint(grouping, seed)
    return {
        name: part.drop(columns=[GROUP_FINGERPRINT_COLUMN])
        for name, part in parts.items()
    }


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(label): int(count)
        for label, count in frame["attack_cat"].value_counts().sort_index().items()
    }


def build_target_isolation_context(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    known_cats=KNOWN_ATTACK_CATS,
    zd_cats=ZERO_DAY_ATTACK_CATS,
) -> dict:
    """Normalize once and build vocabulary-independent canonical identities."""
    class_overlap = sorted(set(known_cats) & set(zd_cats))
    if class_overlap:
        raise ValueError(f"known and OOD class declarations overlap: {class_overlap}")

    train = normalize_labels(train_df.copy())
    test = normalize_labels(test_df.copy())
    validate_declared_classes(train, known_cats, zd_cats)
    validate_declared_classes(test, known_cats, zd_cats)
    if SOURCE_ROLE_COLUMN not in train:
        train[SOURCE_ROLE_COLUMN] = "official_train"
    if SOURCE_ROLE_COLUMN not in test:
        test[SOURCE_ROLE_COLUMN] = "official_test"
    if set(train[SOURCE_ROLE_COLUMN].astype(str)) != {"official_train"}:
        raise ValueError("fold context train rows must have official_train source role")
    if set(test[SOURCE_ROLE_COLUMN].astype(str)) != {"official_test"}:
        raise ValueError("fold context test rows must have official_test source role")
    if SOURCE_FILE_COLUMN not in train:
        train[SOURCE_FILE_COLUMN] = "<in-memory-train>"
    if SOURCE_FILE_COLUMN not in test:
        test[SOURCE_FILE_COLUMN] = "<in-memory-test>"
    train[ROW_ID_COLUMN] = np.arange(len(train), dtype=np.int64)
    test[ROW_ID_COLUMN] = np.arange(len(test), dtype=np.int64)

    schema_reference = train[train["attack_cat"].isin(known_cats)].copy()
    # Schema enumeration needs column names/dtypes only, not a fitted vocabulary.
    for category in CANONICAL_CATEGORY_COLUMNS:
        if category in schema_reference:
            schema_reference[f"{category}_num"] = np.float32(0.0)
    base_feat_cols = sorted(
        column for column in schema_reference
        if column not in MODEL_EXCLUDED_COLUMNS
        and not str(column).startswith("__")
        and pd.api.types.is_numeric_dtype(schema_reference[column])
    )
    schema_reference, feat_cols = engineer_features(
        schema_reference, base_feat_cols.copy()
    )
    feat_cols = list(dict.fromkeys(feat_cols))
    _assert_model_feature_schema(feat_cols)

    train[CANONICAL_FINGERPRINT_COLUMN] = canonical_feature_fingerprints(
        train, base_feat_cols, feat_cols
    ).to_numpy(dtype=np.uint64)
    test[CANONICAL_FINGERPRINT_COLUMN] = canonical_feature_fingerprints(
        test, base_feat_cols, feat_cols
    ).to_numpy(dtype=np.uint64)
    return {
        "train": train,
        "test": test,
        "known_cats": list(known_cats),
        "zd_cats": list(zd_cats),
        "base_feat_cols": list(base_feat_cols),
        "feat_cols": list(feat_cols),
        "feature_schema_sha256": feature_schema_hash(feat_cols),
        "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
    }


def canonical_feature_fingerprints(
    frame: pd.DataFrame,
    base_feat_cols: list[str],
    feat_cols: list[str],
) -> pd.Series:
    """Hash deterministic model features before any fitted preprocessing."""
    prepared = frame.copy()
    non_categorical_base = [
        column for column in base_feat_cols if not column.endswith("_num")
    ]
    prepared, _ = engineer_features(prepared, non_categorical_base.copy())
    canonical = pd.DataFrame(index=prepared.index)
    for column in feat_cols:
        if column.endswith("_num") and column[:-4] in CANONICAL_CATEGORY_COLUMNS:
            raw_column = column[:-4]
            if raw_column not in prepared:
                canonical[column] = "missing:"
            else:
                canonical[column] = prepared[raw_column].map(_normalized_category)
            continue
        if column not in prepared:
            raise ValueError(f"canonical identity is missing model feature: {column}")
        values = pd.to_numeric(prepared[column], errors="coerce").astype(np.float32)
        values = values.mask(values.isna(), np.float32(np.nan))
        canonical[column] = values.mask(values == 0, np.float32(0.0))
    return pd.util.hash_pandas_object(canonical, index=False).astype("uint64")


def _fit_categorical_maps(backbone: pd.DataFrame) -> dict[str, dict[str, int]]:
    return fit_categorical_maps(backbone)


def _prepare_model_frame(
    frame: pd.DataFrame,
    categorical_maps: dict,
    base_feat_cols: list[str],
    feat_cols: list[str],
) -> pd.DataFrame:
    prepared = frame.copy()
    encode_categories(prepared, categorical_maps)
    prepared, _ = engineer_features(prepared, base_feat_cols.copy())
    missing = [column for column in feat_cols if column not in prepared]
    if missing:
        raise ValueError(f"model frame is missing features: {missing}")
    return prepared


def _label_collision_diagnostic(
    labels: np.ndarray,
    fingerprints: np.ndarray,
    known_cats: list[str],
    zd_cats: list[str],
    example_limit: int = 10,
) -> dict:
    table = pd.DataFrame({
        "fingerprint": np.asarray(fingerprints, dtype=np.uint64),
        "label": np.asarray(labels).astype(str),
    })
    label_counts = table.groupby("fingerprint", sort=False)["label"].nunique()
    conflicting = set(label_counts[label_counts > 1].index.astype("uint64").tolist())
    affected = table[table["fingerprint"].isin(conflicting)]
    pair_counts: dict[str, int] = {}
    normal_attack = set()
    known_ood = set()
    examples = []
    known_set, ood_set = set(known_cats), set(zd_cats)
    for fingerprint, group in affected.groupby("fingerprint", sort=True):
        group_labels = sorted(set(group["label"].astype(str)))
        for left, right in itertools.combinations(group_labels, 2):
            key = f"{left} <-> {right}"
            pair_counts[key] = pair_counts.get(key, 0) + 1
        if "Normal" in group_labels and any(label != "Normal" for label in group_labels):
            normal_attack.add(int(fingerprint))
        if set(group_labels) & known_set and set(group_labels) & ood_set:
            known_ood.add(int(fingerprint))
        if len(examples) < example_limit:
            examples.append({
                "fingerprint": str(int(fingerprint)),
                "labels": group_labels,
                "rows": int(len(group)),
            })
    return {
        "fingerprints_with_multiple_labels": int(len(conflicting)),
        "affected_rows": int(len(affected)),
        "label_pair_fingerprint_matrix": dict(sorted(pair_counts.items())),
        "normal_attack_conflicting_fingerprints": int(len(normal_attack)),
        "normal_attack_affected_rows": int(
            affected[affected["fingerprint"].isin(normal_attack)].shape[0]
        ),
        "known_ood_conflicting_fingerprints": int(len(known_ood)),
        "known_ood_affected_rows": int(
            affected[affected["fingerprint"].isin(known_ood)].shape[0]
        ),
        "safe_examples": examples,
    }


def canonical_label_collision_diagnostic(context: dict) -> dict:
    combined = pd.concat([context["train"], context["test"]], ignore_index=True)
    return _label_collision_diagnostic(
        combined["attack_cat"].to_numpy(),
        combined[CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64),
        context["known_cats"],
        context["zd_cats"],
    )


def audit_previous_fuzzers_normal_calibration(context: dict) -> dict:
    """Reproduce the prior seed-42 28-fingerprint/47-row finding for attribution."""
    from .dataset import prepare_official_splits

    legacy = prepare_official_splits(context["train"], context["test"], seed=42)
    normal_idx = int(legacy["label_encoder"].transform(["Normal"])[0])
    target_mask = legacy["y_ood_train"] == "Fuzzers"
    fuzzers_model = set(legacy["fingerprints_ood_train"][target_mask].tolist())
    collision_mask = (legacy["y_calibration"] == normal_idx) & np.asarray([
        value in fuzzers_model for value in legacy["fingerprints_calibration"]
    ])
    row_ids = legacy["row_ids_calibration"][collision_mask]
    collided = context["train"].iloc[row_ids]
    fuzzers = context["train"].query('attack_cat == "Fuzzers"')
    fuzzers_canonical = set(fuzzers[CANONICAL_FINGERPRINT_COLUMN].tolist())
    canonical_match = collided[CANONICAL_FINGERPRINT_COLUMN].isin(fuzzers_canonical)
    raw_columns = [
        column for column in context["train"]
        if column not in {"id", "attack_cat", "label", "label_binary"}
        and not str(column).startswith("__")
    ]
    fuzzers_raw = set(pd.util.hash_pandas_object(fuzzers[raw_columns], index=False))
    collided_raw = pd.util.hash_pandas_object(collided[raw_columns], index=False)
    return {
        "protocol": "previous_P0_shared_backbone_seed42_diagnostic_only",
        "shared_model_input_fingerprints": int(len(set(
            legacy["fingerprints_calibration"][collision_mask].tolist()
        ))),
        "normal_calibration_rows": int(collision_mask.sum()),
        "rows_already_matching_train_fuzzers_before_fitted_transform": int(
            canonical_match.sum()
        ),
        "rows_with_identical_raw_features_except_record_id_and_labels": int(
            sum(value in fuzzers_raw for value in collided_raw)
        ),
        "solely_transformation_induced_rows": int((~canonical_match).sum()),
        "row_ids_sha256": _digest_int64(row_ids),
    }


def _fuzzers_normal_root_cause(context: dict) -> dict:
    combined = pd.concat([context["train"], context["test"]], ignore_index=True)
    selected = combined[combined["attack_cat"].isin(["Fuzzers", "Normal"])].copy()
    grouped = selected.groupby(CANONICAL_FINGERPRINT_COLUMN)["attack_cat"].nunique()
    conflict_fingerprints = set(grouped[grouped > 1].index.astype("uint64").tolist())
    conflicts = selected[selected[CANONICAL_FINGERPRINT_COLUMN].isin(conflict_fingerprints)]
    label_columns = {"attack_cat", "label", "label_binary"}
    raw_columns = [
        column for column in combined.columns
        if column not in label_columns and not str(column).startswith("__")
    ]
    event_columns = [column for column in raw_columns if column != "id"]
    exact_raw_conflicts = 0
    duplicate_event_conflicts = 0
    fully_identical_event_groups = 0
    projection_conflicts = 0
    category_normalization_conflicts = 0
    examples = []
    for fingerprint, group in conflicts.groupby(CANONICAL_FINGERPRINT_COLUMN, sort=True):
        raw_hash = pd.util.hash_pandas_object(group[raw_columns], index=False)
        raw_table = pd.DataFrame({
            "raw_hash": raw_hash.to_numpy(dtype=np.uint64),
            "label": group["attack_cat"].astype(str).to_numpy(),
        })
        is_exact_raw = bool(
            (raw_table.groupby("raw_hash")["label"].nunique() > 1).any()
        )
        if is_exact_raw:
            exact_raw_conflicts += 1
        event_hash = pd.util.hash_pandas_object(group[event_columns], index=False)
        event_table = pd.DataFrame({
            "event_hash": event_hash.to_numpy(dtype=np.uint64),
            "label": group["attack_cat"].astype(str).to_numpy(),
        })
        is_duplicate_event = bool(
            (event_table.groupby("event_hash")["label"].nunique() > 1).any()
        )
        if is_duplicate_event:
            duplicate_event_conflicts += 1
        if event_table["event_hash"].nunique() == 1:
            fully_identical_event_groups += 1
        else:
            projection_conflicts += 1
        category_collapsed = any(
            group[column].astype(str).nunique(dropna=False) > 1
            and group[column].map(_normalized_category).nunique(dropna=False) == 1
            for column in CANONICAL_CATEGORY_COLUMNS
            if column in group
        )
        if category_collapsed:
            category_normalization_conflicts += 1
        if len(examples) < 10:
            varying = [
                column for column in raw_columns
                if group[column].astype(str).nunique(dropna=False) > 1
            ]
            examples.append({
                "fingerprint": str(int(fingerprint)),
                "labels": sorted(set(group["attack_cat"].astype(str))),
                "rows": int(len(group)),
                "raw_record_conflict": is_exact_raw,
                "duplicate_event_ignoring_record_id": is_duplicate_event,
                "varying_nonlabel_fields": varying,
                "label_counts": _class_counts(group),
                "safe_shared_feature_values": {
                    column: str(group.iloc[0][column])
                    for column in ("dur", "proto", "service", "state", "sbytes", "dbytes")
                    if column in group and group[column].nunique(dropna=False) == 1
                },
            })
    return {
        "canonical_fuzzers_normal_conflicting_fingerprints": int(
            len(conflict_fingerprints)
        ),
        "affected_rows": int(len(conflicts)),
        "exact_raw_records_with_conflicting_labels": int(exact_raw_conflicts),
        "duplicate_events_with_conflicting_labels_ignoring_record_id": int(
            duplicate_event_conflicts
        ),
        "groups_with_all_event_fields_identical_except_id_and_labels": int(
            fully_identical_event_groups
        ),
        "feature_projection_conflicts": int(projection_conflicts),
        "categorical_normalization_conflicts": int(
            category_normalization_conflicts
        ),
        "interpretation": (
            "Rows identical apart from the dataset record id are duplicate events with "
            "conflicting labels. Label correctness cannot be adjudicated from these "
            "vectors; conflicts are irreducible for the current deployable feature space."
        ),
        "safe_examples": examples,
    }


def prepare_target_isolated_fold(
    context: dict,
    target_family: str,
    seed: int,
    expected_schema_sha256: str | None = None,
    include_post_transform_label_diagnostic: bool = True,
    include_backdoors_forensics: bool = False,
) -> dict:
    """Build one independent target-purged outer LOFO fold without training."""
    if target_family not in context["zd_cats"]:
        raise ValueError(f"undeclared target family: {target_family}")
    train = context["train"]
    test = context["test"]
    known_cats = context["known_cats"]
    zd_cats = context["zd_cats"]
    base_feat_cols = context["base_feat_cols"]
    feat_cols = context["feat_cols"]
    schema_hash = feature_schema_hash(feat_cols)
    expected_hash = expected_schema_sha256 or P0_SCHEMA_SHA256

    target_train = train[train["attack_cat"] == target_family].copy()
    target_test = test[test["attack_cat"] == target_family].copy()
    if target_test.empty:
        raise ValueError(f"official test contains no rows for target {target_family}")
    target_population = pd.concat([target_train, target_test], ignore_index=True)
    target_canonical = set(
        target_population[CANONICAL_FINGERPRINT_COLUMN].astype("uint64").tolist()
    )

    known_before = train[train["attack_cat"].isin(known_cats)].copy()
    ood_before = train[train["attack_cat"].isin(zd_cats)].copy()
    known_collision = known_before[CANONICAL_FINGERPRINT_COLUMN].isin(target_canonical)
    target_label = ood_before["attack_cat"] == target_family
    surrogate_fp_collision = (
        ~target_label & ood_before[CANONICAL_FINGERPRINT_COLUMN].isin(target_canonical)
    )
    counterfactual_cache = context.setdefault("_counterfactual_split_cache", {})
    if seed not in counterfactual_cache:
        counterfactual_cache[seed] = _split_canonical_known(known_before, seed)
    counterfactual_parts = counterfactual_cache[seed]
    counterfactual_names = {
        "backbone_train": "backbone_eligible",
        "validation": "validation_eligible",
        "meta_known": "meta_known_eligible",
        "calibration": "calibration_eligible",
    }
    known_removed_by_eligible_role = {}
    for source_name, report_name in counterfactual_names.items():
        frame = counterfactual_parts[source_name]
        mask = frame[CANONICAL_FINGERPRINT_COLUMN].isin(target_canonical)
        known_removed_by_eligible_role[report_name] = {
            "rows": int(mask.sum()),
            "fingerprints": int(frame.loc[mask, CANONICAL_FINGERPRINT_COLUMN].nunique()),
        }
    known_remaining = known_before[~known_collision].copy()
    surrogate_remaining = ood_before[~target_label & ~surrogate_fp_collision].copy()
    if known_remaining.empty or surrogate_remaining.empty:
        raise AssertionError(f"target purge emptied a required population: {target_family}")

    known_parts = _split_canonical_known(known_remaining, seed)
    role_frames = {
        "train": known_parts["backbone_train"],
        "val": known_parts["validation"],
        "meta_known": known_parts["meta_known"],
        "calibration": known_parts["calibration"],
        "surrogate_ood": surrogate_remaining,
    }
    categorical_maps = _fit_categorical_maps(role_frames["train"])
    model_frames = {
        name: _prepare_model_frame(
            frame, categorical_maps, base_feat_cols, feat_cols
        )
        for name, frame in role_frames.items()
    }
    target_model_frame = _prepare_model_frame(
        target_population, categorical_maps, base_feat_cols, feat_cols
    )
    test_known = test[test["attack_cat"].isin(known_cats)].copy()
    test_known_model_frame = _prepare_model_frame(
        test_known, categorical_maps, base_feat_cols, feat_cols
    )

    scaler = RobustScaler().fit(
        model_frames["train"][feat_cols].to_numpy(dtype=np.float64)
    )

    def transform(frame: pd.DataFrame) -> np.ndarray:
        return transform_stages(frame[feat_cols].to_numpy(dtype=np.float64), scaler)[1]

    transformed = {name: transform(frame) for name, frame in model_frames.items()}
    transformed_target = transform(target_model_frame)
    transformed_test_known = transform(test_known_model_frame)
    model_fingerprints = {
        name: model_input_fingerprints(values).to_numpy(dtype=np.uint64)
        for name, values in transformed.items()
    }
    target_model_fingerprints = model_input_fingerprints(
        transformed_target
    ).to_numpy(dtype=np.uint64)
    target_model_set = set(target_model_fingerprints.tolist())
    target_canonical_intersections = {
        name: len(target_canonical & set(
            frame[CANONICAL_FINGERPRINT_COLUMN].astype("uint64").tolist()
        ))
        for name, frame in role_frames.items()
    }
    target_model_intersections = {
        name: len(target_model_set & set(values.tolist()))
        for name, values in model_fingerprints.items()
    }
    known_role_names = ("train", "val", "meta_known", "calibration")
    known_row_intersections = _pairwise_intersections({
        name: role_frames[name][ROW_ID_COLUMN].to_numpy(dtype=np.int64)
        for name in known_role_names
    })
    known_canonical_intersections = _pairwise_intersections({
        name: role_frames[name][CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)
        for name in known_role_names
    })
    known_model_intersections = _pairwise_intersections({
        name: model_fingerprints[name] for name in known_role_names
    })
    all_row_intersections = _pairwise_intersections({
        name: frame[ROW_ID_COLUMN].to_numpy(dtype=np.int64)
        for name, frame in role_frames.items()
    })
    all_canonical_intersections = _pairwise_intersections({
        name: frame[CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)
        for name, frame in role_frames.items()
    })
    all_model_intersections = _pairwise_intersections(model_fingerprints)
    audit_frames = {**role_frames, "target": target_population}
    audit_models = {**model_frames, "target": target_model_frame}
    stages = {
        "A_canonical": {
            name: frame[CANONICAL_FINGERPRINT_COLUMN].to_numpy(np.uint64)
            for name, frame in audit_frames.items()
        },
        "B_encoded_float64": {}, "C_scaled_float64": {},
        "D_model_float32": {**model_fingerprints, "target": target_model_fingerprints},
    }
    magnitude = {}
    for name, frame in audit_models.items():
        encoded = frame[feat_cols].to_numpy(np.float64)
        scaled, final = transform_stages(encoded, scaler)
        stages["B_encoded_float64"][name] = matrix_fingerprints(encoded)
        stages["C_scaled_float64"][name] = matrix_fingerprints(scaled)
        magnitude[name] = {"max_absolute_model_input": float(np.abs(final).max()),
                           "coordinates_outside_prior_clip_range": int((np.abs(final) > 10).sum())}
    diagnostics = stage_audit(stages, {
        name: frame["attack_cat"].astype(str).to_numpy() for name, frame in audit_frames.items()
    })
    scientific_gate = collision_gate(
        diagnostics["D_model_float32"]["taxonomy"], target_canonical_intersections,
    )

    le = LabelEncoder().fit(list(known_cats))
    class_group_counts = {
        name: {
            str(label): int(count)
            for label, count in role_frames[name].groupby("attack_cat")[
                CANONICAL_FINGERPRINT_COLUMN
            ].nunique().items()
        }
        for name in known_role_names
    }
    normal_calibration = role_frames["calibration"].query('attack_cat == "Normal"')
    normal_calibration_groups = int(normal_calibration[CANONICAL_FINGERPRINT_COLUMN].nunique())
    fit_roles_usable = all(
        not role_frames[name].empty for name in role_frames
    ) and all(
        set(role_frames[name]["attack_cat"]) == set(known_cats)
        for name in known_role_names
    ) and all(
        count >= 2 for counts in class_group_counts.values() for count in counts.values()
    ) and normal_calibration_groups >= 20
    official_test_excluded = all(
        set(frame[SOURCE_ROLE_COLUMN].astype(str)) == {"official_train"}
        for frame in role_frames.values()
    )
    target_label_absent = all(
        target_family not in set(frame["attack_cat"].astype(str))
        for frame in role_frames.values()
    )
    invariants = {
        "target_label_absent_from_fit": target_label_absent,
        "target_canonical_fingerprints_absent_from_fit": not any(
            target_canonical_intersections.values()
        ),
        "target_model_input_fingerprints_absent_from_fit": not any(
            target_model_intersections.values()
        ),
        "surrogate_target_canonical_fingerprints_absent": (
            target_canonical_intersections["surrogate_ood"] == 0
        ),
        "surrogate_target_model_fingerprints_absent": (
            target_model_intersections["surrogate_ood"] == 0
        ),
        "schema_frozen": (
            schema_hash == expected_hash == P0_SCHEMA_SHA256
            and len(feat_cols) == P0_FEATURE_COUNT
            and tuple(feat_cols) == P0_FEATURE_SCHEMA
        ),
        "known_roles_row_disjoint": not any(known_row_intersections.values()),
        "known_roles_canonical_fingerprint_disjoint": not any(
            known_canonical_intersections.values()
        ),
        "known_roles_model_input_fingerprint_disjoint": not any(
            known_model_intersections.values()
        ),
        "all_fit_roles_row_disjoint": not any(all_row_intersections.values()),
        "all_fit_roles_canonical_fingerprint_disjoint": not any(
            all_canonical_intersections.values()
        ),
        "all_fit_roles_model_input_fingerprint_disjoint": not any(
            all_model_intersections.values()
        ),
        "official_test_excluded_from_fit": official_test_excluded,
        "calibration_has_no_target_equivalent_rows": (
            target_canonical_intersections["calibration"] == 0
            and target_model_intersections["calibration"] == 0
        ),
        "required_roles_usable": fit_roles_usable,
    }
    gate_passed = all(invariants.values()) and scientific_gate["gate"] == "PASS"

    all_train_exposure_canonical = set()
    for frame in role_frames.values():
        all_train_exposure_canonical.update(
            frame[CANONICAL_FINGERPRINT_COLUMN].astype("uint64").tolist()
        )
    test_known_canonical = test_known[CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)
    target_test_canonical = target_test[CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)

    report = {
        "protocol": "target_isolated_outer_lofo_v1",
        "target_family": target_family,
        "seed": int(seed),
        "gate": "PASS" if gate_passed else "FAIL",
        "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
        "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
        "categorical_encoding_version": CATEGORY_ENCODING_VERSION,
        "unicode_data_version": REQUIRED_UNICODE_DATA_VERSION,
        "ambiguity_exclusion_active": False,
        "stage_collision_audit": diagnostics,
        "collision_taxonomy": diagnostics["D_model_float32"]["taxonomy"],
        "scientific_gate": scientific_gate,
        "failed_invariants": [name for name, value in invariants.items() if not value],
        "model_input_magnitude": magnitude,
        "feature_count": len(feat_cols),
        "feature_schema": list(feat_cols),
        "feature_schema_sha256": schema_hash,
        "required_p0_schema_sha256": P0_SCHEMA_SHA256,
        "test_identity_policy": (
            "Only predeclared target-family input identities from official train "
            "and official test control contamination exclusion. No test statistics, "
            "metrics or non-target test rows control fitting or model selection. "
            "This is a test-aware deduplication protocol, not an untouched test set."
        ),
        "target": {
            "target_train_rows": int(len(target_train)),
            "target_test_rows": int(len(target_test)),
            "canonical_target_train_fingerprints": int(target_train[CANONICAL_FINGERPRINT_COLUMN].nunique()),
            "canonical_target_test_fingerprints": int(target_test[CANONICAL_FINGERPRINT_COLUMN].nunique()),
            "canonical_target_fingerprints": int(len(target_canonical)),
            "model_input_target_fingerprints": int(len(target_model_set)),
        },
        "known_before_purge": {
            "rows": int(len(known_before)),
            "fingerprints": int(known_before[CANONICAL_FINGERPRINT_COLUMN].nunique()),
            "per_class_counts": _class_counts(known_before),
        },
        "known_removed": {
            "rows": int(known_collision.sum()),
            "fingerprints": int(
                known_before.loc[known_collision, CANONICAL_FINGERPRINT_COLUMN].nunique()
            ),
            "counterfactual_pre_purge_role_allocation": (
                known_removed_by_eligible_role
            ),
            "allocation_note": (
                "Diagnostic only; these unpurged assignments never enter fitting. "
                "Actual grouped splits are rebuilt from purged known rows."
            ),
        },
        "known_after_purge": {
            "rows": int(len(known_remaining)),
            "fingerprints": int(known_remaining[CANONICAL_FINGERPRINT_COLUMN].nunique()),
            "per_class_counts": _class_counts(known_remaining),
        },
        "split_counts": {
            name: int(len(frame)) for name, frame in role_frames.items()
        },
        "split_per_class_counts": {
            name: _class_counts(frame) for name, frame in role_frames.items()
        },
        "role_usability": {
            "policy": (
                "All five known classes in every known role, at least two canonical "
                "groups per class/role; at least 20 Normal calibration groups for "
                "the existing 5% calibration budget. This is a minimum support check, "
                "not a precision or power guarantee."
            ),
            "per_class_canonical_groups": class_group_counts,
            "normal_calibration_rows": len(normal_calibration),
            "normal_calibration_canonical_groups": normal_calibration_groups,
        },
        "surrogate_ood": {
            "rows_before": int(len(ood_before)),
            "target_label_rows_removed": int(target_label.sum()),
            "non_target_fingerprint_rows_removed": int(surrogate_fp_collision.sum()),
            "rows_after": int(len(surrogate_remaining)),
            "families_remaining": sorted(set(
                surrogate_remaining["attack_cat"].astype(str)
            )),
        },
        "pre_transform_target_intersections": target_canonical_intersections,
        "post_transform_target_intersections": target_model_intersections,
        "post_transform_target_shared_fingerprints_unique": len(
            target_model_set & set().union(*(
                set(values.tolist()) for values in model_fingerprints.values()
            ))
        ),
        "transformation_induced_target_collisions": {
            name: int(target_model_intersections[name])
            for name in role_frames
            if target_canonical_intersections[name] == 0
            and target_model_intersections[name] > 0
        },
        "known_role_row_intersections": known_row_intersections,
        "known_role_canonical_intersections": known_canonical_intersections,
        "known_role_model_input_intersections": known_model_intersections,
        "all_fit_role_row_intersections": all_row_intersections,
        "all_fit_role_canonical_intersections": all_canonical_intersections,
        "all_fit_role_model_input_intersections": all_model_intersections,
        "known_role_shared_model_fingerprints_unique": len(
            _shared_fingerprint_union({
                name: model_fingerprints[name] for name in known_role_names
            })
        ),
        "all_fit_role_shared_canonical_fingerprints_unique": len(
            _shared_fingerprint_union({
                name: frame[CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)
                for name, frame in role_frames.items()
            })
        ),
        "transformation_collision_diagnostics": {"see": "stage_collision_audit"},
        "vocabulary_origin": "target-purged backbone_train only",
        "categorical_vocabulary_sizes": {
            name: int(len(mapping)) for name, mapping in categorical_maps.items()
        },
        "scaler_fit_evidence": {
            "row_count": int(len(role_frames["train"])),
            "row_ids_sha256": _digest_int64(
                role_frames["train"][ROW_ID_COLUMN].to_numpy(dtype=np.int64)
            ),
            "canonical_fingerprint_count": int(
                role_frames["train"][CANONICAL_FINGERPRINT_COLUMN].nunique()
            ),
            "canonical_fingerprint_set_sha256": _digest_uint64(
                role_frames["train"][CANONICAL_FINGERPRINT_COLUMN].to_numpy(dtype=np.uint64)
            ),
        },
        "unseen_canonical_test": {
            "known_rows": int(sum(
                value not in all_train_exposure_canonical
                for value in test_known_canonical
            )),
            "target_rows": int(sum(
                value not in all_train_exposure_canonical
                for value in target_test_canonical
            )),
        },
        "invariants": invariants,
    }

    if include_post_transform_label_diagnostic:
        combined = pd.concat([train, test], ignore_index=True)
        combined_model = _prepare_model_frame(
            combined, categorical_maps, base_feat_cols, feat_cols
        )
        combined_fingerprints = model_input_fingerprints(
            transform(combined_model)
        ).to_numpy(dtype=np.uint64)
        report["post_transform_label_collisions"] = _label_collision_diagnostic(
            combined["attack_cat"].to_numpy(),
            combined_fingerprints,
            known_cats,
            zd_cats,
        )

    if include_backdoors_forensics and target_family == "Backdoors":
        from .backdoors_forensics import backdoors_forensics

        report["backdoors_forensics"] = backdoors_forensics(
            role_frames, target_population, model_frames, target_model_frame, scaler,
            transformed, transformed_target, base_feat_cols, feat_cols, seed,
        )

    return {
        "report": report,
        "gate_passed": gate_passed,
        "feat_cols": feat_cols,
        "feature_schema_sha256": schema_hash,
        "categorical_maps": categorical_maps,
        "model_input_identity_version": MODEL_INPUT_IDENTITY_VERSION,
        "scaler": scaler,
        "label_encoder": le,
        "X_train": transformed["train"],
        "y_train": le.transform(role_frames["train"]["attack_cat"]),
        "X_val": transformed["val"],
        "y_val": le.transform(role_frames["val"]["attack_cat"]),
        "X_meta_known": transformed["meta_known"],
        "y_meta_known": le.transform(role_frames["meta_known"]["attack_cat"]),
        "X_calibration": transformed["calibration"],
        "y_calibration": le.transform(role_frames["calibration"]["attack_cat"]),
        "X_ood_train": transformed["surrogate_ood"],
        "y_ood_train": role_frames["surrogate_ood"]["attack_cat"].to_numpy(),
        "X_test": transformed_test_known,
        "y_test": le.transform(test_known["attack_cat"]),
        "X_ood_test": transform(_prepare_model_frame(
            target_test, categorical_maps, base_feat_cols, feat_cols
        )),
        "y_ood_test": target_test["attack_cat"].to_numpy(),
        "target_family": target_family,
        "target_canonical_fingerprints": np.asarray(
            sorted(target_canonical), dtype=np.uint64
        ),
        "target_model_input_fingerprints": target_model_fingerprints,
        "X_target_identity": transformed_target,
        "fit_role_evidence": {
            name: {
                "row_ids": frame[ROW_ID_COLUMN].to_numpy(dtype=np.int64),
                "source_roles": frame[SOURCE_ROLE_COLUMN].astype(str).to_numpy(),
                "labels": frame["attack_cat"].astype(str).to_numpy(),
                "canonical_fingerprints": frame[CANONICAL_FINGERPRINT_COLUMN].to_numpy(
                    dtype=np.uint64
                ),
                "model_input_fingerprints": model_fingerprints[name],
            }
            for name, frame in role_frames.items()
        },
        "scaler_fit_row_ids": role_frames["train"][ROW_ID_COLUMN].to_numpy(
            dtype=np.int64
        ),
        "scaler_fit_canonical_fingerprints": role_frames["train"][
            CANONICAL_FINGERPRINT_COLUMN
        ].to_numpy(dtype=np.uint64),
        "test_known_unseen_canonical_mask": np.asarray([
            value not in all_train_exposure_canonical
            for value in test_known_canonical
        ], dtype=bool),
        "target_test_unseen_canonical_mask": np.asarray([
            value not in all_train_exposure_canonical
            for value in target_test_canonical
        ], dtype=bool),
    }


def assert_target_fold_retraining_gate(fold: dict) -> dict:
    """Fail closed unless every target-specific data invariant passes."""
    report = fold["report"]
    failed = [
        name for name in REQUIRED_INVARIANTS
        if report.get("invariants", {}).get(name) is not True
    ]
    if report.get("canonical_identity_version") != CANONICAL_IDENTITY_VERSION:
        failed.append("canonical_identity_contract_mismatch")
    if report.get("model_input_identity_version") != MODEL_INPUT_IDENTITY_VERSION:
        failed.append("model_input_contract_mismatch")
    evidence = fold.get("fit_role_evidence", {})
    if set(evidence) != set(FIT_ROLE_NAMES):
        failed.append("missing_fit_role_evidence")
    else:
        target_canonical = set(fold.get("target_canonical_fingerprints", []))
        target_values = fold.get("X_target_identity", np.empty((0, 0)))
        if not target_canonical or len(target_values) == 0:
            failed.append("missing_target_identity_evidence")
        target_model = set(model_input_fingerprints(target_values).tolist())
        canonical_parts, model_parts, row_parts, label_parts = {}, {}, {}, {}
        for name, role in evidence.items():
            key = "ood_train" if name == "surrogate_ood" else name
            values = np.asarray(fold[f"X_{key}"])
            lengths = {len(values), *(len(value) for value in role.values())}
            if len(lengths) != 1 or not len(values):
                failed.append(f"unaligned_or_empty_role/{name}")
            canonical_parts[name] = role["canonical_fingerprints"]
            row_parts[name] = role["row_ids"]
            model_parts[name] = model_input_fingerprints(values).to_numpy()
            if report["target_family"] in set(role["labels"]):
                failed.append(f"target_label/{name}")
            if set(role["source_roles"]) != {"official_train"}:
                failed.append(f"non_training_source/{name}")
            labels = np.asarray(role["labels"]).astype(str)
            label_parts[name] = labels
            if name != "surrogate_ood":
                if set(labels) != set(KNOWN_ATTACK_CATS):
                    failed.append(f"missing_or_undeclared_known_classes/{name}")
                for label in KNOWN_ATTACK_CATS:
                    if len(set(np.asarray(canonical_parts[name])[labels == label])) < 2:
                        failed.append(f"insufficient_class_groups/{name}/{label}")
                if name == "calibration" and len(set(
                    np.asarray(canonical_parts[name])[labels == "Normal"]
                )) < 20:
                    failed.append("insufficient_normal_calibration_groups")
            elif not set(labels).issubset(set(ZERO_DAY_ATTACK_CATS) - {
                report["target_family"]
            }):
                failed.append("undeclared_surrogate_family")
            if target_canonical & set(canonical_parts[name]):
                failed.append(f"target_canonical_collision/{name}")
            if target_model & set(model_parts[name]):
                failed.append(f"target_model_collision/{name}")
            if not np.array_equal(model_parts[name], role["model_input_fingerprints"]):
                failed.append(f"stale_model_input_evidence/{name}")
        if not failed:
            live_taxonomy = collision_taxonomy(
                {**model_parts, "target": np.asarray(
                    model_input_fingerprints(target_values), dtype=np.uint64
                )},
                {**label_parts, "target": np.asarray(
                    [report["target_family"]] * len(target_values)
                )},
            )
            live_target_canonical = {
                name: len(target_canonical & set(np.asarray(values, dtype=np.uint64)))
                for name, values in canonical_parts.items()
            }
            live_scientific_gate = collision_gate(live_taxonomy, live_target_canonical)
            for reason in live_scientific_gate["fail_reasons"]:
                failed.append(
                    "scientific_collision/" + reason["reason"] + "/" +
                    str(reason.get("pair", reason.get("role", "all")))
                )
        for kind, parts in (
            ("row", row_parts), ("canonical", canonical_parts), ("model", model_parts)
        ):
            if any(_pairwise_intersections(parts).values()):
                failed.append(f"all_fit_roles_{kind}_collision")
        if not np.array_equal(
            np.sort(fold.get("scaler_fit_row_ids", [])),
            np.sort(row_parts["train"]),
        ):
            failed.append("scaler_fit_row_id_mismatch")
        if not np.array_equal(
            np.sort(fold.get("scaler_fit_canonical_fingerprints", [])),
            np.sort(canonical_parts["train"]),
        ):
            failed.append("scaler_fit_canonical_evidence_mismatch")
    if (
        feature_schema_hash(fold.get("feat_cols", [])) != P0_SCHEMA_SHA256
        or len(fold.get("feat_cols", [])) != P0_FEATURE_COUNT
        or tuple(fold.get("feat_cols", [])) != P0_FEATURE_SCHEMA
    ):
        failed.append("p0_schema_mismatch")
    if failed:
        raise AssertionError(
            f"target-isolated retraining blocked for {report['target_family']} "
            f"seed={report['seed']}; failed invariants={failed}; "
            "inspect transformation-induced collisions before training"
        )
    return report["invariants"]


def build_data_quality_report(context: dict) -> dict:
    return {
        "version": 1,
        "canonical_identity_version": CANONICAL_IDENTITY_VERSION,
        "canonical_identity_contract": {
            "ordered_fields": list(context["feat_cols"]),
            "numeric_precision": "float32; signed zero canonicalized; NaN retained",
            "categorical_fields": list(CANONICAL_CATEGORY_COLUMNS),
            "missing_category": "missing:; non-missing strings use value: prefix",
            "numeric_missing_policy": (
                "Canonical base numeric NaN remains distinct from zero; deterministic "
                "engineered non-finite values follow existing engineer_features (zero)."
            ),
            "hash": "pandas.hash_pandas_object(index=False), uint64; no row indices",
            "included": (
                "ordered deterministic engineered model features; float32 numeric "
                "values; NFKC/strip/casefold categorical strings"
            ),
            "excluded": (
                "labels, metadata, raw identifiers, fitted scaler, fitted category "
                "indices, clipping, learned representations"
            ),
        },
        "canonical_label_collisions": canonical_label_collision_diagnostic(context),
        "fuzzers_normal_root_cause": _fuzzers_normal_root_cause(context),
        "ambiguity_policy_option": ambiguity_policy_report(context),
    }


def json_sha256(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
