"""UNSW-NB15 loading, feature engineering, RobustScaler fitting and weighted DataLoader creation."""

import hashlib
import json
import os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

KNOWN_ATTACK_CATS  = ['Normal','DoS','Exploits','Reconnaissance','Generic']
ZERO_DAY_ATTACK_CATS = ['Fuzzers','Analysis','Backdoors','Shellcode','Worms']

DEFAULT_TRAIN_FILE = 'UNSW_NB15_training-set.csv'
DEFAULT_TEST_FILE = 'UNSW_NB15_testing-set.csv'
SOURCE_ROLE_COLUMN = '__source_role'
SOURCE_FILE_COLUMN = '__source_file'
FINGERPRINT_COLUMN = '__feature_fingerprint'
ROW_ID_COLUMN = '__official_train_row_id'
METADATA_COLUMN_PREFIX = '__'

MODEL_EXCLUDED_COLUMNS = {
    'attack_cat', 'label', 'label_binary', 'srcip', 'dstip',
    'sport', 'dsport', 'stime', 'ltime', 'id', 'proto', 'service', 'state', 'y',
}

UNSW_RAW_COLUMNS = [
    'srcip','sport','dstip','dsport','proto',
    'state','dur','sbytes','dbytes','sttl',
    'dttl','sloss','dloss','service','sload',
    'dload','spkts','dpkts','swin','dwin',
    'stcpb','dtcpb','smeansz','dmeansz','trans_depth',
    'res_bdy_len','sjit','djit','stime','ltime',
    'sintpkt','dintpkt','tcprtt','synack','ackdat',
    'is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd','is_ftp_login','ct_ftp_cmd',
    'ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
    'ct_dst_sport_ltm','ct_dst_src_ltm','attack_cat','label',
]

SKIP_FILES = {
    'NUSW-NB15_features.csv','UNSW-NB15_features.csv',
    'UNSW-NB15_LIST_EVENTS.csv','UNSW-NB15_GT.csv',
}

# ═══════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════
def _coerce_file_list(files):
    if files is None:
        return []
    if isinstance(files, (str, os.PathLike)):
        return [item.strip() for item in str(files).split(',') if item.strip()]
    return [str(item) for item in files]


def _resolve_role_files(data_dir, files, default_name=None):
    data_dir = Path(data_dir)
    requested = _coerce_file_list(files)
    if not requested and default_name:
        requested = [default_name]
    resolved = []
    for item in requested:
        path = Path(item)
        if not path.is_absolute():
            path = data_dir / path
        if not path.is_file():
            raise FileNotFoundError(f'UNSW-NB15 data file not found: {path}')
        resolved.append(path.resolve())
    return resolved


def _find_unsw_csvs(data_dir):
    """Compatibility helper returning only the official training file."""
    return [str(path) for path in _resolve_role_files(data_dir, None, DEFAULT_TRAIN_FILE)]


def load_unsw_csvs(data_dir, files=None, source_role='official_train'):
    """Load an explicit UNSW-NB15 file role without directory-wide auto-globbing."""
    if not os.path.exists(data_dir) and os.path.exists('/kaggle/input'):
        data_dir = '/kaggle/input'
    default_name = DEFAULT_TRAIN_FILE if files is None else None
    csv_files = _resolve_role_files(data_dir, files, default_name)
    if not csv_files:
        raise ValueError('files must contain at least one explicit UNSW-NB15 CSV path')
    dfs = []
    for path in csv_files:
        name = path.name
        try:
            probe = pd.read_csv(path, nrows=3, low_memory=False,
                                encoding='utf-8', on_bad_lines='skip')
            has_hdr = any('attack_cat' in str(c).lower() for c in probe.columns)
            if has_hdr:
                df = pd.read_csv(path, low_memory=False,
                                 encoding='utf-8', on_bad_lines='skip')
                df.columns = [str(c).strip().lower().replace(' ','_') for c in df.columns]
            else:
                nc = probe.shape[1]
                cols = UNSW_RAW_COLUMNS if nc == 49 else \
                       [c for c in UNSW_RAW_COLUMNS if c not in ('stime','ltime')] if nc == 47 \
                       else [f'col_{i}' for i in range(nc)]
                df = pd.read_csv(path, header=None, names=cols,
                                 low_memory=False, encoding='latin-1', on_bad_lines='skip')
                df.columns = [str(c).strip().lower() for c in df.columns]
            df[SOURCE_ROLE_COLUMN] = source_role
            df[SOURCE_FILE_COLUMN] = str(path)
            print(f'  [OK] {name:45s} {len(df):>8,} rows  role={source_role}')
            dfs.append(df)
        except Exception as e:
            print(f'  [ERR] {name}: {e}')
    df = pd.concat(dfs, ignore_index=True)
    print(f'\n  Total: {len(df):,} rows')
    return df


def load_official_unsw_splits(data_dir, train_files=None, calibration_files=None,
                              test_files=None):
    """Load official train/calibration/test roles as separate dataframes."""
    train_paths = _resolve_role_files(data_dir, train_files, DEFAULT_TRAIN_FILE)
    test_paths = _resolve_role_files(data_dir, test_files, DEFAULT_TEST_FILE)
    calibration_paths = _resolve_role_files(data_dir, calibration_files)
    train_set = set(train_paths)
    test_set = set(test_paths)
    calibration_set = set(calibration_paths)
    overlaps = {
        'train/test': train_set & test_set,
        'train/calibration': train_set & calibration_set,
        'calibration/test': calibration_set & test_set,
    }
    for roles, paths in overlaps.items():
        if paths:
            raise ValueError(f'data role overlap for {roles}: {sorted(map(str, paths))}')

    out = {
        'train': load_unsw_csvs(data_dir, train_paths, source_role='official_train'),
        'test': load_unsw_csvs(data_dir, test_paths, source_role='official_test'),
        'calibration': None,
    }
    if calibration_paths:
        out['calibration'] = load_unsw_csvs(
            data_dir, calibration_paths, source_role='official_calibration'
        )
    return out


def normalize_labels(df):
    ac_col = next((c for c in df.columns if 'attack_cat' in c.lower()), None)
    lb_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if ac_col is None:
        ac_col = df.columns[-2]
    df['attack_cat'] = df[ac_col].astype(str).str.strip()
    df['attack_cat'] = df['attack_cat'].replace(['nan','NaN','',' ','None','-'], 'Normal')
    cat_map = {
        'normal':'Normal','dos':'DoS','exploits':'Exploits','exploit':'Exploits',
        'reconnaissance':'Reconnaissance','generic':'Generic','fuzzers':'Fuzzers',
        'fuzzer':'Fuzzers','analysis':'Analysis','backdoor':'Backdoors',
        'backdoors':'Backdoors','shellcode':'Shellcode','worms':'Worms','worm':'Worms',
    }
    df['attack_cat'] = df['attack_cat'].str.lower().map(
        lambda x: cat_map.get(x.strip(), x.capitalize()))
    if lb_col and lb_col != ac_col:
        df['label_binary'] = pd.to_numeric(df[lb_col], errors='coerce').fillna(0).astype(int)
    else:
        df['label_binary'] = (df['attack_cat'] != 'Normal').astype(int)
    return df


def _encode_categorical_features(df, categorical_maps=None):
    categorical_maps = categorical_maps or {}
    fitted_maps = {}
    for cat in ['proto','service','state']:
        if cat not in df.columns:
            continue
        values = df[cat].astype(str).fillna('unk')
        if cat in categorical_maps:
            mapping = categorical_maps[cat]
        else:
            classes = sorted(values.unique().tolist())
            if 'unk' not in classes:
                classes.append('unk')
            mapping = {v: i for i, v in enumerate(classes)}
        df[f'{cat}_num'] = values.map(lambda x: mapping.get(x, mapping.get('unk', -1))).astype(np.float32)
        fitted_maps[cat] = mapping
    return fitted_maps


def _get_numeric_features(df, categorical_maps=None):
    fitted_maps = _encode_categorical_features(df, categorical_maps)
    cols = []
    for c in sorted(df.columns):
        if c in MODEL_EXCLUDED_COLUMNS or str(c).startswith(METADATA_COLUMN_PREFIX):
            continue
        if c.endswith('_num'):
            cols.append(c); continue
        try:
            if np.issubdtype(df[c].dtype, np.number):
                cols.append(c)
        except: pass
    _assert_model_feature_schema(cols)
    return cols, fitted_maps


def _assert_model_feature_schema(feat_cols):
    """Fail closed if labels, metadata, or duplicate columns reach a model boundary."""
    columns = [str(column) for column in feat_cols]
    forbidden = [
        column for column in columns
        if column.startswith(METADATA_COLUMN_PREFIX)
        or column in MODEL_EXCLUDED_COLUMNS
    ]
    if forbidden:
        raise AssertionError(f'forbidden model features: {forbidden}')
    if len(columns) != len(set(columns)):
        raise AssertionError('model feature schema contains duplicate columns')
    return True


def feature_schema_hash(feat_cols):
    """Hash the ordered model schema for cross-seed and artifact checks."""
    _assert_model_feature_schema(feat_cols)
    payload = json.dumps(list(feat_cols), ensure_ascii=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def engineer_features(df, feat_cols):
    eps = 1e-8
    new = []
    def add(name, vals):
        arr = np.asarray(vals, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, 0.)
        df[name] = arr; new.append(name)

    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        tb = df['sbytes'] + df['dbytes'] + eps
        add('bytes_ratio', df['sbytes']/tb)
        add('log_total_bytes', np.log1p(tb))
        add('log_sbytes', np.log1p(df['sbytes'].clip(lower=0)))
        add('log_dbytes', np.log1p(df['dbytes'].clip(lower=0)))

    if 'spkts' in df.columns and 'dpkts' in df.columns:
        tp = df['spkts'] + df['dpkts'] + eps
        add('pkts_ratio', df['spkts']/tp)
        add('log_total_pkts', np.log1p(tp))

    if 'sbytes' in df.columns and 'dur' in df.columns:
        dur_s = df['dur'].clip(lower=1e-6)
        add('src_bps',     df['sbytes']/dur_s)
        add('log_src_bps', np.log1p(df['sbytes']/dur_s))
        add('pkt_rate',    (df.get('spkts', pd.Series(0,index=df.index))+eps)/dur_s)

    if 'sload' in df.columns and 'dload' in df.columns:
        tl = df['sload'] + df['dload'] + eps
        add('load_asym', (df['sload']-df['dload']).abs()/tl)
        add('log_sload', np.log1p(df['sload'].clip(lower=0)))

    if 'sttl' in df.columns and 'dttl' in df.columns:
        add('ttl_diff', (df['sttl']-df['dttl']).abs())
        add('ttl_sum',  df['sttl']+df['dttl'])

    if 'sloss' in df.columns and 'spkts' in df.columns:
        add('loss_rate_src', df['sloss']/(df['spkts']+eps))
    if 'dloss' in df.columns and 'dpkts' in df.columns:
        add('loss_rate_dst', df['dloss']/(df['dpkts']+eps))

    if 'sjit' in df.columns and 'djit' in df.columns:
        add('jit_ratio', df['sjit']/(df['djit']+eps))
        add('log_sjit',  np.log1p(df['sjit'].clip(lower=0)))

    if 'synack' in df.columns and 'ackdat' in df.columns:
        add('handshake_ratio', df['synack']/(df['ackdat']+eps))
        add('incomplete_tcp',  ((df['synack']>0)&(df['ackdat']==0)).astype(float))

    if 'sintpkt' in df.columns and 'dintpkt' in df.columns:
        add('intpkt_ratio', df['sintpkt']/(df['dintpkt']+eps))

    return df, feat_cols + new


def clean_df(df):
    df = df.replace([np.inf,-np.inf], np.nan)
    n = len(df)
    df = df.drop_duplicates()
    print(f'  Removed {n-len(df):,} duplicates')
    return df


def validate_declared_classes(df, known_cats=KNOWN_ATTACK_CATS,
                              zd_cats=ZERO_DAY_ATTACK_CATS):
    """Fail closed when a dataset contains an undeclared attack family."""
    declared = set(known_cats) | set(zd_cats)
    present = set(df['attack_cat'].astype(str).unique())
    unknown = sorted(present - declared)
    if unknown:
        raise ValueError(
            'Undeclared attack classes detected: '
            f'{unknown}. Add every class explicitly to KNOWN_ATTACK_CATS or '
            'ZERO_DAY_ATTACK_CATS; class roles are never inferred from row counts.'
        )


def feature_fingerprints(df, feature_columns=None):
    """Hash canonical pre-scaled model features, never labels or raw identifiers."""
    columns = list(feature_columns or sorted(
        column for column in df.columns
        if column not in MODEL_EXCLUDED_COLUMNS
        and not str(column).startswith(METADATA_COLUMN_PREFIX)
    ))
    if not columns:
        raise ValueError('cannot fingerprint rows without feature columns')
    missing = [column for column in columns if column not in df]
    if missing:
        raise ValueError(f'cannot fingerprint missing model features: {missing}')
    canonical = df[columns].copy()
    for column in columns:
        if pd.api.types.is_numeric_dtype(canonical[column]):
            values = pd.to_numeric(canonical[column], errors='coerce').astype(np.float32)
            canonical[column] = values.mask(values == 0, np.float32(0.0))
        else:
            # Match categorical preprocessing exactly: no case/whitespace rewrite.
            canonical[column] = canonical[column].astype(str).fillna('unk')
    return pd.util.hash_pandas_object(canonical, index=False).astype('uint64')


def model_input_fingerprints(values):
    """Hash the exact finite float32 vectors presented to the neural network."""
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError('model input fingerprinting requires a 2D matrix')
    if not np.isfinite(matrix).all():
        raise ValueError('model input fingerprinting requires finite values')
    matrix = matrix.copy()
    matrix[matrix == 0] = np.float32(0.0)  # canonicalize negative zero
    return pd.util.hash_pandas_object(
        pd.DataFrame(matrix, columns=range(matrix.shape[1])), index=False
    ).astype('uint64')


def _assert_disjoint_fingerprints(parts):
    names = list(parts)
    for index, left_name in enumerate(names):
        left = set(parts[left_name][FINGERPRINT_COLUMN].astype('uint64'))
        for right_name in names[index + 1:]:
            right = set(parts[right_name][FINGERPRINT_COLUMN].astype('uint64'))
            overlap = left & right
            if overlap:
                raise AssertionError(
                    f'fingerprint leakage between {left_name} and {right_name}: '
                    f'{len(overlap)} shared groups'
                )


def _assert_disjoint_fingerprint_arrays(parts):
    names = list(parts)
    sets = {
        name: set(np.asarray(values, dtype=np.uint64).tolist())
        for name, values in parts.items()
    }
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = sets[left_name] & sets[right_name]
            if overlap:
                raise AssertionError(
                    f'post-transform model-input fingerprint leakage between '
                    f'{left_name} and {right_name}: {len(overlap)} groups'
                )


def _purge_later_known_fingerprint_collisions(result):
    """Keep each exact model-input group in only its highest-priority role."""
    priority = ('train', 'val', 'meta_known', 'calibration')
    seen = set()
    audit = {}
    for key in priority:
        fingerprints = np.asarray(result[f'fingerprints_{key}'], dtype=np.uint64)
        keep = np.asarray([value not in seen for value in fingerprints], dtype=bool)
        removed_fingerprints = set(fingerprints[~keep].tolist())
        before = len(fingerprints)
        for field in ('X', 'y', 'fingerprints', 'source_roles', 'row_ids'):
            result[f'{field}_{key}'] = np.asarray(result[f'{field}_{key}'])[keep]
        seen.update(np.asarray(result[f'fingerprints_{key}'], dtype=np.uint64).tolist())
        audit[key] = {
            'rows_before': int(before),
            'rows_removed_due_to_higher_priority_fingerprint': int((~keep).sum()),
            'fingerprints_removed': int(len(removed_fingerprints)),
            'rows_after': int(keep.sum()),
        }
        if not bool(keep.any()):
            raise AssertionError(
                f'model-input fingerprint purge emptied known partition {key}'
            )
    return audit


def _split_known_by_fingerprint(df_known, seed):
    """Create a grouped 70/10/10/10 known split using ten stratified folds."""
    df_known = df_known.copy()
    if FINGERPRINT_COLUMN not in df_known:
        df_known[FINGERPRINT_COLUMN] = feature_fingerprints(df_known).values
    groups_per_class = df_known.groupby('attack_cat')[FINGERPRINT_COLUMN].nunique()
    too_small = groups_per_class[groups_per_class < 10]
    if not too_small.empty:
        raise ValueError(
            '70/10/10/10 grouped split requires at least 10 unique feature '
            f'fingerprints per known class; insufficient={too_small.to_dict()}'
        )

    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    fold_ids = np.full(len(df_known), -1, dtype=np.int8)
    y = df_known['attack_cat'].to_numpy()
    groups = df_known[FINGERPRINT_COLUMN].to_numpy()
    for fold_id, (_, held_positions) in enumerate(
        splitter.split(np.zeros(len(df_known)), y, groups)
    ):
        fold_ids[held_positions] = fold_id
    if bool((fold_ids < 0).any()):
        raise AssertionError('grouped split did not assign every known row')

    parts = {
        'backbone_train': df_known.iloc[np.flatnonzero(fold_ids < 7)].copy(),
        'validation': df_known.iloc[np.flatnonzero(fold_ids == 7)].copy(),
        'meta_known': df_known.iloc[np.flatnonzero(fold_ids == 8)].copy(),
        'calibration': df_known.iloc[np.flatnonzero(fold_ids == 9)].copy(),
    }
    if any(frame.empty for frame in parts.values()):
        raise AssertionError('70/10/10/10 grouped split produced an empty partition')
    _assert_disjoint_fingerprints(parts)
    return parts


def _assert_fit_roles(parts):
    for name, frame in parts.items():
        roles = set(frame[SOURCE_ROLE_COLUMN].astype(str))
        if 'official_test' in roles:
            raise AssertionError(
                f'official testing rows leaked into fit/calibrate partition {name}'
            )


def prepare_official_splits(train_df, test_df, calibration_df=None,
                            known_cats=KNOWN_ATTACK_CATS,
                            zd_cats=ZERO_DAY_ATTACK_CATS, seed=42):
    """Prepare leakage-resistant official UNSW-NB15 training/evaluation data."""
    class_role_overlap = sorted(set(map(str, known_cats)) & set(map(str, zd_cats)))
    if class_role_overlap:
        raise ValueError(
            f'classes cannot be both known and zero-day targets: {class_role_overlap}'
        )
    if calibration_df is not None:
        raise ValueError(
            'The selected split policy uses an internal 70/10/10/10 split of the '
            'official training set; calibration_files must be empty.'
        )

    train_df = normalize_labels(train_df.copy())
    test_df = normalize_labels(test_df.copy())
    validate_declared_classes(train_df, known_cats, zd_cats)
    validate_declared_classes(test_df, known_cats, zd_cats)
    if SOURCE_ROLE_COLUMN not in train_df:
        train_df[SOURCE_ROLE_COLUMN] = 'official_train'
    if SOURCE_ROLE_COLUMN not in test_df:
        test_df[SOURCE_ROLE_COLUMN] = 'official_test'
    if SOURCE_FILE_COLUMN not in train_df:
        train_df[SOURCE_FILE_COLUMN] = '<in-memory-train>'
    if SOURCE_FILE_COLUMN not in test_df:
        test_df[SOURCE_FILE_COLUMN] = '<in-memory-test>'
    # Preserve a stable identity for every official-training row so the split
    # contract can prove row-disjointness independently of feature equality.
    train_df[ROW_ID_COLUMN] = np.arange(len(train_df), dtype=np.int64)
    test_df[ROW_ID_COLUMN] = np.arange(len(test_df), dtype=np.int64)

    train_known = train_df[train_df['attack_cat'].isin(known_cats)].copy()
    train_ood = train_df[train_df['attack_cat'].isin(zd_cats)].copy()
    test_known = test_df[test_df['attack_cat'].isin(known_cats)].copy()
    test_ood = test_df[test_df['attack_cat'].isin(zd_cats)].copy()
    # The ordered schema and categorical vocabulary are a deterministic contract
    # of the official development role.  They must not depend on a seed-specific
    # split or on any official-test values.  Near-constant filtering is therefore
    # intentionally disabled for this benchmark.
    schema_reference = train_known.copy()
    base_feat_cols, categorical_maps = _get_numeric_features(schema_reference)
    schema_reference, feat_cols = engineer_features(
        schema_reference, base_feat_cols.copy()
    )
    feat_cols = list(dict.fromkeys(feat_cols))
    _assert_model_feature_schema(feat_cols)

    def prepare_features(frame):
        prepared = frame.copy()
        _encode_categorical_features(prepared, categorical_maps)
        prepared, _ = engineer_features(prepared, base_feat_cols.copy())
        missing = [column for column in feat_cols if column not in prepared]
        if missing:
            raise ValueError(f'frame is missing model features: {missing}')
        return prepared

    # Group known partitions by the canonical unscaled model vector. A second
    # post-scaling assertion below catches any equality introduced by clipping.
    known_grouping_view = prepare_features(train_known)
    train_known[FINGERPRINT_COLUMN] = feature_fingerprints(
        known_grouping_view, feat_cols
    ).values
    known_parts = _split_known_by_fingerprint(train_known, seed)

    for name, frame in known_parts.items():
        leaked = sorted(set(frame['attack_cat']) & set(zd_cats))
        if leaked:
            raise AssertionError(f'OOD classes leaked into known partition {name}: {leaked}')
    _assert_fit_roles({**known_parts, 'surrogate_ood_pool': train_ood})

    le = LabelEncoder().fit(list(known_cats))
    all_frames = {
        **known_parts,
        'official_test_known': test_known,
        'train_ood': train_ood,
        'official_test_ood': test_ood,
    }

    transformed_frames = {}
    for name, frame in all_frames.items():
        transformed_frames[name] = prepare_features(frame)

    scaler = RobustScaler()
    scaler.fit(
        transformed_frames['backbone_train'][feat_cols].to_numpy(dtype=np.float32)
    )

    def transform(name):
        values = scaler.transform(
            transformed_frames[name][feat_cols].to_numpy(dtype=np.float32)
        )
        return np.clip(
            np.nan_to_num(values, nan=0.0, posinf=10.0, neginf=-10.0),
            -10.0,
            10.0,
        ).astype(np.float32)

    split_key = {
        'backbone_train': 'train',
        'validation': 'val',
        'meta_known': 'meta_known',
        'calibration': 'calibration',
        'official_test_known': 'test',
    }
    transformed_arrays = {name: transform(name) for name in transformed_frames}
    result = {}
    for frame_name, key in split_key.items():
        frame = transformed_frames[frame_name]
        result[f'X_{key}'] = transformed_arrays[frame_name]
        result[f'y_{key}'] = le.transform(frame['attack_cat'])
        result[f'fingerprints_{key}'] = model_input_fingerprints(
            transformed_arrays[frame_name]
        ).to_numpy(dtype=np.uint64)
        result[f'source_roles_{key}'] = frame[SOURCE_ROLE_COLUMN].astype(str).to_numpy()
        result[f'row_ids_{key}'] = frame[ROW_ID_COLUMN].to_numpy(dtype=np.int64)

    result.update({
        'X_ood_train': transformed_arrays['train_ood'],
        'y_ood_train': transformed_frames['train_ood']['attack_cat'].astype(str).to_numpy(),
        'fingerprints_ood_train': model_input_fingerprints(
            transformed_arrays['train_ood']
        ).to_numpy(dtype=np.uint64),
        'source_roles_ood_train': transformed_frames['train_ood'][SOURCE_ROLE_COLUMN].astype(str).to_numpy(),
        'X_ood_test': transformed_arrays['official_test_ood'],
        'y_ood_test': transformed_frames['official_test_ood']['attack_cat'].astype(str).to_numpy(),
        'fingerprints_ood_test': model_input_fingerprints(
            transformed_arrays['official_test_ood']
        ).to_numpy(dtype=np.uint64),
        'source_roles_ood_test': transformed_frames['official_test_ood'][SOURCE_ROLE_COLUMN].astype(str).to_numpy(),
        'n_features': len(feat_cols),
        'n_classes': len(le.classes_),
        'label_encoder': le,
        'scaler': scaler,
        'feat_cols': feat_cols,
        'feature_schema_sha256': feature_schema_hash(feat_cols),
        'categorical_maps': categorical_maps,
        'known_cats': list(known_cats),
        'zd_cats': list(zd_cats),
        'fit_role_contract': {
            'feature_schema': ['official_train_known'],
            'categorical_vocabulary': ['official_train_known'],
            'scaler': ['backbone_train'],
            'backbone_model': ['backbone_train', 'validation'],
            'hybrid_meta_known': ['meta_known'],
            'hybrid_meta_ood': ['purged_surrogate_ood'],
            'threshold_calibration': ['calibration_normal'],
            'final_evaluation': ['official_test'],
        },
    })
    # Clipping can make previously distinct unscaled rows identical. Keep each
    # exact model-input group only in its highest-priority development role.
    result['known_split_model_input_purge'] = (
        _purge_later_known_fingerprint_collisions(result)
    )
    # Re-check disjointness on exact post-scaling/clipping float32 model inputs.
    _assert_disjoint_fingerprint_arrays({
        'backbone_train': result['fingerprints_train'],
        'validation': result['fingerprints_val'],
        'meta_known': result['fingerprints_meta_known'],
        'calibration': result['fingerprints_calibration'],
    })
    train_fingerprint_rows = np.concatenate([
        result['fingerprints_train'],
        result['fingerprints_val'],
        result['fingerprints_meta_known'],
        result['fingerprints_calibration'],
        result['fingerprints_ood_train'],
    ])
    test_fingerprint_rows = np.concatenate([
        result['fingerprints_test'], result['fingerprints_ood_test']
    ])
    train_fingerprints = set(train_fingerprint_rows.tolist())
    test_fingerprints = set(test_fingerprint_rows.tolist())
    shared = train_fingerprints & test_fingerprints
    result['train_fingerprint_set'] = train_fingerprints
    result['shared_official_fingerprint_count'] = len(shared)
    if shared:
        train_rows = int(sum(value in shared for value in train_fingerprint_rows))
        test_rows = int(sum(value in shared for value in test_fingerprint_rows))
        print(
            '  [WARN] Official train/test model-input overlap: '
            f'{len(shared):,} fingerprints, {train_rows:,} train rows, '
            f'{test_rows:,} test rows. Official test is preserved; an '
            'unseen-fingerprint sensitivity analysis will also be reported.'
        )
    result['test_unseen_fingerprint_mask'] = np.asarray([
        fingerprint not in train_fingerprints
        for fingerprint in result['fingerprints_test']
    ], dtype=bool)
    result['ood_test_unseen_fingerprint_mask'] = np.asarray([
        fingerprint not in train_fingerprints
        for fingerprint in result['fingerprints_ood_test']
    ], dtype=bool)
    result['fingerprint_contamination'] = {
        'fingerprint_semantics': 'exact_post_scaling_clipping_float32_model_input',
        'train_unique_fingerprints': int(len(train_fingerprints)),
        'test_unique_fingerprints': int(len(test_fingerprints)),
        'shared_fingerprints': int(len(shared)),
        'shared_test_rows': int(sum(value in shared for value in test_fingerprint_rows)),
        'unseen_test_rows': int(sum(value not in shared for value in test_fingerprint_rows)),
    }
    result['lofo_known_partition_overlap_audit'] = (
        audit_lofo_known_partition_overlap(result)
    )
    result['isolation_assertions'] = assert_training_isolation(result)
    return result


def build_lofo_surrogate_mask(splits, target_family):
    """Exclude a LOFO target by label and by every target-family fingerprint."""
    labels = np.asarray(splits['y_ood_train']).astype(str)
    fingerprints = np.asarray(splits['fingerprints_ood_train'], dtype=np.uint64)
    target_rows_mask = labels == str(target_family)
    target_fingerprints = set(fingerprints[target_rows_mask].tolist())
    label_eligible = ~target_rows_mask
    fingerprint_collision = np.asarray(
        [fingerprint in target_fingerprints for fingerprint in fingerprints],
        dtype=bool,
    )
    surrogate_mask = label_eligible & ~fingerprint_collision
    surrogate_fingerprints = set(fingerprints[surrogate_mask].tolist())
    intersection = target_fingerprints & surrogate_fingerprints
    if intersection:
        raise AssertionError(
            f'target fingerprint leakage after purge for {target_family}: '
            f'{len(intersection)} shared fingerprints'
        )
    audit = {
        'target_family': str(target_family),
        'target_rows': int(target_rows_mask.sum()),
        'unique_target_fingerprints': int(len(target_fingerprints)),
        'surrogate_rows_before': int(label_eligible.sum()),
        'surrogate_rows_removed_due_to_target_fingerprint': int(
            (label_eligible & fingerprint_collision).sum()
        ),
        'surrogate_rows_after': int(surrogate_mask.sum()),
        'target_surrogate_fingerprint_intersection_after_purge': int(len(intersection)),
    }
    return surrogate_mask, audit


def audit_lofo_known_partition_overlap(splits):
    """Audit target-family model inputs that also occur in known fit roles."""
    partitions = ('train', 'val', 'meta_known', 'calibration')
    observed_known_labels = set()
    for partition in partitions:
        observed_known_labels.update(map(str, splits['label_encoder'].inverse_transform(
            np.asarray(splits[f'y_{partition}'], dtype=np.int64)
        )))
    labels = np.asarray(splits['y_ood_train']).astype(str)
    ood_fingerprints = np.asarray(
        splits['fingerprints_ood_train'], dtype=np.uint64
    )
    audit = {}
    for target_family in splits['zd_cats']:
        target_fingerprints = set(
            ood_fingerprints[labels == str(target_family)].tolist()
        )
        partition_audit = {}
        for partition in partitions:
            values = np.asarray(
                splits[f'fingerprints_{partition}'], dtype=np.uint64
            )
            intersection = target_fingerprints & set(values.tolist())
            partition_audit[partition] = {
                'shared_fingerprints': int(len(intersection)),
                'shared_rows': int(sum(value in target_fingerprints for value in values)),
            }
        normal_idx = int(splits['label_encoder'].transform(['Normal'])[0])
        normal_mask = np.asarray(splits['y_calibration']) == normal_idx
        normal_values = np.asarray(
            splits['fingerprints_calibration'], dtype=np.uint64
        )[normal_mask]
        normal_intersection = target_fingerprints & set(normal_values.tolist())
        partition_audit['calibration_normal'] = {
            'shared_fingerprints': int(len(normal_intersection)),
            'shared_rows': int(sum(
                value in target_fingerprints for value in normal_values
            )),
            'support': int(normal_mask.sum()),
        }
        audit[str(target_family)] = {
            'target_labels_absent_from_known_roles': (
                str(target_family) not in observed_known_labels
            ),
            'observed_known_labels': sorted(observed_known_labels),
            'partitions': partition_audit,
        }
    return audit


def assert_lofo_retraining_gate(splits):
    """Block retraining when target model inputs contaminate any fitted known role."""
    if splits.get('report', {}).get('protocol') == 'target_isolated_outer_lofo_v1':
        from .target_isolation import assert_target_fold_retraining_gate
        return assert_target_fold_retraining_gate(splits)
    audit = audit_lofo_known_partition_overlap(splits)
    violations = []
    for family, family_audit in audit.items():
        if not family_audit['target_labels_absent_from_known_roles']:
            violations.append(f'{family}: target label appears in known roles')
        for partition, values in family_audit['partitions'].items():
            if values['shared_fingerprints']:
                violations.append(
                    f'{family}/{partition}: '
                    f'{values["shared_fingerprints"]} target fingerprints, '
                    f'{values["shared_rows"]} rows'
                )
    if violations:
        raise AssertionError(
            'corrected retraining blocked by target-fingerprint contamination in '
            'known fit roles; choose and document a global-purge or per-target-model '
            f'policy before training. Violations: {violations}'
        )
    return audit


def assert_training_isolation(splits, target_family=None, surrogate_labels=None,
                              surrogate_fingerprints=None,
                              target_fingerprints=None):
    """Runtime assertions for source, fingerprint and target-family isolation."""
    _assert_model_feature_schema(splits['feat_cols'])
    fit_role_keys = [
        'source_roles_train', 'source_roles_val', 'source_roles_meta_known',
        'source_roles_calibration', 'source_roles_ood_train',
    ]
    for key in fit_role_keys:
        roles = set(map(str, splits[key]))
        if 'official_test' in roles:
            raise AssertionError(f'official testing set leaked into fit/calibrate via {key}')

    fingerprint_keys = [
        'fingerprints_train', 'fingerprints_val',
        'fingerprints_meta_known', 'fingerprints_calibration',
    ]
    fingerprint_parts = {
        key: set(np.asarray(splits[key], dtype=np.uint64)) for key in fingerprint_keys
    }
    names = list(fingerprint_parts)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = fingerprint_parts[left_name] & fingerprint_parts[right_name]
            if overlap:
                raise AssertionError(
                    f'known split fingerprint leakage between {left_name} and '
                    f'{right_name}: {len(overlap)} groups'
                )

    row_id_keys = [
        'row_ids_train', 'row_ids_val',
        'row_ids_meta_known', 'row_ids_calibration',
    ]
    row_id_parts = {
        key: set(np.asarray(splits[key], dtype=np.int64)) for key in row_id_keys
    }
    names = list(row_id_parts)
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            overlap = row_id_parts[left_name] & row_id_parts[right_name]
            if overlap:
                raise AssertionError(
                    f'known split row leakage between {left_name} and '
                    f'{right_name}: {len(overlap)} rows'
                )

    target_isolated = None
    if target_family is not None:
        if surrogate_labels is None:
            raise AssertionError('surrogate labels are required for LOFO isolation')
        surrogate = set(map(str, np.asarray(surrogate_labels)))
        if str(target_family) in surrogate:
            raise AssertionError(
                f'target OOD family {target_family} leaked into surrogate OOD'
            )
        if surrogate_fingerprints is None or target_fingerprints is None:
            raise AssertionError(
                'surrogate and target fingerprints are required for LOFO isolation'
            )
        overlap = set(np.asarray(surrogate_fingerprints, dtype=np.uint64)) & set(
            np.asarray(target_fingerprints, dtype=np.uint64)
        )
        if overlap:
            raise AssertionError(
                f'target OOD fingerprint leaked into surrogate OOD: '
                f'{len(overlap)} groups'
            )
        target_isolated = True
    return {
        'official_test_excluded_from_fit': True,
        'official_test_excluded_from_preprocessing': True,
        'metadata_excluded_from_model_features': True,
        'known_split_rows_disjoint': True,
        'known_split_fingerprints_disjoint': True,
        'target_labels_absent_from_known_roles': not bool(
            set(map(str, splits['known_cats'])) & set(map(str, splits['zd_cats']))
        ),
        'target_family_excluded_from_surrogate': target_isolated,
        'target_fingerprints_excluded_from_surrogate': (
            None if target_family is None else True
        ),
    }


def prepare_splits(df, known_cats=KNOWN_ATTACK_CATS, zd_cats=ZERO_DAY_ATTACK_CATS,
                   test_ratio=0.20, val_ratio=0.10, seed=42, zd_augment=1):
    print('\n[DATA SPLIT]')
    df = normalize_labels(df)
    validate_declared_classes(df, known_cats, zd_cats)

    cat_counts = df['attack_cat'].value_counts()
    print('\n  Distribution:')
    for cat, cnt in cat_counts.items():
        m = 'K' if cat in known_cats else ('Z' if cat in zd_cats else '?')
        print(f'    [{m}] {cat:<22} {cnt:>8,}')

    avail      = set(df['attack_cat'].unique())
    act_known  = [c for c in known_cats if c in avail]
    act_zd     = [c for c in zd_cats    if c in avail]

    df_known   = df[df['attack_cat'].isin(act_known)].copy()
    df_zd_full = df[df['attack_cat'].isin(act_zd)].copy()

    print(f'\n  Known  {len(act_known)} classes: {len(df_known):,} samples')
    print(f'  ZD     {len(act_zd)} classes: {len(df_zd_full):,} samples')

    le = LabelEncoder()
    le.fit(act_known)
    df_known['y'] = le.transform(df_known['attack_cat'])
    n_classes = len(act_known)

    min_s = df_known['y'].value_counts().min()
    strat = df_known['y'] if min_s >= 5 else None

    idx_tv, idx_te = train_test_split(df_known.index, test_size=test_ratio,
                                       stratify=strat, random_state=seed)
    strat_tv = df_known.loc[idx_tv,'y'] if strat is not None else None
    idx_tr, idx_va = train_test_split(idx_tv,
                                       test_size=val_ratio/(1-test_ratio),
                                       stratify=strat_tv, random_state=seed)

    print(f'  Train {len(idx_tr):,} | Val {len(idx_va):,} | Test {len(idx_te):,}')
    print(f'  ZD pool: {len(df_zd_full):,}')

    df_train_view = df_known.loc[idx_tr].copy()
    base_feat_cols, categorical_maps = _get_numeric_features(df_train_view)
    df_train_view, engineered_feat_cols = engineer_features(df_train_view, base_feat_cols.copy())

    std = df_train_view[engineered_feat_cols].std()
    feat_cols = [c for c in engineered_feat_cols if std[c] > 1e-8]
    _assert_model_feature_schema(feat_cols)

    _encode_categorical_features(df_known, categorical_maps)
    _encode_categorical_features(df_zd_full, categorical_maps)
    df_known, _ = engineer_features(df_known, base_feat_cols.copy())
    df_zd_full, _ = engineer_features(df_zd_full, base_feat_cols.copy())
    feat_cols = [c for c in feat_cols if c in df_known.columns and c in df_zd_full.columns]
    print(f'  Features: {len(feat_cols)}')

    scaler  = RobustScaler()
    X_tr    = scaler.fit_transform(df_known.loc[idx_tr,feat_cols].values.astype(np.float32))
    X_va    = scaler.transform(df_known.loc[idx_va,feat_cols].values.astype(np.float32))
    X_te    = scaler.transform(df_known.loc[idx_te,feat_cols].values.astype(np.float32))
    X_zd    = scaler.transform(df_zd_full[feat_cols].values.astype(np.float32))

    clip = 10.
    X_tr, X_va, X_te, X_zd = [np.clip(x, -clip, clip) for x in [X_tr, X_va, X_te, X_zd]]
    X_tr, X_va, X_te, X_zd = [
        np.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip)
        for x in [X_tr, X_va, X_te, X_zd]
    ]

    y_tr = df_known.loc[idx_tr,'y'].values
    y_va = df_known.loc[idx_va,'y'].values
    y_te = df_known.loc[idx_te,'y'].values
    y_zd = df_zd_full['attack_cat'].values

    return dict(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        X_zd=X_zd,    y_zd=y_zd,
        n_features=len(feat_cols), n_classes=n_classes,
        label_encoder=le, scaler=scaler, feat_cols=feat_cols,
        categorical_maps=categorical_maps,
        known_cats=act_known, zd_cats=act_zd,
    )


# ═══════════════════════════════════════════════════════════════

class FlowDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


def make_loaders(splits, batch_size=512, num_workers=2,
                 dos_class_idx=None, dos_over=5.0, class_sample_weights=None,
                 seed=42):
    y_tr    = np.asarray(splits['y_train'])
    freq    = np.bincount(y_tr)
    weights = 1.0 / freq[y_tr].astype(np.float32)
    if dos_class_idx is not None:
        weights[y_tr == dos_class_idx] *= dos_over
    for class_idx, multiplier in (class_sample_weights or {}).items():
        weights[y_tr == int(class_idx)] *= float(multiplier)

    tr_ds = FlowDS(splits['X_train'], y_tr)
    va_ds = FlowDS(splits['X_val'],   splits['y_val'])
    te_ds = FlowDS(splits['X_test'],  splits['y_test'])

    sampler = WeightedRandomSampler(torch.FloatTensor(weights), len(y_tr), replacement=True)

    gen = torch.Generator()
    gen.manual_seed(seed)

    def _seed_worker(worker_id):
        worker_seed = (seed + worker_id) % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=gen,
    )
    if num_workers > 0:
        kw.update(
            persistent_workers=True,
            prefetch_factor=2,
        )
    return {
        'train': DataLoader(tr_ds, batch_size=batch_size, sampler=sampler, **kw),
        'val':   DataLoader(va_ds, batch_size=batch_size, shuffle=False, **kw),
        'test':  DataLoader(te_ds, batch_size=batch_size, shuffle=False, **kw),
    }


# ═══════════════════════════════════════════════════════════════
