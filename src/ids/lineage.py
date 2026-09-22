"""Reproducibility metadata for Step 2 scientific reports and artifacts."""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import torch

from .dataset import SOURCE_FILE_COLUMN


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _git_metadata(repo_dir: Path) -> dict:
    def run(*args):
        completed = subprocess.run(
            ['git', *args],
            cwd=repo_dir,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        commit = run('rev-parse', 'HEAD')
        status_lines = run('status', '--porcelain').splitlines()
        branch = run('branch', '--show-current')
        diff = subprocess.run(
            ['git', 'diff', '--binary', 'HEAD'],
            cwd=repo_dir,
            check=True,
            capture_output=True,
        ).stdout
        untracked_paths = run(
            'ls-files', '--others', '--exclude-standard'
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        return {
            'commit_sha': None,
            'branch': None,
            'dirty': None,
            'status': [],
            'working_tree_diff_sha256': None,
            'untracked_files': [],
        }
    untracked_files = []
    for raw_path in untracked_paths:
        path = repo_dir / raw_path
        untracked_files.append({
            'path': raw_path,
            'sha256': _sha256_file(path) if path.is_file() else None,
        })
    return {
        'commit_sha': commit,
        'branch': branch,
        'dirty': bool(status_lines),
        'status': status_lines,
        'working_tree_diff_sha256': hashlib.sha256(diff).hexdigest(),
        'untracked_files': untracked_files,
    }


def _package_versions() -> dict:
    versions = {'python': platform.python_version()}
    for report_name, package_name in (
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('scikit-learn', 'scikit-learn'),
        ('torch', 'torch'),
    ):
        try:
            versions[report_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[report_name] = None
    return versions


def build_experiment_lineage(args, data_roles, splits, repo_dir=None):
    """Build self-contained provenance without reading any evaluation labels."""
    repo_dir = Path(repo_dir or Path(__file__).resolve().parents[2]).resolve()
    dataset_files = {}
    for role, frame in data_roles.items():
        if frame is None:
            dataset_files[role] = []
            continue
        paths = sorted(set(map(str, frame[SOURCE_FILE_COLUMN])))
        records = []
        for raw_path in paths:
            path = Path(raw_path)
            records.append({
                'filename': path.name,
                'path': str(path),
                'sha256': _sha256_file(path) if path.is_file() else None,
            })
        dataset_files[role] = records

    config = {
        key: value if isinstance(value, (str, int, float, bool, type(None))) else str(value)
        for key, value in vars(args).items()
        if not str(key).startswith('_')
    }
    return {
        'git': _git_metadata(repo_dir),
        'invocation': list(sys.argv),
        'execution_environment': {
            'platform': platform.platform(),
            'machine': platform.machine(),
            'selected_device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'cuda_version': torch.version.cuda,
            'cudnn_version': (
                torch.backends.cudnn.version() if torch.backends.cudnn.is_available()
                else None
            ),
            'cuda_device_name': (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
        },
        'seed': int(args.seed),
        'dataset_files': dataset_files,
        'config': config,
        'feature_schema': list(splits['feat_cols']),
        'feature_count': int(splits['n_features']),
        'feature_schema_sha256': str(splits['feature_schema_sha256']),
        'split_counts': {
            'backbone_train': len(splits['X_train']),
            'validation': len(splits['X_val']),
            'meta_known': len(splits['X_meta_known']),
            'calibration': len(splits['X_calibration']),
            'official_train_ood': len(splits['X_ood_train']),
            'official_test_known': len(splits['X_test']),
            'official_test_ood': len(splits['X_ood_test']),
        },
        'known_split_model_input_purge': dict(
            splits['known_split_model_input_purge']
        ),
        'train_test_role_contract': dict(splits['fit_role_contract']),
        'threshold_calibration_population': 'internal calibration partition: Normal only',
        'target_families': list(splits['zd_cats']),
        'fingerprint_contamination': dict(splits['fingerprint_contamination']),
        'lofo_known_partition_overlap_audit': dict(
            splits['lofo_known_partition_overlap_audit']
        ),
        'package_versions': _package_versions(),
    }
