"""Configuration, path resolution and reproducibility helpers for IDS v14."""

from dataclasses import dataclass
import os, random
import numpy as np
import torch

@dataclass
class CFG:
    data_dir: str = '/kaggle/input'
    save_dir: str = '/kaggle/working/checkpoints_v14'
    plot_dir: str = '/kaggle/working/plots_v14'
    train_files: str = ''
    calibration_files: str = ''
    test_files: str = ''
    report_path: str = ''
    demo: bool = False

    # Training
    epochs: int = 100
    batch_size: int = 512
    lr: float = 3e-4
    hidden: int = 256
    ae_hidden: int = 128
    patience: int = 20

    # Loss
    lambda_con: float = 0.3
    focal_gamma: float = 2.0
    dos_weight: float = 3.0
    dos_sampler_weight: float = 1.5
    class_loss_weights: str = "Exploits=3.0,Reconnaissance=3.0"
    class_sampler_weights: str = "Exploits=4.0,Reconnaissance=4.0"
    recon_dos_penalty: float = 2.0

    # Zero-day
    target_fpr: float = 0.05
    adaptive_threshold: bool = False
    n_clusters: int = 25
    zd_augment_factor: int = 1

    num_workers: int = 2
    seed: int = 42


def resolve_paths(cfg):
    base_dir   = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    local_data = os.path.join(base_dir, 'data')
    local_ckpt = os.path.join(base_dir, 'checkpoints')
    local_plot = os.path.join(base_dir, 'plots')

    if getattr(cfg, 'data_dir', None):
        if (not os.path.exists(cfg.data_dir)) and os.path.exists(local_data):
            cfg.data_dir = local_data

    if (
        getattr(cfg, 'save_dir', None)
        and str(cfg.save_dir).startswith('/kaggle/')
        and not os.path.exists('/kaggle/working')
    ):
        cfg.save_dir = local_ckpt
    if (
        getattr(cfg, 'plot_dir', None)
        and str(cfg.plot_dir).startswith('/kaggle/')
        and not os.path.exists('/kaggle/working')
    ):
        cfg.plot_dir = local_plot
    return cfg


def seed_everything(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def get_config():
    in_nb = False
    try:
        shell = get_ipython().__class__.__name__   # noqa
        in_nb = shell in ('ZMQInteractiveShell', 'Shell')
    except NameError:
        pass
    if in_nb:
        return resolve_paths(CFG)
    import argparse
    p = argparse.ArgumentParser()
    for k, v in vars(CFG).items():
        if k.startswith('_'):
            continue
        if isinstance(v, bool):
            p.add_argument(f'--{k}', action='store_true', default=v)
        elif v is not None:
            p.add_argument(f'--{k}', type=type(v), default=v)
    args = p.parse_args()
    for k, v in vars(args).items():
        setattr(CFG, k, v)
    return resolve_paths(CFG)


# ═══════════════════════════════════════════════════════════════
