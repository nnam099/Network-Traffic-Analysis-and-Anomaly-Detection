"""Entry point for IDS v14 UNSW-NB15 training, evaluation, artifact export and plotting."""

import os, json, pickle, time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder

from .config import CFG, get_config, seed_everything
from .dataset import (
    ZERO_DAY_ATTACK_CATS,
    _assert_model_feature_schema,
    assert_lofo_retraining_gate,
    assert_training_isolation,
    build_lofo_surrogate_mask,
    load_official_unsw_splits,
    make_loaders,
    prepare_official_splits,
)
from .lineage import build_experiment_lineage
from .models import IDSModel
from .losses import IDSLoss
from .trainer import normal_class_index, train
from .evaluator import (
    _batch_scores,
    add_hybrid_scores,
    collect_ood_scores,
    evaluate_zero_day_from_scores,
    fit_hybrid_meta_learner,
    build_centroids,
    calibrate,
    compute_adaptive_threshold_trace,
    evaluate_classifier,
    evaluate_zero_day,
)
from .plots import (
    plot_training_curve,
    plot_soc_decision_space,
    plot_per_class_proper,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_threshold_drift,
)

def save_artifacts(model, splits, thresholds, history, centroids, save_dir,
                   hybrid_meta=None, research_protocol=None):
    os.makedirs(save_dir, exist_ok=True)
    _assert_model_feature_schema(splits['feat_cols'])

    pth_path = os.path.join(save_dir, 'ids_v14_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'n_features':       splits['n_features'],
        'n_classes':        splits['n_classes'],
        'hidden':           model.backbone.hidden,      # Luu lai de load dung architecture
        'ae_hidden':        model.ae.enc[4].in_features, # enc[4]=Linear(mid, mid//2) -> in_features=mid=ae_hidden
        'label_classes':    list(splits['label_encoder'].classes_),
        'known_cats':       splits['known_cats'],
        'zd_cats':          splits['zd_cats'],
        'feat_cols':        splits['feat_cols'],
        'categorical_maps': splits.get('categorical_maps', {}),
        'thresholds':       thresholds,
        'hybrid_meta':      hybrid_meta,
        'history':          history,
        'version':          'v14.0',
        'research_protocol': research_protocol,
    }, pth_path)
    print(f'  Model weights -> {pth_path}')

    pkl_path = os.path.join(save_dir, 'ids_v14_pipeline.pkl')
    pipeline = {
        'scaler':        splits['scaler'],
        'label_encoder': splits['label_encoder'],
        'feat_cols':     splits['feat_cols'],
        'feature_names': splits['feat_cols'],
        'known_cats':    splits['known_cats'],
        'zd_cats':       splits['zd_cats'],
        'thresholds':    thresholds,
        'hybrid_meta':   hybrid_meta,
        'categorical_maps': splits.get('categorical_maps', {}),
        'centroids_np':  centroids.cpu().numpy(),
        'n_features':    splits['n_features'],
        'n_classes':     splits['n_classes'],
        'version':       'v14.0',
        'research_protocol': research_protocol,
    }
    with open(pkl_path,'wb') as f:
        pickle.dump(pipeline, f)
    print(f'  Pipeline pkl  -> {pkl_path}')
    return pth_path, pkl_path


def parse_class_weight_overrides(value, label_names):
    """Parse Class=weight pairs into label-encoder indices."""
    if not value:
        return {}
    if isinstance(value, dict):
        items = value.items()
    else:
        items = []
        for part in str(value).split(','):
            part = part.strip()
            if not part:
                continue
            if '=' not in part:
                raise ValueError(f'class weight override must use Class=weight format: {part}')
            name, weight = part.split('=', 1)
            items.append((name.strip(), weight.strip()))

    index_by_name = {str(name): idx for idx, name in enumerate(label_names)}
    overrides = {}
    for name, weight in items:
        if str(name) not in index_by_name:
            print(f'  [WARN] Ignoring class weight override for unknown class: {name}')
            continue
        overrides[index_by_name[str(name)]] = float(weight)
    return overrides


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════
def run_full(args):
    run_started = time.perf_counter()
    print('\n'+'='*70)
    print('IDS v14.0 - leakage-resistant LOFO evaluation | UNSW-NB15')
    print('='*70)

    seed_everything(args.seed)
    if getattr(args, 'adaptive_threshold', False):
        raise ValueError(
            'adaptive_threshold is disabled for scientific benchmarks because '
            'the legacy evaluator updates it from held-out ground-truth labels'
        )

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    print('\n[1/8] Loading explicit official data roles...')
    data_roles = load_official_unsw_splits(
        args.data_dir,
        train_files=getattr(args, 'train_files', '') or None,
        calibration_files=getattr(args, 'calibration_files', '') or None,
        test_files=getattr(args, 'test_files', '') or None,
    )

    print('\n[2/8] Preparing grouped 70/10/10/10 splits...')
    splits = prepare_official_splits(
        data_roles['train'],
        data_roles['test'],
        calibration_df=data_roles['calibration'],
        seed=args.seed,
    )
    assert_lofo_retraining_gate(splits)
    lineage = build_experiment_lineage(args, data_roles, splits)
    print(
        f'  Backbone train {len(splits["X_train"]):,} | '
        f'Validation {len(splits["X_val"]):,} | '
        f'Meta-known {len(splits["X_meta_known"]):,} | '
        f'Calibration {len(splits["X_calibration"]):,}'
    )
    print(
        f'  Official known test {len(splits["X_test"]):,} | '
        f'Train OOD {len(splits["X_ood_train"]):,} | '
        f'Official OOD test {len(splits["X_ood_test"]):,}'
    )

    le = splits['label_encoder']
    label_names = list(le.classes_)
    normal_idx = normal_class_index(label_names)
    dos_idx = None
    recon_idx = None
    if 'DoS' in label_names:
        dos_idx = int(le.transform(['DoS'])[0])
    if 'Reconnaissance' in label_names:
        recon_idx = int(le.transform(['Reconnaissance'])[0])
    class_loss_overrides = parse_class_weight_overrides(
        getattr(args, 'class_loss_weights', ''),
        label_names,
    )
    class_sampler_overrides = parse_class_weight_overrides(
        getattr(args, 'class_sampler_weights', ''),
        label_names,
    )
    if class_loss_overrides:
        print(f'\n  Class loss weight overrides: {class_loss_overrides}')
    if class_sampler_overrides:
        print(f'  Class sampler weight overrides: {class_sampler_overrides}')

    print('\n[3/8] Creating loaders...')
    loaders = make_loaders(splits, batch_size=args.batch_size,
                           num_workers=args.num_workers,
                           dos_class_idx=dos_idx,
                           dos_over=getattr(args, 'dos_sampler_weight', 1.5),
                           class_sample_weights=class_sampler_overrides,
                           seed=args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'\n[4/8] Building model (device={device})...')
    model = IDSModel(n_features=splits['n_features'],
                     n_classes=splits['n_classes'],
                     hidden=args.hidden, ae_hidden=args.ae_hidden)
    model = model.to(device)

    criterion = IDSLoss(
        n_classes=splits['n_classes'], lambda_con=args.lambda_con,
        focal_gamma=args.focal_gamma,  dos_class_idx=dos_idx,
        dos_weight=args.dos_weight,    recon_class_idx=recon_idx,
        recon_dos_penalty=getattr(args, 'recon_dos_penalty', 2.0),
        class_weight_overrides=class_loss_overrides,
        n_features=splits['n_features'],
        device=device,
    )

    print('\n[5/8] Training...')
    model, history = train(
        model,
        loaders,
        args,
        criterion,
        device,
        label_names=label_names,
        normal_idx=normal_idx,
    )

    print('\n[6/8] Freezing centroids, LOFO meta-learners and thresholds...')
    centroids = build_centroids(
        model,
        splits['X_train'],
        splits['y_train'],
        n_clusters=args.n_clusters,
        device=device,
        seed=args.seed,
    )
    normal_calibration = splits['y_calibration'] == normal_idx
    if not bool(normal_calibration.any()):
        raise AssertionError('calibration split contains no Normal rows')

    fold_setups = {}
    fold_artifacts = {}
    all_assertions = dict(splits['isolation_assertions'])
    for target_family in ZERO_DAY_ATTACK_CATS:
        print(f'\n{"-"*70}\nFreezing LOFO target: {target_family}\n{"-"*70}')
        surrogate_mask, purge_audit = build_lofo_surrogate_mask(
            splits, target_family
        )
        fold_assertions = assert_training_isolation(
            splits,
            target_family=target_family,
            surrogate_labels=splits['y_ood_train'][surrogate_mask],
            surrogate_fingerprints=splits['fingerprints_ood_train'][surrogate_mask],
            target_fingerprints=splits['fingerprints_ood_train'][
                splits['y_ood_train'] == target_family
            ],
        )
        all_assertions['target_family_excluded_from_surrogate'] = bool(
            fold_assertions['target_family_excluded_from_surrogate']
        )
        all_assertions['target_fingerprints_excluded_from_surrogate'] = bool(
            fold_assertions['target_fingerprints_excluded_from_surrogate']
        )

        hybrid_meta = fit_hybrid_meta_learner(
            model,
            splits['X_meta_known'],
            splits['X_ood_train'][surrogate_mask],
            device,
            seed=args.seed,
        )
        thresholds = calibrate(
            model,
            splits['X_calibration'][normal_calibration],
            args.target_fpr,
            device,
            centroids,
            hybrid_meta=hybrid_meta,
        )
        thresholds['hybrid_meta'] = hybrid_meta
        surrogate_families = [
            str(family) for family in sorted(set(
                splits['y_ood_train'][surrogate_mask]
            ))
        ]
        fold_setups[target_family] = {
            'hybrid_meta': hybrid_meta,
            'thresholds': thresholds,
            'surrogate_families': surrogate_families,
            'surrogate_support': int(surrogate_mask.sum()),
            'purge_audit': purge_audit,
            'assertions': fold_assertions,
        }
        fold_artifacts[target_family] = {
            'target_family': target_family,
            'surrogate_families': surrogate_families,
            'purge_audit': purge_audit,
            'threshold_calibration_population': {
                'role': 'calibration_normal',
                'support': int(normal_calibration.sum()),
            },
            'assertions': fold_assertions,
            'hybrid_meta': hybrid_meta,
            'thresholds': thresholds,
        }

    # Official test is first accessed for scoring only after every learned
    # component, method choice and threshold has been frozen above.
    print('\n[7/8] Final official-test evaluation with frozen LOFO folds...')
    known_base_scores = collect_ood_scores(model, splits['X_test'], device, centroids)
    ood_base_scores = collect_ood_scores(model, splits['X_ood_test'], device, centroids)
    clf_res = evaluate_classifier(
        model, splits['X_test'], splits['y_test'], label_names, device
    )
    known_unseen = splits['test_unseen_fingerprint_mask']
    clf_sensitivity = evaluate_classifier(
        model,
        splits['X_test'][known_unseen],
        splits['y_test'][known_unseen],
        label_names,
        device,
    )

    def slice_scores(scores, mask):
        return {name: np.asarray(values)[mask] for name, values in scores.items()}

    fold_reports = {}
    for target_family in ZERO_DAY_ATTACK_CATS:
        setup = fold_setups[target_family]
        hybrid_meta = setup['hybrid_meta']
        thresholds = setup['thresholds']
        target_mask = splits['y_ood_test'] == target_family
        if not bool(target_mask.any()):
            raise AssertionError(f'official test has no target OOD rows for {target_family}')
        known_scores = add_hybrid_scores(known_base_scores, hybrid_meta)
        target_scores = add_hybrid_scores(
            slice_scores(ood_base_scores, target_mask), hybrid_meta
        )
        target_labels = splits['y_ood_test'][target_mask]
        official_result = evaluate_zero_day_from_scores(
            known_scores,
            target_scores,
            target_labels,
            thresholds,
            selected_method='hybrid',
            y_known=splits['y_test'],
            known_predictions=clf_res['preds'],
            normal_idx=normal_idx,
        )
        target_unseen = splits['ood_test_unseen_fingerprint_mask'][target_mask]
        sensitivity_result = evaluate_zero_day_from_scores(
            slice_scores(known_scores, known_unseen),
            slice_scores(target_scores, target_unseen),
            target_labels[target_unseen],
            thresholds,
            selected_method='hybrid',
            y_known=splits['y_test'][known_unseen],
            known_predictions=clf_sensitivity['preds'],
            normal_idx=normal_idx,
        )
        official_family = official_result['_per_class'][target_family]
        sensitivity_family = sensitivity_result['_per_class'].get(
            target_family, {'support': 0, 'recall': None}
        )
        official_ood = official_result['ood_metrics']
        sensitivity_ood = sensitivity_result['ood_metrics']
        fold_reports[target_family] = {
            'selected_method': 'hybrid',
            'target_family': target_family,
            'calibration': {
                'normal_support': int(normal_calibration.sum()),
                'target_fpr': float(args.target_fpr),
                'hybrid_threshold': float(thresholds['hybrid']),
            },
            'official_test': {
                'support': official_family['support'],
                'ood_target_recall': float(official_family['recall']),
                'ood_auroc': official_ood['ood_auroc'],
                'ood_auprc': official_ood['ood_auprc'],
                'ood_tpr_at_1pct_fpr': official_ood['ood_tpr_at_1pct_fpr'],
                'ood_tpr_at_5pct_fpr': official_ood['ood_tpr_at_5pct_fpr'],
                'normal_ood_fpr': official_ood['normal_ood_fpr'],
                'known_false_unknown_rate': official_ood['known_false_unknown_rate'],
                'total_alert_fpr': official_ood['total_alert_fpr'],
                'recall': float(official_family['recall']),
                'auroc': official_ood['ood_auroc'],
            },
            'unseen_fingerprint_sensitivity': {
                'support': int(sensitivity_family['support']),
                'ood_target_recall': (
                    None if sensitivity_family['recall'] is None
                    else float(sensitivity_family['recall'])
                ),
                'ood_auroc': sensitivity_ood['ood_auroc'],
                'ood_auprc': sensitivity_ood['ood_auprc'],
                'ood_tpr_at_1pct_fpr': sensitivity_ood['ood_tpr_at_1pct_fpr'],
                'ood_tpr_at_5pct_fpr': sensitivity_ood['ood_tpr_at_5pct_fpr'],
                'normal_ood_fpr': sensitivity_ood['normal_ood_fpr'],
                'known_false_unknown_rate': sensitivity_ood['known_false_unknown_rate'],
                'total_alert_fpr': sensitivity_ood['total_alert_fpr'],
                'recall': (
                    None if sensitivity_family['recall'] is None
                    else float(sensitivity_family['recall'])
                ),
                'auroc': sensitivity_ood['ood_auroc'],
                'known_support': int(known_unseen.sum()),
                'note': (
                    'Exact training model-input fingerprints were excluded. This '
                    'is not a temporal-generalization benchmark.'
                ),
            },
            'surrogate_families': setup['surrogate_families'],
            'surrogate_support': setup['surrogate_support'],
            'purge_audit': setup['purge_audit'],
            'assertions': setup['assertions'],
        }

    def aggregate_fold_metrics(section):
        rows = [values[section] for values in fold_reports.values()]
        valid = [
            row for row in rows
            if row['ood_target_recall'] is not None and row['ood_auroc'] is not None
        ]
        recalls = np.asarray([row['ood_target_recall'] for row in valid], dtype=float)
        aurocs = np.asarray([row['ood_auroc'] for row in valid], dtype=float)
        supports = np.asarray([row['support'] for row in valid], dtype=float)
        if not valid:
            return {
                'macro_recall_mean': None, 'macro_recall_std': None,
                'macro_auroc_mean': None, 'macro_auroc_std': None,
                'pooled_recall': None, 'support': 0, 'families_evaluated': 0,
            }
        return {
            'macro_recall_mean': float(recalls.mean()),
            'macro_recall_std': float(recalls.std(ddof=1)) if len(valid) > 1 else 0.0,
            'macro_auroc_mean': float(aurocs.mean()),
            'macro_auroc_std': float(aurocs.std(ddof=1)) if len(valid) > 1 else 0.0,
            'pooled_recall': float(np.average(recalls, weights=supports)),
            'support': int(supports.sum()),
            'families_evaluated': len(valid),
        }

    aggregate_metrics = {
        'official_test': aggregate_fold_metrics('official_test'),
        'unseen_fingerprint_sensitivity': aggregate_fold_metrics(
            'unseen_fingerprint_sensitivity'
        ),
    }

    print('\n[8/8] Saving seed report and research artifacts...')
    research_protocol = {
        'name': 'official_70_10_10_10_lofo',
        'seed': int(args.seed),
        'selected_method': 'hybrid',
        'folds': fold_artifacts,
        'assertions': all_assertions,
        'lineage': lineage,
    }
    pth_path, pipeline_path = save_artifacts(
        model,
        splits,
        thresholds={},
        history=history,
        centroids=centroids,
        save_dir=args.save_dir,
        hybrid_meta=None,
        research_protocol=research_protocol,
    )
    os.makedirs(args.plot_dir, exist_ok=True)
    plot_training_curve(history, os.path.join(args.plot_dir, 'v14_training_curve.png'))
    plot_confusion_matrix(
        splits['y_test'],
        clf_res['preds'],
        label_names,
        os.path.join(args.plot_dir, 'v14_confusion_matrix.png'),
    )

    elapsed_seconds = time.perf_counter() - run_started
    default_report = os.path.abspath(
        os.path.join(args.save_dir, '..', 'results', f'step2_seed_{args.seed}_lofo.json')
    )
    report_path = getattr(args, 'report_path', '') or default_report
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    report = {
        'version': 'v14.0-step2',
        'scientific_status': 'LEAKAGE_GUARDED_RESEARCH_BENCHMARK',
        'valid_for_scientific_claims': True,
        'protocol': 'official_70_10_10_10_lofo',
        'seed': int(args.seed),
        'elapsed_seconds': float(elapsed_seconds),
        'elapsed_hours': float(elapsed_seconds / 3600.0),
        'n_epochs': len(history),
        'n_features': int(splits['n_features']),
        'feature_schema': list(splits['feat_cols']),
        'feature_schema_sha256': splits['feature_schema_sha256'],
        'known_cats': splits['known_cats'],
        'ood_cats': splits['zd_cats'],
        'assertions': all_assertions,
        'split_rows': {
            'backbone_train': len(splits['X_train']),
            'validation': len(splits['X_val']),
            'meta_known': len(splits['X_meta_known']),
            'calibration': len(splits['X_calibration']),
            'official_test_known': len(splits['X_test']),
            'official_train_ood': len(splits['X_ood_train']),
            'official_test_ood': len(splits['X_ood_test']),
        },
        'known_classification_metrics': {
            key: clf_res[key] for key in (
                'binary_attack_detection_accuracy',
                'binary_attack_detection_auroc',
                'known_multiclass_accuracy',
                'known_macro_f1',
            )
        },
        'unseen_fingerprint_known_classification_metrics': {
            key: clf_sensitivity[key] for key in (
                'binary_attack_detection_accuracy',
                'binary_attack_detection_auroc',
                'known_multiclass_accuracy',
                'known_macro_f1',
            )
        },
        'official_test_known_auc': float(clf_res['auc']),
        'unseen_fingerprint_known_auc': float(clf_sensitivity['auc']),
        'deprecated_metric_aliases': {
            'official_test_known_auc': 'known_classification_metrics.binary_attack_detection_auroc',
            'unseen_fingerprint_known_auc': 'unseen_fingerprint_known_classification_metrics.binary_attack_detection_auroc',
            'folds.*.official_test.recall': 'folds.*.official_test.ood_target_recall',
            'folds.*.official_test.auroc': 'folds.*.official_test.ood_auroc',
        },
        'shared_official_fingerprint_count': int(
            splits['shared_official_fingerprint_count']
        ),
        'fingerprint_contamination': splits['fingerprint_contamination'],
        'known_split_model_input_purge': splits['known_split_model_input_purge'],
        'lineage': lineage,
        'aggregate_metrics': aggregate_metrics,
        'folds': fold_reports,
        'sensitivity_analysis_note': (
            'Rows whose exact post-scaling/clipping float32 model-input fingerprint '
            'appears in official training are excluded. This is not a '
            'temporal-generalization benchmark because '
            'the pre-split CSVs contain no timestamps or flow identifiers.'
        ),
        'model_path': pth_path,
        'pipeline_path': pipeline_path,
    }
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2, default=str)

    print(f'\n{"="*70}')
    print(f'FINAL SUMMARY - STEP 2 SEED {args.seed}')
    print(f'{"="*70}')
    for family, values in fold_reports.items():
        official = values['official_test']
        print(
            f'  {family:<16} recall={official["recall"]:.4f} '
            f'AUROC={official["auroc"]:.4f} support={official["support"]:,}'
        )
    print(f'  Assertions: {all_assertions}')
    print(f'  Elapsed   : {elapsed_seconds/3600.0:.3f} hours')
    print(f'  Report    : {report_path}')
    print(f'  Model     : {pth_path}')
    print(f'  Pipeline  : {pipeline_path}')
    print(f'{"="*70}')
    return model, report, history


# ═══════════════════════════════════════════════════════════════
# DEMO MODE
# ═══════════════════════════════════════════════════════════════
def run_demo(args):
    print('\n'+'='*60)
    print('DEMO MODE - Synthetic UNSW-NB15-like data')
    print('='*60)
    seed_everything(getattr(args, 'seed', 42))
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)

    n_feat=55; n_cls=5; n_zd=5
    N=60000
    X = np.random.randn(N, n_feat).astype(np.float32)
    y = np.random.randint(0, n_cls, N)
    for c in range(n_cls): X[y==c] += c*1.5

    dos_idx_demo = 1
    X[y==dos_idx_demo] = np.random.randn((y==dos_idx_demo).sum(), n_feat)*0.8

    N_zd = 8000
    X_zd = (np.random.randn(N_zd, n_feat)*1.5+3.).astype(np.float32)
    y_zd = np.array([f'ZD_{i%n_zd}' for i in range(N_zd)])

    X_tv,X_te,y_tv,y_te = train_test_split(X,y,test_size=0.2,stratify=y)
    X_tr,X_va,y_tr,y_va = train_test_split(X_tv,y_tv,test_size=0.125,stratify=y_tv)

    sc = RobustScaler().fit(X_tr)
    X_tr = sc.transform(X_tr); X_va = sc.transform(X_va)
    X_te = sc.transform(X_te); X_zd = sc.transform(X_zd)

    le = LabelEncoder()
    le.classes_ = np.array([f'Class_{i}' for i in range(n_cls)])

    splits = dict(
        X_train=X_tr, y_train=y_tr,
        X_val=X_va,   y_val=y_va,
        X_test=X_te,  y_test=y_te,
        X_zd=X_zd,    y_zd=y_zd,
        n_features=n_feat, n_classes=n_cls,
        label_encoder=le, scaler=sc,
        feat_cols=[f'f{i}' for i in range(n_feat)],
        feature_names=[f'f{i}' for i in range(n_feat)],
        known_cats=[f'Class_{i}' for i in range(n_cls)],
        zd_cats=[f'ZD_{i}' for i in range(n_zd)],
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = IDSModel(n_features=n_feat, n_classes=n_cls, hidden=128, ae_hidden=64)
    model  = model.to(device)

    args.epochs   = min(getattr(args,'epochs',10), 10)
    args.patience = min(getattr(args,'patience',5), 5)
    args.lr       = getattr(args, 'lr', 3e-4)
    args.hidden   = getattr(args, 'hidden', 128)

    criterion = IDSLoss(n_classes=n_cls, lambda_con=0.3, focal_gamma=2.0,
                        dos_class_idx=dos_idx_demo, dos_weight=getattr(args, 'dos_weight', 3.0),
                        device=device)
    loaders = make_loaders(splits, batch_size=256, num_workers=0,
                           dos_class_idx=dos_idx_demo,
                           seed=getattr(args, 'seed', 42))
    label_names = [f'Class_{i}' for i in range(n_cls)]
    model, history = train(
        model,
        loaders,
        args,
        criterion,
        device,
        label_names=label_names,
        normal_idx=0,
    )

    hybrid_meta = fit_hybrid_meta_learner(
        model, X_va, X_zd, device, seed=getattr(args, 'seed', 42),
    )
    centroids  = build_centroids(model, X_tr, y_tr, 10, device,
                                 seed=getattr(args, 'seed', 42))
    thresholds = calibrate(
        model, X_va[y_va == 0], args.target_fpr, device, centroids,
        hybrid_meta=hybrid_meta,
    )
    thresholds['hybrid_meta'] = hybrid_meta
    re_val_demo = _batch_scores(model, X_va, device)['ae_re']
    re_thr_demo = float(np.quantile(re_val_demo, 1 - args.target_fpr))
    threshold_trace = None
    if getattr(args, 'adaptive_threshold', False):
        threshold_trace = compute_adaptive_threshold_trace(
            model, X_te, y_te, 0, re_val_demo, args.target_fpr, device,
        )
        thresholds['ae_re'] = float(threshold_trace['final_threshold'])
        re_thr_demo = float(threshold_trace['final_threshold'])
        print(f'\n  Adaptive AE threshold final={re_thr_demo:.6f}')

    clf_res     = evaluate_classifier(model, X_te, y_te, label_names, device)
    clf_res['y_test'] = y_te
    zd_res      = evaluate_zero_day(model, X_te, y_te, X_zd, y_zd,
                                     thresholds, centroids, device,
                                     hybrid_meta=hybrid_meta)

    pth_p, pkl_p = save_artifacts(
        model, splits, thresholds, history, centroids, args.save_dir,
        hybrid_meta=hybrid_meta,
    )

    plot_training_curve(history, os.path.join(args.plot_dir,'v14_training_curve.png'))
    plot_soc_decision_space(model,X_te,y_te,X_zd,0.5,re_thr_demo,device,
                             os.path.join(args.plot_dir,'v14_decision_space.png'),label_names,
                             seed=getattr(args, 'seed', 42))
    per_cls_zd   = zd_res.get('_per_class',{})
    zd_cls_order = sorted(per_cls_zd.keys())
    plot_per_class_proper(label_names, y_te, clf_res['preds'], per_cls_zd, zd_cls_order,
                           os.path.join(args.plot_dir,'v14_per_class_detection.png'), normal_idx=0)
    plot_roc_curves(zd_res, os.path.join(args.plot_dir,'v14_roc_curves.png'))
    plot_confusion_matrix(y_te, clf_res['preds'], label_names,
                          os.path.join(args.plot_dir,'v14_confusion_matrix.png'))
    if threshold_trace is not None:
        plot_threshold_drift(
            threshold_trace,
            os.path.join(args.plot_dir, 'v14_threshold_drift.png'),
        )

    print(f'\nDemo done!  Plots -> {args.plot_dir}')
    print(f'   .pth -> {pth_p}')
    print(f'   .pkl -> {pkl_p}')
    return model, zd_res, history


# ═══════════════════════════════════════════════════════════════


def main():
    cfg = get_config()
    if cfg.demo:
        return run_demo(cfg)
    return run_full(cfg)


if __name__ == '__main__':
    main()
