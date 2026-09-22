"""OOD scoring, centroid construction, calibration and zero-day evaluation helpers."""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
    roc_curve,
)

from .threshold import AdaptiveThreshold

def _sigmoid_np(values):
    values = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-values))


def compute_hybrid_meta_score(ae_re, softmax_score, hybrid_meta):
    if not hybrid_meta:
        raise ValueError('hybrid_meta is required for learned hybrid scoring')
    coef = np.asarray(hybrid_meta.get('coef'), dtype=np.float64).reshape(-1)
    if coef.size < 2:
        raise ValueError('hybrid_meta must contain two coefficients: ae_re and softmax')
    intercept = float(hybrid_meta.get('intercept', 0.0))
    ae_re = np.asarray(ae_re, dtype=np.float64)
    softmax_score = np.asarray(softmax_score, dtype=np.float64)
    return _sigmoid_np(intercept + coef[0] * ae_re + coef[1] * softmax_score).astype(np.float32)


def predict_with_uncertainty(model, x, n_samples=30):
    """
    Estimate predictive uncertainty with Monte Carlo Dropout.

    This intentionally enables training mode for the sampled forward passes so
    Dropout layers stay stochastic, then restores the model's previous mode.
    """
    if n_samples <= 0:
        raise ValueError('n_samples must be positive')

    was_training = bool(model.training)
    model.train()
    samples = []
    try:
        with torch.no_grad():
            for _ in range(int(n_samples)):
                outputs = model(x)
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
                samples.append(torch.softmax(logits, dim=-1))
    finally:
        if not was_training:
            model.eval()

    stacked = torch.stack(samples, dim=0)
    mean_probs = stacked.mean(dim=0).squeeze(0)
    std_probs = stacked.std(dim=0, unbiased=False).squeeze(0)
    entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-12))).sum()
    return mean_probs.detach().cpu(), std_probs.detach().cpu(), float(entropy.detach().cpu().item())


def _hybrid_base_features(model, X, device, batch=512):
    model.eval()
    features = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.FloatTensor(X[i:i+batch]).to(device)
            probs = torch.softmax(model.forward(x)[0], dim=-1)
            softmax_score = (1.0 - probs.max(dim=-1).values).cpu().numpy()
            ae_re = model.ae.recon_error(x).cpu().numpy()
            features.append(np.column_stack([ae_re, softmax_score]))
    if not features:
        return np.empty((0, 2), dtype=np.float32)
    return np.concatenate(features, axis=0).astype(np.float32)


def fit_hybrid_meta_learner(model, X_val_known, X_zd, device, seed=42):
    known_features = _hybrid_base_features(model, X_val_known, device)
    zd_features = _hybrid_base_features(model, X_zd, device)
    X_meta = np.vstack([known_features, zd_features])
    y_meta = np.concatenate([
        np.zeros(len(known_features), dtype=np.int64),
        np.ones(len(zd_features), dtype=np.int64),
    ])
    if X_meta.size == 0 or len(np.unique(y_meta)) < 2:
        raise ValueError('hybrid meta-learner requires known validation and zero-day samples')

    learner = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        solver='lbfgs',
        random_state=seed,
    )
    learner.fit(X_meta, y_meta)
    hybrid_meta = {
        'type': 'logistic_regression',
        'features': ['ae_re', 'softmax'],
        'coef': [float(learner.coef_[0, 0]), float(learner.coef_[0, 1])],
        'intercept': float(learner.intercept_[0]),
        'train_rows': int(len(X_meta)),
        'positive_rows': int(y_meta.sum()),
    }
    print('\n  Hybrid meta-learner weights')
    print(f'    {"feature":<16} {"coef":>12}')
    print(f'    {"-"*29}')
    print(f'    {"ae_re":<16} {hybrid_meta["coef"][0]:>12.6f}')
    print(f'    {"1-max_prob":<16} {hybrid_meta["coef"][1]:>12.6f}')
    print(f'    {"intercept":<16} {hybrid_meta["intercept"]:>12.6f}')
    return hybrid_meta


def _batch_scores(model, X, device, centroids=None, hybrid_meta=None, batch=512):
    model.eval()
    s_sm, s_fvc, s_re, s_hyb = [], [], [], []
    with torch.no_grad():
        for i in range(0,len(X),batch):
            x = torch.FloatTensor(X[i:i+batch]).to(device)
            probs = torch.softmax(model.forward(x)[0], dim=-1)
            softmax_np = (1-probs.max(dim=-1).values).cpu().numpy()
            s_sm.append(softmax_np)
            re = model.ae.recon_error(x)
            re_np = re.cpu().numpy()
            s_re.append(re_np)
            if centroids is not None:
                s_fvc.append(model.fv_cluster_score(x,centroids).cpu().numpy())
            if hybrid_meta is not None:
                s_hyb.append(compute_hybrid_meta_score(re_np, softmax_np, hybrid_meta))
    out = {
        'softmax': np.concatenate(s_sm),
        'ae_re':   np.concatenate(s_re),
    }
    if centroids is not None:
        out['fv_cluster'] = np.concatenate(s_fvc)
    if hybrid_meta is not None:
        out['hybrid']     = np.concatenate(s_hyb)
    return out


def _batch_gradbp(model, X, device, batch=512):
    scores = []
    for i in range(0,len(X),batch):
        x = torch.FloatTensor(X[i:i+batch]).to(device)
        scores.append(model.gradbp_score(x).detach().cpu().numpy())
    return np.concatenate(scores)


def build_centroids(model, X_tr, y_tr, n_clusters=25, device='cpu', seed=42):
    model.eval()
    fvs = []
    with torch.no_grad():
        for i in range(0, len(X_tr), 1024):
            x = torch.FloatTensor(X_tr[i:i+1024]).to(device)
            _, fv = model(x)
            fv_np = F.normalize(fv, dim=-1).cpu().float().numpy()
            fvs.append(fv_np)
    all_fvs = np.concatenate(fvs)
    all_fvs = np.nan_to_num(all_fvs, nan=0.0, posinf=0.0, neginf=0.0)

    centers = []
    for cls in range(len(np.unique(y_tr))):
        m  = y_tr == cls
        cv = all_fvs[m]
        if len(cv) == 0:
            continue
        cv = cv[np.isfinite(cv).all(axis=1)]
        if len(cv) == 0:
            continue
        k = min(n_clusters, len(cv))
        if k == 1:
            centers.append(cv.mean(0, keepdims=True))
        else:
            km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, batch_size=2048)
            centers.append(km.fit(cv).cluster_centers_)
    c = np.concatenate(centers)
    print(f'  Centroids: {len(c)}')
    return torch.FloatTensor(c).to(device)


@torch.no_grad()
def class_prototype_cosine_similarity(model, X, y, class_a, class_b, device='cpu', batch=512):
    model.eval()
    emb, labels = [], []
    y = np.asarray(y)
    for i in range(0, len(X), batch):
        x = torch.FloatTensor(X[i:i+batch]).to(device)
        emb.append(model.get_embed(x).detach().cpu())
        labels.append(torch.LongTensor(y[i:i+batch]))
    emb = torch.cat(emb, dim=0)
    labels = torch.cat(labels, dim=0)
    mask_a = labels == int(class_a)
    mask_b = labels == int(class_b)
    if not mask_a.any() or not mask_b.any():
        raise ValueError('Both classes must have at least one sample')
    proto_a = F.normalize(emb[mask_a].mean(dim=0), dim=0)
    proto_b = F.normalize(emb[mask_b].mean(dim=0), dim=0)
    return float(torch.dot(proto_a, proto_b).item())


def calibrate(model, X_normal_calibration, target_fpr, device, centroids,
              hybrid_meta=None):
    """Calibrate score thresholds using only held-out benign/Normal rows."""
    if len(X_normal_calibration) == 0:
        raise ValueError('threshold calibration requires at least one Normal row')
    scores = _batch_scores(
        model, X_normal_calibration, device, centroids, hybrid_meta=hybrid_meta
    )
    scores['gradbp_l2'] = _batch_gradbp(model, X_normal_calibration, device)
    thr = {}
    print(f'\n  Thresholds @ FPR={target_fpr*100:.0f}%')
    for m, arr in scores.items():
        t = float(np.quantile(arr, 1.0-target_fpr))
        thr[m] = t
        print(f'    {m:<16} thr={t:.6f}  actual_FPR={(arr>t).mean():.4f}')
    return thr


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
def compute_adaptive_threshold_trace(model, X_test, y_test, normal_idx,
                                     seed_re_scores, target_fpr, device,
                                     window_size=1000):
    ae_scores = _batch_scores(model, X_test, device)['ae_re']
    tracker = AdaptiveThreshold(window_size=window_size, target_fpr=target_fpr)
    tracker.update(seed_re_scores)
    thresholds, decisions = [], []
    for re_score, label in zip(ae_scores, y_test):
        decisions.append(tracker(float(re_score)))
        if int(label) == int(normal_idx):
            tracker.update(np.asarray([re_score], dtype=np.float32))
        thresholds.append(tracker.threshold)
    return {
        'ae_scores': ae_scores,
        'thresholds': np.asarray(thresholds, dtype=np.float32),
        'decisions': np.asarray(decisions, dtype=bool),
        'final_threshold': float(tracker.threshold),
    }


def roc_operating_point_at_fpr(fpr_values, tpr_values, thresholds, target_fpr):
    """Return the highest-TPR ROC point that does not exceed the FPR budget."""
    fpr_values = np.asarray(fpr_values, dtype=float)
    tpr_values = np.asarray(tpr_values, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    target_fpr = float(target_fpr)
    if not np.isfinite(target_fpr) or not 0.0 <= target_fpr <= 1.0:
        raise ValueError('target_fpr must be finite and within [0, 1]')
    if not (len(fpr_values) == len(tpr_values) == len(thresholds)):
        raise ValueError('ROC arrays must have identical lengths')
    if not len(fpr_values):
        raise ValueError('ROC arrays must not be empty')
    if not np.isfinite(fpr_values).all() or not np.isfinite(tpr_values).all():
        raise ValueError('ROC FPR/TPR values must be finite')
    if bool(((fpr_values < 0.0) | (fpr_values > 1.0)).any()) or bool(
        ((tpr_values < 0.0) | (tpr_values > 1.0)).any()
    ):
        raise ValueError('ROC FPR/TPR values must be within [0, 1]')
    eligible = np.flatnonzero(fpr_values <= target_fpr)
    if eligible.size == 0:
        raise ValueError('ROC curve has no point within the requested FPR budget')
    eligible_tpr = tpr_values[eligible]
    best_tpr = eligible_tpr.max()
    best = eligible[np.flatnonzero(eligible_tpr == best_tpr)]
    # When multiple thresholds have the same TPR, report the point closest to
    # (but never above) the requested budget.
    index = int(best[np.argmax(fpr_values[best])])
    return {
        'target_fpr': target_fpr,
        'achieved_fpr': float(fpr_values[index]),
        'tpr': float(tpr_values[index]),
        'threshold': float(thresholds[index]),
    }


def evaluate_classifier(model, X_te, y_te, label_names, device):
    model.eval()
    preds, probs_list = [], []
    with torch.no_grad():
        for i in range(0,len(X_te),512):
            x = torch.FloatTensor(X_te[i:i+512]).to(device)
            lg, _ = model(x)
            preds.append(lg.argmax(1).cpu().numpy())
            probs_list.append(torch.softmax(lg,dim=-1).cpu().numpy())
    preds = np.concatenate(preds)
    probs = np.concatenate(probs_list)
    print(classification_report(
        y_te,
        preds,
        labels=list(range(len(label_names))),
        target_names=label_names,
        digits=4,
        zero_division=0,
    ))
    ni = label_names.index('Normal') if 'Normal' in label_names else 0
    bin_ = (y_te!=ni).astype(int)
    bin_pred = (preds!=ni).astype(int)
    score = 1-probs[:,ni]
    try: auc = roc_auc_score(bin_, score)
    except: auc = 0.5
    print(f'  AUC(Normal vs Attack): {auc:.4f}')
    return {
        'preds': preds,
        'probs': probs,
        'binary_attack_detection_accuracy': float(accuracy_score(bin_, bin_pred)),
        'binary_attack_detection_auroc': float(auc),
        'known_multiclass_accuracy': float(accuracy_score(y_te, preds)),
        'known_macro_f1': float(f1_score(
            y_te, preds, labels=list(range(len(label_names))), average='macro',
            zero_division=0,
        )),
        # Backward-compatible alias. New reports identify it as deprecated.
        'auc': float(auc),
    }


def collect_ood_scores(model, X, device, centroids):
    """Collect target-independent base scores once for efficient LOFO evaluation."""
    scores = _batch_scores(model, X, device, centroids, hybrid_meta=None)
    scores['gradbp_l2'] = _batch_gradbp(model, X, device)
    return scores


def add_hybrid_scores(scores, hybrid_meta):
    out = {name: np.asarray(values) for name, values in scores.items()}
    out['hybrid'] = compute_hybrid_meta_score(
        out['ae_re'], out['softmax'], hybrid_meta
    )
    return out


def _alert_policy_metrics(known_scores, threshold, y_known=None,
                          known_predictions=None, normal_idx=0):
    decisions = np.asarray(known_scores) > float(threshold)
    metrics = {
        'known_false_unknown_rate': float(decisions.mean()) if len(decisions) else None,
        'normal_ood_fpr': None,
        'total_alert_fpr': None,
    }
    if y_known is None:
        return metrics
    y_known = np.asarray(y_known)
    normal_mask = y_known == int(normal_idx)
    if not bool(normal_mask.any()):
        return metrics
    metrics['normal_ood_fpr'] = float(decisions[normal_mask].mean())
    if known_predictions is not None:
        classifier_alert = np.asarray(known_predictions) != int(normal_idx)
        metrics['total_alert_fpr'] = float(
            (decisions[normal_mask] | classifier_alert[normal_mask]).mean()
        )
    return metrics


def evaluate_zero_day_from_scores(sk, sz, y_zd, thr, selected_method='hybrid',
                                  y_known=None, known_predictions=None,
                                  normal_idx=0):
    """Report a pre-selected OOD method without tuning on target OOD labels."""
    if selected_method not in sk or selected_method not in sz:
        raise ValueError(f'selected OOD method is unavailable: {selected_method}')
    true = np.concatenate([np.zeros(len(sk[selected_method])), np.ones(len(sz[selected_method]))])
    results = {}
    print(f'\n  {"Method":<16} {"AUC":>8} {"TPR@1%":>10} {"TPR@5%":>10}')
    print(f'  {"-"*48}')
    for method in ['gradbp_l2', 'hybrid', 'ae_re', 'softmax', 'fv_cluster']:
        if method not in sk or method not in sz:
            continue
        scores_all = np.concatenate([sk[method], sz[method]])
        if len(np.unique(true)) < 2:
            auc = None
            auprc = None
            fpr_values = np.asarray([0.0])
            tpr_values = np.asarray([0.0])
            roc_thresholds = np.asarray([np.inf])
        else:
            auc = float(roc_auc_score(true, scores_all))
            auprc = float(average_precision_score(true, scores_all))
            fpr_values, tpr_values, roc_thresholds = roc_curve(true, scores_all)
        op_1 = roc_operating_point_at_fpr(
            fpr_values, tpr_values, roc_thresholds, 0.01
        )
        op_5 = roc_operating_point_at_fpr(
            fpr_values, tpr_values, roc_thresholds, 0.05
        )
        tpr_1, tpr_5 = op_1['tpr'], op_5['tpr']
        auc_display = float('nan') if auc is None else auc
        print(f'  {method:<16} {auc_display:>8.4f} {tpr_1:>10.4f} {tpr_5:>10.4f}')
        results[method] = {
            'ood_auroc': auc,
            'ood_auprc': auprc,
            'ood_tpr_at_1pct_fpr': op_1,
            'ood_tpr_at_5pct_fpr': op_5,
            # Deprecated aliases retained for old plot/report consumers.
            'auc': auc,
            'tpr_1': tpr_1,
            'tpr_5': tpr_5,
            'fpr': fpr_values.tolist(),
            'tpr': tpr_values.tolist(),
        }

    threshold = thr[selected_method]
    print(f'\n  Per-class recall [{selected_method}@thr={threshold:.5f}]:')
    per_class = {}
    for class_name in np.unique(y_zd):
        mask = np.asarray(y_zd) == class_name
        support = int(mask.sum())
        detected = int((np.asarray(sz[selected_method])[mask] > threshold).sum())
        recall = detected / support if support else 0.0
        per_class[str(class_name)] = {'n': support, 'support': support, 'recall': recall}
        bar = '#'*int(recall*20) + '.'*(20-int(recall*20))
        print(f'    {str(class_name):<30} [{bar}] {recall:.1%}  (n={support:,})')

    results['_per_class'] = per_class
    results['_selected_method'] = selected_method
    results['_best_method'] = selected_method
    selected = results[selected_method]
    results['ood_metrics'] = {
        'ood_auroc': selected['ood_auroc'],
        'ood_auprc': selected['ood_auprc'],
        'ood_tpr_at_1pct_fpr': selected['ood_tpr_at_1pct_fpr'],
        'ood_tpr_at_5pct_fpr': selected['ood_tpr_at_5pct_fpr'],
        **_alert_policy_metrics(
            sk[selected_method],
            threshold,
            y_known=y_known,
            known_predictions=known_predictions,
            normal_idx=normal_idx,
        ),
    }
    results['_deprecated_metric_aliases'] = {
        'auc': 'ood_auroc',
        'tpr_1': 'ood_tpr_at_1pct_fpr.tpr',
        'tpr_5': 'ood_tpr_at_5pct_fpr.tpr',
        '_best_method': '_selected_method',
    }
    return results


def evaluate_zero_day(model, X_kn, y_kn, X_zd, y_zd, thr, centroids, device,
                      hybrid_meta=None, selected_method='hybrid'):
    print(f'\n{"="*65}')
    print(f'ZERO-DAY DETECTION  |  Known={len(X_kn):,}  ZD={len(X_zd):,}')
    print(f'{"="*65}')
    sk = add_hybrid_scores(collect_ood_scores(model, X_kn, device, centroids), hybrid_meta)
    sz = add_hybrid_scores(collect_ood_scores(model, X_zd, device, centroids), hybrid_meta)
    results = evaluate_zero_day_from_scores(
        sk, sz, y_zd, thr, selected_method=selected_method
    )
    results['_scores_known']= sk
    results['_scores_zd']   = sz
    return results


# ═══════════════════════════════════════════════════════════════
