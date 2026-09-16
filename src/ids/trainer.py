"""Trainer loop, validation metrics, early stopping and confusion logging for IDS v14."""

import copy, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    tot_loss = ncorr = ntot = 0
    use_amp = (device == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast('cuda'):
                logits, _ = model(X)
                embeds    = model.get_embed(X)
                x_hat     = model.ae(X)
                losses    = criterion(logits, y, embeds, X, x_hat)
                loss      = losses['total']
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, _ = model(X)
            embeds    = model.get_embed(X)
            x_hat     = model.ae(X)
            losses    = criterion(logits, y, embeds, X, x_hat)
            loss      = losses['total']
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        loss_val = loss.item()
        if not np.isfinite(loss_val):
            continue
        tot_loss += loss_val * len(y)
        ncorr    += (logits.argmax(1)==y).sum().item()
        ntot     += len(y)

    if ntot == 0:
        return {'loss': float('nan'), 'acc': 0.0}
    return {'loss': tot_loss/ntot, 'acc': ncorr/ntot}


def normal_class_index(label_names):
    """Return the encoded index for Normal without assuming encoder order."""
    names = [str(name) for name in label_names]
    if 'Normal' not in names:
        raise ValueError('label mapping must include Normal')
    return names.index('Normal')


@torch.no_grad()
def eval_epoch(model, loader, device, normal_idx):
    model.eval()
    all_probs, all_labels = [], []
    for X, y in loader:
        logits, _ = model(X.to(device))
        all_probs.append(torch.softmax(logits,dim=-1).cpu().numpy())
        all_labels.append(y.numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    binary = (labels != normal_idx).astype(int)
    score  = 1 - probs[:,normal_idx]
    try:    auc = roc_auc_score(binary, score)
    except: auc = 0.5
    return {'auc':auc, 'acc':(probs.argmax(1)==labels).mean()}


@torch.no_grad()
def _collect_loader_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    for X, y in loader:
        logits, _ = model(X.to(device))
        preds.append(logits.argmax(1).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(labels), np.concatenate(preds)


def _format_class_name(label_names, idx):
    if label_names and idx < len(label_names):
        return str(label_names[idx])
    return f'Class_{idx}'


def log_top_confusions(model, loader, device, label_names=None, epoch=None, top_k=2):
    labels, preds = _collect_loader_predictions(model, loader, device)
    n_classes = len(label_names) if label_names else int(max(labels.max(), preds.max()) + 1)
    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))
    rows = []
    for true_idx in range(n_classes):
        row_total = int(cm[true_idx].sum())
        if row_total == 0:
            continue
        for pred_idx in range(n_classes):
            if pred_idx == true_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count <= 0:
                continue
            rows.append({
                'true': _format_class_name(label_names, true_idx),
                'pred': _format_class_name(label_names, pred_idx),
                'count': count,
                'row_pct': 100.0 * count / row_total,
            })
    rows = sorted(rows, key=lambda r: (r['count'], r['row_pct']), reverse=True)[:top_k]
    title = 'Top validation confusions'
    if epoch is not None:
        title += f' @ epoch {epoch}'
    print(f'\n  {title}')
    print(f'  {"True":<18} {"Pred":<18} {"Count":>7} {"Row%":>8}')
    print(f'  {"-"*55}')
    if not rows:
        print(f'  {"<none>":<18} {"<none>":<18} {0:>7} {0.0:>7.1f}%')
    for item in rows:
        print(f'  {item["true"]:<18} {item["pred"]:<18} {item["count"]:>7} {item["row_pct"]:>7.1f}%')
    return rows


def train(model, loaders, args, criterion, device, label_names=None, normal_idx=None):
    if normal_idx is None:
        if label_names is None:
            raise ValueError('normal_idx or label_names is required for validation AUC')
        normal_idx = normal_class_index(label_names)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def warmup_cos(ep):
        warmup=5
        if ep<warmup: return ep/warmup
        prog=(ep-warmup)/max(args.epochs-warmup,1)
        return 0.5*(1+np.cos(np.pi*prog))
    sched    = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cos)
    best_auc = 0.; best_sd = None; patience = 0; history = []

    print(f'\n{"="*65}')
    print(f'Training v14 | epochs={args.epochs} lr={args.lr} hidden={args.hidden}')
    print(f'{"="*65}')

    for ep in range(1, args.epochs+1):
        t0  = time.time()
        trm = train_epoch(model, loaders['train'], optimizer, criterion, device)
        vam = eval_epoch(model, loaders['val'], device, normal_idx)
        log_top_confusions(model, loaders['val'], device, label_names=label_names, epoch=ep, top_k=2)
        sched.step()
        is_best = vam['auc'] > best_auc
        if is_best:
            best_auc = vam['auc']
            best_sd  = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
        history.append({**trm, **{f'val_{k}':v for k,v in vam.items()}})
        star = '*' if is_best else ' '
        print(f'{star}[{ep:3d}/{args.epochs}] loss={trm["loss"]:.4f} acc={trm["acc"]:.4f}'
              f'  vAUC={vam["auc"]:.4f}  P={patience}  ({time.time()-t0:.1f}s)')
        if patience >= args.patience:
            print(f'  Early stop at ep {ep}')
            break

    if best_sd: model.load_state_dict(best_sd)
    print(f'\nBest Val AUC = {best_auc:.4f}')
    return model, history


class Trainer:
    """Thin object wrapper around the existing IDS v14 training loop."""

    def __init__(self, model, loaders, args, criterion, device, label_names=None, normal_idx=None):
        self.model = model
        self.loaders = loaders
        self.args = args
        self.criterion = criterion
        self.device = device
        self.label_names = label_names
        self.normal_idx = normal_idx

    def run(self):
        return train(
            self.model,
            self.loaders,
            self.args,
            self.criterion,
            self.device,
            label_names=self.label_names,
            normal_idx=self.normal_idx,
        )


# ═══════════════════════════════════════════════════════════════
