"""Training losses for class imbalance, contrastive separation and autoencoder reconstruction."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(
        self,
        n_classes,
        gamma=2.5,
        label_smooth=0.05,
        class_weights=None,
        dos_class_idx=None,
        recon_class_idx=None,
        recon_dos_penalty=2.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.ls = label_smooth
        self.nc = n_classes
        self.w = class_weights
        self.dos_idx = dos_class_idx
        self.recon_idx = recon_class_idx
        self.recon_dos_penalty = float(recon_dos_penalty)

    def forward(self, logits, labels):
        ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=self.ls, weight=self.w)
        log_probs = F.log_softmax(logits, dim=-1)
        pt = log_probs.gather(1, labels.view(-1, 1)).squeeze(1).exp()
        loss = (1 - pt) ** self.gamma * ce
        if self.dos_idx is not None and self.recon_idx is not None:
            preds = logits.argmax(dim=1)
            recon_as_dos = (labels == self.recon_idx) & (preds == self.dos_idx)
            dos_as_recon = (labels == self.dos_idx) & (preds == self.recon_idx)
            penalty_mask = recon_as_dos | dos_as_recon
            if penalty_mask.any():
                mult = torch.ones_like(loss)
                mult[penalty_mask] = self.recon_dos_penalty
                loss = loss * mult
        return loss.mean()


class SupConLoss(nn.Module):
    def __init__(
        self,
        T=0.15,
        dos_class_idx=None,
        recon_class_idx=None,
        hard_negative_weight=0.20,
        hard_negative_topk=64,
        hard_negative_margin=0.20,
    ):
        super().__init__()
        self.T = T
        self.dos_idx = dos_class_idx
        self.recon_idx = recon_class_idx
        self.hn_weight = float(hard_negative_weight)
        self.hn_topk = int(hard_negative_topk)
        self.hn_margin = float(hard_negative_margin)

    def forward(self, feats, labels):
        B = feats.shape[0]
        sim = torch.matmul(feats, feats.T) / self.T
        sim = sim - sim.detach().max()
        lmat = labels.view(-1, 1)
        pos = (lmat == lmat.T).float().fill_diagonal_(0)
        eye = torch.eye(B, device=feats.device, dtype=torch.bool)
        denom = (torch.exp(sim) * (~eye).float()).sum(1, keepdim=True).clamp(min=1e-8)
        lp = sim - torch.log(denom)
        cnt = pos.sum(1).clamp(min=1e-8)
        loss = -(pos * lp).sum(1) / cnt
        valid = pos.sum(1) > 0
        hard_neg = self._recon_dos_hard_negative_loss(feats, labels)
        if not valid.any():
            return hard_neg
        return loss[valid].mean() + hard_neg

    def _recon_dos_hard_negative_loss(self, feats, labels):
        if self.dos_idx is None or self.recon_idx is None or self.hn_weight <= 0:
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        lmat = labels.view(-1, 1)
        recon_dos = ((lmat == self.recon_idx) & (lmat.T == self.dos_idx)) | (
            (lmat == self.dos_idx) & (lmat.T == self.recon_idx)
        )
        if not recon_dos.any():
            return torch.tensor(0.0, device=feats.device, requires_grad=True)
        cos = torch.matmul(feats, feats.T)
        hard_scores = cos[recon_dos]
        if hard_scores.numel() > self.hn_topk:
            hard_scores = torch.topk(hard_scores, self.hn_topk).values
        penalty = F.softplus((hard_scores - self.hn_margin) / self.T).mean()
        return self.hn_weight * penalty


class IDSLoss(nn.Module):
    def __init__(
        self,
        n_classes,
        lambda_con=0.3,
        focal_gamma=2.0,
        dos_class_idx=None,
        dos_weight=5.0,
        recon_class_idx=None,
        recon_dos_penalty=2.0,
        class_weight_overrides=None,
        n_features=None,
        device="cpu",
    ):
        super().__init__()
        w = torch.ones(n_classes, device=device)
        if dos_class_idx is not None:
            w[dos_class_idx] = dos_weight
        for class_idx, multiplier in (class_weight_overrides or {}).items():
            w[int(class_idx)] = float(multiplier)
        w = w / w.mean()
        self.focal = FocalLoss(
            n_classes,
            gamma=focal_gamma,
            label_smooth=0.05,
            class_weights=w.to(device),
            dos_class_idx=dos_class_idx,
            recon_class_idx=recon_class_idx,
            recon_dos_penalty=recon_dos_penalty,
        )
        self.con = SupConLoss(T=0.15, dos_class_idx=dos_class_idx, recon_class_idx=recon_class_idx)
        self.lam = lambda_con
        self.ae_loss_fn = nn.MSELoss()

    def forward(self, logits, labels, embeds, x, x_hat):
        lf = self.focal(logits, labels)
        lc = self.con(embeds, labels)
        lae = self.ae_loss_fn(x_hat, x)
        lf = torch.nan_to_num(lf, nan=0.0, posinf=10.0)
        lc = torch.nan_to_num(lc, nan=0.0, posinf=10.0)
        lae = torch.nan_to_num(lae, nan=0.0, posinf=10.0)
        total = lf + self.lam * lc + 0.5 * lae
        return {"total": total, "focal": lf, "con": lc, "ae": lae}


# ═══════════════════════════════════════════════════════════════
