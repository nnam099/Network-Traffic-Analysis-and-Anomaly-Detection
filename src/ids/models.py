"""Neural network definitions for the IDS classifier, projection head and autoencoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class IDSBackbone(nn.Module):
    def __init__(self, n_features, hidden=256):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.res1 = ResBlock(hidden)
        self.res2 = ResBlock(hidden)
        self.res3 = ResBlock(hidden)
        self.hidden = hidden

    def forward(self, x):
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        return h


class AutoEncoder(nn.Module):
    def __init__(self, n_features, ae_hidden=128):
        super().__init__()
        mid = ae_hidden
        self.enc = nn.Sequential(
            nn.Linear(n_features, mid * 2),
            nn.GELU(),
            nn.Linear(mid * 2, mid),
            nn.GELU(),
            nn.Linear(mid, mid // 2),
        )
        self.dec = nn.Sequential(
            nn.Linear(mid // 2, mid),
            nn.GELU(),
            nn.Linear(mid, mid * 2),
            nn.GELU(),
            nn.Linear(mid * 2, n_features),
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat

    def recon_error(self, x):
        return (self.forward(x) - x).pow(2).mean(dim=-1)


class ProjectionHead(nn.Sequential):
    def __init__(self, hidden):
        super().__init__(nn.Linear(hidden, 128), nn.GELU(), nn.Linear(128, 64))


class IDSModel(nn.Module):
    def __init__(self, n_features, n_classes, hidden=256, ae_hidden=128):
        super().__init__()
        self.n_classes = n_classes
        self.backbone = IDSBackbone(n_features, hidden)
        self.classifier = nn.Linear(hidden, n_classes)
        self.proj_head = ProjectionHead(hidden)
        self.ae = AutoEncoder(n_features, ae_hidden)
        self.log_temp = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        fv = self.backbone(x)
        logits = self.classifier(fv)
        return logits, fv

    def get_embed(self, x):
        fv = self.backbone(x)
        return F.normalize(self.proj_head(fv), dim=-1)

    def attack_prob(self, x):
        logits, _ = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def energy_score(self, x):
        logits, _ = self.forward(x)
        T = self.log_temp.exp().clamp(0.5, 5.0)
        return -T * torch.logsumexp(logits / T, dim=-1)

    def gradbp_score(self, x):
        x = x.detach()
        fv = self.backbone(x)
        fv.requires_grad_(True)
        logits = self.classifier(fv)
        pred = logits.argmax(dim=-1)
        loss = F.cross_entropy(logits, pred)
        grad = torch.autograd.grad(loss, fv)[0]
        return grad.norm(dim=-1)

    def fv_cluster_score(self, x, centroids):
        _, fv = self.forward(x)
        fv_n = F.normalize(fv, dim=-1)
        cent_n = F.normalize(centroids, dim=-1)
        return torch.cdist(fv_n, cent_n).min(dim=-1).values

    def hybrid_score(self, x, centroids=None, hybrid_meta=None):
        if hybrid_meta is None:
            raise ValueError("hybrid_meta is required for learned hybrid scoring")
        re = self.ae.recon_error(x)
        probs = torch.softmax(self.forward(x)[0], dim=-1)
        cls_s = 1 - probs.max(dim=-1).values
        coef = torch.tensor(hybrid_meta["coef"], dtype=re.dtype, device=re.device)
        intercept = torch.tensor(float(hybrid_meta.get("intercept", 0.0)), dtype=re.dtype, device=re.device)
        return torch.sigmoid(intercept + coef[0] * re + coef[1] * cls_s)


# ═══════════════════════════════════════════════════════════════
