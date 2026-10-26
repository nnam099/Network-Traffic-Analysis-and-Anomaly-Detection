"""Matplotlib plotting functions for IDS v14 training, ROC, decision-space and drift reports."""

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def _attack_probs_batch(model, X, device, batch=512):
    model.eval()
    prob_attack = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            x = torch.FloatTensor(X[i : i + batch]).to(device)
            lg, _ = model(x)
            pr = torch.softmax(lg, dim=-1)
            prob_attack.append((1 - pr[:, 0]).cpu().numpy())
    return np.concatenate(prob_attack)


def plot_soc_decision_space(
    model, X_kn, y_kn_labels, X_zd, p_thr, re_thr, device, save_path, label_names, n_sample=5000, seed=42
):
    model.eval()
    rng = np.random.default_rng(seed)

    def sample(X, y, n):
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        return X[idx], y[idx]

    normal_idx = label_names.index("Normal") if "Normal" in label_names else 0
    mask_benign = y_kn_labels == normal_idx
    mask_katk = ~mask_benign

    Xb, _ = sample(X_kn[mask_benign], y_kn_labels[mask_benign], n_sample)
    Xka, _ = sample(X_kn[mask_katk], y_kn_labels[mask_katk], n_sample)
    Xzd, _ = sample(X_zd, np.zeros(len(X_zd)), n_sample)

    def get_xy(X):
        prob = _attack_probs_batch(model, X, device)
        with torch.no_grad():
            re_list = []
            for i in range(0, len(X), 512):
                x = torch.FloatTensor(X[i : i + 512]).to(device)
                re_list.append(model.ae.recon_error(x).cpu().numpy())
        re = np.concatenate(re_list)
        return prob, re

    pb, reb = get_xy(Xb)
    pka, reka = get_xy(Xka)
    pzd, rezd = get_xy(Xzd)

    fig, ax = plt.subplots(figsize=(11, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.scatter(pb, reb, c="#00BFFF", s=8, alpha=0.35, label="Benign", zorder=2)
    ax.scatter(pka, reka, c="#FF6B6B", s=8, alpha=0.45, label="Known Attack", zorder=3)
    ax.scatter(pzd, rezd, c="#FFA500", s=8, alpha=0.45, label="Zero-Day Attack", zorder=4)

    ax.axvline(p_thr, color="white", linestyle="--", lw=1.2, label=f"P-Thr {p_thr:.2f}")
    ax.axhline(re_thr, color="white", linestyle=":", lw=1.2, label=f"RE-Thr {re_thr:.3f}")

    ax.set_yscale("log")
    ax.set_xlabel("Probability of Attack (Classifier)", color="white", fontsize=12)
    ax.set_ylabel("Reconstruction Error (Anomaly)", color="white", fontsize=12)
    ax.set_title("V14.0 - SOC Decision Space", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")
    ax.legend(facecolor="#1a1f2e", edgecolor="#555", labelcolor="white", fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] SOC Decision Space -> {save_path}")


def plot_per_class_proper(label_names, y_test, preds, per_cls_zd, zd_classes_order, save_path, normal_idx=0):
    known_names, known_recalls = [], []
    for i, name in enumerate(label_names):
        mask = y_test == i
        if mask.sum() == 0:
            continue
        det = (preds[mask] != normal_idx).mean() * 100
        known_names.append(name.upper())
        known_recalls.append(det)

    zd_names = [c.upper() for c in zd_classes_order]
    zd_recalls = [per_cls_zd[c]["recall"] * 100 for c in zd_classes_order]

    fig, axes = plt.subplots(1, 2, figsize=(18, max(7, max(len(known_names), len(zd_names)) * 0.7)))
    fig.patch.set_facecolor("#0d1117")

    def draw_panel(ax, names, vals, color, title):
        ax.set_facecolor("#0d1117")
        bars = ax.barh(names, vals, color=color, height=0.6)
        for bar, val in zip(bars, vals):
            ax.text(
                min(bar.get_width() + 1, 108),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%",
                va="center",
                color="white",
                fontsize=10,
            )
        ax.set_xlim(0, 115)
        ax.set_xlabel("Detection Rate (%)", color="white", fontsize=11)
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    draw_panel(axes[0], known_names, known_recalls, "#00A8E8", "Known Attacks Detection Rate (%)\n(Supervised Head)")
    draw_panel(axes[1], zd_names, zd_recalls, "#FFA500", "Zero-Day Attacks Detection Rate (%)\n(Hybrid Anomaly)")

    fig.suptitle("V14.0 - PER-CLASS DETECTION PERFORMANCE", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] Per-class -> {save_path}")


def plot_training_curve(history, save_path):
    epochs = list(range(1, len(history) + 1))
    losses = [h["loss"] for h in history]
    aucs = [h["val_auc"] for h in history]
    accs = [h["val_acc"] for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0d1117")
    for ax, d, t, c in zip(
        axes, [losses, aucs, accs], ["Train Loss", "Val AUC", "Val Accuracy"], ["#FF6B6B", "#00BFFF", "#FFA500"]
    ):
        ax.set_facecolor("#0d1117")
        ax.plot(epochs, d, c=c, lw=2)
        ax.set_title(t, color="white", fontsize=12)
        ax.tick_params(colors="white")
        ax.set_xlabel("Epoch", color="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    fig.suptitle("V14.0 - Training Curves", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] Training curve -> {save_path}")


def plot_threshold_drift(threshold_trace, save_path):
    thresholds = np.asarray(threshold_trace.get("thresholds", []), dtype=np.float32)
    if thresholds.size == 0:
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(np.arange(len(thresholds)), thresholds, color="#FFA500", lw=1.8)
    ax.set_xlabel("Test timeline index", color="white", fontsize=11)
    ax.set_ylabel("AE threshold", color="white", fontsize=11)
    ax.set_title("V14.0 - Adaptive AE Threshold Drift", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.grid(color="#26313b", linestyle="--", linewidth=0.6, alpha=0.7)
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] Threshold drift -> {save_path}")


def plot_roc_curves(zd_results, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    method_colors = {
        "gradbp_l2": "#FF6B6B",
        "hybrid": "#FFA500",
        "ae_re": "#00BFFF",
        "softmax": "#DDA0DD",
        "fv_cluster": "#F0E68C",
    }
    for m, r in zd_results.items():
        if m.startswith("_") or "fpr" not in r:
            continue
        fpr = np.array(r["fpr"])
        tpr = np.array(r["tpr"])
        auc = r["auc"]
        c = method_colors.get(m, "white")
        ls = "--" if m == "softmax" else "-"
        ax.plot(fpr, tpr, c=c, lw=2, ls=ls, label=f"{m}  AUC={auc:.4f}")

    ax.plot([0, 1], [0, 1], "--", color="#555", lw=1)
    ax.set_xlabel("False Positive Rate", color="white", fontsize=12)
    ax.set_ylabel("True Positive Rate", color="white", fontsize=12)
    ax.set_title("V14.0 - Zero-Day Detection ROC", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")
    ax.legend(facecolor="#1a1f2e", edgecolor="#555", labelcolor="white", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] ROC curves -> {save_path}")


def plot_confusion_matrix(y_true, y_pred, label_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    im = ax.imshow(cm_pct, cmap="Blues")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(colors="white")

    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right", color="white", fontsize=10)
    ax.set_yticklabels(label_names, color="white", fontsize=10)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            v = cm_pct[i, j]
            ax.text(j, i, f"{v:.1f}%", ha="center", va="center", color="white" if v < 50 else "#111", fontsize=9)

    ax.set_xlabel("Predicted", color="white", fontsize=12)
    ax.set_ylabel("True", color="white", fontsize=12)
    ax.set_title("V14.0 - Confusion Matrix (%)", color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [Plot] Confusion matrix -> {save_path}")


# ═══════════════════════════════════════════════════════════════
