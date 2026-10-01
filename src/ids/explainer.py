# -*- coding: utf-8 -*-
"""
explainer.py — SHAP Explainer v15  (tích hợp từ new/explainer.py)
===================================================================
Cải tiến so với v14:
  [FIX]  Scale đúng cách tại explain_alert, không trong __init__
  [NEW]  GradientExplainer (DeepSHAP) nhanh hơn KernelSHAP ~10×
  [NEW]  batch_explain(): giải thích nhiều alert cùng lúc
  [NEW]  attention_importance(): lấy attention weights từ backbone
  [NEW]  plot_bar(): dark-theme bar chart dễ đọc hơn waterfall
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Optional


class SHAPExplainer:
    """
    Giải thích alert bằng SHAP.
    Tự động chọn GradientExplainer (nhanh) hoặc fallback sang KernelExplainer.

    Parameters
    ----------
    model          : IDSModel đã train (eval mode)
    scaler         : RobustScaler đã fit
    feature_names  : list tên feature
    background_data: numpy array raw (chưa scale) ~100-200 samples để tính baseline
    device         : 'cpu' hoặc 'cuda'
    use_gradient   : True = thử GradientExplainer trước (nhanh hơn 10×)
    """

    def __init__(self, model, scaler, feature_names: List[str],
                 background_data: np.ndarray, device: str = 'cpu',
                 use_gradient: bool = True):
        import shap
        self.model         = model
        self.scaler        = scaler
        self.feature_names = feature_names
        self.device        = device
        self._shap         = shap

        # Scale background 1 lần duy nhất ở đây
        bg_scaled = scaler.transform(background_data)
        self._bg_scaled = bg_scaled

        def _predict(X_scaled):
            t = torch.FloatTensor(X_scaled).to(device)
            with torch.no_grad():
                out    = model(t)
                logits = out[0] if isinstance(out, tuple) else out
                probs  = torch.softmax(logits, dim=1)
            return probs.cpu().numpy()

        self.predict_fn = _predict

        if use_gradient:
            try:
                bg_tensor = torch.FloatTensor(bg_scaled).to(device)

                class _Wrapper(torch.nn.Module):
                    def __init__(self, m): super().__init__(); self.m = m
                    def forward(self, x):
                        out = self.m(x)
                        logits = out[0] if isinstance(out, tuple) else out
                        return torch.softmax(logits, dim=1)

                self.explainer = shap.GradientExplainer(
                    _Wrapper(model).to(device), bg_tensor
                )
                self._mode = 'gradient'
                print('[SHAP] GradientExplainer initialized (fast mode)')
            except Exception as e:
                print(f'[SHAP] GradientExplainer failed ({e}), fallback to Kernel')
                self._init_kernel(shap, bg_scaled)
        else:
            self._init_kernel(shap, bg_scaled)

    def _init_kernel(self, shap, bg_scaled):
        self.explainer = shap.KernelExplainer(
            self.predict_fn,
            shap.kmeans(bg_scaled, min(50, len(bg_scaled)))
        )
        self._mode = 'kernel'
        print('[SHAP] KernelExplainer initialized')

    def explain_alert(self, alert_features_raw: np.ndarray, top_k: int = 10) -> dict:
        """
        Giải thích 1 alert cụ thể.

        Parameters
        ----------
        alert_features_raw : numpy array shape (1, n_features) — raw, chưa scale
        top_k              : số feature quan trọng nhất muốn hiển thị

        Returns
        -------
        dict chứa: shap_values, top_features, predicted_class_idx,
                   class_probabilities, summary_text
        """
        alert_scaled = self.scaler.transform(alert_features_raw)

        if self._mode == 'gradient':
            t         = torch.FloatTensor(alert_scaled).to(self.device)
            shap_vals = self.explainer.shap_values(t)
            probs     = self.predict_fn(alert_scaled)[0]
            pred_class = int(np.argmax(probs))
            if isinstance(shap_vals, list):
                sv = np.asarray(shap_vals[pred_class]).flatten()
            else:
                arr = np.asarray(shap_vals)
                if arr.ndim == 3:
                    # SHAP can return (samples, features, outputs) for multi-class models.
                    sv = arr[0, :, pred_class]
                elif arr.ndim == 2:
                    sv = arr[0]
                else:
                    sv = arr.flatten()
        else:
            shap_vals  = self.explainer.shap_values(alert_scaled, nsamples=100)
            probs      = self.predict_fn(alert_scaled)[0]
            pred_class = int(np.argmax(probs))
            sv         = shap_vals[pred_class][0]

        top_indices  = np.argsort(np.abs(sv))[::-1][:top_k]
        fnames       = self.feature_names
        top_features = []
        for i in top_indices:
            fn   = fnames[i] if i < len(fnames) else f'feat_{i}'
            sval = float(sv[i])
            fval = float(alert_features_raw[0][i]) if i < alert_features_raw.shape[1] else 0.
            top_features.append((fn, sval, fval))

        lines = []
        for fname, sval, fval in top_features:
            d = 'tăng nguy cơ' if sval > 0 else 'giảm nguy cơ'
            lines.append(f'  - {fname}: giá_trị={fval:.4f}, SHAP={sval:+.4f} ({d})')

        return {
            'shap_values':         sv,
            'top_features':        top_features,
            'predicted_class_idx': pred_class,
            'class_probabilities': probs.tolist(),
            'summary_text':        '\n'.join(lines),
        }

    def batch_explain(self, X_raw: np.ndarray, top_k: int = 5) -> List[dict]:
        """[NEW] Giải thích nhiều alert cùng lúc."""
        return [self.explain_alert(X_raw[i:i+1], top_k) for i in range(len(X_raw))]

    def attention_importance(self, alert_features_raw: np.ndarray) -> np.ndarray:
        """
        [NEW] Lấy attention weights từ backbone AttentionGate (v15).
        Nhanh hơn SHAP nhiều — phù hợp cho real-time dashboard.

        Returns zero array nếu model không có AttentionGate (tương thích v14).
        """
        if not hasattr(self.model, 'get_attention'):
            return np.zeros(len(self.feature_names))
        alert_scaled = self.scaler.transform(alert_features_raw)
        t = torch.FloatTensor(alert_scaled).to(self.device)
        with torch.no_grad():
            attn = self.model.get_attention(t)
        return attn.cpu().numpy().flatten()

    def attention_summary_text(self, alert_features_raw: np.ndarray, top_k: int = 8) -> str:
        """
        [NEW] Tạo text tóm tắt attention importance để đưa vào LLM prompt.
        Nhanh hơn SHAP → dùng cho real-time triage.
        """
        attn = self.attention_importance(alert_features_raw)
        if attn.sum() == 0:
            return 'N/A (model không có AttentionGate)'
        top_idx = np.argsort(attn)[::-1][:top_k]
        fnames  = self.feature_names
        lines   = []
        for i in top_idx:
            fn   = fnames[i] if i < len(fnames) else f'feat_{i}'
            fval = float(alert_features_raw[0][i]) if i < alert_features_raw.shape[1] else 0.
            lines.append(f'  - {fn}: attention={attn[i]:.4f}, giá_trị={fval:.4f}')
        return '\n'.join(lines)

    def plot_waterfall(self, alert_features_raw: np.ndarray,
                       save_path: str = 'shap_waterfall.png') -> str:
        """Vẽ waterfall chart — đẹp để đưa vào report."""
        result     = self.explain_alert(alert_features_raw)
        sv         = result['shap_values']
        pred_class = result['predicted_class_idx']
        base_val   = 0.0 if self._mode == 'gradient' else \
                     float(self.explainer.expected_value[pred_class])
        alert_scaled = self.scaler.transform(alert_features_raw)
        explanation  = self._shap.Explanation(
            values=sv, base_values=base_val,
            data=alert_scaled[0], feature_names=self.feature_names,
        )
        plt.figure(figsize=(10, 6))
        self._shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[SHAP] Waterfall saved → {save_path}')
        return save_path

    def plot_bar(self, alert_features_raw: np.ndarray,
                 save_path: str = 'shap_bar.png') -> str:
        """
        [NEW] Dark-theme bar chart — dễ đọc hơn waterfall.
        Phù hợp để hiển thị trên SOC dashboard.
        """
        result = self.explain_alert(alert_features_raw)
        top    = result['top_features'][:12]
        names  = [f[0] for f in top]
        vals   = [f[1] for f in top]
        colors = ['#FF6B6B' if v > 0 else '#00BFFF' for v in vals]

        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.barh(names[::-1], vals[::-1], color=colors[::-1], height=0.6)
        ax.axvline(0, color='white', lw=0.8)
        ax.set_xlabel('SHAP Value', color='white')
        ax.set_title('Feature Importance (SHAP)', color='white', fontweight='bold')
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close()
        print(f'[SHAP] Bar chart saved → {save_path}')
        return save_path
