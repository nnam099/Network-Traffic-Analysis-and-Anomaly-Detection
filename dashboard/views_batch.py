from __future__ import annotations

import pandas as pd
import streamlit as st

from ui_safety import render_safety_notice


def render_batch_safety_notice() -> None:
    render_safety_notice()
    st.caption(
        "Batch CSV analysis is best used for prioritization. Low feature coverage or missing flow counters can change OOD rates."
    )


def render_bulk_detection_summary(result_df: pd.DataFrame) -> None:
    total = len(result_df)
    zd_cnt = int(result_df["is_zeroday"].sum())
    st.metric("OOD candidates", zd_cnt)
    st.metric("OOD candidate rate", f"{(zd_cnt / total * 100):.2f}%" if total else "0.00%")
    verdict_counts = (
        result_df["detection"]
        .value_counts()
        .reindex(["Normal", "Known-Attack", "Zero-Day Candidate"], fill_value=0)
        .rename_axis("Label")
        .reset_index(name="Count")
    )
    st.dataframe(verdict_counts, use_container_width=True, hide_index=True)


def compute_normal_fpr_metrics(result_df: pd.DataFrame) -> dict[str, float] | None:
    required = {"ground_truth", "is_zeroday", "classifier_class"}
    if not required.issubset(result_df.columns):
        return None
    normal_rows = result_df.loc[result_df["ground_truth"].astype(str) == "Normal"]
    if normal_rows.empty:
        return None
    verdict_col = "detection" if "detection" in normal_rows.columns else "predicted_class"
    if verdict_col not in normal_rows.columns:
        return None
    normal_ood = normal_rows["is_zeroday"].astype(bool)
    known_classifier_alert = normal_rows["classifier_class"].astype(str).str.strip().str.lower() != "normal"
    normal_alert = normal_ood | known_classifier_alert
    normal_overlap = normal_ood & known_classifier_alert
    verdict_alert = normal_rows[verdict_col].isin(["Known-Attack", "Zero-Day Candidate"])
    if not normal_alert.equals(verdict_alert):
        raise ValueError("dashboard verdict is inconsistent with the OOD/classifier alert union")
    return {
        "normal_ood_fpr": float(normal_ood.mean()),
        "normal_alert_fpr": float(normal_alert.mean()),
        "known_classifier_fpr": float(known_classifier_alert.mean()),
        "normal_ood_classifier_overlap_rate": float(normal_overlap.mean()),
        "normal_ood_classifier_overlap_count": int(normal_overlap.sum()),
    }


def render_ground_truth_summary(result_df: pd.DataFrame) -> None:
    if "ground_truth" not in result_df.columns or not result_df["ground_truth"].astype(str).str.len().any():
        return
    gt_counts = (
        result_df["ground_truth"]
        .value_counts()
        .reindex(["Normal", "Known-Attack"], fill_value=0)
        .rename_axis("Ground Truth")
        .reset_index(name="Count")
    )
    comparable = result_df["correct_vs_ground_truth"].dropna()
    gt_acc = float(comparable.mean()) if len(comparable) else 0.0
    st.metric("Accuracy vs CSV Label", f"{gt_acc * 100:.2f}%")
    fpr_metrics = compute_normal_fpr_metrics(result_df)
    if fpr_metrics is not None:
        c1, c2, c3 = st.columns(3)
        c1.metric("Normal OOD FPR", f"{fpr_metrics['normal_ood_fpr'] * 100:.2f}%")
        c2.metric("Normal alert FPR", f"{fpr_metrics['normal_alert_fpr'] * 100:.2f}%")
        c3.metric("Known-classifier FPR", f"{fpr_metrics['known_classifier_fpr'] * 100:.2f}%")
        st.caption(
            "OOD FPR counts Normal rows flagged as OOD; alert FPR counts any attack alert. "
            "Known-classifier FPR isolates Normal rows assigned to a known attack class. "
            f"OOD/classifier overlap: {fpr_metrics['normal_ood_classifier_overlap_count']:,} rows."
        )
    st.dataframe(gt_counts, use_container_width=True, hide_index=True)
