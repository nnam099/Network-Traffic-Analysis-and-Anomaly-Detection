from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from dashboard_runtime import (
    build_time_window_incidents,
    correlate_alerts,
    filter_alert_history,
)
from inference_runtime import risk_score
from ui_safety import render_safety_notice


STATUS_OPTIONS = ["new", "triaged", "investigating", "confirmed", "false_positive", "closed"]


def render_queue_view(
    history: list[dict],
    update_status: Callable[[str, str, str], None],
) -> None:
    render_safety_notice()
    if not history:
        st.markdown(
            """
            <div class="soc-panel">
                No alerts in the current analyst session. Use Analyze Alert to pull a sample, upload a CSV, or run a manual triage scenario.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown("### Active Alert Queue")
    f1, f2, f3, f4 = st.columns([0.9, 0.9, 0.9, 1.6])
    status_values = sorted({str(a.get("status", "new")) for a in history})
    severity_values = sorted({str(a.get("llm_severity", "N/A")).upper() for a in history})
    status_filter = f1.selectbox("Status", ["All"] + status_values)
    severity_filter = f2.selectbox("Severity", ["All"] + severity_values)
    ood_filter = f3.selectbox("OOD", ["All", "OOD only", "Known only"])
    search_query = f4.text_input("Search alert queue", "")

    filtered_history = filter_alert_history(
        history,
        status=status_filter,
        severity=severity_filter,
        ood_filter=ood_filter,
        query=search_query,
    )
    hist_df = build_history_dataframe(filtered_history)
    st.caption(f"Showing {len(filtered_history):,} of {len(history):,} persisted alerts")
    st.dataframe(hist_df, use_container_width=True, hide_index=True)

    if hist_df.empty:
        st.info("No persisted alerts match the current filters.")
        return

    render_queue_charts(hist_df)
    render_correlation_tables(filtered_history)
    render_alert_disposition(hist_df, history, update_status)


def build_history_dataframe(alerts: list[dict]) -> pd.DataFrame:
    hist_columns = [
        "Alert ID", "Time", "Status", "Source", "Severity", "Risk",
        "Class", "Hybrid Score", "AE Score", "OOD Candidate",
    ]
    hist_rows = [{
        "Alert ID": a["alert_id"],
        "Time": a["timestamp"],
        "Status": a.get("status", "new"),
        "Source": a.get("source", "session"),
        "Severity": a.get("llm_severity", "N/A"),
        "Risk": risk_score(a, a.get("llm_severity")),
        "Class": a["predicted_class"],
        "Hybrid Score": round(a["hybrid_score"], 3),
        "AE Score": round(a.get("ae_score", 0), 3),
        "OOD Candidate": "YES" if a["is_zeroday"] else "NO",
    } for a in alerts]
    hist_df = pd.DataFrame(hist_rows, columns=hist_columns)
    if not hist_df.empty:
        hist_df = hist_df.sort_values(["Risk", "Time"], ascending=[False, False])
    return hist_df


def render_queue_charts(hist_df: pd.DataFrame) -> None:
    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("### Severity Distribution")
        sev_counts = hist_df["Severity"].value_counts().rename_axis("Severity").reset_index(name="Count")
        st.bar_chart(sev_counts.set_index("Severity"))
    with right:
        st.markdown("### OOD Candidate Mix")
        zd_counts = hist_df["OOD Candidate"].value_counts().rename_axis("OOD Candidate").reset_index(name="Count")
        st.bar_chart(zd_counts.set_index("OOD Candidate"))


def render_correlation_tables(filtered_history: list[dict]) -> None:
    correlations = correlate_alerts(filtered_history, min_count=2)
    if correlations:
        st.markdown("### Correlated Alert Groups")
        corr_df = pd.DataFrame([{
            "Group": item["group_type"],
            "Key": item["key"],
            "Alerts": item["alert_count"],
            "OOD": item["ood_count"],
            "Max Risk": int(item["max_risk"]),
            "Latest": item["latest_time"],
            "Sample Alert IDs": ", ".join(item["alert_ids"]),
        } for item in correlations[:25]])
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

    incidents = build_time_window_incidents(filtered_history, window_minutes=15, min_alerts=2)
    if incidents:
        st.markdown("### Incident Windows")
        incident_df = pd.DataFrame([{
            "Incident": item["incident_id"],
            "Group": item["group_type"],
            "Key": item["key"],
            "Window": f"{item['start_time']} -> {item['end_time']}",
            "Alerts": item["alert_count"],
            "OOD": item["ood_count"],
            "High Risk": item["high_count"],
            "Max Risk": int(item["max_risk"]),
            "Severity": item["severity"],
            "Signals": ", ".join(item["primary_classes"] or item["families"]),
            "Focus": item["recommended_focus"],
            "Sample Alert IDs": ", ".join(item["alert_ids"]),
        } for item in incidents[:25]])
        st.dataframe(incident_df, use_container_width=True, hide_index=True)


def render_alert_disposition(
    hist_df: pd.DataFrame,
    history: list[dict],
    update_status: Callable[[str, str, str], None],
) -> None:
    st.markdown("### Alert Disposition")
    selected_alert_id = st.selectbox("Alert", hist_df["Alert ID"].tolist())
    selected_alert = next((a for a in history if a.get("alert_id") == selected_alert_id), {})
    current_status = selected_alert.get("status", "new")
    status_idx = STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0
    d1, d2 = st.columns([0.8, 1.2])
    new_status = d1.selectbox("Status", STATUS_OPTIONS, index=status_idx)
    analyst_note = d2.text_input("Analyst note", value=str(selected_alert.get("analyst_note", "")))
    if st.button("Update alert status", key="update_alert_status"):
        update_status(selected_alert_id, new_status, analyst_note)
        st.rerun()


def queue_summary(history: list[dict]) -> dict[str, int]:
    if not history:
        return {"alerts": 0, "critical_high": 0, "ood": 0, "average_risk": 0}
    return {
        "alerts": len(history),
        "critical_high": sum(1 for a in history if a.get("llm_severity") in ["CRITICAL", "HIGH"]),
        "ood": sum(1 for a in history if a.get("is_zeroday")),
        "average_risk": int(np.mean([risk_score(a, a.get("llm_severity")) for a in history])),
    }
