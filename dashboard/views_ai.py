from __future__ import annotations

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from ui_safety import render_safety_notice


AI_QUESTION_SUGGESTIONS = [
    "Tai sao alert nay bi danh dau la nguy hiem?",
    "MITRE technique nao lien quan nhat?",
    "Buoc xu ly khan cap la gi?",
    "Day co the la false positive khong?",
    "Giai thich feature bat thuong nhat cho toi hieu",
]


def render_ai_context_card(alert_context: dict) -> None:
    render_safety_notice()
    st.markdown(
        f"""
        <div class="soc-panel">
            <span class="soc-badge soc-pill-blue">Context {alert_context.get('alert_id', 'N/A')}</span>
            <span class="soc-badge">{alert_context.get('predicted_class', 'Unknown')}</span>
            <span class="soc-badge">hybrid {float(alert_context.get('hybrid_score', 0)):.3g}</span>
            <span class="soc-badge">ae {float(alert_context.get('ae_score', 0)):.3g}</span>
            <div class="soc-detail-title" style="margin-top:10px;">Current AI Context</div>
            <div class="soc-detail-value">{alert_context.get('predicted_class', alert_context.get('classifier_class', ''))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_question_suggestions() -> None:
    st.markdown("**Goi y cau hoi:**")
    cols = st.columns(len(AI_QUESTION_SUGGESTIONS))
    for index, (col, question) in enumerate(zip(cols, AI_QUESTION_SUGGESTIONS, strict=True)):
        if col.button(question[:25] + "...", key=f"sug_{index}"):
            st.session_state["pending_question"] = question
