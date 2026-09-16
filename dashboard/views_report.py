from __future__ import annotations

import json
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from ui_safety import attach_report_safety_note


def build_export_report(result: dict[str, Any], llm: dict[str, Any] | None = None) -> dict[str, Any]:
    report = attach_report_safety_note({**result, "llm_analysis": llm or {}})
    report.pop("shap_values", None)
    report.pop("probs", None)
    return report


def render_report_download(result: dict[str, Any], llm: dict[str, Any], key: str) -> None:
    if st.button("Export JSON Report", key=f"export_{key}"):
        report = build_export_report(result, llm)
        st.download_button(
            "Download JSON",
            data=json.dumps(report, ensure_ascii=False, indent=2, default=str),
            file_name=f"alert_{result['alert_id']}.json",
            mime="application/json",
        )


def render_raw_report(result: dict[str, Any], llm: dict[str, Any]) -> None:
    st.json(build_export_report(result, llm))
