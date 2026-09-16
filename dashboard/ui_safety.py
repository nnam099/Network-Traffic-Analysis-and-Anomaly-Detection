from __future__ import annotations

from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


SAFETY_COPY = (
    "Zero-day/OOD labels are triage hypotheses, not final security conclusions. "
    "Confirm with packet, endpoint, firewall, SIEM and analyst context before containment."
)


def render_safety_notice(level: str = "info") -> None:
    if st is None:
        raise RuntimeError("streamlit is required to render the dashboard safety notice")
    if level == "warning":
        st.warning(SAFETY_COPY)
    else:
        st.info(SAFETY_COPY)


def attach_report_safety_note(report: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(report)
    enriched["limitations_and_safety"] = {
        "zero_day_candidate_meaning": "OOD hypothesis for analyst review; not a confirmed novel attack.",
        "required_validation": [
            "Check raw flow and packet evidence.",
            "Correlate source, destination, service and time window in SIEM/firewall logs.",
            "Review endpoint or asset context before blocking or escalation.",
        ],
        "model_limitations": [
            "Scores depend on feature quality and threshold calibration.",
            "Real-world CSV normalization can lose directional or timing evidence.",
            "MITRE and LLM output are decision support, not authoritative attribution.",
        ],
    }
    return enriched
