from __future__ import annotations

import os

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None


def render_setup_status(
    model_path: str,
    pipe_path: str,
    data_path: str,
    alert_db_path: str,
    has_explainer: bool,
    has_mitre: bool,
    has_llm: bool,
) -> None:
    st.markdown("### Trang thai hien tai:")
    status_items = {
        f"Model file ({os.path.basename(model_path)})": os.path.exists(model_path),
        f"Pipeline file ({os.path.basename(pipe_path)})": os.path.exists(pipe_path),
        f"Data file ({os.path.basename(data_path)})": os.path.exists(data_path),
        f"Alert DB ({os.path.basename(alert_db_path)})": os.path.exists(alert_db_path),
        "API Key (Bat ky trong .env)": any([
            os.getenv("GROQ_API_KEY"),
            os.getenv("GEMINI_API_KEY"),
            os.getenv("OPENAI_API_KEY"),
            os.getenv("ANTHROPIC_API_KEY"),
        ]),
        "Module explainer.py": has_explainer,
        "Module mitre_mapper.py": has_mitre,
        "Module llm_agent.py": has_llm,
    }
    for item, ok in status_items.items():
        icon = "OK" if ok else "MISSING"
        color = "green" if ok else "red"
        st.markdown(f":{color}[{icon}] {item}")
