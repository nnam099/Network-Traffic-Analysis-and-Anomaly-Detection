from __future__ import annotations

try:
    import streamlit as st
except ModuleNotFoundError:
    st = None

from ui_safety import render_safety_notice


def render_analysis_safety_notice(demo_mode: bool = False) -> None:
    render_safety_notice(level="warning" if demo_mode else "info")
    if demo_mode:
        st.warning("DEMO SANDBOX: Model chua duoc load. Ket qua gia lap khong duoc dua vao analyst queue.")
