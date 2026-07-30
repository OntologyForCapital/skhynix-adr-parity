"""Streamlit 진입점 — 본문은 parity.ui.render_page 가 담당한다."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="SK하이닉스 본주 vs ADR(SKHY) 괴리율",
    page_icon="📊",
    layout="wide",
)

from parity import ui  # noqa: E402  — set_page_config가 첫 스트림릿 호출이어야 한다

ui.render_page()
