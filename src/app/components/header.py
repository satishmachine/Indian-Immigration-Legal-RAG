import base64
from pathlib import Path

import streamlit as st


def _get_logo_base64() -> str:
    """Load Bureau of Immigration seal logo as base64 string."""
    project_root = Path(__file__).resolve().parents[3]
    logo_path = project_root / "assets" / "boi_logo.png"
    if logo_path.exists():
        try:
            with open(logo_path, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:image/png;base64,{encoded}"
        except Exception:
            pass
    return ""


def render_header() -> str:
    """
    Render top header card banner with official Bureau of Immigration logo and tri-color theme.

    Returns:
        Language string ('English').
    """
    logo_src = _get_logo_base64()
    logo_html = (
        f'<img src="{logo_src}" style="width:72px; height:72px; object-fit:contain;" alt="BOI Seal">'
        if logo_src
        else '<span style="font-size:2rem;">🏛️</span>'
    )

    st.markdown(
        f"""
        <div class="boi-header-banner">
            <div class="boi-header-top">
                <div class="boi-logo-wrapper">{logo_html}</div>
                <div class="boi-header-title">
                    <span class="title-orange">Bureau of Immigration</span>
                    <span class="title-white"> – </span>
                    <span class="title-green">Statutory AI Assistant</span>
                </div>
            </div>
            <div class="boi-header-subtitle">
                Grounded Legal Question Answering for Immigration, Citizenship, Passport, Emigration & Foreigners Acts
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return "English"
