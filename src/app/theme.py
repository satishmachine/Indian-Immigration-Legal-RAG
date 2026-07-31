"""
app.theme
=========
Executive Dark-Mode Design System for Indian Immigration Legal AI.
LexisNexis & ChatGPT Enterprise inspired aesthetics.
"""

from __future__ import annotations

import streamlit as st


_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary:   #08080A;
    --bg-sidebar:   #0F0F14;
    --bg-card:      #14141A;
    --bg-hover:     #1C1C24;
    --bg-user-msg:  #1C1C24;

    --border:        #23232E;
    --border-subtle: #191922;

    --accent:        #10B981;
    --accent-hover:  #059669;
    --accent-dim:    rgba(16, 185, 129, 0.10);
    --accent-border: rgba(16, 185, 129, 0.28);
    --accent-text:   #34D399;

    --text-primary:  #F4F4F5;
    --text-muted:    #A1A1AA;
    --text-dim:      #71717A;

    --danger:        #EF4444;
    --danger-dim:    rgba(239, 68, 68, 0.10);
    --danger-border: rgba(239, 68, 68, 0.28);
    --warning:       #F59E0B;

    --sidebar-width: 320px;
    --content-max:   860px;

    --r-sm:  8px;
    --r-md:  12px;
    --r-lg:  16px;

    --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

html, body, [class*="css"] {
    font-family: var(--font-main) !important;
    -webkit-font-smoothing: antialiased;
}

* { box-sizing: border-box; }

/* ── Header & Sidebar Collapsed Control ────────────────────────────────── */
header[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
    z-index: 99999 !important;
    pointer-events: none !important;
}

button[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="collapsedControl"],
[data-testid="collapsedControl"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: fixed !important;
    top: 0.75rem !important;
    left: 0.75rem !important;
    z-index: 999999 !important;
    pointer-events: auto !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 0.4rem 0.6rem !important;
    box-shadow: 0 4px 14px rgba(0,0,0,0.6) !important;
    cursor: pointer !important;
}

button[data-testid="stSidebarCollapseButton"],
[data-testid="stSidebarHeader"] button {
    color: var(--text-primary) !important;
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    pointer-events: auto !important;
}

button[data-testid="stSidebarCollapseButton"] svg,
button[data-testid="stSidebarCollapsedControl"] svg,
[data-testid="stSidebarCollapsedControl"] svg,
button[data-testid="collapsedControl"] svg {
    fill: var(--text-primary) !important;
    color: var(--text-primary) !important;
    width: 20px !important;
    height: 20px !important;
}

/* ── Indian National Flag Tri-Color Inspired Executive Header Card Banner ────── */
.boi-header-banner {
    background: linear-gradient(112deg, rgba(255, 153, 51, 0.24) 0%, rgba(15, 23, 42, 0.96) 50%, rgba(19, 136, 8, 0.24) 100%), #0D0F14 !important;
    border: 1px solid rgba(255, 153, 51, 0.35) !important;
    border-top: 3px solid !important;
    border-image: linear-gradient(to right, #FF9933 0%, #FF9933 35%, #FFFFFF 50%, #138808 65%, #138808 100%) 1 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.35rem 0.55rem 1.35rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
}

.boi-header-top {
    display: flex !important;
    align-items: center !important;
    gap: 1.15rem !important;
    margin-bottom: 0.45rem !important;
}

.boi-logo-wrapper {
    flex-shrink: 0 !important;
    width: 72px !important;
    height: 72px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

.boi-header-title {
    font-size: 1.38rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.012em !important;
    line-height: 1.2 !important;
}

.title-orange {
    color: #FF9933 !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 10px rgba(255, 153, 51, 0.3) !important;
}

.title-white {
    color: #FFFFFF !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 10px rgba(255, 255, 255, 0.3) !important;
}

.title-green {
    color: #22C55E !important;
    font-weight: 800 !important;
    text-shadow: 0 2px 10px rgba(34, 197, 94, 0.3) !important;
}

.boi-header-subtitle {
    text-align: center !important;
    font-size: 0.81rem !important;
    color: #F1F5F9 !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    border-top: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding-top: 0.4rem !important;
    margin-top: 0.1rem !important;
}

/* ── Layout & Sidebar ──────────────────────────────────────────────────── */
.stApp {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border) !important;
    z-index: 999999 !important;
}

section[data-testid="stSidebar"] > div:first-child {
    background-color: var(--bg-sidebar) !important;
}

.block-container {
    max-width: var(--content-max) !important;
    margin: 0 auto !important;
    padding-top: 3.5rem !important;
    padding-bottom: 110px !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
}

/* ── Chat Messages ───────────────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.85rem 0 !important;
    margin-bottom: 0.5rem !important;
}

div[data-testid="stChatMessageAvatarUser"] {
    background-color: #4C1D95 !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
}

div[data-testid="stChatMessageAvatarAssistant"] {
    background-color: var(--accent) !important;
    border-radius: 50% !important;
    width: 32px !important;
    height: 32px !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) .stMarkdown {
    background-color: var(--bg-user-msg) !important;
    border-radius: 18px 18px 4px 18px !important;
    padding: 0.75rem 1.15rem !important;
    display: inline-block !important;
    max-width: 85% !important;
    font-size: 0.935rem !important;
    line-height: 1.6 !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) .stMarkdown p {
    font-size: 0.94rem !important;
    line-height: 1.75 !important;
    color: var(--text-primary) !important;
    margin-bottom: 0.75rem !important;
}

/* ── Harmonized Executive Headings inside Chat Messages ──────────────────── */
div[data-testid="stChatMessage"] h1,
div[data-testid="stChatMessage"] h2 {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    color: #FF9933 !important;
    margin-top: 1.1rem !important;
    margin-bottom: 0.4rem !important;
    padding-bottom: 0.25rem !important;
    border-bottom: 1px solid rgba(255, 153, 51, 0.22) !important;
    letter-spacing: -0.01em !important;
}

div[data-testid="stChatMessage"] h3,
div[data-testid="stChatMessage"] h4 {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
    color: #38BDF8 !important;
    margin-top: 0.85rem !important;
    margin-bottom: 0.3rem !important;
    border-bottom: none !important;
}

div[data-testid="stChatMessage"] ul,
div[data-testid="stChatMessage"] ol {
    margin-top: 0.3rem !important;
    margin-bottom: 0.75rem !important;
    padding-left: 1.25rem !important;
    font-size: 0.93rem !important;
    line-height: 1.65 !important;
}

div[data-testid="stChatMessage"] li {
    margin-bottom: 0.3rem !important;
    font-size: 0.93rem !important;
    color: var(--text-primary) !important;
}

/* ── Fixed Bottom Input Strip ───────────────────────────────────────────── */
div[data-testid="stBottom"] {
    position: fixed !important;
    bottom: 0 !important;
    left: 0 !important;
    right: 0 !important;
    z-index: 9999 !important;
    background-color: var(--bg-primary) !important;
    padding: 0.6rem 0 0.85rem 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] {
    position: static !important;
    max-width: var(--content-max) !important;
    width: 100% !important;
    margin: 0 auto !important;
    padding: 0 1.5rem !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

[data-testid="stChatInput"] {
    background-color: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-md) !important;
    box-shadow: 0 4px 18px rgba(0,0,0,0.5) !important;
    margin: 0 !important;
    padding: 0.15rem 0.5rem !important;
    min-height: 42px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-dim), 0 4px 20px rgba(0,0,0,0.6) !important;
}

[data-testid="stChatInput"] > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

[data-testid="stChatInput"] textarea {
    color: var(--text-primary) !important;
    background: transparent !important;
    font-size: 0.88rem !important;
    min-height: 36px !important;
    padding: 0.35rem 0.4rem !important;
    line-height: 1.35 !important;
    border: none !important;
}

[data-testid="stChatInput"] button {
    background-color: var(--accent) !important;
    border-radius: var(--r-sm) !important;
    color: #fff !important;
    height: 32px !important;
    width: 32px !important;
    min-height: 32px !important;
    border: none !important;
}

/* ── Executive Legal Citation Card (LexisNexis / Westlaw style) ──────────── */
.legal-citation-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: var(--r-sm);
    padding: 0.8rem 1rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.25);
}
.legal-citation-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.legal-citation-act {
    font-size: 0.84rem;
    font-weight: 700;
    color: var(--accent-text);
    letter-spacing: -0.01em;
}
.legal-citation-badge {
    background: var(--accent-dim);
    border: 1px solid var(--accent-border);
    color: var(--accent-text);
    font-size: 0.70rem;
    font-weight: 600;
    padding: 0.12rem 0.45rem;
    border-radius: 4px;
}
.legal-citation-section {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.4rem;
}
.legal-citation-quote {
    font-size: 0.78rem;
    color: var(--text-muted);
    font-style: italic;
    line-height: 1.55;
    background: rgba(0,0,0,0.3);
    border-radius: 6px;
    padding: 0.55rem 0.75rem;
    border-left: 2px solid var(--border);
}

/* ── PDF Viewer Panel ───────────────────────────────────────────────────── */
.pdf-panel-wrapper {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-md);
    padding: 1rem;
    height: 100%;
}
.pdf-panel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 0.6rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.75rem;
}
.pdf-panel-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 0.45rem;
}

/* ── Sidebar System Metrics Card ────────────────────────────────────────── */
.sidebar-metrics-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--r-sm);
    padding: 0.75rem 0.85rem;
    margin: 0.4rem 0.85rem;
}
.sidebar-metric-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 0.76rem;
    padding: 0.22rem 0;
}
.sidebar-metric-label { color: var(--text-dim); }
.sidebar-metric-val { color: var(--text-primary); font-weight: 600; }

/* ── Open PDF in New Web Tab Button ─────────────────────────────────────── */
.pdf-open-newtab-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 100%;
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.18), rgba(37, 99, 235, 0.10));
    border: 1px solid rgba(59, 130, 246, 0.4);
    color: #93C5FD !important;
    padding: 0.55rem 0.85rem;
    border-radius: var(--r-sm);
    font-size: 0.8rem;
    font-weight: 600;
    text-decoration: none !important;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
}
.pdf-open-newtab-btn:hover {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.35), rgba(37, 99, 235, 0.25));
    border-color: #3B82F6;
    color: #FFFFFF !important;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    transform: translateY(-1px);
}

</style>
"""


def apply_custom_theme() -> None:
    """Inject custom executive theme CSS."""
    st.markdown(_CSS, unsafe_allow_html=True)
