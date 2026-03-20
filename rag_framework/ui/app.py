"""
RAG Framework Demo — Streamlit App Entry Point.

Run with:
    streamlit run rag_framework/ui/app.py

Pages:
    Configure  — select and configure each pipeline module
    Evaluate   — run ingestion/query, compare two configs
"""

import streamlit as st

st.set_page_config(
    page_title="RAG Framework Demo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------

PAGES = {
    "⚙️ Configure": "configure",
    "📊 Evaluate": "evaluate",
}

st.sidebar.title("RAG Framework")
st.sidebar.caption("Modular · Configurable · Evaluation-first")
st.sidebar.divider()

selection = st.sidebar.radio("Navigate", list(PAGES.keys()), label_visibility="collapsed")
page_key = PAGES[selection]

st.sidebar.divider()
st.sidebar.info(
    "**How it works**\n\n"
    "1. Configure each pipeline module.\n"
    "2. Export your config as YAML.\n"
    "3. Upload documents and run ingestion.\n"
    "4. Query and compare two configurations."
)

# --------------------------------------------------------------------------
# Page routing
# --------------------------------------------------------------------------

if page_key == "configure":
    from rag_framework.ui.pages.configure import render
    render()
elif page_key == "evaluate":
    from rag_framework.ui.pages.evaluate import render
    render()
