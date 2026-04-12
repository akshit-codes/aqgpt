"""AQGPT RAG package for urbanemissions.info."""

import streamlit as st

from aqgpt_core.rag.pipeline import UrbanEmissionsRAG


@st.cache_resource(show_spinner=False)
def get_rag_pipeline() -> UrbanEmissionsRAG:
    return UrbanEmissionsRAG()


__all__ = [
    "get_rag_pipeline",
]
