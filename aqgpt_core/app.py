"""AQGPT: Streamlit UI Web Page"""

from pathlib import Path
import streamlit as st
from aqgpt_core.config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM, CATEGORIES

st.set_page_config(page_title="AQGPT", layout="wide", initial_sidebar_state="expanded")

def load_src(name):
    base = Path(__file__).parent / "templates"
    html = (base / f"{name}.html").read_text()
    css_path = base / f"{name}.css"
    css = css_path.read_text() if css_path.exists() else ""
    return " ".join(f"<style>{css}</style>{html}".split())

def landing():
    st.markdown(load_src("landing"), unsafe_allow_html=True)

def sidebar():
    with st.sidebar:
        st.markdown("#### 💬 Ask a Question")
        query = st.text_input("question", placeholder="Ask about air quality...", label_visibility="collapsed")
        ask = st.button("Ask", type="primary", use_container_width=True)
        
        st.divider()
        
        st.markdown("#### 📋 Sample Questions")
        
        for cat, questions in CATEGORIES.items():
            with st.expander(cat, expanded=False):
                for q in questions:
                    if st.button(q, key=q, use_container_width=True):
                        st.session_state.query = q
                        st.session_state.asked = True
        
        st.divider()
        
        st.markdown("#### ⚙️ Settings")
        with st.form("settings"):
            c1, c2 = st.columns(2)
            lat = c1.number_input("Lat", value=DEFAULT_LAT, format="%.4f")
            lon = c2.number_input("Lon", value=DEFAULT_LON, format="%.4f")
            radius_km = st.slider("Radius (km)", 1, 50, DEFAULT_RADIUS_KM)
            pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"])
            st.form_submit_button("Apply", use_container_width=True)
        
        return query, ask, lat, lon, radius_km, pollutant

if "asked" not in st.session_state:
    st.session_state.asked = False
if "query" not in st.session_state:
    st.session_state.query = ""

query, ask, lat, lon, radius_km, pollutant = sidebar()

if ask and query:
    st.session_state.query = query
    st.session_state.asked = True

if not st.session_state.asked:
    landing()
else:
    st.markdown(f"**{st.session_state.query}**")