"""AQGPT: Streamlit UI Web Page"""

from pathlib import Path
import streamlit as st
from datetime import datetime, timedelta
from aqgpt_core.config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM, CATEGORIES, VIZ_TYPES
from aqgpt_core.llm import get_text_generator
from aqgpt_core.llm.data_extractor import extract_data_by_viz_type
from aqgpt_core.llm.session_cache import clear_tool_calls_log, get_tool_calls_log

from aqgpt_core.render import (
    render_conditions, render_wind_transport, render_health_advisory,
    render_attribution, render_why_bad, render_spatial_map, render_satellite,
    render_power_plants, render_intervention, render_trends, render_rag
)

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
            time_opt = st.selectbox("Window", ["Last 24h", "Last 7d", "Last 30d"])
            pollutant = st.selectbox("Pollutant", ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"])
            st.form_submit_button("Apply", use_container_width=True)

        return query, ask, lat, lon, radius_km, pollutant, time_opt

if "asked" not in st.session_state:
    st.session_state.asked = False
if "query" not in st.session_state:
    st.session_state.query = ""
if "ai_interventions" not in st.session_state:
    st.session_state.ai_interventions = None
if "last_query_for_interventions" not in st.session_state:
    st.session_state.last_query_for_interventions = None

query, ask, lat, lon, radius_km, pollutant, time_opt = sidebar()
now = datetime.now()
deltas = {"Last 24h": 1, "Last 7d": 7, "Last 30d": 30}
t0 = (now - timedelta(days=deltas[time_opt])).isoformat()
t1 = now.isoformat()

if ask and query:
    st.session_state.query = query
    st.session_state.asked = True
    st.session_state.ai_interventions = None  # Clear cached interventions for new query
    st.session_state.last_query_for_interventions = None

if not st.session_state.asked:
    landing()
else:
    # Clear tool calls log for new query
    clear_tool_calls_log()

    st.markdown(f"**{st.session_state.query}**")
    st.divider()

    # Try to understand query using Gemini first
    llm = get_text_generator()
    routing_query = st.session_state.query.strip()
    context = {
        'current_location': (lat, lon),
        'available_viz_types': list(set(VIZ_TYPES.values())),
        'available_pollutants': ['PM2.5', 'PM10', 'NO2', 'SO2', 'O3', 'CO'],
        'routing_guidance': (
            "Use rag for conceptual or source-backed questions (e.g., 'what is pollution', "
            "'explain PM2.5', 'what does urbanemissions say'). If the user also asks city status "
            "or interventions, include rag alongside local diagnostics visualizations."
        ),
        'routing_examples': [
            "What is pollution? -> ['rag']",
            "What is pollution and how is it in Delhi? -> ['rag', 'conditions', 'why_bad']",
            "What is pollution, how is it in Delhi, and how to improve? -> ['rag', 'conditions', 'why_bad', 'intervention']",
            "What does urbanemissions say about crop burning? -> ['rag']",
        ],
    }

    query_result = llm.understand_query(routing_query, context)

    # Check if there was an error (e.g., location not found)
    if query_result.get("error", False):
        st.error(f"❌ {query_result.get('error_message', 'Unable to process your query')}")
        st.stop()

    # Extract viz_types (now a list)
    viz_types = query_result.get("viz_types", [VIZ_TYPES.get(st.session_state.query, "conditions")])
    if isinstance(viz_types, str):
        viz_types = [viz_types]

    # Override parameters if LLM extracted different ones from query
    if query_result.get("lat"):
        lat = query_result["lat"]
    if query_result.get("lon"):
        lon = query_result["lon"]
    if query_result.get("radius_km"):
        radius_km = query_result["radius_km"]
    if query_result.get("pollutant"):
        pollutant = query_result["pollutant"]

    # Show LLM's understanding and initial answer
    with st.expander("💭 AI Understanding", expanded=True):
        st.write(query_result["answer_summary"])
        if query_result.get("confidence", 1.0) < 0.7:
            st.info(f"Low confidence in understanding ({query_result['confidence']:.0%}). Showing best match for your query.")

    st.divider()

    # Extract data for ALL visualization types in the query (with caching)
    all_data = {}
    for viz_type in viz_types:
        try:
            data = extract_data_by_viz_type(
                viz_type,
                lat,
                lon,
                radius_km,
                pollutant,
                t0,
                t1,
                user_query=st.session_state.query,
            )
            if "error" not in data:
                all_data[viz_type] = data
        except Exception as e:
            st.debug(f"Error extracting {viz_type}: {e}")

    # Generate comprehensive answer based on ALL collected data
    if all_data:
        try:
            comprehensive_answer = llm.generate_answer(all_data, list(all_data.keys()), st.session_state.query)
            if comprehensive_answer:
                st.success(f"**📊 Analysis:** {comprehensive_answer}")
                st.divider()
        except Exception as e:
            st.debug(f"Answer generation error: {e}")

    # Pre-generate interventions if needed (runs once per unique query, cached in session_state)
    ai_interventions = None
    if "intervention" in viz_types:
        # Only regenerate if this is a NEW query or if interventions weren't generated before
        if st.session_state.last_query_for_interventions != st.session_state.query:
            try:
                from aqgpt_core.tools.aq import aq_get_summary
                from aqgpt_core.tools.met import met_get_conditions
                from aqgpt_core.tools.sources import sources_get_all
                from aqgpt_core.tools.attribution import attribution_rank_sources
                from datetime import datetime

                summary = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
                if not summary.empty:
                    mean_pm = summary["mean"].mean()
                    conditions = met_get_conditions(lat, lon, t0, t1)
                    all_src = sources_get_all(lat, lon, radius_km)
                    t1_dt = datetime.fromisoformat(t1)
                    attr = attribution_rank_sources(
                        all_src["summary"], conditions, pollutant,
                        hour=t1_dt.hour,
                        is_winter=t1_dt.month in (10, 11, 12, 1, 2, 3)
                    )

                    st.session_state.ai_interventions = llm.generate_custom_interventions(
                        current_aqi=mean_pm,
                        top_sources=list(attr.keys())[:3],
                        location=(lat, lon),
                        dominant_source=list(attr.keys())[0] if attr else "traffic"
                    )
                    st.session_state.last_query_for_interventions = st.session_state.query
            except Exception as e:
                st.debug(f"Could not pre-generate interventions: {e}")
                st.session_state.ai_interventions = None

        # Use cached interventions from session state
        ai_interventions = st.session_state.ai_interventions

    # Render all visualizations dynamically
    st.markdown("### 📈 Visualizations")

    for viz_type in viz_types:
        with st.container():
            st.markdown(f"#### {viz_type.replace('_', ' ').title()}")

            try:
                if viz_type == "conditions":
                    render_conditions(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "spatial":
                    render_spatial_map(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "attribution":
                    render_attribution(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "satellite":
                    render_satellite(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "power_plants":
                    render_power_plants(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "wind":
                    render_wind_transport(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "why_bad":
                    render_why_bad(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "health":
                    render_health_advisory(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "intervention":
                    render_intervention(lat, lon, radius_km, pollutant, t0, t1, ai_interventions=ai_interventions)
                elif viz_type == "trends":
                    render_trends(lat, lon, radius_km, pollutant, t0, t1)
                elif viz_type == "rag":
                    render_rag(st.session_state.query)
            except Exception as e:
                st.error(f"Error rendering {viz_type}: {str(e)}")
                st.debug(f"Full error: {e}")

            st.divider()

    # Show debug panel with tool calls
    st.markdown("### 🔧 Debug Info")
    with st.expander("Tool Calls & API Requests", expanded=False):
        tool_calls = get_tool_calls_log()

        if tool_calls:
            # Summary
            unique_tools = {}
            cached_count = 0

            for call in tool_calls:
                tool_name = call["tool"]
                unique_tools[tool_name] = unique_tools.get(tool_name, 0) + 1
                if call["from_cache"]:
                    cached_count += 1

            col1, col2, col3 = st.columns(3)
            col1.metric("Total API Calls", len(tool_calls))
            col2.metric("Unique Tools", len(unique_tools))
            col3.metric("Cached Results", cached_count)

            st.divider()

            # Detailed log
            st.markdown("**Detailed Call Log:**")
            for i, call in enumerate(tool_calls, 1):
                status = "✅ Cached" if call["from_cache"] else "🌐 Fresh"
                with st.expander(f"{i}. {call['tool']} {status}", expanded=False):
                    st.json({
                        "tool": call["tool"],
                        "params": call["params"],
                        "source": "cache" if call["from_cache"] else "API",
                        "timestamp": call["timestamp"]
                    })
        else:
            st.info("No tool calls made for this query.")

