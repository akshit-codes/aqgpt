"""AQGPT render functions — all Streamlit visualization logic lives here.

Each function takes live data from the tools layer and renders it
using Streamlit native components, Plotly charts, and Folium maps.

Called by: app.py based on the viz_type determined from the user's query.
Never called directly by the LLM.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import folium
from folium.plugins import HeatMap
from scipy.interpolate import griddata
import streamlit as st
from streamlit_folium import st_folium
from datetime import datetime, timedelta

from aqgpt_core.tools.aq import aq_get_summary, aq_get_timeseries, aq_get_aqi_snapshot
from aqgpt_core.tools.met import met_get_conditions, met_get_timeseries, met_get_stagnation
from aqgpt_core.tools.sources import sources_get_all, sources_get_power_plants
from aqgpt_core.tools.attribution import attribution_rank_sources, attribution_explain
from aqgpt_core.tools.satellite import satellite_get_fires, satellite_get_no2, satellite_get_aod
from aqgpt_core.config import AQI_BREAKPOINTS
from aqgpt_core.rag import get_rag_pipeline
from aqgpt_core.rag.settings import RAG_CHROMA_DIR


def get_aqi_category(value: float, pollutant: str = "PM2.5") -> tuple[str, str]:
    """
    Return (category_name, color_hex) for a pollutant value.
    Uses India NAAQS breakpoints from config.
    """
    breakpoints = AQI_BREAKPOINTS.get(pollutant, AQI_BREAKPOINTS["PM2.5"])
    categories  = [
        ("Good",         "#22c55e"),
        ("Satisfactory", "#84cc16"),
        ("Moderate",     "#eab308"),
        ("Poor",         "#f97316"),
        ("Very Poor",    "#ef4444"),
        ("Severe",       "#b91c1c"),
    ]
    for i, bp in enumerate(breakpoints):
        if value <= bp:
            return categories[i]
    return categories[-1]


def _make_folium_map(lat: float, lon: float, zoom: int = 11) -> folium.Map:
    """
    Create a base Folium map using OpenStreetMap tiles.
    Uses OSM to avoid CDN blocking issues with CartoDB/jQuery.
    """
    return folium.Map(
        location=[lat, lon],
        zoom_start=zoom,
        tiles="OpenStreetMap",
    )


def render_conditions(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render current AQ conditions view.

    Shows AQI card, meteorological context, trend sparkline,
    and station ranking table.
    Called by app.py when viz_type == 'conditions'.
    """
    summary    = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
    timeseries = aq_get_timeseries(lat, lon, radius_km, pollutant, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)

    if summary.empty:
        st.warning("No station data found for this location and time range.")
        return

    # Check if meteorological data is available
    if conditions.get("error"):
        st.error(f"❌ Cannot show full conditions analysis: {conditions.get('error_reason', conditions.get('error'))}")
        # Still show AQI card but skip meteorological context
        current_mean = summary["mean"].mean()
        current_max  = summary["max"].max()
        category, color = get_aqi_category(current_mean, pollutant)

        st.markdown(
            f"<div style='padding:20px;background:#1a1133;border:1px solid #3d2475;"
            f"border-radius:12px;'>"
            f"<div style='font-size:0.8rem;color:#9b92b8;margin-bottom:8px;'>"
            f"Current {pollutant}</div>"
            f"<div style='font-size:3.5rem;font-weight:800;color:{color};"
            f"line-height:1;'>{current_mean:.0f}</div>"
            f"<div style='color:#9b92b8;margin-top:4px;'>µg/m³</div>"
            f"<div style='margin-top:12px;padding-top:8px;border-top:1px solid #2d2640;'>"
            f"<span style='font-weight:600;color:{color};'>{category}</span>"
            f"<span style='color:#4e4668;font-size:0.85rem;'> · Peak: {current_max:.0f}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        return

    current_mean = summary["mean"].mean()
    current_max  = summary["max"].max()
    category, color = get_aqi_category(current_mean, pollutant)

    # ── AQI Card ──────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown(
            f"<div style='padding:20px;background:#1a1133;border:1px solid #3d2475;"
            f"border-radius:12px;'>"
            f"<div style='font-size:0.8rem;color:#9b92b8;margin-bottom:8px;'>"
            f"Current {pollutant}</div>"
            f"<div style='font-size:3.5rem;font-weight:800;color:{color};"
            f"line-height:1;'>{current_mean:.0f}</div>"
            f"<div style='color:#9b92b8;margin-top:4px;'>µg/m³</div>"
            f"<div style='margin-top:12px;padding-top:8px;border-top:1px solid #2d2640;'>"
            f"<span style='font-weight:600;color:{color};'>{category}</span>"
            f"<span style='color:#4e4668;font-size:0.85rem;'> · Peak: {current_max:.0f}</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("**Conditions**")
        wind_speed = conditions.get("wind_speed")
        wind_label = conditions.get("wind_label")
        blh_mean = conditions.get("blh_mean")
        stagnation = conditions.get("stagnation_risk")

        # Check for None values and report if unavailable
        if wind_speed is None or wind_label is None:
            st.warning("⚠️ Wind data unavailable")
        else:
            st.markdown(f"🌬️ Wind: **{wind_label}** @ {wind_speed:.1f} m/s")

        if blh_mean is None:
            st.warning("⚠️ Boundary layer data unavailable")
        else:
            st.markdown(f"📊 BLH: **{blh_mean}** m")

        if stagnation is None:
            st.warning("⚠️ Stagnation risk unavailable")
        else:
            icon = "🔴" if stagnation == "high" else "🟡" if stagnation == "moderate" else "🟢"
            st.markdown(f"{icon} Stagnation: **{stagnation}**")

    with col3:
        if not timeseries.empty and len(timeseries) >= 2:
            today_avg     = timeseries[pollutant].iloc[-1]
            yesterday_avg = timeseries[pollutant].iloc[0]
            delta         = today_avg - yesterday_avg
            st.metric(
                "vs Earlier",
                f"{today_avg:.0f}",
                f"{delta:+.0f}",
                delta_color="inverse"
            )

    # ── Sparkline ─────────────────────────────────────────────────────────────
    if not timeseries.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=timeseries["time"],
            y=timeseries[pollutant],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#8b5cf6", width=2),
            fillcolor="rgba(139,92,246,0.1)",
        ))

        breakpoints      = AQI_BREAKPOINTS.get(pollutant, AQI_BREAKPOINTS["PM2.5"])
        threshold_labels = ["Satisfactory", "Poor", "Very Poor"]
        threshold_colors = ["#84cc16", "#f97316", "#ef4444"]

        for thresh, color, label in zip(breakpoints[1:4], threshold_colors, threshold_labels):
            fig.add_hline(
                y=thresh, line_dash="dot",
                line_color=color, opacity=0.5,
                annotation_text=label,
                annotation_position="right"
            )

        fig.update_layout(
            height=200,
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(gridcolor="#2d2640", title=f"{pollutant} (µg/m³)"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9b92b8", size=10),
        )
        st.plotly_chart(fig, use_container_width=True)

        # AI-generated insight about trends
        from aqgpt_core.llm import get_text_generator
        try:
            trend = "improving" if (timeseries[pollutant].iloc[-1] < timeseries[pollutant].iloc[0]) else "worsening" if (timeseries[pollutant].iloc[-1] > timeseries[pollutant].iloc[0]) else "stable"
            llm = get_text_generator()
            insight = llm.generate_conditions_insight(current_mean, category, conditions['wind_speed'], conditions['wind_label'], conditions['blh_mean'], conditions['stagnation_risk'], trend)
            if insight:
                st.info(f"💡 Insight: {insight}")
        except Exception as e:
            st.debug(f"Could not generate insight: {e}")

    # ── Station Ranking ───────────────────────────────────────────────────────
    if not summary.empty:
        st.markdown("**Station Ranking**")
        ranking = summary.sort_values("mean", ascending=False).head(10).copy()
        ranking = ranking[["station_name", "mean", "max"]].reset_index(drop=True)
        ranking["category"] = ranking["mean"].apply(
            lambda x: get_aqi_category(x, pollutant)[0]
        )
        ranking.columns = ["Station", "Mean (µg/m³)", "Peak (µg/m³)", "Category"]
        ranking["Mean (µg/m³)"] = ranking["Mean (µg/m³)"].astype(str)
        ranking["Peak (µg/m³)"] = ranking["Peak (µg/m³)"].astype(str)
        st.dataframe(ranking, use_container_width=True, hide_index=True)


def render_wind_transport(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render wind transport map showing wind direction, upwind sector,
    and sources colored by upwind/downwind position.
    Called by app.py when viz_type == 'wind'.
    """
    conditions = met_get_conditions(lat, lon, t0, t1)
    met_ts     = met_get_timeseries(lat, lon, t0, t1)
    all_src    = sources_get_all(lat, lon, radius_km)

    # Check if meteorological data is available
    if conditions.get("error"):
        st.error(f"❌ Cannot analyze wind patterns: {conditions.get('error_reason', conditions.get('error'))}")
        return

    wind_dir   = conditions.get("wind_direction")
    wind_speed = conditions.get("wind_speed")
    wind_label = conditions.get("wind_label")

    # Additional check for None values
    if wind_dir is None or wind_speed is None:
        st.error("❌ Meteorological data unavailable for this location/time period. Cannot analyze wind transport.")
        return

    upwind_dir = (wind_dir + 180) % 360

    # ── Map ───────────────────────────────────────────────────────────────────
    m = _make_folium_map(lat, lon, zoom=11)

    # Wind arrows
    if wind_speed > 0.3:
        w_rad    = np.radians(wind_dir)
        n_arrows = 5
        spacing  = radius_km * 0.4 / 111

        for i in range(-n_arrows // 2, n_arrows // 2 + 1):
            for j in range(-n_arrows // 2, n_arrows // 2 + 1):
                if abs(i) + abs(j) > n_arrows // 2 + 1:
                    continue

                arrow_lat = lat + i * spacing
                arrow_lon = lon + j * spacing * 1.2
                arrow_len = 0.008 * min(wind_speed, 10)
                dx        = arrow_len * np.sin(w_rad)
                dy        = arrow_len * np.cos(w_rad)
                end_lat   = arrow_lat + dy
                end_lon   = arrow_lon + dx

                folium.PolyLine(
                    locations=[[arrow_lat, arrow_lon], [end_lat, end_lon]],
                    color="#8b5cf6", weight=3, opacity=0.7,
                ).add_to(m)

                head_size = arrow_len * 0.35
                for angle_offset in [150, 210]:
                    angle = w_rad + np.radians(angle_offset)
                    head  = [
                        end_lat + head_size * np.cos(angle),
                        end_lon + head_size * np.sin(angle),
                    ]
                    folium.PolyLine(
                        [[end_lat, end_lon], head],
                        color="#8b5cf6", weight=3, opacity=0.7
                    ).add_to(m)

    # Upwind sector cone
    cone_len    = radius_km * 1.9 / 111
    cone_angle  = 45
    upwind_rad  = np.radians(upwind_dir)
    cone_points = [[lat, lon]]

    for angle in range(-cone_angle, cone_angle + 1, 5):
        rad = upwind_rad + np.radians(angle)
        cone_points.append([
            lat + cone_len * np.cos(rad),
            lon + cone_len * np.sin(rad),
        ])
    cone_points.append([lat, lon])

    folium.Polygon(
        locations=cone_points,
        color="#f59e0b", weight=2, fill=True,
        fill_color="#f59e0b", fill_opacity=0.15,
        popup="Upwind sector — pollution likely coming from here",
    ).add_to(m)

    # Industry markers colored by upwind/downwind
    for ind in all_src.get("industries", [])[:10]:
        src_dir   = np.degrees(np.arctan2(ind["lon"] - lon, ind["lat"] - lat)) % 360
        is_upwind = abs((src_dir - upwind_dir + 180) % 360 - 180) < 45
        color     = "#ef4444" if is_upwind else "#3b82f6"

        folium.CircleMarker(
            location=[ind["lat"], ind["lon"]],
            radius=10, color=color, weight=2,
            fill=True, fill_color=color, fill_opacity=0.7,
            tooltip=f"🏭 {ind['name']} {'⬆ UPWIND' if is_upwind else ''}",
        ).add_to(m)

    for pow in all_src.get("power_plants", [])[:10]:
        src_dir   = np.degrees(np.arctan2(pow["lon"] - lon, pow["lat"] - lat)) % 360
        is_upwind = abs((src_dir - upwind_dir + 180) % 360 - 180) < 45
        color     = "#ef4444" if is_upwind else "#3b82f6"

        folium.CircleMarker(
            location=[pow["lat"], pow["lon"]],
            radius=10, color=color, weight=2,
            fill=True, fill_color=color, fill_opacity=0.7,
            tooltip=f"⚡ {pow['name']} {'⬆ UPWIND' if is_upwind else ''}",
        ).add_to(m)

    # Query center marker
    folium.CircleMarker(
        location=[lat, lon], radius=12,
        color="white", weight=3,
        fill=True, fill_color="#1a1133", fill_opacity=0.9,
        tooltip="Query center",
    ).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])

    # ── Wind Rose + Upwind Sources ────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Wind Rose**")
        if not met_ts.empty and "wind_dir" in met_ts.columns:
            labels = [
                "N","NNE","NE","ENE","E","ESE","SE","SSE",
                "S","SSW","SW","WSW","W","WNW","NW","NNW"
            ]
            bins = np.arange(0, 361, 22.5)
            met_ts["wind_bin"] = pd.cut(
                met_ts["wind_dir"], bins=bins,
                labels=labels, include_lowest=True
            )
            wind_counts = met_ts["wind_bin"].value_counts().reindex(labels, fill_value=0)

            fig = go.Figure(go.Barpolar(
                r=wind_counts.values, theta=labels,
                marker_color="#8b5cf6",
                marker_line_color="#6d28d9",
                marker_line_width=1,
            ))
            fig.update_layout(
                height=280,
                margin=dict(l=40, r=40, t=20, b=20),
                polar=dict(
                    radialaxis=dict(gridcolor="#2d2640"),
                    angularaxis=dict(
                        gridcolor="#2d2640",
                        direction="clockwise",
                        rotation=90,
                    ),
                    bgcolor="rgba(0,0,0,0)",
                ),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#9b92b8", size=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough met data for wind rose.")

    with col2:
        st.markdown("**Upwind Sources**")
        upwind_sources = []
        for cat, items in [
            ("Industries",   all_src.get("industries", [])),
            ("Kilns",        all_src.get("kilns", [])),
            ("Power Plants", all_src.get("power_plants", [])),
        ]:
            for item in items[:8]:
                src_dir = np.degrees(
                    np.arctan2(item["lon"] - lon, item["lat"] - lat)
                ) % 360
                if abs((src_dir - upwind_dir + 180) % 360 - 180) < 60:
                    upwind_sources.append((
                        cat,
                        item.get("name", "Unknown"),
                        item.get("distance_km", 0),
                    ))

        if upwind_sources:
            for cat, name, dist in sorted(upwind_sources, key=lambda x: x[2])[:15]:
                st.markdown(f"- **{name}** ({cat}, {dist:.1f} km)")
        else:
            st.info("No major sources found in the upwind direction.")

def render_health_advisory(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render health advisory with activity-specific guidance using AI.
    Called by app.py when viz_type == 'health'.
    """
    from aqgpt_core.llm import get_text_generator

    summary = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)

    if summary.empty:
        st.warning("No data available.")
        return

    pm25 = summary["mean"].mean()
    pm25_max = summary["max"].max()
    category, color = get_aqi_category(pm25, pollutant)

    # Use AI to generate health advisory
    llm = get_text_generator()
    try:
        health_advice = llm.generate_health_advisory(pm25, pollutant, (lat, lon))
    except Exception as e:
        st.warning(f"Health advisory generation failed: {e}")
        # Fallback to hardcoded
        health_advice = llm._fallback_health_advisory(pm25, pollutant)

    # Display AI-generated advisory
    border_color = health_advice.get("border_color", color)
    bg_color = health_advice.get("bg_color", "#1a1133")
    header = health_advice.get("header", "Air Quality Advisory")
    health_class = health_advice.get("health_class", category)
    outdoor_ok = health_advice.get("outdoor_ok", False)
    sensitive_risk = health_advice.get("sensitive_risk", "Unknown")
    general_risk = health_advice.get("general_risk", "Unknown")
    exercise_rec = health_advice.get("exercise_rec", "Consult local guidelines")
    mask_rec = health_advice.get("mask_rec", "Check with health authorities")

    st.markdown(
        f"<div style='padding:20px;border-radius:12px;border-left:5px solid {border_color};"
        f"background:linear-gradient(90deg,{bg_color} 0%,#0e0b14 100%);margin-bottom:16px;'>"
        f"<div style='font-size:1.2rem;font-weight:700;margin-bottom:8px;'>{header}</div>"
        f"<div style='color:#9b92b8;'>Current {pollutant}: <b style='color:#f4f0ff'>{pm25:.0f} µg/m³</b> "
        f"({health_class}) · Peak: {pm25_max:.0f} µg/m³</div>"
        f"</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Risk Levels**")
        st.markdown(f"- Sensitive groups: **{sensitive_risk}**")
        st.markdown(f"- General population: **{general_risk}**")
    with col2:
        st.markdown("**Recommendations**")
        st.markdown(f"- Exercise: {exercise_rec}")
        st.markdown(f"- Mask: {mask_rec}")
    with col3:
        st.markdown("**Sensitive Groups**")
        st.markdown("- Children under 14")
        st.markdown("- Adults over 65")
        st.markdown("- Respiratory conditions")
        st.markdown("- Heart disease patients")

    st.divider()
    st.markdown("**Activity Guidance**")

    # AI-suggested activities (with fallback to defaults)
    activities = health_advice.get("activities", [
        {"activity": "🏃 Jogging", "safe": pm25 <= 60, "caution": pm25 <= 90, "note": "Move indoors"},
        {"activity": "🚴 Cycling", "safe": pm25 <= 90, "caution": pm25 <= 120, "note": "Avoid main roads"},
        {"activity": "⚽ Outdoor Sports", "safe": pm25 <= 60, "caution": pm25 <= 90, "note": "Shorten duration"},
        {"activity": "🏫 School Outdoor", "safe": pm25 <= 90, "caution": pm25 <= 120, "note": "Limit to 30 min"},
        {"activity": "👶 Infant Walk", "safe": pm25 <= 30, "caution": pm25 <= 60, "note": "Stay indoors"},
    ])

    # Display activities (convert from bool-based logic if needed)
    cols = st.columns(len(activities))
    for col_idx, (col, activity_data) in enumerate(zip(cols, activities)):
        with col:
            if isinstance(activity_data, dict):
                activity = activity_data.get("activity", "Activity")
                safe = activity_data.get("safe", False)
                caution = activity_data.get("caution", False)
                note = activity_data.get("note", "")
            else:
                # Fallback for old tuple format
                activity, safe, caution, note = activity_data, False, False, ""

            if safe:
                st.success(f"{activity}\n\n✅ Safe")
            elif caution:
                st.warning(f"{activity}\n\n⚠️ Caution\n\n{note}")
            else:
                st.error(f"{activity}\n\n❌ Avoid\n\n{note}")


def render_attribution(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render source attribution with horizontal bar chart and breakdown.
    Called by app.py when viz_type == 'attribution'.
    """
    summary    = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)
    all_src    = sources_get_all(lat, lon, radius_km)

    if summary.empty:
        st.warning("No data available.")
        return

    mean_level = summary["mean"].mean()
    t1_dt      = datetime.fromisoformat(t1)
    attr       = attribution_rank_sources(
        all_src["summary"], conditions, pollutant,
        hour=t1_dt.hour,
        is_winter=t1_dt.month in (10, 11, 12, 1, 2, 3)
    )

    if not attr:
        st.warning("Insufficient data for attribution.")
        return

    attr_df = pd.DataFrame([
        {
            "Source":      k.replace("_", " ").title(),
            "Contribution": v["contribution"],
            "Low":         v["range_low"],
            "High":        v["range_high"],
            "µg/m³":       round(v["contribution"] * mean_level, 1),
        }
        for k, v in attr.items()
    ]).sort_values("Contribution", ascending=False)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        colors = {
            "Traffic":            "#3b82f6",
            "Industry":           "#8b5cf6",
            "Road Dust":          "#f59e0b",
            "Power Plants":       "#ef4444",
            "Residential Biomass": "#10b981",
            "Brick Kilns":        "#ec4899",
        }
        fig = go.Figure()
        for _, row in attr_df.iterrows():
            color = colors.get(row["Source"], "#64748b")
            fig.add_trace(go.Bar(
                x=[row["Contribution"]],
                y=[row["Source"]],
                orientation="h",
                marker_color=color,
                error_x=dict(
                    type="data", symmetric=False,
                    array=[row["High"] - row["Contribution"]],
                    arrayminus=[row["Contribution"] - row["Low"]],
                    color="#475569", thickness=2,
                ),
                text=f"{row['Contribution']:.0%}",
                textposition="outside",
                showlegend=False,
            ))

        fig.update_layout(
            height=300,
            margin=dict(l=0, r=60, t=20, b=20),
            xaxis=dict(
                tickformat=".0%",
                range=[0, max(attr_df["High"].max() * 1.1, 0.5)],
                gridcolor="#2d2640",
            ),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#9b92b8"),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Breakdown**")
        for _, row in attr_df.iterrows():
            st.markdown(
                f"<div style='margin:8px 0;padding:10px;background:#1a1133;"
                f"border-radius:6px;border:1px solid #3d2475;'>"
                f"<div style='font-weight:600;color:#f4f0ff;'>{row['Source']}</div>"
                f"<div style='font-size:0.85rem;color:#9b92b8;'>"
                f"{row['Contribution']:.0%} · ~{row['µg/m³']} µg/m³</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    st.divider()
    st.caption(attribution_explain(attr, conditions, mean_level))

    # AI-generated attribution analysis
    from aqgpt_core.llm import get_text_generator
    try:
        llm = get_text_generator()
        analysis = llm.analyze_attribution(
            traffic=attr.get("traffic", {}).get("contribution", 0),
            industry=attr.get("industry", {}).get("contribution", 0),
            power_plants=attr.get("power_plants", {}).get("contribution", 0),
            dust=attr.get("road_dust", {}).get("contribution", 0),
            other=sum(v.get("contribution", 0) for k, v in attr.items() if k not in ["traffic", "industry", "power_plants", "road_dust"]),
            location=(lat, lon),
            conditions=conditions
        )
        if analysis:
            st.info(f"🔍 Pattern Analysis: {analysis}")
    except Exception as e:
        st.debug(f"Could not generate attribution analysis: {e}")


def render_why_bad(lat, lon, radius_km, pollutant, t0, t1):
    """
    Explain why pollution is high using AI analysis of meteorological factors.
    Called by app.py when viz_type == 'why_bad'.
    """
    from aqgpt_core.llm import get_text_generator

    summary    = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
    met_ts     = met_get_timeseries(lat, lon, t0, t1)
    stagnation = met_get_stagnation(lat, lon, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)

    if summary.empty:
        st.warning("No data available.")
        return

    # Check if meteorological data is available
    if conditions.get("error"):
        st.error(f"❌ Cannot analyze pollution causes: {conditions.get('error_reason', conditions.get('error'))}")
        return

    mean_pm  = summary["mean"].mean()
    category, color = get_aqi_category(mean_pm, pollutant)

    st.markdown(
        f"<h3 style='color:#f4f0ff;'>Why is {pollutant} at "
        f"<span style='color:{color}'>{mean_pm:.0f} µg/m³</span> ({category})?</h3>",
        unsafe_allow_html=True
    )

    # Check that critical meteorological values are not None
    if any(conditions.get(k) is None for k in ["wind_speed", "blh_mean"]):
        st.warning("⚠️ Incomplete meteorological data available. Showing limited analysis.")
        return

    # Use AI to analyze why pollution is high
    llm = get_text_generator()
    try:
        ai_factors = llm.analyze_why_bad(
            {
                "wind_speed": conditions["wind_speed"],
                "wind_direction": conditions.get("wind_direction", 0),
                "blh_mean": conditions["blh_mean"],
                "stagnation_risk": conditions.get("stagnation_risk", "low"),
                "temperature": conditions.get("temperature_mean"),
                "humidity": conditions.get("humidity_mean"),
            },
            mean_pm,
            (lat, lon)
        )
        factors = [
            (
                f.get("factor") or f.get("name") or f.get("title", "Unknown Factor"),
                f.get("description") or f.get("desc") or f.get("reason", ""),
                f.get("contribution_score") or f.get("severity") or f.get("score", 0.5)
            )
            for f in ai_factors
        ]
    except Exception as e:
        st.warning(f"AI factor analysis failed ({e}), using fallback analysis")
        # Fallback to hardcoded logic
        factors = []
        t1_dt = datetime.fromisoformat(t1)

        # Stagnation
        if stagnation["risk"] == "high":
            factors.append(("⚠️ Atmospheric Stagnation", stagnation["reason"], 0.9))
        elif stagnation["risk"] == "moderate":
            factors.append(("🌫️ Moderate Stagnation", stagnation["reason"], 0.6))

        # Wind
        if conditions["wind_speed"] < 2:
            factors.append(("🌬️ Calm Winds",
                f"Wind only {conditions['wind_speed']:.1f} m/s — poor horizontal dispersion.", 0.8))
        elif conditions["wind_speed"] < 4:
            factors.append(("🌬️ Light Winds",
                f"Wind {conditions['wind_speed']:.1f} m/s — moderate dispersion.", 0.6))

        # BLH
        if conditions["blh_mean"] < 500:
            factors.append(("📊 Low Mixing Height",
                f"BLH only {conditions['blh_mean']}m — pollutants trapped near surface.", 0.8))
        elif conditions["blh_mean"] < 800:
            factors.append(("📊 Moderate Mixing Height",
                f"BLH {conditions['blh_mean']}m — limited vertical dispersion.", 0.6))

        # Time of day
        hour = t1_dt.hour
        if 7 <= hour <= 10:
            factors.append(("🚗 Morning Rush Hour", "Peak traffic emissions 7-10 AM.", 0.6))
        elif 17 <= hour <= 21:
            factors.append(("🚗 Evening Rush Hour", "Peak traffic + cooking emissions.", 0.7))
        elif 0 <= hour <= 5:
            factors.append(("🌙 Nighttime Accumulation", "Low BLH and calm conditions at night.", 0.5))

        # Season
        if t1_dt.month in (11, 12, 1, 2):
            factors.append(("❄️ Winter Season",
                "Lower BLH and more heating/biomass emissions in winter.", 0.6))

    # Display factors
    for factor, desc, severity in factors:
        # Map severity score (0-1) to color
        if severity > 0.7:
            border = "#ef4444"
        elif severity > 0.4:
            border = "#f59e0b"
        else:
            border = "#22c55e"

        st.markdown(
            f"<div style='padding:12px;margin:8px 0;background:#1a1133;"
            f"border-left:4px solid {border};border-radius:4px;'>"
            f"<div style='font-weight:600;color:#f4f0ff;'>{factor}</div>"
            f"<div style='font-size:0.9rem;color:#9b92b8;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    if not factors:
        st.info("No specific aggravating factors identified for this time period.")

    # Correlation charts
    if not met_ts.empty:
        st.divider()
        st.markdown("**Meteorological Correlations**")
        timeseries = aq_get_timeseries(lat, lon, radius_km, pollutant, t0, t1)

        if not timeseries.empty and len(timeseries) > 1:
            merged = pd.merge_asof(
                timeseries.sort_values("time"),
                met_ts.sort_values("time"),
                on="time",
                tolerance=pd.Timedelta("1h")
            )

            col1, col2 = st.columns(2)

            with col1:
                if "blh" in merged.columns and merged["blh"].notna().any():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=merged["blh"].dropna(),
                        y=merged.loc[merged["blh"].notna(), pollutant],
                        mode="markers",
                        marker=dict(color="#8b5cf6", size=6, opacity=0.6),
                    ))
                    fig.update_layout(
                        height=250, margin=dict(l=50, r=20, t=30, b=40),
                        title=dict(text=f"{pollutant} vs BLH", font=dict(size=12, color="#f4f0ff")),
                        xaxis=dict(title="BLH (m)", gridcolor="#2d2640"),
                        yaxis=dict(title=f"{pollutant} (µg/m³)", gridcolor="#2d2640"),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9b92b8", size=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if "wind_speed" in merged.columns and merged["wind_speed"].notna().any():
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=merged["wind_speed"].dropna(),
                        y=merged.loc[merged["wind_speed"].notna(), pollutant],
                        mode="markers",
                        marker=dict(color="#f59e0b", size=6, opacity=0.6),
                    ))
                    fig.update_layout(
                        height=250, margin=dict(l=50, r=20, t=30, b=40),
                        title=dict(text=f"{pollutant} vs Wind Speed", font=dict(size=12, color="#f4f0ff")),
                        xaxis=dict(title="Wind Speed (m/s)", gridcolor="#2d2640"),
                        yaxis=dict(title=f"{pollutant} (µg/m³)", gridcolor="#2d2640"),
                        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#9b92b8", size=10),
                    )
                    st.plotly_chart(fig, use_container_width=True)


def render_spatial_map(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render spatial map showing AQI at each station with color coded markers.
    Called by app.py when viz_type == 'spatial'.
    """
    summary = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)

    if summary.empty:
        st.warning("No station data found.")
        return

    m = _make_folium_map(lat, lon, zoom=10)

    for _, row in summary.iterrows():
        if pd.isna(row["mean"]):
            continue
        _, color = get_aqi_category(row["mean"], pollutant)
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=14,
            color="white", weight=2,
            fill=True, fill_color=color, fill_opacity=0.9,
            tooltip=f"{row['station_name']}: {row['mean']:.0f} µg/m³",
            popup=folium.Popup(
                f"<b>{row['station_name']}</b><br>"
                f"{pollutant}: {row['mean']:.0f} µg/m³<br>"
                f"Peak: {row['max']:.0f} µg/m³",
                max_width=200
            ),
        ).add_to(m)

    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color="#8b5cf6", weight=3,
        fill=True, fill_color="#1a1133", fill_opacity=0.9,
        tooltip="Query center",
    ).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])

    st.markdown("**Station Summary**")
    ranking = summary.sort_values("mean", ascending=False).copy()
    ranking["Category"] = ranking["mean"].apply(lambda x: get_aqi_category(x, pollutant)[0])
    ranking = ranking[["station_name", "mean", "max", "Category"]]
    ranking.columns = ["Station", "Mean (µg/m³)", "Peak (µg/m³)", "Category"]
    ranking["Mean (µg/m³)"] = ranking["Mean (µg/m³)"].astype(str)
    ranking["Peak (µg/m³)"] = ranking["Peak (µg/m³)"].astype(str)
    st.dataframe(ranking, use_container_width=True, hide_index=True)


def render_satellite(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render satellite data: TROPOMI NO2 grid, MODIS AOD, VIIRS fires.
    Called by app.py when viz_type == 'satellite'.
    """
    tropomi = satellite_get_no2(lat, lon, radius_km, t0, t1)
    aod     = satellite_get_aod(lat, lon, radius_km, t0, t1)
    fires   = satellite_get_fires(lat, lon, max(radius_km, 150), t0, t1)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**TROPOMI NO2**")
        st.metric("Column Density",
            f"{tropomi['summary']['mean_no2_umol_m2']:.1f} µmol/m²")
        st.caption(f"Resolution: {tropomi['resolution_km']} km · "
                   f"{tropomi['summary']['n_valid_days']} days · "
                   f"Source: {tropomi['source']}")
        if tropomi["summary"]["n_hotspots"] > 0:
            st.warning(f"⚠️ {tropomi['summary']['n_hotspots']} NO2 hotspots detected")

    with col2:
        st.markdown("**MODIS AOD**")
        st.metric("Aerosol Depth", f"{aod['summary']['mean_aod']:.3f}")
        st.caption(f"Source: {aod['source']}")
        if aod["summary"]["high_aod_days"] > 0:
            st.warning(f"⚠️ {aod['summary']['high_aod_days']} high-AOD days")

    with col3:
        st.markdown("**VIIRS Fires**")
        st.metric("Active Fires", fires["summary"]["total_fires"])
        st.metric("Total FRP", f"{fires['summary']['total_frp_mw']:.0f} MW")
        if fires["summary"]["crop_residue_fires"] > 0:
            st.error(f"🔥 {fires['summary']['crop_residue_fires']} crop residue fires")

    # AI-generated satellite interpretation
    st.divider()
    from aqgpt_core.llm import get_text_generator
    try:
        llm = get_text_generator()
        interpretation = llm.interpret_satellite_data(
            no2_hotspots=tropomi["summary"]["n_hotspots"],
            mean_no2=tropomi["summary"]["mean_no2_umol_m2"],
            aod=aod["summary"]["mean_aod"],
            fires=fires["summary"]["total_fires"],
            crop_fires=fires["summary"]["crop_residue_fires"],
            location=(lat, lon)
        )
        if interpretation:
            st.info(f"🛰️ Analysis: {interpretation}")
    except Exception as e:
        st.debug(f"Could not generate satellite interpretation: {e}")

    st.divider()

    no2_grid = tropomi.get("grid_data", [])
    if no2_grid:
        no2_df  = pd.DataFrame(no2_grid)
        no2_avg = no2_df.groupby(["lat", "lon"])["no2_column_mol_m2"].mean().reset_index()

        m_no2   = _make_folium_map(lat, lon, zoom=10)
        no2_min = no2_avg["no2_column_mol_m2"].min()
        no2_max_val = no2_avg["no2_column_mol_m2"].max()

        def no2_color(val):
            if no2_max_val <= no2_min:
                return "#3b82f6"
            norm = (val - no2_min) / (no2_max_val - no2_min)
            if norm < 0.33:
                r, g, b = 59 + int(norm * 3 * 96), 130 + int(norm * 3 * 125), 246
            elif norm < 0.66:
                adj = (norm - 0.33) * 3
                r, g, b = 155 + int(adj * 100), 255, 200 - int(adj * 200)
            else:
                adj = (norm - 0.66) * 3
                r, g, b = 255, 255 - int(adj * 180), 0
            return f"#{min(255,r):02x}{min(255,g):02x}{min(255,b):02x}"

        grid_step = 0.05
        for _, row in no2_avg.iterrows():
            color = no2_color(row["no2_column_mol_m2"])
            folium.Rectangle(
                bounds=[
                    [row["lat"] - grid_step/2, row["lon"] - grid_step/2],
                    [row["lat"] + grid_step/2, row["lon"] + grid_step/2],
                ],
                color=color, fill=True,
                fill_color=color, fill_opacity=0.6, weight=0,
                tooltip=f"NO2: {row['no2_column_mol_m2']*1e6:.1f} µmol/m²",
            ).add_to(m_no2)

        folium.CircleMarker(
            location=[lat, lon], radius=8,
            color="white", weight=2,
            fill=True, fill_color="#1a1133", fill_opacity=0.9,
        ).add_to(m_no2)

        st_folium(m_no2, width=None, height=400, returned_objects=[])

    # Fire map
    if fires["fires"]:
        st.divider()
        st.markdown("**VIIRS Fire Detections**")
        m_fire = _make_folium_map(lat, lon, zoom=7)

        for fire in fires["fires"]:
            color = "#ef4444" if fire["type"] == "crop_residue" else "#f97316"
            folium.CircleMarker(
                location=[fire["lat"], fire["lon"]],
                radius=max(4, min(12, fire["frp"] / 10)),
                color=color, weight=1,
                fill=True, fill_color=color, fill_opacity=0.7,
                tooltip=f"FRP: {fire['frp']:.0f} MW · {fire['type']} · {fire['distance_km']} km",
            ).add_to(m_fire)

        st_folium(m_fire, width=None, height=400, returned_objects=[])


def render_power_plants(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render power plant map with capacity and fuel type.
    Called by app.py when viz_type == 'power_plants'.
    """
    plants = sources_get_power_plants(lat, lon, max(radius_km, 100.0))

    if not plants:
        st.info("No thermal power plants found within 100km.")
        return

    m = _make_folium_map(lat, lon, zoom=8)

    fuel_colors = {
        "Coal": "#ef4444",
        "Gas":  "#3b82f6",
        "Oil":  "#f59e0b",
        "Petcoke": "#8b5cf6",
    }

    for plant in plants:
        color = fuel_colors.get(plant.get("fuel", ""), "#64748b")
        radius = max(8, min(20, plant.get("capacity_mw", 100) / 50))

        folium.CircleMarker(
            location=[plant["lat"], plant["lon"]],
            radius=radius,
            color="white", weight=2,
            fill=True, fill_color=color, fill_opacity=0.85,
            tooltip=(
                f"⚡ {plant['name']}<br>"
                f"Fuel: {plant.get('fuel', 'unknown')}<br>"
                f"Capacity: {plant.get('capacity_mw', 0):.0f} MW<br>"
                f"Distance: {plant['distance_km']} km"
            ),
        ).add_to(m)

    folium.CircleMarker(
        location=[lat, lon], radius=8,
        color="#8b5cf6", weight=3,
        fill=True, fill_color="#1a1133", fill_opacity=0.9,
        tooltip="Query center",
    ).add_to(m)

    st_folium(m, width=None, height=500, returned_objects=[])

    st.markdown("**Power Plants**")
    df = pd.DataFrame(plants)[["name", "fuel", "capacity_mw", "distance_km"]]
    df.columns = ["Name", "Fuel", "Capacity (MW)", "Distance (km)"]
    df["Capacity (MW)"] = df["Capacity (MW)"].round(0).astype(int)
    st.dataframe(df.sort_values("Distance (km)"), use_container_width=True, hide_index=True)


def render_intervention(lat, lon, radius_km, pollutant, t0, t1, ai_interventions=None):
    """
    Render what-if intervention analysis with before/after comparison.
    Called by app.py when viz_type == 'intervention'.

    Args:
        ai_interventions: Pre-generated interventions (from app.py).
                         If None, falls back to hardcoded interventions.
    """
    summary    = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)
    all_src    = sources_get_all(lat, lon, radius_km)

    if summary.empty:
        st.warning("No data available.")
        return

    mean_pm = summary["mean"].mean()
    t1_dt   = datetime.fromisoformat(t1)
    attr    = attribution_rank_sources(
        all_src["summary"], conditions, pollutant,
        hour=t1_dt.hour,
        is_winter=t1_dt.month in (10, 11, 12, 1, 2, 3)
    )

    st.markdown("### Intervention Impact Analysis")

    # Use pre-generated interventions if available; otherwise use hardcoded
    if ai_interventions:
        st.success("💡 AI-Generated Intervention Scenarios (based on your location)")
        st.divider()
        _render_ai_interventions_fragment(ai_interventions, mean_pm)
    else:
        _render_hardcoded_interventions_fragment(attr, mean_pm, pollutant)


@st.fragment
def _render_ai_interventions_fragment(ai_interventions, mean_pm):
    """Fragment that reruns only when intervention selection changes."""
    selected_idx = st.selectbox("Select scenario:", range(len(ai_interventions)), format_func=lambda i: ai_interventions[i].get("name", f"Scenario {i+1}"), key="ai_intervention_select")

    # Render selected AI intervention
    scenario = ai_interventions[selected_idx]
    col1, col2, col3 = st.columns(3)
    col1.metric("Feasibility", scenario.get("feasibility", "N/A").title())
    col2.metric("Estimated Reduction", scenario.get("estimated_reduction", "N/A"))
    col3.metric("Expected AQI After", f"{scenario.get('expected_aqi', mean_pm):.0f} µg/m³")

    st.markdown(f"**Description:** {scenario.get('description', 'N/A')}")
    st.markdown(f"**Timeframe:** {scenario.get('timeframe', 'N/A')}")
    st.markdown(f"**Co-benefits:** {scenario.get('co_benefits', 'N/A')}")


@st.fragment
def _render_hardcoded_interventions_fragment(attr, mean_pm, pollutant):
    """Fragment that reruns only when intervention selection changes."""
    # Fallback to hardcoded interventions
    interventions = {
        "🚗 30% Traffic Reduction":      ("traffic",       0.30, "Odd-even, enhanced public transit"),
        "🏭 50% Industrial Shutdown":    ("industry",      0.50, "Temporary closure of polluting units"),
        "🧱 Brick Kiln Ban":             ("brick_kilns",   0.70, "Seasonal kiln shutdown"),
        "⚡ Power Plant Curtailment":    ("power_plants",  0.40, "Reduce thermal generation"),
        "🚧 Construction Halt":          ("road_dust",     0.50, "Stop construction activities"),
    }

    selected = st.selectbox("Select intervention:", list(interventions.keys()), key="hardcoded_intervention_select")
    source_key, reduction, desc = interventions[selected]

    if source_key in attr:
        contrib    = attr[source_key]["contribution"]
        impact_pct = contrib * reduction
    else:
        impact_pct = 0.05

    impact_ug       = impact_pct * mean_pm
    new_level       = mean_pm - impact_ug
    cat_before, _   = get_aqi_category(mean_pm, pollutant)
    cat_after, _    = get_aqi_category(new_level, pollutant)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current",            f"{mean_pm:.0f} µg/m³",  cat_before)
    col2.metric("After Intervention", f"{new_level:.0f} µg/m³", f"-{impact_ug:.0f}")
    col3.metric("Reduction",          f"{impact_pct*100:.0f}%")
    col4.metric("New Category",       cat_after)

    st.caption(f"*{desc}*")

    # Before / after bar chart
    st.divider()
    st.markdown("**Before / After by Source**")

    before_vals = []
    after_vals  = []
    sources_list = []

    for src, info in attr.items():
        contrib_ug = info["contribution"] * mean_pm
        if source_key == src:
            after_ug = contrib_ug * (1 - reduction)
        else:
            after_ug = contrib_ug * (1 - impact_pct * 0.3)

        sources_list.append(src.replace("_", " ").title())
        before_vals.append(round(contrib_ug, 1))
        after_vals.append(round(after_ug, 1))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Before", x=sources_list, y=before_vals,
        marker_color="#ef4444", opacity=0.8
    ))
    fig.add_trace(go.Bar(
        name="After", x=sources_list, y=after_vals,
        marker_color="#22c55e", opacity=0.8
    ))
    fig.update_layout(
        barmode="group", height=300,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9b92b8"),
        margin=dict(l=40, r=20, t=20, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        yaxis=dict(title=f"{pollutant} (µg/m³)", gridcolor="#2d2640"),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_trends(lat, lon, radius_km, pollutant, t0, t1):
    """
    Render time series trend with diurnal pattern and daily comparison.
    Called by app.py when viz_type == 'trends'.
    Note: Full trend data available in phase 2 with database layer.
    In phase 1 shows current conditions with met context.
    """
    timeseries = aq_get_timeseries(lat, lon, radius_km, pollutant, t0, t1)
    met_ts     = met_get_timeseries(lat, lon, t0, t1)
    conditions = met_get_conditions(lat, lon, t0, t1)

    if timeseries.empty:
        st.warning("No data available.")
        return

    # Main time series
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timeseries["time"],
        y=timeseries[pollutant],
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#8b5cf6", width=2),
        fillcolor="rgba(139,92,246,0.1)",
        hovertemplate="%{x}<br>%{y:.0f} µg/m³<extra></extra>",
    ))

    breakpoints = AQI_BREAKPOINTS.get(pollutant, AQI_BREAKPOINTS["PM2.5"])
    for thresh, color, label in zip(
        breakpoints[1:4],
        ["#84cc16", "#f97316", "#ef4444"],
        ["Satisfactory", "Poor", "Very Poor"]
    ):
        fig.add_hline(
            y=thresh, line_dash="dot",
            line_color=color, opacity=0.4,
            annotation_text=label,
            annotation_position="right"
        )

    fig.update_layout(
        height=250,
        margin=dict(l=50, r=80, t=20, b=40),
        xaxis=dict(gridcolor="#2d2640", title="Time"),
        yaxis=dict(gridcolor="#2d2640", title=f"{pollutant} (µg/m³)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#9b92b8", size=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Met context
    st.divider()
    col1, col2, col3 = st.columns(3)
    col1.metric("Mean Wind", f"{conditions['wind_speed']:.1f} m/s {conditions['wind_label']}")
    col2.metric("Mean BLH",  f"{conditions['blh_mean']} m")
    col3.metric("Stagnation", conditions["stagnation_risk"].title())

    st.info(
        "📊 Full historical trends (7d/30d diurnal patterns, weekend vs weekday) "
        "will be available in phase 2 when the database layer is added."
    )


def render_rag(user_query: str):
    """Render grounded RAG response and linked sources from urbanemissions.info."""
    if not user_query.strip():
        st.warning("Please enter a query for RAG retrieval.")
        return

    rag = get_rag_pipeline()
    indexed_chunks = rag.chunk_count()
    if indexed_chunks == 0:
        st.warning(
            "RAG index is empty in the configured Chroma directory: "
            f"{RAG_CHROMA_DIR}. "
            "Point RAG_CHROMA_DIR to your existing UE index (for example ./ue/chroma_db)."
        )
        return

    answer, sources = rag.query(user_query)

    def _to_superscript_citations(text: str) -> str:
        return re.sub(r"\[(\d+)\]", r"<sup>[\1]</sup>", text)

    st.markdown("**RAG Answer (UrbanEmissions Knowledge Base)**")
    answer_html = _to_superscript_citations(answer)
    st.markdown(
        (
            "<div style='padding:12px 16px;background:#172a1c;border:1px solid #2f6a3f;"
            "border-radius:8px;color:#d9f7e2;'>"
            f"{answer_html}"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.caption(f"Indexed chunks available: {indexed_chunks}")

    st.markdown("**Sources**")
    if not sources:
        st.info("No sources found for this query.")
        return

    for idx, src in enumerate(sources, 1):
        title = src.get("title", "Untitled")
        url = src.get("url", "")
        category = src.get("category", "General")
        snippet = src.get("snippet", "")
        quotes = src.get("quotes", [])

        label = f"[{idx}] {title} ({category})"
        with st.expander(label, expanded=False):
            if url:
                st.markdown(f"Link: [{url}]({url})")
            if snippet:
                st.caption(snippet)
            if quotes:
                st.markdown("**Quotes**")
                for q in quotes[:3]:
                    st.markdown(f"> {q}")