"""Extract key data from visualization fetches for Gemini analysis.

Provides functions to gather data that each visualization would use,
then send to Gemini for comprehensive answer generation.
Includes caching to avoid repeated API calls.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any

from aqgpt_core.tools.aq import aq_get_summary, aq_get_timeseries, aq_get_aqi_snapshot
from aqgpt_core.tools.met import met_get_conditions, met_get_stagnation, met_get_timeseries
from aqgpt_core.tools.satellite import satellite_get_fires, satellite_get_no2, satellite_get_aod
from aqgpt_core.tools.sources import sources_get_all, sources_get_power_plants
from aqgpt_core.tools.attribution import attribution_rank_sources
from aqgpt_core.rag import get_rag_pipeline
from aqgpt_core.llm.session_cache import (
    get_cached_tool_result, cache_tool_result, log_tool_call
)


def _call_tool_cached(tool_name: str, tool_func, params: Dict[str, Any], lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str):
    """Call a tool with caching."""
    result = get_cached_tool_result(lat, lon, radius_km, pollutant, t0, t1, tool_name)

    if result is not None:
        log_tool_call(tool_name, params, from_cache=True)
        return result

    # Call the tool
    result = tool_func(**params)

    # Cache the result
    cache_tool_result(lat, lon, radius_km, pollutant, t0, t1, tool_name, result)
    log_tool_call(tool_name, params, from_cache=False)

    return result


def extract_conditions_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for conditions visualization."""
    try:
        summary = _call_tool_cached(
            "aq_get_summary",
            aq_get_summary,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "pollutant": pollutant, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        timeseries = _call_tool_cached(
            "aq_get_timeseries",
            aq_get_timeseries,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "pollutant": pollutant, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        conditions = _call_tool_cached(
            "met_get_conditions",
            met_get_conditions,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )

        if summary.empty:
            return {"error": "No air quality station data found"}

        # Check if meteorological data failed
        if conditions.get("error"):
            return {"error": f"Unable to complete conditions analysis: {conditions.get('error_reason', conditions.get('error'))}"}

        return {
            "viz_type": "conditions",
            "summary": {
                "mean": float(summary["mean"].mean()),
                "max": float(summary["max"].max()),
                "min": float(summary["min"].min()),
                "n_stations": len(summary),
                "worst_station": summary.nlargest(1, "mean")["station_name"].values[0] if not summary.empty else None,
            },
            "timeseries_length": len(timeseries),
            "conditions": conditions,
        }
    except Exception as e:
        return {"error": f"Conditions data extraction failed: {str(e)}"}


def extract_satellite_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for satellite visualization."""
    try:
        fires = _call_tool_cached(
            "satellite_get_fires",
            satellite_get_fires,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        no2 = _call_tool_cached(
            "satellite_get_no2",
            satellite_get_no2,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        aod = _call_tool_cached(
            "satellite_get_aod",
            satellite_get_aod,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )

        return {
            "viz_type": "satellite",
            "fires": {
                "total": fires.get("summary", {}).get("total_fires", 0),
                "crop_residue": fires.get("summary", {}).get("crop_residue_fires", 0),
                "total_frp_mw": fires.get("summary", {}).get("total_frp_mw", 0),
            },
            "no2": no2.get("summary", {}),
            "aod": aod.get("summary", {}),
        }
    except Exception as e:
        return {"error": str(e)}


def extract_attribution_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for attribution visualization."""
    try:
        sources = _call_tool_cached(
            "sources_get_all",
            sources_get_all,
            {"lat": lat, "lon": lon, "radius_km": radius_km},
            lat, lon, radius_km, pollutant, t0, t1
        )
        conditions = _call_tool_cached(
            "met_get_conditions",
            met_get_conditions,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )

        now = datetime.now()
        hour = now.hour
        is_winter = now.month in [10, 11, 12, 1, 2]

        attribution = attribution_rank_sources(sources["summary"], conditions, pollutant, hour, is_winter)

        return {
            "viz_type": "attribution",
            "sources": {
                "industries": len(sources.get("industries", [])),
                "power_plants": len(sources.get("power_plants", [])),
                "kilns": len(sources.get("kilns", [])),
            },
            "attribution": attribution,
        }
    except Exception as e:
        return {"error": str(e)}


def extract_why_bad_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for why_bad visualization."""
    try:
        summary = _call_tool_cached(
            "aq_get_summary",
            aq_get_summary,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "pollutant": pollutant, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        met_ts = _call_tool_cached(
            "met_get_timeseries",
            met_get_timeseries,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        stagnation = _call_tool_cached(
            "met_get_stagnation",
            met_get_stagnation,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        conditions = _call_tool_cached(
            "met_get_conditions",
            met_get_conditions,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )

        if summary.empty:
            return {"error": "No air quality station data found"}

        # Check if meteorological data failed
        if conditions.get("error"):
            return {"error": f"Unable to analyze pollution causes: {conditions.get('error_reason', conditions.get('error'))}"}

        return {
            "viz_type": "why_bad",
            "pollutant_level": float(summary["mean"].mean()),
            "stagnation": stagnation,
            "conditions": conditions,
            "met_summary": {
                "wind_speed": conditions.get("wind_speed"),
                "wind_direction": conditions.get("wind_direction"),
                "blh_mean": conditions.get("blh_mean"),
                "calm_hours": conditions.get("wind_calm_hours", 0),
            },
        }
    except Exception as e:
        return {"error": f"Why bad data extraction failed: {str(e)}"}


def extract_health_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for health advisory visualization."""
    try:
        summary = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)

        if summary.empty:
            return {"error": "No station data found"}

        return {
            "viz_type": "health",
            "pollutant_level": float(summary["mean"].mean()),
            "pollutant": pollutant,
            "location": {"lat": lat, "lon": lon},
        }
    except Exception as e:
        return {"error": str(e)}


def extract_wind_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for wind transport visualization."""
    try:
        conditions = _call_tool_cached(
            "met_get_conditions",
            met_get_conditions,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        sources = _call_tool_cached(
            "sources_get_all",
            sources_get_all,
            {"lat": lat, "lon": lon, "radius_km": radius_km},
            lat, lon, radius_km, pollutant, t0, t1
        )

        # Check if meteorological data failed
        if conditions.get("error"):
            return {"error": f"Unable to analyze wind patterns: {conditions.get('error_reason', conditions.get('error'))}"}

        return {
            "viz_type": "wind",
            "wind_direction": conditions.get("wind_direction"),
            "wind_speed": conditions.get("wind_speed"),
            "wind_label": conditions.get("wind_label"),
            "sources_count": {
                "industries": len(sources.get("industries", [])),
                "power_plants": len(sources.get("power_plants", [])),
                "kilns": len(sources.get("kilns", [])),
            },
        }
    except Exception as e:
        return {"error": f"Wind data extraction failed: {str(e)}"}


def extract_power_plants_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for power plants visualization."""
    try:
        plants = _call_tool_cached(
            "sources_get_power_plants",
            sources_get_power_plants,
            {"lat": lat, "lon": lon, "radius_km": radius_km},
            lat, lon, radius_km, pollutant, t0, t1
        )

        total_capacity = sum([p.get("capacity_mw", 0) for p in plants]) if plants else 0
        fuel_types = {}
        for plant in plants:
            fuel = plant.get("fuel", "Unknown")
            fuel_types[fuel] = fuel_types.get(fuel, 0) + 1

        return {
            "viz_type": "power_plants",
            "plant_count": len(plants),
            "total_capacity_mw": total_capacity,
            "fuel_mix": fuel_types,
        }
    except Exception as e:
        return {"error": str(e)}


def extract_spatial_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for spatial map visualization."""
    try:
        summary = aq_get_summary(lat, lon, radius_km, pollutant, t0, t1)

        if summary.empty:
            return {"error": "No station data found"}

        return {
            "viz_type": "spatial",
            "n_stations": len(summary),
            "mean_pollution": float(summary["mean"].mean()),
            "max_pollution": float(summary["max"].max()),
            "geographic_spread": "station distribution across area",
        }
    except Exception as e:
        return {"error": str(e)}


def extract_intervention_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for intervention visualization."""
    try:
        summary = _call_tool_cached(
            "aq_get_summary",
            aq_get_summary,
            {"lat": lat, "lon": lon, "radius_km": radius_km, "pollutant": pollutant, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )
        sources = _call_tool_cached(
            "sources_get_all",
            sources_get_all,
            {"lat": lat, "lon": lon, "radius_km": radius_km},
            lat, lon, radius_km, pollutant, t0, t1
        )
        conditions = _call_tool_cached(
            "met_get_conditions",
            met_get_conditions,
            {"lat": lat, "lon": lon, "t0": t0, "t1": t1},
            lat, lon, radius_km, pollutant, t0, t1
        )

        now = datetime.now()
        hour = now.hour
        is_winter = now.month in [10, 11, 12, 1, 2]

        attribution = attribution_rank_sources(sources["summary"], conditions, pollutant, hour, is_winter)

        return {
            "viz_type": "intervention",
            "current_aqi": float(summary["mean"].mean()) if not summary.empty else 0,
            "top_sources": list(attribution.keys())[:3],
        }
    except Exception as e:
        return {"error": str(e)}


def extract_trends_data(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> Dict[str, Any]:
    """Extract data for trends visualization."""
    try:
        timeseries = aq_get_timeseries(lat, lon, radius_km, pollutant, t0, t1)
        met_ts = met_get_timeseries(lat, lon, t0, t1)

        if timeseries.empty:
            return {"error": "No trend data found"}

        return {
            "viz_type": "trends",
            "data_points": len(timeseries),
            "mean_value": float(timeseries[pollutant].mean()) if pollutant in timeseries.columns else 0,
            "max_value": float(timeseries[pollutant].max()) if pollutant in timeseries.columns else 0,
            "min_value": float(timeseries[pollutant].min()) if pollutant in timeseries.columns else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def extract_rag_data(user_query: str, chat_history: list[dict] | None = None) -> Dict[str, Any]:
    """Extract data for RAG visualization and summary synthesis."""
    if not user_query.strip():
        return {"error": "Query is empty"}

    try:
        rag = get_rag_pipeline()
        answer, sources = rag.query(user_query, chat_history=chat_history)
        return {
            "viz_type": "rag",
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
        }
    except Exception as e:
        return {"error": f"RAG retrieval failed: {str(e)}"}


def extract_data_by_viz_type(viz_type: str, lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str, user_query: str = "", chat_history: list[dict] | None = None) -> Dict[str, Any]:
    """Extract data for a given visualization type."""
    extractors = {
        "conditions": extract_conditions_data,
        "satellite": extract_satellite_data,
        "attribution": extract_attribution_data,
        "why_bad": extract_why_bad_data,
        "health": extract_health_data,
        "wind": extract_wind_data,
        "power_plants": extract_power_plants_data,
        "spatial": extract_spatial_data,
        "intervention": extract_intervention_data,
        "trends": extract_trends_data,
    }

    if viz_type == "rag":
        return extract_rag_data(user_query=user_query, chat_history=chat_history)

    extractor = extractors.get(viz_type)
    if extractor:
        return extractor(lat, lon, radius_km, pollutant, t0, t1)

    return {"error": f"Unknown viz_type: {viz_type}"}
