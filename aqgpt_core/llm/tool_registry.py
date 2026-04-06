"""Tool registry and executor for Gemini function calling."""

import json
from datetime import datetime, timedelta
from typing import Any, Callable, Dict

from aqgpt_core.tools.aq import aq_get_aqi_snapshot, aq_get_stations
from aqgpt_core.tools.met import met_get_conditions, met_get_stagnation
from aqgpt_core.tools.satellite import satellite_get_fires, satellite_get_no2, satellite_get_aod
from aqgpt_core.tools.sources import sources_get_all
from aqgpt_core.tools.attribution import attribution_rank_sources

# Tool definitions for Gemini function calling
AVAILABLE_TOOLS = [
    {
        "name": "get_aqi_snapshot",
        "description": "Get current air quality snapshot for a location with mean/max levels and AQI category",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "radius_km": {"type": "number", "description": "Search radius in km"},
                "pollutant": {
                    "type": "string",
                    "enum": ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"],
                    "description": "Pollutant to check"
                },
            },
            "required": ["lat", "lon", "radius_km", "pollutant"],
        },
    },
    {
        "name": "get_weather_conditions",
        "description": "Get meteorological conditions (wind, temperature, boundary layer height, stagnation risk)",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number", "description": "Latitude"},
                "lon": {"type": "number", "description": "Longitude"},
                "t0": {"type": "string", "description": "Start time (ISO format, e.g., 2024-01-01T00:00:00Z)"},
                "t1": {"type": "string", "description": "End time (ISO format)"},
            },
            "required": ["lat", "lon", "t0", "t1"],
        },
    },
    {
        "name": "get_active_fires",
        "description": "Get active fire hotspots detected by satellite (VIIRS) within radius",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
                "t0": {"type": "string", "description": "Start time (ISO format)"},
                "t1": {"type": "string", "description": "End time (ISO format)"},
            },
            "required": ["lat", "lon", "radius_km", "t0", "t1"],
        },
    },
    {
        "name": "get_nearby_sources",
        "description": "Get pollution sources nearby: industries, power plants, kilns, roads",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number", "description": "Search radius in km"},
            },
            "required": ["lat", "lon", "radius_km"],
        },
    },
    {
        "name": "analyze_source_attribution",
        "description": "Analyze which pollution sources contribute what percentage to current AQ",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
                "pollutant": {
                    "type": "string",
                    "enum": ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"],
                },
                "hour": {"type": "integer", "description": "Hour of day (0-23) for context"},
                "is_winter": {"type": "boolean", "description": "Whether to consider winter conditions"},
            },
            "required": ["lat", "lon", "radius_km", "pollutant", "hour", "is_winter"],
        },
    },
    {
        "name": "get_monitoring_stations",
        "description": "Get list of air quality monitoring stations within radius",
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "radius_km": {"type": "number"},
            },
            "required": ["lat", "lon", "radius_km"],
        },
    },
]


def invoke_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool and return results.

    Args:
        tool_name: Name of tool to invoke
        **kwargs: Parameters for the tool

    Returns:
        Dict with tool results or error
    """
    try:
        if tool_name == "get_aqi_snapshot":
            result = aq_get_aqi_snapshot(
                kwargs["lat"], kwargs["lon"], kwargs["radius_km"], kwargs["pollutant"]
            )
            return {"success": True, "data": result}

        elif tool_name == "get_weather_conditions":
            result = met_get_conditions(kwargs["lat"], kwargs["lon"], kwargs["t0"], kwargs["t1"])
            return {"success": True, "data": result}

        elif tool_name == "get_active_fires":
            result = satellite_get_fires(
                kwargs["lat"], kwargs["lon"], kwargs["radius_km"], kwargs["t0"], kwargs["t1"]
            )
            return {"success": True, "data": result}

        elif tool_name == "get_nearby_sources":
            result = sources_get_all(kwargs["lat"], kwargs["lon"], kwargs["radius_km"])
            return {"success": True, "data": result}

        elif tool_name == "analyze_source_attribution":
            sources = sources_get_all(kwargs["lat"], kwargs["lon"], kwargs["radius_km"])
            conditions = met_get_conditions(
                kwargs["lat"], kwargs["lon"],
                (datetime.now() - timedelta(days=1)).isoformat(),
                datetime.now().isoformat()
            )
            result = attribution_rank_sources(
                sources["summary"],
                conditions,
                kwargs["pollutant"],
                kwargs["hour"],
                kwargs["is_winter"],
            )
            return {"success": True, "data": result}

        elif tool_name == "get_monitoring_stations":
            result = aq_get_stations(kwargs["lat"], kwargs["lon"], kwargs["radius_km"])
            return {"success": True, "data": result.to_dict(orient="records") if hasattr(result, 'to_dict') else result}

        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}

    except Exception as e:
        return {"success": False, "error": str(e), "tool": tool_name}
