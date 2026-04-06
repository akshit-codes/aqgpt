"""Session-level data cache to avoid repeated API calls."""

import streamlit as st
from functools import lru_cache
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple


def init_data_cache():
    """Initialize data cache in session state."""
    if "data_cache" not in st.session_state:
        st.session_state.data_cache = {}
    if "tool_calls_log" not in st.session_state:
        st.session_state.tool_calls_log = []


def get_cache_key(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str, tool_name: str) -> str:
    """Generate cache key for a tool invocation."""
    return f"{tool_name}:{lat}:{lon}:{radius_km}:{pollutant}:{t0}:{t1}"


def cache_tool_result(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str, tool_name: str, result: Any):
    """Cache a tool result."""
    init_data_cache()
    key = get_cache_key(lat, lon, radius_km, pollutant, t0, t1, tool_name)
    st.session_state.data_cache[key] = {
        "result": result,
        "timestamp": datetime.now()
    }


def get_cached_tool_result(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str, tool_name: str) -> Any:
    """Get a cached tool result, or None if not cached."""
    init_data_cache()
    key = get_cache_key(lat, lon, radius_km, pollutant, t0, t1, tool_name)
    cached = st.session_state.data_cache.get(key)
    if cached:
        return cached["result"]
    return None


def log_tool_call(tool_name: str, params: Dict[str, Any], from_cache: bool = False):
    """Log a tool call for debugging."""
    init_data_cache()
    st.session_state.tool_calls_log.append({
        "tool": tool_name,
        "params": params,
        "from_cache": from_cache,
        "timestamp": datetime.now().isoformat()
    })


def get_tool_calls_log() -> list:
    """Get the log of all tool calls made."""
    init_data_cache()
    return st.session_state.tool_calls_log


def clear_tool_calls_log():
    """Clear the tool calls log."""
    init_data_cache()
    st.session_state.tool_calls_log = []
