"""Meteorological data from Open-Meteo (free, no API key required). https://open-meteo.com/en/docs"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from aqgpt_core.config import OPEN_METEO_FORECAST, OPEN_METEO_ARCHIVE

_HOURLY_VARS = [
    "wind_speed_10m",
    "wind_direction_10m",
    "temperature_2m",
    "relative_humidity_2m",
    "boundary_layer_height",
]

_WIND_LABELS = [
    "N","NNE","NE","ENE","E","ESE","SE","SSE",
    "S","SSW","SW","WSW","W","WNW","NW","NNW"
]

def _wind_label(degrees: float) -> str:
    return _WIND_LABELS[int(((degrees + 11.25) % 360) / 22.5)]

def _fetch_open_meteo(lat: float, lon: float, t0: str, t1: str) -> pd.DataFrame:
    """
    Internal helper - fetches raw hourly data from Open-Meteo.
    Automatically uses archive endpoint for historical data (>5 days ago)
    and forecast endpoint for recent/current data.
    Returns a DataFrame with columns: time, wind_speed, wind_dir,
    temperature, humidity, blh.
    """
    t0_dt = datetime.fromisoformat(t0)
    t1_dt = datetime.fromisoformat(t1)
    now = datetime.now()

    use_archive = (now - t0_dt).days > 5
    base_url = OPEN_METEO_ARCHIVE if use_archive else OPEN_METEO_FORECAST

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(_HOURLY_VARS),
        "timezone": "Asia/Kolkata",
        "start_date": t0_dt.strftime("%Y-%m-%d"),
        "end_date": t1_dt.strftime("%Y-%m-%d"),
    }

    try:
        r = requests.get(base_url, params=params, timeout=15)
        r.raise_for_status()
        d = r.json()
    except requests.RequestException as e:
        st.warning(f"Open-Meteo request failed: {e}")
        return pd.DataFrame()

    hourly = d.get("hourly", {})
    if not hourly.get("time"):
        return pd.DataFrame()

    df = pd.DataFrame({
        "time":        pd.to_datetime(hourly["time"]),
        "wind_speed":  hourly.get("wind_speed_10m", []),
        "wind_dir":    hourly.get("wind_direction_10m", []),
        "temperature": hourly.get("temperature_2m", []),
        "humidity":    hourly.get("relative_humidity_2m", []),
        "blh":         hourly.get("boundary_layer_height", []),
    })

    return df[(df["time"] >= t0_dt) & (df["time"] <= t1_dt)].reset_index(drop=True)


@st.cache_data(ttl=1800)
def met_get_conditions(lat: float, lon: float, t0: str, t1: str) -> dict:
    """
    Fetch summarised meteorological conditions for a location and time window.

    Called by: LLM (for weather context in any query) and render functions
    (conditions card, wind map, why-is-it-bad).

    Returns a flat dict with:
        wind_speed (m/s), wind_direction (degrees), wind_label (e.g. 'NW'),
        wind_calm_hours (hours below 1 m/s),
        blh_mean (m), blh_low_hours (hours below 500m),
        temperature_mean (C), humidity_mean (%),
        stagnation_risk ('high'/'moderate'/'low')

    t0, t1: ISO format datetime strings e.g. '2025-03-01T00:00:00'
    """
    df = _fetch_open_meteo(lat, lon, t0, t1)

    if df.empty:
        return {
            "wind_speed": 0, "wind_direction": 0, "wind_label": "N",
            "wind_calm_hours": 0, "blh_mean": 500, "blh_low_hours": 0,
            "temperature_mean": 25, "humidity_mean": 50,
            "stagnation_risk": "unknown",
        }

    # Vector mean wind direction
    u = (df["wind_speed"] * np.sin(np.radians(df["wind_dir"]))).mean()
    v = (df["wind_speed"] * np.cos(np.radians(df["wind_dir"]))).mean()
    mean_dir = float((np.degrees(np.arctan2(u, v)) + 360) % 360)
    mean_speed = float(df["wind_speed"].mean())
    calm_hours = int((df["wind_speed"] < 1.0).sum())

    blh = df["blh"].dropna()
    mean_blh = float(blh.mean()) if not blh.empty else 500
    low_blh_hours = int((blh < 500).sum())

    # Stagnation: high if mean BLH < 400m or (BLH < 600 and calm > 8h)
    if mean_blh < 400 or (mean_blh < 600 and calm_hours > 8):
        stagnation = "high"
    elif mean_blh < 700 or calm_hours > 4:
        stagnation = "moderate"
    else:
        stagnation = "low"

    return {
        "wind_speed":       round(mean_speed, 2),
        "wind_direction":   round(mean_dir, 1),
        "wind_label":       _wind_label(mean_dir),
        "wind_calm_hours":  calm_hours,
        "blh_mean":         round(mean_blh),
        "blh_low_hours":    low_blh_hours,
        "temperature_mean": round(float(df["temperature"].mean()), 1),
        "humidity_mean":    round(float(df["humidity"].mean()), 1),
        "stagnation_risk":  stagnation,
    }


@st.cache_data(ttl=1800)
def met_get_timeseries(lat: float, lon: float, t0: str, t1: str) -> pd.DataFrame:
    """
    Fetch raw hourly meteorological timeseries for a location and time window.

    Called by: render functions only (correlation charts, wind rose).
    Not intended for direct LLM use — too much data.

    Returns DataFrame with columns:
        time, wind_speed, wind_dir, temperature, humidity, blh
    """
    return _fetch_open_meteo(lat, lon, t0, t1)


@st.cache_data(ttl=1800)
def met_get_stagnation(lat: float, lon: float, t0: str, t1: str) -> dict:
    """
    Analyse atmospheric stagnation for a location and time window.

    Called by: LLM (for 'why is pollution high' and 'is there stagnation' queries).

    Returns a dict with:
        is_stagnant (bool),
        risk ('high'/'moderate'/'low'/'unknown'),
        reason (human readable string explaining why),
        blh_mean (m), wind_speed (m/s), calm_hours (int)
    """
    conditions = met_get_conditions(lat, lon, t0, t1)

    risk = conditions["stagnation_risk"]
    blh = conditions["blh_mean"]
    speed = conditions["wind_speed"]
    calm = conditions["wind_calm_hours"]

    if risk == "high":
        reason = (
            f"BLH is only {blh}m (pollutants trapped near surface) "
            f"and winds are calm for {calm} hours. "
            "Severe stagnation — pollutants cannot disperse."
        )
    elif risk == "moderate":
        reason = (
            f"BLH is {blh}m and wind speed averages {speed} m/s. "
            "Moderate stagnation — some dispersion but conditions unfavourable."
        )
    else:
        reason = (
            f"BLH is {blh}m and wind speed is {speed} m/s. "
            "Atmosphere is relatively well-mixed."
        )

    return {
        "is_stagnant": risk in ("high", "moderate"),
        "risk":        risk,
        "reason":      reason,
        "blh_mean":    blh,
        "wind_speed":  speed,
        "calm_hours":  calm,
    }