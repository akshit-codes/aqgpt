"""Air quality data from WAQI (World Air Quality Index). https://aqicn.org/api/
Provides real-time AQ readings from CPCB and other ground stations worldwide.
"""

import time
import numpy as np
import pandas as pd
import requests
import streamlit as st

from aqgpt_core.config import WAQI_BASE, WAQI_TOKEN

# WAQI pollutant key mapping
_WAQI_KEY = {
    "PM2.5": "pm25",
    "PM10":  "pm10",
    "NO2":   "no2",
    "SO2":   "so2",
    "O3":    "o3",
    "CO":    "co",
}

def _infer_station_type(name: str) -> str:
    """Guess station type from name — used for attribution weighting."""
    n = name.lower()
    if any(k in n for k in ("highway", "road", "traffic", "ito", "junction")):
        return "traffic"
    if any(k in n for k in ("industrial", "wazirpur", "okhla", "bawana", "manesar")):
        return "industrial"
    return "background"


@st.cache_data(ttl=600)
def aq_get_stations(lat: float, lon: float, radius_km: float) -> pd.DataFrame:
    """
    Fetch all monitoring stations within radius_km of (lat, lon).

    Called by: LLM (to know which stations are available) and render
    functions (map markers, station ranking table).

    Returns DataFrame with columns:
        station_id, station_name, lat, lon, station_type, aqi
    """
    deg = radius_km / 111.0
    bbox = f"{lat - deg},{lon - deg},{lat + deg},{lon + deg}"

    try:
        r = requests.get(
            f"{WAQI_BASE}/map/bounds/",
            params={"latlng": bbox, "token": WAQI_TOKEN},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        st.warning(f"WAQI station fetch failed: {e}")
        return pd.DataFrame()

    if data.get("status") != "ok":
        st.warning(f"WAQI: {data.get('data', 'unknown error')}")
        return pd.DataFrame()

    rows = []
    for s in data.get("data", []):
        aqi_raw = s.get("aqi", "-")
        rows.append({
            "station_id":   str(s["uid"]),
            "station_name": s["station"]["name"],
            "lat":          float(s["lat"]),
            "lon":          float(s["lon"]),
            "aqi":          float(aqi_raw) if str(aqi_raw).lstrip("-").isdigit() else np.nan,
            "station_type": _infer_station_type(s["station"]["name"]),
        })

    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def aq_get_current(lat: float, lon: float, radius_km: float, pollutant: str) -> pd.DataFrame:
    """
    Fetch current pollutant reading per station within radius_km.

    Called by: LLM (for 'what is the air quality right now' queries)
    and render functions (AQI card, conditions view, station ranking).

    Returns DataFrame with columns:
        station_id, station_name, lat, lon, station_type, mean, max, min, std
    Note: for current data mean/max/min/std are all the same value —
    they match the schema expected by app.py render functions.
    """
    stations = aq_get_stations(lat, lon, radius_km)
    if stations.empty:
        return pd.DataFrame()

    waqi_key = _WAQI_KEY.get(pollutant, "pm25")
    rows = []

    for _, stn in stations.iterrows():
        try:
            r = requests.get(
                f"{WAQI_BASE}/feed/@{stn['station_id']}/",
                params={"token": WAQI_TOKEN},
                timeout=8,
            )
            r.raise_for_status()
            d = r.json()

            if d.get("status") != "ok":
                continue

            val = d["data"].get("iaqi", {}).get(waqi_key, {}).get("v")
            if val is not None:
                val = float(val)
                rows.append({
                    "station_id":   stn["station_id"],
                    "station_name": stn["station_name"],
                    "lat":          stn["lat"],
                    "lon":          stn["lon"],
                    "station_type": stn["station_type"],
                    "mean":         val,
                    "max":          val,
                    "min":          val,
                    "std":          0.0,
                })
            time.sleep(0.1)
        except requests.RequestException:
            continue

    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def aq_get_summary(lat: float, lon: float, radius_km: float, pollutant: str, t0: str, t1: str) -> pd.DataFrame:
    """
    Get per-station AQ summary for a time window.

    Called by: render functions (station ranking, intervention analysis,
    health advisory, spatial map).

    For recent windows (<=2 days) uses live WAQI data.
    For historical windows returns current data as approximation
    until database layer is added in phase 2.

    Returns DataFrame with columns:
        station_id, station_name, lat, lon, station_type, mean, max, min, std
    """
    return aq_get_current(lat, lon, radius_km, pollutant)

_POLLUTANT_COL_MAP = {
    "pm2.5": "pm25", "pm25": "pm25",
    "pm10":  "pm10",
    "o3":    "o3",
    "no2":   "no2",
    "so2":   "so2",
    "co":    "co",
    "aqi":   "aqi",
}

@st.cache_data(ttl=600)
def aq_get_timeseries(
    lat: float,
    lon: float,
    radius_km: float,
    pollutant: str,
    t0: str,
    t1: str,
    aqi_csv: str = "./aqgpt_core/data/aqi_data.csv",
) -> pd.DataFrame:
    import math
    from datetime import datetime
    from pathlib import Path

    print(f"\n[timeseries] called: lat={lat}, lon={lon}, radius={radius_km}km, pollutant={pollutant}, t0={t0}, t1={t1}, csv={aqi_csv}")

    # ── 1. Resolve pollutant column name ──────────────────────────────────
    col = _POLLUTANT_COL_MAP.get(pollutant.lower())
    if col is None:
        raise ValueError(f"Unknown pollutant '{pollutant}'. Valid: {list(_POLLUTANT_COL_MAP.keys())}")

    print(f"[timeseries] resolved column: '{col}'")

    # ── 2. Try CSV first ───────────────────────────────────────────────────
    if not Path(aqi_csv).exists():
        print(f"[timeseries] CSV not found at: {Path(aqi_csv).resolve()}")
    else:
        print(f"[timeseries] CSV found at: {Path(aqi_csv).resolve()}")
        df = pd.read_csv(aqi_csv, parse_dates=["fetched_at_utc"])
        print(f"[timeseries] CSV total rows: {len(df)}")
        print(f"[timeseries] CSV time range: {df['fetched_at_utc'].min()} → {df['fetched_at_utc'].max()}")
        print(f"[timeseries] CSV columns: {list(df.columns)}")
        print(f"[timeseries] '{col}' non-null rows: {df[col].notna().sum()}")

        df = df[df["error"].isna() | (df["error"].astype(str).str.strip() == "")].copy()
        df = df[df[col].notna()].copy()
        print(f"[timeseries] after error+notna filter: {len(df)} rows")

        if df.empty:
            print(f"[timeseries] FALLBACK: no valid rows after filter")
        else:
            t0_ts = pd.Timestamp(t0)
            t1_ts = pd.Timestamp(t1)
            print(f"[timeseries] filtering time: {t0_ts} → {t1_ts}")
            df_time = df[(df["fetched_at_utc"] >= t0_ts) & (df["fetched_at_utc"] <= t1_ts)].copy()
            print(f"[timeseries] after time filter: {len(df_time)} rows")

            if df_time.empty:
                print(f"[timeseries] FALLBACK: time range mismatch. CSV has {df['fetched_at_utc'].min()} → {df['fetched_at_utc'].max()}")
            else:
                def haversine_km(lat1, lon1, lat2, lon2):
                    R = 6371.0
                    dlat = math.radians(lat2 - lat1)
                    dlon = math.radians(lon2 - lon1)
                    a = (math.sin(dlat / 2) ** 2
                         + math.cos(math.radians(lat1))
                         * math.cos(math.radians(lat2))
                         * math.sin(dlon / 2) ** 2)
                    return R * 2 * math.asin(math.sqrt(a))

                df_time["_dist_km"] = df_time.apply(
                    lambda r: haversine_km(lat, lon, r["input_lat"], r["input_lon"]), axis=1
                )
                print(f"[timeseries] closest station: {df_time['_dist_km'].min():.1f} km, farthest: {df_time['_dist_km'].max():.1f} km")
                df_near = df_time[df_time["_dist_km"] <= radius_km].copy()
                print(f"[timeseries] after radius filter ({radius_km} km): {len(df_near)} rows")
                
                if df_near.empty:
                    print(f"[timeseries] FALLBACK: no stations within {radius_km} km")
                else:
                    df_near["time"] = df_near["fetched_at_utc"].dt.floor("h")
                    result = (
                        df_near.groupby("time")[col]
                        .mean()
                        .round(1)
                        .reset_index()
                        .rename(columns={col: pollutant})
                        .sort_values("time")
                        .reset_index(drop=True)
                    )
                    print(f"[timeseries] SUCCESS: returning {len(result)} hourly rows")
                    return result

    # ── 3. Fallback ────────────────────────────────────────────────────────
    print(f"[timeseries] hitting live fallback via aq_get_current")
    summary = aq_get_current(lat, lon, radius_km, pollutant)
    if summary.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "time":    [datetime.now()],
        pollutant: [round(float(summary["mean"].mean()), 1)],
    })

def aq_get_aqi_snapshot(lat: float, lon: float, radius_km: float, pollutant: str) -> dict:
    """
    Get a single clean AQI snapshot for a location — designed for LLM use.

    Called by: LLM (for any query needing current AQ context).

    Returns a flat dict with:
        mean_level, max_level, min_level, aqi_category,
        n_stations, worst_station, best_station, pollutant
    """
    from aqgpt_core.config import AQI_BREAKPOINTS

    summary = aq_get_current(lat, lon, radius_km, pollutant)
    if summary.empty:
        return {"error": "No stations found in this area."}

    mean_val = round(float(summary["mean"].mean()), 1)
    breakpoints = AQI_BREAKPOINTS.get(pollutant, AQI_BREAKPOINTS["PM2.5"])
    categories = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]

    category = categories[-1]
    for i, bp in enumerate(breakpoints):
        if mean_val <= bp:
            category = categories[i]
            break

    worst = summary.loc[summary["mean"].idxmax()]
    best  = summary.loc[summary["mean"].idxmin()]

    return {
        "pollutant":     pollutant,
        "mean_level":    mean_val,
        "max_level":     round(float(summary["max"].max()), 1),
        "min_level":     round(float(summary["min"].min()), 1),
        "aqi_category":  category,
        "n_stations":    len(summary),
        "worst_station": worst["station_name"],
        "best_station":  best["station_name"],
    }