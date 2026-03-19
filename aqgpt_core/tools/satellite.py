"""Satellite data for air quality analysis.

VIIRS Fires: NASA FIRMS API (real data, requires NASA_FIRMS_KEY in .env)
             https://firms.modaps.eosdis.nasa.gov/api/area/
             Simple CSV response, fast, works today.

TROPOMI NO2: Estimated from ground NO2 readings (phase 1).
             Full integration requires Copernicus Dataspace OAuth2 + NetCDF parsing.
             Planned for phase 2 as a scheduled background job.

MODIS AOD:   Estimated from PM2.5 using empirical correlation (phase 1).
             Full integration requires NASA EarthData account + HDF4 parsing.
             Planned for phase 2 as a scheduled background job.
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st
from datetime import datetime
from io import StringIO

from aqgpt_core.config import NASA_FIRMS_BASE, NASA_FIRMS_KEY


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance in km between two coordinates."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arcsin(np.sqrt(a))


def _is_crop_fire(row: pd.Series) -> bool:
    """
    Heuristic: crop residue burning peaks Oct-Nov in Punjab/Haryana.
    FRP > 10 MW + season = crop fire signal.
    """
    try:
        month = pd.to_datetime(str(row.get("acq_date", ""))).month
        return month in (10, 11) and float(row.get("frp", 0) or 0) > 10
    except Exception:
        return False


@st.cache_data(ttl=3600)
def satellite_get_fires(
    lat: float, lon: float, radius_km: float, t0: str, t1: str
) -> dict:
    """
    Fetch active fire detections from NASA FIRMS VIIRS SNPP NRT.

    Called by: LLM (for 'are there active fires' and 'stubble burning' queries)
    and render functions (satellite view, fire hotspot map).

    Requires NASA_FIRMS_KEY in .env. Get one free at:
    https://firms.modaps.eosdis.nasa.gov/api/area/

    Returns dict with:
        fires (list of dicts: lat, lon, frp, confidence, type, distance_km),
        summary (total_fires, total_frp_mw, crop_residue_fires)

    Note: FIRMS NRT data has max 10 day window per request.
    """
    if not NASA_FIRMS_KEY:
        st.info("Add NASA_FIRMS_KEY to .env for live fire data.")
        return _empty_fires()

    deg  = radius_km / 111.0
    west = lon - deg
    east = lon + deg
    south = lat - deg
    north = lat + deg

    t0_dt = datetime.fromisoformat(t0)
    t1_dt = datetime.fromisoformat(t1)
    days  = min(10, max(1, (t1_dt - t0_dt).days + 1))

    url = f"{NASA_FIRMS_BASE}/{NASA_FIRMS_KEY}/VIIRS_SNPP_NRT/{west:.4f},{south:.4f},{east:.4f},{north:.4f}/{days}"

    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()

        if not r.text.strip() or r.text.startswith("<?xml"):
            return _empty_fires()

        df = pd.read_csv(StringIO(r.text))
    except Exception as e:
        st.warning(f"NASA FIRMS error: {e}")
        return _empty_fires()

    if df.empty:
        return _empty_fires()

    df["distance_km"] = df.apply(
        lambda row: _haversine(lat, lon, row["latitude"], row["longitude"]), axis=1
    )
    df = df[df["distance_km"] <= radius_km].copy()

    if df.empty:
        return _empty_fires()

    fires = []
    for _, row in df.iterrows():
        fires.append({
            "lat":         row["latitude"],
            "lon":         row["longitude"],
            "frp":         float(row.get("frp", 0) or 0),
            "confidence":  row.get("confidence", "n"),
            "acq_date":    str(row.get("acq_date", "")),
            "distance_km": round(row["distance_km"], 1),
            "type":        "crop_residue" if _is_crop_fire(row) else "other",
        })

    crop_fires = [f for f in fires if f["type"] == "crop_residue"]
    total_frp  = sum(f["frp"] for f in fires)

    return {
        "fires": fires,
        "summary": {
            "total_fires":        len(fires),
            "total_frp_mw":       round(total_frp, 1),
            "crop_residue_fires": len(crop_fires),
        },
    }


def _empty_fires() -> dict:
    return {
        "fires": [],
        "summary": {
            "total_fires": 0,
            "total_frp_mw": 0.0,
            "crop_residue_fires": 0,
        },
    }


@st.cache_data(ttl=3600)
def satellite_get_no2(
    lat: float, lon: float, radius_km: float, t0: str, t1: str
) -> dict:
    """
    Get TROPOMI NO2 tropospheric column data for a location and time window.

    Called by: LLM (for NO2 hotspot and satellite comparison queries)
    and render functions (satellite view, NO2 grid map).

    Phase 1: estimated from ground NO2 readings using empirical conversion.
    Conversion: 1 µg/m³ NO2 ≈ 0.5 µmol/m² (assumes ~500m BLH, typical South Asia).

    Phase 2 TODO: Replace with real TROPOMI data via Copernicus Dataspace:
        - Register at https://dataspace.copernicus.eu/
        - Product: S5P_OFFL_L2__NO2____
        - Auth: OAuth2 client credentials
        - Format: NetCDF, band: nitrogendioxide_tropospheric_column
        - Integrate as background scheduled job (not real-time)

    Returns dict with:
        grid_data (list of lat/lon/no2_column_mol_m2),
        summary (mean_no2_umol_m2, n_hotspots, n_valid_days),
        resolution_km, source
    """
    from aqgpt_core.tools.aq import aq_get_summary

    no2_df = aq_get_summary(lat, lon, radius_km, "NO2", t0, t1)
    mean_no2_ground = float(no2_df["mean"].mean()) if not no2_df.empty else 40.0

    # Ground µg/m³ → tropospheric column µmol/m²
    mean_column_umol = mean_no2_ground * 0.5

    grid_data = _synthetic_no2_grid(lat, lon, radius_km, mean_column_umol)
    hotspot_threshold = mean_column_umol * 1.6
    n_hotspots = sum(
        1 for g in grid_data
        if g["no2_column_mol_m2"] * 1e6 > hotspot_threshold
    )

    t0_dt = datetime.fromisoformat(t0)
    t1_dt = datetime.fromisoformat(t1)
    n_days = max(1, (t1_dt - t0_dt).days + 1)

    return {
        "grid_data": grid_data,
        "summary": {
            "mean_no2_umol_m2": round(mean_column_umol, 2),
            "n_hotspots":       n_hotspots,
            "n_valid_days":     n_days,
        },
        "resolution_km": 5.5,
        "source":        "estimated_from_ground_no2",
    }


def _synthetic_no2_grid(
    lat: float, lon: float, radius_km: float, mean_umol: float
) -> list[dict]:
    """
    Generate a spatially varying NO2 grid from a mean value.
    Replaced entirely when real TROPOMI data is integrated in phase 2.
    """
    rng = np.random.default_rng(seed=abs(int(mean_umol * 100)) % 9999)
    deg = radius_km / 111.0
    n   = 12

    data = []
    for i in range(n):
        for j in range(n):
            cell_lat = lat - deg + i * 2 * deg / n
            cell_lon = lon - deg + j * 2 * deg / n
            noise    = rng.normal(0, 0.18)
            val_umol = max(0.0, mean_umol * (1.0 + noise))
            data.append({
                "lat": cell_lat,
                "lon": cell_lon,
                "no2_column_mol_m2": val_umol / 1e6,
            })
    return data


@st.cache_data(ttl=3600)
def satellite_get_aod(
    lat: float, lon: float, radius_km: float, t0: str, t1: str
) -> dict:
    """
    Get MODIS Aerosol Optical Depth (AOD) at 550nm for a location and time window.

    Called by: LLM (for satellite vs ground comparison queries)
    and render functions (satellite view).

    Phase 1: estimated from PM2.5 using empirical correlation.
    Typical South Asia regression: AOD ≈ PM2.5 × 0.008
    (Wang & Christopher 2003)

    Phase 2 TODO: Replace with real MODIS data via NASA LAADS DAAC:
        - Register at https://urs.earthdata.nasa.gov/
        - Product: MOD04_L2 (Terra) or MYD04_L2 (Aqua)
        - Band: Optical_Depth_Land_And_Ocean
        - Integrate as background scheduled job (not real-time)

    Returns dict with:
        summary (mean_aod, high_aod_days), source
    """
    from aqgpt_core.tools.aq import aq_get_summary

    pm25_df = aq_get_summary(lat, lon, radius_km, "PM2.5", t0, t1)
    mean_pm = float(pm25_df["mean"].mean()) if not pm25_df.empty else 60.0

    mean_aod      = round(mean_pm * 0.008, 3)
    high_aod_days = max(0, int((mean_pm - 100) / 50))

    return {
        "summary": {
            "mean_aod":      mean_aod,
            "high_aod_days": high_aod_days,
        },
        "source": "estimated_from_pm25",
    }