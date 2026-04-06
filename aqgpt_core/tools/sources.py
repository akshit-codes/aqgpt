"""Pollution source data from OpenStreetMap Overpass API and Global Power Plant Database.
https://overpass-api.de/

Provides locations of industrial areas, brick kilns, power plants, and road density
for source attribution and map visualizations.
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st

from aqgpt_core.config import OVERPASS_URL, POWER_PLANTS_CSV

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


def _bbox(lat: float, lon: float, radius_km: float) -> str:
    """Return Overpass bbox string: south,west,north,east."""
    deg = radius_km / 111.0
    return f"{lat - deg},{lon - deg},{lat + deg},{lon + deg}"


def _overpass(query: str) -> list:
    """Execute an Overpass QL query and return elements."""
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=60)
        r.raise_for_status()
        return r.json().get("elements", [])
    except requests.RequestException as e:
        st.warning(f"Overpass API error: {e}")
        return []


def _el_coords(el: dict) -> tuple:
    """Extract lat/lon from a node or way element."""
    if "lat" in el:
        return el["lat"], el["lon"]
    center = el.get("center", {})
    return center.get("lat"), center.get("lon")


@st.cache_data(ttl=86400)
def sources_get_industries(lat: float, lon: float, radius_km: float) -> list[dict]:
    """
    Fetch industrial areas and facilities within radius_km from OpenStreetMap.

    Called by: LLM (for 'which industrial clusters contribute most' queries)
    and render functions (wind transport map, source attribution).

    Returns list of dicts with:
        name, lat, lon, distance_km, type
    Sorted by distance ascending.
    """
    bbox = _bbox(lat, lon, radius_km)
    query = f"""
    [out:json][timeout:25];
    (
      node["landuse"="industrial"]({bbox});
      way["landuse"="industrial"]({bbox});
      node["industrial"]({bbox});
      way["industrial"]({bbox});
    );
    out center tags;
    """
    elements = _overpass(query)
    results = []

    for el in elements:
        elat, elon = _el_coords(el)
        if elat is None:
            continue
        dist = _haversine(lat, lon, elat, elon)
        if dist > radius_km:
            continue
        tags = el.get("tags", {})
        results.append({
            "name":        tags.get("name", tags.get("industrial", "Industrial Area")),
            "lat":         elat,
            "lon":         elon,
            "distance_km": round(dist, 2),
            "type":        tags.get("industrial", "general"),
        })

    return sorted(results, key=lambda x: x["distance_km"])


@st.cache_data(ttl=86400)
def sources_get_kilns(lat: float, lon: float, radius_km: float) -> list[dict]:
    """
    Fetch brick kilns within radius_km from OpenStreetMap.

    Called by: LLM (for brick kiln contribution queries) and
    render functions (source attribution).

    Returns list of dicts with: name, lat, lon, distance_km
    Sorted by distance ascending.
    """
    bbox = _bbox(lat, lon, radius_km)
    query = f"""
    [out:json][timeout:20];
    (
      node["industrial"="kiln"]({bbox});
      way["industrial"="kiln"]({bbox});
      node["craft"="kiln"]({bbox});
    );
    out center tags;
    """
    elements = _overpass(query)
    results = []

    for el in elements:
        elat, elon = _el_coords(el)
        if elat is None:
            continue
        dist = _haversine(lat, lon, elat, elon)
        if dist <= radius_km:
            tags = el.get("tags", {})
            results.append({
                "name":        tags.get("name", "Brick Kiln"),
                "lat":         elat,
                "lon":         elon,
                "distance_km": round(dist, 2),
            })

    return sorted(results, key=lambda x: x["distance_km"])


@st.cache_data(ttl=86400)
def sources_get_power_plants(lat: float, lon: float, radius_km: float) -> list[dict]:
    """
    Fetch thermal power plants within radius_km.

    Called by: LLM (for power plant contribution and satellite queries)
    and render functions (power plant map, source attribution).

    Uses Global Power Plant Database CSV if available at aqgpt_core/data/
    global_power_plants.csv, otherwise falls back to OpenStreetMap.
    Download from: https://datasets.wri.org/dataset/globalpowerplantdatabase

    Returns list of dicts with:
        name, lat, lon, distance_km, capacity_mw, fuel
    Sorted by distance ascending.
    """
    from pathlib import Path
    csv_path = POWER_PLANTS_CSV

    # if csv_path.exists():
    #     return _power_plants_from_csv(lat, lon, radius_km, csv_path)
    return _power_plants_from_osm(lat, lon, radius_km)


def _power_plants_from_csv(lat, lon, radius_km, csv_path) -> list[dict]:
    """Load power plants from Global Power Plant Database CSV."""
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        thermal = {"Coal", "Gas", "Oil", "Petcoke"}
        df = df[df["primary_fuel"].isin(thermal)].copy()
        df["distance_km"] = df.apply(
            lambda r: _haversine(lat, lon, r["latitude"], r["longitude"]), axis=1
        )
        nearby = df[df["distance_km"] <= radius_km].sort_values("distance_km")
        return [
            {
                "name":        row["name"],
                "lat":         row["latitude"],
                "lon":         row["longitude"],
                "distance_km": round(row["distance_km"], 2),
                "capacity_mw": float(row.get("capacity_mw", 0) or 0),
                "fuel":        row["primary_fuel"],
            }
            for _, row in nearby.iterrows()
        ]
    except Exception as e:
        st.warning(f"Power plant CSV error: {e}")
        return _power_plants_from_osm(lat, lon, radius_km)


def _power_plants_from_osm(lat, lon, radius_km) -> list[dict]:
    """Fallback: fetch power plants from OpenStreetMap."""
    bbox = _bbox(lat, lon, radius_km)
    query = f"""
    [out:json][timeout:20];
    (
      node["power"="plant"]({bbox});
      way["power"="plant"]({bbox});
    );
    out center tags;
    """
    elements = _overpass(query)
    results = []

    for el in elements:
        elat, elon = _el_coords(el)
        if elat is None:
            continue
        dist = _haversine(lat, lon, elat, elon)
        if dist <= radius_km:
            tags = el.get("tags", {})
            cap_str = tags.get("plant:output:electricity", "0").replace("MW", "").strip()
            try:
                cap = float(cap_str)
            except ValueError:
                cap = 0.0
            results.append({
                "name":        tags.get("name", "Power Plant"),
                "lat":         elat,
                "lon":         elon,
                "distance_km": round(dist, 2),
                "capacity_mw": cap,
                "fuel":        tags.get("plant:source", "unknown"),
            })

    return sorted(results, key=lambda x: x["distance_km"])


@st.cache_data(ttl=86400)
def sources_get_road_density(lat: float, lon: float, radius_km: float) -> dict:
    """
    Count major roads within radius_km from OpenStreetMap.

    Called by: attribution functions only — not intended for direct LLM use.
    Used as a proxy for traffic emission intensity.

    Returns dict with:
        n_roads (int), density ('low'/'medium'/'high')
    """
    bbox = _bbox(lat, lon, radius_km)
    query = f"""
    [out:json][timeout:15];
    way["highway"~"^(motorway|trunk|primary|secondary)$"]({bbox});
    out count;
    """
    try:
        r = requests.post(OVERPASS_URL, data={"data": query}, timeout=15)
        r.raise_for_status()
        total = int(r.json().get("elements", [{}])[0].get("tags", {}).get("total", 5))
    except Exception:
        total = 5

    density = "high" if total > 15 else "medium" if total > 5 else "low"
    return {"n_roads": total, "density": density}


@st.cache_data(ttl=86400)
def sources_get_all(lat: float, lon: float, radius_km: float) -> dict:
    """
    Fetch all pollution source categories for a location.

    Called by: render functions that need everything (source attribution,
    intervention analysis, wind transport map).
    Not intended for direct LLM use — use individual functions instead.

    Returns dict with:
        industries (list), power_plants (list), kilns (list),
        road_density (dict), summary (counts)
    """
    industries   = sources_get_industries(lat, lon, radius_km)
    kilns        = sources_get_kilns(lat, lon, radius_km)
    power_plants = sources_get_power_plants(lat, lon, max(radius_km, 100.0))
    road_density = sources_get_road_density(lat, lon, radius_km)

    return {
        "industries":    industries,
        "power_plants":  power_plants,
        "kilns":         kilns,
        "road_density":  road_density,
        "summary": {
            "n_industries":   len(industries),
            "n_power_plants": len(power_plants),
            "n_kilns":        len(kilns),
            "n_roads":        road_density["n_roads"],
            "road_density":   road_density["density"],
        },
    }