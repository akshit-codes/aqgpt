"""AQGPT_CORE: API Keys/Tokens, Constants, Base URLs, and Default Values etc."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "demo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NASA_FIRMS_KEY = os.getenv("NASA_FIRMS_KEY", "")

WAQI_BASE = "https://api.waqi.info"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NASA_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

DATA_DIR = Path(__file__).parent / "data"
POWER_PLANTS_CSV = DATA_DIR / "global_power_plants.csv"

DEFAULT_LAT = 23.21364140749712 
DEFAULT_LON = 72.68716264082622
DEFAULT_RADIUS_KM = 20

AQI_BREAKPOINTS = {
    "PM2.5": [30, 60, 90, 120, 250],
    "PM10":  [50, 100, 250, 350, 430],
    "NO2":   [40, 80, 180, 280, 400],
    "SO2":   [40, 80, 380, 800, 1600],
    "O3":    [50, 100, 168, 208, 748],
    "CO":    [1000, 2000, 10000, 17000, 34000],
}

CATEGORIES = {
    "📊 Current Conditions": [
        "What is the air quality right now?",
        "Which stations are worst right now?",
        "Better or worse than yesterday?",
    ],
    "🛰️ Satellite Data": [
        "Show TROPOMI NO2 and fire hotspots",
        "Are there active stubble fires?",
        "Ground vs satellite - what do they show?",
    ],
    "🏭 Sources & Attribution": [
        "What are the main pollution sources?",
        "Traffic vs industry vs dust breakdown",
    ],
    "⚡ Power Plants": [
        "Show thermal power plants on satellite",
        "Coal plants within 100km?",
    ],
    "🌬️ Wind & Transport": [
        "Where is pollution coming from?",
        "Show wind patterns and transport",
    ],
    "❓ Why Is It Bad?": [
        "Why is PM2.5 high right now?",
        "Is there atmospheric stagnation?",
    ],
    "🏥 Health & Safety": [
        "Is outdoor exercise safe?",
        "What mask grade is needed?",
    ],
    "🔧 Interventions": [
        "30% traffic cut - what's the impact?",
        "Best intervention for fastest relief?",
    ],
    "📈 Trends": [
        "Air quality over past week",
        "Diurnal pattern - worst hours?",
    ],
}