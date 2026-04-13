"""AQGPT_CORE: API Keys/Tokens, Constants, Base URLs, and model settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

WAQI_TOKEN = os.getenv("WAQI_TOKEN", "demo")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
NASA_FIRMS_KEY = os.getenv("NASA_FIRMS_KEY", "")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", os.getenv("QWEN_API_BASE", "http://localhost:11434"))
VLLM_API_BASE = os.getenv("VLLM_API_BASE", "http://localhost:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
VLLM_TIMEOUT_SECONDS = int(os.getenv("VLLM_TIMEOUT_SECONDS", "180"))

WAQI_BASE = "https://api.waqi.info"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
NASA_FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
# OVERPASS_URL = "https://overpass.openstreetmap.ru/api/interpreter"

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
    "📚 UrbanEmissions Knowledge": [
        "What does urbanemissions say about PM2.5 sources in India?",
        "Find urbanemissions articles on crop burning and air quality",
    ],
}

VIZ_TYPES = {
    "What is the air quality right now?":       "conditions",
    "Which stations are worst right now?":      "spatial",
    "Better or worse than yesterday?":          "conditions",
    "Show TROPOMI NO2 and fire hotspots":       "satellite",
    "Are there active stubble fires?":          "satellite",
    "Ground vs satellite - what do they show?": "satellite",
    "What are the main pollution sources?":     "attribution",
    "Traffic vs industry vs dust breakdown":    "attribution",
    "Show thermal power plants on satellite":   "power_plants",
    "Coal plants within 100km?":               "power_plants",
    "Where is pollution coming from?":          "wind",
    "Show wind patterns and transport":         "wind",
    "Why is PM2.5 high right now?":            "why_bad",
    "Is there atmospheric stagnation?":         "why_bad",
    "Is outdoor exercise safe?":               "health",
    "What mask grade is needed?":              "health",
    "30% traffic cut - what's the impact?":    "intervention",
    "Best intervention for fastest relief?":   "intervention",
    "Air quality over past week":              "trends",
    "Diurnal pattern - worst hours?":          "trends",
    "What does urbanemissions say about PM2.5 sources in India?": "rag",
    "Find urbanemissions articles on crop burning and air quality": "rag",
}

# ─── Shared Model Configuration ────────────────────────────────────────────────

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

# Backward-compatible legacy env vars kept as fallbacks.
LLM_TEXT_PROVIDER = os.getenv("LLM_TEXT_PROVIDER", "qwen")
LLM_FUNCTION_CALLER = os.getenv("LLM_FUNCTION_CALLER", "qwen")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
GEMINI_FUNCTION_CALLER_MODEL = os.getenv("GEMINI_FUNCTION_CALLER_MODEL", "gemini-2.5-flash")

# These defaults work for both Ollama and vLLM-backed Qwen setups.
QWEN_TEXT_MODEL = os.getenv("QWEN_TEXT_MODEL", "Qwen/Qwen2.5-72B-Instruct")
QWEN_FUNCTION_CALLER_MODEL = os.getenv("QWEN_FUNCTION_CALLER_MODEL", "Qwen/Qwen2.5-72B-Instruct")


@dataclass(frozen=True)
class ModelSelection:
    role: str
    provider: str
    model: str


def _first_env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def _default_model_for_provider(provider: str, *, text: bool) -> str:
    provider = _normalize_provider(provider)
    if provider == "gemini":
        return GEMINI_TEXT_MODEL if text else GEMINI_FUNCTION_CALLER_MODEL
    if provider in {"qwen", "vllm"}:
        return QWEN_TEXT_MODEL if text else QWEN_FUNCTION_CALLER_MODEL
    if provider == "ollama":
        return os.getenv("RAG_GENERATION_MODEL", "mistral")
    return _first_env("AQGPT_MODEL", default="qwen3-235b-a22b")


def _normalize_provider(provider: str) -> str:
    normalized = provider.lower()
    aliases = {
        "qwen_api": "qwen",
        "ollama": "qwen",
    }
    return aliases.get(normalized, normalized)


def resolve_model_selection(role: str) -> ModelSelection:
    """Resolve provider/model for a role using one shared precedence layer.

    Precedence order:
    1. Role-specific unified vars (AQGPT_<ROLE>_PROVIDER / AQGPT_<ROLE>_MODEL)
    2. Global unified vars (AQGPT_MODEL_PROVIDER / AQGPT_MODEL)
    3. Legacy per-feature vars already used by the app
    4. Provider defaults
    """
    normalized_role = role.lower()
    if normalized_role in {"summary", "insights", "text"}:
        provider = _first_env(
            "AQGPT_TEXT_PROVIDER",
            "AQGPT_MODEL_PROVIDER",
            default=LLM_TEXT_PROVIDER,
        )
        provider = _normalize_provider(provider)
        model = _first_env(
            "AQGPT_TEXT_MODEL",
            "AQGPT_MODEL",
            default=_default_model_for_provider(provider, text=True),
        )
        return ModelSelection(role=normalized_role, provider=provider, model=model)

    if normalized_role in {"function_calling", "function_caller", "tools"}:
        provider = _first_env(
            "AQGPT_FUNCTION_PROVIDER",
            "AQGPT_MODEL_PROVIDER",
            default=LLM_FUNCTION_CALLER,
        )
        provider = _normalize_provider(provider)
        model = _first_env(
            "AQGPT_FUNCTION_MODEL",
            "AQGPT_MODEL",
            default=_default_model_for_provider(provider, text=False),
        )
        return ModelSelection(role=normalized_role, provider=provider, model=model)

    if normalized_role in {"rag", "rag_generation"}:
        provider = _first_env(
            "AQGPT_RAG_PROVIDER",
            "AQGPT_MODEL_PROVIDER",
            default=os.getenv("RAG_GENERATION_PROVIDER", "qwen"),
        )
        provider = _normalize_provider(provider)
        model = _first_env(
            "AQGPT_RAG_MODEL",
            "AQGPT_MODEL",
            default=_default_model_for_provider(provider, text=True),
        )
        return ModelSelection(role=normalized_role, provider=provider, model=model)

    raise ValueError(f"Unknown model role: {role!r}")


TEXT_MODEL_CONFIG = resolve_model_selection("text")
FUNCTION_MODEL_CONFIG = resolve_model_selection("function_calling")
RAG_GENERATION_CONFIG = resolve_model_selection("rag_generation")

# RAG retrieval/index settings live in the same central config layer.
RAG_CHROMA_DIR = Path(os.getenv("RAG_CHROMA_DIR", str(WORKSPACE_ROOT / "chroma_db")))
RAG_COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "urbanemissions")
RAG_EMBED_MODEL = _first_env("AQGPT_RAG_EMBED_MODEL", default="all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "6"))
RAG_OLLAMA_BASE_URL = os.getenv("RAG_OLLAMA_BASE_URL", OLLAMA_BASE_URL)
RAG_VLLM_BASE_URL = os.getenv("RAG_VLLM_BASE_URL", VLLM_API_BASE)
MAX_CONTEXT_CHUNKS = int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "8"))
MAX_CHUNKS_PER_URL = int(os.getenv("RAG_MAX_CHUNKS_PER_URL", "2"))
