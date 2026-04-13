"""Qwen via native Ollama for text generation and tool calling."""

from __future__ import annotations

import importlib
import json
from typing import Optional

import streamlit as st

from aqgpt_core.config import (
    DEFAULT_LAT,
    DEFAULT_LON,
    DEFAULT_RADIUS_KM,
    OLLAMA_BASE_URL,
)
from aqgpt_core.llm.base import FunctionCaller, TextGenerator
from aqgpt_core.llm.prompts import (
    ATTRIBUTION_CONTEXT_PROMPT,
    CONDITIONS_SUMMARY_PROMPT,
    HEALTH_ADVISORY_PROMPT,
    INTERVENTION_CUSTOM_PROMPT,
    QUERY_UNDERSTANDING_PROMPT,
    SATELLITE_INTERPRETATION_PROMPT,
    WHY_BAD_PROMPT,
)
from aqgpt_core.llm.tool_registry import AVAILABLE_TOOLS, invoke_tool


def _make_client():
    try:
        ollama = importlib.import_module("ollama")
    except ImportError as exc:
        raise ImportError("ollama package is required for the Qwen/Ollama backend") from exc
    return ollama.Client(host=OLLAMA_BASE_URL)


def _strip_json_wrappers(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _chat(client, model: str, messages: list[dict], **options) -> str:
    response = client.chat(model=model, messages=messages, options=options or None)
    return (response.get("message", {}) or {}).get("content", "") or ""


def _tools_to_ollama_format(tools: list) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in tools
    ]


class QwenTextGenerator(TextGenerator):
    """Qwen-based text generation backed by Ollama."""

    def __init__(self, model: str, provider: str = "qwen"):
        self.client = _make_client()
        self.model = model
        self.provider = provider

    def understand_query(self, query: str, context: dict) -> dict:
        lat = context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[0]
        lon = context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[1]
        routing_guidance = context.get("routing_guidance", "")
        routing_examples = context.get("routing_examples", [])
        examples_block = "\n".join([f"- {e}" for e in routing_examples]) if routing_examples else ""

        prompt = (
            f"{QUERY_UNDERSTANDING_PROMPT}\n\n"
            f"Current location: Lat {lat}, Lon {lon}\n"
            f"Available visualization types: {', '.join(context.get('available_viz_types', []))}\n"
            f"Available pollutants: {', '.join(context.get('available_pollutants', []))}\n\n"
            f"Routing guidance: {routing_guidance}\n"
            f"Routing examples:\n{examples_block}\n\n"
            f'User query: "{query}"\n\n'
            "Return ONLY valid JSON, no markdown or other text."
        )

        try:
            text = _chat(self.client, self.model, [{"role": "user", "content": prompt}])
            result = json.loads(_strip_json_wrappers(text))

            if result.get("location_not_found", False):
                return {
                    "understood_intent": result.get("understood_intent", query),
                    "viz_types": [],
                    "pollutant": None,
                    "lat": None,
                    "lon": None,
                    "error": True,
                    "error_message": result.get("error_message", "Location not found"),
                    "confidence": 0.0,
                    "answer_summary": result.get(
                        "error_message",
                        "I couldn't identify the location you mentioned.",
                    ),
                }

            viz_types = result.get("viz_types", [result.get("viz_type", "conditions")])
            if isinstance(viz_types, str):
                viz_types = [viz_types]

            return {
                "understood_intent": result.get("understood_intent", query),
                "viz_types": viz_types,
                "pollutant": result.get("pollutant"),
                "lat": result.get("lat") or lat,
                "lon": result.get("lon") or lon,
                "time_window": result.get("time_window", "last_24h"),
                "radius_km": result.get("radius_km") or DEFAULT_RADIUS_KM,
                "confidence": result.get("confidence", 0.7),
                "answer_summary": result.get("answer_summary", f"Analyzing: {query}"),
                "error": False,
            }
        except (json.JSONDecodeError, AttributeError) as e:
            st.warning(f"Query understanding error: {e}")
            return {
                "understood_intent": query,
                "viz_types": ["conditions"],
                "pollutant": None,
                "lat": lat,
                "lon": lon,
                "time_window": "last_24h",
                "radius_km": DEFAULT_RADIUS_KM,
                "confidence": 0.3,
                "answer_summary": "Analyzing your query about air quality...",
                "error": False,
            }

    def generate_answer(self, data_context, viz_types=None, user_query: str = "") -> str:
        if isinstance(viz_types, str):
            viz_types = [viz_types]
        if viz_types and not isinstance(data_context, dict):
            data_context = {"single": data_context}

        viz_types_str = ", ".join(viz_types) if viz_types else "general analysis"
        prompt = (
            f'You are an air quality analyst. The user asked: "{user_query}"\n\n'
            f"Based on the following data analysis for {viz_types_str}, "
            "provide a direct, specific answer to their question.\n\n"
            f"Data context: {json.dumps(data_context, default=str)}\n\n"
            "Provide a thorough answer that directly addresses the user's question. "
            "Match the length and depth the user requested if specified, with specific "
            "numbers/findings from the data. Be conversational, insightful, and "
            "action-oriented when relevant. If multiple analysis types are provided, "
            "synthesize them into a cohesive answer."
        )

        try:
            return _chat(self.client, self.model, [{"role": "user", "content": prompt}]) or "Data analysis completed."
        except Exception:
            return f"Analysis completed for: {viz_types_str}"

    def generate_health_advisory(self, pollutant_level: float, pollutant: str, location: tuple) -> dict:
        prompt = (
            HEALTH_ADVISORY_PROMPT.format(
                pollutant_level=pollutant_level,
                pollutant=pollutant,
                location=f"Lat {location[0]}, Lon {location[1]}",
            )
            + "\nReturn ONLY valid JSON."
        )
        try:
            text = _chat(self.client, self.model, [{"role": "user", "content": prompt}])
            return json.loads(_strip_json_wrappers(text))
        except Exception as e:
            st.warning(f"Health advisory generation error: {e}")
            return self._fallback_health_advisory(pollutant_level, pollutant)

    def analyze_why_bad(self, met_data: dict, pollutant_level: float, location: tuple) -> list:
        prompt = (
            WHY_BAD_PROMPT.format(
                met_data=json.dumps(met_data, default=str),
                pollutant_level=pollutant_level,
                location=f"Lat {location[0]}, Lon {location[1]}",
            )
            + "\nReturn ONLY valid JSON."
        )
        try:
            text = _chat(self.client, self.model, [{"role": "user", "content": prompt}])
            result = json.loads(_strip_json_wrappers(text))
            return result.get("factors", [])
        except Exception as e:
            st.warning(f"Why bad analysis error: {e}")
            return []

    def generate_conditions_insight(
        self,
        pollutant_level: float,
        category: str,
        wind_speed: float,
        wind_label: str,
        blh_mean: float,
        stagnation_risk: str,
        trend: str,
    ) -> str:
        prompt = CONDITIONS_SUMMARY_PROMPT.format(
            pollutant_level=pollutant_level,
            category=category,
            wind_speed=wind_speed,
            wind_label=wind_label,
            blh_mean=blh_mean,
            stagnation_risk=stagnation_risk,
            trend=trend,
        )
        try:
            return _chat(self.client, self.model, [{"role": "user", "content": prompt}])
        except Exception:
            return ""

    def interpret_satellite_data(
        self,
        no2_hotspots: int,
        mean_no2: float,
        aod: float,
        fires: int,
        crop_fires: int,
        location: tuple,
    ) -> str:
        prompt = SATELLITE_INTERPRETATION_PROMPT.format(
            no2_hotspots=no2_hotspots,
            mean_no2=mean_no2,
            aod=aod,
            fires=fires,
            crop_fires=crop_fires,
            location=f"Lat {location[0]}, Lon {location[1]}",
        )
        try:
            return _chat(self.client, self.model, [{"role": "user", "content": prompt}])
        except Exception:
            return ""

    def analyze_attribution(
        self,
        traffic: float,
        industry: float,
        power_plants: float,
        dust: float,
        other: float,
        location: tuple,
        conditions: dict,
    ) -> str:
        prompt = ATTRIBUTION_CONTEXT_PROMPT.format(
            traffic=f"{traffic * 100:.0f}",
            industry=f"{industry * 100:.0f}",
            power_plants=f"{power_plants * 100:.0f}",
            dust=f"{dust * 100:.0f}",
            other=f"{other * 100:.0f}",
            location=f"Lat {location[0]}, Lon {location[1]}",
            conditions=json.dumps(conditions, default=str),
        )
        try:
            return _chat(self.client, self.model, [{"role": "user", "content": prompt}])
        except Exception:
            return ""

    def generate_custom_interventions(
        self,
        current_aqi: float,
        top_sources: list,
        location: tuple,
        dominant_source: str,
    ) -> dict:
        prompt = (
            INTERVENTION_CUSTOM_PROMPT.format(
                current_aqi=current_aqi,
                top_sources=", ".join(top_sources),
                location=f"Lat {location[0]}, Lon {location[1]}",
                dominant_source=dominant_source,
            )
            + "\nReturn ONLY valid JSON."
        )
        try:
            text = _chat(self.client, self.model, [{"role": "user", "content": prompt}])
            result = json.loads(_strip_json_wrappers(text))
            return result.get("interventions", [])
        except Exception:
            return []

    @staticmethod
    def _fallback_health_advisory(level: float, pollutant: str) -> dict:
        if level <= 60:
            return {
                "health_class": "Good",
                "border_color": "#22c55e",
                "bg_color": "#dcfce7",
                "outdoor_ok": True,
                "sensitive_risk": "Low",
                "general_risk": "None",
                "exercise_rec": "Safe for all activities",
                "mask_rec": "No mask needed",
                "header": "Air quality is good",
                "activities": [],
            }
        if level <= 120:
            return {
                "health_class": "Moderate",
                "border_color": "#eab308",
                "bg_color": "#fef08a",
                "outdoor_ok": True,
                "sensitive_risk": "Moderate",
                "general_risk": "Low",
                "exercise_rec": "General population OK, sensitive groups should reduce",
                "mask_rec": "Mask recommended for sensitive groups",
                "header": "Air quality is moderate",
                "activities": [],
            }
        return {
            "health_class": "Poor",
            "border_color": "#f97316",
            "bg_color": "#fed7aa",
            "outdoor_ok": False,
            "sensitive_risk": "High",
            "general_risk": "Moderate",
            "exercise_rec": "Limit outdoor activities",
            "mask_rec": "N95 mask recommended",
            "header": "Air quality is poor",
            "activities": [],
        }


class QwenFunctionCaller(FunctionCaller):
    """Qwen function calling backed by Ollama tools."""

    def __init__(self, model: str, provider: str = "qwen"):
        self.client = _make_client()
        self.model = model
        self.provider = provider

    def call_with_tools(
        self,
        query: str,
        tools: Optional[list] = None,
        tool_executor: Optional[callable] = None,
        max_turns: int = 5,
    ) -> dict:
        if tool_executor is None:
            tool_executor = invoke_tool
        if tools is None:
            tools = AVAILABLE_TOOLS

        ollama_tools = _tools_to_ollama_format(tools)
        messages: list[dict] = [{"role": "user", "content": query}]
        tools_called: list[str] = []
        all_results: dict = {}

        for _ in range(max_turns):
            response = self.client.chat(model=self.model, messages=messages, tools=ollama_tools)
            message = (response.get("message", {}) or {})
            tool_calls = message.get("tool_calls") or []

            if not tool_calls:
                return {
                    "response": message.get("content", "") or "",
                    "tools_called": tools_called,
                    "data_fetched": all_results,
                    "success": True,
                }

            messages.append(message)

            for tool_call in tool_calls:
                fn = tool_call.get("function", {}) or {}
                tool_name = fn.get("name", "")
                args = fn.get("arguments", {}) or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}

                result = tool_executor(tool_name, **args)
                tools_called.append(tool_name)
                all_results[tool_name] = result

                messages.append(
                    {
                        "role": "tool",
                        "name": tool_name,
                        "content": json.dumps(result, default=str),
                    }
                )

        return {
            "response": "Max tool call turns reached",
            "tools_called": tools_called,
            "data_fetched": all_results,
            "success": False,
        }
