"""Gemini LLM implementations for text generation and function calling."""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import google.generativeai as genai
import streamlit as st

from aqgpt_core.config import GEMINI_API_KEY, DEFAULT_LAT, DEFAULT_LON, DEFAULT_RADIUS_KM
from aqgpt_core.llm.base import TextGenerator, FunctionCaller
from aqgpt_core.llm.prompts import (
    QUERY_UNDERSTANDING_PROMPT,
    HEALTH_ADVISORY_PROMPT,
    WHY_BAD_PROMPT,
    INTERVENTION_PROMPT,
    ANSWER_SUMMARY_PROMPT,
    CONDITIONS_SUMMARY_PROMPT,
    SATELLITE_INTERPRETATION_PROMPT,
    ATTRIBUTION_CONTEXT_PROMPT,
    INTERVENTION_CUSTOM_PROMPT,
)
from aqgpt_core.llm.tool_registry import AVAILABLE_TOOLS, invoke_tool

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)


class GeminiTextGenerator(TextGenerator):
    """Gemini-based text generation for queries, answers, and insights."""

    def __init__(self, model: str = "gemini-2.5-flash"):
        self.model = genai.GenerativeModel(model)
        self.model_name = model

    def understand_query(self, query: str, context: dict) -> dict:
        """Map user query to visualization types and extract parameters."""
        routing_guidance = context.get("routing_guidance", "")
        routing_examples = context.get("routing_examples", [])
        examples_block = "\n".join([f"- {e}" for e in routing_examples]) if routing_examples else ""

        prompt = f"""{QUERY_UNDERSTANDING_PROMPT}

Current location: Lat {context.get('current_location', (DEFAULT_LAT, DEFAULT_LON))[0]},
                  Lon {context.get('current_location', (DEFAULT_LAT, DEFAULT_LON))[1]}

Available visualization types: {', '.join(context.get('available_viz_types', []))}
Available pollutants: {', '.join(context.get('available_pollutants', []))}

    Routing guidance: {routing_guidance}
    Routing examples:
    {examples_block}

User query: "{query}"

IMPORTANT: If the user asks about a specific location (city/region name) that you cannot identify or don't recognize, return an error instead of using the default location. Set "location_not_found" to true and include a message explaining which location you couldn't find.

Return ONLY valid JSON, no markdown or other text."""

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Remove markdown code fences if present
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            result = json.loads(text)

            # Check if Gemini couldn't find a requested location
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
                    "answer_summary": result.get("error_message", "I couldn't identify the location you mentioned."),
                }

            # Ensure viz_types is a list
            viz_types = result.get("viz_types", [result.get("viz_type", "conditions")])
            if isinstance(viz_types, str):
                viz_types = [viz_types]

            return {
                "understood_intent": result.get("understood_intent", query),
                "viz_types": viz_types,
                "pollutant": result.get("pollutant"),
                "lat": result.get("lat") or context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[0],
                "lon": result.get("lon") or context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[1],
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
                "lat": context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[0],
                "lon": context.get("current_location", (DEFAULT_LAT, DEFAULT_LON))[1],
                "time_window": "last_24h",
                "radius_km": DEFAULT_RADIUS_KM,
                "confidence": 0.3,
                "answer_summary": f"Analyzing your query about air quality...",
                "error": False,
            }

    def generate_answer(self, data_context, viz_types=None, user_query: str = "") -> str:
        """Generate natural language summary of visualization data.

        Handles both single viz_type (legacy) and multiple viz_types (new multi-query).
        """
        # Handle legacy single viz_type case
        if isinstance(viz_types, str):
            viz_types = [viz_types]

        # If data_context is not a dict of dicts, assume it's a single context
        if viz_types and not isinstance(data_context, dict):
            data_context = {"single": data_context}

        viz_types_str = ", ".join(viz_types) if viz_types else "general analysis"

        prompt = f"""You are an air quality analyst. The user asked: "{user_query}"

Based on the following data analysis for {viz_types_str}, provide a direct, specific answer to their question.

Data context: {json.dumps(data_context, default=str)}

Provide a thorough answer that directly addresses the user's question.
Match the length and depth the user requested if specified.
Be conversational, insightful, and action-oriented when relevant.
If multiple analysis types are provided, synthesize them into a cohesive answer."""

        try:
            response = self.model.generate_content(prompt)
            return response.text if response else "Data analysis completed."
        except Exception as e:
            return f"Analysis completed for: {viz_types_str}"

    def generate_health_advisory(self, pollutant_level: float, pollutant: str, location: tuple) -> dict:
        """Generate health recommendations."""
        prompt = HEALTH_ADVISORY_PROMPT.format(
            pollutant_level=pollutant_level,
            pollutant=pollutant,
            location=f"Lat {location[0]}, Lon {location[1]}"
        )
        prompt += "\nReturn ONLY valid JSON."

        try:
            response = self.model.generate_content(prompt, safety_settings=[])
            text = response.text.strip()

            # Remove markdown code fences if present
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            result = json.loads(text)
            return result
        except Exception as e:
            # Fallback to hardcoded if error
            return self._fallback_health_advisory(pollutant_level, pollutant)

    def analyze_why_bad(self, met_data: dict, pollutant_level: float, location: tuple) -> list:
        """Analyze why pollution is high."""
        prompt = WHY_BAD_PROMPT.format(
            met_data=json.dumps(met_data, default=str),
            pollutant_level=pollutant_level,
            location=f"Lat {location[0]}, Lon {location[1]}"
        )
        prompt += "\nReturn ONLY valid JSON with 'factors' array."

        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()

            # Remove markdown code fences if present
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            result = json.loads(text)
            return result.get("factors", [])
        except json.JSONDecodeError as e:
            # Fallback instead of warning - this is expected behavior now
            return []
        except Exception as e:
            return []

    @staticmethod
    def _fallback_health_advisory(level: float, pollutant: str) -> dict:
        """Fallback hardcoded health advisory."""
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
                "activities": []
            }
        elif level <= 120:
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
                "activities": []
            }
        else:
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
                "activities": []
            }

    def generate_conditions_insight(self, pollutant_level: float, category: str, wind_speed: float, wind_label: str, blh_mean: float, stagnation_risk: str, trend: str) -> str:
        """Generate insight for conditions visualization."""
        prompt = CONDITIONS_SUMMARY_PROMPT.format(
            pollutant_level=pollutant_level,
            category=category,
            wind_speed=wind_speed,
            wind_label=wind_label,
            blh_mean=blh_mean,
            stagnation_risk=stagnation_risk,
            trend=trend
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text if response else ""
        except Exception as e:
            return ""

    def interpret_satellite_data(self, no2_hotspots: int, mean_no2: float, aod: float, fires: int, crop_fires: int, location: tuple) -> str:
        """Interpret satellite observations."""
        prompt = SATELLITE_INTERPRETATION_PROMPT.format(
            no2_hotspots=no2_hotspots,
            mean_no2=mean_no2,
            aod=aod,
            fires=fires,
            crop_fires=crop_fires,
            location=f"Lat {location[0]}, Lon {location[1]}"
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text if response else ""
        except Exception as e:
            return ""

    def analyze_attribution(self, traffic: float, industry: float, power_plants: float, dust: float, other: float, location: tuple, conditions: dict) -> str:
        """Analyze source attribution patterns."""
        prompt = ATTRIBUTION_CONTEXT_PROMPT.format(
            traffic=f"{traffic*100:.0f}",
            industry=f"{industry*100:.0f}",
            power_plants=f"{power_plants*100:.0f}",
            dust=f"{dust*100:.0f}",
            other=f"{other*100:.0f}",
            location=f"Lat {location[0]}, Lon {location[1]}",
            conditions=json.dumps(conditions, default=str)
        )
        try:
            response = self.model.generate_content(prompt)
            return response.text if response else ""
        except Exception as e:
            return ""

    def generate_custom_interventions(self, current_aqi: float, top_sources: list, location: tuple, dominant_source: str) -> dict:
        """Generate location-specific intervention scenarios."""
        prompt = INTERVENTION_CUSTOM_PROMPT.format(
            current_aqi=current_aqi,
            top_sources=", ".join(top_sources),
            location=f"Lat {location[0]}, Lon {location[1]}",
            dominant_source=dominant_source
        )
        prompt += "\nReturn ONLY valid JSON."
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            return result.get("interventions", [])
        except Exception as e:
            return []


class GeminiFunctionCaller(FunctionCaller):

    def call_with_tools(
        self,
        query: str,
        tools: Optional[list] = None,
        tool_executor: Optional[callable] = None,
        max_turns: int = 5,
    ) -> dict:
        """Execute query with function calling (tool_use).

        Uses Gemini's native function calling capability to automatically
        decide which tools to invoke and handle multi-turn interactions.
        """
        if tool_executor is None:
            tool_executor = invoke_tool

        # Start conversation
        messages = [{"role": "user", "content": query}]
        tools_called = []
        all_results = {}

        for turn in range(max_turns):
            # Get response from Gemini
            response = self.client.generate_content(messages)

            # Check if model wants to call tools
            tool_calls = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        tool_calls.append(part.function_call)

            if not tool_calls:
                # No more tool calls, return final response
                final_text = ""
                if response.candidates and response.candidates[0].content.parts:
                    final_text = response.candidates[0].content.parts[0].text

                return {
                    "response": final_text,
                    "tools_called": tools_called,
                    "data_fetched": all_results,
                    "success": True,
                }

            # Execute tool calls
            tool_results = []
            for tool_call in tool_calls:
                tool_name = tool_call.name
                args = {k: v for k, v in tool_call.args.items()}

                # Execute tool
                result = tool_executor(tool_name, **args)
                tools_called.append(tool_name)
                all_results[tool_name] = result

                # Add to conversation
                tool_results.append({
                    "role": "function",
                    "name": tool_name,
                    "content": json.dumps(result, default=str),
                })

            # Add assistant response with tool calls to messages
            messages.append({"role": "assistant", "content": response.candidates[0].content})

            # Add tool results to messages
            messages.extend(tool_results)

        return {
            "response": "Max tool call turns reached",
            "tools_called": tools_called,
            "data_fetched": all_results,
            "success": False,
        }
