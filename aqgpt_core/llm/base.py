"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from typing import Any, Optional


class TextGenerator(ABC):
    """Abstract base for text generation (query understanding, answers, insights)."""

    @abstractmethod
    def understand_query(self, query: str, context: dict) -> dict:
        """Understand user query and extract intent.

        Args:
            query: User's natural language question
            context: Dict with current_location, available_viz_types, available_pollutants

        Returns:
            Dict with: viz_type, pollutant, lat, lon, time_window, radius_km, answer_summary
        """
        pass

    @abstractmethod
    def generate_answer(self, data_context, viz_types, user_query: str = "") -> str:
        """Generate natural language answer based on visualization data.

        Args:
            data_context: Dict with viz_type keys and data values, or single dict
            viz_types: List of viz_types being analyzed, or None
            user_query: Original user question (for context)

        Returns:
            Natural language summary (2-3 sentences)
        """
        pass

    @abstractmethod
    def generate_health_advisory(self, pollutant_level: float, pollutant: str, location: tuple) -> dict:
        """Generate health recommendations instead of hardcoded thresholds.

        Returns:
            Dict with category, activities, mask_rec, outdoor_ok, etc.
        """
        pass

    @abstractmethod
    def analyze_why_bad(self, met_data: dict, pollutant_level: float, location: tuple) -> list:
        """Analyze why pollution is high - replaces hardcoded factor detection.

        Returns:
            List of {factor, description, contribution_score} dicts
        """
        pass

    @abstractmethod
    def generate_conditions_insight(self, pollutant_level: float, category: str, wind_speed: float, wind_label: str, blh_mean: float, stagnation_risk: str, trend: str) -> str:
        """Generate insight for conditions visualization."""
        pass

    @abstractmethod
    def interpret_satellite_data(self, no2_hotspots: int, mean_no2: float, aod: float, fires: int, crop_fires: int, location: tuple) -> str:
        """Interpret satellite observations."""
        pass

    @abstractmethod
    def analyze_attribution(self, traffic: float, industry: float, power_plants: float, dust: float, other: float, location: tuple, conditions: dict) -> str:
        """Analyze source attribution patterns."""
        pass

    @abstractmethod
    def generate_custom_interventions(self, current_aqi: float, top_sources: list, location: tuple, dominant_source: str) -> dict:
        """Generate location-specific intervention scenarios."""
        pass

class FunctionCaller(ABC):
    """Abstract base for function calling (tool orchestration)."""

    @abstractmethod
    def call_with_tools(
        self,
        query: str,
        tools: list,
        tool_executor: callable,
        max_turns: int = 5
    ) -> dict:
        """Execute query with function calling (tool_use).

        Args:
            query: User query or internal request
            tools: List of available tool definitions
            tool_executor: Callable that executes tool calls
            max_turns: Max agentic loop turns

        Returns:
            Dict with: response, tools_called, data_fetched
        """
        pass
