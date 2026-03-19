"""Source attribution — heuristic model estimating pollution source contributions.

Pure math, no external API calls required.
This is a rule-based model, not a chemical transport model (CTM).
Uncertainty ranges reflect this — treat outputs as estimates, not measurements.

For a production system, replace with PMF receptor modeling or HYSPLIT trajectories.
"""


def attribution_rank_sources(
    sources_summary: dict,
    conditions: dict,
    pollutant: str = "PM2.5",
    hour: int = 12,
    is_winter: bool = True,
) -> dict:
    """
    Estimate fractional contributions of each pollution source category.

    Called by: LLM (for 'what are the main sources' and 'traffic vs industry'
    queries) and render functions (attribution chart, intervention analysis).

    Parameters:
        sources_summary: output of sources_get_all()['summary']
        conditions: output of met_get_conditions()
        pollutant: pollutant name (currently only PM2.5 modelled)
        hour: hour of day 0-23 for time-of-day adjustments
        is_winter: True for Oct-Mar (affects biomass and dust weights)

    Returns dict of source -> {contribution, range_low, range_high}
    where contribution is a fraction 0-1, all contributions sum to 1.
    Returns empty dict if conditions are insufficient.
    """
    n_industries  = sources_summary.get("n_industries", 5)
    n_power_plants = sources_summary.get("n_power_plants", 2)
    n_roads        = sources_summary.get("n_roads", 10)
    n_kilns        = sources_summary.get("n_kilns", 0)
    wind_speed     = conditions.get("wind_speed", 3.0)
    calm_hours     = conditions.get("wind_calm_hours", 0)

    # ── Base weights ──────────────────────────────────────────────────────────
    weights = {
        "traffic":            0.30,
        "industry":           0.25,
        "road_dust":          0.15,
        "residential_biomass": 0.20,
        "brick_kilns":        0.05,
        "power_plants":       0.05,
    }

    # ── Time of day ───────────────────────────────────────────────────────────
    if 7 <= hour <= 10:
        weights["traffic"]             *= 1.5
        weights["residential_biomass"] *= 1.2
    elif 17 <= hour <= 21:
        weights["traffic"]             *= 1.4
        weights["residential_biomass"] *= 1.4
    elif hour <= 5 or hour >= 22:
        weights["traffic"]             *= 0.4
        weights["residential_biomass"] *= 1.6

    # ── Season ────────────────────────────────────────────────────────────────
    if is_winter:
        weights["residential_biomass"] *= 1.7
        weights["road_dust"]           *= 0.7
    else:
        weights["road_dust"]           *= 1.5
        weights["residential_biomass"] *= 0.5

    # ── Source density ────────────────────────────────────────────────────────
    if n_industries > 10:
        weights["industry"] *= 1.3
    elif n_industries < 3:
        weights["industry"] *= 0.7

    if n_roads > 15:
        weights["traffic"] *= 1.2
    elif n_roads < 5:
        weights["traffic"] *= 0.8

    if n_kilns > 0:
        weights["brick_kilns"] = min(0.15, n_kilns * 0.025)

    if n_power_plants > 0:
        weights["power_plants"] = min(0.12, n_power_plants * 0.04)

    # ── Wind ─────────────────────────────────────────────────────────────────
    if wind_speed < 1.5:
        for k in weights:
            weights[k] *= 1.1
    elif wind_speed > 6:
        weights["traffic"]   *= 0.8
        weights["road_dust"] *= 1.4

    if calm_hours > 10:
        weights["residential_biomass"] *= 1.2

    # ── Normalize ─────────────────────────────────────────────────────────────
    total = sum(weights.values())
    if total == 0:
        return {}

    uncertainty = 0.08
    result = {}
    for source, w in weights.items():
        contrib = w / total
        result[source] = {
            "contribution": round(contrib, 3),
            "range_low":    round(max(0.0, contrib - uncertainty), 3),
            "range_high":   round(min(1.0, contrib + uncertainty), 3),
        }

    return result


def attribution_explain(
    attribution: dict,
    conditions: dict,
    mean_level: float,
) -> str:
    """
    Generate a human readable explanation of attribution results.

    Called by: LLM (as additional context when explaining source contributions).

    Parameters:
        attribution: output of attribution_rank_sources()
        conditions: output of met_get_conditions()
        mean_level: mean pollutant level in µg/m³

    Returns a plain text explanation string.
    """
    if not attribution:
        return "Insufficient data for source attribution."

    ranked = sorted(attribution.items(), key=lambda x: -x[1]["contribution"])
    top_source, top_info = ranked[0]

    lines = [
        f"Top contributor: {top_source.replace('_', ' ').title()} "
        f"({top_info['contribution']*100:.0f}%, "
        f"range {top_info['range_low']*100:.0f}-{top_info['range_high']*100:.0f}%)",
        "",
        "All sources ranked:",
    ]

    for name, info in ranked:
        contrib_ug = info["contribution"] * mean_level
        lines.append(
            f"  {name.replace('_', ' ').title()}: "
            f"{info['contribution']*100:.0f}% "
            f"(~{contrib_ug:.0f} µg/m³, "
            f"range {info['range_low']*100:.0f}-{info['range_high']*100:.0f}%)"
        )

    lines += [
        "",
        f"Wind: {conditions.get('wind_label', '?')} "
        f"at {conditions.get('wind_speed', 0):.1f} m/s",
        f"Stagnation risk: {conditions.get('stagnation_risk', 'unknown')}",
        "",
        "Note: heuristic estimates only, not chemical transport model outputs.",
    ]

    return "\n".join(lines)