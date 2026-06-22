"""
impact_prediction.py
====================
MODULE 9: Environmental Impact Prediction

Calculates environmental impact scores (0–100) for each waste category,
maps items to impact severity classes (Low / Medium / High / Critical),
and estimates CO₂ footprint and ecological damage potential.

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""


# ══════════════════════════════════════════════════════════════════════════════
# IMPACT CLASSIFICATION SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

# Mapping: waste class → impact severity
IMPACT_CLASS_MAP = {
    "Organic Waste": "Low",
    "Paper":         "Low",
    "Glass":         "Medium",
    "Metal":         "Medium",
    "Plastic":       "High",
    "E-Waste":       "Critical",
}

# Impact scores on a 0–100 scale (higher = worse for environment)
IMPACT_SCORE_MAP = {
    "Organic Waste": 15,
    "Paper":         25,
    "Glass":         45,
    "Metal":         55,
    "Plastic":       78,
    "E-Waste":       95,
}

# Severity class → numeric range
SEVERITY_RANGES = {
    "Low":      (0, 30),
    "Medium":   (31, 60),
    "High":     (61, 80),
    "Critical": (81, 100),
}

# Severity class → display color (hex)
SEVERITY_COLORS = {
    "Low":      "#2a9d8f",
    "Medium":   "#e9c46a",
    "High":     "#f4a261",
    "Critical": "#e63946",
}

# Detailed environmental metrics per category
ENVIRONMENTAL_METRICS = {
    "Organic Waste": {
        "decomposition_years": 0.08,          # ~1 month
        "co2_per_kg": 0.5,                    # kg CO₂ per kg waste
        "water_pollution_risk": "Very Low",
        "soil_contamination": "Negligible",
        "landfill_contribution": "Low (biodegradable)",
        "recycling_energy_saving_pct": 60,
        "description": (
            "Organic waste is biodegradable and can be composted to create "
            "nutrient-rich fertilizer. When diverted from landfills, it "
            "significantly reduces methane emissions."
        ),
    },
    "Paper": {
        "decomposition_years": 0.12,          # ~6 weeks
        "co2_per_kg": 1.1,
        "water_pollution_risk": "Low",
        "soil_contamination": "Negligible",
        "landfill_contribution": "Moderate (bulky)",
        "recycling_energy_saving_pct": 70,
        "description": (
            "Paper is highly recyclable and decomposes relatively quickly. "
            "Recycling paper saves 70% of the energy needed to produce "
            "new paper from virgin wood pulp."
        ),
    },
    "Glass": {
        "decomposition_years": 1_000_000,
        "co2_per_kg": 0.8,
        "water_pollution_risk": "Low",
        "soil_contamination": "Low (inert material)",
        "landfill_contribution": "High (does not decompose)",
        "recycling_energy_saving_pct": 30,
        "description": (
            "Glass is inert and non-toxic but takes an extremely long time "
            "to decompose in landfills. It is 100% recyclable with zero "
            "quality loss — infinitely."
        ),
    },
    "Metal": {
        "decomposition_years": 100,
        "co2_per_kg": 4.0,
        "water_pollution_risk": "Medium (corrosion runoff)",
        "soil_contamination": "Medium",
        "landfill_contribution": "High",
        "recycling_energy_saving_pct": 95,
        "description": (
            "Metal recycling is one of the most energy-efficient recycling "
            "processes. Recycling aluminum saves 95% of the energy needed "
            "to produce it from bauxite ore."
        ),
    },
    "Plastic": {
        "decomposition_years": 450,
        "co2_per_kg": 6.0,
        "water_pollution_risk": "High (microplastics)",
        "soil_contamination": "High",
        "landfill_contribution": "Very High",
        "recycling_energy_saving_pct": 80,
        "description": (
            "Plastic pollution is one of the most pressing environmental "
            "crises. Microplastics contaminate oceans, soil, and even the "
            "human food chain. Only ~9% of plastic is actually recycled globally."
        ),
    },
    "E-Waste": {
        "decomposition_years": float("inf"),   # never decomposes safely
        "co2_per_kg": 20.0,
        "water_pollution_risk": "Critical (heavy metals)",
        "soil_contamination": "Critical (lead, mercury, cadmium)",
        "landfill_contribution": "Critical (toxic leaching)",
        "recycling_energy_saving_pct": 85,
        "description": (
            "E-waste is the fastest growing waste stream globally. It contains "
            "toxic heavy metals (lead, mercury, cadmium) that leach into soil "
            "and groundwater. However, it also contains valuable precious metals "
            "(gold, silver, palladium) worth recovering."
        ),
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_impact_class(waste_class: str) -> str:
    """
    Return the impact severity class for a waste category.

    Parameters
    ----------
    waste_class : str
        One of: 'Plastic', 'Paper', 'Glass', 'Metal', 'Organic Waste', 'E-Waste'

    Returns
    -------
    str : 'Low', 'Medium', 'High', or 'Critical'
    """
    normalized = _normalize_class(waste_class)
    return IMPACT_CLASS_MAP.get(normalized, "Unknown")


def get_impact_score(waste_class: str) -> int:
    """
    Return the environmental impact score (0–100) for a waste category.

    Higher scores indicate greater environmental damage potential.
    """
    normalized = _normalize_class(waste_class)
    return IMPACT_SCORE_MAP.get(normalized, -1)


def get_impact_color(waste_class: str) -> str:
    """Return the hex display color for the impact severity level."""
    severity = get_impact_class(waste_class)
    return SEVERITY_COLORS.get(severity, "#888888")


def get_environmental_metrics(waste_class: str) -> dict:
    """
    Return detailed environmental metrics for a waste category.

    Returns
    -------
    dict : Contains decomposition_years, co2_per_kg, water_pollution_risk,
           soil_contamination, landfill_contribution, recycling_energy_saving_pct,
           description, impact_class, impact_score, impact_color
    """
    normalized = _normalize_class(waste_class)
    if normalized not in ENVIRONMENTAL_METRICS:
        return {"error": f"Unknown class: '{waste_class}'"}

    metrics = ENVIRONMENTAL_METRICS[normalized].copy()
    metrics["impact_class"] = IMPACT_CLASS_MAP[normalized]
    metrics["impact_score"] = IMPACT_SCORE_MAP[normalized]
    metrics["impact_color"] = SEVERITY_COLORS[metrics["impact_class"]]
    return metrics


def get_batch_scores(waste_classes: list) -> list:
    """
    Score multiple waste items at once.

    Parameters
    ----------
    waste_classes : list of str

    Returns
    -------
    list of dict : Each containing class, impact_class, impact_score, impact_color
    """
    results = []
    for cls in waste_classes:
        normalized = _normalize_class(cls)
        results.append({
            "class": normalized,
            "impact_class": get_impact_class(normalized),
            "impact_score": get_impact_score(normalized),
            "impact_color": get_impact_color(normalized),
        })
    return results


def get_severity_ranges() -> dict:
    """Return the severity classification ranges."""
    return SEVERITY_RANGES.copy()


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_class(waste_class: str) -> str:
    """Normalize class name to match internal keys."""
    name = waste_class.strip()
    aliases = {
        "Organic":           "Organic Waste",
        "organic":           "Organic Waste",
        "organic waste":     "Organic Waste",
        "Ewaste":            "E-Waste",
        "ewaste":            "E-Waste",
        "e-waste":           "E-Waste",
        "E-waste":           "E-Waste",
        "E Waste":           "E-Waste",
        "Electronic Waste":  "E-Waste",
        "plastic":           "Plastic",
        "paper":             "Paper",
        "glass":             "Glass",
        "metal":             "Metal",
    }
    return aliases.get(name, name)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 9: Environmental Impact Prediction Engine")
    print("=" * 70)

    for cls in IMPACT_CLASS_MAP:
        metrics = get_environmental_metrics(cls)
        print(f"\n{'─' * 60}")
        print(f"  Category            : {cls}")
        print(f"  Impact Class        : {metrics['impact_class']}")
        print(f"  Impact Score        : {metrics['impact_score']} / 100")
        print(f"  CO₂ per kg          : {metrics['co2_per_kg']} kg")
        print(f"  Water Pollution     : {metrics['water_pollution_risk']}")
        print(f"  Soil Contamination  : {metrics['soil_contamination']}")
        print(f"  Recycling Savings   : {metrics['recycling_energy_saving_pct']}% energy")

    # Batch scoring example
    print(f"\n{'=' * 70}")
    print("  Batch Scoring Example:")
    batch = get_batch_scores(["Plastic", "Organic", "E-Waste"])
    for item in batch:
        print(f"    {item['class']:15s} → {item['impact_class']:8s} (Score: {item['impact_score']})")

    print("=" * 70)
