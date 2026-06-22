"""
recommendation_engine.py
========================
MODULE 8: Recycling Recommendation Engine

A rule-based recommendation system that provides detailed recycling
guidance, disposal methods, decomposition timelines, and sustainability
tips for each of the 6 waste categories.

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""

import json
import os

# ══════════════════════════════════════════════════════════════════════════════
# RECYCLING RECOMMENDATION DATABASE
# ══════════════════════════════════════════════════════════════════════════════

RECOMMENDATION_DB = {
    # ── PLASTIC ───────────────────────────────────────────────────────────────
    "Plastic": {
        "category": "Plastic",
        "recyclable": "Yes",
        "recycling_method": "Shred → Melt → Polyester Fiber / New Containers",
        "decomposition_time": "450 Years",
        "impact_level": "High",
        "disposal_instructions": [
            "Rinse out all food residue with water before recycling.",
            "Check the resin identification code (1–7) on the bottom.",
            "PET (1) and HDPE (2) are most widely accepted for curbside recycling.",
            "Remove caps and labels where required by local guidelines.",
            "Flatten bottles and containers to save bin space.",
        ],
        "donts": [
            "Do NOT place plastic bags in curbside recycling — they jam machinery.",
            "Do NOT recycle styrofoam (EPS #6) in standard bins.",
            "Do NOT mix different plastic types in the same batch.",
        ],
        "environmental_tips": [
            "Switch to reusable water bottles and shopping bags.",
            "Buy products with minimal plastic packaging.",
            "Support companies using recycled plastic (rPET).",
        ],
        "sub_items": {
            "Plastic Bottle": {
                "recyclable": "Yes (PET #1)",
                "method": "Shred into flakes → Melt into polyester fiber for clothing",
                "decomposition": "450 Years",
            },
            "Plastic Bag": {
                "recyclable": "Special drop-off only",
                "method": "Collected at grocery stores → Melted into composite lumber",
                "decomposition": "500–1000 Years",
            },
            "Plastic Container": {
                "recyclable": "Yes (HDPE #2)",
                "method": "Washed → Granulated → Remolded into new containers",
                "decomposition": "450 Years",
            },
        },
        "co2_saved_kg": 1.5,
        "fun_fact": "Recycling 1 ton of plastic saves 5,774 kWh of energy and 16.3 barrels of oil.",
    },

    # ── PAPER ─────────────────────────────────────────────────────────────────
    "Paper": {
        "category": "Paper",
        "recyclable": "Yes",
        "recycling_method": "Pulp → De-ink → Re-roll into New Paper/Cardboard",
        "decomposition_time": "2–6 Weeks",
        "impact_level": "Low",
        "disposal_instructions": [
            "Keep paper dry and free of food contamination.",
            "Flatten cardboard boxes before placing in the bin.",
            "Remove plastic windows from envelopes.",
            "Shred confidential documents — shredded paper is still recyclable.",
            "Bundle newspapers and magazines together.",
        ],
        "donts": [
            "Do NOT recycle wax-coated or plastic-laminated paper.",
            "Do NOT include paper towels, tissues, or napkins (compost instead).",
            "Do NOT recycle paper soiled with grease or food (e.g., pizza boxes).",
        ],
        "environmental_tips": [
            "Print double-sided whenever possible.",
            "Use digital documents to reduce paper consumption.",
            "Choose recycled-content paper products.",
        ],
        "sub_items": {
            "Newspaper": {
                "recyclable": "Yes",
                "method": "De-inked and re-pulped into newsprint",
                "decomposition": "6 Weeks",
            },
            "Cardboard Box": {
                "recyclable": "Yes",
                "method": "Flattened → Pulped → New corrugated board",
                "decomposition": "2 Months",
            },
            "Office Paper": {
                "recyclable": "Yes",
                "method": "De-inked → High-grade recycled paper",
                "decomposition": "2–6 Weeks",
            },
        },
        "co2_saved_kg": 0.8,
        "fun_fact": "Paper can be recycled up to 7 times before fibers become too short.",
    },

    # ── GLASS ─────────────────────────────────────────────────────────────────
    "Glass": {
        "category": "Glass",
        "recyclable": "Yes (Infinite Loop)",
        "recycling_method": "Crush → Cullet → Remelt into New Glass Products",
        "decomposition_time": "1 Million Years",
        "impact_level": "Medium",
        "disposal_instructions": [
            "Rinse out residue from jars and bottles.",
            "Remove metal caps and lids (recycle separately).",
            "Separate by color if required (clear, green, brown).",
            "Do NOT break glass intentionally — whole containers are preferred.",
            "Place in designated glass recycling bins.",
        ],
        "donts": [
            "Do NOT include ceramics, mirrors, or window glass — different melting points.",
            "Do NOT recycle light bulbs in glass bins (take to special collection).",
            "Do NOT include Pyrex or tempered glass.",
        ],
        "environmental_tips": [
            "Reuse glass jars for food storage.",
            "Choose glass over plastic when buying beverages.",
            "Support bottle-deposit return schemes.",
        ],
        "sub_items": {
            "Glass Bottle": {
                "recyclable": "Yes (Infinite)",
                "method": "Crushed into cullet → Remelted at 1500°C → New bottles",
                "decomposition": "1 Million Years",
            },
            "Glass Jar": {
                "recyclable": "Yes (Infinite)",
                "method": "Cleaned → Crushed → Remelted with zero quality loss",
                "decomposition": "1 Million Years",
            },
        },
        "co2_saved_kg": 0.5,
        "fun_fact": "Glass is 100% recyclable with zero quality loss, endlessly.",
    },

    # ── METAL ─────────────────────────────────────────────────────────────────
    "Metal": {
        "category": "Metal",
        "recyclable": "Yes",
        "recycling_method": "Crush → Shred → Smelt → New Metal Products",
        "decomposition_time": "50–200 Years",
        "impact_level": "Medium",
        "disposal_instructions": [
            "Rinse out food cans and crush to save space.",
            "Separate aluminum cans from steel/tin cans with a magnet.",
            "Remove paper labels where possible.",
            "Include clean aluminum foil and trays.",
            "Take scrap metal to a dedicated metal recycler for payment.",
        ],
        "donts": [
            "Do NOT include aerosol cans that are not fully empty.",
            "Do NOT recycle paint cans with residual paint (hazardous waste).",
            "Do NOT include electronics or appliances in metal recycling.",
        ],
        "environmental_tips": [
            "Collect aluminum cans — they have high resale value.",
            "Use reusable metal water bottles and cutlery.",
            "Support products made from recycled aluminum.",
        ],
        "sub_items": {
            "Aluminum Can": {
                "recyclable": "Yes",
                "method": "Melted and re-cast in under 60 days",
                "decomposition": "200 Years",
            },
            "Steel Can": {
                "recyclable": "Yes",
                "method": "Magnetically separated → Smelted → New steel",
                "decomposition": "50 Years",
            },
            "Scrap Metal": {
                "recyclable": "Yes (scrap yard)",
                "method": "Sorted → Shredded → Electric arc furnace",
                "decomposition": "100–500 Years",
            },
        },
        "co2_saved_kg": 2.1,
        "fun_fact": "Recycling one aluminum can saves enough energy to run a TV for 3 hours.",
    },

    # ── ORGANIC WASTE ─────────────────────────────────────────────────────────
    "Organic Waste": {
        "category": "Organic Waste",
        "recyclable": "Compostable",
        "recycling_method": "Composting → Nutrient-rich Fertilizer / Biogas",
        "decomposition_time": "2–4 Weeks",
        "impact_level": "Low",
        "disposal_instructions": [
            "Collect food scraps in a kitchen compost bin or biodegradable bag.",
            "Include fruit/vegetable peels, coffee grounds, eggshells, and tea bags.",
            "Add yard waste: leaves, grass clippings, small branches.",
            "Turn compost regularly to aerate and speed decomposition.",
            "Use municipal green/brown bin if available.",
        ],
        "donts": [
            "Do NOT compost meat, dairy, or oily foods in home compost (attracts pests).",
            "Do NOT include diseased plants or chemically-treated wood.",
            "Do NOT add pet waste to food compost.",
        ],
        "environmental_tips": [
            "Start a backyard compost heap — free fertilizer!",
            "Use compost in garden beds to improve soil health.",
            "Reduce food waste by planning meals and storing food properly.",
        ],
        "sub_items": {
            "Food Scraps": {
                "recyclable": "Compostable",
                "method": "Aerobic composting → Humus fertilizer",
                "decomposition": "2–4 Weeks",
            },
            "Yard Waste": {
                "recyclable": "Compostable",
                "method": "Mulched → Compost or biomass energy",
                "decomposition": "1–3 Months",
            },
        },
        "co2_saved_kg": 0.3,
        "fun_fact": "Composting reduces methane emissions by diverting organics from landfills.",
    },

    # ── E-WASTE ───────────────────────────────────────────────────────────────
    "E-Waste": {
        "category": "E-Waste",
        "recyclable": "Specialized Facility Only",
        "recycling_method": "Disassembly → Precious Metal Recovery → Component Recycling",
        "decomposition_time": "Never (Contains Toxic Heavy Metals)",
        "impact_level": "Critical",
        "disposal_instructions": [
            "NEVER place e-waste in regular trash — contains lead, mercury, cadmium.",
            "Locate certified e-waste recyclers (e-Stewards, R2 certified).",
            "Remove batteries separately (recycle at battery drop-off points).",
            "Wipe personal data from devices before recycling.",
            "Check manufacturer take-back programs (Apple, Dell, HP).",
        ],
        "donts": [
            "Do NOT burn e-waste — releases dioxins and heavy metals into the air.",
            "Do NOT disassemble electronics without proper safety equipment.",
            "Do NOT dump e-waste in landfills or water bodies.",
        ],
        "environmental_tips": [
            "Extend device lifespan with repairs instead of replacement.",
            "Donate working electronics to schools or charities.",
            "Buy refurbished electronics to reduce demand for new mining.",
        ],
        "sub_items": {
            "Battery": {
                "recyclable": "Specialized Facility",
                "method": "Sorted by chemistry → Smelted → Metal recovery",
                "decomposition": "100+ Years (toxic leaching)",
            },
            "Circuit Board": {
                "recyclable": "Precious Metal Recovery",
                "method": "Shredded → Refined → Gold, silver, copper extraction",
                "decomposition": "Never",
            },
            "Light Bulb": {
                "recyclable": "Special Collection",
                "method": "Mercury captured → Glass recycled → Metal recovered",
                "decomposition": "Indefinite (mercury contamination)",
            },
            "Mobile Phone": {
                "recyclable": "Manufacturer take-back / e-waste center",
                "method": "Disassembled → Components sorted → Metals recovered",
                "decomposition": "Never",
            },
        },
        "co2_saved_kg": 3.4,
        "fun_fact": "One ton of circuit boards contains 40–800x more gold than one ton of gold ore.",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_recommendation(waste_class: str) -> dict:
    """
    Get the full recycling recommendation for a given waste class.

    Parameters
    ----------
    waste_class : str
        One of: 'Plastic', 'Paper', 'Glass', 'Metal', 'Organic Waste', 'E-Waste'

    Returns
    -------
    dict : Complete recommendation including disposal instructions,
           recycling method, decomposition time, and tips.
    """
    # Normalize input
    normalized = waste_class.strip().title()

    # Handle common aliases
    aliases = {
        "Organic": "Organic Waste",
        "Ewaste": "E-Waste",
        "E Waste": "E-Waste",
        "Electronic Waste": "E-Waste",
        "E-waste": "E-Waste",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in RECOMMENDATION_DB:
        return {
            "error": f"Unknown waste class: '{waste_class}'",
            "valid_classes": list(RECOMMENDATION_DB.keys()),
        }

    return RECOMMENDATION_DB[normalized]


def get_summary(waste_class: str) -> dict:
    """
    Get a concise summary for a waste class (for display in the web app).

    Returns
    -------
    dict with keys: category, recyclable, method, decomposition, impact, tip, co2
    """
    rec = get_recommendation(waste_class)
    if "error" in rec:
        return rec

    return {
        "category": rec["category"],
        "recyclable": rec["recyclable"],
        "method": rec["recycling_method"],
        "decomposition": rec["decomposition_time"],
        "impact": rec["impact_level"],
        "tip": rec["disposal_instructions"][0],
        "co2_saved_kg": rec["co2_saved_kg"],
        "fun_fact": rec["fun_fact"],
    }


def get_all_recommendations() -> dict:
    """Return the complete recommendation database."""
    return RECOMMENDATION_DB


def export_database(filepath: str = "recommendation_database.json") -> str:
    """Export the recommendation database to a JSON file."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(RECOMMENDATION_DB, f, indent=2, ensure_ascii=False)
    return os.path.abspath(filepath)


# ══════════════════════════════════════════════════════════════════════════════
# DEMO / CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 8: Recycling Recommendation Engine")
    print("=" * 70)

    for cls in RECOMMENDATION_DB:
        summary = get_summary(cls)
        print(f"\n{'─' * 50}")
        print(f"  Category          : {summary['category']}")
        print(f"  Recyclable        : {summary['recyclable']}")
        print(f"  Method            : {summary['method']}")
        print(f"  Decomposition     : {summary['decomposition']}")
        print(f"  Impact Level      : {summary['impact']}")
        print(f"  CO₂ Saved/Item    : {summary['co2_saved_kg']} kg")
        print(f"  Fun Fact          : {summary['fun_fact']}")

    print(f"\n{'=' * 70}")
    print("  Recommendation Engine ready.")
    print("=" * 70)
