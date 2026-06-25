"""
chatbot.py
==========
MODULE 12: AI Waste Management Chatbot

A rule-based waste management chatbot with optional Gemini AI integration.
Provides waste disposal guidance, recycling suggestions, and environmental
awareness tips — works fully offline without any API key.

Author  : AI & Data Science Engineering Team
Project : AI-Powered Smart Waste Classification & Recycling System
"""

import re
import random
import os
import time

# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE — Rule-Based Responses
# ══════════════════════════════════════════════════════════════════════════════

GREETING_PATTERNS = [
    r"\b(hi|hello|hey|greetings|howdy|good\s*(morning|afternoon|evening))\b",
]

FAREWELL_PATTERNS = [
    r"\b(bye|goodbye|see\s*you|thanks|thank\s*you|cheers|exit|quit)\b",
]

# Category-specific knowledge base
# Category-specific knowledge base
WASTE_KNOWLEDGE = {
    "plastic": {
        "patterns": [
            r"\bplastic\b", r"\bPET\b", r"\bHDPE\b", r"\bpolythene\b",
            r"\bplastic\s*bottle\b", r"\bplastic\s*bag\b", r"\bplastic\s*container\b",
            r"\bstyrofoam\b", r"\bpolystyrene\b",
        ],
        "responses": [
            (
                "### ♻️ Professional Plastic Recycling & Materials Science Guide\n\n"
                "Plastic polymers represent a diverse class of synthetic organics, categorized by Resin Identification Codes (RIC) 1 through 7.\n\n"
                "| Resin Code | Polymer Type | Common Applications | Recyclability | Circular Economy Value |\n"
                "| :---: | :--- | :--- | :--- | :--- |\n"
                "| **1** | PETE (Polyethylene Terephthalate) | Water bottles, soda cups | **High** | High (Fibers, sheet/film) |\n"
                "| **2** | HDPE (High-Density Polyethylene) | Milk jugs, shampoo bottles | **High** | High (Pipes, decking, packaging) |\n"
                "| **3** | PVC (Polyvinyl Chloride) | Pipes, vinyl siding, cable wire | **Low** | Critical/Toxic (Avoid landfill) |\n"
                "| **4** | LDPE (Low-Density Polyethylene) | Squeeze bottles, shopping bags | **Moderate** | Medium (Plastic lumber) |\n"
                "| **5** | PP (Polypropylene) | Yogurt tubs, bottle caps, straws | **Moderate** | High (Auto parts, industrial fibers) |\n"
                "| **6** | PS (Polystyrene) | Plastic cups, styrofoam packaging | **Low** | Poor (Landfill/Thermal recovery) |\n"
                "| **7** | Other (Polycarbonate, Acrylic, etc.) | Electronics, composite plastics | **Very Low** | Low (Upcycling only) |\n\n"
                "#### 🔍 Disposal & Processing Instructions\n"
                "1. **Decontamination:** Rinse container thoroughly. Residual sugar or organic matter degrades polymer purity during thermal reprocessing.\n"
                "2. **Compaction:** Crush bottles to reduce volume, optimizing logistics and decreasing transportation carbon footprint.\n"
                "3. **Segregation:** Keep films/bags separate. Flexible plastics wrap around mechanical sorting gears, causing system shutdowns.\n\n"
                "#### 📊 Environmental Metrics & Energy Conservation\n"
                "• **Carbon Footprint:** Manufacturing virgin PET emits ~1.89 kg CO₂ per kg of plastic. Using recycled PET (rPET) reduces this emissions factor by **70%**.\n"
                "• **Energy Savings:** Recycling plastic saves approximately **5.77 Billion Joules (1.6 MWh)** of energy per ton compared to refining crude oil."
            ),
        ],
    },
    "paper": {
        "patterns": [
            r"\bpaper\b", r"\bcardboard\b", r"\bnewspaper\b", r"\bmagazine\b",
            r"\bcarton\b", r"\benvelope\b",
        ],
        "responses": [
            (
                "### 📄 Professional Paper & Cellulose Fiber Recovery Guide\n\n"
                "Paper recycling relies on recovering cellulose fibers from wood pulp. Moisture and organic contaminants are the primary challenges in paper recovery.\n\n"
                "| Paper Class | Example Items | Processing Method | Fiber Quality | Recycle Limit |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| **Cardboard (OCC)** | Shipping boxes, package cartons | Pulping & de-inking | **Very High (Long fibers)** | Up to 7 times |\n"
                "| **Mixed Paper** | Newspapers, junk mail, office print | Hydrapulping | **Medium (Shorter fibers)** | Up to 5 times |\n"
                "| **Contaminated Paper** | Pizza boxes, greasy food wrappers | Composting (Organic bin) | **Unusable for paper** | 0 (Compost only) |\n\n"
                "#### 🔍 Disposal & Processing Instructions\n"
                "1. **Keep Dry:** Wet paper fibers weaken and degrade rapidly. Ensure paper products are shielded from moisture.\n"
                "2. **Zero Grease:** Do not recycle paper contaminated with food grease. Grease prevents wood fibers from bonding during the pulping stage.\n"
                "3. **Contaminant Removal:** Remove plastic window inserts from envelopes and steel staples if possible (though pulping filters catch some metal).\n\n"
                "#### 📊 Environmental Metrics & Energy Conservation\n"
                "• **Resource Preservation:** Recycling 1 metric ton of paper saves approximately **17 mature trees**, **26,500 liters (7,000 gallons) of water**, and 3 cubic yards of landfill space.\n"
                "• **Energy Savings:** Paper recycling consumes **40% less energy** than producing paper from raw virgin wood pulp."
            ),
        ],
    },
    "glass": {
        "patterns": [
            r"\bglass\b", r"\bbottle\b.*\bglass\b", r"\bglass\s*jar\b",
            r"\bglass\s*container\b",
        ],
        "responses": [
            (
                "### 🫙 Professional Glass & Silica Recycling Guide\n\n"
                "Glass is a non-crystalline, amorphous solid composed primarily of silica (SiO₂). It is a closed-loop material, meaning it can be recycled indefinitely without degradation of physical properties.\n\n"
                "| Glass Category | Typical Color | Chemical Additives | Sorting Requirement | Remelting Temp |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| **Flint Glass** | Clear | Pure Silica | Separate from colored | ~1,500°C |\n"
                "| **Amber Glass** | Brown | Iron, Sulfur, Carbon | Separate to keep color | ~1,400°C |\n"
                "| **Emerald Glass** | Green | Chromium Oxide | Separate to keep color | ~1,400°C |\n"
                "| **Borosilicate** | Pyrex / Lab Glass | Boron Trioxide | **Do NOT recycle curbside** | ~1,600°C (Contaminant) |\n\n"
                "#### 🔍 Disposal & Processing Instructions\n"
                "1. **Cleanliness:** Empty all residual liquids. Remove metal caps or plastic labels where feasible.\n"
                "2. **Color Sorting:** Group glass by color. Mixed color cullet (crushed glass) has low commercial value and is restricted to aggregate or fiberglass uses.\n"
                "3. **No Pyrex/Ceramics:** Heat-resistant glass (Pyrex) and ceramics have different melting characteristics. Just a single ceramic mug can ruin an entire batch of molten glass.\n\n"
                "#### 📊 Environmental Metrics & Energy Conservation\n"
                "• **Decomposition Rate:** Glass does not degrade naturally; landfill lifetime is estimated at **1,000,000+ years**.\n"
                "• **Energy Savings:** Utilizing recycled glass cullet lowers furnace operating temperatures, saving **20-30% of energy** and reducing CO₂ emissions by 10-15%."
            ),
        ],
    },
    "metal": {
        "patterns": [
            r"\bmetal\b", r"\baluminum\b", r"\baluminium\b", r"\bsteel\b",
            r"\btin\b", r"\bcan\b", r"\bcopper\b", r"\biron\b",
        ],
        "responses": [
            (
                "### 🔩 Professional Metallurgy & Metal Circularity Guide\n\n"
                "Metals are highly valuable secondary raw materials. Recycling metals preserves minerals, prevents massive open-cast mining, and drastically reduces greenhouse gas emissions.\n\n"
                "| Metal Type | Magnetic? | Common Examples | Primary Recovery Process | Energy Savings vs. Virgin |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| **Aluminum** | No | Beverage cans, foil trays | Shredding, melting, ingot casting | **95% Energy Saved** |\n"
                "| **Steel / Tin** | Yes | Food soup cans, steel frames | Magnetic separation, basic oxygen furnace | **75% Energy Saved** |\n"
                "| **Copper** | No | Electrical wires, pipes | Electrolytic refining, smelting | **85% Energy Saved** |\n\n"
                "#### 🔍 Disposal & Processing Instructions\n"
                "1. **Rinse & Dry:** Remove food organic residue to prevent toxic fumes during secondary smelting.\n"
                "2. **Compaction:** Flatten aluminum cans to save bin space and optimize transportation efficiency.\n"
                "3. **Magnetic Check:** Use a kitchen magnet to quickly sort: steel will stick, aluminum will not.\n\n"
                "#### 📊 Environmental Metrics & Energy Conservation\n"
                "• **Energy Comparison:** Producing 1 ton of aluminum from bauxite ore requires **14,000 kWh**. Producing it from recycled cans requires only **700 kWh**.\n"
                "• **Carbon Offset:** Recycling 1 ton of steel prevents the mining of 1.1 tons of iron ore, 630 kg of coal, and avoids **1.8 tons of CO₂ emissions**."
            ),
        ],
    },
    "organic": {
        "patterns": [
            r"\borganic\b", r"\bfood\s*(waste|scrap)\b", r"\bcompost\b",
            r"\bbiodegradable\b", r"\bfruit\b.*\bpeel\b", r"\bvegetable\b",
            r"\bleaves\b", r"\bgrass\b", r"\bgarden\s*waste\b",
        ],
        "responses": [
            (
                "### 🌱 Professional Organic Waste & Bio-Resource Recovery Guide\n\n"
                "Organic waste comprises biodegradable carbonaceous materials. Diverting organic matter from landfills to aerobic composting systems is a critical tool for climate change mitigation.\n\n"
                "| Method | Output Product | Timeframe | Best Input Materials | Materials to Avoid (Home) |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| **Aerobic Composting** | Rich Topsoil Fertilizer | 2–4 months | Fruits, vegetables, coffee grounds, leaves | Meat, dairy, fats, pet waste |\n"
                "| **Anaerobic Digestion** | Biogas (Methane) & Digestate | 2–4 weeks | Municipal food waste, agricultural manure | Excessive woody/lignin materials |\n\n"
                "#### 🔍 Composting & Disposal Guidelines\n"
                "1. **Carbon-to-Nitrogen Balance:** Maintain a healthy ratio of **Browns** (carbon-rich: dry leaves, cardboard) to **Greens** (nitrogen-rich: food scraps, grass clippings) at around 30:1.\n"
                "2. **Moisture & Aeration:** Ensure compost pile is damp but not soggy (like a wrung-out sponge). Turn the pile regularly to supply oxygen to aerobic microbes.\n"
                "3. **Municipal Green Bins:** Use green bins for dairy, meat, and bones only if your local municipal system uses high-temperature industrial composting.\n\n"
                "#### 📊 Environmental Metrics & Carbon Offsets\n"
                "• **Methane Mitigation:** In landfills, organic waste decomposes anaerobically, emitting **Methane (CH₄)**, which is 28-36 times more potent than CO₂. Composting processes it aerobically, emitting minimal greenhouse gases.\n"
                "• **Soil Health:** Compost restores soil organic matter, improves water retention by 20%, and captures atmospheric carbon into the soil."
            ),
        ],
    },
    "ewaste": {
        "patterns": [
            r"\be[\-\s]*waste\b", r"\belectronic\b", r"\bbatter(y|ies)\b",
            r"\bphone\b", r"\bcomputer\b", r"\blaptop\b", r"\btablet\b",
            r"\bcharger\b", r"\bcable\b", r"\blight\s*bulb\b", r"\bLED\b",
            r"\bcircuit\b", r"\bprinter\b", r"\bmonitor\b",
        ],
        "responses": [
            (
                "### ⚡ Professional E-Waste & Precious Metals Recovery Guide\n\n"
                "Electronic waste (e-waste) contains both high-value precious metals and hazardous heavy metals. Safe disposal and recycling are mandatory to prevent toxic leaching and recover rare earth elements.\n\n"
                "| Component | Precious Elements | Toxic Heavy Metals | Disposal Method | Environmental Hazard |\n"
                "| :--- | :--- | :--- | :--- | :--- |\n"
                "| **Circuit Boards** | Gold, Silver, Palladium, Copper | Lead, Beryllium | Certified Recycler | Bioaccumulation, soil toxicity |\n"
                "| **Li-ion Batteries** | Lithium, Cobalt, Nickel, Manganese | Cobalt | Specialized Drop-off | Thermal runaway (fire), water pollution |\n"
                "| **CRT Monitors** | Copper | Lead, Cadmium, Mercury | Hazardous Waste Facility | Neurotoxic dust release |\n\n"
                "#### 🔍 Disposal & Processing Instructions\n"
                "1. **Never Landfill:** E-waste must never be placed in standard municipal trash or regular curbside recycling bins.\n"
                "2. **Data Sanitation:** Perform a factory reset or physically destroy storage drives before dropping off computers or mobile phones.\n"
                "3. **Manufacturer Takeback:** Leverage certified programs (e.g. Apple Trade In, Best Buy recycling) to ensure devices are processed under e-Stewards or R2 standards.\n\n"
                "#### 📊 Environmental Metrics & Material Science\n"
                "• **Resource Density:** One ton of cell phone circuit boards contains **up to 100 times more gold** than one ton of gold ore.\n"
                "• **Toxic Composition:** E-waste represents only ~2% of solid waste in landfills, but contributes **70% of heavy metal contamination** in landfills worldwide."
            ),
        ],
    },
}

# General recycling and environment Q&A
GENERAL_KNOWLEDGE = {
    "recycling_tips": {
        "patterns": [
            r"\brecycl(e|ing)\s*tip\b", r"\bhow\s*to\s*recycle\b",
            r"\brecycling\s*guide\b", r"\bwhat\s*can\s*(i|we)\s*recycle\b",
        ],
        "responses": [
            (
                "### ♻️ Professional Recycling Best Practices & Material Circularity\n\n"
                "Optimizing resource loops requires strict adherence to international waste sorting standards. Use the three-step hierarchy: **Reduce first, Reuse second, Recycle third**.\n\n"
                "#### 📋 System Guidelines for Maximum Circular Efficiency\n"
                "1. **Decontaminate Thoroughly:** A single jar with food residue can contaminate a whole truckload of paper during transport. Rinse containers!\n"
                "2. **Know Your Local Regulations:** Waste management is highly localized. Check with your municipality regarding dual-stream vs. single-stream setups.\n"
                "3. **Avoid 'Wishcycling':** Do not put non-recyclable items (like garden hoses, toys, or plastic film) in the bin hoping they will be recycled. This clogs optical sorting equipment.\n"
                "4. **Flatten & Compress:** Maximize spatial density. Flattening cardboard boxes reduces shipping volume, which lowers transportation fuel emissions.\n\n"
                "#### 📊 Global Recovery Statistics\n"
                "• **Plastic:** Global recycling rate remains under **10%** due to polymer classification limits.\n"
                "• **Aluminum:** Global recycling rate is over **75%**, showcasing the economic efficiency of closed-loop metal recovery.\n"
                "• **Paper:** Around **65%** of paper products are successfully recovered and re-pulped globally."
            ),
        ],
    },
    "environment": {
        "patterns": [
            r"\benvironment\b", r"\bclimate\b", r"\bglobal\s*warming\b",
            r"\bcarbon\s*footprint\b", r"\bsustain\b", r"\bgreen\b",
            r"\beco\s*friendly\b", r"\bplanet\b",
        ],
        "responses": [
            (
                "### 🌍 Advanced Environmental Science & Circular Economy Overview\n\n"
                "Global waste production is a primary driver of climate change, ocean acidification, and biosphere degradation. Transitioning from a linear model (Take-Make-Waste) to a circular economy is crucial for long-term sustainability.\n\n"
                "#### 📊 Global Climate & Resource Statistics\n"
                "• **Carbon Emissions:** Waste management and landfill decay account for approximately **5% of global greenhouse gas emissions**.\n"
                "• **Landfill Methane:** Decomposing landfill mass produces **Landfill Gas (LFG)**, composed of ~50% methane and ~50% CO₂.\n"
                "• **Oceanic Plastisphere:** The Great Pacific Garbage Patch spans over **1.6 million square kilometers** and contains over 1.8 trillion plastic pieces.\n\n"
                "#### 🌱 Circular Economy Framework\n"
                "• **Design Out Waste:** Products should be designed for disassembly and material recovery from the outset.\n"
                "• **Keep Materials in Use:** Reuse products and components to extend their lifecycle before material-level recycling.\n"
                "• **Regenerate Natural Systems:** Compost organic materials to return vital nutrients to agricultural soils."
            ),
        ],
    },
    "reduce": {
        "patterns": [
            r"\breduce\b", r"\bminimize\b", r"\bless\s*waste\b",
            r"\bzero\s*waste\b", r"\bwaste\s*reduction\b",
        ],
        "responses": [
            (
                "### 📉 Waste Reduction & Zero-Waste Methodologies\n\n"
                "The Zero-Waste philosophy aims to guide people in changing their lifestyles and practices to emulate sustainable natural cycles, where all discarded materials are designed to become resources for others to use.\n\n"
                "#### 📋 The 5 R's Zero-Waste Hierarchy\n"
                "1. **Refuse:** Say no to single-use plastics, flyers, and disposable packaging.\n"
                "2. **Reduce:** Decrease overall consumption of goods and choose high-quality, durable items.\n"
                "3. **Reuse:** Swap disposables for reusables (e.g. bags, cups, bottles, containers).\n"
                "4. **Rot:** Compost organic food scraps and yard trimmings to generate soil nutrients.\n"
                "5. **Recycle:** Process remaining sorted recyclables as a last resort.\n\n"
                "#### 💡 Practical Tips for Daily Life\n"
                "• Buy in bulk to reduce packaging waste by up to 30%.\n"
                "• Choose loose fruits and vegetables instead of pre-packaged plastic bags.\n"
                "• Opt for digital bills and documentation to minimize paper waste."
            ),
        ],
    },
    "decomposition": {
        "patterns": [
            r"\bdecompos(e|ition)\b", r"\bhow\s*long\b.*\b(break|degrade|last)\b",
            r"\bbiodegrad\b",
        ],
        "responses": [
            (
                "### ⏳ Material Decomposition Timeline & Environmental Persistence\n\n"
                "Decomposition is the process by which organic substances are broken down into simpler organic matter. Synthetic materials lack natural biological enzymes for decomposition, resulting in extreme environmental persistence.\n\n"
                "| Material Category | Representative Item | Decomposition Time | Environmental Byproducts |\n"
                "| :--- | :--- | :--- | :--- |\n"
                "| **Organic Waste** | Food scraps, peels | **2–4 Weeks** | Nutrient compost / LFG |\n"
                "| **Paper** | Newspapers, napkins | **2–6 Weeks** | Cellulose fibers |\n"
                "| **Textiles** | Cotton shirts | **1–5 Months** | Natural threads |\n"
                "| **Wood** | Plywood, lumber | **10–15 Years** | Humus / Cellulose |\n"
                "| **Metal** | Steel cans | **50 Years** | Iron oxides (Rust) |\n"
                "| **Metal** | Aluminum cans | **200 Years** | Aluminum oxide particles |\n"
                "| **Plastic** | PET Beverage bottles | **450 Years** | Microplastics / Nanoplastics |\n"
                "| **Glass** | Silica bottles | **1 Million Years** | Inert silica sand |\n"
                "| **Synthetic** | Styrofoam (Polystyrene) | **Never** | Toxic monomer leaching |\n"
                "| **E-Waste** | Electronics, batteries | **Never** | Lead, Mercury, Cadmium leaching |"
            ),
        ],
    },
}

# Fallback responses when no pattern matches
FALLBACK_RESPONSES = [
    (
        "### 🤖 Eco-Bot Waste Management Assistant (Offline Mode)\n\n"
        "I am operating in offline rule-based mode because no Gemini API key is configured. I can provide detailed guidance on waste classification and recycling loops.\n\n"
        "#### 🔍 Topics I Can Help You With:\n"
        "• **Plastic:** Polymer codes (PET, HDPE, PVC, PP, etc.) and recovery rules.\n"
        "• **Paper & Cardboard:** Fiber lengths, grease contamination limits.\n"
        "• **Glass:** Color segregation (flint, amber, emerald) and Pyrex hazards.\n"
        "• **Metal:** Metallurgy, energy conservation, magnetic sorting.\n"
        "• **Organic Waste:** Aerobic composting, C:N balance, biogas methane mitigation.\n"
        "• **E-Waste:** Li-ion batteries, circuit board elements, heavy metals.\n"
        "• **General Knowledge:** Reduction/Zero-Waste hierarchy, decomposition timelines, and environmental stats.\n\n"
        "👉 **Try asking me:** *'How do I recycle plastic bottles?'* or *'What is the decomposition time of glass?'*"
    ),
]

GREETING_RESPONSES = [
    "👋 Hello! I'm your AI Waste Management Assistant. Ask me anything about recycling, waste disposal, or environmental impact!",
    "Hi there! ♻️ I'm here to help with waste sorting, recycling guidance, and eco-tips. What would you like to know?",
    "Hey! 🌍 Ready to help you make greener choices. What's on your mind?",
]

FAREWELL_RESPONSES = [
    "Thanks for caring about the environment! 🌱 Every small action counts. Goodbye!",
    "Bye! Remember: Reduce → Reuse → Recycle ♻️",
    "See you! Keep sorting your waste properly — the planet thanks you! 🌍",
]


# ══════════════════════════════════════════════════════════════════════════════
# CHATBOT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class WasteManagementChatbot:
    """
    Rule-based chatbot for waste management queries.
    Falls back to Gemini API when available for advanced/unknown queries.
    """

    def __init__(self, gemini_api_key: str = None):
        """
        Parameters
        ----------
        gemini_api_key : str, optional
            Google Gemini API key for enhanced AI responses.
            If None, chatbot operates in rule-based mode only.
        """
        self.gemini_model = None
        self.gemini_available = False
        self.cache = {}

        if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                
                system_instruction = (
                    "You are Eco-Bot, a highly professional AI Waste Management & Recycling Consultant. "
                    "Provide detailed, structured, and context-aware advice about waste disposal, circular economy, "
                    "composting, and eco-friendly choices. Use clear headers, bold text, bullet points, and tables "
                    "where appropriate to structure your responses. Highlight carbon offsets, materials chemistry "
                    "(e.g., polymer types, metal extraction energy savings), and environmental statistics suitable for "
                    "academic final-year engineering projects and research. Maintain an encouraging and professional tone."
                )
                
                self.gemini_model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                self.gemini_available = True
            except Exception:
                self.gemini_available = False

    def get_response(self, user_message: str) -> str:
        """
        Process user message and return an appropriate response (backwards-compatible wrapper).
        """
        return "".join(list(self.get_response_stream(user_message)))

    def get_response_stream(self, user_message: str, history: list = None):
        """
        Process user message and yield chunks of the response.
        Uses cached responses if available, or falls back to Gemini API / local rules.
        """
        def stream_chunked(text: str, delay: float = 0.008, chunk_size: int = 6):
            for i in range(0, len(text), chunk_size):
                yield text[i:i+chunk_size]
                time.sleep(delay)

        if not user_message or not user_message.strip():
            yield "Please type a message to get started! 😊"
            return

        msg_clean = user_message.strip()
        msg_lower = msg_clean.lower()

        # Check Cache
        if msg_lower in self.cache:
            yield from stream_chunked(self.cache[msg_lower])
            return

        # 1. Check greetings
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                response = random.choice(GREETING_RESPONSES)
                self.cache[msg_lower] = response
                yield from stream_chunked(response)
                return

        # 2. Check farewells
        for pattern in FAREWELL_PATTERNS:
            if re.search(pattern, msg_lower, re.IGNORECASE):
                response = random.choice(FAREWELL_RESPONSES)
                self.cache[msg_lower] = response
                yield from stream_chunked(response)
                return

        # 3. Check waste-specific knowledge
        for category, data in WASTE_KNOWLEDGE.items():
            for pattern in data["patterns"]:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    response = random.choice(data["responses"])
                    self.cache[msg_lower] = response
                    yield from stream_chunked(response)
                    return

        # 4. Check general knowledge
        for topic, data in GENERAL_KNOWLEDGE.items():
            for pattern in data["patterns"]:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    response = random.choice(data["responses"])
                    self.cache[msg_lower] = response
                    yield from stream_chunked(response)
                    return

        # 5. Try Gemini API for unmatched queries
        if self.gemini_available and self.gemini_model:
            try:
                formatted_history = []
                if history:
                    for msg in history:
                        role = "user" if msg.get("role") == "user" else "model"
                        content = msg.get("content", "")
                        if content:
                            formatted_history.append({"role": role, "parts": [content]})
                
                chat = self.gemini_model.start_chat(history=formatted_history)
                response_stream = chat.send_message(msg_clean, stream=True)
                
                full_response = ""
                for chunk in response_stream:
                    if chunk.text:
                        full_response += chunk.text
                        yield chunk.text
                
                # Cache the complete response
                if full_response.strip():
                    self.cache[msg_lower] = full_response
                return
            except Exception as e:
                err_str = str(e)
                err_lower = err_str.lower()
                is_auth_error = (
                    "api key" in err_lower or
                    "api_key" in err_lower or
                    "invalid" in err_lower or
                    "auth" in err_lower or
                    "credential" in err_lower or
                    "unauthorized" in err_lower
                )
                if is_auth_error:
                    self.gemini_available = False  # Disable future queries
                    fallback = (
                        "🔑 **Invalid Gemini API Key.** The key provided was "
                        "rejected by Google. Falling back to local offline mode.\n\n"
                        f"**Eco-Bot (Offline Mode):**\n\n" + random.choice(FALLBACK_RESPONSES)
                    )
                    self.cache[msg_lower] = fallback
                    yield from stream_chunked(fallback)
                else:
                    fallback = f"⚠️ AI service temporarily unavailable. (Error: {e})\n\n" + random.choice(FALLBACK_RESPONSES)
                    yield from stream_chunked(fallback)
                return

        # 6. Fallback
        fallback = random.choice(FALLBACK_RESPONSES)
        self.cache[msg_lower] = fallback
        yield from stream_chunked(fallback)


    @property
    def is_ai_enhanced(self) -> bool:
        """Check if Gemini AI enhancement is available."""
        return self.gemini_available


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_chatbot() -> WasteManagementChatbot:
    """Create a chatbot instance, auto-detecting Gemini API key from env."""
    api_key = os.getenv("GEMINI_API_KEY", None)
    return WasteManagementChatbot(gemini_api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
# CLI DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MODULE 12: AI Waste Management Chatbot")
    print("=" * 60)

    bot = create_chatbot()
    mode = "AI-Enhanced (Gemini)" if bot.is_ai_enhanced else "Rule-Based (Offline)"
    print(f"  Mode: {mode}")
    print("  Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print(bot.get_response("bye"))
            break
        response = bot.get_response(user_input)
        print(f"\n🤖 EcoBot: {response}\n")
