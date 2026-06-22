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
WASTE_KNOWLEDGE = {
    "plastic": {
        "patterns": [
            r"\bplastic\b", r"\bPET\b", r"\bHDPE\b", r"\bpolythene\b",
            r"\bplastic\s*bottle\b", r"\bplastic\s*bag\b", r"\bplastic\s*container\b",
            r"\bstyrofoam\b", r"\bpolystyrene\b",
        ],
        "responses": [
            (
                "♻️ **Plastic Recycling Guide**\n\n"
                "• Rinse containers before recycling\n"
                "• Check the resin code (1-7) on the bottom\n"
                "• PET (#1) and HDPE (#2) are most widely recyclable\n"
                "• **Never** put plastic bags in curbside recycling — they jam machinery\n"
                "• Plastic bottles can be recycled into polyester fiber for clothing\n\n"
                "💡 **Tip:** Switch to reusable bottles and bags to reduce plastic waste by up to 90%!"
            ),
            (
                "🌍 **Plastic Environmental Impact**\n\n"
                "• Decomposition time: **450+ years**\n"
                "• Only ~9% of plastic ever produced has been recycled\n"
                "• 8 million tons of plastic enter oceans annually\n"
                "• Microplastics have been found in drinking water, food, and human blood\n\n"
                "♻️ **Action:** Reduce plastic usage first, reuse second, recycle last."
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
                "📄 **Paper Recycling Guide**\n\n"
                "• Keep paper clean and dry — grease/food ruins recyclability\n"
                "• Flatten cardboard boxes before recycling\n"
                "• Paper can be recycled up to **7 times** before fibers are too short\n"
                "• Remove plastic windows from envelopes\n"
                "• Pizza boxes with grease stains should be composted, not recycled\n\n"
                "💡 **Tip:** Recycling 1 ton of paper saves 17 trees, 7,000 gallons of water!"
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
                "🫙 **Glass Recycling Guide**\n\n"
                "• Glass is **100% recyclable** with zero quality loss — infinitely!\n"
                "• Rinse jars and bottles, remove metal caps\n"
                "• Separate by color if required (clear, green, brown)\n"
                "• Do NOT include ceramics, mirrors, or Pyrex (different melting points)\n"
                "• Decomposition time if landfilled: **1 million years**\n\n"
                "💡 **Tip:** Reuse glass jars for food storage before recycling."
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
                "🔩 **Metal Recycling Guide**\n\n"
                "• Rinse food cans and crush to save bin space\n"
                "• Aluminum is the most valuable recyclable — saves **95% energy**\n"
                "• Use a magnet to sort: sticks = steel, doesn't stick = aluminum\n"
                "• Include clean aluminum foil and trays\n"
                "• Scrap metal can be sold to dedicated recyclers\n\n"
                "💡 **Tip:** Recycling one aluminum can saves enough energy to run a TV for 3 hours!"
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
                "🌱 **Organic Waste & Composting Guide**\n\n"
                "• Food scraps, peels, coffee grounds, eggshells → compost!\n"
                "• Yard waste (leaves, grass, branches) → compost or green bin\n"
                "• Home composting produces free, nutrient-rich fertilizer\n"
                "• **Avoid** composting meat, dairy, and oily foods at home\n"
                "• Decomposition time: **2–4 weeks** (fastest of all categories)\n\n"
                "💡 **Tip:** Composting diverts organics from landfills and reduces methane emissions by 50%!"
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
                "⚡ **E-Waste Disposal Guide**\n\n"
                "• **NEVER** place e-waste in regular trash — contains toxic heavy metals!\n"
                "• Locate certified e-waste recyclers (e-Stewards, R2 certified)\n"
                "• Remove batteries separately and recycle at drop-off points\n"
                "• Wipe personal data before recycling devices\n"
                "• Check manufacturer take-back programs (Apple, Dell, HP, Samsung)\n\n"
                "⚠️ **Warning:** E-waste contains lead, mercury, and cadmium that cause "
                "severe soil and groundwater contamination."
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
                "♻️ **Top 10 Recycling Tips**\n\n"
                "1. **Rinse** containers before recycling\n"
                "2. **Flatten** cardboard and boxes\n"
                "3. **Check** local guidelines — rules vary by area\n"
                "4. **Remove** caps and lids when required\n"
                "5. **Keep it clean** — contamination ruins entire batches\n"
                "6. **No plastic bags** in curbside bins\n"
                "7. **Separate** glass by color if required\n"
                "8. **Compost** food waste instead of trashing it\n"
                "9. **E-waste** goes to specialized facilities only\n"
                "10. **Reduce & Reuse** before recycling!"
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
                "🌍 **Environmental Awareness**\n\n"
                "• Waste management accounts for ~5% of global greenhouse gas emissions\n"
                "• Landfills produce methane — 25x more potent than CO₂\n"
                "• Recycling aluminum saves 95% of energy vs. raw production\n"
                "• The Great Pacific Garbage Patch is now 3x the size of France\n"
                "• By 2050, oceans may contain more plastic than fish by weight\n\n"
                "🌱 **What YOU can do:**\n"
                "• Reduce consumption\n"
                "• Choose reusable products\n"
                "• Sort waste properly\n"
                "• Support circular economy initiatives"
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
                "📉 **Waste Reduction Strategies**\n\n"
                "1. **Refuse** what you don't need\n"
                "2. **Reduce** what you use\n"
                "3. **Reuse** before discarding\n"
                "4. **Repurpose** items creatively\n"
                "5. **Recycle** what you can't reuse\n"
                "6. **Rot** (compost) organic waste\n\n"
                "💡 The most sustainable waste is the waste never created!"
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
                "⏳ **Decomposition Timeline**\n\n"
                "| Material | Time |\n"
                "|---|---|\n"
                "| Organic waste | 2–4 weeks |\n"
                "| Paper | 2–6 weeks |\n"
                "| Cotton cloth | 1–5 months |\n"
                "| Tin can | 50 years |\n"
                "| Aluminum can | 200 years |\n"
                "| Plastic bottle | 450 years |\n"
                "| Glass bottle | 1 million years |\n"
                "| Styrofoam | Never |\n"
                "| E-waste | Never (toxic) |"
            ),
        ],
    },
}

# Fallback responses when no pattern matches
FALLBACK_RESPONSES = [
    (
        "🤔 I'm not sure about that specific topic. I can help with:\n\n"
        "• **Plastic, Paper, Glass, Metal, Organic waste, E-waste** disposal\n"
        "• **Recycling tips** and guides\n"
        "• **Environmental impact** information\n"
        "• **Decomposition timelines**\n"
        "• **Waste reduction** strategies\n\n"
        "Try asking something like: *'How do I recycle plastic bottles?'*"
    ),
    (
        "I specialize in waste management and recycling. Try asking me:\n\n"
        "• *'What should I do with e-waste?'*\n"
        "• *'How long does plastic take to decompose?'*\n"
        "• *'Give me recycling tips'*\n"
        "• *'Tell me about environmental impact'*"
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

        if gemini_api_key and gemini_api_key != "YOUR_GEMINI_API_KEY_HERE":
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
                self.gemini_available = True
            except Exception:
                self.gemini_available = False

    def get_response(self, user_message: str) -> str:
        """
        Process user message and return an appropriate response.

        Uses rule-based matching first. Falls back to Gemini API
        for complex queries if available.

        Parameters
        ----------
        user_message : str
            The user's input message.

        Returns
        -------
        str : The chatbot's response (Markdown formatted).
        """
        if not user_message or not user_message.strip():
            return "Please type a message to get started! 😊"

        msg = user_message.strip().lower()

        # 1. Check greetings
        for pattern in GREETING_PATTERNS:
            if re.search(pattern, msg, re.IGNORECASE):
                return random.choice(GREETING_RESPONSES)

        # 2. Check farewells
        for pattern in FAREWELL_PATTERNS:
            if re.search(pattern, msg, re.IGNORECASE):
                return random.choice(FAREWELL_RESPONSES)

        # 3. Check waste-specific knowledge
        for category, data in WASTE_KNOWLEDGE.items():
            for pattern in data["patterns"]:
                if re.search(pattern, msg, re.IGNORECASE):
                    return random.choice(data["responses"])

        # 4. Check general knowledge
        for topic, data in GENERAL_KNOWLEDGE.items():
            for pattern in data["patterns"]:
                if re.search(pattern, msg, re.IGNORECASE):
                    return random.choice(data["responses"])

        # 5. Try Gemini API for unmatched queries
        if self.gemini_available and self.gemini_model:
            return self._ask_gemini(user_message)

        # 6. Fallback
        return random.choice(FALLBACK_RESPONSES)

    def _ask_gemini(self, user_message: str) -> str:
        """Forward query to Gemini API with waste management context."""
        try:
            system_prompt = (
                "You are an expert waste management and recycling consultant. "
                "Provide helpful, concise, and actionable advice about waste "
                "disposal, recycling methods, environmental impact, and "
                f"sustainability. User asks: {user_message}"
            )
            response = self.gemini_model.generate_content(system_prompt)
            return response.text
        except Exception as e:
            return f"⚠️ AI service temporarily unavailable. Error: {e}\n\n" + random.choice(FALLBACK_RESPONSES)

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
