"""
Eco Lifestyle Agent — LLM Interface
--------------------------------------
Handles communication with:
  1. Google Gemini (gemini-1.5-flash) — primary
  2. OpenAI GPT-4o-mini — fallback
  3. Groq (llama-3.3-70b) — additional option

If no API key is configured, falls back to a rule-based response
so the demo is always usable.
"""

import os
from typing import Optional

# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are EcoBot, a friendly and knowledgeable Eco Lifestyle Agent. 
Your mission is to empower people to adopt a greener, more sustainable lifestyle 
through practical, actionable, and science-backed guidance.

You have access to a curated knowledge base covering:
- Sustainable living tips (plastic reduction, energy saving, water conservation, etc.)
- Eco-friendly product recommendations with brands and impact data
- Recycling guidelines and waste management rules
- Government schemes, subsidies, and programs for green living
- Eco-friendly travel options with carbon footprint data

GUIDELINES:
1. Always be encouraging, positive, and non-judgmental.
2. Provide SPECIFIC, ACTIONABLE advice — not vague generalities.
3. When recommending products, include brand names and tangible impact figures.
4. When mentioning government schemes, include application steps and websites.
5. Use simple language — avoid jargon unless you explain it.
6. Structure longer answers with bullet points or numbered lists.
7. End each response with ONE small, immediate action the user can take today.
8. When context is provided, base your answer primarily on it. You may supplement with 
   general knowledge, but clearly ground your main points in the retrieved context.
9. If you don't know something specific to the user's location, say so and provide 
   a general answer with tips on how to find local information.

Your tone: warm, knowledgeable, like a helpful friend who happens to be an environmental expert.
"""

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(user_query: str, context: str) -> str:
    return f"""Based on the following retrieved information from our eco knowledge base, 
please answer the user's question. Use the context to provide accurate, specific guidance.

RETRIEVED CONTEXT:
{context}

USER QUESTION:
{user_query}

Please provide a helpful, structured, and actionable response:"""


# ── LLM Clients ───────────────────────────────────────────────────────────────

def _call_gemini(messages: list, system: str) -> Optional[str]:
    """Call Google Gemini API."""
    try:
        import google.generativeai as genai  # type: ignore
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return None
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=system
        )
        # Build conversation history
        history = []
        for msg in messages[:-1]:
            history.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [msg["content"]]
            })
        chat = model.start_chat(history=history)
        response = chat.send_message(messages[-1]["content"])
        return response.text
    except Exception as e:
        print(f"[Gemini error] {e}")
        return None


def _call_openai(messages: list, system: str) -> Optional[str]:
    """Call OpenAI API."""
    try:
        from openai import OpenAI  # type: ignore
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        client = OpenAI(api_key=api_key)
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[OpenAI error] {e}")
        return None


def _call_groq(messages: list, system: str) -> Optional[str]:
    """Call Groq API."""
    try:
        from groq import Groq  # type: ignore
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        client = Groq(api_key=api_key)
        full_messages = [{"role": "system", "content": system}] + messages
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=full_messages,
            temperature=0.7,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"[Groq error] {e}")
        return None


# ── Rule-based fallback ───────────────────────────────────────────────────────

def _rule_based_response(user_query: str, context: str) -> str:
    """
    Fallback when no LLM API key is available.
    Extracts and formats the most relevant portions of retrieved context.
    """
    query_lower = user_query.lower()
    
    if not context.strip():
        return (
            "I wasn't able to find specific information on that topic in my knowledge base. "
            "Try rephrasing or ask about: plastic reduction, energy saving, recycling, "
            "eco-friendly products, government schemes, or sustainable travel."
        )
    
    # Grab first 1,200 chars of context and wrap it nicely
    snippet = context[:1200].strip()
    
    intro_map = [
        (["plastic", "bottle", "packaging", "wrap"], "Here's how to reduce plastic in your life:"),
        (["recycle", "recycling", "waste", "bin", "dispose"], "Here are the recycling guidelines:"),
        (["solar", "energy", "electric", "power", "LED", "bulb"], "Here are energy-saving tips:"),
        (["travel", "transport", "flight", "train", "car", "bus", "cycle"], "Here are eco-friendly travel options:"),
        (["product", "buy", "purchase", "recommend", "brand"], "Here are eco-friendly product recommendations:"),
        (["scheme", "subsidy", "government", "policy", "grant", "fund"], "Here are relevant government schemes:"),
        (["water", "shower", "faucet", "tap"], "Here are water conservation tips:"),
        (["food", "compost", "garden", "waste"], "Here are sustainable food and composting tips:"),
    ]
    
    intro = "Here's what I found in the eco knowledge base:"
    for keywords, label in intro_map:
        if any(kw in query_lower for kw in keywords):
            intro = label
            break
    
    return (
        f"🌿 **EcoBot** | *Knowledge Base Response*\n\n"
        f"{intro}\n\n"
        f"{snippet}\n\n"
        f"---\n"
        f"*💡 Tip: Configure a GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY "
        f"in your `.env` file to get AI-powered personalised responses!*"
    )


# ── Main agent function ───────────────────────────────────────────────────────

def get_agent_response(
    user_query: str,
    context: str,
    chat_history: Optional[list] = None
) -> str:
    """
    Generate a response from the eco lifestyle agent.
    
    Tries LLM providers in order: Gemini → OpenAI → Groq → Rule-based fallback.
    
    Args:
        user_query:   The user's current question.
        context:      Retrieved knowledge base context (from EcoRAG).
        chat_history: List of previous {role, content} messages for multi-turn conversation.
    
    Returns:
        Agent response string.
    """
    prompt = build_prompt(user_query, context)
    
    # Build message list
    messages = []
    if chat_history:
        for msg in chat_history[-6:]:  # Keep last 3 turns (6 messages)
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": prompt})
    
    # Try each LLM provider
    for provider_fn in [_call_gemini, _call_openai, _call_groq]:
        result = provider_fn(messages, SYSTEM_PROMPT)
        if result:
            return result
    
    # Rule-based fallback
    return _rule_based_response(user_query, context)
