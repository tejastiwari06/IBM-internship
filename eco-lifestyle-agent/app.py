"""
Eco Lifestyle Agent — Streamlit Web Application
-------------------------------------------------
Run with:  streamlit run app.py
"""

import os
import sys
import streamlit as st

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_engine import EcoRAG
from agent import get_agent_response

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="EcoBot — Eco Lifestyle Agent",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load env vars from .env if present ───────────────────────────────────────

def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

_load_dotenv()

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Global ── */
body, [data-testid="stAppViewContainer"] {
    background: #f0f7f0;
}

/* ── Header banner ── */
.eco-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem 1.5rem;
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 20px rgba(27,94,32,0.25);
}
.eco-header h1 { margin: 0; font-size: 2.2rem; font-weight: 800; }
.eco-header p  { margin: 0.4rem 0 0; opacity: 0.88; font-size: 1.05rem; }

/* ── Chat bubbles ── */
.chat-user {
    background: #e8f5e9;
    border-left: 4px solid #43a047;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
}
.chat-bot {
    background: #ffffff;
    border-left: 4px solid #1b5e20;
    border-radius: 0 12px 12px 12px;
    padding: 0.85rem 1.1rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.chat-role {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.3rem;
}
.user-role  { color: #2e7d32; }
.bot-role   { color: #1b5e20; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #e8f5e9;
}

/* ── Suggestion chips ── */
.chip-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
}
.chip {
    background: #e8f5e9;
    border: 1.5px solid #66bb6a;
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.85rem;
    color: #1b5e20;
    cursor: pointer;
    font-weight: 500;
}

/* ── Stats ── */
.stat-box {
    background: white;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stat-num { font-size: 1.8rem; font-weight: 800; color: #2e7d32; }
.stat-lbl { font-size: 0.8rem; color: #555; }

/* ── Input area ── */
[data-testid="stTextInput"] input {
    border-radius: 30px;
    border: 2px solid #66bb6a;
    padding: 0.6rem 1.2rem;
}
[data-testid="stTextInput"] input:focus {
    border-color: #2e7d32;
    box-shadow: 0 0 0 3px rgba(46,125,50,0.15);
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "rag_loaded" not in st.session_state:
    st.session_state.rag_loaded = False

# ── Load RAG engine ───────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_rag_engine():
    kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    rag = EcoRAG(kb_dir)
    rag.load()
    return rag

with st.spinner("🌱 Loading eco knowledge base..."):
    rag_engine = load_rag_engine()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌿 EcoBot")
    st.markdown("*Your personal AI sustainability guide*")
    st.divider()
    
    # API Key configuration
    st.markdown("### 🔑 LLM Configuration")
    st.markdown("Configure an API key for AI-powered responses:")
    
    api_provider = st.selectbox(
        "Provider", ["Google Gemini (Free)", "OpenAI GPT-4o-mini", "Groq (Free)"],
        key="provider_select"
    )
    
    api_key_input = st.text_input(
        "API Key", type="password", placeholder="Paste your key here",
        key="api_key_input"
    )
    
    if st.button("Apply Key", use_container_width=True, type="primary"):
        if api_key_input.strip():
            if "Gemini" in api_provider:
                os.environ["GEMINI_API_KEY"] = api_key_input.strip()
            elif "OpenAI" in api_provider:
                os.environ["OPENAI_API_KEY"] = api_key_input.strip()
            elif "Groq" in api_provider:
                os.environ["GROQ_API_KEY"] = api_key_input.strip()
            st.success("✅ API key applied for this session!")
        else:
            st.warning("Please enter a key first.")
    
    # API key links
    with st.expander("Get a free API key"):
        st.markdown(
            "- **Gemini**: [aistudio.google.com](https://aistudio.google.com) *(free tier)*\n"
            "- **Groq**: [console.groq.com](https://console.groq.com) *(free tier)*\n"
            "- **OpenAI**: [platform.openai.com](https://platform.openai.com)"
        )
    
    st.divider()
    
    # Stats
    st.markdown("### 📊 Knowledge Base")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{rag_engine.chunk_count}</div>
            <div class="stat-lbl">Knowledge Chunks</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">4</div>
            <div class="stat-lbl">Topic Areas</div>
        </div>""", unsafe_allow_html=True)
    
    st.divider()
    
    # Topics
    st.markdown("### 📚 Topics Covered")
    topics = [
        "♻️ Recycling Guidelines",
        "🌱 Sustainable Living",
        "🛍️ Eco Products",
        "🚆 Green Travel",
        "🏛️ Govt Schemes",
        "💧 Water Conservation",
        "⚡ Energy Saving",
        "🍃 Composting",
    ]
    for topic in topics:
        st.markdown(f"- {topic}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.markdown(
        "<div style='text-align:center;font-size:0.75rem;color:#666;margin-top:1rem'>"
        "Powered by RAG + LLM<br>🌍 Building a sustainable future</div>",
        unsafe_allow_html=True
    )

# ── Main content ──────────────────────────────────────────────────────────────

# Header
st.markdown("""
<div class="eco-header">
    <h1>🌿 EcoBot — Eco Lifestyle Agent</h1>
    <p>Your AI-powered sustainability guide. Ask me anything about eco-friendly living,
    recycling, green products, government schemes, and sustainable travel.</p>
</div>
""", unsafe_allow_html=True)

# ── Suggested questions ───────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "How can I reduce plastic use at home?",
    "What are eco-friendly travel options?",
    "How do I recycle e-waste properly?",
    "What government solar subsidies are available?",
    "Best eco-friendly products for my bathroom?",
    "How can I reduce my food waste?",
    "Tips for saving water at home?",
    "How to start composting as a beginner?",
]

if not st.session_state.messages:
    st.markdown("### 💬 Ask me anything about sustainable living:")
    st.markdown(
        '<div class="chip-grid">' +
        "".join(f'<span class="chip">💬 {q}</span>' for q in SUGGESTED_QUESTIONS) +
        "</div>",
        unsafe_allow_html=True
    )
    # Use buttons for click-through
    cols = st.columns(4)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        with cols[i % 4]:
            if st.button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state["prefill_query"] = q
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────

chat_container = st.container()
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="chat-user">'
                f'<div class="chat-role user-role">👤 You</div>'
                f'{msg["content"]}'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bot">'
                f'<div class="chat-role bot-role">🌿 EcoBot</div>',
                unsafe_allow_html=True
            )
            st.markdown(msg["content"])
            st.markdown("</div>", unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────

st.divider()

with st.form(key="chat_form", clear_on_submit=True):
    prefill = st.session_state.pop("prefill_query", "")
    user_input = st.text_input(
        "Ask EcoBot:",
        value=prefill,
        placeholder="e.g. How can I reduce my carbon footprint? What recyclables go in the blue bin?",
        label_visibility="collapsed",
    )
    col_a, col_b = st.columns([5, 1])
    with col_b:
        submitted = st.form_submit_button("Send 🌿", use_container_width=True, type="primary")

if submitted and user_input.strip():
    query = user_input.strip()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    
    # RAG retrieval
    with st.spinner("🔍 Searching eco knowledge base..."):
        context = rag_engine.retrieve(query, top_k=5)
    
    # LLM response
    with st.spinner("🌿 EcoBot is thinking..."):
        response = get_agent_response(
            user_query=query,
            context=context,
            chat_history=st.session_state.messages[:-1]
        )
    
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="text-align:center;padding:2rem 0 1rem;color:#888;font-size:0.8rem;border-top:1px solid #ddd;margin-top:2rem">
    🌍 EcoBot — Eco Lifestyle Agent &nbsp;|&nbsp; Powered by RAG (Retrieval-Augmented Generation)<br>
    Making sustainability easy and accessible, one conversation at a time.
</div>
""", unsafe_allow_html=True)
