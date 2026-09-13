# 🌿 Eco Lifestyle Agent (RAG-Powered)

An AI-powered sustainability assistant that uses **Retrieval-Augmented Generation (RAG)** to answer natural language questions about eco-friendly living. Built with Python, Streamlit, and a custom TF-IDF RAG engine — no vector database required.

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔍 **RAG Retrieval** | TF-IDF cosine similarity over a curated eco knowledge base |
| 🤖 **Multi-LLM Support** | Google Gemini · OpenAI GPT-4o-mini · Groq LLaMA-3 |
| 📖 **Rich Knowledge Base** | 4 topic areas, 450+ knowledge entries |
| 💬 **Multi-turn Chat** | Maintains conversation context across turns |
| 🔌 **Works Without API Key** | Rule-based fallback delivers answers from raw KB |
| 🎨 **Beautiful UI** | Green-themed Streamlit chat interface |

---

## 📚 Knowledge Base Topics

1. **Sustainable Living Tips** — Plastic reduction, energy saving, water conservation, food waste, composting
2. **Eco-Friendly Products** — Kitchen, bathroom, cleaning, fashion — with brand names and impact data
3. **Recycling Guidelines** — What to recycle, special disposal, recycling symbols explained
4. **Government Schemes** — India (UJALA, FAME, PM Surya Ghar), USA (IRA credits), UK, EU, Australia
5. **Eco Travel** — Carbon footprint comparison, sustainable transport options, ecotourism

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
cd eco-lifestyle-agent
pip install -r requirements.txt
```

### 2. Set your API key (optional but recommended)
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY (free at aistudio.google.com)
```

### 3. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🔑 Getting a Free API Key

| Provider | Free Tier | Link |
|---|---|---|
| Google Gemini | ✅ Yes (generous) | https://aistudio.google.com |
| Groq (LLaMA 3) | ✅ Yes | https://console.groq.com |
| OpenAI | Limited | https://platform.openai.com |

> **No API key?** The app still works using rule-based extraction from the knowledge base.

---

## 🏗️ Architecture

```
eco-lifestyle-agent/
├── app.py                      ← Streamlit UI
├── agent.py                    ← LLM integration (Gemini/OpenAI/Groq + fallback)
├── rag/
│   └── rag_engine.py           ← TF-IDF RAG: load → chunk → index → retrieve
├── knowledge_base/
│   ├── sustainable_living_tips.txt
│   ├── eco_products.txt
│   ├── recycling_guidelines.txt
│   └── government_schemes.txt
├── test_agent.py               ← Validation tests (no API key needed)
├── requirements.txt
└── .env.example
```

### RAG Pipeline

```
User Query
    ↓
TF-IDF Vectorisation (query)
    ↓
Cosine Similarity against all KB chunks
    ↓
Top-5 Chunks retrieved
    ↓
Context assembled (≤3,000 chars)
    ↓
LLM prompt = System + Context + Query
    ↓
EcoBot response
```

---

## 💬 Example Questions

- *"How can I reduce plastic use at home?"*
- *"What are eco-friendly travel options in my city?"*
- *"What government schemes help with solar panel installation in India?"*
- *"How do I properly recycle e-waste and old batteries?"*
- *"What are the best eco-friendly bathroom products?"*
- *"How do I start composting at home?"*
- *"Tips for saving water in daily life?"*
- *"What is my carbon footprint when I fly?"*

---

## 🧪 Running Tests

```bash
python test_agent.py
```

Expected output:
```
🌿 Eco Lifestyle Agent — Validation Tests
==================================================
[1] Loading RAG engine...
  ✓ Loaded 89 knowledge chunks from 4 files

[2] Testing retrieval relevance...
  ✓ [100% relevance] 'How can I reduce plastic use at home?'
  ✓ [100% relevance] 'What government solar subsidies are available?'
  ...

[3] Testing agent fallback (no API key)...
  ✓ Agent fallback returned 412 character response

✅ All tests passed!
```

---

## 🌍 Environmental Impact

This project itself follows eco principles:
- **No GPU required** — runs on any laptop
- **No cloud vector DB** — zero cloud egress emissions for retrieval
- **Minimal dependencies** — fast install, low resource use

---

*Made with 💚 for a sustainable future*
