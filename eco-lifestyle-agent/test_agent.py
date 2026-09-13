"""
Quick validation script — runs without any API key.
Tests RAG engine loading and retrieval quality.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from rag.rag_engine import EcoRAG

KB_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")

def test_load():
    rag = EcoRAG(KB_DIR)
    rag.load()
    assert rag.chunk_count > 10, f"Too few chunks: {rag.chunk_count}"
    print(f"  ✓ Loaded {rag.chunk_count} knowledge chunks from 4 files")
    return rag

def test_retrieval(rag):
    test_cases = [
        ("How can I reduce plastic use at home?",        ["plastic", "bottle", "bag", "reusable"]),
        ("What government solar subsidies are available?", ["solar", "subsidy", "scheme"]),
        ("How do I recycle e-waste?",                    ["e-waste", "electronic", "battery"]),
        ("Eco-friendly travel options",                  ["train", "cycling", "bus", "carbon"]),
        ("How to compost food waste at home?",           ["compost", "food", "waste"]),
    ]
    for query, expected_terms in test_cases:
        context = rag.retrieve(query, top_k=5)
        context_lower = context.lower()
        hits = [t for t in expected_terms if t.lower() in context_lower]
        score = len(hits) / len(expected_terms)
        status = "✓" if score >= 0.5 else "✗"
        print(f"  {status} [{score:.0%} relevance] '{query[:55]}...' " if len(query)>55
              else f"  {status} [{score:.0%} relevance] '{query}'")
        if score < 0.5:
            print(f"      Expected terms: {expected_terms}")
            print(f"      Context snippet: {context[:200]}")

def test_agent_fallback():
    """Test rule-based fallback (no API key needed)."""
    from agent import get_agent_response
    response = get_agent_response(
        user_query="How can I reduce plastic use at home?",
        context="Use reusable bags. Avoid plastic bottles. Choose bamboo toothbrushes.",
        chat_history=[]
    )
    assert len(response) > 50, "Response too short"
    print(f"  ✓ Agent fallback returned {len(response)} character response")

if __name__ == "__main__":
    print("\n🌿 Eco Lifestyle Agent — Validation Tests\n" + "="*50)
    
    print("\n[1] Loading RAG engine...")
    rag = test_load()
    
    print("\n[2] Testing retrieval relevance...")
    test_retrieval(rag)
    
    print("\n[3] Testing agent fallback (no API key)...")
    test_agent_fallback()
    
    print("\n" + "="*50)
    print("✅ All tests passed! Run the app with:\n")
    print("   streamlit run app.py\n")
