"""
RAG Engine for Eco Lifestyle Agent
-----------------------------------
Implements a lightweight, dependency-minimal RAG pipeline:
  1. Document loading & chunking from plain-text knowledge base files
  2. TF-IDF based embedding + cosine similarity retrieval (no external vector DB needed)
  3. Context assembly for LLM prompting
"""

import os
import re
import math
from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Document Loading & Chunking
# ---------------------------------------------------------------------------

def load_documents(kb_dir: str) -> List[Dict]:
    """
    Load all .txt files from the knowledge base directory.
    Returns a list of dicts with keys: source, content
    """
    docs = []
    for filename in os.listdir(kb_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(kb_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            docs.append({"source": filename.replace(".txt", ""), "content": content})
    return docs


def chunk_document(doc: Dict, chunk_size: int = 400, overlap: int = 80) -> List[Dict]:
    """
    Split a document into overlapping text chunks.
    Returns a list of chunk dicts: {source, chunk_id, text}
    """
    words = doc["content"].split()
    chunks = []
    start = 0
    chunk_id = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "source": doc["source"],
            "chunk_id": chunk_id,
            "text": chunk_text
        })
        chunk_id += 1
        start += chunk_size - overlap
    return chunks


def build_knowledge_base(kb_dir: str) -> List[Dict]:
    """Load all docs and split into chunks."""
    docs = load_documents(kb_dir)
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc))
    return all_chunks


# ---------------------------------------------------------------------------
# TF-IDF Vectoriser (no sklearn dependency)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> List[str]:
    """Lowercase, remove punctuation, split into tokens."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def build_tfidf_index(chunks: List[Dict]) -> Tuple[List[Dict], Dict[str, float], List[Dict[str, float]]]:
    """
    Build a TF-IDF index for all chunks.
    Returns:
        chunks          - original chunk list
        idf             - dict of term -> IDF value
        tfidf_vectors   - list of {term: tfidf_score} dicts, one per chunk
    """
    N = len(chunks)
    
    # Term frequency per chunk
    tf_list = []
    df = {}  # document frequency
    
    for chunk in chunks:
        tokens = tokenize(chunk["text"])
        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        # Normalise TF
        total = sum(tf.values()) or 1
        tf = {k: v / total for k, v in tf.items()}
        tf_list.append(tf)
        for term in tf:
            df[term] = df.get(term, 0) + 1
    
    # IDF
    idf = {term: math.log(N / (count + 1)) + 1 for term, count in df.items()}
    
    # TF-IDF vectors
    tfidf_vectors = []
    for tf in tf_list:
        tfidf = {term: tf_score * idf.get(term, 0) for term, tf_score in tf.items()}
        tfidf_vectors.append(tfidf)
    
    return chunks, idf, tfidf_vectors


def cosine_similarity(vec_a: Dict[str, float], vec_b: Dict[str, float]) -> float:
    """Compute cosine similarity between two sparse TF-IDF vectors."""
    dot = sum(vec_a.get(t, 0) * vec_b.get(t, 0) for t in vec_b)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def query_tfidf(
    query: str,
    chunks: List[Dict],
    idf: Dict[str, float],
    tfidf_vectors: List[Dict[str, float]],
    top_k: int = 5
) -> List[Dict]:
    """
    Retrieve top-k most relevant chunks for a query using TF-IDF + cosine similarity.
    Returns a list of chunk dicts with an added 'score' field.
    """
    query_tokens = tokenize(query)
    total = len(query_tokens) or 1
    tf_q = {}
    for token in query_tokens:
        tf_q[token] = tf_q.get(token, 0) + 1
    tf_q = {k: v / total for k, v in tf_q.items()}
    tfidf_q = {term: tf_score * idf.get(term, 0) for term, tf_score in tf_q.items()}

    scores = []
    for i, vec in enumerate(tfidf_vectors):
        score = cosine_similarity(tfidf_q, vec)
        scores.append((score, i))

    scores.sort(key=lambda x: -x[0])
    top_chunks = []
    for score, idx in scores[:top_k]:
        chunk = dict(chunks[idx])
        chunk["score"] = round(score, 4)
        top_chunks.append(chunk)
    return top_chunks


# ---------------------------------------------------------------------------
# Context Assembly
# ---------------------------------------------------------------------------

def assemble_context(retrieved_chunks: List[Dict], max_chars: int = 3000) -> str:
    """
    Combine retrieved chunks into a single context string for the LLM prompt.
    Includes source labels and respects a character budget.
    """
    parts = []
    total = 0
    for chunk in retrieved_chunks:
        source_label = chunk["source"].replace("_", " ").title()
        entry = f"[Source: {source_label}]\n{chunk['text']}"
        if total + len(entry) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                entry = entry[:remaining] + "..."
                parts.append(entry)
            break
        parts.append(entry)
        total += len(entry)
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# EcoRAG — Main retrieval class
# ---------------------------------------------------------------------------

class EcoRAG:
    """
    Lightweight RAG retrieval engine for the Eco Lifestyle Agent.
    No external vector database or heavyweight ML library required.
    """

    def __init__(self, kb_dir: str):
        self.kb_dir = kb_dir
        self.chunks: List[Dict] = []
        self.idf: Dict[str, float] = {}
        self.tfidf_vectors: List[Dict[str, float]] = []
        self._loaded = False

    def load(self):
        """Load knowledge base and build TF-IDF index."""
        raw_chunks = build_knowledge_base(self.kb_dir)
        self.chunks, self.idf, self.tfidf_vectors = build_tfidf_index(raw_chunks)
        self._loaded = True

    def retrieve(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve relevant context for a query.
        Returns a formatted context string ready to be injected into an LLM prompt.
        """
        if not self._loaded:
            self.load()
        results = query_tfidf(query, self.chunks, self.idf, self.tfidf_vectors, top_k=top_k)
        return assemble_context(results)

    @property
    def chunk_count(self) -> int:
        return len(self.chunks)
