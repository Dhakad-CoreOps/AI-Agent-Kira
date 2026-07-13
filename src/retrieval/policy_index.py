"""Local semantic index over the policy documents.

The FAQ agent used to send the entire policy corpus (~50K tokens) with every
question, which Groq's free tier rejects outright (413 Payload Too Large) and
which takes a CPU-only local model many minutes to process. This module fixes
that: documents are chunked and embedded once with a local Ollama embedding
model, and each question retrieves only the handful of most relevant chunks.

Embeddings are computed locally (nomic-embed-text via Ollama) — no API quota
is spent on indexing. The finished index is cached on disk keyed by document
content, so it is only rebuilt when a policy file actually changes. If Ollama
is unavailable, retrieval degrades to keyword-overlap scoring instead of
failing, so the FAQ agent keeps working (just with cruder chunk selection).
"""

import hashlib
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from langchain_ollama import OllamaEmbeddings

from src.agents.base_agent import DATA_DIR
from src.logger import logging

EMBED_MODEL = "nomic-embed-text"
CACHE_PATH = os.path.join(DATA_DIR, ".policy_index_cache.json")

# ~1500 chars ≈ 375 tokens per chunk; top-8 keeps the whole context block
# around 3K tokens — comfortably inside Groq's free-tier request limit.
CHUNK_CHARS = 1500
CHUNK_OVERLAP = 200
TOP_K = 8


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _chunk_text(text: str) -> List[str]:
    """Split on paragraph boundaries into ~CHUNK_CHARS pieces, carrying a
    small overlap so a fact straddling a boundary appears in both chunks."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    current = ""

    for paragraph in paragraphs:
        # A single paragraph larger than a chunk gets hard-split.
        while len(paragraph) > CHUNK_CHARS:
            head, paragraph = paragraph[:CHUNK_CHARS], paragraph[CHUNK_CHARS - CHUNK_OVERLAP :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)

        if len(current) + len(paragraph) + 2 > CHUNK_CHARS and current:
            chunks.append(current)
            current = current[-CHUNK_OVERLAP:] if CHUNK_OVERLAP else ""
        current = f"{current}\n\n{paragraph}" if current else paragraph

    if current:
        chunks.append(current)
    return chunks


def chunk_documents(documents: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    """[(filename, text)] -> [{'source': filename, 'text': chunk}]"""
    return [
        {"source": name, "text": chunk}
        for name, text in documents
        for chunk in _chunk_text(text)
    ]


# ---------------------------------------------------------------------------
# Index build / cache
# ---------------------------------------------------------------------------
def _fingerprint(documents: List[Tuple[str, str]]) -> str:
    h = hashlib.sha1(f"{EMBED_MODEL}|{CHUNK_CHARS}|{CHUNK_OVERLAP}".encode())
    for name, text in sorted(documents):
        h.update(name.encode("utf-8", "replace"))
        h.update(hashlib.sha1(text.encode("utf-8", "replace")).digest())
    return h.hexdigest()


class PolicyIndex:
    """Chunked policy corpus with optional embedding vectors. When vectors are
    missing (Ollama down), retrieve() falls back to keyword scoring."""

    def __init__(self, chunks: List[Dict[str, str]], vectors: Optional[np.ndarray]):
        self.chunks = chunks
        self.vectors = vectors  # shape (n_chunks, dim), L2-normalised, or None
        self._embedder = OllamaEmbeddings(model=EMBED_MODEL)

    # -- scoring ------------------------------------------------------------
    def _semantic_scores(self, query: str) -> Optional[np.ndarray]:
        if self.vectors is None:
            return None
        try:
            q = np.asarray(self._embedder.embed_query(query), dtype=np.float32)
            q /= np.linalg.norm(q) or 1.0
            return self.vectors @ q
        except Exception as e:
            logging.warning(f"Query embedding failed ({e}); using keyword scoring.")
            return None

    def _keyword_scores(self, query: str) -> np.ndarray:
        terms = {w for w in query.lower().split() if len(w) > 3}
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        for i, chunk in enumerate(self.chunks):
            text = chunk["text"].lower()
            scores[i] = sum(text.count(t) for t in terms)
        return scores

    def retrieve(self, query: str, k: int = TOP_K) -> List[Tuple[str, str]]:
        """Top-k (filename, chunk_text) pairs most relevant to the query."""
        if not self.chunks:
            return []
        scores = self._semantic_scores(query)
        if scores is None:
            scores = self._keyword_scores(query)
        order = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i]["source"], self.chunks[i]["text"]) for i in order]


def _load_cache(fingerprint: str, n_chunks: int) -> Optional[np.ndarray]:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("fingerprint") != fingerprint:
            return None
        vectors = np.asarray(cache["vectors"], dtype=np.float32)
        return vectors if len(vectors) == n_chunks else None
    except (OSError, ValueError, KeyError):
        return None


def _save_cache(fingerprint: str, vectors: np.ndarray) -> None:
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "vectors": vectors.tolist()}, f)
    except OSError as e:
        logging.warning(f"Could not persist policy index cache: {e}")


# In-process memo so Streamlit / CLI reuse the index across questions.
_MEMO: Dict[str, PolicyIndex] = {}


def get_policy_index(documents: List[Tuple[str, str]]) -> PolicyIndex:
    """Build (or reuse) the semantic index for these documents. Embeddings are
    cached on disk keyed by document content, so unchanged policies never get
    re-embedded — not even across process restarts."""
    fingerprint = _fingerprint(documents)
    if fingerprint in _MEMO:
        return _MEMO[fingerprint]

    chunks = chunk_documents(documents)
    vectors = _load_cache(fingerprint, len(chunks))

    if vectors is None:
        try:
            logging.info(f"Embedding {len(chunks)} policy chunks with {EMBED_MODEL} (local)")
            raw = OllamaEmbeddings(model=EMBED_MODEL).embed_documents(
                [c["text"] for c in chunks]
            )
            vectors = np.asarray(raw, dtype=np.float32)
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-9)
            _save_cache(fingerprint, vectors)
        except Exception as e:
            logging.warning(
                f"Embedding unavailable ({e}); FAQ retrieval will use keyword scoring. "
                "Is Ollama running with the nomic-embed-text model pulled?"
            )
            vectors = None

    index = PolicyIndex(chunks, vectors)
    _MEMO[fingerprint] = index
    return index
