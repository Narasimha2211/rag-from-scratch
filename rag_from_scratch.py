#!/usr/bin/env python3
"""
My RAG Learning Project (No LangChain/LlamaIndex)

I built this script while learning RAG deeply from first principles.
It demonstrates the full pipeline with minimal abstractions:
1) Document chunking with overlap
2) Embedding generation (local sentence-transformers OR OpenAI API OR fallback)
3) Manual vector store (NumPy) + optional FAISS
4) Top-K retrieval using cosine similarity
5) Context injection prompt for grounded generation

Run examples:
  python rag_from_scratch.py --query "Why do we use overlap in chunking?"
  python rag_from_scratch.py --embedder openai --query "What is cosine similarity?"

Environment variables (optional):
  OPENAI_API_KEY=...           # needed only for --embedder openai or --generate-with-llm
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

# -----------------------------
# 1) DOCUMENT CHUNKING
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 100,
    respect_word_boundaries: bool = False,
) -> List[str]:
    """
    Split text into overlapping character windows.

    WHY overlap helps:
    - Important facts may be split across chunk boundaries.
    - Overlap duplicates boundary content into neighboring chunks,
      improving retrieval recall and reducing missed context.

    Args:
        text: Source document text.
        chunk_size: Number of characters per chunk.
        overlap: Number of characters shared with the next chunk.
        respect_word_boundaries: If True, a chunk that would end mid-word is
            trimmed back to the previous whitespace so words aren't split in
            half (which can otherwise weaken embeddings for boundary words).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        if respect_word_boundaries:
            # A start that lands on whitespace (e.g. right after a trimmed
            # chunk) leaves no earlier boundary to trim back to; skip past it
            # so the next chunk begins on a real word.
            while start < len(text) and text[start].isspace():
                start += 1
            if start >= len(text):
                break

        end = min(start + chunk_size, len(text))
        if respect_word_boundaries and end < len(text) and not text[end].isspace():
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if respect_word_boundaries:
            if end >= len(text):
                # Reached the end of the document; further windows would just be
                # shrinking, redundant suffixes of the tail already captured.
                break
            # Advance from the (possibly trimmed) end, not the fixed step, so the
            # next chunk also starts on a word boundary instead of drifting mid-word.
            start = max(start + 1, end - overlap)
        else:
            start += step

    return chunks


# -----------------------------
# 2) EMBEDDING GENERATION
# -----------------------------

class BaseEmbedder:
    def encode(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(BaseEmbedder):
    """Local embeddings using sentence-transformers (if installed)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is not available. "
                "Install with: pip install sentence-transformers"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.astype(np.float32)


class OpenAIEmbedder(BaseEmbedder):
    """API embeddings using OpenAI (if API key/package available)."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")

        try:
            from openai import OpenAI  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError("openai package not available. Install with: pip install openai") from exc

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def encode(self, texts: List[str]) -> np.ndarray:
        response = self.client.embeddings.create(model=self.model, input=texts)
        vectors = [item.embedding for item in response.data]
        return np.array(vectors, dtype=np.float32)


class HashingEmbedder(BaseEmbedder):
    """
    Lightweight local fallback embedder (dependency-free).

    This is NOT as semantically strong as transformer/API embeddings,
    but keeps the full RAG math visible and runnable anywhere.
    """

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in text.lower().split():
            # Python's built-in hash() is randomized per-process (PYTHONHASHSEED),
            # so the same token would map to different indices across runs.
            # Use a stable hash so embeddings are reproducible.
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % self.dim
            vec[idx] += 1.0
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.vstack([self._embed_one(t) for t in texts]).astype(np.float32)


# -----------------------------
# 3) VECTOR STORE (MANUAL)
# -----------------------------


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


@dataclass
class RetrievedChunk:
    chunk_id: int
    score: float
    text: str


class NumpyVectorStore:
    """
    A simple in-memory vector store:
    - stores normalized embeddings in a matrix (N x D)
    - retrieval uses cosine similarity = dot product (after normalization)
    """

    def __init__(self) -> None:
        self.matrix: np.ndarray | None = None
        self.chunks: List[str] = []

    def add(self, vectors: np.ndarray, chunks: List[str]) -> None:
        if vectors.shape[0] != len(chunks):
            raise ValueError("Number of vectors must match number of chunks")

        vectors = l2_normalize(vectors)
        self.matrix = vectors if self.matrix is None else np.vstack([self.matrix, vectors])
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[RetrievedChunk]:
        if self.matrix is None or len(self.chunks) == 0:
            return []

        query = query_vector.reshape(1, -1)
        query = l2_normalize(query)

        # Cosine similarity when both sides are normalized
        scores = self.matrix @ query.T  # (N, 1)
        scores = scores.ravel()

        top_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        idx = idx[np.argsort(-scores[idx])]

        return [
            RetrievedChunk(chunk_id=int(i), score=float(scores[i]), text=self.chunks[int(i)])
            for i in idx
        ]


# Optional FAISS variant for scale/performance
class FaissVectorStore:
    def __init__(self) -> None:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError("faiss is not available. Install with: pip install faiss-cpu") from exc

        self.faiss = faiss
        self.index = None
        self.chunks: List[str] = []

    def add(self, vectors: np.ndarray, chunks: List[str]) -> None:
        vectors = l2_normalize(vectors).astype(np.float32)
        d = vectors.shape[1]

        if self.index is None:
            # Inner product index + normalized vectors => cosine similarity search
            self.index = self.faiss.IndexFlatIP(d)

        self.index.add(vectors)
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, top_k: int = 3) -> List[RetrievedChunk]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = l2_normalize(query_vector.reshape(1, -1)).astype(np.float32)
        scores, ids = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(chunk_id=int(idx), score=float(score), text=self.chunks[int(idx)]))
        return results


# -----------------------------
# 4) RETRIEVAL + 5) AUGMENTATION
# -----------------------------

def build_grounded_prompt(query: str, retrieved: List[RetrievedChunk]) -> str:
    """
    Context injection prompt:
    - Explicitly tells model to only use retrieved context
    - Forces abstention when evidence is missing

    This is a key anti-hallucination control.
    """
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        context_blocks.append(f"[Chunk {i} | score={item.score:.4f}]\n{item.text}")

    context = "\n\n".join(context_blocks) if context_blocks else "(No context retrieved)"

    return f"""You are a grounded assistant. Use ONLY the context below.
If the answer is not present in the context, reply exactly: "I don't know based on the provided context."
Do not add external facts.

Context:
{context}

Question:
{query}

Answer (grounded in context only):"""


def simple_grounded_answer(query: str, retrieved: List[RetrievedChunk]) -> str:
    """
    Offline fallback generator:
    Selects the best-matching sentence from retrieved context by token overlap.
    """
    if not retrieved:
        return "I don't know based on the provided context."

    stopwords = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "in",
        "on",
        "of",
        "to",
        "for",
        "and",
        "how",
        "does",
        "what",
        "why",
    }

    def tokens(text: str) -> set[str]:
        return {
            t
            for t in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if t and t not in stopwords
        }

    q_tokens = tokens(query)
    best_sentence = ""
    best_score = 0.0

    for item in retrieved:
        sentences = [s.strip() for s in item.text.split(".") if s.strip()]
        for s in sentences:
            s_tokens = tokens(s)
            overlap = len(q_tokens.intersection(s_tokens))
            score = overlap + (0.5 * max(item.score, 0.0))
            if score > best_score:
                best_score = score
                best_sentence = s

    if best_score <= 0.0:
        return "I don't know based on the provided context."
    return best_sentence + "."


def generate_with_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "OPENAI_API_KEY not set. Skipping LLM call."

    try:
        from openai import OpenAI  # type: ignore[import-not-found]
    except Exception:
        return "openai package not available. Install with: pip install openai"

    client = OpenAI(api_key=api_key)
    resp = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        temperature=0.0,
    )
    return resp.output_text.strip()


def choose_embedder(name: str) -> BaseEmbedder:
    if name == "sentence-transformers":
        return SentenceTransformerEmbedder()
    if name == "openai":
        return OpenAIEmbedder()
    if name == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="My from-scratch RAG learning pipeline.")
    parser.add_argument("--doc", type=str, default="data/knowledge_base.txt", help="Path to source text file")
    parser.add_argument("--query", type=str, default="Why do we use overlap in chunking?", help="User query")
    parser.add_argument("--chunk-size", type=int, default=450)
    parser.add_argument("--overlap", type=int, default=90)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--embedder",
        choices=["hashing", "sentence-transformers", "openai"],
        default="hashing",
        help="Embedding backend",
    )
    parser.add_argument("--vector-store", choices=["numpy", "faiss"], default="numpy")
    parser.add_argument("--generate-with-llm", action="store_true", help="Call OpenAI for final generation")
    parser.add_argument(
        "--respect-word-boundaries",
        action="store_true",
        help="Trim chunks back to the previous space instead of splitting a word in half",
    )
    args = parser.parse_args()

    doc_path = Path(args.doc)
    if not doc_path.exists():
        raise FileNotFoundError(f"Document not found: {doc_path}")

    text = doc_path.read_text(encoding="utf-8")

    # 1) Chunking
    chunks = chunk_text(
        text,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        respect_word_boundaries=args.respect_word_boundaries,
    )

    # 2) Embeddings
    embedder = choose_embedder(args.embedder)
    chunk_vectors = embedder.encode(chunks)

    # 3) Store vectors
    if args.vector_store == "faiss":
        store = FaissVectorStore()
    else:
        store = NumpyVectorStore()

    store.add(chunk_vectors, chunks)

    # 4) Retrieval
    query_vector = embedder.encode([args.query])[0]
    retrieved = store.search(query_vector, top_k=args.top_k)

    # 5) Augmentation (context injection)
    prompt = build_grounded_prompt(args.query, retrieved)

    # Generation (LLM optional; fallback answer always available)
    if args.generate_with_llm:
        answer = generate_with_openai(prompt)
    else:
        answer = simple_grounded_answer(args.query, retrieved)

    print("\n=== MY RAG LEARNING PIPELINE ===")
    print(f"Document: {doc_path}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedder: {args.embedder}")
    print(f"Vector store: {args.vector_store}")

    print("\n--- Top Retrieved Chunks ---")
    for item in retrieved:
        preview = item.text.replace("\n", " ")[:180]
        print(f"- id={item.chunk_id:02d}  score={item.score:.4f}  text={preview}...")

    print("\n--- Grounded Prompt (Context Injection) ---")
    print(prompt)

    print("\n--- Final Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
