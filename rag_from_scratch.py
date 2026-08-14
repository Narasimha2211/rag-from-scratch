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
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

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


@dataclass
class SourcedChunk:
    """A chunk tagged with where it came from, for citations in the final answer."""

    text: str
    source: str
    index: int  # position of this chunk within its source document (0-based)


def chunk_document(
    path: str | Path,
    chunk_size: int = 500,
    overlap: int = 100,
    respect_word_boundaries: bool = False,
) -> List[SourcedChunk]:
    """
    Reads and chunks a single file, tagging each chunk with its source
    filename and position so retrieved results can be cited (e.g.
    "knowledge_base.txt#3") instead of showing bare, unattributed text.

    Uses the source filename + chunk index rather than character offsets:
    chunk_text() strips and can trim chunk boundaries (respect_word_boundaries),
    so the returned strings don't align 1:1 with raw offsets into the
    original text.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    pieces = chunk_text(text, chunk_size=chunk_size, overlap=overlap, respect_word_boundaries=respect_word_boundaries)
    return [SourcedChunk(text=piece, source=path.name, index=i) for i, piece in enumerate(pieces)]


def resolve_doc_paths(doc: str | Path) -> List[Path]:
    """
    `doc` may point at a single .txt file or a directory containing .txt
    files. A directory is expanded (non-recursively, sorted for a stable
    ingestion order) into all its .txt files, each ingested as a separate
    source so citations still point at the right file.
    """
    path = Path(doc)
    if path.is_dir():
        paths = sorted(path.glob("*.txt"))
        if not paths:
            raise FileNotFoundError(f"No .txt files found in directory: {path}")
        return paths
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    return [path]


def chunk_documents(
    paths: List[Path],
    chunk_size: int = 500,
    overlap: int = 100,
    respect_word_boundaries: bool = False,
) -> List[SourcedChunk]:
    """Chunks every file in `paths` (see resolve_doc_paths), concatenating results in path order."""
    all_chunks: List[SourcedChunk] = []
    for path in paths:
        all_chunks.extend(
            chunk_document(
                path, chunk_size=chunk_size, overlap=overlap, respect_word_boundaries=respect_word_boundaries
            )
        )
    return all_chunks


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
            from sentence_transformers import SentenceTransformer
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
            from openai import OpenAI
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

    def save(self, path: str | Path) -> None:
        """
        Persist the index to disk as `{path}.npz` (the embedding matrix) plus
        `{path}.chunks.json` (the chunk texts), so a slow embedder (API calls,
        a local transformer model) doesn't have to re-embed on every run.

        Deliberately avoids pickling the chunks (`np.savez` with an object
        array would need `allow_pickle=True` to load) -- plain JSON is enough
        for a list of strings and doesn't risk executing arbitrary code from
        an index file.
        """
        if self.matrix is None:
            raise ValueError("Nothing to save -- call add() first")

        base = Path(path)
        np.savez(base.with_suffix(".npz"), matrix=self.matrix)
        base.with_suffix(".chunks.json").write_text(json.dumps(self.chunks), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "NumpyVectorStore":
        base = Path(path)
        npz_path = base.with_suffix(".npz")
        chunks_path = base.with_suffix(".chunks.json")
        if not npz_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"No saved index found at '{path}' (expected {npz_path} and {chunks_path})")

        store = cls()
        with np.load(npz_path) as data:
            store.matrix = data["matrix"]
        store.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        return store


# Optional FAISS variant for scale/performance
class FaissVectorStore:
    def __init__(self) -> None:
        try:
            import faiss
        except Exception as exc:
            raise RuntimeError("faiss is not available. Install with: pip install faiss-cpu") from exc

        self.faiss = faiss
        self.index: Any = None
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
# 3b) SPARSE RETRIEVAL (BM25) + HYBRID FUSION
# -----------------------------
#
# Dense embedding search is weak on exact keyword/entity matches (rare terms,
# IDs, names) because those get diluted into a fixed-size vector. Sparse
# lexical retrieval (BM25) is weak on synonyms/paraphrase. RAG research
# consistently finds that combining both, fused by rank rather than raw
# score, beats either alone:
#   - Cormack, Clarke & Buettcher (2009), "Reciprocal Rank Fusion outperforms
#     Condorcet and Individual Rank Learning Methods" -- source of the RRF
#     formula used below (k=60).
#   - 2025-2026 RAG surveys/benchmarks (e.g. arXiv:2506.00054) report hybrid
#     retrieval + reranking as the highest-ROI upgrade over naive single-method
#     retrieval, and one 2026 benchmark found BM25 alone outperformed a
#     state-of-the-art dense retriever on most metrics for keyword-heavy
#     (financial) documents.


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


class BM25:
    """
    Okapi BM25 sparse (lexical) retriever, implemented from scratch.

    Scores chunks by term overlap with the query, weighting rarer
    (more discriminative) terms higher via inverse document frequency,
    and normalizing for document length so long chunks aren't favored
    just for containing more words.
    """

    def __init__(self, chunks: List[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks = chunks
        self.doc_tokens = [_tokenize(c) for c in chunks]
        self.doc_lengths = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0

        doc_freq: dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        n_docs = len(self.doc_tokens)
        # "+1" idf variant (as used by e.g. Lucene's BM25Similarity) keeps idf
        # non-negative even for terms that appear in most documents.
        self.idf = {term: math.log(1 + (n_docs - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()}

    def score(self, query: str) -> np.ndarray:
        query_terms = _tokenize(query)
        scores = np.zeros(len(self.doc_tokens), dtype=np.float32)

        for i, toks in enumerate(self.doc_tokens):
            if not toks:
                continue

            term_freq: dict[str, int] = {}
            for t in toks:
                term_freq[t] = term_freq.get(t, 0) + 1

            length_norm = 1 - self.b + self.b * (self.doc_lengths[i] / self.avg_doc_length)

            doc_score = 0.0
            for term in query_terms:
                freq = term_freq.get(term)
                if not freq:
                    continue
                idf = self.idf.get(term, 0.0)
                doc_score += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * length_norm)
            scores[i] = doc_score

        return scores

    def search(self, query: str, top_k: int = 3) -> List[RetrievedChunk]:
        scores = self.score(query)
        if len(scores) == 0:
            return []

        candidate_k = min(top_k, len(scores))
        idx = np.argpartition(-scores, kth=candidate_k - 1)[:candidate_k]
        idx = idx[np.argsort(-scores[idx])]

        return [
            RetrievedChunk(chunk_id=int(i), score=float(scores[i]), text=self.chunks[int(i)])
            for i in idx
            if scores[i] > 0.0  # no lexical overlap at all -- not a real match
        ]


def reciprocal_rank_fusion(
    rankings: List[List[RetrievedChunk]],
    top_k: int = 3,
    k: int = 60,
) -> List[RetrievedChunk]:
    """
    Fuse multiple ranked result lists (e.g. dense + BM25) into one ranking.

    Each chunk earns 1 / (k + rank) from each list it appears in (rank is
    1-indexed); scores are summed across lists. Using rank instead of the
    raw similarity/BM25 score avoids having to make the two scales
    comparable, and lets a chunk that ranks well in *either* retriever surface
    near the top -- this is what lets hybrid search catch cases either
    method misses alone.
    """
    fused_scores: dict[int, float] = {}
    chunk_text_by_id: dict[int, str] = {}

    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            fused_scores[item.chunk_id] = fused_scores.get(item.chunk_id, 0.0) + 1.0 / (k + rank)
            chunk_text_by_id[item.chunk_id] = item.text

    ranked_ids = sorted(fused_scores, key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

    return [RetrievedChunk(chunk_id=cid, score=fused_scores[cid], text=chunk_text_by_id[cid]) for cid in ranked_ids]


class HybridRetriever:
    """Combines a dense vector store with BM25 sparse search via Reciprocal Rank Fusion."""

    def __init__(self, vector_store, bm25: BM25) -> None:
        self.vector_store = vector_store
        self.bm25 = bm25

    def search(
        self,
        query: str,
        query_vector: np.ndarray,
        top_k: int = 3,
        fetch_k: int = 10,
    ) -> List[RetrievedChunk]:
        dense_results = self.vector_store.search(query_vector, top_k=fetch_k)
        sparse_results = self.bm25.search(query, top_k=fetch_k)
        return reciprocal_rank_fusion([dense_results, sparse_results], top_k=top_k)


# -----------------------------
# 3c) RERANKING
# -----------------------------
#
# RAG research consistently pairs hybrid retrieval with reranking as the two
# highest-impact additions over naive single-pass retrieval (2025-2026 RAG
# surveys/benchmarks; one 2026 benchmark measured hybrid + rerank at +17.4%
# relative Recall@5 over hybrid alone). A reranker scores each (query, chunk)
# pair jointly, which a bi-encoder embedder can't do since it encodes the
# query and each chunk independently -- that's why rerankers are applied as a
# second pass over a retriever's shortlist rather than as the retriever
# itself (scoring every chunk in a corpus jointly with the query doesn't
# scale).


class BaseReranker:
    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        raise NotImplementedError


class LexicalOverlapReranker(BaseReranker):
    """
    Dependency-free reranker: re-scores a candidate shortlist with BM25
    restricted to just that shortlist. Works anywhere (no extra install),
    and is still a legitimate second pass -- it lets exact term overlap
    correct a shortlist that a dense-only retriever ranked by embedding
    similarity alone.
    """

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        if not candidates:
            return []

        bm25 = BM25([c.text for c in candidates])
        rescored = bm25.search(query, top_k=len(candidates))

        # BM25 chunk_ids here are positions within `candidates`, not the
        # original chunk_ids -- map back before returning.
        return [
            RetrievedChunk(chunk_id=candidates[r.chunk_id].chunk_id, score=r.score, text=r.text)
            for r in rescored
        ][:top_k]


class CrossEncoderReranker(BaseReranker):
    """
    Cross-encoder reranking using sentence-transformers (if installed).

    A cross-encoder scores the (query, chunk) pair through one joint forward
    pass, which is typically far more accurate than cosine similarity of
    independently-computed embeddings, at the cost of being too slow to run
    over an entire corpus -- hence applying it only to a retriever's
    shortlist.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers is not available. Install with: pip install sentence-transformers"
            ) from exc

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievedChunk], top_k: int = 3) -> List[RetrievedChunk]:
        if not candidates:
            return []

        pairs = [(query, c.text) for c in candidates]
        scores = self.model.predict(pairs)

        rescored = [
            RetrievedChunk(chunk_id=c.chunk_id, score=float(s), text=c.text) for c, s in zip(candidates, scores)
        ]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored[:top_k]


def choose_reranker(name: str | None) -> BaseReranker | None:
    if name is None or name == "none":
        return None
    if name == "lexical":
        return LexicalOverlapReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"Unknown reranker: {name}")


# -----------------------------
# 3d) RETRIEVAL EVALUATION METRICS
# -----------------------------
#
# The README/CHANGELOG cite external benchmarks for why hybrid retrieval and
# reranking help; these two metrics let eval_retrieval.py measure recall and
# rank quality on this project's own data instead of only taking those
# claims on faith.


def recall_at_k(retrieved: List[RetrievedChunk], relevant_ids: set[int], k: int | None = None) -> float:
    """
    Fraction of `relevant_ids` (the gold chunk_ids for a query) that appear
    anywhere in the top-k of `retrieved`. Every retriever/reranker in this
    module already returns best-first, so this just checks membership in a
    prefix -- k defaults to the full list (checking recall at whatever depth
    was actually retrieved) but can be set lower to check recall at a
    shallower cutoff.
    """
    if not relevant_ids:
        raise ValueError("relevant_ids must not be empty")

    top = retrieved[:k] if k is not None else retrieved
    hit_ids = {item.chunk_id for item in top}
    return len(hit_ids & relevant_ids) / len(relevant_ids)


def mean_reciprocal_rank(
    rankings: List[List[RetrievedChunk]], relevant_ids_per_query: List[set[int]]
) -> float:
    """
    Mean, across queries, of 1 / (rank of the first relevant chunk_id in
    that query's ranking) -- 0.0 for a query whose ranking contains none of
    its relevant chunk_ids. Rewards surfacing a relevant chunk early, not
    just somewhere in the list, which recall_at_k doesn't distinguish.
    """
    if len(rankings) != len(relevant_ids_per_query):
        raise ValueError("rankings and relevant_ids_per_query must be the same length")
    if not rankings:
        raise ValueError("rankings must not be empty")

    reciprocal_ranks = []
    for ranking, relevant_ids in zip(rankings, relevant_ids_per_query):
        rr = 0.0
        for rank, item in enumerate(ranking, start=1):
            if item.chunk_id in relevant_ids:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


# -----------------------------
# 4) RETRIEVAL + 5) AUGMENTATION
# -----------------------------

def build_grounded_prompt(
    query: str, retrieved: List[RetrievedChunk], sources: dict[int, str] | None = None
) -> str:
    """
    Context injection prompt:
    - Explicitly tells model to only use retrieved context
    - Forces abstention when evidence is missing

    This is a key anti-hallucination control.

    `sources`, if given, maps chunk_id -> a citation label (e.g.
    "knowledge_base.txt#3") so each context block names where it came from.
    Optional and keyed by chunk_id (not positional) since --load-index and
    tests that build RetrievedChunk directly don't always have source
    metadata available.
    """
    context_blocks = []
    for i, item in enumerate(retrieved, start=1):
        citation = f" | source={sources[item.chunk_id]}" if sources and item.chunk_id in sources else ""
        context_blocks.append(f"[Chunk {i} | score={item.score:.4f}{citation}]\n{item.text}")

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
        from openai import OpenAI
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


def build_vector_store(
    vector_store_name: str, chunk_vectors: np.ndarray, chunks: List[str]
) -> "NumpyVectorStore | FaissVectorStore":
    store: NumpyVectorStore | FaissVectorStore
    store = FaissVectorStore() if vector_store_name == "faiss" else NumpyVectorStore()
    store.add(chunk_vectors, chunks)
    return store


@dataclass
class PipelineResult:
    retrieved: List[RetrievedChunk]
    prompt: str
    answer: str


def retrieve_and_answer(
    query: str,
    embedder: BaseEmbedder,
    store: "NumpyVectorStore | FaissVectorStore",
    chunks: List[str],
    top_k: int = 3,
    retrieval: str = "dense",
    rerank: str | None = "none",
    generate_with_llm: bool = False,
    sources: dict[int, str] | None = None,
) -> PipelineResult:
    """
    Runs retrieval through generation (pipeline steps 4-5) against an
    already-built store. Shared by the CLI and the Streamlit app so the two
    don't drift out of sync on how reranking/hybrid/generation are wired
    together.

    `sources`, if given, maps chunk_id -> citation label and is threaded
    through to build_grounded_prompt() (see there for why it's a dict
    keyed by chunk_id rather than a positional list).
    """
    query_vector = embedder.encode([query])[0]
    reranker = choose_reranker(rerank)
    # When reranking, fetch a broader shortlist first so the reranker has
    # something worth rescoring instead of just re-sorting the final top-k.
    retrieval_k = max(10, top_k * 3) if reranker else top_k

    if retrieval == "hybrid":
        retriever = HybridRetriever(store, BM25(chunks))
        retrieved = retriever.search(query, query_vector, top_k=retrieval_k, fetch_k=max(10, retrieval_k * 3))
    else:
        retrieved = store.search(query_vector, top_k=retrieval_k)

    if reranker:
        retrieved = reranker.rerank(query, retrieved, top_k=top_k)

    prompt = build_grounded_prompt(query, retrieved, sources=sources)
    answer = generate_with_openai(prompt) if generate_with_llm else simple_grounded_answer(query, retrieved)

    return PipelineResult(retrieved=retrieved, prompt=prompt, answer=answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="My from-scratch RAG learning pipeline.")
    parser.add_argument(
        "--doc",
        type=str,
        default="data/knowledge_base.txt",
        help="Path to a source .txt file, or a directory of .txt files to ingest together",
    )
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
    parser.add_argument(
        "--retrieval",
        choices=["dense", "hybrid"],
        default="dense",
        help="'hybrid' fuses dense vector search with BM25 keyword search via Reciprocal Rank Fusion",
    )
    parser.add_argument(
        "--rerank",
        choices=["none", "lexical", "cross-encoder"],
        default="none",
        help="Rescore a broader candidate shortlist before taking the final top-k",
    )
    parser.add_argument(
        "--save-index",
        type=str,
        default=None,
        help="Save the built NumpyVectorStore to this path (as PATH.npz + PATH.chunks.json)",
    )
    parser.add_argument(
        "--load-index",
        type=str,
        default=None,
        help="Load a previously saved NumpyVectorStore instead of re-chunking/re-embedding --doc",
    )
    args = parser.parse_args()

    embedder = choose_embedder(args.embedder)
    store: FaissVectorStore | NumpyVectorStore
    sources: dict[int, str] | None = None

    if args.load_index:
        if args.vector_store == "faiss":
            raise ValueError("--load-index only supports --vector-store numpy")
        store = NumpyVectorStore.load(args.load_index)
        chunks = store.chunks
        source_label = f"loaded index: {args.load_index}"
        # A saved index doesn't persist per-chunk source metadata, so citations
        # aren't available for a loaded index -- sources stays None.
    else:
        doc_paths = resolve_doc_paths(args.doc)
        source_label = args.doc if len(doc_paths) > 1 else str(doc_paths[0])

        # 1) Chunking (tagged with source filename + position for citations)
        sourced_chunks = chunk_documents(
            doc_paths,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            respect_word_boundaries=args.respect_word_boundaries,
        )
        chunks = [c.text for c in sourced_chunks]
        sources = {i: f"{c.source}#{c.index}" for i, c in enumerate(sourced_chunks)}

        # 2) Embeddings
        chunk_vectors = embedder.encode(chunks)

        # 3) Store vectors
        store = build_vector_store(args.vector_store, chunk_vectors, chunks)

        if args.save_index:
            if not isinstance(store, NumpyVectorStore):
                raise ValueError("--save-index only supports --vector-store numpy")
            store.save(args.save_index)

    # 4) Retrieval + 5) Augmentation + generation
    result = retrieve_and_answer(
        args.query,
        embedder,
        store,
        chunks,
        top_k=args.top_k,
        retrieval=args.retrieval,
        rerank=args.rerank,
        generate_with_llm=args.generate_with_llm,
        sources=sources,
    )
    retrieved, prompt, answer = result.retrieved, result.prompt, result.answer

    print("\n=== MY RAG LEARNING PIPELINE ===")
    print(f"Source: {source_label}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Embedder: {args.embedder}")
    print(f"Vector store: {args.vector_store}")
    print(f"Retrieval: {args.retrieval}")
    print(f"Reranker: {args.rerank}")

    print("\n--- Top Retrieved Chunks ---")
    for item in retrieved:
        preview = item.text.replace("\n", " ")[:180]
        citation = f"  source={sources[item.chunk_id]}" if sources and item.chunk_id in sources else ""
        print(f"- id={item.chunk_id:02d}  score={item.score:.4f}{citation}  text={preview}...")

    print("\n--- Grounded Prompt (Context Injection) ---")
    print(prompt)

    print("\n--- Final Answer ---")
    print(answer)


if __name__ == "__main__":
    main()
