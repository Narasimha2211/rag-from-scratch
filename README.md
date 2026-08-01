# My RAG Learning Project (From Scratch)

[![tests](https://github.com/Narasimha2211/rag-from-scratch/actions/workflows/tests.yml/badge.svg)](https://github.com/Narasimha2211/rag-from-scratch/actions/workflows/tests.yml)

I built this project to **learn Retrieval-Augmented Generation (RAG) deeply** by implementing each step myself in Python, without high-level orchestration frameworks.

## Why I built this

I wanted to understand the internals of RAG, not just call a library API. So I implemented:

- document chunking with overlap
- embedding generation (multiple backends)
- vector search math (manual NumPy cosine similarity)
- optional FAISS retrieval
- context injection for grounded generation

This helped me understand how RAG reduces hallucinations by grounding the answer in retrieved context.

---

## What this project includes

### 1) CLI pipeline
File: [rag_from_scratch.py](rag_from_scratch.py)

- Chunking with overlap (optionally word-boundary aware)
- Embedders:
  - `hashing` (offline fallback)
  - `sentence-transformers` (local model)
  - `openai` (API)
- Vector stores:
  - `numpy` (manual cosine similarity)
  - `faiss` (fast ANN-style retrieval)
- Retrieval modes:
  - `dense` — vector similarity search only
  - `hybrid` — dense search + a from-scratch BM25 sparse retriever, fused with
    Reciprocal Rank Fusion (`--retrieval hybrid`)
- Prompt building that enforces:
  - “use only provided context”
  - abstain when evidence is missing

### 2) Streamlit app
File: [streamlit_app.py](streamlit_app.py)

Interactive UI to experiment with:
- chunk size
- overlap
- top-k retrieval
- embedder choice
- vector store choice
- retrieval mode (dense vs. hybrid)
- optional LLM generation

---

## Hybrid retrieval (dense + BM25)

Dense embedding search is weak on exact keyword/entity matches (rare terms,
IDs, names) since they get diluted into a fixed-size vector. Sparse lexical
search (BM25) is weak on synonyms and paraphrase. Recent RAG research
consistently finds that combining both outperforms either alone:

- Cormack, Clarke & Buettcher (2009), *Reciprocal Rank Fusion outperforms
  Condorcet and Individual Rank Learning Methods* — source of the RRF fusion
  formula (`score = Σ 1 / (k + rank)`, `k=60`) used here.
- 2025–2026 RAG surveys (e.g. [arXiv:2506.00054](https://arxiv.org/abs/2506.00054))
  and benchmark studies report hybrid retrieval + reranking as the
  highest-ROI upgrade over naive single-method retrieval — one 2026 benchmark
  found BM25 alone even outperformed a state-of-the-art dense retriever on
  most metrics for keyword-heavy documents.

Try it:

```bash
python rag_from_scratch.py --query "Why do we use overlap in chunking?" --retrieval hybrid
```

Both `BM25` and `reciprocal_rank_fusion` are implemented from scratch in
[rag_from_scratch.py](rag_from_scratch.py), no extra dependency required.

---

## Run locally

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run CLI:

```bash
python rag_from_scratch.py --query "Why do we use overlap in chunking?"
```

3. Run Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

4. Run tests (with coverage):

```bash
pip install -r requirements-dev.txt
pytest -v --cov=rag_from_scratch --cov-report=term-missing
```

Tests and linting (`ruff`) run automatically on every push/PR via [GitHub Actions](.github/workflows/tests.yml).

---

## Sample learning insights

- **Overlap matters** because semantic boundaries rarely match fixed chunk boundaries.
- **Cosine similarity** works best when embeddings are L2-normalized.
- **Grounded prompts** are critical to reduce hallucinations.
- Retrieval quality strongly impacts final answer quality.

