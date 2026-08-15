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

- Chunking with overlap (`--chunk-strategy`):
  - `fixed` (default) — character windows, optionally word-boundary aware
  - `recursive` — packs whole paragraphs/sentences up to `--chunk-size` instead of
    cutting at an arbitrary character offset
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
- Rerankers (`--rerank`):
  - `none` (default)
  - `lexical` — dependency-free, rescoring the shortlist with BM25
  - `cross-encoder` — optional, `sentence-transformers` CrossEncoder
- Diversity (`--mmr`, `--mmr-lambda`): Maximal Marginal Relevance re-selection so the
  final top-k isn't dominated by several near-duplicate chunks
- Multi-document ingestion — `--doc` accepts either a single `.txt` file or a
  directory of them (all ingested together, e.g. `--doc data`)
- Source citations — every retrieved chunk is tagged with where it came from
  (`file.txt#3`), shown in the CLI output and grounded prompt
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
- reranker (none / lexical / cross-encoder)
- optional LLM generation
- multiple uploaded `.txt` documents at once, each cited by name

---

## Hybrid retrieval + reranking

Dense embedding search is weak on exact keyword/entity matches (rare terms,
IDs, names) since they get diluted into a fixed-size vector. Sparse lexical
search (BM25) is weak on synonyms and paraphrase. Recent RAG research
consistently finds that combining both, then reranking the combined
shortlist, outperforms any single method:

- Cormack, Clarke & Buettcher (2009), *Reciprocal Rank Fusion outperforms
  Condorcet and Individual Rank Learning Methods* — source of the RRF fusion
  formula (`score = Σ 1 / (k + rank)`, `k=60`) used here.
- 2025–2026 RAG surveys (e.g. [arXiv:2506.00054](https://arxiv.org/abs/2506.00054))
  and benchmark studies report hybrid retrieval + reranking as the two
  highest-ROI additions over naive single-method retrieval — one 2026
  benchmark measured +17.4% relative Recall@5 from adding reranking on top of
  hybrid retrieval alone, and found BM25 alone even outperformed a
  state-of-the-art dense retriever on most metrics for keyword-heavy documents.

A reranker scores each `(query, chunk)` pair jointly instead of comparing
independently-computed vectors, which is why it's applied as a second pass
over a retriever's shortlist rather than as the retriever itself.

Try it:

```bash
python rag_from_scratch.py --query "Why do we use overlap in chunking?" --retrieval hybrid --rerank lexical
```

`BM25`, `reciprocal_rank_fusion`, and `LexicalOverlapReranker` are all
implemented from scratch in [rag_from_scratch.py](rag_from_scratch.py) —
no extra dependency required. `CrossEncoderReranker` is optional
(`sentence-transformers`).

---

## Measuring retrieval quality on this project's own data

The external benchmarks above are the reason hybrid + reranking is here at
all, but they're not *this* pipeline, on *this* chunking, on *these*
documents. [eval_retrieval.py](eval_retrieval.py) closes that gap: it builds
one index over `data/` (`HashingEmbedder`, dependency-free) and runs the 10
hand-labeled queries in [data/eval_queries.json](data/eval_queries.json)
through `recall_at_k`/`mean_reciprocal_rank` for each retrieval
configuration.

```bash
python eval_retrieval.py
# or: make eval
```

Measured on the two sample documents (`data/knowledge_base.txt`,
`data/bm25_and_hybrid_search.txt`, 14 chunks total):

| config          | recall@5 | MRR   |
|-----------------|----------|-------|
| dense           | 0.950    | 0.767 |
| hybrid          | 0.950    | 0.900 |
| hybrid+lexical  | 0.950    | 1.000 |

Recall@5 barely moves here because the corpus is tiny (5 of 14 chunks is a
lot of headroom), so it isn't a useful signal at this scale — but MRR shows
the effect clearly: hybrid retrieval and reranking don't just find the right
chunk, they rank it higher, and higher-ranked evidence matters more once
`--top-k` is small in a real deployment. This is a 10-query smoke test
against two short documents, not a rigorous benchmark — treat the numbers as
a sanity check that the pipeline's own retrieval choices behave the way the
research they're based on predicts, not as a substitute for evaluating on
your own documents and queries.

---

## Diversity with MMR

Optimizing purely for relevance can waste a limited top-k on several
near-duplicate chunks that all happen to score well, instead of covering the
query from multiple angles. [Maximal Marginal Relevance](https://www.cs.cmu.edu/~jgc/publication/The_Use_of_MMR_Diversity_Based_LTMIR_1998.pdf)
(Carbonell & Goldstein, 1998) greedily re-selects the final top-k, trading
relevance off against similarity to chunks already picked:

```bash
python rag_from_scratch.py --query "Why do we use overlap in chunking?" --mmr --mmr-lambda 0.3
```

`--mmr-lambda` (default `0.5`) controls the trade-off: `1.0` ignores
redundancy entirely (same order as without `--mmr`); lower values favor
diversity more strongly. It composes with `--retrieval hybrid` and
`--rerank`, since `mmr_select` diversifies whatever candidate order they
produce using the underlying chunk embeddings.

---

## Chunking strategies

`--chunk-strategy fixed` (the default) is easy to reason about but cuts at
an arbitrary character offset — even with `--respect-word-boundaries`, a
window can still land in the middle of a sentence, splitting a fact across
two chunks in a way overlap only partially compensates for.

`--chunk-strategy recursive` splits along a hierarchy of natural
boundaries instead — paragraphs first, then sentences — and greedily packs
the resulting units up to `--chunk-size`, carrying trailing sentences into
the next chunk for `--overlap` the same way the fixed strategy does for
characters. A sentence longer than `--chunk-size` (no smaller boundary left
to respect) falls back to a character-level split. The trade-off is chunk
sizes that vary more, since packing stops at a sentence boundary rather
than exactly at `--chunk-size`:

```bash
python rag_from_scratch.py --doc data --chunk-strategy recursive --chunk-size 300
```

`chunk_text_recursive()` (the standalone splitter) and the paragraph/sentence
unit-packing it's built on are implemented from scratch in
[rag_from_scratch.py](rag_from_scratch.py), same as the rest of this project.

---

## Saving/loading an index

Re-embedding on every run is wasteful once you're using a real embedder
(API calls, a local transformer model). Save a built `numpy` index and
reload it later instead:

```bash
python rag_from_scratch.py --embedder sentence-transformers --save-index my_index
python rag_from_scratch.py --load-index my_index --query "..."
```

This writes `my_index.npz` (the embedding matrix) and
`my_index.chunks.json` (the chunk texts) — plain JSON, not pickle, so
loading your own index file can't execute arbitrary code. `--save-index`
and `--load-index` only support `--vector-store numpy`.

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

Tests, linting (`ruff`), and type checking (`mypy`) run automatically on every push/PR via
[GitHub Actions](.github/workflows/tests.yml).

Or use the `Makefile` shortcuts once dev dependencies are installed:

```bash
make check      # lint + typecheck + test (what CI runs)
make coverage   # test with coverage report
make run        # run the CLI with a sample query
make eval       # measure retrieval quality (recall@k, MRR) on data/eval_queries.json
make streamlit  # launch the Streamlit app
```

---

## Sample learning insights

- **Overlap matters** because semantic boundaries rarely match fixed chunk boundaries.
- **Cosine similarity** works best when embeddings are L2-normalized.
- **Grounded prompts** are critical to reduce hallucinations.
- Retrieval quality strongly impacts final answer quality.

