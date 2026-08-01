# Changelog

## Unreleased

- Added hybrid retrieval: a from-scratch `BM25` sparse retriever fused with dense vector
  search via `reciprocal_rank_fusion` (Cormack, Clarke & Buettcher, 2009). Grounded in
  2025–2026 RAG survey/benchmark findings that hybrid + fusion is the highest-ROI upgrade
  over single-method retrieval. Available via CLI `--retrieval hybrid` and a Streamlit
  retrieval-mode selector.
- Added coverage reporting to CI (`pytest-cov`) and a CI status badge in the README.
- Added an optional `respect_word_boundaries` chunking mode (CLI `--respect-word-boundaries`,
  Streamlit checkbox) so chunk edges land on whitespace instead of splitting a word in half.
- Added `ruff` linting with a dedicated CI job; fixed the issues it caught (unused import,
  unsorted imports).
- Pinned minimum versions in `requirements.txt` instead of leaving dependencies unbounded.
- Fixed `HashingEmbedder` using Python's per-process-randomized `hash()`, which made the
  offline fallback embedder non-deterministic across runs; switched to a stable md5-based hash.
- Added a `pytest` suite (`tests/`) covering chunking, normalization, the vector store, and
  prompt/answer construction, plus a GitHub Actions workflow to run it on push/PR.
