# Changelog

## Unreleased

- Wired chunk strategy, deduplication, and MMR into the Streamlit app: sidebar controls
  for `chunk_strategy`, `dedupe`/`dedupe_threshold`, and `mmr`/`mmr_lambda` mirroring the
  CLI's equivalent flags. The upload handler now builds `SourcedChunk`s directly and calls
  `chunk_text_recursive`/`deduplicate_chunks` (previously CLI-only), so the two entry
  points no longer diverge on which chunking/dedup features are available.
- Added `--output {text,json}` (default `text`): `json` prints one JSON object (run
  config, retrieved chunks, prompt, answer) instead of the step-by-step trace, for
  scripting/automation. Covered by a subprocess-based CLI test, matching the existing
  pattern used for `HashingEmbedder`'s determinism test.
- Added chunk deduplication (`--dedupe`, `--dedupe-threshold`, default `0.9`):
  `deduplicate_chunks` drops near-duplicate chunks (by token Jaccard overlap, reusing
  BM25's `_tokenize`) before embedding, keeping the first occurrence. Heavy `--overlap`
  or repeated boilerplate across documents could otherwise waste embedding calls and
  crowd a genuinely different chunk out of a limited `--top-k`.
- Added a recursive/paragraph-aware chunking strategy (`--chunk-strategy {fixed,recursive}`,
  default `fixed`): `chunk_text_recursive` splits along a hierarchy of natural boundaries
  (paragraphs, then sentences) instead of a fixed character offset, then greedily packs the
  resulting units up to `--chunk-size` with `--overlap` carried across the boundary the same
  way the fixed strategy does for characters. Falls back to `chunk_text`'s character windows
  for a single unit longer than `--chunk-size`. Wired through `chunk_document`/`chunk_documents`.
- Added Maximal Marginal Relevance (`--mmr`, `--mmr-lambda`): `mmr_select` greedily
  re-selects the final top-k trading relevance off against redundancy (Carbonell &
  Goldstein, 1998), so results aren't dominated by several near-duplicate chunks.
  Relevance is derived from each candidate's incoming rank (like `reciprocal_rank_fusion`,
  to stay agnostic to dense/BM25/cross-encoder score scales); redundancy always uses
  cosine similarity in embedding space via new `NumpyVectorStore.get_vectors` /
  `FaissVectorStore.get_vectors` methods. Wired into `retrieve_and_answer()`, so it composes
  with `--retrieval hybrid` and `--rerank` for both the CLI and Streamlit.
- Added `eval_retrieval.py`, a runnable evaluation harness measuring `recall_at_k` and
  `mean_reciprocal_rank` (both new, with tests) over 10 hand-labeled queries in
  `data/eval_queries.json`, comparing dense / hybrid / hybrid+lexical-rerank side by side
  (`make eval`). Measured on the two sample documents: MRR goes 0.767 (dense) -> 0.900
  (hybrid) -> 1.000 (hybrid+rerank), backing the hybrid/reranking claims elsewhere in this
  changelog with a number computed on this project's own pipeline instead of only external
  benchmarks. See the README's "Measuring retrieval quality" section for the important caveat
  that this is a 10-query smoke test on two short documents, not a rigorous benchmark.
- Added multi-document ingestion: `--doc` now accepts a directory of `.txt` files (in addition
  to a single file), each ingested and chunked separately so citations stay tied to the right
  source (`resolve_doc_paths`, `chunk_documents`). Streamlit's uploader now accepts multiple
  files for the same effect. Added `data/bm25_and_hybrid_search.txt` as a second sample
  document so there's a real second file to ingest.
- Added source citations: chunks are now tagged with where they came from
  (`chunk_document`/`SourcedChunk`, e.g. `knowledge_base.txt#2`) and both the CLI output and
  the grounded prompt show it via `build_grounded_prompt`'s new `sources` parameter. Not
  available when using `--load-index`, since a saved index doesn't persist source metadata.
- Extracted `retrieve_and_answer()`/`build_vector_store()` in `rag_from_scratch.py` so the CLI
  and Streamlit app share one implementation of retrieval->reranking->generation instead of
  two copies that could silently drift apart as retrieval options were added.
- Added mypy static type checking with a dedicated CI job; fixed the issues it caught
  (implicit `None` typing on `FaissVectorStore.index`, a `store` variable inferred from only
  one branch of an if/else) and dropped now-redundant inline `# type: ignore` comments in
  favor of per-module overrides in `pyproject.toml`.
- Added `NumpyVectorStore.save`/`.load` (`--save-index`/`--load-index`) so a built index can
  be persisted to disk instead of re-embedding on every run. Uses plain JSON for the chunk
  texts (not pickle) so loading an index file can't execute arbitrary code.
- Added reranking (`--rerank {none,lexical,cross-encoder}`): `LexicalOverlapReranker`
  (dependency-free, reuses BM25 over the candidate shortlist) and `CrossEncoderReranker`
  (optional, `sentence-transformers`). Completes the "hybrid retrieval + reranking" pairing
  that 2025–2026 RAG research reports as the two highest-ROI additions over naive retrieval.
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
