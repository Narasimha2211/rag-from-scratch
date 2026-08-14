#!/usr/bin/env python3
"""
Measures retrieval quality (recall@k, MRR) on this project's own sample
documents (data/eval_queries.json), across dense / hybrid / hybrid+rerank
configurations. The README/CHANGELOG cite external RAG benchmarks for why
hybrid retrieval and reranking help; this reports a number computed on this
project's own chunking/embedding choices instead of only citing those
benchmarks.

Uses the dependency-free HashingEmbedder so this runs anywhere with no
extra installs (and matches what CI can exercise).

Run:
  python eval_retrieval.py
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_from_scratch import (
    HashingEmbedder,
    build_vector_store,
    chunk_documents,
    mean_reciprocal_rank,
    recall_at_k,
    resolve_doc_paths,
    retrieve_and_answer,
)

EVAL_QUERIES_PATH = Path("data/eval_queries.json")
DOC_DIR = "data"
CHUNK_SIZE = 450
OVERLAP = 90
TOP_K = 5

# (retrieval mode, reranker) pairs to compare.
CONFIGS = [
    ("dense", "none"),
    ("hybrid", "none"),
    ("hybrid", "lexical"),
]


def load_eval_queries() -> list[dict]:
    return json.loads(EVAL_QUERIES_PATH.read_text(encoding="utf-8"))


def main() -> None:
    eval_queries = load_eval_queries()

    doc_paths = resolve_doc_paths(DOC_DIR)
    sourced_chunks = chunk_documents(doc_paths, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
    chunks = [c.text for c in sourced_chunks]
    label_to_id = {f"{c.source}#{c.index}": i for i, c in enumerate(sourced_chunks)}

    embedder = HashingEmbedder()
    store = build_vector_store("numpy", embedder.encode(chunks), chunks)

    print(f"Indexed {len(chunks)} chunks from {len(doc_paths)} document(s) in '{DOC_DIR}/'")
    print(f"Evaluating {len(eval_queries)} queries at top-{TOP_K}\n")

    header = f"{'config':<20} {'recall@' + str(TOP_K):>10} {'MRR':>8}"
    print(header)
    print("-" * len(header))

    for retrieval, rerank in CONFIGS:
        rankings = []
        relevant_ids_per_query = []
        recalls = []

        for item in eval_queries:
            relevant_ids = {label_to_id[label] for label in item["relevant"]}
            result = retrieve_and_answer(
                item["query"], embedder, store, chunks, top_k=TOP_K, retrieval=retrieval, rerank=rerank
            )
            rankings.append(result.retrieved)
            relevant_ids_per_query.append(relevant_ids)
            recalls.append(recall_at_k(result.retrieved, relevant_ids))

        mean_recall = sum(recalls) / len(recalls)
        mrr = mean_reciprocal_rank(rankings, relevant_ids_per_query)

        label = retrieval if rerank == "none" else f"{retrieval}+{rerank}"
        print(f"{label:<20} {mean_recall:>10.3f} {mrr:>8.3f}")


if __name__ == "__main__":
    main()
