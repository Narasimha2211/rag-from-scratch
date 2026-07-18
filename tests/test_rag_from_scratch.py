import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_from_scratch import (
    HashingEmbedder,
    NumpyVectorStore,
    RetrievedChunk,
    build_grounded_prompt,
    chunk_text,
    l2_normalize,
    simple_grounded_answer,
)

# -----------------------------
# chunk_text
# -----------------------------

def test_chunk_text_basic_overlap():
    text = "abcdefghij"
    chunks = chunk_text(text, chunk_size=4, overlap=2)
    assert chunks == ["abcd", "cdef", "efgh", "ghij", "ij"]


def test_chunk_text_no_overlap():
    text = "abcdefgh"
    chunks = chunk_text(text, chunk_size=4, overlap=0)
    assert chunks == ["abcd", "efgh"]


def test_chunk_text_strips_and_drops_empty_chunks():
    text = "   "
    assert chunk_text(text, chunk_size=4, overlap=0) == []


@pytest.mark.parametrize(
    "chunk_size,overlap",
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_text_invalid_args_raise(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


# -----------------------------
# l2_normalize
# -----------------------------

def test_l2_normalize_unit_length():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    norms = np.linalg.norm(normalized, axis=1)
    np.testing.assert_allclose(norms, [1.0, 1.0], rtol=1e-6)


def test_l2_normalize_handles_zero_vector():
    vectors = np.zeros((1, 3), dtype=np.float32)
    normalized = l2_normalize(vectors)
    assert np.all(np.isfinite(normalized))


# -----------------------------
# HashingEmbedder
# -----------------------------

def test_hashing_embedder_is_deterministic_within_process():
    embedder = HashingEmbedder(dim=64)
    a = embedder.encode(["retrieval augmented generation"])
    b = embedder.encode(["retrieval augmented generation"])
    np.testing.assert_array_equal(a, b)


def test_hashing_embedder_is_deterministic_across_processes():
    # hash() on strings is randomized per-process unless PYTHONHASHSEED is fixed;
    # HashingEmbedder must not depend on it.
    code = (
        "from rag_from_scratch import HashingEmbedder; "
        "e = HashingEmbedder(dim=64); "
        "print(list(e.encode(['overlap matters for retrieval'])[0]))"
    )
    repo_root = str(Path(__file__).resolve().parent.parent)
    results = []
    for seed in ("1", "2"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        results.append(proc.stdout.strip())

    assert results[0] == results[1]


# -----------------------------
# NumpyVectorStore
# -----------------------------

def test_vector_store_search_returns_best_match_first():
    store = NumpyVectorStore()
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.9, 0.1],
        ],
        dtype=np.float32,
    )
    store.add(vectors, ["chunk-a", "chunk-b", "chunk-c"])

    query = np.array([1.0, 0.0], dtype=np.float32)
    results = store.search(query, top_k=2)

    assert len(results) == 2
    assert results[0].text == "chunk-a"
    assert results[0].score >= results[1].score


def test_vector_store_add_rejects_mismatched_lengths():
    store = NumpyVectorStore()
    vectors = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        store.add(vectors, ["only-one-chunk"])


def test_vector_store_search_on_empty_store_returns_empty():
    store = NumpyVectorStore()
    query = np.array([1.0, 0.0], dtype=np.float32)
    assert store.search(query, top_k=3) == []


# -----------------------------
# prompt building / fallback answer
# -----------------------------

def test_build_grounded_prompt_includes_context_and_query():
    retrieved = [RetrievedChunk(chunk_id=0, score=0.5, text="Overlap reduces missed context.")]
    prompt = build_grounded_prompt("Why use overlap?", retrieved)
    assert "Overlap reduces missed context." in prompt
    assert "Why use overlap?" in prompt
    assert "I don't know based on the provided context." in prompt


def test_build_grounded_prompt_with_no_retrieval():
    prompt = build_grounded_prompt("Why use overlap?", [])
    assert "(No context retrieved)" in prompt


def test_simple_grounded_answer_picks_overlapping_sentence():
    retrieved = [
        RetrievedChunk(
            chunk_id=0,
            score=0.9,
            text="Overlap reduces missed context. Cats are unrelated animals.",
        )
    ]
    answer = simple_grounded_answer("Why does overlap reduce missed context?", retrieved)
    assert "overlap" in answer.lower()


def test_simple_grounded_answer_abstains_when_no_evidence():
    assert simple_grounded_answer("anything", []) == "I don't know based on the provided context."
