import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rag_from_scratch import (
    BM25,
    HashingEmbedder,
    HybridRetriever,
    LexicalOverlapReranker,
    NumpyVectorStore,
    RetrievedChunk,
    build_grounded_prompt,
    choose_reranker,
    chunk_text,
    l2_normalize,
    reciprocal_rank_fusion,
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


def test_chunk_text_default_can_split_a_word_in_half():
    text = "hello there friend"
    chunks = chunk_text(text, chunk_size=8, overlap=0)
    assert chunks[0] == "hello th"  # cuts "there" mid-word


def test_chunk_text_respect_word_boundaries_avoids_splitting_words():
    text = "hello there friend"
    chunks = chunk_text(text, chunk_size=8, overlap=0, respect_word_boundaries=True)
    assert chunks == ["hello", "there", "friend"]


def test_chunk_text_respect_word_boundaries_keeps_all_words_intact():
    text = "the quick brown fox jumps over the lazy dog"
    chunks = chunk_text(text, chunk_size=10, overlap=0, respect_word_boundaries=True)
    for chunk in chunks:
        assert all(word in text.split() for word in chunk.split())
    assert " ".join(chunks) == text


def test_chunk_text_respect_word_boundaries_still_splits_a_word_longer_than_chunk_size():
    # No whitespace to trim back to, so a single overlong word is an unavoidable exception.
    text = "supercalifragilistic expialidocious"
    chunks = chunk_text(text, chunk_size=12, overlap=0, respect_word_boundaries=True)
    assert chunks[0] == "supercalifra"


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


def test_vector_store_save_and_load_round_trips(tmp_path):
    store = NumpyVectorStore()
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]], dtype=np.float32)
    store.add(vectors, ["chunk-a", "chunk-b", "chunk-c"])

    index_path = tmp_path / "my_index"
    store.save(index_path)
    loaded = NumpyVectorStore.load(index_path)

    assert loaded.chunks == store.chunks
    np.testing.assert_array_equal(loaded.matrix, store.matrix)

    query = np.array([1.0, 0.0], dtype=np.float32)
    assert [r.text for r in loaded.search(query, top_k=2)] == [r.text for r in store.search(query, top_k=2)]


def test_vector_store_save_without_data_raises(tmp_path):
    store = NumpyVectorStore()
    with pytest.raises(ValueError):
        store.save(tmp_path / "empty_index")


def test_vector_store_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        NumpyVectorStore.load(tmp_path / "does_not_exist")


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


# -----------------------------
# BM25
# -----------------------------

def test_bm25_only_returns_chunks_with_lexical_overlap():
    chunks = [
        "cats are great pets",
        "dogs are great pets",
        "the secret keyword is glorbnax",
    ]
    bm25 = BM25(chunks)
    results = bm25.search("glorbnax", top_k=3)
    assert [r.chunk_id for r in results] == [2]


def test_bm25_idf_weights_rare_terms_higher_than_common_terms():
    chunks = [
        "common word appears here",
        "common word appears here too",
        "common word appears here as well",
        "a rareterm shows up only once",
    ]
    bm25 = BM25(chunks)
    assert bm25.idf["rareterm"] > bm25.idf["common"]


def test_bm25_search_on_no_chunks_returns_empty():
    bm25 = BM25([])
    assert bm25.search("anything", top_k=3) == []


# -----------------------------
# reciprocal_rank_fusion
# -----------------------------

def test_reciprocal_rank_fusion_combines_two_rankings():
    dense = [
        RetrievedChunk(chunk_id=0, score=0.9, text="a"),
        RetrievedChunk(chunk_id=1, score=0.5, text="b"),
    ]
    sparse = [
        RetrievedChunk(chunk_id=1, score=10.0, text="b"),
        RetrievedChunk(chunk_id=2, score=5.0, text="c"),
    ]
    fused = reciprocal_rank_fusion([dense, sparse], top_k=3)
    # chunk 1 ranks in both lists, so it should out-rank chunks that only appear once.
    assert [r.chunk_id for r in fused] == [1, 0, 2]


def test_reciprocal_rank_fusion_respects_top_k():
    dense = [RetrievedChunk(chunk_id=i, score=1.0, text=str(i)) for i in range(5)]
    fused = reciprocal_rank_fusion([dense], top_k=2)
    assert len(fused) == 2


# -----------------------------
# HybridRetriever
# -----------------------------

class _StubDenseStore:
    """A dense retriever stand-in that never surfaces the keyword-match chunk,
    simulating an embedder whose vector space misses an exact/rare term."""

    def __init__(self, ranking):
        self._ranking = ranking

    def search(self, query_vector, top_k=3):
        return self._ranking[:top_k]


def test_hybrid_retriever_surfaces_exact_keyword_match_dense_search_missed():
    chunks = [
        "completely unrelated text about the weather today",
        "another unrelated paragraph about gardening",
        "yet another paragraph about cooking recipes",
        "the secret keyword is glorbnax and nothing else",
    ]
    bm25 = BM25(chunks)
    # Dense search never surfaces chunk 3 at all -- simulates an embedder whose
    # vector space has no signal for the rare term "glorbnax".
    stub_dense = _StubDenseStore(
        [
            RetrievedChunk(chunk_id=0, score=0.9, text=chunks[0]),
            RetrievedChunk(chunk_id=1, score=0.8, text=chunks[1]),
            RetrievedChunk(chunk_id=2, score=0.7, text=chunks[2]),
        ]
    )
    retriever = HybridRetriever(stub_dense, bm25)

    results = retriever.search("glorbnax", query_vector=np.zeros(4, dtype=np.float32), top_k=2, fetch_k=10)

    # Dense-only retrieval (top_k=2 of stub_dense) would never include chunk 3;
    # hybrid must pull it in via the BM25 side despite it ranking last in RRF math
    # among a 3-item dense list.
    assert 3 in [r.chunk_id for r in results]


# -----------------------------
# Reranking
# -----------------------------

def test_lexical_overlap_reranker_reorders_by_query_overlap():
    # Original ranking (e.g. from a weak dense retriever) puts the irrelevant
    # chunk first; the reranker should fix that using lexical overlap.
    candidates = [
        RetrievedChunk(chunk_id=0, score=0.9, text="completely unrelated text about gardening"),
        RetrievedChunk(chunk_id=1, score=0.1, text="the secret keyword is glorbnax"),
    ]
    reranker = LexicalOverlapReranker()
    results = reranker.rerank("what is glorbnax", candidates, top_k=2)
    assert results[0].chunk_id == 1


def test_lexical_overlap_reranker_maps_back_to_original_chunk_ids():
    # candidates[0] has chunk_id=5, candidates[1] has chunk_id=2 -- position in
    # the candidates list must not leak into the returned chunk_id.
    candidates = [
        RetrievedChunk(chunk_id=5, score=0.9, text="apple banana"),
        RetrievedChunk(chunk_id=2, score=0.1, text="the rare keyword is zorptastic"),
    ]
    reranker = LexicalOverlapReranker()
    results = reranker.rerank("zorptastic", candidates, top_k=1)
    assert results[0].chunk_id == 2


def test_lexical_overlap_reranker_on_empty_candidates_returns_empty():
    assert LexicalOverlapReranker().rerank("anything", [], top_k=3) == []


def test_choose_reranker_none_returns_none():
    assert choose_reranker(None) is None
    assert choose_reranker("none") is None


def test_choose_reranker_lexical_returns_lexical_reranker():
    assert isinstance(choose_reranker("lexical"), LexicalOverlapReranker)


def test_choose_reranker_unknown_raises():
    with pytest.raises(ValueError):
        choose_reranker("not-a-real-reranker")
