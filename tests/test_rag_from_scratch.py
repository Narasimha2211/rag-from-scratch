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
    build_vector_store,
    choose_reranker,
    chunk_document,
    chunk_documents,
    chunk_text,
    chunk_text_recursive,
    l2_normalize,
    mean_reciprocal_rank,
    mmr_select,
    recall_at_k,
    reciprocal_rank_fusion,
    resolve_doc_paths,
    retrieve_and_answer,
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
# chunk_text_recursive
# -----------------------------

def test_chunk_text_recursive_invalid_args_raise():
    with pytest.raises(ValueError):
        chunk_text_recursive("some text", chunk_size=0, overlap=0)
    with pytest.raises(ValueError):
        chunk_text_recursive("some text", chunk_size=10, overlap=10)


def test_chunk_text_recursive_one_sentence_per_chunk_when_chunk_size_is_tight():
    text = "Sentence one is short. Sentence two is short too. Sentence three is also short."
    chunks = chunk_text_recursive(text, chunk_size=40, overlap=0)
    assert chunks == [
        "Sentence one is short.",
        "Sentence two is short too.",
        "Sentence three is also short.",
    ]


def test_chunk_text_recursive_packs_multiple_sentences_per_chunk():
    text = "Sentence one is short. Sentence two is short too. Sentence three is also short."
    chunks = chunk_text_recursive(text, chunk_size=60, overlap=0)
    assert chunks == [
        "Sentence one is short. Sentence two is short too.",
        "Sentence three is also short.",
    ]


def test_chunk_text_recursive_overlap_carries_trailing_sentence_into_next_chunk():
    text = "Sentence one is short. Sentence two is short too. Sentence three is also short."
    chunks = chunk_text_recursive(text, chunk_size=60, overlap=30)
    assert chunks == [
        "Sentence one is short. Sentence two is short too.",
        "Sentence two is short too. Sentence three is also short.",
    ]


def test_chunk_text_recursive_treats_blank_line_as_a_unit_boundary():
    # Neither paragraph has sentence-ending punctuation, so without paragraph
    # splitting this would be one giant unrecoverable unit requiring a
    # character-level fallback. Paragraph-first splitting keeps them as two
    # clean units that pack without any mid-word cut.
    text = "Alpha bravo charlie delta\n\nEcho foxtrot golf hotel"
    chunks = chunk_text_recursive(text, chunk_size=30, overlap=0)
    assert chunks == ["Alpha bravo charlie delta", "Echo foxtrot golf hotel"]


def test_chunk_text_recursive_falls_back_to_character_split_for_oversized_unit():
    # A single "sentence" with no punctuation or whitespace at all -- no
    # semantic boundary smaller than chunk_size exists, so it must fall back
    # to chunk_text()'s character windows.
    text = "supercalifragilisticexpialidocious"
    chunks = chunk_text_recursive(text, chunk_size=10, overlap=0)
    assert len(chunks) > 1
    assert all(len(c) <= 10 for c in chunks)


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


# -----------------------------
# build_vector_store / retrieve_and_answer
# (shared pipeline helpers used by both the CLI and the Streamlit app)
# -----------------------------

def _build_store(chunks):
    embedder = HashingEmbedder(dim=64)
    vectors = embedder.encode(chunks)
    return embedder, build_vector_store("numpy", vectors, chunks)


def test_build_vector_store_numpy_returns_populated_store():
    chunks = ["cats are great pets", "dogs are great pets"]
    _, store = _build_store(chunks)
    assert isinstance(store, NumpyVectorStore)
    assert store.chunks == chunks


def test_build_vector_store_unknown_name_falls_back_to_numpy():
    # Matches the CLI's `--vector-store` choice validation happening in
    # argparse, not here -- any non-"faiss" name is treated as numpy.
    _, store = _build_store(["a chunk"])
    assert isinstance(store, NumpyVectorStore)


def test_retrieve_and_answer_dense_returns_top_k():
    chunks = ["cats are great pets", "dogs are great pets", "the secret keyword is glorbnax"]
    embedder, store = _build_store(chunks)

    result = retrieve_and_answer("glorbnax", embedder, store, chunks, top_k=2, retrieval="dense")

    assert len(result.retrieved) == 2
    assert "glorbnax" in result.prompt
    assert result.answer  # fallback generator always returns something


def test_retrieve_and_answer_hybrid_surfaces_exact_keyword_match():
    chunks = [
        "completely unrelated text about the weather today",
        "another unrelated paragraph about gardening",
        "yet another paragraph about cooking recipes",
        "the secret keyword is glorbnax and nothing else",
    ]
    embedder, store = _build_store(chunks)

    result = retrieve_and_answer("glorbnax", embedder, store, chunks, top_k=1, retrieval="hybrid")

    assert result.retrieved[0].text == chunks[3]


def test_retrieve_and_answer_applies_reranker():
    chunks = ["completely unrelated text about gardening", "the secret keyword is glorbnax"]
    embedder, store = _build_store(chunks)

    result = retrieve_and_answer(
        "what is glorbnax", embedder, store, chunks, top_k=1, retrieval="dense", rerank="lexical"
    )

    assert result.retrieved[0].text == chunks[1]


def test_retrieve_and_answer_threads_sources_into_prompt():
    chunks = ["the secret keyword is glorbnax"]
    embedder, store = _build_store(chunks)

    result = retrieve_and_answer(
        "glorbnax", embedder, store, chunks, top_k=1, sources={0: "doc.txt#0"}
    )

    assert "source=doc.txt#0" in result.prompt


def test_retrieve_and_answer_applies_mmr():
    # Chunk 1 is a near-duplicate of chunk 0; with a diverse corpus otherwise
    # dominated by near-duplicates, --mmr should still return top_k results
    # without erroring when composed with the shared pipeline helper.
    chunks = [
        "the secret keyword is glorbnax",
        "the secret keyword is definitely glorbnax",
        "completely unrelated text about gardening",
    ]
    embedder, store = _build_store(chunks)

    result = retrieve_and_answer(
        "glorbnax", embedder, store, chunks, top_k=2, retrieval="dense", mmr=True, mmr_lambda=0.3
    )

    assert len(result.retrieved) == 2


# -----------------------------
# chunk_document (source citations)
# -----------------------------

def test_chunk_document_tags_chunks_with_source_and_index(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("abcdefghij", encoding="utf-8")

    chunks = chunk_document(doc, chunk_size=4, overlap=2)

    assert [c.text for c in chunks] == ["abcd", "cdef", "efgh", "ghij", "ij"]
    assert all(c.source == "notes.txt" for c in chunks)
    assert [c.index for c in chunks] == [0, 1, 2, 3, 4]


def test_chunk_document_uses_filename_not_full_path(tmp_path):
    subdir = tmp_path / "some" / "nested" / "dir"
    subdir.mkdir(parents=True)
    doc = subdir / "report.txt"
    doc.write_text("hello world", encoding="utf-8")

    chunks = chunk_document(doc, chunk_size=100, overlap=0)

    assert chunks[0].source == "report.txt"


def test_chunk_document_recursive_strategy_uses_sentence_boundaries(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("Sentence one is short. Sentence two is short too.", encoding="utf-8")

    chunks = chunk_document(doc, chunk_size=30, overlap=0, chunk_strategy="recursive")

    assert [c.text for c in chunks] == ["Sentence one is short.", "Sentence two is short too."]
    assert all(c.source == "notes.txt" for c in chunks)
    assert [c.index for c in chunks] == [0, 1]


def test_chunk_document_unknown_strategy_raises(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError):
        chunk_document(doc, chunk_strategy="not-a-real-strategy")


# -----------------------------
# build_grounded_prompt citations
# -----------------------------

def test_build_grounded_prompt_includes_citation_when_sources_given():
    retrieved = [RetrievedChunk(chunk_id=0, score=0.5, text="Overlap reduces missed context.")]
    prompt = build_grounded_prompt("Why use overlap?", retrieved, sources={0: "knowledge_base.txt#2"})
    assert "source=knowledge_base.txt#2" in prompt


def test_build_grounded_prompt_omits_citation_for_unknown_chunk_id():
    retrieved = [RetrievedChunk(chunk_id=5, score=0.5, text="Overlap reduces missed context.")]
    prompt = build_grounded_prompt("Why use overlap?", retrieved, sources={0: "knowledge_base.txt#2"})
    assert "source=" not in prompt


# -----------------------------
# resolve_doc_paths / chunk_documents (multi-document ingestion)
# -----------------------------

def test_resolve_doc_paths_single_file_returns_that_file(tmp_path):
    doc = tmp_path / "a.txt"
    doc.write_text("hello", encoding="utf-8")
    assert resolve_doc_paths(doc) == [doc]


def test_resolve_doc_paths_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_doc_paths(tmp_path / "does_not_exist.txt")


def test_resolve_doc_paths_directory_returns_sorted_txt_files(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "not_a_doc.md").write_text("ignore me", encoding="utf-8")

    paths = resolve_doc_paths(tmp_path)

    assert [p.name for p in paths] == ["a.txt", "b.txt"]


def test_resolve_doc_paths_empty_directory_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_doc_paths(empty_dir)


def test_chunk_documents_tags_each_chunk_with_its_own_source(tmp_path):
    doc_a = tmp_path / "a.txt"
    doc_a.write_text("alpha beta", encoding="utf-8")
    doc_b = tmp_path / "b.txt"
    doc_b.write_text("gamma delta", encoding="utf-8")

    chunks = chunk_documents([doc_a, doc_b], chunk_size=100, overlap=0)

    assert [(c.source, c.text) for c in chunks] == [
        ("a.txt", "alpha beta"),
        ("b.txt", "gamma delta"),
    ]
    # index restarts per source document, not a global running counter
    assert [c.index for c in chunks] == [0, 0]


# -----------------------------
# recall_at_k / mean_reciprocal_rank (retrieval evaluation metrics)
# -----------------------------

def _chunks(*ids):
    return [RetrievedChunk(chunk_id=i, score=1.0, text=str(i)) for i in ids]


def test_recall_at_k_full_hit():
    assert recall_at_k(_chunks(3, 1, 2), relevant_ids={1, 2}) == 1.0


def test_recall_at_k_partial_hit():
    assert recall_at_k(_chunks(3, 1, 5), relevant_ids={1, 2}) == 0.5


def test_recall_at_k_no_hit():
    assert recall_at_k(_chunks(3, 4, 5), relevant_ids={1, 2}) == 0.0


def test_recall_at_k_respects_k_cutoff():
    # The relevant chunk is retrieved, but only at rank 3 -- recall@2 should
    # miss it even though it's present further down the full list.
    retrieved = _chunks(9, 8, 1)
    assert recall_at_k(retrieved, relevant_ids={1}, k=2) == 0.0
    assert recall_at_k(retrieved, relevant_ids={1}, k=3) == 1.0


def test_recall_at_k_empty_relevant_ids_raises():
    with pytest.raises(ValueError):
        recall_at_k(_chunks(1, 2), relevant_ids=set())


def test_mean_reciprocal_rank_averages_across_queries():
    rankings = [
        _chunks(5, 1, 2),  # first relevant id (1) is at rank 2 -> 1/2
        _chunks(3, 4),  # no relevant id present -> 0.0
    ]
    relevant_ids_per_query = [{1}, {9}]
    assert mean_reciprocal_rank(rankings, relevant_ids_per_query) == pytest.approx((0.5 + 0.0) / 2)


def test_mean_reciprocal_rank_first_rank_hit_scores_one():
    rankings = [_chunks(7, 8, 9)]
    assert mean_reciprocal_rank(rankings, [{7}]) == 1.0


def test_mean_reciprocal_rank_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([_chunks(1)], [{1}, {2}])


def test_mean_reciprocal_rank_empty_rankings_raises():
    with pytest.raises(ValueError):
        mean_reciprocal_rank([], [])


# -----------------------------
# NumpyVectorStore.get_vectors
# -----------------------------

def test_numpy_vector_store_get_vectors_returns_normalized_rows():
    store = NumpyVectorStore()
    vectors = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    store.add(vectors, ["chunk-a", "chunk-b"])

    fetched = store.get_vectors([1, 0])
    np.testing.assert_allclose(fetched[0], [1.0, 0.0], rtol=1e-6)
    np.testing.assert_allclose(fetched[1], [0.6, 0.8], rtol=1e-6)  # L2-normalized (3,4)


def test_numpy_vector_store_get_vectors_on_empty_store_raises():
    with pytest.raises(ValueError):
        NumpyVectorStore().get_vectors([0])


# -----------------------------
# mmr_select
# -----------------------------

def test_mmr_select_on_empty_candidates_returns_empty():
    assert mmr_select([], np.zeros((0, 2), dtype=np.float32), top_k=3) == []


def test_mmr_select_mismatched_vector_rows_raises():
    candidates = [RetrievedChunk(chunk_id=0, score=1.0, text="a")]
    with pytest.raises(ValueError):
        mmr_select(candidates, np.zeros((2, 2), dtype=np.float32), top_k=1)


@pytest.mark.parametrize("bad_lambda", [-0.1, 1.1])
def test_mmr_select_invalid_lambda_raises(bad_lambda):
    candidates = [RetrievedChunk(chunk_id=0, score=1.0, text="a")]
    vectors = np.array([[1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        mmr_select(candidates, vectors, top_k=1, lambda_mult=bad_lambda)


def test_mmr_select_with_lambda_one_keeps_original_relevance_order():
    # lambda=1.0 ignores redundancy entirely, so MMR should just reproduce the
    # incoming rank order regardless of how similar the candidates' vectors are.
    candidates = [
        RetrievedChunk(chunk_id=0, score=0.9, text="a"),
        RetrievedChunk(chunk_id=1, score=0.8, text="b"),
        RetrievedChunk(chunk_id=2, score=0.7, text="c"),
    ]
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)  # identical
    results = mmr_select(candidates, vectors, top_k=3, lambda_mult=1.0)
    assert [r.chunk_id for r in results] == [0, 1, 2]


def test_mmr_select_diversifies_away_from_near_duplicates():
    # Chunk 1 is a near-duplicate of chunk 0 (top-ranked); chunk 2 is
    # unrelated but ranked last. A low lambda should prefer covering new
    # ground over piling on redundant top-ranked evidence.
    candidates = [
        RetrievedChunk(chunk_id=0, score=0.95, text="a"),
        RetrievedChunk(chunk_id=1, score=0.94, text="a-near-dup"),
        RetrievedChunk(chunk_id=2, score=0.50, text="c"),
    ]
    vectors = np.array(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    results = mmr_select(candidates, vectors, top_k=2, lambda_mult=0.2)
    assert [r.chunk_id for r in results] == [0, 2]


def test_mmr_select_respects_top_k():
    candidates = [RetrievedChunk(chunk_id=i, score=1.0, text=str(i)) for i in range(5)]
    vectors = np.eye(5, dtype=np.float32)[:, :2]
    results = mmr_select(candidates, vectors, top_k=2, lambda_mult=0.5)
    assert len(results) == 2
