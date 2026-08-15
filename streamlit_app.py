from pathlib import Path

import streamlit as st

from rag_from_scratch import (
    SourcedChunk,
    build_vector_store,
    choose_embedder,
    chunk_text,
    chunk_text_recursive,
    deduplicate_chunks,
    retrieve_and_answer,
)

st.set_page_config(page_title="My RAG Learning Lab", page_icon="📚", layout="wide")
st.title("📚 My RAG Learning Lab")
st.caption("Built by me to learn RAG deeply: Chunking → Embedding → Retrieval → Context Injection → Grounded Answer")

with st.sidebar:
    st.header("Settings")
    embedder_name = st.selectbox("Embedder", ["hashing", "sentence-transformers", "openai"], index=0)
    vector_store_name = st.selectbox("Vector Store", ["numpy", "faiss"], index=0)
    chunk_strategy = st.selectbox(
        "Chunk Strategy",
        ["fixed", "recursive"],
        index=0,
        help="'recursive' packs whole paragraphs/sentences up to Chunk Size instead of "
        "cutting at a fixed character offset",
    )
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1200, value=450, step=50)
    overlap = st.slider("Overlap", min_value=0, max_value=400, value=90, step=10)
    top_k = st.slider("Top-K", min_value=1, max_value=8, value=3, step=1)
    respect_word_boundaries = st.checkbox(
        "Respect word boundaries when chunking", value=False, disabled=(chunk_strategy == "recursive")
    )
    dedupe = st.checkbox("Deduplicate near-identical chunks", value=False)
    dedupe_threshold = st.slider(
        "Dedupe Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        disabled=not dedupe,
        help="Jaccard token overlap above which two chunks count as duplicates",
    )
    retrieval_mode = st.selectbox(
        "Retrieval",
        ["dense", "hybrid"],
        index=0,
        help="Hybrid fuses dense vector search with BM25 keyword search via Reciprocal Rank Fusion",
    )
    rerank_name = st.selectbox(
        "Reranker",
        ["none", "lexical", "cross-encoder"],
        index=0,
        help="Rescores a broader candidate shortlist before taking the final top-k",
    )
    use_mmr = st.checkbox(
        "Diversify with MMR", value=False, help="Trades relevance off against redundancy in the final top-k"
    )
    mmr_lambda = st.slider(
        "MMR Lambda", min_value=0.0, max_value=1.0, value=0.5, step=0.05, disabled=not use_mmr
    )
    use_llm = st.checkbox("Generate with OpenAI LLM", value=False)

st.subheader("1) Document(s)")
uploads = st.file_uploader("Upload .txt document(s)", type=["txt"], accept_multiple_files=True)
default_doc_path = Path("data/knowledge_base.txt")

if uploads:
    docs = [(u.name, u.read().decode("utf-8", errors="ignore")) for u in uploads]
elif default_doc_path.exists():
    docs = [(default_doc_path.name, default_doc_path.read_text(encoding="utf-8"))]
else:
    docs = []

source_names = ", ".join(name for name, _ in docs) if docs else "No document found"
st.write(f"Using source(s): **{source_names}**")

query = st.text_input("2) Ask a question", value="Why do we use overlap in chunking?")

if st.button("Run RAG", type="primary"):
    if not docs:
        st.error("No document text available. Upload .txt file(s) or create data/knowledge_base.txt.")
        st.stop()

    try:
        # Chunk each document separately so citations ("name#index") stay
        # tied to the right source file instead of blending across files.
        sourced_chunks: list[SourcedChunk] = []
        for name, text in docs:
            if chunk_strategy == "recursive":
                pieces = chunk_text_recursive(text, chunk_size=chunk_size, overlap=overlap)
            else:
                pieces = chunk_text(
                    text,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    respect_word_boundaries=respect_word_boundaries,
                )
            sourced_chunks.extend(SourcedChunk(text=piece, source=name, index=i) for i, piece in enumerate(pieces))

        chunk_count_before_dedupe = len(sourced_chunks)
        if dedupe:
            sourced_chunks = deduplicate_chunks(sourced_chunks, threshold=dedupe_threshold)

        if not sourced_chunks:
            st.error("Document(s) produced no chunks (empty text?).")
            st.stop()

        chunks = [c.text for c in sourced_chunks]
        sources = {i: f"{c.source}#{c.index}" for i, c in enumerate(sourced_chunks)}

        embedder = choose_embedder(embedder_name)
        chunk_vectors = embedder.encode(chunks)
        store = build_vector_store(vector_store_name, chunk_vectors, chunks)

        result = retrieve_and_answer(
            query,
            embedder,
            store,
            chunks,
            top_k=top_k,
            retrieval=retrieval_mode,
            rerank=rerank_name,
            mmr=use_mmr,
            mmr_lambda=mmr_lambda,
            generate_with_llm=use_llm,
            sources=sources,
        )
        retrieved, prompt, answer = result.retrieved, result.prompt, result.answer

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("3) Retrieval Results")
        dedupe_note = f" (deduped {chunk_count_before_dedupe} → {len(chunks)})" if dedupe else ""
        st.write(f"Chunks created: **{len(chunks)}**{dedupe_note}")
        for i, item in enumerate(retrieved, start=1):
            citation = sources.get(item.chunk_id, "unknown")
            st.markdown(f"**Chunk {i}** | score={item.score:.4f} | id={item.chunk_id} | source={citation}")
            st.write(item.text)
            st.divider()

    with c2:
        st.subheader("4) Context Injection Prompt")
        st.code(prompt)

    st.subheader("5) Final Grounded Answer")
    st.success(answer)

st.info(
    "I built this project to compare different RAG choices (embedders + vector stores) and understand grounded generation. "
    "The app reduces hallucination risk by forcing answers to stay inside retrieved evidence."
)
