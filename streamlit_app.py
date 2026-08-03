from pathlib import Path

import streamlit as st

from rag_from_scratch import (
    BM25,
    FaissVectorStore,
    HybridRetriever,
    NumpyVectorStore,
    build_grounded_prompt,
    choose_embedder,
    choose_reranker,
    chunk_text,
    generate_with_openai,
    simple_grounded_answer,
)

st.set_page_config(page_title="My RAG Learning Lab", page_icon="📚", layout="wide")
st.title("📚 My RAG Learning Lab")
st.caption("Built by me to learn RAG deeply: Chunking → Embedding → Retrieval → Context Injection → Grounded Answer")

with st.sidebar:
    st.header("Settings")
    embedder_name = st.selectbox("Embedder", ["hashing", "sentence-transformers", "openai"], index=0)
    vector_store_name = st.selectbox("Vector Store", ["numpy", "faiss"], index=0)
    chunk_size = st.slider("Chunk Size", min_value=200, max_value=1200, value=450, step=50)
    overlap = st.slider("Overlap", min_value=0, max_value=400, value=90, step=10)
    top_k = st.slider("Top-K", min_value=1, max_value=8, value=3, step=1)
    respect_word_boundaries = st.checkbox("Respect word boundaries when chunking", value=False)
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
    use_llm = st.checkbox("Generate with OpenAI LLM", value=False)

st.subheader("1) Document")
upload = st.file_uploader("Upload .txt document", type=["txt"])
default_doc_path = Path("data/knowledge_base.txt")

if upload is not None:
    text = upload.read().decode("utf-8", errors="ignore")
    source_name = upload.name
else:
    if default_doc_path.exists():
        text = default_doc_path.read_text(encoding="utf-8")
        source_name = str(default_doc_path)
    else:
        text = ""
        source_name = "No document found"

st.write(f"Using source: **{source_name}**")

query = st.text_input("2) Ask a question", value="Why do we use overlap in chunking?")

if st.button("Run RAG", type="primary"):
    if not text.strip():
        st.error("No document text available. Upload a .txt file or create data/knowledge_base.txt.")
        st.stop()

    try:
        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            respect_word_boundaries=respect_word_boundaries,
        )
        embedder = choose_embedder(embedder_name)
        chunk_vectors = embedder.encode(chunks)

        store: FaissVectorStore | NumpyVectorStore
        if vector_store_name == "faiss":
            store = FaissVectorStore()
        else:
            store = NumpyVectorStore()

        store.add(chunk_vectors, chunks)
        query_vector = embedder.encode([query])[0]

        reranker = choose_reranker(rerank_name)
        # When reranking, fetch a broader shortlist first so the reranker has
        # something worth rescoring instead of just re-sorting the final top-k.
        retrieval_k = max(10, top_k * 3) if reranker else top_k

        if retrieval_mode == "hybrid":
            retriever = HybridRetriever(store, BM25(chunks))
            retrieved = retriever.search(
                query, query_vector, top_k=retrieval_k, fetch_k=max(10, retrieval_k * 3)
            )
        else:
            retrieved = store.search(query_vector, top_k=retrieval_k)

        if reranker:
            retrieved = reranker.rerank(query, retrieved, top_k=top_k)

        prompt = build_grounded_prompt(query, retrieved)

        if use_llm:
            answer = generate_with_openai(prompt)
        else:
            answer = simple_grounded_answer(query, retrieved)

    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("3) Retrieval Results")
        st.write(f"Chunks created: **{len(chunks)}**")
        for i, item in enumerate(retrieved, start=1):
            st.markdown(f"**Chunk {i}** | score={item.score:.4f} | id={item.chunk_id}")
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
