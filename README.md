# My RAG Learning Project (From Scratch)

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

- Chunking with overlap
- Embedders:
  - `hashing` (offline fallback)
  - `sentence-transformers` (local model)
  - `openai` (API)
- Vector stores:
  - `numpy` (manual cosine similarity)
  - `faiss` (fast ANN-style retrieval)
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
- optional LLM generation

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

---

## Sample learning insights

- **Overlap matters** because semantic boundaries rarely match fixed chunk boundaries.
- **Cosine similarity** works best when embeddings are L2-normalized.
- **Grounded prompts** are critical to reduce hallucinations.
- Retrieval quality strongly impacts final answer quality.

---

## LinkedIn post idea

I built a RAG project from scratch in Python to learn how retrieval actually works under the hood.

Instead of using high-level frameworks, I implemented chunking, embeddings, manual cosine similarity search, optional FAISS, and grounded prompt injection myself.

This gave me a much better understanding of how to reduce hallucinations and produce evidence-based answers.

#RAG #LLM #Python #NLP #MLOps #GenerativeAI
