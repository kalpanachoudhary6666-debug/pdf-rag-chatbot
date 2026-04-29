# PDF Intelligence Chatbot — Self-RAG Agent with LangGraph

An AI chatbot that doesn't just retrieve and answer — it **thinks, judges, rewrites, and verifies**. Upload any PDF, ask a question, and watch the agent evaluate its own reasoning before giving you a grounded, accurate answer.

[![CI](https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-green.svg)](https://github.com/langchain-ai/langgraph)

---

## What makes this different from a normal chatbot

Most RAG chatbots do this: *retrieve chunks → send to LLM → done.*

This project does something smarter. It uses **Self-RAG** — the agent grades every retrieved document, rewrites the query if the results are bad, generates an answer, and then checks whether that answer is actually supported by the document. If it isn't, it tries again. It only shows you an answer it trusts.

---

## How it works — the full loop

```
You ask a question
        │
        ▼
┌─────────────────────────────────────────────────────┐
│                  LangGraph Agent                    │
│                                                     │
│  1. RETRIEVE                                        │
│     Fetch top-6 chunks from FAISS vector store      │
│     using sentence-transformers embeddings          │
│              │                                      │
│              ▼                                      │
│  2. RE-RANK                                         │
│     CrossEncoder scores every chunk against         │
│     your question → keeps only the best 3          │
│              │                                      │
│              ▼                                      │
│  3. GRADE DOCUMENTS                                 │
│     LLM reads each chunk and asks:                  │
│     "Is this actually relevant to the question?"   │
│     → Relevant chunks pass through                 │
│     → Irrelevant chunks are discarded              │
│              │                                      │
│      ┌───────┴────────┐                             │
│      │                │                             │
│  Enough good       Not enough                       │
│  chunks found      good chunks                      │
│      │                │                             │
│      │                ▼                             │
│      │        4. REWRITE QUERY                      │
│      │           LLM rephrases your question        │
│      │           to search more effectively         │
│      │           → loops back to RETRIEVE           │
│      │           (max 2 rewrites, then continues)   │
│      │                                              │
│      ▼                                              │
│  5. GENERATE                                        │
│     LLM writes the final answer using only          │
│     the graded, relevant chunks as context          │
│              │                                      │
│              ▼                                      │
│  6. CHECK FOR HALLUCINATION                         │
│     LLM verifies: "Is every claim in this answer   │
│     supported by the retrieved documents?"          │
│     → Grounded: show answer ✅                      │
│     → Not grounded: rewrite query & try again 🔄   │
│                                                     │
└─────────────────────────────────────────────────────┘
        │
        ▼
Answer shown with:
  - Source chunks (so you can verify)
  - Reasoning trace (every decision the agent made)
  - Grounded badge ✅ or warning ⚠️
```

---

## The two-stage retrieval (why it's more accurate)

Retrieval works in two stages:

**Stage 1 — Bi-encoder (FAISS)**
Fast vector similarity search across all chunks. Returns top-6 candidates. This is approximate but very fast.

**Stage 2 — CrossEncoder (Re-ranker)**
A more powerful model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores each candidate by reading *both* the question and the chunk together — not separately. This gives much better precision. Top-3 are kept.

This combination gives you the speed of FAISS with the accuracy of a full comparison model.

---

## Query rewriting — what happens when retrieval fails

If the grading step finds that fewer than 2 of the retrieved chunks are actually relevant, the agent doesn't just give up or hallucinate. It rewrites your query:

```
Original:  "What was the revenue last year?"
Rewrite 1: "annual revenue financial results earnings"
Rewrite 2: "total income revenue figures reported period"
```

Each rewrite goes through the full retrieve → rerank → grade loop again. After 2 rewrites the agent generates with whatever it has — but the hallucination check still runs, so you always know if the answer is trusted.

---

## Tech stack

| Layer | Technology |
|---|---|
| Agent orchestration | **LangGraph** — stateful graph with conditional routing |
| LLM | **Groq + LLaMA 3 (8B)** — free tier, very fast |
| Vector search | **FAISS** — local, no external service needed |
| Re-ranking | **CrossEncoder** (Hugging Face, runs locally) |
| Embeddings | **sentence-transformers/all-MiniLM-L6-v2** — local |
| UI | **Streamlit** |
| PDF parsing | **PyPDF** |
| Testing | **pytest** with mocked LLM calls |
| CI/CD | **GitHub Actions** |

---

## Project structure

```
pdf-rag-chatbot/
├── app/
│   ├── pdf_processor.py     # loads and chunks the PDF
│   ├── vector_store.py      # builds and queries the FAISS index
│   ├── rag_chain.py         # orchestrates Self-RAG vs simple RAG
│   └── self_rag.py          # LangGraph graph: all 6 nodes + routing logic
├── tests/
│   ├── test_pdf_processor.py
│   ├── test_vector_store.py
│   ├── test_rag_chain.py
│   └── test_self_rag.py     # 14 tests covering every node and edge case
├── .github/workflows/
│   └── ci.yml               # runs lint (ruff) + tests on every push
├── app.py                   # Streamlit UI with reasoning trace panel
└── requirements.txt
```

---

## Running it locally

```bash
git clone https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot
cd pdf-rag-chatbot

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Get a free Groq API key from [console.groq.com](https://console.groq.com) (takes 30 seconds):

```bash
cp .env.example .env
# open .env and set GROQ_API_KEY=your-key-here
```

```bash
streamlit run app.py
```

Upload a PDF, paste your Groq key in the sidebar, and start asking questions.

---

## Running tests

```bash
pytest
```

All LLM calls are mocked — tests run with no API key and no internet connection.

---

## Key concepts

**Self-RAG** — the model evaluates the quality of its own retrieved context and generation, rather than blindly trusting what was fetched.

**CRAG (Corrective RAG)** — when retrieval quality is low, the query is rewritten and retrieval is retried. This is what prevents hallucinations from bad context.

**LangGraph** — builds the agent as a directed graph where each node is a function and edges are conditional. The loop between retrieve → grade → rewrite is just a cycle in the graph with a counter to prevent infinite loops.

**Hallucination detection** — after the answer is generated, a separate LLM call checks whether every factual claim in the answer is directly supported by the retrieved chunks. This is the final safety net.

---

## Live demo

[pdf-rag-chatbot-kalpana.streamlit.app](https://pdf-rag-chatbot-kalpana.streamlit.app/)

Paste your Groq API key in the sidebar → upload any PDF → ask away.
