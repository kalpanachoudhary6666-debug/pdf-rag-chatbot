# PDF Q&A Chatbot

A chatbot that lets you upload any PDF and ask questions about it in plain English. Built using a RAG (Retrieval-Augmented Generation) pipeline — the app finds the most relevant parts of your document and uses Llama 3.3 (via Groq) to generate accurate answers grounded in that content.

[![CI](https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

---

## What I built and why

I wanted to understand how RAG works in practice — not just theoretically. So I built this end-to-end: the PDF gets split into chunks, each chunk gets converted into a vector embedding (locally, no API needed), and those embeddings are stored in a FAISS index. When you ask a question, the most relevant chunks are retrieved and sent to Llama 3.3 along with your question, so the answer is always grounded in the actual document.

The hardest part was figuring out the right chunking strategy — too large and the retrieval gets noisy, too small and you lose context. I settled on 1000-character chunks with 200-character overlap which gave the best results across different PDF types.

---

## How it works

```
You ask a question
        │
        ▼
Streamlit UI (app.py)
        │
        ▼
RAG Chain retrieves the top 4 most relevant chunks from FAISS
        │
        ▼
Chunks + question sent to Llama 3.3 via Groq API
        │
        ▼
Answer shown with source chunks so you can verify it
```

Embeddings run entirely on your machine using `sentence-transformers/all-MiniLM-L6-v2` — so only the answering step uses an API call.

---

## Tech stack

- **LangChain** — document loading, text splitting, FAISS integration
- **Groq API** — free LLM inference using Llama 3.3 70B
- **FAISS** — local vector search
- **sentence-transformers** — local embeddings, no API key needed
- **Streamlit** — the UI
- **PyPDF** — PDF parsing

---

## Running it locally

```bash
git clone https://github.com/kalpanachoudhary6666-debug/pdf-rag-chatbot.git
cd pdf-rag-chatbot

python -m venv venv
source venv/Scripts/activate   # Windows
# source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

Get a free Groq API key from [console.groq.com](https://console.groq.com) — no credit card needed, takes 30 seconds.

```bash
cp .env.example .env
# open .env and set GROQ_API_KEY=your-key-here
```

```bash
streamlit run app.py
```

---

## Project structure

```
pdf-rag-chatbot/
├── app/
│   ├── pdf_processor.py     # loads and chunks the PDF
│   ├── vector_store.py      # builds and queries the FAISS index
│   └── rag_chain.py         # calls Llama 3.3 via Groq with retrieved context
├── tests/                   # unit tests with mocked API calls
├── .github/workflows/       # CI runs tests on every push
├── app.py                   # Streamlit UI
└── requirements.txt
```

---

## Running tests

```bash
pytest
```

All API calls are mocked so tests run without any API key.

---

## Deploying

Deployed on Streamlit Community Cloud. To deploy your own:

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select this repo → main file: `app.py`
3. Under Advanced settings → Secrets, add:
   ```toml
   GROQ_API_KEY = "your-key-here"
   ```
4. Hit Deploy

---

## Resume bullet

> Built a RAG-based PDF Q&A chatbot using LangChain, FAISS, and Llama 3.3 via Groq API; implemented a local embedding pipeline with sentence-transformers and deployed an interactive Streamlit app for real-time document question answering.
