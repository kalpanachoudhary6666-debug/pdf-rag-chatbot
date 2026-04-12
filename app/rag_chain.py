"""
rag_chain.py
------------
RAG chain using Groq API — free tier, no credit card needed.
Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768

Two modes:
  use_self_rag=True  (default) — Full Self-RAG LangGraph pipeline with:
                                   • CrossEncoder re-ranking
                                   • LLM-based relevance grading
                                   • Automatic query rewriting (up to 2 retries)
                                   • Hallucination checking
  use_self_rag=False            — Simple RAG (fast, fewer Groq calls)

Compatible with Python 3.10+.
"""

import os
from typing import Any, Dict, List, Optional

from groq import Groq
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.self_rag import build_self_rag_graph, get_reranker, run_self_rag


# ─── Simple RAG prompt (fallback mode) ───────────────────────────────────────

_SIMPLE_PROMPT = """You are a helpful assistant that answers questions based strictly \
on the provided PDF document context.

Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say \
"I don't have enough information in the document to answer that."
Do NOT make up answers or use knowledge outside the document.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""


# ─── Build chain ──────────────────────────────────────────────────────────────

def build_rag_chain(
    vector_store: FAISS,
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    use_self_rag: bool = True,
) -> Dict:
    """
    Build a RAG chain using Groq's free LLM API.

    Args:
        vector_store:  Populated FAISS vector store.
        groq_api_key:  Groq API key from console.groq.com (free, no credit card).
        model_name:    Groq model to use.
        temperature:   LLM temperature (0 = deterministic).
        use_self_rag:  If True, uses the full Self-RAG LangGraph pipeline with
                       re-ranking and hallucination checking.
                       If False, uses simple single-pass RAG.

    Returns:
        Chain dict with all components needed by ask_question().
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "Groq API key not provided. Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    # Retrieve more chunks upfront so re-ranker has a good pool to score
    k = 6 if use_self_rag else 4
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )

    # Build Self-RAG graph + load reranker (cached after first load)
    graph = None
    reranker = None
    if use_self_rag:
        reranker = get_reranker()
        graph = build_self_rag_graph(
            retriever=retriever,
            client=client,
            model_name=model_name,
            temperature=temperature,
            reranker=reranker,
        )

    return {
        "client": client,
        "model_name": model_name,
        "temperature": temperature,
        "retriever": retriever,
        "chat_history": [],
        "use_self_rag": use_self_rag,
        "graph": graph,
    }


# ─── Ask question ─────────────────────────────────────────────────────────────

def ask_question(chain: Dict, question: str) -> Dict[str, Any]:
    """
    Ask a question using the RAG chain.

    Routes to Self-RAG (LangGraph) or simple RAG depending on chain config.

    Args:
        chain:    Dict returned by build_rag_chain().
        question: User's question string.

    Returns:
        Dict with keys:
          - answer           (str)
          - source_documents (List[Document])
          - reasoning_steps  (List[str])  ← Self-RAG only, empty in simple mode
          - grounded         (bool)       ← Self-RAG only
          - rewrites_used    (int)        ← Self-RAG only
    """
    if chain.get("use_self_rag") and chain.get("graph") is not None:
        return _ask_self_rag(chain, question)
    return _ask_simple(chain, question)


# ─── Internal: Self-RAG path ──────────────────────────────────────────────────

def _ask_self_rag(chain: Dict, question: str) -> Dict[str, Any]:
    """Run the full LangGraph Self-RAG pipeline."""
    result = run_self_rag(
        compiled_graph=chain["graph"],
        question=question,
        chat_history=list(chain["chat_history"]),
    )

    # Save to memory
    chain["chat_history"].append({"role": "human", "content": question})
    chain["chat_history"].append({"role": "assistant", "content": result["answer"]})

    return result


# ─── Internal: Simple RAG path ────────────────────────────────────────────────

def _ask_simple(chain: Dict, question: str) -> Dict[str, Any]:
    """Single-pass RAG without grading or rewriting."""
    client: Groq = chain["client"]
    model_name: str = chain["model_name"]
    temperature: float = chain["temperature"]
    retriever = chain["retriever"]
    chat_history: List[Dict] = chain["chat_history"]

    source_docs: List[Document] = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)

    history_str = ""
    for msg in chat_history:
        prefix = "Human" if msg["role"] == "human" else "Assistant"
        history_str += f"{prefix}: {msg['content']}\n"

    prompt = _SIMPLE_PROMPT.format(
        context=context,
        chat_history=history_str,
        question=question,
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    answer = response.choices[0].message.content

    chain["chat_history"].append({"role": "human", "content": question})
    chain["chat_history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "source_documents": source_docs,
        "reasoning_steps": [],
        "grounded": True,
        "rewrites_used": 0,
    }
