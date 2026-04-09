"""
rag_chain.py
------------
RAG chain using direct Gemini REST API (v1 stable endpoint).
Bypasses langchain-google-genai to avoid v1beta issues entirely.
Compatible with Python 3.12, 3.13, 3.14.
"""

import os
import requests
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import Optional, Dict, Any, List


PROMPT_TEMPLATE = """You are a helpful assistant that answers questions based strictly on the provided PDF document context.

Use the following pieces of context to answer the question at the end.
If you don't know the answer based on the context, say "I don't have enough information in the document to answer that."
Do NOT make up answers or use knowledge outside the document.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""


def _call_gemini(api_key: str, prompt: str, model: str = "gemini-2.0-flash-lite") -> str:
    """
    Call Gemini using the stable v1 REST API directly.
    No SDK, no v1beta — guaranteed to work with any valid API key.
    """
    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{model}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 2048,
        },
    }
    response = requests.post(url, json=payload, timeout=60)

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error {response.status_code}: {response.text}"
        )

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {data}") from e


def build_rag_chain(
    vector_store: FAISS,
    google_api_key: Optional[str] = None,
    model_name: str = "gemini-2.0-flash-lite",
    temperature: float = 0.0,
) -> Dict:
    """
    Build a RAG chain dict using direct Gemini REST API.

    Args:
        vector_store: Populated FAISS vector store.
        google_api_key: Google API key from aistudio.google.com.
        model_name: Gemini model name.
        temperature: Unused (fixed at 0 in REST call for now).

    Returns:
        Dict with api_key, model_name, retriever, and chat_history.
    """
    api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key not provided.")

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    return {
        "api_key": api_key,
        "model_name": model_name,
        "retriever": retriever,
        "chat_history": [],
    }


def ask_question(chain: Dict, question: str) -> Dict[str, Any]:
    """
    Ask a question using the RAG chain.

    Args:
        chain: Dict returned by build_rag_chain.
        question: User's question string.

    Returns:
        Dict with 'answer' and 'source_documents' keys.
    """
    api_key: str = chain["api_key"]
    model_name: str = chain["model_name"]
    retriever = chain["retriever"]
    chat_history: List[Dict] = chain["chat_history"]

    # Retrieve relevant document chunks
    source_docs: List[Document] = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in source_docs)

    # Format chat history
    history_str = ""
    for msg in chat_history:
        prefix = "Human" if msg["role"] == "human" else "Assistant"
        history_str += f"{prefix}: {msg['content']}\n"

    # Build prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        chat_history=history_str,
        question=question,
    )

    # Call Gemini via stable v1 REST API
    answer = _call_gemini(api_key, prompt, model=model_name)

    # Save to memory
    chain["chat_history"].append({"role": "human", "content": question})
    chain["chat_history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "source_documents": source_docs,
    }
