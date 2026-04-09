"""
rag_chain.py
------------
RAG chain using Groq API — free tier, no credit card needed.
Models: llama-3.3-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768
Compatible with Python 3.12, 3.13, 3.14.
"""

import os
from groq import Groq
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


def build_rag_chain(
    vector_store: FAISS,
    groq_api_key: Optional[str] = None,
    model_name: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
) -> Dict:
    """
    Build a RAG chain using Groq's free LLM API.

    Args:
        vector_store: Populated FAISS vector store.
        groq_api_key: Groq API key from console.groq.com (free).
        model_name: Groq model to use.
        temperature: LLM temperature.

    Returns:
        Dict with client, model_name, retriever, and chat_history.
    """
    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Groq API key not provided. Get a free key at https://console.groq.com")

    client = Groq(api_key=api_key)

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    return {
        "client": client,
        "model_name": model_name,
        "temperature": temperature,
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
    client: Groq = chain["client"]
    model_name: str = chain["model_name"]
    temperature: float = chain["temperature"]
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

    # Call Groq API
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2048,
    )
    answer = response.choices[0].message.content

    # Save to memory
    chain["chat_history"].append({"role": "human", "content": question})
    chain["chat_history"].append({"role": "assistant", "content": answer})

    return {
        "answer": answer,
        "source_documents": source_docs,
    }
