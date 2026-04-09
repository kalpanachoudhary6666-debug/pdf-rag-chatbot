"""
vector_store.py
---------------
Builds and queries a FAISS vector store using local sentence-transformers embeddings.
No API key required for embeddings — runs entirely on your machine.
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List, Optional
import os


# Small, fast, high-quality model — downloads once (~90MB), then cached locally
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_vector_store(chunks: List[Document], google_api_key: Optional[str] = None) -> FAISS:
    """
    Create a FAISS vector store from document chunks.
    Embeddings run locally via sentence-transformers (no API key needed).

    Args:
        chunks: List of chunked LangChain Document objects.
        google_api_key: Not used for embeddings — kept for API compatibility.

    Returns:
        FAISS vector store instance.
    """
    embeddings = _get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def save_vector_store(vector_store: FAISS, path: str) -> None:
    vector_store.save_local(path)


def load_vector_store(path: str, google_api_key: Optional[str] = None) -> FAISS:
    embeddings = _get_embeddings()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def similarity_search(vector_store: FAISS, query: str, k: int = 4) -> List[Document]:
    return vector_store.similarity_search(query, k=k)
