"""
test_vector_store.py
--------------------
Unit tests for the FAISS vector store module.
Mocks _get_embeddings to avoid loading real sentence-transformers models in CI.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.vector_store import build_vector_store, similarity_search


@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="The sky is blue.", metadata={"page": 0}),
        Document(page_content="The ocean is deep.", metadata={"page": 1}),
        Document(page_content="Mountains are tall.", metadata={"page": 2}),
    ]


class TestBuildVectorStore:
    @patch("app.vector_store.FAISS")
    @patch("app.vector_store._get_embeddings")
    def test_builds_store_successfully(self, mock_get_embeddings, mock_faiss_class, sample_chunks):
        mock_embeddings = MagicMock()
        mock_get_embeddings.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_faiss_class.from_documents.return_value = mock_store

        result = build_vector_store(sample_chunks)

        mock_get_embeddings.assert_called_once()
        mock_faiss_class.from_documents.assert_called_once_with(sample_chunks, mock_embeddings)
        assert result == mock_store

    @patch("app.vector_store.FAISS")
    @patch("app.vector_store._get_embeddings")
    def test_returns_faiss_store(self, mock_get_embeddings, mock_faiss_class, sample_chunks):
        mock_store = MagicMock()
        mock_faiss_class.from_documents.return_value = mock_store

        result = build_vector_store(sample_chunks)
        assert result == mock_store

    @patch("app.vector_store.FAISS")
    @patch("app.vector_store._get_embeddings")
    def test_accepts_extra_kwargs(self, mock_get_embeddings, mock_faiss_class, sample_chunks):
        mock_faiss_class.from_documents.return_value = MagicMock()
        # Should not raise even with extra kwargs
        build_vector_store(sample_chunks, google_api_key="unused", groq_api_key="unused")


class TestSimilaritySearch:
    def test_returns_documents(self, sample_chunks):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = sample_chunks[:2]

        results = similarity_search(mock_store, "What color is the sky?", k=2)

        mock_store.similarity_search.assert_called_once_with("What color is the sky?", k=2)
        assert len(results) == 2
        assert all(isinstance(doc, Document) for doc in results)

    def test_default_k_is_4(self, sample_chunks):
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        similarity_search(mock_store, "test query")

        mock_store.similarity_search.assert_called_once_with("test query", k=4)
