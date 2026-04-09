"""
test_vector_store.py
--------------------
Unit tests for the FAISS vector store module.
Uses mocking to avoid real OpenAI API calls in CI.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain.schema import Document

from app.vector_store import build_vector_store, similarity_search


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunks():
    return [
        Document(page_content="The sky is blue.", metadata={"page": 0}),
        Document(page_content="The ocean is deep.", metadata={"page": 1}),
        Document(page_content="Mountains are tall.", metadata={"page": 2}),
    ]


# ── Tests: build_vector_store ──────────────────────────────────────────────────

class TestBuildVectorStore:
    @patch("app.vector_store.FAISS")
    @patch("app.vector_store.OpenAIEmbeddings")
    def test_builds_store_successfully(self, mock_embeddings_class, mock_faiss_class, sample_chunks):
        mock_embeddings = MagicMock()
        mock_embeddings_class.return_value = mock_embeddings

        mock_store = MagicMock()
        mock_faiss_class.from_documents.return_value = mock_store

        result = build_vector_store(sample_chunks, openai_api_key="sk-test-key")

        mock_embeddings_class.assert_called_once_with(openai_api_key="sk-test-key")
        mock_faiss_class.from_documents.assert_called_once_with(sample_chunks, mock_embeddings)
        assert result == mock_store

    def test_raises_without_api_key(self, sample_chunks, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(ValueError, match="OpenAI API key not provided"):
            build_vector_store(sample_chunks, openai_api_key=None)

    @patch("app.vector_store.FAISS")
    @patch("app.vector_store.OpenAIEmbeddings")
    def test_uses_env_var_when_no_key_passed(
        self, mock_embeddings_class, mock_faiss_class, sample_chunks, monkeypatch
    ):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        mock_faiss_class.from_documents.return_value = MagicMock()

        build_vector_store(sample_chunks)

        mock_embeddings_class.assert_called_once_with(openai_api_key="sk-env-key")


# ── Tests: similarity_search ───────────────────────────────────────────────────

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
