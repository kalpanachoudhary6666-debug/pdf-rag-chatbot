"""
test_pdf_processor.py
---------------------
Unit tests for the PDF loading and chunking module.
"""

import pytest
from unittest.mock import patch, MagicMock
from langchain_core.documents import Document

from app.pdf_processor import split_documents, load_pdf_from_bytes


@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content="Artificial intelligence (AI) is intelligence demonstrated by machines. "
                         "It is a branch of computer science focused on building systems that perform tasks "
                         "typically requiring human intelligence such as visual perception, speech recognition, "
                         "decision-making, and language translation.",
            metadata={"page": 0, "source": "sample.pdf"},
        ),
        Document(
            page_content="Machine learning is a subset of AI. Deep learning is a subset of machine learning. "
                         "Neural networks form the backbone of deep learning models. "
                         "These models learn representations from data automatically.",
            metadata={"page": 1, "source": "sample.pdf"},
        ),
    ]


class TestSplitDocuments:
    def test_returns_list(self, sample_documents):
        chunks = split_documents(sample_documents)
        assert isinstance(chunks, list)

    def test_chunks_are_documents(self, sample_documents):
        chunks = split_documents(sample_documents)
        for chunk in chunks:
            assert isinstance(chunk, Document)

    def test_chunk_size_respected(self, sample_documents):
        chunk_size = 100
        chunks = split_documents(sample_documents, chunk_size=chunk_size, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size + 50

    def test_metadata_preserved(self, sample_documents):
        chunks = split_documents(sample_documents)
        for chunk in chunks:
            assert "page" in chunk.metadata
            assert "source" in chunk.metadata

    def test_empty_documents_returns_empty(self):
        chunks = split_documents([])
        assert chunks == []

    def test_overlap_creates_extra_chunks(self, sample_documents):
        chunks_no_overlap = split_documents(sample_documents, chunk_size=100, chunk_overlap=0)
        chunks_with_overlap = split_documents(sample_documents, chunk_size=100, chunk_overlap=50)
        assert len(chunks_with_overlap) >= len(chunks_no_overlap)


class TestLoadPdfFromBytes:
    @patch("app.pdf_processor.PyPDFLoader")
    def test_calls_loader_with_temp_file(self, mock_loader_class):
        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            Document(page_content="Hello world", metadata={"page": 0})
        ]
        mock_loader_class.return_value = mock_loader_instance

        result = load_pdf_from_bytes(b"%PDF-fake-content", "test.pdf")

        mock_loader_class.assert_called_once()
        mock_loader_instance.load.assert_called_once()
        assert len(result) == 1
        assert result[0].page_content == "Hello world"

    @patch("app.pdf_processor.PyPDFLoader")
    def test_temp_file_is_cleaned_up(self, mock_loader_class):
        import os

        created_paths = []

        def capture_path(path):
            created_paths.append(path)
            instance = MagicMock()
            instance.load.return_value = []
            return instance

        mock_loader_class.side_effect = capture_path
        load_pdf_from_bytes(b"%PDF-fake-content")

        for path in created_paths:
            assert not os.path.exists(path), f"Temp file was not cleaned up: {path}"
