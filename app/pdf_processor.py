"""
pdf_processor.py
----------------
Handles loading and chunking PDF documents using LangChain.
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import tempfile
import os


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a PDF from a file path and return a list of LangChain Documents.

    Args:
        file_path: Path to the PDF file.

    Returns:
        List of Document objects (one per page).
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents


def load_pdf_from_bytes(pdf_bytes: bytes, filename: str = "uploaded.pdf") -> List[Document]:
    """
    Load a PDF from raw bytes (useful for Streamlit file uploads).

    Args:
        pdf_bytes: Raw bytes of the PDF file.
        filename: Optional name for the temp file.

    Returns:
        List of Document objects.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_bytes)
        tmp_path = tmp_file.name

    try:
        documents = load_pdf(tmp_path)
    finally:
        os.unlink(tmp_path)

    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Document]:
    """
    Split documents into smaller chunks for embedding.

    Args:
        documents: List of LangChain Document objects.
        chunk_size: Max characters per chunk.
        chunk_overlap: Characters of overlap between consecutive chunks.

    Returns:
        List of chunked Document objects.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks
