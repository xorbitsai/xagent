"""High-level pipelines orchestrating multiple RAG core tools."""

from .document_ingestion import process_document
from .document_search import search_documents
from .web_ingestion import run_web_ingestion

__all__ = ["process_document", "search_documents", "run_web_ingestion"]
