"""Tests for validate_document_processing (parser vs file type compatibility)."""

from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.management.collection_manager import (
    collection_manager,
    validate_document_processing_sync,
)


class _DummyCollection:
    """Minimal CollectionInfo-like object for isolated validation tests."""

    def __init__(
        self,
        *,
        allow_mixed_parse_methods: bool = False,
        skip_config_validation: bool = False,
        collection_locked: bool = False,
    ) -> None:
        self.allow_mixed_parse_methods = allow_mixed_parse_methods
        self.skip_config_validation = skip_config_validation
        self.collection_locked = collection_locked


def test_validate_document_processing_raises_for_incompatible_type_when_collection_missing(
    monkeypatch: Any,
) -> None:
    """When collection does not exist, still validate file type vs parse_method."""

    async def _raise_value_error(collection_name: str) -> Any:  # type: ignore[unused-argument]
        raise ValueError("collection not found")

    monkeypatch.setattr(collection_manager, "get_collection", _raise_value_error)

    with pytest.raises(ValueError) as exc_info:
        validate_document_processing_sync(
            collection_name="kb-docx",
            file_path="/tmp/sample.docx",
            parsing_method="pypdf",
            chunking_method="recursive",
        )

    msg = str(exc_info.value)
    assert "not compatible" in msg
    assert ".docx" in msg
    assert "Supported methods" in msg


def test_validate_document_processing_allows_default_method_without_collection(
    monkeypatch: Any,
) -> None:
    """When parse_method is default, no type compatibility check is run."""

    async def _raise_value_error(collection_name: str) -> Any:  # type: ignore[unused-argument]
        raise ValueError("collection not found")

    monkeypatch.setattr(collection_manager, "get_collection", _raise_value_error)

    validate_document_processing_sync(
        collection_name="kb-docx",
        file_path="/tmp/sample.docx",
        parsing_method="default",
        chunking_method="recursive",
    )


def test_validate_document_processing_respects_allow_mixed(monkeypatch: Any) -> None:
    """When allow_mixed_parse_methods is True, skip parser vs file type check."""

    async def _get_collection(collection_name: str) -> _DummyCollection:  # type: ignore[unused-argument]
        return _DummyCollection(allow_mixed_parse_methods=True)

    monkeypatch.setattr(collection_manager, "get_collection", _get_collection)

    validate_document_processing_sync(
        collection_name="kb-docx",
        file_path="/tmp/sample.docx",
        parsing_method="pypdf",
        chunking_method="recursive",
    )
