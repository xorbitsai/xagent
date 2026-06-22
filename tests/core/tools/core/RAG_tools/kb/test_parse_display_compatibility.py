"""Tests for the KB parse display compatibility facade.

#509 moved parse storage + latest-parse selection into the collection handle;
the facade opens the handle (via its active coordinator, preserving shim
injection) and the display impl keeps the DocumentNotFoundError mapping,
JSON-corruption handling, and element conversion. These tests exercise the real
handle path against the isolated LanceDB store provided by the autouse
``isolate_rag_storage`` fixture in ``tests/conftest.py``.
"""

from __future__ import annotations

import inspect
import json
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import (
    DatabaseOperationError,
    DocumentNotFoundError,
)
from xagent.core.tools.core.RAG_tools.storage.factory import get_vector_index_store


def _seed_parse(
    *,
    collection: str = "docs",
    doc_id: str = "doc-1",
    parse_hash: str,
    created_at: datetime,
    parsed_content: str,
    user_id: int = 1,
) -> None:
    get_vector_index_store().upsert_parses(
        [
            {
                "collection": collection,
                "doc_id": doc_id,
                "parse_hash": parse_hash,
                "parser": "local:default@v1.0.0",
                "created_at": created_at,
                "params_json": "{}",
                "parsed_content": parsed_content,
                "user_id": user_id,
            }
        ]
    )


def _content(text: str, layout_type: str = "text") -> str:
    return json.dumps([{"text": text, "metadata": {"layout_type": layout_type}}])


def _signature_shape(callable_obj: Any) -> list[tuple[str, Any, Any]]:
    return [
        (name, parameter.kind, parameter.default)
        for name, parameter in inspect.signature(callable_obj).parameters.items()
    ]


def test_kb_parse_display_facade_public_surface_imports() -> None:
    """Given the KB package, the parse display facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb
    from xagent.core.tools.core.RAG_tools.kb import (
        KBParseDisplayCompatibilityFacade,
        get_kb_coordinator,
        reset_kb_coordinator_for_tests,
    )

    assert hasattr(kb, "KBParseDisplayCompatibilityFacade")
    reset_kb_coordinator_for_tests()
    coordinator = get_kb_coordinator()
    assert isinstance(
        coordinator.parse_display_compatibility,
        KBParseDisplayCompatibilityFacade,
    )
    assert coordinator.parse_display is coordinator.parse_display_compatibility


def test_parse_display_facade_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.parse import parse_display

    facade = KBParseDisplayCompatibilityFacade()

    assert _signature_shape(facade.reconstruct_parse_result_from_db) == (
        _signature_shape(parse_display.reconstruct_parse_result_from_db)
    )
    assert _signature_shape(facade.paginate_parse_results) == _signature_shape(
        parse_display.paginate_parse_results
    )


def test_public_parse_display_helpers_remain_sync_and_route_through_facade(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Given a public parse display helper call, it routes through the facade."""
    from xagent.core.tools.core.RAG_tools.parse import parse_display

    calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    class _FakeFacade:
        def reconstruct_parse_result_from_db(self, *args: Any, **kwargs: Any):
            calls.append(("reconstruct", args, kwargs))
            return ([{"type": "text", "text": "ok", "metadata": {}}], "hash-1")

        def paginate_parse_results(self, *args: Any, **kwargs: Any):
            calls.append(("paginate", args, kwargs))
            return (["page"], {"page": kwargs["page"]})

    monkeypatch.setattr(
        parse_display,
        "_get_parse_display_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert not inspect.iscoroutinefunction(
        parse_display.reconstruct_parse_result_from_db
    )
    assert not inspect.iscoroutinefunction(parse_display.paginate_parse_results)
    assert parse_display.reconstruct_parse_result_from_db(
        "docs",
        "doc-1",
        parse_hash="hash-1",
        user_id=7,
        is_admin=True,
    ) == ([{"type": "text", "text": "ok", "metadata": {}}], "hash-1")
    assert parse_display.paginate_parse_results([], page=2, page_size=3) == (
        ["page"],
        {"page": 2},
    )
    assert calls == [
        (
            "reconstruct",
            ("docs", "doc-1"),
            {"parse_hash": "hash-1", "user_id": 7, "is_admin": True},
        ),
        ("paginate", ([],), {"page": 2, "page_size": 3}),
    ]


def _facade():
    from xagent.core.tools.core.RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().parse_display_compatibility


def test_parse_display_facade_preserves_sync_tuple_shapes_and_latest_selection() -> (
    None
):
    """Given direct sync calls, latest fallback and explicit hash behavior stay stable."""
    created_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _seed_parse(
        parse_hash="old", created_at=created_at, parsed_content=_content("old body")
    )
    _seed_parse(
        parse_hash="new",
        created_at=created_at + timedelta(seconds=1),
        parsed_content=_content("new body"),
    )
    facade = _facade()

    elements, actual_hash = facade.reconstruct_parse_result_from_db(
        "docs", "doc-1", user_id=1, is_admin=False
    )
    assert actual_hash == "new"
    assert elements == [
        {"type": "text", "text": "new body", "metadata": {"layout_type": "text"}}
    ]

    explicit_elements, explicit_hash = facade.reconstruct_parse_result_from_db(
        "docs", "doc-1", parse_hash="old", user_id=1, is_admin=False
    )
    assert explicit_hash == "old"
    assert explicit_elements[0]["text"] == "old body"

    page_elements, pagination = facade.paginate_parse_results(
        elements + explicit_elements, page=1, page_size=1
    )
    assert len(page_elements) == 1
    assert pagination == {
        "page": 1,
        "page_size": 1,
        "total_elements": 2,
        "total_pages": 2,
        "has_next": True,
        "has_previous": False,
    }


def test_parse_display_lookup_after_rolled_back_ingest_keeps_not_found_behavior() -> (
    None
):
    """Rolled-back ingest leaves no parse row, so lookup stays legacy not-found."""
    facade = _facade()

    with pytest.raises(
        DocumentNotFoundError,
        match="No parse results found for document: doc_id=doc-rolled-back",
    ):
        facade.reconstruct_parse_result_from_db(
            "docs", "doc-rolled-back", user_id=1, is_admin=False
        )


def test_parse_display_explicit_hash_not_found_message() -> None:
    """An explicit missing parse_hash preserves the legacy not-found message."""
    facade = _facade()

    with pytest.raises(
        DocumentNotFoundError,
        match="Parse result not found: doc_id=doc-1, parse_hash=missing",
    ):
        facade.reconstruct_parse_result_from_db(
            "docs", "doc-1", parse_hash="missing", user_id=1, is_admin=False
        )


def test_parse_display_facade_honors_injected_storage_shim() -> None:
    """An injected storage shim routes parse reads through that shim's stores."""
    from xagent.core.tools.core.RAG_tools.kb import KBParseDisplayCompatibilityFacade
    from xagent.core.tools.core.RAG_tools.kb.storage_shim import (
        KBStorageShimCompatibilityFacade,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    created_at = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    _seed_parse(
        parse_hash="h1", created_at=created_at, parsed_content=_content("shim body")
    )

    # A facade with only an injected shim (no coordinator) must back its handle
    # with that shim instead of the process-global coordinator.
    shim = KBStorageShimCompatibilityFacade(
        storage_factory=StorageFactory.get_factory()
    )
    facade = KBParseDisplayCompatibilityFacade(storage_shim=shim)

    elements, actual_hash = facade.reconstruct_parse_result_from_db(
        "docs", "doc-1", user_id=1, is_admin=False
    )
    assert actual_hash == "h1"
    assert elements[0]["text"] == "shim body"


def test_parse_display_facade_preserves_json_corruption_mapping() -> None:
    """Given corrupt parse JSON, facade preserves the legacy DatabaseOperationError."""
    _seed_parse(
        parse_hash="bad",
        created_at=datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        parsed_content="{not-json",
    )
    facade = _facade()

    with pytest.raises(DatabaseOperationError, match="Failed to read parse result"):
        facade.reconstruct_parse_result_from_db(
            "docs", "doc-1", user_id=1, is_admin=False
        )
