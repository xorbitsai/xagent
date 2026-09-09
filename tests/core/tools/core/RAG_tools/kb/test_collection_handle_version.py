"""Tests for the main-pointer lifecycle inline in LanceDBCollectionHandle (#513).

The handle owns collection-scoped main-pointer mechanics: get, set, list, delete.
These tests mirror the assertions in
version_management/test_main_pointer_manager.py:63,165,224.

A lightweight ``_FakeMainPointerStore`` is injected directly into the
``KBCollectionContext`` to avoid real LanceDB I/O (same isolation strategy used
by test_version_compatibility.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VersionManagementError
from xagent.core.tools.core.RAG_tools.core.schemas import StepType
from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
    LanceDBCollectionHandle,
)
from xagent.core.tools.core.RAG_tools.kb.models import (
    KBAccessMode,
    KBBackendCapabilities,
    KBCollectionContext,
    KBStorageBackend,
    KBUserScope,
)
from xagent.core.tools.core.RAG_tools.storage.contracts import MainPointerStore
from xagent.core.tools.core.RAG_tools.storage.factory import (
    get_ingestion_status_store,
    get_metadata_store,
    get_vector_index_store,
)


class _FakeMainPointerStore(MainPointerStore):
    """In-memory MainPointerStore for handle-level tests."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, Optional[str]], dict[str, Any]] = {}

    def _key(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str],
    ) -> tuple[str, str, str, Optional[str]]:
        return (collection, doc_id, step_type, model_tag)

    def get_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        row = self.rows.get(self._key(collection, doc_id, step_type, model_tag))
        return dict(row) if row is not None else None

    def set_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        key = self._key(collection, doc_id, step_type, model_tag)
        existing = self.rows.get(key)
        now = datetime.now(timezone.utc)
        created_at = existing["created_at"] if existing else now
        self.rows[key] = {
            "collection": collection,
            "doc_id": doc_id,
            "step_type": step_type,
            "model_tag": model_tag if model_tag is not None else "",
            "semantic_id": semantic_id,
            "technical_id": technical_id,
            "operator": operator,
            "created_at": created_at,
            "updated_at": now,
        }

    def list_main_pointers(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        rows = [
            dict(row)
            for (row_coll, row_doc, _, _), row in self.rows.items()
            if row_coll == collection and (doc_id is None or row_doc == doc_id)
        ]
        return rows[:limit]

    def delete_main_pointer(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        return (
            self.rows.pop(self._key(collection, doc_id, step_type, model_tag), None)
            is not None
        )

    async def get_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        return self.get_main_pointer(
            collection, doc_id, step_type, model_tag=model_tag, user_id=user_id
        )

    async def set_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        semantic_id: str,
        technical_id: str,
        model_tag: Optional[str] = None,
        operator: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        self.set_main_pointer(
            collection,
            doc_id,
            step_type,
            semantic_id,
            technical_id,
            model_tag=model_tag,
            operator=operator,
            user_id=user_id,
        )

    async def list_main_pointers_async(
        self,
        collection: str,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        return self.list_main_pointers(
            collection, doc_id=doc_id, user_id=user_id, limit=limit
        )

    async def delete_main_pointer_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> bool:
        return self.delete_main_pointer(
            collection, doc_id, step_type, model_tag=model_tag, user_id=user_id
        )


def make_handle(
    collection: str = "coll",
    store: Optional[MainPointerStore] = None,
) -> LanceDBCollectionHandle:
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=store if store is not None else _FakeMainPointerStore(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def test_set_get_roundtrip() -> None:
    """set_main_pointer + get_main_pointer round-trips the pointer correctly.

    Mirrors test_main_pointer_manager.py:63.
    """
    handle = make_handle("roundtrip_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_123",
        technical_id="hash_456",
    )
    result = handle.get_main_pointer("doc-1", "parse")
    assert result is not None
    assert result["semantic_id"] == "parse_123"
    assert result["technical_id"] == "hash_456"
    assert result["collection"] == "roundtrip_coll"
    assert result["doc_id"] == "doc-1"
    assert result["step_type"] == "parse"


def test_delete_returns_bool() -> None:
    """delete_main_pointer returns True when deleted, False when already absent.

    Mirrors test_main_pointer_manager.py:165.
    """
    handle = make_handle("delete_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    # First delete returns True
    result = handle.delete_main_pointer("doc-1", "parse")
    assert result is True
    # Second delete (already absent) returns False
    result2 = handle.delete_main_pointer("doc-1", "parse")
    assert result2 is False


def test_get_missing_returns_none() -> None:
    """get_main_pointer returns None for a row that doesn't exist."""
    handle = make_handle("missing_coll")
    result = handle.get_main_pointer("nonexistent-doc", "parse")
    assert result is None


def test_model_tag_none_normalized() -> None:
    """model_tag=None passes through to the store (store normalizes to '').

    Mirrors test_main_pointer_manager.py:165 backward-compat check.
    """
    handle = make_handle("modeltag_coll")
    handle.set_main_pointer(
        "doc-1",
        "embed",
        semantic_id="embed_123",
        technical_id="embed_hash",
        model_tag=None,
    )
    # Get with model_tag=None should find the row (None passed straight through)
    result = handle.get_main_pointer("doc-1", "embed", model_tag=None)
    assert result is not None
    assert result["semantic_id"] == "embed_123"


def test_set_preserves_created_at() -> None:
    """set_main_pointer preserves created_at on existing rows (store behavior).

    Mirrors test_main_pointer_manager.py:224.
    """
    handle = make_handle("preserve_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="old_parse",
        technical_id="old_hash",
    )
    first = handle.get_main_pointer("doc-1", "parse")
    assert first is not None
    created_at_first = first.get("created_at")

    # Update the pointer
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="new_parse",
        technical_id="new_hash",
    )
    second = handle.get_main_pointer("doc-1", "parse")
    assert second is not None
    assert second["semantic_id"] == "new_parse"
    assert second["technical_id"] == "new_hash"
    # created_at preserved from first write
    assert second.get("created_at") == created_at_first


def test_list_main_pointers() -> None:
    """list_main_pointers returns all pointers for the collection."""
    handle = make_handle("list_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    handle.set_main_pointer(
        "doc-2",
        "chunk",
        semantic_id="chunk_2",
        technical_id="hash_2",
    )
    pointers = handle.list_main_pointers()
    assert len(pointers) == 2
    doc_ids = {p["doc_id"] for p in pointers}
    assert doc_ids == {"doc-1", "doc-2"}


def test_list_main_pointers_filtered_by_doc_id() -> None:
    """list_main_pointers(doc_id=...) returns only that document's rows."""
    handle = make_handle("list_filter_coll")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_1",
        technical_id="hash_1",
    )
    handle.set_main_pointer(
        "doc-2",
        "parse",
        semantic_id="parse_2",
        technical_id="hash_2",
    )
    pointers = handle.list_main_pointers(doc_id="doc-1")
    assert len(pointers) == 1
    assert pointers[0]["doc_id"] == "doc-1"


# ── Task 4: list_candidates handle tests ─────────────────────────────────────


class _FakeVectorIndexStore:
    """Minimal in-memory VectorIndexStore stub for list_candidates handle tests."""

    def __init__(self, candidates: list) -> None:
        self._candidates = candidates

    def list_version_candidate_rows(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag=None,
    ):
        return list(self._candidates)

    async def list_version_candidate_rows_async(
        self,
        collection: str,
        doc_id: str,
        step_type: str,
        model_tag=None,
    ):
        return self.list_version_candidate_rows(
            collection, doc_id, step_type, model_tag
        )


def make_handle_with_vector_store(
    collection: str, candidates: list
) -> "LanceDBCollectionHandle":
    """Make a handle with a fake VectorIndexStore for list_candidates testing."""
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_ingestion_status_store,
        get_metadata_store,
    )

    fake_vis = _FakeVectorIndexStore(candidates)
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=fake_vis,  # type: ignore[arg-type]
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=_FakeMainPointerStore(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context)


def test_list_candidates_sort_before_limit() -> None:
    """Sorting must happen before limit (get correct top-N). Mirrors test_list_candidates.py:408."""

    base_time = datetime(2024, 1, 1)
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "hash_oldest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time,
            "operator": "unknown",
        },
        {
            "semantic_id": "s2",
            "technical_id": "hash_middle",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=5),
            "operator": "unknown",
        },
        {
            "semantic_id": "s3",
            "technical_id": "hash_newer",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=7),
            "operator": "unknown",
        },
        {
            "semantic_id": "s4",
            "technical_id": "hash_newest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=10),
            "operator": "unknown",
        },
        {
            "semantic_id": "s5",
            "technical_id": "hash_second_newest",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": base_time + timedelta(days=8),
            "operator": "unknown",
        },
    ]

    handle = make_handle_with_vector_store("sort_coll", candidates)
    result = handle.list_candidates(
        "doc-1", "parse", limit=3, order_by="created_at desc"
    )

    assert len(result["candidates"]) == 3
    assert result["total_count"] == 5
    assert result["returned_count"] == 3

    technical_ids = [c["technical_id"] for c in result["candidates"]]
    assert technical_ids[0] == "hash_newest"
    assert technical_ids[1] == "hash_second_newest"
    assert technical_ids[2] == "hash_newer"
    assert "hash_oldest" not in technical_ids
    assert "hash_middle" not in technical_ids


def test_list_candidates_state_filter() -> None:
    """State filter narrows candidates. Mirrors test_list_candidates.py:296."""
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "hash1",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
        {
            "semantic_id": "s2",
            "technical_id": "hash2",
            "params_brief": {},
            "stats": {},
            "state": "main",
            "created_at": datetime(2024, 1, 2),
            "operator": "unknown",
        },
    ]
    handle = make_handle_with_vector_store("state_coll", candidates)

    result = handle.list_candidates("doc-1", "parse", state="candidate")
    assert (
        result["total_count"] == 1
    )  # total AFTER state filter (matches _list_candidates_impl behavior)
    assert result["returned_count"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["state"] == "candidate"
    assert result["filters"]["state"] == "candidate"


def test_list_candidates_model_tag_filter() -> None:
    """model_tag is passed through to the store. Mirrors test_list_candidates.py:369."""
    candidates_bge = [
        {
            "semantic_id": "embed_BAAI/bge-large-zh-v1.5_parse_ha",
            "technical_id": "parse_hash1",
            "params_brief": {
                "model": "BAAI/bge-large-zh-v1.5",
                "model_tag": "bge_large",
            },
            "stats": {"upsert_count": 1, "vector_dim": 3},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
    ]

    handle = make_handle_with_vector_store("mtag_coll", candidates_bge)
    result = handle.list_candidates("doc-1", "embed", model_tag="bge_large")

    assert len(result["candidates"]) == 1
    assert result["total_count"] == 1
    assert result["model_tag"] == "bge_large"


def test_list_candidates_result_dict_shape() -> None:
    """Result dict has all required keys. Mirrors test_list_candidates.py:296."""
    candidates = [
        {
            "semantic_id": "s1",
            "technical_id": "h1",
            "params_brief": {},
            "stats": {},
            "state": "candidate",
            "created_at": datetime(2024, 1, 1),
            "operator": "unknown",
        },
    ]
    handle = make_handle_with_vector_store("shape_coll", candidates)
    result = handle.list_candidates(
        "doc-1",
        "parse",
        model_tag=None,
        state=None,
        limit=50,
        order_by="created_at desc",
    )

    assert "candidates" in result
    assert "total_count" in result
    assert "returned_count" in result
    assert "step_type" in result
    assert "model_tag" in result
    assert "filters" in result
    assert result["step_type"] == "parse"
    assert result["filters"]["state"] is None
    assert result["filters"]["limit"] == 50
    assert result["filters"]["order_by"] == "created_at desc"


class _FakeVectorIndexStoreWithCleanup(_FakeVectorIndexStore):
    """Extended fake that records cleanup_cascade_by_scope calls."""

    def __init__(self, candidates: list) -> None:
        super().__init__(candidates)
        self.last_cleanup_args: Optional[Dict[str, Any]] = None
        self.cleanup_return: Dict[str, int] = {
            "embeddings": 0,
            "chunks": 0,
            "parses": 0,
        }
        self.cleanup_side_effect: Optional[Exception] = None

    def cleanup_cascade_by_scope(
        self,
        collection: str,
        doc_id: str,
        scope: str,
        *,
        new_parse_hash=None,
        old_parse_hash=None,
        model_tag=None,
        user_id=None,
        is_admin: bool = True,
        preview_only: bool = True,
        confirm: bool = False,
    ) -> Dict[str, int]:
        if self.cleanup_side_effect is not None:
            raise self.cleanup_side_effect
        self.last_cleanup_args = {
            "collection": collection,
            "doc_id": doc_id,
            "scope": scope,
            "is_admin": is_admin,
            "user_id": user_id,
            "new_parse_hash": new_parse_hash,
            "old_parse_hash": old_parse_hash,
            "model_tag": model_tag,
        }
        return dict(self.cleanup_return)


def make_handle_with_cleanup_store(
    collection: str,
    *,
    cleanup_return: Optional[Dict[str, int]] = None,
    cleanup_side_effect: Optional[Exception] = None,
) -> tuple:
    """Make a handle with a fake cleanup store. Returns (handle, fake_store)."""
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_ingestion_status_store,
        get_metadata_store,
    )

    fake_vis = _FakeVectorIndexStoreWithCleanup([])
    if cleanup_return is not None:
        fake_vis.cleanup_return = cleanup_return
    if cleanup_side_effect is not None:
        fake_vis.cleanup_side_effect = cleanup_side_effect

    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=fake_vis,  # type: ignore[arg-type]
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=_FakeMainPointerStore(),
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context), fake_vis


class TestCleanupCascadeHandle:
    """Tests for cleanup_cascade and per-plane wrappers on LanceDBCollectionHandle."""

    def test_cleanup_cascade_is_admin_none_defaults_to_true(self) -> None:
        """is_admin=None should be promoted to True before calling the store."""
        handle, fake_vis = make_handle_with_cleanup_store(
            "cascade_coll",
            cleanup_return={"embeddings": 0, "chunks": 0, "parses": 0},
        )
        handle.cleanup_cascade("doc1", "parse", is_admin=None, preview_only=True)
        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["is_admin"] is True

    def test_cleanup_document_cascade_wraps_cascade_error(self) -> None:
        """Exceptions from the store should be wrapped in CascadeCleanupError."""
        from xagent.core.tools.core.RAG_tools.core.exceptions import CascadeCleanupError

        handle, fake_vis = make_handle_with_cleanup_store(
            "cascade_err_coll",
            cleanup_side_effect=RuntimeError("store boom"),
        )
        with pytest.raises(CascadeCleanupError):
            handle.cleanup_document_cascade("doc1")

    def test_cleanup_parse_cascade_passes_correct_scope(self) -> None:
        """cleanup_parse_cascade must pass scope='parse' to the store."""
        handle, fake_vis = make_handle_with_cleanup_store(
            "parse_scope_coll",
            cleanup_return={"embeddings": 0, "chunks": 0, "parses": 0},
        )
        handle.cleanup_parse_cascade(
            "doc1",
            old_parse_hash="oldhash",
            new_parse_hash="newhash",
            preview_only=True,
        )
        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["scope"] == "parse"

    def test_cleanup_embed_cascade_uses_embeddings_scope(self) -> None:
        """cleanup_embed_cascade must pass scope='embeddings' to the store."""
        handle, fake_vis = make_handle_with_cleanup_store(
            "embed_scope_coll",
            cleanup_return={"embeddings": 5},
        )
        handle.cleanup_embed_cascade("doc1", preview_only=True)
        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["scope"] == "embeddings"

    def test_cleanup_chunk_cascade_passes_correct_scope(self) -> None:
        """cleanup_chunk_cascade must pass scope='chunk' to the store."""
        handle, fake_vis = make_handle_with_cleanup_store(
            "chunk_scope_coll",
            cleanup_return={"embeddings": 0, "chunks": 2},
        )
        handle.cleanup_chunk_cascade(
            "doc1", new_parse_hash="newhash", preview_only=True
        )
        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["scope"] == "chunk"

    # ── _call_cleanup_cascade_for_step: the step_type → scope mapping ─────────
    # Nothing executed this helper before: promote_version_main's own tests
    # patch it out, and the cleanup_*_cascade tests above go in one level down.

    @pytest.mark.parametrize(
        ("step_type", "expected_scope"),
        [(StepType.PARSE, "parse"), (StepType.CHUNK, "chunk")],
    )
    def test_cascade_for_step_maps_step_type_to_scope(
        self, step_type: StepType, expected_scope: str
    ) -> None:
        """parse/chunk steps route to their scope and carry both hashes."""
        handle, fake_vis = make_handle_with_cleanup_store("step_scope_coll")

        handle._call_cleanup_cascade_for_step(
            "doc1",
            step_type,
            technical_id="newhash",
            old_technical_id="oldhash",
            preview_only=True,
        )

        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["scope"] == expected_scope
        assert fake_vis.last_cleanup_args["new_parse_hash"] == "newhash"
        assert fake_vis.last_cleanup_args["old_parse_hash"] == "oldhash"

    def test_cascade_for_step_embed_passes_model_tag(self) -> None:
        """The embed step routes to the embeddings scope and forwards model_tag."""
        handle, fake_vis = make_handle_with_cleanup_store("step_embed_coll")

        handle._call_cleanup_cascade_for_step(
            "doc1",
            StepType.EMBED,
            technical_id="embed-hash",
            model_tag="bge_large",
            preview_only=True,
        )

        assert fake_vis.last_cleanup_args is not None
        assert fake_vis.last_cleanup_args["scope"] == "embeddings"
        assert fake_vis.last_cleanup_args["model_tag"] == "bge_large"

    def test_cascade_for_step_embed_without_model_tag_raises(self) -> None:
        """Embed cleanup cannot pick a table without a model tag."""
        handle, fake_vis = make_handle_with_cleanup_store("step_embed_bad_coll")

        with pytest.raises(VersionManagementError, match="model_tag is required"):
            handle._call_cleanup_cascade_for_step(
                "doc1", StepType.EMBED, technical_id="embed-hash"
            )

        assert fake_vis.last_cleanup_args is None, "no cleanup may be attempted"

    def test_cascade_for_step_rejects_unknown_step_type(self) -> None:
        """An unmapped step_type is refused, and names itself in the error."""
        handle, fake_vis = make_handle_with_cleanup_store("step_bad_coll")

        with pytest.raises(VersionManagementError, match="Invalid step_type: bogus"):
            handle._call_cleanup_cascade_for_step("doc1", "bogus", technical_id="hash")

        assert fake_vis.last_cleanup_args is None, "no cleanup may be attempted"


# ── Task 6: promote_version_main handle tests ─────────────────────────────────


class _FakeVectorIndexStoreForPromotion(_FakeVectorIndexStoreWithCleanup):
    """Fake store that supports both candidates and cleanup_cascade for promotion tests."""

    def __init__(
        self,
        candidates: list,
        cleanup_return: Optional[Dict[str, int]] = None,
        cleanup_side_effect: Optional[Exception] = None,
    ) -> None:
        super().__init__(candidates)
        if cleanup_return is not None:
            self.cleanup_return = cleanup_return
        if cleanup_side_effect is not None:
            self.cleanup_side_effect = cleanup_side_effect


def make_handle_for_promotion(
    collection: str = "promo_coll",
    *,
    candidates: Optional[list] = None,
    cleanup_return: Optional[Dict[str, int]] = None,
    cleanup_side_effect: Optional[Exception] = None,
    main_pointer_store: Optional[Any] = None,
) -> tuple:
    """Make a handle suitable for promote_version_main tests.

    Returns (handle, fake_vis, fake_mp_store).
    """
    from xagent.core.tools.core.RAG_tools.kb.models import (
        KBAccessMode,
        KBBackendCapabilities,
        KBCollectionContext,
        KBStorageBackend,
        KBUserScope,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_ingestion_status_store,
        get_metadata_store,
    )

    fake_vis = _FakeVectorIndexStoreForPromotion(
        candidates or [],
        cleanup_return=cleanup_return,
        cleanup_side_effect=cleanup_side_effect,
    )
    fake_mp = (
        main_pointer_store
        if main_pointer_store is not None
        else _FakeMainPointerStore()
    )

    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=fake_vis,  # type: ignore[arg-type]
        ingestion_status_store=get_ingestion_status_store(),
        main_pointer_store=fake_mp,
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context), fake_vis, fake_mp


class TestPromoteVersionMainHandle:
    """Tests for promote_version_main inlined in LanceDBCollectionHandle (Task 6).

    Mirrors test_promote_version_main.py:241,296,346 and test_version_compatibility.py:655.
    """

    def _make_candidates(self) -> list:
        from datetime import datetime

        return [
            {
                "semantic_id": "parse_test_v1",
                "technical_id": "abc123",
                "params_brief": {"method": "unstructured"},
                "stats": {"paragraphs_count": 10},
                "state": "candidate",
                "created_at": datetime.now(),
                "operator": "test_user",
            }
        ]

    def test_preview_only(self) -> None:
        """preview_only=True returns promoted=False, preview=True with counts.

        Mirrors test_promote_version_main.py:241.
        """
        handle, fake_vis, _ = make_handle_for_promotion(
            "preview_coll",
            candidates=self._make_candidates(),
            cleanup_return={"parses": 2, "chunks": 10, "embeddings": 50},
        )

        result = handle.promote_version_main(
            "doc-1",
            "parse",
            "abc123",
            preview_only=True,
        )

        assert result["promoted"] is False
        assert result["preview"] is True
        assert result["deleted_counts"]["parses"] == 2
        assert result["deleted_counts"]["chunks"] == 10
        assert result["deleted_counts"]["embeddings"] == 50
        assert "message" in result

    def test_not_confirmed(self) -> None:
        """confirm=False returns preview=True with a message. Mirrors test_promote_version_main.py:296."""
        handle, fake_vis, _ = make_handle_for_promotion(
            "not_confirmed_coll",
            candidates=self._make_candidates(),
            cleanup_return={"parses": 0, "chunks": 0, "embeddings": 0},
        )

        result = handle.promote_version_main(
            "doc-1",
            "parse",
            "abc123",
            preview_only=False,
            confirm=False,
        )

        assert result["promoted"] is False
        assert result["preview"] is True
        assert "Set confirm=True to execute the promotion" in result["message"]

    def test_execute_promotion(self) -> None:
        """confirm=True executes promotion and sets the main pointer.

        Mirrors test_promote_version_main.py:346.
        """
        handle, fake_vis, fake_mp = make_handle_for_promotion(
            "execute_coll",
            candidates=self._make_candidates(),
            cleanup_return={"parses": 2, "chunks": 10, "embeddings": 50},
        )

        result = handle.promote_version_main(
            "doc-1",
            "parse",
            "abc123",
            preview_only=False,
            confirm=True,
            operator="test_user",
        )

        assert result["promoted"] is True
        assert result["preview"] is False
        assert result["deleted_counts"]["parses"] == 2
        assert result["deleted_counts"]["chunks"] == 10
        assert result["deleted_counts"]["embeddings"] == 50
        assert result["operator"] == "test_user"

        # Main pointer must have been advanced
        pointer = fake_mp.get_main_pointer("execute_coll", "doc-1", "parse")
        assert pointer is not None
        assert pointer["technical_id"] == "abc123"
        assert pointer["semantic_id"] == "parse_test_v1"

    def test_no_records_deleted_guard(self) -> None:
        """Raises VersionManagementError when deleted_counts is empty on confirm.

        Mirrors the "No records deleted" guard in promote_version_main.py.
        """
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VersionManagementError,
        )

        handle, fake_vis, _ = make_handle_for_promotion(
            "no_delete_coll",
            candidates=self._make_candidates(),
            cleanup_return={},  # empty → falsy
        )

        with pytest.raises(VersionManagementError, match="No records deleted"):
            handle.promote_version_main(
                "doc-1",
                "parse",
                "abc123",
                preview_only=False,
                confirm=True,
                operator="tester",
            )

    def test_operator_too_long(self) -> None:
        """Operator longer than 32 characters raises VersionManagementError."""
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VersionManagementError,
        )

        handle, fake_vis, _ = make_handle_for_promotion(
            "op_len_coll",
            candidates=self._make_candidates(),
            cleanup_return={"parses": 1},
        )

        with pytest.raises(VersionManagementError, match="Operator name too long"):
            handle.promote_version_main(
                "doc-1",
                "parse",
                "abc123",
                operator="a" * 33,
            )

    def test_failed_promotion_does_not_advance_pointer(self) -> None:
        """Cleanup failure does not advance the main pointer.

        Mirrors test_version_compatibility.py:655.
        """
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            VersionManagementError,
        )

        handle, fake_vis, fake_mp = make_handle_for_promotion(
            "fail_promo_coll",
            candidates=self._make_candidates(),
            cleanup_side_effect=RuntimeError("cleanup failed"),
        )

        with pytest.raises(VersionManagementError, match="cleanup failed"):
            handle.promote_version_main(
                "doc-1",
                "parse",
                "abc123",
                operator="tester",
                confirm=True,
            )

        # Main pointer must NOT have been advanced
        pointer = fake_mp.get_main_pointer("fail_promo_coll", "doc-1", "parse")
        assert pointer is None


# ── Task 7: rollback snapshot/restore/clear handle tests ─────────────────────


class _FakeIngestionStatusStore:
    """In-memory IngestionStatusStore for handle-level snapshot tests."""

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    def _key(self, collection: str, doc_id: str) -> tuple:
        return (collection, doc_id)

    def write_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        *,
        status: str,
        message: Optional[str] = None,
        parse_hash: Optional[str] = None,
        user_id: Optional[int] = None,
    ) -> None:
        self.rows[self._key(collection, doc_id)] = {
            "collection": collection,
            "doc_id": doc_id,
            "status": status,
            "message": message,
            "parse_hash": parse_hash,
            "user_id": user_id,
        }

    def load_ingestion_status(
        self,
        collection: Optional[str] = None,
        doc_id: Optional[str] = None,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> List[Dict[str, Any]]:
        result = []
        for (row_coll, row_doc), row in self.rows.items():
            if collection is not None and row_coll != collection:
                continue
            if doc_id is not None and row_doc != doc_id:
                continue
            result.append(dict(row))
        return result

    def clear_ingestion_status(
        self,
        collection: str,
        doc_id: str,
        user_id: Optional[int] = None,
        is_admin: bool = False,
    ) -> None:
        self.rows.pop(self._key(collection, doc_id), None)

    def rename_collection_status(self, old_name, new_name, user_id, is_admin=False):
        return []

    async def write_ingestion_status_async(
        self, collection, doc_id, *, status, message=None, parse_hash=None, user_id=None
    ):
        self.write_ingestion_status(
            collection,
            doc_id,
            status=status,
            message=message,
            parse_hash=parse_hash,
            user_id=user_id,
        )

    async def load_ingestion_status_async(
        self, collection=None, doc_id=None, user_id=None, is_admin=False
    ):
        return self.load_ingestion_status(
            collection=collection, doc_id=doc_id, user_id=user_id, is_admin=is_admin
        )

    async def clear_ingestion_status_async(
        self, collection, doc_id, user_id=None, is_admin=False
    ):
        self.clear_ingestion_status(
            collection, doc_id, user_id=user_id, is_admin=is_admin
        )


def make_handle_with_status_store(
    collection: str = "snap_coll",
) -> tuple:
    """Return (handle, fake_status_store, fake_mp_store) for snapshot tests."""
    fake_status = _FakeIngestionStatusStore()
    fake_mp = _FakeMainPointerStore()
    context = KBCollectionContext(
        collection=collection,
        user_scope=KBUserScope(user_id=None, is_admin=True),
        access_mode=KBAccessMode.WRITE,
        allow_create=True,
        hide_missing=True,
        metadata_store=get_metadata_store(),
        vector_index_store=get_vector_index_store(),
        ingestion_status_store=fake_status,  # type: ignore[arg-type]
        main_pointer_store=fake_mp,
        backend=KBStorageBackend.LANCEDB,
        capabilities=KBBackendCapabilities.lancedb(),
        collection_info=None,
    )
    return LanceDBCollectionHandle(context), fake_status, fake_mp


def test_status_snapshot_restore_then_clear() -> None:
    """Failed-ingest path: snapshot → simulate failure → restore → clear.

    Asserts the handle's status snapshot primitives produce inspectable state:
    the status row is restorable from the snapshot, and clear removes it.

    Mirrors the failed-ingest rollback invariant (restores-then-clears status;
    main pointer NOT advanced by failed ops).
    """
    handle, fake_status, _ = make_handle_with_status_store("status_snap_coll")

    # Write a pre-existing "processing" status
    handle.write_ingestion_status(
        "doc-snap", status="processing", message="in progress"
    )

    # Capture snapshot (before potential failure)
    snapshot = handle.capture_status_snapshot("doc-snap")
    assert snapshot is not None
    assert len(snapshot) == 1
    assert snapshot[0]["status"] == "processing"

    # Simulate a failure: overwrite status with "failed"
    handle.write_ingestion_status("doc-snap", status="failed", message="boom")

    # Verify mutation happened
    rows = handle.load_ingestion_status(doc_id="doc-snap", is_admin=True)
    assert rows[0]["status"] == "failed"

    # Restore from snapshot
    handle.restore_status_snapshot("doc-snap", snapshot)

    restored = handle.load_ingestion_status(doc_id="doc-snap", is_admin=True)
    assert len(restored) == 1
    assert restored[0]["status"] == "processing"
    assert restored[0]["message"] == "in progress"

    # Clear
    handle.clear_status_snapshot("doc-snap")
    cleared = handle.load_ingestion_status(doc_id="doc-snap", is_admin=True)
    assert cleared == []


def test_main_pointer_snapshot_restore_reverts() -> None:
    """capture_main_pointer_snapshot + restore reverts a mutated pointer.

    Mirrors test_version_compatibility.py:374.
    """
    handle = make_handle("mp_snap_coll")
    handle.set_main_pointer(
        "doc-1", "parse", semantic_id="parse_old", technical_id="old-hash"
    )

    snapshot = handle.capture_main_pointer_snapshot("doc-1", "parse")
    handle.set_main_pointer(
        "doc-1",
        "parse",
        semantic_id="parse_new",
        technical_id="new-hash",
        operator="mutator",
    )

    assert handle.get_main_pointer("doc-1", "parse")["technical_id"] == "new-hash"

    result = handle.restore_main_pointer_snapshot(snapshot, operator="rollback")
    assert result is True

    restored = handle.get_main_pointer("doc-1", "parse")
    assert restored is not None
    assert restored["semantic_id"] == "parse_old"
    assert restored["technical_id"] == "old-hash"


def test_main_pointer_restore_returns_false_for_incomplete() -> None:
    """Incomplete snapshots (missing semantic_id or technical_id) return False.

    Mirrors test_version_compatibility.py:433.
    """
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        KBMainPointerSnapshot,
    )

    handle = make_handle("mp_incomplete_coll")

    missing_semantic = KBMainPointerSnapshot(
        collection="mp_incomplete_coll",
        doc_id="doc-1",
        step_type="parse",
        model_tag=None,
        pointer={"technical_id": "parse-hash"},
    )
    missing_technical = KBMainPointerSnapshot(
        collection="mp_incomplete_coll",
        doc_id="doc-1",
        step_type="parse",
        model_tag=None,
        pointer={"semantic_id": "parse_manual_hash"},
    )

    assert not handle.restore_main_pointer_snapshot(missing_semantic)
    assert not handle.restore_main_pointer_snapshot(missing_technical)
    assert handle.list_main_pointers() == []


def test_candidate_cleanup_snapshot_records_preview_counts() -> None:
    """capture_candidate_cleanup_snapshot records preview counts (preview_only=True).

    Mirrors test_version_compatibility.py:463.
    """
    handle, fake_vis = make_handle_with_cleanup_store(
        "candidate_snap_coll",
        cleanup_return={"chunks": 2, "embeddings": 5},
    )

    snapshot = handle.capture_candidate_cleanup_snapshot(
        "doc-1",
        "parse",
        new_parse_hash="new-hash",
        old_parse_hash="old-hash",
    )

    assert snapshot.cleanup_counts == {"chunks": 2, "embeddings": 5}
    assert snapshot.collection == "candidate_snap_coll"
    assert snapshot.doc_id == "doc-1"
    assert snapshot.scope == "parse"
    # preview_only must have been passed: no real deletion
    assert fake_vis.last_cleanup_args is not None


def test_candidate_cleanup_restore_marks_remaining_side_effects() -> None:
    """Executed candidate cleanup is reported as incomplete, not fake-restored.

    Mirrors test_version_compatibility.py:513.
    """
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        KBVersionCandidateCleanupSnapshot,
    )

    handle = make_handle("candidate_restore_coll")

    snapshot = KBVersionCandidateCleanupSnapshot(
        collection="candidate_restore_coll",
        doc_id="doc-1",
        scope="parse",
        cleanup_counts={"chunks": 2, "embeddings": 5},
        new_parse_hash="new-hash",
        old_parse_hash="old-hash",
    )

    result = handle.restore_candidate_cleanup_snapshot(snapshot, cleanup_executed=True)

    assert result.status == "incomplete"
    assert result.skipped
    assert result.reason == "candidate_cleanup_not_restorable"
    assert not result.restorable
    assert result.side_effects_may_remain
    assert result.cleanup_counts == {"chunks": 2, "embeddings": 5}
    assert result.warnings


def test_candidate_cleanup_restore_does_not_delete_active_artifacts() -> None:
    """Rollback-incomplete restore reports state without issuing more cleanup.

    Mirrors test_version_compatibility.py:542.
    """
    from xagent.core.tools.core.RAG_tools.kb.collection_handle import (
        KBVersionCandidateCleanupSnapshot,
    )

    handle, fake_vis = make_handle_with_cleanup_store("no_delete_coll")

    snapshot = KBVersionCandidateCleanupSnapshot(
        collection="no_delete_coll",
        doc_id="doc-1",
        scope="parse",
        cleanup_counts={"parses": 1, "chunks": 2},
    )
    result = handle.restore_candidate_cleanup_snapshot(snapshot, cleanup_executed=True)

    assert result.status == "incomplete"
    assert result.side_effects_may_remain
    # The fake store should NOT have been called for cleanup on restore
    assert fake_vis.last_cleanup_args is None
