"""Tests for the KB file compatibility facade."""

from __future__ import annotations

from inspect import signature


def test_kb_file_compatibility_public_surface_imports() -> None:
    """Given the KB package, the file facade is publicly importable."""
    import xagent.core.tools.core.RAG_tools.kb as kb

    assert hasattr(kb, "KBFileCompatibilityFacade")
    assert hasattr(kb.get_kb_coordinator(), "file_compatibility")


def test_kb_file_compatibility_methods_match_public_helper_signatures() -> None:
    """Given legacy helpers, facade methods preserve their call signatures."""
    from xagent.core.tools.core.RAG_tools.kb import KBFileCompatibilityFacade
    from xagent.web.services import kb_collection_service, kb_file_service

    facade = KBFileCompatibilityFacade()
    pairs = [
        (
            facade.upsert_uploaded_file_record,
            kb_file_service.upsert_uploaded_file_record,
        ),
        (facade.list_documents_for_user, kb_file_service.list_documents_for_user),
        (
            facade.list_document_records_for_file_ids,
            kb_file_service.list_document_records_for_file_ids,
        ),
        (
            facade.build_uploaded_filename_map,
            kb_file_service.build_uploaded_filename_map,
        ),
        (
            facade.get_document_record_file_id,
            kb_file_service.get_document_record_file_id,
        ),
        (facade.resolve_document_filename, kb_file_service.resolve_document_filename),
        (
            facade.delete_uploaded_file_if_orphaned,
            kb_file_service.delete_uploaded_file_if_orphaned,
        ),
        (
            facade.aggregate_uploaded_file_statuses,
            kb_file_service.aggregate_uploaded_file_statuses,
        ),
        (facade.reconcile_uploaded_files, kb_file_service.reconcile_uploaded_files),
        (
            facade.list_collection_uploaded_file_owner_ids,
            kb_collection_service.list_collection_uploaded_file_owner_ids,
        ),
        (
            facade.delete_collection_physical_dir,
            kb_collection_service.delete_collection_physical_dir,
        ),
        (
            facade.delete_collection_uploaded_files,
            kb_collection_service.delete_collection_uploaded_files,
        ),
        (
            facade.rename_collection_storage,
            kb_collection_service.rename_collection_storage,
        ),
    ]

    for facade_method, public_helper in pairs:
        assert signature(facade_method) == signature(public_helper)


def test_public_file_helper_delegates_through_facade(monkeypatch) -> None:
    """Given a public file helper call, it routes through the coordinator facade."""
    from xagent.web.services import kb_file_service

    class _FakeFacade:
        def get_document_record_file_id(self, record):
            assert record == {"file_id": "legacy"}
            return "facade-file-id"

    monkeypatch.setattr(
        kb_file_service,
        "_get_file_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert (
        kb_file_service.get_document_record_file_id({"file_id": "legacy"})
        == "facade-file-id"
    )


def test_public_collection_helper_delegates_through_facade(monkeypatch) -> None:
    """Given a public collection helper call, it routes through the facade."""
    from xagent.web.services import kb_collection_service

    class _FakeFacade:
        def list_collection_uploaded_file_owner_ids(self, db, *, collection_name: str):
            assert db == "db"
            assert collection_name == "kb"
            return {1, 2}

    monkeypatch.setattr(
        kb_collection_service,
        "_get_file_compatibility_facade",
        lambda: _FakeFacade(),
    )

    assert kb_collection_service.list_collection_uploaded_file_owner_ids(
        "db",
        collection_name="kb",
    ) == {1, 2}


def test_file_status_aggregation_impl_uses_status_private_impl(monkeypatch) -> None:
    """File private impl should not re-enter the public management wrapper."""
    from xagent.web.services import kb_file_service

    class FakeQuery:
        def where(self, _filter: str) -> "FakeQuery":
            return self

        def select(self, _columns: list[str]) -> "FakeQuery":
            return self

        def limit(self, _limit: int) -> "FakeQuery":
            return self

    class FakeTable:
        def search(self) -> FakeQuery:
            return FakeQuery()

    class FakeConnection:
        def open_table(self, table_name: str) -> FakeTable:
            assert table_name == "documents"
            return FakeTable()

    load_calls: list[dict[str, object]] = []

    def fake_load_ingestion_status_impl(**kwargs: object) -> list[dict[str, object]]:
        load_calls.append(dict(kwargs))
        return [{"doc_id": "doc-1", "status": "success"}]

    monkeypatch.setattr(kb_file_service, "get_connection_from_env", FakeConnection)
    monkeypatch.setattr(kb_file_service, "ensure_documents_table", lambda _conn: None)
    monkeypatch.setattr(
        kb_file_service,
        "query_to_list",
        lambda _query: [{"file_id": "file-1", "collection": "kb", "doc_id": "doc-1"}],
    )
    monkeypatch.setattr(
        kb_file_service,
        "_load_ingestion_status_impl",
        fake_load_ingestion_status_impl,
    )
    monkeypatch.setattr(
        kb_file_service, "_load_indexed_doc_refs", lambda *_, **__: set()
    )

    result = kb_file_service._aggregate_uploaded_file_statuses_impl(
        file_ids=["file-1"],
        user_id=7,
        is_admin=False,
        use_cache=False,
    )

    assert result == {"file-1": "SUCCESS"}
    assert load_calls == [
        {"collection": "kb", "user_id": 7, "is_admin": False},
    ]


def test_collection_uploaded_file_cleanup_impl_uses_file_private_impl(
    monkeypatch,
) -> None:
    """Collection private impl should not re-enter the public file wrapper."""
    from xagent.web.services import kb_collection_service

    calls: list[dict[str, object]] = []
    commit_calls: list[bool] = []

    class FakeDb:
        def commit(self) -> None:
            commit_calls.append(True)

    fake_db = FakeDb()

    def fake_delete_uploaded_file_if_orphaned_impl(
        db: object,
        *,
        file_id: str,
        user_id: int,
        remaining_file_ids: set[str],
        after_commit: list[object],
    ) -> bool:
        calls.append(
            {
                "db": db,
                "file_id": file_id,
                "user_id": user_id,
                "remaining_file_ids": set(remaining_file_ids),
            }
        )
        return file_id == "file-a"

    monkeypatch.setattr(
        kb_collection_service,
        "_delete_uploaded_file_if_orphaned_impl",
        fake_delete_uploaded_file_if_orphaned_impl,
    )

    result = kb_collection_service._delete_collection_uploaded_files_impl(
        fake_db,
        user_id=5,
        collection_file_ids={"file-a", "file-b"},
        remaining_file_ids={"file-b"},
        collection_dir=None,
        after_commit=[],
    )

    assert result == 1
    assert commit_calls == []
    assert sorted(calls, key=lambda call: str(call["file_id"])) == [
        {
            "db": fake_db,
            "file_id": "file-a",
            "user_id": 5,
            "remaining_file_ids": {"file-b"},
        },
        {
            "db": fake_db,
            "file_id": "file-b",
            "user_id": 5,
            "remaining_file_ids": {"file-b"},
        },
    ]
