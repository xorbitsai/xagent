"""Tests for the public vector-storage helper surface.

After #510 the storage mechanics (chunk-selection read, vector write/upsert,
model-tag routing, dimension checks, spill-retry, index status) live in the
coordinator-owned collection handle, and the merge-error classifier lives in the
storage layer. These module-level helpers are now thin wrappers that delegate to
the ``KBVectorStorageCompatibilityFacade``.

This file therefore covers only the surviving public contract:
- ``validate_query_vector`` rules (pure validation);
- ``read_chunks_for_embedding`` / ``write_vectors_to_db`` behavior end-to-end
  through the real coordinator + LanceDB store.

The moved mechanics are covered at their new seams:
- handle write/read mechanics: ``kb/test_collection_handle_embedding.py``
- merge fallback / classifier: ``storage/test_lancedb_stores.py``
- column/idempotency/cleanup oracle: ``test_embedding_lifecycle_characterization.py``

Storage isolation/reset is provided by the autouse ``isolate_rag_storage``
fixture in ``tests/conftest.py``.
"""

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VectorValidationError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    ChunkEmbeddingData,
    EmbeddingReadResponse,
    EmbeddingWriteResponse,
)
from xagent.core.tools.core.RAG_tools.vector_storage.vector_manager import (
    read_chunks_for_embedding,
    validate_query_vector,
    write_vectors_to_db,
)


class TestReadChunksForEmbedding:
    def test_read_chunks_no_data(self) -> None:
        result = read_chunks_for_embedding(
            collection="coll",
            doc_id="nonexistent_doc",
            parse_hash="nonexistent_hash",
            model="test_model",
            is_admin=True,
        )
        assert isinstance(result, EmbeddingReadResponse)
        assert result.chunks == []
        assert result.total_count == 0
        assert result.pending_count == 0


class TestWriteVectorsToDb:
    def test_concurrent_writes_same_collection_are_serialized(self) -> None:
        """Regression (PR #915): concurrent page ingestions write to the same
        LanceDB table on separate threads; write_vectors_to_db must serialize
        them per collection so overlapping commits can't raise CommitConflict.
        """
        import threading
        import time
        from unittest.mock import MagicMock, patch

        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            reset_locks_for_testing,
        )

        reset_locks_for_testing()

        state_lock = threading.Lock()
        active = 0
        max_active = 0

        def _fake_write(**kwargs: object) -> EmbeddingWriteResponse:
            nonlocal active, max_active
            with state_lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.02)
            with state_lock:
                active -= 1
            return EmbeddingWriteResponse(
                upsert_count=0, deleted_stale_count=0, index_status="skipped"
            )

        facade = MagicMock()
        facade.write_vectors_to_db.side_effect = _fake_write

        with patch(
            "xagent.core.tools.core.RAG_tools.vector_storage.vector_manager"
            "._get_vector_storage_compatibility_facade",
            return_value=facade,
        ):

            def _writer() -> None:
                write_vectors_to_db(collection="same", embeddings=[])

            threads = [threading.Thread(target=_writer) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        assert facade.write_vectors_to_db.call_count == 4
        # Serialized: never two writers inside the facade call at once.
        assert max_active == 1

    def test_write_vectors_empty_list(self) -> None:
        result = write_vectors_to_db(
            collection="coll", embeddings=[], create_index=True
        )
        assert isinstance(result, EmbeddingWriteResponse)
        assert result.upsert_count == 0
        assert result.deleted_stale_count == 0
        assert result.index_status == "skipped"

    def test_write_then_read_round_trip(self) -> None:
        # Seed a chunk so read-for-embedding can find it, then embed it and
        # confirm it is excluded from the next pending read (end-to-end through
        # the real coordinator + store).
        from datetime import datetime, timezone

        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_vector_index_store,
        )
        from xagent.core.tools.core.RAG_tools.utils.metadata_utils import (
            serialize_metadata,
        )

        get_vector_index_store().upsert_chunks(
            [
                {
                    "collection": "coll",
                    "doc_id": "d1",
                    "parse_hash": "h1",
                    "chunk_id": "c0",
                    "index": 0,
                    "text": "hello",
                    "page_number": None,
                    "section": None,
                    "anchor": None,
                    "json_path": None,
                    "chunk_hash": "ch-c0",
                    "config_hash": "cfg1",
                    "created_at": datetime.now(timezone.utc),
                    "metadata": serialize_metadata({"k": "v"}),
                    "user_id": None,
                }
            ]
        )

        write_result = write_vectors_to_db(
            collection="coll",
            embeddings=[
                ChunkEmbeddingData(
                    doc_id="d1",
                    chunk_id="c0",
                    parse_hash="h1",
                    model="test_model",
                    vector=[0.1, 0.2, 0.3],
                    text="hello",
                    chunk_hash="ch-c0",
                    metadata=None,
                )
            ],
            create_index=False,
        )
        assert write_result.upsert_count == 1

        read_result = read_chunks_for_embedding(
            collection="coll",
            doc_id="d1",
            parse_hash="h1",
            model="test_model",
            is_admin=True,
        )
        assert read_result.total_count == 1
        assert read_result.pending_count == 0


class TestVectorValidation:
    """Validation of the public ``validate_query_vector`` contract."""

    def test_validate_query_vector_valid(self) -> None:
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([0.5, -0.5, 0.0])
        validate_query_vector([1, 2, 3])  # integers are valid

    def test_validate_query_vector_invalid_type(self) -> None:
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector("not a list")  # type: ignore[arg-type]
        with pytest.raises(VectorValidationError, match="query_vector must be a list"):
            validate_query_vector(None)  # type: ignore[arg-type]

    def test_validate_query_vector_empty(self) -> None:
        with pytest.raises(VectorValidationError, match="query_vector cannot be empty"):
            validate_query_vector([])

    def test_validate_query_vector_invalid_elements(self) -> None:
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, "invalid", 3.0])  # type: ignore[list-item]
        with pytest.raises(
            VectorValidationError, match="query_vector must contain only numbers"
        ):
            validate_query_vector([1.0, None, 3.0])  # type: ignore[list-item]

    def test_validate_query_vector_nan_infinity(self) -> None:
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("nan"), 3.0])
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, float("inf"), 3.0])
        with pytest.raises(
            VectorValidationError, match="query_vector contains invalid values"
        ):
            validate_query_vector([1.0, -float("inf"), 3.0])

    def test_validate_query_vector_numpy_scalar_types(self) -> None:
        np = pytest.importorskip("numpy")
        validate_query_vector(
            [np.float64(1.0), np.float32(2.0), np.int32(3)]  # type: ignore[list-item]
        )
        validate_query_vector(
            [np.float64(0.5), np.float32(-0.5), np.int64(0)]  # type: ignore[list-item]
        )
        validate_query_vector([np.float64(1.0), 2.0, np.int32(3)])  # type: ignore[list-item]

    def test_validate_without_connection(self) -> None:
        # Backward-compatible signature: model_tag/conn accepted, ignored.
        validate_query_vector([1.0, 2.0, 3.0])
        validate_query_vector([1.0, 2.0, 3.0], model_tag="test_model")


def _bound_collection_with_a_chunk(model: str | None = "m-old") -> None:
    import asyncio
    from datetime import datetime, timezone

    from xagent.core.tools.core.RAG_tools.core.schemas import CollectionInfo
    from xagent.core.tools.core.RAG_tools.management.collection_manager import (
        collection_manager,
    )
    from xagent.core.tools.core.RAG_tools.storage.factory import (
        get_vector_index_store,
    )

    asyncio.run(
        collection_manager.save_collection(
            CollectionInfo(name="kb", embedding_model_id=model, embedding_dimension=3)
        )
    )
    get_vector_index_store().upsert_chunks(
        [
            {
                "collection": "kb",
                "doc_id": "d1",
                "parse_hash": "h1",
                "chunk_id": "c0",
                "index": 0,
                "text": "hello",
                "page_number": None,
                "section": None,
                "anchor": None,
                "json_path": None,
                "chunk_hash": "ch-c0",
                "config_hash": "cfg1",
                "created_at": datetime.now(timezone.utc),
                "metadata": None,
                "user_id": None,
            }
        ]
    )


def _write_batch(
    create_index: bool = False, *, verify_bound: bool = True
) -> EmbeddingWriteResponse:
    return write_vectors_to_db(
        collection="kb",
        embeddings=[
            ChunkEmbeddingData(
                doc_id="d1",
                chunk_id="c0",
                parse_hash="h1",
                model="m-old",
                vector=[0.1, 0.2, 0.3],
                text="hello",
                chunk_hash="ch-c0",
                metadata=None,
            )
        ],
        create_index=create_index,
        verify_bound=verify_bound,
    )


def _vectorless_chunks(model: str = "m-old") -> int:
    return read_chunks_for_embedding(
        collection="kb", doc_id="d1", parse_hash="h1", model=model, is_admin=True
    ).pending_count


def _freeze_from_another_connection() -> None:
    import os
    from datetime import datetime

    import lancedb

    from xagent.core.tools.core.RAG_tools.management import migration_state

    migration_state._merge_row(
        lancedb.connect(os.environ["LANCEDB_DIR"]),
        {
            "name": "kb",
            "state": migration_state.FROZEN,
            "source_model_id": "m-old",
            "target_model_id": None,
            "updated_at": datetime.now(),
        },
    )


_SWITCH_IN_ANOTHER_PROCESS = """
import sys

from xagent.core.tools.core.RAG_tools.management import collection_manager
from xagent.core.tools.core.RAG_tools.management import migration_state as ms

ms.write_migration_state(
    "kb", ms.CollectionMigrationState(state=ms.FROZEN, source_model_id="m-old")
)
if sys.argv[1] == "switch":
    assert collection_manager.write_collection_binding_sync(
        "kb", "m-new", 3, expected_model_id="m-old"
    )
    ms.clear_migration_state("kb")
"""


class TestWriteAfterMigration:
    """Real LanceDB: a refused write raises and leaves its rows in place."""

    def test_untouched_collection_keeps_the_batch(self) -> None:
        _bound_collection_with_a_chunk()

        assert _write_batch().upsert_count == 1
        assert _vectorless_chunks() == 0

    def test_a_collection_without_a_binding_keeps_the_batch(self) -> None:
        _bound_collection_with_a_chunk(model=None)

        assert _write_batch().upsert_count == 1
        assert _vectorless_chunks() == 0

    def test_a_frozen_collection_without_a_binding_refuses_the_batch(self) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )

        _bound_collection_with_a_chunk(model=None)
        _freeze_from_another_connection()

        with pytest.raises(CollectionMigratingError, match="frozen"):
            _write_batch()

    def test_another_attempts_rows_survive_a_refused_batch_of_the_document(
        self,
    ) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )

        _bound_collection_with_a_chunk()
        assert _write_batch().upsert_count == 1
        _freeze_from_another_connection()

        # A stalled attempt of the same document wakes up and commits the row.
        with pytest.raises(CollectionMigratingError):
            _write_batch()
        assert _vectorless_chunks() == 0

    # The last batch of an ingestion builds the index before the guard reads.
    @pytest.mark.parametrize("create_index", [False, True])
    def test_frozen_collection_refuses_the_batch(self, create_index: bool) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )

        _bound_collection_with_a_chunk()
        _freeze_from_another_connection()

        with pytest.raises(CollectionMigratingError, match="frozen"):
            _write_batch(create_index)
        assert _vectorless_chunks() == 0

    @pytest.mark.parametrize("create_index", [False, True])
    def test_moved_binding_refuses_the_batch(self, create_index: bool) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )
        from xagent.core.tools.core.RAG_tools.management.collection_manager import (
            write_collection_binding_sync,
        )
        from xagent.core.tools.core.RAG_tools.management.migration_state import (
            clear_migration_state,
        )

        _bound_collection_with_a_chunk()
        _freeze_from_another_connection()
        assert write_collection_binding_sync(
            "kb", "m-new", 3, expected_model_id="m-old"
        )
        # Unfrozen by deleting the row: it then reads like a never-migrated one.
        clear_migration_state("kb")

        with pytest.raises(CollectionMigratingError, match="m-new"):
            _write_batch(create_index)
        assert _vectorless_chunks() == 0
        # The retry embeds under the new binding; the old row does not count.
        assert _vectorless_chunks("m-new") == 1

    def test_unreadable_migration_row_refuses_the_batch(self) -> None:
        import pyarrow as pa

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )
        from xagent.core.tools.core.RAG_tools.management.migration_state import (
            MIGRATION_TABLE,
        )
        from xagent.core.tools.core.RAG_tools.storage.factory import (
            get_metadata_store,
        )

        _bound_collection_with_a_chunk()
        get_metadata_store().get_raw_connection().create_table(
            MIGRATION_TABLE, schema=pa.schema([("other", pa.string())])
        )

        with pytest.raises(CollectionMigratingError, match="migration row read: "):
            _write_batch()
        assert _vectorless_chunks() == 0

    def test_unreadable_binding_refuses_the_batch(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )
        from xagent.core.tools.core.RAG_tools.management import collection_manager

        _bound_collection_with_a_chunk()

        def _unreadable(name: str) -> None:
            raise OSError("metadata store gone")

        monkeypatch.setattr(
            collection_manager, "get_collection_if_present_sync", _unreadable
        )

        with pytest.raises(CollectionMigratingError) as raised:
            _write_batch()
        assert "could not be checked (binding read: OSError)" in str(raised.value)
        assert "metadata store gone" not in str(raised.value)
        assert any(
            "metadata store gone" in str(record.exc_info[1])
            for record in caplog.records
            if record.exc_info
        )
        assert _vectorless_chunks() == 0

    def test_an_unverified_write_ignores_a_frozen_collection(self) -> None:
        _bound_collection_with_a_chunk()
        _freeze_from_another_connection()

        assert _write_batch(verify_bound=False).upsert_count == 1
        assert _vectorless_chunks() == 0

    @pytest.mark.parametrize("change", ["freeze", "switch"])
    def test_a_change_made_by_another_process_is_seen(self, change: str) -> None:
        import os
        import subprocess
        import sys
        from pathlib import Path

        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )

        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[6] / "src")
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH")]))
        _bound_collection_with_a_chunk()
        # Warms this process's cached connection before the other process writes.
        assert _write_batch().upsert_count == 1
        subprocess.run(
            [sys.executable, "-c", _SWITCH_IN_ANOTHER_PROCESS, change],
            env=env,
            check=True,
            capture_output=True,
            timeout=120,
        )

        with pytest.raises(
            CollectionMigratingError, match="frozen" if change == "freeze" else "m-new"
        ):
            _write_batch()

    def test_a_switch_finishing_between_the_two_reads_is_caught(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            CollectionMigratingError,
        )
        from xagent.core.tools.core.RAG_tools.kb.vector_storage_compatibility import (
            KBVectorStorageCompatibilityFacade,
        )
        from xagent.core.tools.core.RAG_tools.management import (
            collection_manager,
            migration_state,
        )

        _bound_collection_with_a_chunk()
        _freeze_from_another_connection()
        written: list[bool] = []
        switched: list[bool] = []
        real_write = KBVectorStorageCompatibilityFacade.write_vectors_to_db

        def _write(self: object, **kwargs: object) -> object:
            response = real_write(self, **kwargs)  # type: ignore[arg-type]
            written.append(True)
            return response

        def _then_switch(read):  # type: ignore[no-untyped-def]
            def _read(*args: object, **kwargs: object) -> object:
                result = read(*args, **kwargs)
                if written and not switched:
                    switched.append(True)
                    collection_manager.write_collection_binding_sync(
                        "kb", "m-new", 3, expected_model_id="m-old"
                    )
                    migration_state.clear_migration_state("kb")
                return result

            return _read

        monkeypatch.setattr(
            KBVectorStorageCompatibilityFacade, "write_vectors_to_db", _write
        )
        # Whichever the guard reads first, the switch completes right after it.
        monkeypatch.setattr(
            migration_state,
            "read_migration_state",
            _then_switch(migration_state.read_migration_state),
        )
        monkeypatch.setattr(
            collection_manager,
            "get_collection_if_present_sync",
            _then_switch(collection_manager.get_collection_if_present_sync),
        )

        with pytest.raises(CollectionMigratingError):
            _write_batch()
        assert switched == [True]
        assert _vectorless_chunks() == 0
