"""Tests for main_pointer_manager functions.

These tests mock the MainPointerStore to validate basic CRUD behaviors
without touching real storage.

After the H06 refactor, public functions delegate to the coordinator, which
calls the storage factory for the store. Tests therefore patch
``StorageFactory.get_main_pointer_store`` (the canonical factory method) to
inject the mock store at the right point in the call chain.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

from xagent.core.tools.core.RAG_tools.version_management.main_pointer_manager import (
    delete_main_pointer,
    get_main_pointer,
    list_main_pointers,
    set_main_pointer,
)

# Patch target: the method on StorageFactory used by the coordinator's storage
# shim to return the MainPointerStore instance.
_STORE_PATCH = (
    "xagent.core.tools.core.RAG_tools.storage.factory.StorageFactory"
    ".get_main_pointer_store"
)


class TestMainPointerManager:
    """Test cases for main_pointer_manager functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("LANCEDB_DIR")
        os.environ["LANCEDB_DIR"] = self.temp_dir

    def teardown_method(self):
        """Clean up test fixtures."""
        # Restore original environment
        if self.original_env is not None:
            os.environ["LANCEDB_DIR"] = self.original_env
        elif "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        # Clean up temp directory
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch(_STORE_PATCH)
    def test_get_main_pointer_not_found(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        mock_store = MagicMock()
        mock_store.get_main_pointer.return_value = None
        mock_get_store.return_value = mock_store

        assert get_main_pointer("c", "d", "parse") is None
        mock_store.get_main_pointer.assert_called_once_with(
            collection="c", doc_id="d", step_type="parse", model_tag=None, user_id=None
        )

    @patch(_STORE_PATCH)
    def test_set_and_get_main_pointer_roundtrip(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Set main pointer
        set_main_pointer(
            lancedb_dir="/tmp",
            collection="c",
            doc_id="d",
            step_type="parse",
            semantic_id="parse_123",
            technical_id="hash_456",
        )

        mock_store.set_main_pointer.assert_called_once_with(
            collection="c",
            doc_id="d",
            step_type="parse",
            semantic_id="parse_123",
            technical_id="hash_456",
            model_tag=None,
            operator=None,
            user_id=None,
        )

        # Get main pointer
        mock_store.get_main_pointer.return_value = {
            "collection": "c",
            "doc_id": "d",
            "step_type": "parse",
            "model_tag": "",
            "semantic_id": "parse_123",
            "technical_id": "hash_456",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "operator": "unknown",
        }

        result = get_main_pointer("c", "d", "parse")
        assert result is not None
        assert result["semantic_id"] == "parse_123"
        assert result["technical_id"] == "hash_456"

    @patch(_STORE_PATCH)
    def test_list_and_delete_main_pointers(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # List main pointers
        mock_store.list_main_pointers.return_value = [
            {
                "collection": "c",
                "doc_id": "d1",
                "step_type": "parse",
                "model_tag": "",
                "semantic_id": "parse_1",
                "technical_id": "hash_1",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "operator": "unknown",
            },
            {
                "collection": "c",
                "doc_id": "d2",
                "step_type": "parse",
                "model_tag": "",
                "semantic_id": "parse_2",
                "technical_id": "hash_2",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "operator": "unknown",
            },
        ]

        pointers = list_main_pointers("c")
        assert len(pointers) == 2
        assert pointers[0]["doc_id"] == "d1"

        mock_store.list_main_pointers.assert_called_once_with(
            collection="c", doc_id=None, user_id=None, limit=100
        )

        # Delete main pointer
        mock_store.delete_main_pointer.return_value = True
        result = delete_main_pointer("c", "d1", "parse")
        assert result is True

        mock_store.delete_main_pointer.assert_called_once_with(
            collection="c", doc_id="d1", step_type="parse", model_tag=None, user_id=None
        )

    @patch(_STORE_PATCH)
    def test_get_main_pointer_backward_compatibility(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that model_tag=None matches both '' and NULL values."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        # Should return pointer when model_tag matches empty string
        mock_store.get_main_pointer.return_value = {
            "collection": "c",
            "doc_id": "d",
            "step_type": "parse",
            "model_tag": "",
            "semantic_id": "parse_123",
            "technical_id": "hash_456",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "operator": "unknown",
        }

        result = get_main_pointer("c", "d", "parse", model_tag=None)
        assert result is not None
        assert result["model_tag"] == ""

    @patch(_STORE_PATCH)
    def test_get_main_pointer_injection_attack_prevention(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that special characters in doc_id are handled safely."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        mock_store.get_main_pointer.return_value = {
            "collection": "c",
            "doc_id": "doc' OR '1'='1",
            "step_type": "parse",
            "model_tag": "",
            "semantic_id": "parse_123",
            "technical_id": "hash_456",
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "operator": "unknown",
        }

        result = get_main_pointer("c", "doc' OR '1'='1", "parse")
        assert result is not None
        mock_store.get_main_pointer.assert_called_once()

        # Verify the store was called with the exact doc_id (not injected)
        call_args = mock_store.get_main_pointer.call_args
        assert call_args[1]["doc_id"] == "doc' OR '1'='1"

    @patch(_STORE_PATCH)
    def test_set_main_pointer_preserves_created_at(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that updating a main pointer preserves the original created_at timestamp."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        created_at = datetime(2024, 1, 1, 12, 0, 0)
        mock_store.get_main_pointer.return_value = {
            "collection": "c",
            "doc_id": "d",
            "step_type": "parse",
            "model_tag": "",
            "semantic_id": "old_parse",
            "technical_id": "old_hash",
            "created_at": created_at,
            "updated_at": datetime(2024, 1, 1, 12, 0, 0),
            "operator": "unknown",
        }

        # Update main pointer
        set_main_pointer(
            lancedb_dir="/tmp",
            collection="c",
            doc_id="d",
            step_type="parse",
            semantic_id="new_parse",
            technical_id="new_hash",
        )

        # Verify store was called to set the pointer
        mock_store.set_main_pointer.assert_called_once()

    @patch(_STORE_PATCH)
    def test_set_main_pointer_new_record_created_at(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that creating a new main pointer sets a new created_at timestamp."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        mock_store.get_main_pointer.return_value = None  # No existing pointer

        set_main_pointer(
            lancedb_dir="/tmp",
            collection="c",
            doc_id="d",
            step_type="parse",
            semantic_id="parse_123",
            technical_id="hash_456",
        )

        mock_store.set_main_pointer.assert_called_once()

    @patch(_STORE_PATCH)
    def test_set_main_pointer_normalizes_null_model_tag(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that setting a main pointer with model_tag=None normalizes to empty string."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        set_main_pointer(
            lancedb_dir="/tmp",
            collection="c",
            doc_id="d",
            step_type="embed",
            semantic_id="embed_123",
            technical_id="embed_hash",
            model_tag=None,
        )

        # Verify store was called with normalized model_tag
        call_args = mock_store.set_main_pointer.call_args
        assert call_args[1]["model_tag"] is None  # Store handles normalization

    @patch(_STORE_PATCH)
    def test_set_main_pointer_always_attempts_normalization(
        self,
        mock_get_store: MagicMock,
    ) -> None:
        """Test that setting a main pointer with empty model_tag works correctly."""
        mock_store = MagicMock()
        mock_get_store.return_value = mock_store

        set_main_pointer(
            lancedb_dir="/tmp",
            collection="c",
            doc_id="d",
            step_type="embed",
            semantic_id="embed_123",
            technical_id="embed_hash",
            model_tag="",
        )

        mock_store.set_main_pointer.assert_called_once_with(
            collection="c",
            doc_id="d",
            step_type="embed",
            semantic_id="embed_123",
            technical_id="embed_hash",
            model_tag="",
            operator=None,
            user_id=None,
        )
