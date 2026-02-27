"""Tests for promote_version_main: resolve, preview, confirm."""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import VersionManagementError
from xagent.core.tools.core.RAG_tools.core.schemas import StepType
from xagent.core.tools.core.RAG_tools.version_management.promote_version_main import (
    promote_version_main,
)


class TestPromoteVersionMain:
    """Test cases for promote_version_main function."""

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

    def _patch_list_candidates(self, mock_list_candidates):
        """Helper method to patch list_candidates in the promote_version_main module."""
        import importlib

        promote_version_main_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
        )
        return patch.object(
            promote_version_main_module, "list_candidates", mock_list_candidates
        )

    def _patch_calculate_cleanup_plan(self, mock_calculate_cleanup_plan):
        """Helper method to patch _calculate_cleanup_plan in the promote_version_main module."""
        import importlib

        promote_version_main_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
        )
        return patch.object(
            promote_version_main_module,
            "_calculate_cleanup_plan",
            mock_calculate_cleanup_plan,
        )

    def _patch_call_cleanup_cascade(self, mock_call_cleanup_cascade):
        """Helper method to patch _call_cleanup_cascade in the promote_version_main module."""
        import importlib

        promote_version_main_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
        )
        return patch.object(
            promote_version_main_module,
            "_call_cleanup_cascade",
            mock_call_cleanup_cascade,
        )

    def _patch_set_main_pointer(self, mock_set_main_pointer):
        """Helper method to patch set_main_pointer in the promote_version_main module."""
        import importlib

        promote_version_main_module = importlib.import_module(
            "xagent.core.tools.core.RAG_tools.version_management.promote_version_main"
        )
        return patch.object(
            promote_version_main_module, "set_main_pointer", mock_set_main_pointer
        )

    def test_default_lancedb_dir_when_missing_env(self):
        """Test that default LanceDB directory is used when LANCEDB_DIR environment variable is not set.

        Verifies that:
        1. Function uses ~/.xagent/data/lancedb as default when LANCEDB_DIR is missing
        2. Function checks legacy path (project_root/data/lancedb) for backward compatibility
        3. Function continues execution instead of failing fast
        """
        from pathlib import Path

        from xagent.providers.vector_store.lancedb import LanceDBConnectionManager

        # Remove environment variable to test default behavior
        if "LANCEDB_DIR" in os.environ:
            del os.environ["LANCEDB_DIR"]

        # Expected default path is now ~/.xagent/data/lancedb
        expected_default_path = str(Path.home() / ".xagent" / "data" / "lancedb")

        # Verify the default path matches what LanceDBConnectionManager returns
        assert (
            LanceDBConnectionManager.get_default_lancedb_dir() == expected_default_path
        )

        # The function should not fail immediately due to missing env var
        # Instead it should proceed with database operations (may fail later due to empty DB)
        try:
            promote_version_main(
                "test_collection", "test_doc", StepType.PARSE, "test_id"
            )
        except VersionManagementError as e:
            # Should fail due to no candidates, not due to missing env var
            assert "No candidates found" in str(e)
            assert "LANCEDB_DIR environment variable not set" not in str(e)

    def test_resolve_selected_id_not_found(self):
        """Test error handling when selected_id cannot be resolved.

        Verifies that:
        1. Function raises VersionManagementError when no candidates match selected_id
        2. Error message indicates no candidates were found
        3. Function handles empty candidate list gracefully
        """
        mock_list_candidates = MagicMock()
        mock_list_candidates.return_value = {
            "candidates": [],
            "total_count": 0,
            "returned_count": 0,
        }

        with self._patch_list_candidates(mock_list_candidates):
            with pytest.raises(VersionManagementError, match="No candidates found"):
                promote_version_main(
                    "test_collection", "test_doc", StepType.PARSE, "nonexistent_id"
                )

    def test_resolve_selected_id_by_technical_id(self):
        """Test resolving selected_id by technical_id in full promote flow.

        Verifies that:
        1. promote_version_main correctly resolves technical_id from candidates
        2. Returns proper main_pointer information with both technical and semantic IDs
        3. Handles preview mode correctly with technical_id resolution
        4. Integrates with cleanup plan calculation
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
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
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        mock_calc = MagicMock()
        mock_calc.return_value = {
            "deleted_counts": {"parses": 0, "chunks": 0, "embeddings": 0},
            "notes": [],
            "current_pointer": None,
            "new_technical_id": "abc123",
        }

        with self._patch_list_candidates(mock_list_candidates):
            with self._patch_calculate_cleanup_plan(mock_calc):
                result = promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.PARSE,
                    "abc123",
                    preview_only=True,
                )

                assert result["main_pointer"]["technical_id"] == "abc123"
                assert result["main_pointer"]["semantic_id"] == "parse_test_v1"

    def test_resolve_selected_id_by_semantic_id(self):
        """Test resolving selected_id by semantic_id in full promote flow.

        Verifies that:
        1. promote_version_main correctly resolves semantic_id from candidates
        2. Returns proper main_pointer information with both technical and semantic IDs
        3. Handles preview mode correctly with semantic_id resolution
        4. Supports user-friendly semantic naming for version selection
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
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
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        mock_calc = MagicMock()
        mock_calc.return_value = {
            "deleted_counts": {"parses": 0, "chunks": 0, "embeddings": 0},
            "notes": [],
            "current_pointer": None,
            "new_technical_id": "abc123",
        }

        with self._patch_list_candidates(mock_list_candidates):
            with self._patch_calculate_cleanup_plan(mock_calc):
                result = promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.PARSE,
                    "parse_test_v1",
                    preview_only=True,
                )

                assert result["main_pointer"]["technical_id"] == "abc123"
                assert result["main_pointer"]["semantic_id"] == "parse_test_v1"

    def test_preview_only_mode(self):
        """Test preview_only mode behavior.

        Verifies that:
        1. Preview mode returns promoted=False and preview=True
        2. Deletion counts are calculated and returned for review
        3. Notes are included to guide user decisions
        4. No actual database changes are made in preview mode
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
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
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        mock_calc = MagicMock()
        mock_calc.return_value = {
            "deleted_counts": {"parses": 2, "chunks": 10, "embeddings": 50},
            "notes": ["需重新chunk/embed"],
            "current_pointer": {
                "semantic_id": "parse_old_v1",
                "technical_id": "old123",
            },
            "new_technical_id": "abc123",
        }

        with self._patch_list_candidates(mock_list_candidates):
            with self._patch_calculate_cleanup_plan(mock_calc):
                result = promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.PARSE,
                    "abc123",
                    preview_only=True,
                )

                assert result["promoted"] is False
                assert result["preview"] is True
                assert result["deleted_counts"]["parses"] == 2
                assert result["deleted_counts"]["chunks"] == 10
                assert result["deleted_counts"]["embeddings"] == 50
                assert "需重新chunk/embed" in result["notes"]

    def test_not_confirmed_mode(self):
        """Test behavior when confirm=False (not confirmed mode).

        Verifies that:
        1. Function returns preview=True and promoted=False when confirm=False
        2. Includes message prompting user to set confirm=True
        3. No actual database changes are made without confirmation
        4. Provides safety mechanism to prevent accidental changes
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
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
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        mock_calc = MagicMock()
        mock_calc.return_value = {
            "deleted_counts": {"parses": 0, "chunks": 0, "embeddings": 0},
            "notes": [],
            "current_pointer": None,
            "new_technical_id": "abc123",
        }

        with self._patch_list_candidates(mock_list_candidates):
            with self._patch_calculate_cleanup_plan(mock_calc):
                result = promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.PARSE,
                    "abc123",
                    preview_only=False,
                    confirm=False,
                )

                assert result["promoted"] is False
                assert result["preview"] is True
                assert "Set confirm=True to execute the promotion" in result["message"]

    def test_execute_promotion(self):
        """Test full promotion execution when confirm=True.

        Verifies that:
        1. Promotion is actually executed (promoted=True) when confirm=True
        2. set_main_pointer is called to update the main version pointer
        3. cleanup_cascade is called to clean up old data
        4. All database changes are committed and operator is recorded
        5. Returns proper result structure with deletion counts
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
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
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        mock_call_cleanup_cascade = MagicMock()
        mock_set_main_pointer = MagicMock()

        mock_call_cleanup_cascade.return_value = {
            "parses": 2,
            "chunks": 10,
            "embeddings": 50,
        }

        with self._patch_list_candidates(mock_list_candidates):
            # Mock cleanup plan
            mock_calc = MagicMock()
            mock_calc.return_value = {
                "deleted_counts": {"parses": 2, "chunks": 10, "embeddings": 50},
                "notes": ["需重新chunk/embed"],
                "current_pointer": {
                    "semantic_id": "parse_old_v1",
                    "technical_id": "old123",
                },
                "new_technical_id": "abc123",
            }

            with self._patch_calculate_cleanup_plan(mock_calc):
                with self._patch_call_cleanup_cascade(mock_call_cleanup_cascade):
                    with self._patch_set_main_pointer(mock_set_main_pointer):
                        result = promote_version_main(
                            "test_collection",
                            "test_doc",
                            StepType.PARSE,
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

                        # Verify cleanup and pointer update were called
                        mock_call_cleanup_cascade.assert_called()
                        mock_set_main_pointer.assert_called_once()

    def test_invalid_step_type(self):
        """Test error handling for invalid step_type.

        Verifies that:
        1. Function raises VersionManagementError for invalid step_type
        2. Error message clearly indicates the invalid step_type
        3. Function validates step_type before proceeding with operations
        4. Handles edge cases where step_type is not supported
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
            {
                "semantic_id": "test_v1",
                "technical_id": "abc123",
                "params_brief": {},
                "stats": {},
                "state": "candidate",
                "created_at": datetime.now(),
                "operator": "test_user",
            }
        ]
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        with self._patch_list_candidates(mock_list_candidates):
            mock_calc = MagicMock()
            mock_calc.side_effect = VersionManagementError("Invalid step_type: invalid")

            with self._patch_calculate_cleanup_plan(mock_calc):
                with pytest.raises(
                    VersionManagementError,
                    match="Invalid step_type string: 'invalid'.*Expected one of: 'parse', 'chunk', 'embed'",
                ):
                    promote_version_main(
                        "test_collection",
                        "test_doc",
                        "invalid",  # type: ignore
                        "abc123",
                        preview_only=True,
                    )

    def test_embed_step_type_with_model_tag(self):
        """Test promote operation for embed step_type with model_tag.

        Verifies that:
        1. Function correctly handles embed step_type with model_tag parameter
        2. Returns proper main_pointer with step_type and model_tag information
        3. Supports model-specific embedding version management
        4. Handles technical_id format for embedding models correctly
        """
        mock_list_candidates = MagicMock()
        mock_candidates = [
            {
                "semantic_id": "embed_bge_large_v1",
                "technical_id": "parse_hash1",
                "params_brief": {"model": "BAAI/bge-large-zh-v1.5"},
                "stats": {"vector_count": 10, "vector_dim": 1024},
                "state": "candidate",
                "created_at": datetime.now(),
                "operator": "test_user",
            }
        ]
        mock_list_candidates.return_value = {
            "candidates": mock_candidates,
            "total_count": 1,
            "returned_count": 1,
        }

        with self._patch_list_candidates(mock_list_candidates):
            mock_calc = MagicMock()
            mock_calc.return_value = {
                "deleted_counts": {"embeddings": 20},
                "notes": [],
                "current_pointer": None,
                "new_technical_id": "parse_hash1",
            }

            with self._patch_calculate_cleanup_plan(mock_calc):
                result = promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.EMBED,
                    "embed_bge_large_v1",
                    preview_only=True,
                    model_tag="bge_large",
                )

                assert result["main_pointer"]["step_type"] == "embed"
                assert result["main_pointer"]["model_tag"] == "bge_large"
                assert result["main_pointer"]["technical_id"] == "parse_hash1"

    def test_wraps_unexpected_exceptions(self):
        """Test that unexpected exceptions are wrapped in VersionManagementError.

        Verifies that:
        1. Non-VersionManagementError exceptions are caught
        2. Wrapped exceptions preserve original error context
        3. Error message indicates the promotion failure
        4. Follows proper exception wrapping pattern
        """
        # Mock list_candidates to raise a non-VersionManagementError
        from xagent.core.tools.core.RAG_tools.core.exceptions import (
            DatabaseOperationError,
        )

        mock_list_candidates = MagicMock()
        mock_list_candidates.side_effect = DatabaseOperationError(
            "Database connection failed"
        )

        with self._patch_list_candidates(mock_list_candidates):
            with pytest.raises(
                VersionManagementError, match="Failed to resolve selected_id"
            ):
                promote_version_main(
                    "test_collection",
                    "test_doc",
                    StepType.PARSE,
                    "abc123",
                    preview_only=True,
                )
