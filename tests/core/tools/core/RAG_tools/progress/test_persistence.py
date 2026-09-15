"""Unit tests for ProgressPersistence."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from xagent.core.tools.core.RAG_tools.core.exceptions import ProgressPersistenceError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    DocumentProcessingStatus,
    TaskProgress,
)
from xagent.core.tools.core.RAG_tools.progress.persistence import ProgressPersistence


class TestProgressPersistence:
    """Test ProgressPersistence functionality."""

    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.persistence = ProgressPersistence(storage_dir=self.temp_dir)

    def teardown_method(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test persistence initialization."""
        assert self.persistence.storage_dir == Path(self.temp_dir)
        assert self.persistence.storage_dir.exists()

    def test_save_and_load_task(self):
        """Test saving and loading a single task."""
        # Create a test task
        task = TaskProgress(
            task_id="test_task_123",
            task_type="ingestion",
            status=DocumentProcessingStatus.RUNNING,
            current_step="parse_document",
            overall_progress=0.5,
            start_time=1234567890.0,
            end_time=None,
            metadata={"collection": "docs", "pages": 10},
        )

        # Save task
        self.persistence.save_task_progress(task)

        # Check file was created
        task_file = self.persistence.storage_dir / "test_task_123.json"
        assert task_file.exists()

        # Load task
        loaded_task = self.persistence.load_task_progress("test_task_123")

        assert loaded_task is not None
        assert loaded_task.task_id == task.task_id
        assert loaded_task.task_type == task.task_type
        assert loaded_task.status == task.status
        assert loaded_task.current_step == task.current_step
        assert loaded_task.overall_progress == task.overall_progress
        assert loaded_task.start_time == task.start_time
        assert loaded_task.end_time == task.end_time
        assert loaded_task.metadata == task.metadata

    def test_save_multiple_tasks(self):
        """Test saving multiple tasks."""
        tasks = []

        for i in range(3):
            task = TaskProgress(
                task_id=f"multi_task_{i}",
                task_type="ingestion" if i % 2 == 0 else "search",
                status=DocumentProcessingStatus.RUNNING,
                current_step=f"step_{i}",
                overall_progress=i * 0.2,
                start_time=1234567890.0 + i,
                metadata={"index": i},
            )
            tasks.append(task)
            self.persistence.save_task_progress(task)

        # Check all files were created
        for i in range(3):
            task_file = self.persistence.storage_dir / f"multi_task_{i}.json"
            assert task_file.exists()

        # Load and verify all tasks
        for i, original_task in enumerate(tasks):
            loaded_task = self.persistence.load_task_progress(f"multi_task_{i}")
            assert loaded_task is not None
            assert loaded_task.task_id == original_task.task_id
            assert loaded_task.task_type == original_task.task_type

    def test_load_nonexistent_task(self):
        """Test loading a task that doesn't exist."""
        loaded_task = self.persistence.load_task_progress("nonexistent_task")
        assert loaded_task is None

    def test_delete_task(self):
        """Test deleting a task."""
        # Create and save a task
        task = TaskProgress(
            task_id="delete_test",
            task_type="ingestion",
            status=DocumentProcessingStatus.RUNNING,
        )
        self.persistence.save_task_progress(task)

        # Verify it exists
        task_file = self.persistence.storage_dir / "delete_test.json"
        assert task_file.exists()

        # Delete the task
        self.persistence.delete_task_progress("delete_test")

        # Verify it's gone
        assert not task_file.exists()
        assert self.persistence.load_task_progress("delete_test") is None

    def test_list_all_tasks(self):
        """Test listing all tasks."""
        # Create multiple tasks
        task_ids = ["list_test_1", "list_test_2", "list_test_3"]

        for task_id in task_ids:
            task = TaskProgress(
                task_id=task_id,
                task_type="ingestion",
                status=DocumentProcessingStatus.RUNNING,
            )
            self.persistence.save_task_progress(task)

        # List all tasks
        all_tasks = self.persistence.list_all_tasks()

        assert len(all_tasks) == 3
        returned_ids = [task.task_id for task in all_tasks]
        for task_id in task_ids:
            assert task_id in returned_ids

    def test_list_active_tasks(self):
        """Test listing only active tasks."""
        # Create tasks with different statuses
        tasks_data = [
            ("active_task", DocumentProcessingStatus.RUNNING),
            ("completed_task", DocumentProcessingStatus.SUCCESS),
            ("failed_task", DocumentProcessingStatus.FAILED),
            ("cancelled_task", DocumentProcessingStatus.CANCELLED),
            ("pending_task", DocumentProcessingStatus.PENDING),
        ]

        for task_id, status in tasks_data:
            task = TaskProgress(task_id=task_id, task_type="ingestion", status=status)
            self.persistence.save_task_progress(task)

        # List active tasks (should exclude SUCCESS, FAILED, CANCELLED)
        active_tasks = self.persistence.list_active_tasks()

        active_ids = [task.task_id for task in active_tasks]
        assert "active_task" in active_ids
        assert "pending_task" in active_ids
        assert "completed_task" not in active_ids
        assert "failed_task" not in active_ids
        assert "cancelled_task" not in active_ids

    def test_cleanup_old_tasks(self):
        """Test cleaning up old completed tasks."""
        import time

        # Create tasks with different completion times
        base_time = time.time()

        # Create a completed task (old)
        old_task = TaskProgress(
            task_id="old_completed",
            task_type="ingestion",
            status=DocumentProcessingStatus.SUCCESS,
            end_time=base_time - (25 * 3600),  # 25 hours ago
        )
        self.persistence.save_task_progress(old_task)

        # Create a recently completed task
        recent_task = TaskProgress(
            task_id="recent_completed",
            task_type="ingestion",
            status=DocumentProcessingStatus.SUCCESS,
            end_time=base_time - (1 * 3600),  # 1 hour ago
        )
        self.persistence.save_task_progress(recent_task)

        # Clean up tasks older than 24 hours
        deleted_count = self.persistence.cleanup_old_tasks(max_age_hours=24)

        # Should have deleted 1 old task
        assert deleted_count == 1

        # Check that old task is gone
        assert self.persistence.load_task_progress("old_completed") is None
        assert self.persistence.load_task_progress("recent_completed") is not None

    def test_long_task_id_roundtrip(self):
        """Overlong task IDs must still save, load and delete."""
        long_source = (
            "/" + "/".join(f"deep_segment_{i}" for i in range(30)) + "/doc.pdf"
        )
        task_id = f"ingest_docs_{long_source.replace('/', '_').replace('.', '_')}"
        assert len(task_id) >= 300

        task = TaskProgress(
            task_id=task_id,
            task_type="ingestion",
            status=DocumentProcessingStatus.RUNNING,
        )
        self.persistence.save_task_progress(task)

        file_path = self.persistence._get_task_file_path(task_id)
        assert len(file_path.name.encode("utf-8")) <= 255
        assert file_path.exists()

        loaded = self.persistence.load_task_progress(task_id)
        assert loaded is not None
        assert loaded.task_id == task_id

        assert self.persistence.delete_task_progress(task_id) is True
        assert self.persistence.load_task_progress(task_id) is None

    def test_long_task_ids_do_not_collide(self):
        """Two long IDs sharing a prefix must map to different files."""
        prefix = "ingest_docs_" + "x" * 300
        first, second = f"{prefix}_a", f"{prefix}_b"

        for task_id in (first, second):
            self.persistence.save_task_progress(
                TaskProgress(
                    task_id=task_id,
                    task_type="ingestion",
                    status=DocumentProcessingStatus.RUNNING,
                )
            )

        assert self.persistence._get_task_file_path(
            first
        ) != self.persistence._get_task_file_path(second)
        assert self.persistence.load_task_progress(first).task_id == first
        assert self.persistence.load_task_progress(second).task_id == second

    def test_short_task_id_filename_unchanged(self):
        """Short IDs keep their plain, readable filename."""
        assert (
            self.persistence._get_task_file_path("ingest_docs_report_pdf").name
            == "ingest_docs_report_pdf.json"
        )


def test_multibyte_task_id_filename_stays_under_the_byte_cap(tmp_path):
    """A CJK collection name is 3 bytes per character, so cap bytes not characters."""
    persistence = ProgressPersistence(storage_dir=str(tmp_path))
    # Under the 250-character budget but over it in bytes, so only a byte cap catches it.
    first = "ingest_知识库" + "文档" * 60
    second = "ingest_知识库" + "文档" * 59 + "报告"
    assert len(first) < 250 < len(first.encode("utf-8"))

    first_path = persistence._get_task_file_path(first)
    second_path = persistence._get_task_file_path(second)

    assert len(first_path.name.encode("utf-8")) <= 255
    assert first_path != second_path


def test_task_id_at_the_byte_cap_keeps_its_legacy_filename(tmp_path):
    """Upgrade path: <=250 byte names were legal before, so they must not be rehashed."""
    persistence = ProgressPersistence(storage_dir=str(tmp_path))
    task_id = "ingest_docs_" + "a" * 238
    assert len(task_id.encode("utf-8")) == 250

    legacy_path = tmp_path / f"{task_id}.json"
    legacy_path.write_text(
        json.dumps(
            {
                "task_id": task_id,
                "task_type": "ingestion",
                "status": DocumentProcessingStatus.RUNNING.value,
            }
        ),
        encoding="utf-8",
    )

    loaded = persistence.load_task_progress(task_id)
    assert loaded is not None
    assert loaded.task_id == task_id

    persistence.save_task_progress(loaded)
    assert [path.name for path in tmp_path.glob("*.json")] == [legacy_path.name]

    assert persistence.delete_task_progress(task_id) is True
    assert not legacy_path.exists()


def test_surrogate_task_id_is_hashable_and_save_reports_a_persistence_error(tmp_path):
    """os.fsdecode leaves lone surrogates in real paths; naming must not raise a bare ValueError."""
    persistence = ProgressPersistence(storage_dir=str(tmp_path))
    task_id = "a" * 251 + os.fsdecode(b"\xff")

    path = persistence._get_task_file_path(task_id)
    assert path == persistence._get_task_file_path(task_id)
    assert path != persistence._get_task_file_path("a" * 251 + os.fsdecode(b"\xfe"))

    with pytest.raises(ProgressPersistenceError) as excinfo:
        persistence.save_task_progress(
            TaskProgress(
                task_id=task_id,
                task_type="ingestion",
                status=DocumentProcessingStatus.RUNNING,
            )
        )
    assert isinstance(excinfo.value.__cause__, UnicodeEncodeError)


def test_save_wraps_filename_construction_failures(tmp_path):
    """Path construction must sit inside the try, or its ValueError escapes unwrapped."""
    persistence = ProgressPersistence(storage_dir=str(tmp_path))

    def boom(task_id):
        raise ValueError("unnameable")

    persistence._get_task_file_path = boom

    with pytest.raises(ProgressPersistenceError) as excinfo:
        persistence.save_task_progress(
            TaskProgress(
                task_id="ingest_docs_x",
                task_type="ingestion",
                status=DocumentProcessingStatus.RUNNING,
            )
        )
    assert isinstance(excinfo.value.__cause__, ValueError)
