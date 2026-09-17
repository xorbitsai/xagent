"""Integration tests for complete progress monitoring workflow."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from xagent.core.tools.core.RAG_tools.progress import ProgressTracker


class MockWebSocketConnection:
    """Mock WebSocket connection for integration testing."""

    def __init__(self):
        self.messages = []
        self.connected = True

    async def send_text(self, data: str) -> None:
        self.messages.append(data)

    def is_connected(self) -> bool:
        return self.connected


class TestProgressMonitoringIntegration:
    """Integration tests for complete progress monitoring workflow."""

    @pytest.fixture
    def progress_manager(self):
        """Create a fresh progress manager for each test."""
        # Since it's a singleton now, we might want to be careful.
        # But for integration tests, creating a new instance is better.
        from xagent.core.tools.core.RAG_tools.progress.manager import ProgressManager

        manager = ProgressManager()
        return manager

    @pytest.fixture
    async def mock_broadcaster(self):
        """Create mock broadcaster for testing."""
        from xagent.core.tools.core.RAG_tools.progress import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()

        # Add a mock connection for testing broadcasts
        mock_conn = MockWebSocketConnection()
        await broadcaster.connect("test_integration_task", mock_conn)

        return broadcaster, mock_conn

    def test_complete_ingestion_workflow(self, progress_manager):
        """Test complete document ingestion workflow with progress tracking."""
        task_id = "ingestion_workflow_test"

        # 1. Initialize task
        progress_manager.create_task(
            "ingestion", task_id, metadata={"message": "Starting document ingestion"}
        )

        task = progress_manager.get_task_progress(task_id)
        assert task.status == DocumentProcessingStatus.PENDING
        assert task.metadata["message"] == "Starting document ingestion"
        assert task.task_id == task_id

        # 2. Start processing
        progress_tracker = ProgressTracker(progress_manager, task_id)

        # 3. Parse document step
        with progress_tracker.track_step("parse_document") as parse_tracker:
            # Simulate parsing work
            time.sleep(0.01)  # Small delay to simulate work
            parse_tracker.update(
                message="OCR processing...", current_count=5, total_count=10
            )
            time.sleep(0.01)
            parse_tracker.update(
                message="Layout analysis...", current_count=10, total_count=10
            )
            time.sleep(0.01)
            parse_tracker.complete("Document parsed successfully")

        # Verify task progress updated
        task = progress_manager.get_task_progress(task_id)
        assert task.current_step == "parse_document"

        # 4. Chunk document step
        with progress_tracker.track_step("chunk_document") as chunk_tracker:
            chunk_tracker.update(
                message="Creating chunks...", current_count=25, total_count=50
            )
            time.sleep(0.01)
            chunk_tracker.complete("Document chunked: 50 chunks created")

        # 5. Embedding step
        with progress_tracker.track_step("compute_embeddings") as embed_tracker:
            # Simulate batch processing
            for batch in range(1, 6):
                embed_tracker.update(
                    message=f"Processing embedding batch {batch}/5",
                    current_count=batch * 10,
                    total_count=50,
                    batch=batch,
                    total_batches=5,
                )
                time.sleep(0.005)
            embed_tracker.complete("Embeddings computed for all chunks")

        # 6. Vector storage step
        with progress_tracker.track_step("write_vectors_to_db") as vector_tracker:
            vector_tracker.update(message="Writing vectors to database...")
            time.sleep(0.01)
            vector_tracker.complete("Vectors written to database successfully")

        # 7. Complete the entire task
        progress_manager.complete_task(task_id, success=True)

        # Verify final task state
        final_task = progress_manager.get_task_progress(task_id)
        assert final_task.status == DocumentProcessingStatus.SUCCESS
        assert final_task.overall_progress == 1.0
        assert final_task.end_time is not None

        # Verify task duration
        duration = final_task.end_time - (final_task.start_time or 0)
        assert duration >= 0  # Should have taken some time

    def test_search_workflow(self, progress_manager):
        """Test document search workflow with progress tracking."""
        task_id = "search_workflow_test"

        # 1. Initialize search task
        progress_manager.create_task(
            "search", task_id, metadata={"message": "Starting document search"}
        )

        # 2. Track search steps
        progress_tracker = ProgressTracker(progress_manager, task_id)

        # Query encoding
        with progress_tracker.track_step("encode_query"):
            time.sleep(0.005)  # Simulate encoding time

        # Sparse search
        with progress_tracker.track_step("sparse_search"):
            time.sleep(0.01)  # Simulate search time

        # Dense search
        with progress_tracker.track_step("dense_search"):
            time.sleep(0.01)  # Simulate dense search time

        # Hybrid fusion (if applicable)
        with progress_tracker.track_step("hybrid_search"):
            time.sleep(0.005)  # Simulate fusion time

        # Reranking
        with progress_tracker.track_step("rerank"):
            time.sleep(0.005)  # Simulate reranking time

        # Complete search
        progress_manager.complete_task(task_id, success=True)

        # Verify final task state
        final_task = progress_manager.get_task_progress(task_id)
        assert final_task.status == DocumentProcessingStatus.SUCCESS
        assert final_task.task_type == "search"

    def test_error_handling_and_recovery(self, progress_manager):
        """Test error handling and task recovery scenarios."""
        task_id = "error_handling_test"

        # 1. Start task
        progress_manager.create_task("ingestion", task_id)
        progress_tracker = ProgressTracker(progress_manager, task_id)

        # 2. Step succeeds
        with progress_tracker.track_step("successful_step") as success_tracker:
            success_tracker.complete("Step completed successfully")

        # 3. Step fails
        with progress_tracker.track_step("failing_step") as fail_tracker:
            fail_tracker.fail("Network connection lost")

        # 4. Task continues despite step failure
        with progress_tracker.track_step("recovery_step") as recovery_tracker:
            recovery_tracker.complete("Recovered and continued processing")

        # 5. Manually fail the task
        progress_manager.complete_task(task_id, success=False)

        # Verify task failed
        failed_task = progress_manager.get_task_progress(task_id)
        assert failed_task.status == DocumentProcessingStatus.FAILED

    def test_task_cancellation(self, progress_manager):
        """Test task cancellation workflow."""
        task_id = "cancellation_test"

        # 1. Start task
        progress_manager.create_task("ingestion", task_id)
        progress_tracker = ProgressTracker(progress_manager, task_id)

        # 2. Start some work
        with progress_tracker.track_step("started_step") as started_tracker:
            started_tracker.update(message="Work in progress...")

            # Simulate cancellation during work
            progress_manager.update_task_progress(
                task_id,
                status=DocumentProcessingStatus.CANCELLED,
                metadata={"message": "User requested cancellation"},
            )

            # Step should still complete normally (cancellation is task-level)
            started_tracker.complete("Step finished")

        # Verify task was cancelled
        cancelled_task = progress_manager.get_task_progress(task_id)
        assert cancelled_task.status == DocumentProcessingStatus.CANCELLED
        assert "User requested cancellation" in cancelled_task.metadata["message"]

    @pytest.mark.asyncio
    async def test_realtime_broadcasting(self, progress_manager, mock_broadcaster):
        """Test real-time progress broadcasting to WebSocket connections."""
        # Initialize broadcaster inside the test to ensure it uses the current event loop
        from xagent.core.tools.core.RAG_tools.progress import ProgressBroadcaster

        broadcaster = ProgressBroadcaster()
        mock_conn = MockWebSocketConnection()
        await broadcaster.connect("broadcast_test", mock_conn)

        task_id = "broadcast_test"

        # Mock persistence to avoid FS issues
        progress_manager.persistence = MagicMock()

        # 1. Create task and tracker
        # We need to manually set the broadcaster for this test manager
        progress_manager.broadcaster = broadcaster
        # IMPORTANT: Update the loop reference to the current test loop
        progress_manager._main_loop = asyncio.get_running_loop()

        progress_manager.create_task("ingestion", task_id)
        progress_manager.update_task_progress(
            task_id, status=DocumentProcessingStatus.RUNNING
        )
        progress_tracker = ProgressTracker(progress_manager, task_id)

        # 2. Perform work with progress updates
        with progress_tracker.track_step("broadcast_step") as broadcast_tracker:
            broadcast_tracker.update(message="Starting work...")
            await asyncio.sleep(0.05)  # Allow async operations

            broadcast_tracker.update(
                message="Work in progress...", current_count=50, total_count=100
            )
            await asyncio.sleep(0.05)

            broadcast_tracker.complete("Work completed")

        # Verify messages were sent to WebSocket connection
        assert len(mock_conn.messages) > 0

    def test_concurrent_tasks_isolation(self, progress_manager):
        """Test that concurrent tasks don't interfere with each other."""
        task_ids = [f"concurrent_task_{i}" for i in range(5)]

        # Create multiple tasks
        for task_id in task_ids:
            progress_manager.create_task("ingestion", task_id)
            progress_manager.update_task_progress(
                task_id, status=DocumentProcessingStatus.RUNNING
            )

        # Create trackers and perform work concurrently (simulated)
        trackers = {}
        for task_id in task_ids:
            trackers[task_id] = ProgressTracker(progress_manager, task_id)

            # Perform some work on each task
            with trackers[task_id].track_step("concurrent_step") as step_tracker:
                step_tracker.update(message=f"Working on {task_id}")
                time.sleep(0.001)  # Minimal delay
                step_tracker.complete(f"Completed work on {task_id}")

        # Verify all tasks still exist
        for task_id in task_ids:
            task = progress_manager.get_task_progress(task_id)
            assert task is not None
            assert task.status == DocumentProcessingStatus.RUNNING

    def test_progress_persistence_integration(self, progress_manager, tmp_path):
        """Test integration with progress persistence."""
        from xagent.core.tools.core.RAG_tools.progress import ProgressPersistence

        # Create persistence layer
        persistence = ProgressPersistence(storage_dir=str(tmp_path))

        task_id = "persistence_test"

        # 1. Create and save task
        progress_manager.create_task("ingestion", task_id, metadata={"test": "data"})
        task = progress_manager.get_task_progress(task_id)

        persistence.save_task_progress(task)

        # 2. Load task from persistence
        loaded_task = persistence.load_task_progress(task_id)
        assert loaded_task is not None
        assert loaded_task.task_id == task_id
        assert loaded_task.metadata["test"] == "data"

    def test_resource_cleanup(self, progress_manager):
        """Test proper cleanup of completed/failed tasks."""
        import tempfile

        from xagent.core.tools.core.RAG_tools.progress import ProgressPersistence

        with tempfile.TemporaryDirectory() as temp_dir:
            persistence = ProgressPersistence(storage_dir=temp_dir)

            # Create multiple tasks with different outcomes
            tasks = {
                "completed_task": DocumentProcessingStatus.SUCCESS,
                "failed_task": DocumentProcessingStatus.FAILED,
                "cancelled_task": DocumentProcessingStatus.CANCELLED,
                "active_task": DocumentProcessingStatus.RUNNING,
            }

            # Create and save tasks
            for task_id, status in tasks.items():
                progress_manager.create_task("ingestion", task_id)
                task = progress_manager.get_task_progress(task_id)
                task.status = status
                if status != DocumentProcessingStatus.RUNNING:
                    task.end_time = time.time() - (2 * 24 * 60 * 60)  # 2 days ago
                persistence.save_task_progress(task)

            # Active task should remain when listing active tasks
            active_tasks = persistence.list_active_tasks()
            assert any(t.task_id == "active_task" for t in active_tasks)
            assert len(active_tasks) == 1

    def test_performance_under_load(self, progress_manager):
        """Bound CPU work for many tasks and steps, excluding scheduler stalls."""
        num_tasks = 50
        steps_per_task = 5

        start_time = time.thread_time()

        # Create many tasks
        task_ids = []
        for i in range(num_tasks):
            task_id = f"perf_task_{i:03d}"
            task_ids.append(task_id)
            progress_manager.create_task("ingestion", task_id)
            progress_manager.update_task_progress(
                task_id, status=DocumentProcessingStatus.RUNNING
            )

        # Create trackers and perform work
        for task_id in task_ids:
            progress_tracker = ProgressTracker(progress_manager, task_id)

            for step_num in range(steps_per_task):
                step_name = f"step_{step_num}"
                with progress_tracker.track_step(step_name) as step_tracker:
                    # Minimal work simulation
                    step_tracker.update(message=f"Processing {step_name}")
                    step_tracker.complete(f"Completed {step_name}")

        total_time = time.thread_time() - start_time

        # Verify all work completed
        for task_id in task_ids:
            task = progress_manager.get_task_progress(task_id)
            assert task is not None
            assert task.status == DocumentProcessingStatus.RUNNING

        # CPU regression guard: descheduling is not computation
        # 50 tasks * 5 steps = 250 operations, should be fast
        assert total_time < 5.0  # Less than 5 CPU seconds for all operations

        print(
            f"CPU performance test completed: {num_tasks} tasks, {steps_per_task} steps each, {total_time:.2f}s"
        )
