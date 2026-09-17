"""Integration tests for Progress WebSocket API endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket
from fastapi.testclient import TestClient

from xagent.core.tools.core.RAG_tools.core.schemas import DocumentProcessingStatus
from xagent.core.tools.core.RAG_tools.progress import ProgressManager
from xagent.web.api.progress_ws import progress_ws_router
from xagent.web.models.user import User


class TestProgressWebSocketAPI:
    """Test Progress WebSocket API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with progress WebSocket routes."""
        from fastapi import FastAPI

        from xagent.web.auth_dependencies import get_current_user

        app = FastAPI()
        app.include_router(progress_ws_router)

        # Mock user
        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.username = "testuser"

        # Mock authentication dependency
        app.dependency_overrides[get_current_user] = lambda: mock_user

        return TestClient(app)

    @pytest.fixture
    def mock_progress_manager(self):
        """Mock progress manager for testing."""
        with patch(
            "xagent.web.api.progress_ws.get_progress_manager"
        ) as mock_get_manager:
            mock_manager = MagicMock(spec=ProgressManager)

            # Mock task data
            mock_task = MagicMock()
            mock_task.task_id = "test_task_123"
            mock_task.user_id = 1  # Match mock user id
            mock_task.task_type = "ingestion"
            mock_task.status = DocumentProcessingStatus.RUNNING
            mock_task.current_step = "parse_document"
            mock_task.overall_progress = 0.5
            mock_task.start_time = 1234567890.0
            mock_task.end_time = None
            mock_task.metadata = {"collection": "docs", "pages": 10}

            mock_manager.get_task_progress.return_value = mock_task
            mock_manager.get_active_tasks.return_value = {"test_task_123": mock_task}

            mock_get_manager.return_value = mock_manager
            yield mock_manager

    def test_get_task_progress_success(self, client, mock_progress_manager):
        """Test getting task progress successfully."""
        response = client.get("/api/progress/test_task_123")

        assert response.status_code == 200
        data = response.json()

        assert data["task_id"] == "test_task_123"
        assert data["task_type"] == "ingestion"
        assert data["status"] == "running"
        assert data["current_step"] == "parse_document"
        assert data["overall_progress"] == 0.5
        assert data["start_time"] == 1234567890.0
        assert data["end_time"] is None
        assert data["metadata"]["collection"] == "docs"

    def test_get_task_progress_not_found(self, client, mock_progress_manager):
        """Test getting progress for non-existent task."""
        mock_progress_manager.get_task_progress.return_value = None

        response = client.get("/api/progress/nonexistent_task")

        assert response.status_code == 404
        data = response.json()
        assert "not found" in data["detail"].lower()

    def test_get_task_progress_internal_error(self, client, mock_progress_manager):
        """Test handling of internal errors."""
        mock_progress_manager.get_task_progress.side_effect = Exception(
            "Database error"
        )

        response = client.get("/api/progress/test_task_123")

        assert response.status_code == 500
        data = response.json()
        assert "internal server error" in data["detail"].lower()

    def test_list_active_tasks_success(self, client, mock_progress_manager):
        """Test listing active tasks successfully."""
        response = client.get("/api/progress")

        assert response.status_code == 200
        data = response.json()

        assert "tasks" in data
        assert "count" in data
        assert data["count"] == 1
        assert len(data["tasks"]) == 1

        task = data["tasks"][0]
        assert task["task_id"] == "test_task_123"
        assert task["task_type"] == "ingestion"
        assert task["status"] == "running"

    def test_list_active_tasks_by_type(self, client, mock_progress_manager):
        """Test filtering active tasks by type."""
        response = client.get("/api/progress?task_type=ingestion")

        assert response.status_code == 200
        data = response.json()

        # Should return tasks filtered by type
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_type"] == "ingestion"

    def test_list_active_tasks_empty(self, client, mock_progress_manager):
        """Test listing active tasks when none exist."""
        mock_progress_manager.get_active_tasks.return_value = {}

        response = client.get("/api/progress")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 0
        assert len(data["tasks"]) == 0

    def test_list_active_tasks_error(self, client, mock_progress_manager):
        """Test error handling in listing active tasks."""
        mock_progress_manager.get_active_tasks.side_effect = Exception("Database error")

        response = client.get("/api/progress")

        assert response.status_code == 500
        data = response.json()
        assert "internal server error" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_websocket_connection_and_initial_status(self):
        """Test WebSocket connection and initial status broadcast."""
        from fastapi import WebSocketDisconnect

        from xagent.web.api.progress_ws import progress_websocket_endpoint

        # Mock WebSocket
        mock_websocket = AsyncMock(spec=WebSocket)
        # Configure receive_text to raise WebSocketDisconnect immediately to break the loop
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        # Mock progress manager
        mock_task = MagicMock()
        mock_task.task_id = "ws_test_task"
        mock_task.user_id = 1
        mock_task.task_type = "ingestion"
        mock_task.status = DocumentProcessingStatus.RUNNING
        mock_task.current_step = "parse_document"
        mock_task.overall_progress = 0.5
        mock_task.start_time = 1234567890.0
        mock_task.end_time = None
        mock_task.metadata = {"collection": "docs"}

        mock_user = MagicMock(spec=User)
        mock_user.id = 1

        with (
            patch(
                "xagent.web.api.progress_ws.get_progress_manager"
            ) as mock_get_manager,
            patch(
                "xagent.web.api.progress_ws.progress_broadcaster", spec=True
            ) as mock_broadcaster,
            patch(
                "xagent.web.api.progress_ws.get_authenticated_user",
                new=AsyncMock(return_value=mock_user),
            ),
        ):
            mock_manager = MagicMock()
            mock_manager.get_task_progress.return_value = mock_task

            mock_get_manager.return_value = mock_manager

            # Call the WebSocket endpoint
            await progress_websocket_endpoint(
                mock_websocket, "ws_test_task", "fake_token"
            )

            # Verify WebSocket was accepted
            mock_websocket.accept.assert_called_once()

            # Verify broadcaster connect was called
            mock_broadcaster.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_with_invalid_token(self):
        """Test WebSocket connection with invalid token."""
        from fastapi import WebSocketDisconnect

        from xagent.web.api.progress_ws import progress_websocket_endpoint

        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.receive_text.side_effect = WebSocketDisconnect()

        with (
            patch(
                "xagent.web.api.progress_ws.get_progress_manager"
            ) as mock_get_manager,
            patch("xagent.web.api.progress_ws.progress_broadcaster", spec=True),
            patch(
                "xagent.web.api.progress_ws.get_authenticated_user",
                new=AsyncMock(return_value=None),
            ),
        ):
            mock_manager = MagicMock()
            mock_manager.get_task_progress.return_value = None
            mock_get_manager.return_value = mock_manager

            await progress_websocket_endpoint(
                mock_websocket, "test_task", "invalid_token"
            )

            # In our implementation, it should NOT accept if token is invalid
            mock_websocket.accept.assert_not_called()
            mock_websocket.close.assert_called()

    def test_progress_status_enum_serialization(self, client, mock_progress_manager):
        """Test that progress status enums are properly serialized."""
        # Test different status values
        test_cases = [
            DocumentProcessingStatus.PENDING,
            DocumentProcessingStatus.RUNNING,
            DocumentProcessingStatus.SUCCESS,
            DocumentProcessingStatus.FAILED,
            DocumentProcessingStatus.CANCELLED,
        ]

        for status in test_cases:
            mock_progress_manager.get_task_progress.return_value.status = status

            response = client.get("/api/progress/test_task_123")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == status.value

    def test_progress_data_serialization(self, client, mock_progress_manager):
        """Test that complex progress data is properly serialized."""
        # Create complex metadata
        complex_metadata = {
            "collection": "test_docs",
            "source_file": "large_document.pdf",
            "pages": 500,
            "chunks_created": 1250,
            "embeddings_generated": 1250,
            "processing_stats": {
                "parse_time": 45.67,
                "chunk_time": 23.45,
                "embed_time": 156.78,
                "total_time": 225.9,
            },
            "model_info": {
                "embedding_model": "text-embedding-ada-002",
                "dimensions": 1536,
                "chunk_size": 1000,
                "chunk_overlap": 200,
            },
        }

        mock_progress_manager.get_task_progress.return_value.metadata = complex_metadata

        response = client.get("/api/progress/test_task_123")
        assert response.status_code == 200

        data = response.json()
        assert data["metadata"] == complex_metadata
        assert data["metadata"]["processing_stats"]["total_time"] == 225.9

    def test_concurrent_api_calls(self, client, mock_progress_manager):
        """Test concurrent API calls don't interfere with each other."""
        import threading

        results = []
        errors = []

        def api_call_worker(task_id: str):
            try:
                response = client.get(f"/api/progress/{task_id}")
                results.append((task_id, response.status_code, response.json()))
            except Exception as e:
                errors.append((task_id, str(e)))

        # Start multiple concurrent requests
        threads = []
        task_ids = [f"concurrent_task_{i}" for i in range(10)]

        for task_id in task_ids:
            t = threading.Thread(target=api_call_worker, args=[task_id])
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join(timeout=5)

        # Verify all requests succeeded
        assert len(errors) == 0
        assert len(results) == 10

        for task_id, status_code, data in results:
            assert status_code == 200
            assert data["task_id"] == "test_task_123"  # All return the same mock data

    def test_api_response_format_consistency(self, client, mock_progress_manager):
        """Test that API responses maintain consistent format."""
        response = client.get("/api/progress/test_task_123")

        assert response.status_code == 200
        data = response.json()

        # Required fields
        required_fields = [
            "task_id",
            "task_type",
            "status",
            "current_step",
            "overall_progress",
            "start_time",
            "end_time",
            "metadata",
        ]

        for field in required_fields:
            assert field in data

        # Type checks
        assert isinstance(data["task_id"], str)
        assert isinstance(data["task_type"], str)
        assert isinstance(data["status"], str)
        assert isinstance(data["overall_progress"], (int, float))
        assert isinstance(data["metadata"], dict)

        # Nullable fields
        assert data["end_time"] is None or isinstance(data["end_time"], (int, float))
        assert data["start_time"] is None or isinstance(
            data["start_time"], (int, float)
        )

    def test_websocket_url_validation(self):
        """Test WebSocket URL parameter validation."""
        # This would be tested with actual WebSocket client
        # For now, just verify the endpoint accepts the parameters

        # The endpoint expects:
        # - task_id: str (path parameter)
        # - token: str (query parameter, optional)

        # In a real test, we would:
        # 1. Connect to ws://host:port/ws/progress/{task_id}?token={token}
        # 2. Verify connection is accepted
        # 3. Send/receive messages
        # 4. Verify disconnection

        # For this unit test, we rely on the async test above
        pass

    def test_error_response_format(self, client, mock_progress_manager):
        """Test that error responses follow consistent format."""
        # Test 404 error
        mock_progress_manager.get_task_progress.return_value = None

        response = client.get("/api/progress/nonexistent")
        assert response.status_code == 404

        error_data = response.json()
        assert "detail" in error_data
        assert isinstance(error_data["detail"], str)

        # Test 500 error
        mock_progress_manager.get_task_progress.side_effect = Exception("Test error")

        response = client.get("/api/progress/test_task_123")
        assert response.status_code == 500

        error_data = response.json()
        assert "detail" in error_data
        assert "internal server error" in error_data["detail"].lower()
