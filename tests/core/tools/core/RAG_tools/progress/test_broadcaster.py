"""Unit tests for ProgressBroadcaster."""

from __future__ import annotations

import json

import pytest

from xagent.core.tools.core.RAG_tools.progress.realtime import ProgressBroadcaster


class MockWebSocketConnection:
    """Mock WebSocket connection for testing."""

    def __init__(self):
        self.sent_messages = []
        self.connected = True

    async def send_text(self, data: str) -> None:
        """Mock send_text method."""
        self.sent_messages.append(data)

    def is_connected(self) -> bool:
        """Mock connection check."""
        return self.connected

    def disconnect(self):
        """Mock disconnect."""
        self.connected = False


class TestProgressBroadcaster:
    """Test ProgressBroadcaster functionality."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Test connecting and disconnecting WebSocket connections."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task_id = "test_task_123"
        connection = MockWebSocketConnection()

        # Connect
        await broadcaster.connect(task_id, connection)

        # Check connection was added
        assert task_id in broadcaster._connections
        assert connection in broadcaster._connections[task_id]

        # Disconnect
        await broadcaster.disconnect(task_id, connection)

        # Check connection was removed
        assert (
            task_id not in broadcaster._connections
            or connection not in broadcaster._connections[task_id]
        )

    @pytest.mark.asyncio
    async def test_multiple_connections_same_task(self):
        """Test multiple connections to the same task."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task_id = "multi_conn_task"
        conn1 = MockWebSocketConnection()
        conn2 = MockWebSocketConnection()
        conn3 = MockWebSocketConnection()

        # Connect multiple clients
        await broadcaster.connect(task_id, conn1)
        await broadcaster.connect(task_id, conn2)
        await broadcaster.connect(task_id, conn3)

        # Check all connections are tracked
        assert len(broadcaster._connections[task_id]) == 3
        assert conn1 in broadcaster._connections[task_id]
        assert conn2 in broadcaster._connections[task_id]
        assert conn3 in broadcaster._connections[task_id]

    @pytest.mark.asyncio
    async def test_broadcast_event(self):
        """Test broadcasting events to all connections of a task."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task_id = "broadcast_test"
        conn1 = MockWebSocketConnection()
        conn2 = MockWebSocketConnection()

        # Connect two clients
        await broadcaster.connect(task_id, conn1)
        await broadcaster.connect(task_id, conn2)

        # Broadcast a message
        event_type = "test_event"
        test_data = {"status": "running", "progress": 0.5}
        await broadcaster.broadcast_event(task_id, event_type, test_data)

        # Check both connections received the message
        assert len(conn1.sent_messages) == 1
        assert len(conn2.sent_messages) == 1

        # Check message content
        sent_data = json.loads(conn1.sent_messages[0])
        assert sent_data["event_type"] == event_type
        assert sent_data["data"] == test_data

    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test handling of connection failures during broadcast."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task_id = "failure_test"

        # Create a connection that will fail
        failing_conn = MockWebSocketConnection()

        # Override send_text to raise an exception
        async def failing_send_text(data: str):
            raise ConnectionError("Connection lost")

        failing_conn.send_text = failing_send_text

        # Create a normal connection
        normal_conn = MockWebSocketConnection()

        # Connect both
        await broadcaster.connect(task_id, failing_conn)
        await broadcaster.connect(task_id, normal_conn)

        # Broadcast - should handle the failing connection gracefully
        await broadcaster.broadcast_event(task_id, "test")

        # Normal connection should still receive the message
        assert len(normal_conn.sent_messages) == 1

        # Failing connection should have been removed
        assert (
            task_id not in broadcaster._connections
            or failing_conn not in broadcaster._connections[task_id]
        )

    @pytest.mark.asyncio
    async def test_isolate_different_tasks(self):
        """Test that broadcasts to different tasks are isolated."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task1 = "task_1"
        task2 = "task_2"

        conn1 = MockWebSocketConnection()
        conn2 = MockWebSocketConnection()

        # Connect to different tasks
        await broadcaster.connect(task1, conn1)
        await broadcaster.connect(task2, conn2)

        # Broadcast to task1
        await broadcaster.broadcast_event(task1, "update1")

        # Only conn1 should receive the message
        assert len(conn1.sent_messages) == 1
        assert len(conn2.sent_messages) == 0

        # Broadcast to task2
        await broadcaster.broadcast_event(task2, "update2")

        # Now conn2 should receive the message, conn1 unchanged
        assert len(conn1.sent_messages) == 1
        assert len(conn2.sent_messages) == 1

    @pytest.mark.asyncio
    async def test_get_connection_count(self):
        """Test getting connection count for tasks."""
        broadcaster = ProgressBroadcaster()
        async with broadcaster._lock:
            broadcaster._connections.clear()

        task1 = "count_test_1"
        task2 = "count_test_2"

        # Add connections
        conn1 = MockWebSocketConnection()
        conn2 = MockWebSocketConnection()
        conn3 = MockWebSocketConnection()

        await broadcaster.connect(task1, conn1)
        await broadcaster.connect(task1, conn2)
        await broadcaster.connect(task2, conn3)

        # Check counts
        assert await broadcaster.get_connection_count(task1) == 2
        assert await broadcaster.get_connection_count(task2) == 1
        assert await broadcaster.get_connection_count("nonexistent") == 0
