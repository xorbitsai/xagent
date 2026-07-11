"""Test cases for TaskTracker and TaskTrackerManager."""

from unittest.mock import MagicMock, patch

import pytest

from xagent.core.model.chat.token_context import add_token_usage, get_token_usage
from xagent.web.models.task import Task, TaskStatus
from xagent.web.tracking import TaskTracker, TaskTrackerManager


class TestTaskTracker:
    """Test cases for TaskTracker."""

    @pytest.fixture
    def db_session(self):
        """Fixture providing a mock database session."""
        session = MagicMock()

        # Mock Task object
        mock_task = MagicMock(spec=Task)
        mock_task.id = 123
        mock_task.status = TaskStatus.RUNNING
        mock_task.input_tokens = 0
        mock_task.output_tokens = 0
        mock_task.total_tokens = 0
        mock_task.llm_calls = 0
        mock_task.token_usage_details = None

        # Mock query to return the task
        session.query.return_value.filter.return_value.first.return_value = mock_task

        return session

    @pytest.fixture
    def task_tracker(self, db_session):
        """Fixture providing a TaskTracker instance."""
        return TaskTracker(task_id=123, db_session=db_session)

    @pytest.mark.asyncio
    async def test_init_task_tracker(self, db_session):
        """Test TaskTracker initialization."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        assert tracker.task_id == 123
        assert tracker.db_session == db_session
        assert tracker.update_interval_seconds == 15  # default
        assert not tracker.is_tracking

    @pytest.mark.asyncio
    async def test_complete_tracking_reports_only_current_turn_delta(self, db_session):
        """The usage-record hook must receive only this turn's delta, not the
        re-seeded prior-turn baseline (multi-turn tasks seed from the DB)."""
        from xagent.core.model.chat.token_context import add_tool_call_usage
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 42
        # Prior-turn state seeded from the DB row.
        task.input_tokens = 100
        task.output_tokens = 50
        task.llm_calls = 1
        task.token_usage_details = [
            {"type": "input", "tokens": 100, "model": "m", "call_type": "chat"},
            {"type": "output", "tokens": 50, "model": "m", "call_type": "chat"},
        ]

        captured = {}

        def _hook(db, user_id, delta_details, delta_actions):
            captured.update(
                user_id=user_id, details=delta_details, actions=delta_actions
            )

        quota_hooks.set_usage_record_hook(_hook)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            await tracker.start_tracking()
            # This turn's usage, appended on top of the seeded baseline.
            add_token_usage(
                input_tokens=10, output_tokens=5, model="m", call_type="chat"
            )
            add_tool_call_usage(3)
            await tracker.complete_tracking()
        finally:
            quota_hooks.set_usage_record_hook(None)

        assert captured["user_id"] == 42
        assert captured["actions"] == 3  # only this turn's tool calls (baseline was 0)
        # Only the two entries appended this turn, not the two seeded ones.
        assert len(captured["details"]) == 2
        assert sorted(d["tokens"] for d in captured["details"]) == [5, 10]

    @pytest.mark.asyncio
    async def test_interrupt_reason_for_quota_passes_turn_delta(self, db_session):
        """The per-step quota gate must see the same this-turn delta the metering
        path computes, and surface the gate's reason (or None when open)."""
        from xagent.core.model.chat.token_context import add_tool_call_usage
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 42
        task.input_tokens = 100
        task.output_tokens = 50
        task.llm_calls = 1
        task.token_usage_details = [
            {"type": "input", "tokens": 100, "model": "m", "call_type": "chat"},
        ]

        captured = {}

        def _gate(db, user_id, delta_details, delta_actions):
            captured.update(
                user_id=user_id, details=delta_details, actions=delta_actions
            )
            return "over credits" if delta_actions >= 2 else None

        quota_hooks.set_run_progress_gate_hook(_gate)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            # Before tracking starts, the checker is a no-op (fails open).
            assert tracker.interrupt_reason_for_quota() is None
            await tracker.start_tracking()
            add_token_usage(
                input_tokens=10, output_tokens=5, model="m", call_type="chat"
            )
            add_tool_call_usage(2)
            reason = tracker.interrupt_reason_for_quota()
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

        assert reason == "over credits"
        # The reason is recorded so the run's caller can surface why it stopped.
        assert tracker.quota_interrupt_reason == "over credits"
        assert captured["user_id"] == 42
        assert captured["actions"] == 2  # only this turn's tool calls
        # Only this turn's input+output entries, not the seeded baseline one.
        assert len(captured["details"]) == 2
        assert sorted(d["tokens"] for d in captured["details"]) == [5, 10]

    @pytest.mark.asyncio
    async def test_quota_gate_caches_user_id_and_logs_once(self, db_session, caplog):
        """F1: user_id is cached at construction (not re-read per step). F3: the
        fail-open warning is logged once per run, not once per step."""
        import logging

        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 7

        def _boom(db, user_id, dd, da):
            raise RuntimeError("gate infra down")

        quota_hooks.set_run_progress_gate_hook(_boom)
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            assert tracker._user_id == 7  # F1: cached at construction
            await tracker.start_tracking()
            with caplog.at_level(logging.WARNING):
                assert tracker.interrupt_reason_for_quota() is None
                assert tracker.interrupt_reason_for_quota() is None
                assert tracker.interrupt_reason_for_quota() is None
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

        warnings = [r for r in caplog.records if "failed open" in r.getMessage()]
        assert len(warnings) == 1  # F3: one log despite three failing calls
        assert tracker.quota_interrupt_reason is None  # never tripped → no reason

    @pytest.mark.asyncio
    async def test_runtime_should_interrupt_drives_tracker_gate(self, db_session):
        """End-to-end seam: the runtime's should_interrupt — the exact call the
        pattern loop makes at each safe point — wired to the tracker's real gate
        method fires and records the reason when a registered hook trips."""
        from xagent.core.agent.runtime import PatternRuntime
        from xagent.web.services import quota_hooks

        task = db_session.query.return_value.filter.return_value.first.return_value
        task.user_id = 5

        quota_hooks.set_run_progress_gate_hook(lambda db, uid, dd, da: "Out of credits")
        try:
            tracker = TaskTracker(task_id=123, db_session=db_session)
            await tracker.start_tracking()
            runtime = PatternRuntime(
                interrupt_checker=tracker.interrupt_reason_for_quota
            )
            assert await runtime.should_interrupt() is True
            assert runtime.interrupt_reason == "Out of credits"
            assert tracker.quota_interrupt_reason == "Out of credits"
        finally:
            quota_hooks.set_run_progress_gate_hook(None)

    @pytest.mark.asyncio
    async def test_init_task_tracker_with_custom_interval(self, db_session):
        """Test TaskTracker with custom update interval."""
        tracker = TaskTracker(
            task_id=123, db_session=db_session, update_interval_seconds=60
        )

        assert tracker.update_interval_seconds == 60

    @pytest.mark.asyncio
    async def test_init_task_tracker_task_not_found(self, db_session):
        """Test TaskTracker with non-existent task."""
        # Mock query to return None (task not found)
        db_session.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ValueError, match="Task 123 not found"):
            TaskTracker(task_id=123, db_session=db_session)

    @pytest.mark.asyncio
    async def test_start_tracking(self, task_tracker):
        """Test starting token tracking."""
        # Add some tokens before starting
        add_token_usage(input_tokens=10, output_tokens=5)

        await task_tracker.start_tracking()

        # Should reset token usage
        usage = get_token_usage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_start_tracking_uses_existing_task_totals(self, db_session):
        mock_task = db_session.query.return_value.filter.return_value.first.return_value
        mock_task.input_tokens = 120
        mock_task.output_tokens = 80
        mock_task.llm_calls = 4
        mock_task.token_usage_details = [{"type": "input", "tokens": 120}]

        tracker = TaskTracker(task_id=123, db_session=db_session)
        await tracker.start_tracking()

        usage = get_token_usage()
        assert usage.input_tokens == 120
        assert usage.output_tokens == 80
        assert usage.llm_calls == 4
        assert usage.details == [{"type": "input", "tokens": 120}]

    @pytest.mark.asyncio
    async def test_start_tracking_already_tracking(self, task_tracker, caplog):
        """Test starting tracking when already tracking."""
        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.start_tracking()

            # Try to start again
            await task_tracker.start_tracking()

        # Should log warning
        assert mock_warning.called
        assert "already being tracked" in mock_warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_periodic_update(self, task_tracker):
        """Test periodic database update."""
        await task_tracker.start_tracking()

        # Add some tokens
        add_token_usage(input_tokens=100, output_tokens=50)

        # Perform periodic update
        await task_tracker.periodic_update()

        # Verify database was updated
        task_tracker.task.input_tokens = 100
        task_tracker.task.output_tokens = 50
        task_tracker.task.total_tokens = 150
        task_tracker.task.llm_calls = 1

        task_tracker.db_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_periodic_update_not_tracking(self, task_tracker, caplog):
        """Test periodic update when not tracking."""
        # Don't start tracking

        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.periodic_update()

        # Should log warning
        assert mock_warning.called
        assert "not being tracked" in mock_warning.call_args.args[0]

    @pytest.mark.asyncio
    async def test_complete_tracking(self, task_tracker):
        """Test completing tracking."""
        await task_tracker.start_tracking()

        # Add some tokens
        add_token_usage(input_tokens=200, output_tokens=100)
        add_token_usage(input_tokens=50, output_tokens=25)

        # Complete tracking
        usage = await task_tracker.complete_tracking()

        # Verify usage was returned
        assert usage.input_tokens == 250
        assert usage.output_tokens == 125
        assert usage.total_tokens == 375
        assert usage.llm_calls == 2

        # Verify database was updated
        task_tracker.task.input_tokens = 250
        task_tracker.task.output_tokens = 125
        task_tracker.task.total_tokens = 375
        task_tracker.task.llm_calls = 2

        assert not task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_complete_tracking_not_started(self, task_tracker):
        """Test completing tracking without starting."""
        # Don't start tracking

        with pytest.raises(RuntimeError, match="not being tracked"):
            await task_tracker.complete_tracking()

    @pytest.mark.asyncio
    async def test_get_current_usage(self, task_tracker):
        """Test getting current usage without stopping."""
        await task_tracker.start_tracking()

        add_token_usage(input_tokens=30, output_tokens=15)

        usage = task_tracker.get_current_usage()

        assert usage.input_tokens == 30
        assert usage.output_tokens == 15
        assert task_tracker.is_tracking  # Should still be tracking

    @pytest.mark.asyncio
    async def test_start_stop_periodic_updates(self, task_tracker):
        """Test starting and stopping periodic background updates."""
        await task_tracker.start_tracking()

        # Start periodic updates
        await task_tracker.start_periodic_updates()

        assert task_tracker.is_tracking

        # Stop periodic updates
        await task_tracker.stop_periodic_updates()

        assert not task_tracker.is_tracking

    @pytest.mark.asyncio
    async def test_start_periodic_updates_already_active(self, task_tracker, caplog):
        """Test starting periodic updates when already active."""
        with patch("xagent.web.tracking.task_tracker.logger.warning") as mock_warning:
            await task_tracker.start_tracking()
            await task_tracker.start_periodic_updates()

            # Try to start again
            await task_tracker.start_periodic_updates()

        # Should log warning
        assert mock_warning.called
        assert "already active" in mock_warning.call_args.args[0]


class TestTaskTrackerManager:
    """Test cases for TaskTrackerManager."""

    @pytest.fixture
    def manager(self):
        """Fixture providing a TaskTrackerManager."""
        return TaskTrackerManager()

    @pytest.fixture
    def mock_session(self):
        """Fixture providing a mock database session."""
        session = MagicMock()

        # Mock Task object
        mock_task = MagicMock(spec=Task)
        mock_task.id = 1
        mock_task.status = TaskStatus.RUNNING

        session.query.return_value.filter.return_value.first.return_value = mock_task

        return session

    @pytest.mark.asyncio
    async def test_get_or_create_tracker_new(self, manager, mock_session):
        """Test creating a new tracker."""
        tracker = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        assert tracker is not None
        assert tracker.task_id == 1
        assert 1 in manager._trackers

    @pytest.mark.asyncio
    async def test_get_or_create_tracker_existing(self, manager, mock_session):
        """Test getting existing tracker."""
        # Create tracker first
        tracker1 = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Get same tracker again
        tracker2 = manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Should return the same instance
        assert tracker1 is tracker2

    @pytest.mark.asyncio
    async def test_get_tracker(self, manager, mock_session):
        """Test getting tracker without creating."""
        # Non-existent tracker
        tracker = manager.get_tracker(task_id=1)
        assert tracker is None

        # Create tracker
        manager.get_or_create_tracker(task_id=1, db_session=mock_session)

        # Now it exists
        tracker = manager.get_tracker(task_id=1)
        assert tracker is not None

    @pytest.mark.asyncio
    async def test_complete_tracker(self, manager, mock_session):
        """Test completing a specific tracker."""
        # Create and start tracker
        tracker = manager.get_or_create_tracker(task_id=1, db_session=mock_session)
        await tracker.start_tracking()

        # Add tokens
        add_token_usage(input_tokens=10, output_tokens=5)

        # Complete the tracker
        usage = await manager.complete_tracker(task_id=1)

        # Verify usage
        assert usage.input_tokens == 10
        assert usage.output_tokens == 5

        # Tracker should be removed
        assert 1 not in manager._trackers

    @pytest.mark.asyncio
    async def test_complete_tracker_nonexistent(self, manager):
        """Test completing non-existent tracker."""
        usage = await manager.complete_tracker(task_id=999)

        # Should return None
        assert usage is None

    @pytest.mark.asyncio
    async def test_complete_all(self, manager):
        """Test completing all trackers."""
        # Create multiple trackers with independent token tracking
        # Note: In real usage, each task would have its own execution context
        # Here we simulate by manually tracking tokens per task

        # Trackers are created but tokens accumulate in shared context
        for i in range(1, 4):
            tracker = manager.get_or_create_tracker(task_id=i, db_session=MagicMock())
            await tracker.start_tracking()

        # Complete all (will have 0 tokens since we didn't add any after last reset)
        results = await manager.complete_all()

        # Verify all trackers completed
        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results

        # All trackers should be removed
        assert len(manager._trackers) == 0

    @pytest.mark.asyncio
    async def test_multiple_tasks_independent(self, manager):
        """Test that multiple task trackers can be created and managed independently."""
        # Mock different sessions
        session1 = MagicMock()
        session2 = MagicMock()

        for i, session in enumerate([session1, session2], 1):
            mock_task = MagicMock(spec=Task)
            mock_task.id = i
            session.query.return_value.filter.return_value.first.return_value = (
                mock_task
            )

        # Create two trackers
        tracker1 = manager.get_or_create_tracker(task_id=1, db_session=session1)
        tracker2 = manager.get_or_create_tracker(task_id=2, db_session=session2)

        # Verify both are tracked independently
        assert tracker1.task_id == 1
        assert tracker2.task_id == 2
        assert len(manager._trackers) == 2


class TestTaskTrackerIntegration:
    """Integration tests for TaskTracker with real token tracking."""

    @pytest.fixture
    def db_session(self):
        """Fixture providing a mock database session."""
        session = MagicMock()

        mock_task = MagicMock(spec=Task)
        mock_task.id = 123
        mock_task.status = TaskStatus.RUNNING
        mock_task.input_tokens = 0
        mock_task.output_tokens = 0
        mock_task.total_tokens = 0
        mock_task.llm_calls = 0
        mock_task.token_usage_details = None

        session.query.return_value.filter.return_value.first.return_value = mock_task

        return session

    @pytest.mark.asyncio
    async def test_full_tracking_workflow(self, db_session):
        """Test complete tracking workflow."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        # Start tracking
        await tracker.start_tracking()

        # Simulate LLM calls
        add_token_usage(
            input_tokens=100, output_tokens=50, model="gpt-4", call_type="chat"
        )
        add_token_usage(
            input_tokens=200, output_tokens=100, model="gpt-4", call_type="chat"
        )

        # Check current usage
        usage = tracker.get_current_usage()
        assert usage.input_tokens == 300
        assert usage.output_tokens == 150

        # Complete tracking
        final_usage = await tracker.complete_tracking()

        # Verify final stats
        assert final_usage.input_tokens == 300
        assert final_usage.output_tokens == 150
        assert final_usage.llm_calls == 2  # Two add_token_usage calls
        assert len(final_usage.details) == 4  # 2 input + 2 output entries

        # Verify database was updated
        db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_tracking_with_details(self, db_session):
        """Test tracking with detailed token information."""
        tracker = TaskTracker(task_id=123, db_session=db_session)

        await tracker.start_tracking()

        # Add tokens with details
        add_token_usage(
            input_tokens=100,
            output_tokens=50,
            model="gpt-4",
            call_type="chat",
        )
        add_token_usage(
            input_tokens=50,
            output_tokens=25,
            model="gpt-3.5-turbo",
            call_type="stream_chat",
        )

        final_usage = await tracker.complete_tracking()

        # Verify details are tracked (4 entries: 2 input + 2 output)
        assert len(final_usage.details) == 4

        # Details are accumulated, check both models are present
        models = [d.get("model") for d in final_usage.details]
        assert "gpt-4" in models
        assert "gpt-3.5-turbo" in models

        # Check both input and output types are present
        types = [d.get("type") for d in final_usage.details]
        assert "input" in types
        assert "output" in types


def test_check_run_progress_gate_guards():
    """The quota-gate seam is a no-op when no hook is registered or user_id is
    None, and otherwise forwards to the hook and returns its reason."""
    from xagent.web.services import quota_hooks

    # No hook registered → None regardless of args.
    quota_hooks.set_run_progress_gate_hook(None)
    assert quota_hooks.check_run_progress_gate("db", 1, [], 0) is None

    seen = []
    quota_hooks.set_run_progress_gate_hook(
        lambda db, uid, dd, da: (seen.append(uid), "OVER")[1]
    )
    try:
        # user_id None short-circuits before the hook is called.
        assert quota_hooks.check_run_progress_gate("db", None, [], 0) is None
        assert seen == []
        # Otherwise the hook runs and its reason is returned verbatim.
        assert quota_hooks.check_run_progress_gate("db", 7, [], 3) == "OVER"
        assert seen == [7]
    finally:
        quota_hooks.set_run_progress_gate_hook(None)
