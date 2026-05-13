"""Test for the tracing system."""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock

from xagent.core.agent.trace import (
    ConsoleTraceHandler,
    TraceAction,
    TraceCategory,
    TraceEvent,
    TraceEventType,
    Tracer,
    TraceScope,
    trace_action_start,
    trace_dag_plan_start,
    trace_error,
    trace_info,
    trace_step_start,
    trace_task_start,
)


class TestTraceEventType(unittest.TestCase):
    """Test the new TraceEventType class."""

    def test_event_type_creation(self):
        """Test creating trace event types."""
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        self.assertEqual(event_type.scope, TraceScope.TASK)
        self.assertEqual(event_type.action, TraceAction.START)
        self.assertEqual(event_type.category, TraceCategory.DAG)
        self.assertEqual(event_type.value, "task_start_dag")

    def test_event_type_equality(self):
        """Test event type equality."""
        event1 = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)
        event2 = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)
        event3 = TraceEventType(TraceScope.STEP, TraceAction.START, TraceCategory.DAG)

        self.assertEqual(event1, event2)
        self.assertNotEqual(event1, event3)

    def test_event_type_hash(self):
        """Test event type hashing."""
        event1 = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)
        event2 = TraceEventType(TraceScope.TASK, TraceAction.START, TraceCategory.DAG)

        self.assertEqual(hash(event1), hash(event2))

    def test_predefined_event_types(self):
        """Test predefined event types."""
        from xagent.core.agent.trace import (
            ACTION_START_TOOL,
            STEP_START_DAG,
            TASK_START_DAG,
        )

        self.assertEqual(TASK_START_DAG.scope, TraceScope.TASK)
        self.assertEqual(TASK_START_DAG.action, TraceAction.START)
        self.assertEqual(TASK_START_DAG.category, TraceCategory.DAG)

        self.assertEqual(STEP_START_DAG.scope, TraceScope.STEP)
        self.assertEqual(STEP_START_DAG.action, TraceAction.START)
        self.assertEqual(STEP_START_DAG.category, TraceCategory.DAG)

        self.assertEqual(ACTION_START_TOOL.scope, TraceScope.ACTION)
        self.assertEqual(ACTION_START_TOOL.action, TraceAction.START)
        self.assertEqual(ACTION_START_TOOL.category, TraceCategory.TOOL)


class TestTraceEvent(unittest.TestCase):
    """Test the new TraceEvent class."""

    def test_task_event_creation(self):
        """Test creating task-level events."""
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        event = TraceEvent(event_type, task_id="task123")

        self.assertEqual(event.event_type, event_type)
        self.assertEqual(event.task_id, "task123")
        self.assertIsNone(event.step_id)

    def test_step_event_creation(self):
        """Test creating step-level events."""
        event_type = TraceEventType(
            TraceScope.STEP, TraceAction.START, TraceCategory.DAG
        )
        event = TraceEvent(event_type, task_id="task123", step_id="step456")

        self.assertEqual(event.event_type, event_type)
        self.assertEqual(event.task_id, "task123")
        self.assertEqual(event.step_id, "step456")

    def test_action_event_creation(self):
        """Test creating action-level events."""
        event_type = TraceEventType(
            TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL
        )
        event = TraceEvent(event_type, task_id="task123", step_id="step456")

        self.assertEqual(event.event_type, event_type)
        self.assertEqual(event.task_id, "task123")
        self.assertEqual(event.step_id, "step456")

    def test_task_event_validation(self):
        """Test task event validation."""
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )

        # Should raise error without task_id
        with self.assertRaises(ValueError) as cm:
            TraceEvent(event_type)
        self.assertIn("requires task_id", str(cm.exception))

        # Should work with task_id
        event = TraceEvent(event_type, task_id="task123")
        self.assertEqual(event.task_id, "task123")

    def test_step_event_validation(self):
        """Test step event validation."""
        event_type = TraceEventType(
            TraceScope.STEP, TraceAction.START, TraceCategory.DAG
        )

        # Should raise error without step_id
        with self.assertRaises(ValueError) as cm:
            TraceEvent(event_type, task_id="task123")
        self.assertIn("requires step_id", str(cm.exception))

        # Should work with step_id
        event = TraceEvent(event_type, task_id="task123", step_id="step456")
        self.assertEqual(event.step_id, "step456")

    def test_action_event_validation(self):
        """Test action event validation."""
        event_type = TraceEventType(
            TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL
        )

        # Should raise error without step_id
        with self.assertRaises(ValueError) as cm:
            TraceEvent(event_type, task_id="task123")
        self.assertIn("requires step_id", str(cm.exception))

        # Should work with step_id
        event = TraceEvent(event_type, task_id="task123", step_id="step456")
        self.assertEqual(event.step_id, "step456")

    def test_event_to_dict(self):
        """Test converting event to dictionary."""
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        event = TraceEvent(event_type, task_id="task123", data={"key": "value"})

        result = event.to_dict()

        self.assertEqual(result["event_type"], "task_start_dag")
        self.assertEqual(result["scope"], "task")
        self.assertEqual(result["action"], "start")
        self.assertEqual(result["category"], "dag")
        self.assertEqual(result["task_id"], "task123")
        self.assertIsNone(result["step_id"])
        self.assertEqual(result["data"]["key"], "value")


class TestTracer(unittest.TestCase):
    """Test the Tracer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracer = Tracer()

    def test_trace_event(self):
        """Test tracing events."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer.add_handler(mock_handler)

        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        event_id = asyncio.run(self.tracer.trace_event(event_type, task_id="task123"))

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        # Verify the event was passed correctly
        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertIsInstance(call_args, TraceEvent)
        self.assertEqual(call_args.task_id, "task123")

    def test_multiple_handlers(self):
        """Test tracing with multiple handlers."""
        mock_handler1 = MagicMock()
        mock_handler1.handle_event = AsyncMock()
        mock_handler2 = MagicMock()
        mock_handler2.handle_event = AsyncMock()
        self.tracer.add_handler(mock_handler1)
        self.tracer.add_handler(mock_handler2)

        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        asyncio.run(self.tracer.trace_event(event_type, task_id="task123"))

        mock_handler1.handle_event.assert_called_once()
        mock_handler2.handle_event.assert_called_once()

    def test_handler_error_handling(self):
        """Test error handling when handlers fail."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock(side_effect=Exception("Handler error"))
        self.tracer.add_handler(mock_handler)

        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        event_id = asyncio.run(self.tracer.trace_event(event_type, task_id="task123"))

        self.assertIsNotNone(event_id)  # Should still return an ID
        mock_handler.handle_event.assert_called_once()

    def test_required_persistence_propagates_handler_errors(self):
        """Test required persistence fails when a handler fails."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock(side_effect=Exception("Handler error"))
        self.tracer.add_handler(mock_handler)

        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self.tracer.trace_event(
                    event_type,
                    task_id="task123",
                    require_persisted=True,
                )
            )

    def test_required_persistence_requires_handler(self):
        """Test required persistence fails without a handler."""
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self.tracer.trace_event(
                    event_type,
                    task_id="task123",
                    require_persisted=True,
                )
            )


class TestConsoleTraceHandler(unittest.TestCase):
    """Test the ConsoleTraceHandler class."""

    def test_task_event_handling(self):
        """Test handling task-level events."""
        handler = ConsoleTraceHandler()
        event_type = TraceEventType(
            TraceScope.TASK, TraceAction.START, TraceCategory.DAG
        )
        event = TraceEvent(event_type, task_id="task123", data={"key": "value"})

        # This should not raise an exception
        asyncio.run(handler.handle_event(event))

    def test_step_event_handling(self):
        """Test handling step-level events."""
        handler = ConsoleTraceHandler()
        event_type = TraceEventType(
            TraceScope.STEP, TraceAction.START, TraceCategory.DAG
        )
        event = TraceEvent(
            event_type, task_id="task123", step_id="step456", data={"key": "value"}
        )

        # This should not raise an exception
        asyncio.run(handler.handle_event(event))

    def test_action_event_handling(self):
        """Test handling action-level events."""
        handler = ConsoleTraceHandler()
        event_type = TraceEventType(
            TraceScope.ACTION, TraceAction.START, TraceCategory.TOOL
        )
        event = TraceEvent(
            event_type, task_id="task123", step_id="step456", data={"key": "value"}
        )

        # This should not raise an exception
        asyncio.run(handler.handle_event(event))


class TestConvenienceFunctions(unittest.TestCase):
    """Test the convenience functions."""

    def test_trace_task_start(self):
        """Test trace_task_start function."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_task_start(
                self.tracer, "task123", TraceCategory.DAG, {"key": "value"}
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.TASK)
        self.assertEqual(call_args.event_type.action, TraceAction.START)
        self.assertEqual(call_args.event_type.category, TraceCategory.DAG)
        self.assertEqual(call_args.task_id, "task123")
        self.assertEqual(call_args.data["key"], "value")

    def test_trace_step_start(self):
        """Test trace_step_start function."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_step_start(
                self.tracer, "task123", "step456", TraceCategory.DAG, {"key": "value"}
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.STEP)
        self.assertEqual(call_args.event_type.action, TraceAction.START)
        self.assertEqual(call_args.event_type.category, TraceCategory.DAG)
        self.assertEqual(call_args.task_id, "task123")
        self.assertEqual(call_args.step_id, "step456")
        self.assertEqual(call_args.data["key"], "value")

    def test_trace_action_start(self):
        """Test trace_action_start function."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_action_start(
                self.tracer, "task123", "step456", TraceCategory.TOOL, {"key": "value"}
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.ACTION)
        self.assertEqual(call_args.event_type.action, TraceAction.START)
        self.assertEqual(call_args.event_type.category, TraceCategory.TOOL)
        self.assertEqual(call_args.task_id, "task123")
        self.assertEqual(call_args.step_id, "step456")
        self.assertEqual(call_args.data["key"], "value")

    def test_trace_error_task_level(self):
        """Test trace_error function at task level."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_error(
                self.tracer,
                "task123",
                None,
                "TypeError",
                "Something went wrong",
                "traceback here",
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.TASK)
        self.assertEqual(call_args.event_type.action, TraceAction.ERROR)
        self.assertEqual(call_args.event_type.category, TraceCategory.GENERAL)
        self.assertEqual(call_args.task_id, "task123")
        self.assertIsNone(call_args.step_id)
        self.assertEqual(call_args.data["error_type"], "TypeError")
        self.assertEqual(call_args.data["error_message"], "Something went wrong")
        self.assertEqual(call_args.data["traceback"], "traceback here")

    def test_trace_error_step_level(self):
        """Test trace_error function at step level."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_error(
                self.tracer, "task123", "step456", "TypeError", "Something went wrong"
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.STEP)
        self.assertEqual(call_args.event_type.action, TraceAction.ERROR)
        self.assertEqual(call_args.event_type.category, TraceCategory.GENERAL)
        self.assertEqual(call_args.task_id, "task123")
        self.assertEqual(call_args.step_id, "step456")

    def test_trace_info(self):
        """Test trace_info function."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_info(
                self.tracer, "task123", "step456", TraceCategory.LLM, {"key": "value"}
            )
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.STEP)
        self.assertEqual(call_args.event_type.action, TraceAction.INFO)
        self.assertEqual(call_args.event_type.category, TraceCategory.LLM)
        self.assertEqual(call_args.task_id, "task123")
        self.assertEqual(call_args.step_id, "step456")
        self.assertEqual(call_args.data["key"], "value")


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility functions."""

    def test_trace_dag_plan_start(self):
        """Test backward compatibility function."""
        mock_handler = MagicMock()
        mock_handler.handle_event = AsyncMock()
        self.tracer = Tracer()
        self.tracer.add_handler(mock_handler)

        event_id = asyncio.run(
            trace_dag_plan_start(self.tracer, "task123", {"key": "value"})
        )

        self.assertIsNotNone(event_id)
        mock_handler.handle_event.assert_called_once()

        call_args = mock_handler.handle_event.call_args[0][0]
        self.assertEqual(call_args.event_type.scope, TraceScope.TASK)
        self.assertEqual(call_args.event_type.action, TraceAction.START)
        self.assertEqual(call_args.event_type.category, TraceCategory.DAG_PLAN)


if __name__ == "__main__":
    unittest.main()
