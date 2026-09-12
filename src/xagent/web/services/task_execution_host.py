"""Shared task ingress versus already-claimed worker execution."""

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from ...config import get_shared_task_execution_enabled

_in_claimed_command: ContextVar[bool] = ContextVar(
    "xagent_claimed_command", default=False
)


def enqueues_task_turns() -> bool:
    # A MESSAGE has already crossed the durable boundary. Nesting START would
    # reuse its (task_id, turn_id) command identity and block the same queue.
    return get_shared_task_execution_enabled() and not _in_claimed_command.get()


@contextmanager
def claimed_command_execution() -> Iterator[None]:
    token = _in_claimed_command.set(True)
    try:
        yield
    finally:
        _in_claimed_command.reset(token)
