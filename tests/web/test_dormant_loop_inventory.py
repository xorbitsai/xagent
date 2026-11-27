"""Keep the activation/shutdown loop inventory explicit for overlap releases."""

from __future__ import annotations

import ast
import inspect
import textwrap

from xagent.web import app
from xagent.web.channels.feishu.bot import FeishuChannelManager
from xagent.web.channels.slack.bot import SlackChannelManager
from xagent.web.channels.telegram.bot import TelegramChannelManager
from xagent.web.composio_auth import run_missing_account_loop, run_reconcile_loop
from xagent.web.services.orphan_upload_gc import run_orphan_upload_gc_loop
from xagent.web.services.task_auto_resume import run_task_auto_resume_loop
from xagent.web.services.task_cleanup_obligations import run_cleanup_obligation_loop
from xagent.web.services.task_lease_recovery import run_task_lease_recovery_loop
from xagent.web.services.task_retention_purge import run_retention_purge_loop
from xagent.web.services.uploaded_file_recovery import (
    run_uploaded_file_compensation_recovery_loop,
)


def _calls(function) -> set[str]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }


def test_shutdown_loop_inventory_is_classified() -> None:
    # A new stop path needs an explicit singleton/per-process classification.
    singleton = {
        "stop_task_command_dispatcher",  # lease-owned work remains claimable
        "stop_orphan_upload_gc_task",
        "stop_composio_reconcile_task",
        "stop_composio_missing_account_task",
        "stop_retention_purge_task",
        "stop_task_cleanup_retry_task",
        "stop_uploaded_file_recovery_task",
        "stop_task_auto_resume_task",
        "stop_task_lease_recovery_task",
    }
    per_process = {"stop_browser_session_sweep_task"}
    actual = {
        name for name in _calls(app._stop_runtime_loops) if name.startswith("stop_")
    }
    assert actual == singleton | per_process


def test_every_registered_singleton_loop_has_a_tick_fence_and_wakeup() -> None:
    loops = (
        app._run_trigger_dispatcher,
        run_task_lease_recovery_loop,
        run_task_auto_resume_loop,
        run_orphan_upload_gc_loop,
        run_uploaded_file_compensation_recovery_loop,
        run_cleanup_obligation_loop,
        run_retention_purge_loop,
        run_reconcile_loop,
        run_missing_account_loop,
    )
    for loop in loops:
        calls = _calls(loop)
        assert {"tick_allowed", "wait_tick"} <= calls, loop.__qualname__
    for manager in (
        TelegramChannelManager,
        FeishuChannelManager,
        SlackChannelManager,
    ):
        assert "tick_allowed" in _calls(manager._sync_bots_async)
        assert "wait_tick" in _calls(manager.start)
