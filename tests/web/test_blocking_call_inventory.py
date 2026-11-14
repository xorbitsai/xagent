"""Every thread-starting call in ``web`` and ``core/tools`` is tracked or audited.

``asyncio.to_thread`` and friends start a thread that a cancelled awaiter does
not stop, and the thread may still write after ``app.retire_writers`` reported
success. The guarantee of ``retire_writers`` is therefore exactly: the calls
that go through ``run_tracked`` (web) or ``run_blocking`` (core), plus the
files audited below. This test fails on any thread-starting call that is
neither, so a new one cannot slip in unreviewed.

Detected: ``to_thread``, ``run_in_executor``, ``anyio.to_thread.run_sync``,
``run_in_thread_with_telemetry``, ``threading.Thread(...)`` and
``ThreadPoolExecutor(...)`` (and so every ``submit`` on a pool built there).
Not detected, on purpose: ``run_db_io_cancellation_safe`` and
``run_auth_db_worker``. They hold the worker until it has returned, even when
the awaiting task is cancelled, so a cancelled and awaited task owns no live
thread.

Each entry below is keyed by file with the exact number of call sites audited.
The count is exact in both directions: adding a call to an audited file fails,
and so does tracking one (lower the number, so the audit stays true).

The reasons name which of the documented ``retire_writers`` "Not covered"
classes a file belongs to, or why it cannot write at all. They were checked
against the code, not copied from a template; if a call site changes meaning,
change the reason with it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "xagent"
SCANNED = ("web", "core/tools")
THREAD_CALLS = {
    "to_thread",
    "run_in_executor",
    "run_in_thread_with_telemetry",
    "Thread",
    "ThreadPoolExecutor",
    "run_sync",
}
# The tracker itself.
IMPLEMENTATION = {"web/services/writer_workers.py"}

HTTP = (
    "HTTP route handler thread. HTTP producers are listed under 'Not covered' "
    "in retire_writers: the host stops routing HTTP traffic before it relies "
    "on the retire."
)
EXEC = (
    "Runs inside a running execution (task coordinator or a tool call). It may "
    "write, but executions are owned by the task coordinators and "
    "background_task_manager that execution_status() reports and shutdown "
    "drains; retire_writers lists them under 'Not covered'."
)
FILES = (
    "Filesystem or workspace-file work for a running execution or an HTTP "
    "route (no database, vector-store or shared-state write); both are listed "
    "under 'Not covered' in retire_writers."
)
READ = "Read-only or pure computation: no database, vector-store or shared write."
NETWORK = "Outbound network call (DNS, provider or Google API read): writes no local shared state."

ALLOWED: dict[str, tuple[int, str]] = {
    # -- web/api: HTTP routes ---------------------------------------------------
    "web/api/admin_users.py": (6, HTTP),
    "web/api/auth.py": (11, HTTP),
    "web/api/chat.py": (8, HTTP),
    "web/api/files.py": (7, HTTP),
    "web/api/kb.py": (16, HTTP),
    "web/api/mcp.py": (5, HTTP),
    "web/api/model.py": (2, HTTP),
    "web/api/skill_hub.py": (6, HTTP),
    "web/api/tools.py": (2, HTTP),
    "web/api/triggers.py": (8, HTTP),
    "web/api/v1/skills.py": (1, HTTP),
    "web/api/v1/tasks.py": (1, HTTP),
    "web/api/workforces.py": (2, HTTP),
    # -- web: lifecycle and telemetry -------------------------------------------
    "web/app.py": (
        2,
        "Two calls: the startup check of the auth-attempt storage, which is "
        "awaited before the lifespan starts any writer, and the temp-file "
        "walk, which deletes temp files only and is stopped through its own "
        "event (documented in retire_writers).",
    ),
    "web/worker.py": (
        4,
        "Worker-process startup checks (schema, key and engine validation), "
        "awaited before the worker serves; the web retire does not run in a "
        "worker and never overlaps them.",
    ),
    "web/runtime_telemetry.py": (
        1,
        "Telemetry exporter shutdown; no application data.",
    ),
    "web/tracing.py": (
        1,
        "Reads one committed tool outcome from the checkpoint store for a "
        "running execution (load); writes nothing.",
    ),
    "web/tools/config.py": (1, NETWORK),
    "web/tools/mcp/google_api_proxy.py": (1, NETWORK),
    "web/tools/mcp/google_slides.py": (3, NETWORK),
    "web/tracking/standalone_usage.py": (
        1,
        "The host's usage gate hook, called before work starts; the usage "
        "delivery that writes is tracked separately (run_tracked in "
        "UsageScope).",
    ),
    # -- web/channels: reads and sends; the writing threads are tracked ---------
    "web/channels/feishu/bot.py": (
        4,
        "Two file downloads into a temp or staging directory and two message "
        "create/patch calls to the Feishu API; the database-writing threads of "
        "this file (input lookup and acceptance, upload staging) are tracked.",
    ),
    "web/channels/feishu/trace_handler.py": (
        2,
        "Message create/patch calls to the Feishu API; no local shared write.",
    ),
    "web/channels/telegram/bot.py": (
        1,
        "Resolves the user's ASR model in one short read-only session; the "
        "writing threads of this file (input lookup and acceptance, upload "
        "staging) are tracked.",
    ),
    # -- web/models, web/services: helpers and execution-owned work -------------
    "web/models/auth_database.py": (
        1,
        "run_auth_db_worker: drains the worker before a cancellation "
        "propagates, so a cancelled and awaited task owns no live thread.",
    ),
    "web/services/db_runtime.py": (
        1,
        "run_db_io_cancellation_safe itself: drains the worker before a "
        "cancellation propagates, so a cancelled and awaited task owns no "
        "live thread.",
    ),
    "web/services/agent_management.py": (
        2,
        "Runtime-key delivery and its compensation, both awaited through "
        "await_task_settlement (the worker is drained before a cancellation "
        "propagates); reached from the agent routes (api/agents.py, "
        "api/v1/agents.py) and workforce creation, which are HTTP producers. "
        "The builder's create_agent/update_agent tools use no thread.",
    ),
    "web/services/durable_sandbox_lifecycle.py": (
        4,
        "DurableSandboxDeleteCoordinator has no caller in this repository "
        "(an embedding host constructs it); it is not part of any loop "
        "retire_writers stops.",
    ),
    "web/services/gmail_triggers.py": (
        5,
        "Gmail Pub/Sub webhook collection (collect_gmail_pubsub_events): "
        "webhook ingress is an HTTP producer, listed under 'Not covered'. The "
        "scheduled watch renewal and provisioning sweeps are tracked.",
    ),
    "web/services/mcp_oauth.py": (1, NETWORK),
    "web/services/task_coordinator_runtime.py": (2, EXEC),
    "web/services/task_delete.py": (
        5,
        "Task deletion, called by the delete routes (api/chat.py, "
        "api/v1/tasks.py): " + HTTP,
    ),
    "web/services/task_event_trace_handler.py": (
        3,
        "Reads one stream event's display data, the prior-message check and "
        "the task description for a websocket trace; read-only.",
    ),
    "web/services/task_execution.py": (7, EXEC),
    "web/services/task_execution_controller.py": (2, EXEC),
    "web/services/task_lease_service.py": (
        1,
        "Lease acquisition, awaited through await_task_settlement. The lease "
        "is the cross-process mutex for a run, so a stray acquire is "
        "arbitrated by the lease itself; the callers are coordinators and "
        "command attempts that execution_status() reports.",
    ),
    "web/services/task_orchestrator.py": (4, EXEC),
    "web/services/task_runtime.py": (
        2,
        "The bounded executor for task-runtime hooks of running executions "
        "(shutdown_task_runtime_hook_executor joins it at shutdown): " + EXEC,
    ),
    "web/services/trace_database.py": (
        3,
        "Trace-event persistence of running executions, awaited through "
        "drain_async_task_cancellation_safe and closed by "
        "close_trace_database_runtime at shutdown: " + EXEC,
    ),
    "web/services/trace_handlers.py": (1, READ),
    "web/services/trigger_providers/gmail.py": (
        3,
        "Trigger register/unregister from the trigger HTTP routes, and the "
        "webhook OIDC verification: HTTP producers.",
    ),
    "web/services/triggers.py": (
        2,
        "Two pools whose results are awaited inside the call (the `with` "
        "block joins them before it returns), used by the synchronous "
        "trigger CRUD that the HTTP routes run. The retire-relevant trigger "
        "run threads of this file are tracked.",
    ),
    "web/services/uploaded_file_cleanup_publication.py": (
        3,
        "Restores or copies a stored file to local disk and enters/exits the "
        "per-file cleanup lock, for HTTP routes and executions: " + FILES,
    ),
    # -- core/tools/adapters: tool calls of a running execution ------------------
    "core/tools/adapters/vibe/a2a_agent_tool.py": (1, NETWORK),
    "core/tools/adapters/vibe/command_executor.py": (1, EXEC),
    "core/tools/adapters/vibe/computer.py": (1, EXEC),
    "core/tools/adapters/vibe/factory.py": (3, EXEC),
    "core/tools/adapters/vibe/javascript_executor.py": (1, EXEC),
    "core/tools/adapters/vibe/mcp_adapter.py": (3, EXEC),
    "core/tools/adapters/vibe/output_filter_wrapper.py": (1, EXEC),
    "core/tools/adapters/vibe/python_executor.py": (1, EXEC),
    "core/tools/adapters/vibe/sandboxed_tool/sandboxed_tool_wrapper.py": (4, EXEC),
    "core/tools/adapters/vibe/vision_tool.py": (1, FILES),
    # -- core/tools/core: RAG storage, tools ------------------------------------
    "core/tools/core/browser_use.py": (1, FILES),
    "core/tools/core/RAG_tools/kb/collection_handle.py": (
        1,
        "A pool that runs the dense query beside the sparse one and is joined "
        "by its `with` block before the search returns; read-only. The "
        "document-delete write of this file goes through run_blocking.",
    ),
    "core/tools/core/RAG_tools/kb/coordinator.py": (
        1,
        "_run_in_separate_loop: a thread that is joined immediately, so the "
        "caller does not return while it runs. Every to_thread of this file "
        "goes through run_blocking.",
    ),
    "core/tools/core/RAG_tools/management/collection_manager.py": (
        3,
        "Collection statistics aggregation (read-only) and a thread joined "
        "immediately in the sync-to-async bridge.",
    ),
    "core/tools/core/RAG_tools/management/collections.py": (2, READ),
    "core/tools/core/RAG_tools/management/write_leases.py": (
        3,
        "Three daemon helpers: the heartbeat that renews the operation's own "
        "lease rows while its scope runs (stopped and joined before release, "
        "and the async decorators release it off the event loop), plus the "
        "fire-and-forget compaction and expired-row GC of the lease table, "
        "neither of which writes application data. All belong to the "
        "HTTP-route and execution classes retire_writers lists under "
        "'Not covered'.",
    ),
    "core/tools/core/RAG_tools/management/migration_state.py": (
        1,
        "The freeze guard reads one state row (and opens the metadata "
        "connection to read it). Lease writes go off the loop through "
        "run_blocking, and a scope releases on its own exit.",
    ),
    "core/tools/core/RAG_tools/pipelines/document_ingestion.py": (
        2,
        "Embedding encode calls: a provider call and a pool joined by its "
        "`with` block inside the ingestion call. The ingestion call itself "
        "(the writer) is reached through run_blocking from the builder and "
        "otherwise from HTTP routes and executions.",
    ),
    "core/tools/core/RAG_tools/storage/file_reference.py": (
        2,
        "Enter/exit of the file-reference protection lock around a document "
        "upsert that the caller awaits inline; the lock itself writes no row.",
    ),
    "core/tools/core/RAG_tools/storage/lancedb_stores.py": (
        6,
        "Opens a LanceDB connection and reads collection metadata; creates no "
        "table. The metadata-table creation goes through run_blocking.",
    ),
    "core/tools/core/RAG_tools/web_crawler/crawler.py": (1, NETWORK),
    "core/tools/core/document_search.py": (1, READ),
    "core/tools/core/exa_web_search.py": (2, NETWORK),
    "core/tools/core/excel.py": (2, FILES),
    "core/tools/core/image_tool.py": (6, FILES),
    "core/tools/core/pptx_conversion.py": (3, FILES),
    "core/tools/core/pptx_render.py": (10, FILES),
    "core/tools/core/video_tool.py": (9, FILES),
    "core/tools/core/vision_tool.py": (5, FILES),
}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
        owner = func.value
        if name == "run_sync":
            # anyio.to_thread.run_sync only.
            return (
                name
                if isinstance(owner, ast.Attribute) and owner.attr == "to_thread"
                else None
            )
        if name == "Thread":
            return (
                name
                if isinstance(owner, ast.Name) and owner.id == "threading"
                else None
            )
        return name if name in THREAD_CALLS else None
    if isinstance(func, ast.Name) and func.id in THREAD_CALLS - {"run_sync"}:
        return func.id
    return None


def _scan() -> dict[str, list[int]]:
    found: dict[str, list[int]] = {}
    for sub in SCANNED:
        for path in sorted((SRC / sub).rglob("*.py")):
            rel = path.relative_to(SRC).as_posix()
            if rel in IMPLEMENTATION:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            lines = sorted(
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and _call_name(node)
            )
            if lines:
                found[rel] = lines
    return found


FOUND = _scan()

_FIX = (
    "\nA thread started like this keeps running after its awaiting task is "
    "cancelled and can still write after retire_writers() reported success.\n"
    "- If it can write to the database, a vector store or other shared state: "
    "in xagent.web use `run_tracked(fn, ...)` "
    "(xagent.web.services.writer_workers); in xagent.core use "
    "`run_blocking(fn, ...)` (xagent.core.utils.blocking).\n"
    "- If it cannot (read-only, filesystem scratch, network), or belongs to a "
    "class listed under 'Not covered' in app.retire_writers, add or raise the "
    "entry for the file in ALLOWED in this test, with a reason that is true "
    "for every call in the file, and say so in the PR."
)


def test_every_thread_starting_call_is_tracked_or_audited():
    unlisted = {rel: lines for rel, lines in FOUND.items() if rel not in ALLOWED}
    assert not unlisted, (
        "Untracked, unaudited thread-starting calls:\n"
        + "\n".join(
            f"  {rel}: lines {lines}" for rel, lines in sorted(unlisted.items())
        )
        + _FIX
    )


@pytest.mark.parametrize("rel", sorted(ALLOWED))
def test_an_audited_file_keeps_exactly_its_audited_call_count(rel):
    count, reason = ALLOWED[rel]
    actual = FOUND.get(rel, [])
    assert (SRC / rel).is_file(), f"{rel} no longer exists: remove its entry"
    assert len(actual) <= count, (
        f"{rel} has {len(actual)} thread-starting calls (lines {actual}) but "
        f"{count} were audited." + _FIX
    )
    assert len(actual) == count, (
        f"{rel} now has {len(actual)} thread-starting calls, {count} audited: "
        "lower the audited count (a call was removed or moved onto "
        "run_tracked/run_blocking) so the audit stays exact."
    )
    assert len(reason) > 40, f"{rel}: the reason must say why the file is safe"


def test_the_scan_sees_the_known_call_shapes():
    """The detector itself must keep finding each shape it claims to find."""
    tree = ast.parse(
        "import asyncio, anyio, threading\n"
        "from concurrent.futures import ThreadPoolExecutor\n"
        "async def f(loop):\n"
        "    await asyncio.to_thread(g)\n"
        "    await loop.run_in_executor(None, g)\n"
        "    await anyio.to_thread.run_sync(g)\n"
        "    await run_in_thread_with_telemetry('x', g)\n"
        "    threading.Thread(target=g)\n"
        "    ThreadPoolExecutor(1)\n"
        "    other.run_sync(g)\n"
        "    other.Thread(g)\n"
    )
    names = sorted(
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node))
    )
    assert names == sorted(
        [
            "to_thread",
            "run_in_executor",
            "run_sync",
            "run_in_thread_with_telemetry",
            "Thread",
            "ThreadPoolExecutor",
        ]
    )


def test_the_tracked_call_sites_are_still_there():
    """The files the docstring of retire_writers promises are tracked still use the trackers."""
    expected = {
        "web/app.py": "run_tracked(",
        "web/api/websocket.py": "run_tracked(",
        "web/channels/feishu/bot.py": "run_tracked(",
        "web/channels/slack/bot.py": "run_tracked(",
        "web/channels/telegram/bot.py": "run_tracked(",
        "web/services/channel_runtime.py": "run_tracked(",
        "web/services/shared_channel_execution.py": "run_tracked(",
        "web/services/gmail_provisioning.py": "start_tracked_thread(",
        "web/services/triggers.py": "run_tracked(",
        "web/services/task_cleanup_obligations.py": "run_tracked(",
        "web/services/task_retention_purge.py": "run_tracked(",
        "core/tools/core/RAG_tools/kb/coordinator.py": "run_blocking(",
        "core/tools/core/RAG_tools/kb/api_compatibility.py": "run_blocking(",
        "core/tools/core/RAG_tools/pipelines/web_ingestion.py": "run_blocking(",
        "core/tools/adapters/vibe/file_ingestion_tool.py": "run_blocking(",
        "core/tools/adapters/vibe/function.py": "run_blocking(",
    }
    missing = [
        rel
        for rel, needle in expected.items()
        if needle not in (SRC / rel).read_text(encoding="utf-8")
    ]
    assert not missing, f"no longer routed through its tracker: {missing}"
