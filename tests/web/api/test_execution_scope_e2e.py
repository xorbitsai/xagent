"""#757 cross-cutting acceptance: every subsystem consumes the same scope.

Drives the real ``get_agent_for_task`` build through the resolver path (and
the persisted-snapshot path for delegated tasks) and checks each
consumption point in one pass, so no execution path reaches one subsystem
scoped and another unscoped:

* sandbox lifecycle key (container family),
* sandbox mount workspace-config (base dir + segments),
* AgentService workspace base dir + carried scope segments,
* the recorded scope fingerprint,

plus disjoint namespaces between two scopes under one platform user and
byte-for-byte unchanged unscoped behavior. That memory result sets are
likewise disjoint through the same contextvar mechanism is pinned in
``tests/web/test_execution_scope_memory.py``.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tests.shared.execution_scope import register_scope_resolver
from xagent.config import get_uploads_dir
from xagent.core.execution_scope import (
    DeferToSnapshot,
    ExecutionScope,
    scope_fingerprint,
    set_execution_scope_snapshot_loader,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.workspace import scoped_user_root
from xagent.sandbox.base import SandboxMountIntent
from xagent.web.api.chat import AgentServiceManager, create_default_tools
from xagent.web.models.agent import AgentStatus
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.llm_utils import AgentRuntimeFields
from xagent.web.services.task_setup_snapshot import (
    RuntimeUserFields,
    TaskSetupSnapshot,
    _TaskFields,
)

SCOPE_A = ExecutionScope(
    sandbox_key_suffix="tenant-a",
    workspace_segments=("tenant-a",),
    memory_dimensions={"tenant": "a"},
)
SCOPE_B = ExecutionScope(
    sandbox_key_suffix="tenant-b",
    workspace_segments=("tenant-b",),
    memory_dimensions={"tenant": "b"},
)

# Two end users of one client application (#79-01): same sandbox suffix and
# same mount prefix, but deeper (full) workspace_segments differ. The prefix
# is what lets them share one container.
SCOPE_EU7 = ExecutionScope(
    sandbox_key_suffix="client-3",
    workspace_segments=("client-3", "eu-7"),
    sandbox_mount_segments=("client-3",),
)
SCOPE_EU8 = ExecutionScope(
    sandbox_key_suffix="client-3",
    workspace_segments=("client-3", "eu-8"),
    sandbox_mount_segments=("client-3",),
)


def _make_user() -> User:
    return User(id=1, username="e2e-user", password_hash="hash", is_admin=False)


def _make_task_row(task_id: int) -> Task:
    return Task(
        id=task_id,
        user_id=1,
        title="e2e task",
        description="x",
        status=TaskStatus.PENDING,
        agent_id=7,
        agent_type="standard",
    )


def _build_snapshot(task_id: int) -> TaskSetupSnapshot:
    return TaskSetupSnapshot(
        task=_TaskFields(
            id=task_id,
            user_id=1,
            status=TaskStatus.PENDING,
            agent_id=7,
            agent_config=None,
            model_name=None,
            compact_model_name=None,
            execution_mode="flash",
            agent_type="standard",
        ),
        runtime_user=RuntimeUserFields(id=1, is_admin=False),
        has_reconstructable_history=False,
        task_pattern="single_call",
        task_llm=None,
        task_fast_llm=None,
        task_vision_llm=None,
        task_compact_llm=None,
        agent=AgentRuntimeFields(
            id=7,
            name="e2e-agent",
            status=AgentStatus.PUBLISHED,
            instructions="be terse",
        ),
        agent_config={
            "llms": (None, None, None, None),
            "execution_mode": "flash",
            "instructions": "be terse",
            "skills": [],
            "knowledge_bases": [],
            "tool_categories": ["basic"],
        },
        excluded_agent_id=7,
    )


def _build_db_mock(task_row: Task) -> MagicMock:
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = task_row
    return db


class _Build:
    """Everything one ``get_agent_for_task`` build touched, per subsystem."""

    def __init__(self) -> None:
        self.sandbox_lifecycle: tuple[str, str] | None = None
        self.sandbox_mount_intent: SandboxMountIntent | None = None
        self.agent_service_kwargs: dict[str, Any] | None = None
        self.recorded_sandbox_key: str | None = None
        self.recorded_fingerprint: Any = None


async def _run_build(manager: AgentServiceManager, task_id: int) -> _Build:
    build = _Build()
    manager._default_llm = MagicMock()

    fake_sandbox_manager = MagicMock()

    async def _lease_provider(
        lifecycle_type, lifecycle_id, *, mount_intent=None, prepare_root=None
    ):
        build.sandbox_lifecycle = (lifecycle_type, lifecycle_id)
        build.sandbox_mount_intent = mount_intent
        return AsyncMock()

    fake_sandbox_manager.get_or_create_lease_provider = AsyncMock(
        side_effect=_lease_provider
    )

    with ExitStack() as stack:
        stack.enter_context(
            patch.object(manager, "_load_persisted_conversation_history")
        )
        stack.enter_context(
            patch.object(manager, "_load_persisted_execution_context", new=AsyncMock())
        )
        stack.enter_context(
            patch("xagent.web.api.chat.create_task_tracer", return_value=MagicMock())
        )
        stack.enter_context(
            patch(
                "xagent.web.api.chat.create_default_tools",
                new=AsyncMock(return_value=([], MagicMock())),
            )
        )
        stack.enter_context(
            patch(
                "xagent.web.sandbox_manager.get_sandbox_manager",
                return_value=fake_sandbox_manager,
            )
        )
        agent_service_mock = stack.enter_context(
            patch("xagent.web.api.chat.AgentService")
        )
        try:
            await manager.get_agent_for_task(
                task_id=task_id,
                db=_build_db_mock(_make_task_row(task_id)),
                user=_make_user(),
                task_setup_snapshot=_build_snapshot(task_id),
            )
        except Exception:
            pass

    if agent_service_mock.call_args is not None:
        build.agent_service_kwargs = agent_service_mock.call_args.kwargs
    build.recorded_sandbox_key = manager._agent_sandbox_keys.get(task_id)
    build.recorded_fingerprint = manager._agent_scope_fingerprints.get(task_id)
    return build


@pytest.mark.asyncio
async def test_scoped_build_applies_the_scope_to_every_subsystem() -> None:
    """No partially-applied scope: one resolved scope reaches the sandbox
    lifecycle key, the sandbox mount intent, the workspace base dir, the
    carried segments, and the cache fingerprint — in the same build.

    SCOPE_A does not set ``isolate_external_dirs``, so the CA mount
    intent folds to the unscoped user root (build_chat_workspace_binding's
    covering-absorbs-root rule): the container family still narrows via
    ``sandbox_key_suffix``, but the Actor-logical workspace base dir (below)
    is what actually isolates task 42's files under the tenant-a subtree.
    """
    register_scope_resolver(lambda task_id: SCOPE_A)
    build = await _run_build(AgentServiceManager(), 42)

    scoped_base = str(scoped_user_root(get_uploads_dir(), 1, ("tenant-a",)))
    unscoped_root = str(get_uploads_dir() / "user_1")
    assert build.sandbox_lifecycle == ("user", "1:tenant-a")
    assert build.sandbox_mount_intent.mount_root == unscoped_root
    assert build.recorded_sandbox_key == "user:1:tenant-a"
    assert build.agent_service_kwargs["workspace_base_dir"] == scoped_base
    assert build.agent_service_kwargs["scope_segments"] == ("tenant-a",)
    assert build.recorded_fingerprint == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_two_scopes_under_one_user_are_disjoint_everywhere() -> None:
    def resolver(task_id: str):
        return {"42": SCOPE_A, "43": SCOPE_B}.get(task_id)

    register_scope_resolver(resolver)
    manager = AgentServiceManager()
    build_a = await _run_build(manager, 42)
    build_b = await _run_build(manager, 43)
    register_scope_resolver(None)
    build_unscoped = await _run_build(AgentServiceManager(), 44)

    keys = {
        build_a.recorded_sandbox_key,
        build_b.recorded_sandbox_key,
        build_unscoped.recorded_sandbox_key,
    }
    assert keys == {"user:1:tenant-a", "user:1:tenant-b", "user:1"}
    base_dirs = {
        build.agent_service_kwargs["workspace_base_dir"]
        for build in (build_a, build_b, build_unscoped)
    }
    assert len(base_dirs) == 3
    fingerprints = {
        build.recorded_fingerprint for build in (build_a, build_b, build_unscoped)
    }
    assert len(fingerprints) == 3 and None in fingerprints


@pytest.mark.asyncio
async def test_delegated_task_builds_scoped_from_persisted_snapshot() -> None:
    """A delegated task the embedder does not own: the snapshot drives the build.

    This is what :class:`DeferToSnapshot` exists for. The resolver cannot
    supply the namespace because it does not know it -- that is why it defers
    -- so its fallback claims no scoping at all, and it says so explicitly
    with ``snapshot_defines_namespace=True``. Without that declaration the
    same abstention fails closed, because an abstention that claims no
    namespace authority must not hand one out (see
    ``tests/core/test_execution_scope.py``).
    """
    register_scope_resolver(
        lambda task_id: DeferToSnapshot(
            fallback=ExecutionScope(strict_memory_isolation=True),
            snapshot_defines_namespace=True,
        ),
    )
    set_execution_scope_snapshot_loader(
        lambda task_id: SCOPE_A if task_id == "42" else None
    )
    build = await _run_build(AgentServiceManager(), 42)

    assert build.sandbox_lifecycle == ("user", "1:tenant-a")
    assert build.recorded_sandbox_key == "user:1:tenant-a"
    assert build.agent_service_kwargs["scope_segments"] == ("tenant-a",)
    assert build.recorded_fingerprint == scope_fingerprint(SCOPE_A)


@pytest.mark.asyncio
async def test_resolver_authoritative_unscoped_ignores_persisted_snapshot() -> None:
    """``None`` from the resolver is authoritative unscoped for the task, not
    an abstention: a persisted snapshot is not consulted at all, unlike the
    :class:`DeferToSnapshot` case above."""
    register_scope_resolver(lambda task_id: None)
    set_execution_scope_snapshot_loader(
        lambda task_id: SCOPE_A if task_id == "42" else None
    )
    build = await _run_build(AgentServiceManager(), 42)

    assert build.sandbox_lifecycle == ("user", "1")
    assert build.recorded_sandbox_key == "user:1"
    assert build.agent_service_kwargs["scope_segments"] == ()
    assert build.recorded_fingerprint is None


@pytest.mark.asyncio
async def test_unscoped_build_is_byte_identical_to_pre_757_behavior() -> None:
    build = await _run_build(AgentServiceManager(), 42)

    legacy_base = str(get_uploads_dir() / "user_1")
    assert build.sandbox_lifecycle == ("user", "1")
    assert build.sandbox_mount_intent.mount_root == legacy_base
    assert build.recorded_sandbox_key == "user:1"
    assert build.agent_service_kwargs["workspace_base_dir"] == legacy_base
    assert build.agent_service_kwargs["scope_segments"] == ()
    assert build.recorded_fingerprint is None


@pytest.mark.asyncio
async def test_tool_workspace_lives_inside_the_tree_the_sandbox_binds(
    tmp_path, monkeypatch
) -> None:
    """One spelling reaches both the sandbox mount and the tool workspace.

    The sandbox binds a canonical, lexically normalized path -- the mount
    intent's own identity domain, since desired-vs-observed comparison has to
    be lexical -- while ``TaskWorkspace`` ``resolve()``s the base dir it is
    handed. A configured uploads dir carrying a ``..`` segment separates the
    two: composing the tool base dir from the raw spelling hands the workspace
    a path the mount intent never reports, and the desired spec then depends
    on which producer wrote it.
    """
    base = tmp_path / "base"
    base.mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", f"{base}/nonexistent/..")
    register_scope_resolver(lambda task_id: SCOPE_A)

    build = await _run_build(AgentServiceManager(), 42)
    mount_root = build.sandbox_mount_intent.mount_root
    assert mount_root == str(base / "user_1")
    assert build.agent_service_kwargs["workspace_base_dir"] == str(
        base / "user_1" / "tenant-a"
    )

    # The tool side through its own production path: the real
    # create_default_tools composes the config and the real factory
    # materializes the workspace from it. Only the tool creation itself and
    # the session factory are mocked -- what the workspace is rooted at is
    # not.
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
                new=AsyncMock(return_value=[]),
            )
        )
        stack.enter_context(
            patch("xagent.web.models.database.get_session_local", return_value=None)
        )
        _tools, tool_config = await create_default_tools(
            db=None,
            user=_make_user(),
            task_id="web_task_42",
            workspace_owner_id=1,
            scope=SCOPE_A,
        )
    workspace_config = tool_config.get_workspace_config()
    workspace = ToolFactory._create_workspace(workspace_config)

    # The spelling, not only the directory it happens to resolve to: the
    # desired spec is compared byte-wise, so a base dir carrying a ``..``
    # segment makes the same workspace look like a different configuration.
    assert workspace_config["base_dir"] == str(base / "user_1" / "tenant-a")
    assert workspace is not None
    assert workspace.base_dir.is_relative_to(Path(mount_root).resolve())


@pytest.mark.asyncio
async def test_mount_prefix_shares_sandbox_root_across_deeper_segments() -> None:
    """#79-01 through the real build seam: two scopes sharing a sandbox
    suffix and mount prefix but differing in deeper workspace_segments land
    on the same sandbox lifecycle key and the same physical mount root,
    while their workspace base dir and carried segments stay disjoint at
    the full segments.

    Neither scope sets ``isolate_external_dirs``, so the ``client-3``
    mount prefix folds away into the wider unscoped user root (the same
    covering-absorbs-root rule as the plain-suffix case above) — the two
    scopes end up sharing that wider root rather than the narrower
    ``client-3`` prefix, which still proves the point (one container,
    reachable from a shared physical mount) even more strongly.
    """

    def resolver(task_id: str):
        return {"42": SCOPE_EU7, "43": SCOPE_EU8}.get(task_id)

    register_scope_resolver(resolver)
    manager = AgentServiceManager()
    build_eu7 = await _run_build(manager, 42)
    build_eu8 = await _run_build(manager, 43)

    unscoped_root = str(get_uploads_dir() / "user_1")
    # Same container family and same mount root — the prerequisite for reuse.
    assert build_eu7.sandbox_lifecycle == ("user", "1:client-3")
    assert build_eu8.sandbox_lifecycle == ("user", "1:client-3")
    assert build_eu7.sandbox_mount_intent.mount_root == unscoped_root
    assert build_eu8.sandbox_mount_intent.mount_root == unscoped_root
    # Full segments still diverge: workspace base dir and carried segments
    # place each end user in its own subtree — of the client-3 prefix (the
    # Actor-logical workspace root always uses the full workspace_segments,
    # independent of how the sandbox mount folded).
    client_prefix = str(scoped_user_root(get_uploads_dir(), 1, ("client-3",)))
    base_eu7 = build_eu7.agent_service_kwargs["workspace_base_dir"]
    base_eu8 = build_eu8.agent_service_kwargs["workspace_base_dir"]
    assert base_eu7 != base_eu8
    assert base_eu7.startswith(client_prefix) and base_eu8.startswith(client_prefix)
    assert build_eu7.agent_service_kwargs["scope_segments"] == ("client-3", "eu-7")
    assert build_eu8.agent_service_kwargs["scope_segments"] == ("client-3", "eu-8")
    # Distinct tasks: the differing mount is not the only differentiator, the
    # full segments keep the fingerprints (hence the caches) apart.
    assert build_eu7.recorded_fingerprint != build_eu8.recorded_fingerprint


@pytest.mark.asyncio
async def test_prefix_shared_mount_passes_config_equivalence_gate() -> None:
    """Close the chain the unit tests leave open: feed the two builds'
    mount intents through the real ``SandboxManager`` and confirm the
    shared mount is accepted by ``_ensure_config_equivalent`` (no
    ``SandboxRuntimeConflictError``), whereas mounting the full segments
    would be rejected — the exact rejection #79-01 removes."""
    from xagent.sandbox.base import (
        ResolvedSandboxRuntimeSpec,
        SandboxRuntimeConflictError,
    )
    from xagent.web.sandbox_manager import SandboxManager

    def resolver(task_id: str):
        return {"42": SCOPE_EU7, "43": SCOPE_EU8}.get(task_id)

    register_scope_resolver(resolver)
    manager = AgentServiceManager()
    build_eu7 = await _run_build(manager, 42)
    build_eu8 = await _run_build(manager, 43)

    lifecycle_id = "1:client-3"

    def _spec_for(mount_root: str) -> ResolvedSandboxRuntimeSpec:
        # Derive the mount volume the way _make_default_volumes does, so the
        # spec is a genuine consequence of each scope's mount root rather
        # than a hand-built literal.
        [(mount_path, _create)] = SandboxManager._workspace_mount_paths(
            "user", lifecycle_id, SandboxMountIntent(mount_root=mount_root)
        )
        return ResolvedSandboxRuntimeSpec.from_parts(
            template_type="image",
            image="img:v1",
            volumes=[(str(mount_path), "/sandbox/workspace", "rw")],
        )

    # The actual fact under test: two distinct scopes (different Actor
    # segments, eu-7 vs eu-8) independently compute the exact same mount
    # root -- that's the shared-prefix fold, not a hand-picked identical
    # input. Because the root is genuinely equal, the specs built from it
    # are equal too, which is why the gate below accepts the reuse (contrast
    # the full-segments specs, which never share a root and are rejected).
    assert (
        build_eu7.sandbox_mount_intent.mount_root
        == build_eu8.sandbox_mount_intent.mount_root
    )
    spec_eu7 = _spec_for(build_eu7.sandbox_mount_intent.mount_root)
    spec_eu8 = _spec_for(build_eu8.sandbox_mount_intent.mount_root)
    assert spec_eu7 == spec_eu8
    SandboxManager._ensure_config_equivalent(
        f"user::{lifecycle_id}", spec_eu7, spec_eu8
    )  # must not raise

    # Contrast: had the mount stayed at the full segments, the two specs
    # would differ and the reuse would be rejected.
    full_spec_eu7 = _spec_for(build_eu7.agent_service_kwargs["workspace_base_dir"])
    full_spec_eu8 = _spec_for(build_eu8.agent_service_kwargs["workspace_base_dir"])
    assert not SandboxManager._config_equivalent(full_spec_eu7, full_spec_eu8)
    with pytest.raises(SandboxRuntimeConflictError):
        SandboxManager._ensure_config_equivalent(
            f"user::{lifecycle_id}", full_spec_eu7, full_spec_eu8
        )
