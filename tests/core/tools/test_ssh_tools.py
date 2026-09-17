from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.ssh import (
    ActorRef,
    BoundTargetInfo,
    PrincipalRef,
    ResolvedSshTarget,
    SensitiveSshCredential,
    SshError,
    SshErrorCode,
    SshExecutionContext,
    SshSecretHandle,
)
from xagent.core.ssh.executor import SshExecuteOutcome
from xagent.core.ssh.runner import SshRunResult
from xagent.core.tools.adapters.vibe.ssh_tools import (
    SshDownloadTool,
    SshExecuteTool,
    SshListTargetsTool,
    SshUploadTool,
    _agent_id_for_task,
    _egress_from_env,
    _numeric_task_id,
)
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.database import Base
from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.user import User
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)


def _ctx() -> SshExecutionContext:
    return SshExecutionContext(
        actor=ActorRef(actor_type="user", actor_id="u"),
        execution_principal=PrincipalRef(principal_type="user", principal_id="u"),
        agent_id=1,
        task_id=None,
        turn_id=None,
        request_id="r",
    )


class _Provider:
    def __init__(self, resolved=None, targets=None, error=None):
        self._resolved = resolved
        self._targets = targets or []
        self._error = error

    async def resolve(self, context, target_alias):
        if self._error is not None:
            raise self._error
        return self._resolved

    async def read_version(self, secret_handle):
        raise NotImplementedError

    async def list_bound_targets(self, context):
        return self._targets


def _resolved(caps=("execute",)) -> ResolvedSshTarget:
    return ResolvedSshTarget(
        target_public_id="t",
        hostname="h",
        port=22,
        username="d",
        remote_root=None,
        capabilities=frozenset(caps),
        approval_policy="always",
        secret_handle=SshSecretHandle(credential_id="c", version_id="v"),
        known_hosts="h ssh-ed25519 AAAA\n",
        credential_public_id="c",
        credential_version_id="v",
        host_key_fingerprint="SHA256:x",
    )


async def test_list_targets_returns_aliases() -> None:
    provider = _Provider(
        targets=[
            BoundTargetInfo(
                alias="prod", display_name="Prod", capabilities=frozenset({"execute"})
            ),
        ]
    )
    tool = SshListTargetsTool(provider=provider, context=_ctx())
    out = await tool.run_json_async({})
    assert out["targets"][0]["alias"] == "prod"
    assert "execute" in out["targets"][0]["capabilities"]


class _Executor:
    def __init__(self, outcome=None, error=None):
        self._outcome = outcome
        self._error = error

    async def execute(self, context, *, target_alias, command, timeout_seconds):
        if self._error is not None:
            raise self._error
        return self._outcome


async def test_execute_returns_outcome() -> None:
    outcome = SshExecuteOutcome(
        exit_code=0, stdout="ok", stderr="", truncated=False, duration_ms=5
    )
    tool = SshExecuteTool(executor=_Executor(outcome=outcome), context=_ctx())
    out = await tool.run_json_async({"target": "prod", "command": "uptime"})
    assert out["ok"] is True
    assert out["exit_code"] == 0
    assert out["stdout"] == "ok"
    assert out["truncated"] is False


async def test_execute_surfaces_ssh_error() -> None:
    tool = SshExecuteTool(
        executor=_Executor(error=SshError(SshErrorCode.TARGET_DISABLED, "disabled")),
        context=_ctx(),
    )
    out = await tool.run_json_async({"target": "prod", "command": "uptime"})
    assert out["ok"] is False
    assert out["error_code"] == "ssh_target_disabled"


def test_execute_sync_not_supported() -> None:
    tool = SshExecuteTool(executor=_Executor(), context=_ctx())
    with pytest.raises(NotImplementedError):
        tool.run_json_sync({"target": "prod", "command": "x"})


class _RecordingTransferExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def upload(
        self, context, *, target_alias, local_path, remote_path, overwrite
    ):
        self.calls.append(("upload", target_alias, local_path, remote_path, overwrite))

    async def download(
        self, context, *, target_alias, remote_path, local_path, overwrite
    ):
        self.calls.append(
            ("download", target_alias, remote_path, local_path, overwrite)
        )


class _FakeWorkspace:
    """Minimal workspace: resolves under a root and rejects escapes / missing
    files, mirroring TaskWorkspace's containment contract for these tools."""

    def __init__(self, root):
        self.root = root

    def _contained(self, p: str):
        resolved = (self.root / p).resolve()
        if not str(resolved).startswith(str(self.root.resolve())):
            raise ValueError("path escapes workspace")
        return resolved

    def resolve_path_with_search(self, p: str):
        resolved = self._contained(p)
        if not resolved.exists():
            raise FileNotFoundError(p)
        return resolved

    def resolve_path(self, p: str, default_dir: str = "output"):
        return self._contained(p)


async def test_upload_tool_passes_resolved_path_to_executor(tmp_path) -> None:
    (tmp_path / "f.txt").write_text("x")
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(
        executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx()
    )
    out = await tool.run_json_async(
        {"target": "prod", "local_path": "f.txt", "remote_path": "/srv/f.txt"}
    )
    assert out["ok"] is True
    assert ex.calls[0][0] == "upload"
    assert ex.calls[0][3] == "/srv/f.txt"


async def test_upload_tool_rejects_workspace_escape(tmp_path) -> None:
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(
        executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx()
    )
    out = await tool.run_json_async(
        {"target": "prod", "local_path": "../../etc/passwd", "remote_path": "/srv/x"}
    )
    assert out["ok"] is False
    assert out["error_code"] == "ssh_operation_not_allowed"
    assert ex.calls == []  # executor never reached — no connection, no secret


async def test_download_tool_writes_into_workspace(tmp_path) -> None:
    ex = _RecordingTransferExecutor()
    tool = SshDownloadTool(
        executor=ex, workspace=_FakeWorkspace(tmp_path), context=_ctx()
    )
    out = await tool.run_json_async(
        {"target": "prod", "remote_path": "/srv/f", "local_path": "out.txt"}
    )
    assert out["ok"] is True
    assert ex.calls[0][0] == "download"


async def test_transfer_tool_without_workspace_fails_closed() -> None:
    ex = _RecordingTransferExecutor()
    tool = SshUploadTool(executor=ex, workspace=None, context=_ctx())
    out = await tool.run_json_async(
        {"target": "prod", "local_path": "f", "remote_path": "/srv/f"}
    )
    assert out["ok"] is False
    assert ex.calls == []


def test_egress_from_env_allowlist(monkeypatch) -> None:
    monkeypatch.setenv("XAGENT_SSH_ALLOW_CIDRS", "127.0.0.0/8, 10.0.0.0/8")
    assert _egress_from_env().allow_cidrs == ("127.0.0.0/8", "10.0.0.0/8")


def test_egress_from_env_default_empty(monkeypatch) -> None:
    monkeypatch.delenv("XAGENT_SSH_ALLOW_CIDRS", raising=False)
    assert _egress_from_env().allow_cidrs == ()


def test_numeric_task_id_parses_workspace_prefixed_id() -> None:
    # config.get_task_id() hands the tool a workspace-scoped string, not the
    # bare DB id — e.g. "web_task_30". Non-task ids ("tools_list") yield None.
    assert _numeric_task_id("web_task_30") == 30
    assert _numeric_task_id(30) == 30
    assert _numeric_task_id("30") == 30
    assert _numeric_task_id("tools_list") is None
    assert _numeric_task_id(None) is None


def test_numeric_task_id_rejects_ids_that_merely_end_in_digits() -> None:
    # A trailing-digit search would read the uuid suffix as somebody else's
    # task id, and the caller derives the owner scope from whatever row it hits.
    assert _numeric_task_id("agent_7_a1b2cd34") is None
    assert _numeric_task_id("preview_ab12cd34") is None
    assert _numeric_task_id("web_task_30x") is None


@pytest.fixture
def task_db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'ssh-tools.db'}")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autoflush=False, bind=engine, expire_on_commit=False)
    try:
        yield session_factory
    finally:
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def _new_user(db, name: str) -> User:
    user = User(username=name, email=f"{name}@example.test", password_hash="x")
    db.add(user)
    db.flush()
    return user


def _new_agent(db, user_id: int, **overrides) -> Agent:
    agent = Agent(
        user_id=user_id,
        name=overrides.pop("name", "ssh-agent"),
        status=overrides.pop("status", AgentStatus.DRAFT),
        **overrides,
    )
    db.add(agent)
    db.flush()
    return agent


def _new_task(db, user_id: int, **overrides) -> Task:
    task = Task(
        user_id=user_id,
        title="SSH task",
        description="SSH task",
        status=TaskStatus.PENDING,
        **overrides,
    )
    db.add(task)
    db.commit()
    db.refresh(task)
    return task


@pytest.mark.parametrize("status", [AgentStatus.DRAFT, AgentStatus.PUBLISHED])
def test_agent_id_for_task_authorizes_owned_direct_agent(task_db, status) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        agent = _new_agent(db, int(owner.id), status=status)
        task = _new_task(db, int(owner.id), agent_id=int(agent.id))
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) == int(agent.id)


@pytest.mark.parametrize("status", [AgentStatus.DRAFT, AgentStatus.PUBLISHED])
def test_agent_id_for_task_authorizes_owned_preview_agent(task_db, status) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        agent = _new_agent(db, int(owner.id), status=status)
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": int(agent.id)},
        )
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) == int(agent.id)


def test_agent_id_for_task_rejects_cross_owner_published_direct_agent(task_db) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        other = _new_user(db, "other")
        published = _new_agent(db, int(other.id), status=AgentStatus.PUBLISHED)
        task = _new_task(db, int(owner.id), agent_id=int(published.id))
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) is None


@pytest.mark.parametrize("candidate", ["01", "abc", True, 2**31, str(2**31)])
def test_agent_id_for_task_rejects_malformed_preview_candidate(
    task_db, candidate
) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": candidate},
        )
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) is None


def test_agent_id_for_task_rejects_cross_scope_preview_agent(task_db) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        other = _new_user(db, "other")
        foreign = _new_agent(db, int(other.id))
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": int(foreign.id)},
        )
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) is None


def test_agent_id_for_task_rejects_archived_agent(task_db) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        archived = _new_agent(db, int(owner.id), status=AgentStatus.ARCHIVED)
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": int(archived.id)},
        )
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) is None


def test_agent_id_for_task_does_not_fallback_after_denied_primary_agent(
    task_db,
) -> None:
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        other = _new_user(db, "other")
        denied_primary = _new_agent(db, int(other.id), name="foreign")
        owned_preview = _new_agent(db, int(owner.id), name="owned")
        task = _new_task(
            db,
            int(owner.id),
            agent_id=int(denied_primary.id),
            agent_config={"preview_agent_id": int(owned_preview.id)},
        )
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(owner.id)) is None


def test_agent_id_for_task_allows_team_admin_private_and_member_team_visible(
    task_db,
) -> None:
    db = task_db()
    try:
        admin = _new_user(db, "admin")
        member = _new_user(db, "member")
        private = _new_agent(
            db, int(member.id), team_id=77, visibility="admins", name="private"
        )
        shared = _new_agent(
            db, int(admin.id), team_id=77, visibility="team", name="shared"
        )
        legacy = _new_agent(db, int(member.id), team_id=None, name="legacy")
        admin_task = _new_task(
            db, int(admin.id), agent_config={"preview_agent_id": int(private.id)}
        )
        member_task = _new_task(
            db, int(member.id), agent_config={"preview_agent_id": int(shared.id)}
        )
        legacy_task = _new_task(
            db, int(member.id), agent_config={"preview_agent_id": int(legacy.id)}
        )
    finally:
        db.close()

    set_agent_team_scope_hook(
        lambda _db, user_id: AgentTeamScope(
            team_id=77, is_team_admin=user_id == int(admin.id)
        )
    )
    try:
        assert _agent_id_for_task(task_db, int(admin_task.id), int(admin.id)) == int(
            private.id
        )
        assert _agent_id_for_task(task_db, int(member_task.id), int(member.id)) == int(
            shared.id
        )
        assert _agent_id_for_task(task_db, int(legacy_task.id), int(member.id)) == int(
            legacy.id
        )
    finally:
        set_agent_team_scope_hook(None)


def test_agent_id_for_task_rejects_a_task_owned_by_another_user(task_db) -> None:
    """The owner scope must come from the executing user, not from whatever row
    the task id happens to hit."""
    db = task_db()
    try:
        owner = _new_user(db, "owner")
        intruder = _new_user(db, "intruder")
        agent = _new_agent(db, int(owner.id))
        task = _new_task(db, int(owner.id), agent_id=int(agent.id))
    finally:
        db.close()

    assert _agent_id_for_task(task_db, int(task.id), int(intruder.id)) is None


async def test_binding_authorized_creator_emits_nothing_without_a_binding(
    task_db,
) -> None:
    """The contract behind ``BINDING_AUTHORIZED_CATEGORIES``: admission is
    unconditional at the selection gates, so the creator itself must refuse to
    emit anything for an agent with no bound target."""
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.core.tools.adapters.vibe.base import BINDING_AUTHORIZED_CATEGORIES
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    assert BINDING_AUTHORIZED_CATEGORIES == frozenset({"ssh"}), (
        "a new member needs the same no-authorization-no-tools proof"
    )

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        agent = _new_agent(db, int(owner.id))
        task = _new_task(db, int(owner.id), agent_id=int(agent.id))
    finally:
        db.close()

    provider = _RecordingTargetProvider()  # no bound targets
    set_ssh_target_provider_hook(lambda _sf: provider)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: int(owner.id),
        get_task_id=lambda: f"web_task_{int(task.id)}",
        get_workspace_config=lambda: None,
    )
    try:
        assert await ssh_tools.create_ssh_tools(config) == []
        assert provider.list_calls == 1
    finally:
        set_ssh_target_provider_hook(None)


async def test_create_ssh_tools_emits_nothing_for_a_delegated_sub_agent(
    task_db,
) -> None:
    """A delegated sub-agent's task id (``agent_{id}_{uuid8}``) carries no
    resolvable task, so it must not reach the provider at all."""
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        agent = _new_agent(db, int(owner.id))
        _new_task(db, int(owner.id), agent_id=int(agent.id))
    finally:
        db.close()

    provider = _RecordingTargetProvider()
    set_ssh_target_provider_hook(lambda _sf: provider)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: int(owner.id),
        get_task_id=lambda: f"agent_{int(agent.id)}_a1b2cd34",
        get_workspace_config=lambda: None,
    )
    try:
        assert await ssh_tools.create_ssh_tools(config) == []
        assert provider.list_calls == 0
    finally:
        set_ssh_target_provider_hook(None)


class _RecordingTargetProvider(_Provider):
    def __init__(
        self,
        *,
        resolved=None,
        targets=None,
        credential=None,
        events=None,
    ) -> None:
        super().__init__(resolved=resolved, targets=targets)
        self._credential = credential
        self._events = events
        self.list_calls = 0
        self.resolve_calls = 0
        self.list_contexts = []
        self.resolve_contexts = []

    async def resolve(self, context, target_alias):
        self.resolve_calls += 1
        self.resolve_contexts.append(context)
        if self._events is not None:
            self._events.append(("resolve", context, target_alias))
        return await super().resolve(context, target_alias)

    async def list_bound_targets(self, context):
        self.list_calls += 1
        self.list_contexts.append(context)
        if self._events is not None:
            self._events.append(("list", context))
        return await super().list_bound_targets(context)

    async def read_version(self, secret_handle):
        if self._events is not None:
            self._events.append(("read_version", secret_handle))
        if self._credential is None:
            raise NotImplementedError
        return self._credential


class _RecordingRunner:
    def __init__(self, events) -> None:
        self._events = events

    async def execute(self, **kwargs) -> SshRunResult:
        self._events.append(("run", kwargs["command"]))
        return SshRunResult(exit_code=0, stdout="ok", stderr="", truncated=False)


@pytest.mark.parametrize("status", [AgentStatus.DRAFT, AgentStatus.PUBLISHED])
async def test_create_ssh_tools_propagates_authorized_preview_context_to_execute(
    task_db,
    monkeypatch,
    status,
) -> None:
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        _new_agent(db, int(owner.id), name="id-separator")
        preview = _new_agent(db, int(owner.id), name="preview", status=status)
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": int(preview.id)},
        )
        owner_id = int(owner.id)
        preview_id = int(preview.id)
        task_id = int(task.id)
    finally:
        db.close()
    assert preview_id != owner_id

    events: list[tuple] = []
    provider = _RecordingTargetProvider(
        resolved=_resolved(),
        targets=[
            BoundTargetInfo(
                alias="prod",
                display_name="Production",
                capabilities=frozenset({"execute"}),
            )
        ],
        credential=SensitiveSshCredential(b"KEY", "", "ssh-ed25519"),
        events=events,
    )

    def _factory(session_factory):
        events.append(("factory", session_factory))
        return provider

    runner = _RecordingRunner(events)
    set_ssh_target_provider_hook(_factory)
    monkeypatch.setattr(ssh_tools, "_make_ssh_sandbox_lease", lambda *_args: None)
    monkeypatch.setattr(ssh_tools, "AsyncsshRunner", lambda: runner)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: owner_id,
        get_task_id=lambda: f"web_task_{task_id}",
        get_workspace_config=lambda: None,
    )
    try:
        tools = await ssh_tools.create_ssh_tools(config)
        assert [tool.name for tool in tools] == [
            "ssh_list_targets",
            "ssh_execute",
            "ssh_upload",
            "ssh_download",
        ]

        expected_context = SshExecutionContext(
            actor=ActorRef(actor_type="user", actor_id=str(owner_id)),
            execution_principal=PrincipalRef(
                principal_type="user", principal_id=str(owner_id)
            ),
            agent_id=preview_id,
            task_id=task_id,
            turn_id=None,
            request_id=f"web_task_{task_id}",
        )
        assert provider.list_contexts == [expected_context]
        assert events == [("factory", task_db), ("list", expected_context)]

        execute_tool = next(tool for tool in tools if tool.name == "ssh_execute")
        assert isinstance(execute_tool, SshExecuteTool)

        async def _resolve_public(_hostname, _port):
            return ["8.8.8.8"]

        execute_tool._executor._resolver = _resolve_public
        outcome = await execute_tool.run_json_async(
            {"target": "prod", "command": "uptime"}
        )

        assert outcome["ok"] is True
        assert outcome["stdout"] == "ok"
        assert provider.resolve_contexts == [expected_context]
        assert provider.resolve_contexts[0] is provider.list_contexts[0]
        assert events == [
            ("factory", task_db),
            ("list", expected_context),
            ("resolve", expected_context, "prod"),
            ("read_version", _resolved().secret_handle),
            ("run", "uptime"),
        ]
    finally:
        set_ssh_target_provider_hook(None)


async def test_create_ssh_tools_denies_cross_scope_preview_before_provider_targets(
    task_db,
) -> None:
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        other = _new_user(db, "other")
        foreign = _new_agent(db, int(other.id))
        task = _new_task(
            db,
            int(owner.id),
            agent_config={"preview_agent_id": int(foreign.id)},
        )
    finally:
        db.close()

    provider = _RecordingTargetProvider()
    factory_calls: list[object] = []

    def _factory(session_factory):
        factory_calls.append(session_factory)
        return provider

    set_ssh_target_provider_hook(_factory)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: int(owner.id),
        get_task_id=lambda: f"web_task_{int(task.id)}",
        get_workspace_config=lambda: None,
    )
    try:
        assert await ssh_tools.create_ssh_tools(config) == []
        assert factory_calls == [task_db]
        assert provider.list_calls == 0
        assert provider.resolve_calls == 0
    finally:
        set_ssh_target_provider_hook(None)


@pytest.mark.parametrize(
    ("case", "preview_candidate"),
    [
        ("cross-owner-direct", None),
        ("malformed-preview", "abc"),
        ("oversized-preview", "1" * 5000),
        ("nonexistent-preview", "2147483647"),
        ("archived-direct", None),
        ("denied-primary-no-preview-fallback", None),
    ],
)
async def test_create_ssh_tools_denies_unresolved_task_agent_before_provider_targets(
    task_db,
    case,
    preview_candidate,
) -> None:
    """Persisted task identity must be authorized before provider target access."""
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        if case == "cross-owner-direct":
            other = _new_user(db, "other")
            foreign = _new_agent(db, int(other.id), status=AgentStatus.PUBLISHED)
            task = _new_task(db, int(owner.id), agent_id=int(foreign.id))
        elif case in {"malformed-preview", "oversized-preview", "nonexistent-preview"}:
            task = _new_task(
                db,
                int(owner.id),
                agent_config={"preview_agent_id": preview_candidate},
            )
        elif case == "archived-direct":
            archived = _new_agent(db, int(owner.id), status=AgentStatus.ARCHIVED)
            task = _new_task(db, int(owner.id), agent_id=int(archived.id))
        else:
            other = _new_user(db, "other")
            denied_primary = _new_agent(db, int(other.id), name="foreign")
            owned_preview = _new_agent(db, int(owner.id), name="owned")
            task = _new_task(
                db,
                int(owner.id),
                agent_id=int(denied_primary.id),
                agent_config={"preview_agent_id": int(owned_preview.id)},
            )
    finally:
        db.close()

    provider = _RecordingTargetProvider()
    factory_calls: list[object] = []

    def _factory(session_factory):
        factory_calls.append(session_factory)
        return provider

    set_ssh_target_provider_hook(_factory)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: int(owner.id),
        get_task_id=lambda: f"web_task_{int(task.id)}",
        get_workspace_config=lambda: None,
    )
    try:
        assert await ssh_tools.create_ssh_tools(config) == []
        assert factory_calls == [task_db]
        assert provider.list_calls == 0
        assert provider.resolve_calls == 0
    finally:
        set_ssh_target_provider_hook(None)


async def test_create_ssh_tools_propagates_task_owner_scope_failure(task_db) -> None:
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    db = task_db()
    try:
        owner = _new_user(db, "owner")
        agent = _new_agent(db, int(owner.id))
        task = _new_task(db, int(owner.id), agent_id=int(agent.id))
    finally:
        db.close()

    provider = _RecordingTargetProvider()
    set_ssh_target_provider_hook(lambda _session_factory: provider)

    def _raise_scope_failure(*_args):
        raise RuntimeError("team scope unavailable")

    set_agent_team_scope_hook(_raise_scope_failure)
    config = SimpleNamespace(
        get_session_factory=lambda: task_db,
        get_user_id=lambda: int(owner.id),
        get_task_id=lambda: f"web_task_{int(task.id)}",
        get_workspace_config=lambda: None,
    )
    try:
        with pytest.raises(RuntimeError, match="team scope unavailable"):
            await ssh_tools.create_ssh_tools(config)
        assert provider.list_calls == 0
        assert provider.resolve_calls == 0
    finally:
        set_agent_team_scope_hook(None)
        set_ssh_target_provider_hook(None)


class _FakeLease:
    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    async def __aenter__(self):
        return self._sandbox

    async def __aexit__(self, *a) -> None:
        return None


class _FakeProvider:
    def __init__(self, sandbox) -> None:
        self._sandbox = sandbox

    def lease(self, *, concurrency_safe: bool):
        return _FakeLease(self._sandbox)


class _FakeManager:
    def __init__(self, sandbox=None, *, capacity_error=False) -> None:
        self._sandbox = sandbox
        self._capacity_error = capacity_error
        self.calls: list[tuple[str, str]] = []

    async def get_or_create_lease_provider(self, lifecycle_type, lifecycle_id, **_):
        self.calls.append((lifecycle_type, lifecycle_id))
        if self._capacity_error:
            from xagent.web.sandbox_manager import SandboxCapacityError

            raise SandboxCapacityError(cap=1, in_use=1)
        return _FakeProvider(self._sandbox)


def test_ssh_sandbox_lease_none_without_manager(monkeypatch) -> None:
    import xagent.web.sandbox_manager as sm
    from xagent.core.tools.adapters.vibe.ssh_tools import _make_ssh_sandbox_lease

    monkeypatch.setattr(sm, "get_sandbox_manager", lambda: None)
    assert _make_ssh_sandbox_lease(30, 1) is None


async def test_ssh_sandbox_lease_leases_dedicated_ssh_sandbox(monkeypatch) -> None:
    import xagent.web.sandbox_manager as sm
    from xagent.core.tools.adapters.vibe.ssh_tools import _make_ssh_sandbox_lease

    sentinel = object()
    manager = _FakeManager(sentinel)
    monkeypatch.setattr(sm, "get_sandbox_manager", lambda: manager)
    factory = _make_ssh_sandbox_lease(30, 1)
    assert factory is not None
    async with factory() as sandbox:
        assert sandbox is sentinel
    # Leased under a task-scoped ssh lifecycle, distinct from the agent sandbox.
    assert manager.calls == [("ssh", "30")]


async def test_ssh_sandbox_lease_task_none_falls_back_to_agent(monkeypatch) -> None:
    import xagent.web.sandbox_manager as sm
    from xagent.core.tools.adapters.vibe.ssh_tools import _make_ssh_sandbox_lease

    manager = _FakeManager(object())
    monkeypatch.setattr(sm, "get_sandbox_manager", lambda: manager)
    async with _make_ssh_sandbox_lease(None, 7)():
        pass
    assert manager.calls == [("ssh", "agent-7")]


async def test_ssh_sandbox_lease_capacity_fails_closed(monkeypatch) -> None:
    import xagent.web.sandbox_manager as sm
    from xagent.core.tools.adapters.vibe.ssh_tools import _make_ssh_sandbox_lease

    monkeypatch.setattr(
        sm, "get_sandbox_manager", lambda: _FakeManager(capacity_error=True)
    )
    factory = _make_ssh_sandbox_lease(30, 1)
    with pytest.raises(SshError) as exc:
        async with factory():
            pass
    assert exc.value.code == SshErrorCode.SANDBOX_UNAVAILABLE


async def test_create_ssh_tools_skips_on_boxlite_backend(monkeypatch) -> None:
    # Boxlite buffers command output unbounded, so ssh_execute there is a
    # host-memory DoS; SSH tools must not be emitted under that backend (M2).
    import xagent.web.sandbox_manager as sm
    from xagent.core.tools.adapters.vibe import ssh_tools
    from xagent.web.services.ssh_runtime import set_ssh_target_provider_hook

    provider = _Provider(targets=[SimpleNamespace()])  # one bound target
    set_ssh_target_provider_hook(lambda _sf: provider)
    monkeypatch.setattr(ssh_tools, "_agent_id_for_task", lambda _sf, _tid, _uid: 1)
    monkeypatch.setattr(sm, "get_sandbox_manager", lambda: _FakeManager(object()))
    monkeypatch.setenv("SANDBOX_IMPLEMENTATION", "boxlite")
    config = SimpleNamespace(
        get_session_factory=lambda: object(),
        get_user_id=lambda: 42,
        get_task_id=lambda: "web_task_30",
        get_workspace_config=lambda: None,
    )
    try:
        assert await ssh_tools.create_ssh_tools(config) == []
    finally:
        set_ssh_target_provider_hook(None)


def test_ssh_creator_registered_under_ssh_category() -> None:
    # Managed SSH tools get their own assignable "ssh" category so the agent
    # editor can auto-enable it when a target is bound (mirrors connectors).
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    ToolRegistry._import_tool_modules()
    entry = next(
        e for e in ToolRegistry._tool_creators if e[0].__name__ == "create_ssh_tools"
    )
    assert entry[1] == frozenset({"ssh"})


def test_ssh_tools_carry_ssh_category() -> None:
    # Category must be SSH (not OTHER) so compute_allowed_names admits them
    # when the "ssh" category is selected.
    assert (
        SshExecuteTool(executor=_Executor(), context=_ctx()).metadata.category.value
        == "ssh"
    )
    assert (
        SshListTargetsTool(provider=_Provider(), context=_ctx()).metadata.category.value
        == "ssh"
    )
