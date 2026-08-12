import asyncio
import functools
import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from types import MappingProxyType, SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy import inspect as sqlalchemy_inspect
from sqlalchemy import select
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.state import InstanceState
from sqlalchemy.pool import QueuePool

from xagent.core.execution_scope import ExecutionScope
from xagent.core.task_runtime import TaskRuntimeContext
from xagent.core.tools.adapters.vibe.config import (
    MCPConfigLoadError,
    ToolFactoryRuntimeSessionBoundaryError,
)
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.mcp import MCPServer
from xagent.web.models.tool_config import ToolConfig
from xagent.web.models.user import User
from xagent.web.services.tool_credentials import (
    set_user_tool_allowlist_hook,
    set_user_tool_overrides_hook,
)
from xagent.web.tools.config import WebToolConfig


def _factory():
    engine = create_engine("sqlite://")  # in-memory, fresh
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


_FACTORY_MODEL_VALUE_FIELDS = (
    "vision_model",
    "image_generate_model",
    "image_edit_model",
    "video_model",
    "asr_model",
    "tts_model",
    "sound_effect_model",
    "music_model",
)
_FACTORY_MODEL_MAPPING_FIELDS = (
    "image_models",
    "video_models",
    "asr_models",
    "tts_models",
    "sound_effect_models",
    "music_models",
)


def _factory_model_snapshot(generation):
    from xagent.web.tools.config import (
        _ToolFactoryRuntimeLoadPlan,
        _ToolFactoryRuntimeSnapshot,
    )

    values = {field_name: object() for field_name in _FACTORY_MODEL_VALUE_FIELDS}
    for field_name in _FACTORY_MODEL_MAPPING_FIELDS:
        values[field_name] = {
            "shared": object(),
            f"only-{generation}": object(),
        }
    plan = _ToolFactoryRuntimeLoadPlan(
        user_id=None,
        task_id=None,
        connector_runtime_turn_id=None,
        load_policy=False,
        load_basic=False,
        load_sql=False,
        load_custom_api=False,
        load_vision=True,
        load_image=True,
        load_video=True,
        load_audio=True,
        published_agent_policy=None,
    )
    return _ToolFactoryRuntimeSnapshot(plan=plan, **values), values


def _assert_factory_model_values(cfg, expected):
    for field_name in _FACTORY_MODEL_VALUE_FIELDS:
        getter = getattr(cfg, f"get_{field_name}")
        assert getter() is expected[field_name]
    for field_name in _FACTORY_MODEL_MAPPING_FIELDS:
        getter = getattr(cfg, f"get_{field_name}")
        actual = getter()
        expected_mapping = expected[field_name]
        assert actual.keys() == expected_mapping.keys()
        assert all(actual[key] is value for key, value in expected_mapping.items())


def _assert_factory_model_getters_are_neutral(cfg):
    for field_name in _FACTORY_MODEL_VALUE_FIELDS:
        assert getattr(cfg, f"get_{field_name}")() is None
    for field_name in _FACTORY_MODEL_MAPPING_FIELDS:
        assert getattr(cfg, f"get_{field_name}")() == {}


_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _graph_reaches_identity(root, target):
    visited = set()

    def _visit(value):
        if value is target:
            return True
        if id(value) in visited:
            return False
        if is_dataclass(value) and not isinstance(value, type):
            children = (getattr(value, item.name) for item in fields(value))
        elif type(value) in {dict, _MAPPING_PROXY_TYPE}:
            children = (child for key, item in value.items() for child in (key, item))
        elif type(value) in {list, tuple, set, frozenset}:
            children = iter(value)
        else:
            return False
        visited.add(id(value))
        return any(_visit(child) for child in children)

    return _visit(root)


def _assert_identities_not_reachable(root, forbidden_by_name):
    for name, forbidden in forbidden_by_name.items():
        assert not _graph_reaches_identity(root, forbidden), (
            f"retained model state reaches forbidden {name}"
        )


class _Chain:
    """Minimal chainable query stub: filter/join return self, terminals empty."""

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _NonEmptyMappingWithHostileBool(Mapping[str, object]):
    def __init__(self, value):
        self._value = value

    def __bool__(self):
        raise AssertionError("mapping truthiness must not be consulted")

    def __getitem__(self, key):
        if key != "model":
            raise KeyError(key)
        return self._value

    def __iter__(self):
        return iter(("model",))

    def __len__(self):
        return 1


class _ListChain:
    """Minimal chainable query stub with a fixed ``all()`` result."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def all(self):
        return list(self._rows)

    def first(self):
        return self._rows[0] if self._rows else None


class _StaticRowsSession:
    def __init__(self, rows):
        self._rows = list(rows)

    def query(self, *a, **k):
        return _ListChain(self._rows)


class _TrackingSession:
    """Records whether ``.query`` was driven (i.e. the session was used)."""

    def __init__(self):
        self.query_calls = 0
        self.closed = False

    def query(self, *a, **k):
        self.query_calls += 1
        return _Chain()

    def close(self):
        self.closed = True

    def connection(self):
        return object()

    def rollback(self):
        return None


class _FailingQuerySession:
    def __init__(self):
        self.query_calls = 0

    def query(self, *args, **kwargs):
        self.query_calls += 1
        raise RuntimeError("database-secret")


class _PostgresAbortSession:
    """Models Postgres' abort-until-transaction-reset behavior."""

    def __init__(self, checkout_error: Exception | None = None):
        self.aborted = False
        self.closed = False
        self.checkout_error = checkout_error

    def connection(self):
        if self.checkout_error is not None:
            raise self.checkout_error
        return object()

    def query(self, *_args, **_kwargs):
        self.assert_usable()
        return _ListChain([SimpleNamespace(id=1)])

    def close(self):
        self.closed = True

    def swallow_statement_failure(self) -> None:
        self.aborted = True

    def assert_usable(self) -> None:
        if self.aborted:
            raise RuntimeError("current transaction is aborted")


def test_checked_out_session_runner_owns_checkout_and_close():
    from xagent.web.tools.config import _run_with_checked_out_session

    events: list[str] = []

    class Session:
        def connection(self):
            events.append("checkout")
            return object()

        def close(self):
            events.append("close")

    session = Session()

    def operation(db):
        assert db is session
        events.append("operation")
        return "loaded"

    result = _run_with_checked_out_session(lambda: session, operation)

    assert result == "loaded"
    assert events == ["checkout", "operation", "close"]


def test_checked_out_session_runner_closes_after_operation_failure():
    from xagent.web.tools.config import _run_with_checked_out_session

    class Session:
        closed = False

        def connection(self):
            return object()

        def close(self):
            self.closed = True

    session = Session()

    def fail_operation(_db):
        raise RuntimeError("loader failed")

    with pytest.raises(RuntimeError, match="loader failed"):
        _run_with_checked_out_session(lambda: session, fail_operation)

    assert session.closed


def test_checked_out_session_runner_propagates_close_failure():
    class Session:
        def connection(self):
            return object()

        def close(self):
            raise RuntimeError("session close failed")

    from xagent.web.tools.config import _run_with_checked_out_session

    with pytest.raises(RuntimeError, match="session close failed"):
        _run_with_checked_out_session(lambda: Session(), lambda _db: "loaded")


def test_get_session_factory_prefers_injected_factory():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    assert cfg.get_session_factory() is factory


def test_factory_built_get_db_is_lazy_and_closed_by_close():
    factory = _factory()
    cfg = WebToolConfig(db=None, request=None, db_factory=factory)
    db1 = cfg.get_db()
    db2 = cfg.get_db()
    assert db1 is db2  # cached, single construction-time session
    cfg.close()
    # closing twice is safe
    cfg.close()


def test_live_db_path_unchanged():
    sentinel = object()
    cfg = WebToolConfig(db=sentinel, request=None)
    assert cfg.get_db() is sentinel
    cfg.close()  # must not raise; caller owns the request session


@pytest.mark.asyncio
async def test_create_default_tools_uses_worker_session_factory_without_live_db(
    monkeypatch,
):
    """The chat bootstrap delegates all runtime preparation to ToolFactory."""
    from xagent.web.api.chat import create_default_tools

    session_factory = object()
    captured: dict[str, object] = {}

    class _FakeToolConfig:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def set_task_runtime_contribution(self, contribution) -> None:
            self.task_runtime_contribution = contribution

        async def refresh_runtime_policy(self) -> None:
            raise AssertionError("create_default_tools must not pre-refresh policy")

    async def create_tools(config):
        return ["prepared-tool"]

    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: session_factory,
    )
    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)

    tools, config = await create_default_tools(
        None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
    )

    assert tools == ["prepared-tool"]
    assert captured["db"] is None
    assert captured["db_factory"] is session_factory


@pytest.mark.asyncio
async def test_create_default_tools_skips_runtime_workspace_without_providers(
    monkeypatch,
):
    import xagent.web.api.chat as chat_module
    from xagent.web.api.chat import create_default_tools

    class _FakeToolConfig:
        def __init__(self, **kwargs):
            self.runtime_contribution = None

        def set_task_runtime_contribution(self, contribution) -> None:
            self.runtime_contribution = contribution

    async def create_tools(config):
        return []

    def unexpected_workspace(_config):
        raise AssertionError("workspace must not be created without providers")

    async def unexpected_runtime(_context):
        raise AssertionError("runtime build must not run without providers")

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: object(),
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)
    monkeypatch.setattr(ToolFactory, "create_workspace", unexpected_workspace)
    monkeypatch.setattr(chat_module, "build_task_runtime", unexpected_runtime)
    monkeypatch.setattr(chat_module, "registered_task_extensions", lambda: ())

    tools, config = await create_default_tools(
        None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
        task_runtime_context=TaskRuntimeContext(
            task_id=11,
            user_id=7,
            source="internal",
            session_factory=lambda: object(),
        ),
    )

    assert tools == []
    assert config.runtime_contribution.tools == ()


@pytest.mark.asyncio
async def test_create_default_tools_degrades_when_runtime_provider_build_fails(
    monkeypatch,
    caplog,
):
    import xagent.web.api.chat as chat_module
    from xagent.web.api.chat import create_default_tools
    from xagent.web.services.task_runtime import TaskRuntimeExtensionError

    class _FakeToolConfig:
        def __init__(self, **kwargs):
            self.runtime_contribution = None
            self.runtime_workspace = None

        def get_workspace_config(self):
            return {"task_id": "web_task_11"}

        def set_task_runtime_contribution(self, contribution) -> None:
            self.runtime_contribution = contribution

        # Both runtime setters are concrete no-ops on ``BaseToolConfig``, so
        # every real config has them; the double has to as well.
        def set_task_runtime_workspace(self, workspace) -> None:
            self.runtime_workspace = workspace

    async def create_tools(config):
        return ["core-tool"]

    async def fail_runtime(_context):
        raise TaskRuntimeExtensionError(
            "broken_runtime",
            "build_runtime",
            RuntimeError("provider unavailable"),
        )

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: object(),
    )
    monkeypatch.setattr(ToolFactory, "create_all_tools", create_tools)
    monkeypatch.setattr(
        ToolFactory,
        "create_workspace",
        lambda _config: SimpleNamespace(id="workspace"),
    )
    monkeypatch.setattr(chat_module, "build_task_runtime", fail_runtime)
    monkeypatch.setattr(
        chat_module,
        "registered_task_extensions",
        lambda: ("broken_runtime",),
    )

    with caplog.at_level("ERROR"):
        tools, config = await create_default_tools(
            None,
            user=SimpleNamespace(id=7, is_admin=False),
            task_id="web_task_11",
            task_runtime_context=TaskRuntimeContext(
                task_id=11,
                user_id=7,
                source="internal",
                session_factory=lambda: object(),
            ),
        )

    assert tools == ["core-tool"]
    assert config.runtime_contribution.tools == ()
    assert "broken_runtime" in caplog.text


@pytest.mark.asyncio
async def test_create_default_tools_isolates_runtime_tool_name_collision(
    monkeypatch,
    caplog,
):
    import xagent.web.api.chat as chat_module
    from xagent.core.task_runtime import (
        TaskRuntimeContribution,
        merge_task_runtime_contributions,
    )
    from xagent.core.tools.adapters.vibe.config import (
        ToolConfig as StandaloneToolConfig,
    )
    from xagent.web.api.chat import create_default_tools

    core_tool = SimpleNamespace(
        name="computer",
        metadata=SimpleNamespace(category="other"),
    )
    runtime_tool = SimpleNamespace(
        name="computer",
        metadata=SimpleNamespace(category="other"),
    )

    class _FakeToolConfig(StandaloneToolConfig):
        def __init__(self, **kwargs):
            super().__init__({})
            self.runtime_contribution = TaskRuntimeContribution()
            self.runtime_workspace = None

        def get_workspace_config(self):
            return {"task_id": "web_task_11"}

        def set_task_runtime_contribution(self, contribution) -> None:
            self.runtime_contribution = contribution

        def get_task_runtime_contribution(self):
            return self.runtime_contribution

        def set_task_runtime_workspace(self, workspace) -> None:
            self.runtime_workspace = workspace

        def get_task_runtime_workspace(self):
            return self.runtime_workspace

    async def create_registered_tools(config):
        return [core_tool]

    async def build_runtime(_context):
        return merge_task_runtime_contributions(
            {
                "desktop_runtime": TaskRuntimeContribution(
                    tools=(runtime_tool,),
                    environment="Control the desktop.",
                    preferred_input_modalities=("image",),
                )
            }
        )

    monkeypatch.setattr("xagent.web.tools.config.WebToolConfig", _FakeToolConfig)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: object(),
    )
    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        create_registered_tools,
    )
    monkeypatch.setattr(
        ToolFactory,
        "create_workspace",
        lambda _config: SimpleNamespace(id="workspace"),
    )
    monkeypatch.setattr(chat_module, "build_task_runtime", build_runtime)
    monkeypatch.setattr(
        chat_module,
        "registered_task_extensions",
        lambda: ("desktop_runtime",),
    )

    with caplog.at_level("WARNING"):
        tools, config = await create_default_tools(
            None,
            user=SimpleNamespace(id=7, is_admin=False),
            task_id="web_task_11",
            task_runtime_context=TaskRuntimeContext(
                task_id=11,
                user_id=7,
                source="internal",
                session_factory=lambda: object(),
            ),
        )

    assert tools == [core_tool]
    assert config.runtime_contribution == TaskRuntimeContribution()
    assert "Dropping task runtime extension 'desktop_runtime'" in caplog.text


@pytest.mark.asyncio
async def test_create_default_tools_prefetches_excluded_agent_policy_once(monkeypatch):
    """The prefetched agent policy must include the excluded agent ID."""
    from xagent.web.api.chat import create_default_tools
    from xagent.web.tools.config import _ToolFactoryRuntimeSnapshot

    plans = []
    session_factory = object()

    def load_runtime_snapshot(session_factory, plan, policy_snapshot=None):
        plans.append(plan)
        return _ToolFactoryRuntimeSnapshot(plan=plan)

    async def create_tools(config, apply_user_override_filter=True):
        return []

    monkeypatch.setattr(
        "xagent.web.tools.config._load_tool_factory_runtime_snapshot",
        load_runtime_snapshot,
    )
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local",
        lambda: session_factory,
    )
    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", create_tools)

    tools, _ = await create_default_tools(
        db=None,
        user=SimpleNamespace(id=7, is_admin=False),
        task_id="web_task_11",
        excluded_agent_id=41,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=None),
    )

    assert tools == []
    assert len(plans) == 1, [
        plan.published_agent_policy.excluded_agent_ids for plan in plans
    ]
    assert 41 in plans[0].published_agent_policy.excluded_agent_ids


def _saturated_tool_config(
    tmp_path, *, pool_timeout: float
) -> tuple[object, object, WebToolConfig]:
    engine = create_engine(
        f"sqlite:///{tmp_path / 'tool-factory.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=pool_timeout,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    held_connection = engine.connect()
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )
    return engine, held_connection, cfg


@pytest.mark.asyncio
async def test_tool_factory_credential_prefetch_waits_off_event_loop(tmp_path):
    """Credential checkout must not freeze unrelated async work."""
    engine, held_connection, cfg = _saturated_tool_config(tmp_path, pool_timeout=0.5)
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    async def build_tools() -> list:
        return await ToolFactory.create_all_tools(cfg)

    ticker_task = asyncio.create_task(ticker())
    build_task = asyncio.create_task(build_tools())
    try:
        await asyncio.sleep(0.08)
        assert ticks >= 4
        assert not build_task.done()

        held_connection.close()
        await build_task
    finally:
        if not held_connection.closed:
            held_connection.close()
        if not build_task.done():
            build_task.cancel()
            await asyncio.gather(build_task, return_exceptions=True)
        stop.set()
        await ticker_task
        cfg.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_tool_factory_prefetch_propagates_pool_timeout(tmp_path):
    """A build-time checkout timeout must stop the build and reach its owner."""
    engine, held_connection, cfg = _saturated_tool_config(tmp_path, pool_timeout=0.05)
    try:
        with pytest.raises(SQLAlchemyTimeoutError):
            await ToolFactory.create_all_tools(cfg)
    finally:
        held_connection.close()
        cfg.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_tool_factory_releases_live_read_session_before_worker_checkout(
    monkeypatch,
    tmp_path,
):
    """A request SELECT must not starve the worker on a one-slot pool."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'live-request-tool-factory.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.05,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    cfg = WebToolConfig(
        db=live_db,
        request=None,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )

    async def create_tools(config, apply_user_override_filter=True):
        return []

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", create_tools)

    try:
        assert live_db.query(ToolConfig).all() == []
        assert engine.pool.checkedout() == 1

        assert await ToolFactory.create_all_tools(cfg) == []
        assert engine.pool.checkedout() == 0
        assert cfg._live_db is None

        assert live_db.query(ToolConfig).all() == []
        assert engine.pool.checkedout() == 1
        live_db.rollback()
        assert engine.pool.checkedout() == 0
    finally:
        live_db.close()
        cfg.close()
        engine.dispose()


@pytest.mark.parametrize("factory_owned", [False, True])
def test_verified_factory_handoff_detaches_clean_sessions(factory_owned, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'verified-handoff.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    request = SimpleNamespace(user=object())
    cfg = WebToolConfig(
        db=None if factory_owned else factory(),
        db_factory=factory if factory_owned else None,
        request=request,
        user=request.user,
        user_id=1,
    )
    try:
        cfg.db.query(ToolConfig).all()
        assert engine.pool.checkedout() == 1

        cfg.handoff_factory_runtime()

        assert engine.pool.checkedout() == 0
        assert cfg._live_db is None
        assert cfg._lazy_db is None
        assert cfg.request is None
        assert cfg._user is None
    finally:
        cfg.close()
        engine.dispose()


def test_verified_factory_handoff_preserves_pending_caller_state(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'pending-handoff.db'}")
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    request = SimpleNamespace(user=object())
    cfg = WebToolConfig(
        db=live_db,
        request=request,
        user=request.user,
        user_id=1,
    )
    try:
        live_db.add(
            User(
                username="pending-handoff-user",
                password_hash="hash",
                is_admin=False,
            )
        )

        with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
            cfg.handoff_factory_runtime()

        assert cfg._live_db is live_db
        assert cfg.request is request
        assert cfg._user is request.user
        assert list(live_db.new)
    finally:
        live_db.rollback()
        live_db.close()
        cfg.close()
        engine.dispose()


@pytest.mark.parametrize("rollback_fails", [False, True])
def test_verified_factory_handoff_terminally_closes_failed_lazy_session(
    rollback_fails,
):
    class FailingLazySession:
        new = (object(),)
        dirty = ()
        deleted = ()
        info = {}

        def __init__(self, rollback_fails: bool) -> None:
            self.rollback_calls = 0
            self.close_calls = 0
            self.invalidate_calls = 0
            self.rollback_fails = rollback_fails

        def in_transaction(self) -> bool:
            return True

        def rollback(self) -> None:
            self.rollback_calls += 1
            if self.rollback_fails:
                raise RuntimeError("rollback failed")

        def close(self) -> None:
            self.close_calls += 1
            raise RuntimeError("close failed")

        def invalidate(self) -> None:
            self.invalidate_calls += 1

    lazy_db = FailingLazySession(rollback_fails)
    cfg = WebToolConfig(db=None, db_factory=lambda: lazy_db, request=None, user_id=1)
    assert cfg.db is lazy_db

    with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
        cfg.handoff_factory_runtime()

    assert lazy_db.rollback_calls == 1
    assert lazy_db.close_calls == 1
    assert lazy_db.invalidate_calls == 1
    assert cfg._lazy_db is None


def test_dual_lazy_cleanup_failure_retains_retry_ownership():
    class RetryableLazySession:
        new = (object(),)
        dirty = ()
        deleted = ()
        info = {}

        def __init__(self) -> None:
            self.close_failures_remaining = 1
            self.invalidate_failures_remaining = 1

        def in_transaction(self) -> bool:
            return True

        def rollback(self) -> None:
            return None

        def close(self) -> None:
            if self.close_failures_remaining:
                self.close_failures_remaining -= 1
                raise RuntimeError("close failed")

        def invalidate(self) -> None:
            if self.invalidate_failures_remaining:
                self.invalidate_failures_remaining -= 1
                raise RuntimeError("invalidate failed")

    lazy_db = RetryableLazySession()
    cfg = WebToolConfig(db=None, db_factory=lambda: lazy_db, request=None, user_id=1)
    assert cfg.db is lazy_db

    with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
        cfg.handoff_factory_runtime()

    assert cfg._lazy_db is lazy_db

    cfg.close()

    assert cfg._lazy_db is None


def test_dual_lazy_cleanup_failure_retains_real_pool_owner_until_retry(
    monkeypatch, tmp_path
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'dual-lazy-cleanup.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    ToolConfig.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    lazy_db = factory()
    lazy_db.query(ToolConfig).all()
    assert engine.pool.checkedout() == 1
    real_close = lazy_db.close
    real_invalidate = lazy_db.invalidate

    def fail_close():
        raise RuntimeError("close failed")

    def fail_invalidate():
        raise RuntimeError("invalidate failed")

    monkeypatch.setattr(
        "xagent.web.models.database.release_db_connection_if_clean",
        lambda _db: False,
    )
    monkeypatch.setattr(lazy_db, "close", fail_close)
    monkeypatch.setattr(lazy_db, "invalidate", fail_invalidate)
    cfg = WebToolConfig(db=None, db_factory=lambda: lazy_db, request=None, user_id=1)
    cfg._lazy_db = lazy_db
    try:
        with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
            cfg.handoff_factory_runtime()

        assert cfg._lazy_db is lazy_db
        lazy_db.connection()
        assert engine.pool.checkedout() == 1

        monkeypatch.setattr(lazy_db, "close", real_close)
        monkeypatch.setattr(lazy_db, "invalidate", real_invalidate)
        cfg.close()

        assert cfg._lazy_db is None
        assert engine.pool.checkedout() == 0
    finally:
        lazy_db.close()
        engine.dispose()


def test_verified_factory_handoff_preserves_live_state_while_cleaning_lazy_failure():
    class PendingSession:
        new = (object(),)
        dirty = ()
        deleted = ()
        info = {}

        def __init__(self, *, close_fails: bool = False) -> None:
            self.rollback_calls = 0
            self.close_calls = 0
            self.invalidate_calls = 0
            self.close_fails = close_fails

        def in_transaction(self) -> bool:
            return True

        def rollback(self) -> None:
            self.rollback_calls += 1

        def close(self) -> None:
            self.close_calls += 1
            if self.close_fails:
                raise RuntimeError("close failed")

        def invalidate(self) -> None:
            self.invalidate_calls += 1

    live_db = PendingSession()
    lazy_db = PendingSession(close_fails=True)
    request = SimpleNamespace(user=object())
    cfg = WebToolConfig(
        db=live_db,
        db_factory=lambda: lazy_db,
        request=request,
        user=request.user,
        user_id=1,
    )
    cfg._lazy_db = lazy_db

    with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
        cfg.handoff_factory_runtime()

    assert live_db.rollback_calls == 0
    assert live_db.close_calls == 0
    assert cfg._live_db is live_db
    assert cfg.request is request
    assert cfg._user is request.user
    assert lazy_db.rollback_calls == 1
    assert lazy_db.close_calls == 1
    assert lazy_db.invalidate_calls == 1
    assert cfg._lazy_db is None


def _contains_orm_instance(value, seen=None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return False
    seen.add(value_id)

    if isinstance(sqlalchemy_inspect(value, raiseerr=False), InstanceState):
        return True
    if isinstance(value, dict):
        return any(
            _contains_orm_instance(item, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_orm_instance(item, seen) for item in value)
    if isinstance(value, functools.partial):
        return (
            _contains_orm_instance(value.func, seen)
            or _contains_orm_instance(value.args, seen)
            or _contains_orm_instance(value.keywords or {}, seen)
        )
    closure = getattr(value, "__closure__", None)
    if closure:
        return any(_contains_orm_instance(cell.cell_contents, seen) for cell in closure)
    return False


def test_orm_capture_walker_detects_partial_function_closure():
    server = MCPServer(
        name="walker-negative-control",
        managed="external",
        transport="streamable_http",
    )

    def captures_mapped_server():
        return server

    assert _contains_orm_instance(functools.partial(captures_mapped_server))


def test_delegated_mcp_refresh_callback_detaches_real_orm_and_closes_its_session(
    monkeypatch, tmp_path
):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'delegated-refresh.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        connect_args={"check_same_thread": False},
    )
    MCPServer.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as seed_db:
        seed_db.add(
            MCPServer(
                name="detached-refresh-server",
                managed="external",
                transport="streamable_http",
                url="https://mcp.example.test",
            )
        )
        seed_db.commit()
    construction_db = factory()
    server = construction_db.scalar(
        select(MCPServer).where(MCPServer.name == "detached-refresh-server")
    )
    assert server is not None
    server_id = int(server.id)
    assert engine.pool.checkedout() == 1
    cfg = WebToolConfig(
        db=construction_db,
        db_factory=factory,
        request=None,
        user_id=7,
        task_id="42",
        connector_runtime_turn_id="turn-1",
    )
    bindings = [
        {
            "source": {"input_type": "secrets", "key": "authorization"},
            "target": {"target_type": "transport_headers", "key": "Authorization"},
        }
    ]
    operation_sessions = []

    def load_runtime_view(*, db, **_kwargs):
        operation_sessions.append(db)
        assert db is not construction_db
        assert db.scalar(select(MCPServer.id)) is not None
        return {f"mcp:{server_id}": {"secrets": {"authorization": "fresh"}}}

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        load_runtime_view,
    )
    try:
        refresh = cfg._build_delegated_mcp_refresh_callback(
            server=server,
            runtime_bindings=bindings,
            allow_delegated_authorization=True,
        )
        assert isinstance(refresh, functools.partial)
        assert not _contains_orm_instance(refresh)

        construction_db.expire(server)
        construction_db.expunge(server)
        construction_db.close()
        refreshed = refresh()

        assert refreshed["headers"]["Authorization"] == "fresh"
        assert len(operation_sessions) == 1
        assert engine.pool.checkedout() == 0
    finally:
        cfg.close()
        construction_db.close()
        engine.dispose()


@pytest.mark.asyncio
async def test_direct_tool_factory_build_loads_policy_hook_once(monkeypatch, tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'direct-tool-policy.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as db:
        user = User(username="direct-policy-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        user_id = int(user.id)

    hook_calls = 0

    def load_overrides(_db, _user):
        nonlocal hook_calls
        hook_calls += 1
        return {"calculator": {"enabled": False}}

    set_user_tool_overrides_hook(load_overrides)
    set_user_tool_allowlist_hook(lambda _db, _user: None)

    async def create_tools(config, apply_user_override_filter=True):
        return []

    monkeypatch.setattr(ToolFactory, "_create_all_tools_prepared", create_tools)
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=user_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=[]),
    )

    try:
        assert await ToolFactory.create_all_tools(cfg) == []
        assert hook_calls == 1
    finally:
        cfg.close()
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)
        engine.dispose()


@pytest.mark.asyncio
async def test_factory_runtime_snapshot_is_rebuilt_for_each_build(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    await ToolFactory.create_all_tools(cfg)
    await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 2
    assert all(session.closed for session in sessions)
    assert cfg._factory_runtime_snapshot is None


@pytest.mark.asyncio
async def test_handoff_retains_loaded_model_values_without_database_fallback():
    from xagent.web.tools.config import (
        _ToolFactoryRuntimeLoadPlan,
        _ToolFactoryRuntimeSnapshot,
    )

    image_adapter = object()
    video_adapter = object()
    tts_adapter = object()
    music_adapter = object()
    credential_map = {("provider", "api_key"): "secret"}
    sql_connections = {"database": "postgresql://secret"}
    custom_api_configs = [{"id": object()}]
    published_agent_records = [object()]
    plan = _ToolFactoryRuntimeLoadPlan(
        user_id=1,
        task_id=None,
        connector_runtime_turn_id=None,
        load_policy=False,
        load_basic=False,
        load_sql=False,
        load_custom_api=False,
        load_vision=True,
        load_image=True,
        load_video=False,
        load_audio=True,
        published_agent_policy=None,
    )
    snapshot = _ToolFactoryRuntimeSnapshot(
        plan=plan,
        tool_credentials=credential_map,
        sql_connections=sql_connections,
        custom_api_configs=custom_api_configs,
        vision_model=None,
        image_models={"image": image_adapter},
        image_generate_model=None,
        image_edit_model=image_adapter,
        video_models={"video": video_adapter},
        video_model=video_adapter,
        asr_models={},
        asr_model=None,
        tts_models={"tts": tts_adapter},
        tts_model=tts_adapter,
        sound_effect_models={},
        sound_effect_model=None,
        music_models={"music": music_adapter},
        music_model=music_adapter,
        published_agent_records=published_agent_records,
    )
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=1,
    )
    cfg._factory_runtime_snapshot = snapshot
    cfg.handoff_factory_runtime()

    retained = cfg._retained_factory_model_state
    assert is_dataclass(retained)
    assert retained.__dataclass_params__.frozen
    assert {item.name for item in fields(retained)} == {
        "load_vision",
        "load_image",
        "load_video",
        "load_audio",
        "vision_model",
        "image_models",
        "image_generate_model",
        "image_edit_model",
        "video_models",
        "video_model",
        "asr_models",
        "asr_model",
        "tts_models",
        "tts_model",
        "sound_effect_models",
        "sound_effect_model",
        "music_models",
        "music_model",
    }
    mapping_proxy_type = type(MappingProxyType({}))
    for name in (
        "image_models",
        "video_models",
        "asr_models",
        "tts_models",
        "sound_effect_models",
        "music_models",
    ):
        assert isinstance(getattr(retained, name), mapping_proxy_type)
    forbidden_state = {
        "snapshot": snapshot,
        "plan": plan,
        "credential map": credential_map,
        "SQL connections": sql_connections,
        "Custom API configs": custom_api_configs,
        "published-agent records": published_agent_records,
    }
    _assert_identities_not_reachable(vars(retained), forbidden_state)

    positive_control = object()
    positive_control_graph = {
        "nested": [
            (
                MappingProxyType(
                    {"set": {frozenset({positive_control})}},
                ),
            ),
        ],
    }
    assert _graph_reaches_identity(positive_control_graph, positive_control)
    with pytest.raises(AssertionError, match="positive control"):
        _assert_identities_not_reachable(
            positive_control_graph,
            {"positive control": positive_control},
        )
    assert not _graph_reaches_identity(
        SimpleNamespace(hidden=positive_control),
        positive_control,
    )

    mutated_retained = replace(retained)
    object.__setattr__(
        mutated_retained,
        "_hidden_construction_state",
        {"snapshot": snapshot},
    )
    with pytest.raises(AssertionError, match="snapshot"):
        _assert_identities_not_reachable(
            vars(mutated_retained),
            {"snapshot": snapshot},
        )
    _assert_identities_not_reachable(vars(retained), forbidden_state)

    assert cfg.get_vision_model() is None
    assert cfg.get_image_generate_model() is None
    assert cfg.get_image_edit_model() is image_adapter
    assert cfg.get_video_model() is None
    assert cfg.get_asr_model() is None
    assert cfg.get_tts_model() is tts_adapter
    assert cfg.get_sound_effect_model() is None
    assert cfg.get_music_model() is music_adapter

    mapping_getters = (
        (cfg.get_image_models, {"image": image_adapter}),
        (cfg.get_video_models, {}),
        (cfg.get_asr_models, {}),
        (cfg.get_tts_models, {"tts": tts_adapter}),
        (cfg.get_sound_effect_models, {}),
        (cfg.get_music_models, {"music": music_adapter}),
    )
    for getter, expected in mapping_getters:
        returned = getter()
        assert returned == expected
        returned["mutation"] = object()
        assert getter() == expected
        assert getter() is not returned

    cfg.close()

    assert cfg._retained_factory_model_state is None
    assert cfg.get_vision_model() is None
    assert cfg.get_image_generate_model() is None
    assert cfg.get_image_edit_model() is None
    assert cfg.get_video_model() is None
    assert cfg.get_asr_model() is None
    assert cfg.get_tts_model() is None
    assert cfg.get_sound_effect_model() is None
    assert cfg.get_music_model() is None
    for getter, _expected in mapping_getters:
        assert getter() == {}


def test_handoff_replaces_retained_model_generation_without_merging():
    snapshot_a, values_a = _factory_model_snapshot("a")
    snapshot_b, values_b = _factory_model_snapshot("b")
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=None,
    )

    cfg._apply_factory_runtime_snapshot(snapshot_a)
    cfg.handoff_factory_runtime()
    retained_a = cfg._retained_factory_model_state
    assert retained_a is not None
    _assert_factory_model_values(cfg, values_a)

    cfg._apply_factory_runtime_snapshot(snapshot_b)
    cfg.handoff_factory_runtime()
    retained_b = cfg._retained_factory_model_state
    assert retained_b is not None
    assert retained_b is not retained_a
    _assert_factory_model_values(cfg, values_b)


@pytest.mark.asyncio
async def test_policy_refresh_and_construction_discard_preserve_retained_generation():
    snapshot_a, _values_a = _factory_model_snapshot("a")
    snapshot_b, values_b = _factory_model_snapshot("b")
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=None,
    )
    cfg._apply_factory_runtime_snapshot(snapshot_b)
    cfg.handoff_factory_runtime()
    retained_b = cfg._retained_factory_model_state
    assert retained_b is not None

    # Construction cleanup and policy refresh must not erase the last handoff.
    cfg._apply_factory_runtime_snapshot(snapshot_a)
    cfg.discard_prepared_factory_runtime()
    assert cfg._retained_factory_model_state is retained_b
    _assert_factory_model_values(cfg, values_b)

    await cfg.refresh_runtime_policy()
    assert cfg._retained_factory_model_state is retained_b
    _assert_factory_model_values(cfg, values_b)


def test_abort_discards_new_construction_without_erasing_retained_generation():
    snapshot_a, values_a = _factory_model_snapshot("a")
    snapshot_b, _values_b = _factory_model_snapshot("b")
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=None,
    )
    cfg._apply_factory_runtime_snapshot(snapshot_a)
    cfg.handoff_factory_runtime()
    retained_a = cfg._retained_factory_model_state
    assert retained_a is not None

    cfg._apply_factory_runtime_snapshot(snapshot_b)
    cfg.abort_factory_runtime()

    assert cfg._factory_runtime_snapshot is None
    assert cfg._retained_factory_model_state is retained_a
    assert cfg._factory_runtime_handed_off is True
    _assert_factory_model_values(cfg, values_a)


def test_abort_without_retained_generation_leaves_neutral_handed_off_state():
    snapshot, _values = _factory_model_snapshot("initial")
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=None,
    )
    cfg._apply_factory_runtime_snapshot(snapshot)

    cfg.abort_factory_runtime()

    assert cfg._factory_runtime_snapshot is None
    assert cfg._retained_factory_model_state is None
    assert cfg._factory_runtime_handed_off is True
    _assert_factory_model_getters_are_neutral(cfg)


@pytest.mark.parametrize(
    ("lazy_outcome", "expected_close_calls", "expected_invalidate_calls"),
    [
        ("absent", 0, 0),
        ("close-succeeds", 1, 0),
        ("invalidate-succeeds", 1, 1),
        ("cleanup-fails", 1, 1),
    ],
)
def test_close_clears_retained_generation_for_every_lazy_cleanup_outcome(
    lazy_outcome,
    expected_close_calls,
    expected_invalidate_calls,
):
    class MatrixLazySession:
        def __init__(self, outcome):
            self.outcome = outcome
            self.close_calls = 0
            self.invalidate_calls = 0

        def close(self):
            self.close_calls += 1
            if self.outcome in {"invalidate-succeeds", "cleanup-fails"}:
                raise RuntimeError("close failed")

        def invalidate(self):
            self.invalidate_calls += 1
            if self.outcome == "cleanup-fails":
                raise RuntimeError("invalidate failed")

    snapshot, _values = _factory_model_snapshot("close")
    cfg = WebToolConfig(
        db=None,
        db_factory=lambda: (_ for _ in ()).throw(AssertionError("db fallback")),
        request=None,
        user_id=None,
    )
    cfg._apply_factory_runtime_snapshot(snapshot)
    cfg.handoff_factory_runtime()
    lazy_db = None if lazy_outcome == "absent" else MatrixLazySession(lazy_outcome)
    cfg._lazy_db = lazy_db

    if lazy_outcome == "cleanup-fails":
        with pytest.raises(
            ToolFactoryRuntimeSessionBoundaryError,
            match="Tool runtime cleanup could not be completed",
        ):
            cfg.close()
    else:
        cfg.close()

    assert cfg._retained_factory_model_state is None
    assert cfg._factory_runtime_handed_off is True
    _assert_factory_model_getters_are_neutral(cfg)
    if lazy_db is not None:
        assert lazy_db.close_calls == expected_close_calls
        assert lazy_db.invalidate_calls == expected_invalidate_calls

    if lazy_outcome == "cleanup-fails":
        lazy_db.outcome = "close-succeeds"
    cfg.close()
    assert cfg._retained_factory_model_state is None
    assert cfg._lazy_db is None
    _assert_factory_model_getters_are_neutral(cfg)


def test_failed_verified_handoff_does_not_replace_retained_generation(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'retained-handoff-failure.db'}")
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    live_db = factory()
    snapshot_a, _values_a = _factory_model_snapshot("a")
    snapshot_b, _values_b = _factory_model_snapshot("b")
    request = SimpleNamespace(user=object())
    cfg = WebToolConfig(
        db=live_db,
        db_factory=factory,
        request=request,
        user=request.user,
        user_id=None,
    )
    try:
        cfg._apply_factory_runtime_snapshot(snapshot_a)
        cfg.handoff_factory_runtime()
        retained_a = cfg._retained_factory_model_state
        assert retained_a is not None

        live_db.add(
            User(
                username="retained-handoff-failure",
                password_hash="hash",
                is_admin=False,
            )
        )
        cfg._live_db = live_db
        cfg.request = request
        cfg._user = request.user
        cfg._apply_factory_runtime_snapshot(snapshot_b)

        with pytest.raises(ToolFactoryRuntimeSessionBoundaryError):
            cfg.handoff_factory_runtime()

        assert cfg._retained_factory_model_state is retained_a
        assert cfg._factory_runtime_snapshot is snapshot_b
        assert cfg._live_db is live_db
        assert cfg.request is request
        assert cfg._user is request.user
    finally:
        live_db.rollback()
        live_db.close()
        cfg._live_db = None
        cfg.request = None
        cfg._user = None
        cfg.close()
        engine.dispose()


def test_standalone_unhanded_off_model_getter_uses_legacy_loader_once(monkeypatch):
    image_adapter = object()
    loader_calls = 0
    cfg = WebToolConfig(db=object(), request=None, user_id=None)

    def load_image_models():
        nonlocal loader_calls
        loader_calls += 1
        return {"image": image_adapter}

    monkeypatch.setattr(cfg, "_load_image_models", load_image_models)

    first = cfg.get_image_models()
    second = cfg.get_image_models()

    assert loader_calls == 1
    assert first == {"image": image_adapter}
    assert second == {"image": image_adapter}
    assert first is not second
    assert first["image"] is image_adapter
    assert second["image"] is image_adapter


def test_model_getters_delegate_to_shared_resolver(monkeypatch):
    vision_model = object()
    image_model = object()
    cfg = WebToolConfig(db=object(), request=None, user_id=None)
    resolver_calls = []

    def resolve_factory_model_field(**kwargs):
        resolver_calls.append(kwargs)
        if kwargs["field_name"] == "vision_model":
            return vision_model
        return {"image": image_model}

    monkeypatch.setattr(
        cfg,
        "_resolve_factory_model_field",
        resolve_factory_model_field,
    )

    assert cfg.get_vision_model() is vision_model
    assert cfg.get_image_models() == {"image": image_model}
    assert resolver_calls[0]["terminal_neutral"] is None
    assert resolver_calls[1]["terminal_neutral"] == {}


def test_unhanded_off_mapping_loader_avoids_truthiness_and_preserves_values(
    monkeypatch,
):
    image_model = object()
    loader_calls = 0
    cfg = WebToolConfig(db=object(), request=None, user_id=None)

    def load_image_models():
        nonlocal loader_calls
        loader_calls += 1
        return _NonEmptyMappingWithHostileBool(image_model)

    monkeypatch.setattr(cfg, "_load_image_models", load_image_models)

    first = cfg.get_image_models()
    second = cfg.get_image_models()

    assert loader_calls == 1
    assert type(first) is dict
    assert first["model"] is image_model
    assert second["model"] is image_model
    assert first is not second
    first["mutation"] = object()
    assert "mutation" not in second


def test_unhanded_off_invalid_mapping_loader_result_remains_fail_loud(monkeypatch):
    cfg = WebToolConfig(db=object(), request=None, user_id=None)

    monkeypatch.setattr(cfg, "_load_image_models", lambda: None)

    with pytest.raises(TypeError):
        cfg.get_image_models()


def test_empty_unhanded_off_mapping_loader_is_cached_and_returns_fresh_dicts(
    monkeypatch,
):
    loader_calls = 0
    cfg = WebToolConfig(db=object(), request=None, user_id=None)

    def load_image_models():
        nonlocal loader_calls
        loader_calls += 1
        return {}

    monkeypatch.setattr(cfg, "_load_image_models", load_image_models)

    first = cfg.get_image_models()
    second = cfg.get_image_models()

    assert loader_calls == 1
    assert first == second == {}
    assert first is not second


def test_close_neutralizes_prefilled_model_mapping_caches_without_loading(monkeypatch):
    cfg = WebToolConfig(db=object(), request=None, user_id=None)
    mapping_getters = (
        (cfg.get_image_models, "_cached_image_configs", "_load_image_models"),
        (cfg.get_video_models, "_cached_video_configs", "_load_video_models"),
        (cfg.get_asr_models, "_cached_asr_models", "_load_asr_models"),
        (cfg.get_tts_models, "_cached_tts_models", "_load_tts_models"),
        (
            cfg.get_sound_effect_models,
            "_cached_sound_effect_models",
            "_load_sound_effect_models",
        ),
        (cfg.get_music_models, "_cached_music_models", "_load_music_models"),
    )

    def fail_if_called():
        raise AssertionError("terminal config attempted a model loader")

    for _getter, cache_name, loader_name in mapping_getters:
        setattr(cfg, cache_name, {"prefilled": object()})
        monkeypatch.setattr(cfg, loader_name, fail_if_called)

    cfg.close()

    for getter, _cache_name, _loader_name in mapping_getters:
        assert getter() == {}


def test_explicit_vision_model_remains_authoritative_after_close():
    explicit_vision_model = object()
    cfg = WebToolConfig(
        db=object(),
        request=None,
        user_id=None,
        vision_model=explicit_vision_model,
    )

    cfg.close()

    assert cfg.get_vision_model() is explicit_vision_model


@pytest.mark.asyncio
async def test_policy_refresh_defers_full_factory_inputs_until_build(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    await cfg.refresh_runtime_policy()
    assert len(sessions) == 0
    assert cfg._factory_runtime_snapshot is None
    assert cfg._pending_runtime_policy is not None

    await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 1
    assert sessions[0].closed
    assert cfg._factory_runtime_snapshot is None
    assert cfg._pending_runtime_policy is None


@pytest.mark.asyncio
async def test_factory_runtime_snapshot_is_released_when_build_raises(monkeypatch):
    sessions: list[_TrackingSession] = []

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: None,
    )

    async def fail_build(_cls, _config):
        raise RuntimeError("registered tool build failed")

    monkeypatch.setattr(
        ToolRegistry,
        "create_registered_tools",
        classmethod(fail_build),
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        workspace_config={"task_id": "_mock_"},
        task_id="_mock_",
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["basic"]),
    )

    with pytest.raises(RuntimeError, match="registered tool build failed"):
        await ToolFactory.create_all_tools(cfg)

    assert len(sessions) == 1
    assert sessions[0].closed
    assert cfg._factory_runtime_snapshot is None


@pytest.mark.asyncio
async def test_factory_prepare_snapshots_selected_sync_factory_inputs(
    monkeypatch,
):
    """After full prepare, selected synchronous getters read only cached values."""
    main_thread_id = threading.get_ident()
    loader_thread_ids: list[int] = []
    session = _TrackingSession()

    def record(value):
        loader_thread_ids.append(threading.get_ident())
        return value

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        lambda *_args: record("credential"),
    )
    monkeypatch.setattr(
        "xagent.web.tools.config.get_sql_connection_map",
        lambda *_args: record({"WAREHOUSE": "sqlite:///warehouse.db"}),
    )

    model_values = {
        "get_default_vision_model": object(),
        "get_image_models": {"image": object()},
        "get_default_image_generate_model": object(),
        "get_default_image_edit_model": object(),
        "get_video_models": {"video": object()},
        "get_default_video_model": object(),
        "get_asr_models": {"asr": object()},
        "get_default_asr_model": object(),
        "get_tts_models": {"tts": object()},
        "get_default_tts_model": object(),
        "get_sound_effect_models": {"sound": object()},
        "get_default_sound_effect_model": object(),
        "get_music_models": {"music": object()},
        "get_default_music_model": object(),
    }
    for name, value in model_values.items():
        monkeypatch.setattr(
            f"xagent.web.services.model_service.{name}",
            lambda *_args, _value=value, **_kwargs: record(_value),
        )

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=lambda: session,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=[
                "basic",
                "database",
                "image",
                "video",
                "audio",
                "vision",
                "mcp:custom-api",
            ]
        ),
    )

    await cfg.prepare_factory_runtime()

    def fail_factory():
        raise AssertionError("factory getter attempted a second database checkout")

    cfg._db_factory = fail_factory
    assert cfg.get_tool_credential("web_search", "api_key") == "credential"
    assert cfg.get_sql_connections() == {"WAREHOUSE": "sqlite:///warehouse.db"}
    assert cfg.get_custom_api_configs() == []
    assert cfg.get_vision_model() is model_values["get_default_vision_model"]
    assert (
        cfg.get_image_generate_model()
        is model_values["get_default_image_generate_model"]
    )
    assert cfg.get_image_edit_model() is model_values["get_default_image_edit_model"]
    assert cfg.get_video_model() is model_values["get_default_video_model"]
    assert cfg.get_asr_model() is model_values["get_default_asr_model"]
    assert cfg.get_tts_model() is model_values["get_default_tts_model"]
    assert (
        cfg.get_sound_effect_model() is model_values["get_default_sound_effect_model"]
    )
    assert cfg.get_music_model() is model_values["get_default_music_model"]
    mapping_getters = (
        (cfg.get_image_models, "get_image_models"),
        (cfg.get_video_models, "get_video_models"),
        (cfg.get_asr_models, "get_asr_models"),
        (cfg.get_tts_models, "get_tts_models"),
        (cfg.get_sound_effect_models, "get_sound_effect_models"),
        (cfg.get_music_models, "get_music_models"),
    )
    for getter, model_value_name in mapping_getters:
        expected = model_values[model_value_name]
        returned = getter()
        assert returned == expected
        assert returned is not expected
        assert all(returned[key] is value for key, value in expected.items())
    assert loader_thread_ids
    assert all(thread_id != main_thread_id for thread_id in loader_thread_ids)


@pytest.mark.asyncio
async def test_close_neutralizes_old_generation_before_public_prepare_installs_next(
    monkeypatch,
):
    sessions: list[_TrackingSession] = []
    generation_a_model = object()
    generation_b_model = object()
    cached_model = object()
    current_model = generation_a_model

    def session_factory() -> _TrackingSession:
        session = _TrackingSession()
        sessions.append(session)
        return session

    def load_image_models(*_args, **_kwargs):
        return {"image": current_model}

    monkeypatch.setattr(
        "xagent.web.services.model_service.get_image_models",
        load_image_models,
    )
    monkeypatch.setattr(
        "xagent.web.services.model_service.get_default_image_generate_model",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xagent.web.services.model_service.get_default_image_edit_model",
        lambda *_args, **_kwargs: None,
    )
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["image"]),
    )

    try:
        await cfg.prepare_factory_runtime()
        assert cfg.get_image_models()["image"] is generation_a_model
        cfg.handoff_factory_runtime()
        assert cfg.get_image_models()["image"] is generation_a_model

        cfg._cached_image_configs = {"image": cached_model}
        cfg.close()
        assert cfg.get_image_models() == {}

        current_model = generation_b_model
        await cfg.prepare_factory_runtime()

        prepared_models = cfg.get_image_models()
        assert prepared_models["image"] is generation_b_model
        assert prepared_models["image"] is not generation_a_model
        assert prepared_models["image"] is not cached_model
        cfg.handoff_factory_runtime()
        assert cfg.get_image_models()["image"] is generation_b_model
        assert sessions
        assert all(session.closed for session in sessions)
    finally:
        cfg.close()


@pytest.mark.asyncio
async def test_factory_prefetch_isolates_later_read_from_swallowed_sql_failure(
    monkeypatch,
):
    """One optional loader must not poison a later independent DB read."""
    sessions: list[_PostgresAbortSession] = []
    loader_sessions: dict[str, _PostgresAbortSession] = {}
    video_model = object()

    def session_factory() -> _PostgresAbortSession:
        session = _PostgresAbortSession()
        sessions.append(session)
        return session

    def load_broken_images(db: _PostgresAbortSession, _user_id):
        # ``get_image_models`` currently catches SQL errors internally. Model
        # the resulting Postgres transaction state without leaking the error to
        # the snapshot loader that would otherwise know to recover it.
        loader_sessions["image"] = db
        db.swallow_statement_failure()
        return {}

    def load_videos(db: _PostgresAbortSession, _user_id):
        loader_sessions["video"] = db
        db.assert_usable()
        return {"video": video_model}

    monkeypatch.setattr(
        "xagent.web.services.model_service.get_image_models",
        load_broken_images,
    )
    monkeypatch.setattr(
        "xagent.web.services.model_service.get_video_models",
        load_videos,
    )
    monkeypatch.setattr(
        "xagent.web.services.model_service.get_default_video_model",
        lambda *_args, **_kwargs: None,
    )

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=["image", "video"]
        ),
    )

    try:
        await cfg.prepare_factory_runtime()

        assert cfg.get_image_models() == {}
        assert cfg.get_video_models() == {"video": video_model}
        assert loader_sessions["image"] is not loader_sessions["video"]
        assert all(session.closed for session in sessions)
    finally:
        cfg.release_prepared_factory_runtime()
        cfg.close()


@pytest.mark.parametrize(
    ("failing_getter", "expected_input_name"),
    [
        ("get_asr_models", "audio:asr-models"),
        ("get_tts_models", "audio:tts-models"),
        ("get_sound_effect_models", "audio:sound-effect-models"),
        ("get_music_models", "audio:music-models"),
        ("get_default_asr_model", "audio:default-asr"),
        ("get_default_tts_model", "audio:default-tts"),
        ("get_default_sound_effect_model", "audio:default-sound-effect"),
        ("get_default_music_model", "audio:default-music"),
    ],
)
@pytest.mark.asyncio
async def test_audio_prefetch_logs_the_specific_failed_input(
    monkeypatch,
    caplog,
    failing_getter,
    expected_input_name,
):
    from xagent.web.services import model_service

    collection_getters = (
        "get_asr_models",
        "get_tts_models",
        "get_sound_effect_models",
        "get_music_models",
    )
    default_getters = (
        "get_default_asr_model",
        "get_default_tts_model",
        "get_default_sound_effect_model",
        "get_default_music_model",
    )
    for getter_name in collection_getters:
        monkeypatch.setattr(
            model_service,
            getter_name,
            lambda *_args, _getter_name=getter_name, **_kwargs: {
                _getter_name: object()
            },
        )
    for getter_name in default_getters:
        monkeypatch.setattr(
            model_service,
            getter_name,
            lambda *_args, **_kwargs: None,
        )

    def fail_loader(*_args, **_kwargs):
        raise RuntimeError("audio loader failed")

    monkeypatch.setattr(model_service, failing_getter, fail_loader)
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=_TrackingSession,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(tool_categories=["audio"]),
    )
    try:
        with caplog.at_level(logging.WARNING, logger="xagent.web.tools.config"):
            await cfg.prepare_factory_runtime()

        assert f"Failed to prefetch {expected_input_name} tool input" in caplog.text
    finally:
        cfg.release_prepared_factory_runtime()
        cfg.close()


@pytest.mark.asyncio
async def test_factory_prefetch_recovers_before_later_read_after_required_failure(
    monkeypatch,
):
    """A required input failure must not mark an unrelated input unavailable."""
    sessions: list[_PostgresAbortSession] = []
    loader_sessions: dict[str, _PostgresAbortSession] = {}

    def session_factory() -> _PostgresAbortSession:
        session = _PostgresAbortSession()
        sessions.append(session)
        return session

    def load_broken_credential(db: _PostgresAbortSession, *_args):
        loader_sessions["basic"] = db
        raise RuntimeError("credential query failed")

    def load_sql_connections(db: _PostgresAbortSession, _user_id):
        loader_sessions["database"] = db
        db.assert_usable()
        return {"WAREHOUSE": "sqlite:///warehouse.db"}

    monkeypatch.setattr(
        "xagent.web.tools.config.resolve_tool_credential",
        load_broken_credential,
    )
    monkeypatch.setattr(
        "xagent.web.tools.config.get_sql_connection_map",
        load_sql_connections,
    )

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=["basic", "database"]
        ),
    )

    try:
        await cfg.prepare_factory_runtime()

        with pytest.raises(RuntimeError, match="credential snapshot is unavailable"):
            cfg.get_tool_credential("web_search", "api_key")
        assert cfg.get_sql_connections() == {"WAREHOUSE": "sqlite:///warehouse.db"}
        assert loader_sessions["basic"] is not loader_sessions["database"]
        assert all(session.closed for session in sessions)
    finally:
        cfg.release_prepared_factory_runtime()
        cfg.close()


@pytest.mark.asyncio
async def test_factory_prefetch_propagates_later_input_checkout_timeout(monkeypatch):
    """A later input's checkout timeout must not degrade to an empty model set."""
    sessions: list[_PostgresAbortSession] = []
    checkout_timeout = SQLAlchemyTimeoutError("later checkout timed out")
    video_loader_called = False

    def session_factory() -> _PostgresAbortSession:
        session = _PostgresAbortSession(
            checkout_error=checkout_timeout if sessions else None
        )
        sessions.append(session)
        return session

    def load_video_models(*_args):
        nonlocal video_loader_called
        video_loader_called = True
        return {}

    monkeypatch.setattr(
        "xagent.web.services.model_service.get_image_models",
        lambda *_args: {},
    )
    monkeypatch.setattr(
        "xagent.web.services.model_service.get_video_models",
        load_video_models,
    )

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=session_factory,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=["image", "video"]
        ),
    )

    try:
        with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
            await cfg.prepare_factory_runtime()

        assert exc_info.value is checkout_timeout
        assert video_loader_called is False
        assert len(sessions) == 2
        assert all(session.closed for session in sessions)
    finally:
        cfg.release_prepared_factory_runtime()
        cfg.close()


def test_runtime_policy_isolates_override_read_before_allowlist():
    """A swallowed override failure must not erase the independent allowlist."""
    from xagent.web.tools.config import _load_tool_runtime_policy_snapshot

    sessions: list[_PostgresAbortSession] = []
    hook_sessions: dict[str, _PostgresAbortSession] = {}

    def session_factory() -> _PostgresAbortSession:
        session = _PostgresAbortSession()
        sessions.append(session)
        return session

    def load_overrides(db: _PostgresAbortSession, _user):
        hook_sessions["overrides"] = db
        db.swallow_statement_failure()
        return {}

    def load_allowlist(db: _PostgresAbortSession, _user):
        hook_sessions["allowlist"] = db
        db.assert_usable()
        return ["file"]

    set_user_tool_overrides_hook(load_overrides)
    set_user_tool_allowlist_hook(load_allowlist)
    try:
        snapshot = _load_tool_runtime_policy_snapshot(session_factory, 1)

        assert snapshot.tool_overrides == {}
        assert snapshot.tool_allowlist == ["file"]
        assert hook_sessions["overrides"] is not hook_sessions["allowlist"]
        assert all(session.closed for session in sessions)
    finally:
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)


def test_runtime_policy_propagates_later_input_checkout_timeout():
    """The allowlist checkout must not hide a pool timeout as no policy."""
    from xagent.web.tools.config import _load_tool_runtime_policy_snapshot

    sessions: list[_PostgresAbortSession] = []
    checkout_timeout = SQLAlchemyTimeoutError("allowlist checkout timed out")
    allowlist_hook_called = False

    def session_factory() -> _PostgresAbortSession:
        session = _PostgresAbortSession(
            checkout_error=checkout_timeout if sessions else None
        )
        sessions.append(session)
        return session

    def load_allowlist(_db, _user):
        nonlocal allowlist_hook_called
        allowlist_hook_called = True
        return ["file"]

    set_user_tool_overrides_hook(lambda _db, _user: {})
    set_user_tool_allowlist_hook(load_allowlist)
    try:
        with pytest.raises(SQLAlchemyTimeoutError) as exc_info:
            _load_tool_runtime_policy_snapshot(session_factory, 1)

        assert exc_info.value is checkout_timeout
        assert allowlist_hook_called is False
        assert len(sessions) == 2
        assert all(session.closed for session in sessions)
    finally:
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)


@pytest.mark.asyncio
async def test_default_model_prefetch_returns_every_pool_checkout(
    monkeypatch,
    tmp_path,
):
    from xagent.web.models import database
    from xagent.web.models.database import Base
    from xagent.web.services import model_service

    engine = create_engine(
        f"sqlite:///{tmp_path / 'default-models.db'}",
        poolclass=QueuePool,
        pool_size=2,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", factory)

    checkouts = 0
    checkins = 0
    max_checked_out = 0

    def record_checkout(*_args) -> None:
        nonlocal checkouts, max_checked_out
        checkouts += 1
        max_checked_out = max(max_checked_out, engine.pool.checkedout())

    def record_checkin(*_args) -> None:
        nonlocal checkins
        checkins += 1

    event.listen(engine, "checkout", record_checkout)
    event.listen(engine, "checkin", record_checkin)  # codespell:ignore checkin

    collection_getters = (
        "get_image_models",
        "get_video_models",
        "get_asr_models",
        "get_tts_models",
        "get_sound_effect_models",
        "get_music_models",
    )
    for getter_name in collection_getters:
        monkeypatch.setattr(
            f"xagent.web.services.model_service.{getter_name}",
            lambda *_args: {"configured": object()},
        )

    default_getters = (
        "get_default_vision_model",
        "get_default_image_generate_model",
        "get_default_image_edit_model",
        "get_default_video_model",
        "get_default_asr_model",
        "get_default_tts_model",
        "get_default_sound_effect_model",
        "get_default_music_model",
    )
    default_calls: list[str] = []
    for getter_name in default_getters:
        real_getter = getattr(model_service, getter_name)

        def record_default_call(
            *args,
            _getter_name=getter_name,
            _real_getter=real_getter,
            **kwargs,
        ):
            default_calls.append(_getter_name)
            return _real_getter(*args, **kwargs)

        monkeypatch.setattr(model_service, getter_name, record_default_call)

    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=1,
        task_id="_mock_",
        workspace_config={"task_id": "_mock_"},
        include_mcp_tools=False,
        tool_selection_spec=ToolSelectionSpec.from_raw(
            tool_categories=["vision", "image", "video", "audio"]
        ),
    )
    try:
        await cfg.prepare_factory_runtime()

        assert default_calls == list(default_getters)
        assert checkouts == checkins == len(collection_getters) + len(default_getters)
        assert max_checked_out == 1
        assert engine.pool.checkedout() == 0
    finally:
        cfg.close()
        engine.dispose()


def test_legacy_default_model_resolvers_close_owned_pool_connections(
    monkeypatch,
    tmp_path,
):
    from xagent.web.models import database
    from xagent.web.models.database import Base
    from xagent.web.services import model_service

    engine = create_engine(
        f"sqlite:///{tmp_path / 'legacy-default-models.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.1,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    monkeypatch.setattr(database, "_SessionLocal", factory)

    checkouts = 0
    checkins = 0

    def record_checkout(*_args) -> None:
        nonlocal checkouts
        checkouts += 1

    def record_checkin(*_args) -> None:
        nonlocal checkins
        checkins += 1

    event.listen(engine, "checkout", record_checkout)
    event.listen(engine, "checkin", record_checkin)  # codespell:ignore checkin

    default_getters = (
        model_service.get_default_vision_model,
        model_service.get_default_image_generate_model,
        model_service.get_default_image_edit_model,
        model_service.get_default_video_model,
        model_service.get_default_asr_model,
        model_service.get_default_tts_model,
        model_service.get_default_sound_effect_model,
        model_service.get_default_music_model,
    )
    try:
        for getter in default_getters:
            assert getter() is None
            assert engine.pool.checkedout() == 0

        assert checkouts == checkins == len(default_getters)
    finally:
        engine.dispose()


@pytest.mark.asyncio
async def test_runtime_policy_refresh_waits_for_pool_off_event_loop(tmp_path):
    """A saturated policy-query pool must not freeze unrelated coroutines."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'tool-policy.db'}",
        poolclass=QueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.5,
        connect_args={"check_same_thread": False},
    )
    User.__table__.create(bind=engine)
    factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    with factory() as db:
        user = User(username="policy-user", password_hash="hash", is_admin=False)
        db.add(user)
        db.commit()
        user_id = int(user.id)

    def policy_hook(db, user):
        assert db.query(User.id).filter(User.id == user.id).scalar() == user_id
        return {"calculator": {"enabled": False}}

    set_user_tool_overrides_hook(policy_hook)
    set_user_tool_allowlist_hook(lambda _db, _user: ["file"])
    held_connection = engine.connect()
    cfg = WebToolConfig(
        db=None,
        request=None,
        db_factory=factory,
        user_id=user_id,
        user=SimpleNamespace(id=user_id, is_admin=False),
    )
    ticks = 0
    stop = asyncio.Event()

    async def ticker() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.01)

    ticker_task = asyncio.create_task(ticker())
    try:
        await asyncio.sleep(0.02)
        ticks_before_wait = ticks
        refresh_task = asyncio.create_task(cfg.refresh_runtime_policy())
        await asyncio.sleep(0.08)
        assert ticks - ticks_before_wait >= 4
        assert not refresh_task.done()

        held_connection.close()
        await refresh_task
        assert cfg.get_user_tool_overrides() == {"calculator": {"enabled": False}}
        assert cfg.get_user_tool_allowlist() == ["file"]
    finally:
        if not held_connection.closed:
            held_connection.close()
        stop.set()
        await ticker_task
        cfg.close()
        set_user_tool_overrides_hook(None)
        set_user_tool_allowlist_hook(None)
        engine.dispose()


def test_legacy_oauth_session_uses_engine_when_caller_is_connection_bound():
    engine = create_engine("sqlite://")
    connection = engine.connect()
    caller_db = Session(bind=connection)
    cfg = WebToolConfig(db=caller_db, request=None, user_id=1)

    oauth_db = cfg._new_legacy_oauth_session()
    try:
        assert caller_db.get_bind() is connection
        assert oauth_db.get_bind() is engine
    finally:
        oauth_db.close()
        caller_db.close()
        connection.close()
        engine.dispose()


def test_custom_api_loader_uses_factory_session():
    # Factory-only (nested child) config: the loader must mint/reuse the lazy
    # factory session via get_db(), not read the None live ``self.db`` and
    # silently swallow ``None.query`` into an empty tool list.
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    cfg.get_custom_api_configs()
    assert sess.query_calls >= 1


def test_mcp_loader_uses_factory_session():
    sess = _TrackingSession()
    cfg = WebToolConfig(db=None, request=None, db_factory=lambda: sess, user_id=1)
    asyncio.run(cfg._load_mcp_server_configs())
    assert sess.query_calls >= 1


def test_mcp_config_scan_failure_raises_safe_typed_error():
    cfg = WebToolConfig(
        db=_FailingQuerySession(),
        request=None,
        user_id=1,
        include_mcp_tools=True,
    )

    with pytest.raises(MCPConfigLoadError) as exc_info:
        asyncio.run(cfg._load_mcp_server_configs())

    assert exc_info.value.summaries[0].server_name == "MCP server"
    assert exc_info.value.summaries[0].reason == "config_load_failed"
    assert "database-secret" not in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_failed_mcp_config_refresh_never_reuses_stale_cache():
    session = _FailingQuerySession()
    cfg = WebToolConfig(
        db=session,
        request=None,
        user_id=1,
        include_mcp_tools=True,
    )
    cfg._cached_mcp_configs = [{"name": "stale", "config": {"token": "secret"}}]
    cfg._mcp_hook_generation_at_load = -1

    for _ in range(2):
        with pytest.raises(MCPConfigLoadError):
            asyncio.run(cfg.get_mcp_server_configs())

    assert session.query_calls == 2


def test_connector_runtime_turn_switch_invalidates_runtime_caches():
    cfg = WebToolConfig(
        db=None,
        request=None,
        connector_runtime_turn_id="turn-1",
    )
    cfg._connector_runtime_view = {"custom_api:1": {"secrets": {"token": "old"}}}
    cfg._cached_mcp_configs = [{"id": 1, "connector_runtime": {"context": {}}}]

    assert cfg.set_connector_runtime_turn_id("turn-1") is False
    assert cfg._connector_runtime_view is not None
    assert cfg._cached_mcp_configs is not None

    assert cfg.set_connector_runtime_turn_id("turn-2") is True
    assert cfg._connector_runtime_turn_id == "turn-2"
    assert cfg._connector_runtime_view is None
    assert cfg._cached_mcp_configs is None


@dataclass(frozen=True, eq=False)
class _ScopeWithTurnPayload(ExecutionScope):
    """Scope subclass carrying turn-only data outside the namespace fields.

    ``__eq__`` deliberately compares only the inherited namespace fields (the
    same ones ``ExecutionScope.__eq__`` compares), ignoring ``turn_marker``
    and class identity -- mirroring a resolver that hands back a richer scope
    object for the same namespace.
    """

    turn_marker: str = field(default="", compare=False)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionScope):
            return NotImplemented
        namespace = (
            "sandbox_key_suffix",
            "workspace_segments",
            "sandbox_mount_segments",
            "strict_memory_isolation",
            "isolate_external_dirs",
        )
        return [getattr(self, f) for f in namespace] == [
            getattr(other, f) for f in namespace
        ] and dict(self.memory_dimensions) == dict(other.memory_dimensions)

    def __hash__(self) -> int:
        return hash((self.sandbox_key_suffix, self.workspace_segments))


def _prime_scope_derived_caches(cfg: WebToolConfig) -> None:
    cfg._cached_mcp_configs = [{"id": 1, "connector_runtime": {"context": {}}}]
    cfg._factory_runtime_snapshot = object()
    cfg._pending_runtime_policy = object()


def _assert_scope_derived_caches_primed(cfg: WebToolConfig) -> None:
    assert cfg._cached_mcp_configs is not None
    assert cfg._factory_runtime_snapshot is not None
    assert cfg._pending_runtime_policy is not None


def _assert_scope_derived_caches_dropped(cfg: WebToolConfig) -> None:
    assert cfg._cached_mcp_configs is None
    assert cfg._factory_runtime_snapshot is None
    assert cfg._pending_runtime_policy is None


def test_set_execution_scope_swaps_the_scope_and_drops_scope_derived_caches():
    scope_a = ExecutionScope(sandbox_key_suffix="tenant-a")
    cfg = WebToolConfig(db=None, request=None, execution_scope=scope_a)
    _prime_scope_derived_caches(cfg)

    # Same scope object: no-op, every scope-derived cache untouched.
    assert cfg.set_execution_scope(scope_a) is False
    assert cfg.get_execution_scope() is scope_a
    _assert_scope_derived_caches_primed(cfg)

    # A fresh base-class instance comparing equal is also a no-op: the
    # persisted-snapshot path decodes a fresh equal instance every turn and
    # must not force a tool rebuild.
    assert (
        cfg.set_execution_scope(ExecutionScope(sandbox_key_suffix="tenant-a")) is False
    )
    assert cfg.get_execution_scope() is scope_a
    _assert_scope_derived_caches_primed(cfg)

    # Different scope: swaps and drops every scope-derived cache.
    scope_b = ExecutionScope(sandbox_key_suffix="tenant-b")
    assert cfg.set_execution_scope(scope_b) is True
    assert cfg.get_execution_scope() is scope_b
    _assert_scope_derived_caches_dropped(cfg)

    # Repeating the same scope is a no-op again, leaving state alone.
    cfg._cached_mcp_configs = ["sentinel"]
    assert cfg.set_execution_scope(scope_b) is False
    assert cfg.get_execution_scope() is scope_b
    assert cfg._cached_mcp_configs == ["sentinel"]

    # compare=False trap: a subclass instance that compares equal to scope_b
    # by value (same namespace fields) but carries a different type must
    # still be swapped in -- a resolver returning a richer scope for an
    # unchanged namespace must not be dropped as a no-op equality match.
    scope_b_with_turn_payload = _ScopeWithTurnPayload(
        sandbox_key_suffix="tenant-b", turn_marker="turn-9"
    )
    assert scope_b_with_turn_payload == scope_b
    assert type(scope_b_with_turn_payload) is not type(scope_b)

    assert cfg.set_execution_scope(scope_b_with_turn_payload) is True
    assert cfg.get_execution_scope() is scope_b_with_turn_payload
    assert cfg._cached_mcp_configs is None


def test_set_execution_scope_swaps_equal_same_subclass_instances():
    # Two instances of the SAME subclass differing only in payload the
    # subclass excludes from equality: successive turns of a resolver that
    # returns a richer scope per turn. Value equality cannot prove freshness
    # for a subclass, so the setter must swap.
    turn_1 = _ScopeWithTurnPayload(sandbox_key_suffix="tenant-c", turn_marker="turn-1")
    cfg = WebToolConfig(db=None, request=None, execution_scope=turn_1)
    _prime_scope_derived_caches(cfg)

    # The identical object stays a no-op even for a subclass.
    assert cfg.set_execution_scope(turn_1) is False
    _assert_scope_derived_caches_primed(cfg)

    turn_2 = _ScopeWithTurnPayload(sandbox_key_suffix="tenant-c", turn_marker="turn-2")
    assert turn_2 == turn_1
    assert type(turn_2) is type(turn_1)

    assert cfg.set_execution_scope(turn_2) is True
    assert cfg.get_execution_scope() is turn_2
    _assert_scope_derived_caches_dropped(cfg)


def test_connector_runtime_view_resolution_errors_fail_closed(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    cfg = WebToolConfig(
        db=object(),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
    )

    try:
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            cfg._load_connector_runtime_view()
        assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
        assert exc_info.value.status_code == 503
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert str(exc_info.value.__cause__) == "database unavailable"
        assert cfg._connector_runtime_view is None
    finally:
        cfg.close()


def test_mcp_config_loader_propagates_runtime_view_resolution_error(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    for name in (
        "load_user_env_overrides",
        "load_shared_env_overrides",
        "load_user_env_sources",
    ):
        monkeypatch.setattr(
            f"xagent.web.services.mcp_runtime.{name}", lambda *_a, **_k: {}
        )

    server = SimpleNamespace(
        id=7,
        name="ShiftCare",
        transport="streamable_http",
        description="runtime connector",
        runtime_bindings=[],
        allow_delegated_authorization=False,
        runtime_input_schema=None,
    )
    cfg = WebToolConfig(
        db=_StaticRowsSession([server]),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
        include_mcp_tools=True,
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        asyncio.run(cfg._load_mcp_server_configs())

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_custom_api_config_loader_propagates_runtime_view_resolution_error(monkeypatch):
    def _raise_runtime_lookup_error(**_kwargs):
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(
        "xagent.web.services.connector_runtime.load_connector_runtime_view",
        _raise_runtime_lookup_error,
    )
    api = SimpleNamespace(
        id=11,
        name="ShiftCare",
        description="runtime API",
        url="https://api.example.test",
        method="GET",
        headers={},
        body=None,
        env={},
        runtime_input_schema=None,
        runtime_bindings=[],
        allow_delegated_authorization=False,
    )
    cfg = WebToolConfig(
        db=_StaticRowsSession([api]),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        cfg.get_custom_api_configs()

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert isinstance(exc_info.value.__cause__, RuntimeError)
