import asyncio
import logging
import threading
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from xagent.core.tools.adapters.vibe.config import MCPConfigLoadError
from xagent.core.tools.adapters.vibe.connector_runtime import (
    ERROR_CONNECTOR_RUNTIME_UNAVAILABLE,
    ConnectorRuntimeError,
)
from xagent.core.tools.adapters.vibe.factory import ToolFactory, ToolRegistry
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
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


class _Chain:
    """Minimal chainable query stub: filter/join return self, terminals empty."""

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
        return self

    def all(self):
        return []

    def first(self):
        return None


class _ListChain:
    """Minimal chainable query stub with a fixed ``all()`` result."""

    def __init__(self, rows):
        self._rows = list(rows)

    def filter(self, *a, **k):
        return self

    def join(self, *a, **k):
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

        assert live_db.query(ToolConfig).all() == []
        assert engine.pool.checkedout() == 1
        cfg.release_db_connection()
        assert engine.pool.checkedout() == 0
    finally:
        live_db.close()
        cfg.close()
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
    assert cfg.get_image_models() is model_values["get_image_models"]
    assert (
        cfg.get_image_generate_model()
        is model_values["get_default_image_generate_model"]
    )
    assert cfg.get_image_edit_model() is model_values["get_default_image_edit_model"]
    assert cfg.get_video_models() is model_values["get_video_models"]
    assert cfg.get_video_model() is model_values["get_default_video_model"]
    assert cfg.get_asr_models() is model_values["get_asr_models"]
    assert cfg.get_asr_model() is model_values["get_default_asr_model"]
    assert cfg.get_tts_models() is model_values["get_tts_models"]
    assert cfg.get_tts_model() is model_values["get_default_tts_model"]
    assert cfg.get_sound_effect_models() is model_values["get_sound_effect_models"]
    assert (
        cfg.get_sound_effect_model() is model_values["get_default_sound_effect_model"]
    )
    assert cfg.get_music_models() is model_values["get_music_models"]
    assert cfg.get_music_model() is model_values["get_default_music_model"]
    assert loader_thread_ids
    assert all(thread_id != main_thread_id for thread_id in loader_thread_ids)


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
        db=_StaticRowsSession([SimpleNamespace(custom_api=api)]),
        request=None,
        task_id="web_task_123",
        user_id=1,
        connector_runtime_turn_id="turn-1",
    )

    with pytest.raises(ConnectorRuntimeError) as exc_info:
        cfg.get_custom_api_configs()

    assert exc_info.value.code == ERROR_CONNECTOR_RUNTIME_UNAVAILABLE
    assert isinstance(exc_info.value.__cause__, RuntimeError)
