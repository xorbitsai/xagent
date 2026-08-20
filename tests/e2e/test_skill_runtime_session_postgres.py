from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from collections.abc import Iterator
from functools import partial
from pathlib import Path
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
from typing import Any
from uuid import uuid4

import psycopg2
import pytest
from sqlalchemy import text
from sqlalchemy.exc import TimeoutError as SQLAlchemyTimeoutError
from sqlalchemy.orm import Session
from sqlalchemy.pool import QueuePool

from tests.e2e.app_harness import (
    create_e2e_user,
    disable_external_app_services,
    init_e2e_db,
    run_e2e_app_client,
)
from tests.e2e.minio_harness import (
    _docker_available,
    _docker_client,
    run_container_with_dynamic_ports,
)
from xagent.core.agent.service import AgentService
from xagent.core.tools.adapters.vibe.factory import ToolFactory
from xagent.core.tools.adapters.vibe.selection_spec import ToolSelectionSpec
from xagent.web.models.database import get_engine
from xagent.web.models.skill import UserSkill, UserSkillFile
from xagent.web.models.user import User
from xagent.web.tools.config import WebToolConfig

pytestmark = [pytest.mark.e2e, pytest.mark.docker]

POSTGRES_PASSWORD = "xagent_test"
POSTGRES_DATABASE = "xagent_test"
SKILL_INDEX_SENTINEL = "Pool handoff regression fixture"
QUESTION = "Which deployment target should I use?"
EXPECTED_SKILL_INDEX_LINE = (
    "- session-safe: Pool handoff regression fixture "
    "When to use: Test authenticated Skill database reads"
)
EXPECTED_LOAD_SKILL_PARAMETERS = {
    "properties": {
        "skill_name": {
            "description": (
                "Exact name of the skill to load, as listed in the skill index."
            ),
            "type": "string",
        }
    },
    "required": ["skill_name"],
    "type": "object",
}
AVAILABLE_TOOLS_MODEL_GETTERS = (
    "get_vision_model",
    "get_image_models",
    "get_video_models",
    "get_asr_models",
    "get_tts_models",
    "get_sound_effect_models",
    "get_music_models",
)


@pytest.fixture
def postgres_url() -> Iterator[str]:
    if not _docker_available():
        pytest.skip("Requires reachable Docker daemon")

    client = _docker_client()
    container, host_ports = run_container_with_dynamic_ports(
        client,
        "postgres:16-bookworm",
        name=f"xagent-postgres-e2e-{uuid4().hex[:12]}",
        container_ports=("5432/tcp",),
        tmpfs={"/var/lib/postgresql/data": "rw,size=256m"},
        environment={
            "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
            "POSTGRES_DB": POSTGRES_DATABASE,
        },
    )
    host_port = host_ports["5432/tcp"]

    database_url = (
        "postgresql+psycopg2://postgres:"
        f"{POSTGRES_PASSWORD}@127.0.0.1:{host_port}/{POSTGRES_DATABASE}"
    )
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            try:
                connection = psycopg2.connect(
                    host="127.0.0.1",
                    port=host_port,
                    user="postgres",
                    password=POSTGRES_PASSWORD,
                    dbname=POSTGRES_DATABASE,
                    connect_timeout=1,
                )
            except psycopg2.OperationalError:
                time.sleep(0.25)
            else:
                connection.close()
                break
        else:
            raise RuntimeError("PostgreSQL did not become ready")

        yield database_url
    finally:
        container.remove(force=True)


def _configure_postgres_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    tmp_path: Path,
    postgres_url: str,
) -> None:
    monkeypatch.setenv("DATABASE_URL", postgres_url)
    monkeypatch.setenv("XAGENT_DB_POOL_SIZE", "1")
    monkeypatch.setenv("XAGENT_DB_MAX_OVERFLOW", "0")
    monkeypatch.setenv("XAGENT_DB_POOL_TIMEOUT_SECONDS", "1")
    monkeypatch.setenv("XAGENT_FILE_STORAGE_STARTUP_SYNC_ENABLED", "false")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("XAGENT_FILE_MATERIALIZE_DIR", str(tmp_path / "materialized"))
    monkeypatch.setenv("LANCEDB_DIR", str(tmp_path / "lancedb"))
    monkeypatch.setenv("LANCEDB_PATH", str(tmp_path / "lancedb-path"))
    monkeypatch.setenv("LANCEDB_AUTO_MIGRATE", "false")


def _seed_personal_skill(db: Any, *, user_id: int) -> str:
    name = "session-safe"
    content = b"""---
name: session-safe
description: Pool handoff regression fixture
when_to_use: Test authenticated Skill database reads
---

# Session Safe
"""
    skill = UserSkill(
        user_id=user_id,
        name=name,
        origin="custom",
        created_by_user_id=user_id,
        updated_by_user_id=user_id,
    )
    skill.files.append(
        UserSkillFile(
            path="SKILL.md",
            content=content,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            media_type="text/markdown",
        )
    )
    db.add(skill)
    db.commit()
    return name


class _WaitingLLM:
    model_name = "waiting-regression"
    context_window = 32_000

    def __init__(self, *, call_id: str) -> None:
        self.call_id = call_id
        self.calls: list[dict[str, Any]] = []

    async def chat(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return {
            "content": "I need one required choice.",
            "tool_calls": [
                {
                    "id": self.call_id,
                    "function": {
                        "name": "ask_user_question",
                        "arguments": json.dumps(
                            {
                                "message": QUESTION,
                                "interactions": [],
                            }
                        ),
                    },
                }
            ],
        }


def _owned_references(value: object) -> list[object]:
    if value is None or type(value) in {str, bytes, int, float, bool}:
        return []
    if isinstance(value, ModuleType | type):
        return []
    if isinstance(value, dict):
        return [*value.keys(), *value.values()]
    if isinstance(value, list | tuple | set | frozenset | deque):
        return list(value)
    if isinstance(value, partial):
        return [value.func, *value.args, *(value.keywords or {}).values()]
    if isinstance(value, MethodType):
        return [value.__self__, value.__func__]
    if isinstance(value, FunctionType):
        references: list[object] = [
            *(value.__defaults__ or ()),
            *(value.__kwdefaults__ or {}).values(),
        ]
        for cell in value.__closure__ or ():
            try:
                references.append(cell.cell_contents)
            except ValueError:
                pass
        return references
    try:
        return list(vars(value).values())
    except TypeError:
        return []


def _assert_resources_not_reachable(
    *,
    roots: dict[str, object],
    forbidden: dict[str, object],
) -> None:
    forbidden_by_id = {id(value): name for name, value in forbidden.items()}
    queue = deque((name, value) for name, value in roots.items())
    seen: set[int] = set()
    while queue:
        path, value = queue.popleft()
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        forbidden_name = forbidden_by_id.get(value_id)
        assert forbidden_name is None, f"{path} retains caller {forbidden_name}"
        for index, reference in enumerate(_owned_references(value)):
            queue.append((f"{path}[{index}]", reference))


def _assert_detached_retained_service(
    *,
    caller_db: Session,
    request: object,
    user: User,
    user_id: int,
    config: WebToolConfig,
    service: AgentService,
) -> None:
    assert get_engine().pool.checkedout() == 0, "verified factory handoff leaked"
    assert config is service.tool_config
    assert config._live_db is None
    assert config._lazy_db is None
    assert config.request is None
    assert config._user is None
    assert service.skill_scope_context.user_id == user_id
    assert not hasattr(service.skill_scope_context, "db")
    assert not hasattr(service.skill_scope_context, "request")
    assert not hasattr(service.skill_scope_context, "user")
    assert caller_db is not config._live_db
    assert request is not config.request
    assert user is not config._user
    _assert_resources_not_reachable(
        roots={
            "tools": service.tools,
            "config": config,
            "service": service,
        },
        forbidden={
            "Session": caller_db,
            "request": request,
            "ORM User": user,
        },
    )


def test_authenticated_skill_routes_handoff_one_slot_postgres_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postgres_url: str,
) -> None:
    _configure_postgres_app(
        monkeypatch,
        tmp_path=tmp_path,
        postgres_url=postgres_url,
    )
    disable_external_app_services(monkeypatch)
    SessionLocal = init_e2e_db()
    with SessionLocal() as db:
        user = create_e2e_user(db, username="skill-runtime-user")
        skill_name = _seed_personal_skill(db, user_id=user.id)

    with run_e2e_app_client(
        monkeypatch,
        username=user.username,
        user_id=user.id,
    ) as app:
        skills_response = app.client.get("/api/skills/", headers=app.headers)
        assert skills_response.status_code == 200, skills_response.text
        assert skill_name in {item["name"] for item in skills_response.json()}
        assert get_engine().pool.checkedout() == 0

        installed_response = app.client.get(
            "/api/skill-hub/installed",
            headers=app.headers,
        )
        assert installed_response.status_code == 200, installed_response.text
        assert skill_name in {item["name"] for item in installed_response.json()}
        assert get_engine().pool.checkedout() == 0


def test_available_tools_route_uses_retained_models_before_request_db_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postgres_url: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _configure_postgres_app(
        monkeypatch,
        tmp_path=tmp_path,
        postgres_url=postgres_url,
    )
    disable_external_app_services(monkeypatch)
    session_factory = init_e2e_db()
    with session_factory() as seed_db:
        seeded_user = create_e2e_user(
            seed_db,
            username="available-tools-pool-user",
        )

    engine = get_engine()
    pool = engine.pool
    assert engine.dialect.name == "postgresql"
    assert isinstance(pool, QueuePool)
    assert pool.size() == 1
    assert pool._max_overflow == 0  # noqa: SLF001
    assert pool.timeout() == 1
    assert pool.checkedout() == 0

    getter_call_counts: dict[int, dict[str, int]] = {}
    factory_returns: list[tuple[WebToolConfig, bool, bool, dict[str, int]]] = []
    real_getters = {
        name: getattr(WebToolConfig, name) for name in AVAILABLE_TOOLS_MODEL_GETTERS
    }

    def _getter_spy(name: str, real_getter: Any) -> Any:
        def _delegate(config: WebToolConfig) -> Any:
            counts = getter_call_counts.setdefault(id(config), {})
            counts[name] = counts.get(name, 0) + 1
            return real_getter(config)

        return _delegate

    for getter_name, real_getter in real_getters.items():
        monkeypatch.setattr(
            WebToolConfig,
            getter_name,
            _getter_spy(getter_name, real_getter),
        )

    real_create_all_tools = ToolFactory.create_all_tools

    async def _factory_spy(
        config: WebToolConfig,
        apply_user_override_filter: bool = True,
    ) -> list[Any]:
        tools = await real_create_all_tools(
            config,
            apply_user_override_filter=apply_user_override_filter,
        )
        factory_returns.append(
            (
                config,
                config._retained_factory_model_state is not None,
                config._factory_runtime_handed_off,
                dict(getter_call_counts.get(id(config), {})),
            )
        )
        return tools

    monkeypatch.setattr(
        ToolFactory,
        "create_all_tools",
        staticmethod(_factory_spy),
    )

    def _assert_route_getters_called_once(
        config: WebToolConfig,
        factory_counts: dict[str, int],
    ) -> None:
        total_counts = getter_call_counts[id(config)]
        assert total_counts == {
            name: factory_counts.get(name, 0) + 1
            for name in AVAILABLE_TOOLS_MODEL_GETTERS
        }

    with run_e2e_app_client(
        monkeypatch,
        username=seeded_user.username,
        user_id=seeded_user.id,
    ) as app:
        baseline_response = app.client.get(
            "/api/tools/available",
            headers=app.headers,
        )
        assert baseline_response.status_code == 200, baseline_response.text
        baseline_body = baseline_response.json()
        assert isinstance(baseline_body["tools"], list)
        assert isinstance(baseline_body["count"], int)
        assert baseline_body["count"] == len(baseline_body["tools"])
        assert baseline_body["count"] > 0
        assert all(
            isinstance(tool, dict)
            and isinstance(tool.get("name"), str)
            and isinstance(tool.get("usage_count"), int)
            for tool in baseline_body["tools"]
        )
        assert len(factory_returns) == 1

        (
            baseline_config,
            baseline_retained_at_factory_return,
            baseline_handed_off_at_factory_return,
            baseline_factory_counts,
        ) = factory_returns[0]
        assert type(baseline_config) is WebToolConfig
        assert baseline_retained_at_factory_return is True
        assert baseline_handed_off_at_factory_return is True
        _assert_route_getters_called_once(
            baseline_config,
            baseline_factory_counts,
        )
        assert baseline_config._retained_factory_model_state is None
        assert baseline_config._lazy_db is None
        assert pool.checkedout() == 0

        with app.session_factory() as probe_db:
            assert probe_db.execute(text("SELECT 1")).scalar_one() == 1
        assert pool.checkedout() == 0

        real_handoff = WebToolConfig.handoff_factory_runtime

        def _restore_legacy_post_handoff_reads(config: WebToolConfig) -> None:
            # Preserve verified detach while restoring only the stale read mode.
            real_handoff(config)
            config._retained_factory_model_state = None
            config._factory_runtime_handed_off = False

        monkeypatch.setattr(
            WebToolConfig,
            "handoff_factory_runtime",
            _restore_legacy_post_handoff_reads,
        )
        caplog.clear()
        tools_logger = logging.getLogger("xagent.web.api.tools")
        capture_directly = caplog.handler not in logging.getLogger().handlers
        if capture_directly:
            tools_logger.addHandler(caplog.handler)
        try:
            with (
                caplog.at_level(logging.ERROR, logger="xagent.web.api.tools"),
                pytest.raises(SQLAlchemyTimeoutError, match="QueuePool limit"),
            ):
                app.client.get(
                    "/api/tools/available",
                    headers=app.headers,
                )
        finally:
            if capture_directly:
                tools_logger.removeHandler(caplog.handler)

        assert len(factory_returns) == 2
        (
            mutation_config,
            mutation_retained_at_factory_return,
            mutation_handed_off_at_factory_return,
            mutation_factory_counts,
        ) = factory_returns[1]
        assert type(mutation_config) is type(baseline_config)
        assert mutation_config is not baseline_config
        assert mutation_retained_at_factory_return is False
        assert mutation_handed_off_at_factory_return is False
        _assert_route_getters_called_once(
            mutation_config,
            mutation_factory_counts,
        )
        usage_errors = [
            record.getMessage()
            for record in caplog.records
            if record.name == "xagent.web.api.tools"
            and record.getMessage().startswith("Failed to fetch tool usage stats:")
        ]
        assert len(usage_errors) == 1
        assert "QueuePool limit of size 1 overflow 0 reached" in usage_errors[0]

        # Route cleanup must release the lazy checkout even after the timeout.
        assert mutation_config._retained_factory_model_state is None
        assert mutation_config._factory_runtime_handed_off is True
        assert mutation_config._lazy_db is None
        assert pool.checkedout() == 0
        with app.session_factory() as probe_db:
            assert probe_db.execute(text("SELECT 1")).scalar_one() == 1
        assert pool.checkedout() == 0


@pytest.mark.asyncio
async def test_retained_agent_services_wait_without_holding_postgres_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    postgres_url: str,
) -> None:
    from xagent.web.api.chat import create_default_tools

    _configure_postgres_app(
        monkeypatch,
        tmp_path=tmp_path,
        postgres_url=postgres_url,
    )
    disable_external_app_services(monkeypatch)
    session_factory = init_e2e_db()
    with session_factory() as seed_db:
        seeded_user = create_e2e_user(
            seed_db,
            username="retained-skill-runtime-user",
        )
        skill_name = _seed_personal_skill(seed_db, user_id=seeded_user.id)

    engine = get_engine()
    pool = engine.pool
    assert engine.dialect.name == "postgresql"
    assert isinstance(pool, QueuePool)
    assert pool.size() == 1
    assert pool._max_overflow == 0  # noqa: SLF001
    assert pool.timeout() == 1
    assert pool.checkedout() == 0

    real_create_all_tools = ToolFactory.create_all_tools
    factory_calls: list[WebToolConfig] = []

    async def _count_create_all_tools(
        config: WebToolConfig,
        apply_user_override_filter: bool = True,
    ) -> list[Any]:
        factory_calls.append(config)
        return await real_create_all_tools(
            config,
            apply_user_override_filter=apply_user_override_filter,
        )

    real_load_mcp = WebToolConfig._load_mcp_server_configs
    mcp_loads: list[tuple[WebToolConfig, list[dict[str, Any]]]] = []

    async def _spy_load_mcp(
        config: WebToolConfig,
    ) -> list[dict[str, Any]]:
        loaded = await real_load_mcp(config)
        mcp_loads.append((config, loaded))
        return loaded

    monkeypatch.setattr(
        ToolFactory,
        "create_all_tools",
        staticmethod(_count_create_all_tools),
    )
    monkeypatch.setattr(
        WebToolConfig,
        "_load_mcp_server_configs",
        _spy_load_mcp,
    )

    selection = ToolSelectionSpec.from_raw(tool_categories=["skill", "mcp"])
    retained: list[
        tuple[list[Any], WebToolConfig, AgentService, _WaitingLLM, Session]
    ] = []
    try:
        for index in range(2):
            caller_db = session_factory()
            orm_user = caller_db.get(User, seeded_user.id)
            assert orm_user is not None
            assert orm_user.username == seeded_user.username
            assert pool.checkedout() == 1
            request = SimpleNamespace(
                marker=f"request-{index}-{uuid4().hex}",
                user=orm_user,
            )

            tools, config = await create_default_tools(
                caller_db,
                request=request,
                user=orm_user,
                task_id=f"retained-skill-runtime-{index}",
                allowed_skills=[skill_name],
                tool_selection_spec=selection,
            )
            llm = _WaitingLLM(call_id=f"ask-{index}")
            service = AgentService(
                name=f"retained-skill-runtime-{index}",
                id=f"retained-skill-runtime-{index}",
                pattern="react",
                llm=llm,
                tools=tools,
                tool_config=config,
                enable_workspace=False,
            )

            _assert_detached_retained_service(
                caller_db=caller_db,
                request=request,
                user=orm_user,
                user_id=seeded_user.id,
                config=config,
                service=service,
            )
            retained.append((tools, config, service, llm, caller_db))

        assert len(factory_calls) == 2
        assert len(mcp_loads) == 2
        assert all(loaded == [] for _config, loaded in mcp_loads)
        assert [config for config, _loaded in mcp_loads] == [
            retained[0][1],
            retained[1][1],
        ]

        for index, (tools, config, service, llm, _caller_db) in enumerate(retained):
            result = await service.execute_task(
                f"Run the session-safe skill, service {index}.",
                task_id=f"retained-skill-runtime-{index}",
            )

            assert result["success"] is False
            assert result["status"] == "waiting_for_user"
            assert result["message"] == QUESTION
            assert result["message_type"] == "question"
            assert result["interactions"] == []
            assert result["chat_response"] == {
                "message": QUESTION,
                "interactions": [],
            }
            assert (
                service.get_execution_status(f"retained-skill-runtime-{index}")[
                    "status"
                ]
                == "waiting_for_user"
            )
            assert len(llm.calls) == 1
            first_call = llm.calls[0]
            system_text = "\n".join(
                str(message.get("content", ""))
                for message in first_call["messages"]
                if message.get("role") == "system"
            )
            assert SKILL_INDEX_SENTINEL in system_text
            assert EXPECTED_SKILL_INDEX_LINE in system_text
            load_skill_schemas = [
                schema
                for schema in first_call["tools"]
                if schema["function"]["name"] == "load_skill"
            ]
            assert load_skill_schemas == [
                {
                    "type": "function",
                    "function": {
                        "name": "load_skill",
                        "description": (
                            "Load a skill's full instructions into the system context.\n"
                            "\n"
                            "The available skills are listed in the system context "
                            "with one-line summaries. Call this when one of them "
                            "clearly matches the current task; its detailed guidance "
                            "becomes available from the next step on. Do not load "
                            "skills that are unrelated to the task."
                        ),
                        "parameters": EXPECTED_LOAD_SKILL_PARAMETERS,
                    },
                }
            ]
            assert service.tools == tools
            assert config is service.tool_config
            assert pool.checkedout() == 0

        assert len(factory_calls) == 2
        with session_factory() as probe_db:
            assert probe_db.execute(text("SELECT 1")).scalar_one() == 1
        assert pool.checkedout() == 0

        real_handoff = WebToolConfig.handoff_factory_runtime
        mutation_db = session_factory()
        mutation_config: WebToolConfig | None = None
        try:
            mutation_user = mutation_db.get(User, seeded_user.id)
            assert mutation_user is not None
            mutation_request = SimpleNamespace(
                marker=f"mutation-{uuid4().hex}",
                user=mutation_user,
            )
            monkeypatch.setattr(
                WebToolConfig,
                "handoff_factory_runtime",
                WebToolConfig.discard_prepared_factory_runtime,
            )
            _mutation_tools, mutation_config = await create_default_tools(
                mutation_db,
                request=mutation_request,
                user=mutation_user,
                task_id="retained-skill-runtime-mutation",
                allowed_skills=[skill_name],
                tool_selection_spec=selection,
            )
            with pytest.raises(AssertionError, match="verified factory handoff leaked"):
                _assert_detached_retained_service(
                    caller_db=mutation_db,
                    request=mutation_request,
                    user=mutation_user,
                    user_id=seeded_user.id,
                    config=mutation_config,
                    service=AgentService(
                        name="retained-skill-runtime-mutation",
                        id="retained-skill-runtime-mutation",
                        pattern="react",
                        llm=_WaitingLLM(call_id="ask-mutation"),
                        tools=_mutation_tools,
                        tool_config=mutation_config,
                        enable_workspace=False,
                    ),
                )
        finally:
            monkeypatch.setattr(
                WebToolConfig,
                "handoff_factory_runtime",
                real_handoff,
            )
            mutation_db.close()
            if mutation_config is not None:
                mutation_config.close()
    finally:
        for _tools, config, _service, _llm, caller_db in retained:
            caller_db.close()
            config.close()
        engine.dispose()
