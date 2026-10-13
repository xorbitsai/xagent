"""Tool-list usage counts are optional and must not break the listing (#3007)."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Session

from tests.web.services.task_database_shared import engine as engine_fixture
from tests.web.services.task_database_shared import task_id as task_id_fixture
from xagent.web.api import tools as tools_api
from xagent.web.models.database import release_db_connection_if_clean
from xagent.web.models.task import TraceEvent
from xagent.web.models.tool_config import ToolConfig

engine = engine_fixture
task_id = task_id_fixture


def _seed(db: Session, task_id: int) -> None:
    db.add(
        ToolConfig(
            tool_name="search",
            tool_type="basic",
            category="basic",
            display_name="Search",
            enabled=False,
        )
    )
    db.add(
        TraceEvent(
            task_id=task_id,
            event_id="usage-1",
            event_type="tool_execution_end",
            timestamp=datetime(2026, 10, 12, tzinfo=timezone.utc),
            data={"tool_name": "search", "success": True},
        )
    )
    db.commit()


def _broken_usage_query(db: Session):
    # A real database error: on PostgreSQL it aborts the transaction.
    return db.query(sa.literal_column("missing_column").label("tool_name"))


def test_usage_counts_are_read_without_touching_the_request_transaction(
    engine, task_id
):
    with Session(engine) as db:
        _seed(db, task_id)
        timeout_before = None
        if engine.dialect.name == "postgresql":
            with engine.connect() as connection:
                timeout_before = connection.exec_driver_sql(
                    "SHOW statement_timeout"
                ).scalar()

        assert tools_api._load_tool_usage_counts(db) == {"search": 1}

        # The statistics timeout is scoped to the statistics query only.
        if timeout_before is not None:
            timeout_after = (
                db.connection().exec_driver_sql("SHOW statement_timeout").scalar()
            )
            assert timeout_after == timeout_before
        # A read-only lookup must leave the session releasable before slow awaits.
        assert db.in_transaction()
        assert release_db_connection_if_clean(db) is True
        assert [row.tool_name for row in db.query(ToolConfig).all()] == ["search"]


def test_failed_usage_query_leaves_tool_metadata_readable(
    engine, task_id, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tools_api, "_tool_usage_query", _broken_usage_query)
    with Session(engine) as db:
        _seed(db, task_id)

        assert tools_api._load_tool_usage_counts(db) == {}
        assert [row.tool_name for row in db.query(ToolConfig).all()] == ["search"]


@pytest.mark.postgresql
@pytest.mark.parametrize("engine", ["postgresql"], indirect=True)
def test_slow_usage_query_is_cancelled_by_the_statistics_timeout(
    engine, task_id, monkeypatch: pytest.MonkeyPatch
):
    def slow_usage_query(db: Session):
        return db.query(
            sa.literal("slow").label("tool_name"),
            sa.literal(1).label("usage_count"),
        ).select_from(sa.func.pg_sleep(2))

    monkeypatch.setattr(tools_api, "_tool_usage_query", slow_usage_query)
    monkeypatch.setattr(tools_api, "_USAGE_STATS_STATEMENT_TIMEOUT_MS", 50)
    with Session(engine) as db:
        _seed(db, task_id)

        assert tools_api._load_tool_usage_counts(db) == {}
        assert [row.tool_name for row in db.query(ToolConfig).all()] == ["search"]


@pytest.mark.asyncio
async def test_listing_applies_tool_policy_after_usage_statistics_fail(
    engine, task_id, monkeypatch: pytest.MonkeyPatch
):
    # The production sequence from #3007: the statistics statement fails, then
    # the route reads ToolConfig on the same request session.
    class Config:
        def __init__(self, **_kwargs) -> None:
            pass

        def get_vision_model(self):
            return None

        def get_image_models(self):
            return {}

        get_video_models = get_asr_models = get_tts_models = get_image_models
        get_sound_effect_models = get_music_models = get_image_models
        get_user_tool_overrides = get_image_models

        def close(self) -> None:
            return None

    async def create_all_tools(_config, apply_user_override_filter: bool = True):
        return [
            SimpleNamespace(
                name="search",
                description="",
                metadata=SimpleNamespace(category=SimpleNamespace(value="basic")),
            )
        ]

    monkeypatch.setattr(tools_api, "WebToolConfig", Config)
    monkeypatch.setattr(
        "xagent.core.tools.adapters.vibe.factory.ToolFactory.create_all_tools",
        create_all_tools,
    )
    monkeypatch.setattr("xagent.web.sandbox_manager.get_sandbox_manager", lambda: None)
    monkeypatch.setattr(tools_api, "get_default_tool_configs", lambda: [])
    monkeypatch.setattr(tools_api, "_tool_usage_query", _broken_usage_query)
    with Session(engine) as db:
        _seed(db, task_id)

        response = await tools_api.get_available_tools(
            current_user=SimpleNamespace(id=7, is_admin=False),
            db=db,
        )

    [tool] = response["tools"]
    assert tool["usage_count"] == 0
    # The seeded policy row disables the tool, so this proves the read ran.
    assert tool["enabled"] is False
    assert tool["status"] == "disabled"
