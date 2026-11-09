"""SQLAlchemyModelHub.load's model_name fallback is ordered and scopeable."""

from __future__ import annotations

from typing import Any, Iterator

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table
from xagent.core.model.storage.error import ModelNotFoundError

SHARED_NAME = "text-embedding-v4"


@pytest.fixture
def hub() -> Iterator[SQLAlchemyModelHub]:
    engine = create_engine("sqlite:///:memory:")

    # SQLite returns unordered SELECTs in reverse, so a fallback that relies on
    # scan order instead of ORDER BY picks a different row here.
    @event.listens_for(engine, "connect")
    def _reverse_unordered(dbapi_connection: Any, _record: Any) -> None:
        dbapi_connection.execute("PRAGMA reverse_unordered_selects = ON")

    base = declarative_base()
    model = create_model_table(base)
    base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    model_hub = SQLAlchemyModelHub(session, model)
    for model_id in ("first", "second", "third"):
        model_hub.store(
            EmbeddingModelConfig(
                id=model_id,
                model_name=SHARED_NAME,
                model_provider="dashscope",
                api_key=f"{model_id}-key",
            )
        )
    yield model_hub
    model_hub.close()
    engine.dispose()


def test_name_fallback_without_scope_picks_the_oldest_row(
    hub: SQLAlchemyModelHub,
) -> None:
    assert hub.load(SHARED_NAME).id == "first"


def test_name_fallback_hands_the_scope_every_match_in_primary_key_order(
    hub: SQLAlchemyModelHub,
) -> None:
    seen: list[str] = []

    def scope(db: Session, matches: list[Any]) -> list[Any]:
        assert db is hub.db
        seen.extend(row.model_id for row in matches)
        return matches[::-1]

    assert hub.load(SHARED_NAME, name_match_scope=scope).id == "third"
    assert seen == ["first", "second", "third"]


def test_name_fallback_raises_when_the_scope_allows_no_match(
    hub: SQLAlchemyModelHub,
) -> None:
    with pytest.raises(ModelNotFoundError):
        hub.load(SHARED_NAME, name_match_scope=lambda _db, _matches: [])


def test_exact_model_id_does_not_consult_the_scope(hub: SQLAlchemyModelHub) -> None:
    def scope(_db: Session, _matches: list[Any]) -> list[Any]:
        raise AssertionError("scope must not run for an exact model_id")

    assert hub.load("second", name_match_scope=scope).api_key == "second-key"


def test_name_fallback_skips_inactive_rows(hub: SQLAlchemyModelHub) -> None:
    hub.db.query(hub.Model).filter(hub.Model.model_id == "first").update(
        {hub.Model.is_active: False}
    )
    hub.db.commit()

    assert hub.load(SHARED_NAME).id == "second"
