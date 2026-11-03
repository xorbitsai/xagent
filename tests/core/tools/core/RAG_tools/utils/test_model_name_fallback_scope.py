"""A bare model_name only resolves to a model the requesting user may use.

Callers such as the KB ingestion tools pass the literal ``"text-embedding-v4"``.
That is a ``model_name`` many users' rows can share, so resolving it must never
pick another user's private row (their API key and their bill).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.core.model.model import EmbeddingModelConfig
from xagent.core.tools.core.RAG_tools.core.exceptions import EmbeddingAdapterError
from xagent.core.tools.core.RAG_tools.utils import model_resolver
from xagent.core.tools.core.RAG_tools.utils.user_scope import user_scope_context
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserModel
from xagent.web.services import model_service

SHARED_NAME = "text-embedding-v4"


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Session]:
    url = f"sqlite:///{tmp_path / 'models.db'}"
    engine = create_engine(url)
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    monkeypatch.setattr(model_resolver, "get_default_db_url", lambda: url)
    for env in ("DASHSCOPE_EMBEDDING_MODEL", "DASHSCOPE_API_KEY"):
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setattr(model_service, "_visible_user_ids_hook", None)
    monkeypatch.setattr(model_service, "_model_visibility_hook", None)
    model_resolver._reset_model_hub_cache()
    yield session
    model_resolver._reset_model_hub_cache()
    session.close()
    engine.dispose()


def _user(db: Session, name: str, *, is_admin: bool = False) -> int:
    user = User(username=name, password_hash="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    return int(user.id)


def _model(
    db: Session,
    model_id: str,
    owner_id: int,
    *,
    model_name: str = SHARED_NAME,
    shared: bool = False,
    managed_by: Optional[str] = None,
) -> None:
    row = Model(
        model_id=model_id,
        category="embedding",
        model_provider="dashscope",
        model_name=model_name,
        api_key=f"{model_id}-key",
        managed_by=managed_by,
    )
    db.add(row)
    db.flush()
    db.add(
        UserModel(user_id=owner_id, model_id=row.id, is_owner=True, is_shared=shared)
    )
    db.commit()


def _resolve(user_id: Optional[int], name: str = SHARED_NAME) -> str:
    with user_scope_context(user_id, False):
        cfg, _adapter = model_resolver.resolve_embedding_adapter(name)
    return cfg.id


@pytest.fixture
def tenants(db: Session) -> dict[str, int]:
    """Four rows share one model_name; the lowest ids are the riskiest picks."""
    ids = {
        "admin": _user(db, "admin", is_admin=True),
        "alice": _user(db, "alice"),
        "bob": _user(db, "bob"),
        "carol": _user(db, "carol"),
    }
    _model(db, "alice-private", ids["alice"])
    _model(db, "admin-custom", ids["admin"], shared=True)
    _model(
        db,
        "platform/text-embedding-v4",
        ids["admin"],
        shared=True,
        managed_by="platform",
    )
    _model(db, "bob-private", ids["bob"])
    return ids


def test_user_without_own_row_gets_the_platform_row_not_another_tenants(
    tenants: dict[str, int],
) -> None:
    assert _resolve(tenants["carol"]) == "platform/text-embedding-v4"


def test_users_own_row_wins_over_shared_rows(tenants: dict[str, int]) -> None:
    assert _resolve(tenants["bob"]) == "bob-private"
    assert _resolve(tenants["alice"]) == "alice-private"


def test_without_a_user_only_admin_shared_rows_count(tenants: dict[str, int]) -> None:
    assert _resolve(None) == "platform/text-embedding-v4"


def test_hidden_platform_row_falls_back_to_other_shared_rows(
    tenants: dict[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        model_service,
        "_model_visibility_hook",
        lambda _db, _uid, row: not str(row.model_id).startswith("platform/"),
    )
    assert _resolve(tenants["carol"]) == "admin-custom"


def test_another_tenants_private_row_is_never_resolved(
    db: Session, tenants: dict[str, int]
) -> None:
    _model(db, "alice-bge", tenants["alice"], model_name="bge-m3")

    for user_id in (tenants["carol"], None):
        with pytest.raises(EmbeddingAdapterError):
            _resolve(user_id, "bge-m3")


def test_exact_model_id_still_resolves_directly(tenants: dict[str, int]) -> None:
    assert _resolve(tenants["carol"], "admin-custom") == "admin-custom"


def test_direct_hub_load_is_scoped_too(tenants: dict[str, int]) -> None:
    with user_scope_context(tenants["carol"], False):
        cfg = model_resolver._load_model_from_hub(
            SHARED_NAME, EmbeddingModelConfig, EmbeddingAdapterError
        )
    assert cfg.id == "platform/text-embedding-v4"
