"""``GET /v1/models`` and ``GET /v1/agents/{id}/assignable-models``, over /v1.

The list is ``usable_models`` for the holder, so it must agree with what the
create path accepts and with what ``GET /v1/agents/{id}`` reports as
unresolved.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web import models as _all_models  # noqa: F401  (registers every table)
from xagent.web.api.v1.agents import list_agent_assignable_models
from xagent.web.api.v1.models import list_models
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserModel
from xagent.web.schemas.v1 import V1Model, V1ModelForAgent
from xagent.web.services import agent_management
from xagent.web.services.agent_management import (
    AgentManagementService,
    InvalidAgentModelConfigError,
)
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.model_service import (
    set_model_visibility_hook,
    set_visible_user_ids_hook,
    usable_models,
)

from ..conftest import _direct_db_session, client
from .test_agent_detail import (
    _CELL_NAMES,
    NOT_FOUND,
    _add_model,
    _add_user,
    _build_world,
    _get,
    _seed_agent,
    _set_models,
    _user_id,
)
from .test_management import _bearer, _personal_key
from .test_skills import _KEY_KINDS, _error, _headers_for

pytestmark = pytest.mark.usefixtures("_test_db")

MODEL_KEYS = {
    "id",
    "model_id",
    "model_name",
    "model_provider",
    "category",
    "abilities",
    "is_platform",
    "owned_by_you",
}


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    set_visible_user_ids_hook(None)
    set_model_visibility_hook(None)
    set_agent_team_scope_hook(None)


def _list(key: str):
    return client.get("/v1/models", headers=_bearer(key))


def _list_for_agent(key: str, agent_id: Any):
    return client.get(f"/v1/agents/{agent_id}/assignable-models", headers=_bearer(key))


def _set_columns(model_pk: int, **columns: Any) -> None:
    db = _direct_db_session()
    try:
        row = db.get(Model, model_pk)
        for name, value in columns.items():
            setattr(row, name, value)
        db.commit()
    finally:
        db.close()


# ===== only a live personal key opens the route =====


@pytest.mark.parametrize("kind", _KEY_KINDS)
def test_models_route_accepts_only_a_live_personal_key(kind):
    assert len(_KEY_KINDS) == 4

    _error(client.get("/v1/models", headers=_headers_for(kind)), 401, "invalid_api_key")


# ===== the fields =====


def test_model_fields_are_exactly_and_carry_their_values():
    key = _personal_key()
    me = _user_id("admin")
    other = _add_user("other")
    owned_platform = _add_model("owned-platform", links=((me, True, True),))
    shared_platform = _add_model(
        "shared-platform",
        category="embedding",
        links=((other, True, True), (me, False, False)),
    )
    owned_plain = _add_model(
        "owned-plain", provider="router", links=((me, True, False),)
    )
    shared_plain = _add_model("shared-plain", links=((other, True, True),))
    # The holder only sees a shared row from a visible user.
    set_visible_user_ids_hook(lambda _db, _uid: [other])
    _set_columns(owned_platform, managed_by="platform", abilities=["chat", "vision"])
    _set_columns(shared_platform, managed_by="platform", abilities=[])
    _set_columns(owned_plain, managed_by=None, abilities=["chat"])
    _set_columns(shared_plain, managed_by="user", abilities=["vision"])

    resp = _list(key)

    assert resp.status_code == 200, resp.text
    rows = {row["id"]: row for row in resp.json()}

    def expected(pk, name, provider, category, abilities, platform, owned):
        return {
            "id": pk,
            "model_id": name,
            "model_name": name,
            "model_provider": provider,
            "category": category,
            "abilities": abilities,
            "is_platform": platform,
            "owned_by_you": owned,
        }

    assert rows == {
        owned_platform: expected(
            owned_platform,
            "owned-platform",
            "openai",
            "llm",
            ["chat", "vision"],
            True,
            True,
        ),
        shared_platform: expected(
            shared_platform, "shared-platform", "openai", "embedding", [], True, False
        ),
        owned_plain: expected(
            owned_plain, "owned-plain", "router", "llm", ["chat"], False, True
        ),
        shared_plain: expected(
            shared_plain, "shared-plain", "openai", "llm", ["vision"], False, False
        ),
    }
    assert all(set(row) == MODEL_KEYS for row in rows.values())


@pytest.mark.parametrize(
    ("stored", "listed"),
    [
        (["chat", "vision"], ["chat", "vision"]),
        (["chat", 1, None, 2.5], ["chat", "1", "None", "2.5"]),
        ("chat", ["chat"]),
        (None, []),
        ({"chat": True}, []),
        (7, []),
    ],
    ids=["strings", "mixed-elements", "string", "null", "object", "number"],
)
def test_abilities_are_listed_as_strings(stored, listed):
    key = _personal_key()
    me = _user_id("admin")
    pk = _add_model("with-abilities", links=((me, True, False),))
    _set_columns(pk, abilities=stored)

    (row,) = _list(key).json()

    assert row["abilities"] == listed


@pytest.mark.parametrize("schema", [V1Model, V1ModelForAgent])
def test_every_model_field_is_described(schema):
    assert all(field.description for field in schema.model_fields.values())


# ===== which models are listed =====


def test_listing_has_every_category_in_usable_models_order():
    world = _build_world(False)
    resp = _list(world.key)

    db = _direct_db_session()
    try:
        expected = [row.id for row in usable_models(db, world.holder)]
    finally:
        db.close()
    assert [row["id"] for row in resp.json()] == expected


@pytest.mark.parametrize("cell", _CELL_NAMES)
def test_listing_matches_create_validation(cell):
    world = _build_world(False)
    model_id = world.models[cell]
    listed = {row["id"] for row in _list(world.key).json()}

    db = _direct_db_session()
    try:
        try:
            AgentManagementService(db)._validate_models(
                {"general": model_id}, user_id=world.holder
            )
            accepted = True
        except InvalidAgentModelConfigError:
            accepted = False
    finally:
        db.close()

    assert (model_id in listed) == accepted, cell


@pytest.mark.parametrize("holder_is_owner", [True, False], ids=["H=O", "H!=O"])
@pytest.mark.parametrize("cell", [*_CELL_NAMES, "nonexistent-999999"])
def test_unresolved_models_match_the_model_listing(cell, holder_is_owner):
    world = _build_world(holder_is_owner)
    model_id = 999_999 if cell == "nonexistent-999999" else world.models[cell]
    listed = {row["id"]: row["category"] for row in _list(world.key).json()}
    _set_models(world.agent_id, {"general": model_id})

    unresolved = _get(world.key, world.agent_id).json()["unresolved_models"]

    assert (unresolved == ["general"]) == (listed.get(model_id) != "llm"), cell


# ===== statements do not grow with the number of models =====


def _statements(engine: Any, call: Callable[[], Any]) -> tuple[Any, int]:
    seen: list[str] = []

    def capture(_conn, _cursor, statement, *_rest):
        seen.append(statement)

    event.listen(engine, "before_cursor_execute", capture)
    try:
        result = call()
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    return result, len(seen)


def _owned_by_another_user(factory: Any, holder_id: int) -> int:
    """A team agent the holder manages and another user owns."""
    scope = AgentTeamScope(team_id=1, is_team_admin=False)
    set_agent_team_scope_hook(
        lambda _db, user_id: scope if int(user_id) == holder_id else None
    )
    with factory() as db:
        owner = User(username="owner", password_hash="x", is_admin=False)
        db.add(owner)
        db.flush()
        agent = Agent(user_id=owner.id, name="team agent", team_id=1)
        db.add(agent)
        db.commit()
        return int(agent.id)


def test_agent_listing_query_count_does_not_grow_with_models():
    from xagent.web.models.database import get_session_local

    factory = get_session_local()
    holder_id = _seed_shared_json_models(factory, 0, 3)
    agent_id = _owned_by_another_user(factory, holder_id)
    rows, small_count = _statements(
        factory.kw["bind"], lambda: _call_list_for_agent(holder_id, agent_id)
    )
    assert [row.category for row in rows] == ["llm", "llm"]
    assert [row.owner_can_use for row in rows] == [True, True]

    _seed_shared_json_models(factory, 3, 30)
    rows, large_count = _statements(
        factory.kw["bind"], lambda: _call_list_for_agent(holder_id, agent_id)
    )

    assert len(rows) == 15
    assert small_count > 0 and small_count == large_count


# ===== the listing for one agent =====


@pytest.mark.parametrize("kind", _KEY_KINDS)
def test_agent_listing_accepts_only_a_live_personal_key(kind):
    resp = client.get("/v1/agents/1/assignable-models", headers=_headers_for(kind))

    _error(resp, 401, "invalid_api_key")


@pytest.mark.parametrize("holder_is_owner", [True, False], ids=["H=O", "H!=O"])
def test_agent_listing_marks_what_the_owner_can_use(holder_is_owner):
    world = _build_world(holder_is_owner)
    named = {pk: name for name, pk in world.models.items()}

    resp = _list_for_agent(world.key, world.agent_id)

    assert resp.status_code == 200, resp.text
    rows = resp.json()
    # Only chat models the holder can use, each marked for the owner. A model
    # the owner cannot use is marked, not left out.
    assert {named[row["id"]]: row["owner_can_use"] for row in rows} == (
        {"m1": True, "m2": True, "m3": True, "m4": True}
        if holder_is_owner
        else {"m1": False, "m3": True, "m4": True}
    )
    assert all(set(row) == MODEL_KEYS | {"owner_can_use"} for row in rows)
    assert all(type(row["owner_can_use"]) is bool for row in rows)
    # The listing without an agent never carries the mark.
    plain = _list(world.key).json()
    assert plain and all(set(row) == MODEL_KEYS for row in plain)


@pytest.mark.parametrize("bad", ["0", "-1", str(2**31), "abc"])
def test_agent_listing_refuses_an_id_out_of_range(bad):
    world = _build_world(False)
    _error(_list_for_agent(world.key, bad), 422, "invalid_input")


def test_an_agent_the_holder_cannot_manage_is_not_found():
    world = _build_world(False)
    someone_elses = _seed_agent(_user_id("tess"))

    hidden = _list_for_agent(world.key, someone_elses)

    assert hidden.status_code == 404 and hidden.json() == NOT_FOUND
    assert hidden.content == _list_for_agent(world.key, 999_999).content


# ===== read failures are errors, not empty answers =====


def test_listing_read_failure_is_internal_error():
    world = _build_world(False)
    assert _list(world.key).json(), "the failure must be hiding a non-empty list"
    boom = OperationalError("SELECT secret_table", {}, Exception("secret detail"))

    with patch(
        "xagent.web.services.model_service.rank_name_matched_models", side_effect=boom
    ):
        resp = _list(world.key)

    _error(resp, 500, "internal_error")
    assert "secret" not in resp.text


def test_no_usable_model_is_an_empty_list():
    key = _personal_key()
    _add_model("someone-elses", links=((_add_user("someone"), True, False),))

    resp = _list(key)

    assert resp.status_code == 200
    assert resp.json() == []


# ===== the queries run on PostgreSQL =====


def _seed_shared_json_models(factory: Any, start: int, stop: int) -> int:
    """Add shared models ``start`` to ``stop - 1`` with JSON abilities.

    Each model is linked to two users, so a join of ``models`` and
    ``user_models`` would repeat it; de-duplicating that join needs equality
    on the JSON ``abilities`` column, which PostgreSQL does not define. The
    users are created on the first call. Returns the holder's user id.
    """
    with factory() as db:
        if start == 0:
            db.add_all(
                [
                    User(username="sysadmin", password_hash="x", is_admin=True),
                    User(username="holder", password_hash="x", is_admin=False),
                ]
            )
            db.flush()
        users = {u.username: int(u.id) for u in db.query(User)}
        for i in range(start, stop):
            model = Model(
                model_id=f"shared-json-{i}",
                category="llm" if i % 2 == 0 else "image",
                model_provider="openai",
                model_name=f"shared-json-{i}",
                api_key="test-key",
                abilities=["chat", "vision"],
                is_active=True,
            )
            db.add(model)
            db.flush()
            db.add_all(
                [
                    UserModel(
                        user_id=users["sysadmin"],
                        model_id=model.id,
                        is_owner=True,
                        is_shared=True,
                    ),
                    UserModel(
                        user_id=users["holder"],
                        model_id=model.id,
                        is_owner=False,
                        is_shared=False,
                    ),
                ]
            )
        db.commit()
        return users["holder"]


def _call_list(holder_id: int) -> list[Any]:
    return asyncio.run(
        list_models(
            authed=(
                SimpleNamespace(id=holder_id),
                SimpleNamespace(key_prefix="xag_personal_t"),
            )
        )
    )


def _call_list_for_agent(holder_id: int, agent_id: int) -> list[Any]:
    authed = (
        SimpleNamespace(id=holder_id),
        SimpleNamespace(key_prefix="xag_personal_t"),
    )
    return asyncio.run(list_agent_assignable_models(agent_id=agent_id, authed=authed))


def _read_shared_json_models(factory: Any) -> None:
    holder_id = _seed_shared_json_models(factory, 0, 3)
    rows, small_count = _statements(factory.kw["bind"], lambda: _call_list(holder_id))
    assert small_count > 0
    assert [row.abilities for row in rows] == [["chat", "vision"]] * 3
    assert [row.category for row in rows] == ["llm", "image", "llm"]
    assert not any(row.owned_by_you for row in rows)
    with factory() as db:
        for row in rows:
            AgentManagementService(db)._validate_models(
                {"general": row.id}, user_id=holder_id
            )

    _seed_shared_json_models(factory, 3, 30)
    rows, large_count = _statements(factory.kw["bind"], lambda: _call_list(holder_id))

    assert len(rows) == 30
    assert small_count == large_count


def test_listing_query_count_does_not_grow_with_models():
    """Shared JSON models on SQLite, so the query path runs without PostgreSQL."""
    from xagent.web.models.database import get_session_local

    _read_shared_json_models(get_session_local())


@pytest.mark.postgresql
def test_shared_json_models_are_listed_on_postgresql(monkeypatch):
    with disposable_database_factory("xagent_v1_models_listing") as make_database:
        engine = make_database("listing")
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        monkeypatch.setattr(agent_management, "get_session_local", lambda: factory)

        _read_shared_json_models(factory)
