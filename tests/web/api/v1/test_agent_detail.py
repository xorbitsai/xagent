"""``GET /v1/agents/{agent_id}``: one agent's full configuration over /v1.

The route reads the agent through the same function the other agent
sub-resources use, so the set of readable agents is the set ``GET /v1/agents``
lists. ``unresolved_models`` is computed for the key holder and is checked here
against what slot resolution at run time does for the same user.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

import pytest
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from tests.shared.postgres_disposable import disposable_database_factory
from xagent.web import models as _all_models  # noqa: F401  (registers every table)
from xagent.web.api.v1.agents import get_agent
from xagent.web.api.v1.errors import _DEFAULT_MESSAGES, V1ErrorCode
from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserModel
from xagent.web.services import agent_management, agent_store
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.hot_path_cache import (
    InMemoryTTLCache,
    set_cache_backend_for_testing,
)
from xagent.web.services.llm_utils import (
    AutoModelUnavailableError,
    load_agent_builder_config,
)
from xagent.web.services.model_service import (
    set_model_visibility_hook,
    set_visible_user_ids_hook,
    slot_assignable_models,
    usable_models,
)

from ..conftest import _admin_headers, _direct_db_session, _register_second_user, client
from .test_management import _bearer, _personal_key
from .test_skills import _KEY_KINDS, _error, _headers_for, _key_for

pytestmark = pytest.mark.usefixtures("_test_db")

SLOTS = ("general", "small_fast", "visual", "compact")
NOT_FOUND = {
    "error": {
        "code": "agent_not_found",
        "message": _DEFAULT_MESSAGES[V1ErrorCode.AGENT_NOT_FOUND],
    }
}
# The keys the response carries, and nothing else.
DETAIL_KEYS = {
    "id",
    "user_id",
    "team_id",
    "visibility",
    "name",
    "description",
    "instructions",
    "execution_mode",
    "models",
    "unresolved_models",
    "knowledge_bases",
    "skills",
    "tool_categories",
    "suggested_prompts",
    "prompt_context_fields",
    "logo_url",
    "status",
    "published_at",
    "created_at",
    "updated_at",
    "widget_enabled",
    "allowed_domains",
    "share_enabled",
    "share_updated_at",
}
FORBIDDEN_KEYS = {
    "widget_key",
    "share_token",
    "base_url",
    "api_key",
    "model_name",
    "model_id",
    "template_id",
    "widget_voice_enabled",
}


@pytest.fixture(autouse=True)
def _reset_hooks():
    yield
    set_agent_team_scope_hook(None)
    set_visible_user_ids_hook(None)
    set_model_visibility_hook(None)


# ===== helpers =====


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _add_user(username: str, *, is_admin: bool = False) -> int:
    db = _direct_db_session()
    try:
        user = User(username=username, password_hash="x", is_admin=is_admin)
        db.add(user)
        db.commit()
        return int(user.id)
    finally:
        db.close()


def _seed_agent(
    owner_id: int,
    *,
    team_id: int | None = None,
    visibility: str = "team",
    origin: str = AgentOrigin.USER.value,
    models: Any = None,
    **columns: Any,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=owner_id,
            name=f"agent {db.query(Agent).count()}",
            team_id=team_id,
            visibility=visibility,
            origin=origin,
            models=models,
            **columns,
        )
        db.add(agent)
        db.commit()
        return int(agent.id)
    finally:
        db.close()


def _set_models(agent_id: int, models: Any) -> None:
    db = _direct_db_session()
    try:
        db.get(Agent, agent_id).models = models
        db.commit()
    finally:
        db.close()


def _get(key: str, agent_id: int):
    return client.get(f"/v1/agents/{agent_id}", headers=_bearer(key))


def _add_model(
    name: str,
    *,
    category: str = "llm",
    active: bool = True,
    provider: str = "openai",
    model_name: str | None = None,
    links: tuple[tuple[int, bool, bool], ...] = (),
) -> int:
    """Insert a model and its link rows ``(user_id, is_owner, is_shared)``."""
    db = _direct_db_session()
    try:
        model = Model(
            model_id=name,
            category=category,
            model_provider=provider,
            model_name=model_name or name,
            api_key="test-key",
            abilities=["chat"],
            is_active=active,
        )
        db.add(model)
        db.flush()
        for user_id, is_owner, is_shared in links:
            db.add(
                UserModel(
                    user_id=user_id,
                    model_id=model.id,
                    is_owner=is_owner,
                    is_shared=is_shared,
                )
            )
        db.commit()
        return int(model.id)
    finally:
        db.close()


def _walk_keys(value: Any):
    if isinstance(value, dict):
        for key, inner in value.items():
            yield key
            yield from _walk_keys(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from _walk_keys(inner)


# ===== only a live personal key opens the route =====


@pytest.mark.parametrize("kind", _KEY_KINDS)
def test_new_agent_routes_accept_only_a_live_personal_key(kind):
    assert len(_KEY_KINDS) == 4
    resp = client.get("/v1/agents/1", headers=_headers_for(kind))

    _error(resp, 401, "invalid_api_key")


# ===== who can read =====

_COLUMNS = [
    "own-personal",
    "same-team-others-personal",
    "team-visible-others",
    "admins-only-others",
    "team-visible-own",
    "admins-only-own",
    "other-team",
    "workforce-manager-own",
    "nonexistent",
]
_EXPECTED = {
    "no-hook": {
        "own-personal": 200,
        "same-team-others-personal": 404,
        "workforce-manager-own": 404,
        "nonexistent": 404,
    },
    "member": dict(zip(_COLUMNS, [200, 404, 200, 404, 200, 404, 404, 404, 404])),
    "team-admin": dict(zip(_COLUMNS, [200, 404, 200, 200, 200, 200, 404, 404, 404])),
}
_CELLS = [
    (role, column, status)
    for role, row in _EXPECTED.items()
    for column, status in row.items()
]


def _column_spec(column: str, me: int, bob: int) -> dict[str, Any]:
    return {
        "own-personal": {"owner_id": me},
        "same-team-others-personal": {"owner_id": bob},
        "team-visible-others": {"owner_id": bob, "team_id": 1},
        "admins-only-others": {"owner_id": bob, "team_id": 1, "visibility": "admins"},
        "team-visible-own": {"owner_id": me, "team_id": 1},
        "admins-only-own": {"owner_id": me, "team_id": 1, "visibility": "admins"},
        "other-team": {"owner_id": bob, "team_id": 2},
        "workforce-manager-own": {
            "owner_id": me,
            "origin": AgentOrigin.WORKFORCE_GENERATED_MANAGER.value,
        },
    }[column]


def _seed_column(column: str, me: int, bob: int) -> int:
    if column == "nonexistent":
        return 999_999
    return _seed_agent(**_column_spec(column, me, bob))


def _install_role(role: str, me: int) -> None:
    if role == "no-hook":
        return
    scope = AgentTeamScope(team_id=1, is_team_admin=role == "team-admin")
    set_agent_team_scope_hook(
        lambda _db, user_id: scope if int(user_id) == me else None
    )


assert len(_CELLS) == 22


@pytest.mark.parametrize(
    ("role", "column", "status"), _CELLS, ids=[f"{r}-{c}" for r, c, _ in _CELLS]
)
def test_detail_visibility_matrix(role, column, status):
    key = _personal_key()
    me = _user_id("admin")
    bob = _add_user("bob")
    _install_role(role, me)
    agent_id = _seed_column(column, me, bob)

    resp = _get(key, agent_id)

    assert resp.status_code == status, resp.text
    if status == 404:
        assert resp.json() == NOT_FOUND
        # An agent that is hidden looks exactly like one that does not exist.
        assert resp.content == _get(key, 999_998).content
    else:
        body = resp.json()
        spec = _column_spec(column, me, bob)
        assert body["id"] == agent_id
        assert body["team_id"] == spec.get("team_id")
        assert body["visibility"] == spec.get("visibility", "team")


@pytest.mark.parametrize("role", list(_EXPECTED))
def test_detail_set_equals_list_set(role):
    key = _personal_key()
    me = _user_id("admin")
    bob = _add_user("bob")
    _install_role(role, me)
    # The table has no cells for team agents when no team hook is installed.
    columns = [column for column in _COLUMNS if column in _EXPECTED[role]]
    seeded = [_seed_column(column, me, bob) for column in columns]

    readable = {
        agent_id for agent_id in seeded if _get(key, agent_id).status_code == 200
    }
    listed = {
        row["id"] for row in client.get("/v1/agents", headers=_bearer(key)).json()
    }

    assert readable, "the comparison must not be between two empty sets"
    assert readable == listed


@pytest.mark.parametrize("bad", ["0", "-1", str(2**31), "abc", "from-template"])
def test_agent_id_must_be_a_positive_32_bit_integer(bad):
    _error(
        client.get(f"/v1/agents/{bad}", headers=_bearer(_personal_key())),
        422,
        "invalid_input",
    )


# ===== the fields =====


def test_detail_fields_are_exactly():
    key = _personal_key()
    me = _user_id("admin")
    model = _add_model("shared-llm", links=((me, True, True),))
    agent_id = _seed_agent(me, models={"general": model})

    body = _get(key, agent_id).json()

    assert set(body) == DETAIL_KEYS
    assert not FORBIDDEN_KEYS & set(_walk_keys(body))


def test_detail_empty_values_come_back_empty():
    key = _personal_key()
    agent_id = _seed_agent(_user_id("admin"))

    body = _get(key, agent_id).json()

    assert body["models"] is None
    assert body["unresolved_models"] == []
    assert body["description"] is None
    assert body["instructions"] is None
    for name in ("knowledge_bases", "skills", "suggested_prompts", "allowed_domains"):
        assert body[name] == []


# ===== the model world shared by the reference-kind tests =====


@dataclass
class World:
    key: str
    holder: int
    agent_id: int
    models: dict[str, int]
    hidden: list[int]


def _build_world(holder_is_owner: bool) -> World:
    """Users, a team, and the eleven kinds of model reference.

    ``hana`` holds the key. ``ollie`` is the agent's owner unless the holder is.
    ``tara`` is the administrator of team 1, which ``hana`` and ``ollie`` belong
    to; ``tess`` is a third person; ``admin`` is a system administrator.
    """
    _admin_headers()
    sysadmin = _user_id("admin")
    headers = _register_second_user("hana")
    key = _key_for(headers)["full_key"]
    hana = _user_id("hana")
    ollie, tara, tess = (_add_user(n) for n in ("ollie", "tara", "tess"))
    scopes = {
        hana: AgentTeamScope(team_id=1, is_team_admin=False),
        ollie: AgentTeamScope(team_id=1, is_team_admin=False),
        tara: AgentTeamScope(team_id=1, is_team_admin=True),
    }
    set_agent_team_scope_hook(lambda _db, user_id: scopes.get(int(user_id)))

    def visible_sharers(db, user_id):
        admins = [int(u.id) for u in db.query(User).filter(User.is_admin)]
        scope = scopes.get(int(user_id)) if user_id is not None else None
        if scope is None:
            return admins
        return admins + [
            uid
            for uid, other in scopes.items()
            if other.team_id == scope.team_id and other.is_team_admin
        ]

    set_visible_user_ids_hook(visible_sharers)

    m2_owner = hana if holder_is_owner else ollie
    models = {
        "m1": _add_model("m1-holder-private", links=((hana, True, False),)),
        "m2": _add_model("m2-owner-private", links=((m2_owner, True, False),)),
        "m3": _add_model("m3-sysadmin-shared", links=((sysadmin, True, True),)),
        "m4": _add_model("m4-team-admin-shared", links=((tara, True, True),)),
        "m5": _add_model("m5-inactive", active=False, links=((sysadmin, True, True),)),
        "m6": 999_999,
        "m7": _add_model(
            "m7-hidden-image", category="image", links=((sysadmin, True, True),)
        ),
        "m8": _add_model("m8-holder-nonowner-link", links=((hana, False, False),)),
        "m9": _add_model("m9-third-person", links=((tess, True, False),)),
        "m10": _add_model(
            "m10-sysadmin-image", category="image", links=((sysadmin, True, True),)
        ),
        "m11": _add_model(
            "m11-team-admin-image", category="image", links=((tara, True, True),)
        ),
    }
    hidden = [models["m7"]]
    set_model_visibility_hook(
        lambda _db, user_id, model: user_id is None or int(model.id) not in hidden
    )
    if holder_is_owner:
        agent_id = _seed_agent(hana)
    else:
        agent_id = _seed_agent(ollie, team_id=1)
    return World(key=key, holder=hana, agent_id=agent_id, models=models, hidden=hidden)


_CELL_NAMES = [f"m{i}" for i in range(1, 12)]
# What the holder can put in a slot, whoever owns the agent.
_HOLDER_RESOLVES = {
    True: {"m1", "m2", "m3", "m4"},
    False: {"m1", "m3", "m4"},
}


assert len(_CELL_NAMES) == 11


def test_unresolved_models_follow_the_slot_order():
    world = _build_world(False)
    ok, bad = world.models["m3"], world.models["m9"]
    _set_models(
        world.agent_id,
        {"compact": bad, "visual": bad, "small_fast": ok, "general": bad},
    )

    body = _get(world.key, world.agent_id).json()

    assert body["unresolved_models"] == ["general", "visual", "compact"]


# ===== stored shapes =====

_USABLE, _UNUSABLE, _USABLE_AS_STRING = "usable", "unusable", "usable-as-string"
# (name, stored value, expected `models`, expected `unresolved_models`)
#
# A stored string, boolean, float or out-of-range integer is always listed.
# How slot resolution treats such a value depends on the database dialect and
# driver (it may resolve, fail, or find nothing), so only the read is asserted:
# it reports the slot as needing a rewrite with an integer id, even when the
# integer the value spells would be usable.
_SHAPES = [
    ("none", None, None, []),
    ("empty-dict", {}, {}, []),
    ("usable-id", {"general": _USABLE}, {"general": _USABLE}, []),
    ("unusable-id", {"general": _UNUSABLE}, {"general": _UNUSABLE}, ["general"]),
    ("null-slot", {"general": None}, {"general": None}, []),
    ("list", [1, 2], None, []),
    ("string", "x", None, []),
    ("unknown-key", {"foo": 3}, {"foo": 3}, []),
    (
        "numeric-string",
        {"general": _USABLE_AS_STRING},
        {"general": _USABLE_AS_STRING},
        ["general"],
    ),
    ("leading-zero", {"general": "0123"}, {"general": "0123"}, ["general"]),
    (
        "huge-string",
        {"general": "9999999999"},
        {"general": "9999999999"},
        ["general"],
    ),
    ("beyond-int32", {"general": 2**31}, {"general": 2**31}, ["general"]),
    ("true", {"general": True}, {"general": True}, ["general"]),
    ("float", {"general": 1.0}, {"general": 1.0}, ["general"]),
]


@pytest.mark.parametrize(
    ("stored", "expected_models", "expected_unresolved"),
    [row[1:] for row in _SHAPES],
    ids=[row[0] for row in _SHAPES],
)
def test_detail_models_stored_shapes(stored, expected_models, expected_unresolved):
    assert len(_SHAPES) == 14
    world = _build_world(False)
    ids = {
        _USABLE: world.models["m3"],
        _UNUSABLE: world.models["m9"],
        _USABLE_AS_STRING: str(world.models["m3"]),
    }

    def fill(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                k: ids.get(v, v) if isinstance(v, str) else v for k, v in value.items()
            }
        return value

    _set_models(world.agent_id, fill(stored))

    body = _get(world.key, world.agent_id).json()

    assert json.dumps(body["models"], sort_keys=True) == json.dumps(
        fill(expected_models), sort_keys=True
    )
    assert body["unresolved_models"] == expected_unresolved


# ===== the read is bound to slot resolution at run time =====


def _resolved_at_run_time(world: World) -> dict[str, bool]:
    """Which slots of the agent a task started by the holder resolves a model for."""
    db = _direct_db_session()
    try:
        agent = db.get(Agent, world.agent_id)
        llms = load_agent_builder_config(agent, db, world.holder)["llms"]
    finally:
        db.close()
    return {slot: llm is not None for slot, llm in zip(SLOTS, llms)}


@pytest.mark.parametrize("holder_is_owner", [True, False], ids=["H=O", "H!=O"])
@pytest.mark.parametrize("cell", [*_CELL_NAMES, "nonexistent-999999"])
def test_unresolved_models_match_the_runtime_resolution(cell, holder_is_owner):
    world = _build_world(holder_is_owner)
    model_id = 999_999 if cell == "nonexistent-999999" else world.models[cell]
    resolves = cell in _HOLDER_RESOLVES[holder_is_owner]

    for slot in SLOTS:
        _set_models(world.agent_id, {slot: model_id})
        resolved = _resolved_at_run_time(world)

        listed = _get(world.key, world.agent_id).json()["unresolved_models"]

        assert (slot in listed) == (not resolved[slot]), (cell, slot)
        assert listed == ([] if resolves else [slot]), (cell, slot)


@pytest.mark.parametrize(
    "value", [None, "", 0, False], ids=["null", "empty", "0", "false"]
)
def test_unset_slot_values_are_not_listed_and_fall_back(value):
    world = _build_world(False)
    _set_models(world.agent_id, {slot: value for slot in SLOTS})

    # Slot resolution returns nothing for a falsy value without a query.
    assert _resolved_at_run_time(world) == {slot: False for slot in SLOTS}
    assert _get(world.key, world.agent_id).json()["unresolved_models"] == []


def test_a_chat_model_hidden_by_the_hook_is_listed_and_not_resolved():
    """Run time applies the model visibility hook, so the read agrees with it."""
    world = _build_world(False)
    hidden_llm = _add_model("hidden-llm", links=((_user_id("admin"), True, True),))
    world.hidden.append(hidden_llm)
    _set_models(world.agent_id, {"general": hidden_llm})

    assert _resolved_at_run_time(world)["general"] is False
    assert _get(world.key, world.agent_id).json()["unresolved_models"] == ["general"]


def test_a_model_that_fails_to_build_is_not_listed_yet_falls_back():
    """Documents current run-time behavior; it is not asserted as intended.

    The read does not build the model. Slot resolution at run time does, and
    when building fails it logs the error and resolves no model for the
    slot, so the slot is not listed although a task falls back.
    """
    world = _build_world(False)
    _set_models(world.agent_id, {"general": world.models["m3"]})
    assert _resolved_at_run_time(world)["general"] is True

    with patch(
        "xagent.web.services.llm_utils.create_base_llm",
        side_effect=RuntimeError("the provider client cannot be built"),
    ):
        assert _resolved_at_run_time(world)["general"] is False
        assert _get(world.key, world.agent_id).json()["unresolved_models"] == []


def test_an_auto_model_without_candidates_is_not_listed():
    """Known difference: the read checks the Auto row, not its candidates."""
    world = _build_world(False)
    auto = _add_model(
        "auto-router-1",
        provider="router",
        model_name="auto",
        links=((_user_id("admin"), True, True),),
    )
    _set_models(world.agent_id, {"general": auto})

    with pytest.raises(AutoModelUnavailableError):
        _resolved_at_run_time(world)
    assert _get(world.key, world.agent_id).json()["unresolved_models"] == []


# ===== the slot rule has one owner =====


def test_slot_assignable_models_are_the_usable_llm_models():
    world = _build_world(False)
    db = _direct_db_session()
    try:
        usable = {m.id: m.category for m in usable_models(db, world.holder)}
        slotted = {m.id for m in slot_assignable_models(db, world.holder)}
        among = {
            m.id
            for m in usable_models(
                db, world.holder, among_ids=[world.models["m1"], world.models["m9"]]
            )
        }
        none = usable_models(db, world.holder, among_ids=[])
    finally:
        db.close()

    m = world.models
    assert set(usable) == {m["m1"], m["m3"], m["m4"], m["m10"], m["m11"]}
    assert slotted == {m["m1"], m["m3"], m["m4"]}
    assert among == {m["m1"]}
    assert none == []


# ===== reads go to the database =====


@pytest.fixture
def real_cache():
    """A working cache backend: the default one stores nothing."""
    backend = InMemoryTTLCache()
    set_cache_backend_for_testing(backend)
    try:
        yield backend
    finally:
        set_cache_backend_for_testing(None)


def test_detail_reads_the_database_not_the_cache(real_cache):
    key = _personal_key()
    admin_headers = _admin_headers()
    agent_id = _seed_agent(_user_id("admin"), instructions="before")
    web = f"/api/agents/{agent_id}"
    # The web detail read fills the cache for this viewer and agent.
    assert client.get(web, headers=admin_headers).json()["instructions"] == "before"

    db = _direct_db_session()
    try:
        db.get(Agent, agent_id).instructions = "after"
        db.commit()
    finally:
        db.close()
    # The cache really holds the old value, so reading it would be visible.
    assert client.get(web, headers=admin_headers).json()["instructions"] == "before"

    with patch.object(agent_store, "cache_get") as cache_reads:
        resp = _get(key, agent_id)

    assert resp.json()["instructions"] == "after"
    assert cache_reads.call_count == 0


# ===== read failures are errors, not empty answers =====


@pytest.mark.parametrize(
    "failing",
    [
        "xagent.web.services.model_service.rank_name_matched_models",
        "xagent.web.services.agent_store.AgentStore.get_owned_agent",
    ],
    ids=["model-check", "agent-row"],
)
def test_detail_read_failure_is_internal_error(failing):
    world = _build_world(False)
    _set_models(world.agent_id, {"general": world.models["m1"]})
    boom = OperationalError("SELECT secret_table", {}, Exception("secret detail"))

    with patch(failing, side_effect=boom):
        resp = _get(world.key, world.agent_id)

    _error(resp, 500, "internal_error")
    assert "secret" not in resp.text


# ===== the coarse query runs on PostgreSQL =====


def _read_shared_json_model(factory: Callable[[], Any]) -> None:
    """Read an agent whose slots point at a shared model with JSON abilities.

    The model has two link rows, so a join of ``models`` and ``user_models``
    would repeat it; de-duplicating that join needs equality on the JSON
    ``abilities`` column, which PostgreSQL does not define.
    """
    with factory() as db:
        sysadmin = User(username="sysadmin", password_hash="x", is_admin=True)
        holder = User(username="holder", password_hash="x", is_admin=False)
        db.add_all([sysadmin, holder])
        db.flush()
        model = Model(
            model_id="shared-json",
            category="llm",
            model_provider="openai",
            model_name="shared-json",
            api_key="test-key",
            abilities=["chat", "vision"],
            is_active=True,
        )
        db.add(model)
        db.flush()
        db.add_all(
            [
                UserModel(
                    user_id=sysadmin.id,
                    model_id=model.id,
                    is_owner=True,
                    is_shared=True,
                ),
                UserModel(
                    user_id=holder.id,
                    model_id=model.id,
                    is_owner=False,
                    is_shared=False,
                ),
            ]
        )
        agent = Agent(
            user_id=holder.id,
            name="postgres agent",
            models={"general": model.id, "visual": model.id},
        )
        db.add(agent)
        db.commit()
        holder_id, agent_id, model_pk = int(holder.id), int(agent.id), int(model.id)
        with_model = usable_models(db, holder_id)
        assert [row.id for row in with_model] == [model_pk]

    detail = asyncio.run(
        get_agent(
            agent_id=agent_id,
            authed=(
                SimpleNamespace(id=holder_id),
                SimpleNamespace(key_prefix="xag_personal_t"),
            ),
        )
    )

    assert detail.id == agent_id
    assert detail.models == {"general": model_pk, "visual": model_pk}
    assert detail.unresolved_models == []


def test_shared_json_model_is_readable():
    """The same read on SQLite, so the query path is covered without PostgreSQL."""
    from xagent.web.models.database import get_session_local

    _read_shared_json_model(get_session_local())


@pytest.mark.postgresql
def test_shared_json_model_is_readable_on_postgresql(monkeypatch):
    with disposable_database_factory("xagent_v1_agent_detail") as make_database:
        engine = make_database("detail")
        Base.metadata.create_all(bind=engine)
        factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        monkeypatch.setattr(agent_management, "get_session_local", lambda: factory)

        _read_shared_json_model(factory)
