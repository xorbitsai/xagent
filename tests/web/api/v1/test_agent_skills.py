"""Integration tests for reading and replacing an agent's skill list over /v1."""

import logging

import pytest

from xagent.web.models.agent import Agent, AgentOrigin
from xagent.web.models.user import User
from xagent.web.services.agent_team_scope import (
    AgentTeamScope,
    set_agent_team_scope_hook,
)
from xagent.web.services.skill_management import SkillManagementRuntime

from ..conftest import _admin_headers, _direct_db_session, _register_second_user, client
from .test_management import _bearer, _personal_key
from .test_skills import (  # noqa: F401  (_isolated_skill_library is an autouse fixture)
    _NAMES,
    _create,
    _error,
    _isolated_skill_library,
    _key_for,
)

pytestmark = pytest.mark.usefixtures("_test_db")


@pytest.fixture
def team_scopes():
    """Team scopes by user id, as an application that groups users installs."""
    scopes: dict[int, AgentTeamScope] = {}
    set_agent_team_scope_hook(lambda _db, user_id: scopes.get(int(user_id)))
    yield scopes
    set_agent_team_scope_hook(None)


# ===== helpers =====


def _user_id(username: str = "admin") -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _seed_agent(
    owner: str = "admin",
    skills=(),
    *,
    team_id: int | None = None,
    visibility: str = "team",
    origin: str = AgentOrigin.USER.value,
) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(owner),
            name=f"agent {db.query(Agent).count()}",
            skills=list(skills),
            team_id=team_id,
            visibility=visibility,
            origin=origin,
        )
        db.add(agent)
        db.commit()
        return int(agent.id)
    finally:
        db.close()


def _stored_skills(agent_id: int) -> list[str]:
    db = _direct_db_session()
    try:
        return list(db.get(Agent, agent_id).skills or [])
    finally:
        db.close()


def _set_visibility(agent_id: int, visibility: str) -> None:
    db = _direct_db_session()
    try:
        db.get(Agent, agent_id).visibility = visibility
        db.commit()
    finally:
        db.close()


def _get(key: str, agent_id: int):
    return client.get(f"/v1/agents/{agent_id}/skills", headers=_bearer(key))


def _put(key: str, agent_id: int, skills):
    if isinstance(skills, bytes):
        return client.put(
            f"/v1/agents/{agent_id}/skills",
            headers={**_bearer(key), "content-type": "application/json"},
            content=skills,
        )
    return client.put(
        f"/v1/agents/{agent_id}/skills", headers=_bearer(key), json={"skills": skills}
    )


def _count_visible_name_reads(monkeypatch) -> list[int]:
    reads: list[int] = []
    real = SkillManagementRuntime.visible_names

    async def counted(self, *, user_id):
        reads.append(user_id)
        return await real(self, user_id=user_id)

    monkeypatch.setattr(SkillManagementRuntime, "visible_names", counted)
    return reads


# ===== the list is reachable exactly where the agent is =====

# (cell, the caller's role in team 1, how the agent is stored, reachable)
_ACCESS = [
    ("own", "member", {"owner": "admin"}, True),
    ("team-visible", "member", {"owner": "bob", "team_id": 1}, True),
    (
        "admins-only-to-a-member",
        "member",
        {"owner": "bob", "team_id": 1, "visibility": "admins"},
        False,
    ),
    (
        "admins-only-to-an-admin",
        "admin",
        {"owner": "bob", "team_id": 1, "visibility": "admins"},
        True,
    ),
    ("other-team", "admin", {"owner": "bob", "team_id": 2}, False),
    (
        "workforce-manager",
        "member",
        {"owner": "admin", "origin": AgentOrigin.WORKFORCE_GENERATED_MANAGER.value},
        False,
    ),
    ("nonexistent", "member", None, False),
]


@pytest.mark.parametrize("method", ["GET", "PUT"])
@pytest.mark.parametrize(
    ("cell", "role", "stored_as", "reachable"), _ACCESS, ids=[c[0] for c in _ACCESS]
)
def test_the_list_is_reachable_exactly_where_the_agent_is(
    cell, role, stored_as, reachable, method, team_scopes, monkeypatch
):
    assert len(_ACCESS) == 7
    key = _personal_key()
    _register_second_user("bob")
    team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=role == "admin")
    agent_id = (
        999_999 if stored_as is None else _seed_agent(skills=["seeded"], **stored_as)
    )
    reads = _count_visible_name_reads(monkeypatch)

    resp = _get(key, agent_id) if method == "GET" else _put(key, agent_id, ["added"])

    if not reachable:
        _error(resp, 404, "agent_not_found")
        assert reads == [], "access is refused before any skill is read"
        if stored_as is not None:
            assert _stored_skills(agent_id) == ["seeded"]
        return
    assert resp.status_code == 200, resp.text
    if method == "GET":
        assert resp.json() == {"agent_id": agent_id, "skills": ["seeded"]}
    else:
        assert _stored_skills(agent_id) == ["added"]


def test_access_lost_before_the_locked_write_refuses_it(team_scopes, monkeypatch):
    key = _personal_key()
    _register_second_user("bob")
    team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=False)
    agent_id = _seed_agent("bob", ["seeded"], team_id=1)
    real = SkillManagementRuntime.visible_names

    async def restricted_meanwhile(self, *, user_id):
        names = await real(self, user_id=user_id)
        _set_visibility(agent_id, "admins")
        return names

    monkeypatch.setattr(SkillManagementRuntime, "visible_names", restricted_meanwhile)

    _error(_put(key, agent_id, ["added"]), 404, "agent_not_found")
    assert _stored_skills(agent_id) == ["seeded"]


def test_get_refuses_an_agent_id_beyond_the_id_column():
    _error(_get(_personal_key(), 2**31), 422, "invalid_input")


# ===== only added names are checked, and only against the caller's skills =====

_BAD = ["bad name!", "a\tb", "x" * 300, "four?", "five.", "six six"]
# At most five names, each cut to 100 characters of its repr.
_BAD_MESSAGE = (
    "These names are neither valid personal skill names nor the names of "
    "skills this key can see: 'bad name!', 'a\\tb', '" + "x" * 99 + ", 'four?', "
    "'five.'"
)

# (row, stored before, request, status, stored after, unresolved_for_caller)
_PUT_ROWS = [
    ("holder-has-it", [], ["mine"], 200, ["mine"], []),
    ("compliant-but-unseen", [], ["unseen"], 200, ["unseen"], ["unseen"]),
    ("visible-external", [], ["pdf.tools"], 200, ["pdf.tools"], []),
    ("noncompliant-and-unseen", ["mine"], ["mine", *_BAD], 400, ["mine"], None),
    (
        "keeps-an-unseen-name",
        ["unseen"],
        ["unseen", "mine"],
        200,
        ["unseen", "mine"],
        [],
    ),
    ("keeps-a-noncompliant-name", ["old name!"], ["old name!"], 200, ["old name!"], []),
    (
        "team-admin-adds-a-members-skill",
        [],
        ["bobs-skill"],
        200,
        ["bobs-skill"],
        ["bobs-skill"],
    ),
    ("over-100-items", ["mine"], [f"s{i}" for i in range(101)], 422, ["mine"], None),
    ("item-over-1000-chars", ["mine"], ["a" * 1001], 422, ["mine"], None),
    ("lone-surrogate", ["mine"], b'{"skills": ["\\ud800"]}', 422, ["mine"], None),
    ("agent-id-over-int32", ["mine"], ["unseen"], 422, ["mine"], None),
    ("empty", ["mine"], [], 200, [], []),
    (
        "duplicates",
        [],
        ["unseen", "mine", "unseen"],
        200,
        ["unseen", "mine"],
        ["unseen"],
    ),
]


@pytest.mark.parametrize(
    ("row", "before", "request_skills", "status", "after", "unresolved"),
    _PUT_ROWS,
    ids=[r[0] for r in _PUT_ROWS],
)
def test_put_checks_and_reports_only_the_names_it_adds(
    row, before, request_skills, status, after, unresolved, team_scopes
):
    from xagent.skills.library import (
        CompositeSkillLibraryProvider,
        SkillRecord,
        StaticRecordsProvider,
        set_skill_library_provider,
    )
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider

    assert len(_PUT_ROWS) == 13
    external = SkillRecord(
        name="pdf.tools", source="external", files={"SKILL.md": b"# PDF tools\n"}
    )
    set_skill_library_provider(
        CompositeSkillLibraryProvider(
            [StaticRecordsProvider([external]), XagentPersonalDbSkillProvider()]
        )
    )
    key = _personal_key()
    assert _create(key, "mine").status_code == 200
    if row == "team-admin-adds-a-members-skill":
        bob = _key_for(_register_second_user("bob"))["full_key"]
        assert _create(bob, "bobs-skill").status_code == 200
        team_scopes[_user_id()] = AgentTeamScope(team_id=1, is_team_admin=True)
        agent_id = _seed_agent("bob", before, team_id=1)
    else:
        agent_id = _seed_agent("admin", before)

    resp = _put(
        key, 2**31 if row == "agent-id-over-int32" else agent_id, request_skills
    )

    if status == 200:
        assert resp.status_code == 200, resp.text
        assert resp.json() == {
            "agent_id": agent_id,
            "skills": after,
            "unresolved_for_caller": unresolved,
        }
    else:
        error = _error(resp, status, "invalid_input")
        if status == 400:
            assert error["message"] == _BAD_MESSAGE
    assert _stored_skills(agent_id) == after


@pytest.mark.parametrize("name", _NAMES)
def test_an_added_name_must_match_in_full(name):
    assert len(_NAMES) == 5
    key = _personal_key()
    agent_id = _seed_agent("admin", ["kept"])

    _error(_put(key, agent_id, ["kept", name]), 400, "invalid_input")

    assert _stored_skills(agent_id) == ["kept"]


# ===== a failing skill read refuses the write =====


def test_a_failing_skill_read_refuses_the_put_without_writing():
    from xagent.skills.library import set_skill_library_provider

    class Failing:
        async def list_records(self, context):
            raise RuntimeError("provider down")

    key = _personal_key()
    agent_id = _seed_agent("admin", ["kept"])
    set_skill_library_provider(Failing())

    _error(_put(key, agent_id, ["kept", "added"]), 500, "internal_error")
    assert _stored_skills(agent_id) == ["kept"]


# ===== the write holds this agent's row =====


def test_the_write_locks_only_this_agents_row(monkeypatch):
    from sqlalchemy.orm import Query

    key = _personal_key()
    _register_second_user("bob")
    target = _seed_agent("admin", ["kept"])
    # Rows a lock without the id condition would also select.
    _seed_agent("admin", ["kept"])
    _seed_agent("bob", ["kept"])
    locked: list[list[int]] = []
    real_with_for_update = Query.with_for_update

    def spy(self, *args, **kwargs):
        # The rows this query selects, read with the same filter.
        locked.append(sorted(int(row.id) for row in self.all()))
        return real_with_for_update(self, *args, **kwargs)

    monkeypatch.setattr(Query, "with_for_update", spy)

    assert _put(key, target, ["kept", "added"]).status_code == 200

    assert locked == [[target]]
    assert _stored_skills(target) == ["kept", "added"]


# ===== a successful PUT leaves one log line =====


@pytest.mark.parametrize("outcome", ["written", "refused"])
def test_a_successful_put_logs_one_line(outcome, caplog):
    _admin_headers()
    issued = _key_for(_register_second_user("logger"))
    agent_id = _seed_agent("logger", ["kept"])
    added = ["secret-one", "secret-two"] if outcome == "written" else ["secret one!"]
    caplog.clear()

    with caplog.at_level(logging.INFO):
        resp = _put(issued["full_key"], agent_id, ["kept", *added])

    assert resp.status_code == (200 if outcome == "written" else 400), resp.text
    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("personal key agent skills write:")
    ]
    if outcome == "refused":
        assert lines == []
        return
    assert lines == [
        f"personal key agent skills write: key_prefix={issued['key_prefix']} "
        f"user_id={_user_id('logger')} agent_id={agent_id} added=2"
    ]
    assert "secret" not in "\n".join(r.getMessage() for r in caplog.records)
