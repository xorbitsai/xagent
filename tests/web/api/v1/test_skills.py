"""Integration tests for the /v1 personal skill endpoints."""

import asyncio
import io
import logging
import zipfile
from types import SimpleNamespace
from urllib.parse import quote

import pytest
from fastapi import HTTPException

from xagent.web.api.v1.errors import _DEFAULT_MESSAGES, V1ErrorCode
from xagent.web.models.skill import UserSkill, UserSkillFile
from xagent.web.models.user import User

from ..conftest import (
    _admin_headers,
    _direct_db_session,
    _register_second_user,
    app_for_tests,
    client,
)
from .test_auth import _create_agent_and_key
from .test_management import _bearer, _personal_key

pytestmark = pytest.mark.usefixtures("_test_db")

SKILL_MD = """---
description: Alpha does alpha things
when_to_use: When alpha is asked for
tags: [one, two]
---
# Alpha
Body text.
"""
OTHER_MD = "# Other\n\n## Description\nA different skill.\n"
# Makes the frontmatter extractor blow the stack, which is what deep YAML does.
UNPARSABLE = "# Alpha\nBOOM-unparsable\n"
SUMMARY_KEYS = {"name", "description", "when_to_use", "tags", "source"}


@pytest.fixture(autouse=True)
def _isolated_skill_library(monkeypatch):
    """Only the shipped skills plus the database layer; no host directories."""
    from xagent.skills import utils as skill_utils
    from xagent.skills.library import set_skill_library_provider
    from xagent.skills.manager import SkillManager

    monkeypatch.delenv("XAGENT_EXTERNAL_SKILLS_LIBRARY_DIRS", raising=False)
    monkeypatch.setattr(
        skill_utils,
        "_get_default_skill_dirs",
        lambda: [SkillManager.get_builtin_root()],
    )
    set_skill_library_provider(None)
    yield
    set_skill_library_provider(None)


@pytest.fixture
def unparsable_marker(monkeypatch):
    from xagent.skills.parser import SkillParser

    real_extract = SkillParser._extract_frontmatter

    def exploding_extract(content):
        if "BOOM-unparsable" in content:
            raise RecursionError("simulated deep-YAML stack exhaustion")
        return real_extract(content)

    monkeypatch.setattr(
        SkillParser, "_extract_frontmatter", staticmethod(exploding_extract)
    )


# ===== helpers =====


def _key_for(headers) -> dict:
    resp = client.post("/api/me/personal-keys", headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _bob_key() -> str:
    _admin_headers()
    return _key_for(_register_second_user("bob"))["full_key"]


def _create(key, name="alpha", skill_md=SKILL_MD):
    return client.post(
        "/v1/skills", headers=_bearer(key), json={"name": name, "skill_md": skill_md}
    )


def _upload(key, filename, data, name=None):
    return client.post(
        "/v1/skills/upload",
        headers=_bearer(key),
        files={"file": (filename, data)},
        data={} if name is None else {"name": name},
    )


def _update(key, name="alpha", skill_md=OTHER_MD):
    return client.put(
        f"/v1/skills/{quote(name, safe='')}",
        headers=_bearer(key),
        json={"skill_md": skill_md},
    )


def _delete(key, name="alpha"):
    return client.delete(f"/v1/skills/{quote(name, safe='')}", headers=_bearer(key))


def _counts() -> tuple[int, int]:
    db = _direct_db_session()
    try:
        return db.query(UserSkill).count(), db.query(UserSkillFile).count()
    finally:
        db.close()


def _stored(name="alpha", path="SKILL.md", user="admin") -> bytes | None:
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == user).one()
        skill = (
            db.query(UserSkill)
            .filter(UserSkill.user_id == owner.id, UserSkill.name == name)
            .first()
        )
        if skill is None:
            return None
        return next((bytes(f.content) for f in skill.files if f.path == path), None)
    finally:
        db.close()


def _seed(name: str, files: dict[str, bytes], user="admin", origin="custom") -> None:
    """Insert a row directly, the way an older release or another writer could."""
    db = _direct_db_session()
    try:
        owner = db.query(User).filter(User.username == user).one()
        skill = UserSkill(user_id=owner.id, name=name, origin=origin)
        db.add(skill)
        db.flush()
        for path, content in files.items():
            db.add(
                UserSkillFile(
                    skill_id=skill.id,
                    path=path,
                    content=content,
                    size_bytes=len(content),
                    sha256="0" * 64,
                )
            )
        db.commit()
    finally:
        db.close()


def _error(resp, status: int, code: str) -> dict:
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error"}, body
    assert body["error"]["code"] == code, body
    return body["error"]


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


# ===== only a live personal key opens these routes =====

_ROUTES = [
    ("GET", "/v1/skills"),
    ("GET", "/v1/skills/{name}"),
    ("POST", "/v1/skills"),
    ("POST", "/v1/skills/upload"),
    ("PUT", "/v1/skills/{name}"),
    ("DELETE", "/v1/skills/{name}"),
    ("GET", "/v1/agents/{agent_id}/skills"),
    ("PUT", "/v1/agents/{agent_id}/skills"),
]
_KEY_KINDS = ["no-key", "agent-key", "revoked-personal-key", "workspace-key"]


def _headers_for(kind: str) -> dict:
    if kind == "no-key":
        return {}
    if kind == "agent-key":
        return _bearer(_create_agent_and_key()[1])
    if kind == "revoked-personal-key":
        _admin_headers()
        holder = _register_second_user("revoked-holder")
        issued = _key_for(holder)
        resp = client.delete(f"/api/me/personal-keys/{issued['id']}", headers=holder)
        assert resp.status_code == 200, resp.text
        return _bearer(issued["full_key"])
    # A well-formed key of a kind this deployment does not issue.
    return _bearer("xag_workspace_" + "a" * 6 + "_" + "b" * 32)


@pytest.mark.parametrize("kind", _KEY_KINDS)
@pytest.mark.parametrize(("method", "path"), _ROUTES)
def test_skill_routes_accept_only_a_live_personal_key(method, path, kind):
    mounted = sorted(
        (verb.upper(), template)
        for template, operations in app_for_tests.openapi()["paths"].items()
        if template == "/v1/skills"
        or template.startswith("/v1/skills/")
        or (template.startswith("/v1/agents/") and template.endswith("/skills"))
        for verb in operations
    )
    assert len(mounted) == 8
    assert mounted == sorted(_ROUTES)

    headers = _headers_for(kind)
    url = path.replace("{name}", "alpha").replace("{agent_id}", "1")
    if path.endswith("/upload"):
        resp = client.request(
            method, url, headers=headers, files={"file": ("a.md", b"x")}
        )
    else:
        resp = client.request(
            method, url, headers=headers, json={"name": "alpha", "skill_md": SKILL_MD}
        )

    _error(resp, 401, "invalid_api_key")


# ===== the listing is the set the holder's tasks run with =====


async def _runtime_skills(username: str = "admin") -> dict[str, dict]:
    """The skills a task run for ``username`` loads, keyed by name."""
    from xagent.skills.utils import create_skill_manager
    from xagent.web.tools.config import WebToolConfig

    db = _direct_db_session()
    try:
        holder = db.query(User).filter(User.username == username).one().id
    finally:
        db.close()
    config = WebToolConfig(db=None, request=SimpleNamespace(), user_id=holder)
    manager = create_skill_manager(context=config.get_skill_scope_context())
    await manager.ensure_initialized()
    return {
        s["name"]: await manager.get_skill(s["name"])
        for s in await manager.list_skills()
    }


def _v1_source(entry: dict) -> str:
    """The ``source`` /v1 has to report for a skill manager entry."""
    if entry["scope"] == "personal":
        return "personal"
    if entry["scope"] == "team":
        return "team"
    if entry["source"] == "builtin":
        return "builtin"
    return "external"


def _assert_listing_is_the_runtime_set(key: str) -> tuple[list[dict], list[str]]:
    """``GET /v1/skills`` lists exactly what a task run for the holder loads,
    with the same ``source`` there and on ``GET /v1/skills/{name}``, and the
    names an agent skill list update checks against are that same set.

    Returns the listing and the body of each listed skill's GET.
    """
    from xagent.web.services.skill_management import SkillManagementRuntime

    listed = client.get("/v1/skills", headers=_bearer(key))
    assert listed.status_code == 200, listed.text
    assert isinstance(listed.json(), list)
    runtime = asyncio.run(_runtime_skills())
    assert {item["name"] for item in listed.json()} == set(runtime)
    holder = client.get("/v1/me", headers=_bearer(key)).json()["user_id"]
    visible = asyncio.run(SkillManagementRuntime().visible_names(user_id=holder))
    assert visible == set(runtime)
    bodies = []
    for item in listed.json():
        expected = _v1_source(runtime[item["name"]])
        assert item["source"] == expected
        viewed = client.get(
            f"/v1/skills/{quote(item['name'], safe='')}", headers=_bearer(key)
        )
        assert viewed.status_code == 200, viewed.text
        assert viewed.json()["source"] == expected
        bodies.append(viewed.text)
    return listed.json(), bodies


def test_listing_is_the_set_the_holder_runs_tasks_with():
    from xagent.skills.library import (
        CompositeSkillLibraryProvider,
        SkillRecord,
        StaticRecordsProvider,
        set_skill_library_provider,
    )
    from xagent.skills.personal_db import XagentPersonalDbSkillProvider

    def record(name, source):
        return SkillRecord(
            name=name,
            source=source,
            scope=source,
            files={"SKILL.md": f"# {name}\n".encode()},
            path=f"/srv/internal/{name}",
        )

    set_skill_library_provider(
        CompositeSkillLibraryProvider(
            [
                StaticRecordsProvider(
                    [
                        record("shared-name", "builtin"),
                        record("plain-builtin", "builtin"),
                        record("team-only", "team"),
                        record("pdf.tools", "external"),
                    ]
                ),
                XagentPersonalDbSkillProvider(),
            ]
        )
    )
    key = _personal_key()
    assert _create(key, "shared-name").status_code == 200
    assert _create(key, "mine-only").status_code == 200

    listed, bodies = _assert_listing_is_the_runtime_set(key)

    assert [(item["name"], item["source"]) for item in listed] == [
        ("mine-only", "personal"),
        ("shared-name", "personal"),
        ("pdf.tools", "external"),
        ("plain-builtin", "builtin"),
        ("team-only", "team"),
    ]
    assert not [body for body in bodies if "/srv/internal" in body]
    _error(
        client.get("/v1/skills/missing", headers=_bearer(key)), 404, "skill_not_found"
    )


def test_listing_is_the_runtime_set_on_the_default_provider_chain(
    monkeypatch, tmp_path
):
    """No provider registered, as in a deployment that registers none: the
    filesystem roots first, then the holder's personal skills."""
    from xagent.skills import utils as skill_utils
    from xagent.skills.library import get_skill_library_provider
    from xagent.skills.manager import SkillManager

    assert get_skill_library_provider() is None
    host_root = tmp_path / "host-skills"
    (host_root / "host-only").mkdir(parents=True)
    (host_root / "host-only" / "SKILL.md").write_text("# Host only\n\nOn disk.\n")
    monkeypatch.setattr(
        skill_utils,
        "_get_default_skill_dirs",
        lambda: [SkillManager.get_builtin_root(), host_root],
    )
    key = _personal_key()
    builtins = sorted(
        name
        for name, entry in asyncio.run(_runtime_skills()).items()
        if entry["source"] == "builtin"
    )
    assert builtins, "the shipped skills are part of the default chain"
    shadowed = builtins[0]
    assert _create(key, shadowed).status_code == 200
    assert _create(key, "mine-only").status_code == 200

    listed, _ = _assert_listing_is_the_runtime_set(key)

    sources = {item["name"]: item["source"] for item in listed}
    assert sources[shadowed] == "personal"
    assert sources["mine-only"] == "personal"
    assert sources["host-only"] == "external"
    assert {name for name, source in sources.items() if source == "builtin"} == (
        set(builtins) - {shadowed}
    )


# ===== writes only touch the holder's own personal skills =====


@pytest.mark.parametrize(
    "bob_first", [False, True], ids=["alice-row-first", "bob-row-first"]
)
@pytest.mark.parametrize("operation", ["update", "delete"])
def test_writes_only_touch_the_holders_own_personal_skills(operation, bob_first):
    # Both row orders: a query that forgot the owner would hit whichever row
    # comes first, so a single order can hide it.
    alice, bob = _personal_key(), _bob_key()
    for key in (bob, alice) if bob_first else (alice, bob):
        assert _create(key, "shared", SKILL_MD).status_code == 200
    builtin = next(
        item["name"]
        for item in client.get("/v1/skills", headers=_bearer(alice)).json()
        if item["source"] == "builtin"
    )
    act = _update if operation == "update" else _delete

    assert act(alice, "shared").status_code == 200

    if operation == "update":
        assert _stored("shared", user="admin") == OTHER_MD.encode()
    else:
        assert _stored("shared", user="admin") is None
    assert _stored("shared", user="bob") == SKILL_MD.encode(), "Bob's row is untouched"
    _error(act(alice, builtin), 404, "skill_not_found")
    assert (
        client.get(f"/v1/skills/{builtin}", headers=_bearer(alice)).status_code == 200
    )


def test_an_invalid_edit_is_refused_before_the_name_is_looked_up():
    key = _personal_key()
    builtin = next(
        item["name"]
        for item in client.get("/v1/skills", headers=_bearer(key)).json()
        if item["source"] == "builtin"
    )
    before = _counts()

    error = _error(_update(key, builtin, "  \n"), 400, "invalid_input")

    assert error["message"] == "SKILL.md is empty."
    assert _counts() == before
    assert (
        client.get(f"/v1/skills/{builtin}", headers=_bearer(key)).json()["source"]
        == "builtin"
    )


# ===== every failure is a /v1 envelope =====


def _trigger(case, key, monkeypatch):
    if case == "400":
        return _create(key, "blank", "   ")
    if case == "404":
        return _update(key, "missing")
    if case == "409":
        assert _create(key, "dup", SKILL_MD).status_code == 200
        return _create(key, "dup", OTHER_MD)
    if case == "413":
        monkeypatch.setenv("XAGENT_MAX_UPLOAD_SIZE", "1K")
        return _upload(key, "big.md", b"a" * 2000, name="big")
    if case == "other":
        from xagent.web.api import skill_hub

        def refuse(**_kwargs):
            raise HTTPException(status_code=502, detail="upstream detail: secret")

        monkeypatch.setattr(skill_hub, "_write_personal_skill", refuse)
        return _create(key, "boom")
    # A multipart body with no boundary cannot be parsed at all; the framework
    # answers before any route code runs.
    return client.post(
        "/v1/skills/upload",
        headers={**_bearer(key), "content-type": "multipart/form-data"},
        content=b"--xyz\r\nbroken",
    )


_TRANSLATIONS = [
    ("400", 400, "invalid_input", "SKILL.md is empty."),
    ("404", 404, "skill_not_found", "Skill not found."),
    (
        "409",
        409,
        "skill_already_exists",
        "A personal skill named 'dup' already exists.",
    ),
    ("413", 413, "invalid_input", "Upload exceeds 1 KiB limit."),
    ("other", 500, "internal_error", "Internal server error."),
]


@pytest.mark.parametrize(
    "case", [row[0] for row in _TRANSLATIONS] + ["malformed-multipart"]
)
def test_every_failure_is_a_v1_envelope(case, monkeypatch, caplog):
    assert len(_TRANSLATIONS) == 5
    key = _personal_key()

    resp = _trigger(case, key, monkeypatch)

    if case == "malformed-multipart":
        # Known exception: the framework answers before any /v1 code runs.
        assert resp.status_code == 400
        assert set(resp.json()) == {"detail"}
        return
    _, status, code, message = next(row for row in _TRANSLATIONS if row[0] == case)
    error = _error(resp, status, code)
    assert error["message"] == message
    if case == "other":
        assert "secret" not in resp.text
        assert any(
            r.levelno == logging.ERROR and "secret" in r.getMessage()
            for r in caplog.records
        )


# ===== a holder deleted after authentication is refused, with no write =====


@pytest.mark.parametrize("operation", ["create", "upload", "update", "delete"])
def test_a_holder_deleted_after_authentication_is_refused_without_a_write(operation):
    from xagent.web.api.v1.deps import (
        PersonalApiKeySnapshot,
        UserPrincipalSnapshot,
        get_user_from_personal_key,
    )

    ghost = (
        UserPrincipalSnapshot(id=999_999, username="ghost", email=None, is_admin=False),
        PersonalApiKeySnapshot(key_prefix="ghost0"),
    )
    app_for_tests.dependency_overrides[get_user_from_personal_key] = lambda: ghost
    try:
        send = {
            "create": lambda: _create("unused"),
            "upload": lambda: _upload("unused", "alpha.md", SKILL_MD.encode(), "alpha"),
            "update": lambda: _update("unused"),
            "delete": lambda: _delete("unused"),
        }[operation]
        resp = send()
    finally:
        del app_for_tests.dependency_overrides[get_user_from_personal_key]

    _error(resp, 401, "invalid_api_key")
    assert _counts() == (0, 0)


# ===== a refused create or upload writes nothing =====


@pytest.mark.parametrize(
    ("surface", "name", "payload", "status", "setting"),
    [
        ("create", "bad name!", SKILL_MD, 422, None),
        ("create", "ok", "  \n", 400, None),
        ("create", "ok", UNPARSABLE, 400, None),
        ("create", "ok", "a" * 200_001, 422, None),
        ("create", "taken", OTHER_MD, 409, None),
        ("upload", "bad name!", SKILL_MD.encode(), 400, None),
        ("upload", "ok", b"  \n", 400, None),
        ("upload", "ok", b"\xff\xfe not utf-8", 400, None),
        ("upload", "ok", b"a" * 200_001, 400, None),
        ("upload", "ok", b"a" * 2000, 413, "1K"),
        ("upload", "taken", OTHER_MD.encode(), 409, None),
    ],
)
def test_refused_writes_leave_no_rows(
    surface, name, payload, status, setting, monkeypatch, unparsable_marker
):
    key = _personal_key()
    assert _create(key, "taken", SKILL_MD).status_code == 200
    before = _counts()
    if setting:
        monkeypatch.setenv("XAGENT_MAX_UPLOAD_SIZE", setting)

    if surface == "create":
        resp = _create(key, name, payload)
    else:
        resp = _upload(key, "skill.md", payload, name=name)

    assert resp.status_code == status, resp.text
    assert "error" in resp.json()
    assert _counts() == before


@pytest.mark.parametrize("filename", ["notes.txt", "broken.zip"])
def test_unsupported_and_unreadable_uploads_write_nothing(filename):
    key = _personal_key()
    resp = _upload(key, filename, b"not a skill", name="ok")
    _error(resp, 400, "invalid_input")
    assert _counts() == (0, 0)


# ===== a same-bytes retry is a success, never a second row =====


@pytest.mark.parametrize(
    ("first", "second", "status"),
    [
        ("create", "create", 200),
        ("upload", "upload", 200),
        ("upload", "create", 409),
        ("create", "upload", 409),
    ],
)
def test_same_bytes_retry_is_idempotent_within_one_origin(first, second, status):
    key = _personal_key()

    def send(surface, text):
        if surface == "create":
            return _create(key, "alpha", text)
        return _upload(key, "alpha.md", text.encode(), name="alpha")

    assert send(first, SKILL_MD).status_code == 200
    assert _counts() == (1, 1)

    replay = send(second, SKILL_MD)

    assert replay.status_code == status, replay.text
    assert _counts() == (1, 1)
    if first == second:
        assert replay.json()["name"] == "alpha"
        _error(send(first, OTHER_MD), 409, "skill_already_exists")
        assert _stored() == SKILL_MD.encode()


# ===== an edit is validated before it is stored =====


@pytest.mark.parametrize(
    ("kind", "skill_md", "status"),
    [
        ("blank", "  \n\t ", 400),
        ("zero-width", "​ ﻿⁠", 400),
        ("unparsable", UNPARSABLE, 400),
        ("legacy-attachment", OTHER_MD, 200),
    ],
)
def test_edit_is_validated_before_it_is_stored(
    kind, skill_md, status, unparsable_marker
):
    key = _personal_key()
    if kind == "legacy-attachment":
        # Stored before attachments were bounded; a fresh write would refuse it.
        _seed("alpha", {"SKILL.md": SKILL_MD.encode(), "template.md": b"a" * 900_000})
    else:
        assert _create(key).status_code == 200

    resp = _update(key, "alpha", skill_md)

    assert resp.status_code == status, resp.text
    if status == 200:
        assert _stored() == skill_md.encode()
        assert _stored(path="template.md") == b"a" * 900_000
    else:
        assert _error(resp, 400, "invalid_input")
        assert _stored() == SKILL_MD.encode()


# ===== no internal paths; write responses agree with the next read =====


def test_responses_carry_no_internal_paths_and_writes_agree_with_reads():
    from xagent.skills.manager import SkillManager

    key = _personal_key()
    assert _create(key, "edited", OTHER_MD).status_code == 200
    writes = {
        "created": _create(key, "created", SKILL_MD),
        "uploaded": _upload(
            key,
            "bundle.zip",
            _zip({"zipped/SKILL.md": SKILL_MD.encode(), "zipped/notes.md": b"n"}),
            name="",  # an empty name means "derive it", as on the web route
        ),
        "edited": _update(key, "edited", SKILL_MD),
    }
    expected_names = {"created": "created", "uploaded": "zipped", "edited": "edited"}

    for label, resp in writes.items():
        assert resp.status_code == 200, resp.text
        written = resp.json()
        assert set(written) == SUMMARY_KEYS
        assert written["name"] == expected_names[label]
        assert written["source"] == "personal"
        read = client.get(f"/v1/skills/{written['name']}", headers=_bearer(key)).json()
        assert {k: read[k] for k in SUMMARY_KEYS} == written
    assert client.get("/v1/skills/zipped", headers=_bearer(key)).json()["files"] == [
        "SKILL.md",
        "notes.md",
    ]

    from xagent.web.schemas.v1 import V1SkillDetail

    listed = client.get("/v1/skills", headers=_bearer(key)).json()
    assert {item["source"] for item in listed} == {"personal", "builtin"}
    for item in listed:
        detail = client.get(f"/v1/skills/{item['name']}", headers=_bearer(key))
        assert set(detail.json()) == set(V1SkillDetail.model_fields)
        assert "db://" not in detail.text
        assert str(SkillManager.get_builtin_root()) not in detail.text


# ===== a failing provider is a 500, never an empty list =====


@pytest.mark.parametrize("path", ["/v1/skills", "/v1/skills/alpha"])
def test_provider_failure_is_not_folded_into_an_empty_result(path):
    from xagent.skills.library import set_skill_library_provider

    class Failing:
        async def list_records(self, context):
            raise RuntimeError("provider down")

    key = _personal_key()
    set_skill_library_provider(Failing())

    _error(client.get(path, headers=_bearer(key)), 500, "internal_error")


# ===== every error code has default text =====


def test_every_error_code_has_a_default_message():
    assert set(_DEFAULT_MESSAGES) == set(V1ErrorCode)


# ===== a name must match in full =====

_NAMES = ["abc\n", "abc\r", " abc", "a" * 101, "pdf.tools"]
_OPERATIONS = ["create", "upload", "update", "delete"]


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("name", _NAMES)
def test_names_that_do_not_match_in_full_are_refused(name, operation):
    assert len(_NAMES) == 5 and len(_OPERATIONS) == 4
    key = _personal_key()
    if operation in ("update", "delete"):
        # A row that already carries such a name must stay out of reach.
        _seed(name, {"SKILL.md": SKILL_MD.encode()})
    before = _counts()

    if operation == "create":
        resp = _create(key, name)
        _error(resp, 422, "invalid_input")
    elif operation == "upload":
        resp = _upload(key, "skill.md", SKILL_MD.encode(), name=name)
        _error(resp, 400, "invalid_input")
    elif operation == "update":
        _error(_update(key, name), 404, "skill_not_found")
        assert _stored(name) == SKILL_MD.encode()
    else:
        _error(_delete(key, name), 404, "skill_not_found")

    assert _counts() == before


# ===== each successful write leaves one log line =====


@pytest.mark.parametrize(
    "action", ["create", "create-replay", "upload", "upload-replay", "update", "delete"]
)
def test_each_successful_write_logs_one_line(action, caplog):
    _admin_headers()
    issued = _key_for(_register_second_user("logger"))
    key = issued["full_key"]
    db = _direct_db_session()
    try:
        holder_id = db.query(User).filter(User.username == "logger").one().id
    finally:
        db.close()
    secret = "SECRET-BODY-TEXT"
    body = SKILL_MD + secret
    base = action.split("-")[0]
    if base != "create" and base != "upload":
        assert _create(key, "alpha", body).status_code == 200
    send = {
        "create": lambda: _create(key, "alpha", body),
        "upload": lambda: _upload(key, "alpha.md", body.encode(), name="alpha"),
        "update": lambda: _update(key, "alpha", body + "!"),
        "delete": lambda: _delete(key, "alpha"),
    }[base]
    if action.endswith("replay"):
        assert send().status_code == 200
    caplog.clear()

    with caplog.at_level(logging.INFO):
        assert send().status_code == 200

    lines = [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("personal key skill write:")
    ]
    assert len(lines) == 1, lines
    assert f"key_prefix={issued['key_prefix']}" in lines[0]
    assert f"user_id={holder_id}" in lines[0]
    assert f"action={base}" in lines[0] and "name='alpha'" in lines[0]
    assert secret not in "\n".join(r.getMessage() for r in caplog.records)


def test_a_refused_write_logs_no_write_line(caplog):
    _admin_headers()
    key = _key_for(_register_second_user("refused"))["full_key"]
    assert _create(key, "alpha", SKILL_MD).status_code == 200
    caplog.clear()

    with caplog.at_level(logging.INFO):
        assert _create(key, "alpha", SKILL_MD + "different").status_code == 409

    assert not [
        r
        for r in caplog.records
        if r.getMessage().startswith("personal key skill write:")
    ]
