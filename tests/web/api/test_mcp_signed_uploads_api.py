"""Create, update and echo of a connector's signed-upload settings over HTTP."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.mcp import MCPServer, UserMCPServer
from xagent.web.models.user import User

from .conftest import _admin_headers, _direct_db_session, _register_second_user, client

pytestmark = pytest.mark.usefixtures("_test_db")

URL = "https://mcp.example.test/mcp"
NAME = "upload-connector"
BAD = {"tool_name": "attach_document"}
SIGNED_UPLOADS = [
    {
        "tool_name": "attach_document",
        "file_argument": "document_file_id",
        "issuing_tool_name": "request_upload",
        "allowed_upload_hosts": ["uploads.example.test"],
        "upload_path_prefix": "/staged/",
        "content_types": {".pdf": "application/pdf"},
    }
]
TRANSPORT_DETAIL = (
    "Invalid configuration: signed_uploads are supported only for the sse and "
    "streamable_http transports; to use another transport, set signed_uploads "
    "to [] in the same request"
)
# The errors signed_upload_settings_error gives for the values used below,
# written out so a change of wording shows up here.
MISSING_KEYS_ERROR = "signed_uploads[0] must have exactly the documented keys"
NOT_A_LIST_ERROR = "signed_uploads must be a list"
NO_HOST_ERROR = "the connection URL has no hostname"
OWNER_ONLY_DETAIL = "Only the server owner can change the shared configuration"


def _without_content_types() -> list[dict[str, Any]]:
    entry = copy.deepcopy(SIGNED_UPLOADS[0])
    del entry["content_types"]
    return [entry]


def _invalid_detail(error: str) -> str:
    return "Invalid configuration: " + error


def _user_id(username: str) -> int:
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == username).one().id)
    finally:
        db.close()


def _create(headers: dict[str, str], **fields: Any):
    body: dict[str, Any] = {
        "name": NAME,
        "transport": "streamable_http",
        "description": "Uploads",
        "config": {"url": URL},
    }
    body.update(fields)
    return client.post("/api/mcp/servers", json=body, headers=headers)


def _create_ok(headers: dict[str, str], **fields: Any) -> int:
    resp = _create(headers, **fields)
    assert resp.status_code == 201, resp.text
    return int(resp.json()["id"])


def _row(server_id: int) -> dict[str, Any]:
    from sqlalchemy import text

    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        is_null = db.execute(
            text("SELECT signed_uploads IS NULL FROM mcp_servers WHERE id = :id"),
            {"id": server_id},
        ).scalar_one()
        return {
            "name": server.name,
            "transport": server.transport,
            "url": server.url,
            "description": server.description,
            "signed_uploads": server.signed_uploads,
            "signed_uploads_is_null": bool(is_null),
        }
    finally:
        db.close()


def _store(server_id: int, value: Any) -> None:
    db = _direct_db_session()
    try:
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()
        server.signed_uploads = value
        db.commit()
    finally:
        db.close()


def _link_non_owner(username: str, server_id: int) -> None:
    db = _direct_db_session()
    try:
        db.add(
            UserMCPServer(
                user_id=_user_id(username),
                mcpserver_id=server_id,
                is_owner=False,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


def _assert_names_signed_uploads(resp) -> None:
    assert any("signed_uploads" in err["loc"] for err in resp.json()["detail"])


def _server_named(items: list[dict[str, Any]], server_id: int) -> dict[str, Any]:
    matches = [
        item
        for item in items
        if item["id"] == server_id and item["transport"] != "custom_api"
    ]
    assert len(matches) == 1
    return matches[0]


_REJECT_CASES = [
    "create-not-a-list",
    "create-entry-not-an-object",
    "create-missing-key",
    "create-stdio",
    "create-websocket",
    "create-url-without-host",
    "update-not-a-list",
    "update-entry-not-an-object",
    "update-missing-key",
    "update-transport-to-stdio",
    "update-transport-to-websocket",
    "update-url-without-host",
]


def _reject_on_create(case: str, owner: dict[str, str]) -> None:
    if case == "create-not-a-list":
        resp = _create(owner, signed_uploads={"tool_name": "attach_document"})
        assert resp.status_code == 422, resp.text
        _assert_names_signed_uploads(resp)
    elif case == "create-entry-not-an-object":
        resp = _create(owner, signed_uploads=["attach_document"])
        assert resp.status_code == 422, resp.text
        _assert_names_signed_uploads(resp)
    elif case == "create-missing-key":
        raw = _without_content_types()
        resp = _create(owner, signed_uploads=raw)
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == (
            "Invalid configuration: "
            "signed_uploads[0] must have exactly the documented keys"
        )
    elif case == "create-stdio":
        resp = _create(
            owner,
            transport="stdio",
            config={"command": "python"},
            signed_uploads=SIGNED_UPLOADS,
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == TRANSPORT_DETAIL
    elif case == "create-websocket":
        resp = _create(
            owner,
            transport="websocket",
            config={"url": "wss://mcp.example.test/ws"},
            signed_uploads=SIGNED_UPLOADS,
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == TRANSPORT_DETAIL
    else:
        no_host = "https:///mcp"
        resp = _create(owner, config={"url": no_host}, signed_uploads=SIGNED_UPLOADS)
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == (
            "Invalid configuration: the connection URL has no hostname"
        )

    db = _direct_db_session()
    try:
        assert db.query(MCPServer).filter(MCPServer.name == NAME).count() == 0
    finally:
        db.close()


def _reject_on_update(case: str, owner: dict[str, str]) -> None:
    server_id = _create_ok(owner, signed_uploads=SIGNED_UPLOADS)
    before = _row(server_id)
    assert before["signed_uploads"] == SIGNED_UPLOADS

    expected_status = 400
    expected_detail: str | None
    if case == "update-not-a-list":
        payload: dict[str, Any] = {"signed_uploads": {"tool_name": "attach_document"}}
        expected_status, expected_detail = 422, None
    elif case == "update-entry-not-an-object":
        payload = {"signed_uploads": ["attach_document"]}
        expected_status, expected_detail = 422, None
    elif case == "update-missing-key":
        raw = _without_content_types()
        payload = {"signed_uploads": raw}
        expected_detail = _invalid_detail(MISSING_KEYS_ERROR)
    elif case == "update-transport-to-stdio":
        payload = {"transport": "stdio", "config": {"command": "python"}}
        expected_detail = TRANSPORT_DETAIL
    elif case == "update-transport-to-websocket":
        payload = {"transport": "websocket"}
        expected_detail = TRANSPORT_DETAIL
    else:
        no_host = "https:///mcp"
        payload = {"config": {"url": no_host}}
        expected_detail = _invalid_detail(NO_HOST_ERROR)

    resp = client.put(f"/api/mcp/servers/{server_id}", json=payload, headers=owner)
    assert resp.status_code == expected_status, resp.text
    if expected_status == 422:
        _assert_names_signed_uploads(resp)
    else:
        assert resp.json()["detail"] == expected_detail
    assert _row(server_id) == before


@pytest.mark.parametrize("case", _REJECT_CASES)
def test_save_rejects_invalid_signed_uploads(case: str) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    if case.startswith("create-"):
        _reject_on_create(case, owner)
    else:
        _reject_on_update(case, owner)


@pytest.mark.parametrize(
    "case",
    [
        "create-valid",
        "create-valid-sse",
        "create-omitted",
        "update-to-stdio-clearing",
        "update-clear-to-null",
    ],
)
def test_save_accepts_signed_uploads(case: str) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")

    if case in ("create-valid", "create-valid-sse"):
        transport = "sse" if case == "create-valid-sse" else "streamable_http"
        resp = _create(owner, transport=transport, signed_uploads=SIGNED_UPLOADS)
        assert resp.status_code == 201, resp.text
        body = resp.json()
        assert body["transport"] == transport
        assert body["signed_uploads"] == SIGNED_UPLOADS
        assert body["signed_uploads_error"] is None
        assert _row(body["id"])["signed_uploads"] == SIGNED_UPLOADS
    elif case == "create-omitted":
        server_id = _create_ok(owner)
        assert _row(server_id)["signed_uploads_is_null"] is True
    elif case == "update-to-stdio-clearing":
        server_id = _create_ok(owner, signed_uploads=SIGNED_UPLOADS)
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={
                "transport": "stdio",
                "config": {"command": "python"},
                "signed_uploads": [],
            },
            headers=owner,
        )
        assert resp.status_code == 200, resp.text
        row = _row(server_id)
        assert row["transport"] == "stdio"
        assert row["signed_uploads"] == []
    else:
        server_id = _create_ok(owner, signed_uploads=SIGNED_UPLOADS)
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={"signed_uploads": None},
            headers=owner,
        )
        assert resp.status_code == 200, resp.text
        assert _row(server_id)["signed_uploads_is_null"] is True


@pytest.mark.parametrize(
    ("created", "switched"),
    [("streamable_http", "sse"), ("sse", "streamable_http")],
)
def test_switching_between_sse_and_streamable_http_keeps_valid_settings(
    created: str, switched: str
) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner, transport=created, signed_uploads=SIGNED_UPLOADS)

    resp = client.put(
        f"/api/mcp/servers/{server_id}", json={"transport": switched}, headers=owner
    )

    assert resp.status_code == 200, resp.text
    assert resp.json()["signed_uploads"] == SIGNED_UPLOADS
    assert resp.json()["signed_uploads_error"] is None
    row = _row(server_id)
    assert row["transport"] == switched
    assert row["signed_uploads"] == SIGNED_UPLOADS


@pytest.mark.parametrize(
    "case",
    [
        "owner-rename",
        "owner-description",
        "owner-same-transport",
        "owner-same-url",
        "owner-new-url",
        "non-owner-personal",
        "non-owner-same-description",
    ],
)
def test_save_checks_signed_uploads_only_when_they_or_the_connection_change(
    case: str,
) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    _store(server_id, BAD)
    caller = owner
    if case.startswith("non-owner"):
        caller = _register_second_user("carol", "carolpass1")
        _link_non_owner("carol", server_id)

    payloads: dict[str, dict[str, Any]] = {
        "owner-rename": {"name": "upload-connector-renamed"},
        "owner-description": {"description": "New description"},
        "owner-same-transport": {"transport": "streamable_http"},
        "owner-same-url": {"config": {"url": URL}},
        "owner-new-url": {"config": {"url": "https://other.example.test/mcp"}},
        "non-owner-personal": {"user_env": {"K": "v"}, "is_active": False},
        "non-owner-same-description": {"description": "Uploads"},
    }
    resp = client.put(
        f"/api/mcp/servers/{server_id}", json=payloads[case], headers=caller
    )

    if case == "owner-new-url":
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == _invalid_detail(NOT_A_LIST_ERROR)
        assert _row(server_id)["url"] == URL
        return

    assert resp.status_code == 200, resp.text
    row = _row(server_id)
    assert row["signed_uploads"] == BAD
    if case == "owner-rename":
        assert row["name"] == "upload-connector-renamed"
        body = resp.json()
        assert body["signed_uploads"] == BAD
        assert body["signed_uploads_error"] == NOT_A_LIST_ERROR


@pytest.mark.parametrize(
    ("stored", "requested", "expected_status"),
    [
        pytest.param("omit", SIGNED_UPLOADS, 403, id="set-on-empty"),
        pytest.param("omit", [], 200, id="empty-list-on-null"),
        pytest.param(SIGNED_UPLOADS, SIGNED_UPLOADS, 200, id="same-value"),
        pytest.param([], None, 200, id="null-on-empty-list"),
        pytest.param(SIGNED_UPLOADS, [], 403, id="clear-to-empty-list"),
        pytest.param(SIGNED_UPLOADS, None, 403, id="clear-to-null"),
    ],
)
def test_non_owner_cannot_change_signed_uploads(
    stored: Any, requested: Any, expected_status: int
) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    if stored == "omit":
        server_id = _create_ok(owner)
    else:
        server_id = _create_ok(owner, signed_uploads=stored)
    other = _register_second_user("carol", "carolpass1")
    _link_non_owner("carol", server_id)
    before = _row(server_id)

    resp = client.put(
        f"/api/mcp/servers/{server_id}",
        json={"signed_uploads": requested},
        headers=other,
    )

    assert resp.status_code == expected_status, resp.text
    if expected_status == 403:
        assert resp.json()["detail"] == OWNER_ONLY_DETAIL
    assert _row(server_id) == before


def test_non_owner_resending_stored_invalid_signed_uploads_is_not_checked() -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    stored = [{"tool_name": "attach_document"}]
    _store(server_id, stored)
    other = _register_second_user("carol", "carolpass1")
    _link_non_owner("carol", server_id)
    before = _row(server_id)

    resp = client.put(
        f"/api/mcp/servers/{server_id}",
        json={"signed_uploads": stored},
        headers=other,
    )

    # A non-owner's copy is only compared with the stored value and never
    # saved, so it is not checked.  The owner sending the same value gets 400:
    # test_owner_saving_the_stored_invalid_signed_uploads_is_still_checked.
    assert resp.status_code == 200, resp.text
    assert resp.json()["signed_uploads"] == stored
    assert resp.json()["signed_uploads_error"] == MISSING_KEYS_ERROR
    assert _row(server_id) == before


def test_non_owner_empty_list_against_a_stored_non_list_is_a_change() -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    _store(server_id, {})
    other = _register_second_user("carol", "carolpass1")
    _link_non_owner("carol", server_id)
    before = _row(server_id)

    resp = client.put(
        f"/api/mcp/servers/{server_id}",
        json={"signed_uploads": []},
        headers=other,
    )

    # Only null and [] mean no settings.  A stored {} is neither, so [] is a
    # change and the non-owner is refused.  The row stays as it is either way,
    # because a non-owner's value is never saved.
    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == OWNER_ONLY_DETAIL
    assert _row(server_id) == before


@pytest.mark.parametrize("case", ["admin-valid", "admin-missing-key"])
def test_platform_admin_can_change_signed_uploads(case: str) -> None:
    admin = _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    _link_non_owner("admin", server_id)
    before = _row(server_id)
    assert before["signed_uploads_is_null"] is True

    if case == "admin-valid":
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={"signed_uploads": SIGNED_UPLOADS},
            headers=admin,
        )
        assert resp.status_code == 200, resp.text
        assert _row(server_id)["signed_uploads"] == SIGNED_UPLOADS
    else:
        raw = _without_content_types()
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={"signed_uploads": raw},
            headers=admin,
        )
        assert resp.status_code == 400, resp.text
        assert resp.json()["detail"] == _invalid_detail(MISSING_KEYS_ERROR)
        assert _row(server_id) == before


def test_owner_saving_the_stored_invalid_signed_uploads_is_still_checked() -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    stored = [{"tool_name": "attach_document"}]
    _store(server_id, stored)
    before = _row(server_id)
    assert before["signed_uploads"] == stored

    resp = client.put(
        f"/api/mcp/servers/{server_id}",
        json={"signed_uploads": stored},
        headers=owner,
    )

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == _invalid_detail(MISSING_KEYS_ERROR)
    assert _row(server_id) == before


def _add_custom_api(owner_name: str) -> None:
    db = _direct_db_session()
    try:
        api = CustomApi(
            name="lookup-api", url="https://api.example.test/lookup", method="GET"
        )
        db.add(api)
        db.flush()
        db.add(
            UserCustomApi(
                user_id=_user_id(owner_name),
                custom_api_id=api.id,
                is_owner=True,
                is_active=True,
            )
        )
        db.commit()
    finally:
        db.close()


@pytest.mark.parametrize(
    ("stored", "expected_error"),
    [
        ({"tool_name": "attach_document"}, NOT_A_LIST_ERROR),
        ([{"tool_name": "attach_document"}], MISSING_KEYS_ERROR),
    ],
    ids=["not-an-array", "missing-keys"],
)
def test_list_and_detail_echo_invalid_signed_uploads(
    stored: Any, expected_error: str
) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner)
    _store(server_id, stored)
    _add_custom_api("bob")

    listing = client.get("/api/mcp/servers", headers=owner)
    assert listing.status_code == 200, listing.text
    items = listing.json()
    listed = _server_named(items, server_id)
    assert listed["signed_uploads"] == stored
    assert listed["signed_uploads_error"] == expected_error

    custom = [item for item in items if item["transport"] == "custom_api"]
    assert len(custom) == 1
    assert custom[0]["signed_uploads"] is None
    assert custom[0]["signed_uploads_error"] is None

    detail = client.get(f"/api/mcp/servers/{server_id}", headers=owner)
    assert detail.status_code == 200, detail.text
    assert detail.json()["signed_uploads"] == listed["signed_uploads"]
    assert detail.json()["signed_uploads_error"] == listed["signed_uploads_error"]


def test_list_and_detail_echo_usable_and_inert_signed_uploads() -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    usable_id = _create_ok(owner, signed_uploads=SIGNED_UPLOADS)
    inert_id = _create_ok(
        owner,
        name="stdio-connector",
        transport="stdio",
        config={"command": "python"},
    )
    _store(inert_id, SIGNED_UPLOADS)

    listing = client.get("/api/mcp/servers", headers=owner)
    assert listing.status_code == 200, listing.text
    items = listing.json()

    for server_id in (usable_id, inert_id):
        listed = _server_named(items, server_id)
        assert listed["signed_uploads"] == SIGNED_UPLOADS
        assert listed["signed_uploads_error"] is None

        detail = client.get(f"/api/mcp/servers/{server_id}", headers=owner)
        assert detail.status_code == 200, detail.text
        assert detail.json()["signed_uploads"] == SIGNED_UPLOADS
        assert detail.json()["signed_uploads_error"] is None


# ---------------------------------------------------------------------------
# Issuer entries
# ---------------------------------------------------------------------------

ISSUER_ENTRY = {
    "issuing_tool_name": "request_upload",
    "allowed_upload_hosts": ["uploads.example.test"],
    "upload_path_prefix": "/staged/",
    "content_types": {".pdf": "application/pdf"},
}
ISSUER_ENTRIES = [ISSUER_ENTRY]
# A replacement entry for another issuing tool, which may sit next to an issuer.
OTHER_REPLACEMENT = {**SIGNED_UPLOADS[0], "issuing_tool_name": "issue_other"}
# A replacement entry that only hides the issuing tool: its tool is one the
# connector does not list.
HIDE_ONLY_ENTRIES = [
    {
        **ISSUER_ENTRY,
        "tool_name": "unlisted_tool",
        "file_argument": "document_file_id",
    }
]


def _also_a_replacement_issuer(position: int) -> str:
    return _invalid_detail(
        f"signed_uploads[{position}].issuing_tool_name "
        "is also the issuing tool of a replacement entry"
    )


@pytest.mark.parametrize(
    "case",
    [
        "create-issuer",
        "create-issuer-and-replacement",
        "update-issuer-replaces-replacement",
        "update-issuer-rewritten-as-hide-only",
        "update-hide-only-back-to-issuer",
    ],
)
def test_api_accepts_issuer_entries(case: str) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    expected: list[dict[str, Any]] = ISSUER_ENTRIES

    if case.startswith("create-"):
        if case == "create-issuer-and-replacement":
            expected = [ISSUER_ENTRY, OTHER_REPLACEMENT]
        resp = _create(owner, signed_uploads=expected)
        assert resp.status_code == 201, resp.text
        assert resp.json()["signed_uploads"] == expected
        assert resp.json()["signed_uploads_error"] is None
        server_id = int(resp.json()["id"])
    else:
        stored: list[dict[str, Any]] = {
            "update-issuer-replaces-replacement": SIGNED_UPLOADS,
            "update-issuer-rewritten-as-hide-only": ISSUER_ENTRIES,
            "update-hide-only-back-to-issuer": HIDE_ONLY_ENTRIES,
        }[case]
        if case == "update-issuer-rewritten-as-hide-only":
            expected = HIDE_ONLY_ENTRIES
        elif case == "update-hide-only-back-to-issuer":
            expected = ISSUER_ENTRIES
        server_id = _create_ok(owner, signed_uploads=stored)
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={"signed_uploads": expected},
            headers=owner,
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["signed_uploads"] == expected
        assert resp.json()["signed_uploads_error"] is None

    assert _row(server_id)["signed_uploads"] == expected
    detail = client.get(f"/api/mcp/servers/{server_id}", headers=owner)
    assert detail.status_code == 200, detail.text
    assert detail.json()["signed_uploads"] == expected
    assert detail.json()["signed_uploads_error"] is None


_CLASHES = [
    pytest.param(
        [ISSUER_ENTRY, SIGNED_UPLOADS[0]],
        _also_a_replacement_issuer(0),
        id="issuer-first",
    ),
    pytest.param(
        [SIGNED_UPLOADS[0], ISSUER_ENTRY],
        _also_a_replacement_issuer(1),
        id="issuer-last",
    ),
]


@pytest.mark.parametrize("mode", ["create", "update"])
@pytest.mark.parametrize(("clashing", "detail"), _CLASHES)
def test_save_rejects_clashing_issuer_entries(
    mode: str, clashing: Any, detail: str
) -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    if mode == "create":
        resp = _create(owner, signed_uploads=clashing)
        server_id = None
    else:
        server_id = _create_ok(owner, signed_uploads=ISSUER_ENTRIES)
        before = _row(server_id)
        resp = client.put(
            f"/api/mcp/servers/{server_id}",
            json={"signed_uploads": clashing},
            headers=owner,
        )

    assert resp.status_code == 400, resp.text
    assert resp.json()["detail"] == detail
    if server_id is not None:
        assert _row(server_id) == before
    else:
        db = _direct_db_session()
        try:
            assert db.query(MCPServer).filter(MCPServer.name == NAME).count() == 0
        finally:
            db.close()


def test_non_owner_cannot_change_issuer_entries() -> None:
    _admin_headers()
    owner = _register_second_user("bob", "bobpass1")
    server_id = _create_ok(owner, signed_uploads=ISSUER_ENTRIES)
    other = _register_second_user("carol", "carolpass1")
    _link_non_owner("carol", server_id)
    before = _row(server_id)
    changed = [{**ISSUER_ENTRY, "upload_path_prefix": "/other/"}]

    resp = client.put(
        f"/api/mcp/servers/{server_id}",
        json={"signed_uploads": changed},
        headers=other,
    )

    assert resp.status_code == 403, resp.text
    assert resp.json()["detail"] == OWNER_ONLY_DETAIL
    assert _row(server_id) == before
