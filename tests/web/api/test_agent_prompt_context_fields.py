"""API tests for agent prompt_context_fields declarations."""

import pytest

from xagent.web.models.agent import Agent
from xagent.web.services.hot_path_cache import invalidate_agent_cache

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

_FIELDS = [
    {
        "name": "account_id",
        "max_length": 64,
        "description": "Customer account identifier",
    },
    {"name": "plan.tier", "max_length": 16, "description": "Current plan name"},
]


def _create(headers: dict[str, str], name: str, fields=None) -> dict:
    body = {"name": name, "execution_mode": "balanced"}
    if fields is not None:
        body["prompt_context_fields"] = fields
    response = client.post("/api/agents", headers=headers, json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_create_echoes_declarations() -> None:
    headers = _admin_headers()
    created = _create(headers, "declared", _FIELDS)

    assert created["prompt_context_fields"] == _FIELDS


def test_create_without_declarations_defaults_to_empty() -> None:
    headers = _admin_headers()
    assert _create(headers, "undeclared")["prompt_context_fields"] == []


def test_update_replaces_declarations_and_omission_keeps_them() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "updatable", _FIELDS)["id"]

    kept = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"description": "changed"},
    )
    assert kept.status_code == 200, kept.text
    assert kept.json()["prompt_context_fields"] == _FIELDS

    replaced = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": [{"name": "only", "max_length": 1}]},
    )
    assert replaced.status_code == 200, replaced.text
    assert replaced.json()["prompt_context_fields"] == [
        {"name": "only", "max_length": 1, "description": None}
    ]


def test_update_with_empty_list_clears_declarations() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "clearable", _FIELDS)["id"]

    cleared = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": []},
    )

    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["prompt_context_fields"] == []


def test_update_with_null_leaves_declarations_unchanged() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "null-keeps", _FIELDS)["id"]

    kept = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": None},
    )

    assert kept.status_code == 200, kept.text
    assert kept.json()["prompt_context_fields"] == _FIELDS


def test_malformed_stored_declarations_read_as_empty() -> None:
    headers = _admin_headers()
    created = _create(headers, "malformed-stored", _FIELDS)
    agent_id = created["id"]
    db = _direct_db_session()
    try:
        agent = db.get(Agent, agent_id)
        agent.prompt_context_fields = [{"name": "no-max-length"}]
        db.commit()
    finally:
        db.close()
    invalidate_agent_cache(int(created["user_id"]), agent_id)

    fetched = client.get(f"/api/agents/{agent_id}", headers=headers)
    updated = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"description": "changed"},
    )

    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["prompt_context_fields"] == []
    assert updated.status_code == 200, updated.text
    assert updated.json()["prompt_context_fields"] == []


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param([{"name": "1bad", "max_length": 8}], id="bad-name-start"),
        pytest.param([{"name": "has space", "max_length": 8}], id="bad-name-space"),
        pytest.param([{"name": "a" * 65, "max_length": 8}], id="bad-name-length"),
        pytest.param(
            [
                {"name": "dup", "max_length": 8},
                {"name": "dup", "max_length": 4},
            ],
            id="duplicate-name",
        ),
        pytest.param(
            [{"name": f"f{i}", "max_length": 8} for i in range(33)],
            id="too-many",
        ),
        pytest.param([{"name": "zero", "max_length": 0}], id="max-length-zero"),
        pytest.param([{"name": "huge", "max_length": 1025}], id="max-length-1025"),
        pytest.param([{"name": "str", "max_length": "8"}], id="max-length-str"),
        pytest.param([{"name": "bool", "max_length": True}], id="max-length-bool"),
        pytest.param([{"name": "float", "max_length": 8.0}], id="max-length-float"),
        pytest.param(
            [{"name": "long-desc", "max_length": 8, "description": "x" * 201}],
            id="description-201",
        ),
        pytest.param(
            [{"name": "newline-desc", "max_length": 8, "description": "a\nb"}],
            id="description-newline",
        ),
        pytest.param(
            [{"name": "tab-desc", "max_length": 8, "description": "a\tb"}],
            id="description-tab",
        ),
        pytest.param(
            [{"name": "backtick-desc", "max_length": 8, "description": "a`b"}],
            id="description-backtick",
        ),
        pytest.param(
            [{"name": "ls-desc", "max_length": 8, "description": "a\u2028b"}],
            id="description-line-separator",
        ),
        pytest.param(
            [{"name": "ps-desc", "max_length": 8, "description": "a\u2029b"}],
            id="description-paragraph-separator",
        ),
        pytest.param(
            [{"name": "bidi-desc", "max_length": 8, "description": "a\u202eb"}],
            id="description-bidi",
        ),
    ],
)
def test_invalid_declarations_are_422(fields) -> None:
    headers = _admin_headers()
    response = client.post(
        "/api/agents",
        headers=headers,
        json={"name": "invalid", "prompt_context_fields": fields},
    )

    assert response.status_code == 422, response.text


def test_description_is_echoed_stripped() -> None:
    headers = _admin_headers()
    created = _create(
        headers,
        "described",
        [{"name": "tier", "max_length": 8, "description": "  Customer tier  "}],
    )

    assert created["prompt_context_fields"] == [
        {"name": "tier", "max_length": 8, "description": "Customer tier"}
    ]


def test_blank_description_becomes_null() -> None:
    headers = _admin_headers()
    created = _create(
        headers,
        "blank-desc",
        [{"name": "tier", "max_length": 8, "description": "   "}],
    )

    assert created["prompt_context_fields"] == [
        {"name": "tier", "max_length": 8, "description": None}
    ]


def test_absent_description_echoes_null() -> None:
    headers = _admin_headers()
    created = _create(headers, "no-desc", [{"name": "tier", "max_length": 8}])

    assert created["prompt_context_fields"] == [
        {"name": "tier", "max_length": 8, "description": None}
    ]


def test_update_rejects_lax_max_length() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "strict-update")["id"]

    for value in ("8", True, 8.0):
        response = client.put(
            f"/api/agents/{agent_id}",
            headers=headers,
            json={"prompt_context_fields": [{"name": "x", "max_length": value}]},
        )
        assert response.status_code == 422, (value, response.text)


def test_update_invalid_declarations_are_422() -> None:
    headers = _admin_headers()
    agent_id = _create(headers, "invalid-update")["id"]

    response = client.put(
        f"/api/agents/{agent_id}",
        headers=headers,
        json={"prompt_context_fields": [{"name": "dup", "max_length": 1}] * 2},
    )

    assert response.status_code == 422, response.text


def test_boundary_declarations_are_accepted() -> None:
    headers = _admin_headers()
    fields = [
        {"name": "A" + "b" * 63, "max_length": 1024, "description": "x" * 200},
        {"name": "_", "max_length": 1, "description": None},
    ]
    created = _create(headers, "boundary", fields)

    assert created["prompt_context_fields"] == fields
