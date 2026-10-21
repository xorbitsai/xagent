"""Unsigned prompt-context entry on the widget task-create path.

Only the widget guest task-create accepts top-level ``prompt_context`` values;
they are filtered against the agent's declarations, stamped unsigned, and
stored on ``agent_config``. Every other entry point ignores the field.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from xagent.web.api.chat import chat_router
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.task import Task
from xagent.web.models.user import User
from xagent.web.schemas.chat import TaskCreateRequest, WidgetTaskCreateRequest

from .conftest import (
    _admin_headers,
    _direct_db_session,
    _setup_admin,
    app_for_tests,
    client,
    patch_schedule_bg,
)
from .test_share_rate_limit_endpoints import _widget_guest_headers
from .test_workforce_widget import _create_workforce, _enable_widget

# The api conftest app deliberately omits the authenticated chat router.
app_for_tests.include_router(chat_router)

pytestmark = pytest.mark.usefixtures("_test_db")

DECLARATIONS = [
    {"name": "product.sku", "max_length": 16},
    {"name": "customer.tier", "max_length": 8},
    {"name": "tiny", "max_length": 3},
]


def _user_id() -> int:
    _setup_admin()
    db = _direct_db_session()
    try:
        return int(db.query(User).filter(User.username == "admin").one().id)
    finally:
        db.close()


def _create_widget_agent(
    declarations: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name="Prompt Context Widget Agent",
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            prompt_context_fields=DECLARATIONS
            if declarations is None
            else declarations,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        agent_id = int(agent.id)
    finally:
        db.close()
    enabled = client.put(
        f"/api/agents/{agent_id}",
        headers=_admin_headers(),
        json={"widget_enabled": True, "allowed_domains": ["*"]},
    )
    assert enabled.status_code == 200, enabled.text
    db = _direct_db_session()
    try:
        key = str(db.query(Agent).filter(Agent.id == agent_id).one().widget_key)
    finally:
        db.close()
    return _widget_guest_headers(key)


def _create_share_agent(token: str) -> dict[str, str]:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=_user_id(),
            name="Prompt Context Share Agent",
            description="d",
            instructions="i",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
            share_enabled=True,
            share_token=token,
        )
        db.add(agent)
        db.commit()
    finally:
        db.close()
    response = client.post("/api/share/auth", json={"share_token": token})
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def _task_agent_config(task_id: int) -> dict[str, Any]:
    db = _direct_db_session()
    try:
        task = db.query(Task).filter(Task.id == task_id).one()
        return dict(task.agent_config or {})
    finally:
        db.close()


def _create_widget_task(headers: dict[str, str], **body: Any) -> Any:
    return client.post(
        "/api/widget/chat/task/create",
        headers=headers,
        json={"title": "hello", "description": "hello", **body},
    )


def test_declared_values_are_stored_unsigned() -> None:
    headers = _create_widget_agent()

    response = _create_widget_task(
        headers,
        prompt_context={"product.sku": "A-100", "customer.tier": "vip"},
    )

    assert response.status_code == 200, response.text
    assert _task_agent_config(int(response.json()["task_id"]))["prompt_context"] == {
        "data": {"product.sku": "A-100", "customer.tier": "vip"},
        "signed": False,
    }


def test_undeclared_and_over_length_values_are_dropped() -> None:
    headers = _create_widget_agent()

    response = _create_widget_task(
        headers,
        prompt_context={
            "product.sku": "A-100",
            "customer.tier": "vip",
            "undeclared": "x",
            "tiny": "too-long",
        },
    )

    assert response.status_code == 200, response.text
    assert _task_agent_config(int(response.json()["task_id"]))["prompt_context"] == {
        "data": {"product.sku": "A-100", "customer.tier": "vip"},
        "signed": False,
    }


def test_all_values_dropped_means_no_key() -> None:
    headers = _create_widget_agent()

    response = _create_widget_task(
        headers,
        prompt_context={"undeclared": "x", "product.sku": "x" * 17},
    )

    assert response.status_code == 200, response.text
    assert "prompt_context" not in _task_agent_config(int(response.json()["task_id"]))


def test_agent_config_prompt_context_cannot_bypass_the_filter() -> None:
    headers = _create_widget_agent()

    response = _create_widget_task(
        headers,
        prompt_context={"product.sku": "A-100"},
        agent_config={
            "prompt_context": {
                "data": {"forged": "yes"},
                "signed": True,
            }
        },
    )

    assert response.status_code == 200, response.text
    assert _task_agent_config(int(response.json()["task_id"]))["prompt_context"] == {
        "data": {"product.sku": "A-100"},
        "signed": False,
    }


@pytest.mark.parametrize("value", [["A-100"], "A-100", 3, True])
def test_non_object_prompt_context_is_a_422(value: object) -> None:
    headers = _create_widget_agent()

    response = _create_widget_task(headers, prompt_context=value)

    assert response.status_code == 422


def test_only_the_widget_request_model_declares_prompt_context() -> None:
    assert "prompt_context" in WidgetTaskCreateRequest.model_fields
    assert "prompt_context" not in TaskCreateRequest.model_fields


def test_workforce_widget_drops_values_and_logs_only_the_count(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    workforce_id = _create_workforce("Prompt Context Workforce")
    headers = _widget_guest_headers(_enable_widget(workforce_id))
    patch_schedule_bg(monkeypatch)
    caplog.set_level(logging.INFO, logger="xagent.web.api.public_chat_access")

    response = _create_widget_task(
        headers,
        prompt_context={"product.sku": "A-100", "customer.tier": "vip"},
    )

    assert response.status_code == 200, response.text
    assert "prompt_context" not in _task_agent_config(int(response.json()["task_id"]))
    assert f"Workforce widget {workforce_id} dropped 2 prompt_context field(s)" in (
        caplog.text
    )
    assert "A-100" not in caplog.text
    assert "product.sku" not in caplog.text


def test_authenticated_task_create_ignores_top_level_prompt_context() -> None:
    headers = _admin_headers()

    response = client.post(
        "/api/chat/task/create",
        headers=headers,
        json={
            "title": "hello",
            "description": "hello",
            "prompt_context": {"product.sku": "A-100"},
        },
    )

    assert response.status_code == 200, response.text
    assert "prompt_context" not in _task_agent_config(int(response.json()["task_id"]))


def test_share_task_create_ignores_top_level_prompt_context() -> None:
    headers = _create_share_agent("prompt-context-share")

    response = client.post(
        "/api/share/chat/task/create",
        headers=headers,
        json={
            "title": "hello",
            "description": "hello",
            "prompt_context": {"product.sku": "A-100"},
        },
    )

    assert response.status_code == 200, response.text
    assert "prompt_context" not in _task_agent_config(int(response.json()["task_id"]))


def test_no_prompt_context_value_reaches_the_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    headers = _create_widget_agent()
    caplog.set_level(logging.DEBUG)

    response = _create_widget_task(
        headers,
        prompt_context={"product.sku": "SKU-LOG-SENTINEL", "customer.tier": "vip"},
    )

    assert response.status_code == 200, response.text
    assert "SKU-LOG-SENTINEL" not in caplog.text
