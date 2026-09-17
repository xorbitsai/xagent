"""Widget-guest WS per-message revalidation (#992).

The widget WS endpoint re-derives access from the *live* agent/workforce on
every inbound message and closes with ``4003`` when it is gone
(``public_chat_websocket_endpoint``). That was an explicit acceptance criterion
of the agent (#988/#989) and workforce (#985) widget work, but only the HTTP
``POST /api/widget/chat/task/create`` re-check was covered — nothing exercised
the socket, and no test in the suite even reached
``/api/widget/chat/ws/{task_id}``.

These tests drive the real endpoint end to end: a guest with a valid token holds
a live socket, the owner revokes the widget (disable, key rotation, or
unpublishing the owner — #1055), and the next inbound message must drop the
connection with ``4003``. Unlike
``test_public_chat_websocket_db_boundary.py`` — which mocks
``_authorize_public_chat_websocket`` to prove the close *plumbing* — nothing is
stubbed on the auth path here, so the revocation checks in
``ensure_widget_agent_available`` / ``ensure_widget_workforce_available`` are
what the assertions actually depend on.

Boundary of that guarantee, so it is explicit for the next reader: revalidation
fires only on *inbound* messages. A revoked guest that simply stops sending
keeps receiving outbound and broadcast frames on its open socket — "dropped on
the next message" is the whole contract, not "dropped when revoked".

Connect-time revocation — the guest reloads the page after being revoked and
reconnects, the far more common sequence — is covered by
``test_widget_ws_reconnect_after_revocation_reports_4003`` below. Since #1057 the
widget endpoint accepts the handshake before auth (like the share path), so that
denial is a post-accept ``4003`` carrying the real reason rather than the generic
pre-accept ``4001`` uvicorn used to collapse into a bare HTTP 403.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect

from xagent.web.api.public_chat_access import create_public_chat_access_token
from xagent.web.models.agent import Agent, AgentStatus
from xagent.web.models.user import User
from xagent.web.services import workforce_runs as workforce_runs_service

from .conftest import _admin_headers, _direct_db_session, client

pytestmark = pytest.mark.usefixtures("_test_db")

GUEST_ID = "guest-ws-revalidation"


def _owner_id() -> int:
    """Bootstrap the admin owner if needed and return its user id."""
    _admin_headers()
    db = _direct_db_session()
    try:
        user = db.query(User).filter(User.username == "admin").one()
        return int(user.id)
    finally:
        db.close()


def _create_published_agent(user_id: int, name: str) -> int:
    db = _direct_db_session()
    try:
        agent = Agent(
            user_id=user_id,
            name=name,
            description=f"{name} description",
            instructions=f"{name} instructions",
            execution_mode="balanced",
            status=AgentStatus.PUBLISHED,
        )
        db.add(agent)
        db.commit()
        db.refresh(agent)
        return int(agent.id)
    finally:
        db.close()


def _enable_agent_widget(agent_id: int) -> str:
    """Turn on an agent's widget and return its key."""
    response = client.put(
        f"/api/agents/{agent_id}",
        headers=_admin_headers(),
        json={"widget_enabled": True, "allowed_domains": ["*"]},
    )
    assert response.status_code == 200, response.text
    db = _direct_db_session()
    try:
        agent = db.query(Agent).filter(Agent.id == agent_id).one()
        assert agent.widget_key
        return str(agent.widget_key)
    finally:
        db.close()


def _create_workforce(name: str) -> int:
    headers = _admin_headers()
    owner_id = _owner_id()
    manager_agent_id = _create_published_agent(owner_id, f"{name} Manager")
    worker_agent_id = _create_published_agent(owner_id, f"{name} Worker")
    created = client.post(
        "/api/workforces",
        headers=headers,
        json={
            "name": name,
            "description": "Coordinates widget WS revalidation tests",
            "manager_agent_id": manager_agent_id,
            "workers": [
                {
                    "source_type": "existing",
                    "agent_id": worker_agent_id,
                    "alias": "worker-1",
                    "assignment_instructions": "Handle everything",
                    "enabled": True,
                    "sort_order": 1,
                }
            ],
        },
    )
    assert created.status_code == 200, created.text
    workforce_id = int(created.json()["id"])
    published = client.post(f"/api/workforces/{workforce_id}/publish", headers=headers)
    assert published.status_code == 200, published.text
    return workforce_id


def _enable_workforce_widget(workforce_id: int) -> str:
    """Turn on a workforce deployment's widget and return its key."""
    response = client.put(
        f"/api/workforces/{workforce_id}/widget",
        headers=_admin_headers(),
        json={"widget_enabled": True, "allowed_domains": ["*"]},
    )
    assert response.status_code == 200, response.text
    key = response.json()["widget_key"]
    assert isinstance(key, str) and key
    return str(key)


def _authenticate_widget_guest(widget_key: str) -> str:
    """Exchange a widget key for a guest access token (direct-visit flow)."""
    response = client.post(
        "/api/widget/auth",
        json={"guest_id": GUEST_ID, "widget_key": widget_key},
    )
    assert response.status_code == 200, response.text
    return str(response.json()["access_token"])


def _stub_workforce_turn_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Let ``create_workforce_run`` build its task/run rows without executing.

    Patches ``schedule_claimed_create_turn`` specifically: that is what
    ``create_workforce_run`` reaches (via ``_start_normalized_workforce_run``),
    *not* ``begin_turn``. Stubbing ``begin_turn`` — as the sibling widget and
    share-link suites do — is inert: the real turn still starts and its
    background task then races test-DB teardown. The caller only reads
    ``background_task`` off the result, so a bare namespace is enough.
    """

    async def _stub(**_kwargs: Any) -> SimpleNamespace:
        return SimpleNamespace(background_task=None)

    monkeypatch.setattr(
        workforce_runs_service.TaskTurnOrchestrator,
        "schedule_claimed_create_turn",
        _stub,
    )


@dataclass(frozen=True)
class _WidgetGuest:
    """A widget guest holding a token for a task it is allowed to open."""

    token: str
    task_id: int
    # Owner-side revocations, each of which must invalidate ``token``. Field
    # names match the ``revocation`` parametrize ids so the test can dispatch
    # by id without a lookup table.
    disable: Callable[[], None]
    rotate: Callable[[], None]
    unpublish: Callable[[], None]


def _agent_widget_guest() -> _WidgetGuest:
    owner_id = _owner_id()
    agent_id = _create_published_agent(owner_id, "WS Revalidation Widget Agent")
    token = _authenticate_widget_guest(_enable_agent_widget(agent_id))

    created = client.post(
        "/api/widget/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello", "agent_id": agent_id},
    )
    assert created.status_code == 200, created.text

    def disable() -> None:
        response = client.put(
            f"/api/agents/{agent_id}",
            headers=_admin_headers(),
            json={"widget_enabled": False},
        )
        assert response.status_code == 200, response.text

    def rotate() -> None:
        response = client.post(
            f"/api/agents/{agent_id}/widget-key/rotate", headers=_admin_headers()
        )
        assert response.status_code == 200, response.text

    def unpublish() -> None:
        response = client.post(
            f"/api/agents/{agent_id}/unpublish", headers=_admin_headers()
        )
        assert response.status_code == 200, response.text

    return _WidgetGuest(
        token=token,
        task_id=int(created.json()["task_id"]),
        disable=disable,
        rotate=rotate,
        unpublish=unpublish,
    )


def _workforce_widget_guest(monkeypatch: pytest.MonkeyPatch) -> _WidgetGuest:
    workforce_id = _create_workforce("WS Revalidation Widget Workforce")
    token = _authenticate_widget_guest(_enable_workforce_widget(workforce_id))
    _stub_workforce_turn_start(monkeypatch)

    created = client.post(
        "/api/widget/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello"},
    )
    assert created.status_code == 200, created.text

    def disable() -> None:
        response = client.put(
            f"/api/workforces/{workforce_id}/widget",
            headers=_admin_headers(),
            json={"widget_enabled": False},
        )
        assert response.status_code == 200, response.text

    def rotate() -> None:
        response = client.post(
            f"/api/workforces/{workforce_id}/widget-key/rotate",
            headers=_admin_headers(),
        )
        assert response.status_code == 200, response.text

    def unpublish() -> None:
        response = client.post(
            f"/api/workforces/{workforce_id}/unpublish", headers=_admin_headers()
        )
        assert response.status_code == 200, response.text

    return _WidgetGuest(
        token=token,
        task_id=int(created.json()["task_id"]),
        disable=disable,
        rotate=rotate,
        unpublish=unpublish,
    )


def _build_guest(channel: str, monkeypatch: pytest.MonkeyPatch) -> _WidgetGuest:
    if channel == "agent":
        return _agent_widget_guest()
    return _workforce_widget_guest(monkeypatch)


def _receive_until(ws: WebSocketTestSession, event_type: str) -> bool:
    """Read frames until one carries ``event_type``.

    The connect handshake replays the task's historical stream, so the reply to
    an inbound message can be queued behind an unknown number of earlier frames.
    A close arriving first raises ``WebSocketDisconnect`` out of here, which is
    the point: it says the socket died on the message under test.
    """
    while True:
        if json.loads(ws.receive_text()).get("type") == event_type:
            return True


def _drain_until_disconnect(ws: WebSocketTestSession) -> WebSocketDisconnect:
    """Consume buffered frames and return the close that ends the socket.

    The connect handshake replays the task's historical stream, so an unknown
    number of frames may already be queued ahead of the close; reading exactly
    once would assert against a data frame instead of the denial.
    """
    with pytest.raises(WebSocketDisconnect) as disconnected:
        while True:
            ws.receive_text()
    return disconnected.value


# Bound the run: if the revalidation is ever dropped, the revoked message falls
# through to the real chat handler and ``_drain_until_disconnect`` blocks on a
# close that never comes. Without this the regression would hang the suite
# instead of failing it (there is no global pytest timeout).
@pytest.mark.timeout(60)
@pytest.mark.parametrize("channel", ["agent", "workforce"])
@pytest.mark.parametrize("revocation", ["disable", "rotate", "unpublish"])
def test_widget_ws_closes_4003_when_widget_is_revoked_mid_session(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    revocation: str,
) -> None:
    """Revoking a widget must drop live guest sockets on their next message.

    The guest JWT has a 30-day TTL and the socket outlives any single request,
    so the receive loop — not the token — is what enforces revocation here.
    All three revocation levers are covered: disabling the widget, rotating its
    key (which the guest token pins via its ``widget_key`` claim), and
    unpublishing the owner — taking an agent/workforce out of service must stop
    its embedded widget from serving, not just hide it from the product surfaces
    (#1055).
    """
    guest = _build_guest(channel, monkeypatch)
    url = f"/api/widget/chat/ws/{guest.task_id}?token={guest.token}"

    with client.websocket_connect(url) as ws:
        # Control: prove the socket survives an inbound message while the widget
        # is still live, so the close below is attributable to the revocation and
        # not to sending anything at all. ``intervention`` is the cheapest
        # handled type that answers (no run is started), and receiving its reply
        # is what makes this an assertion rather than a claim — a close provoked
        # by *this* message would surface here instead of below.
        ws.send_text(json.dumps({"type": "intervention", "action": "noop"}))
        assert _receive_until(ws, "intervention_processed")

        revoke: Callable[[], None] = getattr(guest, revocation)
        revoke()

        ws.send_text(json.dumps({"type": "chat", "message": "still there?"}))
        denied = _drain_until_disconnect(ws)

    assert denied.code == 4003
    assert denied.reason == "Widget is unavailable"


@pytest.mark.timeout(60)
@pytest.mark.parametrize("channel", ["agent", "workforce"])
@pytest.mark.parametrize("revocation", ["disable", "rotate", "unpublish"])
def test_widget_ws_reconnect_after_revocation_reports_4003(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    revocation: str,
) -> None:
    """Connect-time counterpart of the mid-session test above: a revoked guest
    reloads the page and reconnects, so the ``ensure_widget_*_available`` denial
    happens at connect-time auth and surfaces as a post-accept ``4003`` (see the
    module docstring for the accept-first rationale). All three revocation levers
    are covered, mirroring the mid-session case.
    """
    guest = _build_guest(channel, monkeypatch)
    revoke: Callable[[], None] = getattr(guest, revocation)
    revoke()

    url = f"/api/widget/chat/ws/{guest.task_id}?token={guest.token}"
    # Accept-first (#1057): the connection upgrades, then connect-time auth
    # rejects it, so the denial surfaces at receive rather than at
    # context-manager entry (a pre-accept close would raise on the ``with``).
    with client.websocket_connect(url) as ws:
        denied = _drain_until_disconnect(ws)

    assert denied.code == 4003
    assert denied.reason == "Widget is unavailable"


@pytest.mark.parametrize("channel", ["agent", "workforce"])
@pytest.mark.parametrize(
    "user_id_claim",
    [
        # A string that would coerce cleanly -- the interesting string case,
        # since one that is not a number fails everywhere anyway.
        pytest.param("{owner_id}", id="numeric_string"),
        # ``bool`` is an ``int`` subclass, so ``isinstance(True, int)`` passes
        # and ``not True`` is False: a guard that does not exclude ``bool``
        # admits ``true``, which SQLAlchemy renders as ``User.id = 1`` and which
        # then compares equal to owner id 1 in the ownership check
        # (``1 == True``).
        pytest.param(True, id="bool_true"),
    ],
)
def test_widget_guest_token_with_non_int_user_id_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    channel: str,
    user_id_claim: object,
) -> None:
    """``user_id`` must be exactly an int, mirroring ``get_share_chat_user``.

    Widget guest tokens are server-minted, so no caller can produce these
    payloads without the signing key — this is defense in depth. But untyped,
    the two widget branches disagree about a string ``user_id`` (the agent
    branch fails closed on an int/str owner comparison; the workforce branch
    coerced it and admitted the request), and both admit ``true`` as user 1.
    """
    # A regression that admits the token would otherwise start a real workforce
    # run before reaching the assertion below.
    _stub_workforce_turn_start(monkeypatch)
    owner_id = _owner_id()
    payload: dict[str, Any] = {
        "sub": "admin",
        "user_id": user_id_claim.format(owner_id=owner_id)
        if isinstance(user_id_claim, str)
        else user_id_claim,
        "channel_id": None,
        "guest_id": GUEST_ID,
        "auth_mode": "widget",
    }

    if channel == "agent":
        agent_id = _create_published_agent(owner_id, "Typed Claim Widget Agent")
        payload["widget_agent_id"] = agent_id
        payload["widget_key"] = _enable_agent_widget(agent_id)
    else:
        workforce_id = _create_workforce("Typed Claim Widget Workforce")
        payload["widget_workforce_id"] = workforce_id
        payload["widget_key"] = _enable_workforce_widget(workforce_id)

    token = create_public_chat_access_token(payload)
    response = client.post(
        "/api/widget/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello"},
    )

    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Invalid widget token"


def test_widget_guest_token_with_bool_entity_id_claim_is_rejected() -> None:
    """The entity-id claims get the same exact-int treatment as ``user_id``.

    ``widget_agent_id`` / ``widget_workforce_id`` select *which* entity the guest
    is scoped to, so a ``bool`` there is an entity substitution rather than an
    identity one: SQLAlchemy renders ``true`` as ``= 1``, resolving whichever
    agent happens to hold id 1. The agent below is created first precisely so it
    *is* id 1 — with a plain isinstance guard the forged claim resolves to it and
    the request succeeds, which is what makes this more than a smoke test.
    """
    owner_id = _owner_id()
    agent_id = _create_published_agent(owner_id, "Bool Id Claim Widget Agent")
    assert agent_id == 1, "test relies on this agent holding the id `true` resolves to"
    widget_key = _enable_agent_widget(agent_id)

    token = create_public_chat_access_token(
        {
            "sub": "admin",
            "user_id": owner_id,
            "channel_id": None,
            "guest_id": GUEST_ID,
            "auth_mode": "widget",
            "widget_agent_id": True,
            "widget_key": widget_key,
        }
    )
    response = client.post(
        "/api/widget/chat/task/create",
        headers={"Authorization": f"Bearer {token}"},
        json={"title": "hello", "description": "hello"},
    )

    assert response.status_code == 401, response.text
    assert response.json()["detail"] == "Invalid widget token"
