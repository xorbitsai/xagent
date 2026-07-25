from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
import websockets
from playwright.async_api import BrowserContext, async_playwright
from websockets.asyncio.server import ServerConnection
from websockets.exceptions import ConnectionClosed

from xagent.core.computer.policy import (
    ComputerPolicyOutcome,
    DefaultComputerActionPolicy,
)
from xagent.core.computer.schema import (
    ComputerAction,
    ComputerActionBatch,
    ComputerActionType,
    ComputerElement,
    ComputerElementSource,
    ComputerEnvironmentType,
    ComputerObservation,
    ComputerTarget,
    Viewport,
)
from xagent.core.context_ref import ContextReference, ContextReferencePurpose

pytestmark = pytest.mark.skipif(
    os.getenv("XAGENT_RUN_BROWSER_EXTENSION_E2E") != "1",
    reason="Set XAGENT_RUN_BROWSER_EXTENSION_E2E=1 for the headed Chrome test.",
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTENSION_ROOT = PROJECT_ROOT / "browser-extension"


class RelayHarness:
    def __init__(self) -> None:
        self.connection: ServerConnection | None = None
        self.messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def handler(self, connection: ServerConnection) -> None:
        self.connection = connection
        hello = self._load(await connection.recv())
        await self.messages.put(hello)
        await connection.send(
            json.dumps(
                {
                    "type": "ready",
                    "protocol_version": 1,
                    "paired": True,
                    "session_token": "saved-relay-session",
                }
            )
        )
        try:
            async for raw in connection:
                await self.messages.put(self._load(raw))
        except ConnectionClosed:
            pass

    async def wait_for(
        self,
        message_type: str,
        *,
        predicate: Any | None = None,
    ) -> dict[str, Any]:
        async with asyncio.timeout(10):
            while True:
                message = await self.messages.get()
                if message.get("type") != message_type:
                    continue
                if predicate is None or predicate(message):
                    return message

    async def observe(self, frame_id: str) -> dict[str, Any]:
        if self.connection is None:
            raise RuntimeError("Extension relay is not connected.")
        request_id = uuid4().hex
        await self.connection.send(
            json.dumps(
                {
                    "type": "command",
                    "protocol_version": 1,
                    "request_id": request_id,
                    "command": "observe",
                    "payload": {"frame_id": frame_id},
                }
            )
        )
        return await self.wait_for(
            "response",
            predicate=lambda message: message.get("request_id") == request_id,
        )

    @staticmethod
    def _load(raw: str | bytes) -> dict[str, Any]:
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise TypeError("Relay message must be an object.")
        return value


@asynccontextmanager
async def login_server() -> AsyncIterator[str]:
    html = b"""<!doctype html>
<html>
  <head><title>Takeover Login</title></head>
  <body>
    <main>
      <h1>Sign in</h1>
      <form id="login">
        <label>Username <input id="username" autocomplete="username"></label>
        <label>Password <input id="password" type="password"
          autocomplete="current-password"></label>
        <button id="sign-in" type="submit">Sign in</button>
      </form>
    </main>
    <script>
      document.querySelector("#login").addEventListener("submit", (event) => {
        event.preventDefault();
        document.title = "Signed in";
        document.querySelector("main").innerHTML = "<h1>Welcome</h1>";
      });
    </script>
  </body>
</html>"""

    async def handle(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            try:
                await reader.readuntil(b"\r\n\r\n")
            except asyncio.IncompleteReadError:
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(html)}\r\n".encode()
                + b"Connection: close\r\n\r\n"
                + html
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = int(server.sockets[0].getsockname()[1])
    try:
        yield f"http://127.0.0.1:{port}/login"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_extension_login_requires_user_takeover(tmp_path: Path) -> None:
    subprocess.run(
        ["npm", "run", "build"],
        cwd=EXTENSION_ROOT,
        check=True,
    )
    relay = RelayHarness()

    async with websockets.serve(relay.handler, "127.0.0.1", 0) as relay_server:
        relay_port = int(relay_server.sockets[0].getsockname()[1])
        relay_url = f"ws://127.0.0.1:{relay_port}/ws/browser-relay"
        async with login_server() as login_url:
            async with async_playwright() as playwright:
                executable = Path(
                    os.getenv(
                        "XAGENT_BROWSER_EXTENSION_CHROMIUM_PATH",
                        playwright.chromium.executable_path,
                    )
                )
                if not executable.exists():
                    pytest.skip(
                        "Install Chromium or set "
                        "XAGENT_BROWSER_EXTENSION_CHROMIUM_PATH."
                    )
                context = await playwright.chromium.launch_persistent_context(
                    str(tmp_path / "chrome-profile"),
                    executable_path=str(executable),
                    headless=False,
                    args=[
                        (f"--disable-extensions-except={EXTENSION_ROOT / 'dist'}"),
                        f"--load-extension={EXTENSION_ROOT / 'dist'}",
                        "--no-first-run",
                    ],
                )
                try:
                    await _exercise_login_takeover(
                        context=context,
                        relay=relay,
                        relay_url=relay_url,
                        login_url=login_url,
                    )
                finally:
                    await context.close()


async def _exercise_login_takeover(
    *,
    context: BrowserContext,
    relay: RelayHarness,
    relay_url: str,
    login_url: str,
) -> None:
    worker = (
        context.service_workers[0]
        if context.service_workers
        else await context.wait_for_event("serviceworker")
    )
    extension_id = worker.url.split("/")[2]
    login_page = await context.new_page()
    await login_page.goto(login_url)
    popup = await context.new_page()
    await popup.goto(f"chrome-extension://{extension_id}/popup.html")
    await popup.locator("#setup-code").fill(
        json.dumps(
            {
                "websocket_url": relay_url,
                "pairing_token": "pair-once",
            }
        )
    )
    await popup.locator("#connect").click()
    await popup.locator("#status-badge").filter(has_text="Connected").wait_for()

    hello = await relay.wait_for("hello")
    assert hello["pairing_token"] == "pair-once"
    assert "session_token" not in hello

    await login_page.bring_to_front()
    status = await popup.evaluate(
        "() => chrome.runtime.sendMessage({type: 'attach_active_tab'})"
    )
    assert status["attached"] is True
    await relay.wait_for(
        "status",
        predicate=lambda message: message.get("attached") is True,
    )

    first = await relay.observe("frame-before-takeover")
    assert first["success"] is True
    serialized = json.dumps(first)
    assert "super-secret" not in serialized
    raw_observation = first["result"]["observation"]
    password = next(
        element
        for element in raw_observation["elements"]
        if element.get("metadata", {}).get("input_type") == "password"
    )
    assert password["metadata"]["sensitive"] is True

    observation = _policy_observation(raw_observation, password)
    decision = await DefaultComputerActionPolicy().evaluate(
        ComputerActionBatch(
            session_id="extension-e2e",
            expected_frame_id=observation.frame_id,
            actions=[
                ComputerAction(
                    type=ComputerActionType.TYPE,
                    target=ComputerTarget(element_id=password["element_id"]),
                    text="agent-must-not-type-this",
                )
            ],
        ),
        observation,
    )
    assert decision.outcome is ComputerPolicyOutcome.BLOCK
    assert await login_page.locator("#password").input_value() == ""

    await login_page.locator("#username").fill("user@example.com")
    await login_page.locator("#password").fill("super-secret")
    await login_page.locator("#sign-in").click()
    await login_page.get_by_role("heading", name="Welcome").wait_for()

    after_takeover = await relay.observe("frame-after-takeover")
    assert after_takeover["success"] is True
    assert after_takeover["result"]["observation"]["title"] == "Signed in"
    assert "super-secret" not in json.dumps(after_takeover)

    previous_connection = relay.connection
    assert previous_connection is not None
    await previous_connection.close(code=1011, reason="transient test failure")
    reconnect_hello = await relay.wait_for("hello")
    assert reconnect_hello["session_token"] == "saved-relay-session"
    assert "pairing_token" not in reconnect_hello
    await popup.locator("#status-badge").filter(has_text="Connected").wait_for()
    reconnected_status = await popup.evaluate(
        "() => chrome.runtime.sendMessage({type: 'get_status'})"
    )
    assert reconnected_status["hasSession"] is True
    assert reconnected_status["attached"] is True

    assert relay.connection is not None
    await relay.connection.send(
        json.dumps(
            {
                "type": "error",
                "protocol_version": 1,
                "error": "Browser relay session is invalid or expired.",
            }
        )
    )
    await popup.locator("#status-badge").filter(has_text="Not paired").wait_for()
    invalidated_status = await popup.evaluate(
        "() => chrome.runtime.sendMessage({type: 'get_status'})"
    )
    assert invalidated_status["hasSession"] is False
    assert invalidated_status["attached"] is False


def _policy_observation(
    raw: dict[str, Any],
    password: dict[str, Any],
) -> ComputerObservation:
    frame_id = "frame-before-takeover"
    return ComputerObservation(
        session_id="extension-e2e",
        frame_id=frame_id,
        environment=ComputerEnvironmentType.BROWSER,
        viewport=Viewport.model_validate(raw["viewport"]),
        screenshot=ContextReference(
            file_ref={
                "file_id": "extension-e2e-frame",
                "filename": "frame.png",
                "mime_type": "image/png",
            },
            purpose=ContextReferencePurpose.OBSERVATION,
            frame_id=frame_id,
        ),
        elements=[
            ComputerElement(
                element_id=password["element_id"],
                source=ComputerElementSource.DOM,
                bounds=password["bounds"],
                label=password.get("label"),
                role=password.get("role"),
                text=password.get("text"),
                metadata=password.get("metadata", {}),
            )
        ],
        active_url=raw.get("active_url"),
        title=raw.get("title"),
    )
