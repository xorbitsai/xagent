"""Protocol-level coverage for the standalone FXMacroData MCP example.

Run with: PYTHONPATH=src python -m unittest discover -s tests/scripts
"""

import asyncio
import importlib.util
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import uvicorn
import yaml
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "docs/examples/fxmacrodata/mcp_server.yaml"
SPEC = importlib.util.spec_from_file_location(
    "fxmacrodata_example", ROOT / "scripts/fxmacrodata_example.py"
)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)

TEST_KEY = "test-key-123"
DELAY_MESSAGE = "Free access is delayed by 15 minutes."


class ExampleConfigTest(unittest.TestCase):
    def test_example_config_is_keyless_https(self) -> None:
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        self.assertEqual(data["url"], "https://mcp.fxmacrodata.com")
        self.assertNotIn("Authorization", data["headers"])
        with mock.patch.dict(os.environ, {}, clear=True):
            config = example.load_config(CONFIG, example.read_api_key())
        self.assertEqual(config.headers, {"User-Agent": "xagent/fxmacrodata-example"})

    def test_key_adds_bearer_header(self) -> None:
        config = example.load_config(CONFIG, TEST_KEY)
        self.assertEqual(config.headers["Authorization"], f"Bearer {TEST_KEY}")

    def test_invalid_key_is_rejected_without_echo(self) -> None:
        for value in ("bad key", "line\nbreak", "café"):
            with mock.patch.dict(os.environ, {"FXMACRODATA_API_KEY": value}):
                with self.assertRaises(ValueError) as caught:
                    example.read_api_key()
            self.assertNotIn(value.strip(), str(caught.exception))

    def test_blank_key_means_keyless(self) -> None:
        with mock.patch.dict(os.environ, {"FXMACRODATA_API_KEY": "  "}):
            self.assertIsNone(example.read_api_key())

    def test_plain_http_remote_url_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mcp_server.yaml"
            data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
            data["url"] = "http://example.com/mcp"
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            for api_key in (None, TEST_KEY):
                with self.assertRaisesRegex(ValueError, "https"):
                    example.load_config(path, api_key)

    def test_currency_must_be_three_letters(self) -> None:
        self.assertEqual(example.normalize_currency(" USD "), "usd")
        for value in ("US", "USD1", "U$D", ""):
            with self.assertRaises(ValueError):
                example.normalize_currency(value)

    def test_notices_found_in_nested_json_text_once(self) -> None:
        payload = {"result": {"freemium_delay": {"message": DELAY_MESSAGE}}}
        result = {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "structured_content": payload,
        }
        notices = example.collect_notices(result)
        self.assertEqual(list(dict.fromkeys(notices)), [DELAY_MESSAGE])
        self.assertEqual(example.collect_notices({"content": "not json"}), [])


class FXMacroDataExampleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        server = FastMCP("example-fixture", stateless_http=True, json_response=True)

        @server.tool()
        def release_calendar(currency: str) -> dict[str, Any]:
            self.calls.append({"tool": "release_calendar", "currency": currency})
            row = {"release": "inflation", "announcement_datetime": 1791981000}
            # Echo the credential so the tests prove it never reaches stdout.
            return {"ok": True, "result": {"data": [row]}, "seen": self.last_auth()}

        @server.tool()
        def latest_announcements(currency: str) -> dict[str, Any]:
            self.calls.append({"tool": "latest_announcements", "currency": currency})
            if currency == "eur":
                raise ValueError(f"subscription_required for {self.last_auth()}")
            delay = {"applied": True, "message": DELAY_MESSAGE}
            row = {"indicator": "inflation", "val": 2.9}
            return {"ok": True, "result": {"data": [row], "freemium_delay": delay}}

        app = server.streamable_http_app()

        async def record_request(request: Any, call_next: Any) -> Any:
            self.requests.append(
                {
                    "user_agent": request.headers.get("user-agent"),
                    "authorization": request.headers.get("authorization"),
                }
            )
            return await call_next(request)

        app.add_middleware(BaseHTTPMiddleware, dispatch=record_request)

        self.socket = socket.socket()
        self.socket.bind(("127.0.0.1", 0))
        port = self.socket.getsockname()[1]
        self.server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="on")
        )
        self.task = asyncio.create_task(self.server.serve(sockets=[self.socket]))
        async with asyncio.timeout(5):
            while not self.server.started:
                if self.task.done():
                    await self.task
                await asyncio.sleep(0.01)

        self.directory = tempfile.TemporaryDirectory()
        self.config = Path(self.directory.name) / "mcp_server.yaml"
        data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
        data["url"] = f"http://127.0.0.1:{port}/mcp"
        self.config.write_text(yaml.safe_dump(data), encoding="utf-8")

    async def asyncTearDown(self) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, timeout=5)
        self.socket.close()
        self.directory.cleanup()

    def last_auth(self) -> str | None:
        return self.requests[-1]["authorization"] if self.requests else None

    async def run_example(
        self, currency: str, api_key: str | None
    ) -> tuple[int, str, str]:
        env = {k: v for k, v in os.environ.items() if k != "FXMACRODATA_API_KEY"}
        if api_key is not None:
            env["FXMACRODATA_API_KEY"] = api_key
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "scripts/fxmacrodata_example.py"),
            "--config",
            str(self.config),
            "--currency",
            currency,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        assert process.returncode is not None
        return process.returncode, stdout.decode(), stderr.decode()

    async def test_keyless_run_through_agent_loader(self) -> None:
        code, stdout, stderr = await self.run_example("USD", None)
        self.assertEqual(code, 0, stderr)
        result = json.loads(stdout)
        self.assertEqual(
            set(result), {"release_calendar", "latest_announcements", "notices"}
        )
        self.assertEqual(result["notices"], [DELAY_MESSAGE])
        self.assertIn("inflation", json.dumps(result["release_calendar"]))
        self.assertEqual(
            [c["tool"] for c in self.calls],
            ["release_calendar", "latest_announcements"],
        )
        self.assertEqual({c["currency"] for c in self.calls}, {"usd"})
        self.assertTrue(self.requests)
        for request in self.requests:
            self.assertEqual(request["user_agent"], "xagent/fxmacrodata-example")
            self.assertIsNone(request["authorization"])

    async def test_key_is_sent_as_bearer_and_never_printed(self) -> None:
        code, stdout, stderr = await self.run_example("USD", TEST_KEY)
        self.assertEqual(code, 0, stderr)
        for request in self.requests:
            self.assertEqual(request["authorization"], f"Bearer {TEST_KEY}")
        self.assertNotIn(TEST_KEY, stdout + stderr)
        self.assertIn("[redacted]", stdout)

    async def test_locked_result_fails_without_the_key(self) -> None:
        code, stdout, stderr = await self.run_example("EUR", TEST_KEY)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, "")
        self.assertIn("latest_announcements failed", stderr)
        self.assertIn("subscription_required", stderr)
        self.assertNotIn(TEST_KEY, stderr)

    async def test_invalid_currency_is_rejected_before_connecting(self) -> None:
        with self.assertRaises(ValueError):
            await example.fetch(self.config, "US1")
        self.assertEqual(self.requests, [])


if __name__ == "__main__":
    unittest.main()
