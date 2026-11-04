"""Protocol-level coverage for the standalone Parallel MCP example.

Run with: PYTHONPATH=src python -m unittest discover -s tests/scripts
"""

import asyncio
import importlib.util
import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import uvicorn
import yaml
from mcp.server.fastmcp import FastMCP
from starlette.middleware.base import BaseHTTPMiddleware

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "parallel_search_example", ROOT / "scripts/parallel_search_example.py"
)
assert SPEC is not None and SPEC.loader is not None
example = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(example)


class ParallelSearchExampleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.requests: list[dict[str, Any]] = []
        server = FastMCP("example-fixture", stateless_http=True, json_response=True)

        @server.tool()
        def web_search(
            objective: str, search_queries: list[str], session_id: str
        ) -> dict[str, Any]:
            self.calls.append({"tool": "web_search", "session_id": session_id})
            if objective == "fail":
                raise ValueError("fixture search failure")
            return {
                "results": [
                    {
                        "url": "https://docs.python.org/3/library/asyncio-task.html",
                        "title": "Coroutines and Tasks",
                        "excerpts": ["TaskGroup waits for its tasks on exit."],
                    }
                ],
                "queries": search_queries,
            }

        @server.tool()
        def web_fetch(
            urls: list[str], objective: str, session_id: str
        ) -> dict[str, Any]:
            self.calls.append({"tool": "web_fetch", "session_id": session_id})
            return {"url": urls[0], "excerpts": [objective, "Added in version 3.11."]}

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
        original = ROOT / "docs/examples/parallel-search/mcp_server.yaml"
        data = yaml.safe_load(original.read_text(encoding="utf-8"))
        data["url"] = f"http://127.0.0.1:{port}/mcp"
        self.config.write_text(yaml.safe_dump(data), encoding="utf-8")

    async def asyncTearDown(self) -> None:
        self.server.should_exit = True
        await asyncio.wait_for(self.task, timeout=5)
        self.socket.close()
        self.directory.cleanup()

    async def test_search_and_fetch_through_agent_loader(self) -> None:
        url = "https://docs.python.org/3/library/asyncio-task.html"
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(ROOT / "scripts/parallel_search_example.py"),
            "--config",
            str(self.config),
            "--query",
            "Python TaskGroup documentation",
            "--fetch-url",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15)
        self.assertEqual(process.returncode, 0, stderr.decode())
        result = json.loads(stdout)
        self.assertEqual(set(result), {"web_search", "web_fetch"})
        for value in result.values():
            self.assertFalse(value["is_error"])
            self.assertIn(url, json.dumps(value))
        self.assertEqual([c["tool"] for c in self.calls], ["web_search", "web_fetch"])
        self.assertEqual(self.calls[0]["session_id"], self.calls[1]["session_id"])
        self.assertEqual(len(self.calls[0]["session_id"]), 32)
        self.assertTrue(self.requests)
        for request in self.requests:
            self.assertEqual(request["user_agent"], "xagent/parallel-search-example")
            self.assertIsNone(request["authorization"])
        self.assertEqual(len(self.calls), 2)

    async def test_server_error_is_not_successful_output(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "web_search failed"):
            await example.search(self.config, "fail")
        self.assertEqual([c["tool"] for c in self.calls], ["web_search"])


if __name__ == "__main__":
    unittest.main()
