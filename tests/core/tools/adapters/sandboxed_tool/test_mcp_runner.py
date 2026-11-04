"""Tests for mcp_runner.py helper functions and main()."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from unittest.mock import AsyncMock, patch

from xagent.core.tools.adapters.vibe.sandboxed_tool.mcp_runner import (
    _load_connection,
    main,
)


def test_stdio_runner_imports_without_web_fetch_dependencies():
    script = textwrap.dedent(
        """
        import sys

        class BlockWebDependencies:
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in {'html2text', 'bs4', 'yarl'}:
                    raise ModuleNotFoundError(fullname)
                return None

        sys.meta_path.insert(0, BlockWebDependencies())
        from xagent.core.tools.adapters.vibe.sandboxed_tool.mcp_runner import (
            _load_connection,
        )
        assert callable(_load_connection)
        assert 'xagent.core.tools.core.mcp.http' not in sys.modules
        assert 'xagent.core.tools.core.web_content' not in sys.modules
        print('stdio runner imported')
        """
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, sys.path))},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "stdio runner imported" in result.stdout


class _FakeTool:
    """Minimal fake MCP tool for serialization tests."""

    def __init__(self, name: str) -> None:
        self.name = name

    def model_dump(self, mode: str = "json") -> dict[str, object]:
        assert mode == "json"
        return {
            "name": self.name,
            "description": f"Tool {self.name}",
            "inputSchema": {"type": "object", "properties": {}},
        }


class TestLoadConnection:
    """Tests for _load_connection()."""

    def test_roundtrip(self):
        connection = {"transport": "stdio", "command": "npx", "args": ["demo"]}
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as config:
            json.dump(connection, config)
            config.flush()
            assert _load_connection(config.name) == connection


class TestMain:
    """Tests for mcp_runner.main()."""

    def test_happy_path(self, tmp_path):
        result_file = str(tmp_path / "result.json")
        connection_file = tmp_path / "connection.json"
        connection_file.write_text(
            json.dumps({"transport": "stdio", "command": "npx", "args": ["demo"]})
        )
        argv = [
            "--connection-file",
            str(connection_file),
            "--result-file",
            result_file,
        ]

        with (
            patch(
                "xagent.core.tools.adapters.vibe.sandboxed_tool.mcp_runner.load_mcp_tools",
                new=AsyncMock(return_value=[_FakeTool("echo")]),
            ),
            patch("sys.argv", ["mcp_runner"] + argv),
        ):
            main()

        result = json.loads((tmp_path / "result.json").read_text())
        assert result == [
            {
                "name": "echo",
                "description": "Tool echo",
                "inputSchema": {"type": "object", "properties": {}},
            }
        ]
