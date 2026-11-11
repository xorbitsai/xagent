"""The Composio switch lookup must import without the web stack.

The connector smoke job runs ``connector_smoke l1`` in a venv that has only
``httpx`` and ``sqlalchemy``. ``get_builtin_public_mcp_app_rows`` reads the
Composio per-app switches, so it has to work with ``fastapi``, ``cryptography``
and the other web dependencies missing. A subprocess keeps the block from
leaking into this process's ``sys.modules``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_BLOCKED = ("cryptography", "fastapi", "starlette", "requests", "yarl", "aiohttp")

_SCRIPT = textwrap.dedent(
    """
    import importlib.abc
    import sys

    BLOCKED = {blocked!r}

    class _Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
            return None

    sys.meta_path.insert(0, _Blocker())
    sys.path.insert(0, {src!r})

    from xagent.web.composio_apps import composio_app
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    assert composio_app("google", "gmail") is None
    assert get_builtin_public_mcp_app_rows()
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in BLOCKED)
    assert not leaked, leaked
    assert "xagent.web.composio_auth" not in sys.modules
    print("ok")
    """
).format(blocked=_BLOCKED, src=str(SRC))


def test_switch_lookup_and_builtin_rows_need_no_web_dependencies():
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True,
        text=True,
        timeout=120,
        env={"PATH": "", "XAGENT_GMAIL_AUTH_BACKEND": "native"},
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "ok"
