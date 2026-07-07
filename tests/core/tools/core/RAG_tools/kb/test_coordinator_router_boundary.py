"""#514 - Router boundary guard for KBCoordinator.

The coordinator must stay a router: it must not import backend-specific store
modules or FastAPI/web route modules. This is a forward-looking static guard
(it catches future import-based leaks). The current runtime-shim routing is
covered separately by the fake-handle routing tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xagent.core.tools.core.RAG_tools.kb.coordinator as coordinator_module

# ponytail: hardcoded list with a known ceiling — a NEW backend store impl
# (e.g. qdrant_stores.py) is NOT auto-detected and must be appended here.
FORBIDDEN_SUBSTRINGS = (
    "lancedb",
    "schema_manager",
    "fastapi",
    "starlette",
)
FORBIDDEN_PREFIXES = ("xagent.web",)


def _imported_modules(source: str):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            yield node.lineno, module
            for alias in node.names:
                yield node.lineno, f"{module}.{alias.name}" if module else alias.name


def _is_forbidden(module: str) -> bool:
    if not module:
        return False
    if any(sub in module for sub in FORBIDDEN_SUBSTRINGS):
        return True
    if any(module == p or module.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
        return True
    return False


def test_coordinator_imports_no_backend_or_web_modules() -> None:
    source = Path(coordinator_module.__file__).read_text()
    offenders = [
        (lineno, mod) for lineno, mod in _imported_modules(source) if _is_forbidden(mod)
    ]
    assert offenders == [], (
        "KBCoordinator must not import backend-store or web/route modules "
        f"(keep it a router): {offenders}"
    )


def test_guard_flags_forbidden_imports() -> None:
    # Positive control: the checker must actually fire, so a future refactor
    # that neuters it is caught.
    snippet = (
        "from fastapi import APIRouter\n"
        "import lancedb\n"
        "from x.lancedb_stores import Y\n"
        "from xagent.core.tools.core.RAG_tools.storage import lancedb_stores\n"
    )
    flagged = [mod for _, mod in _imported_modules(snippet) if _is_forbidden(mod)]
    assert "fastapi" in flagged
    assert "lancedb" in flagged
    assert "x.lancedb_stores" in flagged
    assert "xagent.core.tools.core.RAG_tools.storage.lancedb_stores" in flagged
