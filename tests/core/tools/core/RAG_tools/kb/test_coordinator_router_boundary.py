"""#514 - Router boundary guard for KBCoordinator.

The coordinator must stay a router: it must not import backend-specific store
modules or FastAPI/web route modules. This is a forward-looking static guard
(it catches future import-based leaks). The current runtime-shim routing is
covered separately by the fake-handle routing tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import xagent.core.tools.core.RAG_tools.kb.collection_handle as handle_module
import xagent.core.tools.core.RAG_tools.kb.coordinator as coordinator_module

# A new backend store module (e.g. qdrant_stores.py) is not auto-detected and
# must be appended here.
FORBIDDEN_SUBSTRINGS = (
    "lancedb",
    "schema_manager",
    "fastapi",
    "starlette",
)
FORBIDDEN_PREFIXES = ("xagent.web",)


def _resolve_relative(module: str, level: int, package: str) -> str:
    """Turn ``from ..x import y`` into the absolute module it names.

    Without this a relative import reaches no prefix rule, so a route module
    imported as ``from ......web.api.kb import router`` would slip the guard.
    """
    if not level:
        return module
    parts = package.split(".")
    # More dots than the package is deep is not importable, but the guard reads
    # source: without the clamp the negative index would slice from the end and
    # resolve to a plausible-looking module that no rule matches.
    base = parts[: max(0, len(parts) - (level - 1))]
    return ".".join([*base, module]) if module else ".".join(base)


def _imported_modules(source: str, package: str = ""):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_relative(node.module or "", node.level, package)
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
        (lineno, mod)
        for lineno, mod in _imported_modules(source, coordinator_module.__package__)
        if _is_forbidden(mod)
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


# The handle owns backend mechanics, so backend imports are expected there; what
# it must never need is a web framework or a route module (#514 acceptance:
# "handle methods do not require FastAPI, HTTP response, or request objects").
HANDLE_FORBIDDEN_SUBSTRINGS = ("fastapi", "starlette")
HANDLE_FORBIDDEN_PREFIXES = ("xagent.web",)


def _is_forbidden_for_handle(module: str) -> bool:
    if not module:
        return False
    if any(sub in module for sub in HANDLE_FORBIDDEN_SUBSTRINGS):
        return True
    return any(
        module == p or module.startswith(p + ".") for p in HANDLE_FORBIDDEN_PREFIXES
    )


def test_handle_imports_no_web_or_route_modules() -> None:
    source = Path(handle_module.__file__).read_text()
    offenders = [
        (lineno, mod)
        for lineno, mod in _imported_modules(source, handle_module.__package__)
        if _is_forbidden_for_handle(mod)
    ]
    assert offenders == [], (
        f"KBCollectionHandle must stay callable without a web framework: {offenders}"
    )


def test_handle_guard_flags_web_imports() -> None:
    package = "xagent.core.tools.core.RAG_tools.kb"
    snippet = (
        "from fastapi import HTTPException\n"
        "from starlette.responses import JSONResponse\n"
        "from xagent.web.api.kb import router\n"
        "from ......web.api.kb import router\n"
        "from ..storage import lancedb_stores\n"
    )
    flagged = [
        mod
        for _, mod in _imported_modules(snippet, package)
        if _is_forbidden_for_handle(mod)
    ]
    assert "fastapi" in flagged
    assert "starlette.responses" in flagged
    assert "xagent.web.api.kb" in flagged
    # The relative spelling of the same route module must not slip through.
    assert flagged.count("xagent.web.api.kb") == 2
    # Backend imports are legitimate below the boundary.
    assert "xagent.core.tools.core.RAG_tools.storage.lancedb_stores" not in flagged


def test_relative_resolution_clamps_impossible_depth() -> None:
    package = "xagent.core.tools.core.RAG_tools.kb"

    assert _resolve_relative("web.api.kb", 6, package) == "xagent.web.api.kb"
    # Two past the package depth. Without the clamp this is parts[:-1], which
    # keeps five segments and resolves to a module no rule matches; level=99
    # would not show it, since an out-of-range negative slice is already empty.
    assert _resolve_relative("web.api.kb", 8, package) == "web.api.kb"
