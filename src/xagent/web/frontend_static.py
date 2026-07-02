"""Serve the built frontend static export from the FastAPI backend.

In the single-process (pip / uvx) deployment the frontend is shipped as a
Next.js static export and served by this backend, so no Node runtime is needed.
The multi-container Docker deployment instead runs the Next.js server directly;
there the export directory is absent and the backend stays API-only.

Next static export emits one HTML file per route. Dynamic routes ([id]/[token])
are emitted as a ``__shell__`` placeholder file (see the server wrappers under
``frontend/src/app``); the matching client page reads the real id from the URL.
This module reconstructs the route table from those files so an arbitrary URL
resolves to the correct HTML shell.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

# Path prefixes owned by the backend; never shadowed by the SPA catch-all so an
# unknown API path still returns a JSON 404 rather than index.html.
_API_PREFIXES = (
    "api",
    "v1",
    "uploads",
    "ws",
    "health",
    "ready",
    "docs",
    "redoc",
    "openapi.json",
)

_SHELL_TOKEN = "__shell__"


def _build_shell_patterns(dist_dir: Path) -> list[tuple[re.Pattern[str], Path]]:
    """Turn every ``__shell__`` export file into a (URL regex -> file) rule.

    ``agent/__shell__.html``          -> ``^agent/[^/]+$``
    ``workforces/__shell__/run.html`` -> ``^workforces/[^/]+/run$``
    """
    patterns: list[tuple[re.Pattern[str], Path]] = []
    for html in sorted(dist_dir.rglob("*.html")):
        rel = html.relative_to(dist_dir).with_suffix("")  # drop .html
        segments = rel.as_posix().split("/")
        if _SHELL_TOKEN not in segments:  # placeholder may be a dir or the file
            continue
        regex = ["[^/]+" if seg == _SHELL_TOKEN else re.escape(seg) for seg in segments]
        patterns.append((re.compile("^" + "/".join(regex) + "$"), html))
    return patterns


def mount_frontend(app: FastAPI, dist_dir: Path) -> bool:
    """Serve the frontend static export from ``dist_dir`` on ``app``.

    Returns True if the frontend was mounted, False if the directory is absent
    (in which case the backend stays API-only). Must be called after all API
    routers are registered so the catch-all only receives unmatched paths.
    """
    if not dist_dir.is_dir() or not (dist_dir / "index.html").is_file():
        logger.info(
            "Frontend static export not found at %s; serving API only", dist_dir
        )
        return False

    # Hashed build assets: let StaticFiles handle them directly.
    next_assets = dist_dir / "_next"
    if next_assets.is_dir():
        app.mount("/_next", StaticFiles(directory=str(next_assets)), name="next-assets")

    shell_patterns = _build_shell_patterns(dist_dir)
    not_found = dist_dir / "404.html"

    def _resolve(rel_path: str) -> Path | None:
        # Exact asset (favicon, images, etc.)
        candidate = dist_dir / rel_path
        if rel_path and candidate.is_file():
            return candidate
        # Static route emitted as "<name>.html"
        html = dist_dir / f"{rel_path}.html"
        if rel_path and html.is_file():
            return html
        # Directory index
        index = candidate / "index.html"
        if candidate.is_dir() and index.is_file():
            return index
        # Dynamic route -> shell
        for pattern, target in shell_patterns:
            if pattern.match(rel_path):
                return target
        return None

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> Response:
        if full_path == "" or full_path == "/":
            return FileResponse(dist_dir / "index.html")

        first_segment = full_path.split("/", 1)[0]
        if first_segment in _API_PREFIXES:
            # Owned by the backend but unmatched above -> genuine 404.
            return Response(status_code=404)

        resolved = _resolve(full_path)
        if resolved is not None:
            return FileResponse(resolved)

        if not_found.is_file():
            return FileResponse(not_found, status_code=404)
        return Response(status_code=404)

    logger.info(
        "Serving frontend static export from %s (%d dynamic-route shells)",
        dist_dir,
        len(shell_patterns),
    )
    return True
