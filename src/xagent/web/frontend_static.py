"""Serve the built frontend static export from the FastAPI backend.

In the single-process (pip / uvx) deployment the frontend is shipped as a
Next.js static export and served by this backend, so no Node runtime is needed.
The multi-container Docker deployment instead runs the Next.js server directly;
there the export directory is absent and the backend stays API-only.

Next static export emits one HTML file per route. Dynamic routes ([id]/[token])
are emitted as a ``__shell__`` placeholder file (see the server wrappers under
``frontend/src/app``). This module resolves real URLs to those shells and binds
their Flight route parameters so the client page's ``useParams()`` receives
the requested id, rather than the build-time placeholder.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .browser_security import page_security_headers

logger = logging.getLogger(__name__)

_SHELL_TOKEN = "__shell__"


def _collect_backend_prefixes(app: FastAPI) -> frozenset[str]:
    """Collect the literal first path segments owned by the backend.

    Derived from the registered route table rather than hand-maintained, so it
    can't silently drift from the real routes. Any request whose first segment
    is in this set but reached the SPA catch-all (i.e. matched no route) gets a
    JSON 404 instead of the SPA shell. Must be called after all routers/mounts
    are registered and before the catch-all is added (the catch-all's own
    ``{full_path:path}`` segment is dynamic and therefore skipped here).
    """
    prefixes: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        if not path or not path.startswith("/"):
            continue
        first = path[1:].split("/", 1)[0]
        # Skip empty (root) and dynamic segments like "{full_path:path}".
        if not first or first.startswith("{"):
            continue
        prefixes.add(first)
    return frozenset(prefixes)


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


def _render_shell(file: Path, value: str) -> str:
    """Bind a shell's dynamic segment in its serialized Next Flight data."""
    # Flight interprets a leading "$" as a reference unless it is doubled.
    flight_value = "$" + value if value.startswith("$") else value
    literal = json.dumps(flight_value, ensure_ascii=True)
    # HTML embeds the Flight stream inside an inline script. Keep user input
    # out of HTML syntax even after decoding the outer JavaScript string.
    for char, escaped in (("<", r"\u003c"), (">", r"\u003e"), ("&", r"\u0026")):
        literal = literal.replace(char, escaped)
    placeholder = json.dumps(_SHELL_TOKEN)
    if file.suffix == ".html":
        # self.__next_f.push([1, "<Flight data>"]) adds a second JSON layer.
        placeholder = json.dumps(placeholder)[1:-1]
        literal = json.dumps(literal)[1:-1]
    return file.read_text(encoding="utf-8").replace(placeholder, literal)


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
    resolved_root = dist_dir.resolve()
    backend_prefixes = _collect_backend_prefixes(app)

    def _file_response(
        file: Path, *, status_code: int = 200, shell_value: str | None = None
    ) -> Response:
        headers = None
        if file.suffix.lower() == ".html":
            route = file.resolve().relative_to(resolved_root).as_posix()
            headers = page_security_headers(
                embeddable=route == "widget.html" or route.startswith("widget/")
            )
        if shell_value is not None:
            return Response(
                _render_shell(file, shell_value),
                status_code=status_code,
                headers=headers,
                media_type="text/html" if file.suffix == ".html" else "text/plain",
            )
        return FileResponse(file, status_code=status_code, headers=headers)

    def _resolve(rel_path: str) -> tuple[Path, str | None] | None:
        # Reject path traversal: the request path must stay within dist_dir.
        try:
            (resolved_root / rel_path).resolve().relative_to(resolved_root)
        except (ValueError, RuntimeError):
            return None

        # Exact asset (favicon, images, and static routes' own .html/.txt files).
        candidate = dist_dir / rel_path
        if rel_path and candidate.is_file():
            return candidate, None

        # Next prefetches a route's RSC flight payload at "<route>.txt". Match it
        # against the route table on the base path and serve the .txt variant, so
        # dynamic-route prefetches get the shell's flight payload (not the HTML,
        # which would break client-side soft navigation).
        is_rsc = rel_path.endswith(".txt")
        lookup = rel_path[:-4] if is_rsc else rel_path
        ext = ".txt" if is_rsc else ".html"

        # Static route emitted as "<name>.html" / "<name>.txt".
        static_file = dist_dir / f"{lookup}{ext}"
        if lookup and static_file.is_file():
            return static_file, None

        # Directory index (HTML navigations only).
        if not is_rsc:
            index = candidate / "index.html"
            if candidate.is_dir() and index.is_file():
                return index, None

        # Dynamic route -> shell, in the format the request asked for.
        for pattern, target in shell_patterns:
            if pattern.match(lookup):
                # Exported pages currently have one dynamic path segment,
                # including nested routes such as workforces/<id>/run.
                segments = target.relative_to(dist_dir).with_suffix("").parts
                value = lookup.split("/")[segments.index(_SHELL_TOKEN)]
                if is_rsc:
                    txt = target.with_suffix(".txt")
                    return (txt if txt.is_file() else target), value
                return target, value
        return None

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str) -> Response:
        if full_path == "":
            return _file_response(dist_dir / "index.html")

        first_segment = full_path.split("/", 1)[0]
        if first_segment in backend_prefixes:
            # Owned by the backend but unmatched above -> genuine 404. Mirror
            # Starlette's default 404 body so API clients see a consistent shape
            # whether or not the frontend is mounted.
            return JSONResponse({"detail": "Not Found"}, status_code=404)

        resolved = _resolve(full_path)
        if resolved is not None:
            file, shell_value = resolved
            if shell_value is not None:
                return await run_in_threadpool(
                    _file_response, file, shell_value=shell_value
                )
            return _file_response(file)

        if not_found.is_file():
            return _file_response(not_found, status_code=404)
        return JSONResponse({"detail": "Not Found"}, status_code=404)

    logger.info(
        "Serving frontend static export from %s (%d dynamic-route shells)",
        dist_dir,
        len(shell_patterns),
    )
    return True
