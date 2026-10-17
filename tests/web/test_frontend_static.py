"""Tests for serving the frontend static export (``xagent.web.frontend_static``).

Uses a synthetic export directory rather than a real Next.js build, so the
route-resolution logic can be exercised without a Node toolchain.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from xagent.web.browser_security import page_security_headers
from xagent.web.frontend_static import (
    _build_shell_patterns,
    _collect_backend_prefixes,
    mount_frontend,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def dist_dir(tmp_path: Path) -> Path:
    """A synthetic Next.js static export mirroring the real output shape."""
    d = tmp_path / "frontend_dist"
    _write(d / "index.html", "<html>index</html>")
    _write(d / "404.html", "<html>not found</html>")
    # Static routes emit both an HTML file and an RSC .txt payload.
    _write(d / "tools.html", "<html>tools</html>")
    _write(d / "tools.txt", "RSC:tools")
    _write(d / "login.html", "<html>login</html>")
    _write(d / "login.txt", "RSC:login")
    # Single-segment dynamic route shell.
    _write(d / "agent" / "__shell__.html", "<html>agent-shell</html>")
    _write(d / "agent" / "__shell__.txt", "RSC:agent-shell")
    # Nested dynamic route shell.
    _write(d / "workforces" / "__shell__" / "run.html", "<html>run-shell</html>")
    _write(d / "workforces" / "__shell__" / "run.txt", "RSC:run-shell")
    # Hashed build asset.
    _write(d / "_next" / "static" / "chunks" / "main.js", "console.log(1)")
    _write(d / "favicon.ico", "icon-bytes")
    _write(d / "widget" / "chat" / "__shell__.html", "<html>widget</html>")
    return d


@pytest.mark.parametrize(
    "path", ["/", "/tools", "/tools.html", "/agent/123", "/unknown"]
)
def test_spa_html_has_enforced_baseline_csp(dist_dir, path):
    response = TestClient(_app_with_backend_routes(dist_dir)).get(path)
    policy = response.headers["content-security-policy"]
    assert "object-src 'none'" in policy
    assert "base-uri 'none'" in policy
    assert "form-action 'self'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "unsafe-eval" not in policy
    assert response.headers["x-frame-options"] == "DENY"


def test_spa_widget_keeps_embedding_but_not_unsafe_eval(dist_dir):
    response = TestClient(_app_with_backend_routes(dist_dir)).get("/widget/chat/token")
    policy = response.headers["content-security-policy"]
    assert "frame-ancestors" not in policy
    assert "x-frame-options" not in response.headers
    assert "unsafe-eval" not in policy
    assert "object-src 'none'" in policy


def test_widget_traversal_cannot_make_regular_pages_embeddable(dist_dir):
    response = TestClient(_app_with_backend_routes(dist_dir)).get(
        "/widget/..%2Ftools.html"
    )
    assert response.status_code == 200
    assert response.text == "<html>tools</html>"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert response.headers["x-frame-options"] == "DENY"


@pytest.mark.parametrize("embeddable", [False, True])
def test_packaged_and_next_production_policies_match(embeddable):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for cross-runtime policy parity")
    result = subprocess.run(
        [
            node,
            "--input-type=module",
            "-e",
            "import {browserSecurityHeaders} from './browser-security.mjs';"
            f"console.log(JSON.stringify(Object.fromEntries(browserSecurityHeaders({{embeddable: {str(embeddable).lower()}}}).map(x=>[x.key,x.value]))))",
        ],
        cwd=Path(__file__).resolve().parents[2] / "frontend",
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == page_security_headers(embeddable=embeddable)


def test_spa_rsc_and_javascript_are_not_treated_as_html(dist_dir):
    client = TestClient(_app_with_backend_routes(dist_dir))
    for path in ("/login.txt", "/_next/static/chunks/main.js"):
        assert "content-security-policy" not in client.get(path).headers


def _app_with_backend_routes(dist: Path) -> FastAPI:
    """An app whose backend routes are registered before the frontend mount."""
    app = FastAPI()

    @app.get("/api/agents/{agent_id}")
    async def _agents(agent_id: str):  # pragma: no cover - trivial
        return JSONResponse({"id": agent_id})

    @app.get("/preview/{legacy_path:path}")
    async def _preview(legacy_path: str):  # pragma: no cover - trivial
        return JSONResponse({"p": legacy_path})

    @app.get("/health")
    async def _health():  # pragma: no cover - trivial
        return JSONResponse({"ok": True})

    mounted = mount_frontend(app, dist)
    assert mounted is True
    return app


class TestBuildShellPatterns:
    def test_maps_single_and_nested_dynamic_shells(self, dist_dir: Path):
        patterns = _build_shell_patterns(dist_dir)
        matchers = {p.pattern: target for p, target in patterns}
        assert "^agent/[^/]+$" in matchers
        assert "^workforces/[^/]+/run$" in matchers

    def test_matches_ids_but_not_extra_segments(self, dist_dir: Path):
        patterns = dict((p.pattern, p) for p, _ in _build_shell_patterns(dist_dir))
        agent = patterns["^agent/[^/]+$"]
        assert agent.match("agent/abc123")
        assert not agent.match("agent/abc/extra")
        assert not agent.match("agent")


class TestCollectBackendPrefixes:
    def test_derives_prefixes_from_route_table(self, dist_dir: Path):
        app = _app_with_backend_routes(dist_dir)
        prefixes = _collect_backend_prefixes(app)
        assert {"api", "preview", "health"} <= prefixes
        # Mounted assets are included; the dynamic catch-all segment is not.
        assert "_next" in prefixes
        assert "{full_path:path}" not in prefixes


class TestServeSpa:
    def test_root_serves_index(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/")
        assert r.status_code == 200
        assert "index" in r.text

    def test_static_route(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/tools")
        assert r.status_code == 200
        assert "tools" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_dynamic_route_serves_html_shell(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/agent/real-id-123")
        assert r.status_code == 200
        assert "agent-shell" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_nested_dynamic_route_serves_html_shell(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/workforces/w1/run")
        assert r.status_code == 200
        assert "run-shell" in r.text

    def test_dynamic_route_rsc_prefetch_serves_txt_shell(self, dist_dir: Path):
        """N1 regression: a `<route>.txt` prefetch must get the shell's flight
        payload, not the HTML shell."""
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/agent/real-id-123.txt")
        assert r.status_code == 200
        assert r.text == "RSC:agent-shell"
        assert "text/html" not in r.headers["content-type"]

    def test_nested_dynamic_rsc_prefetch_serves_txt_shell(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/workforces/w1/run.txt")
        assert r.status_code == 200
        assert r.text == "RSC:run-shell"

    def test_static_route_txt_served_exactly(self, dist_dir: Path):
        """A real static .txt asset is served as-is, not reinterpreted."""
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/login.txt")
        assert r.status_code == 200
        assert r.text == "RSC:login"

    def test_hashed_asset_served(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/_next/static/chunks/main.js")
        assert r.status_code == 200
        assert "console.log" in r.text

    def test_exact_asset(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/favicon.ico")
        assert r.status_code == 200

    def test_unknown_api_prefix_returns_json_404(self, dist_dir: Path):
        """N2 regression: unmatched backend paths get Starlette's 404 shape."""
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/api/does-not-exist")
        assert r.status_code == 404
        assert "application/json" in r.headers["content-type"]
        assert r.json() == {"detail": "Not Found"}

    def test_real_api_route_not_shadowed(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/api/agents/42")
        assert r.status_code == 200
        assert r.json() == {"id": "42"}

    def test_unknown_spa_path_serves_404_html(self, dist_dir: Path):
        client = TestClient(_app_with_backend_routes(dist_dir))
        r = client.get("/totally/unknown/page")
        assert r.status_code == 404
        assert "not found" in r.text
        assert "text/html" in r.headers["content-type"]

    def test_path_traversal_is_rejected(self, dist_dir: Path, tmp_path: Path):
        """The resolver must not serve files outside dist_dir."""
        secret = tmp_path / "secret.txt"
        secret.write_text("TOPSECRET", encoding="utf-8")
        client = TestClient(_app_with_backend_routes(dist_dir))
        for path in ("/%2e%2e/secret.txt", "/..%2f..%2fsecret.txt"):
            r = client.get(path)
            assert "TOPSECRET" not in r.text
            assert r.status_code == 404


class TestMountFrontend:
    def test_returns_false_when_export_absent(self, tmp_path: Path):
        app = FastAPI()
        assert mount_frontend(app, tmp_path / "missing") is False

    def test_returns_false_when_index_missing(self, tmp_path: Path):
        d = tmp_path / "frontend_dist"
        d.mkdir()
        (d / "_next").mkdir()
        app = FastAPI()
        assert mount_frontend(app, d) is False
