"""Real provider smoke: collections list fields + ingest + search (no stub pipeline).

Complements stub-heavy ``test_frontend_display_e2e.py`` for the default
embedding/search stack.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail

pytestmark = [pytest.mark.e2e, pytest.mark.real_rag]


class TestFrontendDisplayRealProviderSmoke:
    """``GET /api/kb/collections`` display fields + search alignment after ingest."""

    def test_collections_list_and_search_after_real_ingest(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        tmp_path: Path,
    ) -> None:
        token_phrase = "REAL_DISPLAY_SMOKE_TOKEN_A1B2C3"
        fp = tmp_path / "display_smoke.txt"
        fp.write_text(
            f"Frontend display smoke. Keyword: {token_phrase}.",
            encoding="utf-8",
        )
        collection = "real_display_smoke_collection"
        with open(fp, "rb") as f:
            ingest = client.post(
                "/api/kb/ingest",
                files={"file": ("display_smoke.txt", f, "text/plain")},
                data={"collection": collection},
                headers=auth_headers,
            )
        assert ingest.status_code == 200, http_detail(ingest)

        listed = client.get("/api/kb/collections", headers=auth_headers)
        assert listed.status_code == 200, http_detail(listed)
        cols = listed.json().get("collections", [])
        row = next((c for c in cols if c.get("name") == collection), None)
        assert row is not None, http_detail(listed)
        assert "name" in row and row["name"] == collection
        assert "documents" in row
        assert int(row.get("documents", 0)) >= 1

        search = client.post(
            "/api/kb/search",
            data={
                "collection": collection,
                "query_text": token_phrase,
                "top_k": "8",
            },
            headers=auth_headers,
        )
        assert search.status_code == 200, http_detail(search)
        body = search.json()
        results = body.get("results") or []
        if results:
            joined = " ".join(
                str(r.get("content") or r.get("text", "")) for r in results
            )
            assert token_phrase in joined, http_detail(search)
