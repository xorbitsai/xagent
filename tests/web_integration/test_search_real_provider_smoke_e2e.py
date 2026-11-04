"""Minimal real embedding + search smoke tests (no stub RAG pipeline).

Uses the default configured embedding provider; intended as a lightweight
real-RAG gate alongside contract_stub-heavy ``test_search_functionality_e2e.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail

pytestmark = [pytest.mark.e2e, pytest.mark.real_rag]


class TestSearchRealProviderSmoke:
    """Single end-to-end path: ingest text → search with Form API → 200 + optional hits."""

    def test_ingest_then_search_form_succeeds(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        tmp_path: Path,
    ) -> None:
        """Real ingest and search; asserts HTTP contract and stable JSON shape."""
        token_phrase = "REAL_SEARCH_SMOKE_TOKEN_Z9Y8X7"
        fp = tmp_path / "smoke_search.txt"
        fp.write_text(
            f"Smoke search document. Unique phrase for retrieval: {token_phrase}.",
            encoding="utf-8",
        )
        collection = "real_search_smoke_collection"
        with open(fp, "rb") as f:
            ingest = client.post(
                "/api/kb/ingest",
                files={"file": ("smoke_search.txt", f, "text/plain")},
                data={"collection": collection},
                headers=auth_headers,
            )
        assert ingest.status_code == 200, http_detail(ingest)
        ing = ingest.json()
        assert ing.get("status") in {"success", "partial"}, http_detail(ingest)

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
        assert "results" in body
        results = body.get("results") or []
        if results:
            joined = " ".join(
                str(r.get("content") or r.get("text", "")) for r in results
            )
            assert token_phrase in joined, http_detail(search)
