"""E2E tests for KB collection-level access control (403/404 contracts).

These tests complement ``test_multitenancy_isolation_e2e.py`` by asserting
single, explicit HTTP outcomes on routes that use
:func:`xagent.web.api.kb._ensure_collection_access` (and related rename rules).

Ingest uses a **stub embedding pipeline** so CI does not depend on external
embedding keys; assertions are HTTP semantics (403/404), not provider quality.
Real embedding coverage lives in ``real_rag`` suites (e.g. multitenancy E2E).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.web_integration.http_helpers import http_detail

pytestmark = [pytest.mark.e2e, pytest.mark.contract_stub]


def _register_and_login(
    client: TestClient, username: str, password: str, email: str
) -> str:
    """Register (idempotent) and return JWT access token."""
    reg = client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "email": email},
    )
    assert reg.status_code in (200, 400)
    login = client.post(
        "/api/auth/login",
        json={"username": username, "password": password},
    )
    assert login.status_code == 200
    return str(login.json()["access_token"])


def _write_sample_txt(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("access-control e2e sample content", encoding="utf-8")
    return path


def _ingest_txt(
    client: TestClient,
    token: str,
    collection: str,
    file_path: Path,
) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as handle:
        resp = client.post(
            "/api/kb/ingest",
            files={"file": (file_path.name, handle, "text/plain")},
            data={"collection": collection},
            headers=headers,
        )
    assert resp.status_code == 200, http_detail(resp)


class TestKbAccessControlContract:
    """Strict status-code checks for KB access boundaries."""

    def test_parse_result_cross_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Another tenant cannot read parse results for a foreign collection name."""
        t1 = _register_and_login(
            client, "ac_parse_t1", "pw-t1-", "ac_parse_t1@example.com"
        )
        t2 = _register_and_login(
            client, "ac_parse_t2", "pw-t2-", "ac_parse_t2@example.com"
        )

        coll = "ac_parse_coll_t1"
        sample = _write_sample_txt(tmp_path / "ac_parse_doc.txt")
        _ingest_txt(client, t1, coll, sample)

        doc_id = "legit-doc-id-01"
        url = f"/api/kb/collections/{coll}/parses/{doc_id}/parse_result"
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_parse_result_unknown_collection_returns_404(
        self, client: TestClient
    ) -> None:
        """A collection name that does not exist anywhere yields 404 (not 403)."""
        token = _register_and_login(
            client, "ac_parse_404", "pw-404-", "ac_parse_404@example.com"
        )
        missing = "ac_no_such_collection_xyz"
        doc_id = "legit-doc-id-02"
        url = f"/api/kb/collections/{missing}/parses/{doc_id}/parse_result"
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_rename_target_name_taken_by_other_tenant_returns_409(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Renaming into a taken name is a naming conflict, not an access violation."""
        t1 = _register_and_login(client, "ac_rn_t1", "pw-r1-", "ac_rn_t1@example.com")
        t2 = _register_and_login(client, "ac_rn_t2", "pw-r2-", "ac_rn_t2@example.com")

        coll_a = "ac_rename_source_coll"
        coll_b = "ac_rename_target_coll"
        sample1 = _write_sample_txt(tmp_path / "ac_rn_a.txt")
        sample2 = _write_sample_txt(tmp_path / "ac_rn_b.txt")
        _ingest_txt(client, t1, coll_a, sample1)
        _ingest_txt(client, t2, coll_b, sample2)

        resp = client.put(
            f"/api/kb/collections/{coll_a}",
            data={"new_name": coll_b},
            headers={"Authorization": f"Bearer {t1}"},
        )
        assert resp.status_code == 409
        assert "name unavailable" in resp.json()["detail"]
        assert "Access denied" not in resp.json()["detail"]

    def test_documents_check_cross_tenant_returns_403(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Duplicate-check is read-only, so a foreign collection is a denial.

        It creates nothing, so it must not answer with a naming-conflict status.
        """
        t1 = _register_and_login(client, "ac_chk_t1", "pw-c1-", "ac_chk_t1@example.com")
        t2 = _register_and_login(client, "ac_chk_t2", "pw-c2-", "ac_chk_t2@example.com")

        coll = "ac_check_foreign_coll"
        sample = _write_sample_txt(tmp_path / "ac_chk.txt")
        _ingest_txt(client, t1, coll, sample)

        resp = client.post(
            f"/api/kb/collections/{coll}/documents/check",
            json={"filenames": ["any.txt"]},
            headers={"Authorization": f"Bearer {t2}"},
        )
        assert resp.status_code == 403
        assert "Access denied" in resp.json()["detail"]

    def test_save_collection_config_cross_tenant_returns_409(
        self, client: TestClient, tmp_path: Path
    ) -> None:
        """Saving config under a taken name reports a conflict, not a denial.

        The settings panel of an existing knowledge base drives this endpoint, and it
        passes ``allow_create=True``, so a foreign name used to surface as
        "Access denied" here too.
        """
        t1 = _register_and_login(client, "ac_cfg_t1", "pw-g1-", "ac_cfg_t1@example.com")
        t2 = _register_and_login(client, "ac_cfg_t2", "pw-g2-", "ac_cfg_t2@example.com")

        coll = "ac_config_foreign_coll"
        sample = _write_sample_txt(tmp_path / "ac_cfg.txt")
        _ingest_txt(client, t1, coll, sample)

        resp = client.post(
            f"/api/kb/collections/{coll}/config",
            json={"embedding_model_id": "text-embedding-v4"},
            headers={"Authorization": f"Bearer {t2}"},
        )
        detail = resp.json()["detail"]
        assert resp.status_code == 409
        assert "name unavailable" in detail
        assert "Access denied" not in detail
        # The wording claims no ownership. This is not an anti-enumeration
        # measure: the 409-vs-success distinction already reveals that the name
        # is taken, whatever the sentence says.
        assert "already exists" not in detail

    @pytest.mark.parametrize(
        "slug, path, style",
        [
            ("f", "/api/kb/ingest", "file"),
            ("fj", "/api/kb/ingest/jobs", "file"),
            ("w", "/api/kb/ingest-web", "web"),
            ("wj", "/api/kb/ingest-web/jobs", "web"),
            ("c", "/api/kb/ingest-cloud", "cloud"),
        ],
    )
    def test_ingest_into_name_taken_by_other_tenant_returns_409(
        self, client: TestClient, tmp_path: Path, slug: str, path: str, style: str
    ) -> None:
        """The path from #1139: creating a knowledge base by importing content.

        Every ingest entry point takes ``allow_create=True``, and the creation
        dialog drives all of them, so each must report a taken name as a conflict
        rather than accuse the caller of an access violation.
        """
        t1 = _register_and_login(
            client, f"ac_ing_{slug}_t1", "pw-i1-", f"ac_ing_{slug}_t1@example.com"
        )
        t2 = _register_and_login(
            client, f"ac_ing_{slug}_t2", "pw-i2-", f"ac_ing_{slug}_t2@example.com"
        )

        coll = f"ac_ingest_foreign_coll_{slug}"
        sample = _write_sample_txt(tmp_path / "ac_ing.txt")
        _ingest_txt(client, t1, coll, sample)

        headers = {"Authorization": f"Bearer {t2}"}
        if style == "file":
            with open(sample, "rb") as handle:
                resp = client.post(
                    path,
                    files={"file": (sample.name, handle, "text/plain")},
                    data={"collection": coll},
                    headers=headers,
                )
        elif style == "web":
            resp = client.post(
                path,
                data={"collection": coll, "start_url": "https://example.com/docs"},
                headers=headers,
            )
        else:
            resp = client.post(
                path,
                json={
                    "collection": coll,
                    "files": [
                        {
                            "provider": "google_drive",
                            "fileId": "file-1",
                            "fileName": "alpha.pdf",
                        }
                    ],
                },
                headers=headers,
            )

        detail = resp.json()["detail"]
        assert resp.status_code == 409
        assert "name unavailable" in detail
        assert "Access denied" not in detail
        # Wording only: it does not name an owner. See the note above -- the
        # status code itself is what discloses that the name exists.
        assert "already exists" not in detail
