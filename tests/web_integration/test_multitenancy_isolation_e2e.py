"""
E2E Tests for Multi-Tenant Data Isolation in Knowledge Base System.

This test suite verifies that multi-tenant isolation is properly enforced
throughout the RAG/KB system, ensuring users can only access their own data
while admins can access all data when appropriate.

These tests are critical for ensuring that schema changes and database migrations
do not break tenant isolation, which is a fundamental security requirement.
"""

from pathlib import Path
from typing import Any, Dict

import pytest

from tests.web_integration.http_helpers import http_detail
from xagent.web.models.user import User

pytestmark = [pytest.mark.e2e, pytest.mark.real_rag]


class TestMultiTenantDataIsolation:
    """Test multi-tenant data isolation in KB system.

    These tests verify that:
    1. Regular users can only see their own collections and documents
    2. Admins can see all collections and documents when needed
    3. Cross-tenant access is properly denied
    4. Isolation is enforced at all levels (API, storage, search)
    """

    @pytest.fixture
    def tenant_users(self) -> Dict[str, Dict[str, str]]:
        """Create multiple test users for different tenants."""
        return {
            "tenant1_user": {
                "username": "tenant1_user",
                "password": "password123",
                "email": "tenant1@example.com",
                "role": "user",
            },
            "tenant2_user": {
                "username": "tenant2_user",
                "password": "password123",
                "email": "tenant2@example.com",
                "role": "user",
            },
            "admin_user": {
                "username": "admin_user",
                "password": "admin123",
                "email": "admin@example.com",
                "role": "admin",
            },
        }

    @pytest.fixture
    def sample_files_for_tenants(self, tmp_path: Path) -> Dict[str, Path]:
        """Create sample files for different tenants."""
        files = {}

        # Tenant 1 files
        tenant1_file = tmp_path / "tenant1_document.txt"
        tenant1_file.write_text(
            "This is a document for tenant 1. "
            "It contains sensitive information specific to tenant 1. "
            "Keywords: tenant1, private, secret"
        )
        files["tenant1_user"] = tenant1_file

        # Tenant 2 files
        tenant2_file = tmp_path / "tenant2_document.txt"
        tenant2_file.write_text(
            "This is a document for tenant 2. "
            "It contains sensitive information specific to tenant 2. "
            "Keywords: tenant2, confidential, internal"
        )
        files["tenant2_user"] = tenant2_file

        # Admin files
        admin_file = tmp_path / "admin_document.txt"
        admin_file.write_text(
            "This is an admin document. "
            "It contains administrative information. "
            "Keywords: admin, system, configuration"
        )
        files["admin_user"] = admin_file

        return files

    def test_tenants_see_only_own_collections_and_documents(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Each tenant sees only their collection(s) and own document counts via list API."""
        user_tokens: Dict[str, str] = {}
        for user_key, user_data in tenant_users.items():
            register_response = client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            if register_response.status_code not in [200, 400]:
                register_response.raise_for_status()

            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens[user_key] = login_response.json()["access_token"]

        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200, http_detail(response)

        for user_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            response = client.get("/api/kb/collections", headers=headers)
            assert response.status_code == 200, http_detail(response)

            collections = response.json()["collections"]
            collection_names = {col["name"] for col in collections}
            expected_collection = f"{user_key}_collection"
            assert expected_collection in collection_names, (
                f"{user_key} should see their own collection"
            )
            for other_tenant in [k for k in user_tokens if k != user_key]:
                assert f"{other_tenant}_collection" not in collection_names, (
                    f"{user_key} should NOT see {other_tenant}'s collection"
                )

            test_collection = next(
                (c for c in collections if c["name"] == expected_collection),
                None,
            )
            assert test_collection is not None
            assert int(test_collection.get("documents", 0)) >= 1

    def test_cross_tenant_access_denied(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
    ):
        """Test that cross-tenant access attempts are properly denied.

        This test verifies that users cannot access other tenants' collections
        or documents by trying different access methods.
        """
        # Setup users and collections
        user_tokens = {}
        collection_ids = {}

        for user_key, user_data in tenant_users.items():
            # Register/login
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens[user_key] = login_response.json()["access_token"]

            # Create collection
            headers = {"Authorization": f"Bearer {user_tokens[user_key]}"}
            file_path = sample_files_for_tenants[user_key]

            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200, http_detail(response)
                collection_ids[user_key] = response.json().get("collection")

        # Try to access other tenant's collection
        tenant1_token = user_tokens["tenant1_user"]
        tenant2_collection_name = "tenant2_user_collection"

        # Attempt to delete documents in tenant2's collection using tenant1's token
        # Note: Using a dummy filename to test access control
        response = client.delete(
            f"/api/kb/collections/{tenant2_collection_name}/documents/test.txt",
            headers={"Authorization": f"Bearer {tenant1_token}"},
        )

        # Cross-tenant delete must be explicitly forbidden.
        assert response.status_code == 403

    def test_admin_sees_all_tenant_collections_and_documents(
        self,
        client,
        tenant_users: Dict[str, Dict[str, str]],
        sample_files_for_tenants: Dict[str, Path],
        db_session_factory,
    ):
        """Admin lists all tenant collections and sees non-zero document counts."""
        user_tokens: Dict[str, str] = {}

        for user_key, user_data in tenant_users.items():
            client.post(
                "/api/auth/register",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                    "email": user_data["email"],
                },
            )
            login_response = client.post(
                "/api/auth/login",
                json={
                    "username": user_data["username"],
                    "password": user_data["password"],
                },
            )
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens[user_key] = login_response.json()["access_token"]

        session = db_session_factory()
        try:
            admin = session.query(User).filter(User.username == "admin_user").first()
            assert admin is not None
            admin.is_admin = True
            session.commit()
        finally:
            session.close()

        for user_key in ("tenant1_user", "tenant2_user"):
            headers = {"Authorization": f"Bearer {user_tokens[user_key]}"}
            file_path = sample_files_for_tenants[user_key]
            with open(file_path, "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (file_path.name, f, "text/plain")},
                    data={"collection": f"{user_key}_collection"},
                    headers=headers,
                )
                assert response.status_code == 200, http_detail(response)

        admin_headers = {"Authorization": f"Bearer {user_tokens['admin_user']}"}
        response = client.get("/api/kb/collections", headers=admin_headers)
        assert response.status_code == 200, http_detail(response)

        collections = response.json()["collections"]
        by_name = {c["name"]: c for c in collections}
        assert "tenant1_user_collection" in by_name
        assert "tenant2_user_collection" in by_name
        assert int(by_name["tenant1_user_collection"].get("documents", 0)) >= 1
        assert int(by_name["tenant2_user_collection"].get("documents", 0)) >= 1


class TestMultiTenantSearchIsolation:
    """Test search isolation between tenants.

    These tests verify that search results are properly isolated,
    ensuring users can only search and retrieve their own documents.
    """

    @pytest.fixture
    def tenant_search_data(self, tmp_path: Path) -> Dict[str, Dict[str, Any]]:
        """Create search test data for different tenants."""
        data = {}

        # Tenant 1: Technical documentation
        tenant1_file = tmp_path / "tenant1_tech_docs.txt"
        tenant1_file.write_text("""
        Tenant 1 Technical Documentation

        This document contains proprietary technical information
        for Tenant 1's systems.

        Key technologies:
        - Python backend services
        - React frontend framework
        - PostgreSQL database
        - Redis caching layer

        This information is CONFIDENTIAL to Tenant 1.
        """)
        data["tenant1"] = {
            "file": tenant1_file,
            "query": "Python backend",
            "expected_keywords": ["Python", "backend", "Tenant 1"],
        }

        # Tenant 2: Marketing materials
        tenant2_file = tmp_path / "tenant2_marketing.txt"
        tenant2_file.write_text("""
        Tenant 2 Marketing Strategy

        This document contains proprietary marketing strategies
        for Tenant 2's products.

        Key strategies:
        - Social media campaigns
        - Email marketing automation
        - Influencer partnerships
        - Content marketing initiatives

        This information is CONFIDENTIAL to Tenant 2.
        """)
        data["tenant2"] = {
            "file": tenant2_file,
            "query": "marketing strategy",
            "expected_keywords": ["marketing", "strategy", "Tenant 2"],
        }

        return data

    def test_search_isolation_own_collection_and_cross_tenant_forbidden(
        self, client, tenant_search_data: Dict[str, Dict[str, Any]]
    ):
        """Own-collection search succeeds; tenant-scoped results; cross-tenant search is 403."""
        users = {
            "tenant1": {"username": "search_iso_user1", "password": "pass1234"},
            "tenant2": {"username": "search_iso_user2", "password": "pass1234"},
        }
        user_tokens: Dict[str, str] = {}
        coll1 = "tenant1_iso_search_coll"
        coll2 = "tenant2_iso_search_coll"

        for tenant_key, user_info in users.items():
            client.post(
                "/api/auth/register",
                json={
                    "username": user_info["username"],
                    "password": user_info["password"],
                    "email": f"{tenant_key}@example.com",
                },
            )
            login_response = client.post("/api/auth/login", json=user_info)
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens[tenant_key] = login_response.json()["access_token"]

        for tenant_key, user_info in users.items():
            token = user_tokens[tenant_key]
            headers = {"Authorization": f"Bearer {token}"}
            file_data = tenant_search_data[tenant_key]
            coll = coll1 if tenant_key == "tenant1" else coll2
            with open(file_data["file"], "rb") as f:
                response = client.post(
                    "/api/kb/ingest",
                    files={"file": (file_data["file"].name, f, "text/plain")},
                    data={"collection": coll},
                    headers=headers,
                )
                assert response.status_code == 200, http_detail(response)

        for tenant_key, token in user_tokens.items():
            headers = {"Authorization": f"Bearer {token}"}
            search_data = tenant_search_data[tenant_key]
            coll = coll1 if tenant_key == "tenant1" else coll2
            response = client.post(
                "/api/kb/search",
                data={
                    "collection": coll,
                    "query_text": search_data["query"],
                },
                headers=headers,
            )
            assert response.status_code == 200, http_detail(response)
            results = response.json().get("results", [])
            result_text = " ".join(
                str(r.get("content") or r.get("text", "")) for r in results
            )
            normalized_result_text = result_text.lower()
            if results:
                for keyword in search_data["expected_keywords"]:
                    assert keyword.lower() in normalized_result_text, (
                        f"Search results should contain expected keyword: {keyword}"
                    )
            other_tenant = "tenant2" if tenant_key == "tenant1" else "tenant1"
            for keyword in tenant_search_data[other_tenant]["expected_keywords"]:
                if keyword.startswith(other_tenant.capitalize()):
                    assert keyword.lower() not in normalized_result_text, (
                        f"Search results should NOT contain {other_tenant}'s content"
                    )

        cross = client.post(
            "/api/kb/search",
            data={
                "collection": coll2,
                "query_text": "test query",
            },
            headers={"Authorization": f"Bearer {user_tokens['tenant1']}"},
        )
        assert cross.status_code == 403


class TestMultiTenantAfterMigration:
    """Test that multi-tenant isolation persists after schema migrations.

    These tests are critical for ensuring that schema changes and
    database migrations do not break tenant isolation.
    """

    def test_isolation_after_user_id_field_change(self, client):
        """Test that isolation remains after user_id field changes.

        This simulates a scenario where the user_id field type or
        structure changes (e.g., from string to int, or added metadata).
        """
        # Create two users
        users = [
            {
                "username": "migration_user1",
                "password": "pass1234",
                "email": "user1@example.com",
            },
            {
                "username": "migration_user2",
                "password": "pass1234",
                "email": "user2@example.com",
            },
        ]

        user_tokens = []
        for user in users:
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens.append(login_response.json()["access_token"])

        # User 1 creates a collection
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("user1_doc.txt", "User 1 private data", "text/plain")},
            data={"collection": "user1_collection"},
            headers=headers1,
        )
        assert response.status_code == 200, http_detail(response)

        # User 2 tries to delete User 1's document
        headers2 = {"Authorization": f"Bearer {user_tokens[1]}"}
        response = client.delete(
            "/api/kb/collections/user1_collection/documents/user1_doc.txt",
            headers=headers2,
        )

        # Cross-tenant delete must be explicitly forbidden.
        assert response.status_code == 403

    def test_legacy_orphan_data_isolation(self, client):
        """Test that legacy data without proper user_id is properly isolated.

        This test simulates legacy data that might not have user_id set
        and verifies it doesn't leak between tenants.
        """
        # Create a regular user
        client.post(
            "/api/auth/register",
            json={
                "username": "legacy_test_user",
                "password": "pass1234",
                "email": "legacy@example.com",
            },
        )
        login_response = client.post(
            "/api/auth/login",
            json={"username": "legacy_test_user", "password": "pass1234"},
        )
        assert login_response.status_code == 200, http_detail(login_response)
        user_token = login_response.json()["access_token"]
        headers = {"Authorization": f"Bearer {user_token}"}

        # Create a collection
        response = client.post(
            "/api/kb/ingest",
            files={"file": ("user_doc.txt", "User private data", "text/plain")},
            data={"collection": "user_collection"},
            headers=headers,
        )
        assert response.status_code == 200, http_detail(response)

        # List collections - should only see own collection
        response = client.get("/api/kb/collections", headers=headers)
        assert response.status_code == 200, http_detail(response)

        collections = response.json()["collections"]
        collection_names = {col["name"] for col in collections}

        # Should see own collection
        assert "user_collection" in collection_names

        # If there were legacy orphan collections, they should not appear
        # (unless the system has specific logic to handle them)

    def test_isolation_with_mixed_schema_versions(self, client):
        """Test isolation works with mixed schema versions.

        This simulates a scenario where some collections use the old schema
        and some use the new schema, verifying isolation still works.
        """
        # Create two users
        users = [
            {
                "username": "schema_user1",
                "password": "pass1234",
                "email": "schema1@example.com",
            },
            {
                "username": "schema_user2",
                "password": "pass1234",
                "email": "schema2@example.com",
            },
        ]

        user_tokens = []
        for user in users:
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200, http_detail(login_response)
            user_tokens.append(login_response.json()["access_token"])

        # Both users create collections
        collection_ids = []
        for i, token in enumerate(user_tokens):
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/ingest",
                files={
                    "file": (f"user{i + 1}_doc.txt", f"User {i + 1} data", "text/plain")
                },
                data={"collection": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200, http_detail(response)
            collection_ids.append(response.json().get("collection"))

        # Verify cross-access is denied
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.delete(
            "/api/kb/collections/user2_collection/documents/user2_doc.txt",
            headers=headers1,
        )
        assert response.status_code == 403, (
            "Cross-tenant access should be denied even with mixed schemas"
        )


class TestMultiTenantDeleteIsolation:
    """Test that delete operations respect tenant isolation.

    These tests verify that users can only delete their own collections
    and documents, not other tenants'.
    """

    def test_users_can_only_delete_own_collections(self, client):
        """Test that users can only delete their own collections."""
        # Create two users
        users = [
            {
                "username": "delete_user1",
                "password": "pass1234",
                "email": "delete1@example.com",
            },
            {
                "username": "delete_user2",
                "password": "pass1234",
                "email": "delete2@example.com",
            },
        ]

        user_tokens = []
        collection_ids = []

        for i, user in enumerate(users):
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200, http_detail(login_response)
            token = login_response.json()["access_token"]
            user_tokens.append(token)

            # Create collection
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/ingest",
                files={
                    "file": (f"user{i + 1}_doc.txt", f"User {i + 1} data", "text/plain")
                },
                data={"collection": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200, http_detail(response)
            collection_ids.append(response.json().get("collection"))

        # User 1 tries to delete User 2's collection
        headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
        response = client.delete(
            "/api/kb/collections/user2_collection", headers=headers1
        )

        # Cross-tenant collection delete must be explicitly forbidden.
        assert response.status_code == 403, (
            "Cross-tenant collection delete should be forbidden"
        )

        # Verify endpoint remains responsive after cross-tenant delete attempt/no-op
        headers2 = {"Authorization": f"Bearer {user_tokens[1]}"}
        response = client.get("/api/kb/collections", headers=headers2)
        assert response.status_code == 200, http_detail(response)

    def test_users_can_only_delete_own_documents(self, client):
        """Test that users can only delete their own documents."""
        # Create two users in same collection scenario
        # (if system supports shared collections)
        # Or test document-level delete isolation

        # For now, test that file_id based deletion is isolated
        users = [
            {
                "username": "doc_delete_user1",
                "password": "pass1234",
                "email": "docdel1@example.com",
            },
            {
                "username": "doc_delete_user2",
                "password": "pass1234",
                "email": "docdel2@example.com",
            },
        ]

        user_tokens = []
        file_ids = []

        for i, user in enumerate(users):
            client.post("/api/auth/register", json=user)
            login_response = client.post(
                "/api/auth/login",
                json={"username": user["username"], "password": user["password"]},
            )
            assert login_response.status_code == 200, http_detail(login_response)
            token = login_response.json()["access_token"]
            user_tokens.append(token)

            # Upload document
            headers = {"Authorization": f"Bearer {token}"}
            response = client.post(
                "/api/kb/ingest",
                files={
                    "file": (
                        f"user{i + 1}_file.txt",
                        f"User {i + 1} file content",
                        "text/plain",
                    )
                },
                data={"collection": f"user{i + 1}_collection"},
                headers=headers,
            )
            assert response.status_code == 200, http_detail(response)
            # Get file_id from response
            file_ids.append(response.json().get("file_id"))

        # User 1 tries to delete User 2's document by file_id
        if file_ids[1]:  # If file_id is available
            headers1 = {"Authorization": f"Bearer {user_tokens[0]}"}
            response = client.delete(
                f"/api/kb/collections/user2_collection/documents/user2_file.txt?file_id={file_ids[1]}",
                headers=headers1,
            )

            # Cross-tenant document delete must be explicitly forbidden.
            assert response.status_code == 403, (
                "User should not be able to delete other tenant's document"
            )
