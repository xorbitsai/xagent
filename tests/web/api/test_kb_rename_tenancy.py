"""Tenant-scoped collection rename behavior (TDD).

Non-admin rename is rejected when multiple users occupy the same collection name.
Admin rename always scopes to an explicit target user.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
)
from xagent.web.api.auth import hash_password
from xagent.web.api.kb import kb_router
from xagent.web.auth_config import JWT_ALGORITHM, JWT_SECRET_KEY
from xagent.web.models.database import Base, get_db
from xagent.web.models.user import User


def _auth_headers(user: User) -> dict[str, str]:
    payload = {
        "sub": user.username,
        "user_id": user.id,
        "type": "access",
        "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        "iat": datetime.now(timezone.utc),
    }
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def rename_users_env(tmp_path):
    """App with two regular users and one admin user."""
    import os
    import shutil

    db_path = tmp_path / "test.db"
    temp_lancedb_dir = tmp_path / "lancedb"
    temp_lancedb_dir.mkdir(parents=True, exist_ok=True)

    from xagent.core.tools.core.RAG_tools.storage.factory import StorageFactory

    previous_lancedb_dir = os.environ.get("LANCEDB_DIR")
    os.environ["LANCEDB_DIR"] = str(temp_lancedb_dir)
    StorageFactory.get_factory().reset_all()

    test_engine = create_engine(f"sqlite:///{db_path}")
    TestingSessionLocal = sessionmaker(bind=test_engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(kb_router)
    app.dependency_overrides[get_db] = override_get_db
    Base.metadata.create_all(bind=test_engine)

    session = TestingSessionLocal()
    regular = User(
        username="regular",
        password_hash=hash_password("test"),
        is_admin=False,
    )
    other = User(
        username="other",
        password_hash=hash_password("test"),
        is_admin=False,
    )
    admin = User(
        username="admin",
        password_hash=hash_password("test"),
        is_admin=True,
    )
    session.add_all([regular, other, admin])
    session.commit()
    session.refresh(regular)
    session.refresh(other)
    session.refresh(admin)

    yield (
        app,
        _auth_headers(regular),
        regular,
        _auth_headers(admin),
        admin,
        other,
        TestingSessionLocal,
    )

    session.close()
    test_engine.dispose()
    StorageFactory.get_factory().reset_all()
    if previous_lancedb_dir is None:
        os.environ.pop("LANCEDB_DIR", None)
    else:
        os.environ["LANCEDB_DIR"] = previous_lancedb_dir
    shutil.rmtree(temp_lancedb_dir, ignore_errors=True)


def _visible_only_old(old_name: str) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=[
            CollectionInfo(name=old_name, documents=1, document_names=[]),
        ],
        total_count=1,
        message="ok",
        warnings=[],
    )


def test_kb_rename_non_admin_rejects_shared_collection_name(rename_users_env) -> None:
    """Non-admin must not rename when multiple users hold the same collection name."""
    app, headers, user, _, _, _, _ = rename_users_env
    client = TestClient(app)
    old_name = "shared_project"
    new_name = "shared_project_new"

    mock_metadata_store = MagicMock()
    mock_metadata_store.count_users_with_collection_config = AsyncMock(return_value=2)

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new_callable=AsyncMock,
            return_value=_visible_only_old(old_name),
        ),
        patch(
            "xagent.web.api.kb.get_metadata_store",
            return_value=mock_metadata_store,
        ),
        patch("xagent.web.api.kb.rename_collection_storage") as mock_physical,
        patch("xagent.web.api.kb.get_vector_index_store") as mock_vector_factory,
    ):
        response = client.put(
            f"/api/kb/collections/{old_name}",
            data={"new_name": new_name},
            headers=headers,
        )

    assert response.status_code == 409, response.text
    detail = response.json()["detail"].lower()
    assert "shared" in detail or "multiple" in detail
    mock_physical.assert_not_called()
    mock_vector_factory.return_value.rename_collection_data.assert_not_called()
    mock_metadata_store.rename_collection.assert_not_called()


def test_kb_rename_non_admin_succeeds_when_sole_collection_occupant(
    rename_users_env,
) -> None:
    """Non-admin may rename when they are the only config occupant for the old name."""
    app, headers, user, _, _, _, _ = rename_users_env
    client = TestClient(app)
    old_name = "solo_project"
    new_name = "solo_project_new"

    mock_metadata_store = MagicMock()
    mock_metadata_store.count_users_with_collection_config = AsyncMock(return_value=1)
    mock_metadata_store.rename_collection = AsyncMock()

    mock_vector_store = MagicMock()
    mock_vector_store.list_document_records.return_value = []
    mock_vector_store.rename_collection_data.return_value = []

    mock_rename_result = MagicMock()
    mock_rename_result.status = "not_found"
    mock_rename_result.error = None
    mock_rename_result.old_collection_dir = None
    mock_rename_result.new_collection_dir = None

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new_callable=AsyncMock,
            return_value=_visible_only_old(old_name),
        ),
        patch(
            "xagent.web.api.kb.get_metadata_store",
            return_value=mock_metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_vector_index_store",
            return_value=mock_vector_store,
        ),
        patch(
            "xagent.web.api.kb.rename_collection_storage",
            return_value=mock_rename_result,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.load_ingestion_status",
            return_value=[],
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.write_ingestion_status",
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.clear_ingestion_status",
        ),
    ):
        response = client.put(
            f"/api/kb/collections/{old_name}",
            data={"new_name": new_name},
            headers=headers,
        )

    assert response.status_code == 200, response.text
    mock_vector_store.rename_collection_data.assert_called_once_with(
        collection_name=old_name,
        new_name=new_name,
        user_id=int(user.id),
        is_admin=False,
    )
    mock_metadata_store.rename_collection.assert_awaited_once_with(
        old_name=old_name,
        new_name=new_name,
        user_id=int(user.id),
        is_admin=False,
    )


def test_kb_rename_admin_requires_target_user_id(rename_users_env) -> None:
    """Admin rename must specify which user's collection scope to update."""
    app, _, _, admin_headers, _, _, _ = rename_users_env
    client = TestClient(app)

    with patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock):
        response = client.put(
            "/api/kb/collections/admin_old",
            data={"new_name": "admin_new"},
            headers=admin_headers,
        )

    assert response.status_code == 422, response.text
    assert "target_user_id" in response.json()["detail"].lower()


def test_kb_rename_admin_scopes_physical_vector_and_metadata_to_target_user(
    rename_users_env,
) -> None:
    """Admin passes target_user_id; all rename side effects use that user only."""
    app, _, _, admin_headers, _, target_user, _ = rename_users_env
    client = TestClient(app)
    old_name = "admin_scope_old"
    new_name = "admin_scope_new"
    target_user_id = int(target_user.id)

    mock_metadata_store = MagicMock()
    mock_metadata_store.count_users_with_collection_config = AsyncMock(return_value=2)
    mock_metadata_store.rename_collection = AsyncMock()

    mock_vector_store = MagicMock()
    mock_vector_store.list_document_records.return_value = []
    mock_vector_store.rename_collection_data.return_value = []

    mock_rename_result = MagicMock()
    mock_rename_result.status = "not_found"
    mock_rename_result.error = None
    mock_rename_result.old_collection_dir = None
    mock_rename_result.new_collection_dir = None

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new_callable=AsyncMock,
            return_value=_visible_only_old(old_name),
        ),
        patch(
            "xagent.web.api.kb.get_metadata_store",
            return_value=mock_metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_vector_index_store",
            return_value=mock_vector_store,
        ),
        patch(
            "xagent.web.api.kb.rename_collection_storage",
            return_value=mock_rename_result,
        ) as mock_physical,
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.load_ingestion_status",
            return_value=[],
        ) as mock_load_status,
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.write_ingestion_status",
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.clear_ingestion_status",
        ),
    ):
        response = client.put(
            f"/api/kb/collections/{old_name}",
            data={"new_name": new_name, "target_user_id": str(target_user_id)},
            headers=admin_headers,
        )

    assert response.status_code == 200, response.text
    mock_physical.assert_called_once()
    assert mock_physical.call_args.kwargs["user_id"] == target_user_id
    mock_vector_store.rename_collection_data.assert_called_once_with(
        collection_name=old_name,
        new_name=new_name,
        user_id=target_user_id,
        is_admin=False,
    )
    mock_metadata_store.rename_collection.assert_awaited_once_with(
        old_name=old_name,
        new_name=new_name,
        user_id=target_user_id,
        is_admin=True,
    )
    mock_load_status.assert_called_once_with(
        collection=old_name,
        user_id=target_user_id,
        is_admin=False,
    )


def test_kb_rename_migrates_ingestion_status_only_for_scoped_user(
    rename_users_env,
) -> None:
    """Ingestion status migration must not pull other users' rows for the old name."""
    app, headers, user, _, _, _, _ = rename_users_env
    client = TestClient(app)
    old_name = "status_scope_old"
    new_name = "status_scope_new"

    mock_metadata_store = MagicMock()
    mock_metadata_store.count_users_with_collection_config = AsyncMock(return_value=1)
    mock_metadata_store.rename_collection = AsyncMock()

    mock_vector_store = MagicMock()
    mock_vector_store.list_document_records.return_value = []
    mock_vector_store.rename_collection_data.return_value = []

    mock_rename_result = MagicMock()
    mock_rename_result.status = "not_found"
    mock_rename_result.error = None
    mock_rename_result.old_collection_dir = None
    mock_rename_result.new_collection_dir = None

    status_rows = [
        {
            "collection": old_name,
            "doc_id": "doc-a",
            "status": "success",
            "message": "",
            "parse_hash": "h1",
            "user_id": int(user.id),
        }
    ]

    with (
        patch("xagent.web.api.kb._ensure_collection_access", new_callable=AsyncMock),
        patch(
            "xagent.web.api.kb._list_collections_with_retry",
            new_callable=AsyncMock,
            return_value=_visible_only_old(old_name),
        ),
        patch(
            "xagent.web.api.kb.get_metadata_store",
            return_value=mock_metadata_store,
        ),
        patch(
            "xagent.web.api.kb.get_vector_index_store",
            return_value=mock_vector_store,
        ),
        patch(
            "xagent.web.api.kb.rename_collection_storage",
            return_value=mock_rename_result,
        ),
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.load_ingestion_status",
            return_value=status_rows,
        ) as mock_load_status,
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.write_ingestion_status",
        ) as mock_write_status,
        patch(
            "xagent.core.tools.core.RAG_tools.management.status.clear_ingestion_status",
        ) as mock_clear_status,
    ):
        response = client.put(
            f"/api/kb/collections/{old_name}",
            data={"new_name": new_name},
            headers=headers,
        )

    assert response.status_code == 200, response.text
    mock_load_status.assert_called_once_with(
        collection=old_name,
        user_id=int(user.id),
        is_admin=False,
    )
    mock_write_status.assert_called_once()
    mock_clear_status.assert_called_once_with(
        old_name,
        "doc-a",
        user_id=int(user.id),
        is_admin=False,
    )
