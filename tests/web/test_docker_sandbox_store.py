"""Test Docker sandbox store"""

import json
import subprocess
import sys
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.sandbox import (
    DEFAULT_SANDBOX_IMAGE,
    SandboxConfig,
    SandboxInfo,
    SandboxSnapshot,
    SandboxTemplate,
)
from xagent.web.models.database import Base
from xagent.web.models.sandbox import SandboxInfo as SandboxInfoModel
from xagent.web.models.sandbox import SandboxSnapshot as SandboxSnapshotModel
from xagent.web.sandbox_store import SANDBOX_TYPE_DOCKER, DBDockerStore

# Test database setup - using in-memory database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create database session"""
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    yield db
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def store(db_session):
    """Create DBDockerStore instance and mock _get_db_session"""
    store = DBDockerStore()
    # Mock _get_db_session method to return test database session
    store._get_db_session = lambda: db_session
    return store


@pytest.fixture(scope="function")
def sample_sandbox_info():
    """Create sample SandboxInfo with all config fields"""
    template = SandboxTemplate(
        type="image",
        image=DEFAULT_SANDBOX_IMAGE,
    )
    config = SandboxConfig(
        working_dir="/workspace",
        cpus=2,
        memory=1024,
        env={"PYTHONPATH": "/app", "DEBUG": "true"},
        volumes=[("/host/data", "/container/data", "rw")],
        network_isolated=True,
        ports=[(8080, 80), (8443, 443)],
    )
    return SandboxInfo(
        name="test-sandbox",
        state="running",
        template=template,
        config=config,
        created_at=datetime.now().isoformat(),
    )


@pytest.fixture(scope="function")
def sample_snapshot():
    """Create sample SandboxSnapshot metadata."""
    return SandboxSnapshot(
        snapshot_id="test-snapshot",
        metadata={
            "image_tag": "xagent-sandbox-snapshot:test-snapshot",
            "image_id": "sha256:test-image",
        },
        created_at=datetime.now().isoformat(),
    )


class TestDBDockerStore:
    """Test DBDockerStore functionality."""

    def test_add_info_new(self, db_session, store, sample_sandbox_info):
        """Test adding new sandbox info"""
        store.add_info("test-sandbox", sample_sandbox_info)

        # Verify record exists in database
        model = (
            db_session.query(SandboxInfoModel)
            .filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model is not None
        assert model.name == "test-sandbox"
        assert model.state == "running"
        assert model.sandbox_type == SANDBOX_TYPE_DOCKER

        # Verify template and config JSON
        template_data = json.loads(model.template)
        assert template_data["type"] == "image"
        assert template_data["image"] == DEFAULT_SANDBOX_IMAGE

        config_data = json.loads(model.config)
        assert config_data["cpus"] == 2
        assert config_data["memory"] == 1024

    def test_add_info_update_existing(self, db_session, store, sample_sandbox_info):
        """Test updating existing sandbox info"""
        # Add first
        store.add_info("test-sandbox", sample_sandbox_info)

        # Modify and update
        sample_sandbox_info.state = "stopped"
        sample_sandbox_info.config.cpus = 4
        store.add_info("test-sandbox", sample_sandbox_info)

        # Verify update
        model = (
            db_session.query(SandboxInfoModel)
            .filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model.state == "stopped"
        config_data = json.loads(model.config)
        assert config_data["cpus"] == 4

    def test_update_info_state(self, db_session, store, sample_sandbox_info):
        """Test updating sandbox state"""
        # Add first
        store.add_info("test-sandbox", sample_sandbox_info)

        # Update state
        store.update_info_state("test-sandbox", "stopped")

        # Verify
        model = (
            db_session.query(SandboxInfoModel)
            .filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model.state == "stopped"

    def test_delete_info(self, db_session, store, sample_sandbox_info):
        """Test deleting sandbox info"""
        # Add first
        store.add_info("test-sandbox", sample_sandbox_info)

        # Delete
        store.delete_info("test-sandbox")

        # Verify deleted
        model = (
            db_session.query(SandboxInfoModel)
            .filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model is None

    def test_get_snapshot(self, db_session, store, sample_snapshot):
        """Test getting persisted snapshot info."""
        store.add_snapshot(sample_snapshot)

        snapshot = store.get_snapshot("test-snapshot")

        assert snapshot is not None
        assert snapshot.snapshot_id == "test-snapshot"
        assert snapshot.metadata == sample_snapshot.metadata
        assert snapshot.created_at == sample_snapshot.created_at

    def test_add_snapshot_update_existing(self, db_session, store, sample_snapshot):
        """Test updating an existing snapshot record."""
        store.add_snapshot(sample_snapshot)

        updated_snapshot = sample_snapshot.model_copy(
            update={
                "metadata": {
                    "image_tag": "xagent-sandbox-snapshot:test-snapshot-v2",
                    "image_id": "sha256:updated-image",
                }
            }
        )
        store.add_snapshot(updated_snapshot)

        model = (
            db_session.query(SandboxSnapshotModel)
            .filter(
                SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxSnapshotModel.snapshot_id == "test-snapshot",
            )
            .first()
        )
        assert model is not None
        assert json.loads(model.metadata_json)["image_tag"] == (
            "xagent-sandbox-snapshot:test-snapshot-v2"
        )

    def test_list_snapshots(self, store):
        """Test listing snapshots ordered by snapshot ID."""
        snapshot_b = SandboxSnapshot(
            snapshot_id="snapshot-b",
            metadata={"image_tag": "xagent-sandbox-snapshot:snapshot-b"},
            created_at=datetime.now().isoformat(),
        )
        snapshot_a = SandboxSnapshot(
            snapshot_id="snapshot-a",
            metadata={"image_tag": "xagent-sandbox-snapshot:snapshot-a"},
            created_at=datetime.now().isoformat(),
        )

        store.add_snapshot(snapshot_b)
        store.add_snapshot(snapshot_a)

        snapshots = store.list_snapshots()

        assert [snapshot.snapshot_id for snapshot in snapshots] == [
            "snapshot-a",
            "snapshot-b",
        ]

    def test_delete_snapshot(self, db_session, store, sample_snapshot):
        """Test deleting snapshot metadata."""
        store.add_snapshot(sample_snapshot)

        store.delete_snapshot("test-snapshot")

        model = (
            db_session.query(SandboxSnapshotModel)
            .filter(
                SandboxSnapshotModel.sandbox_type == SANDBOX_TYPE_DOCKER,
                SandboxSnapshotModel.snapshot_id == "test-snapshot",
            )
            .first()
        )
        assert model is None


def test_sandbox_store_imports_without_boxlite():
    code = (
        "import sys; sys.modules['boxlite'] = None\n"
        "import xagent.web.sandbox_store as m\n"
        "assert m.DBDockerStore\n"
        "assert not hasattr(m, 'DBBoxliteStore')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
