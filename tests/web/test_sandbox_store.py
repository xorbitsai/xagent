"""Test sandbox store functionality"""

import json
from datetime import datetime

import pytest

try:
    import boxlite  # noqa: F401
except ImportError:
    pytest.skip(
        "boxlite not installed, skipping sandbox store tests", allow_module_level=True
    )

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.sandbox import SandboxConfig, SandboxInfo, SandboxTemplate
from xagent.web.models.database import Base
from xagent.web.models.sandbox import SandboxInfo as SandboxInfoModel
from xagent.web.sandbox_store import SANDBOX_TYPE_BOXLITE, DBBoxliteStore

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
    """Create DBBoxliteStore instance and mock _get_db_session"""
    store = DBBoxliteStore()
    # Mock _get_db_session method to return test database session
    store._get_db_session = lambda: db_session
    return store


@pytest.fixture(scope="function")
def sample_sandbox_info():
    """Create sample SandboxInfo with all config fields"""
    template = SandboxTemplate(
        type="image",
        image="python:3.11-slim",
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


class TestDBBoxliteStore:
    """Test DBBoxliteStore functionality"""

    def test_add_info_new(self, db_session, store, sample_sandbox_info):
        """Test adding new sandbox info"""
        store.add_info("test-sandbox", sample_sandbox_info)

        # Verify record exists in database
        model = (
            db_session.query(SandboxInfoModel)
            .filter(
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model is not None
        assert model.name == "test-sandbox"
        assert model.state == "running"
        assert model.sandbox_type == SANDBOX_TYPE_BOXLITE

        # Verify template and config JSON
        template_data = json.loads(model.template)
        assert template_data["type"] == "image"
        assert template_data["image"] == "python:3.11-slim"

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
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model.state == "stopped"
        config_data = json.loads(model.config)
        assert config_data["cpus"] == 4

    def test_get_info_exists(self, db_session, store, sample_sandbox_info):
        """Test getting existing sandbox info"""
        # Add first
        store.add_info("test-sandbox", sample_sandbox_info)

        # Get
        info = store.get_info("test-sandbox")

        assert info is not None
        assert info.name == "test-sandbox"
        assert info.state == "running"
        assert info.template.type == "image"
        assert info.template.image == "python:3.11-slim"
        assert info.config.cpus == 2
        assert info.config.memory == 1024
        assert info.config.env == {"PYTHONPATH": "/app", "DEBUG": "true"}

    def test_get_info_not_exists(self, db_session, store):
        """Test getting non-existent sandbox info"""
        info = store.get_info("non-existent")
        assert info is None

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
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model.state == "stopped"

    def test_update_info_state_not_exists(self, db_session, store):
        """Test updating non-existent sandbox state (should not raise error)"""
        # Should not raise exception
        store.update_info_state("non-existent", "stopped")

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
                SandboxInfoModel.sandbox_type == SANDBOX_TYPE_BOXLITE,
                SandboxInfoModel.name == "test-sandbox",
            )
            .first()
        )
        assert model is None

    def test_delete_info_not_exists(self, db_session, store):
        """Test deleting non-existent sandbox info (should not raise error)"""
        # Should not raise exception
        store.delete_info("non-existent")

    def test_model_to_info_conversion(self, db_session, store, sample_sandbox_info):
        """Test database model to SandboxInfo conversion"""
        # Add and get
        store.add_info("test-sandbox", sample_sandbox_info)
        info = store.get_info("test-sandbox")

        # Verify all fields are correctly converted
        assert isinstance(info, SandboxInfo)
        assert isinstance(info.template, SandboxTemplate)
        assert isinstance(info.config, SandboxConfig)
        assert info.name == sample_sandbox_info.name
        assert info.state == sample_sandbox_info.state
        assert info.template.type == sample_sandbox_info.template.type
        assert info.template.image == sample_sandbox_info.template.image
        assert info.config.cpus == sample_sandbox_info.config.cpus
        assert info.config.memory == sample_sandbox_info.config.memory

    def test_all_config_fields_saved(self, db_session, store, sample_sandbox_info):
        """Test all config fields are correctly saved and retrieved"""
        # Add
        store.add_info("test-sandbox", sample_sandbox_info)

        # Get
        info = store.get_info("test-sandbox")

        # Verify all config fields
        assert info is not None
        assert info.config.working_dir == "/workspace"
        assert info.config.cpus == 2
        assert info.config.memory == 1024
        assert info.config.env == {"PYTHONPATH": "/app", "DEBUG": "true"}
        assert info.config.volumes == [("/host/data", "/container/data", "rw")]
        assert info.config.network_isolated is True
        assert info.config.ports == [(8080, 80), (8443, 443)]
