import pytest
from cryptography.fernet import Fernet
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from xagent.core.model.model import (
    ChatModelConfig,
    VectorDBConfig,
    VectorDBType,
)
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table

Base = declarative_base()
Model = create_model_table(Base)


@pytest.fixture(autouse=True)
def setup_encryption_key(monkeypatch):
    # Set a valid Base64-encoded 32-byte key for Fernet encryption
    key = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
    monkeypatch.setenv("ENCRYPTION_KEY", key)
    return key


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


def test_vector_db_config_roundtrip(db_session):
    hub = SQLAlchemyModelHub(db_session, Model)

    # 1. Store a VectorDBConfig with extra config
    config_id = "test-weaviate"
    vdb_config = VectorDBConfig(
        id=config_id,
        model_name="My Weaviate",
        base_url="http://localhost:8080",
        api_key="test-key",
        db_type=VectorDBType.WEAVIATE_SAAS,
        config={"grpc_port": 50051, "secure": True},
    )

    hub.store(vdb_config)

    # 2. Load it back
    loaded = hub.load(config_id)

    assert isinstance(loaded, VectorDBConfig)
    assert loaded.id == config_id
    assert loaded.db_type == VectorDBType.WEAVIATE_SAAS
    assert loaded.config["grpc_port"] == 50051
    assert loaded.config["secure"] is True
    assert loaded.api_key == "test-key"


def test_vector_db_config_list(db_session):
    hub = SQLAlchemyModelHub(db_session, Model)

    # Store multiple configs (must provide api_key to avoid IntegrityError on _api_key_encrypted)
    hub.store(
        VectorDBConfig(
            id="v1", model_name="V1", db_type=VectorDBType.LANCEDB, api_key=""
        )
    )
    hub.store(
        VectorDBConfig(
            id="v2",
            model_name="V2",
            db_type=VectorDBType.WEAVIATE_SAAS,
            config={"k": "v"},
            api_key="",
        )
    )

    configs = hub.list()
    assert len(configs) >= 2
    assert isinstance(configs["v1"], VectorDBConfig)
    assert isinstance(configs["v2"], VectorDBConfig)
    assert configs["v2"].config["k"] == "v"


def test_vector_db_config_fallback(db_session, setup_encryption_key):
    hub = SQLAlchemyModelHub(db_session, Model)

    # Generate a valid encrypted token
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"dummy-key").decode()

    # Manually insert a record with invalid db_type in model_provider
    db_record = Model(
        model_id="invalid-db",
        category="vector_db",
        model_provider="unknown_db",
        model_name="Unknown",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()

    # Should fallback to LANCEDB and not crash
    loaded = hub.load("invalid-db")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.db_type == VectorDBType.LANCEDB
    assert loaded.api_key == "dummy-key"


def test_vector_db_legacy_weaviate_local_alias(db_session, setup_encryption_key):
    hub = SQLAlchemyModelHub(db_session, Model)

    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"dummy-key").decode()

    # Legacy provider value should be mapped to the new canonical enum value.
    db_record = Model(
        model_id="legacy-weaviate-local",
        category="vector_db",
        model_provider="weaviate_local",
        model_name="LegacyWeaviateLocal",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()

    loaded = hub.load("legacy-weaviate-local")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.db_type == VectorDBType.WEAVIATE


def test_vector_db_config_with_other_models(db_session):
    """list() returns correct mix of ChatModelConfig and VectorDBConfig."""
    hub = SQLAlchemyModelHub(db_session, Model)
    hub.store(
        ChatModelConfig(
            id="chat1",
            model_name="GPT-4",
            model_provider="openai",
            api_key="sk-test",
        )
    )
    hub.store(
        VectorDBConfig(
            id="v1",
            model_name="V1",
            db_type=VectorDBType.LANCEDB,
            api_key="",
        )
    )
    configs = hub.list()
    assert "chat1" in configs
    assert "v1" in configs
    assert isinstance(configs["chat1"], ChatModelConfig)
    assert isinstance(configs["v1"], VectorDBConfig)
    assert configs["chat1"].model_name == "GPT-4"
    assert configs["v1"].db_type == VectorDBType.LANCEDB


def test_vector_db_config_empty_config(db_session):
    """Roundtrip with config={} or no extra config."""
    hub = SQLAlchemyModelHub(db_session, Model)
    hub.store(
        VectorDBConfig(
            id="empty-cfg",
            model_name="Empty",
            db_type=VectorDBType.CHROMADB,
            config={},
            api_key="",
        )
    )
    loaded = hub.load("empty-cfg")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.config == {}
    assert loaded.db_type == VectorDBType.CHROMADB


def test_vector_db_config_non_dict_abilities(db_session, setup_encryption_key):
    """When DB has non-dict abilities for vector_db, config becomes {} and no crash."""
    hub = SQLAlchemyModelHub(db_session, Model)
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"key").decode()
    db_record = Model(
        model_id="bad-abilities",
        category="vector_db",
        model_provider="lancedb",
        model_name="Bad",
        _api_key_encrypted=valid_encrypted,
        abilities=["list", "not", "dict"],  # wrong type for vector_db config
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()
    loaded = hub.load("bad-abilities")
    assert isinstance(loaded, VectorDBConfig)
    assert loaded.config == {}
    assert loaded.db_type == VectorDBType.LANCEDB


def test_vector_db_config_delete(db_session):
    """delete() removes vector_db config; load() then raises."""
    hub = SQLAlchemyModelHub(db_session, Model)
    hub.store(
        VectorDBConfig(
            id="to-delete",
            model_name="Del",
            db_type=VectorDBType.LANCEDB,
            api_key="",
        )
    )
    assert hub.exists("to-delete")
    hub.delete("to-delete")
    assert not hub.exists("to-delete")
    with pytest.raises(ValueError, match="Model not found"):
        hub.load("to-delete")


def test_vector_db_config_normalize_case(db_session, setup_encryption_key):
    """Provider value is normalized to lowercase (e.g. LanceDB -> lancedb)."""
    hub = SQLAlchemyModelHub(db_session, Model)
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"k").decode()
    db_record = Model(
        model_id="case-test",
        category="vector_db",
        model_provider="LanceDB",
        model_name="Case",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()
    loaded = hub.load("case-test")
    assert loaded.db_type == VectorDBType.LANCEDB


def test_vector_db_config_normalize_whitespace_legacy(db_session, setup_encryption_key):
    """Legacy value with whitespace (e.g. ' weaviate_local ') normalizes to weaviate."""
    hub = SQLAlchemyModelHub(db_session, Model)
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"k").decode()
    db_record = Model(
        model_id="ws-test",
        category="vector_db",
        model_provider=" weaviate_local ",
        model_name="WS",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()
    loaded = hub.load("ws-test")
    assert loaded.db_type == VectorDBType.WEAVIATE


def test_vector_db_config_empty_string_provider(db_session, setup_encryption_key):
    """Empty or whitespace-only model_provider falls back to lancedb."""
    hub = SQLAlchemyModelHub(db_session, Model)
    cipher = Fernet(setup_encryption_key.encode())
    valid_encrypted = cipher.encrypt(b"k").decode()
    db_record = Model(
        model_id="empty-provider",
        category="vector_db",
        model_provider="",
        model_name="EmptyProvider",
        _api_key_encrypted=valid_encrypted,
        is_active=True,
    )
    db_session.add(db_record)
    db_session.commit()
    loaded = hub.load("empty-provider")
    assert loaded.db_type == VectorDBType.LANCEDB
