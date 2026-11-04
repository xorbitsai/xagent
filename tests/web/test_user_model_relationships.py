"""Test user-model relationship functionality"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User, UserDefaultModel, UserModel

# Test database setup - use in-memory database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def test_db():
    """Create test database"""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def db_session(test_db):
    """Create database session"""
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="function")
def admin_user(db_session):
    """Create admin user"""
    user = User(username="admin", password_hash="hashed_admin_pass", is_admin=True)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def regular_user(db_session):
    """Create regular user"""
    user = User(username="regularuser", password_hash="hashed_pass", is_admin=False)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture(scope="function")
def sample_model(db_session):
    """Create sample model"""
    model = Model(
        model_id="test-model",
        category="llm",
        model_provider="openai",
        model_name="gpt-4",
        api_key="test-key",
        abilities=["chat", "tool_calling"],
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


class TestUserModelRelationships:
    """Test user-model relationship functionality"""

    def test_create_user_model_relationship(
        self, db_session, regular_user, sample_model
    ):
        """Test creating user-model relationship"""
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)
        db_session.commit()
        db_session.refresh(user_model)

        assert user_model.user_id == regular_user.id
        assert user_model.model_id == sample_model.id
        assert user_model.is_owner is True
        assert user_model.can_edit is True
        assert user_model.can_delete is True
        assert user_model.is_shared is False

    def test_model_relationships(self, db_session, regular_user, sample_model):
        """Test model relationships"""
        # Create user-model relationship
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)
        db_session.commit()

        # Test relationships
        assert len(sample_model.user_models) == 1
        assert sample_model.user_models[0].user_id == regular_user.id
        assert len(regular_user.user_models) == 1
        assert regular_user.user_models[0].model_id == sample_model.id

    def test_user_default_model_configuration(
        self, db_session, regular_user, sample_model
    ):
        """Test user default model configuration"""
        # Create user-model relationship first
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)

        # Create default configuration
        default_config = UserDefaultModel(
            user_id=regular_user.id, model_id=sample_model.id, config_type="general"
        )
        db_session.add(default_config)
        db_session.commit()

        # Verify configuration
        assert default_config.user_id == regular_user.id
        assert default_config.model_id == sample_model.id
        assert default_config.config_type == "general"

    def test_multiple_default_configurations(
        self, db_session, regular_user, sample_model
    ):
        """Test multiple default configurations for different model types"""
        # Create user-model relationship
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)

        # Create multiple default configurations
        configs = [
            UserDefaultModel(
                user_id=regular_user.id, model_id=sample_model.id, config_type="general"
            ),
            UserDefaultModel(
                user_id=regular_user.id,
                model_id=sample_model.id,
                config_type="small_fast",
            ),
            UserDefaultModel(
                user_id=regular_user.id, model_id=sample_model.id, config_type="visual"
            ),
            UserDefaultModel(
                user_id=regular_user.id, model_id=sample_model.id, config_type="compact"
            ),
        ]

        for config in configs:
            db_session.add(config)
        db_session.commit()

        # Verify all configurations exist
        db_configs = (
            db_session.query(UserDefaultModel)
            .filter(UserDefaultModel.user_id == regular_user.id)
            .all()
        )
        assert len(db_configs) == 4

        config_types = [config.config_type for config in db_configs]
        assert "general" in config_types
        assert "small_fast" in config_types
        assert "visual" in config_types
        assert "compact" in config_types

    def test_user_model_uniqueness(self, db_session, regular_user, sample_model):
        """Test that user-model relationship is unique"""
        # Create first relationship
        user_model1 = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model1)
        db_session.commit()

        # Try to create duplicate relationship
        user_model2 = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=False,
            can_edit=False,
            can_delete=False,
            is_shared=False,
        )
        db_session.add(user_model2)

        # This should raise an integrity error due to unique constraint
        with pytest.raises(Exception):
            db_session.commit()

    def test_default_model_uniqueness(self, db_session, regular_user, sample_model):
        """Test that default model configuration is unique per user and type"""
        # Create user-model relationship
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)

        # Create first default configuration
        default_config1 = UserDefaultModel(
            user_id=regular_user.id, model_id=sample_model.id, config_type="general"
        )
        db_session.add(default_config1)
        db_session.commit()

        # Try to create duplicate configuration
        default_config2 = UserDefaultModel(
            user_id=regular_user.id, model_id=sample_model.id, config_type="general"
        )
        db_session.add(default_config2)

        # This should raise an integrity error due to unique constraint
        with pytest.raises(Exception):
            db_session.commit()

    def test_cascade_delete_user(self, db_session, regular_user, sample_model):
        """Test that deleting user cascades to relationships"""
        # Create user-model relationship
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)

        # Create default configuration
        default_config = UserDefaultModel(
            user_id=regular_user.id, model_id=sample_model.id, config_type="general"
        )
        db_session.add(default_config)
        db_session.commit()

        # Delete user
        db_session.delete(regular_user)
        db_session.commit()

        # Verify relationships are deleted
        remaining_user_models = (
            db_session.query(UserModel)
            .filter(UserModel.user_id == regular_user.id)
            .all()
        )
        assert len(remaining_user_models) == 0

        remaining_defaults = (
            db_session.query(UserDefaultModel)
            .filter(UserDefaultModel.user_id == regular_user.id)
            .all()
        )
        assert len(remaining_defaults) == 0

    def test_cascade_delete_model(self, db_session, regular_user, sample_model):
        """Test that deleting model cascades to relationships"""
        # Create user-model relationship
        user_model = UserModel(
            user_id=regular_user.id,
            model_id=sample_model.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=False,
        )
        db_session.add(user_model)

        # Create default configuration
        default_config = UserDefaultModel(
            user_id=regular_user.id, model_id=sample_model.id, config_type="general"
        )
        db_session.add(default_config)
        db_session.commit()

        # Delete model
        db_session.delete(sample_model)
        db_session.commit()

        # Verify relationships are deleted
        remaining_user_models = (
            db_session.query(UserModel)
            .filter(UserModel.model_id == sample_model.id)
            .all()
        )
        assert len(remaining_user_models) == 0

        remaining_defaults = (
            db_session.query(UserDefaultModel)
            .filter(UserDefaultModel.model_id == sample_model.id)
            .all()
        )
        assert len(remaining_defaults) == 0

    def test_user_model_query_filters(
        self, db_session, admin_user, regular_user, sample_model
    ):
        """Test querying user models with various filters"""
        # Create multiple relationships
        relationships = [
            UserModel(
                user_id=admin_user.id,
                model_id=sample_model.id,
                is_owner=True,
                can_edit=True,
                can_delete=True,
                is_shared=True,
            ),
            UserModel(
                user_id=regular_user.id,
                model_id=sample_model.id,
                is_owner=False,
                can_edit=False,
                can_delete=False,
                is_shared=True,
            ),
        ]

        for rel in relationships:
            db_session.add(rel)
        db_session.commit()

        # Test filtering by owner
        owner_models = db_session.query(UserModel).filter(UserModel.is_owner).all()
        assert len(owner_models) == 1
        assert owner_models[0].user_id == admin_user.id

        # Test filtering by shared models
        shared_models = db_session.query(UserModel).filter(UserModel.is_shared).all()
        assert len(shared_models) == 2

        # Test filtering by editable models
        editable_models = db_session.query(UserModel).filter(UserModel.can_edit).all()
        assert len(editable_models) == 1
        assert editable_models[0].user_id == admin_user.id
