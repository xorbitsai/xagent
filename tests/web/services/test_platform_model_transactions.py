from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.model.model import ChatModelConfig, EmbeddingModelConfig
from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import UserDefaultModel, UserModel
from xagent.web.services.llm_utils import (
    PLATFORM_MODEL_MANAGER,
    CoreStorage,
    ModelWriteMode,
    PlatformModelIdentityError,
    PlatformModelStore,
)
from xagent.web.services.model_store import ModelStore


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'models.db'}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False)
    try:
        yield session_factory
    finally:
        engine.dispose()


def _config(model_id: str, *, embedding: bool = False):
    common = {
        "id": model_id,
        "model_provider": "openai",
        "api_key": "platform-key",
        "base_url": "https://api.openai.com/v1",
    }
    if embedding:
        return EmbeddingModelConfig(
            **common,
            model_name="text-embedding-3-small",
            abilities=["embedding"],
            dimension=1536,
        )
    return ChatModelConfig(**common, model_name="gpt-4", abilities=["chat"])


def test_default_platform_create_remains_active_and_committed(sessions):
    with sessions() as db, sessions() as observer:
        created = PlatformModelStore(db).create(_config("platform/default-commit"))

        observed = observer.query(Model).filter_by(id=created.id).one()
        assert observed.is_active is True
        assert observed.managed_by == PLATFORM_MODEL_MANAGER


def test_staged_catalog_commits_model_link_and_default_together(sessions):
    with sessions() as db, sessions() as observer:
        created = PlatformModelStore(db).create(
            _config("platform/staged-embedding", embedding=True),
            is_active=False,
            write_mode=ModelWriteMode.STAGE,
        )
        assert created.id is not None
        assert created.is_active is False
        assert observer.query(Model).filter_by(id=created.id).first() is None

        model_store = ModelStore(db)
        link = model_store.create_user_model_link(
            user_id=1,
            model_id=created.id,
            is_shared=True,
            write_mode=ModelWriteMode.STAGE,
        )
        default = model_store.set_user_default_model(
            user_id=1,
            model_id=created.id,
            config_type="embedding",
            user_model=link,
            write_mode=ModelWriteMode.STAGE,
        )
        assert link.id is not None
        assert default.id is not None

        with patch(
            "xagent.web.services.model_store.invalidate_model_cache"
        ) as invalidate:
            db.commit()
            invalidate.assert_called_once_with(None)

        observer.close()
        with sessions() as committed_observer:
            assert (
                committed_observer.query(Model).filter_by(id=created.id).one().is_active
                is False
            )
            assert committed_observer.query(UserModel).filter_by(id=link.id).one()
            assert (
                committed_observer.query(UserDefaultModel)
                .filter_by(id=default.id)
                .one()
            )


def test_staged_catalog_rollback_discards_everything(sessions):
    with sessions() as db:
        created = PlatformModelStore(db).create(
            _config("platform/rolled-back", embedding=True),
            is_active=False,
            write_mode=ModelWriteMode.STAGE,
        )
        model_store = ModelStore(db)
        link = model_store.create_user_model_link(
            user_id=1,
            model_id=created.id,
            is_shared=False,
            write_mode=ModelWriteMode.STAGE,
        )
        model_store.set_user_default_model(
            user_id=1,
            model_id=created.id,
            config_type="embedding",
            user_model=link,
            write_mode=ModelWriteMode.STAGE,
        )

        with patch(
            "xagent.web.services.model_store.invalidate_model_cache"
        ) as invalidate:
            try:
                raise RuntimeError("injected catalog failure")
            except RuntimeError:
                db.rollback()
            invalidate.assert_not_called()

        assert (
            db.query(Model).filter_by(model_id="platform/rolled-back").first() is None
        )
        assert db.query(UserModel).filter_by(model_id=created.id).first() is None
        assert db.query(UserDefaultModel).filter_by(model_id=created.id).first() is None


def test_link_and_default_helpers_commit_by_default(sessions):
    with sessions() as db, sessions() as observer:
        created = PlatformModelStore(db).create(
            _config("platform/helper-compatibility")
        )
        model_store = ModelStore(db)
        link = model_store.create_user_model_link(
            user_id=1, model_id=created.id, is_shared=False
        )
        default = model_store.set_user_default_model(
            user_id=1,
            model_id=created.id,
            config_type="general",
            user_model=link,
        )

        assert observer.query(UserModel).filter_by(id=link.id).one()
        assert observer.query(UserDefaultModel).filter_by(id=default.id).one()


def test_ordinary_store_cannot_forge_or_mutate_platform_identity(sessions):
    with sessions() as db:
        ordinary_store = CoreStorage(db, Model)
        with pytest.raises(PlatformModelIdentityError, match="reserved"):
            ordinary_store.store(_config("platform/forged"))

        created = PlatformModelStore(db).create(_config("platform/protected"))
        with pytest.raises(PlatformModelIdentityError, match="ordinary"):
            ordinary_store.set_model_active(created.model_id, False)
