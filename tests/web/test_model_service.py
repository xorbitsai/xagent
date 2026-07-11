"""Test model service functionality"""

from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.web.models.database import Base
from xagent.web.models.model import Model
from xagent.web.models.user import User
from xagent.web.services.model_service import (
    _is_model_visible_to_user,
    get_asr_models,
    get_default_model,
    get_default_music_model,
    get_default_sound_effect_model,
    get_image_models,
    get_music_models,
    get_sound_effect_models,
    get_tts_models,
    get_vision_model,
)

# Test database setup - use in-memory database
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
        model_id="test-openai-model",
        category="llm",
        model_provider="openai",
        model_name="gpt-4",
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
        temperature=0.7,
        abilities=["chat", "tool_calling"],
    )
    db_session.add(model)
    db_session.commit()
    db_session.refresh(model)
    return model


class TestModelService:
    """Test model service functionality"""

    def test_get_default_model_user_specific(self):
        """Test getting user-specific default model"""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Create mock objects
            mock_user_default = MagicMock()
            mock_user_default.user_id = 1
            mock_user_default.config_type = "general"
            mock_user_default.model = MagicMock()
            mock_user_default.model.model_id = "test-model"

            # Setup query result
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = mock_user_default

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(1)

            assert result == mock_llm
            mock_create.assert_called_once_with(mock_user_default.model)

    def test_get_default_model_admin_shared(self):
        """Test getting admin shared default model — admin fallback uses _get_visible_user_ids."""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
            patch(
                "xagent.web.services.model_service._get_visible_user_ids"
            ) as mock_visible,
        ):
            # _get_visible_user_ids returns admin user IDs
            mock_visible.return_value = [1]

            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Setup query to return None for user-specific query and shared default for shared query
            mock_query_result = MagicMock()
            mock_query_result.first.return_value = None
            mock_query_result.all.return_value = [MagicMock()]
            mock_db.query.return_value.join.return_value.filter.return_value = (
                mock_query_result
            )

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(2)  # regular user

            assert result == mock_llm
            mock_create.assert_called_once()
            # Verify _get_visible_user_ids was called for admin fallback
            mock_visible.assert_called_with(mock_db, 2)

    def test_get_default_model_no_user_id(self):
        """Test getting default model without user ID - falls through to visible users fallback."""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
            patch(
                "xagent.web.services.model_service._get_visible_user_ids"
            ) as mock_visible,
        ):
            mock_visible.return_value = [1]

            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # No user-specific default
            mock_query_result = MagicMock()
            mock_query_result.first.return_value = None
            mock_query_result.all.return_value = [MagicMock()]
            mock_db.query.return_value.join.return_value.filter.return_value = (
                mock_query_result
            )

            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(None)

            assert result == mock_llm
            mock_visible.assert_called_with(mock_db, None)

    def test_get_default_model_no_configuration(self):
        """Test getting default model when no configuration exists - should return None"""
        with patch("xagent.web.models.database.get_db") as mock_get_db:
            # Setup mock database session
            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # Setup query to return no results
            mock_db.query.return_value.join.return_value.filter.return_value.first.return_value = None
            mock_db.query.return_value.join.return_value.filter.return_value.all.return_value = []

            result = get_default_model(1)
            assert result is None

    def test_model_service_multiple_users(self):
        """Test model service with multiple users — both fall through to admin shared default."""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
            patch(
                "xagent.web.services.model_service._get_visible_user_ids"
            ) as mock_visible,
        ):
            mock_visible.return_value = [1]  # admin user ID is visible to everyone

            # Setup mock database session - create a new session for each call
            mock_db1 = MagicMock()
            mock_db2 = MagicMock()
            mock_get_db.side_effect = [iter([mock_db1]), iter([mock_db2])]

            # Setup query to return None for individual user defaults and shared default for shared query
            mock_query_result1 = MagicMock()
            mock_query_result1.first.return_value = None
            mock_query_result1.all.return_value = [MagicMock()]
            mock_db1.query.return_value.join.return_value.filter.return_value = (
                mock_query_result1
            )

            mock_query_result2 = MagicMock()
            mock_query_result2.first.return_value = None
            mock_query_result2.all.return_value = [MagicMock()]
            mock_db2.query.return_value.join.return_value.filter.return_value = (
                mock_query_result2
            )

            # Setup mock LLM creation
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            # Both users should get the same shared model
            admin_result = get_default_model(1)  # admin user
            regular_result = get_default_model(2)  # regular user

            assert admin_result == mock_llm
            assert regular_result == mock_llm

            # Should have been called twice (once for each user)
            assert mock_create.call_count == 2
            # _get_visible_user_ids should have been called for both users
            assert mock_visible.call_count == 2

    def test_is_model_visible_to_user_fails_closed_on_exception(self):
        """Visibility check must return False (deny) when the query raises."""
        with patch(
            "xagent.web.services.model_service._get_visible_user_ids"
        ) as mock_visible:
            mock_visible.side_effect = RuntimeError("DB connection lost")
            mock_db = MagicMock()
            # Ownership check (step 1) returns None so code reaches step 2
            mock_db.query.return_value.filter.return_value.first.return_value = None
            result = _is_model_visible_to_user(mock_db, 1, 1)
            assert result is False

    def test_get_default_model_stale_default_skipped(self):
        """Stale user default (model no longer visible) falls through to admin fallback."""
        with (
            patch("xagent.web.models.database.get_db") as mock_get_db,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
            patch(
                "xagent.web.services.model_service._get_visible_user_ids"
            ) as mock_visible,
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
        ):
            mock_visible.return_value = [1]
            mock_visibility.return_value = False  # model NOT visible — stale

            mock_db = MagicMock()
            mock_get_db.return_value = iter([mock_db])

            # UserDefaultModel found, but visibility check fails
            mock_user_default = MagicMock()
            mock_user_default.user_id = 42
            mock_user_default.config_type = "general"
            mock_user_default.model = MagicMock()
            mock_user_default.model.id = 42

            mock_first_result = MagicMock()
            mock_first_result.first.return_value = mock_user_default
            mock_all_result = MagicMock()
            mock_all_result.all.return_value = [MagicMock()]

            # Two filter calls: one for user-specific (returns None after visibility fails),
            # then admin fallback
            mock_db.query.return_value.join.return_value.filter.side_effect = [
                mock_first_result,  # user-specific query
                mock_all_result,  # admin fallback query
            ]

            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_default_model(42)

            assert result == mock_llm
            # Visibility check was called
            mock_visibility.assert_called_once_with(mock_db, 42, 42)
            # Admin fallback was used (create_llm called)
            mock_create.assert_called_once()
            # _get_visible_user_ids was called for admin fallback
            mock_visible.assert_called_with(mock_db, 42)

    def test_embedding_model_stale_default_falls_through_to_system(self, monkeypatch):
        """Stale embedding default (not visible) must fall through to visible system fallback."""
        from contextvars import copy_context

        from xagent.web.user_isolated_memory import current_user_id

        mock_db = MagicMock()

        def mock_get_db():
            yield mock_db

        # Set up user context
        ctx = copy_context()
        ctx.run(current_user_id.set, 1)

        def run_in_context():
            # Build mock visible system fallback model
            system_fallback = MagicMock()
            system_fallback.id = 200
            system_fallback.model_id = "system-embedding-model"

            # user_default query returns a row
            mock_user_default = MagicMock()
            mock_user_default.model_id = 99

            # embedding_model query returns the stale model
            stale_embedding = MagicMock()
            stale_embedding.id = 99
            stale_embedding.model_id = "stale-embedding-model"

            # Mock query chain:
            # 1st call: UserDefaultModel.filter().first() → user_default
            # 2nd call: DBModel.filter().first() → stale_embedding (stale, visibility fails)
            # 3rd call: DBModel.filter().all() → [system_fallback]
            mock_filter = mock_db.query.return_value.filter.return_value
            mock_filter.first.side_effect = [
                mock_user_default,
                stale_embedding,
            ]
            mock_filter.all.return_value = [system_fallback]

            monkeypatch.setattr(
                "xagent.web.services.model_service._is_model_visible_to_user",
                lambda db, model_id, user_id: model_id != 99,
            )
            monkeypatch.setattr(
                "xagent.web.dynamic_memory_store.get_db",
                mock_get_db,
            )

            from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager

            manager = DynamicMemoryStoreManager()
            result = manager._get_embedding_model_from_db()

            # Must fall through to system fallback, not stale model
            assert result is system_fallback
            assert result.model_id == "system-embedding-model"

        ctx.run(run_in_context)

    def test_get_vision_model_filters_by_visibility(self):
        """get_vision_model returns the first visible vision-capable model."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch("xagent.web.services.llm_utils._create_llm_instance") as mock_create,
        ):
            mock_db = MagicMock()

            # Two vision-capable models
            visible_model = MagicMock()
            visible_model.id = 1
            invisible_model = MagicMock()
            invisible_model.id = 2

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model,
                invisible_model,
            ]

            # First model is visible, second is not
            mock_visibility.side_effect = [True, False]
            mock_llm = MagicMock()
            mock_create.return_value = mock_llm

            result = get_vision_model(mock_db, user_id=42)

            assert result == mock_llm
            mock_create.assert_called_once_with(visible_model)
            assert mock_visibility.call_count == 1  # stops after first visible

    def test_get_vision_model_returns_none_when_no_visible(self):
        """get_vision_model returns None when no vision model is visible."""
        with patch(
            "xagent.web.services.model_service._is_model_visible_to_user"
        ) as mock_visibility:
            mock_db = MagicMock()

            invisible_model = MagicMock()
            invisible_model.id = 1
            mock_db.query.return_value.filter.return_value.all.return_value = [
                invisible_model
            ]
            mock_visibility.return_value = False

            result = get_vision_model(mock_db, user_id=42)

            assert result is None

    def test_get_image_models_filters_by_visibility(self):
        """get_image_models excludes image models not visible to the user."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch(
                "xagent.web.services.model_service.DashScopeImageModel"
            ) as mock_dashscope,
        ):
            mock_db = MagicMock()

            visible_model = MagicMock()
            visible_model.id = 1
            visible_model.api_key = "key1"
            visible_model.base_url = "http://url1"
            visible_model.model_provider = "dashscope"
            visible_model.model_name = "visible-model"
            visible_model.model_id = "mid-1"
            visible_model.abilities = ["generate"]

            invisible_model = MagicMock()
            invisible_model.id = 2
            invisible_model.api_key = "key2"
            invisible_model.base_url = "http://url2"
            invisible_model.model_provider = "dashscope"
            invisible_model.model_name = "invisible-model"
            invisible_model.model_id = "mid-2"
            invisible_model.abilities = ["generate"]

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model,
                invisible_model,
            ]

            mock_visibility.side_effect = [True, False]
            mock_instance = MagicMock()
            mock_dashscope.return_value = mock_instance

            result = get_image_models(mock_db, user_id=42)

            assert len(result) == 1
            assert "mid-1" in result
            assert "mid-2" not in result

    def test_get_asr_models_filters_by_visibility(self):
        """get_asr_models excludes speech models not visible to the user."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch(
                "xagent.core.model.asr.adapter.get_asr_model_instance"
            ) as mock_asr_instance,
        ):
            mock_db = MagicMock()

            visible_model = MagicMock()
            visible_model.id = 1
            visible_model.abilities = ["asr"]
            visible_model.api_key = "key1"
            visible_model.base_url = "http://url1"
            visible_model.model_provider = "xinference"
            visible_model.model_name = "visible-asr"

            invisible_model = MagicMock()
            invisible_model.id = 2
            invisible_model.abilities = ["asr"]
            invisible_model.api_key = "key2"
            invisible_model.base_url = "http://url2"
            invisible_model.model_provider = "xinference"
            invisible_model.model_name = "invisible-asr"

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model,
                invisible_model,
            ]

            mock_visibility.side_effect = [True, False]
            mock_instance = MagicMock()
            mock_asr_instance.return_value = mock_instance

            result = get_asr_models(mock_db, user_id=42)

            assert len(result) == 1
            assert "visible-asr" in result
            assert "invisible-asr" not in result

    def test_get_tts_models_filters_by_visibility(self):
        """get_tts_models excludes speech models not visible to the user."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch(
                "xagent.core.model.tts.adapter.get_tts_model_instance"
            ) as mock_tts_instance,
        ):
            mock_db = MagicMock()

            visible_model = MagicMock()
            visible_model.id = 1
            visible_model.abilities = ["tts"]
            visible_model.api_key = "key1"
            visible_model.base_url = "http://url1"
            visible_model.model_provider = "xinference"
            visible_model.model_name = "visible-tts"

            invisible_model = MagicMock()
            invisible_model.id = 2
            invisible_model.abilities = ["tts"]
            invisible_model.api_key = "key2"
            invisible_model.base_url = "http://url2"
            invisible_model.model_provider = "xinference"
            invisible_model.model_name = "invisible-tts"

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model,
                invisible_model,
            ]

            mock_visibility.side_effect = [True, False]
            mock_instance = MagicMock()
            mock_tts_instance.return_value = mock_instance

            result = get_tts_models(mock_db, user_id=42)

            assert len(result) == 1
            assert "visible-tts" in result
            assert "invisible-tts" not in result

    def test_get_tts_models_loads_elevenlabs_without_base_url(self):
        """ElevenLabs TTS models use SDK defaults and should not require base_url."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch(
                "xagent.core.model.tts.adapter.get_tts_model_instance"
            ) as mock_tts_instance,
        ):
            mock_db = MagicMock()

            visible_model = MagicMock()
            visible_model.id = 1
            visible_model.abilities = ["tts"]
            visible_model.api_key = "key1"
            visible_model.base_url = None
            visible_model.model_provider = "elevenlabs"
            visible_model.model_name = "eleven_v3"

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model
            ]

            mock_visibility.return_value = True
            mock_instance = MagicMock()
            mock_tts_instance.return_value = mock_instance

            result = get_tts_models(mock_db, user_id=42)

            assert result == {"eleven_v3": mock_instance}

    def test_get_asr_models_loads_elevenlabs_without_base_url(self):
        """ElevenLabs ASR models use SDK defaults and should not require base_url."""
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user"
            ) as mock_visibility,
            patch(
                "xagent.core.model.asr.adapter.get_asr_model_instance"
            ) as mock_asr_instance,
        ):
            mock_db = MagicMock()

            visible_model = MagicMock()
            visible_model.id = 1
            visible_model.abilities = ["asr"]
            visible_model.api_key = "key1"
            visible_model.base_url = None
            visible_model.model_provider = "elevenlabs"
            visible_model.model_name = "scribe_v2"

            mock_db.query.return_value.filter.return_value.all.return_value = [
                visible_model
            ]

            mock_visibility.return_value = True
            mock_instance = MagicMock()
            mock_asr_instance.return_value = mock_instance

            result = get_asr_models(mock_db, user_id=42)

            assert result == {"scribe_v2": mock_instance}

    def test_get_sound_effect_models_uses_independent_category(self):
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user",
                return_value=True,
            ),
            patch(
                "xagent.core.model.sound_effect.get_sound_effect_model_instance"
            ) as mock_instance_factory,
        ):
            mock_db = MagicMock()
            db_model = MagicMock()
            db_model.id = 1
            db_model.model_id = "sound-effect-default"
            db_model.model_name = "eleven_text_to_sound_v2"
            db_model.category = "sound_effect"
            db_model.is_active = True
            mock_db.query.return_value.filter.return_value.all.return_value = [db_model]
            model = MagicMock()
            mock_instance_factory.return_value = model

            result = get_sound_effect_models(mock_db, user_id=42)

            assert result == {"sound-effect-default": model}
            mock_instance_factory.assert_called_once_with(db_model)

    def test_get_music_models_uses_independent_category(self):
        with (
            patch(
                "xagent.web.services.model_service._is_model_visible_to_user",
                return_value=True,
            ),
            patch(
                "xagent.core.model.music.get_music_model_instance"
            ) as mock_instance_factory,
        ):
            mock_db = MagicMock()
            db_model = MagicMock()
            db_model.id = 1
            db_model.model_id = "music-default"
            db_model.model_name = "music_v2"
            db_model.category = "music"
            db_model.is_active = True
            mock_db.query.return_value.filter.return_value.all.return_value = [db_model]
            model = MagicMock()
            mock_instance_factory.return_value = model

            result = get_music_models(mock_db, user_id=42)

            assert result == {"music-default": model}
            mock_instance_factory.assert_called_once_with(db_model)

    @pytest.mark.parametrize(
        "get_default",
        [get_default_sound_effect_model, get_default_music_model],
    )
    def test_audio_generation_default_closes_database_generator(self, get_default):
        mock_db = MagicMock()
        filter_query = (
            mock_db.query.return_value.join.return_value.join.return_value.filter
        )
        shared_query = filter_query.return_value
        shared_query.limit.return_value.all.return_value = []
        db_gen = MagicMock()
        db_gen.__next__.return_value = mock_db

        with patch(
            "xagent.web.models.database.get_db",
            return_value=db_gen,
        ):
            result = get_default(user_id=None)

        assert result is None
        assert any(
            getattr(getattr(condition, "right", None), "value", None) == '"generate"'
            for condition in filter_query.call_args.args
        )
        db_gen.close.assert_called_once_with()

    def test_embedding_fallback_skips_invisible_returns_none(self, monkeypatch):
        """System fallback returns None when no visible embedding model exists."""
        from contextvars import copy_context

        from xagent.web.user_isolated_memory import current_user_id

        mock_db = MagicMock()

        def mock_get_db():
            yield mock_db

        ctx = copy_context()
        ctx.run(current_user_id.set, 1)

        def run_in_context():
            # No user default
            mock_filter = mock_db.query.return_value.filter.return_value
            mock_filter.first.return_value = None

            # Two active embeddings, neither visible
            embedding1 = MagicMock()
            embedding1.id = 1
            embedding1.model_id = "private-embed-1"
            embedding2 = MagicMock()
            embedding2.id = 2
            embedding2.model_id = "private-embed-2"
            mock_filter.all.return_value = [embedding1, embedding2]

            monkeypatch.setattr(
                "xagent.web.services.model_service._is_model_visible_to_user",
                lambda db, model_id, user_id: False,
            )
            monkeypatch.setattr(
                "xagent.web.dynamic_memory_store.get_db",
                mock_get_db,
            )

            from xagent.web.dynamic_memory_store import DynamicMemoryStoreManager

            manager = DynamicMemoryStoreManager()
            result = manager._get_embedding_model_from_db()

            assert result is None

        ctx.run(run_in_context)
