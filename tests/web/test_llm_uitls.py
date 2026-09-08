from unittest.mock import Mock, patch

import pytest
from sqlalchemy.orm import Session

from xagent.core.model import ChatModelConfig
from xagent.web.models import Model, UserDefaultModel, UserModel
from xagent.web.services.llm_utils import UserAwareModelStorage, create_llm_from_env


@pytest.fixture
def mock_db():
    return Mock(spec=Session)


@pytest.fixture
def mock_core_storage(monkeypatch):
    core = Mock()
    monkeypatch.setattr("xagent.web.services.llm_utils.CoreStorage", lambda *args: core)
    return core


@pytest.fixture
def storage(mock_db, mock_core_storage):
    return UserAwareModelStorage(mock_db)


class TestGetLLMByNameWithAccess:
    def test_returns_llm_when_model_exists_no_user(self, storage, mock_core_storage):
        mock_llm = Mock()
        mock_model = Mock(
            spec=ChatModelConfig, id=1, model_id="test-model", model_name="test-model"
        )  # Added model_name
        mock_core_storage.load.return_value = ChatModelConfig(
            id="model", model_name="model"
        )
        mock_core_storage.get_db_model.return_value = mock_model
        mock_core_storage.create_llm_instance.return_value = mock_llm

        result = storage.get_llm_by_name_with_access("test-model")

        assert result == mock_llm
        mock_core_storage.load.assert_called_once_with("test-model")

    def test_returns_llm_when_user_has_access(
        self, storage, mock_db, mock_core_storage
    ):
        mock_llm = Mock()
        mock_model = Mock(
            spec=ChatModelConfig, id=1, model_id="test-model", model_name="test-model"
        )  # Added model_name
        mock_user_model = Mock(spec=UserModel)
        mock_core_storage.load.return_value = ChatModelConfig(
            id="model", model_name="model"
        )

        mock_core_storage.get_db_model.return_value = mock_model
        mock_core_storage.create_llm_instance.return_value = mock_llm
        mock_db.query.return_value.filter.return_value.first.return_value = (
            mock_user_model
        )

        result = storage.get_llm_by_name_with_access("test-model", user_id=1)

        assert result == mock_llm

    def test_returns_none_when_invalid_model_config(
        self, storage, mock_db, mock_core_storage
    ):
        """Returns None when model_config is not a ChatModelConfig (dict instead of object)."""
        mock_model = Mock(
            spec=ChatModelConfig, id=1, model_id="test-model", model_name="test-model"
        )  # Added model_name
        mock_core_storage.load.return_value = {"model": "config"}
        mock_core_storage.get_db_model.return_value = mock_model
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = storage.get_llm_by_name_with_access("test-model", user_id=1)

        assert result is None

    def test_returns_none_when_invalid_model_provider(self, storage, mock_core_storage):
        mock_model = Mock(
            spec=Model, id=1, model_id="test-model", model_name="test-model"
        )  # Wrong type
        mock_core_storage.load.return_value = {"model": "config"}
        mock_core_storage.get_db_model.return_value = mock_model

        result = storage.get_llm_by_name_with_access("test-model")

        assert result is None

    def test_returns_none_when_model_config_is_not_chat_model(
        self, storage, mock_core_storage, mock_db
    ):
        """Test that non-ChatModelConfig types are rejected"""
        from xagent.core.model.model import EmbeddingModelConfig

        # Setup: load returns an EmbeddingModelConfig instead of ChatModelConfig
        mock_embedding_config = EmbeddingModelConfig(
            id="embedding-model", model_name="text-embedding-ada-002"
        )
        mock_db_model = Mock(
            id=1, model_id="embedding-model", model_name="embedding-model"
        )

        mock_core_storage.load.return_value = mock_embedding_config
        mock_core_storage.get_db_model.return_value = mock_db_model

        result = storage.get_llm_by_name_with_access("embedding-model")

        assert result is None
        mock_core_storage.create_llm_instance.assert_not_called()

    def test_accepts_valid_chat_model_config(self, storage, mock_core_storage, mock_db):
        """Test that valid ChatModelConfig is accepted (positive case)"""
        mock_llm = Mock()
        mock_chat_config = ChatModelConfig(id="gpt-4", model_name="gpt-4")
        mock_db_model = Mock(id=1, model_id="gpt-4", model_name="gpt-4")

        mock_core_storage.load.return_value = mock_chat_config
        mock_core_storage.get_db_model.return_value = mock_db_model
        mock_core_storage.create_llm_instance.return_value = mock_llm

        result = storage.get_llm_by_name_with_access("gpt-4")

        assert result == mock_llm
        mock_core_storage.create_llm_instance.assert_called_once_with(mock_chat_config)

    def test_returns_llm_via_shared_model_step2(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test two-step lookup: step 1 misses, step 2 finds shared model from visible user."""
        mock_llm = Mock()
        mock_chat_config = ChatModelConfig(id="shared-model", model_name="shared-model")
        mock_db_model = Mock(id=1, model_id="shared-model", model_name="shared-model")

        mock_core_storage.load.return_value = mock_chat_config
        mock_core_storage.get_db_model.return_value = mock_db_model
        mock_core_storage.create_llm_instance.return_value = mock_llm

        # Mock _get_visible_user_ids to return admin user IDs
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [10],  # admin user ID
        )

        # Step 1: own UserModel query returns None
        # Step 2: shared UserModel query returns a shared model
        mock_shared_user_model = Mock(spec=UserModel)
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            None,  # step 1: no own UserModel
            mock_shared_user_model,  # step 2: found shared UserModel
        ]

        result = storage.get_llm_by_name_with_access("shared-model", user_id=1)

        assert result == mock_llm
        assert mock_db.query.return_value.filter.return_value.first.call_count == 2

    def test_returns_none_when_shared_model_not_found_step2(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test two-step lookup: both steps miss, returns None."""
        mock_chat_config = ChatModelConfig(
            id="private-model", model_name="private-model"
        )
        mock_db_model = Mock(id=1, model_id="private-model", model_name="private-model")

        mock_core_storage.load.return_value = mock_chat_config
        mock_core_storage.get_db_model.return_value = mock_db_model

        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [10],
        )

        # Both steps return None
        mock_db.query.return_value.filter.return_value.first.side_effect = [
            None,  # step 1
            None,  # step 2
        ]

        result = storage.get_llm_by_name_with_access("private-model", user_id=1)

        assert result is None


class TestGetConfiguredDefaults:
    def test_returns_user_specific_defaults(self, storage, mock_db, mock_core_storage):
        """Test that user-specific defaults are returned when available"""
        mock_llm = Mock()
        mock_model = Mock(model_id="model-1")
        mock_config = Mock(spec=ChatModelConfig)

        # Setup user defaults
        mock_general = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="general"
        )
        mock_fast = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="small_fast"
        )
        mock_vision = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="visual"
        )
        mock_compact = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="compact"
        )

        # Mock query chain for each config type
        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [
            mock_general,
            mock_fast,
            mock_vision,
            mock_compact,
        ]

        mock_core_storage.load.return_value = mock_config
        mock_core_storage.create_llm_instance.return_value = mock_llm

        result = storage.get_configured_defaults(user_id=1)

        assert result == (mock_llm, mock_llm, mock_llm, mock_llm)
        assert mock_core_storage.load.call_count == 4
        assert mock_core_storage.create_llm_instance.call_count == 4

    def test_all_explicit_models_do_not_load_unrelated_defaults(self, storage):
        names = ["general", "fast", "vision", "compact"]
        llms = {name: Mock(model_name=name) for name in names}

        with (
            patch.object(
                storage,
                "get_llm_by_name_with_access",
                side_effect=lambda name, user_id: llms[name],
            ),
            patch.object(
                storage,
                "get_configured_defaults",
                side_effect=AssertionError("unrelated defaults must not be loaded"),
            ) as get_defaults,
        ):
            result = storage.resolve_llms_from_names(names, user_id=7)

        assert result == tuple(llms[name] for name in names)
        get_defaults.assert_not_called()

    def test_falls_back_to_admin_defaults(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test fallback to admin shared defaults when user defaults missing"""
        mock_llm = Mock()
        mock_model = Mock(model_id="admin-model")
        mock_config = Mock(spec=ChatModelConfig)

        # Mock _get_visible_user_ids at the source module
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [1],
        )

        # No user-specific defaults
        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [None, None, None, None]

        # Admin defaults available
        mock_admin_general = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="general"
        )
        mock_admin_fast = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="small_fast"
        )
        mock_admin_vision = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="visual"
        )
        mock_admin_compact = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="compact"
        )

        mock_filter.all.return_value = [
            mock_admin_general,
            mock_admin_fast,
            mock_admin_vision,
            mock_admin_compact,
        ]

        mock_core_storage.load.return_value = mock_config
        mock_core_storage.create_llm_instance.return_value = mock_llm

        result = storage.get_configured_defaults(user_id=1)

        assert result == (mock_llm, mock_llm, mock_llm, mock_llm)
        assert mock_core_storage.load.call_count == 4
        assert mock_core_storage.create_llm_instance.call_count == 4

    def test_partial_user_defaults_fills_from_admin(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test that missing user defaults are filled from admin defaults"""
        mock_user_llm = Mock()
        mock_admin_llm = Mock()
        mock_user_model = Mock(model_id="user-model")
        mock_admin_model = Mock(model_id="admin-model")
        mock_user_config = Mock(spec=ChatModelConfig)
        mock_admin_config = Mock(spec=ChatModelConfig)

        # Mock _get_visible_user_ids at the source module
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [1],
        )

        # User has only general and fast defaults
        mock_general = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="general"
        )
        mock_fast = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="small_fast"
        )

        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [mock_general, mock_fast, None, None]

        # Admin has vision and compact
        mock_admin_vision = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="visual"
        )
        mock_admin_compact = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="compact"
        )
        mock_filter.all.return_value = [mock_admin_vision, mock_admin_compact]

        def load_side_effect(model_id):
            if model_id == "user-model":
                return mock_user_config
            return mock_admin_config

        def create_instance_side_effect(config):
            if config == mock_user_config:
                return mock_user_llm
            return mock_admin_llm

        mock_core_storage.load.side_effect = load_side_effect
        mock_core_storage.create_llm_instance.side_effect = create_instance_side_effect

        result = storage.get_configured_defaults(user_id=1)

        assert result == (mock_user_llm, mock_user_llm, mock_admin_llm, mock_admin_llm)

    def test_falls_back_to_env_when_no_models(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test fallback to environment variables when no models configured"""
        mock_env_llm = Mock()

        # No user defaults
        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [None, None, None, None]

        # No admin defaults
        mock_filter.all.return_value = []

        # Mock environment fallback
        monkeypatch.setattr(
            "xagent.web.services.llm_utils.create_llm_from_env", lambda: mock_env_llm
        )

        result = storage.get_configured_defaults(user_id=1)

        assert result == (mock_env_llm, mock_env_llm, mock_env_llm, mock_env_llm)

    def test_uses_default_llm_for_missing_specialized(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test that default LLM is used when specialized LLMs are missing"""
        mock_default_llm = Mock()
        mock_model = Mock(model_id="default-model")
        mock_config = Mock(spec=ChatModelConfig)

        # Mock _get_visible_user_ids at the source module
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [1],
        )

        # Only general default available
        mock_general = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="general"
        )

        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [mock_general, None, None, None]

        # No admin defaults
        mock_filter.all.return_value = []

        mock_core_storage.load.return_value = mock_config
        mock_core_storage.create_llm_instance.return_value = mock_default_llm

        result = storage.get_configured_defaults(user_id=1)

        # All should use the default LLM
        assert result == (
            mock_default_llm,
            mock_default_llm,
            mock_default_llm,
            mock_default_llm,
        )

    def test_stale_user_default_falls_back_to_admin(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """When user default exists but visibility check fails, fall back to admin shared."""
        mock_admin_llm = Mock()
        mock_user_model = Mock(model_id="user-stale-model")
        mock_admin_model = Mock(model_id="admin-shared-model")
        mock_config = Mock(spec=ChatModelConfig)

        # _get_visible_user_ids returns admin
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [1],
        )
        # _is_model_visible_to_user always returns False (stale defaults)
        monkeypatch.setattr(
            "xagent.web.services.model_service._is_model_visible_to_user",
            lambda db, model_id, user_id: False,
        )

        # User has defaults for all 4 types — but all fail visibility
        mock_general = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="general"
        )
        mock_fast = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="small_fast"
        )
        mock_vision = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="visual"
        )
        mock_compact = Mock(
            spec=UserDefaultModel, model=mock_user_model, config_type="compact"
        )

        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.first.side_effect = [
            mock_general,
            mock_fast,
            mock_vision,
            mock_compact,
        ]

        # Admin shared defaults available
        mock_admin_general = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="general"
        )
        mock_admin_fast = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="small_fast"
        )
        mock_admin_vision = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="visual"
        )
        mock_admin_compact = Mock(
            spec=UserDefaultModel, model=mock_admin_model, config_type="compact"
        )
        mock_filter.all.return_value = [
            mock_admin_general,
            mock_admin_fast,
            mock_admin_vision,
            mock_admin_compact,
        ]

        def load_side_effect(model_id):
            return mock_config

        def create_instance_side_effect(config):
            return mock_admin_llm

        mock_core_storage.load.side_effect = load_side_effect
        mock_core_storage.create_llm_instance.side_effect = create_instance_side_effect

        result = storage.get_configured_defaults(user_id=1)

        # All should come from admin fallback, not user defaults
        assert result == (
            mock_admin_llm,
            mock_admin_llm,
            mock_admin_llm,
            mock_admin_llm,
        )
        # load + create_llm called 4 times (all admin fallback)
        assert mock_core_storage.load.call_count == 4
        assert mock_core_storage.create_llm_instance.call_count == 4

    def test_handles_exception_gracefully(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test that exceptions are handled and fallback to env is used"""
        mock_env_llm = Mock()

        # Simulate exception during query
        mock_db.query.side_effect = Exception("Database error")

        monkeypatch.setattr(
            "xagent.web.services.llm_utils.create_llm_from_env", lambda: mock_env_llm
        )

        result = storage.get_configured_defaults(user_id=1)

        assert result == (mock_env_llm, mock_env_llm, mock_env_llm, mock_env_llm)

    def test_no_user_id_uses_admin_defaults(
        self, storage, mock_db, mock_core_storage, monkeypatch
    ):
        """Test that when no user_id provided, only admin defaults are used"""
        mock_llm = Mock()
        mock_model = Mock(model_id="admin-model")
        mock_config = Mock(spec=ChatModelConfig)

        # Mock _get_visible_user_ids at the source module
        monkeypatch.setattr(
            "xagent.web.services.model_service._get_visible_user_ids",
            lambda db, user_id: [1],
        )

        # Admin defaults
        mock_admin_general = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="general"
        )
        mock_admin_fast = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="small_fast"
        )
        mock_admin_vision = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="visual"
        )
        mock_admin_compact = Mock(
            spec=UserDefaultModel, model=mock_model, config_type="compact"
        )

        mock_query = mock_db.query.return_value
        mock_join = mock_query.join.return_value
        mock_filter = mock_join.filter.return_value
        mock_filter.all.return_value = [
            mock_admin_general,
            mock_admin_fast,
            mock_admin_vision,
            mock_admin_compact,
        ]

        mock_core_storage.load.return_value = mock_config
        mock_core_storage.create_llm_instance.return_value = mock_llm

        result = storage.get_configured_defaults(user_id=None)

        assert result == (mock_llm, mock_llm, mock_llm, mock_llm)
        # Should not query for user-specific defaults
        assert mock_filter.first.call_count == 0


class TestCreateLLMFromEnv:
    def test_creates_deepseek_llm_from_env(self, monkeypatch):
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL_NAME",
            "ZHIPU_API_KEY",
            "ZHIPU_BASE_URL",
            "ZHIPU_MODEL_NAME",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-api-key")
        monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-pro")
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        mock_llm = Mock()
        with patch(
            "xagent.web.services.llm_utils.DeepSeekLLM", return_value=mock_llm
        ) as mock_deepseek_llm:
            result = create_llm_from_env()

        mock_deepseek_llm.assert_called_once_with(
            model_name="deepseek-v4-pro",
            api_key="deepseek-api-key",
            base_url="https://api.deepseek.com",
        )
        assert result is mock_llm

    def test_ignores_deepseek_placeholder_api_key(self, monkeypatch):
        for key in (
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_MODEL_NAME",
            "ZHIPU_API_KEY",
            "ZHIPU_BASE_URL",
            "ZHIPU_MODEL_NAME",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("DEEPSEEK_API_KEY", "your-deepseek-api-key")

        with patch("xagent.web.services.llm_utils.DeepSeekLLM") as mock_deepseek_llm:
            result = create_llm_from_env()

        mock_deepseek_llm.assert_not_called()
        assert result is None

    def test_openai_placeholder_does_not_block_deepseek(self, monkeypatch):
        for key in (
            "OPENAI_BASE_URL",
            "OPENAI_MODEL_NAME",
            "ZHIPU_API_KEY",
            "ZHIPU_BASE_URL",
            "ZHIPU_MODEL_NAME",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("OPENAI_API_KEY", "your-openai-api-key")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-api-key")
        monkeypatch.setenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")

        mock_llm = Mock()
        with (
            patch("xagent.web.services.llm_utils.OpenAILLM") as mock_openai_llm,
            patch(
                "xagent.web.services.llm_utils.DeepSeekLLM", return_value=mock_llm
            ) as mock_deepseek_llm,
        ):
            result = create_llm_from_env()

        mock_openai_llm.assert_not_called()
        mock_deepseek_llm.assert_called_once_with(
            model_name="deepseek-v4-flash",
            api_key="deepseek-api-key",
            base_url=None,
        )
        assert result is mock_llm
