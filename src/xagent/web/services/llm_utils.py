"""LLM resolution utilities for task creation with multi-tenant support"""

import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from ...core.model.chat.basic.adapter import create_base_llm
from ...core.model.chat.basic.base import BaseLLM
from ...core.model.chat.basic.claude import ClaudeLLM
from ...core.model.chat.basic.gemini import GeminiLLM
from ...core.model.chat.basic.openai import OpenAILLM
from ...core.model.chat.basic.zhipu import ZhipuLLM
from ...core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelConfig,
    RerankModelConfig,
)
from ..models.model import Model
from ..models.user import UserDefaultModel, UserModel

logger = logging.getLogger(__name__)


def _create_llm_instance(db_model: Model) -> BaseLLM:
    if db_model.category == "llm":
        config: ModelConfig = ChatModelConfig(
            id=db_model.model_id,
            model_name=db_model.model_name,
            model_provider=db_model.model_provider,
            api_key=db_model.api_key,
            base_url=db_model.base_url,
            default_temperature=db_model.temperature,
            abilities=db_model.abilities,
            description=db_model.description,
        )
    elif db_model.category == "embedding":
        config = EmbeddingModelConfig(
            id=db_model.model_id,
            model_name=db_model.model_name,
            model_provider=db_model.model_provider,
            dimension=db_model.dimension,
            api_key=db_model.api_key,
            base_url=db_model.base_url,
            abilities=db_model.abilities,
            description=db_model.description,
        )
    elif db_model.category == "rerank":
        config = RerankModelConfig(
            id=db_model.model_id,
            model_name=db_model.model_name,
            api_key=db_model.api_key,
            base_url=db_model.base_url,
            abilities=db_model.abilities,
            description=db_model.description,
        )
    else:
        raise ValueError(f"Unknown model category: {db_model.category}")

    return create_base_llm(config)


class CoreStorage:
    """Direct database-based model storage without ModelHub abstraction."""

    def __init__(self, db: Session, model_class: type[Model]):
        """
        Initialize CoreStorage.

        Args:
            db: SQLAlchemy database session
            model_class: Model ORM class
        """
        self.db = db
        self.Model = model_class

    def _db_model_to_config(self, db_model: Model) -> ModelConfig:
        """Convert database model to ModelConfig."""
        common = {
            "id": db_model.model_id,
            "model_name": db_model.model_name,
            "api_key": db_model.api_key,
            "base_url": db_model.base_url,
            "abilities": db_model.abilities,
            "description": db_model.description,
            "max_retries": db_model.max_retries
            if db_model.max_retries is not None
            else 10,
        }

        if db_model.category == "llm":
            return ChatModelConfig(
                **common,
                model_provider=db_model.model_provider,
                default_temperature=db_model.temperature,
                default_max_tokens=db_model.max_tokens,
            )
        elif db_model.category == "image":
            from ...core.model.model import ImageModelConfig

            return ImageModelConfig(
                **common,
                model_provider=db_model.model_provider,
                default_max_tokens=db_model.max_tokens,
            )
        elif db_model.category == "embedding":
            return EmbeddingModelConfig(
                **common,
                model_provider=db_model.model_provider,
                dimension=db_model.dimension,
            )
        elif db_model.category == "rerank":
            return RerankModelConfig(**common)
        elif db_model.category == "speech":
            from ...core.model.model import SpeechModelConfig

            return SpeechModelConfig(
                **common,
                model_provider=db_model.model_provider,
            )
        else:
            raise ValueError(f"Unknown model category: {db_model.category}")

    def load(self, model_id: str) -> ModelConfig:
        """Load model configuration by model_id or model_name."""
        db_model = self.get_db_model(model_id)
        if not db_model:
            raise ValueError(f"Model not found: {model_id}")

        return self._db_model_to_config(db_model)

    def exists(self, model_id: str) -> bool:
        """Check if model exists."""
        return self.get_db_model(model_id) is not None

    def store(self, model: ModelConfig) -> None:
        """Store model configuration to database."""

        db_data: dict[str, Any] = {
            "model_id": model.id,
            "model_name": model.model_name,
            "api_key": model.api_key,
            "base_url": model.base_url,
            "abilities": model.abilities,
            "description": model.description,
            "max_retries": model.max_retries,
            "is_active": True,
        }

        if isinstance(model, ChatModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "temperature": model.default_temperature,
                    "max_tokens": model.default_max_tokens,
                    "category": "llm",
                }
            )
        elif isinstance(model, EmbeddingModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "dimension": model.dimension,
                    "category": "embedding",
                }
            )
        elif isinstance(model, RerankModelConfig):
            db_data.update(
                {
                    "model_provider": "none",
                    "category": "rerank",
                }
            )
        else:
            # Try ImageModelConfig or SpeechModelConfig
            from ...core.model.model import ImageModelConfig, SpeechModelConfig

            if isinstance(model, ImageModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "max_tokens": model.default_max_tokens,
                        "category": "image",
                    }
                )
            elif isinstance(model, SpeechModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "category": "speech",
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

        db_model = self.Model(**db_data)
        self.db.add(db_model)
        self.db.commit()

    def delete(self, model_id: str) -> None:
        """Delete model by model_id."""
        db_model = (
            self.db.query(self.Model).filter(self.Model.model_id == model_id).first()
        )
        if db_model:
            self.db.delete(db_model)
            self.db.commit()

    def list(self) -> dict[str, ModelConfig]:
        """List all active models."""
        db_models = self.db.query(self.Model).filter(self.Model.is_active).all()
        result: dict[str, ModelConfig] = {}

        for db_model in db_models:
            try:
                config = self._db_model_to_config(db_model)
                result[str(db_model.model_id)] = config
            except ValueError:
                # Skip models with unknown categories
                continue

        return result

    def create_llm_instance(self, model_config: ModelConfig) -> Optional[BaseLLM]:
        """Create LLM instance from ModelConfig"""
        try:
            if not isinstance(model_config, ChatModelConfig):
                logger.warning(f"Model is not a chat model: {model_config.model_name}")
                return None

            return create_base_llm(model_config)
        except Exception as e:
            logger.error(f"Error creating LLM instance: {e}")
            return None

    def get_llm_by_id(self, model_id: str) -> Optional[BaseLLM]:
        """Get LLM instance by model_id"""
        try:
            # Strip whitespace from model_id
            model_id = model_id.strip() if isinstance(model_id, str) else model_id
            model_config = self.load(model_id)
            return self.create_llm_instance(model_config)
        except ValueError:
            return None

    def get_llm_by_name(self, model_name: str) -> Optional[BaseLLM]:
        """Get LLM instance by model_name (alias for get_llm_by_id)"""
        return self.get_llm_by_id(model_name)

    def get_all_active_models(self) -> dict[str, ModelConfig]:
        """Get all active models."""
        return self.list()

    def add_model(
        self,
        model_id: str,
        model_provider: str,
        model_name: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        abilities: Optional[List[str]] = None,
        description: Optional[str] = None,
        category: str = "llm",
    ) -> None:
        """Add a new model to storage"""
        # Strip whitespace from string fields
        model_id = model_id.strip() if isinstance(model_id, str) else model_id
        model_provider = (
            model_provider.strip()
            if isinstance(model_provider, str)
            else model_provider
        )
        model_name = model_name.strip() if isinstance(model_name, str) else model_name
        api_key = api_key.strip() if isinstance(api_key, str) else api_key
        base_url = base_url.strip() if isinstance(base_url, str) else base_url
        description = (
            description.strip() if isinstance(description, str) else description
        )

        if category == "llm":
            model_config: ModelConfig = ChatModelConfig(
                id=model_id,
                model_name=model_name,
                model_provider=model_provider,
                api_key=api_key,
                base_url=base_url,
                default_temperature=temperature,
                abilities=abilities,
                description=description,
            )
        elif category == "embedding":
            model_config = EmbeddingModelConfig(
                id=model_id,
                model_name=model_name,
                model_provider=model_provider,
                api_key=api_key,
                base_url=base_url,
                abilities=abilities,
                description=description,
            )
        elif category == "rerank":
            model_config = RerankModelConfig(
                id=model_id,
                model_name=model_name,
                api_key=api_key,
                base_url=base_url,
                abilities=abilities,
                description=description,
            )
        else:
            raise ValueError(f"Unsupported category: {category}")

        self.store(model_config)

    def update_model(self, model_id: str, **kwargs: Any) -> bool:
        """Update model configuration"""
        try:
            # Strip whitespace from model_id
            model_id = model_id.strip() if isinstance(model_id, str) else model_id

            model_config = self.load(model_id)

            # Strip whitespace from string fields
            for key, value in kwargs.items():
                if isinstance(value, str):
                    kwargs[key] = value.strip()

            # Update fields
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)

            # Delete old and store updated
            self.delete(model_id)
            self.store(model_config)
            return True
        except ValueError:
            return False

    def delete_model(self, model_id: str) -> bool:
        """Delete a model (uses parent's delete method)"""
        try:
            self.delete(model_id)
            return True
        except Exception:
            return False

    def get_db_model(self, model_id: Union[str, int]) -> Model | None:
        """Get a model by id, model_id or model_name"""
        if isinstance(model_id, str):
            model_id = model_id.strip()

        db_model = None

        # Try by integer ID first if applicable
        if isinstance(model_id, int) or (
            isinstance(model_id, str) and model_id.isdigit()
        ):
            try:
                int_id = int(model_id)
                db_model = (
                    self.db.query(self.Model).filter(self.Model.id == int_id).first()
                )
            except ValueError:
                pass

        if db_model:
            return db_model

        # Try to find by model_id
        db_model = (
            self.db.query(self.Model)
            .filter(self.Model.model_id == str(model_id))
            .first()
        )
        # If not found by model_id, try by model_name
        if not db_model:
            db_model = (
                self.db.query(self.Model)
                .filter(self.Model.model_name == str(model_id))
                .first()
            )
        return db_model

    def set_model_active(self, model_id: str, is_active: bool) -> bool:
        """Set model active status"""
        # Strip whitespace from model_id
        model_id = model_id.strip() if isinstance(model_id, str) else model_id

        db_model = (
            self.db.query(self.Model).filter(self.Model.model_id == model_id).first()
        )

        if not db_model:
            return False

        db_model.is_active = bool(is_active)  # type: ignore[assignment]
        self.db.commit()
        return True


class UserAwareModelStorage:
    """
    Extends core model storage with user access control.

    This wraps the core standalone storage and adds user-specific logic.
    """

    def __init__(self, db: Session):
        """
        Initialize user-aware model storage.

        Args:
            db: Database session
        """
        self.db = db
        self.core_storage = CoreStorage(db, Model)

    def get_llm_by_id(
        self, model_id: str, user_id: Optional[int] = None
    ) -> Optional[BaseLLM]:
        """
        Get LLM instance by model ID with user access control.
        Alias for get_llm_by_name_with_access since it handles both ID and name.
        """
        return self.get_llm_by_name_with_access(model_id, user_id)

    def get_llm_by_name_with_access(
        self, model_name: str, user_id: Optional[int] = None
    ) -> Optional[BaseLLM]:
        """
        Get LLM instance by model name with user access control.

        Args:
            model_name: Model identifier (model_id or model_name)
            user_id: User ID to check access. If None, only checks if model exists and is active.

        Returns:
            LLM instance if found and accessible, None otherwise
        """
        try:
            # Try to get by model_id first, then by model_name
            logger.info(f"Looking for model: {model_name} for user {user_id}")
            model_config = self.core_storage.load(model_name)
            db_model = self.core_storage.get_db_model(model_name)
            if not db_model:
                logger.warning(f"Cannot find model for id: {model_name}")
                return None
            logger.info(
                f"Found model: id={db_model.id}, model_id={db_model.model_id}, model_name={db_model.model_name}"
            )
            if not isinstance(model_config, ChatModelConfig):
                logger.warning(f"Invalid model type: {type(db_model).__name__}")
                return None

            # If user_id is provided, check access permissions
            if user_id is not None:
                logger.info(
                    f"Checking user access: user_id={user_id}, model_id={db_model.id}"
                )
                user_model = (
                    self.db.query(UserModel)
                    .filter(
                        UserModel.user_id == user_id,
                        UserModel.model_id == db_model.id,
                    )
                    .first()
                )

                if not user_model:
                    logger.warning(
                        f"User {user_id} does not have access to model '{model_name}' (db_model.id={db_model.id})"
                    )
                    return None
                else:
                    logger.info(f"User {user_id} has access to model '{model_name}'")

            return self.core_storage.create_llm_instance(model_config)
        except Exception as e:
            logger.error(f"Error getting LLM instance for model '{model_name}': {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def get_configured_defaults(
        self, user_id: Optional[int] = None
    ) -> Tuple[
        Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM]
    ]:
        """
        Get configured default LLMs for a user.

        Args:
            user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

        Returns:
            Tuple of (default_llm, fast_llm, vision_llm, compact_llm)
        """
        try:
            default_llm = None
            fast_llm = None
            vision_llm = None
            compact_llm = None

            # Try to get user-specific defaults first
            if user_id:
                # Get general default model
                general_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "general",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if general_default and general_default.model:
                    model_config = self.core_storage.load(
                        general_default.model.model_id
                    )
                    default_llm = self.core_storage.create_llm_instance(model_config)

                # Get small/fast model
                fast_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "small_fast",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if fast_default and fast_default.model:
                    model_config = self.core_storage.load(fast_default.model.model_id)
                    fast_llm = self.core_storage.create_llm_instance(model_config)

                # Get vision model
                vision_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "visual",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if vision_default and vision_default.model:
                    model_config = self.core_storage.load(vision_default.model.model_id)
                    vision_llm = self.core_storage.create_llm_instance(model_config)

                # Get compact model
                compact_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "compact",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if compact_default and compact_default.model:
                    model_config = self.core_storage.load(
                        compact_default.model.model_id
                    )
                    compact_llm = self.core_storage.create_llm_instance(model_config)

            # If user-specific defaults are not complete, try admin shared defaults
            if not default_llm or not fast_llm or not vision_llm or not compact_llm:
                # Get admin defaults (shared models)
                admin_defaults = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.config_type.in_(
                            ["general", "small_fast", "visual", "compact"]
                        ),
                        UserModel.is_shared,
                    )
                    .all()
                )

                for admin_default in admin_defaults:
                    model_config = self.core_storage.load(admin_default.model.model_id)
                    if not default_llm and admin_default.config_type == "general":
                        default_llm = self.core_storage.create_llm_instance(
                            model_config
                        )
                    elif not fast_llm and admin_default.config_type == "small_fast":
                        fast_llm = self.core_storage.create_llm_instance(model_config)
                    elif not vision_llm and admin_default.config_type == "visual":
                        vision_llm = self.core_storage.create_llm_instance(model_config)
                    elif not compact_llm and admin_default.config_type == "compact":
                        compact_llm = self.core_storage.create_llm_instance(
                            model_config
                        )

            # Fallback to environment variables if no configured models
            if not default_llm:
                default_llm = create_llm_from_env()
                if default_llm:
                    logger.info("Using environment variables for default LLM")

            if not fast_llm:
                fast_llm = default_llm

            if not vision_llm:
                vision_llm = default_llm

            if not compact_llm:
                compact_llm = default_llm

            return default_llm, fast_llm, vision_llm, compact_llm

        except Exception as e:
            logger.error(f"Error getting configured defaults: {e}")
            # Final fallback to environment variables
            default_llm = create_llm_from_env()
            return default_llm, default_llm, default_llm, default_llm

    def resolve_llms_from_names(
        self, llm_names: Optional[List[Optional[str]]], user_id: Optional[int] = None
    ) -> Tuple[
        Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM]
    ]:
        """
        Resolve LLM instances from names with user access control.

        Args:
            llm_names: List of exactly 4 LLM names in fixed order: [default, fast_small, vision, compact]
            user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

        Returns:
            Tuple of (default_llm, fast_llm, vision_llm, compact_llm)
        """
        logger.info(
            f"resolve_llms_from_names called with llm_names: {llm_names}, user_id: {user_id}"
        )

        if not llm_names:
            logger.info("No llm_names provided, using configured defaults")
            return self.get_configured_defaults(user_id)

        if len(llm_names) != 4:
            logger.error(
                f"Expected exactly 4 LLM names, got {len(llm_names)}. Using configured defaults."
            )
            return self.get_configured_defaults(user_id)

        # Extract model names
        default_name = llm_names[0]
        fast_name = llm_names[1]
        vision_name = llm_names[2]
        compact_name = llm_names[3]

        # Get default LLM (required)
        if not default_name:
            logger.error(
                "Default model name is required but not provided. Using configured defaults."
            )
            return self.get_configured_defaults(user_id)

        default_llm = self.get_llm_by_name_with_access(default_name, user_id)
        if not default_llm:
            logger.warning(
                f"Default LLM '{default_name}' not found or no access, falling back to configured default"
            )
            default_llm, _, _, _ = self.get_configured_defaults(user_id)

        # Get specialized LLMs - load defaults once for efficiency
        _, default_fast_llm, default_vision_llm, default_compact_llm = (
            self.get_configured_defaults(user_id)
        )

        # Get fast LLM (optional)
        fast_llm = None
        if fast_name:
            fast_llm = self.get_llm_by_name_with_access(fast_name, user_id)
            if not fast_llm:
                logger.warning(
                    f"Fast LLM '{fast_name}' not found or no access, using configured fast default"
                )
                fast_llm = default_fast_llm

        # Get vision LLM (optional)
        vision_llm = None
        if vision_name:
            vision_llm = self.get_llm_by_name_with_access(vision_name, user_id)
            if not vision_llm:
                logger.warning(
                    f"Vision LLM '{vision_name}' not found or no access, using configured vision default"
                )
                vision_llm = default_vision_llm

        # Get compact LLM (optional)
        compact_llm = None
        if compact_name:
            logger.info(f"Looking for compact LLM: {compact_name}")
            compact_llm = self.get_llm_by_name_with_access(compact_name, user_id)
            if not compact_llm:
                logger.warning(
                    f"Compact LLM '{compact_name}' not found or no access, using configured compact default"
                )
                compact_llm = default_compact_llm
            else:
                logger.info(f"Found compact LLM: {compact_llm.model_name}")
        else:
            logger.info(
                "No compact LLM specified in llm_names, using configured default"
            )
            compact_llm = default_compact_llm

        logger.info(
            f"resolve_llms_from_names returning: default={default_llm.model_name if default_llm else None}, "
            f"fast={fast_llm.model_name if fast_llm else None}, "
            f"vision={vision_llm.model_name if vision_llm else None}, "
            f"compact={compact_llm.model_name if compact_llm else None}"
        )

        return default_llm, fast_llm, vision_llm, compact_llm


# Backward-compatible wrapper functions
def resolve_llms_from_names(
    llm_names: Optional[List[Optional[str]]], db: Session, user_id: Optional[int] = None
) -> Tuple[Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM]]:
    """
    Backward-compatible wrapper for resolve_llms_from_names.

    Args:
        llm_names: List of exactly 4 LLM names
        db: Database session
        user_id: User ID for access control

    Returns:
        Tuple of (default_llm, fast_llm, vision_llm, compact_llm)
    """
    storage = UserAwareModelStorage(db)
    return storage.resolve_llms_from_names(llm_names, user_id)


def resolve_llms_for_user(
    db: Session, user_id: int
) -> Tuple[Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM]]:
    """
    Backward-compatible wrapper for getting user defaults.

    Args:
        db: Database session
        user_id: User ID

    Returns:
        Tuple of (default_llm, fast_llm, vision_llm, compact_llm)
    """
    logger.info(f"resolve_llms_for_user called with user_id: {user_id}")
    storage = UserAwareModelStorage(db)
    return storage.get_configured_defaults(user_id)


def get_llm_by_name(
    model_name: str, db: Session, user_id: Optional[int] = None
) -> Optional[BaseLLM]:
    """
    Backward-compatible wrapper for getting LLM by name.

    Args:
        model_name: Model identifier
        db: Database session
        user_id: User ID for access control

    Returns:
        LLM instance or None
    """
    storage = UserAwareModelStorage(db)
    return storage.get_llm_by_name_with_access(model_name, user_id)


def create_llm_from_env() -> Optional[BaseLLM]:
    """
    Create LLM instance from environment variables.

    Returns:
        LLM instance or None
    """
    # Try OpenAI first
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
            base_url = os.getenv("OPENAI_BASE_URL")
            return OpenAILLM(
                model_name=model_name,
                api_key=openai_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error creating OpenAI LLM from env: {e}")

    # Try Zhipu
    zhipu_key = os.getenv("ZHIPU_API_KEY")
    if zhipu_key:
        try:
            model_name = os.getenv("ZHIPU_MODEL_NAME", "glm-4")
            base_url = os.getenv("ZHIPU_BASE_URL")
            return ZhipuLLM(
                model_name=model_name,
                api_key=zhipu_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error creating Zhipu LLM from env: {e}")

    # Try Gemini
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if gemini_key:
        try:
            model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")
            base_url = os.getenv("GEMINI_BASE_URL")
            return GeminiLLM(
                model_name=model_name,
                api_key=gemini_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error creating Gemini LLM from env: {e}")

    # Try Claude
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if claude_key:
        try:
            model_name = os.getenv("CLAUDE_MODEL_NAME", "claude-3-5-sonnet-20241022")
            base_url = os.getenv("CLAUDE_BASE_URL")
            return ClaudeLLM(
                model_name=model_name,
                api_key=claude_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error creating Claude LLM from env: {e}")

    return None


def make_normalize_model_id(core_storage: CoreStorage) -> Callable:
    def normalize_model_id(model_id: Any, model_name: Any) -> Optional[str]:
        if model_id:
            db_model = core_storage.get_db_model(model_id)
            if db_model:
                return str(db_model.model_id)
            # Preserve stored identifier even if the backing model row no longer exists.
            # This avoids API inconsistencies when models are deleted/migrated.
            return str(model_id).strip() if isinstance(model_id, str) else str(model_id)
        if model_name:
            db_model = core_storage.get_db_model(str(model_name))
            return str(db_model.model_id) if db_model else None
        return None

    return normalize_model_id
