"""LLM resolution utilities for task creation with multi-tenant support"""

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, List, Optional, Tuple, Union

from sqlalchemy.orm import Session, joinedload

from ...core.model.chat.basic.adapter import create_base_llm
from ...core.model.chat.basic.base import BaseLLM
from ...core.model.chat.basic.claude import ClaudeLLM
from ...core.model.chat.basic.deepseek import DeepSeekLLM
from ...core.model.chat.basic.gemini import GeminiLLM
from ...core.model.chat.basic.openai import OpenAILLM
from ...core.model.chat.basic.zhipu import ZhipuLLM
from ...core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ModelConfig,
    MusicModelConfig,
    RerankModelConfig,
    SoundEffectModelConfig,
    VideoModelConfig,
)
from ...core.model.providers import (
    ROUTER_PROVIDER,
    is_auto_router_model,
    is_placeholder_api_key,
)
from ..models.auto_model import AutoModelCandidate, AutoModelConfig
from ..models.model import Model
from ..models.user import UserDefaultModel, UserModel

logger = logging.getLogger(__name__)

PLATFORM_MODEL_ID_PREFIX = "platform/"
PLATFORM_MODEL_MANAGER = "platform"


class PlatformModelIdentityError(ValueError):
    """Raised when an ordinary write crosses the platform model boundary."""


class AutoModelUnavailableError(RuntimeError):
    """Raised when a configured Auto model has no usable downstream model."""


class ModelWriteMode(Enum):
    """Control whether a model write owns or joins the current transaction."""

    COMMIT = "commit"
    STAGE = "stage"


def is_platform_model_id(model_id: object) -> bool:
    """Return whether an identifier belongs to the reserved platform namespace."""

    return isinstance(model_id, str) and model_id.strip().startswith(
        PLATFORM_MODEL_ID_PREFIX
    )


def _ensure_ordinary_model(model: Model) -> None:
    if is_platform_model_id(model.model_id) or (
        model.managed_by == PLATFORM_MODEL_MANAGER
    ):
        raise PlatformModelIdentityError(
            "Platform-managed models cannot be mutated through ordinary model storage"
        )


def _create_llm_instance(db_model: Model) -> BaseLLM:
    if db_model.category == "llm":
        config: ModelConfig = ChatModelConfig(
            id=db_model.model_id,
            model_name=db_model.model_name,
            model_provider=db_model.model_provider,
            api_key=db_model.api_key,
            base_url=db_model.base_url,
            default_temperature=db_model.temperature,
            context_window=db_model.context_window,
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
            model_provider=getattr(db_model, "model_provider", "dashscope")
            or "dashscope",
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
                context_window=db_model.context_window,
            )
        elif db_model.category == "image":
            from ...core.model.model import ImageModelConfig

            return ImageModelConfig(
                **common,
                model_provider=db_model.model_provider,
                default_max_tokens=db_model.max_tokens,
            )
        elif db_model.category == "video":
            return VideoModelConfig(
                **common,
                model_provider=db_model.model_provider,
            )
        elif db_model.category == "embedding":
            return EmbeddingModelConfig(
                **common,
                model_provider=db_model.model_provider,
                dimension=db_model.dimension,
            )
        elif db_model.category == "rerank":
            return RerankModelConfig(
                **common,
                model_provider=db_model.model_provider,
            )
        elif db_model.category == "speech":
            from ...core.model.model import SpeechModelConfig

            return SpeechModelConfig(
                **common,
                model_provider=db_model.model_provider,
            )
        elif db_model.category == "sound_effect":
            return SoundEffectModelConfig(
                **common,
                model_provider=db_model.model_provider,
            )
        elif db_model.category == "music":
            return MusicModelConfig(
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

        if is_platform_model_id(model.id):
            raise PlatformModelIdentityError(
                "Model IDs beginning with 'platform/' are reserved"
            )
        self._store(model)

    def _store(
        self,
        model: ModelConfig,
        *,
        managed_by: str | None = None,
        is_active: bool = True,
        write_mode: ModelWriteMode = ModelWriteMode.COMMIT,
    ) -> Model:
        """Persist a model, with provenance available only to trusted wrappers."""

        if not isinstance(write_mode, ModelWriteMode):
            raise TypeError("write_mode must be a ModelWriteMode")

        db_data: dict[str, Any] = {
            "model_id": model.id,
            "model_name": model.model_name,
            "api_key": model.api_key,
            "base_url": model.base_url,
            "abilities": model.abilities,
            "description": model.description,
            "max_retries": model.max_retries,
            "is_active": is_active,
            "managed_by": managed_by,
        }

        if isinstance(model, ChatModelConfig):
            db_data.update(
                {
                    "model_provider": model.model_provider,
                    "temperature": model.default_temperature,
                    "max_tokens": model.default_max_tokens,
                    "context_window": model.context_window,
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
                    "model_provider": model.model_provider,
                    "category": "rerank",
                }
            )
        else:
            # Try ImageModelConfig or SpeechModelConfig
            from ...core.model.model import (
                ImageModelConfig,
                MusicModelConfig,
                SoundEffectModelConfig,
                SpeechModelConfig,
                VideoModelConfig,
            )

            if isinstance(model, ImageModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "max_tokens": model.default_max_tokens,
                        "category": "image",
                    }
                )
            elif isinstance(model, VideoModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "category": "video",
                    }
                )
            elif isinstance(model, SpeechModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "category": "speech",
                    }
                )
            elif isinstance(model, SoundEffectModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "category": "sound_effect",
                    }
                )
            elif isinstance(model, MusicModelConfig):
                db_data.update(
                    {
                        "model_provider": model.model_provider,
                        "category": "music",
                    }
                )
            else:
                raise ValueError(f"Unsupported model type: {type(model)}")

        db_model = self.Model(**db_data)
        self.db.add(db_model)
        if write_mode is ModelWriteMode.STAGE:
            # Staged callers need the database ID for links/defaults while
            # retaining ownership of the surrounding transaction.
            self.db.flush()
        else:
            self.db.commit()
        return db_model

    def delete(self, model_id: str) -> None:
        """Delete model by model_id."""
        db_model = (
            self.db.query(self.Model).filter(self.Model.model_id == model_id).first()
        )
        if db_model:
            _ensure_ordinary_model(db_model)
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

    def create_llm_instance(
        self,
        model_config: ModelConfig,
        downstream_resolver: Optional[Callable[[str], BaseLLM]] = None,
    ) -> Optional[BaseLLM]:
        """Create LLM instance from ModelConfig.

        ``downstream_resolver`` is forwarded to virtual router models so Auto
        can dispatch through the selected saved model configuration.
        """
        try:
            if not isinstance(model_config, ChatModelConfig):
                logger.warning(f"Model is not a chat model: {model_config.model_name}")
                return None

            # Keep the bare create_base_llm(config) call for ordinary models;
            # only virtual "auto" models need the downstream resolver.
            if downstream_resolver is not None:
                return create_base_llm(model_config, downstream_resolver)
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
        elif category == "video":
            model_config = VideoModelConfig(
                id=model_id,
                model_name=model_name,
                model_provider=model_provider,
                api_key=api_key,
                base_url=base_url,
                abilities=abilities or ["generate"],
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

            db_model = self.get_db_model(model_id)
            if db_model is None:
                raise ValueError(f"Model not found: {model_id}")
            _ensure_ordinary_model(db_model)
            model_config = self._db_model_to_config(db_model)

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

        _ensure_ordinary_model(db_model)

        db_model.is_active = bool(is_active)  # type: ignore[assignment]
        self.db.commit()
        return True


class PlatformModelStore:
    """Trusted persistence boundary for host-managed platform models."""

    def __init__(self, db: Session, model_class: type[Model] = Model):
        self.db = db
        self.Model = model_class
        self._storage = CoreStorage(db, model_class)

    def create(
        self,
        model: ModelConfig,
        *,
        is_active: bool = True,
        write_mode: ModelWriteMode = ModelWriteMode.COMMIT,
    ) -> Model:
        """Create a platform model without tenant ownership or default links.

        ``STAGE`` flushes the INSERT so the returned row has a stable database
        ID, but leaves final commit or rollback ownership with the caller.
        """

        if not is_platform_model_id(model.id):
            raise PlatformModelIdentityError(
                "Platform-managed model IDs must begin with 'platform/'"
            )
        existing = (
            self.db.query(self.Model).filter(self.Model.model_id == model.id).first()
        )
        if existing is not None:
            raise PlatformModelIdentityError(f"Model ID is already claimed: {model.id}")

        return self._storage._store(
            model,
            managed_by=PLATFORM_MODEL_MANAGER,
            is_active=is_active,
            write_mode=write_mode,
        )

    def get(self, model_id: str) -> Model | None:
        """Read an exactly matching platform-managed row."""

        return (
            self.db.query(self.Model)
            .filter(
                self.Model.model_id == model_id,
                self.Model.managed_by == PLATFORM_MODEL_MANAGER,
            )
            .first()
        )


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
                # Step 1: own UserModel (must be owner)
                user_model = (
                    self.db.query(UserModel)
                    .filter(
                        UserModel.user_id == user_id,
                        UserModel.model_id == db_model.id,
                        UserModel.is_owner.is_(True),
                    )
                    .first()
                )
                # Step 2: shared from visible users
                if not user_model:
                    from .model_service import _get_visible_user_ids

                    visible_ids = _get_visible_user_ids(self.db, user_id)
                    user_model = (
                        self.db.query(UserModel)
                        .filter(
                            UserModel.model_id == db_model.id,
                            UserModel.user_id.in_(visible_ids),
                            UserModel.is_shared.is_(True),
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

            if is_auto_router_model(
                model_config.model_provider, model_config.model_name
            ):
                if model_config.model_provider == ROUTER_PROVIDER:
                    configured_model, resolver = self._build_configured_router_resolver(
                        model_config, db_model
                    )
                    return self.core_storage.create_llm_instance(
                        configured_model,
                        downstream_resolver=resolver,
                    )
                return self.core_storage.create_llm_instance(
                    model_config,
                    downstream_resolver=self._build_openrouter_resolver(model_config),
                )
            return self.core_storage.create_llm_instance(model_config)
        except AutoModelUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Error getting LLM instance for model '{model_name}': {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def _build_openrouter_resolver(
        self, openrouter_cfg: ChatModelConfig
    ) -> Callable[[str], BaseLLM]:
        """Build the "auto" downstream resolver from its own OpenRouter config.

        The ``auto`` model *is* an OpenRouter model, so the chosen slug is run
        with that same config's credentials + base_url. Returns a closure that,
        given a slug, builds the concrete downstream LLM.
        """

        def _resolve(slug: str) -> BaseLLM:
            child = openrouter_cfg.model_copy(
                update={"id": f"router:{slug}", "model_name": slug}
            )
            llm = self.core_storage.create_llm_instance(child)
            if llm is None:
                raise RuntimeError(
                    f"failed to build OpenRouter downstream LLM for {slug!r}"
                )
            return llm

        return _resolve

    def _build_configured_router_resolver(
        self, router_cfg: ChatModelConfig, router_model: Model
    ) -> tuple[ChatModelConfig, Callable[[str], BaseLLM]]:
        """Snapshot the user's Auto bindings into a resolver for this run."""

        from .auto_model_service import (
            AUTO_ROUTER_CONFIG_NAME,
            AutoModelConfigurationError,
            AutoModelDependencyError,
            load_router_profile_catalog,
            validate_candidate_modalities,
        )
        from .model_service import _is_model_visible_to_user

        config = (
            self.db.query(AutoModelConfig)
            .options(
                joinedload(AutoModelConfig.candidates).joinedload(
                    AutoModelCandidate.target_model
                )
            )
            .filter(AutoModelConfig.router_model_id == router_model.id)
            .first()
        )
        if config is None or not config.candidates:
            raise AutoModelUnavailableError("Auto model has no configured candidates")

        targets_by_profile: dict[str, ChatModelConfig] = {}
        fallback_profile: str | None = None
        for candidate in config.candidates:
            target = candidate.target_model
            if (
                target is None
                or not target.is_active
                or target.category != "llm"
                or is_auto_router_model(target.model_provider, target.model_name)
                or not _is_model_visible_to_user(
                    self.db, target.id, int(config.user_id)
                )
            ):
                logger.warning(
                    "Skipping unavailable Auto candidate %r",
                    candidate.routing_model_id,
                )
                continue
            target_cfg = self.core_storage._db_model_to_config(target)
            if not isinstance(target_cfg, ChatModelConfig):
                logger.warning(
                    "Skipping non-chat Auto candidate %r",
                    candidate.routing_model_id,
                )
                continue
            targets_by_profile[candidate.routing_model_id] = target_cfg
            if candidate.target_model_id == config.fallback_model_id:
                fallback_profile = candidate.routing_model_id

        if not targets_by_profile:
            raise AutoModelUnavailableError(
                "Auto model has no active configured candidates"
            )

        try:
            catalog = load_router_profile_catalog()
            for profile_id, target_cfg in targets_by_profile.items():
                validate_candidate_modalities(
                    catalog, profile_id, target_cfg.abilities or []
                )
        except (AutoModelConfigurationError, AutoModelDependencyError) as exc:
            raise AutoModelUnavailableError(str(exc)) from exc

        profile_ids = list(targets_by_profile)
        configured = router_cfg.model_copy(
            update={
                "router_config_name": AUTO_ROUTER_CONFIG_NAME,
                "router_candidate_models": profile_ids,
                "router_fallback_model": fallback_profile,
            }
        )

        def _resolve(profile_id: str) -> BaseLLM:
            target_cfg = targets_by_profile.get(profile_id)
            if target_cfg is None:
                raise RuntimeError(
                    f"xrouter selected unbound Auto profile {profile_id!r}"
                )
            llm = self.core_storage.create_llm_instance(target_cfg)
            if llm is None:
                raise RuntimeError(
                    f"failed to build Auto downstream LLM for {profile_id!r}"
                )
            return llm

        return configured, _resolve

    def _create_default_model(
        self, db_model: Model, user_id: Optional[int]
    ) -> Optional[BaseLLM]:
        """Create a default model, hydrating configured Auto when necessary."""

        model_id = str(db_model.model_id)
        model_config = self.core_storage.load(model_id)
        if (
            getattr(model_config, "model_provider", None) == ROUTER_PROVIDER
            and getattr(model_config, "model_name", None) == "auto"
        ):
            return self.get_llm_by_name_with_access(model_id, user_id)
        return self.core_storage.create_llm_instance(model_config)

    def get_configured_defaults(
        self,
        user_id: Optional[int] = None,
        *,
        config_types: tuple[str, ...] = ("general", "small_fast", "visual", "compact"),
        fallback_llm: Optional[BaseLLM] = None,
    ) -> Tuple[
        Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM], Optional[BaseLLM]
    ]:
        """
        Get configured default LLMs for a user.

        Args:
            user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.
            config_types: Only these default slots are instantiated.
            fallback_llm: Explicit general LLM to use when a requested specialized slot is unset.

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
                from ..models.model import Model as DBModel

                general_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        DBModel,
                        UserDefaultModel.model_id == DBModel.id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "general",
                        DBModel.is_active,
                    )
                    .first()
                )

                if (
                    "general" in config_types
                    and general_default
                    and general_default.model
                ):
                    from .model_service import _is_model_visible_to_user

                    if _is_model_visible_to_user(
                        self.db, general_default.model.id, user_id
                    ):
                        default_llm = self._create_default_model(
                            general_default.model, user_id
                        )

                # Get small/fast model
                fast_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        DBModel,
                        UserDefaultModel.model_id == DBModel.id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "small_fast",
                        DBModel.is_active,
                    )
                    .first()
                )

                if "small_fast" in config_types and fast_default and fast_default.model:
                    from .model_service import _is_model_visible_to_user

                    if _is_model_visible_to_user(
                        self.db, fast_default.model.id, user_id
                    ):
                        fast_llm = self._create_default_model(
                            fast_default.model, user_id
                        )

                # Get vision model
                vision_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        DBModel,
                        UserDefaultModel.model_id == DBModel.id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "visual",
                        DBModel.is_active,
                    )
                    .first()
                )

                if "visual" in config_types and vision_default and vision_default.model:
                    from .model_service import _is_model_visible_to_user

                    if _is_model_visible_to_user(
                        self.db, vision_default.model.id, user_id
                    ):
                        vision_llm = self._create_default_model(
                            vision_default.model, user_id
                        )

                # Get compact model
                compact_default = (
                    self.db.query(UserDefaultModel)
                    .join(
                        DBModel,
                        UserDefaultModel.model_id == DBModel.id,
                    )
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "compact",
                        DBModel.is_active,
                    )
                    .first()
                )

                if (
                    "compact" in config_types
                    and compact_default
                    and compact_default.model
                ):
                    from .model_service import _is_model_visible_to_user

                    if _is_model_visible_to_user(
                        self.db, compact_default.model.id, user_id
                    ):
                        compact_llm = self._create_default_model(
                            compact_default.model, user_id
                        )

            # If user-specific defaults are not complete, try visible users' shared defaults
            if not default_llm or not fast_llm or not vision_llm or not compact_llm:
                from .model_service import _get_visible_user_ids

                visible_ids = _get_visible_user_ids(self.db, user_id)
                # Get visible users' shared defaults
                admin_defaults = (
                    self.db.query(UserDefaultModel)
                    .join(
                        UserModel,
                        UserDefaultModel.model_id == UserModel.model_id,
                    )
                    .filter(
                        UserDefaultModel.config_type.in_(config_types),
                        UserModel.is_shared.is_(True),
                        UserDefaultModel.user_id.in_(visible_ids),
                    )
                    .all()
                )

                for admin_default in admin_defaults:
                    if not default_llm and admin_default.config_type == "general":
                        default_llm = self._create_default_model(
                            admin_default.model, user_id
                        )
                    elif not fast_llm and admin_default.config_type == "small_fast":
                        fast_llm = self._create_default_model(
                            admin_default.model, user_id
                        )
                    elif not vision_llm and admin_default.config_type == "visual":
                        vision_llm = self._create_default_model(
                            admin_default.model, user_id
                        )
                    elif not compact_llm and admin_default.config_type == "compact":
                        compact_llm = self._create_default_model(
                            admin_default.model, user_id
                        )

            requested_defaults = {
                "general": default_llm,
                "small_fast": fast_llm,
                "visual": vision_llm,
                "compact": compact_llm,
            }
            if any(requested_defaults[kind] is None for kind in config_types):
                default_llm = default_llm or fallback_llm
                if default_llm is None and "general" not in config_types:
                    default_llm, _, _, _ = self.get_configured_defaults(
                        user_id, config_types=("general",)
                    )
                if default_llm is None:
                    default_llm = create_llm_from_env()

            if "small_fast" in config_types and not fast_llm:
                fast_llm = default_llm

            if "visual" in config_types and not vision_llm:
                vision_llm = default_llm

            if "compact" in config_types and not compact_llm:
                compact_llm = default_llm

            return default_llm, fast_llm, vision_llm, compact_llm

        except AutoModelUnavailableError:
            raise
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
            default_llm, _, _, _ = self.get_configured_defaults(
                user_id, config_types=("general",)
            )

        # Get fast LLM (optional)
        fast_llm = None
        if fast_name:
            fast_llm = self.get_llm_by_name_with_access(fast_name, user_id)

        # Get vision LLM (optional)
        vision_llm = None
        if vision_name:
            vision_llm = self.get_llm_by_name_with_access(vision_name, user_id)

        # Get compact LLM (optional)
        compact_llm = None
        if compact_name:
            logger.info(f"Looking for compact LLM: {compact_name}")
            compact_llm = self.get_llm_by_name_with_access(compact_name, user_id)

        default_fast_llm = None
        default_vision_llm = None
        default_compact_llm = None
        missing_defaults = tuple(
            kind
            for kind, needed in (
                ("small_fast", bool(fast_name) and fast_llm is None),
                ("visual", bool(vision_name) and vision_llm is None),
                ("compact", compact_llm is None),
            )
            if needed
        )
        if missing_defaults:
            _, default_fast_llm, default_vision_llm, default_compact_llm = (
                self.get_configured_defaults(
                    user_id, config_types=missing_defaults, fallback_llm=default_llm
                )
            )

        if fast_name and not fast_llm:
            logger.warning(
                f"Fast LLM '{fast_name}' not found or no access, using configured fast default"
            )
            fast_llm = default_fast_llm

        if vision_name and not vision_llm:
            logger.warning(
                f"Vision LLM '{vision_name}' not found or no access, using configured vision default"
            )
            vision_llm = default_vision_llm

        if compact_name:
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
    if openai_key and not is_placeholder_api_key(openai_key):
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

    # Try DeepSeek
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_key and not is_placeholder_api_key(deepseek_key):
        try:
            model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash")
            base_url = os.getenv("DEEPSEEK_BASE_URL")
            return DeepSeekLLM(
                model_name=model_name,
                api_key=deepseek_key,
                base_url=base_url,
            )
        except Exception as e:
            logger.error(f"Error creating DeepSeek LLM from env: {e}")

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


@dataclass(frozen=True)
class AgentRuntimeFields:
    """Primitive subset of the ``Agent`` row produced by
    ``resolve_task_runtime_config_core``.

    Snapshot consumers (``load_task_setup_snapshot_sync``) expose
    this directly on ``TaskSetupSnapshot.agent``; main-loop
    consumers (``_reconstruct_agent_from_history``) read the same
    fields off the resolved ``RuntimeConfig``. One dataclass, both
    code paths -- no parallel definitions to drift.
    """

    id: int
    name: str
    status: Any  # AgentStatus enum (typed loosely to avoid an import cycle)
    instructions: Optional[str]
    # ``None`` for a personal/legacy agent and for every task with no
    # ``agent_id``. Read by the team-scope connector-visibility hook seam;
    # a fail-closed default keeps an un-migrated construction site personal.
    team_id: Optional[int] = None
    # The Agent row's own creator, never the runner of this task. The one
    # construction site in this module fills it unconditionally from the
    # Agent row's ``user_id``, for a team agent and a personal one alike -- unlike
    # ``team_id`` above, a personal agent still has a creator. The ``None``
    # default is what a task with no Agent row gets (``TaskSetupSnapshot``
    # leaves ``agent`` unset there) and what an un-migrated future
    # construction site would get; it is a fail-closed default, not a
    # description of the personal-agent case. Named for what it is -- the
    # agent's creator -- not "owner", which this module already uses
    # elsewhere for the running principal.
    agent_creator_user_id: Optional[int] = None


@dataclass(frozen=True)
class RuntimeConfig:
    """Resolved task runtime configuration: the LLM tuple, execution
    pattern, and optional agent-builder overlay returned by
    ``resolve_task_runtime_config_core``.

    Frozen dataclass instead of a free-form dict because:

      - typo'd field access fails at type-check time, not silently
        returning ``None`` at runtime
      - the contract is greppable / introspectable
      - both consumers (main-loop wrapper + off-loop snapshot loader)
        access fields by name; the dataclass lets ``mypy`` follow
        the type through both branches

    ``agent_config`` stays a ``dict | None`` because it carries
    ``saved_model_ids`` / ``saved_model_descriptors`` (downstream
    diagnostics) plus ``skills`` / ``knowledge_bases`` / etc.;
    promoting that to a dataclass is a separate refactor.
    """

    llms: Tuple[
        Optional[BaseLLM],
        Optional[BaseLLM],
        Optional[BaseLLM],
        Optional[BaseLLM],
    ]
    task_pattern: str
    agent_config: Optional[dict]
    has_agent_builder_config: bool
    excluded_agent_id: Optional[int]
    agent_fields: Optional[AgentRuntimeFields]
    # Workforce runtime if the task carries ``workforce_run_id`` in its
    # ``agent_config``. ``Any`` instead of ``WorkforceTaskRuntime`` to
    # keep this module free of the workforce import cycle -- callers
    # that need the typed view import it themselves.
    workforce: Any = None


def _load_agent_for_task_runtime(
    session: Session,
    task_row: Any,
    workforce: Any,
) -> Any | None:
    from ..models.agent import (
        Agent,
        AgentStatus,
        is_workforce_generated_manager_agent,
    )
    from ..models.user import User
    from .agent_access import list_accessible_published_agents
    from .agent_team_scope import get_agent_team_scope, owned_agent_clause

    task_agent_id = getattr(task_row, "agent_id", None)
    if task_agent_id is None:
        return None

    agent = session.query(Agent).filter(Agent.id == task_agent_id).first()
    if agent is None:
        return None
    if is_workforce_generated_manager_agent(agent):
        if workforce is not None and workforce.manager_agent_id == task_agent_id:
            return agent
        return None
    # Team-scoped ownership keyed on the task owner: a teammate's task may run
    # a team-visible agent; admins-only stays hidden from non-admins.
    task_user_id = int(task_row.user_id)
    owned = (
        session.query(Agent)
        .filter(
            Agent.id == task_agent_id,
            owned_agent_clause(
                task_user_id, get_agent_team_scope(session, task_user_id)
            ),
        )
        .first()
    )
    if owned is not None:
        return owned
    if workforce is not None and workforce.manager_agent_id == task_agent_id:
        return agent
    if getattr(agent.status, "value", agent.status) != AgentStatus.PUBLISHED.value:
        return None

    user = session.query(User).filter(User.id == task_row.user_id).first()
    if user is None:
        return None
    visible_agent_ids = {
        int(item.id)
        for item in list_accessible_published_agents(
            session,
            user,
            purpose="agent_list",
        )
    }
    return agent if int(agent.id) in visible_agent_ids else None


def resolve_task_runtime_config_core(
    task_row: Any,
    session: Session,
    *,
    user_id: Optional[int],
) -> RuntimeConfig:
    """Resolve a task's LLM tuple, agent-builder overlay, and execution
    pattern in one shot. Pure function over an open SQLAlchemy session
    -- no event loop, no logging, no fallback.

    Single source of truth shared by both paths that need to bootstrap
    a task's runtime configuration:

      - ``AgentServiceManager._resolve_task_runtime_config`` (main
        loop, used by ``_reconstruct_agent_from_history``) -- wraps
        this with logging + the ``_pick_default_llm_with_warning``
        fallback.
      - ``load_task_setup_snapshot_sync`` (worker thread, used by
        ``_schedule_bg._runner`` on the normal-creation path) --
        wraps this with primitive ``_TaskFields`` snapshotting so no
        ORM ``Task`` row escapes the loader's session.

    Does NOT apply the ``_pick_default_llm_with_warning`` fallback --
    that helper raises ``HTTPException``, which is unsafe to call from
    a worker thread. Callers handle the fallback step.
    """
    from ...config import (
        get_agent_pattern_for_execution_mode,
        get_default_task_execution_mode,
    )
    from ..models.agent import AgentStatus

    task_execution_mode = getattr(task_row, "execution_mode", None)
    if not task_execution_mode:
        task_execution_mode = get_default_task_execution_mode(
            agent_id=getattr(task_row, "agent_id", None),
        )
    task_pattern = get_agent_pattern_for_execution_mode(task_execution_mode)

    # Inline LLM-id normalization (the legacy
    # ``AgentServiceManager._get_task_llm_ids`` body).
    core_storage = CoreStorage(session, Model)
    normalize = make_normalize_model_id(core_storage)
    llm_ids = [
        normalize(
            getattr(task_row, "model_id", None),
            getattr(task_row, "model_name", None),
        ),
        normalize(
            getattr(task_row, "small_fast_model_id", None),
            getattr(task_row, "small_fast_model_name", None),
        ),
        normalize(
            getattr(task_row, "visual_model_id", None),
            getattr(task_row, "visual_model_name", None),
        ),
        normalize(
            getattr(task_row, "compact_model_id", None),
            getattr(task_row, "compact_model_name", None),
        ),
    ]
    storage = UserAwareModelStorage(session)
    task_llm, task_fast_llm, task_vision_llm, task_compact_llm = (
        storage.resolve_llms_from_names(llm_ids, user_id)
    )

    agent_config: Optional[dict] = None
    has_agent_builder_config = False
    excluded_agent_id: Optional[int] = None
    agent_fields: Optional[AgentRuntimeFields] = None

    # Workforce runtime resolution -- pure query against the same
    # session, no side effect. Workforce and policy-visible agent modes
    # change two things below: (1) the Agent access check can allow a
    # cross-user manager/shared agent, and (2) workforce tasks keep their
    # own execution_mode instead of inheriting from agent_config.
    from .workforce_runtime import resolve_workforce_task_runtime

    workforce = resolve_workforce_task_runtime(session, task_row)

    if task_row.agent_id is not None:
        agent_row = _load_agent_for_task_runtime(session, task_row, workforce)
        if agent_row is not None:
            agent_config = load_agent_builder_config(
                agent_row, session, int(task_row.user_id)
            )
            has_agent_builder_config = True
            # Slot-wise overlay -- an agent slot wins when set,
            # otherwise the task's own LLM stays. Mirrors the legacy
            # ``_merge_agent_builder_llms``.
            baseline_llms = (
                task_llm,
                task_fast_llm,
                task_vision_llm,
                task_compact_llm,
            )
            task_llm, task_fast_llm, task_vision_llm, task_compact_llm = (
                agent_llm or baseline
                for baseline, agent_llm in zip(baseline_llms, agent_config["llms"])
            )

            if workforce is None:
                # Non-workforce: agent_config.execution_mode overrides
                # the task's own mode (legacy agent-builder behavior).
                agent_execution_mode = agent_config.get("execution_mode", "balanced")
                task_pattern = get_agent_pattern_for_execution_mode(
                    agent_execution_mode
                )
            # workforce mode: keep task_pattern computed above from
            # task.execution_mode; do not let agent_config override.

            if agent_row.status == AgentStatus.PUBLISHED:
                excluded_agent_id = int(agent_row.id)

            agent_fields = AgentRuntimeFields(
                id=int(agent_row.id),
                name=str(agent_row.name),
                status=agent_row.status,
                instructions=(
                    str(agent_row.instructions)
                    if agent_row.instructions is not None
                    else None
                ),
                team_id=(
                    int(agent_row.team_id) if agent_row.team_id is not None else None
                ),
                agent_creator_user_id=int(agent_row.user_id),
            )
    else:
        # Inline agent_config path: task carries its own agent config
        # dict (no Agent row reference). Used by build-preview tasks
        # routed through normal task flow.
        inline_agent_config = load_task_inline_agent_config(task_row)
        if inline_agent_config is not None:
            agent_config = inline_agent_config
            inline_execution_mode = agent_config.get("execution_mode") or "balanced"
            task_pattern = get_agent_pattern_for_execution_mode(inline_execution_mode)

    return RuntimeConfig(
        llms=(task_llm, task_fast_llm, task_vision_llm, task_compact_llm),
        task_pattern=task_pattern,
        agent_config=agent_config,
        has_agent_builder_config=has_agent_builder_config,
        excluded_agent_id=excluded_agent_id,
        agent_fields=agent_fields,
        workforce=workforce,
    )


def load_task_inline_agent_config(task: Any) -> Optional[dict]:
    """Build an inline agent_config dict from ``task.agent_config`` JSON.

    Returns ``None`` when the task carries no inline agent fields (the
    common case: the task references an Agent row by ``agent_id``).

    Inline configs are used by build-preview tasks (#459) that get
    routed through normal task flow with their config embedded in the
    Task row rather than a separate Agent row.

    Shape matches ``load_agent_builder_config`` so downstream consumers
    can treat them uniformly.
    """
    raw = getattr(task, "agent_config", None)
    if not isinstance(raw, dict):
        return None

    if not any(
        key in raw
        for key in ("instructions", "knowledge_bases", "skills", "tool_categories")
    ):
        return None

    return {
        "llms": (None, None, None, None),
        "execution_mode": getattr(task, "execution_mode", None) or "balanced",
        "instructions": raw.get("instructions"),
        "skills": raw.get("skills") or [],
        "knowledge_bases": raw.get("knowledge_bases") or [],
        "tool_categories": raw.get("tool_categories"),
        "memory_similarity_threshold": raw.get("memory_similarity_threshold"),
        "is_preview": raw.get("is_preview"),
        "preview_agent_id": raw.get("preview_agent_id"),
    }


def load_agent_builder_config(agent: Any, db: Session, user_id: int) -> dict:
    """Eagerly load Agent Builder configuration into a primitive dict.

    Single source of truth shared by:

      - ``AgentServiceManager._load_agent_builder_config`` (the
        legacy in-method caller in ``chat.py``)
      - ``load_task_setup_snapshot_sync._load_agent_builder_config_sync``
        (the off-loop snapshot loader)

    The two used to keep nearly-identical copies of this logic, which
    risked drift -- e.g. the snapshot version had a defensive non-
    dict ``agent.models`` guard the in-method copy lacked. Centralizing
    here keeps the LLM-resolution contract in one place.

    Returns:
        dict with keys: ``llms`` (tuple of 4 BaseLLM | None for
        ``general`` / ``small_fast`` / ``visual`` / ``compact``),
        ``execution_mode``, ``instructions``, ``skills``,
        ``knowledge_bases``, ``tool_categories``.
    """
    storage = UserAwareModelStorage(db)

    raw_models: Any = agent.models or {}
    if raw_models and not isinstance(raw_models, dict):
        # JSON column with a dict-shaped contract (slot -> DBModel.id).
        # A non-dict here means upstream data corruption or hand-edit;
        # log loudly so on-call can trace it back instead of seeing a
        # silent "no LLMs resolved" downstream.
        logger.warning(
            "Agent %s has non-dict models field (%s); treating as empty",
            agent.id,
            type(raw_models).__name__,
        )
    models: dict[str, Any] = dict(raw_models) if isinstance(raw_models, dict) else {}

    # Captures the resolved ``DBModel`` row per slot so downstream
    # fallback diagnostics (``_pick_default_llm_with_warning``) can log
    # human-readable model identifiers instead of opaque PKs.
    saved_model_descriptors: dict[str, dict[str, Any]] = {}

    def _resolve(slot: str) -> Optional[BaseLLM]:
        db_row_id = models.get(slot)
        if not db_row_id:
            return None
        db_model = db.query(Model).filter(Model.id == db_row_id).first()
        if not db_model:
            return None
        saved_model_descriptors[slot] = {
            "pk": db_model.id,
            "model_id": str(db_model.model_id),
            "model_name": getattr(db_model, "model_name", None),
        }
        return storage.get_llm_by_name_with_access(str(db_model.model_id), user_id)

    llms = (
        _resolve("general"),
        _resolve("small_fast"),
        _resolve("visual"),
        _resolve("compact"),
    )

    return {
        "llms": llms,
        "saved_model_ids": dict(models),
        "saved_model_descriptors": saved_model_descriptors,
        "execution_mode": agent.execution_mode,
        "instructions": agent.instructions,
        "skills": list(agent.skills or []),
        "knowledge_bases": list(agent.knowledge_bases or []),
        "tool_categories": list(agent.tool_categories)
        if agent.tool_categories is not None
        else None,
    }
