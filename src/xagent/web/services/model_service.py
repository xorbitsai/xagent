"""
Model service for providing model-related utilities.

This service provides centralized functionality for model resolution and management
across the xagent system with multi-tenant support.
"""

import logging
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from xagent.core.model.image.base import BaseImageModel

from ...core.model.chat.basic.base import BaseLLM
from ...core.model.image.dashscope import DashScopeImageModel
from ...core.model.image.gemini import GeminiImageModel
from ...core.model.image.openai import OpenAIImageModel
from ...core.model.image.xinference import XinferenceImageModel

logger = logging.getLogger(__name__)


def get_default_vision_model(user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get the default vision model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default vision model or None if not available
    """
    try:
        # Try to get from database (requires web context)
        from ..models.database import get_db
        from ..models.user import User, UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        # This won't work in non-web contexts, so we'll fallback to environment
        try:
            # Try to get a database session (this might fail in CLI contexts)
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                vision_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "visual",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if vision_default and vision_default.model:
                    return _create_llm_instance(vision_default.model)

            # Fallback to admin defaults first, then other shared defaults
            admin_vision_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "visual",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_vision_defaults:
                return _create_llm_instance(admin_vision_defaults[0].model)

            # If no admin defaults, fallback to any shared defaults
            vision_models = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(UserDefaultModel.config_type == "visual", UserModel.is_shared)
                .limit(1)
                .all()
            )

            if vision_models:
                return _create_llm_instance(vision_models[0].model)

        except Exception as e:
            logger.warning(f"Failed to get vision model from database: {e}")
            pass

    except ImportError:
        pass  # Web modules not available

    # No fallback to environment variables - require database configuration
    return None


def get_default_model(user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get the default general model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default general model or None if not available
    """
    try:
        from ..models.database import get_db
        from ..models.user import User, UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                general_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "general",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if general_default and general_default.model:
                    return _create_llm_instance(general_default.model)

            # Fallback to admin defaults first, then other shared defaults
            admin_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "general",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_defaults:
                return _create_llm_instance(admin_defaults[0].model)

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(UserDefaultModel.config_type == "general", UserModel.is_shared)
                .limit(1)
                .all()
            )

            if shared_defaults:
                return _create_llm_instance(shared_defaults[0].model)

        except Exception as e:
            logger.warning(f"Failed to get default model from database: {e}")
            pass

    except ImportError:
        pass

    # No fallback to environment variables - require database configuration
    return None


def get_fast_model(user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get the default fast/small model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default fast/small model or None if not available
    """
    try:
        from ..models.database import get_db
        from ..models.user import User, UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                fast_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "small_fast",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if fast_default and fast_default.model:
                    return _create_llm_instance(fast_default.model)

            # Fallback to admin defaults first, then other shared defaults
            admin_fast_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "small_fast",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_fast_defaults:
                return _create_llm_instance(admin_fast_defaults[0].model)

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "small_fast", UserModel.is_shared
                )
                .limit(1)
                .all()
            )

            if shared_defaults:
                return _create_llm_instance(shared_defaults[0].model)

        except Exception as e:
            logger.warning(f"Failed to get fast model from database: {e}")
            pass

    except ImportError:
        pass

    # For fast model, return None if not configured
    return None


def get_compact_model(user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get the default compact model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default compact model or None if not available
    """
    try:
        from ..models.database import get_db
        from ..models.user import User, UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                compact_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "compact",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if compact_default and compact_default.model:
                    return _create_llm_instance(compact_default.model)

            # Fallback to admin defaults first, then other shared defaults
            admin_compact_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "compact",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_compact_defaults:
                return _create_llm_instance(admin_compact_defaults[0].model)

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(UserDefaultModel.config_type == "compact", UserModel.is_shared)
                .limit(1)
                .all()
            )

            if shared_defaults:
                return _create_llm_instance(shared_defaults[0].model)

        except Exception as e:
            logger.warning(f"Failed to get compact model from database: {e}")
            pass

    except ImportError:
        pass

    # For compact model, return None if not configured
    return None


def get_embedding_model(user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get the default embedding model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default embedding model or None if not available
    """
    try:
        from ..models.database import get_db
        from ..models.user import User, UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                embedding_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "embedding",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if embedding_default and embedding_default.model:
                    return _create_llm_instance(embedding_default.model)

            # Fallback to admin defaults first, then other shared defaults
            admin_embedding_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "embedding",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_embedding_defaults:
                return _create_llm_instance(admin_embedding_defaults[0].model)

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "embedding", UserModel.is_shared
                )
                .limit(1)
                .all()
            )

            if shared_defaults:
                return _create_llm_instance(shared_defaults[0].model)

        except Exception as e:
            logger.warning(f"Failed to get embedding model from database: {e}")
            pass

    except ImportError:
        pass

    # No fallback to environment variables - require database configuration
    return None


def get_vision_model(db: Session, user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get vision model from database.

    Args:
        db: Database session
        user_id: User ID (currently ignored as models are shared)

    Returns:
        Vision model instance or None if not found
    """
    try:
        from sqlalchemy import or_

        from ..models.model import Model as DBModel
        from .llm_utils import _create_llm_instance

        # Query models that have vision ability in their abilities JSON field
        db_model = (
            db.query(DBModel)
            .filter(
                DBModel.category == "llm",
                DBModel.is_active,
                or_(
                    DBModel.abilities.contains(["vision"]),
                    DBModel.abilities.like('%"vision"%'),
                ),
            )
            .first()
        )

        if db_model:
            return _create_llm_instance(db_model)
        return None

    except Exception as e:
        logger.error(f"Failed to get vision model from database: {e}")
        return None


def get_image_models(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get image models from database.

    Args:
        db: Database session
        user_id: User ID (currently ignored as models are shared)

    Returns:
        Dictionary of image model instances
    """
    image_models: dict[str, BaseImageModel] = {}
    image_model: BaseImageModel
    try:
        from ..models.model import Model as DBModel

        db_models = (
            db.query(DBModel)
            .filter(
                DBModel.category == "image",
                DBModel.is_active,
            )
            .all()
        )

        for db_model in db_models:
            if not (
                api_key := str(db_model.api_key)
                if db_model.api_key is not None
                else None
            ):
                raise ValueError("Image model API key cannot be empty")
            if not (
                base_url := str(db_model.base_url)
                if db_model.base_url is not None
                else None
            ):
                raise ValueError("Image model base URL cannot be empty")
            model_provider = str(db_model.model_provider).strip().lower()
            try:
                if model_provider == "dashscope":
                    image_model = DashScopeImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate"]),  # pyright: ignore[reportArgumentType]
                    )
                    image_models[str(db_model.model_name)] = image_model
                    logger.info(f"Added image model: {db_model.model_name}")
                elif model_provider == "gemini":
                    image_model = GeminiImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate"]),  # pyright: ignore[reportArgumentType]
                    )
                    image_models[str(db_model.model_name)] = image_model
                    logger.info(f"Added image model: {db_model.model_name}")
                elif model_provider == "openai":
                    image_model = OpenAIImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate", "edit"]),  # pyright: ignore[reportArgumentType]
                    )
                    image_models[str(db_model.model_name)] = image_model
                    logger.info(f"Added image model: {db_model.model_name}")
                elif model_provider == "xinference":
                    image_model = XinferenceImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate", "edit"]),  # pyright: ignore[reportArgumentType]
                    )
                    image_models[str(db_model.model_name)] = image_model
                    logger.info(f"Added image model: {db_model.model_name}")
            except Exception as e:
                logger.warning(
                    f"Failed to create image model for {db_model.model_name}: {e}"
                )

    except Exception as e:
        logger.error(f"Failed to get image models from database: {e}")

    return image_models


def get_models_by_category(category: str, db: Session) -> list:
    """
    Get models by category from database.

    Args:
        category: Model category ('vision', 'image', 'llm', etc.)
        db: Database session

    Returns:
        List of database model records
    """
    try:
        from ..models.model import Model as DBModel

        models = (
            db.query(DBModel)
            .filter(DBModel.category == category, DBModel.is_active.is_(True))
            .all()
        )

        logger.info(f"Found {len(models)} models for category '{category}'")
        return models

    except Exception as e:
        logger.error(f"Error getting models by category '{category}': {e}")
        return []


def get_default_image_generate_model(
    user_id: Optional[int] = None,
) -> Optional[BaseImageModel]:
    """
    Get the default image generation model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default image generation model or None if not available
    """
    try:
        from ...core.model.image.adapter import get_image_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import User, UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                image_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .join(DBModel, UserModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "image",
                        UserModel.user_id == user_id,
                        DBModel.abilities.contains("generate"),
                    )
                    .first()
                )

                if image_default and image_default.model:
                    try:
                        return get_image_model_instance(image_default.model)
                    except Exception as e:
                        logger.warning(f"Failed to create image model instance: {e}")

            # Fallback to admin defaults first, then other shared defaults
            admin_image_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "image",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                    DBModel.abilities.contains("generate"),
                )
                .limit(1)
                .all()
            )

            if admin_image_defaults:
                try:
                    return get_image_model_instance(admin_image_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create image model instance: {e}")

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "image",
                    UserModel.is_shared,
                    DBModel.abilities.contains("generate"),
                )
                .limit(1)
                .all()
            )

            if shared_defaults:
                try:
                    return get_image_model_instance(shared_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create image model instance: {e}")

        except Exception as e:
            logger.warning(
                f"Failed to get default image generation model from database: {e}"
            )
            pass

    except ImportError:
        pass

    return None


def get_default_image_edit_model(
    user_id: Optional[int] = None,
) -> Optional[BaseImageModel]:
    """
    Get the default image editing model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default image editing model or None if not available
    """
    try:
        from ...core.model.image.adapter import get_image_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import User, UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                image_default = (
                    db.query(UserDefaultModel)
                    .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                    .join(DBModel, UserModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "image_edit",
                        UserModel.user_id == user_id,
                    )
                    .first()
                )

                if image_default and image_default.model:
                    try:
                        return get_image_model_instance(image_default.model)
                    except Exception as e:
                        logger.warning(f"Failed to create image model instance: {e}")

            # Fallback to admin defaults first, then other shared defaults
            admin_image_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "image_edit",
                    UserModel.is_shared,
                    UserDefaultModel.user_id.in_(
                        db.query(User.id).filter(User.is_admin)
                    ),
                )
                .limit(1)
                .all()
            )

            if admin_image_defaults:
                try:
                    return get_image_model_instance(admin_image_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create image model instance: {e}")

            # If no admin defaults, fallback to any shared defaults
            shared_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "image_edit",
                    UserModel.is_shared,
                )
                .limit(1)
                .all()
            )

            if shared_defaults:
                try:
                    return get_image_model_instance(shared_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create image model instance: {e}")

        except Exception as e:
            logger.warning(
                f"Failed to get default image editing model from database: {e}"
            )
            pass

    except ImportError:
        pass

    return None


def get_default_embedding_model(user_id: Optional[int] = None) -> Optional[str]:
    """
    Get the default embedding model ID for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The embedding model ID or None if not available
    """
    from ..models.database import get_db
    from ..models.user import User, UserDefaultModel, UserModel

    db = next(get_db())

    # If user_id is provided, get user-specific default
    if user_id:
        embedding_default = (
            db.query(UserDefaultModel)
            .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == "embedding",
                UserModel.user_id == user_id,
            )
            .first()
        )

        if embedding_default and embedding_default.model:
            return str(embedding_default.model.model_id)

    # Admin defaults
    admin_embedding_defaults = (
        db.query(UserDefaultModel)
        .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
        .filter(
            UserDefaultModel.config_type == "embedding",
            UserModel.is_shared,
            UserDefaultModel.user_id.in_(db.query(User.id).filter(User.is_admin)),
        )
        .limit(1)
        .all()
    )

    if admin_embedding_defaults:
        return str(admin_embedding_defaults[0].model.model_id)

    # Any shared defaults
    embedding_models = (
        db.query(UserDefaultModel)
        .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
        .filter(UserDefaultModel.config_type == "embedding", UserModel.is_shared)
        .limit(1)
        .all()
    )

    if embedding_models:
        return str(embedding_models[0].model.model_id)

    return None
