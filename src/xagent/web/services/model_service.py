"""
Model service for providing model-related utilities.

This service provides centralized functionality for model resolution and management
across the xagent system with multi-tenant support.
"""

import json
import logging
from typing import Any, Dict, Optional, cast

from sqlalchemy.orm import Session

from xagent.core.model.image.base import BaseImageModel
from xagent.web.api.model import DBModel

from ...core.model.chat.basic.base import BaseLLM
from ...core.model.image.dashscope import DashScopeImageModel
from ...core.model.image.gemini import GeminiImageModel
from ...core.model.image.openai import OpenAIImageModel
from ...core.model.image.xinference import XinferenceImageModel
from ...core.model.video.base import BaseVideoModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook infrastructure for dynamic model sharing
# ---------------------------------------------------------------------------

_visible_user_ids_hook = None  # (db: Session, user_id: int) -> list[int]


def set_visible_user_ids_hook(hook: Any) -> None:
    """Set a custom hook that returns user IDs whose models are visible."""
    global _visible_user_ids_hook
    _visible_user_ids_hook = hook


def _get_visible_user_ids(db: Session, user_id: Optional[int] = None) -> list[int]:
    """Return user IDs whose shared models are visible to *user_id*.

    Without a hook this returns all admin user IDs (legacy behaviour).
    When *user_id* is ``None`` (standalone / non-web context), only admin IDs
    are returned so that standalone callers fall back to admin defaults safely.

    Hook implementers MUST handle ``user_id=None`` gracefully — e.g. treat it
    as an unauthenticated context and return only system-admin IDs.
    """
    if _visible_user_ids_hook is not None:
        return list(_visible_user_ids_hook(db, user_id))
    from ..models.user import User

    return [uid for (uid,) in db.query(User.id).filter(User.is_admin).all()]


def build_user_model_visibility_filter(user_id: int, visible_ids: list[int]) -> Any:
    """Return SQLAlchemy filter for UserModel rows visible to *user_id*.

    The filter matches rows owned by *user_id* OR rows that are shared
    and owned by any user in *visible_ids*.
    """
    from sqlalchemy import and_, or_

    from ..models.user import UserModel

    return or_(
        UserModel.user_id == user_id,
        and_(UserModel.user_id.in_(visible_ids), UserModel.is_shared.is_(True)),
    )


def _is_model_visible_to_user(
    db: Session, model_id: Any, user_id: Optional[int] = None
) -> bool:
    """Check whether a DBModel row is visible to *user_id*.

    Returns True when the user has an own ``is_owner`` UserModel **or**
    when a shared UserModel exists from a visible user.  In non-web /
    standalone contexts (*user_id* is None) visibility is always granted.
    """
    if user_id is None:
        return True
    try:
        from ..models.user import UserModel

        # Step 1: user's own (owner) UserModel
        own = (
            db.query(UserModel)
            .filter(
                UserModel.model_id == model_id,
                UserModel.user_id == user_id,
                UserModel.is_owner.is_(True),
            )
            .first()
        )
        if own is not None:
            return True

        # Step 2: shared from visible users
        visible_ids = _get_visible_user_ids(db, user_id)
        shared = (
            db.query(UserModel)
            .filter(
                UserModel.model_id == model_id,
                UserModel.user_id.in_(visible_ids),
                UserModel.is_shared.is_(True),
            )
            .first()
        )
        return shared is not None
    except Exception:
        logger.warning(
            "Visibility check failed for model_id=%s user_id=%s — defaulting to hidden",
            model_id,
            user_id,
        )
        return False


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
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        # This won't work in non-web contexts, so we'll fallback to environment
        try:
            # Try to get a database session (this might fail in CLI contexts)
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                vision_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "visual",
                        DBModel.is_active,
                    )
                    .first()
                )

                if vision_default and vision_default.model:
                    if _is_model_visible_to_user(db, vision_default.model.id, user_id):
                        return _create_llm_instance(vision_default.model)

            # Fallback to visible users' shared defaults
            admin_vision_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "visual",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_vision_defaults:
                return _create_llm_instance(admin_vision_defaults[0].model)

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
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                general_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "general",
                        DBModel.is_active,
                    )
                    .first()
                )

                if general_default and general_default.model:
                    if _is_model_visible_to_user(db, general_default.model.id, user_id):
                        return _create_llm_instance(general_default.model)

            # Fallback to visible users' shared defaults
            admin_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "general",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_defaults:
                return _create_llm_instance(admin_defaults[0].model)

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
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                fast_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "small_fast",
                        DBModel.is_active,
                    )
                    .first()
                )

                if fast_default and fast_default.model:
                    if _is_model_visible_to_user(db, fast_default.model.id, user_id):
                        return _create_llm_instance(fast_default.model)

            # Fallback to visible users' shared defaults
            admin_fast_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "small_fast",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_fast_defaults:
                return _create_llm_instance(admin_fast_defaults[0].model)

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
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                compact_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "compact",
                        DBModel.is_active,
                    )
                    .first()
                )

                if compact_default and compact_default.model:
                    if _is_model_visible_to_user(db, compact_default.model.id, user_id):
                        return _create_llm_instance(compact_default.model)

            # Fallback to visible users' shared defaults
            admin_compact_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "compact",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_compact_defaults:
                return _create_llm_instance(admin_compact_defaults[0].model)

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
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel
        from .llm_utils import _create_llm_instance

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                embedding_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "embedding",
                        DBModel.is_active,
                    )
                    .first()
                )

                if embedding_default and embedding_default.model:
                    if _is_model_visible_to_user(
                        db, embedding_default.model.id, user_id
                    ):
                        return _create_llm_instance(embedding_default.model)

            # Fallback to visible users' shared defaults
            admin_embedding_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "embedding",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_embedding_defaults:
                return _create_llm_instance(admin_embedding_defaults[0].model)

        except Exception as e:
            logger.warning(f"Failed to get embedding model from database: {e}")
            pass

    except ImportError:
        pass

    # No fallback to environment variables - require database configuration
    return None


def get_vision_model(db: Session, user_id: Optional[int] = None) -> Optional[BaseLLM]:
    """
    Get vision model from database filtered by current user visibility.

    Args:
        db: Database session
        user_id: User ID for visibility filtering. If None, all models are visible.

    Returns:
        Vision model instance or None if not found
    """
    try:
        from sqlalchemy import String, cast, or_

        from ..models.model import Model as DBModel
        from .llm_utils import _create_llm_instance

        # Query models that have vision ability in their abilities JSON field
        db_models = (
            db.query(DBModel)
            .filter(
                DBModel.category == "llm",
                DBModel.is_active,
                or_(
                    cast(DBModel.abilities, String).contains('"vision"'),
                    cast(DBModel.abilities, String).like('%"vision"%'),
                ),
            )
            .all()
        )

        for db_model in db_models:
            if user_id is not None and not _is_model_visible_to_user(
                db, db_model.id, user_id
            ):
                continue
            return _create_llm_instance(db_model)
        return None

    except Exception as e:
        logger.error(f"Failed to get vision model from database: {e}")
        return None


def _add_image_model_with_id(
    models_dict: dict[str, Any], instance: Any, db_model: DBModel
) -> None:
    setattr(instance, "model_id", str(db_model.model_id))
    models_dict[str(db_model.model_id)] = instance
    logger.info(
        f"Added image model: model_id={db_model.model_id}, model_name={db_model.model_name}"
    )


def get_image_models(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get image models from database filtered by current user visibility.

    Args:
        db: Database session
        user_id: User ID for visibility filtering. If None, all models are visible.

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
            if user_id is not None and not _is_model_visible_to_user(
                db, db_model.id, user_id
            ):
                continue

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
                    _add_image_model_with_id(image_models, image_model, db_model)
                elif model_provider == "gemini":
                    image_model = GeminiImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate"]),  # pyright: ignore[reportArgumentType]
                    )
                    _add_image_model_with_id(image_models, image_model, db_model)
                elif model_provider == "openai":
                    image_model = OpenAIImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate", "edit"]),  # pyright: ignore[reportArgumentType]
                    )
                    _add_image_model_with_id(image_models, image_model, db_model)
                elif model_provider == "xinference":
                    image_model = XinferenceImageModel(
                        model_name=str(db_model.model_name),
                        api_key=api_key,
                        base_url=base_url,
                        abilities=list(db_model.abilities or ["generate", "edit"]),  # pyright: ignore[reportArgumentType]
                    )
                    _add_image_model_with_id(image_models, image_model, db_model)
            except Exception as e:
                logger.warning(
                    f"Failed to create image model for {db_model.model_name}: {e}"
                )

    except Exception as e:
        logger.error(f"Failed to get image models from database: {e}")

    return image_models


def get_video_models(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """Get video models from database filtered by current user visibility."""
    video_models: dict[str, Any] = {}
    try:
        from ...core.model.video.adapter import get_video_model_instance
        from ..models.model import Model as DBModel

        db_models = (
            db.query(DBModel)
            .filter(
                DBModel.category == "video",
                DBModel.is_active,
            )
            .all()
        )

        for db_model in db_models:
            if user_id is not None and not _is_model_visible_to_user(
                db, db_model.id, user_id
            ):
                continue
            try:
                video_model = get_video_model_instance(db_model)
                setattr(video_model, "model_id", str(db_model.model_id))
                video_models[str(db_model.model_id)] = video_model
                logger.info(
                    "Added video model: model_id=%s, model_name=%s",
                    db_model.model_id,
                    db_model.model_name,
                )
            except Exception as e:
                logger.warning(
                    "Failed to create video model %s: %s",
                    db_model.model_name,
                    e,
                )
    except Exception as e:
        logger.error(f"Failed to load video models: {e}")
        db.rollback()

    return video_models


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
        from sqlalchemy import String, cast

        from ...core.model.image.adapter import get_image_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                image_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "image",
                        DBModel.is_active,
                        cast(DBModel.abilities, String).contains('"generate"'),
                    )
                    .first()
                )

                if image_default and image_default.model:
                    if _is_model_visible_to_user(db, image_default.model.id, user_id):
                        try:
                            instance = get_image_model_instance(image_default.model)
                            setattr(
                                instance, "model_id", str(image_default.model.model_id)
                            )
                            return instance
                        except Exception as e:
                            logger.warning(
                                f"Failed to create image model instance: {e}"
                            )

            # Fallback to visible users' shared defaults
            admin_image_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "image",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                    cast(DBModel.abilities, String).contains('"generate"'),
                )
                .limit(1)
                .all()
            )

            if admin_image_defaults:
                try:
                    instance = get_image_model_instance(admin_image_defaults[0].model)
                    setattr(
                        instance,
                        "model_id",
                        str(admin_image_defaults[0].model.model_id),
                    )
                    return instance
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
        from ..models.user import UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                image_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "image_edit",
                        DBModel.is_active,
                    )
                    .first()
                )

                if image_default and image_default.model:
                    if _is_model_visible_to_user(db, image_default.model.id, user_id):
                        try:
                            instance = get_image_model_instance(image_default.model)
                            setattr(
                                instance, "model_id", str(image_default.model.model_id)
                            )
                            return instance
                        except Exception as e:
                            logger.warning(
                                f"Failed to create image model instance: {e}"
                            )

            # Fallback to visible users' shared defaults
            admin_image_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .filter(
                    UserDefaultModel.config_type == "image_edit",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_image_defaults:
                try:
                    instance = get_image_model_instance(admin_image_defaults[0].model)
                    setattr(
                        instance,
                        "model_id",
                        str(admin_image_defaults[0].model.model_id),
                    )
                    return instance
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


def get_default_video_model(
    user_id: Optional[int] = None,
) -> Optional[BaseVideoModel]:
    """
    Get the default video generation model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default video generation model or None if not available
    """
    try:
        from sqlalchemy import String, cast

        from ...core.model.video.adapter import get_video_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel

        try:
            db = next(get_db())

            if user_id:
                video_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "video",
                        DBModel.category == "video",
                        DBModel.is_active,
                        cast(DBModel.abilities, String).contains('"generate"'),
                    )
                    .first()
                )

                if video_default and video_default.model:
                    if _is_model_visible_to_user(db, video_default.model.id, user_id):
                        try:
                            instance = get_video_model_instance(video_default.model)
                            setattr(
                                instance, "model_id", str(video_default.model.model_id)
                            )
                            return instance
                        except Exception as e:
                            logger.warning(
                                f"Failed to create video model instance: {e}"
                            )

            shared_video_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "video",
                    DBModel.category == "video",
                    DBModel.is_active,
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                    cast(DBModel.abilities, String).contains('"generate"'),
                )
                .limit(1)
                .all()
            )

            if shared_video_defaults:
                try:
                    instance = get_video_model_instance(shared_video_defaults[0].model)
                    setattr(
                        instance,
                        "model_id",
                        str(shared_video_defaults[0].model.model_id),
                    )
                    return instance
                except Exception as e:
                    logger.warning(f"Failed to create video model instance: {e}")

        except Exception as e:
            logger.warning(f"Failed to get default video model from database: {e}")
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
    from ..models.model import Model as DBModel
    from ..models.user import UserDefaultModel, UserModel

    db = next(get_db())

    # If user_id is provided, get user-specific default
    if user_id:
        embedding_default = (
            db.query(UserDefaultModel)
            .join(DBModel, UserDefaultModel.model_id == DBModel.id)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == "embedding",
                DBModel.is_active,
            )
            .first()
        )

        if embedding_default and embedding_default.model:
            if _is_model_visible_to_user(db, embedding_default.model.id, user_id):
                return str(embedding_default.model.model_id)

    # Visible users' shared defaults
    admin_embedding_defaults = (
        db.query(UserDefaultModel)
        .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
        .filter(
            UserDefaultModel.config_type == "embedding",
            UserModel.is_shared.is_(True),
            UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
        )
        .limit(1)
        .all()
    )

    if admin_embedding_defaults:
        return str(admin_embedding_defaults[0].model.model_id)

    return None


def get_default_rerank_model(user_id: Optional[int] = None) -> Optional[str]:
    """Get the default rerank model ID for a specific user.

    Mirrors :func:`get_default_embedding_model` for rerank models. Looks up
    the user's configured default rerank model from ``UserDefaultModel``
    and returns the underlying ``Model.model_id`` (string).
    """
    from ..models.database import get_db
    from ..models.model import Model as DBModel
    from ..models.user import UserDefaultModel, UserModel

    db = next(get_db())
    try:
        if user_id:
            rerank_default = (
                db.query(UserDefaultModel)
                .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.user_id == user_id,
                    UserDefaultModel.config_type == "rerank",
                    DBModel.is_active,
                )
                .first()
            )
            if rerank_default and rerank_default.model:
                if _is_model_visible_to_user(db, rerank_default.model.id, user_id):
                    return str(rerank_default.model.model_id)

        admin_rerank_defaults = (
            db.query(UserDefaultModel)
            .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
            .filter(
                UserDefaultModel.config_type == "rerank",
                UserModel.is_shared.is_(True),
                UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
            )
            .limit(1)
            .all()
        )
        if admin_rerank_defaults:
            return str(admin_rerank_defaults[0].model.model_id)
    except Exception:
        logger.exception("get_default_rerank_model failed")
    finally:
        db.close()
    return None


def _get_models_by_category(
    db: Session, ability: str, model_type: str, user_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Get models by category and ability from database filtered by visibility.

    Generic helper function to load models (ASR, TTS, etc.) from database.

    Args:
        db: Database session
        ability: Model ability to filter by (e.g., "asr", "tts")
        model_type: Model type for error messages (e.g., "ASR", "TTS")
        user_id: User ID for visibility filtering. If None, all models are visible.

    Returns:
        Dictionary of model instances
    """
    models: dict[str, Any] = {}
    try:
        from ..models.model import Model as DBModel

        db_models = (
            db.query(DBModel)
            .filter(
                DBModel.category == "speech",
                DBModel.is_active,
            )
            .all()
        )

        for db_model in db_models:
            if user_id is not None and not _is_model_visible_to_user(
                db, db_model.id, user_id
            ):
                continue
            abilities: Any = getattr(db_model, "abilities", None)
            if isinstance(abilities, str):
                try:
                    abilities = json.loads(abilities)
                except (TypeError, json.JSONDecodeError):
                    abilities = []
            if (
                not isinstance(abilities, (list, tuple, set))
                or ability not in abilities
            ):
                continue

            # api_key may be empty for self-hosted Xinference (no auth);
            # the adapter handles that gracefully so we don't pre-validate
            # it here. Skip the model only when base_url is missing —
            # without that there's no way to reach the service. Raising
            # here would abort loading ALL models of this category in one
            # go (the outer try/except returns {}), so any per-model issue
            # is downgraded to a per-model warning + skip.
            base_url = cast(Optional[str], getattr(db_model, "base_url", None))
            if not base_url:
                logger.warning(
                    f"Skipping {model_type} model {db_model.model_name}: "
                    "base_url is empty"
                )
                continue

            model_provider = str(db_model.model_provider).strip().lower()
            try:
                model: Any = None
                if model_provider == "xinference":
                    # Import appropriate adapter based on model type
                    if ability == "asr":
                        from ...core.model.asr.adapter import get_asr_model_instance

                        model = get_asr_model_instance(db_model)
                    elif ability == "tts":
                        from ...core.model.tts.adapter import get_tts_model_instance

                        model = get_tts_model_instance(db_model)
                    else:
                        raise ValueError(f"Unsupported model ability: {ability}")

                    models[str(db_model.model_name)] = model
                    logger.info(f"Added {model_type} model: {db_model.model_name}")
                else:
                    logger.warning(
                        f"Unsupported {model_type} model provider: {model_provider}"
                    )
            except Exception as e:
                logger.warning(
                    f"Failed to create {model_type} model {db_model.model_name}: {e}"
                )

    except Exception as e:
        logger.error(f"Failed to load {model_type} models: {e}")
        db.rollback()

    return models


def get_asr_models(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get ASR (speech-to-text) models from database filtered by visibility.

    Args:
        db: Database session
        user_id: User ID for visibility filtering. If None, all models are visible.

    Returns:
        Dictionary of ASR model instances
    """
    return _get_models_by_category(db, "asr", "ASR", user_id=user_id)


def get_tts_models(db: Session, user_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Get TTS (text-to-speech) models from database filtered by visibility.

    Args:
        db: Database session
        user_id: User ID for visibility filtering. If None, all models are visible.

    Returns:
        Dictionary of TTS model instances
    """
    return _get_models_by_category(db, "tts", "TTS", user_id=user_id)


def get_default_asr_model(user_id: Optional[int] = None) -> Optional[Any]:
    """
    Get the default ASR model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default ASR model or None if not available
    """
    try:
        from ...core.model.asr.adapter import get_asr_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                asr_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "asr",
                        DBModel.is_active,
                    )
                    .first()
                )

                if asr_default and asr_default.model:
                    if _is_model_visible_to_user(db, asr_default.model.id, user_id):
                        try:
                            return get_asr_model_instance(asr_default.model)
                        except Exception as e:
                            logger.warning(f"Failed to create ASR model instance: {e}")

            # Visible users' shared defaults
            admin_asr_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "asr",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_asr_defaults:
                try:
                    return get_asr_model_instance(admin_asr_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create ASR model instance: {e}")

        except Exception as e:
            logger.warning(f"Database query failed for ASR model: {e}")

    except Exception as e:
        logger.error(f"Failed to get default ASR model: {e}")

    return None


def get_default_tts_model(user_id: Optional[int] = None) -> Optional[Any]:
    """
    Get the default TTS model for a specific user.

    Args:
        user_id: User ID for multi-tenant model resolution. If None, uses admin defaults.

    Returns:
        The default TTS model or None if not available
    """
    try:
        from ...core.model.tts.adapter import get_tts_model_instance
        from ..models.database import get_db
        from ..models.model import Model as DBModel
        from ..models.user import UserDefaultModel, UserModel

        try:
            db = next(get_db())

            # If user_id is provided, get user-specific default
            if user_id:
                tts_default = (
                    db.query(UserDefaultModel)
                    .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                    .filter(
                        UserDefaultModel.user_id == user_id,
                        UserDefaultModel.config_type == "tts",
                        DBModel.is_active,
                    )
                    .first()
                )

                if tts_default and tts_default.model:
                    if _is_model_visible_to_user(db, tts_default.model.id, user_id):
                        try:
                            return get_tts_model_instance(tts_default.model)
                        except Exception as e:
                            logger.warning(f"Failed to create TTS model instance: {e}")

            # Visible users' shared defaults
            admin_tts_defaults = (
                db.query(UserDefaultModel)
                .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
                .join(DBModel, UserModel.model_id == DBModel.id)
                .filter(
                    UserDefaultModel.config_type == "tts",
                    UserModel.is_shared.is_(True),
                    UserDefaultModel.user_id.in_(_get_visible_user_ids(db, user_id)),
                )
                .limit(1)
                .all()
            )

            if admin_tts_defaults:
                try:
                    return get_tts_model_instance(admin_tts_defaults[0].model)
                except Exception as e:
                    logger.warning(f"Failed to create TTS model instance: {e}")

        except Exception as e:
            logger.warning(f"Database query failed for TTS model: {e}")

    except Exception as e:
        logger.error(f"Failed to get default TTS model: {e}")

    return None
