"""Model DB/cache boundary for web model management paths."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session, joinedload

from ..models.model import Model as DBModel
from ..models.user import User, UserDefaultModel, UserModel
from ..schemas.model import ModelWithAccessInfo, UserDefaultModelResponse
from .hot_path_cache import (
    cache_get,
    cache_set,
    invalidate_model_cache,
    model_list_key,
    user_default_model_key,
    user_default_models_key,
)
from .llm_utils import CoreStorage

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CONFIG_TYPES = [
    "general",
    "small_fast",
    "visual",
    "compact",
    "embedding",
    "image",
    "image_edit",
    "asr",
    "tts",
    "rerank",
]


class ModelSharingConflictError(ValueError):
    """Raised when a model sharing state transition violates model constraints."""


class ModelStore:
    """Owns model reads/writes that participate in hot-path cache policy."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def serialize_model_with_access(
        self,
        db_model: DBModel,
        user_model: UserModel,
        *,
        requesting_user_id: int | None = None,
    ) -> dict[str, Any]:
        is_owner = (
            user_model.is_owner and user_model.user_id == requesting_user_id
            if requesting_user_id is not None
            else user_model.is_owner
        )

        return {
            "id": db_model.id,
            "model_id": db_model.model_id,
            "category": db_model.category,
            "model_provider": db_model.model_provider,
            "model_name": db_model.model_name,
            "base_url": db_model.base_url,
            "temperature": db_model.temperature,
            "context_window": db_model.context_window,
            "dimension": db_model.dimension,
            "abilities": db_model.abilities,
            "description": db_model.description,
            "created_at": db_model.created_at.isoformat()
            if db_model.created_at
            else None,
            "updated_at": db_model.updated_at.isoformat()
            if db_model.updated_at
            else None,
            "is_active": db_model.is_active,
            "is_owner": is_owner,
            "can_edit": is_owner and user_model.can_edit,
            "can_delete": is_owner and user_model.can_delete,
            "is_shared": user_model.is_shared,
        }

    def model_has_shared_visibility(self, model_id: int) -> bool:
        return (
            self.db.query(UserModel.id)
            .filter(UserModel.model_id == model_id, UserModel.is_shared.is_(True))
            .first()
            is not None
        )

    def list_models(
        self,
        *,
        user_id: int,
        skip: int,
        limit: int,
        model_provider: str | None,
        category: str | None,
        is_active: bool | None,
    ) -> list[ModelWithAccessInfo]:
        cache_key = model_list_key(
            user_id,
            skip=skip,
            limit=limit,
            model_provider=model_provider,
            category=category,
            is_active=is_active,
        )
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return [ModelWithAccessInfo.model_validate(item) for item in cached]

        from .model_service import (
            _get_visible_user_ids,
            build_user_model_visibility_filter,
        )

        visible_ids = _get_visible_user_ids(self.db, user_id)
        best_user_model_id = (
            self.db.query(UserModel.id)
            .filter(
                UserModel.model_id == DBModel.id,
                build_user_model_visibility_filter(user_id, visible_ids),
            )
            .order_by(
                (UserModel.user_id == user_id).desc(),
                UserModel.is_owner.desc(),
            )
            .limit(1)
            .correlate(DBModel)
            .scalar_subquery()
        )

        query = (
            self.db.query(DBModel, UserModel)
            .join(UserModel, DBModel.id == UserModel.model_id)
            .filter(UserModel.id == best_user_model_id)
        )
        if model_provider:
            query = query.filter(DBModel.model_provider == model_provider)
        if category:
            query = query.filter(DBModel.category == category)
        if is_active is not None:
            query = query.filter(DBModel.is_active == is_active)

        result = [
            ModelWithAccessInfo.model_validate(
                self.serialize_model_with_access(
                    db_model,
                    user_model,
                    requesting_user_id=user_id,
                )
            )
            for db_model, user_model in query.offset(skip).limit(limit).all()
        ]
        cache_set(cache_key, [item.model_dump(mode="json") for item in result])
        return result

    def get_user_default_models(self, user: User) -> list[dict[str, Any]]:
        user_id = int(user.id)
        cache_key = user_default_models_key(user_id)
        cached = cache_get(cache_key)
        if isinstance(cached, list):
            return cached

        user_defaults = (
            self.db.query(UserDefaultModel)
            .options(joinedload(UserDefaultModel.model))
            .join(DBModel, UserDefaultModel.model_id == DBModel.id)
            .filter(UserDefaultModel.user_id == user.id, DBModel.is_active)
            .all()
        )
        user_defaults_by_type: dict[str, UserDefaultModel] = {}
        for user_default in user_defaults:
            user_defaults_by_type[str(user_default.config_type)] = user_default

        result: list[dict[str, Any]] = []

        from .model_service import (
            _get_visible_user_ids,
            build_user_model_visibility_filter,
        )

        visible_ids = _get_visible_user_ids(self.db, user_id)

        default_model_ids = [int(default.model_id) for default in user_defaults]
        visible_user_models_by_model_id: dict[int, UserModel] = {}
        if default_model_ids:
            visible_user_models = (
                self.db.query(UserModel)
                .options(joinedload(UserModel.model))
                .filter(
                    UserModel.model_id.in_(default_model_ids),
                    build_user_model_visibility_filter(user_id, visible_ids),
                )
                .order_by(
                    UserModel.model_id,
                    (UserModel.user_id == user_id).desc(),
                    UserModel.is_owner.desc(),
                )
                .all()
            )
            for visible_user_model in visible_user_models:
                visible_user_models_by_model_id.setdefault(
                    int(visible_user_model.model_id), visible_user_model
                )

        fallback_defaults_by_type: dict[str, UserDefaultModel] = {}
        fallback_defaults = (
            self.db.query(UserDefaultModel)
            .options(joinedload(UserDefaultModel.model))
            .join(DBModel, UserDefaultModel.model_id == DBModel.id)
            .join(UserModel, UserDefaultModel.model_id == UserModel.model_id)
            .filter(
                UserDefaultModel.user_id.in_(visible_ids),
                UserDefaultModel.config_type.in_(DEFAULT_MODEL_CONFIG_TYPES),
                DBModel.is_active,
                UserModel.is_shared.is_(True),
            )
            .order_by(
                UserDefaultModel.config_type,
                (UserDefaultModel.user_id == user_id).desc(),
            )
            .all()
        )
        for fallback_candidate in fallback_defaults:
            fallback_defaults_by_type.setdefault(
                str(fallback_candidate.config_type), fallback_candidate
            )

        for config_type in DEFAULT_MODEL_CONFIG_TYPES:
            user_model: UserModel | None = None
            if config_type in user_defaults_by_type:
                user_default = user_defaults_by_type[config_type]
                user_model = visible_user_models_by_model_id.get(
                    int(user_default.model_id)
                )
                if user_model:
                    result.append(
                        self._default_model_payload(
                            user_default,
                            self.serialize_model_with_access(
                                user_model.model,
                                user_model,
                                requesting_user_id=user_id,
                            ),
                        )
                    )
                    continue

            if not user_model:
                fallback_default_for_type = fallback_defaults_by_type.get(config_type)
                if fallback_default_for_type:
                    logger.info(
                        "User %s has no %s default, using visible user default for display",
                        user.username,
                        config_type,
                    )
                    result.append(
                        self._default_model_payload(
                            fallback_default_for_type,
                            self._shared_fallback_model_payload(
                                fallback_default_for_type.model
                            ),
                        )
                    )

        cache_set(cache_key, result)
        return result

    def get_user_default_model(
        self, user_id: int, config_type: str
    ) -> UserDefaultModelResponse | None:
        cache_key = user_default_model_key(user_id, config_type)
        cached = cache_get(cache_key)
        if isinstance(cached, dict):
            return UserDefaultModelResponse.model_validate(cached)

        user_default = (
            self.db.query(UserDefaultModel)
            .join(DBModel, UserDefaultModel.model_id == DBModel.id)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == config_type,
                DBModel.is_active,
            )
            .first()
        )
        if not user_default:
            return None

        response = UserDefaultModelResponse.model_validate(user_default)
        cache_set(cache_key, response.model_dump(mode="json"))
        return response

    def create_user_model_link(
        self, *, user_id: int, model_id: int, is_shared: bool
    ) -> UserModel:
        user_model = UserModel(
            user_id=user_id,
            model_id=model_id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_shared=is_shared,
        )
        self.db.add(user_model)
        self.db.commit()
        invalidate_model_cache(None if is_shared else user_id)
        return user_model

    def set_user_default_model(
        self,
        *,
        user_id: int,
        model_id: int,
        config_type: str,
        user_model: UserModel,
    ) -> UserDefaultModel:
        # Use a single atomic upsert to avoid UNIQUE-constraint races
        # when the client (e.g. React strict mode, double click) fires
        # two set_default requests for the same (user_id, config_type)
        # before either has committed. delete-then-insert is racy
        # under SQLite because there is no row-level locking.
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert

        bind = self.db.get_bind()
        dialect = bind.dialect.name if bind is not None else ""
        stmt: Any = None
        if dialect == "sqlite":
            stmt = sqlite_insert(UserDefaultModel).values(
                user_id=user_id, model_id=model_id, config_type=config_type
            )
        elif dialect == "postgresql":
            stmt = pg_insert(UserDefaultModel).values(
                user_id=user_id, model_id=model_id, config_type=config_type
            )
        if stmt is not None:
            stmt = stmt.on_conflict_do_update(
                index_elements=["user_id", "config_type"],
                set_={"model_id": model_id},
            )
            self.db.execute(stmt)
            self.db.commit()
        else:
            # Fallback: delete + insert (kept for any other dialect)
            existing_default = (
                self.db.query(UserDefaultModel)
                .filter(
                    UserDefaultModel.user_id == user_id,
                    UserDefaultModel.config_type == config_type,
                )
                .first()
            )
            if existing_default:
                self.db.delete(existing_default)
                self.db.flush()
            self.db.add(
                UserDefaultModel(
                    user_id=user_id, model_id=model_id, config_type=config_type
                )
            )
            self.db.commit()

        user_default = (
            self.db.query(UserDefaultModel)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == config_type,
            )
            .first()
        )
        new_default_is_shared = bool(user_model.is_shared)
        invalidate_model_cache(None if new_default_is_shared else user_id)
        return user_default  # type: ignore[return-value]  # Always exists after commit

    def delete_user_default_model(
        self, *, user_id: int, config_type: str
    ) -> UserDefaultModel | None:
        user_default = (
            self.db.query(UserDefaultModel)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == config_type,
            )
            .first()
        )
        if user_default is None:
            return None

        default_was_shared = self.model_has_shared_visibility(
            int(user_default.model_id)
        )
        self.db.delete(user_default)
        self.db.commit()
        invalidate_model_cache(None if default_was_shared else user_id)
        return user_default

    def commit_model_update(
        self, *, user_id: int, db_model: DBModel, invalidate_globally: bool
    ) -> None:
        self.db.commit()
        self.db.refresh(db_model)
        invalidate_model_cache(None if invalidate_globally else user_id)

    def set_model_sharing(
        self,
        *,
        user_id: int,
        db_model: DBModel,
        user_model: UserModel,
        share_with_users: bool,
    ) -> None:
        model_id = int(db_model.id)
        currently_shared = self.model_has_shared_visibility(model_id)

        if share_with_users:
            user_model.is_shared = True  # type: ignore[assignment]
        elif currently_shared:
            owner_defaults = (
                self.db.query(UserDefaultModel)
                .filter(
                    UserDefaultModel.model_id == model_id,
                    UserDefaultModel.user_id == user_id,
                )
                .count()
            )
            if owner_defaults > 0:
                raise ModelSharingConflictError(
                    "Cannot un-share: you have this model as your default. "
                    "Change default first."
                )

            user_model.is_shared = False  # type: ignore[assignment]
            self.db.query(UserModel).filter(
                UserModel.model_id == model_id, UserModel.is_owner.is_(False)
            ).delete()
            self.db.query(UserDefaultModel).filter(
                UserDefaultModel.model_id == model_id,
                UserDefaultModel.user_id != user_id,
            ).delete()

        self.db.commit()
        self.db.refresh(db_model)
        self.db.refresh(user_model)
        invalidate_model_cache(None)

    def delete_model(
        self, *, model_storage: CoreStorage, user_model: UserModel
    ) -> None:
        model_id = int(user_model.model.id)
        self.db.query(UserModel).filter(UserModel.model_id == model_id).delete()
        self.db.query(UserDefaultModel).filter(
            UserDefaultModel.model_id == model_id
        ).delete()
        model_storage.delete(user_model.model.model_id)
        invalidate_model_cache(None)

    def invalidate_after_user_delete(self) -> None:
        invalidate_model_cache(None)

    def _default_model_payload(
        self, user_default: UserDefaultModel, model_payload: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            "id": user_default.id,
            "user_id": user_default.user_id,
            "model_id": user_default.model_id,
            "config_type": user_default.config_type,
            "created_at": user_default.created_at.isoformat()
            if user_default.created_at
            else None,
            "updated_at": user_default.updated_at.isoformat()
            if user_default.updated_at
            else None,
            "model": model_payload,
        }

    def _shared_fallback_model_payload(self, db_model: DBModel) -> dict[str, Any]:
        return {
            "id": db_model.id,
            "model_id": db_model.model_id,
            "category": db_model.category,
            "model_provider": db_model.model_provider,
            "model_name": db_model.model_name,
            "base_url": db_model.base_url,
            "temperature": db_model.temperature,
            "context_window": db_model.context_window,
            "dimension": db_model.dimension,
            "abilities": db_model.abilities,
            "description": db_model.description,
            "created_at": db_model.created_at.isoformat()
            if db_model.created_at
            else None,
            "updated_at": db_model.updated_at.isoformat()
            if db_model.updated_at
            else None,
            "is_active": db_model.is_active,
            "is_owner": False,
            "can_edit": False,
            "can_delete": False,
            "is_shared": True,
        }
