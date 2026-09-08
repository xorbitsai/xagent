"""Persistence and validation for the per-user virtual Auto model."""

from __future__ import annotations

from typing import Any, Iterable

from sqlalchemy.orm import Session, joinedload

from ...core.model.providers import (
    AUTO_MODEL_NAME,
    ROUTER_PROVIDER,
    is_auto_router_model,
)
from ..models.auto_model import AutoModelCandidate, AutoModelConfig
from ..models.model import Model as DBModel
from ..models.user import UserDefaultModel, UserModel
from ..schemas.model import AutoModelConfigUpdate, ModelWithAccessInfo
from .hot_path_cache import invalidate_model_cache
from .model_store import ModelStore

AUTO_ROUTER_CONFIG_NAME = "auto"
AUTO_ROUTER_MODEL_ID_PREFIX = "auto-router-"


def is_reserved_auto_router_model_id(model_id: object) -> bool:
    """Return whether an ID belongs to the service-owned Auto namespace."""

    return isinstance(model_id, str) and model_id.strip().startswith(
        AUTO_ROUTER_MODEL_ID_PREFIX
    )


class AutoModelConfigurationError(ValueError):
    """Raised when an Auto model configuration is invalid."""


class AutoModelDependencyError(RuntimeError):
    """Raised when the installed xrouter package cannot provide profiles."""


def load_router_profile_catalog() -> Any:
    """Load xrouter's bundled profile catalog without importing it at startup."""

    try:
        from xrouter_llm import default_models_dir, load_benchmark_profiles
    except ImportError as exc:
        raise AutoModelDependencyError(
            "Auto model configuration needs xrouter-llm>=0.3.3"
        ) from exc
    return load_benchmark_profiles(default_models_dir())


def list_router_profiles() -> list[dict[str, Any]]:
    catalog = load_router_profile_catalog()
    return [
        {
            "id": profile.model_id,
            "provider": profile.provider,
            "aliases": list(profile.aliases),
            "input_modalities": list(profile.input_modalities),
            "context_window": profile.context_length,
        }
        for profile in sorted(catalog.profiles(), key=lambda item: item.model_id)
    ]


def validate_candidate_modalities(
    catalog: Any, profile_id: str, abilities: Iterable[str]
) -> None:
    """A profile must describe the saved endpoint's declared input capabilities."""
    if profile_id not in catalog.known_model_ids():
        raise AutoModelConfigurationError(f"Unknown Auto profile {profile_id!r}")
    profile = catalog.get(profile_id)
    modality_abilities = {"image": "vision", "audio": "audio", "video": "video"}
    profile_modalities = set(profile.input_modalities) & modality_abilities.keys()
    target_modalities = {
        modality
        for modality, ability in modality_abilities.items()
        if ability in abilities
    }
    if profile_modalities != target_modalities:
        raise AutoModelConfigurationError(
            f"Profile {profile_id!r} input modalities do not match the candidate model's abilities. "
            "Choose a matching profile or correct the model's abilities."
        )


class AutoModelService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_config(self, user_id: int) -> AutoModelConfig | None:
        return (
            self.db.query(AutoModelConfig)
            .options(
                joinedload(AutoModelConfig.router_model),
                joinedload(AutoModelConfig.candidates).joinedload(
                    AutoModelCandidate.target_model
                ),
            )
            .filter(AutoModelConfig.user_id == user_id)
            .first()
        )

    def upsert_config(
        self, *, user_id: int, request: AutoModelConfigUpdate
    ) -> AutoModelConfig:
        catalog = load_router_profile_catalog()
        known_profile_ids = set(catalog.known_model_ids())
        requested_profile_ids = {
            candidate.routing_model_id for candidate in request.candidates
        }
        unknown_profile_ids = sorted(requested_profile_ids - known_profile_ids)
        if unknown_profile_ids:
            raise AutoModelConfigurationError(
                "Unknown xrouter model profile(s): " + ", ".join(unknown_profile_ids)
            )

        target_ids = {candidate.target_model_id for candidate in request.candidates}
        targets = self._accessible_target_models(user_id=user_id, model_ids=target_ids)
        missing_target_ids = sorted(target_ids - set(targets))
        if missing_target_ids:
            raise AutoModelConfigurationError(
                "Candidate model(s) are inactive, missing, or inaccessible: "
                + ", ".join(str(model_id) for model_id in missing_target_ids)
            )

        for candidate in request.candidates:
            validate_candidate_modalities(
                catalog,
                candidate.routing_model_id,
                targets[candidate.target_model_id].abilities or [],
            )

        config = (
            self.db.query(AutoModelConfig)
            .filter(AutoModelConfig.user_id == user_id)
            .first()
        )
        if config is None:
            router_model = self._create_router_model(user_id, targets.values())
            self.db.flush()
            config = AutoModelConfig(
                user_id=user_id,
                router_model_id=router_model.id,
                strategy="balanced",
            )
            self.db.add(config)
            self.db.flush()
        else:
            router_model = config.router_model
            self._update_router_model_abilities(router_model, targets.values())
            # Normalize legacy quality/cost records. Strategy is intentionally no
            # longer user-selectable until xrouter exposes distinct single-model
            # routing policies that work with an explicit candidate set.
            config.strategy = "balanced"  # type: ignore[assignment]
            self.db.query(AutoModelCandidate).filter(
                AutoModelCandidate.config_id == config.id
            ).delete(synchronize_session=False)
            self.db.flush()

        config.fallback_model_id = request.fallback_model_id  # type: ignore[assignment]
        for candidate in request.candidates:
            self.db.add(
                AutoModelCandidate(
                    config_id=config.id,
                    routing_model_id=candidate.routing_model_id,
                    target_model_id=candidate.target_model_id,
                )
            )

        general_default = (
            self.db.query(UserDefaultModel)
            .filter(
                UserDefaultModel.user_id == user_id,
                UserDefaultModel.config_type == "general",
            )
            .first()
        )
        if request.set_as_default is True:
            if general_default is None:
                self.db.add(
                    UserDefaultModel(
                        user_id=user_id,
                        model_id=config.router_model_id,
                        config_type="general",
                    )
                )
            else:
                general_default.model_id = config.router_model_id
        elif (
            request.set_as_default is False
            and general_default is not None
            and int(general_default.model_id) == int(config.router_model_id)
        ):
            self.db.delete(general_default)

        self.db.commit()
        invalidate_model_cache(user_id)
        result = self.get_config(user_id)
        if result is None:  # pragma: no cover - defensive after committed insert
            raise RuntimeError("Auto model configuration disappeared after save")
        return result

    def serialize_config(
        self, config: AutoModelConfig | None, *, user_id: int
    ) -> dict[str, Any]:
        if config is None:
            return {
                "configured": False,
                "strategy": "balanced",
                "fallback_model_id": None,
                "auto_model": None,
                "candidates": [],
            }

        access_by_model_id = self._accessible_user_models(
            user_id=user_id,
            model_ids={
                int(config.router_model_id),
                *(int(candidate.target_model_id) for candidate in config.candidates),
            },
        )
        router_access = access_by_model_id.get(int(config.router_model_id))
        if router_access is None:
            raise AutoModelConfigurationError(
                "Auto model is not accessible to its owner"
            )

        store = ModelStore(self.db)
        auto_model = ModelWithAccessInfo.model_validate(
            store.serialize_model_with_access(
                config.router_model,
                router_access,
                requesting_user_id=user_id,
            )
        )
        candidates = []
        for candidate in config.candidates:
            target_access = access_by_model_id.get(int(candidate.target_model_id))
            if target_access is None:
                continue
            candidates.append(
                {
                    "routing_model_id": candidate.routing_model_id,
                    "target_model_id": candidate.target_model_id,
                    "target_model": store.serialize_model_with_access(
                        candidate.target_model,
                        target_access,
                        requesting_user_id=user_id,
                    ),
                }
            )
        return {
            "configured": True,
            "strategy": "balanced",
            "fallback_model_id": config.fallback_model_id,
            "auto_model": auto_model,
            "candidates": candidates,
        }

    def _accessible_target_models(
        self, *, user_id: int, model_ids: set[int]
    ) -> dict[int, DBModel]:
        access = self._accessible_user_models(user_id=user_id, model_ids=model_ids)
        result: dict[int, DBModel] = {}
        for model_id, user_model in access.items():
            model = user_model.model
            if (
                model.category == "llm"
                and model.is_active
                and not is_auto_router_model(model.model_provider, model.model_name)
            ):
                result[model_id] = model
        return result

    def _accessible_user_models(
        self, *, user_id: int, model_ids: set[int]
    ) -> dict[int, UserModel]:
        from .model_service import (
            _get_visible_user_ids,
            build_user_model_visibility_filter,
        )

        if not model_ids:
            return {}
        visible_ids = _get_visible_user_ids(self.db, user_id)
        rows = (
            self.db.query(UserModel)
            .options(joinedload(UserModel.model))
            .filter(
                UserModel.model_id.in_(model_ids),
                build_user_model_visibility_filter(user_id, visible_ids),
            )
            .order_by(
                UserModel.model_id,
                (UserModel.user_id == user_id).desc(),
                UserModel.is_owner.desc(),
            )
            .all()
        )
        result: dict[int, UserModel] = {}
        for row in rows:
            result.setdefault(int(row.model_id), row)
        return result

    def _create_router_model(self, user_id: int, targets: Iterable[DBModel]) -> DBModel:
        model_id = f"{AUTO_ROUTER_MODEL_ID_PREFIX}{user_id}"
        if self.db.query(DBModel.id).filter(DBModel.model_id == model_id).first():
            raise AutoModelConfigurationError(
                f"Reserved Auto model ID {model_id!r} is already in use"
            )
        router_model = DBModel(
            model_id=model_id,
            category="llm",
            model_provider=ROUTER_PROVIDER,
            model_name=AUTO_MODEL_NAME,
            description="Routes each request to one of your configured models.",
            is_active=True,
        )
        router_model.api_key = ""  # type: ignore[assignment]
        self._update_router_model_abilities(router_model, targets)
        self.db.add(router_model)
        self.db.flush()
        self.db.add(
            UserModel(
                user_id=user_id,
                model_id=router_model.id,
                is_owner=True,
                can_edit=False,
                can_delete=False,
                is_shared=False,
            )
        )
        return router_model

    @staticmethod
    def _update_router_model_abilities(
        router_model: DBModel, targets: Iterable[DBModel]
    ) -> None:
        ability_sets = [set(target.abilities or []) for target in targets]
        abilities = {"chat"}
        if ability_sets:
            abilities.update(set.intersection(*ability_sets))
        router_model.abilities = sorted(abilities)  # type: ignore[assignment]
