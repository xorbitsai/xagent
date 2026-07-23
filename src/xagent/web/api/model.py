"""Model management API route handlers"""

import asyncio
import logging
import time
import urllib.parse
from inspect import isawaitable
from typing import Any, List, Optional, cast

from fastapi import (
    APIRouter,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from sqlalchemy.orm import Session

from xagent.core.model.chat.basic.deepseek import DEEPSEEK_SUPPORTED_MODELS
from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ImageModelConfig,
    ModelConfig,
    MusicModelConfig,
    RerankModelConfig,
    SoundEffectModelConfig,
    VideoModelConfig,
)
from xagent.core.model.providers import (
    canonical_provider_name,
    default_base_url_for_provider,
    is_auto_router_model,
    provider_requires_base_url,
)
from xagent.core.utils.security import redact_sensitive_text

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.model import Model as DBModel
from ..models.user import User, UserDefaultModel, UserModel
from ..schemas.model import (
    ModelConnectionTestRequest,
    ModelCreate,
    ModelTestRequest,
    ModelTestResponse,
    ModelUpdate,
    ModelWithAccessInfo,
    UserDefaultModelCreate,
    UserDefaultModelResponse,
)
from ..services.llm_utils import CoreStorage
from ..services.model_store import ModelSharingConflictError, ModelStore
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hook infrastructure for dynamic model sharing
# ---------------------------------------------------------------------------

_can_share_hook = None  # (user: User) -> bool


def set_can_share_hook(hook: Any) -> None:
    """Set a custom hook that determines whether a user can share models."""
    global _can_share_hook
    _can_share_hook = hook


def _can_user_share(user: User) -> bool:
    """Check whether *user* is allowed to share models.

    Without a hook this falls back to ``user.is_admin`` (legacy behaviour).
    """
    if _can_share_hook is not None:
        return bool(_can_share_hook(user))
    return bool(user.is_admin)


model_router = APIRouter(prefix="/api/models", tags=["models"])

MAX_TRANSCRIBE_UPLOAD_BYTES = 25 * 1024 * 1024


def _decode_model_identifier(model_id: str) -> str:
    """Decode a model identifier from the URL path."""

    return urllib.parse.unquote(model_id)


def _resolve_accessible_model(
    db: Session, user: User, model_id: str
) -> tuple[CoreStorage, DBModel, UserModel]:
    """Resolve a model and the current user's access relationship.

    Two-step lookup:
    1. Check if the user owns the model (UserModel.user_id == user.id).
    2. If not, check if any visible user shares this model.
    """

    from ..services.model_service import _get_visible_user_ids

    decoded_model_id = _decode_model_identifier(model_id)
    model_storage = CoreStorage(db, DBModel)
    db_model = model_storage.get_db_model(decoded_model_id)
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Step 1: own UserModel
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == db_model.id,
            UserModel.user_id == user.id,
            UserModel.is_owner.is_(True),
        )
        .first()
    )
    if user_model:
        return model_storage, db_model, user_model

    # Step 2: shared from visible users
    visible_ids = _get_visible_user_ids(db, int(user.id))
    shared = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == db_model.id,
            UserModel.user_id.in_(visible_ids),
            UserModel.is_shared.is_(True),
        )
        .first()
    )
    if shared:
        return model_storage, db_model, shared

    raise HTTPException(status_code=404, detail="Model not found or access denied")


def _normalize_provider_model_id(model_id: str) -> str:
    """Normalize provider model IDs for matching."""

    normalized = model_id.strip()
    if normalized.startswith("models/"):
        return normalized[7:]
    return normalized


def _find_provider_model(
    models: List[dict[str, Any]], target_model_name: str
) -> Optional[dict[str, Any]]:
    """Find a provider model entry by ID after normalizing provider-specific prefixes."""

    normalized_target = _normalize_provider_model_id(target_model_name)
    for model in models:
        model_id = str(model.get("id", "")).strip()
        if _normalize_provider_model_id(model_id) == normalized_target:
            return model
    return None


def _validate_requested_abilities(
    requested_abilities: Optional[List[str]], provider_model: Optional[dict[str, Any]]
) -> None:
    """Validate that the fetched provider model supports the requested abilities."""

    if not requested_abilities or not provider_model:
        return

    available_abilities = provider_model.get("abilities") or provider_model.get(
        "model_ability"
    )
    if not available_abilities:
        return

    available = {str(ability) for ability in available_abilities}
    missing = sorted(set(requested_abilities) - available)
    if missing:
        raise ValueError(
            f"Model '{provider_model.get('id', '')}' does not support abilities: {', '.join(missing)}"
        )


def _model_has_ability(model: Any, ability: str) -> bool:
    abilities = getattr(model, "abilities", None) or []
    if isinstance(abilities, str):
        return ability in abilities
    if not isinstance(abilities, (list, tuple, set)):
        return False
    return ability in {str(item) for item in abilities}


def _iter_accessible_asr_models(db: Session, user: User) -> list[DBModel]:
    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    user_id = int(user.id)
    visible_ids = _get_visible_user_ids(db, user_id)
    rows = (
        db.query(DBModel, UserModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(
            DBModel.category == "speech",
            DBModel.is_active,
            build_user_model_visibility_filter(user_id, visible_ids),
        )
        .order_by(
            (UserModel.user_id == user_id).desc(),
            UserModel.is_owner.desc(),
            DBModel.id.asc(),
        )
        .all()
    )

    models: list[DBModel] = []
    seen_model_ids: set[int] = set()
    for db_model, _user_model in rows:
        model_pk = int(db_model.id)
        if model_pk in seen_model_ids or not _model_has_ability(db_model, "asr"):
            continue
        seen_model_ids.add(model_pk)
        models.append(db_model)
    return models


def _resolve_asr_model_for_transcription(
    db: Session,
    user: User,
    requested_model_id: Optional[str] = None,
) -> DBModel:
    if requested_model_id:
        _, db_model, _user_model = _resolve_accessible_model(
            db, user, requested_model_id
        )
        if db_model.category != "speech" or not _model_has_ability(db_model, "asr"):
            raise HTTPException(
                status_code=400,
                detail="Selected model does not support ASR transcription",
            )
        return db_model

    user_id = int(user.id)
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
    default_model: Optional[DBModel] = None
    if asr_default and asr_default.model:
        default_model = cast(DBModel, asr_default.model)
        if not _model_has_ability(default_model, "asr"):
            default_model = None

    if default_model:
        try:
            _resolve_accessible_model(db, user, str(default_model.model_id))
            return default_model
        except HTTPException:
            logger.warning(
                "Ignoring invisible default ASR model %s for user %s",
                default_model.model_id,
                user_id,
            )

    accessible_models = _iter_accessible_asr_models(db, user)
    if accessible_models:
        return accessible_models[0]

    raise HTTPException(status_code=404, detail="No ASR model is configured")


def _format_from_upload(file: UploadFile) -> Optional[str]:
    filename = file.filename or ""
    if "." in filename:
        suffix = filename.rsplit(".", 1)[-1].strip().lower()
        if suffix:
            return suffix

    content_type = (file.content_type or "").lower()
    if "/" in content_type:
        subtype = content_type.split("/", 1)[1].split(";", 1)[0].strip()
        if subtype:
            return "mp3" if subtype == "mpeg" else subtype
    return None


async def _read_transcribe_upload_with_size_limit(file: UploadFile) -> bytes:
    audio_bytes = await file.read(MAX_TRANSCRIBE_UPLOAD_BYTES + 1)
    if len(audio_bytes) > MAX_TRANSCRIBE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large")
    return audio_bytes


def _validate_provider_model_name(provider: str, model_name: str) -> None:
    """Validate provider-specific curated model names before saving."""

    if canonical_provider_name(provider) != "deepseek":
        return

    if model_name not in DEEPSEEK_SUPPORTED_MODELS:
        supported = ", ".join(DEEPSEEK_SUPPORTED_MODELS)
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported DeepSeek model '{model_name}'. Supported models: {supported}",
        )


async def _validate_provider_model_listing(
    provider: str,
    model_name: str,
    api_key: Optional[str],
    base_url: Optional[str],
    requested_abilities: Optional[List[str]] = None,
) -> None:
    """Validate provider connectivity by fetching the provider model list."""

    import asyncio

    from ..services.model_list_service import (
        PROVIDER_FETCHERS,
        fetch_models_from_provider,
    )

    provider_id = provider.lower().strip()
    if provider_id not in PROVIDER_FETCHERS:
        raise ValueError(
            f"Connection test is not supported for provider '{provider}' in this category yet"
        )

    models = await asyncio.wait_for(
        fetch_models_from_provider(provider, api_key or "", base_url),
        timeout=10.0,
    )
    # "auto" is a virtual OpenRouter model routed in-process by xrouter-llm; it is
    # not a real OpenRouter slug, so the fetch above only confirms connectivity —
    # skip the model-membership and per-model ability checks.
    if is_auto_router_model(provider, model_name):
        return
    provider_model = _find_provider_model(models, model_name)
    if provider_model is None:
        raise ValueError(f"Model '{model_name}' was not found in provider '{provider}'")

    _validate_requested_abilities(requested_abilities, provider_model)


def _is_default_config_type_compatible(model: Any, config_type: str) -> bool:
    category_by_config_type = {
        "general": "llm",
        "small_fast": "llm",
        "visual": "llm",
        "compact": "llm",
        "embedding": "embedding",
        "image": "image",
        "image_edit": "image",
        "video": "video",
        "asr": "speech",
        "tts": "speech",
        "speech": "speech",
        "sound_effect": "sound_effect",
        "music": "music",
        "rerank": "rerank",
    }

    expected_category = category_by_config_type.get(config_type)
    if expected_category is None:
        return False
    current_category = str(getattr(model, "category", ""))
    if current_category != expected_category:
        return False

    abilities = getattr(model, "abilities", None) or []
    if not isinstance(abilities, list):
        abilities = []

    required_abilities_by_config_type = {
        "visual": {"vision"},
        "image_edit": {"edit"},
        "video": {"generate"},
        "asr": {"asr"},
        "tts": {"tts"},
        "speech": {"asr", "tts"},
        "sound_effect": {"generate"},
        "music": {"generate"},
    }

    required_abilities = required_abilities_by_config_type.get(config_type)
    if not required_abilities:
        return True

    current_abilities = {str(ability) for ability in abilities}
    return required_abilities.issubset(current_abilities)


@model_router.post("/", response_model=ModelWithAccessInfo)
@model_router.post("/register", response_model=ModelWithAccessInfo)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ModelWithAccessInfo:
    """Create a new model configuration"""

    # Debug logging
    logger.info(f"🔍 Creating model: {model.model_id}")
    logger.info(f"  Category: {model.category}")
    logger.info(f"  Provider: {model.model_provider}")
    logger.info(f"  Abilities: {model.abilities}")
    logger.info(f"  Model name: {model.model_name}")

    # Check if model_id already exists
    model_storage = CoreStorage(db, DBModel)

    if model_storage.exists(model.model_id):
        raise HTTPException(status_code=400, detail="Model ID already exists")

    # Only users with sharing permission can share models
    if model.share_with_users and not _can_user_share(user):
        raise HTTPException(
            status_code=403,
            detail="Only administrators can share models with all users",
        )

    model_provider = canonical_provider_name(model.model_provider)
    base_url = model.base_url or default_base_url_for_provider(model_provider)
    if (
        model.category == "video"
        and model_provider == "volcengine-ark"
        and not model.base_url
        and model.model_name.startswith("dreamina-")
    ):
        from xagent.core.model.video.ark import ARK_BYTEPLUS_BASE_URL

        base_url = ARK_BYTEPLUS_BASE_URL
    _validate_provider_model_name(model_provider, model.model_name)

    if model.category == "llm":
        config: ModelConfig = ChatModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            default_temperature=model.temperature,
            context_window=model.context_window,
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
        )
    elif model.category == "embedding":
        config = EmbeddingModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
            dimension=model.dimension,
        )
    elif model.category == "image":
        config = ImageModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            default_temperature=model.temperature,
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
        )
    elif model.category == "video":
        config = VideoModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            timeout=1800.0,
            abilities=model.abilities or ["generate"],
            description=model.description,
        )
    elif model.category == "speech":
        from xagent.core.model.model import SpeechModelConfig

        config = SpeechModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
            language=model.language,
            voice=model.voice,
            format=model.format,
            sample_rate=model.sample_rate,
        )
    elif model.category == "sound_effect":
        config = SoundEffectModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            timeout=180.0,
            abilities=model.abilities or ["generate"],
            description=model.description,
        )
    elif model.category == "music":
        config = MusicModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=base_url,
            api_key=model.api_key or "",
            timeout=600.0,
            abilities=model.abilities or ["generate"],
            description=model.description,
        )
    elif model.category == "rerank":
        # DashScope rerank has model-family-specific endpoints; let the
        # adapter derive the correct URL when the form leaves it blank.
        from xagent.core.model.rerank.dashscope import _default_url_for

        rerank_base_url = base_url
        if model_provider == "dashscope" and model.model_name:
            rerank_base_url = _default_url_for(model.model_name)

        config = RerankModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model_provider,
            base_url=rerank_base_url,
            api_key=model.api_key or "",
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
            top_n=model.top_n,
            instruct=model.instruct,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model category")

    model_storage.store(config)

    db_model = model_storage.get_db_model(model.model_id)
    assert db_model

    # Create user model relationship
    is_share: bool = model.share_with_users and _can_user_share(user)
    ModelStore(db).create_user_model_link(
        user_id=int(user.id),
        model_id=int(db_model.id),
        is_shared=is_share,
    )

    # No pre-creation for other users — dynamic discovery handles visibility.

    assert db_model

    # Create response object with proper field mapping
    response_data = {
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
        "created_at": db_model.created_at.isoformat() if db_model.created_at else None,
        "updated_at": db_model.updated_at.isoformat() if db_model.updated_at else None,
        "is_active": db_model.is_active,
        "is_owner": True,
        "can_edit": True,
        "can_delete": True,
        "is_shared": is_share,
    }
    return ModelWithAccessInfo.model_validate(response_data)


@model_router.get("/", response_model=List[ModelWithAccessInfo])
async def list_models(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_provider: Optional[str] = Query(None, description="Filter by model type"),
    category: Optional[str] = Query(
        None, description="Filter by category (llm, image)"
    ),
    is_active: Optional[bool] = Query(True, description="Filter by active status"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[ModelWithAccessInfo]:
    """List all model configurations accessible to the current user"""
    return ModelStore(db).list_models(
        user_id=int(user.id),
        skip=skip,
        limit=limit,
        model_provider=model_provider,
        category=category,
        is_active=is_active,
    )


@model_router.get("/user-default")
async def get_user_default_models(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list:
    """Get all user's default model configurations with per-type admin fallback"""
    try:
        return ModelStore(db).get_user_default_models(user)
    except Exception as e:
        logger.error(f"Error getting user default models: {e}")
        # Return an empty list even if an error occurs, instead of 404
        return []


@model_router.get("/by-id/{model_id:path}", response_model=ModelWithAccessInfo)
async def get_model_by_path(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> ModelWithAccessInfo:
    """Get a specific model configuration, including slash-containing model IDs."""

    _, db_model, user_model = _resolve_accessible_model(db, user, model_id)
    return ModelWithAccessInfo.model_validate(
        ModelStore(db).serialize_model_with_access(
            db_model, user_model, requesting_user_id=int(user.id)
        )
    )


@model_router.put("/by-id/{model_id:path}", response_model=ModelWithAccessInfo)
async def update_model_by_path(
    model_id: str,
    model_update: ModelUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ModelWithAccessInfo:
    """Update a model configuration, including slash-containing model IDs."""

    return await update_model(model_id, model_update, db, user)


@model_router.delete("/by-id/{model_id:path}")
async def delete_model_by_path(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    """Delete a model configuration, including slash-containing model IDs."""

    return await delete_model(model_id, db, user)


@model_router.post("/test-connection", response_model=ModelTestResponse)
async def test_model_connection(
    request: ModelConnectionTestRequest,
    user: User = Depends(get_current_user),
) -> ModelTestResponse:
    """Test connection with provided model parameters before saving."""
    from xagent.core.model.chat.basic.adapter import create_base_llm
    from xagent.core.model.embedding.adapter import create_embedding_adapter
    from xagent.core.model.image.adapter import create_image_model
    from xagent.core.model.xinference_base import BaseXinferenceModel

    start_time = time.time()
    timeout_seconds = 10.0
    try:
        provider = canonical_provider_name(request.model_provider)
        base_url = request.base_url or default_base_url_for_provider(provider)
        if (
            request.category == "video"
            and provider == "volcengine-ark"
            and not request.base_url
            and request.model_name.startswith("dreamina-")
        ):
            from xagent.core.model.video.ark import ARK_BYTEPLUS_BASE_URL

            base_url = ARK_BYTEPLUS_BASE_URL

        if request.category == "llm":
            # For some reasoning models (like o1, o3, claude reasoning variants), temperature might be deprecated
            # and max_tokens might be replaced by max_completion_tokens. We use a more minimal test strategy here.
            model_name_lower = request.model_name.lower()
            is_reasoning_model = (
                model_name_lower.startswith(("o1", "o3", "gpt-5"))
                or "-o1" in model_name_lower
                or "-o3" in model_name_lower
                or "-gpt-5" in model_name_lower
                or "thinking" in model_name_lower
                or "reasoner" in model_name_lower
            )
            is_deepseek_model = provider == "deepseek"

            config_kwargs: dict[str, Any] = {
                "id": "test-model",
                "model_provider": provider,
                "model_name": request.model_name,
                "api_key": request.api_key,
                "base_url": base_url,
            }

            # Add temperature only if it's not a known reasoning model that rejects it
            if not is_reasoning_model:
                config_kwargs["default_temperature"] = request.temperature or 0.7

            config = ChatModelConfig(**config_kwargs)
            llm = create_base_llm(config)

            # Test chat connection with a small but non-trivial token budget.
            # ``max_tokens=1`` is unsafe for reasoning models that aren't
            # caught by the name-based heuristic above (e.g. qwen3-thinking
            # variants advertised as ``qwen3.x_*``): they would consume the
            # single token in ``reasoning_content`` and return an empty
            # ``content``, which providers report as an invalid response.
            # 16 tokens is still cheap and gives reasoning models room to
            # produce at least the start of an answer.
            chat_kwargs: dict[str, Any] = {"max_tokens": 16}

            # Claude models and OpenAI o1/o3 handle max_tokens differently or deprecate temperature
            if is_reasoning_model:
                chat_kwargs = {}  # let the adapter handle defaults
            if is_deepseek_model:
                chat_kwargs["thinking"] = {"type": "disabled"}

            await asyncio.wait_for(
                llm.chat([{"role": "user", "content": "Hello"}], **chat_kwargs),
                timeout=timeout_seconds,
            )

        elif request.category == "embedding":
            embedding_config = EmbeddingModelConfig(
                id="test-model",
                model_provider=provider,
                model_name=request.model_name,
                api_key=request.api_key,
                base_url=base_url,
                dimension=request.dimension or 1536,
                abilities=request.abilities or ["embedding"],
            )
            embedding_model = create_embedding_adapter(embedding_config)
            await asyncio.wait_for(
                asyncio.to_thread(embedding_model.encode, "hello"),
                timeout=timeout_seconds,
            )

        elif request.category == "image":
            image_config = ImageModelConfig(
                id="test-model",
                model_provider=provider,
                model_name=request.model_name,
                api_key=request.api_key,
                base_url=base_url,
                abilities=request.abilities or ["generate"],
            )
            create_image_model(image_config)
            await asyncio.wait_for(
                _validate_provider_model_listing(
                    provider=provider,
                    model_name=request.model_name,
                    api_key=request.api_key,
                    base_url=base_url,
                    requested_abilities=request.abilities,
                ),
                timeout=timeout_seconds,
            )

        elif request.category == "video":
            from xagent.core.model.video.adapter import create_video_model

            video_config = VideoModelConfig(
                id="test-model",
                model_name=request.model_name,
                model_provider=provider,
                api_key=request.api_key,
                base_url=base_url,
                abilities=request.abilities or ["generate"],
                timeout=1800.0,
            )
            video_model = create_video_model(video_config)
            validator = getattr(video_model, "validate_configuration", None)
            if callable(validator):
                validator()

        elif request.category == "speech":
            if provider not in {"xinference", "elevenlabs"}:
                raise ValueError(
                    f"Unsupported speech provider for testing: {request.model_provider}"
                )

            if provider == "elevenlabs":
                requested_abilities = request.abilities
                unsupported_abilities = sorted(
                    set(requested_abilities or []) - {"asr", "tts"}
                )
                if unsupported_abilities:
                    raise ValueError(
                        "ElevenLabs speech connection test only supports ASR/TTS abilities: "
                        + ", ".join(unsupported_abilities)
                    )
            else:
                requested_abilities = request.abilities or ["asr"]

            await asyncio.wait_for(
                _validate_provider_model_listing(
                    provider=provider,
                    model_name=request.model_name,
                    api_key=request.api_key,
                    base_url=base_url,
                    requested_abilities=requested_abilities,
                ),
                timeout=timeout_seconds,
            )

            if provider == "elevenlabs":
                response_time = time.time() - start_time
                return ModelTestResponse(
                    model_id=request.model_name,
                    status="passed",
                    response_time=response_time,
                    message="Connection successful",
                    error=None,
                )

            probe_model = BaseXinferenceModel(
                model=request.model_name,
                model_uid=request.model_name,
                base_url=base_url,
                api_key=request.api_key or None,
            )
            try:
                await asyncio.wait_for(
                    probe_model._ensure_model_handle(), timeout=timeout_seconds
                )
            finally:
                await probe_model.aclose()

        elif request.category == "sound_effect":
            if provider != "elevenlabs":
                raise ValueError(
                    "Sound effect connection testing currently supports ElevenLabs"
                )
            from xagent.core.model.sound_effect import create_sound_effect_model

            sound_effect_model = create_sound_effect_model(
                SoundEffectModelConfig(
                    id="test-model",
                    model_name=request.model_name,
                    model_provider=provider,
                    api_key=request.api_key,
                    base_url=base_url,
                    abilities=request.abilities or ["generate"],
                )
            )
            try:
                await asyncio.wait_for(
                    sound_effect_model.validate_connection(), timeout=timeout_seconds
                )
            finally:
                await sound_effect_model.aclose()

        elif request.category == "music":
            if provider != "elevenlabs":
                raise ValueError(
                    "Music connection testing currently supports ElevenLabs"
                )
            from xagent.core.model.music import create_music_model

            music_model = create_music_model(
                MusicModelConfig(
                    id="test-model",
                    model_name=request.model_name,
                    model_provider=provider,
                    api_key=request.api_key,
                    base_url=base_url,
                    abilities=request.abilities or ["generate"],
                )
            )
            try:
                await asyncio.wait_for(
                    music_model.validate_connection(), timeout=timeout_seconds
                )
            finally:
                await music_model.aclose()

        elif request.category == "rerank":
            from xagent.core.model.rerank.adapter import _create_rerank_model
            from xagent.core.model.rerank.dashscope import _default_url_for

            # The DashScope rerank endpoint differs between model families
            # (qwen3-rerank uses the OpenAI-compatible URL, gte-rerank-v2
            # uses the legacy WebAPI). Derive the URL from the model name
            # so a stale ``base_url`` from the form cannot break the
            # connectivity probe.
            rerank_base_url = base_url
            if provider == "dashscope" and request.model_name:
                rerank_base_url = _default_url_for(request.model_name)

            rerank_config = RerankModelConfig(
                id="test-model",
                model_provider=provider,
                model_name=request.model_name,
                api_key=request.api_key,
                base_url=rerank_base_url,
                top_n=request.top_n,
                instruct=request.instruct,
            )
            rerank_model = _create_rerank_model(rerank_config)

            def _probe_rerank() -> list:
                return list(
                    rerank_model.compress(
                        documents=["hello", "world"],
                        query="hello",
                    )
                )

            await asyncio.wait_for(
                asyncio.to_thread(_probe_rerank),
                timeout=timeout_seconds,
            )

        else:
            raise ValueError(f"Unsupported category for testing: {request.category}")

        response_time = time.time() - start_time
        return ModelTestResponse(
            model_id=request.model_name,
            status="passed",
            response_time=response_time,
            message="Connection successful",
            error=None,
        )

    except asyncio.TimeoutError:
        logger.error(f"Model connection test timed out for {request.model_name}")
        response_time = time.time() - start_time
        return ModelTestResponse(
            model_id=request.model_name,
            status="failed",
            response_time=response_time,
            message="Connection timed out",
            error=f"Connection timed out after {int(timeout_seconds)} seconds. Please check your network connection and provider status.",
        )
    except Exception as e:
        logger.error(f"Model connection test failed: {e}")
        response_time = time.time() - start_time
        safe_error = redact_sensitive_text(str(e))
        return ModelTestResponse(
            model_id=request.model_name,
            status="failed",
            response_time=response_time,
            message="Connection failed",
            error=safe_error,
        )


@model_router.post("/speech/transcribe")
async def transcribe_speech_input(
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Transcribe a short UI voice input clip with an accessible ASR model."""

    from xagent.core.model.asr.adapter import get_asr_model_instance

    try:
        audio_bytes = await _read_transcribe_upload_with_size_limit(file)
    finally:
        await file.close()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    db_model = _resolve_asr_model_for_transcription(db, user, model_id)
    asr_model = get_asr_model_instance(db_model)
    try:
        result = await asyncio.wait_for(
            asr_model.transcribe(
                audio_bytes,
                language=language,
                format=_format_from_upload(file),
            ),
            timeout=180.0,
        )
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Transcription timed out") from exc
    except Exception as exc:
        safe_error = redact_sensitive_text(str(exc))
        logger.error(
            "Speech transcription failed for model %s: %s",
            db_model.model_id,
            safe_error,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Speech transcription failed: {safe_error}",
        ) from exc
    finally:
        close = getattr(asr_model, "aclose", None)
        if callable(close):
            result_to_close = close()
            if isawaitable(result_to_close):
                await result_to_close

    text = getattr(result, "text", result)
    return {
        "text": str(text or ""),
        "model_id": db_model.model_id,
        "model_name": db_model.model_name,
    }


@model_router.post("/test", response_model=List[ModelTestResponse])
async def test_models(
    test_request: Optional[ModelTestRequest] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[ModelTestResponse]:
    """Test model configurations"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    model_storage = CoreStorage(db, DBModel)
    visible_ids = _get_visible_user_ids(db, int(user.id))

    if test_request and test_request.model_ids:
        # Test specific models that user has access to
        models = (
            db.query(DBModel)
            .join(UserModel, DBModel.id == UserModel.model_id)
            .filter(
                DBModel.model_id.in_(test_request.model_ids),
                DBModel.is_active,
                build_user_model_visibility_filter(int(user.id), visible_ids),
            )
            .all()
        )
    else:
        # Test all active models that user has access to
        models = (
            db.query(DBModel)
            .join(UserModel, DBModel.id == UserModel.model_id)
            .filter(
                DBModel.is_active,
                build_user_model_visibility_filter(int(user.id), visible_ids),
            )
            .all()
        )

    if not models:
        return []

    test_results = []
    test_message = "Test message - are you working?"

    for model in models:
        start_time = time.time()

        try:
            llm = model_storage.get_llm_by_id(str(model.model_id))
            if not llm:
                test_results.append(
                    ModelTestResponse(
                        model_id=model.model_id,
                        status="failed",
                        response_time=None,
                        message="Failed to create LLM instance",
                        error="Unsupported model type",
                    )
                )
                continue

            # Test with a simple message. Use a small but non-trivial token
            # budget so that reasoning models (e.g. qwen3-thinking,
            # deepseek-r1) have room to produce at least the start of an
            # answer instead of getting truncated mid-thought, which would
            # otherwise surface as "Invalid response" from the provider.
            test_messages = [{"role": "user", "content": test_message}]
            await llm.chat(test_messages, max_tokens=16)
            response_time = time.time() - start_time

            test_results.append(
                ModelTestResponse(
                    model_id=model.model_id,
                    status="passed",
                    response_time=response_time,
                    message="Model test successful",
                    error=None,
                )
            )

        except Exception as e:
            response_time = time.time() - start_time
            safe_error = redact_sensitive_text(str(e))
            logger.error(
                "Error testing model %s: %s",
                model.model_id,
                safe_error,
            )
            test_results.append(
                ModelTestResponse(
                    model_id=model.model_id,
                    status="failed",
                    response_time=response_time,
                    message="Model test failed",
                    error=safe_error,
                )
            )

    return test_results


@model_router.get("/types/available")
async def get_available_model_providers() -> dict:
    """Get available model providers"""

    return {
        "model_providers": [
            {
                "type": "openai",
                "name": "OpenAI",
                "description": "OpenAI API compatible models",
                "examples": ["gpt-4", "gpt-4o", "gpt-3.5-turbo"],
            },
            {
                "type": "zhipu",
                "name": "Zhipu AI",
                "description": "Zhipu AI models",
                "examples": ["glm-4", "glm-4-air", "glm-3-turbo"],
            },
            {
                "type": "deepseek",
                "name": "DeepSeek",
                "description": "DeepSeek v4 models with tool calling and thinking mode",
                "examples": ["deepseek-v4-flash", "deepseek-v4-pro"],
            },
        ]
    }


@model_router.get("/categories")
async def list_model_categories(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """List all model categories accessible to the current user"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    # Get distinct categories from user's accessible models
    visible_ids = _get_visible_user_ids(db, int(user.id))
    categories = (
        db.query(DBModel.category)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(build_user_model_visibility_filter(int(user.id), visible_ids))
        .filter(DBModel.is_active)
        .distinct()
        .all()
    )

    return {
        "categories": [cat[0] for cat in categories],
    }


@model_router.get("/providers")
async def list_model_providers(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """List all model providers accessible to the current user"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    # Get distinct providers from user's accessible models
    visible_ids = _get_visible_user_ids(db, int(user.id))
    providers = (
        db.query(DBModel.model_provider)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(build_user_model_visibility_filter(int(user.id), visible_ids))
        .filter(DBModel.is_active)
        .distinct()
        .all()
    )

    return {
        "providers": [prov[0] for prov in providers],
    }


@model_router.get("/abilities")
async def list_model_abilities(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """List all model abilities across accessible models"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    # Get all models to collect abilities
    visible_ids = _get_visible_user_ids(db, int(user.id))
    models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(build_user_model_visibility_filter(int(user.id), visible_ids))
        .filter(DBModel.is_active)
        .all()
    )

    abilities_set: set[str] = set()
    for model in models:
        if model.abilities:
            abilities_set.update(model.abilities)

    return {
        "abilities": sorted(list(abilities_set)),
    }


@model_router.get("/summary")
async def get_models_summary(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Get summary statistics of accessible models"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    # Get all accessible models
    visible_ids = _get_visible_user_ids(db, int(user.id))
    models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(build_user_model_visibility_filter(int(user.id), visible_ids))
        .filter(DBModel.is_active)
        .all()
    )

    # Count by category
    category_counts: dict[str, int] = {}
    provider_counts: dict[str, int] = {}
    total_models = len(models)

    for model in models:
        # Count by category
        cat = str(model.category)
        prov = str(model.model_provider)
        category_counts[cat] = category_counts.get(cat, 0) + 1
        # Count by provider
        provider_counts[prov] = provider_counts.get(prov, 0) + 1

    return {
        "total_models": total_models,
        "by_category": category_counts,
        "by_provider": provider_counts,
    }


@model_router.get(
    "/default/{model_provider}", response_model=Optional[ModelWithAccessInfo]
)
async def get_default_model(
    model_provider: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Optional[ModelWithAccessInfo]:
    """Get the default model for a specific type"""

    # Map model_provider to config_type
    config_type_map = {
        "llm": "general",
        "embedding": "embedding",
        "image": "image",
        "image_edit": "image_edit",
        "video": "video",
        "asr": "asr",
        "tts": "tts",
        "speech": "speech",
        "sound_effect": "sound_effect",
        "music": "music",
        "rerank": "rerank",
    }

    config_type = config_type_map.get(model_provider, "general")

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == config_type,
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.get("/default/general", response_model=Optional[ModelWithAccessInfo])
async def get_general_default_model(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Optional[ModelWithAccessInfo]:
    """Get the general default model (config_type='general')"""

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == "general",
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.get("/default/small-fast", response_model=Optional[ModelWithAccessInfo])
async def get_small_fast_default_model(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Optional[ModelWithAccessInfo]:
    """Get the small/fast default model (config_type='small_fast')"""

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == "small_fast",
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.get("/default/visual", response_model=Optional[ModelWithAccessInfo])
async def get_visual_default_model(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Optional[ModelWithAccessInfo]:
    """Get the visual default model (config_type='visual')"""

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == "visual",
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.get("/default/compact", response_model=Optional[ModelWithAccessInfo])
async def get_compact_default_model(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Optional[ModelWithAccessInfo]:
    """Get the compact default model (config_type='compact')"""

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == "compact",
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.get("/default/embedding", response_model=Optional[ModelWithAccessInfo])
async def get_embedding_default_model(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> Optional[ModelWithAccessInfo]:
    """Get the default embedding model (config_type='embedding')"""

    # Get user's default model configuration
    user_default = (
        db.query(UserDefaultModel)
        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == "embedding",
            DBModel.is_active,
        )
        .first()
    )

    if not user_default:
        return None

    # Get user model relationship for access info (two-step: own or shared)

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == user_default.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        return None

    is_owner = user_model.user_id == user.id
    model_data = {
        "id": user_default.model.id,
        "model_id": user_default.model.model_id,
        "category": user_default.model.category,
        "model_provider": user_default.model.model_provider,
        "model_name": user_default.model.model_name,
        "base_url": user_default.model.base_url,
        "temperature": user_default.model.temperature,
        "dimension": user_default.model.dimension,
        "abilities": user_default.model.abilities,
        "description": user_default.model.description,
        "created_at": user_default.model.created_at.isoformat()
        if user_default.model.created_at
        else None,
        "updated_at": user_default.model.updated_at.isoformat()
        if user_default.model.updated_at
        else None,
        "is_active": user_default.model.is_active,
        "is_owner": is_owner,
        "can_edit": is_owner and user_model.can_edit,
        "can_delete": is_owner and user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


# User Default Model Configuration Endpoints


@model_router.post("/user-default", response_model=UserDefaultModelResponse)
async def set_user_default_model(
    config: UserDefaultModelCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> UserDefaultModelResponse:
    """Set a user's default model configuration"""

    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    # Check if user has access to the model (own or shared from visible users)
    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == config.model_id,
            build_user_model_visibility_filter(int(user.id), visible_ids),
        )
        .first()
    )

    if not user_model:
        raise HTTPException(status_code=404, detail="Model not found or access denied")

    # Get the model to check its abilities
    model = db.query(DBModel).filter(DBModel.id == config.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Respect the config_type selected by the client so users can choose
    # distinct defaults for multi-capability speech models.
    config_type = config.config_type

    if not _is_default_config_type_compatible(model, config_type):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Config type '{config_type}' is incompatible with model category "
                f"'{model.category}'"
            ),
        )

    user_default = ModelStore(db).set_user_default_model(
        user_id=int(user.id),
        model_id=int(config.model_id),
        config_type=config_type,
        user_model=user_model,
    )

    # If this is an embedding model configuration, trigger memory store check
    if config.config_type == "embedding":
        try:
            from ..dynamic_memory_store import get_memory_store_manager

            manager = get_memory_store_manager()
            with UserContext(int(user.id)):
                if manager.check_embedding_model_change():
                    logger.info(
                        f"Memory store updated for user {user.id} after setting default embedding model"
                    )
        except Exception as e:
            logger.error(
                f"Error updating memory store after setting default embedding model: {e}"
            )

    return UserDefaultModelResponse.model_validate(user_default)


@model_router.get(
    "/user-default/{config_type}", response_model=Optional[UserDefaultModelResponse]
)
async def get_user_default_model(
    config_type: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Optional[UserDefaultModelResponse]:
    """Get a user's default model configuration for a specific type"""
    return ModelStore(db).get_user_default_model(int(user.id), config_type)


@model_router.delete("/user-default/{config_type}")
async def delete_user_default_model(
    config_type: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Delete a user's default model configuration"""

    user_default = ModelStore(db).delete_user_default_model(
        user_id=int(user.id), config_type=config_type
    )

    if not user_default:
        raise HTTPException(status_code=404, detail="Default configuration not found")

    return {"message": "Default configuration deleted successfully"}


# ---------------------------------------------------------------------------
# Catch-all /{model_id} routes — MUST come after all fixed-path routes to
# avoid matching "categories", "providers", "summary", etc.
# ---------------------------------------------------------------------------


@model_router.get("/{model_id}", response_model=ModelWithAccessInfo)
async def get_model(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> ModelWithAccessInfo:
    """Get a specific model configuration"""
    _, db_model, user_model = _resolve_accessible_model(db, user, model_id)
    return ModelWithAccessInfo.model_validate(
        ModelStore(db).serialize_model_with_access(
            db_model, user_model, requesting_user_id=int(user.id)
        )
    )


@model_router.put("/{model_id}", response_model=ModelWithAccessInfo)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ModelWithAccessInfo:
    """Update a model configuration"""
    _, db_model_ref, user_model = _resolve_accessible_model(db, user, model_id)

    # Permission: only owner can edit
    if user_model.user_id != user.id or not user_model.can_edit:
        raise HTTPException(status_code=403, detail="No permission to edit this model")

    # Get the database model
    db_model = user_model.model

    model_store = ModelStore(db)

    # Constraint 2: shared models cannot change category or abilities
    is_shared = model_store.model_has_shared_visibility(int(db_model.id))
    if is_shared:
        locked = []
        if (
            model_update.category is not None
            and model_update.category != db_model.category
        ):
            locked.append("category")
        if model_update.abilities is not None and set(model_update.abilities) != set(
            db_model.abilities or []
        ):
            locked.append("abilities")
        if locked:
            raise HTTPException(
                409,
                detail=f"Cannot change {' and '.join(locked)} of a shared model. Un-share first.",
            )

    share_with_users = model_update.share_with_users
    if share_with_users and not _can_user_share(user):
        raise HTTPException(
            status_code=403, detail="Only administrators can enable global sharing"
        )

    # Update model configuration in-place
    update_data = model_update.model_dump(exclude_unset=True)
    if "model_provider" in update_data:
        update_data["model_provider"] = canonical_provider_name(
            update_data["model_provider"]
        )
    effective_provider = update_data.get("model_provider", db_model.model_provider)
    effective_model_name = update_data.get("model_name", db_model.model_name)
    _validate_provider_model_name(effective_provider, effective_model_name)

    for field, value in update_data.items():
        # Don't update api_key with empty string
        if field == "api_key" and value == "":
            continue
        # Skip share_with_users as it's handled separately
        if field == "share_with_users":
            continue
        # Only set fields that exist on the model
        if hasattr(db_model, field):
            setattr(db_model, field, value)

    if share_with_users is not None:
        try:
            model_store.set_model_sharing(
                user_id=int(user.id),
                db_model=db_model,
                user_model=user_model,
                share_with_users=share_with_users,
            )
        except ModelSharingConflictError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
    else:
        model_store.commit_model_update(
            user_id=int(user.id),
            db_model=db_model,
            invalidate_globally=is_shared,
        )

    # Return updated model with access info
    return ModelWithAccessInfo.model_validate(
        model_store.serialize_model_with_access(
            db_model, user_model, requesting_user_id=int(user.id)
        )
    )


@model_router.delete("/{model_id}")
async def delete_model(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    """Delete a model configuration"""
    model_storage, _, user_model = _resolve_accessible_model(db, user, model_id)

    # Permission: only owner can delete
    if user_model.user_id != user.id or not user_model.can_delete:
        raise HTTPException(
            status_code=403, detail="No permission to delete this model"
        )

    # Constraint 1: owner cannot delete own default model
    owner_defaults = (
        db.query(UserDefaultModel)
        .filter(
            UserDefaultModel.model_id == user_model.model.id,
            UserDefaultModel.user_id == user.id,
        )
        .count()
    )
    if owner_defaults > 0:
        raise HTTPException(
            409,
            detail="Cannot delete: you have this model as your default. Change default first.",
        )

    ModelStore(db).delete_model(model_storage=model_storage, user_model=user_model)

    return {"message": "Model deleted successfully"}


# Public endpoints (no authentication required)


@model_router.get("/public/list")
async def list_public_models(
    category: Optional[str] = Query(None, description="Filter by category"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    limit: int = Query(50, ge=1, le=100),
    db: Session = Depends(get_db),
) -> dict:
    """List public model information (no authentication required).

    Returns only basic model information without sensitive data like API keys.
    """

    query = db.query(DBModel).filter(DBModel.is_active)

    if category:
        query = query.filter(DBModel.category == category)
    if provider:
        query = query.filter(DBModel.model_provider == provider)

    models = query.limit(limit).all()

    result = []
    for model in models:
        model_data: dict[str, Any] = {
            "id": model.id,
            "model_id": model.model_id,
            "category": model.category,
            "model_provider": model.model_provider,
            "model_name": model.model_name,
            "abilities": model.abilities,
            "description": model.description,
        }
        # Add category-specific fields
        if model.category == "llm":
            model_data["temperature"] = model.temperature
            model_data["max_tokens"] = model.max_tokens
            model_data["context_window"] = model.context_window
        elif model.category == "embedding":
            model_data["dimension"] = model.dimension

        result.append(model_data)

    return {
        "models": result,
        "count": len(result),
    }


@model_router.get("/public/categories")
async def list_public_categories(
    db: Session = Depends(get_db),
) -> dict:
    """List all available model categories (no authentication required)."""

    categories = db.query(DBModel.category).filter(DBModel.is_active).distinct().all()

    return {
        "categories": [cat[0] for cat in categories],
    }


@model_router.get("/public/providers")
async def list_public_providers(
    db: Session = Depends(get_db),
) -> dict:
    """List all available model providers (no authentication required)."""

    providers = (
        db.query(DBModel.model_provider).filter(DBModel.is_active).distinct().all()
    )

    return {
        "providers": [prov[0] for prov in providers],
    }


@model_router.get("/public/summary")
async def get_public_summary(
    db: Session = Depends(get_db),
) -> dict:
    """Get public summary of available models (no authentication required)."""

    total_models = db.query(DBModel).filter(DBModel.is_active).count()

    # Count by category
    category_counts = {}
    for cat in [
        "llm",
        "embedding",
        "rerank",
        "image",
        "video",
        "speech",
        "sound_effect",
        "music",
    ]:
        count = (
            db.query(DBModel)
            .filter(DBModel.category == cat)
            .filter(DBModel.is_active)
            .count()
        )
        category_counts[cat] = count

    return {
        "total_models": total_models,
        "by_category": category_counts,
    }


# Provider model fetching endpoints


@model_router.get("/providers/supported")
async def list_supported_providers() -> dict:
    """Get list of supported model providers with their information."""

    from ..services.model_list_service import get_supported_providers

    providers = get_supported_providers()

    return {
        "providers": providers,
    }


@model_router.post("/providers/{provider}/models")
async def fetch_provider_models(
    provider: str,
    api_key: str = Body(...),
    base_url: Optional[str] = Body(None),
    category: Optional[str] = Body(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Fetch available models from a specific provider.

    Requires the provider's API key. For providers like Azure OpenAI,
    base_url is also required. When category helps route to the correct fetcher
    for provider+category combinations (e.g. xinference+rerank).
    """
    api_key = api_key.strip()
    base_url = base_url.strip() if base_url else base_url

    # Validate provider
    from ..services.model_list_service import (
        PROVIDER_FETCHERS,
        fetch_models_from_provider,
    )

    # Try provider+category combination first (e.g. "xinference-rerank"),
    # then fallback to base provider name
    combined_key = f"{provider}-{category}" if category else None
    if combined_key and combined_key in PROVIDER_FETCHERS:
        provider_to_use = combined_key
    elif provider.lower() not in PROVIDER_FETCHERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported providers: {list(PROVIDER_FETCHERS.keys())}",
        )
    else:
        provider_to_use = provider.lower()

    # Providers that mark base_url as mandatory (e.g. Azure OpenAI,
    # Xinference, OpenAI-Compatible) must not silently fall back to a
    # provider-side default when it's omitted.
    if not base_url and (
        provider_to_use == "azure_openai" or provider_requires_base_url(provider)
    ):
        raise HTTPException(
            status_code=400,
            detail=f"base_url is required for the {provider} provider",
        )

    try:
        models = await fetch_models_from_provider(provider_to_use, api_key, base_url)

        return {
            "provider": provider,
            "models": models,
            "count": len(models),
        }
    except Exception as e:
        safe_error = redact_sensitive_text(str(e))
        logger.error(
            "Error fetching models from %s: %s",
            provider,
            safe_error,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models from {provider}: {safe_error}",
        )


@model_router.post("/providers/fetch")
async def fetch_multiple_providers_models(
    providers: List[str],
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Fetch available models from multiple providers at once.

    Uses API keys from existing model configurations in the database.
    This endpoint will use the API keys stored in your configured models
    to fetch available models from each provider.
    """

    # Get all models configured for the user (own or shared from visible users)

    from ..services.model_list_service import (
        PROVIDER_FETCHERS,
        fetch_models_from_provider,
    )
    from ..services.model_service import (
        _get_visible_user_ids,
        build_user_model_visibility_filter,
    )

    visible_ids = _get_visible_user_ids(db, int(user.id))
    user_models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(build_user_model_visibility_filter(int(user.id), visible_ids))
        .filter(DBModel.is_active)
        .filter(DBModel.api_key.isnot(None))
        .all()
    )

    # Group by provider
    provider_keys: dict[str, str] = {}
    provider_base_urls: dict[str, str] = {}

    for model in user_models:
        provider = str(model.model_provider).lower()
        # Use first available API key for each provider
        if provider not in provider_keys and model.api_key:
            provider_keys[provider] = str(model.api_key)
            if model.base_url:
                provider_base_urls[provider] = str(model.base_url)

    # Filter to requested providers
    if providers:
        provider_keys = {
            k: v
            for k, v in provider_keys.items()
            if k in [p.lower() for p in providers]
        }

    results: dict[str, Any] = {}

    for provider, api_key in provider_keys.items():
        if provider not in PROVIDER_FETCHERS:
            results[provider] = {"error": "Unsupported provider", "models": []}
            continue

        base_url = provider_base_urls.get(provider)

        try:
            models = await fetch_models_from_provider(provider, api_key, base_url)
            results[provider] = {
                "models": models,
                "count": len(models),
            }
        except Exception as e:
            safe_error = redact_sensitive_text(str(e))
            logger.error(
                "Error fetching from %s: %s",
                provider,
                safe_error,
            )
            results[provider] = {
                "error": safe_error,
                "models": [],
            }

    return {
        "results": results,
    }


@model_router.get("/xinference/tts-models")
async def list_xinference_tts_models(
    base_url: str = Query(..., description="Xinference server base URL"),
    api_key: Optional[str] = Query(None, description="Optional API key"),
) -> dict:
    """Get available TTS models from Xinference server.

    Returns a list of TTS/audio models running on the Xinference server,
    along with their model abilities that can be used for the 'abilities' field
    when registering a model.

    For TTS models, use abilities: ["tts"]
    For ASR models, use abilities: ["asr"]
    For models with both capabilities, use: ["tts", "asr"]
    """
    try:
        from xagent.core.model.tts.xinference import XinferenceTTS

        models = XinferenceTTS.list_available_models(base_url=base_url, api_key=api_key)

        # Map model abilities to xagent abilities format
        result_models = []
        for model in models:
            model_ability = model.get("model_ability", [])

            # Determine xagent abilities based on model capabilities
            abilities = []
            if any(ability.startswith("text2audio") for ability in model_ability):
                abilities.append("tts")
            if any(ability.startswith("audio2text") for ability in model_ability):
                abilities.append("asr")

            result_models.append(
                {
                    "id": model["id"],
                    "model_uid": model["model_uid"],
                    "model_type": model["model_type"],
                    "model_ability": model_ability,
                    "description": model["description"],
                    "abilities": abilities,  # Suggested abilities for xagent
                    "category": "speech",
                    "model_provider": "xinference",
                }
            )

        return {
            "models": result_models,
            "count": len(result_models),
        }

    except Exception as e:
        logger.error(f"Error fetching Xinference TTS models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch TTS models from Xinference: {str(e)}",
        )
