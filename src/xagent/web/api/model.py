"""Model management API route handlers"""

import logging
import time
from typing import Any, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from xagent.core.model.model import (
    ChatModelConfig,
    EmbeddingModelConfig,
    ImageModelConfig,
    ModelConfig,
)

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.model import Model as DBModel
from ..models.user import User, UserDefaultModel, UserModel
from ..schemas.model import (
    ModelCreate,
    ModelTestRequest,
    ModelTestResponse,
    ModelUpdate,
    ModelWithAccessInfo,
    UserDefaultModelCreate,
    UserDefaultModelResponse,
)
from ..services.llm_utils import CoreStorage
from ..user_isolated_memory import UserContext

logger = logging.getLogger(__name__)

# Create router
model_router = APIRouter(prefix="/api/models", tags=["models"])


@model_router.post("/", response_model=ModelWithAccessInfo)
@model_router.post("/register", response_model=ModelWithAccessInfo)
async def create_model(
    model: ModelCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ModelWithAccessInfo:
    """Create a new model configuration"""

    # Check if model_id already exists
    model_storage = CoreStorage(db, DBModel)

    if model_storage.exists(model.model_id):
        raise HTTPException(status_code=400, detail="Model ID already exists")

    # Only admin can share models with all users
    if model.share_with_users and not user.is_admin:
        raise HTTPException(
            status_code=403,
            detail="Only administrators can share models with all users",
        )

    if model.category == "llm":
        config: ModelConfig = ChatModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model.model_provider,
            base_url=model.base_url,
            api_key=model.api_key,
            default_temperature=model.temperature,
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
        )
    elif model.category == "embedding":
        config = EmbeddingModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model.model_provider,
            base_url=model.base_url,
            api_key=model.api_key,
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
            dimension=model.dimension,
        )
    elif model.category == "image":
        config = ImageModelConfig(
            id=model.model_id,
            model_name=model.model_name,
            model_provider=model.model_provider,
            base_url=model.base_url,
            api_key=model.api_key,
            default_temperature=model.temperature,
            timeout=180.0,
            abilities=model.abilities,
            description=model.description,
        )
    else:
        raise HTTPException(status_code=400, detail="Invalid model category")

    model_storage.store(config)

    db_model = model_storage.get_db_model(model.model_id)
    assert db_model

    # Create user model relationship
    user_model = UserModel(
        user_id=user.id,
        model_id=db_model.id,
        is_owner=True,
        can_edit=True,
        can_delete=True,
        is_shared=model.share_with_users and user.is_admin,
    )
    db.add(user_model)
    db.commit()

    is_share: bool = model.share_with_users and bool(user.is_admin)
    # If admin is sharing the model, create relationships for all users
    if is_share:
        all_users = db.query(User).filter(User.id != user.id).all()
        for other_user in all_users:
            shared_user_model = UserModel(
                user_id=other_user.id,
                model_id=db_model.id,
                is_owner=False,
                can_edit=False,
                can_delete=False,
                is_shared=True,
            )
            db.add(shared_user_model)
        db.commit()

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

    # Get models that user has access to (owned or shared)
    query = (
        db.query(DBModel, UserModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
    )
    if model_provider:
        query = query.filter(DBModel.model_provider == model_provider)

    if category:
        query = query.filter(DBModel.category == category)

    if is_active is not None:
        query = query.filter(DBModel.is_active == is_active)

    models = query.offset(skip).limit(limit).all()

    result = []
    for db_model, user_model in models:
        model_data = {
            "id": db_model.id,
            "model_id": db_model.model_id,
            "category": db_model.category,
            "model_provider": db_model.model_provider,
            "model_name": db_model.model_name,
            "base_url": db_model.base_url,
            "temperature": db_model.temperature,
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
            "is_owner": user_model.is_owner,
            "can_edit": user_model.can_edit,
            "can_delete": user_model.can_delete,
            "is_shared": user_model.is_shared,
        }
        result.append(ModelWithAccessInfo.model_validate(model_data))

    return result


@model_router.get("/user-default")
async def get_user_default_models(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> list:
    """Get all user's default model configurations with per-type admin fallback"""

    try:
        # Get all possible config types
        all_config_types = [
            "general",
            "small_fast",
            "visual",
            "compact",
            "embedding",
            "image",
            "image_edit",
        ]

        # Get user's own defaults
        user_defaults_by_type: dict[str, UserDefaultModel] = {}
        user_defaults = (
            db.query(UserDefaultModel)
            .join(DBModel, UserDefaultModel.model_id == DBModel.id)
            .filter(UserDefaultModel.user_id == user.id, DBModel.is_active)
            .all()
        )

        # Organize user defaults by config type
        for ud in user_defaults:
            user_defaults_by_type[str(ud.config_type)] = ud

        result = []

        # Process each config type
        for config_type in all_config_types:
            if config_type in user_defaults_by_type:
                # User has their own default for this type
                ud = user_defaults_by_type[config_type]

                # Get user model relationship for access info
                user_model = (
                    db.query(UserModel)
                    .filter(
                        UserModel.user_id == user.id, UserModel.model_id == ud.model_id
                    )
                    .first()
                )

                if user_model:
                    model_data = {
                        "id": ud.id,
                        "user_id": ud.user_id,
                        "model_id": ud.model_id,
                        "config_type": ud.config_type,
                        "created_at": ud.created_at.isoformat()
                        if ud.created_at
                        else None,
                        "updated_at": ud.updated_at.isoformat()
                        if ud.updated_at
                        else None,
                        "model": {
                            "id": user_model.model.id,
                            "model_id": user_model.model.model_id,
                            "category": user_model.model.category,
                            "model_provider": user_model.model.model_provider,
                            "model_name": user_model.model.model_name,
                            "base_url": user_model.model.base_url,
                            "temperature": user_model.model.temperature,
                            "dimension": user_model.model.dimension,
                            "abilities": user_model.model.abilities,
                            "description": user_model.model.description,
                            "created_at": user_model.model.created_at.isoformat()
                            if user_model.model.created_at
                            else None,
                            "updated_at": user_model.model.updated_at.isoformat()
                            if user_model.model.updated_at
                            else None,
                            "is_active": user_model.model.is_active,
                            "is_owner": user_model.is_owner,
                            "can_edit": user_model.can_edit,
                            "can_delete": user_model.can_delete,
                            "is_shared": user_model.is_shared,
                        },
                    }
                    result.append(model_data)
            else:
                # User has no default for this type, try admin fallback
                admin_user = db.query(User).filter(User.is_admin).first()
                if admin_user:
                    admin_default = (
                        db.query(UserDefaultModel)
                        .join(DBModel, UserDefaultModel.model_id == DBModel.id)
                        .join(
                            UserModel, UserDefaultModel.model_id == UserModel.model_id
                        )
                        .filter(
                            UserDefaultModel.user_id == admin_user.id,
                            UserDefaultModel.config_type == config_type,
                            DBModel.is_active,
                            UserModel.is_shared,
                        )
                        .first()
                    )

                    if admin_default:
                        logger.info(
                            f"User {user.username} has no {config_type} default, using admin default for display"
                        )

                        model_data = {
                            "id": admin_default.id,
                            "user_id": admin_default.user_id,
                            "model_id": admin_default.model_id,
                            "config_type": admin_default.config_type,
                            "created_at": admin_default.created_at.isoformat()
                            if admin_default.created_at
                            else None,
                            "updated_at": admin_default.updated_at.isoformat()
                            if admin_default.updated_at
                            else None,
                            "model": {
                                "id": admin_default.model.id,
                                "model_id": admin_default.model.model_id,
                                "category": admin_default.model.category,
                                "model_provider": admin_default.model.model_provider,
                                "model_name": admin_default.model.model_name,
                                "base_url": admin_default.model.base_url,
                                "temperature": admin_default.model.temperature,
                                "dimension": admin_default.model.dimension,
                                "abilities": admin_default.model.abilities,
                                "description": admin_default.model.description,
                                "created_at": admin_default.model.created_at.isoformat()
                                if admin_default.model.created_at
                                else None,
                                "updated_at": admin_default.model.updated_at.isoformat()
                                if admin_default.model.updated_at
                                else None,
                                "is_active": admin_default.model.is_active,
                                "is_owner": False,  # Admin models are not owned by non-admin users
                                "can_edit": False,
                                "can_delete": False,
                                "is_shared": True,
                            },
                        }
                        result.append(model_data)

        return result
    except Exception as e:
        logger.error(f"Error getting user default models: {e}")
        # Return an empty list even if an error occurs, instead of 404
        return []


@model_router.get("/{model_id}", response_model=ModelWithAccessInfo)
async def get_model(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> ModelWithAccessInfo:
    """Get a specific model configuration"""
    # Decode URL-encoded model_id (supports double-encoding for slashes)
    import urllib.parse

    decoded_model_id = urllib.parse.unquote(model_id)

    model_storage = CoreStorage(db, DBModel)

    # Check if model exists (resolves by ID, model_id, or model_name)
    db_model = model_storage.get_db_model(decoded_model_id)
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if user has access to this model
    user_model = (
        db.query(UserModel)
        .filter(UserModel.model_id == db_model.id, UserModel.user_id == user.id)
        .first()
    )

    if not user_model:
        raise HTTPException(status_code=404, detail="Model not found or access denied")

    model_data = {
        "id": user_model.model.id,
        "model_id": user_model.model.model_id,
        "category": user_model.model.category,
        "model_provider": user_model.model.model_provider,
        "model_name": user_model.model.model_name,
        "base_url": user_model.model.base_url,
        "temperature": user_model.model.temperature,
        "dimension": user_model.model.dimension,
        "abilities": user_model.model.abilities,
        "description": user_model.model.description,
        "created_at": user_model.model.created_at.isoformat()
        if user_model.model.created_at
        else None,
        "updated_at": user_model.model.updated_at.isoformat()
        if user_model.model.updated_at
        else None,
        "is_active": user_model.model.is_active,
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.put("/{model_id}", response_model=ModelWithAccessInfo)
async def update_model(
    model_id: str,
    model_update: ModelUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ModelWithAccessInfo:
    """Update a model configuration"""
    # Decode URL-encoded model_id (supports double-encoding for slashes)
    import urllib.parse

    decoded_model_id = urllib.parse.unquote(model_id)

    model_storage = CoreStorage(db, DBModel)

    # Resolve model first
    db_model_resolved = model_storage.get_db_model(decoded_model_id)
    if not db_model_resolved:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if user has access to this model and can edit
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.model_id == db_model_resolved.id, UserModel.user_id == user.id
        )
        .first()
    )

    if not user_model:
        raise HTTPException(status_code=404, detail="Model not found or access denied")

    if not user_model.can_edit:
        raise HTTPException(status_code=403, detail="No permission to edit this model")

    # Get the database model
    db_model = user_model.model

    # Handle admin sharing updates
    if model_update.share_with_users is not None:
        # Only check admin permission when enabling sharing (share_with_users=True)
        # Allow non-admin users to disable sharing (share_with_users=False)
        if model_update.share_with_users and not user.is_admin:
            raise HTTPException(
                status_code=403, detail="Only administrators can enable global sharing"
            )

        # Update sharing status for all user models
        db.query(UserModel).filter(UserModel.model_id == db_model.id).update(
            {"is_shared": model_update.share_with_users}
        )

        # If sharing is enabled, create relationships for users who don't have access
        if model_update.share_with_users:
            existing_user_ids = [
                um.user_id
                for um in db.query(UserModel)
                .filter(UserModel.model_id == db_model.id)
                .all()
            ]

            all_users = db.query(User).filter(User.id.notin_(existing_user_ids)).all()
            for other_user in all_users:
                shared_user_model = UserModel(
                    user_id=other_user.id,
                    model_id=db_model.id,
                    is_owner=False,
                    can_edit=False,
                    can_delete=False,
                    is_shared=True,
                )
                db.add(shared_user_model)

    # Update model configuration in-place
    update_data = model_update.model_dump(exclude_unset=True)
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

    # Commit database changes
    db.commit()
    db.refresh(db_model)

    # Return updated model with access info
    model_data = {
        "id": db_model.id,
        "model_id": db_model.model_id,
        "category": db_model.category,
        "model_provider": db_model.model_provider,
        "model_name": db_model.model_name,
        "base_url": db_model.base_url,
        "temperature": db_model.temperature,
        "dimension": db_model.dimension,
        "abilities": db_model.abilities,
        "description": db_model.description,
        "created_at": db_model.created_at.isoformat() if db_model.created_at else None,
        "updated_at": db_model.updated_at.isoformat() if db_model.updated_at else None,
        "is_active": db_model.is_active,
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
        "is_shared": user_model.is_shared,
    }

    return ModelWithAccessInfo.model_validate(model_data)


@model_router.delete("/{model_id}")
async def delete_model(
    model_id: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)
) -> dict:
    """Delete a model configuration"""
    # Decode URL-encoded model_id (supports double-encoding for slashes)
    import urllib.parse

    decoded_model_id = urllib.parse.unquote(model_id)

    model_storage = CoreStorage(db, DBModel)

    # Resolve model first
    db_model = model_storage.get_db_model(decoded_model_id)
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Check if user has access to this model and can delete
    user_model = (
        db.query(UserModel)
        .filter(UserModel.model_id == db_model.id, UserModel.user_id == user.id)
        .first()
    )

    if not user_model:
        raise HTTPException(status_code=404, detail="Model not found or access denied")

    if not user_model.can_delete:
        raise HTTPException(
            status_code=403, detail="No permission to delete this model"
        )

    # Delete all user model relationships
    db.query(UserModel).filter(UserModel.model_id == user_model.model.id).delete()

    # Delete all user default model configurations
    db.query(UserDefaultModel).filter(
        UserDefaultModel.model_id == user_model.model.id
    ).delete()

    # Delete the model using CoreStorage
    model_storage.delete(decoded_model_id)

    return {"message": "Model deleted successfully"}


@model_router.post("/test", response_model=List[ModelTestResponse])
async def test_models(
    test_request: Optional[ModelTestRequest] = None,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[ModelTestResponse]:
    """Test model configurations"""
    model_storage = CoreStorage(db, DBModel)

    if test_request and test_request.model_ids:
        # Test specific models that user has access to
        models = (
            db.query(DBModel)
            .join(UserModel, DBModel.id == UserModel.model_id)
            .filter(
                DBModel.model_id.in_(test_request.model_ids),
                DBModel.is_active,
                UserModel.user_id == user.id,
            )
            .all()
        )
    else:
        # Test all active models that user has access to
        models = (
            db.query(DBModel)
            .join(UserModel, DBModel.id == UserModel.model_id)
            .filter(DBModel.is_active, UserModel.user_id == user.id)
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

            # Test with a simple message and minimal tokens for speed
            test_messages = [{"role": "user", "content": test_message}]
            await llm.chat(test_messages, max_tokens=1)
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
            logger.error(f"Error testing model {model.model_id}: {e}")
            test_results.append(
                ModelTestResponse(
                    model_id=model.model_id,
                    status="failed",
                    response_time=response_time,
                    message="Model test failed",
                    error=str(e),
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
        ]
    }


@model_router.get("/categories")
async def list_model_categories(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """List all model categories accessible to the current user"""

    # Get distinct categories from user's accessible models
    categories = (
        db.query(DBModel.category)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
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

    # Get distinct providers from user's accessible models
    providers = (
        db.query(DBModel.model_provider)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
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

    # Get all models to collect abilities
    models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
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

    # Get all accessible models
    models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Get user model relationship for access info
    user_model = (
        db.query(UserModel)
        .filter(
            UserModel.user_id == user.id, UserModel.model_id == user_default.model_id
        )
        .first()
    )

    if not user_model:
        return None

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
        "is_owner": user_model.is_owner,
        "can_edit": user_model.can_edit,
        "can_delete": user_model.can_delete,
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

    # Check if user has access to the model
    user_model = (
        db.query(UserModel)
        .filter(UserModel.user_id == user.id, UserModel.model_id == config.model_id)
        .first()
    )

    if not user_model:
        raise HTTPException(status_code=404, detail="Model not found or access denied")

    # Remove existing configuration for this config_type
    db.query(UserDefaultModel).filter(
        UserDefaultModel.user_id == user.id,
        UserDefaultModel.config_type == config.config_type,
    ).delete()

    # Create new default configuration
    user_default = UserDefaultModel(
        user_id=user.id, model_id=config.model_id, config_type=config.config_type
    )

    db.add(user_default)
    db.commit()
    db.refresh(user_default)

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

    return UserDefaultModelResponse.model_validate(user_default)


@model_router.delete("/user-default/{config_type}")
async def delete_user_default_model(
    config_type: str,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Delete a user's default model configuration"""

    user_default = (
        db.query(UserDefaultModel)
        .filter(
            UserDefaultModel.user_id == user.id,
            UserDefaultModel.config_type == config_type,
        )
        .first()
    )

    if not user_default:
        raise HTTPException(status_code=404, detail="Default configuration not found")

    db.delete(user_default)
    db.commit()

    return {"message": "Default configuration deleted successfully"}


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
    for cat in ["llm", "embedding", "rerank", "image"]:
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
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Fetch available models from a specific provider.

    Requires the provider's API key. For providers like Azure OpenAI,
    base_url is also required.
    """

    # Validate provider
    from ..services.model_list_service import (
        PROVIDER_FETCHERS,
        fetch_models_from_provider,
    )

    if provider.lower() not in PROVIDER_FETCHERS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported provider: {provider}. Supported providers: {list(PROVIDER_FETCHERS.keys())}",
        )

    # For Azure OpenAI, base_url is required
    if provider.lower() == "azure_openai" and not base_url:
        raise HTTPException(
            status_code=400,
            detail="base_url is required for Azure OpenAI provider",
        )

    try:
        models = await fetch_models_from_provider(provider, api_key, base_url)

        return {
            "provider": provider,
            "models": models,
            "count": len(models),
        }
    except Exception as e:
        logger.error(f"Error fetching models from {provider}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch models from {provider}: {str(e)}",
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

    from ..services.model_list_service import (
        PROVIDER_FETCHERS,
        fetch_models_from_provider,
    )

    # Get all models configured for the user
    user_models = (
        db.query(DBModel)
        .join(UserModel, DBModel.id == UserModel.model_id)
        .filter(UserModel.user_id == user.id)
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
            logger.error(f"Error fetching from {provider}: {e}")
            results[provider] = {
                "error": str(e),
                "models": [],
            }

    return {
        "results": results,
    }
