from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError, model_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..builtin_mcp_registry import (
    get_builtin_execution_fields,
    get_builtin_public_mcp_app,
    is_builtin_public_mcp_app,
)
from ..models.database import get_db
from ..models.oauth_provider import OAuthProvider
from ..models.public_mcp import PublicMCPApp, PublicMCPAppAudit
from ..models.user import User

admin_mcp_router = APIRouter(prefix="/api/admin/mcp", tags=["Admin MCP"])

_LAUNCH_CONFIG_DESCRIPTION = (
    "Runtime launch metadata such as command, arguments, required environment "
    "variable names, and mappings. Do not include credentials or secret values; "
    "configure them through the connector credential flow."
)


def verify_admin(user: User = Depends(get_current_user)) -> User:
    if not getattr(user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required"
        )
    return user


# Pydantic schemas
class OAuthProviderBase(BaseModel):
    provider_name: str
    name: str
    client_id: str
    client_secret: str
    auth_url: str
    token_url: str
    redirect_uri: Optional[str] = None
    userinfo_url: Optional[str] = None
    user_id_path: Optional[str] = "id"
    email_path: Optional[str] = "email"
    default_scopes: Optional[List[str]] = None


class OAuthProviderCreate(OAuthProviderBase):
    pass


class OAuthProviderUpdate(BaseModel):
    provider_name: Optional[str] = None
    name: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    auth_url: Optional[str] = None
    token_url: Optional[str] = None
    redirect_uri: Optional[str] = None
    userinfo_url: Optional[str] = None
    user_id_path: Optional[str] = None
    email_path: Optional[str] = None
    default_scopes: Optional[List[str]] = None


class OAuthProviderResponse(OAuthProviderBase):
    id: int


class PublicMCPAppBase(BaseModel):
    app_id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    transport: str = "oauth"
    provider_name: Optional[str] = None
    category: Optional[str] = None
    oauth_scopes: Optional[List[str]] = None
    is_visible_in_connector: bool = True
    launch_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=_LAUNCH_CONFIG_DESCRIPTION,
    )


class PublicMCPAppCreate(PublicMCPAppBase):
    # Validator lives on the write model only, not PublicMCPAppBase — otherwise
    # PublicMCPAppResponse would inherit it and re-run on response serialization,
    # turning one legacy/partial DB row into a full-list 500 on read.
    @model_validator(mode="after")
    def _enforce_auth_classification(self) -> "PublicMCPAppCreate":
        # Reuse the single source of truth (classify_app_auth) rather than
        # re-deriving the rule here. Reject an entry that declares a partial
        # launch_config (either the key-based or the remote-OAuth shape) yet
        # still classifies as "unconnectable" — the write-time constraint
        # issue #764 asked for. Covers four asymmetric shapes:
        # command-without-required_env, required_env-without-command,
        # url-without-auth.type=mcp_oauth, and auth.type=mcp_oauth-without-url.
        from ..mcp_apps import classify_app_auth

        launch = self.launch_config or {}
        declares_key_shape = bool(launch.get("command") or launch.get("required_env"))
        auth = launch.get("auth")
        declares_remote_oauth_shape = bool(launch.get("url")) or (
            isinstance(auth, dict) and auth.get("type") == "mcp_oauth"
        )
        if (declares_key_shape or declares_remote_oauth_shape) and (
            classify_app_auth(self.transport, self.launch_config) == "unconnectable"
        ):
            raise ValueError(
                "A key-based catalog app must declare both launch_config.command "
                "and launch_config.required_env; a remote-OAuth catalog app must "
                "declare both launch_config.url and launch_config.auth.type == "
                "'mcp_oauth' (with transport one of sse/websocket/streamable_http), "
                "otherwise it cannot be connected."
            )
        return self


class PublicMCPAppResponse(PublicMCPAppBase):
    id: int
    is_builtin: bool


class PublicMCPAppUpdate(BaseModel):
    app_id: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    icon: Optional[str] = None
    transport: Optional[str] = None
    provider_name: Optional[str] = None
    category: Optional[str] = None
    oauth_scopes: Optional[List[str]] = None
    is_visible_in_connector: Optional[bool] = None
    launch_config: Optional[Dict[str, Any]] = Field(
        default=None,
        description=_LAUNCH_CONFIG_DESCRIPTION,
    )


_PUBLIC_MCP_APP_FIELDS = tuple(PublicMCPAppBase.model_fields)
_BUILTIN_PROTECTED_FIELDS = frozenset(
    {
        "app_id",
        "name",
        "transport",
        "provider_name",
        "oauth_scopes",
        "launch_config",
    }
)
_PUBLIC_MCP_AUDIT_REQUEST_ID_MAX_LENGTH = 128


def _public_mcp_app_values(app: PublicMCPApp) -> Dict[str, Any]:
    return deepcopy({field: getattr(app, field) for field in _PUBLIC_MCP_APP_FIELDS})


def _public_mcp_audit_request_id(request: Request) -> str:
    request_id = request.headers.get("x-request-id")
    if request_id:
        return request_id[:_PUBLIC_MCP_AUDIT_REQUEST_ID_MAX_LENGTH]
    return uuid4().hex


def _record_public_mcp_app_audit(
    db: Session,
    *,
    actor: User,
    request: Request,
    action: str,
    app_id: str,
    before_values: Dict[str, Any] | None,
    after_values: Dict[str, Any] | None,
) -> None:
    """Add one catalog audit row to the caller's write transaction."""
    db.add(
        PublicMCPAppAudit(
            actor_user_id=int(actor.id),
            action=action,
            app_id=app_id,
            before_values=before_values,
            after_values=after_values,
            request_id=_public_mcp_audit_request_id(request),
        )
    )


def _commit_public_mcp_app_write(
    db: Session,
    *,
    integrity_error_detail: str | None = None,
) -> None:
    """Commit one catalog write and always restore the session on failure."""
    try:
        db.commit()
    except Exception as error:
        db.rollback()
        if integrity_error_detail is not None and isinstance(error, IntegrityError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=integrity_error_detail,
            ) from None
        raise


def _public_mcp_app_response(app: PublicMCPApp) -> Dict[str, Any]:
    values = _public_mcp_app_values(app)
    execution_fields = get_builtin_execution_fields(app.app_id)
    if execution_fields is not None:
        values.update(execution_fields)
    return {
        "id": app.id,
        **values,
        "is_builtin": execution_fields is not None,
    }


def _validate_public_mcp_app_values(
    values: Dict[str, Any], *, enforce_connect_shape: bool = True
) -> None:
    # F4: PublicMCPAppCreate's _enforce_auth_classification rejects a
    # connect-shape that classifies as "unconnectable" (see that validator's
    # docstring). Applying it unconditionally on every edit would let a shape
    # rule added after a row was created (or a row otherwise grandfathered
    # in) permanently block that row's unrelated fields (e.g. icon) from ever
    # being edited again. Skip that one check — via PublicMCPAppBase, which
    # PublicMCPAppCreate extends with no other override — when this edit
    # doesn't touch the fields the shape check reads (transport/launch_config).
    model = PublicMCPAppCreate if enforce_connect_shape else PublicMCPAppBase
    try:
        model.model_validate(values)
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid MCP app configuration",
        ) from None


def _apply_public_mcp_app_update(db_app: PublicMCPApp, changes: Dict[str, Any]) -> None:
    canonical = get_builtin_public_mcp_app(db_app.app_id)
    persisted = _public_mcp_app_values(db_app)
    enforce_connect_shape = bool({"transport", "launch_config"} & changes.keys())

    if canonical is not None:
        for field in _BUILTIN_PROTECTED_FIELDS.intersection(changes):
            if changes[field] != canonical[field]:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Built-in MCP app field '{field}' is managed by code",
                )
        writable_changes = {
            field: value
            for field, value in changes.items()
            if field not in _BUILTIN_PROTECTED_FIELDS
        }
        merged = {
            **persisted,
            **writable_changes,
            **{field: canonical[field] for field in _BUILTIN_PROTECTED_FIELDS},
        }
        _validate_public_mcp_app_values(
            merged, enforce_connect_shape=enforce_connect_shape
        )
    else:
        if "app_id" in changes and changes["app_id"] != db_app.app_id:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="MCP app ID is immutable",
            )
        merged = {**persisted, **changes}
        _validate_public_mcp_app_values(
            merged, enforce_connect_shape=enforce_connect_shape
        )
        writable_changes = {
            field: value for field, value in changes.items() if field != "app_id"
        }

    for field, value in writable_changes.items():
        setattr(db_app, field, value)


# --- OAuth Providers ---
@admin_mcp_router.get("/providers", response_model=List[OAuthProviderResponse])
async def list_providers(
    db: Session = Depends(get_db), _: User = Depends(verify_admin)
) -> Any:
    providers = db.query(OAuthProvider).all()
    results = []
    for p in providers:
        p_dict = {c.name: getattr(p, c.name) for c in p.__table__.columns}
        p_dict["client_id"] = "********"
        p_dict["client_secret"] = "********"
        results.append(p_dict)
    return results


@admin_mcp_router.post("/providers", response_model=OAuthProviderResponse)
async def create_provider(
    provider: OAuthProviderCreate,
    db: Session = Depends(get_db),
    _: User = Depends(verify_admin),
) -> Any:
    existing_provider = (
        db.query(OAuthProvider)
        .filter(OAuthProvider.provider_name == provider.provider_name)
        .first()
    )
    if existing_provider:
        raise HTTPException(status_code=400, detail="Provider already exists")

    provider_data = provider.model_dump()
    provider_data["client_id"] = encrypt_value(provider_data["client_id"])
    provider_data["client_secret"] = encrypt_value(provider_data["client_secret"])
    db_provider = OAuthProvider(**provider_data)
    db.add(db_provider)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Provider already exists") from None
    db.refresh(db_provider)

    # Return masked data
    response_dict = {
        c.name: getattr(db_provider, c.name) for c in db_provider.__table__.columns
    }
    response_dict["client_id"] = "********"
    response_dict["client_secret"] = "********"
    return response_dict


@admin_mcp_router.put("/providers/{provider_id}", response_model=OAuthProviderResponse)
async def update_provider(
    provider_id: int,
    provider: OAuthProviderUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(verify_admin),
) -> Any:
    db_provider = (
        db.query(OAuthProvider).filter(OAuthProvider.id == provider_id).first()
    )
    if not db_provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    provider_data = provider.model_dump(exclude_unset=True)

    if (
        "provider_name" in provider_data
        and provider_data["provider_name"] is not None
        and provider_data["provider_name"] != db_provider.provider_name
    ):
        existing_provider = (
            db.query(OAuthProvider)
            .filter(OAuthProvider.provider_name == provider_data["provider_name"])
            .first()
        )
        if existing_provider:
            raise HTTPException(status_code=400, detail="Provider already exists")

    if "client_id" in provider_data:
        if provider_data["client_id"] is None:
            provider_data.pop("client_id")
        else:
            provider_data["client_id"] = encrypt_value(provider_data["client_id"])

    if "client_secret" in provider_data:
        if provider_data["client_secret"] is None:
            provider_data.pop("client_secret")
        else:
            provider_data["client_secret"] = encrypt_value(
                provider_data["client_secret"]
            )

    for key, value in provider_data.items():
        setattr(db_provider, key, value)

    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(status_code=400, detail="Provider already exists") from None
    db.refresh(db_provider)

    # Return masked data
    response_dict = {
        c.name: getattr(db_provider, c.name) for c in db_provider.__table__.columns
    }
    response_dict["client_id"] = "********"
    response_dict["client_secret"] = "********"
    return response_dict


@admin_mcp_router.delete("/providers/{provider_id}")
async def delete_provider(
    provider_id: int, db: Session = Depends(get_db), _: User = Depends(verify_admin)
) -> dict:
    db_provider = (
        db.query(OAuthProvider).filter(OAuthProvider.id == provider_id).first()
    )
    if not db_provider:
        raise HTTPException(status_code=404, detail="Provider not found")
    linked_apps_count = (
        db.query(PublicMCPApp)
        .filter(PublicMCPApp.provider_name == db_provider.provider_name)
        .count()
    )
    if linked_apps_count > 0:
        raise HTTPException(
            status_code=409,
            detail=(
                "Provider is referenced by one or more MCP apps. "
                "Remove or update linked apps before deleting this provider."
            ),
        )
    db.delete(db_provider)
    db.commit()
    return {"success": True}


# --- Public MCP Apps ---
@admin_mcp_router.get("/apps", response_model=list[PublicMCPAppResponse])
async def list_apps(
    db: Session = Depends(get_db), _: User = Depends(verify_admin)
) -> Any:
    apps = db.query(PublicMCPApp).all()
    return [_public_mcp_app_response(app) for app in apps]


@admin_mcp_router.post("/apps", response_model=PublicMCPAppResponse)
async def create_app(
    app: PublicMCPAppCreate,
    request: Request,
    db: Session = Depends(get_db),
    actor: User = Depends(verify_admin),
) -> Any:
    if is_builtin_public_mcp_app(app.app_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Built-in MCP app IDs are reserved",
        )
    existing_app = (
        db.query(PublicMCPApp).filter(PublicMCPApp.app_id == app.app_id).first()
    )
    if existing_app:
        raise HTTPException(status_code=400, detail="App already exists")
    db_app = PublicMCPApp(**app.model_dump())
    db.add(db_app)
    _record_public_mcp_app_audit(
        db,
        actor=actor,
        request=request,
        action="create",
        app_id=db_app.app_id,
        before_values=None,
        after_values=_public_mcp_app_values(db_app),
    )
    _commit_public_mcp_app_write(db, integrity_error_detail="App already exists")
    db.refresh(db_app)
    return _public_mcp_app_response(db_app)


@admin_mcp_router.put(
    "/apps/{app_id}",
    response_model=PublicMCPAppResponse,
    description=(
        "Full replacement of a public MCP app. Built-in identity and execution "
        "fields must match their code-owned canonical values; use PATCH for "
        "partial updates to editable presentation fields."
    ),
)
async def update_app(
    app_id: int,
    app: PublicMCPAppCreate,
    request: Request,
    db: Session = Depends(get_db),
    actor: User = Depends(verify_admin),
) -> Any:
    db_app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_id).first()
    if not db_app:
        raise HTTPException(status_code=404, detail="App not found")

    before_values = _public_mcp_app_values(db_app)
    _apply_public_mcp_app_update(db_app, app.model_dump())
    _record_public_mcp_app_audit(
        db,
        actor=actor,
        request=request,
        action="update",
        app_id=db_app.app_id,
        before_values=before_values,
        after_values=_public_mcp_app_values(db_app),
    )

    _commit_public_mcp_app_write(db)
    db.refresh(db_app)
    return _public_mcp_app_response(db_app)


@admin_mcp_router.patch(
    "/apps/{app_id}",
    response_model=PublicMCPAppResponse,
    description=(
        "Partial update of a public MCP app. For built-in apps, only editable "
        "presentation fields such as description, icon, category, and visibility "
        "may be changed."
    ),
)
async def patch_app(
    app_id: int,
    app: PublicMCPAppUpdate,
    request: Request,
    db: Session = Depends(get_db),
    actor: User = Depends(verify_admin),
) -> Any:
    db_app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_id).first()
    if not db_app:
        raise HTTPException(status_code=404, detail="App not found")

    before_values = _public_mcp_app_values(db_app)
    _apply_public_mcp_app_update(db_app, app.model_dump(exclude_unset=True))
    _record_public_mcp_app_audit(
        db,
        actor=actor,
        request=request,
        action="update",
        app_id=db_app.app_id,
        before_values=before_values,
        after_values=_public_mcp_app_values(db_app),
    )

    _commit_public_mcp_app_write(db)
    db.refresh(db_app)
    return _public_mcp_app_response(db_app)


@admin_mcp_router.delete("/apps/{app_id}")
async def delete_app(
    app_id: int,
    request: Request,
    db: Session = Depends(get_db),
    actor: User = Depends(verify_admin),
) -> dict:
    db_app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_id).first()
    if not db_app:
        raise HTTPException(status_code=404, detail="App not found")
    if is_builtin_public_mcp_app(db_app.app_id):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Built-in MCP apps are managed by code",
        )
    before_values = _public_mcp_app_values(db_app)
    _record_public_mcp_app_audit(
        db,
        actor=actor,
        request=request,
        action="delete",
        app_id=db_app.app_id,
        before_values=before_values,
        after_values=None,
    )
    db.delete(db_app)
    _commit_public_mcp_app_write(db)
    return {"success": True}
