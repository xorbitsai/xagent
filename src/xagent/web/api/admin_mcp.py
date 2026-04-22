from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.oauth_provider import OAuthProvider
from ..models.public_mcp import PublicMCPApp
from ..models.user import User

admin_mcp_router = APIRouter(prefix="/api/admin/mcp", tags=["Admin MCP"])


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
    launch_config: Optional[Dict[str, Any]] = None


class PublicMCPAppCreate(PublicMCPAppBase):
    pass


class PublicMCPAppResponse(PublicMCPAppBase):
    id: int


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
    provider_data = provider.model_dump()
    provider_data["client_id"] = encrypt_value(provider_data["client_id"])
    provider_data["client_secret"] = encrypt_value(provider_data["client_secret"])
    db_provider = OAuthProvider(**provider_data)
    db.add(db_provider)
    db.commit()
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
    provider: OAuthProviderCreate,
    db: Session = Depends(get_db),
    _: User = Depends(verify_admin),
) -> Any:
    db_provider = (
        db.query(OAuthProvider).filter(OAuthProvider.id == provider_id).first()
    )
    if not db_provider:
        raise HTTPException(status_code=404, detail="Provider not found")

    provider_data = provider.model_dump()

    if provider_data["client_id"] == "********":
        provider_data.pop("client_id")
    else:
        provider_data["client_id"] = encrypt_value(provider_data["client_id"])

    if provider_data["client_secret"] == "********":
        provider_data.pop("client_secret")
    else:
        provider_data["client_secret"] = encrypt_value(provider_data["client_secret"])

    for key, value in provider_data.items():
        setattr(db_provider, key, value)

    db.commit()
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
    db.delete(db_provider)
    db.commit()
    return {"success": True}


# --- Public MCP Apps ---
@admin_mcp_router.get("/apps", response_model=list[PublicMCPAppResponse])
async def list_apps(
    db: Session = Depends(get_db), _: User = Depends(verify_admin)
) -> Any:
    apps = db.query(PublicMCPApp).all()
    return apps


@admin_mcp_router.post("/apps", response_model=PublicMCPAppResponse)
async def create_app(
    app: PublicMCPAppCreate,
    db: Session = Depends(get_db),
    _: User = Depends(verify_admin),
) -> Any:
    db_app = PublicMCPApp(**app.model_dump())
    db.add(db_app)
    db.commit()
    db.refresh(db_app)
    return db_app


@admin_mcp_router.put("/apps/{app_id}", response_model=PublicMCPAppResponse)
async def update_app(
    app_id: int,
    app: PublicMCPAppCreate,
    db: Session = Depends(get_db),
    _: User = Depends(verify_admin),
) -> Any:
    db_app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_id).first()
    if not db_app:
        raise HTTPException(status_code=404, detail="App not found")

    for key, value in app.model_dump().items():
        setattr(db_app, key, value)

    db.commit()
    db.refresh(db_app)
    return db_app


@admin_mcp_router.delete("/apps/{app_id}")
async def delete_app(
    app_id: int, db: Session = Depends(get_db), _: User = Depends(verify_admin)
) -> dict:
    db_app = db.query(PublicMCPApp).filter(PublicMCPApp.id == app_id).first()
    if not db_app:
        raise HTTPException(status_code=404, detail="App not found")
    db.delete(db_app)
    db.commit()
    return {"success": True}
