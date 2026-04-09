"""Cloud Storage API Endpoints"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, Query
from google.auth.transport.requests import Request  # type: ignore
from google.oauth2.credentials import Credentials  # type: ignore
from googleapiclient.discovery import build  # type: ignore
from sqlalchemy.orm import Session

from ..auth_dependencies import get_current_user
from ..models.database import get_db
from ..models.user import User
from ..models.user_oauth import UserOAuth

logger = logging.getLogger(__name__)

cloud_router = APIRouter(prefix="/api/cloud", tags=["Cloud Storage"])

# Google OAuth Constants
GOOGLE_TOKEN_URI = "https://oauth2.googleapis.com/token"


def get_google_credentials(
    user_id: int, db: Session, account_id: Optional[int] = None
) -> Any:
    """Get Google Credentials for user, refreshing if necessary"""
    query = db.query(UserOAuth).filter(
        UserOAuth.user_id == user_id, UserOAuth.provider == "google-drive"
    )

    if account_id:
        query = query.filter(UserOAuth.id == account_id)

    oauth_account = query.first()

    if not oauth_account:
        if account_id:
            raise HTTPException(
                status_code=404, detail="Selected Google Drive account not found"
            )
        raise HTTPException(
            status_code=401, detail="Google Drive account not connected"
        )

    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not client_id or not client_secret:
        raise HTTPException(
            status_code=500, detail="Google OAuth configuration missing"
        )

    creds = Credentials(
        token=oauth_account.access_token,
        refresh_token=oauth_account.refresh_token,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=oauth_account.scope.split(" ") if oauth_account.scope else None,
    )

    # Check if token needs refresh
    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            # Update token in DB
            oauth_account.access_token = creds.token
            if creds.expiry:
                oauth_account.expires_at = creds.expiry
            db.commit()
        except Exception as e:
            logger.error(f"Failed to refresh Google token: {e}")
            raise HTTPException(
                status_code=401,
                detail="Google Drive session expired. Please reconnect.",
            )

    return creds


@cloud_router.get("/accounts")
async def list_connected_accounts(
    provider: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List connected cloud accounts"""
    query = db.query(UserOAuth).filter(UserOAuth.user_id == user.id)

    if provider:
        query = query.filter(UserOAuth.provider == provider)

    accounts = query.all()

    return [
        {
            "id": acc.id,
            "provider": acc.provider,
            "email": acc.email,
            "created_at": acc.created_at,
        }
        for acc in accounts
    ]


@cloud_router.delete("/accounts/{account_id}")
async def delete_connected_account(
    account_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Dict[str, Any]:
    """Delete a connected cloud account"""
    account = (
        db.query(UserOAuth)
        .filter(UserOAuth.id == account_id, UserOAuth.user_id == user.id)
        .first()
    )

    if not account:
        raise HTTPException(status_code=404, detail="Account not found")

    db.delete(account)
    db.commit()

    return {"success": True, "message": "Account deleted successfully"}


@cloud_router.get("/google-drive/drives")
async def list_google_drives(
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List Google Drives (My Drive + Shared Drives)"""
    try:
        creds = get_google_credentials(cast(int, user.id), db, account_id)

        # Build Drive API service
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        drives_list = [{"id": "root", "name": "My Drive", "kind": "drive#drive"}]

        # List Shared Drives
        try:
            results = service.drives().list(pageSize=100).execute()
            shared_drives = results.get("drives", [])
            drives_list.extend(shared_drives)
        except Exception:
            # logger.warning(f"Failed to list shared drives: {e}")
            pass

        return drives_list

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing Google Drives: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@cloud_router.get("/google-drive/files")
async def list_google_drive_files(
    folder_id: str = "root",
    account_id: Optional[int] = Query(None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List files in Google Drive folder"""
    try:
        creds = get_google_credentials(cast(int, user.id), db, account_id)

        # Build Drive API service
        # cache_discovery=False to avoid file system issues and timeouts
        service = build("drive", "v3", credentials=creds, cache_discovery=False)

        # Query for files in folder, not trashed
        # fields reference: https://developers.google.com/drive/api/v3/reference/files/list
        query = f"'{folder_id}' in parents and trashed = false"

        # Include drives support for Shared Drives
        supports_all_drives = True
        include_items_from_all_drives = True

        results = (
            service.files()
            .list(
                q=query,
                pageSize=100,
                fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                orderBy="folder,name",
                supportsAllDrives=supports_all_drives,
                includeItemsFromAllDrives=include_items_from_all_drives,
            )
            .execute()
        )

        files = results.get("files", [])

        # Map to frontend format
        cloud_files = []
        for file in files:
            mime_type = file.get("mimeType")
            is_folder = mime_type == "application/vnd.google-apps.folder"

            # Format size
            size_str = None
            if "size" in file:
                size_bytes = int(file["size"])
                if size_bytes < 1024:
                    size_str = f"{size_bytes} B"
                elif size_bytes < 1024 * 1024:
                    size_str = f"{size_bytes / 1024:.1f} KB"
                elif size_bytes < 1024 * 1024 * 1024:
                    size_str = f"{size_bytes / (1024 * 1024):.1f} MB"
                else:
                    size_str = f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"

            # Format date (ISO 8601 to YYYY-MM-DD)
            updated_at = file.get("modifiedTime", "")
            if updated_at:
                try:
                    # Simple slice for YYYY-MM-DD, or use datetime parsing if needed
                    updated_at = updated_at.split("T")[0]
                except Exception:
                    pass

            cloud_files.append(
                {
                    "id": file.get("id"),
                    "name": file.get("name"),
                    "type": "folder" if is_folder else "file",
                    "size": size_str,
                    "updatedAt": updated_at,
                    "mimeType": mime_type,  # Optional, helpful for debugging
                }
            )

        return cloud_files

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing Google Drive files: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
