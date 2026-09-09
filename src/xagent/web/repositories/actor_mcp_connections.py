from __future__ import annotations

from typing import Any, cast
from uuid import UUID

from sqlalchemy import delete, exists, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Query, Session

from ..models.actor_mcp_connection import ActorMCPServerConnection
from ..models.public_mcp import PublicMCPApp


def scoped_actor_mcp_connection_query(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> Query[ActorMCPServerConnection]:
    """Build the only permitted base query for actor MCP connections."""
    return db.query(ActorMCPServerConnection).filter(
        ActorMCPServerConnection.user_id == user_id,
        ActorMCPServerConnection.resource_owner_key == resource_owner_key,
    )


def get_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    catalog_app_generation: UUID | None = None,
    expected_lifecycle_generation: UUID,
    for_update: bool = False,
) -> ActorMCPServerConnection | None:
    query = scoped_actor_mcp_connection_query(
        db,
        user_id=user_id,
        resource_owner_key=resource_owner_key,
    ).filter(
        ActorMCPServerConnection.app_id == app_id,
        ActorMCPServerConnection.lifecycle_generation == expected_lifecycle_generation,
        *(
            (ActorMCPServerConnection.catalog_app_generation == catalog_app_generation,)
            if catalog_app_generation is not None
            else ()
        ),
        exists().where(
            PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
            PublicMCPApp.generation == ActorMCPServerConnection.catalog_app_generation,
        ),
    )
    if for_update:
        query = query.with_for_update()
    return cast(
        ActorMCPServerConnection | None,
        query.populate_existing().one_or_none(),
    )


def get_actor_mcp_connection_metadata_row(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
) -> ActorMCPServerConnection | None:
    return cast(
        ActorMCPServerConnection | None,
        scoped_actor_mcp_connection_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .filter(
            ActorMCPServerConnection.app_id == app_id,
            exists().where(
                PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
                PublicMCPApp.generation
                == ActorMCPServerConnection.catalog_app_generation,
            ),
        )
        .populate_existing()
        .one_or_none(),
    )


def get_current_actor_mcp_connection_for_create(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    catalog_app_generation: UUID,
) -> ActorMCPServerConnection | None:
    """Read a current-lifecycle row only to converge an idempotent create."""
    return (
        scoped_actor_mcp_connection_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .filter(
            ActorMCPServerConnection.app_id == app_id,
            ActorMCPServerConnection.catalog_app_generation == catalog_app_generation,
            exists().where(
                PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
                PublicMCPApp.generation
                == ActorMCPServerConnection.catalog_app_generation,
            ),
        )
        .populate_existing()
        .one_or_none()
    )


def actor_mcp_connection_identity_exists_for_create(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
) -> bool:
    """Classify a create collision without exposing an unfenced row."""
    return (
        scoped_actor_mcp_connection_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .filter(ActorMCPServerConnection.app_id == app_id)
        .first()
        is not None
    )


def list_actor_mcp_connections(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
) -> list[ActorMCPServerConnection]:
    return (
        scoped_actor_mcp_connection_query(
            db,
            user_id=user_id,
            resource_owner_key=resource_owner_key,
        )
        .filter(
            exists().where(
                PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
                PublicMCPApp.generation
                == ActorMCPServerConnection.catalog_app_generation,
            )
        )
        .order_by(ActorMCPServerConnection.id)
        .populate_existing()
        .all()
    )


def add_actor_mcp_connection(
    db: Session, connection: ActorMCPServerConnection
) -> ActorMCPServerConnection:
    db.add(connection)
    db.flush()
    return connection


def update_actor_mcp_connection_env(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    expected_lifecycle_generation: UUID,
    encrypted_env: dict[str, str] | None,
) -> bool:
    result = cast(
        CursorResult[Any],
        db.execute(
            update(ActorMCPServerConnection)
            .where(
                ActorMCPServerConnection.user_id == user_id,
                ActorMCPServerConnection.resource_owner_key == resource_owner_key,
                ActorMCPServerConnection.app_id == app_id,
                ActorMCPServerConnection.lifecycle_generation
                == expected_lifecycle_generation,
                exists().where(
                    PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
                    PublicMCPApp.generation
                    == ActorMCPServerConnection.catalog_app_generation,
                ),
            )
            .values(encrypted_env=encrypted_env)
            .execution_options(synchronize_session=False)
        ),
    )
    return bool(result.rowcount == 1)


def hard_delete_actor_mcp_connection(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str,
    app_id: str,
    expected_lifecycle_generation: UUID,
) -> bool:
    result = cast(
        CursorResult[Any],
        db.execute(
            delete(ActorMCPServerConnection)
            .where(
                ActorMCPServerConnection.user_id == user_id,
                ActorMCPServerConnection.resource_owner_key == resource_owner_key,
                ActorMCPServerConnection.app_id == app_id,
                ActorMCPServerConnection.lifecycle_generation
                == expected_lifecycle_generation,
                exists().where(
                    PublicMCPApp.app_id == ActorMCPServerConnection.app_id,
                    PublicMCPApp.generation
                    == ActorMCPServerConnection.catalog_app_generation,
                ),
            )
            .execution_options(synchronize_session=False)
        ),
    )
    return bool(result.rowcount == 1)
