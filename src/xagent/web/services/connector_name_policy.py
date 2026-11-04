"""Uniqueness of connector names across MCP servers and Custom APIs.

A task selects connectors with ``mcp:<name>``, and the selector is compared
against every connector name through ``normalize_mcp_server_name`` (strip,
spaces and hyphens to underscore, lower case). That selector addresses MCP
servers and Custom APIs alike, so two names that fold to the same key are
selected together. Each table only enforces uniqueness of the raw name; this
module adds the cross-table, folded check that name-writing routes run before
they write.

Callers: the routes that create or rename an MCP server or a Custom API from
a name the user typed. Provisioning that derives the name from the connector
catalog must not call it, because an existing folded twin would otherwise keep
every user from connecting the official app.

Catalog apps get their server row only when someone first connects them, so a
typed name is also checked against the catalog itself
(``folds_to_catalog_app_name``): otherwise a name taken before that first
connect would collide with the row provisioning creates later.

The check is global. One tenant's name blocks every tenant from using any
case, space, hyphen or underscore variant of it, across both connector types.
It has to be global because team sharing makes a row visible to a team without
any name write, so the only moment a collision can be refused is when a name
is written, and at that moment only the global scope is enforceable.

Known differences and limits. On SQLite, and on PostgreSQL with a C/POSIX
ctype, they can only cause a missed conflict (a colliding pair the check did
not catch). With a non-C PostgreSQL locale, case folding of some non-ASCII
names may differ from Python's, which can cause either a missed conflict or a
rejection:

* The comparison runs in SQL so no table is read into memory. SQL ``trim``
  removes only spaces, while Python ``str.strip`` also removes tabs,
  newlines and other Unicode whitespace. Create and rename refuse Custom API
  names whose edge whitespace includes anything besides spaces, and no MCP
  name starts or ends with whitespace other than spaces, so this difference
  only affects rows written before that check existed.
* SQLite ``lower`` folds ASCII only, and PostgreSQL ``lower`` folds ASCII
  only under a C/POSIX ctype; Python ``str.lower`` folds all of Unicode.
* The check reads before the caller writes, so two concurrent writers can both
  pass it. The per-table unique constraint on the raw name still holds.

The check also reveals whether some connector, possibly another tenant's,
already has a name that folds to the requested one. It answers yes or no and
never echoes the stored spelling.
"""

from __future__ import annotations

from sqlalchemy import and_, func
from sqlalchemy.orm import Session
from sqlalchemy.sql.elements import ColumnElement

from ...core.tools.adapters.vibe.connector_runtime import (
    CONNECTOR_TYPE_CUSTOM_API,
    CONNECTOR_TYPE_MCP,
    ConnectorRef,
)
from ...core.tools.adapters.vibe.selection_spec import normalize_mcp_server_name
from ..models.custom_api import CustomApi
from ..models.mcp import MCPServer
from ..models.public_mcp import PublicMCPApp


def _sql_fold(column: ColumnElement) -> ColumnElement:
    """The database expression of ``normalize_mcp_server_name``."""
    return func.lower(func.replace(func.replace(func.trim(column), " ", "_"), "-", "_"))


def has_folded_connector_name_conflict(
    db: Session, name: str, *, exclude: ConnectorRef | None = None
) -> bool:
    """Whether ``name`` folds to the same key as the name of an existing MCP
    server or Custom API, other than the row identified by ``exclude``.

    ``exclude`` matches on connector type and id together: an MCP server and a
    Custom API can share a numeric id, and only the row being renamed is
    exempt. Read failures propagate to the caller.
    """
    key = normalize_mcp_server_name(name)
    for model, connector_type in (
        (MCPServer, CONNECTOR_TYPE_MCP),
        (CustomApi, CONNECTOR_TYPE_CUSTOM_API),
    ):
        condition = _sql_fold(model.name) == key
        if exclude is not None and exclude.connector_type == connector_type:
            condition = and_(condition, model.id != exclude.connector_id)
        if db.query(model.id).filter(condition).first() is not None:
            return True
    return False


def folded_name_conflict_detail(name: str) -> str:
    """The error text for a rejected name; echoes only the requested name."""
    return (
        f"'{name}' conflicts with an existing connector name. Connector names "
        "are compared case-insensitively, treating spaces, hyphens and "
        "underscores as the same character, and must be unique across MCP "
        "servers and custom APIs."
    )


def folds_to_catalog_app_name(db: Session, name: str) -> bool:
    """Whether ``name`` folds to the same key as the app id or display name
    of any catalog app, including apps hidden from the connector list.

    Provisioning names a catalog app's server row after one of those two
    spellings, so the app reserves both whether or not its row exists yet.
    The catalog table is the source because provisioning only creates rows
    for apps it can read from it; built-in apps are seeded into it. Read
    failures propagate to the caller.
    """
    key = normalize_mcp_server_name(name)
    if not key:
        return False
    columns = PublicMCPApp.__table__.c
    condition = (_sql_fold(columns.app_id) == key) | (_sql_fold(columns.name) == key)
    return db.query(PublicMCPApp.id).filter(condition).first() is not None


def catalog_app_name_detail(name: str) -> str:
    """The error text for a name rejected by ``folds_to_catalog_app_name``."""
    return (
        f"'{name}' is reserved for a catalog app; connect it from the catalog "
        "instead. Connector names are compared case-insensitively, treating "
        "spaces, hyphens and underscores as the same character."
    )


def has_unfoldable_edge_whitespace(name: str) -> bool:
    """Whether leading or trailing whitespace of ``name`` contains anything
    other than plain spaces (U+0020).

    A Custom API create or rename must not store such a name: SQL ``trim``
    removes only spaces, so the stored name would fold differently in the
    database than ``normalize_mcp_server_name`` folds it in task selection,
    and the folded uniqueness check would miss collisions with it.
    """
    return name.strip() != name.strip(" ")


def unfoldable_edge_whitespace_detail() -> str:
    """The error text for a name rejected by ``has_unfoldable_edge_whitespace``."""
    return (
        "Connector names cannot start or end with tabs, line breaks or other "
        "whitespace besides spaces."
    )
