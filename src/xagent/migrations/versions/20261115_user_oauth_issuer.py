"""Record the issuing OAuth provider on each native UserOAuth credential.

``user_oauth.provider`` is the connecting app's key, not the provider that
issued the tokens: a custom app ``acme-mail`` backed by Google stores
``provider="acme-mail"``. Revocation used to infer the issuer from the app's
current ``public_mcp_apps.provider_name``, which an admin can delete,
retarget (Google -> GitHub), or delete and re-create under the same id. The
issuer is a fact known when the callback writes the tokens, so it now lives
on the credential itself as ``user_oauth.oauth_provider`` (nullable
``String(50)``, an ``oauth_providers.provider_name`` such as ``google``).

Backfill rule -- an existing row gets an issuer only when it is certain:

* Composio-backed rows (``token_type = 'composio'``) stay NULL: their grant
  belongs to Composio's own OAuth app, never this deployment's client.
* For every other row, the candidate issuers of its key are collected:

  1. the key itself, when it is a configured provider
     (``oauth_providers.provider_name``) -- a bare provider login stores its
     grant under the provider name (``google``, ``github``);
  2. every non-NULL ``provider_name`` the catalog app with that id has ever
     had: the current ``public_mcp_apps`` row plus every ``before_values``
     and ``after_values`` snapshot in ``public_mcp_app_audits`` (creates,
     updates, deletes, re-creates).

  Exactly one candidate -> that provider. Zero candidates (no catalog row,
  no audit trail, not a provider name) or more than one (a retargeted app,
  an app re-created for another provider, an app id that collides with a
  different provider's name) -> NULL. A NULL ``provider_name`` in the
  history is not a candidate: an app without an OAuth provider cannot have
  driven a native OAuth callback, so it cannot have issued the row's tokens.

NULL means "issuer unknown" and is permanent for that row until it is
reconnected; provider-side revocation refuses to guess for it and counts it
as a possible reference of every provider instead.

Limits of the evidence: catalog edits made before
``20260715_add_public_mcp_app_audits`` left no audit row, and the backfill
assumes each row was connected through its app's configured provider (the
UI always does; a hand-crafted login for another provider is not
detectable after the fact).

Portable across PostgreSQL and SQLite: the history is read through Core
selects (JSON snapshots decoded by SQLAlchemy) and resolved in Python, then
applied with one ``UPDATE`` per resolved key that only touches rows whose
issuer is still NULL, so a re-run is a no-op.

Revision ID: 20261115_user_oauth_issuer
Revises: 20261114_oauth_revocation_fences
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any, Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20261115_user_oauth_issuer"
down_revision: Union[str, None] = "20261114_oauth_revocation_fences"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "user_oauth"
COLUMN = "oauth_provider"
# Mirrors models.user_oauth.COMPOSIO_TOKEN_TYPE; migrations never import
# application models, whose shape may move on after this revision.
COMPOSIO_TOKEN_TYPE = "composio"

_user_oauth = sa.table(
    TABLE,
    sa.column("provider", sa.String),
    sa.column("token_type", sa.String),
    sa.column(COLUMN, sa.String),
)
_oauth_providers = sa.table("oauth_providers", sa.column("provider_name", sa.String))
_public_mcp_apps = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
)
_public_mcp_app_audits = sa.table(
    "public_mcp_app_audits",
    sa.column("app_id", sa.String),
    sa.column("before_values", sa.JSON),
    sa.column("after_values", sa.JSON),
)


def _snapshot_provider_name(snapshot: object) -> str | None:
    """The ``provider_name`` one audit snapshot recorded, if any."""
    if not isinstance(snapshot, Mapping):
        return None
    value = snapshot.get("provider_name")
    return value if isinstance(value, str) and value else None


def _catalog_provider_history(
    bind: sa.engine.Connection, tables: set[str]
) -> dict[str, set[str]]:
    """Every non-NULL ``provider_name`` each catalog app id has ever had."""
    history: defaultdict[str, set[str]] = defaultdict(set)
    if "public_mcp_apps" in tables:
        for app_id, provider_name in bind.execute(
            sa.select(_public_mcp_apps.c.app_id, _public_mcp_apps.c.provider_name)
        ):
            if provider_name:
                history[str(app_id)].add(str(provider_name))
    if "public_mcp_app_audits" in tables:
        for app_id, before, after in bind.execute(
            sa.select(
                _public_mcp_app_audits.c.app_id,
                _public_mcp_app_audits.c.before_values,
                _public_mcp_app_audits.c.after_values,
            )
        ):
            for snapshot in (before, after):
                if provider_name := _snapshot_provider_name(snapshot):
                    history[str(app_id)].add(provider_name)
    return history


def resolve_issuers(
    keys: Iterable[str],
    *,
    provider_names: set[str],
    app_history: Mapping[str, set[str]],
) -> dict[str, str]:
    """Map each ``user_oauth.provider`` key whose issuer is certain to it.

    Pure so the rule can be tested without a database: a key resolves only
    when its candidates -- itself if it is a configured provider name, plus
    every provider its catalog app has ever named -- are exactly one.
    """
    resolved: dict[str, str] = {}
    for key in keys:
        candidates = set(app_history.get(key, ()))
        if key in provider_names:
            candidates.add(key)
        if len(candidates) == 1:
            resolved[key] = next(iter(candidates))
    return resolved


def _backfill(bind: sa.engine.Connection, tables: set[str]) -> None:
    native = sa.or_(
        _user_oauth.c.token_type.is_(None),
        _user_oauth.c.token_type != COMPOSIO_TOKEN_TYPE,
    )
    keys = [
        str(key)
        for (key,) in bind.execute(
            sa.select(_user_oauth.c.provider)
            .where(_user_oauth.c.oauth_provider.is_(None), native)
            .distinct()
        )
    ]
    if not keys:
        return
    provider_names: set[str] = set()
    if "oauth_providers" in tables:
        provider_names = {
            str(name)
            for (name,) in bind.execute(sa.select(_oauth_providers.c.provider_name))
        }
    issuers = resolve_issuers(
        keys,
        provider_names=provider_names,
        app_history=_catalog_provider_history(bind, tables),
    )
    for key, issuer in issuers.items():
        bind.execute(
            sa.update(_user_oauth)
            .where(
                _user_oauth.c.provider == key,
                _user_oauth.c.oauth_provider.is_(None),
                native,
            )
            .values({COLUMN: issuer})
        )


def _columns(inspector: Any) -> set[str]:
    return {column["name"] for column in inspector.get_columns(TABLE)}


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    # A fresh install creates user_oauth from the models after Alembic runs,
    # already with this column and no rows to backfill.
    if TABLE not in tables:
        return
    if COLUMN not in _columns(inspector):
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(50), nullable=True))
    _backfill(bind, tables)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE in inspector.get_table_names() and COLUMN in _columns(inspector):
        op.drop_column(TABLE, COLUMN)
