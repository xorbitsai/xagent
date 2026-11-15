"""Record the issuing OAuth provider on each native UserOAuth credential.

``user_oauth.provider`` is the connecting app's key, not the provider that
issued the tokens: a custom app ``acme-mail`` backed by Google stores
``provider="acme-mail"``. Revocation used to infer the issuer from the app's
current ``public_mcp_apps.provider_name``, which an admin can delete,
retarget (Google -> GitHub), or delete and re-create under the same id. The
issuer is a fact known when the callback writes the tokens, so it now lives
on the credential itself as ``user_oauth.oauth_provider`` (nullable
``String(50)``, an ``oauth_providers.provider_name`` such as ``google``).

Backfill rule -- an existing row gets an issuer only from durable evidence,
never from a catalog history that may be missing part of the app's life.
Catalog auditing (``public_mcp_app_audits``, actions ``create`` / ``update``
/ ``delete``) only began with ``20260715_add_public_mcp_app_audits``; an
app's earlier provider changes left no trace, so "one provider observed" is
not proof on its own. The evidence accepted, first match wins:

1. **Shared Google grant.** A row that a ``google_oauth_grants`` row owns
   was written by the shared Google callback writer -> ``google``.
2. **Built-in app id** (``BUILTIN_APP_PROVIDERS``, frozen from the code
   registry at this revision) -> its canonical provider, when every provider
   the current catalog row and every audit snapshot of that id name is that
   canonical one. A built-in's provider has been normalized and is protected
   from admin edits (and the app from deletion and id reuse) since auditing
   began, so it can never be pointed elsewhere again. A row from a pre-audit
   detour through another provider therefore holds a token the app can never
   use: mislabelling it cannot cost a live connection.
3. **Custom app with complete audited history** -> its provider, when the
   app id's earliest audit row is its ``create``, the credential was created
   at or after that ``create`` (an older row may belong to a pre-audit
   incarnation of the same id), and every provider named from then on --
   all audit snapshots plus the current catalog row -- is one single value.

Everything else stays NULL, notably:

* a custom app that existed before auditing began (no ``create`` audit
  first), whatever single provider is visible today -- a pre-audit retarget
  would be invisible;
* a custom app whose history names more than one provider (retargeted, or
  re-created for another provider);
* a bare provider key that is not a built-in app id and has no shared-grant
  link (for example a bare ``google`` login): an app whose id equals a
  provider name may have written that key before auditing began;
* rows of a table without ``created_at`` under the custom-app rule;
* Composio-backed rows (``token_type = 'composio'``): their grant belongs to
  Composio's own OAuth app, never this deployment's client. A table that
  predates ``token_type`` holds no Composio rows, so all of its rows are
  treated as native.

A NULL ``provider_name`` in the history is not a candidate: an app without
an OAuth provider cannot have driven a native OAuth callback.

NULL means "issuer unknown" and lasts until the row is reconnected:
provider-side revocation refuses to guess for it and counts it as a
possible reference of every provider, so it can block a revoke but never be
cut off by one.

Remaining assumption: each row was connected through its app's configured
provider at the time (the UI always does; a hand-crafted login for another
provider is not detectable after the fact).

Portable across PostgreSQL and SQLite: the evidence is read through Core
selects (JSON snapshots decoded by SQLAlchemy) and resolved in Python, then
applied with ``UPDATE``s that only touch rows whose issuer is still NULL, so
a re-run is a no-op.

Revision ID: 20261115_user_oauth_issuer
Revises: 20261114_oauth_revocation_fences
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
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
GOOGLE = "google"
# Every built-in catalog app with an OAuth provider, as
# builtin_mcp_registry.get_builtin_public_mcp_app_rows() defines it at this
# revision. Frozen here so the backfill does not change with later code.
BUILTIN_APP_PROVIDERS: dict[str, str] = {
    "deputy": "deputy",
    "employment-hero": "employment-hero",
    "excel": "microsoft",
    "facebook": "meta",
    "github": "github",
    "gmail": GOOGLE,
    "google-ads": GOOGLE,
    "google-analytics": GOOGLE,
    "google-calendar": GOOGLE,
    "google-docs": GOOGLE,
    "google-drive": GOOGLE,
    "google-search-console": GOOGLE,
    "google-sheets": GOOGLE,
    "google-slides": GOOGLE,
    "instagram": "meta",
    "intercom": "intercom",
    "jira": "jira",
    "linear": "linear",
    "linkedin": "linkedin",
    "meta-ads": "meta",
    "myob": "myob",
    "onedrive": "microsoft",
    "outlook": "microsoft",
    "planner": "microsoft",
    "powerpoint": "microsoft",
    "salesforce": "salesforce",
    "sharepoint": "microsoft",
    "slack": "slack",
    "teams": "microsoft",
    "whatsapp": "meta",
    "word": "microsoft",
    "xero": "xero",
    "zendesk": "zendesk",
    "zoom": "zoom",
}
_UPDATE_CHUNK = 500

_user_oauth = sa.table(
    TABLE,
    sa.column("id", sa.Integer),
    sa.column("provider", sa.String),
    sa.column("token_type", sa.String),
    sa.column("created_at", sa.DateTime(timezone=True)),
    sa.column(COLUMN, sa.String),
)
_google_oauth_grants = sa.table(
    "google_oauth_grants", sa.column("oauth_account_id", sa.Integer)
)
_public_mcp_apps = sa.table(
    "public_mcp_apps",
    sa.column("app_id", sa.String),
    sa.column("provider_name", sa.String),
)
_public_mcp_app_audits = sa.table(
    "public_mcp_app_audits",
    sa.column("id", sa.Integer),
    sa.column("action", sa.String),
    sa.column("app_id", sa.String),
    sa.column("before_values", sa.JSON),
    sa.column("after_values", sa.JSON),
    sa.column("created_at", sa.DateTime(timezone=True)),
)


@dataclass
class AppHistory:
    """What the catalog proves about one app id.

    ``providers`` is every non-NULL provider the current row and the audit
    snapshots name. ``audited_since`` is when the id's audit trail began with
    its ``create`` -- the point from which its history is complete -- or
    ``None`` when the trail starts later in the app's life (or not at all).
    """

    providers: set[str] = field(default_factory=set)
    audited_since: datetime | None = None
    first_action: str | None = None


def _snapshot_provider_name(snapshot: object) -> str | None:
    """The ``provider_name`` one audit snapshot recorded, if any."""
    if not isinstance(snapshot, Mapping):
        return None
    value = snapshot.get("provider_name")
    return value if isinstance(value, str) and value else None


def _catalog_history(
    bind: sa.engine.Connection, tables: set[str]
) -> dict[str, AppHistory]:
    histories: dict[str, AppHistory] = {}
    if "public_mcp_app_audits" in tables:
        audits = bind.execute(
            sa.select(
                _public_mcp_app_audits.c.app_id,
                _public_mcp_app_audits.c.action,
                _public_mcp_app_audits.c.created_at,
                _public_mcp_app_audits.c.before_values,
                _public_mcp_app_audits.c.after_values,
            ).order_by(_public_mcp_app_audits.c.id)
        )
        for app_id, action, created_at, before, after in audits:
            history = histories.setdefault(str(app_id), AppHistory())
            if history.first_action is None:
                history.first_action = str(action)
                if action == "create":
                    history.audited_since = created_at
            history.providers.update(
                name
                for name in map(_snapshot_provider_name, (before, after))
                if name is not None
            )
    if "public_mcp_apps" in tables:
        for app_id, provider_name in bind.execute(
            sa.select(_public_mcp_apps.c.app_id, _public_mcp_apps.c.provider_name)
        ):
            history = histories.setdefault(str(app_id), AppHistory())
            if provider_name:
                history.providers.add(str(provider_name))
    return histories


def resolve_issuer(
    key: str,
    created_at: datetime | None,
    *,
    history: AppHistory | None,
) -> str | None:
    """The issuer of one non-Composio row without a shared-grant link, or
    ``None`` when the evidence does not prove it (rules 2 and 3 of the
    module docstring). Pure so the rule can be tested without a database."""
    observed = history.providers if history is not None else set()
    canonical = BUILTIN_APP_PROVIDERS.get(key)
    if canonical is not None:
        return canonical if observed <= {canonical} else None
    if (
        history is None
        or history.audited_since is None
        or created_at is None
        or created_at < history.audited_since
        or len(observed) != 1
    ):
        return None
    return next(iter(observed))


def _assign(bind: sa.engine.Connection, issuer: str, ids: list[int]) -> None:
    for start in range(0, len(ids), _UPDATE_CHUNK):
        bind.execute(
            sa.update(_user_oauth)
            .where(
                _user_oauth.c.id.in_(ids[start : start + _UPDATE_CHUNK]),
                _user_oauth.c.oauth_provider.is_(None),
            )
            .values({COLUMN: issuer})
        )


def _backfill(bind: sa.engine.Connection, tables: set[str], columns: set[str]) -> None:
    # A user_oauth that predates ``token_type`` (an old schema being carried
    # through the chain) cannot hold Composio rows, so every row is native.
    native = (
        sa.or_(
            _user_oauth.c.token_type.is_(None),
            _user_oauth.c.token_type != COMPOSIO_TOKEN_TYPE,
        )
        if "token_type" in columns
        else sa.true()
    )
    unknown = sa.and_(_user_oauth.c.oauth_provider.is_(None), native)

    if "google_oauth_grants" in tables:
        bind.execute(
            sa.update(_user_oauth)
            .where(
                unknown,
                _user_oauth.c.id.in_(
                    sa.select(_google_oauth_grants.c.oauth_account_id)
                ),
            )
            .values({COLUMN: GOOGLE})
        )

    created_at = _user_oauth.c.created_at if "created_at" in columns else sa.null()
    rows = bind.execute(
        sa.select(_user_oauth.c.id, _user_oauth.c.provider, created_at).where(unknown)
    ).all()
    if not rows:
        return
    histories = _catalog_history(bind, tables)
    by_issuer: dict[str, list[int]] = {}
    for row_id, key, row_created_at in rows:
        issuer = resolve_issuer(
            str(key), row_created_at, history=histories.get(str(key))
        )
        if issuer is not None:
            by_issuer.setdefault(issuer, []).append(int(row_id))
    for issuer, ids in by_issuer.items():
        _assign(bind, issuer, ids)


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
    columns = _columns(inspector)
    if COLUMN not in columns:
        op.add_column(TABLE, sa.Column(COLUMN, sa.String(50), nullable=True))
    _backfill(bind, tables, columns)


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if TABLE in inspector.get_table_names() and COLUMN in _columns(inspector):
        op.drop_column(TABLE, COLUMN)
