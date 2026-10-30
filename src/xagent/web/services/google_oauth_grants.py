"""Safe storage and runtime resolution for shared Google OAuth grants."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

from sqlalchemy import and_, or_
from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.orm import Session, aliased
from sqlalchemy.orm.exc import ObjectDeletedError

from ..models.google_oauth_grant import (
    GoogleOAuthAppAssociation,
    GoogleOAuthGrant,
)
from ..models.user_oauth import (
    UserOAuth,
    is_undecryptable_oauth_token,
)
from .user_oauth import (
    normalize_user_oauth_resource_owner_key,
    user_oauth_owner_clause,
)

GOOGLE_SHARED_OAUTH_PROVIDER = "google"


class GoogleOAuthGrantConflict(ValueError):
    """The requested association would cross a grant isolation boundary."""


class GoogleOAuthRefreshTokenRequired(ValueError):
    """Google must be asked for consent before this identity can be stored."""


class GoogleOAuthScopeConsentRequired(ValueError):
    """The user declined one or more scopes required by the connector."""

    def __init__(self, missing_scopes: Iterable[str]):
        self.missing_scopes = tuple(sorted(set(missing_scopes)))
        super().__init__("Google did not grant every requested shared scope")


@dataclass(frozen=True)
class GoogleOAuthAuthorizationContext:
    """Existing authority that may be reused by one Google authorization."""

    has_refresh_token: bool
    associated_scopes: frozenset[str]


def google_oauth_client_fingerprint(client_id: str) -> str:
    """Return a stable, non-secret lookup key for one Google OAuth client."""
    normalized = str(client_id or "").strip()
    if not normalized:
        raise ValueError("Google OAuth client id must not be blank")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalize_google_oauth_scopes(scopes: str | Iterable[str] | None) -> frozenset[str]:
    """Normalize Google's space-delimited scopes into an exact set."""
    if scopes is None:
        return frozenset()
    values = [scopes] if isinstance(scopes, str) else scopes
    normalized: set[str] = set()
    for value in values:
        normalized.update(part for part in re.split(r"[\s,]+", str(value)) if part)
    return frozenset(normalized)


def google_oauth_grant_covers_scopes(
    oauth_account: UserOAuth, required_scopes: str | Iterable[str] | None
) -> bool:
    """Return whether the stored grant proves it covers every required scope."""
    required = normalize_google_oauth_scopes(required_scopes)
    if not required:
        return True
    granted = normalize_google_oauth_scopes(oauth_account.scope)
    return bool(granted) and required.issubset(granted)


def google_oauth_authorization_context(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    oauth_client_id: str,
    app_id: str,
) -> GoogleOAuthAuthorizationContext:
    """Return reusable authority for one Google authorization request.

    Stored scope strings are deliberately not trusted: rollback mirrors can
    retain scopes for detached apps. Instead, rebuild the union from current,
    valid associations and the canonical app catalog. If account selection is
    ambiguous, only an existing association for the requested app selects a
    grant; otherwise the current app remains the sole requested authority.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    client_fingerprint = google_oauth_client_fingerprint(oauth_client_id)
    _normalized_nonempty(app_id, field="Google app id")
    shared_rows = (
        db.query(GoogleOAuthGrant, UserOAuth)
        .join(UserOAuth, UserOAuth.id == GoogleOAuthGrant.oauth_account_id)
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.oauth_client_fingerprint == client_fingerprint,
            _grant_owner_clause(owner_key),
            UserOAuth.user_id == int(user_id),
            user_oauth_owner_clause(owner_key),
            UserOAuth.provider == GOOGLE_SHARED_OAUTH_PROVIDER,
            UserOAuth.provider_user_id == GoogleOAuthGrant.google_subject,
            UserOAuth.access_token != "",
        )
        .all()
    )
    has_refresh_token = False
    valid_apps_by_grant: dict[int, set[str]] = {}
    for grant, account in shared_rows:
        refresh_token = account.refresh_token
        if refresh_token and not is_undecryptable_oauth_token(refresh_token):
            has_refresh_token = True
        grant_id = int(grant.id)
        valid_apps_by_grant[grant_id] = {
            str(legacy_account.provider)
            for legacy_account in legacy_accounts_for_shared_google_oauth(
                db, oauth_account_id=int(account.id)
            )
        }

    app_grant_ids = [
        grant_id
        for grant_id, app_ids in valid_apps_by_grant.items()
        if app_id in app_ids
    ]
    target_grant_id: int | None = None
    if len(app_grant_ids) == 1:
        target_grant_id = app_grant_ids[0]
    elif len(shared_rows) == 1:
        target_grant_id = int(shared_rows[0][0].id)

    associated_scopes: set[str] = set()
    if target_grant_id is not None:
        from ..mcp_apps import get_app_by_id

        for associated_app_id in valid_apps_by_grant.get(target_grant_id, set()):
            app = get_app_by_id(db, associated_app_id)
            if app is None or str(app.get("provider") or "").lower() != "google":
                continue
            app_scopes = _strict_google_app_scopes(app)
            if app_scopes is not None:
                associated_scopes.update(app_scopes)

    return GoogleOAuthAuthorizationContext(
        has_refresh_token=has_refresh_token,
        associated_scopes=frozenset(associated_scopes),
    )


def _normalized_nonempty(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


def _grant_owner_clause(owner_key: str | None) -> Any:
    return (
        GoogleOAuthGrant.resource_owner_key.is_(None)
        if owner_key is None
        else GoogleOAuthGrant.resource_owner_key == owner_key
    )


def _validate_shared_account(
    oauth_account: UserOAuth,
    *,
    user_id: int,
    resource_owner_key: str | None,
    google_subject: str,
) -> None:
    if oauth_account.id is None:
        raise ValueError("shared Google OAuth account must be persisted")
    if int(oauth_account.user_id) != int(user_id):
        raise GoogleOAuthGrantConflict("shared Google grant user does not match")
    if oauth_account.resource_owner_key != resource_owner_key:
        raise GoogleOAuthGrantConflict("shared Google grant owner does not match")
    if oauth_account.provider != GOOGLE_SHARED_OAUTH_PROVIDER:
        raise GoogleOAuthGrantConflict(
            "shared Google credential must use provider='google'"
        )
    if str(oauth_account.provider_user_id or "") != google_subject:
        raise GoogleOAuthGrantConflict("shared Google grant subject does not match")


def _strict_google_app_scopes(app: Mapping[str, Any]) -> frozenset[str] | None:
    raw_scopes = app.get("oauth_scopes")
    if raw_scopes is None:
        return None
    if not isinstance(raw_scopes, (list, tuple, set)) or any(
        not isinstance(scope, str) or not scope.strip() for scope in raw_scopes
    ):
        return None
    return normalize_google_oauth_scopes(raw_scopes)


def _connected_google_apps_covered_by_bare_account(
    db: Session, *, user_id: int, oauth_account: UserOAuth
) -> tuple[tuple[str, frozenset[str]], ...]:
    """Return personal Google apps whose existing bare authority was valid."""
    if oauth_account.resource_owner_key is not None:
        return ()

    from ..mcp_apps import get_app_for_mcp_server
    from ..models.mcp import MCPServer, UserMCPServer

    granted_scopes = normalize_google_oauth_scopes(oauth_account.scope)
    apps_by_id: dict[str, frozenset[str]] = {}
    servers = (
        db.query(MCPServer)
        .join(UserMCPServer, UserMCPServer.mcpserver_id == MCPServer.id)
        .filter(
            UserMCPServer.user_id == int(user_id),
            UserMCPServer.is_active.is_(True),
        )
        .all()
    )
    for server in servers:
        app = get_app_for_mcp_server(db, server)
        if app is None or str(app.get("provider") or "").lower() != "google":
            continue
        app_id = app.get("id")
        required_scopes = _strict_google_app_scopes(app)
        if (
            isinstance(app_id, str)
            and app_id
            and required_scopes is not None
            and required_scopes.issubset(granted_scopes)
        ):
            apps_by_id[app_id] = required_scopes
    return tuple(sorted(apps_by_id.items()))


def _copy_shared_google_credentials(
    source: UserOAuth, target: UserOAuth, *, preserve_missing_refresh: bool
) -> None:
    for field in (
        "access_token",
        "refresh_token",
        "expires_at",
        "token_type",
        "scope",
        "email",
        "instance_url",
    ):
        value = getattr(source, field)
        if (
            field == "refresh_token"
            and preserve_missing_refresh
            and (not value or is_undecryptable_oauth_token(value))
        ):
            continue
        setattr(target, field, value)


def create_google_oauth_grant_association(
    db: Session,
    *,
    oauth_account: UserOAuth,
    oauth_client_id: str,
    app_id: str,
    legacy_oauth_account: UserOAuth | None = None,
) -> GoogleOAuthAppAssociation:
    """Create one explicit app link without committing the caller's transaction.

    Existing links are returned idempotently. Any attempt to reuse an identity
    with a different shared credential, or to attach a legacy row from another
    owner/Google identity, fails closed.
    """
    user_id = int(oauth_account.user_id)
    owner_key = normalize_user_oauth_resource_owner_key(
        oauth_account.resource_owner_key
    )
    subject = _normalized_nonempty(
        oauth_account.provider_user_id, field="Google subject"
    )
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    client_fingerprint = google_oauth_client_fingerprint(oauth_client_id)
    _validate_shared_account(
        oauth_account,
        user_id=user_id,
        resource_owner_key=owner_key,
        google_subject=subject,
    )

    # Serialize association creation on the credential row. Two apps can be
    # connected concurrently, so locking only an already-existing grant row
    # would leave the first insert (where no row exists yet) racy.
    account_query = db.query(UserOAuth).filter(UserOAuth.id == int(oauth_account.id))
    if db.get_bind().dialect.name == "sqlite":
        account_query.update({UserOAuth.id: UserOAuth.id}, synchronize_session=False)
    else:
        account_query = account_query.with_for_update()
    locked_account = account_query.populate_existing().one()
    _validate_shared_account(
        locked_account,
        user_id=user_id,
        resource_owner_key=owner_key,
        google_subject=subject,
    )

    grant = (
        db.query(GoogleOAuthGrant)
        .filter(GoogleOAuthGrant.oauth_account_id == int(oauth_account.id))
        .with_for_update()
        .one_or_none()
    )
    if grant is not None and (
        int(grant.user_id) != user_id
        or grant.resource_owner_key != owner_key
        or grant.google_subject != subject
        or grant.oauth_client_fingerprint != client_fingerprint
    ):
        raise GoogleOAuthGrantConflict(
            "shared credential is already bound to another Google grant boundary"
        )

    identity_grant = (
        db.query(GoogleOAuthGrant)
        .filter(
            GoogleOAuthGrant.user_id == user_id,
            GoogleOAuthGrant.google_subject == subject,
            GoogleOAuthGrant.oauth_client_fingerprint == client_fingerprint,
            (
                GoogleOAuthGrant.resource_owner_key.is_(None)
                if owner_key is None
                else GoogleOAuthGrant.resource_owner_key == owner_key
            ),
        )
        .with_for_update()
        .one_or_none()
    )
    if grant is None:
        grant = identity_grant
    elif identity_grant is not None and int(identity_grant.id) != int(grant.id):
        raise GoogleOAuthGrantConflict(
            "Google identity/client resolves to conflicting shared grants"
        )

    if grant is None:
        grant = GoogleOAuthGrant(
            user_id=user_id,
            resource_owner_key=owner_key,
            google_subject=subject,
            oauth_client_fingerprint=client_fingerprint,
            oauth_account_id=int(oauth_account.id),
        )
        db.add(grant)
        db.flush()
    elif int(grant.oauth_account_id) != int(oauth_account.id):
        raise GoogleOAuthGrantConflict(
            "Google identity/client already belongs to a different shared credential"
        )

    legacy_account_id: int | None = None
    if legacy_oauth_account is not None:
        legacy_subject = str(legacy_oauth_account.provider_user_id or "")
        if (
            int(legacy_oauth_account.user_id) != user_id
            or legacy_oauth_account.resource_owner_key != owner_key
            or legacy_oauth_account.provider != normalized_app_id
            or not legacy_subject
            or legacy_subject != subject
        ):
            raise GoogleOAuthGrantConflict(
                "legacy Google credential crosses the shared grant boundary"
            )
        if legacy_oauth_account.id is None:
            raise ValueError("legacy Google OAuth account must be persisted")
        legacy_account_id = int(legacy_oauth_account.id)
        if legacy_account_id == int(oauth_account.id):
            raise GoogleOAuthGrantConflict(
                "shared Google credential cannot be its own legacy reference"
            )

    association = (
        db.query(GoogleOAuthAppAssociation)
        .filter(
            GoogleOAuthAppAssociation.grant_id == int(grant.id),
            GoogleOAuthAppAssociation.app_id == normalized_app_id,
        )
        .one_or_none()
    )
    if association is None:
        if legacy_account_id is not None:
            existing_legacy_association = (
                db.query(GoogleOAuthAppAssociation)
                .filter(
                    GoogleOAuthAppAssociation.legacy_oauth_account_id
                    == legacy_account_id
                )
                .one_or_none()
            )
            if existing_legacy_association is not None:
                raise GoogleOAuthGrantConflict(
                    "legacy Google credential is already associated with another app"
                )
        association = GoogleOAuthAppAssociation(
            grant_id=int(grant.id),
            app_id=normalized_app_id,
            legacy_oauth_account_id=legacy_account_id,
        )
        try:
            # Translate a concurrent winner of either association uniqueness
            # race without leaving the caller's transaction unusable.
            with db.begin_nested():
                db.add(association)
                db.flush()
        except IntegrityError as exc:
            raise GoogleOAuthGrantConflict(
                "Google app or legacy credential is already associated"
            ) from exc
    elif legacy_account_id is not None and association.legacy_oauth_account_id not in {
        None,
        legacy_account_id,
    }:
        raise GoogleOAuthGrantConflict(
            "Google app association already references another legacy credential"
        )
    elif legacy_account_id is not None:
        setattr(association, "legacy_oauth_account_id", legacy_account_id)
    return association


def _lock_google_oauth_accounts(db: Session, account_ids: Iterable[int]) -> None:
    """Lock shared credential rows in stable order before their grant rows."""
    normalized_ids = sorted({int(account_id) for account_id in account_ids})
    if not normalized_ids:
        return
    account_query = db.query(UserOAuth).filter(UserOAuth.id.in_(normalized_ids))
    if db.get_bind().dialect.name == "sqlite":
        account_query.update({UserOAuth.id: UserOAuth.id}, synchronize_session=False)
    else:
        account_query.order_by(UserOAuth.id).with_for_update().all()


def lock_google_shared_accounts_for_app_replacement(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    app_id: str,
) -> None:
    """Lock old shared rows before a callback replaces an app's legacy row.

    The generic callback owns legacy-row replacement, while the shared-grant
    service owns grant/association updates. Taking the shared account lock at
    the outer boundary keeps the end-to-end order account-first when a user
    switches Google subjects and prevents a runtime mirror (shared then legacy)
    from deadlocking with callback replacement (legacy then shared).
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    account_ids = [
        int(account_id)
        for (account_id,) in db.query(GoogleOAuthGrant.oauth_account_id)
        .join(
            GoogleOAuthAppAssociation,
            GoogleOAuthAppAssociation.grant_id == GoogleOAuthGrant.id,
        )
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            _grant_owner_clause(owner_key),
            GoogleOAuthAppAssociation.app_id == normalized_app_id,
        )
        .all()
    ]
    _lock_google_oauth_accounts(db, account_ids)


def _lock_google_identity_grants(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    google_subject: str,
) -> list[GoogleOAuthGrant]:
    """Lock an identity's accounts first, then return its locked grants."""
    identity_filter = (
        GoogleOAuthGrant.user_id == int(user_id),
        GoogleOAuthGrant.google_subject == google_subject,
        _grant_owner_clause(resource_owner_key),
    )
    account_ids = [
        int(account_id)
        for (account_id,) in db.query(GoogleOAuthGrant.oauth_account_id)
        .filter(*identity_filter)
        .all()
    ]
    _lock_google_oauth_accounts(db, account_ids)
    return cast(
        list[GoogleOAuthGrant],
        db.query(GoogleOAuthGrant)
        .filter(*identity_filter)
        .order_by(GoogleOAuthGrant.id)
        .with_for_update()
        .all(),
    )


def _remove_stale_legacy_associations(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    app_id: str,
) -> None:
    """Drop links orphaned when this app switched to another Google subject."""
    legacy_account = aliased(UserOAuth)
    stale_legacy_clause = or_(
        GoogleOAuthAppAssociation.legacy_oauth_account_id.is_(None),
        legacy_account.id.is_(None),
        legacy_account.user_id != GoogleOAuthGrant.user_id,
        legacy_account.provider != GoogleOAuthAppAssociation.app_id,
        legacy_account.provider_user_id.is_(None),
        legacy_account.provider_user_id != GoogleOAuthGrant.google_subject,
        and_(
            GoogleOAuthGrant.resource_owner_key.is_(None),
            legacy_account.resource_owner_key.is_not(None),
        ),
        and_(
            GoogleOAuthGrant.resource_owner_key.is_not(None),
            or_(
                legacy_account.resource_owner_key.is_(None),
                legacy_account.resource_owner_key
                != GoogleOAuthGrant.resource_owner_key,
            ),
        ),
    )
    association_filter = (
        GoogleOAuthGrant.user_id == int(user_id),
        _grant_owner_clause(resource_owner_key),
        GoogleOAuthAppAssociation.app_id == app_id,
        stale_legacy_clause,
    )
    account_ids = [
        int(account_id)
        for (account_id,) in db.query(GoogleOAuthGrant.oauth_account_id)
        .join(
            GoogleOAuthAppAssociation,
            GoogleOAuthAppAssociation.grant_id == GoogleOAuthGrant.id,
        )
        .outerjoin(
            legacy_account,
            legacy_account.id == GoogleOAuthAppAssociation.legacy_oauth_account_id,
        )
        .filter(*association_filter)
        .all()
    ]
    _lock_google_oauth_accounts(db, account_ids)
    associations = (
        db.query(GoogleOAuthAppAssociation)
        .join(
            GoogleOAuthGrant,
            GoogleOAuthGrant.id == GoogleOAuthAppAssociation.grant_id,
        )
        .outerjoin(
            legacy_account,
            legacy_account.id == GoogleOAuthAppAssociation.legacy_oauth_account_id,
        )
        .filter(*association_filter)
        .with_for_update(of=GoogleOAuthAppAssociation)
        .all()
    )
    grant_ids = {int(association.grant_id) for association in associations}
    for association in associations:
        db.delete(association)
    if associations:
        db.flush()
        _delete_orphaned_google_grants(db, grant_ids=grant_ids)


def write_google_shared_oauth_callback(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    oauth_client_id: str,
    app_id: str,
    google_subject: str,
    access_token: str,
    refresh_token: str | None,
    granted_scopes: str | Iterable[str] | None,
    requested_scopes: str | Iterable[str] | None,
    email: str | None,
    token_type: str,
    expires_at: datetime | None,
    legacy_oauth_account: UserOAuth,
) -> tuple[UserOAuth, UserOAuth, GoogleOAuthAppAssociation]:
    """Persist a shared callback while retaining one rollback-compatible row.

    Existing shared credentials are updated in place. The app-specific legacy
    row mirrors the same token material so disabling the rollout flag takes
    effect immediately and does not require another Google authorization.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    subject = _normalized_nonempty(google_subject, field="Google subject")
    client_fingerprint = google_oauth_client_fingerprint(oauth_client_id)

    if (
        int(legacy_oauth_account.user_id) != int(user_id)
        or legacy_oauth_account.resource_owner_key != owner_key
        or legacy_oauth_account.provider != normalized_app_id
    ):
        raise GoogleOAuthGrantConflict(
            "legacy Google credential crosses the callback boundary"
        )

    legacy_refresh_token = legacy_oauth_account.refresh_token
    _remove_stale_legacy_associations(
        db,
        user_id=int(user_id),
        resource_owner_key=owner_key,
        app_id=normalized_app_id,
    )
    identity_grants = _lock_google_identity_grants(
        db,
        user_id=int(user_id),
        resource_owner_key=owner_key,
        google_subject=subject,
    )
    exact_grants = [
        candidate
        for candidate in identity_grants
        if candidate.oauth_client_fingerprint == client_fingerprint
    ]
    if len(exact_grants) > 1:
        raise GoogleOAuthGrantConflict(
            "Google identity resolves to multiple current-client grants"
        )
    grant = exact_grants[0] if exact_grants else None
    client_fingerprint_migrated = False
    if grant is None:
        # A deployment may rotate its Google client id. The callback proves a
        # fresh authorization under the new client, so migrate the sole
        # identity row in place rather than replacing its credential row and
        # cascading every app association.
        if len(identity_grants) == 1:
            grant = identity_grants[0]
            setattr(grant, "oauth_client_fingerprint", client_fingerprint)
            client_fingerprint_migrated = True
        elif len(identity_grants) > 1:
            raise GoogleOAuthGrantConflict(
                "Google identity resolves to multiple client grants"
            )

    shared_account: UserOAuth
    adopted_bare_apps: tuple[tuple[str, frozenset[str]], ...] = ()
    if grant is None:
        # A pre-rollout bare Google login already occupies this provider and
        # subject uniqueness boundary. Adopt it in place so enabling shared
        # grants cannot turn a valid existing connection into a 500.
        existing_shared_account = (
            db.query(UserOAuth)
            .filter(
                UserOAuth.user_id == int(user_id),
                user_oauth_owner_clause(owner_key),
                UserOAuth.provider == GOOGLE_SHARED_OAUTH_PROVIDER,
                UserOAuth.provider_user_id == subject,
            )
            .with_for_update()
            .one_or_none()
        )
        if existing_shared_account is None:
            candidate = UserOAuth(
                user_id=int(user_id),
                provider=GOOGLE_SHARED_OAUTH_PROVIDER,
                resource_owner_key=owner_key,
                provider_user_id=subject,
                access_token=access_token,
            )
            try:
                with db.begin_nested():
                    db.add(candidate)
                    db.flush()
                shared_account = candidate
            except IntegrityError as exc:
                # A caller outside the HTTP callback may not hold the user-row
                # lock. Recover the concurrent winner without poisoning the
                # surrounding credential transaction.
                concurrent_winner = (
                    db.query(UserOAuth)
                    .filter(
                        UserOAuth.user_id == int(user_id),
                        user_oauth_owner_clause(owner_key),
                        UserOAuth.provider == GOOGLE_SHARED_OAUTH_PROVIDER,
                        UserOAuth.provider_user_id == subject,
                    )
                    .with_for_update()
                    .one_or_none()
                )
                if concurrent_winner is None:
                    raise GoogleOAuthGrantConflict(
                        "shared Google credential could not be created"
                    ) from exc
                shared_account = concurrent_winner
        else:
            shared_account = existing_shared_account
            adopted_bare_apps = _connected_google_apps_covered_by_bare_account(
                db, user_id=int(user_id), oauth_account=shared_account
            )
    else:
        shared_account = (
            db.query(UserOAuth)
            .filter(UserOAuth.id == int(grant.oauth_account_id))
            .with_for_update()
            .populate_existing()
            .one()
        )
        _validate_shared_account(
            shared_account,
            user_id=int(user_id),
            resource_owner_key=owner_key,
            google_subject=subject,
        )

    shared_refresh_token = shared_account.refresh_token
    resolved_refresh_token = refresh_token or (
        None
        if client_fingerprint_migrated
        else shared_refresh_token or legacy_refresh_token
    )
    if not resolved_refresh_token or is_undecryptable_oauth_token(
        resolved_refresh_token
    ):
        raise GoogleOAuthRefreshTokenRequired

    scopes = set(normalize_google_oauth_scopes(granted_scopes))
    requested = set(normalize_google_oauth_scopes(requested_scopes))
    if not requested.issubset(scopes):
        raise GoogleOAuthScopeConsentRequired(requested.difference(scopes))
    scope_value = " ".join(sorted(scopes))

    for account in (shared_account, legacy_oauth_account):
        setattr(account, "provider_user_id", subject)
        setattr(account, "access_token", access_token)
        setattr(account, "refresh_token", resolved_refresh_token)
        setattr(account, "token_type", token_type)
        setattr(account, "scope", scope_value)
        setattr(account, "email", email)
        setattr(account, "expires_at", expires_at)
    db.flush()

    association = create_google_oauth_grant_association(
        db,
        oauth_account=shared_account,
        oauth_client_id=oauth_client_id,
        app_id=normalized_app_id,
        legacy_oauth_account=legacy_oauth_account,
    )

    # Once a pre-rollout bare row becomes grant-owned it can no longer serve
    # as an implicit provider fallback. Preserve only connected sibling apps
    # that the old bare scope set had already authorized, and keep every new
    # association behind the same owner/subject/client boundary.
    for sibling_app_id, sibling_required_scopes in adopted_bare_apps:
        if sibling_app_id == normalized_app_id:
            continue
        sibling_rows = (
            db.query(UserOAuth)
            .filter(
                UserOAuth.user_id == int(user_id),
                user_oauth_owner_clause(owner_key),
                UserOAuth.provider == sibling_app_id,
            )
            .order_by(UserOAuth.id.desc())
            .all()
        )
        sibling_legacy = next(
            (row for row in sibling_rows if str(row.provider_user_id or "") == subject),
            None,
        )
        shared_covers_sibling = google_oauth_grant_covers_scopes(
            shared_account, sibling_required_scopes
        )
        if sibling_legacy is not None and not shared_covers_sibling:
            # This app already has an independent credential. Leave it intact
            # instead of replacing a working fallback with an under-scoped
            # shared grant.
            continue
        if sibling_legacy is None:
            # Do not make the adopted bare identity replace an app-specific
            # connection that already points at another Google account.
            if sibling_rows:
                continue
            sibling_legacy = UserOAuth(
                user_id=int(user_id),
                provider=sibling_app_id,
                resource_owner_key=owner_key,
                provider_user_id=subject,
                access_token=access_token,
            )
            db.add(sibling_legacy)
            db.flush()
        setattr(sibling_legacy, "provider_user_id", subject)
        if shared_covers_sibling:
            _copy_shared_google_credentials(
                shared_account, sibling_legacy, preserve_missing_refresh=False
            )
        else:
            # Consuming the bare row removes its implicit provider fallback.
            # Keep an explicit reconnect tombstone so the sibling is not
            # silently presented as never connected.
            setattr(sibling_legacy, "access_token", "")
            setattr(sibling_legacy, "refresh_token", None)
            setattr(sibling_legacy, "expires_at", None)
            setattr(sibling_legacy, "scope", " ".join(sorted(sibling_required_scopes)))
            setattr(sibling_legacy, "email", shared_account.email)
        create_google_oauth_grant_association(
            db,
            oauth_account=shared_account,
            oauth_client_id=oauth_client_id,
            app_id=sibling_app_id,
            legacy_oauth_account=sibling_legacy,
        )

    # A reconnect repairs the whole accepted shared authority immediately,
    # before a feature-flag rollback or a direct legacy consumer can run.
    # The helper skips catalog apps whose required scopes are not covered.
    mirror_shared_google_oauth_credentials(db, oauth_account=shared_account)
    return shared_account, legacy_oauth_account, association


def legacy_accounts_for_shared_google_oauth(
    db: Session, *, oauth_account_id: int
) -> list[UserOAuth]:
    """Return valid legacy mirrors linked to one shared credential."""
    return cast(
        list[UserOAuth],
        (
            db.query(UserOAuth)
            .join(
                GoogleOAuthAppAssociation,
                GoogleOAuthAppAssociation.legacy_oauth_account_id == UserOAuth.id,
            )
            .join(
                GoogleOAuthGrant,
                GoogleOAuthGrant.id == GoogleOAuthAppAssociation.grant_id,
            )
            .filter(
                GoogleOAuthGrant.oauth_account_id == int(oauth_account_id),
                UserOAuth.user_id == GoogleOAuthGrant.user_id,
                UserOAuth.provider == GoogleOAuthAppAssociation.app_id,
                UserOAuth.provider_user_id == GoogleOAuthGrant.google_subject,
                or_(
                    and_(
                        GoogleOAuthGrant.resource_owner_key.is_(None),
                        UserOAuth.resource_owner_key.is_(None),
                    ),
                    UserOAuth.resource_owner_key == GoogleOAuthGrant.resource_owner_key,
                ),
            )
            .populate_existing()
            .all()
        ),
    )


def gmail_legacy_accounts_for_shared_google_oauth(
    db: Session, *, oauth_account_id: int
) -> list[UserOAuth]:
    """Return ordinary Gmail rows whose watches depend on a shared grant."""
    return [
        account
        for account in legacy_accounts_for_shared_google_oauth(
            db, oauth_account_id=oauth_account_id
        )
        if account.provider == "gmail" and account.resource_owner_key is None
    ]


def scope_valid_gmail_accounts_for_shared_google_oauth(
    db: Session, *, oauth_account: UserOAuth
) -> list[UserOAuth]:
    """Return ordinary Gmail mirrors the current shared scopes can authorize."""
    return [
        account
        for account in _scope_valid_legacy_accounts_for_shared_google_oauth(
            db, oauth_account=oauth_account
        )
        if account.provider == "gmail" and account.resource_owner_key is None
    ]


def _scope_valid_legacy_accounts_for_shared_google_oauth(
    db: Session, *, oauth_account: UserOAuth
) -> list[UserOAuth]:
    """Resolve scope policy through the canonical catalog execution view."""
    if (
        oauth_account.id is None
        or oauth_account.provider != GOOGLE_SHARED_OAUTH_PROVIDER
    ):
        return []

    from ..mcp_apps import get_app_by_id

    scope_valid_accounts: list[UserOAuth] = []
    for legacy_account in legacy_accounts_for_shared_google_oauth(
        db, oauth_account_id=int(oauth_account.id)
    ):
        app = get_app_by_id(db, str(legacy_account.provider))
        if app is None or str(app.get("provider") or "").lower() != "google":
            continue
        required_scopes = _strict_google_app_scopes(app)
        if required_scopes is None or not google_oauth_grant_covers_scopes(
            oauth_account, required_scopes
        ):
            continue
        scope_valid_accounts.append(legacy_account)
    return scope_valid_accounts


def mirror_shared_google_oauth_credentials(
    db: Session,
    *,
    oauth_account: UserOAuth,
    skip_gmail_tombstones: bool = False,
) -> int:
    """Keep scope-valid rollback rows current after callback or refresh."""
    if oauth_account.id is None:
        return 0
    legacy_account_ids = [
        int(account.id)
        for account in legacy_accounts_for_shared_google_oauth(
            db, oauth_account_id=int(oauth_account.id)
        )
    ]
    # Runtime refresh and callback both hold the shared row before entering
    # this helper. Lock mirrors in stable order and force a database reload so
    # a Gmail mailbox transition committed during provider I/O cannot be
    # revived from the session identity map.
    _lock_google_oauth_accounts(db, legacy_account_ids)
    scope_valid_accounts = _scope_valid_legacy_accounts_for_shared_google_oauth(
        db, oauth_account=oauth_account
    )
    mirrored = 0
    mirrored_accounts: list[UserOAuth] = []
    for legacy_account in scope_valid_accounts:
        if (
            skip_gmail_tombstones
            and legacy_account.provider == "gmail"
            and not legacy_account.access_token
            and legacy_account.refresh_token is None
        ):
            continue
        _copy_shared_google_credentials(
            oauth_account, legacy_account, preserve_missing_refresh=True
        )
        mirrored += 1
        mirrored_accounts.append(legacy_account)
    if mirrored_accounts:
        # Callers may immediately run another populate_existing() query (for
        # example, Gmail post-callback provisioning). Persist the mirror inside
        # this transaction first so that re-read cannot discard pending values.
        db.flush(mirrored_accounts)
    return mirrored


def _lock_shared_google_accounts_for_detach(
    db: Session,
    *,
    user_id: int,
    association_filter: Any,
) -> None:
    """Take the callback-compatible account-first lock order before detach."""
    account_ids = sorted(
        {
            int(oauth_account_id)
            for (oauth_account_id,) in db.query(GoogleOAuthGrant.oauth_account_id)
            .join(
                GoogleOAuthAppAssociation,
                GoogleOAuthAppAssociation.grant_id == GoogleOAuthGrant.id,
            )
            .filter(
                GoogleOAuthGrant.user_id == int(user_id),
                GoogleOAuthGrant.resource_owner_key.is_(None),
                association_filter,
            )
            .all()
        }
    )
    _lock_google_oauth_accounts(db, account_ids)


def detach_google_shared_oauth_app(
    db: Session,
    *,
    user_id: int,
    app_id: str,
) -> int:
    """Detach an app and delete only shared credentials no sibling still uses."""
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    return _detach_google_shared_associations(
        db,
        user_id=int(user_id),
        association_filter=GoogleOAuthAppAssociation.app_id == normalized_app_id,
    )


def _detach_google_shared_associations(
    db: Session,
    *,
    user_id: int,
    association_filter: Any,
) -> int:
    """Detach matching ordinary associations under the shared lock order."""
    _lock_shared_google_accounts_for_detach(
        db,
        user_id=int(user_id),
        association_filter=association_filter,
    )
    associations = (
        db.query(GoogleOAuthAppAssociation)
        .join(
            GoogleOAuthGrant,
            GoogleOAuthGrant.id == GoogleOAuthAppAssociation.grant_id,
        )
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.resource_owner_key.is_(None),
            association_filter,
        )
        .with_for_update()
        .all()
    )
    grant_ids = {int(association.grant_id) for association in associations}
    for association in associations:
        db.delete(association)
    db.flush()

    _delete_orphaned_google_grants(db, grant_ids=grant_ids)
    return len(associations)


def detach_google_shared_oauth_legacy_account(
    db: Session,
    *,
    user_id: int,
    legacy_oauth_account_id: int,
) -> int:
    """Detach the exact visible legacy account selected by cloud account UI."""
    return _detach_google_shared_associations(
        db,
        user_id=int(user_id),
        association_filter=GoogleOAuthAppAssociation.legacy_oauth_account_id
        == int(legacy_oauth_account_id),
    )


def _delete_orphaned_google_grants(db: Session, *, grant_ids: Iterable[int]) -> None:
    """Delete shared credentials after their last explicit app link is gone."""

    for grant_id in grant_ids:
        sibling_exists = (
            db.query(GoogleOAuthAppAssociation.id)
            .filter(
                GoogleOAuthAppAssociation.grant_id == grant_id,
                GoogleOAuthAppAssociation.legacy_oauth_account_id.is_not(None),
            )
            .first()
            is not None
        )
        if sibling_exists:
            continue
        grant = (
            db.query(GoogleOAuthGrant)
            .filter(GoogleOAuthGrant.id == grant_id)
            .with_for_update()
            .one_or_none()
        )
        if grant is None:
            continue
        oauth_account = (
            db.query(UserOAuth)
            .filter(UserOAuth.id == int(grant.oauth_account_id))
            .one_or_none()
        )
        # Remove the boundary and credential together so token material cannot
        # become an unreferenced secret at rest.
        db.delete(grant)
        if oauth_account is not None:
            # Delete explicitly as well as relying on ON DELETE CASCADE so
            # SQLite/unit-test connections with foreign keys disabled cannot
            # retain an orphan boundary row.
            db.flush()
            db.delete(oauth_account)


def resolve_google_shared_oauth_account(
    db: Session,
    *,
    user_id: int,
    resource_owner_key: str | None,
    oauth_client_id: str,
    app_id: str,
    required_scopes: str | Iterable[str] | None = None,
    lock: bool = False,
) -> UserOAuth | None:
    """Resolve the newest association inside one exact owner namespace.

    The newest association is authoritative. An under-scoped, tombstoned, or
    unreadable newest credential must fall back to the app's legacy row rather
    than silently selecting an older Google identity.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    client_fingerprint = google_oauth_client_fingerprint(oauth_client_id)
    legacy_account = aliased(UserOAuth)
    query = (
        db.query(GoogleOAuthGrant, UserOAuth)
        .join(
            GoogleOAuthAppAssociation,
            GoogleOAuthAppAssociation.grant_id == GoogleOAuthGrant.id,
        )
        .join(UserOAuth, UserOAuth.id == GoogleOAuthGrant.oauth_account_id)
        .join(
            legacy_account,
            legacy_account.id == GoogleOAuthAppAssociation.legacy_oauth_account_id,
        )
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.oauth_client_fingerprint == client_fingerprint,
            GoogleOAuthAppAssociation.app_id == normalized_app_id,
            _grant_owner_clause(owner_key),
            UserOAuth.user_id == int(user_id),
            user_oauth_owner_clause(owner_key),
            UserOAuth.provider == GOOGLE_SHARED_OAUTH_PROVIDER,
            UserOAuth.provider_user_id == GoogleOAuthGrant.google_subject,
            legacy_account.user_id == int(user_id),
            (
                legacy_account.resource_owner_key.is_(None)
                if owner_key is None
                else legacy_account.resource_owner_key == owner_key
            ),
            legacy_account.provider == normalized_app_id,
            legacy_account.provider_user_id == GoogleOAuthGrant.google_subject,
        )
        .order_by(GoogleOAuthAppAssociation.id.desc())
    )
    candidate = query.first()
    if candidate is None:
        return None
    _grant, oauth_account = candidate
    if (
        not oauth_account.access_token
        or is_undecryptable_oauth_token(oauth_account.access_token)
        or not google_oauth_grant_covers_scopes(oauth_account, required_scopes)
    ):
        return None
    if not lock:
        return cast(UserOAuth, oauth_account)

    account_id = int(oauth_account.id)
    try:
        if db.get_bind().dialect.name == "sqlite":
            updated = (
                db.query(UserOAuth)
                .filter(UserOAuth.id == account_id)
                .update({UserOAuth.id: UserOAuth.id}, synchronize_session=False)
            )
            if updated != 1:
                db.rollback()
                return None
            db.expire_all()
            locked_candidate = query.filter(UserOAuth.id == account_id).first()
        else:
            locked_candidate = (
                query.filter(UserOAuth.id == account_id)
                .with_for_update(of=UserOAuth)
                .first()
            )
    except (InvalidRequestError, ObjectDeletedError):
        db.rollback()
        return None
    if locked_candidate is None:
        return None
    _locked_grant, locked_account = locked_candidate
    if (
        not locked_account.access_token
        or is_undecryptable_oauth_token(locked_account.access_token)
        or not google_oauth_grant_covers_scopes(locked_account, required_scopes)
    ):
        return None
    return cast(UserOAuth, locked_account)
