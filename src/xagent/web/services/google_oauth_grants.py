"""Safe storage and runtime resolution for shared Google OAuth grants."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
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

    scopes: frozenset[str]
    has_refresh_token: bool


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
    """Return reusable scopes/token state inside the exact owner/client boundary.

    The app's legacy row participates only as a rollout bridge. It lets the
    first shared callback preserve an already-issued refresh token when Google
    omits one, without treating another app's legacy credential as shared.
    """
    owner_key = normalize_user_oauth_resource_owner_key(resource_owner_key)
    client_fingerprint = google_oauth_client_fingerprint(oauth_client_id)
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    shared_accounts = (
        db.query(UserOAuth)
        .join(GoogleOAuthGrant, GoogleOAuthGrant.oauth_account_id == UserOAuth.id)
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
    legacy_accounts = (
        db.query(UserOAuth)
        .filter(
            UserOAuth.user_id == int(user_id),
            user_oauth_owner_clause(owner_key),
            UserOAuth.provider == normalized_app_id,
        )
        .all()
    )
    scopes: set[str] = set()
    for account in [*shared_accounts, *legacy_accounts]:
        scopes.update(normalize_google_oauth_scopes(account.scope))
    has_refresh_token = False
    for account in shared_accounts:
        refresh_token = account.refresh_token
        if refresh_token and not is_undecryptable_oauth_token(refresh_token):
            has_refresh_token = True
    return GoogleOAuthAuthorizationContext(
        scopes=frozenset(scopes), has_refresh_token=has_refresh_token
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
    exact_grant = (
        db.query(GoogleOAuthGrant)
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.google_subject == subject,
            GoogleOAuthGrant.oauth_client_fingerprint == client_fingerprint,
            _grant_owner_clause(owner_key),
        )
        .with_for_update()
        .one_or_none()
    )
    grant = exact_grant
    if grant is None:
        # A deployment may rotate its Google client id. The callback proves a
        # fresh authorization under the new client, so migrate the sole
        # identity row in place rather than replacing its credential row and
        # cascading every app association.
        identity_grants = (
            db.query(GoogleOAuthGrant)
            .filter(
                GoogleOAuthGrant.user_id == int(user_id),
                GoogleOAuthGrant.google_subject == subject,
                _grant_owner_clause(owner_key),
            )
            .with_for_update()
            .all()
        )
        if len(identity_grants) == 1:
            grant = identity_grants[0]
            setattr(grant, "oauth_client_fingerprint", client_fingerprint)
        elif len(identity_grants) > 1:
            raise GoogleOAuthGrantConflict(
                "Google identity resolves to multiple client grants"
            )

    shared_account: UserOAuth
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
    resolved_refresh_token = (
        refresh_token or shared_refresh_token or legacy_refresh_token
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


def mirror_shared_google_oauth_credentials(
    db: Session, *, oauth_account: UserOAuth
) -> int:
    """Keep rollback-compatible per-app rows current after token refresh."""
    if (
        oauth_account.id is None
        or oauth_account.provider != GOOGLE_SHARED_OAUTH_PROVIDER
    ):
        return 0
    legacy_accounts = legacy_accounts_for_shared_google_oauth(
        db, oauth_account_id=int(oauth_account.id)
    )
    for legacy_account in legacy_accounts:
        for field in (
            "access_token",
            "refresh_token",
            "expires_at",
            "token_type",
            "scope",
            "email",
            "instance_url",
        ):
            value = getattr(oauth_account, field)
            if field == "refresh_token" and (
                not value or is_undecryptable_oauth_token(value)
            ):
                # A damaged shared row must not erase the legacy recovery
                # token that exists specifically to make rollback possible.
                continue
            setattr(legacy_account, field, value)
    return len(legacy_accounts)


def detach_google_shared_oauth_app(
    db: Session,
    *,
    user_id: int,
    app_id: str,
) -> int:
    """Detach an app and delete only shared credentials no sibling still uses."""
    normalized_app_id = _normalized_nonempty(app_id, field="Google app id")
    associations = (
        db.query(GoogleOAuthAppAssociation)
        .join(
            GoogleOAuthGrant,
            GoogleOAuthGrant.id == GoogleOAuthAppAssociation.grant_id,
        )
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.resource_owner_key.is_(None),
            GoogleOAuthAppAssociation.app_id == normalized_app_id,
        )
        .with_for_update()
        .all()
    )
    grant_ids = {int(association.grant_id) for association in associations}
    for association in associations:
        db.delete(association)
    db.flush()

    for grant_id in grant_ids:
        sibling_exists = (
            db.query(GoogleOAuthAppAssociation.id)
            .filter(GoogleOAuthAppAssociation.grant_id == grant_id)
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
    return len(associations)


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
