"""Safe storage and runtime resolution for shared Google OAuth grants."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from typing import cast

from sqlalchemy.exc import IntegrityError, InvalidRequestError
from sqlalchemy.orm import Session
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


def _normalized_nonempty(value: object, *, field: str) -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


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
    query = (
        db.query(GoogleOAuthGrant, UserOAuth)
        .join(
            GoogleOAuthAppAssociation,
            GoogleOAuthAppAssociation.grant_id == GoogleOAuthGrant.id,
        )
        .join(UserOAuth, UserOAuth.id == GoogleOAuthGrant.oauth_account_id)
        .filter(
            GoogleOAuthGrant.user_id == int(user_id),
            GoogleOAuthGrant.oauth_client_fingerprint == client_fingerprint,
            GoogleOAuthAppAssociation.app_id == normalized_app_id,
            (
                GoogleOAuthGrant.resource_owner_key.is_(None)
                if owner_key is None
                else GoogleOAuthGrant.resource_owner_key == owner_key
            ),
            UserOAuth.user_id == int(user_id),
            user_oauth_owner_clause(owner_key),
            UserOAuth.provider == GOOGLE_SHARED_OAUTH_PROVIDER,
            UserOAuth.provider_user_id == GoogleOAuthGrant.google_subject,
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
