"""
Custom API Management API Endpoints

Provides REST API endpoints for managing Custom API configurations
in the web application.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ...core.tools.adapters.vibe.connector_runtime import (
    ConnectorRuntimeError,
    validate_runtime_config_declaration,
)
from ...core.utils.encryption import encrypt_value
from ..auth_dependencies import get_current_user
from ..models.custom_api import CustomApi, UserCustomApi
from ..models.database import get_db
from ..models.user import User

if TYPE_CHECKING:
    from ..services.connector_team_scope import ConnectorAccess
    from .mcp import _TeamOwnedUserApi

logger = logging.getLogger(__name__)


# Pydantic models for API
class CustomApiCreate(BaseModel):
    """Request model for creating a Custom API."""

    name: str = Field(..., min_length=1, max_length=100, description="API name")
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field("GET", description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    body: Optional[str] = Field(None, description="HTTP body (JSON template)")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    runtime_input_schema: Optional[Dict[str, Any]] = Field(
        None, description="Runtime input declarations"
    )
    runtime_bindings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Runtime binding declarations"
    )
    allow_delegated_authorization: bool = Field(
        False, description="Allow runtime Authorization header binding"
    )
    is_active: bool = Field(
        True, description="Whether this caller's own link row to the API is active"
    )


class CustomApiUpdate(BaseModel):
    """Request model for updating a Custom API."""

    name: Optional[str] = Field(
        None, min_length=1, max_length=100, description="API name"
    )
    description: Optional[str] = Field(None, description="API description")
    url: Optional[str] = Field(
        None, min_length=1, max_length=500, description="API URL"
    )
    method: Optional[str] = Field(None, description="HTTP method")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    body: Optional[str] = Field(None, description="HTTP body (JSON template)")
    env: Optional[Dict[str, str]] = Field(
        None, description="Environment variables (secrets)"
    )
    runtime_input_schema: Optional[Dict[str, Any]] = Field(
        None, description="Runtime input declarations"
    )
    runtime_bindings: Optional[List[Dict[str, Any]]] = Field(
        None, description="Runtime binding declarations"
    )
    allow_delegated_authorization: Optional[bool] = Field(
        None, description="Allow runtime Authorization header binding"
    )
    is_active: Optional[bool] = Field(
        None, description="Whether this caller's own link row to the API is active"
    )


class CustomApiResponse(BaseModel):
    """Response model for Custom API."""

    id: int
    user_id: int
    name: str
    description: Optional[str]
    url: Optional[str]
    method: Optional[str]
    headers: Optional[Dict[str, str]]
    body: Optional[str]
    env: Optional[Dict[str, str]]  # Will return masked values
    runtime_input_schema: Optional[Dict[str, Any]]
    runtime_bindings: Optional[List[Dict[str, Any]]]
    allow_delegated_authorization: bool
    is_active: bool = Field(
        ...,
        description=(
            "Whether this caller's own link row to the API is active. A "
            "caller who has no personal link row -- a team member reaching "
            "a connector their team links -- has no row to hold this, and "
            "receives the stand-in association's constant instead of a "
            "stored value."
        ),
    )
    is_default: bool = Field(
        ...,
        description=(
            "Whether this caller's own link row marks the API as their "
            "default. Carries the same caveat as ``is_active``: with no "
            "personal link row the value is the stand-in association's "
            "constant, not a stored one."
        ),
    )
    created_at: str
    updated_at: str

    class Config:
        from_attributes = True
        json_encoders = {datetime: lambda v: v.isoformat()}


# Create router
custom_api_router = APIRouter(prefix="/api/custom-apis", tags=["Custom API Management"])


def _db_api_to_response(
    api: CustomApi,
    user_api: "UserCustomApi | _TeamOwnedUserApi",
) -> CustomApiResponse:
    """Convert database CustomApi to response model with masked env values.

    ``is_active`` and ``is_default`` are read off ``user_api``, which is the
    caller's own link row when one exists and the ``_TeamOwnedUserApi``
    stand-in when the caller has no usable one. The stand-in holds class
    constants rather than stored columns, so for such a caller these two
    fields report those constants -- the same constants the aggregate
    connector list carries for that caller whenever it lists the connector
    at all, since it builds its response from the same stand-in.
    """

    # Mask env values for frontend
    masked_env = None
    if api.env and isinstance(api.env, dict):
        masked_env = {k: "********" for k in api.env.keys()}

    return CustomApiResponse(
        id=api.id,
        user_id=user_api.user_id,
        name=api.name,
        description=api.description,
        url=api.url,
        method=api.method,
        headers=api.headers,
        body=api.body,
        env=masked_env,
        runtime_input_schema=api.runtime_input_schema,
        runtime_bindings=api.runtime_bindings,
        allow_delegated_authorization=bool(api.allow_delegated_authorization),
        is_active=user_api.is_active,
        is_default=user_api.is_default,
        created_at=str(api.created_at.isoformat()),
        updated_at=str(api.updated_at.isoformat()),
    )


def _process_env_vars(
    env: Optional[Dict[str, str]], existing_env: Optional[Dict[str, str]] = None
) -> Optional[Dict[str, str]]:
    """Encrypt environment variables, retaining masks only for the same key."""
    if not env:
        return env

    encrypted_env = {}
    existing_env = existing_env or {}

    for k, v in env.items():
        if v == "********":
            # Retain existing encrypted value if masked
            if k in existing_env:
                encrypted_env[k] = existing_env[k]
            else:
                raise ValueError(
                    f"Masked secret '{k}' has no stored value; provide a new value"
                )
        else:
            encrypted_env[k] = encrypt_value(v)

    return encrypted_env


@custom_api_router.get("", response_model=List[CustomApiResponse])
async def list_custom_apis(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> List[CustomApiResponse]:
    """List all Custom APIs for the current user."""
    user_apis = (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == current_user.id).all()
    )

    responses = []
    for user_api in user_apis:
        if user_api.custom_api:
            responses.append(_db_api_to_response(user_api.custom_api, user_api))

    return responses


@custom_api_router.post(
    "", response_model=CustomApiResponse, status_code=status.HTTP_201_CREATED
)
async def create_custom_api(
    api_data: CustomApiCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Create a new Custom API."""

    # Check if name already exists
    existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Custom API with name '{api_data.name}' already exists",
        )

    # A masked value is a same-key retention token, never a transferable secret.
    try:
        encrypted_env = _process_env_vars(api_data.env)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid environment variables: {exc}",
        ) from exc
    try:
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema=api_data.runtime_input_schema,
            runtime_bindings=api_data.runtime_bindings,
            allow_delegated_authorization=api_data.allow_delegated_authorization,
            static_headers=api_data.headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid runtime configuration: {exc}",
        ) from exc

    # Create CustomApi
    new_api = CustomApi(
        name=api_data.name,
        description=api_data.description,
        url=api_data.url,
        method=api_data.method,
        headers=api_data.headers,
        body=api_data.body,
        env=encrypted_env,
        runtime_input_schema=api_data.runtime_input_schema,
        runtime_bindings=api_data.runtime_bindings,
        allow_delegated_authorization=api_data.allow_delegated_authorization,
    )

    db.add(new_api)
    db.flush()

    # Create UserCustomApi link
    user_api = UserCustomApi(
        user_id=current_user.id,
        custom_api_id=new_api.id,
        is_owner=True,
        can_edit=True,
        can_delete=True,
        is_active=api_data.is_active,
    )

    db.add(user_api)

    db.commit()
    db.refresh(new_api)
    db.refresh(user_api)

    return _db_api_to_response(new_api, user_api)


def _http_from_connector_runtime(exc: ConnectorRuntimeError) -> HTTPException:
    """Map the connector team seam's typed error onto this module's HTTP answer.

    Four call sites need it -- ``get_custom_api``, and ``update_custom_api``
    for its pre-lock resolution, its post-lock re-check (which makes its raw
    hook call through ``_recheck_team_access_under_definition_lock``) and its rename
    hook -- so that every one of them answers with the status and message
    the seam declares instead of letting the error reach the generic
    handler as a 500. Each call site wraps only the seam call it makes,
    rather than the whole route body, so the mapping itself lives here in
    one place.
    """
    return HTTPException(status_code=exc.status_code, detail=exc.safe_message)


def _resolve_custom_api_for_request(
    db: Session,
    user_id: int,
    api_id: int,
    *,
    skip_resolution_when: "Callable[[UserCustomApi], bool] | None" = None,
) -> "tuple[UserCustomApi | _TeamOwnedUserApi, CustomApi, ConnectorAccess | None]":
    """Resolve the caller's association, the definition row, and the caller's
    team access verdict, for ``GET``/``PUT /api/custom-apis/{api_id}``.

    Looks up the caller's own personal link row first, with the same query
    both routes have always run. When that row exists and its ``custom_api``
    relationship resolves, the association and the definition row both come
    from it and nothing else runs. When it does not -- no row, or a row whose
    relationship is unexpectedly empty -- the definition row is looked up on
    its own, because a team-owned API's shared row must still be found even
    though this caller has no personal link to it, and the caller's team
    access verdict decides what happens next:

    - no working personal row and no team access (``access is None``) -> 404,
      the same outcome every caller without an association has always gotten.
    - no working personal row but the caller's team links the API -> the
      ``_TeamOwnedUserApi`` stand-in takes the association's place. It carries
      the caller's own id and the flag defaults that apply when no personal
      row exists, so the response contract stays the shape it always was.

    ``skip_resolution_when`` lets a caller declare when its own working
    personal row already decides the answer on its own, so resolving a verdict
    would only add an unnecessary hook call: ``get_custom_api`` passes a
    predicate that is always true, because it never reads the verdict at all
    and a personal row -- edit rights or not -- already decides what it
    returns; ``update_custom_api`` passes one that checks ``can_edit``,
    because only a personal row with ``can_edit=True`` decides the edit answer
    on its own -- a ``can_edit=False`` row does not, since a granting team
    verdict can still widen it. Left unset (the default), resolution is never
    skipped, which is what a caller with no working personal row always needs
    -- the verdict is the gate there and must stay fail-closed.

    Raises ``ConnectorRuntimeError`` when access resolution itself fails;
    callers translate that into an ``HTTPException``.
    """
    from ..services.connector_team_scope import resolve_one_connector_access_or_raise
    from .mcp import _TeamOwnedUserApi

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == user_id,
        )
        .first()
    )
    if user_api is not None and user_api.custom_api is not None:
        api: Optional[CustomApi] = user_api.custom_api
    else:
        user_api = None
        api = db.query(CustomApi).filter(CustomApi.id == api_id).first()

    already_decided = user_api is not None and (
        skip_resolution_when is not None and skip_resolution_when(user_api)
    )

    access: "ConnectorAccess | None" = None
    if api is not None and not already_decided:
        access = resolve_one_connector_access_or_raise(
            db, int(user_id), ("custom_api", int(api.id))
        )

    if user_api is None and access is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    resolved_user_api: "UserCustomApi | _TeamOwnedUserApi" = (
        user_api if user_api is not None else _TeamOwnedUserApi(int(user_id))
    )
    return resolved_user_api, cast(CustomApi, api), access


def _recheck_team_access_under_definition_lock(
    db: Session, user_id: int, api_id: int
) -> "ConnectorAccess | None":
    """Re-resolve the caller's team access verdict for ``update_custom_api``,
    on behalf of a caller whose freshly-read personal link row does not
    grant the edit on its own.

    Lives as its own function, separate from ``update_custom_api``, so that
    this call site and the route's rename hook call are each attributed to a
    distinct function by the call-site table's accounting
    (``connector_team_scope.py``'s "Call sites and what the caller holds"
    table, checked against the source by
    ``test_the_call_site_table_and_the_call_sites_agree``): that accounting
    is keyed by enclosing function name, one row per function, and
    ``update_custom_api`` already owns the rename hook's row.

    Called only from inside ``update_custom_api``'s ``FOR UPDATE`` block, on
    a payload that writes the ``custom_apis`` definition row, after that
    lock has been taken and before anything this request has staged is
    committed. Declares ``caller_holds_lock=True`` on the call it makes: a
    hook that ends this transaction would release that lock without the
    caller finding out, and the writes staged above would then commit
    against a row somebody else may have moved.

    Raises ``ConnectorRuntimeError`` when access resolution itself fails;
    the caller translates that into an ``HTTPException`` and rolls back.
    """
    from ..services.connector_team_scope import resolve_one_connector_access_or_raise

    return resolve_one_connector_access_or_raise(
        db, user_id, ("custom_api", api_id), caller_holds_lock=True
    )


@custom_api_router.get("/{api_id}", response_model=CustomApiResponse)
def get_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Get a specific Custom API by ID."""

    try:
        # This route never reads the team verdict at all (see
        # _db_api_to_response), so a working personal row -- edit rights or
        # not -- already decides everything this route returns; resolving a
        # verdict for such a caller would only add an unnecessary hook call.
        user_api, api, _team_access = _resolve_custom_api_for_request(
            db,
            int(current_user.id),
            api_id,
            skip_resolution_when=lambda _user_api: True,
        )
    except ConnectorRuntimeError as exc:
        raise _http_from_connector_runtime(exc) from exc

    return _db_api_to_response(api, user_api)


@custom_api_router.put("/{api_id}", response_model=CustomApiResponse)
def update_custom_api(
    api_id: int,
    api_data: CustomApiUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CustomApiResponse:
    """Update an existing Custom API."""

    try:
        # A personal row with can_edit=True already decides the edit answer on
        # its own (the gate just below), so resolving a verdict for that row
        # would only add an unnecessary hook call. A can_edit=False personal
        # row does not decide it, because a granting team verdict can still
        # widen it, so that caller's verdict is resolved.
        #
        # The definition row this resolution loads is deliberately dropped:
        # the fresh single-table read further down is the one every field
        # below reads and writes.
        user_api, _pre_lock_api, team_access = _resolve_custom_api_for_request(
            db,
            int(current_user.id),
            api_id,
            skip_resolution_when=lambda ua: bool(ua.can_edit),
        )
    except ConnectorRuntimeError as exc:
        raise _http_from_connector_runtime(exc) from exc

    # Two independent grants of the edit right: the caller's own link row, or
    # a team verdict that grants edit on a connector the caller's team links.
    # A caller with no personal row at all reaches this with the stand-in,
    # whose can_edit is False, so only the verdict can admit that caller.
    is_stand_in = not isinstance(user_api, UserCustomApi)
    if not (
        bool(user_api.can_edit) or (team_access is not None and team_access.can_edit)
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to edit this Custom API",
        )

    # is_active lives on the personal association row; a caller with no
    # personal row (the stand-in) has none to hold it, so a request that
    # carries the field at all must be rejected outright -- writing it onto
    # the stand-in would only set a shadowing instance attribute that
    # persists nothing, and the response below would then read that shadow
    # back and report a change that never happened. This checks whether the
    # field is present, the same test ``writes_definition_row`` below makes
    # with ``model_fields_set``, not whether its value is ``None``: an
    # explicit ``{"is_active": null}`` carries the field and must be refused
    # the same as any other value would be.
    if is_stand_in and "is_active" in api_data.model_fields_set:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No personal connection exists to configure is_active for this API",
        )

    # Which row a request writes decides which row it locks. Every field of
    # ``CustomApiUpdate`` except ``is_active`` writes the shared
    # ``CustomApi`` definition row; ``is_active`` writes this caller's own
    # ``UserCustomApi`` link row and nothing else. Locking the definition
    # row for a payload that never writes it made an activate/deactivate
    # queue behind an unrelated edit of the same connector -- a wait that
    # request has no write to justify, and one that surfaces as an error
    # rather than a delay wherever a lock timeout is configured.
    #
    # ``model_fields_set`` decides this, not the values: an explicitly-null
    # ``runtime_input_schema`` is written to the definition row below even
    # though its value is ``None``, while an absent field is not written at
    # all. The set is a superset of the writes below -- a payload carrying
    # ``description=None`` is counted here and then skipped by the write at
    # its own ``is not None`` guard -- so this takes the lock in a few
    # cases that did not need it and skips it in none that do.
    fields_set = api_data.model_fields_set
    writes_definition_row = bool(fields_set - {"is_active"})
    # This flag also gates the whole post-lock re-authorization further down
    # -- the fresh read of the caller's link row and the fresh resolution of
    # the team verdict -- not only the lock itself: adding a future field to
    # the ``{"is_active"}`` exclusion set above, because it too writes only
    # the link row, would silently skip that re-authorization as well, not
    # just the lock, for any payload that sets only that field.

    # A fresh single-table read of the definition row, on both paths. The
    # gate above reached this row through the personal link row's
    # relationship, or through a plain lookup for a caller who has no
    # personal link row, and neither is a statement this route can add a
    # locking clause to; this is a separate statement, so a row deleted
    # between the two yields None here (handled as the same 404) rather
    # than surfacing as an unrelated error out of the write path or out of
    # ``db.refresh`` below. ``populate_existing()`` makes this statement's
    # row the one the rest of this route reads and responds with: without
    # it the already-identity-mapped instance the gate loaded would be
    # returned unrefreshed, and every field below would still be that
    # earlier snapshot.
    #
    # ``FOR UPDATE`` is added only on the path that writes this row, so a
    # request that writes it still waits for another request holding it.
    # That clause is a PostgreSQL/MySQL row lock only: SQLAlchemy renders
    # no locking clause at all on SQLite -- the statement it emits there is
    # byte-for-byte the one it emits without this call -- so on a SQLite
    # deployment the read-modify-write below is not serialized and two
    # concurrent edits of one connector can still interleave. The
    # single-statement conditional ``UPDATE``s elsewhere in this repository
    # (services/task_interaction_staging.py,
    # services/chat_history_service.py) are safe on SQLite without a lock
    # because one statement is atomic there; that reasoning does not carry
    # to this route, which reads the row, computes in Python, and writes it
    # back. Closing the SQLite window needs the dual-dialect fence
    # ``_lock_user_row_for_preferences_update`` (api/auth.py) and
    # ``acquire_runtime_key_transition_fence`` (services/api_keys.py) use
    # -- a no-op ``UPDATE`` that takes SQLite's writer lock -- and is left
    # to a change of its own.
    definition_query = (
        db.query(CustomApi).filter(CustomApi.id == api_id).populate_existing()
    )
    if writes_definition_row:
        definition_query = definition_query.with_for_update()
    current_api = definition_query.first()
    if current_api is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )
    # Declared as ``Any`` from here on, for mypy's sake: the column-typed
    # attributes below (name, description, env, ...) are all mutated
    # directly by this route, which mypy rejects against the ORM's declared
    # column types.
    api = cast(Any, current_api)

    if writes_definition_row:
        # The access gate above ran before the lock statement; the lock
        # statement waits. Everything the gate decided -- that this caller
        # may edit this connector, whether that came from the caller's own
        # link row or from the team access verdict -- was therefore
        # established before a wait of unbounded length, and nothing has
        # re-established it since. A supported admin user deletion
        # (``admin_users.py``, which removes a user's association rows and
        # leaves every definition row standing), a connector-team delete
        # that removes only the caller's link, or a revocation of the
        # caller's team access inside the installing application's own
        # tables can each commit inside that wait. The request would then
        # resume unauthorized, write the shared definition row anyway, and
        # commit it -- a durable shared mutation the caller was no longer
        # allowed to make.
        #
        # So both halves of the gate's decision are re-established here from
        # fresh reads and the same combination is recomputed: the caller's
        # own link row is re-read, and the team verdict is re-resolved
        # whenever that fresh read does not grant the edit on its own. A
        # payload that writes only the caller's own link row skips this
        # block entirely, and correctly so -- it took no lock above, so its
        # gate was never separated from its writes by a wait of unbounded
        # length.
        #
        # ``populate_existing()`` on the definition query above refreshes
        # the row of that statement and nothing else, so it does not cover
        # the link row: that needs its own statement. This one is a
        # single-table read of the caller's link, with
        # ``populate_existing()`` so its columns are overwritten with the
        # database's current values rather than the ones the gate loaded.
        # When it returns a row, that row replaces ``user_api`` for the rest
        # of the route -- the link-row write below and the response both
        # read it. When it returns nothing, ``user_api`` becomes a freshly
        # constructed stand-in rather than the one this route resolved
        # before the lock: that earlier one may itself have been backed by
        # a personal row that has since been deleted, and this statement is
        # what finds that out.
        # A revocation that commits after these statements is the window
        # this route had before it took any lock at all: in-process, with
        # no wait in it. Closing that one needs an authorization fence of
        # its own and is not attempted here.
        current_user_api = (
            db.query(UserCustomApi)
            .filter(
                UserCustomApi.custom_api_id == api_id,
                UserCustomApi.user_id == current_user.id,
            )
            .populate_existing()
            .first()
        )
        still_can_edit = current_user_api is not None and bool(
            current_user_api.can_edit
        )
        if not still_can_edit:
            # The freshly-read personal row does not grant the edit on its
            # own -- either it is gone, or its can_edit has been cleared --
            # so the team verdict is re-resolved regardless of what the gate
            # decided before the lock: unlike the narrower condition this
            # replaces, this one also catches a caller whose personal row
            # was deleted out from under a gate decision that never went
            # through the team verdict at all. An owner (the fresh row still
            # grants edit on its own) and a payload that writes only
            # ``is_active`` (this whole block is skipped) both still cost
            # nothing extra; a caller who reached the gate on the team
            # verdict and still holds it spends the one extra hook call it
            # always did; a request with no access hook installed spends
            # nothing, because the hook slot being empty makes resolution
            # return ``{}`` without a query. Only a request whose personal
            # row or team verdict actually changed while it waited for the
            # lock now gets a different -- and correct -- answer than it did
            # before this change.
            #
            # This re-check assumes READ COMMITTED, PostgreSQL's default,
            # which this codebase sets no isolation_level on its engine to
            # change: it needs a fresh snapshot to see a link the application
            # revoked and committed after this request's pre-lock read. Under
            # REPEATABLE READ or SERIALIZABLE the re-read reuses this
            # transaction's original snapshot, sees the pre-lock answer again,
            # and the re-check degrades to a no-op -- it would stop refusing,
            # not start refusing wrongly.
            try:
                rechecked = _recheck_team_access_under_definition_lock(
                    db, int(current_user.id), api_id
                )
            except ConnectorRuntimeError as exc:
                db.rollback()
                raise _http_from_connector_runtime(exc) from exc
            if rechecked is not None and rechecked.can_edit:
                still_can_edit = True
            elif team_access is not None and team_access.can_edit:
                # The gate's own read once granted this edit through the
                # team verdict; that access was revoked while this request
                # waited for the lock.
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=(
                        "Your team's access to this Custom API changed while "
                        "this edit was in flight"
                    ),
                )
            elif current_user_api is None:
                # No personal row survived the wait, and neither the fresh
                # nor the pre-lock team verdict grants the edit: the same
                # 404 a caller with no association to this connector has
                # always gotten, whether that caller reached the gate
                # through a personal row that is now gone or never had one.
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Custom API not found",
                )
            else:
                db.rollback()
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You do not have permission to edit this Custom API",
                )

        if current_user_api is not None:
            user_api = current_user_api
        else:
            # The gate's own read may have resolved a personal row that has
            # since been deleted; carrying that ORM object forward would
            # raise ``StaleDataError`` from ``db.commit()`` below on a
            # payload that writes the definition row, or
            # ``ObjectDeletedError`` while the response is built on a
            # payload that does not -- and the latter fails only after this
            # request's other writes have already committed. A freshly
            # constructed stand-in never touches the database at all, so
            # neither can happen.
            from .mcp import _TeamOwnedUserApi

            user_api = _TeamOwnedUserApi(int(current_user.id))

        # A second is_active guard, under the lock. The one above the lock
        # (see the ``is_stand_in`` check earlier in this route) catches a
        # caller who already had no personal row when the gate ran; this one
        # catches a caller whose personal row existed then and was deleted
        # while this request waited for the lock. Same presence test as the
        # guard above: the field being in the request is what matters, not
        # its value.
        if current_user_api is None and "is_active" in api_data.model_fields_set:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No personal connection exists to configure is_active for this API",
            )

    # Read only after the statement above: rename_team_connector's "old"
    # argument must be the name this transaction's own read established --
    # under the lock, on the path that renames -- not whatever the gate's
    # read further up saw. A concurrent committed rename in
    # between would otherwise make this stale, and the rewrite below would
    # look for a name that no longer exists anywhere, leaving the previous
    # renamer's selectors dangling with no error.
    old_name = str(api.name)

    # Check name uniqueness if name is changed
    if api_data.name and api_data.name != api.name:
        existing = db.query(CustomApi).filter(CustomApi.name == api_data.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Custom API with name '{api_data.name}' already exists",
            )
        api.name = api_data.name

    # Update fields
    if api_data.description is not None:
        api.description = api_data.description
    if api_data.url is not None:
        api.url = api_data.url
    if api_data.method is not None:
        api.method = api_data.method
    if api_data.headers is not None:
        api.headers = api_data.headers
    if api_data.body is not None:
        api.body = api_data.body

    # Process env variables
    if api_data.env is not None:
        existing_env: Dict[str, str] = api.env if isinstance(api.env, dict) else {}
        try:
            processed_env = _process_env_vars(api_data.env, existing_env)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid environment variables: {exc}",
            ) from exc
        api.env = processed_env

    runtime_input_schema = (
        api_data.runtime_input_schema
        if "runtime_input_schema" in fields_set
        else api.runtime_input_schema
    )
    runtime_bindings = (
        api_data.runtime_bindings
        if "runtime_bindings" in fields_set
        else api.runtime_bindings
    )
    allow_delegated_authorization = (
        bool(api_data.allow_delegated_authorization)
        if "allow_delegated_authorization" in fields_set
        else bool(api.allow_delegated_authorization)
    )
    try:
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema=runtime_input_schema,
            runtime_bindings=runtime_bindings,
            allow_delegated_authorization=allow_delegated_authorization,
            static_headers=api.headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid runtime configuration: {exc}",
        ) from exc
    if "runtime_input_schema" in fields_set:
        api.runtime_input_schema = runtime_input_schema
    if "runtime_bindings" in fields_set:
        api.runtime_bindings = runtime_bindings
    if "allow_delegated_authorization" in fields_set:
        api.allow_delegated_authorization = allow_delegated_authorization

    from ..services.connector_team_scope import rename_team_connector

    # The rename hook is application-installed and raises the seam's own typed
    # error; answer with the status that error declares rather than letting it
    # reach the generic handler as a 500. Everything this request staged is
    # rolled back first, so a refused rename leaves nothing behind.
    try:
        rename_team_connector(
            db,
            int(current_user.id),
            "custom_api",
            int(api_id),
            old_name,
            str(api.name),
            # This transaction holds the definition row FOR UPDATE and has not
            # committed anything yet. A hook that ends it releases that lock
            # without this route finding out, and the writes staged above would
            # then commit against a row somebody else may have moved.
            caller_holds_lock=True,
        )
    except ConnectorRuntimeError as exc:
        db.rollback()
        raise _http_from_connector_runtime(exc) from exc

    # Update UserCustomApi link. A caller with no personal row never reaches
    # this: a payload carrying is_active is rejected by one of two guards --
    # the one above the lock, for a caller who already had no personal row
    # when the gate ran, or the one under the lock, for a caller whose
    # personal row existed then and was deleted while this request waited
    # for the lock.
    if api_data.is_active is not None:
        user_api.is_active = api_data.is_active

    db.commit()
    db.refresh(api)

    return _db_api_to_response(api, user_api)


@custom_api_router.delete("/{api_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_custom_api(
    api_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> None:
    """Delete a Custom API."""

    user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .first()
    )

    if not user_api or not user_api.custom_api:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )

    if not user_api.can_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this Custom API",
        )

    # One lock order across every row this pair of routes touches. An
    # ``update_custom_api`` call that writes the definition row locks this
    # same row first, calls ``rename_team_connector`` afterwards, and
    # writes the ``UserCustomApi`` link row afterwards too; both branches below delete
    # the link row first and the definition row second, inside one
    # transaction. Taking the lock here -- before ``delete_team_connector``
    # rather than after it -- puts both routes in one order on both sides of
    # the hook boundary: definition row first, then the link row and
    # whatever rows a connector team hook locks. With the lock after the
    # hook instead, the two routes waited in opposite directions across that
    # boundary and a concurrent edit/delete pair on the same connector could
    # deadlock (PostgreSQL 40P01, surfacing to the caller as HTTP 500).
    # ``populate_existing`` matches the PUT's own lock: it is the same
    # identity-mapped instance the relationship read above returned, with
    # its column values overwritten by whatever this statement reads under
    # the lock, so the deletion below acts on this transaction's own view
    # of the row rather than a snapshot from before the lock was taken.
    # This is also a fresh statement, so a row deleted between the access
    # read above and here still yields None (handled as the same 404)
    # rather than reaching ``db.delete`` with nothing to delete.
    #
    # Two costs come with taking it here rather than last. The two 403
    # refusals below read ``delete_team_connector``'s answer and so now
    # happen after this statement: a request that is going to be refused
    # does briefly hold this row, until the raised ``HTTPException``
    # propagates out and the request's session is closed without
    # committing. And the hook's own work now runs inside the lock, so this
    # route holds the row for longer than it did with the lock last.
    #
    # On the success path, a hook that calls a helper which ends this
    # session's own transaction -- ``release_db_connection_if_clean``
    # (``models/database.py``) is the one this repository has -- releases
    # this lock without the route ever knowing: at this point in the route
    # the session has run nothing but ``SELECT``s, which is exactly the
    # condition that helper treats as safe to roll back, so it takes the
    # rollback branch, raises nothing, and the route carries on to delete
    # and commit believing it still holds the row it locked above. A hook
    # that raises instead is not this case: the exception propagates out of
    # this route unhandled, so the request is aborted rather than
    # continuing to commit, and the transaction's own rollback on session
    # close is the ordinary way this lock is released.
    #
    # What this statement orders is this route against ``update_custom_api``
    # on the same connector. A PUT that writes the definition row cannot
    # form a cycle with this route, whatever the hook locks, because both
    # reach the hook with this row already held and so cannot be inside the
    # hook at the same time. A PUT that writes only the caller's own
    # ``UserCustomApi`` link row takes no lock on this row at all: it reads
    # the definition row without locking it and waits only for the link
    # row, so it has nothing this route waits for and cannot close a cycle
    # either. A hook that additionally locks rows of its own in some other
    # order relative to this repository's statements is outside what this
    # statement arranges; only the installing application can order those.
    #
    # ``FOR UPDATE`` here is a PostgreSQL/MySQL row lock only: SQLAlchemy
    # renders no locking clause at all on SQLite, so on a SQLite deployment
    # this statement does not order this route against anything. Closing
    # that window needs the dual-dialect fence
    # ``acquire_runtime_key_transition_fence`` (services/api_keys.py) uses
    # -- a no-op ``UPDATE`` that takes SQLite's writer lock -- and is left
    # to a change of its own.
    locked_api = (
        db.query(CustomApi)
        .filter(CustomApi.id == api_id)
        .populate_existing()
        .with_for_update()
        .first()
    )
    if locked_api is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )
    api = locked_api

    # This route's gate ran before the lock statement, and the lock
    # statement waits. The gate's two answers -- the caller has a link to
    # this connector, and that link grants delete -- were both established
    # before that wait, and this route's effects are the widest in the
    # pair: with no connector-team hook installed it takes the ``db.delete(api)``
    # branch below, which removes the shared definition row and cascades
    # away every other user's link to it. A supported admin user deletion
    # or a team delete of the caller's own link can commit inside the
    # wait, and the request would then answer 204 and delete the shared
    # row for a caller who no longer had any link to it.
    #
    # Re-read the link row here, before ``delete_team_connector`` is
    # called and before anything is deleted, so a revoked request produces
    # its refusal with zero shared effect: no hook call, no delete, no
    # commit. ``populate_existing()`` on the lock statement above refreshes
    # the definition row only, so the link row needs this separate
    # statement; the object it returns replaces ``user_api`` for the
    # team-owned branch's own ``db.delete``. A revocation that commits
    # after this statement is the window this route had before it took any
    # lock at all: in-process, with no wait in it. Closing that one needs
    # the authorization fence designed for the team-edit changes and is
    # not attempted here. Same order and same answers as the gate: gone is
    # a 404, present but no longer permitted is a 403.
    current_user_api = (
        db.query(UserCustomApi)
        .filter(
            UserCustomApi.custom_api_id == api_id,
            UserCustomApi.user_id == current_user.id,
        )
        .populate_existing()
        .first()
    )
    if current_user_api is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Custom API not found",
        )
    if not current_user_api.can_delete:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this Custom API",
        )
    user_api = current_user_api

    from ..services.connector_team_scope import delete_team_connector

    team_delete = delete_team_connector(
        db,
        int(current_user.id),
        "custom_api",
        int(api_id),
        # This transaction holds the definition row FOR UPDATE and has run
        # nothing but SELECTs, which is exactly the state in which a helper
        # that "returns a clean connection" rolls it back -- releasing the
        # lock while this route goes on to delete.
        caller_holds_lock=True,
    )
    if team_delete.blocked_reason:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=team_delete.blocked_reason,
        )
    if team_delete.team_owned and not team_delete.authorized:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only a team admin can delete a team Custom API",
        )

    if team_delete.team_owned:
        db.delete(user_api)
        db.flush([user_api])
        with db.no_autoflush:
            remaining = (
                db.query(UserCustomApi)
                .filter(UserCustomApi.custom_api_id == api_id)
                .first()
            )
        if remaining is None and team_delete.delete_definition:
            db.delete(api)
    else:
        db.delete(api)  # Will cascade to UserCustomApi
    db.commit()

    return None
