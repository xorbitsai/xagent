import ast
import importlib
import inspect
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.api.custom_api import (
    CustomApiCreate,
    CustomApiResponse,
    CustomApiUpdate,
    _process_env_vars,
    create_custom_api,
    delete_custom_api,
    get_custom_api,
    list_custom_apis,
    update_custom_api,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorDeleteDecision,
    set_connector_team_hooks,
)

_SEAM_MODULE = "xagent.web.api.custom_api"

# Every top-level function in this module that can reach an installed
# connector team hook. Written out so the discovery below cannot pass by
# finding nothing.
_SEAM_REACHING_FUNCTIONS = {
    "_resolve_custom_api_for_request",
    "_recheck_team_access_under_definition_lock",
    "get_custom_api",
    "update_custom_api",
    "delete_custom_api",
}


def _functions_reaching_the_connector_seam() -> dict[str, ast.AST]:
    """Every top-level function in this module that can reach an installed
    connector team hook.

    Seeded on the functions that import ``connector_team_scope`` in their own
    body, which is how every call site in this module reaches the seam, then
    closed transitively over plain-name calls, because one route reaches it
    only through a helper (``get_custom_api`` through
    ``_resolve_custom_api_for_request``). A seed-only check would miss exactly
    the route this test exists for.
    """
    module = importlib.import_module(_SEAM_MODULE)
    tree = ast.parse(inspect.getsource(module))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    reaching = {
        name
        for name, node in functions.items()
        if any(
            isinstance(child, ast.ImportFrom)
            and child.module is not None
            and child.module.endswith("connector_team_scope")
            for child in ast.walk(node)
        )
    }
    changed = True
    while changed:
        changed = False
        for name, node in functions.items():
            if name in reaching:
                continue
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            if called & reaching:
                reaching.add(name)
                changed = True
    return {name: functions[name] for name in reaching}


def test_the_discovery_of_seam_reaching_functions_is_not_vacuous():
    """Pins the enumeration itself, so the assertion below cannot pass by
    finding nothing."""
    assert set(_functions_reaching_the_connector_seam()) == _SEAM_REACHING_FUNCTIONS


def test_no_function_that_reaches_the_connector_seam_is_a_coroutine():
    """An installed connector team hook may be slow -- the seam is designed on
    the assumption that the installing application answers from its own
    tables. FastAPI runs a coroutine route on the event loop thread itself, so
    a slow hook call inside an ``async def`` stalls every other request the
    process is serving, not just this one; a plain ``def`` goes to the
    threadpool instead, where a slow call occupies one worker.

    Enumerated by reachability rather than by a hand-written list of routes:
    an earlier fix for this same risk class swept siblings along the "takes a
    row lock" axis and therefore missed a route that calls a hook without
    taking one.
    """
    offenders = [
        name
        for name, node in _functions_reaching_the_connector_seam().items()
        if isinstance(node, ast.AsyncFunctionDef)
    ]
    assert offenders == [], (
        "these functions can reach an installed connector team hook while "
        f"running on the event loop thread: {sorted(offenders)}"
    )


def test_custom_api_models_env_validation():
    # Valid creation
    api = CustomApiCreate(name="test", env={"key": "val"})
    assert api.env == {"key": "val"}

    # Missing env is allowed (handled by database default or just none)
    api = CustomApiCreate(name="test")
    assert api.env is None

    # Empty env dict is allowed to clear secrets
    api_empty = CustomApiCreate(name="test", env={})
    assert api_empty.env == {}

    # Same for update
    api_update = CustomApiUpdate(name="test", env={})
    assert api_update.env == {}

    runtime_api = CustomApiCreate(
        name="runtime",
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    assert runtime_api.runtime_input_schema == {
        "context": {"account_id": {"type": "string"}}
    }
    assert runtime_api.runtime_bindings is not None


def test_custom_api_response_requires_runtime_projection_fields():
    """Custom API response mappers must project every persisted runtime field."""
    response_data = {
        "id": 1,
        "user_id": 1,
        "name": "runtime",
        "description": None,
        "url": None,
        "method": "GET",
        "headers": None,
        "body": None,
        "env": None,
        "is_active": True,
        "is_default": False,
        "created_at": "2026-07-14T00:00:00",
        "updated_at": "2026-07-14T00:00:00",
    }

    with pytest.raises(ValidationError) as exc_info:
        CustomApiResponse(**response_data)

    missing_fields = {error["loc"] for error in exc_info.value.errors()}
    assert missing_fields == {
        ("runtime_input_schema",),
        ("runtime_bindings",),
        ("allow_delegated_authorization",),
    }


def test_process_env_vars():
    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        # Test None
        assert _process_env_vars(None) is None

        # Test encrypting new values
        env = {"key1": "val1", "key2": "val2"}
        res = _process_env_vars(env)
        assert res == {"key1": "enc_val1", "key2": "enc_val2"}

        # Test keeping masked values
        env_with_mask = {"key1": "********", "key3": "val3"}
        existing = {"key1": "enc_old1", "key2": "enc_old2"}
        res_masked = _process_env_vars(env_with_mask, existing)
        assert res_masked == {"key1": "enc_old1", "key3": "enc_val3"}

        # A mask cannot be moved to a new key identity.
        with pytest.raises(ValueError, match="new_key"):
            _process_env_vars({"new_key": "********"}, existing)


@pytest.mark.asyncio
async def test_list_custom_apis():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10, name="test_api", created_at=datetime.now(), updated_at=datetime.now()
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    db.query().filter().all.return_value = [mock_user_api]

    res = await list_custom_apis(current_user=user, db=db)
    assert len(res) == 1
    assert res[0].name == "test_api"
    assert res[0].id == 10


@pytest.mark.asyncio
async def test_create_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    api_data = CustomApiCreate(
        name="new_api",
        description="desc",
        env={"k1": "v1"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
        is_active=True,
    )

    # Mock no existing api
    db.query().filter().first.return_value = None

    # Create mock CustomApi object with datetimes so isoformat() doesn't fail
    CustomApi(
        id=1, name="new_api", created_at=datetime.now(), updated_at=datetime.now()
    )

    # Create mock UserCustomApi object to pair with our custom api mock
    UserCustomApi(
        user_id=1, custom_api_id=1, is_owner=True, is_active=True, is_default=False
    )

    # Update db.add to populate created_at/updated_at fields on our mock
    def mock_add(obj):
        if isinstance(obj, CustomApi):
            obj.id = 1
            obj.created_at = datetime.now()
            obj.updated_at = datetime.now()
        elif isinstance(obj, UserCustomApi):
            obj.user_id = 1
            obj.custom_api_id = 1
            obj.is_active = True
            obj.is_default = False

    db.add.side_effect = mock_add

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        res = await create_custom_api(api_data, current_user=user, db=db)

        assert res.name == "new_api"
        assert res.env == {"k1": "********"}  # Response should mask env
        assert res.runtime_input_schema == {
            "context": {"account_id": {"type": "string"}}
        }
        assert res.runtime_bindings == [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ]
        db.add.assert_called()
        db.commit.assert_called()


@pytest.mark.asyncio
async def test_create_custom_api_duplicate_name():
    db = MagicMock(spec=Session)
    user = User(id=1)

    api_data = CustomApiCreate(name="existing_api")

    # Mock existing api
    db.query().filter().first.return_value = CustomApi(name="existing_api")

    with pytest.raises(HTTPException) as exc_info:
        await create_custom_api(api_data, current_user=user, db=db)
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_create_custom_api_rejects_runtime_static_header_conflict():
    db = MagicMock(spec=Session)
    user = User(id=1)
    api_data = CustomApiCreate(
        name="runtime_api",
        headers={"X-Account-ID": "static"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )
    db.query().filter().first.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await create_custom_api(api_data, current_user=user, db=db)

    assert exc_info.value.status_code == 400
    assert "Invalid runtime configuration" in str(exc_info.value.detail)


def test_get_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10, name="test_api", created_at=datetime.now(), updated_at=datetime.now()
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    db.query().filter().first.return_value = mock_user_api

    res = get_custom_api(10, current_user=user, db=db)
    assert res.id == 10
    assert res.name == "test_api"


def test_get_custom_api_not_found():
    db = MagicMock(spec=Session)
    user = User(id=1)
    db.query().filter().first.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        get_custom_api(99, current_user=user, db=db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10,
        name="old_name",
        env={"k1": "enc_old1"},
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[],
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )

    # Return user api on first query
    # Return None for existing name check
    db.query().filter().first.side_effect = [mock_user_api, None]
    # The row lock's own fresh query is a separate mock chain
    # (.populate_existing().with_for_update() sits between .filter() and
    # .first()), so it needs its own return value rather than sharing the
    # side_effect list above.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The post-lock re-read of the link row is a third mock chain, distinct
    # from both of the above: unrevoked, it must return the same
    # still-``can_edit`` link the access gate saw, or the route's
    # unstubbed ``MagicMock`` default -- truthy for any attribute access,
    # including ``.can_edit`` -- would make the permission check below a
    # no-op no matter what it is actually asked to verify.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    api_data = CustomApiUpdate(
        name="new_name",
        env={"k1": "********", "k2": "v2"},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
    )

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        update_custom_api(10, api_data, current_user=user, db=db)

        assert mock_api.name == "new_name"
        assert mock_api.env == {"k1": "enc_old1", "k2": "enc_v2"}
        assert mock_api.runtime_input_schema == {
            "context": {"account_id": {"type": "string"}}
        }
        assert mock_api.runtime_bindings == [
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ]
        db.commit.assert_called()


@pytest.mark.asyncio
async def test_update_custom_api_env_replacement_deletes_only_the_omitted_secret():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(
        id=10,
        name="records",
        env={"BEARER_TOKEN": "enc_bearer", "TENANT": "enc_tenant"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The post-lock re-read of the link row -- see the comment in
    # test_update_custom_api.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        update_custom_api(
            10,
            CustomApiUpdate(env={"TENANT": "********"}),
            current_user=user,
            db=db,
        )

    assert mock_api.env == {"TENANT": "enc_tenant"}


@pytest.mark.asyncio
async def test_update_custom_api_rejects_renamed_masked_secret():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(
        id=10,
        name="records",
        env={"TOKEN": "encrypted-token"},
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The post-lock re-read of the link row -- see the comment in
    # test_update_custom_api.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    with pytest.raises(HTTPException) as exc_info:
        update_custom_api(
            10,
            CustomApiUpdate(env={"RENAMED_TOKEN": "********"}),
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 400
    assert mock_api.env == {"TOKEN": "encrypted-token"}
    db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_update_custom_api_explicit_null_clears_runtime_config():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10,
        name="old_name",
        runtime_input_schema={"context": {"account_id": {"type": "string"}}},
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
        allow_delegated_authorization=True,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    mock_user_api = UserCustomApi(
        user_id=1,
        custom_api_id=10,
        can_edit=True,
        is_active=True,
        is_default=False,
        custom_api=mock_api,
    )
    db.query().filter().first.return_value = mock_user_api
    # The row lock's own fresh query is a separate mock chain -- see the
    # comment in test_update_custom_api.
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The post-lock re-read of the link row -- see the comment in
    # test_update_custom_api.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    api_data = CustomApiUpdate(
        runtime_input_schema=None,
        runtime_bindings=None,
        allow_delegated_authorization=False,
    )

    update_custom_api(10, api_data, current_user=user, db=db)

    assert mock_api.runtime_input_schema is None
    assert mock_api.runtime_bindings is None
    assert mock_api.allow_delegated_authorization is False
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_delete_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(id=10)
    mock_user_api = UserCustomApi(
        user_id=1, custom_api_id=10, can_delete=True, custom_api=mock_api
    )

    db.query().filter().first.return_value = mock_user_api
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The post-lock re-read of the link row -- see the comment in
    # test_update_custom_api.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    delete_custom_api(10, current_user=user, db=db)

    db.delete.assert_called_once_with(mock_api)
    db.commit.assert_called()


@pytest.mark.asyncio
async def test_delete_team_custom_api_flushes_only_current_user_link():
    db = MagicMock(spec=Session)
    user = User(id=1)
    mock_api = CustomApi(id=10)
    mock_user_api = UserCustomApi(
        user_id=1, custom_api_id=10, can_delete=True, custom_api=mock_api
    )
    db.query().filter().first.side_effect = [mock_user_api, None]
    db.query().filter().populate_existing().with_for_update().first.return_value = (
        mock_api
    )
    # The lock-order re-read of the link row is a third mock chain, distinct
    # from the access gate's plain ``.filter().first()`` above: unrevoked,
    # it must return the same still-``can_delete`` link the gate saw.
    db.query().filter().populate_existing().first.return_value = mock_user_api

    decision = ConnectorDeleteDecision(
        team_owned=True,
        authorized=True,
        delete_definition=True,
    )
    with patch(
        "xagent.web.services.connector_team_scope.delete_team_connector",
        return_value=decision,
    ):
        delete_custom_api(10, current_user=user, db=db)

    db.flush.assert_called_once_with([mock_user_api])
    assert db.no_autoflush.__enter__.called
    assert db.delete.call_args_list == [call(mock_user_api), call(mock_api)]
    db.commit.assert_called_once()


def test_the_locking_routes_are_sync_defs_so_a_lock_wait_never_holds_the_event_loop():
    """``update_custom_api`` can run a ``SELECT ... FOR UPDATE`` that waits
    indefinitely on a concurrent writer holding the same definition row;
    ``delete_custom_api`` always does. FastAPI runs a coroutine route on the
    event loop thread itself, so such a wait inside an ``async def`` route
    stalls every other request the process is serving, not just this one.
    Declaring them as plain ``def`` puts them in the threadpool instead,
    where the wait occupies one worker.
    """
    from xagent.web.api import custom_api as custom_api_api

    assert not inspect.iscoroutinefunction(custom_api_api.update_custom_api)
    assert not inspect.iscoroutinefunction(custom_api_api.delete_custom_api)


def _lock_order_session_factory():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine), engine


def _seed_owned_api_for_lock_order(
    session_factory, *, name: str, description: str | None = None
) -> tuple[int, int]:
    db = session_factory()
    owner = User(username=f"user-{name}", password_hash="x", is_admin=False)
    db.add(owner)
    db.flush()
    api = CustomApi(
        name=name, url="https://example.test/api", method="GET", description=description
    )
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=owner.id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    owner_id, api_id = int(owner.id), int(api.id)
    db.close()
    return owner_id, api_id


def _count_custom_apis_selects_before_first_delete(statements: list[str]) -> int:
    """How many ``SELECT``s against ``custom_apis`` land before the first
    ``DELETE`` of either table.

    The route's own not-found guard (``not user_api or not
    user_api.custom_api``) always lazy-loads the ``custom_api`` relationship,
    which is one such ``SELECT`` on its own -- with or without the lock
    statement this test exists to pin. So *presence* of a ``custom_apis``
    ``SELECT`` before the delete is true either way and proves nothing; the
    *count* is what distinguishes them -- one without the lock statement,
    two with it, because the lock statement is its own separate ``Query``
    execution against the database, issued in addition to the relationship
    load above rather than instead of it. ``populate_existing()`` does not
    decide whether that second statement is sent; it decides which column
    values end up on the already-identity-mapped row once it is: with it,
    the lock statement's freshly-read columns overwrite what the
    relationship load put there, rather than being discarded in favor of
    it.
    """
    count = 0
    for statement in statements:
        upper = statement.strip().upper()
        if upper.startswith("DELETE"):
            break
        if upper.startswith("SELECT") and "FROM CUSTOM_APIS" in upper:
            count += 1
    return count


def _updated_table_names(statements: list[str]) -> list[str]:
    """The table name out of each recorded ``UPDATE`` statement, in order.

    Matched off the second token rather than a substring check: SQLite
    renders ``UPDATE user_custom_apis SET ...``, and a plain ``"CUSTOM_APIS"
    in statement`` check would also match that table's own name, since it
    contains the shorter table name as a substring.
    """
    tables = []
    for statement in statements:
        tokens = statement.strip().split()
        if tokens and tokens[0].upper() == "UPDATE":
            tables.append(tokens[1].strip('"').upper())
    return tables


def _assert_is_active_edit_writes_link_row_only(
    *,
    seed_name: str,
    payload_kwargs: dict,
    expected_description,
    seed_description: str | None = None,
) -> None:
    """Shared body for the two ``is_active``-plus-sibling-field tests
    below: run the edit, assert the response and the persisted link row,
    then assert that no ``UPDATE`` reached ``custom_apis`` while exactly
    one reached ``user_custom_apis``.
    """
    session_factory, engine = _lock_order_session_factory()
    owner_id, api_id = _seed_owned_api_for_lock_order(
        session_factory, name=seed_name, description=seed_description
    )
    db = session_factory()
    current_user = SimpleNamespace(id=owner_id, is_admin=False)
    statements: list[str] = []

    def record_query(conn, cursor, statement, parameters, context, executemany):
        del conn, cursor, parameters, context, executemany
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", record_query)
    try:
        response = update_custom_api(
            api_id,
            CustomApiUpdate(is_active=False, **payload_kwargs),
            current_user=current_user,
            db=db,
        )
    finally:
        event.remove(engine, "before_cursor_execute", record_query)
        db.close()

    assert isinstance(response, CustomApiResponse)
    assert response.is_active is False

    with session_factory() as fresh:
        link = (
            fresh.query(UserCustomApi)
            .filter(
                UserCustomApi.custom_api_id == api_id,
                UserCustomApi.user_id == owner_id,
            )
            .one()
        )
        assert link.is_active is False
        api = fresh.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert api.name == seed_name
        assert api.description == expected_description

    updated_tables = _updated_table_names(statements)
    assert updated_tables.count("CUSTOM_APIS") == 0, (
        f"an is_active edit for {seed_name!r} must not write the "
        f"definition row -- saw UPDATEs against {updated_tables!r}"
    )
    assert updated_tables.count("USER_CUSTOM_APIS") == 1


def test_an_is_active_edit_with_an_explicit_null_description_writes_no_definition_field():
    """``model_fields_set`` decides ``writes_definition_row``, not the
    values -- so a payload pairing ``is_active`` with an explicit-null
    ``description`` still counts ``description`` into that set and takes
    the lock, even though the write below it is skipped by its own
    ``is not None`` guard. Despite taking the lock, only the caller's own
    link row may see an ``UPDATE``.

    The seed's own description is non-``None`` (``"seed-desc"``): if the
    guard were ever bypassed, assigning the payload's ``None`` over that
    would be an observable change, not a same-value no-op silently skipped
    by SQLAlchemy's own dirty-tracking.
    """
    _assert_is_active_edit_writes_link_row_only(
        seed_name="is-active-null-description",
        seed_description="seed-desc",
        payload_kwargs={"description": None},
        expected_description="seed-desc",
    )


def test_an_is_active_edit_repeating_the_current_name_writes_no_definition_field():
    """Same shape as the explicit-null-description case above, with
    ``name`` as the sibling field: it counts into ``fields_set`` and takes
    the lock, but the name-change guard (``api_data.name != api.name``) is
    false, so the definition row must still see no ``UPDATE``.
    """
    _assert_is_active_edit_writes_link_row_only(
        seed_name="is-active-repeated-name",
        payload_kwargs={"name": "is-active-repeated-name"},
        expected_description=None,
    )


class TestDeleteLockOrderMatchesThePutsLockOrder:
    """An ``update_custom_api`` call that writes the definition row locks
    it first, calls ``rename_team_connector`` after that lock, and writes
    the ``UserCustomApi`` link row after that. For the two routes to share one
    lock order, ``delete_custom_api`` must take the same definition-row
    lock before it calls ``delete_team_connector`` and before it deletes
    the link row, in both of its branches. The hook boundary matters as
    much as the two tables do: a hook that locks rows of its own sees both
    routes arrive holding the definition row, so the two cannot be inside
    the hook at the same time on the same connector.

    SQLite silently drops ``FOR UPDATE`` (it is a no-op on this dialect), so
    nothing here demonstrates that the lock actually blocks a second writer
    -- that proof lives in test_custom_api_edit_lock_postgresql.py, against
    a real server. What this proves instead is statement *order*, which is
    dialect-independent and exercisable without one.
    """

    def _run(self, *, team_owned: bool) -> list[str]:
        session_factory, engine = _lock_order_session_factory()
        owner_id, api_id = _seed_owned_api_for_lock_order(
            session_factory,
            name="lock-order-team" if team_owned else "lock-order-cascade",
        )
        db = session_factory()
        current_user = SimpleNamespace(id=owner_id, is_admin=False)

        statements: list[str] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", record_query)
        try:
            if team_owned:

                def deleted_hook(_db, _user_id, _connector_type, _connector_id):
                    return ConnectorDeleteDecision(
                        team_owned=True, authorized=True, delete_definition=True
                    )

                set_connector_team_hooks(deleted=deleted_hook)
                try:
                    delete_custom_api(api_id, current_user=current_user, db=db)
                finally:
                    set_connector_team_hooks()
            else:
                delete_custom_api(api_id, current_user=current_user, db=db)
        finally:
            event.remove(engine, "before_cursor_execute", record_query)
            db.close()
        return statements

    def test_lock_order_team_owned_branch(self):
        statements = self._run(team_owned=True)
        assert _count_custom_apis_selects_before_first_delete(statements) == 2, (
            "expected the not-found guard's relationship load AND the new "
            "lock statement's own SELECT against custom_apis, both before "
            "the first DELETE"
        )

    def test_lock_order_cascade_branch(self):
        statements = self._run(team_owned=False)
        assert _count_custom_apis_selects_before_first_delete(statements) == 2, (
            "expected the not-found guard's relationship load AND the new "
            "lock statement's own SELECT against custom_apis, both before "
            "the first DELETE"
        )

    def test_the_lock_is_taken_before_the_delete_hook_is_called(self):
        """``delete_team_connector`` runs with the definition row already
        held. That is what puts this route in the same direction as
        ``update_custom_api``, which calls ``rename_team_connector`` after
        its own definition-row lock, on the path that takes one -- and is
        therefore what stops the two from forming a cycle through rows the
        hook locks. The hook counts the
        ``custom_apis`` ``SELECT``s issued so far: the not-found guard's
        relationship load, plus the lock's own query.
        """
        session_factory, engine = _lock_order_session_factory()
        owner_id, api_id = _seed_owned_api_for_lock_order(
            session_factory, name="lock-order-before-hook"
        )
        db = session_factory()
        current_user = SimpleNamespace(id=owner_id, is_admin=False)

        statements: list[str] = []
        selects_seen_by_the_hook: list[int] = []

        def record_query(conn, cursor, statement, parameters, context, executemany):
            del conn, cursor, parameters, context, executemany
            statements.append(statement)

        def deleted_hook(_db, _user_id, _connector_type, _connector_id):
            selects_seen_by_the_hook.append(
                sum(
                    1
                    for statement in statements
                    if statement.strip().upper().startswith("SELECT")
                    and "FROM CUSTOM_APIS" in statement.strip().upper()
                )
            )
            return ConnectorDeleteDecision()

        event.listen(engine, "before_cursor_execute", record_query)
        set_connector_team_hooks(deleted=deleted_hook)
        try:
            delete_custom_api(api_id, current_user=current_user, db=db)
        finally:
            set_connector_team_hooks()
            event.remove(engine, "before_cursor_execute", record_query)
            db.close()

        assert selects_seen_by_the_hook == [2], (
            "expected the delete hook to be called once, with both the "
            "not-found guard's relationship load and the lock statement's "
            "own SELECT against custom_apis already issued"
        )


def _with_for_update_lines(fn: "ast.FunctionDef") -> list[int]:
    """Line numbers of every ``with_for_update(...)`` call inside ``fn``."""
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "with_for_update"
    ]


def _call_lines_by_name(fn: "ast.FunctionDef", name: str) -> list[int]:
    """Line numbers of every plain ``name(...)`` call inside ``fn``."""
    return [
        node.lineno
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == name
    ]


def _route_ast(name: str) -> "ast.FunctionDef":
    from xagent.web.api import custom_api as custom_api_api

    tree = ast.parse(inspect.getsource(custom_api_api))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} not found in xagent.web.api.custom_api")


@pytest.mark.parametrize(
    ("route", "hook"),
    [
        ("update_custom_api", "rename_team_connector"),
        ("delete_custom_api", "delete_team_connector"),
    ],
)
def test_the_definition_row_lock_statement_precedes_the_team_hook_call(route, hook):
    """``delete_custom_api`` always takes the definition-row lock before it
    calls into a connector team hook; ``update_custom_api`` takes that lock
    only for a payload that writes the definition row, and takes it before
    the hook when it does.

    Statement position is the thing under test, so it is read off the
    source rather than off executed SQL: SQLAlchemy renders no locking
    clause at all on SQLite, so a run against the suite's own engine
    cannot tell this lock statement from a plain read. That the clause
    really blocks a second
    writer is proved against a real server in
    test_custom_api_edit_lock_postgresql.py; what is proved here is the
    order the two routes share, which is what keeps a concurrent edit and
    delete of one connector from waiting on each other in opposite
    directions across the hook boundary.
    """
    fn = _route_ast(route)
    lock_lines = _with_for_update_lines(fn)
    assert len(lock_lines) == 1, (
        f"expected exactly one with_for_update call in {route}, found {len(lock_lines)}"
    )
    hook_lines = _call_lines_by_name(fn, hook)
    assert len(hook_lines) == 1, (
        f"expected exactly one {hook} call in {route}, found {len(hook_lines)}"
    )
    assert lock_lines[0] < hook_lines[0], (
        f"{route} must take the definition-row lock before it calls {hook}"
    )
