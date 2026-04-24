from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from sqlalchemy.orm import Session

from xagent.web.api.custom_api import (
    CustomApiCreate,
    CustomApiUpdate,
    _process_env_vars,
    create_custom_api,
    delete_custom_api,
    get_custom_api,
    list_custom_apis,
    update_custom_api,
)
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.user import User


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

        # Test masked value without existing
        res_missing = _process_env_vars({"new_key": "********"}, existing)
        assert res_missing == {}


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
        name="new_api", description="desc", env={"k1": "v1"}, is_active=True
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
async def test_get_custom_api():
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

    res = await get_custom_api(10, current_user=user, db=db)
    assert res.id == 10
    assert res.name == "test_api"


@pytest.mark.asyncio
async def test_get_custom_api_not_found():
    db = MagicMock(spec=Session)
    user = User(id=1)
    db.query().filter().first.return_value = None

    with pytest.raises(HTTPException) as exc_info:
        await get_custom_api(99, current_user=user, db=db)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_update_custom_api():
    db = MagicMock(spec=Session)
    user = User(id=1)

    mock_api = CustomApi(
        id=10,
        name="old_name",
        env={"k1": "enc_old1"},
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

    api_data = CustomApiUpdate(name="new_name", env={"k1": "********", "k2": "v2"})

    with patch(
        "xagent.web.api.custom_api.encrypt_value", side_effect=lambda x: f"enc_{x}"
    ):
        await update_custom_api(10, api_data, current_user=user, db=db)

        assert mock_api.name == "new_name"
        assert mock_api.env == {"k1": "enc_old1", "k2": "enc_v2"}
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

    await delete_custom_api(10, current_user=user, db=db)

    db.delete.assert_called_once_with(mock_api)
    db.commit.assert_called()
