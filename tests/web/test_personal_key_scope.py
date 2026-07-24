"""Contract tests for the application-owned personal-key access seam."""

from types import SimpleNamespace

import pytest

from xagent.web.services.personal_key_scope import (
    PersonalKeyAccessScope,
    get_personal_key_access_scope,
    set_personal_key_scope_hook,
)


@pytest.fixture(autouse=True)
def _clear_scope_hook():
    set_personal_key_scope_hook(None)
    yield
    set_personal_key_scope_hook(None)


def test_default_scope_is_limited_to_the_actor():
    scope = get_personal_key_access_scope(None, SimpleNamespace(id=17))

    assert scope == PersonalKeyAccessScope(
        owner_user_ids=(17,), can_manage_others=False
    )


def test_hook_scope_always_includes_the_actor():
    set_personal_key_scope_hook(
        lambda _db, _actor: PersonalKeyAccessScope(
            owner_user_ids=(29,), can_manage_others=True
        )
    )

    scope = get_personal_key_access_scope(None, SimpleNamespace(id=17))

    assert scope == PersonalKeyAccessScope(
        owner_user_ids=(17, 29), can_manage_others=True
    )


def test_non_manager_hook_cannot_expose_other_owners():
    set_personal_key_scope_hook(
        lambda _db, _actor: PersonalKeyAccessScope(
            owner_user_ids=(17, 29), can_manage_others=False
        )
    )

    scope = get_personal_key_access_scope(None, SimpleNamespace(id=17))

    assert scope == PersonalKeyAccessScope(
        owner_user_ids=(17,), can_manage_others=False
    )


@pytest.mark.parametrize(
    ("owner_user_ids", "can_manage_others"),
    [
        ((29,), 1),
        ((29,), "yes"),
        ([29], True),
        ((True,), True),
        (("29",), True),
        ((29.0,), True),
        ((0,), True),
        ((-29,), True),
    ],
)
def test_malformed_hook_scope_fails_closed(owner_user_ids, can_manage_others):
    set_personal_key_scope_hook(
        lambda _db, _actor: PersonalKeyAccessScope(
            owner_user_ids=owner_user_ids,
            can_manage_others=can_manage_others,
        )
    )

    scope = get_personal_key_access_scope(None, SimpleNamespace(id=17))

    assert scope == PersonalKeyAccessScope(
        owner_user_ids=(17,), can_manage_others=False
    )


@pytest.mark.parametrize("error_type", [AttributeError, TypeError, ValueError])
def test_hook_contract_errors_fail_closed_to_the_actor(error_type):
    def _raise_contract_error(_db, _actor):
        raise error_type("invalid scope contract")

    set_personal_key_scope_hook(_raise_contract_error)

    scope = get_personal_key_access_scope(None, SimpleNamespace(id=17))

    assert scope == PersonalKeyAccessScope(
        owner_user_ids=(17,), can_manage_others=False
    )


def test_unexpected_hook_errors_remain_explicit():
    def _raise_policy_outage(_db, _actor):
        raise RuntimeError("policy backend unavailable")

    set_personal_key_scope_hook(_raise_policy_outage)

    with pytest.raises(RuntimeError, match="policy backend unavailable"):
        get_personal_key_access_scope(None, SimpleNamespace(id=17))
