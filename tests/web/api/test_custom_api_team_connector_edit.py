"""The edit right on a team-linked Custom API: ``GET``/``PUT
/api/custom-apis/{api_id}`` resolve a caller with no personal row through
the connector access hook instead of 404ing outright, ``can_edit`` falls
back to that verdict for a caller with no personal row, an ``is_active``
payload from such a caller rejects outright instead of writing a shadow
attribute the response then reads back, a raising hook surfaces as its
declared status rather than a 500, and the verdict is re-resolved once
more after the definition row's lock is taken, refusing the write if it
no longer grants what the pre-lock answer granted.

Every test installs the access hook through ``snapshot_connector_team_hooks``
so no hook state leaks between tests or into suites that run after this one.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from xagent.core.tools.adapters.vibe.connector_runtime import ConnectorRuntimeError
from xagent.web.api.custom_api import (
    CustomApiUpdate,
    get_custom_api,
    update_custom_api,
)
from xagent.web.api.mcp import _custom_api_to_mcp_response, _TeamOwnedUserApi
from xagent.web.models.custom_api import CustomApi, UserCustomApi
from xagent.web.models.database import Base
from xagent.web.models.user import User
from xagent.web.services.connector_team_scope import (
    ConnectorAccess,
    ConnectorHookSessionBoundaryError,
    set_connector_team_hooks,
    snapshot_connector_team_hooks,
)


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()


def _make_user(db, user_id: int, *, is_admin: bool = False) -> User:
    user = User(
        id=user_id, username=f"user-{user_id}", password_hash="x", is_admin=is_admin
    )
    db.add(user)
    db.commit()
    return user


def _make_owned_api(db, owner_id: int, *, name: str = "shared-api") -> CustomApi:
    api = CustomApi(name=name, url="https://example.test/api", method="GET")
    db.add(api)
    db.flush()
    db.add(
        UserCustomApi(
            user_id=owner_id,
            custom_api_id=api.id,
            is_owner=True,
            can_edit=True,
            can_delete=True,
            is_active=True,
        )
    )
    db.commit()
    return api


def _get(api_id, current_user, db):
    return get_custom_api(api_id, current_user=current_user, db=db)


def _put(api_id, payload, current_user, db):
    return update_custom_api(api_id, payload, current_user=current_user, db=db)


def _sequenced_access_hook(*answers):
    """An access hook that answers differently on successive calls, so a
    test can make the second (post-lock) resolution disagree with the
    first. ``None`` in the sequence means an empty answer -- the batch
    contract's way of saying "the caller's team does not link this". An
    entry that is an exception instance is raised instead of returned, so a
    test can make the second resolution fail outright. The last entry
    repeats for any further call. Records every call's ``refs`` on
    ``.calls`` so a test can pin how many round trips the route pays."""
    calls: list[object] = []

    def hook(db, user_id, refs):
        calls.append(refs)
        index = min(len(calls) - 1, len(answers) - 1)
        answer = answers[index]
        if isinstance(answer, BaseException):
            raise answer
        if answer is None:
            return {}
        return {ref: answer for ref in refs}

    hook.calls = calls
    return hook


class TestGateHelperOnGetAndPut:
    def test_get_404s_for_an_unrelated_user_with_no_link_and_no_team_access(self, db):
        owner = _make_user(db, 1)
        stranger = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=lambda db, user_id, refs: {})
            with pytest.raises(HTTPException) as exc:
                _get(api.id, stranger, db)
        assert exc.value.status_code == 404

    def test_get_returns_the_stand_in_for_a_team_member_with_no_personal_row(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = _get(api.id, member, db)

        assert response.id == api.id
        assert response.user_id == member.id

    def test_get_owner_behaviour_is_unchanged_with_no_hook_installed(self, db):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            response = _get(api.id, owner, db)

        assert response.id == api.id
        assert response.user_id == owner.id


class TestPutWiringForATeamEditor:
    def test_team_editor_edit_is_durable_and_creates_no_association_row(self, db):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = _put(
                api_id,
                CustomApiUpdate(description="edited by the team"),
                editor,
                db,
            )

        assert response.description == "edited by the team"

        # Durability, not staging -- a same-session query would still see
        # an uncommitted UPDATE even if the route never committed.
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == "edited by the team"

        # The edit did not fabricate a personal association for the team
        # editor -- that would be a get-or-create write on an
        # authorization path.
        assert (
            db.query(UserCustomApi).filter(UserCustomApi.user_id == editor.id).first()
            is None
        )

    def test_a_member_with_a_personal_row_edits_the_shared_config_durably(self, db):
        """A caller whose own personal row does not grant edit, widened by a
        granting team verdict. ``can_edit=False`` on the personal row is the
        point -- it is what keeps ``_resolve_custom_api_for_request``'s
        ``skip_resolution_when=lambda ua: bool(ua.can_edit)`` from
        short-circuiting before the verdict is even resolved."""
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="both-rows-custom-api")
        api_id = api.id
        db.add(
            UserCustomApi(
                user_id=member.id,
                custom_api_id=api_id,
                is_owner=False,
                can_edit=False,
                is_active=True,
            )
        )
        db.commit()

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            response = _put(
                api_id,
                CustomApiUpdate(description="widened-by-the-team"),
                member,
                db,
            )

        assert response.description == "widened-by-the-team"

        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == "widened-by-the-team"
        assert (
            db.query(UserCustomApi)
            .filter(
                UserCustomApi.user_id == member.id,
                UserCustomApi.custom_api_id == api_id,
            )
            .count()
            == 1
        )

    def test_view_only_team_member_cannot_tamper_the_shared_config(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                _put(
                    api.id,
                    CustomApiUpdate(description="should not land"),
                    member,
                    db,
                )
        assert exc.value.status_code == 403


def test_a_denying_verdict_stand_in_is_403_on_an_empty_payload_too(db):
    """The route's edit gate has no payload-shape carve-out: it requires the
    edit right for every payload, including one that sets no field at all, so
    a caller with no personal row whose team verdict denies edit is refused
    before anything is read or written. Pinned here so the gate cannot be
    narrowed to specific payload shapes without this failing.
    """
    owner = _make_user(db, 1)
    member = _make_user(db, 2)
    api = _make_owned_api(db, owner.id, name="denying-stand-in-target")
    api_id = api.id
    # Captured as plain values, not read off ``api`` after the call: ``api``
    # and the ``refreshed`` row below share the same identity-mapped Python
    # object in this session, so comparing one against the other after the
    # call would be comparing the object with itself and could never fail.
    original_name = str(api.name)
    original_description = str(api.description) if api.description is not None else None

    with snapshot_connector_team_hooks():
        set_connector_team_hooks(
            access=lambda db, user_id, refs: {
                ref: ConnectorAccess(team_owned=True, can_edit=False) for ref in refs
            }
        )
        with pytest.raises(HTTPException) as exc:
            _put(api_id, CustomApiUpdate(), member, db)
    assert exc.value.status_code == 403

    db.rollback()
    refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
    assert refreshed.name == original_name
    assert refreshed.description == original_description
    assert (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == member.id).count() == 0
    )


def test_a_granting_verdict_stand_in_succeeds_on_an_empty_payload_without_a_recheck(
    db,
):
    """The granting counterpart of the test above, and the one path that
    reaches a 200 while skipping both the definition row's lock and the
    post-lock re-authorization: an empty payload writes no field of the
    shared definition row, so it takes no lock, and having taken no lock it
    has no unbounded wait to re-establish the gate's decision across.

    The hook is sequenced to grant on its first answer and to deny on every
    later one. If the route ever re-resolved the verdict on this payload, the
    second answer would refuse the request with a 403; the request returning
    normally, together with the single recorded hook call, is what pins the
    skip.
    """
    owner = _make_user(db, 1)
    member = _make_user(db, 2)
    api = _make_owned_api(db, owner.id, name="granting-stand-in-target")
    api_id = api.id
    original_name = str(api.name)
    original_description = str(api.description) if api.description is not None else None

    hook = _sequenced_access_hook(
        ConnectorAccess(team_owned=True, can_edit=True),
        None,
    )
    with snapshot_connector_team_hooks():
        set_connector_team_hooks(access=hook)
        response = _put(api_id, CustomApiUpdate(), member, db)

    # 1. one resolution, not two: the post-lock re-authorization never ran.
    assert len(hook.calls) == 1

    # 2. the shared definition row is untouched -- read back from the
    # database after a rollback rather than off the in-session object.
    db.rollback()
    refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
    assert refreshed.name == original_name
    assert refreshed.description == original_description

    # 3. no personal link row was created for a caller who had none.
    assert (
        db.query(UserCustomApi).filter(UserCustomApi.user_id == member.id).count() == 0
    )

    # 4. the response is the stand-in's own view of the connector.
    assert response.id == api_id
    assert response.user_id == member.id


class TestIsActiveRejectionForAStandIn:
    def test_is_active_from_a_caller_with_no_personal_row_is_400_not_a_silent_drop(
        self, db
    ):
        owner = _make_user(db, 1)
        editor = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="unchanged-name")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
                }
            )
            with pytest.raises(HTTPException) as exc:
                _put(
                    api_id,
                    CustomApiUpdate(is_active=False),
                    editor,
                    db,
                )

        # 1. the declared status.
        assert exc.value.status_code == 400
        assert "personal connection" in str(exc.value.detail)

        # 2. nothing persisted -- the exception was raised before any
        # commit, so a same-session rollback-then-requery must still show
        # no personal association row for this caller.
        db.rollback()
        assert (
            db.query(UserCustomApi).filter(UserCustomApi.user_id == editor.id).first()
            is None
        )

        # 3. the response body does not claim the change -- the call
        # raised rather than returning, so no ``CustomApiResponse`` ever
        # left the route carrying an ``is_active`` value nothing wrote.
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == "unchanged-name"


class TestStandInFlagsAgreeAcrossBothResponseSurfaces:
    def test_a_stand_in_caller_gets_the_same_two_flags_from_both_surfaces(self, db):
        """``is_active`` and ``is_default`` live on the caller's own link row.
        A caller with no such row is answered from a stand-in association
        holding class constants, and two separate response constructors read
        it: the single-connector ``GET`` here, and the aggregate connector
        list's ``_custom_api_to_mcp_response``. Whatever those constants are,
        both surfaces must report the same pair for the same caller -- pinned
        as an equality rather than as literal ``True``/``False`` so that
        changing what a stand-in reports stays a one-line change in one place
        instead of a test failure that reads like a regression.
        """
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="two-surface-target")
        api_id = api.id

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(
                access=lambda db, user_id, refs: {
                    ref: ConnectorAccess(team_owned=True, can_edit=False)
                    for ref in refs
                }
            )
            detail = _get(api_id, member, db)

        definition = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        aggregate = _custom_api_to_mcp_response(
            definition, _TeamOwnedUserApi(int(member.id))
        )

        assert (
            db.query(UserCustomApi).filter(UserCustomApi.user_id == member.id).first()
            is None
        ), "the caller under test must have no personal link row"
        assert detail.is_active == aggregate.is_active
        assert detail.is_default == aggregate.is_default


class TestTypedErrorArm:
    """A raising hook still surfaces its declared status for a caller with
    no working personal row -- the verdict is genuinely the gate for that
    population and must stay fail-closed. An owner's row already decides
    ``GET``'s answer (it never reads the verdict at all) and ``PUT``'s
    (``can_edit`` is already ``True``), so neither ever calls the hook for
    an owner's row; that population is pinned separately, below, in
    ``TestOwnerIsImmuneToAHookFailure``."""

    def test_get_surfaces_a_raising_hooks_declared_status(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                _get(api.id, member, db)

        assert exc.value.status_code == 503

    def test_put_surfaces_a_raising_hooks_declared_status_and_leaves_the_row_unchanged(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="pristine")
        api_id = api.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                _put(
                    api_id,
                    CustomApiUpdate(name="should-not-land"),
                    member,
                    db,
                )

        assert exc.value.status_code == 503

        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == "pristine"

    def test_put_passes_through_a_planted_connector_runtime_error_by_its_own_status(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id)

        def boom(*_a, **_k):
            raise ConnectorRuntimeError("planted", "planted failure", status_code=409)

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            with pytest.raises(HTTPException) as exc:
                _put(
                    api.id,
                    CustomApiUpdate(description="irrelevant"),
                    member,
                    db,
                )

        assert exc.value.status_code == 409
        assert exc.value.detail == "planted failure"

    def test_a_raising_rename_hook_surfaces_its_declared_status_not_a_500(self, db):
        """The rename hook runs after the definition row has already been
        rewritten in the session. Its typed failure must reach the client as
        the status the seam declares, not as a generic 500, and the staged
        rename must not survive the refusal."""
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id, name="rename-hook-raises")
        api_id = api.id
        original_name = str(api.name)

        def boom(*_a, **_k):
            raise ConnectorRuntimeError(
                "connector_runtime_unavailable",
                "Connector team scope is unavailable.",
                status_code=503,
            )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(renamed=boom)
            with pytest.raises(HTTPException) as exc:
                update_custom_api(
                    api_id,
                    CustomApiUpdate(name="renamed-by-the-test"),
                    current_user=owner,
                    db=db,
                )

        assert exc.value.status_code == 503
        # Zero side effects: the rename that triggered the hook is rolled
        # back with everything else this request had staged.
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name


class TestOwnerIsImmuneToAHookFailure:
    """An owner's row already decides both routes' answers on its own --
    ``GET`` never reads the verdict at all, and ``PUT``'s ``can_edit`` is
    already ``True`` -- so neither ever calls the hook for an owner's row.
    A hook that would raise must therefore never surface: both routes
    return their normal success status, unaffected by whatever the hook
    would have done."""

    def test_get_and_put_succeed_for_an_owner_even_though_the_hook_would_raise(
        self, db
    ):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id, name="owner-immune")
        api_id = api.id

        def boom(*_a, **_k):
            raise ValueError("hook exploded")

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=boom)
            get_response = _get(api_id, owner, db)
            put_response = _put(
                api_id,
                CustomApiUpdate(description="edited by the owner"),
                owner,
                db,
            )

        assert get_response.id == api_id
        assert put_response.description == "edited by the owner"


class TestTheVerdictIsRevalidatedUnderTheDefinitionLock:
    """The verdict granting a caller edit access is resolved before the
    route's definition-row lock exists, and the installing application can
    revoke the link at any moment through its own tables, which that lock
    does not cover. Every payload below writes the shared definition row, so
    every one of them takes the lock and must therefore re-establish the
    verdict under it before committing.
    """

    def _run(self, db, *, hook):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="revalidated-under-lock")
        api_id = api.id
        # Captured as plain values before the call, not read off ``api``
        # afterwards: ``api`` and the requery below share the same
        # identity-mapped Python object in this session, so comparing one
        # against the other after the call would be comparing the object
        # with itself and could never fail.
        original_name = str(api.name)
        original_description = (
            str(api.description) if api.description is not None else None
        )

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            result = {}
            try:
                result["response"] = _put(
                    api_id,
                    CustomApiUpdate(description="edited-while-in-flight"),
                    member,
                    db,
                )
            except HTTPException as exc:
                result["error"] = exc
            return api, api_id, result, original_name, original_description

    def test_revoked_between_resolution_and_lock_is_refused(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True), None
        )
        _api, api_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0

    def test_downgraded_to_not_editable_between_resolution_and_lock_is_refused(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=False),
        )
        _api, api_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 403
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0

    def test_still_granted_on_recheck_commits_durably(self, db):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ConnectorAccess(team_owned=True, can_edit=True),
        )
        (
            _api,
            api_id,
            result,
            _original_name,
            _original_description,
        ) = self._run(db, hook=hook)

        assert "error" not in result
        assert result["response"].description == "edited-while-in-flight"

        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == "edited-while-in-flight"

    def test_recheck_that_raises_surfaces_the_hooks_own_status_with_zero_side_effects(
        self, db
    ):
        hook = _sequenced_access_hook(
            ConnectorAccess(team_owned=True, can_edit=True),
            ValueError("hook exploded during recheck"),
        )
        _api, api_id, result, original_name, original_description = self._run(
            db, hook=hook
        )

        assert result["error"].status_code == 503
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.name == original_name
        assert refreshed.description == original_description
        assert db.query(UserCustomApi).filter(UserCustomApi.user_id == 2).count() == 0


class TestTheRecheckCostsExactlyOneExtraHookCall:
    """How many times a single request calls the application's access hook.
    A caller admitted by a team verdict pays one call at the gate and one
    more under the lock; a caller whose own link row already grants edit
    pays none at all."""

    def test_a_granting_stand_in_editing_the_shared_config_pays_two_calls(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="cost-stand-in-shared")
        api_id = api.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            _put(api_id, CustomApiUpdate(description="shared-edit"), member, db)

        assert len(hook.calls) == 2

    def test_an_owner_pays_zero_calls(self, db):
        owner = _make_user(db, 1)
        api = _make_owned_api(db, owner.id, name="cost-owner")
        api_id = api.id
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            _put(api_id, CustomApiUpdate(description="owner-edit"), owner, db)

        assert len(hook.calls) == 0


class TestPostLockRecheckDeclaresTheLock:
    def test_a_hook_that_commits_on_the_post_lock_call_is_refused_as_a_boundary_violation(
        self, db
    ):
        """The post-lock re-check declares ``caller_holds_lock=True`` on the
        call it makes through ``_recheck_team_access_under_definition_lock``: a hook
        that ends this request's own transaction while it holds the
        definition row's lock must be refused as the seam's own boundary
        violation, not answered as if the lock were still intact.
        """
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="boundary-probe")
        api_id = api.id
        original_description = (
            str(api.description) if api.description is not None else None
        )

        calls: list[object] = []

        def hook(db, user_id, refs):
            calls.append(refs)
            if len(calls) == 2:
                db.commit()
            return {
                ref: ConnectorAccess(team_owned=True, can_edit=True) for ref in refs
            }

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            with pytest.raises(ConnectorHookSessionBoundaryError):
                _put(
                    api_id,
                    CustomApiUpdate(description="should-not-land"),
                    member,
                    db,
                )

        assert len(calls) == 2
        db.rollback()
        refreshed = db.query(CustomApi).filter(CustomApi.id == api_id).one()
        assert refreshed.description == original_description


class TestHookCallCountAcrossPayloadShapes:
    """How many times a single request calls the access hook, across the
    two shapes not already covered by ``TestTheRecheckCostsExactlyOneExtraHookCall``
    above: a payload that writes only ``is_active`` never takes the lock, so
    a personal row that needs the team verdict to grant the edit asks once,
    at the gate, and never again; and a deployment with no access hook
    installed asks nothing at either point, because the hook slot being
    empty answers ``{}`` without ever reaching the ``hook`` object a test
    installs. The other two shapes -- an owner's row that already grants
    the edit, and a stand-in caller who writes the shared definition row --
    are the same call counts ``TestTheRecheckCostsExactlyOneExtraHookCall``
    already covers above.
    """

    def test_is_active_only_payload_from_a_can_edit_false_personal_row_pays_one_call(
        self, db
    ):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="count-is-active-only")
        api_id = api.id
        db.add(
            UserCustomApi(
                user_id=member.id,
                custom_api_id=api_id,
                is_owner=False,
                can_edit=False,
                is_active=True,
            )
        )
        db.commit()
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            _put(api_id, CustomApiUpdate(is_active=False), member, db)

        assert len(hook.calls) == 1

    def test_no_hook_installed_pays_zero_calls_regardless_of_payload(self, db):
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="count-no-hook")
        api_id = api.id
        # Never installed through ``set_connector_team_hooks``, so nothing
        # in the seam can reach it; used only to prove that fact rather
        # than to answer anything.
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks()
            with pytest.raises(HTTPException) as exc:
                _put(
                    api_id,
                    CustomApiUpdate(description="stand-in-write"),
                    member,
                    db,
                )

        # No access hook installed means the caller's team can never be
        # shown to link this connector, so the pre-lock resolution itself
        # refuses with the same 404 an unrelated caller with no personal
        # row has always gotten, before this route's own 403 is reached.
        assert exc.value.status_code == 404
        assert len(hook.calls) == 0


class TestReachablePathAWritesIsActiveWithoutRetakingTheLock:
    def test_a_can_edit_false_personal_row_widened_by_the_team_writes_is_active_in_one_call(
        self, db
    ):
        """The reachable reading of the gap this route used to have: a
        caller with a personal row whose own ``can_edit`` is ``False``, who
        can still write ``is_active`` because the team verdict grants the
        edit. The payload sets no field of the shared definition row, so it
        never takes that row's lock and never runs the post-lock re-check
        -- there is nothing to re-establish, because nothing that could
        move under an unbounded wait was ever locked.
        """
        owner = _make_user(db, 1)
        member = _make_user(db, 2)
        api = _make_owned_api(db, owner.id, name="reachable-path-a")
        api_id = api.id
        db.add(
            UserCustomApi(
                user_id=member.id,
                custom_api_id=api_id,
                is_owner=False,
                can_edit=False,
                is_active=True,
            )
        )
        db.commit()
        hook = _sequenced_access_hook(ConnectorAccess(team_owned=True, can_edit=True))

        with snapshot_connector_team_hooks():
            set_connector_team_hooks(access=hook)
            response = _put(api_id, CustomApiUpdate(is_active=False), member, db)

        assert response.is_active is False
        assert len(hook.calls) == 1

        db.rollback()
        link = (
            db.query(UserCustomApi)
            .filter(
                UserCustomApi.user_id == member.id,
                UserCustomApi.custom_api_id == api_id,
            )
            .one()
        )
        assert link.is_active is False
