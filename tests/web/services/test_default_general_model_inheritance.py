"""The ``general`` model slot is filled on both agent write paths.

Server-side creation paths pass no model config (``workforce_creator``
hardcodes ``models=None``), which persisted as an unset slot: the builder
rendered "--" and its required-field guard refused to save, and vibe
delegation failed outright. Every create path reaches
``AgentStore.add_agent``; ``migration/loaders.py`` builds its ``Agent``
outside the store and calls the same helper itself.

The resolution order is the one the LLM-returning resolvers use: the user's
own default, then a visible user's shared default.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from xagent.migration.bundle import MigrationBundle, PersonaItem
from xagent.migration.loaders import MigrationLoader
from xagent.web.models.agent import Agent
from xagent.web.models.database import Base
from xagent.web.models.model import Model as DBModel
from xagent.web.models.user import User, UserDefaultModel, UserModel
from xagent.web.services.agent_store import AgentStore
from xagent.web.services.model_service import with_default_general_model


@pytest.fixture(autouse=True)
def _no_visibility_hook() -> Iterator[None]:
    """``set_visible_user_ids_hook`` is module-global and deployments install
    their own at import time, so pin the admin-default behaviour these cases
    assert on."""
    from xagent.web.services.model_service import set_visible_user_ids_hook

    set_visible_user_ids_hook(None)
    yield
    set_visible_user_ids_hook(None)


@pytest.fixture()
def db() -> Iterator[Session]:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def _user_row(db: Session, username: str, *, is_admin: bool = False) -> User:
    user = User(username=username, password_hash="x", is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def _user(db: Session, username: str, *, is_admin: bool = False) -> int:
    return int(_user_row(db, username, is_admin=is_admin).id)


def _model(db: Session, model_id: str, *, is_active: bool = True) -> int:
    model = DBModel(
        model_id=model_id,
        category="llm",
        model_provider="p",
        model_name=model_id,
        api_key="k",
        is_active=is_active,
    )
    db.add(model)
    db.commit()
    db.refresh(model)
    return int(model.id)


def _own(db: Session, user_id: int, model_pk: int, *, is_shared: bool = False) -> None:
    db.add(
        UserModel(
            user_id=user_id, model_id=model_pk, is_owner=True, is_shared=is_shared
        )
    )
    db.commit()


def _default(db: Session, user_id: int, model_pk: int) -> None:
    db.add(UserDefaultModel(user_id=user_id, model_id=model_pk, config_type="general"))
    db.commit()


@pytest.fixture()
def owner(db: Session) -> tuple[int, int]:
    """A user whose visible, active model is their ``general`` default."""
    user_id = _user(db, "owner")
    model_pk = _model(db, "gpt-x")
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)
    return user_id, model_pk


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        pytest.param(None, {"general": "PK"}, id="omitted_is_filled"),
        pytest.param({}, {"general": "PK"}, id="empty_dict_is_filled"),
        pytest.param(
            {"compact": 7}, {"compact": 7, "general": "PK"}, id="other_slots_kept"
        ),
        # An explicit None means "no main model"; a stated id is a choice.
        pytest.param({"general": None}, {"general": None}, id="explicit_none_kept"),
        pytest.param({"general": 42}, {"general": 42}, id="stated_choice_kept"),
        # Template YAML reaches this layer unvalidated; a typo must not 500.
        pytest.param("gpt-4", "gpt-4", id="non_dict_passes_through"),
    ],
)
def test_fills_only_an_omitted_slot(
    db: Session, owner: tuple[int, int], payload: Any, expected: Any
) -> None:
    user_id, model_pk = owner
    if isinstance(expected, dict):
        expected = {k: (model_pk if v == "PK" else v) for k, v in expected.items()}
    assert with_default_general_model(db, payload, user_id=user_id) == expected


def test_a_default_for_another_slot_is_not_used(db: Session) -> None:
    user_id = _user(db, "visual_only")
    model_pk = _model(db, "vision-llm")
    _own(db, user_id, model_pk)
    db.add(UserDefaultModel(user_id=user_id, model_id=model_pk, config_type="visual"))
    db.commit()
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_no_default_anywhere_leaves_config_unset(db: Session) -> None:
    user_id = _user(db, "no_default")
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_inactive_default_is_not_used(db: Session) -> None:
    user_id = _user(db, "inactive_default")
    model_pk = _model(db, "retired", is_active=False)
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_invisible_default_is_not_used(db: Session) -> None:
    """A default row survives the model becoming invisible (a demotion, a
    team move); injecting it would slip past ``_validate_models`` and leave
    the builder's Main Model blank while its save guard sees a value."""
    user_id = _user(db, "orphaned_default")
    stranger_id = _user(db, "stranger")
    model_pk = _model(db, "someone-elses")
    _own(db, stranger_id, model_pk)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_a_non_visible_sharer_does_not_qualify(db: Session) -> None:
    """``is_shared`` alone is not visibility: the sharer also has to be in the
    user's visible set, which ``_get_visible_user_ids`` resolves (admins by
    default, whatever the deployment's hook says otherwise)."""
    user_id = _user(db, "borrower")
    sharer_id = _user(db, "sharer")
    model_pk = _model(db, "shared-llm")
    _own(db, sharer_id, model_pk, is_shared=True)
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_a_visible_sharers_shared_default_is_inherited(db: Session) -> None:
    """A user whose only usable default is an admin-shared one must not be
    treated as having none -- that is the bug this PR fixes, on another path
    (``agent_tool.py`` already resolves through this fallback)."""
    user_id = _user(db, "no_own_default")
    admin_id = _user(db, "sharing_admin", is_admin=True)
    model_pk = _model(db, "admin-llm")
    _own(db, admin_id, model_pk, is_shared=True)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }


def test_an_unshared_admin_default_is_not_inherited(db: Session) -> None:
    user_id = _user(db, "outsider")
    admin_id = _user(db, "private_admin", is_admin=True)
    model_pk = _model(db, "admin-private-llm")
    _own(db, admin_id, model_pk, is_shared=False)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_inactive_shared_default_is_not_inherited(db: Session) -> None:
    user_id = _user(db, "late_comer")
    admin_id = _user(db, "retiring_admin", is_admin=True)
    model_pk = _model(db, "admin-retired-llm", is_active=False)
    _own(db, admin_id, model_pk, is_shared=True)
    _default(db, admin_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_add_agent_fills_the_slot(db: Session, owner: tuple[int, int]) -> None:
    user_id, model_pk = owner
    agent = AgentStore(db).add_agent(
        user_id=user_id, name="server-made", description=None, instructions=None
    )
    db.commit()
    db.refresh(agent)
    assert agent.models == {"general": model_pk}


def test_a_stranger_pointing_at_a_shared_model_does_not_qualify(
    db: Session,
) -> None:
    """The shared layer keys on who OWNS the default row, not on who shared
    the model -- otherwise any unrelated user adopting an admin-shared model
    as their own default would hand it to everyone."""
    user_id = _user(db, "bystander")
    admin_id = _user(db, "silent_admin", is_admin=True)
    stranger_id = _user(db, "stranger_with_taste")
    model_pk = _model(db, "adopted-llm")
    _own(db, admin_id, model_pk, is_shared=True)
    # The admin shares the model but never made it their own default.
    _default(db, stranger_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_own_default_wins_over_a_visible_one(db: Session) -> None:
    """Both layers are one query now, so the ordering is what keeps the
    user's own choice ahead of an inherited one."""
    user_id = _user(db, "has_own")
    admin_id = _user(db, "also_has_one", is_admin=True)
    own_pk = _model(db, "mine")
    admin_pk = _model(db, "theirs")
    _own(db, user_id, own_pk)
    _own(db, admin_id, admin_pk, is_shared=True)
    _default(db, user_id, own_pk)
    _default(db, admin_id, admin_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {"general": own_pk}


def test_one_visible_admin_s_default_on_another_s_shared_model(db: Session) -> None:
    """Owner and sharer both have to be visible, but need not be the same user.

    ``_is_model_visible_to_user`` considers this model visible, so requiring
    one user to be both would silently drop a usable default.
    """
    user_id = _user(db, "onlooker")
    naming_admin = _user(db, "admin_who_names", is_admin=True)
    sharing_admin = _user(db, "admin_who_shares", is_admin=True)
    model_pk = _model(db, "cross-admin-llm")
    _own(db, sharing_admin, model_pk, is_shared=True)
    _default(db, naming_admin, model_pk)

    from xagent.web.services.model_service import _is_model_visible_to_user

    assert _is_model_visible_to_user(db, model_pk, user_id) is True
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }


def test_a_default_row_does_not_borrow_a_stranger_s_sharing(db: Session) -> None:
    """Visibility of the model is judged for the caller, not for whoever named
    it as their default.

    A visible admin naming a model that only an *invisible* third party shares
    would otherwise inject something ``_is_model_visible_to_user`` rejects --
    the one thing this resolver must never do.
    """
    user_id = _user(db, "asker")
    admin_id = _user(db, "naming_admin", is_admin=True)
    outsider_id = _user(db, "invisible_sharer")
    model_pk = _model(db, "borrowed-llm")
    # Shared by someone outside the visible set; the admin never holds it.
    _own(db, outsider_id, model_pk, is_shared=True)
    _default(db, admin_id, model_pk)

    from xagent.web.services.model_service import _is_model_visible_to_user

    assert _is_model_visible_to_user(db, model_pk, user_id) is False
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_a_read_error_degrades_to_an_unset_slot(db: Session, monkeypatch) -> None:
    """A read failure must not turn into a failed create.

    The resolver runs inside ``add_agent`` before flush, and the create paths
    do not catch this: ``api/agents.py`` would 500 it, and
    ``workforce_creator``'s ``begin_nested`` retry catches IntegrityError
    alone. Degrading to an unset slot is the pre-existing behaviour anyway.
    """
    user_id = _user(db, "unlucky")
    model_pk = _model(db, "fine-llm")
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)

    import xagent.web.services.model_service as module

    def _boom(*_args: Any, **_kwargs: Any) -> list[int]:
        raise RuntimeError("connection reset")

    # Inside the savepoint, where a real read failure would land.
    monkeypatch.setattr(module, "_get_visible_user_ids", _boom)
    assert with_default_general_model(db, None, user_id=user_id) is None

    agent = AgentStore(db).add_agent(
        user_id=user_id, name="degraded", description=None, instructions=None
    )
    db.commit()
    db.refresh(agent)
    assert agent.models is None


def test_the_resolver_runs_inside_a_savepoint(db: Session, monkeypatch) -> None:
    """The savepoint is what makes the degrade real.

    A failed statement aborts the surrounding transaction on PostgreSQL, so
    swallowing the exception without rolling back would only move the error to
    the caller's next statement. SQLite does not abort, so what gets pinned
    here is the nesting itself rather than a recovery.
    """
    user_id = _user(db, "savepointed")
    model_pk = _model(db, "fine-llm")
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)

    import xagent.web.services.model_service as module

    nested: list[bool] = []
    real = module._query_default_model_id

    def _spy(db_: Session, *args: Any, **kwargs: Any) -> Any:
        nested.append(db_.in_nested_transaction())
        return real(db_, *args, **kwargs)

    monkeypatch.setattr(module, "_query_default_model_id", _spy)

    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }
    assert nested == [True]


def test_the_fallback_survives_a_nested_savepoint(db: Session) -> None:
    """``workforce_creator`` calls ``add_agent`` inside ``db.begin_nested()``;
    the resolver's own savepoint has to nest inside that one."""
    user_id = _user(db, "workforce_owner")
    model_pk = _model(db, "workforce-llm")
    _own(db, user_id, model_pk)
    _default(db, user_id, model_pk)

    with db.begin_nested():
        agent = AgentStore(db).add_agent(
            user_id=user_id, name="nested", description=None, instructions=None
        )
    db.commit()
    db.refresh(agent)
    assert agent.models == {"general": model_pk}


def test_an_invisible_own_default_falls_through_to_the_shared_layer(
    db: Session,
) -> None:
    """An own default pointing at a model the user can no longer see must not
    short-circuit the shared layer (``agent_tool`` skips such a slot and keeps
    filling from the shared defaults)."""
    user_id = _user(db, "demoted")
    stranger_id = _user(db, "former_teammate")
    admin_id = _user(db, "sharing_admin_2", is_admin=True)
    gone_pk = _model(db, "no-longer-visible")
    shared_pk = _model(db, "still-visible")
    _own(db, stranger_id, gone_pk)
    _own(db, admin_id, shared_pk, is_shared=True)
    _default(db, user_id, gone_pk)
    _default(db, admin_id, shared_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": shared_pk
    }


def _imported_agent(db: Session, user: User) -> Agent:
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.persona = PersonaItem(instructions="You are helpful.")
    MigrationLoader(db, user=user).load(bundle)
    return db.query(Agent).filter(Agent.user_id == user.id).one()


def test_imported_agent_inherits_the_owners_default(db: Session) -> None:
    """The loader builds its Agent outside ``add_agent``, so a fallback wired
    into the store alone would miss this path."""
    user = _user_row(db, "importer")
    model_pk = _model(db, "gpt-x")
    _own(db, int(user.id), model_pk)
    _default(db, int(user.id), model_pk)
    assert _imported_agent(db, user).models == {"general": model_pk}


def test_imported_agent_without_a_default_keeps_its_empty_config(db: Session) -> None:
    user = _user_row(db, "importer_no_default")
    assert _imported_agent(db, user).models == {}


def test_the_newest_visible_default_wins(db: Session) -> None:
    """``uq_user_default_model`` is per (user, config_type), so two visible
    admins can each hold a ``general`` default -- the ordering has to pick one
    deterministically rather than leave it to insertion order."""
    user_id = _user(db, "no_default_of_own")
    first_admin = _user(db, "admin_one", is_admin=True)
    second_admin = _user(db, "admin_two", is_admin=True)
    first_pk = _model(db, "older-shared")
    second_pk = _model(db, "newer-shared")
    _own(db, first_admin, first_pk, is_shared=True)
    _own(db, second_admin, second_pk, is_shared=True)
    _default(db, first_admin, first_pk)
    _default(db, second_admin, second_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": second_pk
    }


def test_a_legacy_non_owner_row_is_not_visibility(db: Session) -> None:
    """A borrowed row is not visibility. ``_is_model_visible_to_user`` -- what
    decides at run time whether a configured id loads -- requires ``is_owner``
    on an own row; ``build_user_model_visibility_filter`` does not, and
    ``model_store`` deletes such rows, so they have existed. Injecting one
    would leave the agent a ``general`` slot that cannot resolve."""
    user_id = _user(db, "holds_a_stale_row")
    stranger_id = _user(db, "gone_team")
    model_pk = _model(db, "left-behind")
    _own(db, stranger_id, model_pk)
    db.add(UserModel(user_id=user_id, model_id=model_pk, is_owner=False))
    db.commit()
    _default(db, user_id, model_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_another_config_type_resolves_on_its_own(db: Session) -> None:
    """Only ``general`` is reached through ``with_default_general_model``; the
    parameter is what lets the sibling resolvers converge here later."""
    from xagent.web.services.model_service import resolve_default_model_id

    user_id = _user(db, "visual_default")
    model_pk = _model(db, "vision-x")
    _own(db, user_id, model_pk)
    db.add(UserDefaultModel(user_id=user_id, model_id=model_pk, config_type="visual"))
    db.commit()
    assert resolve_default_model_id(db, user_id, "visual") == model_pk


def test_is_active_is_read_off_the_default_row(db: Session) -> None:
    """An unrelated active model must not stand in for the retired one the
    default actually points at."""
    user_id = _user(db, "retired_default_only")
    retired_pk = _model(db, "retired-llm", is_active=False)
    other_pk = _model(db, "unrelated-live")
    _own(db, user_id, retired_pk)
    _own(db, user_id, other_pk)
    _default(db, user_id, retired_pk)
    assert with_default_general_model(db, None, user_id=user_id) is None


def test_an_own_shared_row_matches_the_run_time_check(db: Session) -> None:
    """An admin is in their own visible set, so a non-owner own row that is
    shared satisfies ``_is_model_visible_to_user`` through its second step.
    Spelling the predicate out per-branch is what keeps the two agreeing --
    correcting the shared filter with an extra clause did not."""
    from xagent.web.services.model_service import _is_model_visible_to_user

    user_id = _user(db, "admin_borrower", is_admin=True)
    model_pk = _model(db, "shared-to-self")
    db.add(
        UserModel(user_id=user_id, model_id=model_pk, is_owner=False, is_shared=True)
    )
    db.commit()
    _default(db, user_id, model_pk)
    assert _is_model_visible_to_user(db, model_pk, user_id) is True
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": model_pk
    }


def test_a_newer_invisible_default_does_not_hide_an_older_visible_one(
    db: Session,
) -> None:
    """``uq_user_default_model`` is per user, so the newest-first ordering is
    only reached across users -- where the newest row can be one whose model is
    not shared. Visibility sits in the WHERE clause, so that row is not a
    candidate rather than a candidate that has to be skipped."""
    user_id = _user(db, "picks_the_older")
    older_admin = _user(db, "shares_properly", is_admin=True)
    newer_admin = _user(db, "keeps_it_private", is_admin=True)
    visible_pk = _model(db, "shared-llm")
    private_pk = _model(db, "unshared-llm")
    _own(db, older_admin, visible_pk, is_shared=True)
    _own(db, newer_admin, private_pk)
    _default(db, older_admin, visible_pk)
    _default(db, newer_admin, private_pk)
    assert with_default_general_model(db, None, user_id=user_id) == {
        "general": visible_pk
    }
