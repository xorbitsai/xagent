"""A Composio trip that reaches the wrong region fails closed (xagent-saas#2461).

With connector OAuth routing on, the edge routes Composio's verification and
denial trips (which carry no ``state``) by one fixed-name route cookie. A
member of workspaces in two regions who starts a connect in one region and
then another in the second, in one browser, overwrites that cookie, so the
first flow's verification reaches the second region.

Two databases stand in for the two regions here. They share every secret
(the JWT key, ``ENCRYPTION_KEY``, ``COMPOSIO_USER_ID_SECRET``) and one
Composio project, and both regions' first user has id 1, so the colliding
case derives the same Composio user id in both: the worst case.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app
from xagent.web.mcp_apps import ensure_builtin_oauth_server_definition
from xagent.web.models.actor_oauth_flow import ActorOAuthFlowState
from xagent.web.models.database import Base
from xagent.web.models.mcp import UserMCPServer
from xagent.web.models.public_mcp import PublicMCPApp
from xagent.web.models.user import User

from . import test_composio_facebook_connect as facebook_connect
from .test_composio_facebook_connect import (
    OWNER,
    Browser,
    _flow_cookies,
    _grants,
    _start,
    _verify_url,
)

composio = facebook_connect.composio
composio_env = facebook_connect.composio_env

OTHER_REGION_OWNER = "toby:workspace-9:member-alice"


def _region(path: Path) -> tuple[Session, User]:
    """One region's database, with the Facebook connector installed."""
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def _no_driver_transactions(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _emit_begin(connection):
        connection.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine, autoflush=False, autocommit=False)()
    user = User(username="workspace-account", password_hash="hash")
    db.add(user)
    app = get_builtin_public_mcp_app("facebook")
    assert app is not None
    db.add(PublicMCPApp(**app))
    db.commit()
    server = ensure_builtin_oauth_server_definition(db, app_id="facebook")
    db.add(
        UserMCPServer(
            user_id=int(user.id),
            mcpserver_id=int(server.id),
            is_owner=True,
            is_active=True,
        )
    )
    db.commit()
    db.refresh(user)
    return db, user


@pytest.fixture
def regions(tmp_path):
    sg = _region(tmp_path / "sg.db")
    au = _region(tmp_path / "au.db")
    assert sg[1].id == au[1].id == 1
    yield sg, au
    for db, _user in (sg, au):
        db.close()
        db.get_bind().dispose()


def _begin(db: Session, user: User, browser: Browser, owner: str) -> None:
    start = _start(db, user, browser, owner)
    assert browser.get(db, start.headers["location"]).status_code == 302


@pytest.mark.parametrize(
    "au_owner",
    [OWNER, OTHER_REGION_OWNER],
    ids=["same-composio-user-id", "different-composio-user-id"],
)
def test_a_verification_routed_to_another_region_binds_nothing(
    regions, composio_env, composio, au_owner
):
    (sg_db, sg_user), (au_db, au_user) = regions
    browser = Browser()
    _begin(sg_db, sg_user, browser, OWNER)
    sg_account = composio.created[-1]
    _begin(au_db, au_user, browser, au_owner)
    au_account = composio.created[-1]

    response = browser.get(au_db, _verify_url(composio, sg_account))

    assert response.status_code == 400
    assert _grants(au_db) == []
    assert _grants(sg_db) == []
    assert composio.paths("DELETE") == []
    # The other region's own flow is untouched and still completes, binding
    # only the account it created.
    assert au_db.query(ActorOAuthFlowState).count() == 1
    assert browser.get(au_db, _verify_url(composio, au_account)).status_code == 200
    assert [grant.access_token for grant in _grants(au_db)] == [au_account]
    assert _grants(sg_db) == []


def test_a_begin_in_another_region_ends_this_regions_flow_in_the_browser(
    regions, composio_env, composio
):
    """A flow whose row lives in another region's database is not one this
    region can verify, so the second region's begin clears its cookie: the
    first connect fails closed rather than waiting to be misrouted."""
    (sg_db, sg_user), (au_db, au_user) = regions
    browser = Browser()
    _begin(sg_db, sg_user, browser, OWNER)
    (sg_cookie,) = _flow_cookies(browser)

    _begin(au_db, au_user, browser, OTHER_REGION_OWNER)

    assert sg_cookie not in _flow_cookies(browser)
    assert len(_flow_cookies(browser)) == 1
    # The first region's own row is left for its expiry; nothing is spent.
    assert sg_db.query(ActorOAuthFlowState).count() == 1


def test_a_denial_routed_to_another_region_ends_nothing_there(
    regions, composio_env, composio
):
    (sg_db, sg_user), (au_db, au_user) = regions
    browser = Browser()
    _begin(sg_db, sg_user, browser, OWNER)
    sg_account = composio.created[-1]
    _begin(au_db, au_user, browser, OTHER_REGION_OWNER)

    response = browser.get(
        au_db,
        f"{facebook_connect.CALLBACK_URL}?status=failed&error=access_denied"
        f"&connected_account_id={sg_account}",
    )

    assert response.status_code == 400
    # The other region's own flow is not ended by a denial for an account
    # it did not create.
    assert au_db.query(ActorOAuthFlowState).count() == 1
    assert len(_flow_cookies(browser)) == 1
    assert _grants(au_db) == [] and _grants(sg_db) == []
