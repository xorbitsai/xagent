"""How a stored ``MCPServer`` row is resolved back to its catalog app.

``get_app_for_mcp_server`` decides which app's credentials a disconnect
deletes and which app a connector listing reports, so the rule it encodes is
load-bearing: identify the row by something stable, never by a value that is
both mutable and non-unique.

``PublicMCPApp.app_id`` is unique; ``PublicMCPApp.name`` is neither unique nor
immutable (the admin API can rename an app). Resolving by name therefore had
two failure modes, both fixed here and pinned below:

* an id-named row -- the convention the catalog connect helpers write --
  resolved to nothing, so callers silently skipped whatever they do with the
  result (for the disconnect path, deleting the user's OAuth credentials);
* a rename could move the answer to a *different* app between one caller's
  check and its later use.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from xagent.web.mcp_apps import get_app_for_mcp_server
from xagent.web.models.public_mcp import PublicMCPApp


@pytest.fixture()
def catalog_db():
    engine = create_engine("sqlite:///:memory:")
    PublicMCPApp.__table__.create(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def _app(db, app_id, name, **kwargs):
    db.add(
        PublicMCPApp(
            app_id=app_id,
            name=name,
            transport=kwargs.get("transport", "oauth"),
            provider_name=kwargs.get("provider_name", f"prov-{app_id}"),
            category="Email",
            oauth_scopes=[],
            is_visible_in_connector=kwargs.get("visible", True),
            launch_config={},
        )
    )
    db.commit()


class _Row:
    """The two attributes the resolver reads off a stored server row."""

    def __init__(self, name, auth=None):
        self.name = name
        self.auth = auth


class TestAnUnstampedRow:
    """Both provisioning conventions write ``MCPServer.name``: the catalog
    connect helpers store the app id, the builtin OAuth flow the display
    name. Neither may be unresolvable."""

    def test_a_display_named_row_resolves(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(catalog_db, _Row("Acme Mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_an_id_named_row_resolves(self, catalog_db):
        """The regression: this returned ``None``, so a disconnect of such a
        row reported success while leaving the OAuth credentials stored."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(catalog_db, _Row("acme-mail"))
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_a_name_matching_nothing_resolves_to_none(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        assert get_app_for_mcp_server(catalog_db, _Row("no-such-thing")) is None

    @pytest.mark.parametrize(
        "id_owner_first", [True, False], ids=["id-owner-first", "name-owner-first"]
    )
    def test_a_cross_namespace_collision_refuses(self, catalog_db, id_owner_first):
        """Two apps answer to this name -- one by ``app_id``, one by display
        name -- and the row itself says nothing about which provisioned it.

        Picking either would be a guess. ``app_id`` being unique proves only
        that at most one app carries that id; it does not prove this row came
        from that app rather than from the other app's display name, and the
        two readings are equally legal because both naming conventions are
        supported. An earlier revision of this file asserted the opposite
        ("the id must win") and would have locked in a resolver that deleted
        the wrong app's credentials for a legacy display-named row.

        Seeded in both catalog insertion orders: the refusal must not depend
        on scan order either.
        """
        if id_owner_first:
            _app(catalog_db, "acme-mail", "Acme Mail")
            _app(catalog_db, "legacy-app", "acme-mail", visible=False)
        else:
            _app(catalog_db, "legacy-app", "acme-mail", visible=False)
            _app(catalog_db, "acme-mail", "Acme Mail")

        assert get_app_for_mcp_server(catalog_db, _Row("acme-mail")) is None

    @pytest.mark.parametrize(
        "duplicate_first", [True, False], ids=["duplicate-first", "original-first"]
    )
    def test_two_apps_sharing_a_display_name_refuse(self, catalog_db, duplicate_first):
        """``PublicMCPApp.name`` has no uniqueness constraint, so the name
        alone can have two owners. The base resolver answered with whichever
        row an unordered ``.first()`` returned; both orders must refuse."""
        if duplicate_first:
            _app(catalog_db, "second-app", "Acme Mail", visible=False)
            _app(catalog_db, "acme-mail", "Acme Mail")
        else:
            _app(catalog_db, "acme-mail", "Acme Mail")
            _app(catalog_db, "second-app", "Acme Mail", visible=False)

        assert get_app_for_mcp_server(catalog_db, _Row("Acme Mail")) is None

    def test_one_app_matched_by_both_columns_still_resolves(self, catalog_db):
        """An app whose id and display name are the same string matches twice
        but has a single owner, so it is unambiguous and must resolve."""
        _app(catalog_db, "acme", "acme")
        resolved = get_app_for_mcp_server(catalog_db, _Row("acme"))
        assert resolved is not None and resolved["id"] == "acme"


class TestAStampedRow:
    def test_the_stamp_decides(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        _app(catalog_db, "other-app", "Other App")
        resolved = get_app_for_mcp_server(
            catalog_db, _Row("Other App", auth={"app_id": "acme-mail"})
        )
        assert resolved is not None and resolved["id"] == "acme-mail"

    @pytest.mark.parametrize(
        "stamp", [pytest.param("", id="empty"), pytest.param(7, id="non-string")]
    )
    def test_a_malformed_stamp_refuses_rather_than_falling_back(
        self, catalog_db, stamp
    ):
        """A present-but-invalid stamp must not fall back to the row's name:
        that fallback is how one connector's teardown selects another
        connector's credentials."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        assert (
            get_app_for_mcp_server(
                catalog_db, _Row("Acme Mail", auth={"app_id": stamp})
            )
            is None
        )

    def test_an_unknown_stamp_resolves_to_none(self, catalog_db):
        _app(catalog_db, "acme-mail", "Acme Mail")
        row = _Row("Acme Mail", auth={"app_id": "no-such-app"})
        assert get_app_for_mcp_server(catalog_db, row) is None

    def test_a_stamped_row_survives_a_rename_of_another_app_onto_its_name(
        self, catalog_db
    ):
        """The mutable-name race, at the resolver level: renaming app B onto
        the stored server name must not move a stamped row's answer to B."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        _app(catalog_db, "other-app", "Other App")
        row = _Row("acme-mail", auth={"app_id": "acme-mail"})

        renamed = (
            catalog_db.query(PublicMCPApp)
            .filter(PublicMCPApp.app_id == "other-app")
            .one()
        )
        renamed.name = "acme-mail"
        catalog_db.commit()

        resolved = get_app_for_mcp_server(catalog_db, row)
        assert resolved is not None and resolved["id"] == "acme-mail"

    def test_auth_without_an_app_id_key_takes_the_name_path(self, catalog_db):
        """Selection is by key presence, not truthiness: a provider-only blob
        carries no stamp and must still resolve by name."""
        _app(catalog_db, "acme-mail", "Acme Mail")
        resolved = get_app_for_mcp_server(
            catalog_db, _Row("Acme Mail", auth={"provider": "acme"})
        )
        assert resolved is not None and resolved["id"] == "acme-mail"


class TestTheChangedCallers:
    """Caller-level guards.

    The truth table above pins the resolver, but both production call sites
    could be reverted to the old name-only lookup without failing any of it.
    These exercise the real rows -- ``MCPServer``, ``UserMCPServer``,
    ``PublicMCPApp``, ``UserOAuth`` -- through the endpoints themselves.
    """

    @pytest.fixture()
    def db(self):
        from sqlalchemy.pool import StaticPool

        from xagent.web.models.database import Base

        engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        Base.metadata.create_all(bind=engine)
        with Session(engine) as session:
            yield session
        engine.dispose()

    def _connected_id_named_app(self, db):
        """A catalog app connected through the id-naming convention: the
        shared row is named after ``app_id`` while the display name differs.
        """
        from xagent.web.models.mcp import MCPServer, UserMCPServer
        from xagent.web.models.user import User
        from xagent.web.models.user_oauth import UserOAuth

        user = User(username="someone", password_hash="h", is_admin=False)
        db.add(user)
        db.commit()
        db.refresh(user)
        _app(db, "acme-mail", "Acme Mail Displayed")
        server = MCPServer(
            name="acme-mail",
            transport="oauth",
            managed=False,
            auth={"app_id": "acme-mail"},
        )
        db.add(server)
        db.flush()
        db.add(
            UserMCPServer(
                user_id=int(user.id),
                mcpserver_id=server.id,
                is_active=True,
                is_owner=True,
            )
        )
        db.add(
            UserOAuth(
                user_id=int(user.id),
                provider="acme-mail",
                access_token="live-token",
                email="someone@acme.example",
            )
        )
        db.commit()
        return user, int(server.id)

    async def test_teardown_deletes_the_credential_of_an_id_named_row(self, db):
        """The regression this PR exists for: resolving by display name found
        nothing for an id-named row, so the cleanup was skipped and the token
        outlived a successful teardown."""
        from xagent.web.api.mcp import delete_mcp_server
        from xagent.web.models.user_oauth import UserOAuth

        user, server_id = self._connected_id_named_app(db)

        await delete_mcp_server(server_id=server_id, current_user=user, db=db)

        assert db.query(UserOAuth).filter(UserOAuth.user_id == user.id).count() == 0

    async def test_teardown_keeps_an_unrelated_apps_credential(self, db):
        """Scope check: resolving the right app must not widen the deletion."""
        from xagent.web.api.mcp import delete_mcp_server
        from xagent.web.models.user_oauth import UserOAuth

        user, server_id = self._connected_id_named_app(db)
        _app(db, "other-app", "Other App")
        db.add(
            UserOAuth(
                user_id=int(user.id),
                provider="other-app",
                access_token="untouched",
                email="other@acme.example",
            )
        )
        db.commit()

        await delete_mcp_server(server_id=server_id, current_user=user, db=db)

        survivors = {o.provider for o in db.query(UserOAuth).all()}
        assert survivors == {"other-app"}

    def test_listing_enrichment_names_the_app_of_an_id_named_row(self, db):
        """``_enrich_oauth_server_info`` reported app_id, provider and the
        connected account as absent for every id-named row."""
        from xagent.web.api.mcp import _enrich_oauth_server_info
        from xagent.web.models.mcp import MCPServer

        user, server_id = self._connected_id_named_app(db)
        server = db.query(MCPServer).filter(MCPServer.id == server_id).one()

        app_id, provider, connected_account = _enrich_oauth_server_info(
            db, server, {"acme-mail": "someone@acme.example"}
        )

        assert app_id == "acme-mail"
        assert provider == "prov-acme-mail"
        assert connected_account == "someone@acme.example"
