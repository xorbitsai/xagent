"""Tests for adding the Slack channels:join OAuth scope."""

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260825_add_slack_channels_join_scope.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_slack_channels_join_scope_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, scopes: list[str], description: str | None = None):
    connection.execute(
        text(
            """
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                description TEXT,
                oauth_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO public_mcp_apps (app_id, description, oauth_scopes) "
            "VALUES ('slack', :description, :scopes)"
        ),
        {"description": description, "scopes": json.dumps(scopes)},
    )


def _create_oauth_providers_table(connection, default_scopes: list[str]):
    connection.execute(
        text(
            """
            CREATE TABLE oauth_providers (
                id INTEGER PRIMARY KEY,
                provider_name VARCHAR(50) NOT NULL UNIQUE,
                default_scopes JSON
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO oauth_providers (provider_name, default_scopes) "
            "VALUES ('slack', :scopes), ('hubspot', :other_scopes)"
        ),
        {
            "scopes": json.dumps(default_scopes),
            "other_scopes": json.dumps(["oauth"]),
        },
    )


def _create_user_oauth_table(connection):
    connection.execute(
        text(
            """
            CREATE TABLE user_oauth (
                id INTEGER PRIMARY KEY,
                user_id INTEGER NOT NULL,
                provider VARCHAR(50) NOT NULL,
                access_token VARCHAR NOT NULL
            )
            """
        )
    )
    connection.execute(
        text(
            "INSERT INTO user_oauth (user_id, provider, access_token) "
            "VALUES (1, 'slack', 'old-slack-token'), (1, 'hubspot', 'old-hubspot-token')"
        )
    )


def _access_tokens(connection):
    return dict(
        connection.execute(text("SELECT provider, access_token FROM user_oauth")).all()
    )


def _row(connection):
    row = connection.execute(
        text(
            "SELECT description, oauth_scopes FROM public_mcp_apps WHERE app_id='slack'"
        )
    ).one()
    description, scopes = row[0], row[1]
    return description, json.loads(scopes) if isinstance(scopes, str) else scopes


def _scopes(connection):
    return _row(connection)[1]


def _provider_default_scopes(connection, provider_name: str):
    scopes = connection.execute(
        text("SELECT default_scopes FROM oauth_providers WHERE provider_name = :p"),
        {"p": provider_name},
    ).scalar()
    return json.loads(scopes) if isinstance(scopes, str) else scopes


def test_upgrade_adds_channels_join_scope_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert scopes == migration.CURRENT_SCOPES
        assert "channels:join" in scopes
        assert description == migration.CURRENT_DESCRIPTION


def test_upgrade_is_idempotent(tmp_path):
    """Re-run as two genuinely separate migration runs (each its own
    transaction), not two calls stacked inside one uncommitted transaction —
    the latter wouldn't exercise re-running against already-persisted state,
    which is what idempotency is actually about."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_upgrade_does_not_disconnect_existing_user_oauth_grants(tmp_path):
    """Slack doesn't retroactively grant a newly-requested scope to an
    already-issued token, but it also doesn't revoke what that token could
    already do — adding channels:join must not force every existing user to
    reconnect just to keep using the tools they already had access to."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_user_oauth_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        tokens = _access_tokens(connection)
        assert tokens["slack"] == "old-slack-token"
        assert tokens["hubspot"] == "old-hubspot-token"


def test_downgrade_restores_previous_scopes_and_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        description, scopes = _row(connection)
        assert scopes == migration.PREVIOUS_SCOPES
        assert description == migration.PREVIOUS_DESCRIPTION


def test_upgrade_downgrade_upgrade_round_trip(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_upgrade_without_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when the table doesn't exist


def test_upgrade_preserves_customized_description(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES, "A custom description")
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        description, scopes = _row(connection)
        assert description == "A custom description"
        # oauth_scopes is builtin-protected, so it always updates regardless
        # of the description guard — see _set_slack_scopes's own docstring.
        assert scopes == migration.CURRENT_SCOPES


def test_downgrade_preserves_customized_description(tmp_path):
    """Mirrors test_downgrade_preserves_customized_provider_default_scopes:
    an operator customization made after upgrade() must survive downgrade()
    too, not just upgrade()."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.PREVIOUS_SCOPES, migration.PREVIOUS_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE public_mcp_apps SET description = :d WHERE app_id = 'slack'"
                ),
                {"d": "A custom description"},
            )
            migration.downgrade()
        description, _ = _row(connection)
        assert description == "A custom description"


def test_upgrade_updates_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # Set comparison, not list equality: the merge appends this
        # migration's delta rather than reproducing CURRENT_SCOPES's exact
        # order, and scope order has no OAuth significance.
        assert set(_provider_default_scopes(connection, "slack")) == set(
            migration.CURRENT_SCOPES
        )
        # A different provider row must be untouched.
        assert _provider_default_scopes(connection, "hubspot") == ["oauth"]


def test_downgrade_restores_provider_default_scopes(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert (
            _provider_default_scopes(connection, "slack") == migration.PREVIOUS_SCOPES
        )


def test_upgrade_without_oauth_providers_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise when oauth_providers is missing


def test_downgrade_without_oauth_providers_table_is_a_noop(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(
            connection, migration.CURRENT_SCOPES, migration.CURRENT_DESCRIPTION
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()  # must not raise when oauth_providers is missing


def test_upgrade_without_default_scopes_column_on_oauth_providers_is_a_noop(
    tmp_path,
):
    """oauth_providers can exist without a default_scopes column (an older
    schema, or a reduced admin deployment) -- _columns_present's guard must
    make this a no-op, not a crash on a column that isn't there."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE
                )
                """
            )
        )
        connection.execute(
            text("INSERT INTO oauth_providers (provider_name) VALUES ('slack')")
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise despite the missing column


def test_upgrade_merges_channels_join_into_customized_provider_default_scopes(
    tmp_path,
):
    """The old "skip unless it exactly equals PREVIOUS_SCOPES" guard
    permanently dropped channels:join from the app-id-less authorize path
    for any workspace that had ever customized this column — merging by
    delta instead must add channels:join while still preserving the
    customization, not silently keep the list frozen forever."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, ["chat:write", "custom:scope"])
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # Exact set, not just "these are present": a regression that merged
        # in the whole CURRENT_SCOPES list (e.g. add_scopes=CURRENT_SCOPES
        # instead of the delta _ADDED_SCOPES) would still contain
        # "chat:write"/"channels:join" and pass a presence-only check, but
        # would also pollute the row with channels:read/groups:read/etc.
        # that were never part of this customization.
        assert set(_provider_default_scopes(connection, "slack")) == {
            "chat:write",
            "custom:scope",
            "channels:join",
        }
        # The app-facing oauth_scopes column is unaffected by this guard —
        # it always updates (see _set_slack_scopes's own docstring).
        assert _scopes(connection) == migration.CURRENT_SCOPES


def test_downgrade_removes_channels_join_but_preserves_other_customization(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            connection.execute(
                text(
                    "UPDATE oauth_providers SET default_scopes = :s "
                    "WHERE provider_name = 'slack'"
                ),
                {"s": json.dumps(["chat:write", "custom:scope", "channels:join"])},
            )
            migration.downgrade()
        # Exact set: the row was overwritten to exactly these 3 scopes
        # right before downgrade() ran, so removal must strip exactly
        # channels:join and nothing else — not the whole
        # PREVIOUS_SCOPES-to-CURRENT_SCOPES delta by accident, and must
        # not touch the unrelated customization either.
        assert set(_provider_default_scopes(connection, "slack")) == {
            "chat:write",
            "custom:scope",
        }


def test_upgrade_updates_provider_default_scopes_when_no_row_exists(tmp_path):
    """No 'slack' provider row means there's nothing to merge into — must
    not raise, and must not fabricate a row."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    default_scopes JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise with no matching row
        assert _provider_default_scopes(connection, "slack") is None


def test_upgrade_merges_into_a_row_with_null_default_scopes(tmp_path):
    """A "slack" row can exist with default_scopes literally NULL (the
    column is nullable, and the admin PATCH endpoint's schema allows
    clearing it) -- that has no informative starting value to merge a delta
    into, so upgrade must seed the full CURRENT_SCOPES set (mirroring
    20260812's equivalent None-handling), not write only this migration's
    own ["channels:join"] delta and leave every other scope missing from
    the app-id-less authorize path."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    default_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO oauth_providers (provider_name, default_scopes) "
                "VALUES ('slack', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        # Exact list, not just set equality: this branch does
        # `list(full_scopes_if_null)`, a verbatim order-preserving copy, so
        # an exact comparison is strictly stronger and just as easy to
        # write — it also catches an accidental reorder/dedup that a set
        # comparison would silently let through.
        assert _provider_default_scopes(connection, "slack") == migration.CURRENT_SCOPES


def test_downgrade_merges_into_a_row_with_null_default_scopes(tmp_path):
    """Symmetric with the upgrade case above: a NULL default_scopes row has
    no delta to remove channels:join from either, so downgrade must seed
    the full PREVIOUS_SCOPES set rather than leaving an empty list."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.CURRENT_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    default_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO oauth_providers (provider_name, default_scopes) "
                "VALUES ('slack', NULL)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        # Exact list -- see the upgrade counterpart's comment above.
        assert (
            _provider_default_scopes(connection, "slack") == migration.PREVIOUS_SCOPES
        )


def test_upgrade_preserves_unparsable_default_scopes_without_overwriting(tmp_path):
    """default_scopes can hold a JSON string that isn't itself a scope list
    (e.g. double-encoded, or written by something other than this
    migration's own list writes) -- SQLAlchemy's JSON column type happily
    deserializes the column's outer JSON into that Python str, but this
    function's own json.loads(current) on the str value then fails. That
    must leave the row untouched rather than coercing to an empty list and
    overwriting it — either a delta-merge or a full write from that empty
    list would silently discard whatever was actually stored, taking away
    the operator's only chance to notice and fix it. (A raw, non-JSON-at-all
    string can't be used here: SQLAlchemy's own row deserializer raises
    before this function's code ever runs, which is exercised by the
    coverage this test's docstring intentionally does not claim.)"""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    default_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO oauth_providers (provider_name, default_scopes) "
                "VALUES ('slack', :scopes)"
            ),
            {"scopes": json.dumps("not-a-scope-list")},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise
        raw = connection.execute(
            text(
                "SELECT default_scopes FROM oauth_providers "
                "WHERE provider_name = 'slack'"
            )
        ).scalar()
        assert json.loads(raw) == "not-a-scope-list"


def test_upgrade_preserves_default_scopes_that_decode_to_a_non_list_value(tmp_path):
    """default_scopes can hold JSON that parses successfully but isn't a
    list at all (a JSON object here) -- the malformed-JSON-string branch
    above only guards the "fails to parse" case; a value that parses fine
    but isn't a list must get the same warn-and-skip treatment, not
    silently fall through to being treated as an empty list and then
    overwritten with just ["channels:join"], discarding whatever was
    actually stored."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        connection.execute(
            text(
                """
                CREATE TABLE oauth_providers (
                    id INTEGER PRIMARY KEY,
                    provider_name VARCHAR(50) NOT NULL UNIQUE,
                    default_scopes JSON
                )
                """
            )
        )
        connection.execute(
            text(
                "INSERT INTO oauth_providers (provider_name, default_scopes) "
                "VALUES ('slack', :scopes)"
            ),
            {"scopes": json.dumps({"unexpectedly": "an object"})},
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()  # must not raise
        raw = connection.execute(
            text(
                "SELECT default_scopes FROM oauth_providers "
                "WHERE provider_name = 'slack'"
            )
        ).scalar()
        assert json.loads(raw) == {"unexpectedly": "an object"}


def test_upgrade_merges_into_a_customization_committed_before_this_migration_runs(
    tmp_path,
):
    """Regression coverage for the race this guard is meant to survive: an
    admin customization already committed to the row before upgrade() ever
    reads it must not be lost, and channels:join must still be added on top
    of it — exercising the same merge path a customization landing between
    a naive SELECT and UPDATE would need to survive."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            connection.execute(
                text(
                    "UPDATE oauth_providers SET default_scopes = :s "
                    "WHERE provider_name = 'slack'"
                ),
                {"s": json.dumps([*migration.PREVIOUS_SCOPES, "custom:scope"])},
            )
            migration.upgrade()
        # Exact set: must add exactly channels:join on top of the
        # pre-existing customization, not the whole CURRENT_SCOPES list.
        assert set(_provider_default_scopes(connection, "slack")) == set(
            migration.PREVIOUS_SCOPES
        ) | {"custom:scope", "channels:join"}


def test_merge_provider_default_scopes_is_idempotent(tmp_path):
    """Re-run as two genuinely separate migration runs (each its own
    transaction, not two upgrade() calls stacked inside one uncommitted
    transaction) must not duplicate channels:join or otherwise change an
    already-merged list."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, migration.PREVIOUS_SCOPES)
        _create_oauth_providers_table(connection, migration.PREVIOUS_SCOPES)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        scopes = _provider_default_scopes(connection, "slack")
        assert scopes.count("channels:join") == 1


def test_migration_fields_match_registry():
    from xagent.web.builtin_mcp_registry import (
        get_builtin_oauth_provider_rows,
        get_builtin_public_mcp_app_rows,
    )

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_row["oauth_scopes"]
    assert migration.CURRENT_DESCRIPTION == registry_row["description"]
    registry_provider = next(
        r for r in get_builtin_oauth_provider_rows() if r["provider_name"] == "slack"
    )
    assert migration.CURRENT_SCOPES == registry_provider["default_scopes"]


def test_previous_fields_match_prior_migrations_current_fields():
    """This migration's PREVIOUS_SCOPES/PREVIOUS_DESCRIPTION are hand-copied
    from 20260812_add_slack_history_reactions_files_scopes's CURRENT_SCOPES/
    CURRENT_DESCRIPTION, not derived from it -- nothing else checks that copy
    stays correct. 20260812's own test file used to assert its CURRENT_*
    fields exactly match the live registry, which transitively caught this
    file's PREVIOUS_* going stale too; this PR loosened that assertion to a
    subset check (and dropped the description check outright) since 20260825
    now layers changes on top of the registry's final values. Without this
    test, a future edit to either migration's constants that breaks the
    handoff would pass every existing test while permanently breaking
    _set_slack_description_if_unchanged's `WHERE description ==
    expected_current` guard on any real database that already ran
    20260812 -- the description would never get updated by this migration."""
    prior_migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260812_add_slack_history_reactions_files_scopes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_slack_history_reactions_files_scopes_migration", prior_migration_file
    )
    prior_migration = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(prior_migration)

    migration = _load_migration_module()
    assert migration.PREVIOUS_SCOPES == prior_migration.CURRENT_SCOPES
    assert migration.PREVIOUS_DESCRIPTION == prior_migration.CURRENT_DESCRIPTION


def test_upgrade_after_actually_running_20260812_upgrade(tmp_path):
    """Every other test in this file constructs 20260825's starting fixture
    directly (e.g. _create_oauth_providers_table(connection,
    migration.PREVIOUS_SCOPES)) rather than by running
    20260812_add_slack_history_reactions_files_scopes's real upgrade()
    first -- so a regression in what 20260812 actually leaves in the row
    (as opposed to what its constants claim) would pass every other test
    here. This runs the real two-migration sequence a fresh, or
    partially-migrated, database goes through."""
    prior_migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260812_add_slack_history_reactions_files_scopes.py"
    )
    spec = importlib.util.spec_from_file_location(
        "add_slack_history_reactions_files_scopes_migration", prior_migration_file
    )
    prior_migration = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(prior_migration)

    migration = _load_migration_module()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    with engine.begin() as connection:
        _create_table(
            connection,
            prior_migration.PREVIOUS_SCOPES,
            prior_migration.PREVIOUS_DESCRIPTION,
        )
        _create_oauth_providers_table(connection, prior_migration.PREVIOUS_SCOPES)
        with patch.object(prior_migration, "op", _operations(connection)):
            prior_migration.upgrade()
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()

        # oauth_scopes always gets overwritten unconditionally with the
        # literal CURRENT_SCOPES list (see _set_slack_scopes's docstring),
        # so this is exact-list-safe.
        assert _scopes(connection) == migration.CURRENT_SCOPES
        # Set, not list, equality for default_scopes: the delta-merge
        # appends channels:join at the end rather than reproducing
        # CURRENT_SCOPES's exact (inserted-in-the-middle) order.
        assert set(_provider_default_scopes(connection, "slack")) == set(
            migration.CURRENT_SCOPES
        )
