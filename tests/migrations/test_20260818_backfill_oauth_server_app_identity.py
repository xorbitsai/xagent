"""Tests for migration 20260818_backfill_oauth_server_app_identity (#1429).

Following this repo's migration-test convention: the two tables are built
in their pre-migration shape with SQLAlchemy Core, seeded, and only the
migration under test is run against them (via MigrationContext/Operations,
so no full alembic history replay is needed).

What must hold:

- an auth-less oauth row named exactly after an OAuth app is stamped with
  that app's ``app_id`` -- and only that: no ``provider`` is written;
- a provider-only row is never stamped: the stored name is the only signal,
  because no writer this codebase shipped produces a provider-without-app_id
  row and the population that does exist carries no auth at all;
- rows already carrying a nonblank ``app_id`` are untouched (idempotence);
- rows resolving to nothing (orphans, non-oauth-app name matches) are left
  exactly as they were, preserving the name-fallback shim's behavior;
- non-oauth-transport server rows are never candidates, and neither are
  rows whose whole ``auth`` payload is a non-dict or whose stored
  ``provider`` is present but unusable;
- one ``app_id`` never lands on two rows -- including when the colliding
  partner was stamped before the run, or by an earlier run.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

TARGET_REVISION = "20260818_backfill_oauth_server_app_identity"


def _migration_module() -> ModuleType:
    import xagent.migrations as migrations_pkg

    migrations_dir = Path(next(iter(migrations_pkg.__path__)))
    path = migrations_dir / "versions" / f"{TARGET_REVISION}.py"
    spec = importlib.util.spec_from_file_location(TARGET_REVISION, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pre_migration_metadata() -> sa.MetaData:
    """The two tables reduced to the columns the migration reads/writes."""
    metadata = sa.MetaData()
    # Uniqueness mirrors production (mcp_servers.name, public_mcp_apps.app_id)
    # so a fixture cannot seed a state the real schema would reject -- which
    # is also what makes the same-name-server cases below honest: they must use
    # distinct names, exactly like production data.
    sa.Table(
        "mcp_servers",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(100), nullable=False, unique=True),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("auth", sa.JSON, nullable=True),
    )
    sa.Table(
        "public_mcp_apps",
        metadata,
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("app_id", sa.String(100), nullable=False, unique=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("transport", sa.String(50), nullable=False),
        sa.Column("provider_name", sa.String(50), nullable=True),
    )
    return metadata


@pytest.fixture()
def seeded_engine():
    engine = create_engine("sqlite:///:memory:")
    metadata = _pre_migration_metadata()
    metadata.create_all(engine)
    try:
        yield engine, metadata
    finally:
        engine.dispose()


def _run_upgrade(engine) -> None:
    _run(engine, "upgrade")


def _run_downgrade(engine) -> None:
    _run(engine, "downgrade")


def _run(engine, direction: str) -> None:
    """Both directions go through the same Operations context Alembic gives a
    version module, so a downgrade that started using `op` would be exercised
    the same way an upgrade is rather than only appearing to work."""
    module = _migration_module()
    with engine.begin() as conn:
        migration_context = MigrationContext.configure(conn)
        with Operations.context(migration_context):
            getattr(module, direction)()


def _auth_by_name(engine, metadata) -> dict[str, object]:
    servers = metadata.tables["mcp_servers"]
    with engine.connect() as conn:
        return {row.name: row.auth for row in conn.execute(sa.select(servers)).all()}


def _seed(engine, metadata, apps: list[dict], servers: list[dict]) -> None:
    with engine.begin() as conn:
        if apps:
            conn.execute(metadata.tables["public_mcp_apps"].insert(), apps)
        if servers:
            conn.execute(metadata.tables["mcp_servers"].insert(), servers)


def test_an_auth_less_row_is_stamped_by_exact_display_name(seeded_engine):
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Acme Drive"]
    assert auth == {"app_id": "acme-drive"}


def test_a_provider_only_row_is_never_stamped(seeded_engine):
    """There is no provider fallback: the stored name is the only signal. No
    writer this codebase shipped produces a provider-without-app_id row
    (_oauth_auth_metadata writes app_id first and unconditionally; the generic
    servers API cannot author transport="oauth" -- MCPServerConfig rejects it),
    and the population that does exist carries no auth at all, so a
    provider-keyed pass would defend nothing while being the only place two
    rows could compete for one identity."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            # Blank app_id plus a provider that would have resolved uniquely.
            {
                "name": "Old Acme Name",
                "transport": "oauth",
                "auth": {"app_id": "   ", "provider": "acme"},
            },
            # No app_id at all, provider only.
            {
                "name": "Older Acme Name",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Old Acme Name"] == {"app_id": "   ", "provider": "acme"}
    assert auth["Older Acme Name"] == {"provider": "acme"}


def test_rows_already_stamped_and_orphans_are_untouched(seeded_engine):
    engine, metadata = seeded_engine
    stamped = {"app_id": "acme-drive", "provider": "acme", "extra": "kept"}
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": dict(stamped)},
            {"name": "no-such-app", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)
    # Idempotence: a second run changes nothing either.
    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == stamped
    assert auth["no-such-app"] is None


def test_a_provider_conflict_refuses_the_name_match(seeded_engine):
    """A row named after app A whose own auth.provider names a different
    provider is the conflict _ensure_server_matches_oauth_app refuses with a
    ValueError. Stamping A's app_id would create a row *no* app claims — it
    fails A's provider gate and the other provider's app_id gate — so the
    migration refuses too, leaving the read-time provider fallback exactly as
    it was."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            {
                "app_id": "other-app",
                "name": "Other App",
                "transport": "oauth",
                "provider_name": "other",
            },
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "other"},
            }
        ],
    )

    _run_upgrade(engine)

    # Untouched: the conflicting evidence is left for a human (or the
    # provisioning writer's own refusal path) to resolve.
    assert _auth_by_name(engine, metadata)["Acme Drive"] == {"provider": "other"}


def test_a_non_dict_auth_payload_is_left_alone(seeded_engine):
    """Garbage (a scalar/list auth on an oauth row) is not a candidate: such a
    row resolves fine today through the name fallback, and this migration is
    irreversible, so destroying a value it does not have to touch is the wrong
    default."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": "garbage-string"}
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == "garbage-string"


def test_offline_sql_mode_raises_instead_of_stamping():
    """Alembic emits alembic_version bookkeeping even for an empty migration
    body, so a silent offline no-op would advance the version while touching
    no row — permanently skipping the backfill on the next online upgrade.
    The offline branch must fail loudly instead, for both dialects.

    Scope: this pins the raise, which is the whole guard. The bookkeeping
    hazard itself lives in Alembic's env.py/CLI path, which this harness does
    not drive — an end-to-end `alembic upgrade --sql` test would need a real
    config and script directory, and the raise here is what makes that path
    unreachable in the first place."""
    module = _migration_module()
    for dialect in ("sqlite", "postgresql"):
        migration_context = MigrationContext.configure(
            dialect_name=dialect, opts={"as_sql": True}
        )
        with (
            Operations.context(migration_context),
            pytest.raises(RuntimeError, match="offline"),
        ):
            module.upgrade()


def test_duplicate_exact_names_are_ambiguous_and_resolve_nothing(seeded_engine):
    """Two OAuth apps sharing one exact display name poison that name key: an
    auth-less row under it stays untouched rather than being stamped with
    whichever app happened to be seeded first."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": None,
            },
            {
                "app_id": "acme-drive-eu",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": None,
            },
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] is None


def test_a_non_string_app_id_is_malformed_and_restamped(seeded_engine):
    """get_app_for_mcp_server rejects a non-string auth.app_id outright, so a
    row carrying one is permanently unresolvable — it must stay a candidate
    and a successful name match overwrites the malformed value."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": {"app_id": 123}}],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)["Acme Drive"]
    assert auth == {"app_id": "acme-drive"}


def test_identities_are_matched_raw_never_trimmed(seeded_engine):
    """Identities are opaque and every read path compares them exactly, so the
    migration must not trim: a padded catalog app_id is stamped verbatim
    (that raw value is what get_app_by_id can resolve), and a
    whitespace-variant row name does not match at all — under-matching leaves
    the name-fallback shim in charge."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": " acme-drive ",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            # A second app so the variant-name row below has an identity
            # available to it: were the lookup to trim, it would resolve this
            # app rather than being refused by the claim guard, which would
            # otherwise mask the bug.
            {
                "app_id": "acme-mail",
                "name": "Acme Mail",
                "transport": "oauth",
                "provider_name": "mail",
            },
        ],
        servers=[
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
            {"name": "Acme Mail ", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive"] == {"app_id": " acme-drive "}
    # Trailing space: no match, and "acme-mail" is claimed by nobody, so only a
    # trimming lookup could have stamped this row.
    assert auth["Acme Mail "] is None


def test_an_identity_another_row_already_carries_is_refused(seeded_engine):
    """`claimed` seeds from the identities rows already carry, so an app whose
    app_id is already on one row is never stamped onto a second -- two rows
    sharing an identity would make _lookup_oauth_server_for_app, which reads an
    unordered query, pick nondeterministically between them.

    Reachable through the API, not only by out-of-band edits: rename the app
    away (the writer, finding no row under the new name, creates and stamps a
    fresh one), then rename it back, and a legacy unstamped row's name once
    again resolves an app another row already carries."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            # Carries the identity already, and is not a candidate.
            {
                "name": "Acme Drive (renamed away)",
                "transport": "oauth",
                "auth": {"app_id": "acme-drive"},
            },
            # Would resolve to the same app by exact name.
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive (renamed away)"] == {"app_id": "acme-drive"}
    assert auth["Acme Drive"] is None


def test_lookup_map_construction_skips_the_right_apps(seeded_engine):
    """Two map-building branches: an app whose app_id is blank is unusable as
    an identity and contributes nothing, and app-side ``transport`` is
    case-folded (it is a shape enum, not an identity). A name-keyed map means a
    blank-named app is simply unreachable."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            # Blank app_id: unusable as an identity, contributes nothing.
            {
                "app_id": "   ",
                "name": "Blank Id App",
                "transport": "oauth",
                "provider_name": "blank",
            },
            # Mixed-case transport still counts as oauth on the app side.
            {
                "app_id": "cased-app",
                "name": "Cased App",
                "transport": "OAuth",
                "provider_name": None,
            },
        ],
        servers=[
            {"name": "Blank Id App", "transport": "oauth", "auth": None},
            {"name": "Cased App", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Blank Id App"] is None
    assert auth["Cased App"] == {"app_id": "cased-app"}


def test_unrelated_auth_keys_survive_the_stamp(seeded_engine):
    """Only app_id is added: everything already in the auth dict -- including
    a blank provider and an encrypted secret -- is carried through
    untouched."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "  ", "access_token": "encrypted-blob"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {
        "app_id": "acme-drive",
        "provider": "  ",
        "access_token": "encrypted-blob",
    }


def test_a_stamp_survives_a_rerun_unchanged(seeded_engine):
    """Idempotence across runs, not just within one: run 2 -- what a downgrade
    (a no-op) plus upgrade produces -- must neither restamp the row nor let the
    identity it now carries be handed to anything else."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)
    after_first = _auth_by_name(engine, metadata)
    assert after_first["Acme Drive"] == {"app_id": "acme-drive"}

    _run_downgrade(engine)
    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata) == after_first


def test_a_name_owned_by_a_non_oauth_app_is_refused(seeded_engine):
    """A name matching a *non-OAuth* app is evidence about this row, not an
    absence of evidence: the row is refused rather than treated as unmatched,
    so no later rule can claim it for an app of the wrong shape."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-notes",
                "name": "Acme Notes",
                "transport": "stdio",
                "provider_name": "acme",
            },
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
        ],
        servers=[
            {
                "name": "Acme Notes",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Notes"] == {"provider": "acme"}


def test_a_name_carried_by_both_shapes_refuses_the_oauth_app(seeded_engine):
    """The one case where the cross-shape gate changes the outcome: a single
    display name carried by an OAuth app *and* a non-OAuth app. Without the
    gate the OAuth app would simply win the name; with it the ownership is
    contested, so the row keeps its read-time name fallback."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            {
                "app_id": "acme-drive-stdio",
                "name": "Acme Drive",
                "transport": "stdio",
                "provider_name": None,
            },
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] is None


def test_a_malformed_non_string_provider_refuses_the_row(seeded_engine):
    """A provider that is not a string at all cannot be compared by the
    conflict gate, and _is_oauth_server_for_app would reject the row against
    any app once stamped — so a stamp buys nothing and the row is left
    alone."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": ["acme"]},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {"provider": ["acme"]}


def test_a_mixed_case_transport_row_is_not_a_candidate(seeded_engine):
    """The five gates that make an oauth row work compare `transport ==
    "oauth"` unfolded (tools/config.py's credential injection,
    _is_oauth_server_for_app, _enrich_oauth_server_info,
    _ensure_server_matches_oauth_app), so a row stored "OAuth" is dead to all
    of them. Stamping it would be worthless to that row and could burn the
    app's one identity: the `claimed` guard would then refuse the genuine row,
    with no downgrade to undo it. The candidate filter therefore matches
    exactly, like the writer and the readers."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {"name": "Acme Drive", "transport": "OAuth", "auth": None},
            {"name": "Acme Drive (real)", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    # The dead row is untouched, and its presence did not deny the identity to
    # a working row (this one loses only because its name does not match).
    assert auth["Acme Drive"] is None
    assert auth["Acme Drive (real)"] is None


def test_an_identity_on_a_non_oauth_transport_row_is_still_claimed(seeded_engine):
    """The claim set is read *without* a transport filter, deliberately wider
    than the candidate query. get_app_for_mcp_server resolves from auth.app_id
    with no transport check at all, and the OAuth-disconnect cleanup path walks
    a user's whole server list through it -- so a row stored "OAuth" (or any
    other transport) still makes its app_id taken, even though that row could
    never serve the connector itself. Filtering the claim set by transport left
    such an id invisible and let a second row be stamped with it."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            # Not a candidate (transport is not exactly "oauth"), but it does
            # carry the identity, and the transport-blind reader sees it.
            {
                "name": "Acme Drive (mixed case)",
                "transport": "OAuth",
                "auth": {"app_id": "acme-drive"},
            },
            # Would resolve to the same app by exact name.
            {"name": "Acme Drive", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Drive (mixed case)"] == {"app_id": "acme-drive"}
    assert auth["Acme Drive"] is None


def test_a_provider_differing_only_in_case_is_not_a_conflict(seeded_engine):
    """The conflict gate compares providers the way the runtime does
    (_normalize_app_key: casefold, strip, whitespace-to-hyphen). Comparing
    them exactly would permanently refuse a stamp over a difference
    _is_oauth_server_for_app already treats as none -- and this migration runs
    once."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "  Acme "},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {
        "provider": "  Acme ",
        "app_id": "acme-drive",
    }


def test_a_missing_table_is_a_logged_no_op(seeded_engine, caplog):
    """Alembic commits the version bump either way, so a skip here is
    permanent: it must at least say so."""
    import logging

    engine, _metadata = seeded_engine
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE public_mcp_apps"))

    with caplog.at_level(logging.WARNING):
        _run_upgrade(engine)

    assert any("search_path" in r.getMessage() for r in caplog.records)


def test_an_auth_list_payload_is_left_alone(seeded_engine):
    """The non-dict branch covers a JSON list too, not just a scalar."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": ["junk"]}],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == ["junk"]


def test_a_blank_app_side_provider_is_not_a_conflict(seeded_engine):
    """An app with no provider cannot contradict anything, so a row that has
    one is still stamped."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "   ",
            }
        ],
        servers=[
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            }
        ],
    )

    _run_upgrade(engine)

    assert _auth_by_name(engine, metadata)["Acme Drive"] == {
        "provider": "acme",
        "app_id": "acme-drive",
    }


def test_downgrade_is_a_no_op(seeded_engine):
    """Documented as irreversible: the downgrade must not attempt to strip
    stamps it cannot tell apart from the writer's own."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            }
        ],
        servers=[{"name": "Acme Drive", "transport": "oauth", "auth": None}],
    )
    _run_upgrade(engine)
    before = _auth_by_name(engine, metadata)

    _run_downgrade(engine)

    assert _auth_by_name(engine, metadata) == before


def test_non_oauth_shapes_are_never_candidates(seeded_engine):
    """A stdio server row named like an app, and an oauth row named after a
    *non-oauth* app, are both left alone: the first is a different transport,
    the second would stamp a cross-shape identity."""
    engine, metadata = seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-notes",
                "name": "Acme Notes",
                "transport": "stdio",
                "provider_name": None,
            }
        ],
        servers=[
            # mcp_servers.name is unique in production, so the two shapes must
            # be seeded under distinct names; the stdio row is a candidate by
            # neither transport nor name, and the oauth row's name resolves
            # only a stdio app, which is the cross-shape refusal.
            {"name": "Acme Notes Local", "transport": "stdio", "auth": None},
            {"name": "Acme Notes", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    servers = metadata.tables["mcp_servers"]
    with engine.connect() as conn:
        rows = conn.execute(sa.select(servers)).all()
    assert all(row.auth is None for row in rows)


# Its own schema, not the shared public one. The CI Postgres job runs every
# step against a single database, and later steps depend on the real,
# fully-migrated shape of mcp_servers/public_mcp_apps -- dropping and stubbing
# those in `public` would corrupt them. The migration reads through
# `inspector.get_table_names()` and unqualified table literals, both of which
# follow search_path, so pointing search_path at a private schema isolates the
# whole run without the migration needing to know.
_PG_SCHEMA = "backfill_app_identity_test"


def _postgres_url() -> str | None:
    # Both spellings, matching the sibling migration suites: CI sets the first,
    # while a local run may already export the second.
    return os.getenv("XAGENT_TEST_POSTGRES_URL") or os.getenv(
        "POSTGRES_TEST_DATABASE_URL"
    )


@pytest.fixture
def postgres_seeded_engine():
    """The two tables on a real PostgreSQL server, in a throwaway schema.

    The chain-level CI job upgrades an *empty* database, so without this the
    backfill's data path would never run on PostgreSQL at all -- and JSON
    round-tripping, the ordered read and the re-read guard are exactly the
    parts that could differ from SQLite.
    """
    url = _postgres_url()
    if not url:
        pytest.skip("XAGENT_TEST_POSTGRES_URL is not set")
    engine = create_engine(
        url,
        connect_args={"options": f"-csearch_path={_PG_SCHEMA}"},
    )
    with engine.begin() as conn:
        conn.execute(text(f"DROP SCHEMA IF EXISTS {_PG_SCHEMA} CASCADE"))
        conn.execute(text(f"CREATE SCHEMA {_PG_SCHEMA}"))
    metadata = _pre_migration_metadata()
    metadata.create_all(bind=engine)
    try:
        yield engine, metadata
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SCHEMA IF EXISTS {_PG_SCHEMA} CASCADE"))
        engine.dispose()


@pytest.mark.postgresql
def test_the_backfill_runs_on_postgresql(postgres_seeded_engine):
    """Stamp, refusal and collision behavior on the real backend: the JSON
    write round-trips; a name match is stamped; a row whose name resolves an
    app another row already carries is refused; a provider-only row and an
    orphan are both left alone."""
    engine, metadata = postgres_seeded_engine
    _seed(
        engine,
        metadata,
        apps=[
            {
                "app_id": "acme-drive",
                "name": "Acme Drive",
                "transport": "oauth",
                "provider_name": "acme",
            },
            {
                "app_id": "acme-mail",
                "name": "Acme Mail",
                "transport": "oauth",
                "provider_name": "mail",
            },
        ],
        servers=[
            {"name": "Acme Mail", "transport": "oauth", "auth": None},
            {
                "name": "Acme Drive v2",
                "transport": "oauth",
                "auth": {"app_id": "acme-drive", "provider": "acme"},
            },
            {
                "name": "Acme Drive",
                "transport": "oauth",
                "auth": {"provider": "acme"},
            },
            {"name": "orphan", "transport": "oauth", "auth": None},
        ],
    )

    _run_upgrade(engine)

    auth = _auth_by_name(engine, metadata)
    assert auth["Acme Mail"] == {"app_id": "acme-mail"}
    assert auth["Acme Drive v2"] == {"app_id": "acme-drive", "provider": "acme"}
    # Its name resolves acme-drive, but the row above already carries that
    # identity, so it is refused and keeps its pre-migration state.
    assert auth["Acme Drive"] == {"provider": "acme"}
    assert auth["orphan"] is None
