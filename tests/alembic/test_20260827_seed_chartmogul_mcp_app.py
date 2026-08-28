"""Tests for the ChartMogul MCP connector seed migration."""

import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

# Ground-truth literal, independent of migration.ROW/the registry constant:
# the other cross-file checks below (test_seed_row_matches_registry,
# test_dockerfile_vendor_path_matches_registry) only assert that the
# Dockerfile, the registry, and this migration agree with *each other* --
# none of them would catch all three drifting to the same wrong value
# together. Mirrors the hardcoded-literal convention already used by e.g.
# test_20260818_seed_stripe_mcp_app.py.
EXPECTED_VENDOR_PATH = "/opt/xagent/vendor/chartmogul-mcp-server"


@pytest.fixture(autouse=True)
def _clear_vendor_path_env(monkeypatch):
    """get_builtin_public_mcp_app_rows() -> get_chartmogul_mcp_vendor_path()
    reads XAGENT_CHARTMOGUL_MCP_VENDOR_PATH from the process environment,
    unlike every other builtin connector's launch_config, which is a static
    literal. Any test in this file that touches the live registry would
    otherwise silently fail in an environment that happens to have this var
    set (e.g. a shell that sourced a deployment .env, or a CI job that
    carries a --build-arg override's env through to its test step) --
    confirmed by reproducing it directly. Clearing it here, autouse, means
    no test in this file has to remember to do it individually."""
    from xagent.config import CHARTMOGUL_MCP_VENDOR_PATH

    monkeypatch.delenv(CHARTMOGUL_MCP_VENDOR_PATH, raising=False)


def _load_migration_module():
    migration_file = (
        Path(__file__).parent.parent.parent
        / "src/xagent/migrations/versions/20260827_seed_chartmogul_mcp_app.py"
    )
    spec = importlib.util.spec_from_file_location(
        "seed_chartmogul_migration", migration_file
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _operations(connection):
    return Operations(MigrationContext.configure(connection))


def _create_table(connection, *, omit_description=False):
    description_column = "" if omit_description else "description TEXT,"
    connection.execute(
        text(
            f"""
            CREATE TABLE public_mcp_apps (
                id INTEGER PRIMARY KEY,
                app_id VARCHAR(100) NOT NULL UNIQUE,
                name VARCHAR(200) NOT NULL,
                {description_column}
                icon VARCHAR(1000),
                transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                provider_name VARCHAR(50),
                category VARCHAR(100),
                oauth_scopes JSON,
                is_visible_in_connector BOOLEAN NOT NULL DEFAULT 1,
                launch_config JSON
            )
            """
        )
    )


def _app_ids(connection):
    return set(connection.execute(text("SELECT app_id FROM public_mcp_apps")).scalars())


def test_upgrade_inserts_chartmogul(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
        assert "chartmogul" in _app_ids(connection)
        row = connection.execute(
            text(
                "SELECT transport, provider_name, is_visible_in_connector, "
                "launch_config FROM public_mcp_apps WHERE app_id='chartmogul'"
            )
        ).first()
        assert row[0] == "stdio"
        assert row[1] is None
        # The table defaults is_visible_in_connector to TRUE, so this must be
        # asserted against the actual persisted value, not just the dict
        # equality test_seed_row_matches_registry does -- a dropped/mistyped
        # key in ROW would otherwise insert visible without anything here
        # catching it. Mirrors test_upgrade_inserts_chrome's identical check.
        assert row[2] == 0
        assert EXPECTED_VENDOR_PATH in str(row[3])
        assert "CHARTMOGUL_TOKEN" in str(row[3])


def test_upgrade_is_idempotent(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.upgrade()  # second run must not raise or duplicate
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='chartmogul'")
        ).scalar()
        assert rows == 1


def test_upgrade_warns_and_still_inserts_when_column_missing(tmp_path, caplog):
    """If a table predates one of ROW's keys (shouldn't happen here, but the
    column-filter exists defensively), the row must still be inserted, and
    the drop must be logged rather than silent -- app_id then already
    exists, so a later run can never self-heal the missing column."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection, omit_description=True)
        with patch.object(migration, "op", _operations(connection)):
            with caplog.at_level("WARNING"):
                migration.upgrade()
        assert "chartmogul" in _app_ids(connection)
        assert any(
            "description" in message and "chartmogul" in message
            for message in caplog.messages
        ), (
            f"expected a warning naming the dropped 'description' column, got: {caplog.text!r}"
        )


def test_upgrade_raises_on_foreign_app_id_collision(tmp_path):
    """'chartmogul' had no special meaning before this migration -- nothing
    stopped an operator from hand-creating a custom PublicMCPApp with this
    exact app_id beforehand. Silently adopting that row (the old behavior)
    would let the builtin registry overlay ChartMogul's real
    transport/launch_config onto someone else's connector, and let a later
    downgrade delete it outright. upgrade() must instead fail loudly and
    leave the foreign row completely untouched."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('chartmogul', 'Operator ChartMogul', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            with pytest.raises(RuntimeError, match="did not create"):
                migration.upgrade()
        row = connection.execute(
            text(
                "SELECT COUNT(*), MIN(is_visible_in_connector), MIN(name) "
                "FROM public_mcp_apps WHERE app_id='chartmogul'"
            )
        ).first()
        assert row[0] == 1  # no duplicate inserted
        assert row[1] == 1  # left exactly as the operator set it
        assert row[2] == "Operator ChartMogul"  # other fields left alone


def test_downgrade_leaves_a_non_matching_row_in_place(tmp_path):
    """downgrade() only ever runs after a successful upgrade() (which now
    raises on any non-matching collision), so in practice a foreign row
    can't coexist with a completed upgrade -- but downgrade()'s own
    name/description/transport match is what makes that guarantee hold
    without a dedicated tracking column, so pin it directly rather than
    relying only on upgrade()'s behavior."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        connection.execute(
            text(
                "INSERT INTO public_mcp_apps "
                "(app_id, name, description, transport, is_visible_in_connector) "
                "VALUES ('chartmogul', 'Operator ChartMogul', 'hand-made', "
                "'stdio', 1)"
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            migration.downgrade()
        row = connection.execute(
            text(
                "SELECT name, is_visible_in_connector FROM public_mcp_apps "
                "WHERE app_id='chartmogul'"
            )
        ).first()
        # Doesn't match ROW's name/description/transport -- left in place,
        # not deleted.
        assert row is not None
        assert row[0] == "Operator ChartMogul"
        assert row[1] == 1


def test_upgrade_raises_if_visibility_column_is_missing(tmp_path):
    """Exercises the RuntimeError path directly (this suite runs without
    -O, so the assert-vs-raise distinction is otherwise never actually
    executed). A table missing is_visible_in_connector must fail loudly
    rather than seed the chartmogul row visible via the column-filter's
    silent drop."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                CREATE TABLE public_mcp_apps (
                    id INTEGER PRIMARY KEY,
                    app_id VARCHAR(100) NOT NULL UNIQUE,
                    name VARCHAR(200) NOT NULL,
                    description TEXT,
                    icon VARCHAR(1000),
                    transport VARCHAR(50) NOT NULL DEFAULT 'oauth',
                    provider_name VARCHAR(50),
                    category VARCHAR(100),
                    oauth_scopes JSON,
                    launch_config JSON
                )
                """
            )
        )
        with patch.object(migration, "op", _operations(connection)):
            try:
                migration.upgrade()
                raised = False
            except RuntimeError as exc:
                raised = True
                assert "is_visible_in_connector" in str(exc)
        assert raised, "upgrade() must raise when the visibility column is missing"
        assert "chartmogul" not in _app_ids(connection)


def test_seed_row_classifies_api_key():
    """The ChartMogul entry must classify as "api_key" -- an
    "unconnectable" classification would make the catalog entry dead on
    arrival in the connector UI once it's flipped visible."""
    from xagent.web.mcp_apps import classify_app_auth

    migration = _load_migration_module()
    assert (
        classify_app_auth(migration.ROW["transport"], migration.ROW["launch_config"])
        == "api_key"
    )


def test_downgrade_then_upgrade_round_trip(tmp_path):
    """A downgrade only deletes the catalog row (leftover
    MCPServer/UserMCPServer rows are intentionally left in place per the
    downgrade docstring), so a subsequent upgrade must cleanly re-seed it
    rather than hitting the existing-row early return or a uniqueness
    conflict."""
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
            migration.upgrade()
        rows = connection.execute(
            text("SELECT COUNT(*) FROM public_mcp_apps WHERE app_id='chartmogul'")
        ).scalar()
        assert rows == 1
        assert "chartmogul" in _app_ids(connection)


def test_seed_row_matches_registry():
    """The migration snapshot and the runtime registry must define the same
    chartmogul row (the migration is a frozen copy; this catches drift)."""
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    migration = _load_migration_module()
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chartmogul"
    )
    assert migration.ROW == registry_row


def test_registry_launch_config_reflects_a_custom_vendor_path_env(monkeypatch):
    """The Dockerfile/config tests each check their own half of the ARG ->
    ENV -> get_chartmogul_mcp_vendor_path() -> registry chain in isolation,
    but none of them actually set a custom env value and ask the live
    registry for its launch_config -- so a registry-side hardcode, a
    renamed config key, or a stage that re-clobbers the ENV could still
    leave every one of those tests green. Chain all of it end to end."""
    from xagent.config import CHARTMOGUL_MCP_VENDOR_PATH
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    monkeypatch.setenv(CHARTMOGUL_MCP_VENDOR_PATH, "/custom/chartmogul-path")
    registry_row = next(
        r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "chartmogul"
    )
    assert "/custom/chartmogul-path" in registry_row["launch_config"]["args"]


def test_downgrade_removes_chartmogul(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        _create_table(connection)
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        assert "chartmogul" not in _app_ids(connection)


def test_dockerfile_vendor_path_matches_registry():
    """The Dockerfile's CHARTMOGUL_MCP_VENDOR_PATH build ARG default must
    match config.get_chartmogul_mcp_vendor_path()'s own default -- the
    registry's launch_config reads that function at runtime (covered by
    test_seed_row_matches_registry, and test_config.py separately pins the
    function's default in isolation), so this only needs to check the
    Dockerfile side of the pair. Passing the ARG through as
    XAGENT_CHARTMOGUL_MCP_VENDOR_PATH at build time (asserted below) then
    makes the *built image* self-consistent regardless of either default.
    Same shape as test_dockerfile_npx_cache_pin_matches_registry for the
    chrome-devtools connector's npx version pin.
    """
    dockerfile = (
        Path(__file__).parent.parent.parent / "docker/Dockerfile.backend"
    ).read_text()
    match = re.search(r'ARG CHARTMOGUL_MCP_VENDOR_PATH="([^"]+)"', dockerfile)
    assert match is not None
    assert match.group(1) == EXPECTED_VENDOR_PATH

    # The ARG declaration alone doesn't prove the RUN block actually uses
    # it -- someone could hardcode a different literal path in the clone/cd
    # commands while leaving this default untouched. Assert the RUN block
    # references it as a shell variable, not just that the ARG exists.
    assert (
        'git clone "$CHARTMOGUL_MCP_REPO_URL" "$CHARTMOGUL_MCP_VENDOR_PATH"'
        in dockerfile
    )
    assert 'cd "$CHARTMOGUL_MCP_VENDOR_PATH"' in dockerfile

    # The built image must actually pass the ARG through as the env var
    # config.get_chartmogul_mcp_vendor_path() reads -- without this, the
    # running container would silently fall back to its Python-side default
    # even if someone built with a different --build-arg
    # CHARTMOGUL_MCP_VENDOR_PATH, pointing launch_config at a path the
    # image never cloned into.
    assert (
        'ENV XAGENT_CHARTMOGUL_MCP_VENDOR_PATH="$CHARTMOGUL_MCP_VENDOR_PATH"'
        in dockerfile
    )


def test_chartmogul_mcp_ref_is_a_pinned_commit():
    """CHARTMOGUL_MCP_REF must be a full commit SHA, not a branch/tag name --
    upstream ships neither, and this PR's stated safety rationale (Dockerfile
    comment) depends on the ref never silently tracking new upstream commits.
    """
    dockerfile = (
        Path(__file__).parent.parent.parent / "docker/Dockerfile.backend"
    ).read_text()
    match = re.search(r'ARG CHARTMOGUL_MCP_REF="([^"]+)"', dockerfile)
    assert match is not None
    assert re.fullmatch(r"[0-9a-f]{40}", match.group(1)), (
        f"CHARTMOGUL_MCP_REF={match.group(1)!r} is not a full 40-character commit SHA"
    )


def test_dockerfile_vendoring_precedes_cache_sensitive_copies():
    """The ChartMogul clone+sync RUN block is deliberately placed right
    after the `uv` binary COPY it needs, and before every other COPY in the
    `runtime` stage (frontend tools, Playwright, backend-build's own
    outputs) -- per the comment above it in the Dockerfile -- so a frontend
    dependency bump, a Playwright version bump, or an xagent source change
    doesn't also bust this unrelated third-party clone's build cache. That
    ordering is an invariant nothing else enforces -- a future edit could
    silently move it back (regressing build times, not correctness) with no
    test or build failure to catch it.
    """
    dockerfile = (
        Path(__file__).parent.parent.parent / "docker/Dockerfile.backend"
    ).read_text()
    # Anchored on the executable `git clone` inside the RUN block itself,
    # not the `ARG INSTALL_CHARTMOGUL` declaration above it -- an ARG can
    # sit anywhere before its first use, so a future edit could move the
    # actual clone+sync RUN block down past the COPY instructions below while leaving
    # the ARG in place, and an ARG-anchored check would stay green through
    # exactly the regression this test exists to catch.
    vendoring_index = dockerfile.index(
        'git clone "$CHARTMOGUL_MCP_REPO_URL" "$CHARTMOGUL_MCP_VENDOR_PATH"'
    )
    for marker in (
        "COPY --from=runtime-frontend-tools /opt/xagent/frontend /opt/xagent/frontend",
        "COPY --from=runtime-playwright /ms-playwright /ms-playwright",
        "COPY --from=backend-build /opt/xagent/src /opt/xagent/src",
    ):
        assert vendoring_index < dockerfile.index(marker), (
            f"the ChartMogul vendoring block must come before {marker!r}, "
            "or that copy's cache invalidation will also bust its build cache"
        )


def test_upgrade_and_downgrade_no_op_without_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    migration = _load_migration_module()
    with engine.begin() as connection:
        with patch.object(migration, "op", _operations(connection)):
            migration.upgrade()
            migration.downgrade()
        table_names = set(
            connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).scalars()
        )
        assert "public_mcp_apps" not in table_names
