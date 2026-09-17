"""add the durable task interaction request table

The table records one blocking interaction request raised by a task and
its lifecycle through to answer or expiry. The model landed on its own
(#1209): until this revision, only a create_all-built schema carried it,
so any database maintained through the alembic revision chain was missing
the table entirely. This revision is model-only plumbing -- it has zero
production consumers; the primitive that writes rows here and the code
that reads them land in later PRs.

upgrade()/downgrade() fork on context.as_sql because offline (--sql)
generation runs against a MockConnection, where reflection is
unavailable. The offline branch therefore emits the DDL unconditionally
and the online branch keeps existence guards -- the same shape as
20260726_add_task_telegram_user_id.py and 20260808_add_task_lease_attempt_id.py,
not the inspector-only shape #1137 used, which raises under --sql on both
dialects.

The online branch carries two guards: the table already existing makes
upgrade() a no-op, which is what keeps a create_all-first startup
idempotent against a subsequent `alembic upgrade head`; a missing parent
table (any of tasks, trace_events, users) also makes it a no-op, because
the two backends fail asymmetrically on a dangling foreign key reference
-- PostgreSQL raises UndefinedTable while creating the table, SQLite lets
the CREATE TABLE succeed and only fails on a later INSERT.

All twenty-three CHECK constraints, both UNIQUE constraints, and all
three named foreign keys are rendered inline inside one op.create_table
on both PostgreSQL and SQLite, so no ALTER TABLE ADD CONSTRAINT is ever
emitted for this table. downgrade() is therefore a single op.drop_table:
every constraint disappears with the table, and there is nothing to drop
by name. The name-for-name contract between this file and the model's
__table_args__ (see TaskInteractionRequest's docstring) is instead
enforced by the create_all/migration parity tests, not by the downgrade
path.

downgrade() is state-based, not provenance-based: it drops whatever table
is there, including one a create_all built rather than this revision --
same as the sibling column migrations (e.g.
20260804_add_task_checkpoint_trace_event_anchor.py).

The offline SQLite script below carries no transaction wrapper of its
own, so a mid-script failure would still let a trailing
`alembic_version` update land and the stamp would need manual repair.
This revision's offline SQLite upgrade is only three statements (one
CREATE TABLE, two CREATE INDEX) and its downgrade is one DROP TABLE, so
that risk has no known failure mode here.

Revision ID: 20260809_add_task_interaction_requests
Revises: 20260806_seed_chrome_mcp_app
Create Date: 2026-08-09

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260809_add_task_interaction_requests"
down_revision: Union[str, None] = "20260806_seed_chrome_mcp_app"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

TABLE = "task_interaction_requests"
PARENT_TABLES = ("tasks", "trace_events", "users")
INDEXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("ix_task_interaction_requests_task_status", ("task_id", "status")),
    (
        "ix_task_interaction_requests_resume_trace_event_id",
        ("resume_trace_event_id",),
    ),
)
# Name, expression pairs. Order and text are copied verbatim from
# TaskInteractionRequest.__table_args__ (src/xagent/web/models/task_interaction.py)
# -- the model is this list's source of truth, and the create_all/migration
# parity tests compare both name and expression against it.
CHECKS: tuple[tuple[str, str], ...] = (
    (
        "ck_task_interaction_requests_status",
        "status IN ('active','answered','terminated')",
    ),
    ("ck_task_interaction_requests_kind", "kind IN ('clarification')"),
    (
        "ck_task_interaction_requests_origin",
        "origin IN ('internal','sdk','a2a','trigger','widget','shared_link')",
    ),
    (
        "ck_task_interaction_requests_resume_checkpoint_type",
        "resume_checkpoint_type IN ('agent_execution_checkpoint')",
    ),
    (
        "ck_task_interaction_requests_resume_locator_format",
        "resume_locator_format IN ('trace_event_pk_v1')",
    ),
    (
        "ck_task_interaction_requests_terminal_reason",
        "terminal_reason IS NULL OR terminal_reason IN "
        "('deadline_elapsed','run_superseded','answered_via_legacy_resume')",
    ),
    ("ck_task_interaction_requests_protocol_version_floor", "protocol_version >= 1"),
    (
        "ck_task_interaction_requests_active_slot_value",
        "active_slot IS NULL OR active_slot = 1",
    ),
    (
        "ck_task_interaction_requests_active_slot_pairs_status",
        "(status = 'active') = (active_slot IS NOT NULL)",
    ),
    (
        "ck_task_interaction_requests_active_anchor",
        "status <> 'active' OR resume_trace_event_id IS NOT NULL",
    ),
    (
        "ck_task_interaction_requests_active_protocol",
        "active_slot IS NULL OR protocol_version = 1",
    ),
    (
        "ck_task_interaction_requests_terminal_pairs_status",
        "(status = 'terminated') = (terminal_reason IS NOT NULL)",
    ),
    (
        "ck_task_interaction_requests_terminated_at_pairs_status",
        "(status = 'terminated') = (terminated_at IS NOT NULL)",
    ),
    (
        "ck_task_interaction_requests_response_pairs_status",
        "(status = 'answered') = (response_payload IS NOT NULL)",
    ),
    (
        "ck_task_interaction_requests_responded_at_pairs_status",
        "(status = 'answered') = (responded_at IS NOT NULL)",
    ),
    (
        "ck_task_interaction_requests_responder_pairs_responded_at",
        "(responded_at IS NULL) = (responder_identity IS NULL)",
    ),
    ("ck_task_interaction_requests_expiry_after_creation", "expires_at > created_at"),
    ("ck_task_interaction_requests_run_id_nonempty", "run_id <> ''"),
    (
        "ck_task_interaction_requests_resume_event_id_nonempty",
        "resume_event_id <> ''",
    ),
    (
        "ck_task_interaction_requests_resume_execution_id_nonempty",
        "resume_execution_id <> ''",
    ),
    (
        "ck_task_interaction_requests_resume_run_partition_nonempty",
        "resume_run_partition <> ''",
    ),
    (
        "ck_task_interaction_requests_request_idempotency_key_nonempty",
        "request_idempotency_key <> ''",
    ),
    (
        "ck_task_interaction_requests_responder_identity_nonempty",
        "responder_identity <> ''",
    ),
)


# The schema of the *visible* parent relations (tasks/trace_events/users).
# version_table_schema names only the Alembic version table, and
# current_schema() is merely the first entry on search_path, so neither
# identifies the schema an unqualified reference actually resolves to. Ask
# PostgreSQL which one it resolves -- same lookup as
# 20260808_add_task_lease_attempt_id.py's _target_schema.
POSTGRES_VISIBLE_TABLE_SCHEMA_SQL = sa.text(
    """
    SELECT ns.nspname
    FROM pg_catalog.pg_class AS cls
    JOIN pg_catalog.pg_namespace AS ns ON ns.oid = cls.relnamespace
    WHERE cls.oid = pg_catalog.to_regclass(:table_name)
    """
)


def _target_schema() -> str | None:
    """The schema holding the parent relations this table hangs off.

    Resolved from a parent, not from TABLE: TABLE does not exist yet on the
    run that creates it, so to_regclass() on it would always be NULL.
    """
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        resolved = bind.execute(
            POSTGRES_VISIBLE_TABLE_SCHEMA_SQL, {"table_name": PARENT_TABLES[0]}
        ).scalar()
        if resolved:
            return str(resolved)
    schema = op.get_context().version_table_schema
    return str(schema) if schema else None


def _create_table(schema: str | None) -> None:
    op.create_table(
        TABLE,
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("task_id", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("protocol_version", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("active_slot", sa.Integer(), nullable=True),
        sa.Column("origin", sa.String(length=20), nullable=False),
        sa.Column("request_payload", sa.JSON(none_as_null=True), nullable=False),
        sa.Column("response_payload", sa.JSON(none_as_null=True), nullable=True),
        sa.Column("request_idempotency_key", sa.String(length=64), nullable=False),
        sa.Column("resume_trace_event_id", sa.Integer(), nullable=True),
        sa.Column("resume_event_id", sa.String(length=255), nullable=False),
        sa.Column("resume_execution_id", sa.String(length=255), nullable=False),
        sa.Column("resume_locator_format", sa.String(length=32), nullable=False),
        sa.Column("resume_checkpoint_type", sa.String(length=64), nullable=False),
        sa.Column("resume_run_partition", sa.String(length=64), nullable=False),
        sa.Column("responder_user_id", sa.Integer(), nullable=True),
        sa.Column("responder_identity", sa.String(length=255), nullable=True),
        sa.Column("terminal_reason", sa.String(length=32), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # onupdate=func.now() is a Python-side behavior SQLAlchemy applies on
        # UPDATE; it has no DDL representation, so only server_default is
        # replicated here.
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("responded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("terminated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "task_id", "active_slot", name="uq_task_interaction_active_slot"
        ),
        sa.UniqueConstraint(
            "task_id",
            "run_id",
            "request_idempotency_key",
            name="uq_task_interaction_request_identity",
        ),
        *[sa.CheckConstraint(expr, name=name) for name, expr in CHECKS],
        sa.ForeignKeyConstraint(
            ["task_id"],
            ["tasks.id"],
            name="fk_task_interaction_requests_task_id",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["resume_trace_event_id"],
            ["trace_events.id"],
            name="fk_task_interaction_requests_resume_trace_event_id",
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["responder_user_id"],
            ["users.id"],
            name="fk_task_interaction_requests_responder_user_id",
            ondelete="SET NULL",
        ),
        schema=schema,
    )
    for index_name, columns in INDEXES:
        op.create_index(index_name, TABLE, list(columns), unique=False, schema=schema)


def upgrade() -> None:
    context = op.get_context()

    # Offline (--sql) generation has a MockConnection, so reflection is
    # unavailable. Emit the unconditional DDL instead of inspecting. This is
    # the 20260726 shape, not the inspector-only shape #1137 used -- the
    # latter raises under --sql on both dialects.
    if context.as_sql:
        _create_table(None)
        return

    # Resolve the schema the parent tables actually live in before doing any
    # reflection: both guards below must inspect the same schema the table
    # gets created in, or a stale same-named table (or same-named parents) on
    # an earlier search_path entry can shadow the real target -- see the
    # module docstring.
    schema = _target_schema()
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names(schema=schema))

    # Guard 1: a create_all-first startup already has this table: makes
    # `alembic upgrade head` idempotent against it.
    if TABLE in tables:
        return

    # Guard 2: any parent table missing means the three foreign keys below
    # would reference nothing. PostgreSQL raises UndefinedTable if this ran
    # anyway; SQLite would let CREATE TABLE succeed and only fail later, on
    # the first INSERT.
    if not set(PARENT_TABLES) <= tables:
        return

    _create_table(schema)


def downgrade() -> None:
    context = op.get_context()

    if context.as_sql:
        op.drop_table(TABLE)
        return

    schema = _target_schema()
    if TABLE not in sa.inspect(op.get_bind()).get_table_names(schema=schema):
        return

    op.drop_table(TABLE, schema=schema)
