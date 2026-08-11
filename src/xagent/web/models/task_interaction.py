"""The durable, authoritative record of a blocking task interaction request."""

from sqlalchemy import (
    JSON,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.sql import func

from .database import Base

# ---------------------------------------------------------------------------
# Column-domain constants for this table's `origin` and `protocol_version`
# columns.
#
# Attribution: these four names -- the vocabulary, the protocol version, the
# normalizer, and the literal-subset constant -- live here rather than in an
# interaction-rollout gate module or anywhere else, because they define what
# a row in ``task_interaction_requests`` *means*, not whether some rollout
# gate currently lets a new native row through. ``origin`` and
# ``protocol_version`` are columns on this table; ``INTERACTION_ORIGIN_VOCABULARY``
# mirrors the ``ck_task_interaction_requests_origin`` CHECK a few lines
# below, and ``INTERACTION_PROTOCOL_VERSION`` mirrors
# ``ck_task_interaction_requests_active_protocol``. A rollout gate built on
# top of these names can be deleted outright and this table's column-domain
# mapping still has to exist for whatever reads or writes rows here -- so
# the mapping is owned by the table, not by whichever gate happens to
# consume it today.
#
# Two call surfaces, two normalization rules, by design: ``Task.source`` is
# a value written by application code and persisted, so an unexpected
# reading of it (extra whitespace, wrong case) is itself a fact worth
# surfacing -- ``normalize_interaction_origin`` below does not strip or
# lowercase, so a stray ``" SDK "`` reads back as unknown instead of being
# silently folded into the known ``"sdk"`` bucket. The
# ``XAGENT_INTERACTION_NATIVE_SOURCES`` environment variable parsed in
# ``config.py`` is a human-typed operator input and *is* stripped and
# lowercased there. Do not unify the two -- each rule is pinned by its own
# tests and guards a different kind of input.
# ---------------------------------------------------------------------------
INTERACTION_ORIGIN_VOCABULARY = frozenset(
    {"internal", "sdk", "a2a", "trigger", "widget", "shared_link"}
)
INTERACTION_PROTOCOL_VERSION = 1

# Ordering overlay for the ``ck_task_interaction_requests_origin`` CHECK
# below: offline SQL must be byte-deterministic and the already-frozen
# migration rendered the IN-list in exactly this order, so the order cannot
# be derived by sorting. Membership is owned by
# ``INTERACTION_ORIGIN_VOCABULARY`` above -- this tuple only fixes the
# rendering order, and the assert keeps the pair from ever diverging.
ORIGIN_VALUES: tuple[str, ...] = (
    "internal",
    "sdk",
    "a2a",
    "trigger",
    "widget",
    "shared_link",
)
assert frozenset(ORIGIN_VALUES) == INTERACTION_ORIGIN_VOCABULARY, (
    "ORIGIN_VALUES is an ordering overlay over INTERACTION_ORIGIN_VOCABULARY; "
    "change membership there, then mirror it here"
)

# ``Task.source`` literals actually written by producers today -- a proper
# subset of INTERACTION_ORIGIN_VOCABULARY, never an alias for it. The
# vocabulary can (and does, via "internal") contain a value no call site
# ever passes explicitly, produced instead by the column default and by
# normalize_interaction_origin's falsy fallback below. Retiring a
# ``Task.source`` value -- stop producing it, keep reading rows that
# already carry it -- is a legitimate, expected move, and doing so narrows
# this constant without narrowing INTERACTION_ORIGIN_VOCABULARY or the
# origin CHECK. Because of that this constant must only ever be asserted as
# a subset of the vocabulary, never as equal to it.
TASK_SOURCE_LITERALS = frozenset({"a2a", "sdk", "shared_link", "trigger", "widget"})


def normalize_interaction_origin(source: str | None) -> str | None:
    """Map a ``Task.source`` reading to an origin vocabulary member.

    Returns ``None`` for anything not recognized -- callers must treat that
    as "unknown", not silently fall back to ``"internal"`` or any other
    bucket. Deliberately does not ``strip()`` or ``.lower()`` the input (see
    the module-level comment above): a value that only matches after
    normalization is exactly the signal this function exists to preserve.

    - ``None`` or ``""`` (falsy) -> ``"internal"``, matching this column's
      default for rows that predate this vocabulary.
    - An exact (byte-for-byte) match against
      :data:`INTERACTION_ORIGIN_VOCABULARY` -> that value.
    - Anything else, including whitespace-only strings and near-matches
      that only differ in case or surrounding whitespace -> ``None``.
    """
    if not source:
        return "internal"
    if source in INTERACTION_ORIGIN_VOCABULARY:
        return source
    return None


class TaskInteractionRequest(Base):  # type: ignore
    """One blocking interaction request raised by a task and its lifecycle to answer or expiry.

    Zero production consumers as of this table's introduction: the primitive
    that writes rows here (``stage_interaction_request`` / the
    ``interaction_handoff`` context manager) and the code that reads them
    land separately, so this class is model-only until those callers exist.

    Active slot: ``UNIQUE(task_id, active_slot)`` caps a task at one
    ``active`` row at a time. ``active_slot`` is ``1`` on that row and SQL
    ``NULL`` on every ``answered`` / ``terminated`` row, relying on
    NULL-distinct uniqueness so terminal rows can coexist without bound.
    This table depends on PostgreSQL's default NULLS DISTINCT behavior --
    do not add ``postgresql_nulls_not_distinct=True`` to
    ``uq_task_interaction_active_slot``, or every terminal row after the
    first would collide.

    That cap reads as "one active slot per root Task" only because one
    agent tree maps to exactly one ``tasks`` row today; the schema carries
    no root concept of its own. If a root discriminator column is ever
    added here it must be a ``NOT NULL`` sentinel -- an empty string, not a
    nullable column -- because the same NULL-distinct behavior this
    constraint relies on would otherwise void it entirely on the root
    side, accepting a second active root row.

    ``protocol_version`` is governed by two constraints:
    ``ck_task_interaction_requests_protocol_version_floor`` (``>= 1``, every
    row) and ``ck_task_interaction_requests_active_protocol`` (``= 1``,
    active rows only). A terminated or answered row may carry a future
    version by design -- that is what lets the protocol evolve without
    forbidding a later version from ever being written as a historical row,
    while still keeping anything that is not a version number at all out of
    every row, and keeping the active slot pinned to the one version this
    table currently knows how to interpret live.

    RESTRICT-for-active: ``resume_trace_event_id`` is
    ``ON DELETE SET NULL``, but ``ck_task_interaction_requests_active_anchor``
    requires a non-NULL anchor on every ``active`` row, so a delete that
    would SET NULL an active row's anchor is rejected by that CHECK instead
    of going through. The error surface is a CHECK violation, not a foreign
    key violation. A terminated row's anchor may still be cleared: the two
    columns touched by ON DELETE SET NULL (``resume_trace_event_id`` and
    ``responder_user_id``) are deliberately absent from every two-way paired
    CHECK below, because a two-way pairing would make SET NULL fail the same
    way on rows where clearing the FK is meant to succeed. Read together:
    terminal rows let SET NULL through (a cleared anchor is acceptable
    historical wear); active rows make SET NULL collide with a CHECK, which
    behaves like RESTRICT.

    ``JSON(none_as_null=True)`` is confined to this table, carried by both of
    its JSON columns. ``response_payload`` uses it so a Python ``None`` lands
    as SQL NULL instead of the JSON scalar ``null``: consumers must test "has
    this request been answered" with SQL ``IS NULL``, never with
    ``payload is None`` in Python, since a legitimate JSON ``null`` answer
    value must still read as answered. ``request_payload`` uses it so its
    ``NOT NULL`` constraint actually fires on a Python ``None`` -- without
    the flag, serialization runs before binding and a Python ``None`` would
    reach the column as the JSON text ``null``, which ``NOT NULL`` does not
    reject.

    ``terminal_reason`` vocabulary: each of the three members is spoken for
    by exactly one intended writer, and none of those writers exists yet.
    ``deadline_elapsed`` and ``run_superseded`` are the two reclaim
    branches, arriving with the staging primitive; ``answered_via_legacy_resume``
    arrives with the transitional close on the legacy answer path. It is
    not the same outcome as ``answered`` -- that path recovers a free-text
    chat message, not a protocol v1 structured response, and synthesizing a
    payload for it would fabricate a contract the client never produced.
    ``operator_cancelled`` and ``task_terminated`` have no writer yet and are
    intentionally not in the vocabulary; the old name
    ``superseded_by_legacy_resume`` must not come back.

    ``origin``'s IN-list is a frozen superset of ``Task.source`` literals,
    not a mirror: retiring a ``Task.source`` value must not narrow this
    CHECK, or historical rows written with the retired value become
    unrecreatable; adding a new ``Task.source`` value needs this CHECK
    updated plus a migration. The staging primitive validates ``origin`` in
    Python before INSERT -- the CHECK is the backstop, not the primary
    validation.

    Clock source for ``ck_task_interaction_requests_expiry_after_creation``:
    both operands now come from the caller's ``now`` -- the staging
    primitive (``stage_interaction_request`` in ``task_interaction_staging.py``)
    binds ``created_at`` explicitly to the same ``now`` it already validates
    and uses for the reclaim UPDATE, rather than leaving it to
    ``server_default=func.now()``. ``server_default`` stays on the column
    for every other writer this table may eventually get; it is simply not
    what this table's one production writer today relies on. That retires
    the earlier single-operand analysis and the per-backend slack it
    documented here (PostgreSQL's non-advancing in-transaction ``now()``;
    SQLite's second-truncated ``CURRENT_TIMESTAMP`` text comparison): with
    both operands bound from the same Python value, the CHECK is exactly
    the Python predicate the primitive already enforces
    (``expires_at > now``) at microsecond granularity, verified directly
    against both backends.

    The price of that precision is caller-clock skew inheritance: a stored
    row's ``created_at`` is only as accurate as the caller's own clock, the
    same exposure ``terminated_at`` and ``updated_at`` already carry on this
    table (both are also bound from caller-supplied values, never a server
    clock). A caller running fast can make a row's ``created_at`` land after
    its own server-defaulted ``updated_at`` from the same INSERT -- no
    constraint on this table pairs the two columns, and nothing in this
    module reads ``updated_at``, so that skew is inert here. What the CHECK
    actually asserts is that the row is internally consistent under one
    clock -- the caller's -- not that any column agrees with wall-clock
    time on the server.

    ``expires_at`` must always be bound as an aware **UTC** datetime by the
    caller -- that obligation is portable, not SQLite-specific. Neither
    backend rejects a naive bind: SQLite stores the digits verbatim, and
    PostgreSQL resolves a naive value against the session's ``TimeZone``
    setting, landing on the right instant only because that setting
    happens to be UTC. With a non-UTC session ``TimeZone``, PostgreSQL
    silently stores the wrong instant instead of raising (verified against
    a live PostgreSQL instance: the same naive value round-trips eight
    hours off under a +08:00 session ``TimeZone``). SQLite has a second,
    separate failure mode on top of that: ``DateTime(timezone=True)`` drops
    tzinfo on bind entirely, so even an *aware*-but-non-UTC value (e.g.
    ``+08:00``) is silently stored as local wall-clock time -- the CHECK
    still passes, and the row is accepted with the wrong instant. That
    aware-non-UTC corruption is SQLite-specific; the caller's UTC
    obligation is not. PostgreSQL converts an aware non-UTC value correctly
    (verified: a ``+08:00`` bind reads back as the UTC instant); only
    SQLite drops the offset. That asymmetry is exactly what the two
    suites' round-trip tests measure -- they assert different outcomes on
    purpose.
    ``expires_at`` is authoritative only for reclamation statements; readers
    must never use it to hide a row that is otherwise still ``active``.

    Constraint names are part of the contract with the follow-up migration:
    model and migration are two implementations of the same invariant, and
    they must carry identical names or the migration's downgrade has nothing
    to drop by name (mirrors the naming discipline in
    ``AgentTrigger.__table_args__``, trigger.py).

    This table is a leaf -- nothing references it -- so its foreign keys do
    not join the ``tasks`` <-> ``trace_events`` cycle that forces
    ``last_checkpoint_trace_event_id`` (task.py) to be ``use_alter=True``.
    ``create_all`` / ``drop_all`` for this table have been verified to run
    cleanly on both SQLite and PostgreSQL without that flag.
    """

    __tablename__ = "task_interaction_requests"
    __table_args__ = (
        # Uniqueness is always UniqueConstraint, never a unique Index: both
        # backends' get_unique_constraints() report it the same way.
        UniqueConstraint(
            "task_id", "active_slot", name="uq_task_interaction_active_slot"
        ),
        UniqueConstraint(
            "task_id",
            "run_id",
            "request_idempotency_key",
            name="uq_task_interaction_request_identity",
        ),
        Index("ix_task_interaction_requests_task_status", "task_id", "status"),
        Index(
            "ix_task_interaction_requests_resume_trace_event_id",
            "resume_trace_event_id",
        ),
        # ---- vocabulary CHECKs: closed IN-lists ----
        CheckConstraint(
            "status IN ('active','answered','terminated')",
            name="ck_task_interaction_requests_status",
        ),
        CheckConstraint(
            "kind IN ('clarification')",
            name="ck_task_interaction_requests_kind",
        ),
        CheckConstraint(
            "origin IN (" + ",".join(f"'{v}'" for v in ORIGIN_VALUES) + ")",
            name="ck_task_interaction_requests_origin",
        ),
        CheckConstraint(
            "resume_checkpoint_type IN ('agent_execution_checkpoint')",
            name="ck_task_interaction_requests_resume_checkpoint_type",
        ),
        CheckConstraint(
            "resume_locator_format IN ('trace_event_pk_v1')",
            name="ck_task_interaction_requests_resume_locator_format",
        ),
        CheckConstraint(
            "terminal_reason IS NULL OR terminal_reason IN "
            "('deadline_elapsed','run_superseded','answered_via_legacy_resume')",
            name="ck_task_interaction_requests_terminal_reason",
        ),
        # ---- numeric domain floor ----
        # A different category from the slot-scoped "= 1" pin below: the
        # floor rejects values that are not version numbers at all, on every
        # row, while the slot CHECK keeps future versions out of the v1 slot
        # without forbidding them as historical rows.
        CheckConstraint(
            "protocol_version >= 1",
            name="ck_task_interaction_requests_protocol_version_floor",
        ),
        # ---- slot admission CHECKs: active_slot scoping ----
        CheckConstraint(
            "active_slot IS NULL OR active_slot = 1",
            name="ck_task_interaction_requests_active_slot_value",
        ),
        CheckConstraint(
            "(status = 'active') = (active_slot IS NOT NULL)",
            name="ck_task_interaction_requests_active_slot_pairs_status",
        ),
        CheckConstraint(
            "status <> 'active' OR resume_trace_event_id IS NOT NULL",
            name="ck_task_interaction_requests_active_anchor",
        ),
        CheckConstraint(
            "active_slot IS NULL OR protocol_version = 1",
            name="ck_task_interaction_requests_active_protocol",
        ),
        # ---- paired CHECKs ----
        # Whether a pair can be written both ways depends on whether either
        # column in it can be unilaterally cleared by some FK's
        # ON DELETE SET NULL. The only columns affected by SET NULL on this
        # table are resume_trace_event_id and responder_user_id, and neither
        # one enters a two-way pairing below. Every pair not subject to SET
        # NULL is written as a single biconditional; the two SET NULL-
        # affected columns are the only ones outside that form.
        CheckConstraint(
            "(status = 'terminated') = (terminal_reason IS NOT NULL)",
            name="ck_task_interaction_requests_terminal_pairs_status",
        ),
        CheckConstraint(
            "(status = 'terminated') = (terminated_at IS NOT NULL)",
            name="ck_task_interaction_requests_terminated_at_pairs_status",
        ),
        CheckConstraint(
            "(status = 'answered') = (response_payload IS NOT NULL)",
            name="ck_task_interaction_requests_response_pairs_status",
        ),
        CheckConstraint(
            "(status = 'answered') = (responded_at IS NOT NULL)",
            name="ck_task_interaction_requests_responded_at_pairs_status",
        ),
        # Renamed from ck_task_interaction_requests_responder_identity_pairs_responded_at
        # (66 chars): that name exceeds PostgreSQL's 63 character identifier
        # limit and SQLAlchemy raises IdentifierError on create_all rather
        # than truncating it. Semantics are unchanged; see
        # test_no_constraint_name_exceeds_the_postgres_identifier_limit for
        # the structural guard against this happening again.
        CheckConstraint(
            "(responded_at IS NULL) = (responder_identity IS NULL)",
            name="ck_task_interaction_requests_responder_pairs_responded_at",
        ),
        CheckConstraint(
            "expires_at > created_at",
            name="ck_task_interaction_requests_expiry_after_creation",
        ),
        # ---- empty-string CHECKs ----
        # NOT NULL alone does not stop an "or ''" write; on the one nullable
        # column here, neither does its pairing CHECK -- '' satisfies
        # "IS NOT NULL" while naming no one.
        CheckConstraint(
            "run_id <> ''", name="ck_task_interaction_requests_run_id_nonempty"
        ),
        CheckConstraint(
            "resume_event_id <> ''",
            name="ck_task_interaction_requests_resume_event_id_nonempty",
        ),
        CheckConstraint(
            "resume_execution_id <> ''",
            name="ck_task_interaction_requests_resume_execution_id_nonempty",
        ),
        CheckConstraint(
            "resume_run_partition <> ''",
            name="ck_task_interaction_requests_resume_run_partition_nonempty",
        ),
        CheckConstraint(
            "request_idempotency_key <> ''",
            name="ck_task_interaction_requests_request_idempotency_key_nonempty",
        ),
        CheckConstraint(
            "responder_identity <> ''",
            name="ck_task_interaction_requests_responder_identity_nonempty",
        ),
        # Deleted: ck_task_interaction_requests_run_partition_matches. Forcing
        # run_id == resume_run_partition would turn the kind of corruption
        # #1071 is meant to detect into a row that cannot be written at all,
        # which would get misclassified downstream as a slot conflict
        # instead of the corruption it actually is. The comparison instead
        # happens in the primitive's plain-Python validation.
    )

    id = Column(Integer, primary_key=True)
    task_id = Column(
        Integer,
        ForeignKey(
            "tasks.id",
            ondelete="CASCADE",
            name="fk_task_interaction_requests_task_id",
        ),
        nullable=False,
    )
    run_id = Column(String(64), nullable=False)
    kind = Column(String(32), nullable=False)
    protocol_version = Column(Integer, nullable=False)
    status = Column(String(32), nullable=False)
    active_slot = Column(Integer, nullable=True)
    # Frozen at ask time from task.source, falling back to "internal".
    origin = Column(String(20), nullable=False)

    # NOT NULL alone cannot stop a Python None: JSON serialization runs
    # before binding, so None would land as the JSON text 'null'. none_as_null
    # makes it a real SQL NULL, and the NOT NULL fires.
    request_payload = Column(JSON(none_as_null=True), nullable=False)
    # The flag is confined to this table and carried by both of its JSON
    # columns (see class docstring): response_payload uses it to tell "no
    # answer yet" from "answered with JSON null", request_payload uses it so
    # NOT NULL actually fires.
    response_payload = Column(JSON(none_as_null=True), nullable=True)

    request_idempotency_key = Column(
        String(64), nullable=False
    )  # validated against COMMAND_ID_PATTERN by task_interaction_service.create()

    resume_trace_event_id = Column(
        Integer,
        ForeignKey(
            "trace_events.id",
            ondelete="SET NULL",
            name="fk_task_interaction_requests_resume_trace_event_id",
        ),
        nullable=True,
    )
    resume_event_id = Column(String(255), nullable=False)
    resume_execution_id = Column(String(255), nullable=False)
    resume_locator_format = Column(String(32), nullable=False)
    resume_checkpoint_type = Column(String(64), nullable=False)
    resume_run_partition = Column(String(64), nullable=False)

    responder_user_id = Column(
        Integer,
        ForeignKey(
            "users.id",
            ondelete="SET NULL",
            name="fk_task_interaction_requests_responder_user_id",
        ),
        nullable=True,
    )
    # "user:{id}" / "guest:{guest_id}" -- the prefix is a namespace, not
    # decoration.
    responder_identity = Column(String(255), nullable=True)
    terminal_reason = Column(String(32), nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    # Authoritative only for reclamation statements; readers must not hide a
    # row based on this column (see class docstring).
    expires_at = Column(DateTime(timezone=True), nullable=False)
    responded_at = Column(DateTime(timezone=True), nullable=True)
    terminated_at = Column(DateTime(timezone=True), nullable=True)
