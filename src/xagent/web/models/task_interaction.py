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

    ``response_payload`` uses ``JSON(none_as_null=True)`` -- the only column
    in the repository with that flag -- so a Python ``None`` lands as SQL
    NULL instead of the JSON scalar ``null``. Consumers must test "has this
    request been answered" with SQL ``IS NULL``, never with
    ``payload is None`` in Python, since a legitimate JSON ``null`` answer
    value must still read as answered.

    ``terminal_reason`` vocabulary: the three members each have a named
    writer path. ``answered_via_legacy_resume`` is
    not the same outcome as ``answered`` -- that path recovers a free-text
    chat message, not a protocol v1 structured response, and synthesizing a
    payload for it would fabricate a contract the client never produced.
    ``operator_cancelled`` and ``task_terminated`` have no writer yet and are
    intentionally not in the vocabulary; the old name
    ``superseded_by_legacy_resume`` must not come back.

    Clock source for ``ck_task_interaction_requests_expiry_after_creation``:
    ``created_at`` is bound by ``server_default=func.now()``, the
    database's transaction-start clock,
    so the direction of the comparison is safe -- truncation can only make
    ``created_at`` earlier, never later. That said, SQLite's
    ``CURRENT_TIMESTAMP`` only has second precision, so once a TTL shrinks to
    sub-second granularity this CHECK stops protecting against a
    zero-or-negative TTL on SQLite specifically; this needs revisiting before
    any such TTL ships.

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
    obligation is not.
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
            "origin IN ('internal','sdk','a2a','trigger','widget','shared_link')",
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
        # one enters a two-way pairing below.
        CheckConstraint(
            "(status = 'terminated') = (terminal_reason IS NOT NULL)",
            name="ck_task_interaction_requests_terminal_pairs_status",
        ),
        CheckConstraint(
            "status <> 'answered' OR response_payload IS NOT NULL",
            name="ck_task_interaction_requests_answered_pairs_response",
        ),
        CheckConstraint(
            "status = 'answered' OR response_payload IS NULL",
            name="ck_task_interaction_requests_unanswered_has_no_response",
        ),
        CheckConstraint(
            "status <> 'answered' OR responded_at IS NOT NULL",
            name="ck_task_interaction_requests_answered_pairs_responded_at",
        ),
        CheckConstraint(
            "status = 'answered' OR responded_at IS NULL",
            name="ck_task_interaction_requests_unanswered_has_no_responded_at",
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
        # ---- empty-string CHECKs: NOT NULL alone does not stop an "or ''" write ----
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

    request_payload = Column(JSON, nullable=False)  # NOT NULL: no none_as_null needed
    # The only none_as_null=True column in the repository (see class
    # docstring): lets consumers tell "no answer yet" (SQL NULL) apart from
    # "answered with a JSON null value" using SQL IS NULL.
    response_payload = Column(JSON(none_as_null=True), nullable=True)

    request_idempotency_key = Column(
        String(64), nullable=False
    )  # validated against COMMAND_ID_PATTERN by the first production writer

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
