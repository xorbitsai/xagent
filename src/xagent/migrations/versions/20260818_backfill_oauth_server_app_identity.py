"""backfill auth.app_id onto unstamped OAuth-transport server rows

Rows provisioned before ``_ensure_user_mcp_server`` wrote OAuth metadata
carry no ``auth`` payload at all. Every read path that resolves a
server row back to its catalog app prefers the stable ``auth.app_id`` and
falls back to the row's *exact current display name*
(``get_app_for_mcp_server``) -- so an unstamped row's identity lives and
dies by that name: once it stops matching, ``/api/mcp/servers``
enrichment reports no ``app_id``, the connector picker persists the app's
current display name, and the runtime -- which knows the row only under
its stored name -- resolves the selection to zero tools, silently (#1429,
surfaced reviewing #1403).

Which apps can drift that way is worth stating precisely, because the
at-risk population is real and was produced by the normal API, not by
out-of-band edits:

- Genuine **builtin** apps cannot be renamed at all: ``name``,
  ``transport`` and ``provider_name`` are in
  ``_BUILTIN_PROTECTED_FIELDS`` (``admin_mcp.py``) and a PATCH touching
  them 409s, while ``_app_to_dict`` serves the registry's name
  regardless of what the row stores.
- **Admin-created** OAuth catalog apps *are* freely renameable, and
  ``classify_app_auth`` labels them ``builtin_oauth`` too (purely on
  ``transport == "oauth"``). Admin app CRUD shipped in ``c8642b8c``
  (2026-04-24); ``_ensure_user_mcp_server`` did not write ``auth`` at
  all until ``cb724dbb`` (2026-06-26). For those ~63 days the ordinary
  OAuth-connect flow, on an app an admin could rename the next day,
  produced exactly the unstamped row this migration exists for. Any
  deployment that connected such an app in that window still carries
  those rows.
- After ``cb724dbb`` every provisioned row carries the stamp, so nothing
  new joins the population; what remains is that historical residue plus
  whatever direct database edits or a code-level registry rename leave
  behind.

The stamp is derived while the name still matches, the only moment it
can be derived safely:

- Candidates are ``mcp_servers`` rows with ``transport = 'oauth'`` whose
  ``auth`` is missing or lacks a nonblank *string* ``app_id`` — a
  non-string value (e.g. an integer) is malformed metadata the read path
  rejects outright, so such a row stays a candidate and a successful match
  overwrites the malformed value. A row whose whole ``auth`` payload is a
  non-dict, or whose stored ``provider`` is present but not a usable
  string, is left alone instead: both resolve fine today through the name
  fallback, and this migration is irreversible.
- Identities are opaque and matched/stored as **raw strings**, never
  trimmed or coerced: the identity paths this migration feeds compare them
  exactly (``get_app_by_id``, ``get_app_by_name``, and through them
  ``get_app_for_mcp_server``), so stamping a trimmed variant of a padded
  catalog id would write a value the exact lookup can never resolve.
  Elsewhere in the connector listing, names *are* matched normalized, and
  ``public_mcp_apps.name`` carries no uniqueness constraint, so
  whitespace/case variants are producible — they simply do not match here,
  and under-matching leaves the name-fallback shim in charge.
- Each is matched against ``public_mcp_apps`` (builtin apps are seeded
  into that table too) by its exact stored display name -- the value
  ``_ensure_user_mcp_server`` named the row after, *unless* a builtin's
  stored name has drifted from the code registry ``_app_to_dict`` serves
  it from, in which case this simply does not match and the row is left
  alone. A name shared by two OAuth apps resolves nothing: picking either
  would be a guess.
- The stored name is the **only** signal used to resolve an app. There is
  deliberately no provider fallback: no writer this codebase has ever
  shipped produces a row carrying a provider but no ``app_id``
  (``_oauth_auth_metadata`` writes ``app_id`` first and unconditionally,
  and the generic servers API cannot author ``transport="oauth"`` at all --
  ``MCPServerConfig`` rejects it), so that population has never existed,
  while the population that *does* exist -- rows written with no ``auth``
  at all before ``cb724dbb`` -- carries no provider to fall back to. A
  provider-keyed pass would defend nothing real, and it was the only place
  where two rows could compete for one identity.
- This migration cannot stop a *future* rename from recreating the
  condition: ``_ensure_user_mcp_server`` still looks its row up by display
  name, so the next connect after a rename creates a second row that the
  unconditional writer stamps with the same ``app_id``. That write-path
  root cause is tracked in #1569; what follows is only about not creating
  the condition here.
- An app already carried by another row is never handed to a second one.
  ``_ensure_user_mcp_server`` creates a *new* row when a rename makes the
  old one unfindable, so orphan pairs for one app are this migration's own
  target population, and two rows sharing an identity would make
  ``_lookup_oauth_server_for_app`` -- which reads an unordered query --
  pick nondeterministically between them for configure/disconnect. The
  refused row keeps the behavior it had before this migration.
- Only apps whose own ``transport`` is ``oauth`` are eligible: a
  same-named non-OAuth app is a different connector shape, and stamping
  its id onto an oauth-transport row would manufacture exactly the
  cross-shape identity confusion this migration exists to remove. Such a
  name is *refused*, not merely treated as unmatched -- it is evidence
  about the row rather than an absence of evidence.
- Rows that resolve to nothing are left untouched, keeping the exact-name
  fallback in ``get_app_for_mcp_server`` as their compatibility shim. That
  is a deliberate trade rather than a free win: ``get_app_for_mcp_server``
  never falls back to the name once ``app_id`` is present, so a stamp that
  later goes dangling (an admin deletes a custom OAuth app and recreates
  it under the same name with a new id) resolves to nothing where an
  unstamped row would still have matched by name. Stamping only
  unambiguous matches is what keeps that trade narrow.
- A drifted builtin row is classified here from its *stored* columns,
  while runtime reads take ``transport``/``provider_name``/``name`` from
  the code registry (``_app_to_dict``) and only warn about drift. The two
  can therefore disagree about a drifted row; the blast radius is one
  unambiguous stamp on a row whose stored shape says ``oauth``.
- Only rows whose stored name still matches can be stamped. In a
  deployment where the drift already happened, nothing here can rescue the
  row: this hardens the population still resolvable today, and cannot
  retroactively repair one whose identity signal is already gone.
- A row whose own ``auth.provider`` contradicts the name-resolved app's
  provider is refused — the same conflict the provisioning writer
  (``_ensure_server_matches_oauth_app``) refuses with a ValueError. This is
  a guard on the name path, not a remnant of any fallback:
  ``_is_oauth_server_for_app`` checks a stored provider *even when the
  app_id matches*, so stamping here would leave a row that matches no app
  at all, where today it still matches by name.
- Only ``app_id`` is written, into a *copy* of the existing auth dict;
  nothing else is added or touched. A ``provider`` is deliberately not
  written: ``app_id`` alone resolves the app, while
  ``_is_oauth_server_for_app`` checks a stored provider *even when the
  app_id matches*, so writing one sourced from a possibly-drifted
  ``public_mcp_apps.provider_name`` could break a match that works today.
- Rows already carrying a nonblank ``app_id`` are not candidates, and the
  identities they already carry seed the claim set, so a rerun -- including
  after the no-op downgrade -- neither re-stamps them nor hands their
  identity to the next-best row. That is what makes the idempotence claim
  hold across runs, not only within one.

Offline (``--sql``) mode **raises** instead of no-opping: Alembic emits
the ``alembic_version`` bookkeeping even for an empty migration body, so
a silently-skipped offline run would advance the version while touching
no row — and a later online upgrade would then skip this revision
forever. Failing loudly forces the one correct path: run this revision
online. The schema is untouched either way.

The downgrade is a deliberate no-op. Removing ``auth.app_id`` would need
to distinguish stamped rows from rows the post-metadata writer created,
which the data cannot express -- and an extra stable identity is harmless
to every pre-migration reader (they all prefer ``app_id`` and only fall
back to names when it is absent).

Revision ID: 20260818_backfill_oauth_server_app_identity
Revises: 20260820_merge_jira_posthog_heads
Create Date: 2026-08-18
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import NamedTuple

import sqlalchemy as sa
from alembic import op

logger = logging.getLogger(__name__)

# revision identifiers, used by Alembic.
revision: str = "20260818_backfill_oauth_server_app_identity"
down_revision: str | None = "20260820_merge_jira_posthog_heads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


class _CatalogApp(NamedTuple):
    """A catalog app's ``app_id``, plus its provider.

    ``app_id`` is the identity that gets stamped. ``provider`` is *not* an
    identity signal -- nothing resolves an app by it -- and is carried only so
    the name path can refuse a row whose own stored provider contradicts it.
    """

    app_id: str
    provider: str | None


class _Candidate(NamedTuple):
    """An oauth row still needing a stamp, with its usable stored provider."""

    row: sa.RowMapping
    auth: dict
    provider: str | None


def _same_provider(left: str, right: str) -> bool:
    """Whether two provider strings name the same provider.

    Normalized the way the runtime compares them (``_normalize_app_key``:
    casefold, strip, whitespace-to-hyphen), not exactly. Providers are not
    identities here -- they are only ever used to detect a *contradiction* --
    so comparing them more strictly than the runtime does would refuse a row
    over a difference the runtime already treats as no difference, and this
    migration runs once and cannot be undone.
    """
    return "-".join(left.strip().lower().split()) == "-".join(
        right.strip().lower().split()
    )


def _nonblank_str(value: object) -> str | None:
    """The value itself when it is a str with non-whitespace content, else None.

    Raw, never trimmed or coerced: identities are opaque and every downstream
    comparison is exact (``get_app_by_id`` compares ``PublicMCPApp.app_id``
    directly; ``get_app_for_mcp_server`` rejects a non-string ``auth.app_id``).
    Stamping a trimmed variant of a padded catalog id would write a value the
    exact lookup can never resolve — strictly worse than not stamping. This
    helper decides only *whether* a usable string exists; the string used for
    matching and for storage is always the raw one.
    """
    if isinstance(value, str) and value.strip():
        return value
    return None


def upgrade() -> None:
    if op.get_context().as_sql:
        # Fail loudly rather than no-op: returning here would still let
        # Alembic emit the alembic_version bookkeeping, so an operator who
        # applies the generated --sql script advances the version while no
        # row was read or updated — and a later online upgrade then skips
        # this revision forever, leaving the legacy rows unstamped.
        raise RuntimeError(
            "20260818_backfill_oauth_server_app_identity is a data migration "
            "and cannot run in offline (--sql) mode: applying generated SQL "
            "would advance alembic_version without performing the backfill. "
            "Run `alembic upgrade` online for this revision."
        )

    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    # A bare database has neither table until create_all runs; there is
    # nothing to backfill there. Logged rather than returned silently: Alembic
    # still commits the version bump, so a skip here is permanent, and that is
    # exactly the "advance the version without backfilling" shape the offline
    # branch above raises over. It is benign only because a database with no
    # mcp_servers has no rows to stamp -- an operator seeing this line on a
    # populated deployment is looking at a search_path problem.
    if "mcp_servers" not in tables or "public_mcp_apps" not in tables:
        logger.warning(
            "%s: mcp_servers/public_mcp_apps not visible on this connection's "
            "search_path; nothing backfilled and the revision is marked applied",
            revision,
        )
        return

    mcp_servers = sa.table(
        "mcp_servers",
        sa.column("id", sa.Integer),
        sa.column("name", sa.String),
        sa.column("transport", sa.String),
        sa.column("auth", sa.JSON),
    )
    public_mcp_apps = sa.table(
        "public_mcp_apps",
        sa.column("app_id", sa.String),
        sa.column("name", sa.String),
        sa.column("transport", sa.String),
        sa.column("provider_name", sa.String),
    )

    # Exact display name -> app, for OAuth apps only.
    # A name shared by two OAuth apps is recorded as ambiguous: it then
    # resolves nothing rather than picking whichever app was seeded first.
    # Keys and stored values are the raw strings — identities are opaque, and
    # the identity paths compare them exactly. Only ``transport`` is folded,
    # because it is a shape enum rather than an identity -- and a little more
    # tolerantly than classify_app_auth, which lowercases without stripping.
    apps_by_name: dict[str, _CatalogApp] = {}
    ambiguous_names: set[str] = set()
    names_of_other_shapes: set[str] = set()
    for app_row in bind.execute(sa.select(public_mcp_apps)).mappings():
        if str(app_row["transport"] or "").strip().lower() != "oauth":
            # Recorded, not ignored: a row whose name belongs to a non-oauth
            # app must be told apart from a row whose name belongs to nothing.
            # Both miss apps_by_name, but the first is positive evidence of a
            # *different-shape* app. It only changes the outcome when one name
            # is carried by both an oauth and a non-oauth app -- then this is
            # what stops the oauth app from claiming the row on a name whose
            # ownership is genuinely contested.
            other_name = _nonblank_str(app_row["name"])
            if other_name:
                names_of_other_shapes.add(other_name)
            continue
        app_id = _nonblank_str(app_row["app_id"])
        if not app_id:
            continue
        entry = _CatalogApp(app_id, _nonblank_str(app_row["provider_name"]))
        name = _nonblank_str(app_row["name"])
        if name:
            if name in apps_by_name:
                ambiguous_names.add(name)
            apps_by_name[name] = entry

    # Materialized before the loop: the loop UPDATEs the same table, and
    # stepping a live pysqlite cursor across a mutating table has formally
    # unspecified row visibility. The table is small (one row per connector).
    # Ordered by id. No longer load-bearing for correctness -- with the
    # provider pass gone, mcp_servers.name is unique and each name maps to one
    # app, so two candidates can never contest one identity -- but it keeps the
    # UPDATE order and the per-row audit log reproducible across runs and
    # backends, which matters for a migration with no downgrade.
    candidates = (
        bind.execute(
            sa.select(mcp_servers)
            # Exact, matching the writer and every runtime reader: the five
            # gates that make an oauth row *work* (tools/config.py's credential
            # injection, _is_oauth_server_for_app, _enrich_oauth_server_info,
            # _ensure_server_matches_oauth_app) all compare `transport ==
            # "oauth"` unfolded. A row stored "OAuth" passes none of them, so
            # it is a dead row: stamping it would be worthless to it and, worse,
            # could burn the app's one identity -- the `claimed` guard would
            # then refuse the genuine row, irreversibly (downgrade is a no-op).
            # Folding here would make this migration the only place with a
            # wider notion of "oauth" than everything consuming its output.
            .where(mcp_servers.c.transport == "oauth")
            .order_by(mcp_servers.c.id)
        )
        .mappings()
        .all()
    )

    # Every identity any row already carries, read *without* a transport
    # filter. Deliberately wider than the candidate query above, because the
    # reader that matters here is wider too: get_app_for_mcp_server resolves
    # from auth.app_id with no transport check at all, and the OAuth-disconnect
    # cleanup path walks a user's whole server list through it. So a row whose
    # transport is "OAuth" -- or anything else -- still makes its app_id taken,
    # even though that row could never serve the connector itself. Filtering
    # this by transport left such an id invisible to the claim guard, which
    # could then hand it to a second row: the one thing this migration promises
    # it will not do.
    already_stamped: set[str] = set()
    for row in bind.execute(
        sa.select(mcp_servers.c.auth).where(mcp_servers.c.auth.is_not(None))
    ).mappings():
        raw = row["auth"]
        if isinstance(raw, dict):
            existing_any = _nonblank_str(raw.get("app_id"))
            if existing_any:
                already_stamped.add(existing_any)

    # Split the candidates into rows already carrying a usable identity and
    # rows still needing one. A nonblank *string* app_id is the only shape the
    # read path accepts (get_app_for_mcp_server rejects a non-string app_id
    # outright), so that is what counts as "already stamped".
    #
    # A non-dict auth payload is garbage on an oauth row, but it is left
    # alone rather than replaced: such a row resolves fine today through the
    # name fallback, this migration is irreversible (downgrade is a no-op),
    # and destroying a value we do not have to is the wrong default for a
    # data migration. NULL auth is the primary target and is not that case.
    pending: list[_Candidate] = []
    for row in candidates:
        raw_auth = row["auth"]
        if raw_auth is not None and not isinstance(raw_auth, dict):
            continue
        auth: dict = raw_auth if isinstance(raw_auth, dict) else {}
        if _nonblank_str(auth.get("app_id")):
            continue
        raw_provider = auth.get("provider")
        if raw_provider is not None and not isinstance(raw_provider, str):
            # Present but not a string at all (a list, a number). The
            # provider-conflict gate below cannot compare it, and
            # _is_oauth_server_for_app would reject the row against any app
            # once stamped, so a stamp would buy nothing. Refuse, matching
            # _ensure_server_matches_oauth_app's strictness about
            # contradictory stored metadata. A *blank* string is different:
            # it carries no claim, so it counts as absent, like a missing key.
            logger.warning(
                "%s: mcp_servers row %s has a malformed auth.provider; "
                "leaving it unstamped",
                revision,
                row["id"],
            )
            continue
        pending.append(_Candidate(row, auth, _nonblank_str(raw_provider)))

    # One pass: a row is resolved by its exact stored name and nothing else.
    # There is deliberately no provider fallback -- no writer this codebase has
    # ever shipped produces a row with a provider but no app_id
    # (_oauth_auth_metadata writes app_id first and unconditionally; the
    # generic servers API cannot author transport="oauth" at all), so such a
    # population has never existed, and the population that *does* exist --
    # rows written with no auth at all before cb724dbb -- carries no provider
    # to fall back to. A provider-keyed pass would defend nothing real while
    # being the only place two rows could compete for one identity.
    #
    # `claimed` still starts from the identities rows already carry: the
    # colliding partner is the row the writer stamped when a rename left the
    # old one orphaned, and it never enters `pending`. Seeding it is also what
    # makes idempotence hold across a downgrade-then-upgrade rerun.
    claimed: set[str] = set(already_stamped)

    def claim(candidate: _Candidate, app: _CatalogApp) -> str | None:
        """Reserve the identity for this row, or refuse it as already taken.

        Reserving as we go is what lets resolution and the write share one
        pass: a candidate's decision only ever depends on the candidates
        already processed, never on later ones.
        """
        if app.app_id in claimed:
            logger.warning(
                "%s: app_id %r is already carried by another row; leaving "
                "mcp_servers row %s to its pre-migration name fallback",
                revision,
                app.app_id,
                candidate.row["id"],
            )
            return None
        claimed.add(app.app_id)
        return app.app_id

    stamped = 0
    for candidate in pending:
        row_name = str(candidate.row["name"]) if candidate.row["name"] else ""
        if row_name in names_of_other_shapes:
            # The name belongs to an app of another transport -- evidence about
            # this row, so it refuses rather than resolving to nothing.
            continue
        if row_name in ambiguous_names:
            # Two OAuth apps share this exact name; picking either would be a
            # guess, so the row keeps its read-time name fallback.
            continue
        app = apps_by_name.get(row_name)
        if app is None:
            continue
        if (
            candidate.provider
            and app.provider
            and not _same_provider(candidate.provider, app.provider)
        ):
            # The row's own provider contradicts the name-resolved app -- the
            # same conflict _ensure_server_matches_oauth_app refuses with a
            # ValueError. This is a guard on the name path, not a leftover of
            # any fallback: _is_oauth_server_for_app checks a stored provider
            # *even when the app_id matches*, so stamping here would leave a
            # row that matches no app at all, where today it still matches by
            # name.
            continue
        app_id = claim(candidate, app)
        if app_id is None:
            continue

        row_id = candidate.row["id"]
        # Re-read immediately before writing: candidates were materialized up
        # front, and this migration deliberately targets rows a live
        # provisioning flow can still be writing to. Without this, a blind
        # UPDATE of the whole auth column would silently discard whatever was
        # committed in between -- most plausibly a provider the writer added.
        current = bind.execute(
            sa.select(mcp_servers.c.auth).where(mcp_servers.c.id == row_id)
        ).scalar()
        if current != candidate.auth and not (current is None and candidate.auth == {}):
            logger.warning(
                "%s: mcp_servers row %s changed under us; leaving it unstamped",
                revision,
                row_id,
            )
            continue
        # Only app_id is written. The row's own provider (when it has one) is
        # left as-is and none is added: app_id alone resolves the app, while
        # _is_oauth_server_for_app checks a stored provider *even when the
        # app_id matches*, so writing one sourced from a possibly-drifted
        # public_mcp_apps.provider_name could break a match that works today.
        new_auth = dict(candidate.auth)
        new_auth["app_id"] = app_id
        bind.execute(
            sa.update(mcp_servers)
            .where(mcp_servers.c.id == row_id)
            .values(auth=new_auth)
        )
        # Audit at WARNING, not INFO, on purpose: Alembic loads version files
        # by bare filename, so this logger is not the configured `alembic`
        # qualname logger and alembic.ini's root=WARN drops INFO from it
        # entirely. Stamping is irreversible (downgrade is a no-op), so an
        # operator reconstructing or reversing a bad backfill needs these lines
        # to actually appear.
        logger.warning(
            "%s: stamped mcp_servers row %s (name %r) with app_id %r",
            revision,
            row_id,
            candidate.row["name"],
            app_id,
        )
        stamped += 1

    logger.warning(
        "%s: stamped %s of %s unstamped oauth row(s); %s left unstamped",
        revision,
        stamped,
        len(pending),
        len(pending) - stamped,
    )


def downgrade() -> None:
    # Deliberate no-op: stamped rows are indistinguishable from rows the
    # post-metadata writer created, and the extra stable identity is harmless
    # to every pre-migration reader (see module docstring).
    pass
