"""Personal-skill writes are validated before they commit.

The alternative -- commit, reload, and delete the row if it cannot be proved
to be ours -- needs the compensating delete to identify the row it is undoing,
and neither a name nor a primary key does that: names are reusable and SQLite
recycles the id of a deleted row. Validating first removes the need to undo
anything, so there is no delete that can hit the wrong row.

These drive the real writer and the real routes against a live SQLite session
and assert on the table.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from xagent.skills.library import SkillRecord, SkillScopeContext
from xagent.web.api import skill_hub
from xagent.web.api.skill_hub import (
    _assert_bundle_parses,
    _delete_personal_skill,
    _write_personal_skill,
)

SKILL_MD = b"# Test Skill\n\n## Description\nA test skill.\n"


def _factory(tmp_path, name="skills.db", user_ids=(7,)):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from xagent.web.models.database import Base
    from xagent.web.models.skill import UserSkill, UserSkillFile
    from xagent.web.models.user import User

    engine = create_engine(f"sqlite:///{tmp_path / name}")
    Base.metadata.create_all(
        engine, tables=[User.__table__, UserSkill.__table__, UserSkillFile.__table__]
    )
    factory = sessionmaker(bind=engine)
    with factory() as db:
        for uid in user_ids:
            db.execute(
                User.__table__.insert().values(
                    id=uid, username=f"u{uid}", password_hash="h", is_admin=False
                )
            )
        db.commit()
    return factory


def _user(uid=7):
    return SimpleNamespace(id=uid)


def _rows(db):
    from xagent.web.models.skill import UserSkill

    return sorted((int(r.id), str(r.name)) for r in db.query(UserSkill).all())


# ── validation happens before the write ──────────────────────────────────────


class TestValidationPrecedesTheWrite:
    """Whatever ``SkillManager.reload`` would skip must be refused up front."""

    @pytest.mark.parametrize(
        "files,expected",
        [
            ({"SKILL.md": b"\xe9\xff\xfe latin-1"}, "UTF-8"),
            ({"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe not utf8"}, "UTF-8"),
        ],
        ids=["skill-md-not-utf8", "template-md-not-utf8"],
    )
    def test_an_unloadable_bundle_never_reaches_the_table(
        self, tmp_path, files, expected
    ):
        db = _factory(tmp_path)()
        with pytest.raises(HTTPException) as exc:
            _write_personal_skill(db=db, user=_user(), name="bad", files=files)
        assert exc.value.status_code == 400
        assert expected in exc.value.detail
        assert _rows(db) == [], "nothing may be written when validation fails"
        db.close()

    def test_the_name_stays_free_after_a_refused_write(self, tmp_path):
        """The failure the pre-commit check exists to prevent: a row that is
        invisible to every name-keyed verb but still squats the name."""
        db = _factory(tmp_path)()
        with pytest.raises(HTTPException):
            _write_personal_skill(
                db=db, user=_user(), name="alpha", files={"SKILL.md": b"\xff\xfe"}
            )
        _write_personal_skill(
            db=db, user=_user(), name="alpha", files={"SKILL.md": SKILL_MD}
        )
        assert _rows(db) == [(1, "alpha")]
        db.close()

    def test_validation_runs_on_the_normalized_bundle(self, tmp_path):
        """Normalization rewrites paths, so the raw input and the bundle that
        actually gets stored can parse differently.

        ``/template.md`` is invisible to the parser as written -- it only
        decodes the exact keys ``SKILL.md`` and ``template.md`` -- but
        normalization strips the leading slash, and the stored bundle then has
        a ``template.md`` the parser does decode and chokes on. Validating the
        raw input would wave this through and commit a row reload cannot load,
        which is the whole failure the pre-commit check exists to stop.
        """
        db = _factory(tmp_path)()
        raw = {"SKILL.md": SKILL_MD, "/template.md": b"\xff\xfe"}

        from xagent.skills.parser import SkillParser

        SkillParser.parse_bundle(name="c", files=raw)  # premise: raw parses

        with pytest.raises(HTTPException) as exc:
            _write_personal_skill(db=db, user=_user(), name="alpha", files=raw)
        assert exc.value.status_code == 400
        assert _rows(db) == [], "the stored form is what must be judged"
        db.close()

    def test_archive_cruft_is_dropped_rather_than_validated(self, tmp_path):
        """The other half of normalizing first: cruft never reaches the parser
        and never reaches the table, so a stray .DS_Store cannot fail a write."""
        db = _factory(tmp_path)()
        _write_personal_skill(
            db=db,
            user=_user(),
            name="alpha",
            files={"SKILL.md": SKILL_MD, ".DS_Store": b"\x00\x01\xff"},
        )
        from xagent.web.models.skill import UserSkillFile

        stored = sorted(f.path for f in db.query(UserSkillFile).all())
        assert stored == ["SKILL.md"]
        db.close()


class TestAssertBundleParses:
    """The validator tracks the parser rather than restating its rules."""

    def test_a_good_bundle_passes(self):
        _assert_bundle_parses({"SKILL.md": SKILL_MD})

    def test_it_refuses_exactly_what_the_parser_refuses(self):
        """The guarantee that makes a committed row trustworthy: reload skips
        a record only when parse_bundle raises, so anything the validator lets
        through must be something parse_bundle accepts."""
        from xagent.skills.parser import SkillParser

        cases = [
            {"SKILL.md": SKILL_MD},
            {"SKILL.md": b"---\nname: x\n---\n\n# X\n"},
            {"SKILL.md": b"\xe9\xff\xfe"},
            {"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe"},
            {"SKILL.md": b"---\n: : :\n---\n# X\n"},
            {"SKILL.md": SKILL_MD, "logo.png": b"\x89PNG\r\n\x1a\n"},
            {"other.md": b"x"},
        ]
        for files in cases:
            try:
                SkillParser.parse_bundle(name="candidate", files=files)
                parser_ok = True
            except Exception:
                parser_ok = False
            try:
                _assert_bundle_parses(files)
                validator_ok = True
            except HTTPException:
                validator_ok = False
            assert parser_ok == validator_ok, files

    def test_a_non_utf8_decoded_file_names_the_offending_file(self):
        """parse_bundle decodes SKILL.md and template.md; the message has to
        say which of them failed, since the exception itself does not."""
        with pytest.raises(HTTPException) as exc:
            _assert_bundle_parses({"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe"})
        assert "template.md" in exc.value.detail

    def test_an_unexpected_parser_failure_keeps_its_5xx(self, monkeypatch):
        """A service defect must not be presented to the caller as bad input.

        The broad catch used to turn any exception into a 400, so a bug in the
        parser looked like a malformed bundle and lost its 5xx signal. Nothing
        has been written when this runs, so letting it propagate cannot leave
        a partial write behind.
        """
        from xagent.skills import parser as parser_mod

        def boom(*a, **k):
            raise RuntimeError("a defect, not bad input")

        monkeypatch.setattr(parser_mod.SkillParser, "parse_bundle", boom)
        with pytest.raises(RuntimeError):
            _assert_bundle_parses({"SKILL.md": SKILL_MD})

    def test_a_missing_skill_md_is_a_400(self):
        """The parser's own ValueError is genuine bad input and stays a 400."""
        with pytest.raises(HTTPException) as exc:
            _assert_bundle_parses({"other.md": b"x"})
        assert exc.value.status_code == 400

    def test_the_culprit_named_is_one_the_parser_actually_decodes(self):
        """parse_bundle only decodes SKILL.md and template.md. Scanning every
        file blamed a binary asset the parser never reads and hid the real
        cause, sending the user to delete a perfectly good logo."""
        with pytest.raises(HTTPException) as exc:
            _assert_bundle_parses(
                {
                    "SKILL.md": SKILL_MD,
                    "logo.png": b"\x89PNG\r\n\x1a\n\xff",
                    "template.md": b"\xff\xfe",
                }
            )
        assert "template.md" in exc.value.detail
        assert "logo.png" not in exc.value.detail

    def test_a_non_decoded_asset_is_not_rejected(self):
        """parse_bundle only decodes SKILL.md and template.md, so a binary
        asset is not a load failure and must not be refused. Validating more
        than the parser does would reject bundles reload would happily serve."""
        _assert_bundle_parses({"SKILL.md": SKILL_MD, "logo.png": b"\x89PNG\r\n\x1a\n"})


# ── deletion addressing ──────────────────────────────────────────────────────


class TestDeletePersonalSkill:
    @pytest.mark.parametrize("target", ["alpha", "beta"], ids=["first", "second"])
    def test_deleting_by_name_removes_exactly_that_row(self, tmp_path, target):
        """Both positions are exercised on purpose. Deleting only the first
        row happens to look right when the name filter is missing entirely --
        the query would return row 1 either way -- so a same-owner wrong-row
        delete would go unnoticed."""
        db = _factory(tmp_path)()
        for name in ("alpha", "beta"):
            _write_personal_skill(
                db=db, user=_user(), name=name, files={"SKILL.md": SKILL_MD}
            )
        _delete_personal_skill(db=db, user=_user(), name=target)
        survivor = "beta" if target == "alpha" else "alpha"
        assert [name for _, name in _rows(db)] == [survivor]
        db.close()

    def test_another_owners_skill_is_refused(self, tmp_path):
        db = _factory(tmp_path, user_ids=(7, 8))()
        _write_personal_skill(
            db=db, user=_user(8), name="theirs", files={"SKILL.md": SKILL_MD}
        )
        with pytest.raises(HTTPException) as exc:
            _delete_personal_skill(db=db, user=_user(7), name="theirs")
        assert exc.value.status_code == 404
        assert _rows(db) == [(1, "theirs")], "no cross-owner delete"
        db.close()


# ── the duplicate-name race ──────────────────────────────────────────────────


class TestDuplicateNameRace:
    """The pre-check SELECT and the INSERT are not atomic across sessions.

    The interleaving that matters is: *both* attempts pass the pre-check, and
    only then does either insert. Synchronising after the first flush cannot
    produce it -- under SQLite the first flusher holds the write lock, so the
    second attempt can fail before it ever reaches the barrier, and the test
    then passes for a reason unrelated to what it claims. The barrier here sits
    between the pre-check and the flush.
    """

    @staticmethod
    def _run_race(factory, *, tags=("a", "b"), name="dup"):
        precheck_done = threading.Barrier(len(tags), timeout=30)
        results: dict[str, object] = {}
        reached: set[str] = set()
        lock = threading.Lock()

        def attempt(tag: str) -> None:
            db = factory()
            original_query = db.query
            synced = False

            def query_then_sync(*args, **kwargs):
                nonlocal synced
                out = original_query(*args, **kwargs)
                if synced:
                    return out
                synced = True

                class _Synced:
                    """Block after the pre-check SELECT has been evaluated.

                    ``.first()`` is what runs the statement, so waiting inside
                    it -- not at ``query()`` -- guarantees this attempt really
                    saw an empty table before the other one inserts.
                    """

                    def filter(self, *a, **k):
                        self._inner = out.filter(*a, **k)
                        return self

                    def first(self):
                        row = self._inner.first()
                        with lock:
                            reached.add(tag)
                        precheck_done.wait()
                        return row

                return _Synced()

            db.query = query_then_sync  # type: ignore[method-assign]
            try:
                _write_personal_skill(
                    db=db, user=_user(), name=name, files={"SKILL.md": SKILL_MD}
                )
                results[tag] = "committed"
            except HTTPException as exc:
                results[tag] = f"http-{exc.status_code}"
            except BaseException as exc:
                results[tag] = f"raised-{type(exc).__name__}: {exc}"
            finally:
                db.close()

        threads = [threading.Thread(target=attempt, args=(t,)) for t in tags]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)
        assert not any(t.is_alive() for t in threads), "a racing thread hung"
        return results, reached

    def test_both_prechecks_really_run_before_either_insert(self, tmp_path):
        """The premise the outcome assertion rests on. If a writer could fail
        before the barrier -- the flaw in the original test -- this fails."""
        _, reached = self._run_race(_factory(tmp_path))
        assert reached == {"a", "b"}, (
            "both attempts must clear the pre-check before either inserts; "
            f"only {sorted(reached)} got there"
        )

    def test_the_loser_gets_409_and_one_row_survives(self, tmp_path):
        """Without the IntegrityError guard the loser leaks a 500."""
        factory = _factory(tmp_path)
        results, _ = self._run_race(factory)
        assert sorted(results.values()) == ["committed", "http-409"], results

        db = factory()
        assert _rows(db) == [(1, "dup")]
        from xagent.web.models.skill import UserSkillFile

        assert db.query(UserSkillFile).count() == 1, "no orphaned file rows"
        db.close()

    def test_an_unrelated_integrity_error_is_not_reported_as_a_duplicate(
        self, tmp_path
    ):
        """The guard must not swallow every IntegrityError. A foreign-key
        violation is a bug, not a name conflict, and has to stay loud.

        Raised by the real INSERT against a real constraint, not by a
        monkeypatched ``flush``: patching flush makes the *pre-check query's*
        autoflush raise, so the error never reaches the guard at all and the
        test passes whatever the guard does.
        """
        from sqlalchemy import text
        from sqlalchemy.exc import IntegrityError

        db = _factory(tmp_path)()
        db.execute(text("PRAGMA foreign_keys=ON"))
        with pytest.raises(IntegrityError) as exc:
            _write_personal_skill(
                db=db, user=_user(9999), name="orphan", files={"SKILL.md": SKILL_MD}
            )
        assert "foreign key" in str(exc.value).lower(), str(exc.value)
        db.close()

    def test_the_guard_only_recognises_the_skill_name_constraint(self):
        from sqlalchemy.exc import IntegrityError

        from xagent.web.api.skill_hub import _is_skill_name_unique_violation

        def err(message):
            return IntegrityError("INSERT", {}, Exception(message))

        assert _is_skill_name_unique_violation(err("uq_user_skill_name violated"))
        assert _is_skill_name_unique_violation(
            err("UNIQUE constraint failed: user_skills.user_id, user_skills.name")
        )
        assert not _is_skill_name_unique_violation(err("FOREIGN KEY constraint failed"))
        assert not _is_skill_name_unique_violation(
            err(
                "UNIQUE constraint failed: user_skill_files.skill_id, "
                "user_skill_files.path"
            )
        ), "the per-file constraint is a different failure"
        assert not _is_skill_name_unique_violation(
            err("NOT NULL constraint failed: user_skills.name")
        )


# ── route behaviour ──────────────────────────────────────────────────────────


def _install_manager(monkeypatch, loader):
    class _Manager:
        async def get_skill(self, name):
            return loader(name)

    async def _scoped(*args):
        return _Manager()

    monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped)


async def _create(db, name="alpha", user_id=7):
    return await skill_hub.create_skill(
        skill_hub.CreateSkillRequest(
            name=name, skill_md=SKILL_MD.decode(), scope="personal"
        ),
        SimpleNamespace(),
        SkillScopeContext(user_id=user_id, metadata={}),
        db,
        SimpleNamespace(id=user_id),
    )


class TestCreateAcceptsAnyCompliantProvider:
    """A committed row is never removed on the strength of how a provider
    chose to represent it.

    ``SkillRecord.path`` is ``str | None`` and provider-defined, and
    ``_all_file_names`` exists so a provider may list an asset without
    eagerly loading its bytes. A custom provider replaces the default one
    wholesale, so inferring "is this row mine?" from the default's display
    path or from an in-memory bundle would reject a valid readback -- and,
    under a compensating design, delete the skill the user just created.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "record",
        [
            SkillRecord(
                name="alpha",
                source="personal",
                scope="personal",
                files={"SKILL.md": SKILL_MD},
                path="saas://skills/alpha",
            ),
            SkillRecord(
                name="alpha",
                source="personal",
                scope="personal",
                files={"SKILL.md": SKILL_MD},
            ),
            SkillRecord(
                name="alpha",
                source="personal",
                scope="personal",
                files={"SKILL.md": SKILL_MD},
                path="db://personal/1",
                _all_file_names=["SKILL.md", "logo.png"],
            ),
        ],
        ids=["provider-owned-path", "no-path", "lazy-assets"],
    )
    async def test_the_write_survives_and_is_returned(
        self, monkeypatch, tmp_path, record
    ):
        db = _factory(tmp_path)()
        _install_manager(
            monkeypatch,
            lambda name: {
                "name": name,
                "scope": record.scope,
                "source": record.source,
                "path": record.path,
                "_record": record,
            },
        )
        summary = await _create(db)
        assert summary.name == "alpha"
        assert _rows(db) == [(1, "alpha")], "a valid write is never removed"
        db.close()

    @pytest.mark.asyncio
    async def test_an_invisible_skill_still_reports_success(
        self, monkeypatch, tmp_path
    ):
        """The write is durable and was parsed before it landed, so the
        request succeeded and only the read side is behind.

        Answering 5xx here was guidance the client could not act on: a replay
        re-enters the (user, name) pre-check and gets a deterministic 409, and
        generic 5xx retry logic would replay a non-idempotent create.
        """
        db = _factory(tmp_path)()
        _install_manager(monkeypatch, lambda name: None)

        summary = await _create(db)

        assert summary.name == "alpha"
        assert summary.source == "user"
        assert summary.description == "A test skill.", (
            "the summary comes from the bytes that were validated"
        )
        assert _rows(db) == [(1, "alpha")]
        db.close()

    @pytest.mark.asyncio
    async def test_a_second_create_after_an_invisible_one_still_conflicts(
        self, monkeypatch, tmp_path
    ):
        """The 409 that made 'retry' bad advice is still correct behaviour --
        the point is that the first call no longer asks for the retry."""
        db = _factory(tmp_path)()
        _install_manager(monkeypatch, lambda name: None)
        await _create(db)
        with pytest.raises(HTTPException) as exc:
            await _create(db)
        assert exc.value.status_code == 409
        assert _rows(db) == [(1, "alpha")]
        db.close()

    async def test_a_cancelled_readback_leaves_the_row_intact(
        self, monkeypatch, tmp_path
    ):
        """Nothing to compensate: the row is valid whether or not the client
        is still listening, so a disconnect cannot cost the user their skill."""
        db = _factory(tmp_path)()

        async def explode(*args, **kwargs):
            raise asyncio.CancelledError()

        monkeypatch.setattr(skill_hub, "_get_scoped_manager", explode)
        with pytest.raises(asyncio.CancelledError):
            await _create(db)
        assert _rows(db) == [(1, "alpha")]
        db.close()

    @pytest.mark.asyncio
    async def test_a_bad_bundle_is_refused_before_any_row_exists(
        self, monkeypatch, tmp_path
    ):
        """``/create`` always sends valid UTF-8, so drive the refusal through
        the writer the way an upload or an install would reach it."""
        db = _factory(tmp_path)()
        _install_manager(monkeypatch, lambda name: None)
        with pytest.raises(HTTPException) as exc:
            _write_personal_skill(
                db=db,
                user=_user(),
                name="alpha",
                files={"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe"},
            )
        assert exc.value.status_code == 400
        assert _rows(db) == []

        # And the name is still free afterwards.
        _install_manager(
            monkeypatch,
            lambda name: {"name": name, "scope": "personal", "path": "db://personal/1"},
        )
        await _create(db)
        assert _rows(db) == [(1, "alpha")]
        db.close()


class TestDeleteKeepsProviderNamespaces:
    """``/installed/{name}`` resolves every name through the manager.

    ``:`` is not reserved by the provider contracts and the team create path
    does not run ``_validate_skill_name``, so a provider may own a record
    genuinely named ``id:42``. An earlier revision parsed that prefix as a row
    id and deleted the caller's unrelated personal row 42 instead; there is no
    such prefix now, and no route addresses a personal row by primary key.
    """

    @pytest.mark.asyncio
    async def test_a_provider_skill_named_like_a_row_id_goes_to_the_provider(
        self, monkeypatch, tmp_path
    ):
        db = _factory(tmp_path)()
        _write_personal_skill(
            db=db, user=_user(), name="my-important-skill", files={"SKILL.md": SKILL_MD}
        )
        row_id = _rows(db)[0][0]

        _install_manager(
            monkeypatch,
            lambda name: {
                "name": name,
                "scope": "team",
                "source": "team",
                "path": f"team://{name}",
            },
        )
        seen: list[tuple] = []

        async def _spy(_provider, method, _ctx, **kwargs):
            seen.append((method, kwargs.get("name")))

        monkeypatch.setattr(skill_hub, "invoke_skill_write_provider", _spy)

        await skill_hub.delete_installed(
            f"id:{row_id}",
            SimpleNamespace(),
            SkillScopeContext(user_id=7, metadata={}),
            db,
            _user(),
        )
        assert seen == [("delete_skill", f"id:{row_id}")], (
            "the name must reach the provider that owns it"
        )
        assert _rows(db) == [(row_id, "my-important-skill")], (
            "the caller's unrelated personal row must survive"
        )
        db.close()

    def test_no_route_addresses_a_personal_row_by_primary_key(self):
        """A numeric primary key is not a safe public recovery identity: the
        column has no AUTOINCREMENT, so SQLite hands a deleted row's id to the
        next insert, and a client retrying a lost 204 would delete whatever
        took its place. Recovery for rows the manager cannot load is tracked
        separately rather than shipped with that hazard.
        """
        from xagent.web.api.skill_hub import router

        paths = [
            r.path for r in router.routes if "DELETE" in getattr(r, "methods", set())
        ]
        assert paths == ["/api/skill-hub/installed/{name}"], paths

    @pytest.mark.asyncio
    async def test_a_row_the_manager_cannot_load_is_reported_missing(
        self, monkeypatch, tmp_path
    ):
        """Documents the limitation this PR leaves open: such a row is
        unreachable by name. It is a stuck row, not a deleted one."""
        db = _factory(tmp_path)()
        _write_personal_skill(
            db=db, user=_user(), name="broken", files={"SKILL.md": SKILL_MD}
        )
        _install_manager(monkeypatch, lambda name: None)
        with pytest.raises(HTTPException) as exc:
            await skill_hub.delete_installed(
                "broken",
                SimpleNamespace(),
                SkillScopeContext(user_id=7, metadata={}),
                db,
                _user(),
            )
        assert exc.value.status_code == 404
        assert _rows(db) == [(1, "broken")], "the row is stuck, not lost"
        db.close()
