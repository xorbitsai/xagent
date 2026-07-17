"""Tests for ``xagent migrate`` (OpenClaw / Hermes -> xagent).

Covers adapter parsing (pure, filesystem fixtures) and the end-to-end loader
against a fresh SQLite database.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator

import pytest

from xagent.migration.adapters import detect_sources
from xagent.migration.adapters.base import load_skill_dir
from xagent.migration.adapters.hermes import HermesAdapter
from xagent.migration.adapters.openclaw import OpenClawAdapter
from xagent.migration.bundle import (
    ArchivedItem,
    MigrationBundle,
    ScheduleItem,
    SkillItem,
)
from xagent.migration.loaders import (
    CRON_UNSUPPORTED_REASON,
    HEARTBEAT_UNSUPPORTED_REASON,
    LoadReport,
    MigrationLoader,
)
from xagent.migration.runner import build_preview, resolve_adapters, write_archive


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


# --------------------------------------------------------------------------
# Fixtures: synthetic source homes
# --------------------------------------------------------------------------


@pytest.fixture
def openclaw_home(tmp_path: Path) -> Path:
    root = tmp_path / "openclaw"
    ws = root / "workspace"
    _write(ws / "SOUL.md", "You are Clawbot. Be terse.")
    _write(ws / "IDENTITY.md", "Name: Clawbot")
    _write(ws / "TOOLS.md", "legacy tool notes")
    _write(ws / "HEARTBEAT.md", "# Heartbeat\n- Check HN each morning\n")
    _write(
        ws / "skills" / "hn-digest" / "SKILL.md",
        "---\ndescription: Summarize HN\n---\n## Description\nSummarize HN.\n",
    )
    _write(ws / "skills" / "hn-digest" / "template.md", "body")
    _write(
        root / "openclaw.json",
        # JSON5: comment + trailing comma exercise the tolerant parser.
        '{\n  // config\n  "agents": {"defaults": {"name": "Clawbot"}},\n'
        '  "cron": [\n'
        '    {"name": "brief", "prompt": "morning brief", "schedule": "0 8 * * *"},\n'
        '    {"name": "poll", "prompt": "poll inbox", "interval_seconds": 900},\n'
        "  ],\n}\n",
    )
    return root


@pytest.fixture
def hermes_home(tmp_path: Path) -> Path:
    root = tmp_path / "hermes"
    _write(root / "SOUL.md", "I am the Hermes persona.")
    _write(root / "skills" / "greet" / "SKILL.md", "---\ndescription: greet\n---\n")
    _write(
        root / "cron" / "jobs.json",
        json.dumps(
            {
                "jobs": [
                    {"name": "hn", "prompt": "summarize HN", "schedule": "0 9 * * *"},
                    {"name": "tick", "prompt": "tick", "interval_seconds": 1800},
                ]
            }
        ),
    )
    return root


# --------------------------------------------------------------------------
# Adapter parsing
# --------------------------------------------------------------------------


def test_openclaw_adapter_parses_footprint(openclaw_home: Path) -> None:
    bundle = OpenClawAdapter(root=openclaw_home).parse()

    assert bundle.source == "openclaw"
    assert bundle.agent_name == "Clawbot"
    # Persona merges SOUL.md + IDENTITY.md.
    assert bundle.persona is not None
    assert "Clawbot" in bundle.persona.instructions
    assert "Name: Clawbot" in bundle.persona.instructions

    skills = {s.name: s for s in bundle.skills}
    assert "hn-digest" in skills
    assert set(skills["hn-digest"].files) == {"SKILL.md", "template.md"}
    assert skills["hn-digest"].description == "Summarize HN"

    by_name = {s.name: s for s in bundle.schedules}
    # Interval job is importable; cron-expression job carries the expression.
    assert by_name["poll"].interval_seconds == 900
    assert by_name["brief"].cron_expression == "0 8 * * *"
    assert by_name["brief"].interval_seconds is None
    # HEARTBEAT.md becomes a natural-language schedule.
    assert any(s.natural_language for s in bundle.schedules)

    # TOOLS.md is archived, not silently dropped.
    assert any(a.name == "TOOLS.md" for a in bundle.archived)


def test_hermes_adapter_parses_footprint(hermes_home: Path) -> None:
    bundle = HermesAdapter(root=hermes_home).parse()

    assert bundle.source == "hermes"
    assert bundle.persona is not None
    assert "Hermes persona" in bundle.persona.instructions
    assert [s.name for s in bundle.skills] == ["greet"]

    by_name = {s.name: s for s in bundle.schedules}
    assert by_name["tick"].interval_seconds == 1800
    assert by_name["hn"].cron_expression == "0 9 * * *"


def test_missing_source_home_yields_empty_bundle(tmp_path: Path) -> None:
    bundle = OpenClawAdapter(root=tmp_path / "does-not-exist").parse()
    assert bundle.is_empty()


def test_malformed_openclaw_config_does_not_crash(tmp_path: Path) -> None:
    root = tmp_path / "openclaw"
    _write(root / "openclaw.json", "{ this is : not json ]")
    _write(root / "workspace" / "SOUL.md", "persona")
    # Parsing should degrade gracefully: persona survives, no schedules, and
    # the dropped config is surfaced as a warning rather than silence.
    bundle = OpenClawAdapter(root=root).parse()
    assert bundle.persona is not None
    assert bundle.schedules == []
    assert any("openclaw.json" in w for w in bundle.warnings)


def test_jsonish_fallback_preserves_slashes_inside_strings(tmp_path: Path) -> None:
    """A ``//`` inside a string value must survive JSON5 comment stripping."""
    root = tmp_path / "openclaw"
    _write(
        root / "openclaw.json",
        "{\n"
        "  // line comment forces the tolerant fallback\n"
        "  /* so does this block comment */\n"
        '  "agents": {"defaults": {"name": "https://claw.example/bot"}},\n'
        '  "cron": [\n'
        '    {"name": "brief", "prompt": "read https://news.example //daily",\n'
        '     "interval_seconds": 600},\n'
        "  ],\n"
        "}\n",
    )
    bundle = OpenClawAdapter(root=root).parse()
    assert bundle.warnings == []
    assert bundle.agent_name == "https://claw.example/bot"
    (schedule,) = bundle.schedules
    assert schedule.prompt == "read https://news.example //daily"
    assert schedule.interval_seconds == 600


def test_hermes_agent_name_comes_from_config_yaml(tmp_path: Path) -> None:
    root = tmp_path / "hermes"
    _write(root / "SOUL.md", "persona")
    _write(root / "config.yaml", "model: foo\nagent:\n  name: Custom Hermes\n")
    assert HermesAdapter(root=root).parse().agent_name == "Custom Hermes"

    # An unreadable config falls back to the generic name instead of crashing.
    _write(root / "config.yaml", "agent: [unclosed")
    assert HermesAdapter(root=root).parse().agent_name == "Hermes Agent"


def test_detect_sources_and_resolve_adapters(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / ".openclaw").mkdir()
    (tmp_path / ".hermes").mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert sorted(a.key for a in detect_sources()) == ["hermes", "openclaw"]

    # --source-dir without --from is ambiguous and must be rejected.
    with pytest.raises(ValueError):
        resolve_adapters(None, tmp_path / ".openclaw")


def test_openclaw_config_with_utf8_bom_parses(tmp_path: Path) -> None:
    """Files written on Windows often carry a BOM; it must not nuke the config."""
    root = tmp_path / "openclaw"
    root.mkdir()
    (root / "openclaw.json").write_bytes(
        b'\xef\xbb\xbf{"agents": {"defaults": {"name": "Clawbot"}}}'
    )
    bundle = OpenClawAdapter(root=root).parse()
    assert bundle.agent_name == "Clawbot"


def test_build_preview_splits_importable_and_archived(openclaw_home: Path) -> None:
    bundle = OpenClawAdapter(root=openclaw_home).parse()
    preview = build_preview(bundle)
    assert "poll" in preview["schedules_importable"]
    # cron-expression + heartbeat schedules are archived (no interval).
    assert "brief" in preview["schedules_archived"]


# --------------------------------------------------------------------------
# Loader (end to end against a real SQLite DB)
# --------------------------------------------------------------------------


@pytest.fixture
def db_session() -> Iterator[object]:
    from xagent.web.models.database import Base, get_engine, get_session_local, init_db

    temp_dir = tempfile.mkdtemp()
    db_url = f"sqlite:///{os.path.join(temp_dir, 'test.db')}"
    init_db(db_url=db_url)
    session = get_session_local()()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=get_engine())
        shutil.rmtree(temp_dir, ignore_errors=True)


def _make_user(db) -> object:
    from xagent.web.models.user import User

    user = User(
        username="alice",
        email="alice@example.com",
        password_hash="x",
        is_admin=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def test_loader_imports_agent_skills_and_interval_schedule(
    db_session, openclaw_home: Path
) -> None:
    from xagent.web.models.agent import Agent
    from xagent.web.models.skill import UserSkill
    from xagent.web.models.trigger import AgentTrigger

    user = _make_user(db_session)
    bundle = OpenClawAdapter(root=openclaw_home).parse()

    report = MigrationLoader(db_session, user=user).load(bundle)

    # Agent created with persona as instructions.
    agent = db_session.query(Agent).filter(Agent.user_id == user.id).one()
    assert agent.name == report.agent_name == "Clawbot"
    assert "Clawbot" in (agent.instructions or "")

    # Skill imported into user_skills.
    skills = db_session.query(UserSkill).filter(UserSkill.user_id == user.id).all()
    assert {s.name for s in skills} == {"hn-digest"}
    assert skills[0].origin == "imported"

    # Interval schedule became a scheduled trigger; cron ones were archived.
    triggers = (
        db_session.query(AgentTrigger).filter(AgentTrigger.user_id == user.id).all()
    )
    assert [t.type for t in triggers] == ["scheduled"]
    assert triggers[0].config.get("interval_seconds") == 900
    assert "poll" in report.schedules_imported
    assert any(CRON_UNSUPPORTED_REASON in a.reason for a in bundle.archived)


def test_loader_skill_conflict_strategies(db_session, hermes_home: Path) -> None:
    from xagent.web.models.skill import UserSkill

    user = _make_user(db_session)

    # First import creates "greet".
    bundle1 = HermesAdapter(root=hermes_home).parse()
    report1 = MigrationLoader(db_session, user=user).load(bundle1)
    assert "greet" in report1.skills_imported

    # Second import with skip leaves one "greet"; the skip names its reason.
    bundle2 = HermesAdapter(root=hermes_home).parse()
    report2 = MigrationLoader(db_session, user=user, skill_conflict="skip").load(
        bundle2
    )
    assert report2.skills_skipped == ["greet (already exists)"]

    # Third import with rename creates "greet-imported".
    bundle3 = HermesAdapter(root=hermes_home).parse()
    report3 = MigrationLoader(db_session, user=user, skill_conflict="rename").load(
        bundle3
    )
    assert "greet-imported" in report3.skills_imported

    # Fourth import with overwrite replaces "greet" in place.
    bundle4 = HermesAdapter(root=hermes_home).parse()
    report4 = MigrationLoader(db_session, user=user, skill_conflict="overwrite").load(
        bundle4
    )
    assert report4.skills_imported == ["greet"]
    greet_rows = (
        db_session.query(UserSkill)
        .filter(UserSkill.user_id == user.id, UserSkill.name == "greet")
        .all()
    )
    assert len(greet_rows) == 1

    names = {
        s.name for s in db_session.query(UserSkill).filter(UserSkill.user_id == user.id)
    }
    assert names == {"greet", "greet-imported"}


def test_rerunning_migration_reuses_the_migrated_agent(
    db_session, hermes_home: Path
) -> None:
    from xagent.web.models.agent import Agent

    user = _make_user(db_session)
    report1 = MigrationLoader(db_session, user=user).load(
        HermesAdapter(root=hermes_home).parse()
    )

    # Re-run after the source persona changed: no duplicate agent appears, and
    # the reused agent picks up the new instructions.
    _write(hermes_home / "SOUL.md", "I am the updated Hermes persona.")
    report2 = MigrationLoader(db_session, user=user).load(
        HermesAdapter(root=hermes_home).parse()
    )

    agents = db_session.query(Agent).filter(Agent.user_id == user.id).all()
    assert len(agents) == 1
    assert report1.agent_reused is False
    assert report2.agent_reused is True
    assert report2.agent_name == report1.agent_name == "Hermes Agent"
    assert "updated Hermes persona" in (agents[0].instructions or "")


def test_agent_name_uniquified_against_users_own_agents(
    db_session, hermes_home: Path
) -> None:
    """A user's own same-named agent is never hijacked; re-runs still reuse."""
    from xagent.web.models.agent import Agent

    user = _make_user(db_session)
    own = Agent(
        user_id=user.id,
        name="Hermes Agent",
        description="hand-made, not a migration product",
        execution_mode="balanced",
        models={},
        knowledge_bases=[],
        skills=[],
        tool_categories=[],
        suggested_prompts=[],
    )
    db_session.add(own)
    db_session.commit()

    MigrationLoader(db_session, user=user).load(HermesAdapter(root=hermes_home).parse())
    report2 = MigrationLoader(db_session, user=user).load(
        HermesAdapter(root=hermes_home).parse()
    )

    names = sorted(
        a.name for a in db_session.query(Agent).filter(Agent.user_id == user.id)
    )
    assert names == ["Hermes Agent", "Hermes Agent (2)"]
    assert report2.agent_reused is True
    assert report2.agent_name == "Hermes Agent (2)"


def test_skills_only_bundle_creates_no_agent(db_session) -> None:
    """UserSkills are user-scoped; an empty agent would just be clutter."""
    from xagent.web.models.agent import Agent

    user = _make_user(db_session)
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.skills = [
        SkillItem(
            name="solo",
            files={"SKILL.md": b"---\ndescription: d\n---\n"},
            source_path="x",
        ),
    ]

    report = MigrationLoader(db_session, user=user).load(bundle)

    assert report.agent_name is None
    assert report.skills_imported == ["solo"]
    assert db_session.query(Agent).filter(Agent.user_id == user.id).count() == 0


def test_write_archive_persists_items(tmp_path: Path, openclaw_home: Path) -> None:
    bundle = OpenClawAdapter(root=openclaw_home).parse()
    archive_dir = tmp_path / "archive"
    written = write_archive(bundle, archive_dir)
    assert written
    assert (archive_dir / "REASON.txt").exists()
    assert (archive_dir / "TOOLS.md").read_bytes() == b"legacy tool notes"


def test_write_archive_survives_uncreatable_archive_dir(tmp_path: Path, caplog) -> None:
    """A failing mkdir degrades to a warning: the DB import already committed."""
    import logging

    bundle = MigrationBundle(source="openclaw", source_root="x")
    bundle.archived = [ArchivedItem(name="TOOLS.md", reason="r", content=b"data")]
    # A regular file squatting on a path component makes mkdir raise OSError.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        written = write_archive(bundle, blocker / "archive")

    assert written == []
    assert any("archive" in r.getMessage().lower() for r in caplog.records)


def test_write_archive_never_clobbers_on_name_collision(tmp_path: Path) -> None:
    """Fallback names are re-checked, so pathological name sets can't overwrite."""
    bundle = MigrationBundle(source="openclaw", source_root="x")
    bundle.archived = [
        ArchivedItem(name="TOOLS.md", reason="r", content=b"first"),
        ArchivedItem(name="2-TOOLS.md", reason="r", content=b"second"),
        ArchivedItem(name="TOOLS.md", reason="r", content=b"third"),
    ]

    written = write_archive(bundle, tmp_path / "archive")

    assert len(set(written)) == 3
    contents = sorted(Path(p).read_bytes() for p in written)
    assert contents == [b"first", b"second", b"third"]


def test_loader_one_bad_skill_does_not_poison_the_run(db_session) -> None:
    """A failing skill is reported, and everything after it still imports."""
    user = _make_user(db_session)
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.skills = [
        # No SKILL.md -> rejected by the Skill Hub writer's validation.
        SkillItem(name="broken", files={"notes.md": b"n"}, source_path="x"),
        SkillItem(
            name="good",
            files={"SKILL.md": b"---\ndescription: d\n---\n"},
            source_path="x",
        ),
    ]
    bundle.schedules = [ScheduleItem(name="tick", prompt="tick", interval_seconds=60)]

    report = MigrationLoader(db_session, user=user).load(bundle)

    assert len(report.errors) == 1
    assert "broken" in report.errors[0]
    assert report.skills_imported == ["good"]
    assert report.schedules_imported == ["tick"]
    assert not report.ok


def test_loader_normalizes_skill_names_to_hub_rules(db_session) -> None:
    from xagent.web.models.skill import UserSkill

    user = _make_user(db_session)
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.skills = [
        SkillItem(
            name="my skill!",
            files={"SKILL.md": b"---\ndescription: d\n---\n"},
            source_path="x",
        ),
    ]

    report = MigrationLoader(db_session, user=user).load(bundle)

    assert report.skills_imported == ["my-skill"]
    names = {
        s.name for s in db_session.query(UserSkill).filter(UserSkill.user_id == user.id)
    }
    assert names == {"my-skill"}


def test_skills_colliding_after_normalization_report_the_collision(
    db_session,
) -> None:
    """Two source skills that normalize to one name: the skip says which."""
    user = _make_user(db_session)
    files = {"SKILL.md": b"---\ndescription: d\n---\n"}
    bundle = MigrationBundle(source="hermes", source_root="x")
    bundle.skills = [
        SkillItem(name="my skill", files=dict(files), source_path="x"),
        SkillItem(name="my-skill", files=dict(files), source_path="x"),
    ]

    report = MigrationLoader(db_session, user=user).load(bundle)

    assert report.skills_imported == ["my-skill"]
    (skipped,) = report.skills_skipped
    assert skipped == "my-skill (name collides with 'my skill' after normalization)"


def test_rerunning_migration_does_not_duplicate_triggers(
    db_session, hermes_home: Path
) -> None:
    from xagent.web.models.trigger import AgentTrigger

    user = _make_user(db_session)
    MigrationLoader(db_session, user=user).load(HermesAdapter(root=hermes_home).parse())
    report2 = MigrationLoader(db_session, user=user).load(
        HermesAdapter(root=hermes_home).parse()
    )

    triggers = (
        db_session.query(AgentTrigger).filter(AgentTrigger.user_id == user.id).all()
    )
    assert len(triggers) == 1
    assert report2.schedules_imported == []
    assert report2.schedules_skipped == ["tick"]


def test_heartbeat_schedules_archived_with_their_own_reason(
    db_session, openclaw_home: Path
) -> None:
    user = _make_user(db_session)
    bundle = OpenClawAdapter(root=openclaw_home).parse()

    MigrationLoader(db_session, user=user).load(bundle)

    reasons = {a.name: a.reason for a in bundle.archived}
    assert reasons["brief"] == CRON_UNSUPPORTED_REASON
    assert reasons["heartbeat-1"] == HEARTBEAT_UNSUPPORTED_REASON


def test_load_skill_dir_skips_hidden_files(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skill"
    _write(skill_dir / "SKILL.md", "---\ndescription: d\n---\n")
    _write(skill_dir / ".DS_Store", "junk")
    _write(skill_dir / ".git" / "config", "junk")

    item = load_skill_dir("skill", skill_dir)

    assert item is not None
    assert set(item.files) == {"SKILL.md"}


def test_load_skill_dir_skips_symlinks(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    skill_dir = tmp_path / "skill"
    _write(skill_dir / "SKILL.md", "---\ndescription: d\n---\n")
    try:
        (skill_dir / "leak.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks not available on this platform/user")

    item = load_skill_dir("skill", skill_dir)

    assert item is not None
    assert set(item.files) == {"SKILL.md"}


def test_resolve_user_selection_rules(db_session) -> None:
    from xagent.migration.cli import _resolve_user
    from xagent.web.models.user import User

    with pytest.raises(SystemExit):  # no users yet
        _resolve_user(db_session, None)

    admin = _make_user(db_session)  # alice, admin
    assert _resolve_user(db_session, None).id == admin.id  # sole user

    bob = User(username="bob", email="bob@example.com", password_hash="x")
    db_session.add(bob)
    db_session.commit()
    # Multiple users but a single admin: the admin is auto-selected.
    assert _resolve_user(db_session, None).id == admin.id
    # An explicit --user always wins; unknown names fail loudly.
    assert _resolve_user(db_session, "bob").username == "bob"
    with pytest.raises(SystemExit):
        _resolve_user(db_session, "nobody")

    carol = User(
        username="carol", email="carol@example.com", password_hash="x", is_admin=True
    )
    db_session.add(carol)
    db_session.commit()
    with pytest.raises(SystemExit):  # two admins: ambiguous
        _resolve_user(db_session, None)


def test_confirm_prompt_names_the_target_user(monkeypatch) -> None:
    """The y/N prompt restates the target account, not just a generic question."""
    from xagent.migration import cli

    prompts: list[str] = []

    def fake_input(prompt: str = "") -> str:
        prompts.append(prompt)
        return "y"

    monkeypatch.setattr("builtins.input", fake_input)
    assert cli._confirm("alice") is True
    assert "alice" in prompts[0]


def test_print_report_distinguishes_outcomes(capsys) -> None:
    from xagent.migration.cli import _print_report

    report = LoadReport(agent_name="Clawbot", agent_reused=True)
    report.errors.append("skill 'x': boom")
    report.skills_skipped.append("y (already exists)")
    _print_report(report, [])
    out = capsys.readouterr().out
    assert "finished with errors" in out
    assert "agent reused" in out
    assert "! skill 'x': boom" in out
    # Skipped skills are listed with their reason, not just counted.
    assert "- y (already exists)" in out

    _print_report(LoadReport(agent_name=None), [])
    out = capsys.readouterr().out
    assert "Migration complete." in out
    assert "none needed (skills only)" in out


def test_cli_run_imports_end_to_end(
    db_session, hermes_home: Path, monkeypatch, capsys
) -> None:
    """The non-dry-run CLI path: resolve user, load, and report."""
    import argparse

    from xagent.migration import cli
    from xagent.web.models.agent import Agent

    user = _make_user(db_session)
    # run() closes the session when done, detaching `user` - keep the raw id.
    user_id = int(user.id)
    # Point the CLI at the already-initialized test database.
    monkeypatch.setattr("xagent.web.models.database.init_db", lambda *a, **k: None)
    monkeypatch.setattr(
        "xagent.web.models.database.get_session_local", lambda: lambda: db_session
    )
    args = argparse.Namespace(
        source="hermes",
        source_dir=hermes_home,
        username="alice",
        skill_conflict="skip",
        dry_run=False,
        yes=True,
    )

    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert "Importing into xagent user: alice" in out
    assert "agent created        : Hermes Agent" in out
    agent = db_session.query(Agent).filter(Agent.user_id == user_id).one()
    assert agent.name == "Hermes Agent"


def test_dry_run_never_initializes_the_db(
    openclaw_home: Path, monkeypatch, capsys
) -> None:
    """A fresh install (no users yet) must still be able to preview."""
    import argparse

    from xagent.migration import cli

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("dry-run must not initialize the DB")

    monkeypatch.setattr("xagent.web.models.database.init_db", _boom)
    args = argparse.Namespace(
        source="openclaw",
        source_dir=openclaw_home,
        username=None,
        skill_conflict="skip",
        dry_run=True,
        yes=False,
    )

    assert cli.run(args) == 0
    out = capsys.readouterr().out
    assert "dry-run: no changes made." in out
