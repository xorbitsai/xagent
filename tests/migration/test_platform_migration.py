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

from xagent.migration.adapters.hermes import HermesAdapter
from xagent.migration.adapters.openclaw import OpenClawAdapter
from xagent.migration.loaders import CRON_UNSUPPORTED_REASON, MigrationLoader
from xagent.migration.runner import build_preview, write_archive


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
    # Parsing should degrade gracefully: persona survives, no schedules.
    bundle = OpenClawAdapter(root=root).parse()
    assert bundle.persona is not None
    assert bundle.schedules == []


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

    # Second import with skip leaves one "greet".
    bundle2 = HermesAdapter(root=hermes_home).parse()
    report2 = MigrationLoader(db_session, user=user, skill_conflict="skip").load(
        bundle2
    )
    assert report2.skills_skipped == ["greet"]

    # Third import with rename creates "greet-imported".
    bundle3 = HermesAdapter(root=hermes_home).parse()
    report3 = MigrationLoader(db_session, user=user, skill_conflict="rename").load(
        bundle3
    )
    assert "greet-imported" in report3.skills_imported

    names = {
        s.name for s in db_session.query(UserSkill).filter(UserSkill.user_id == user.id)
    }
    assert names == {"greet", "greet-imported"}


def test_loader_makes_agent_name_unique(db_session, hermes_home: Path) -> None:
    from xagent.web.models.agent import Agent

    user = _make_user(db_session)
    MigrationLoader(db_session, user=user).load(HermesAdapter(root=hermes_home).parse())
    MigrationLoader(db_session, user=user).load(HermesAdapter(root=hermes_home).parse())

    names = sorted(
        a.name for a in db_session.query(Agent).filter(Agent.user_id == user.id)
    )
    assert names == ["Hermes Agent", "Hermes Agent (2)"]


def test_write_archive_persists_items(tmp_path: Path, openclaw_home: Path) -> None:
    bundle = OpenClawAdapter(root=openclaw_home).parse()
    archive_dir = tmp_path / "archive"
    written = write_archive(bundle, archive_dir)
    assert written
    assert (archive_dir / "REASON.txt").exists()
    assert (archive_dir / "TOOLS.md").read_bytes() == b"legacy tool notes"
