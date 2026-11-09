"""A fake identity adapter and tools for the smoke V2 engine tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, cast

from xagent.core.tools.adapters.vibe.mcp_adapter import MCPLoadResult, MCPWriteHint
from xagent.web.connector_checks.declarations import ConnectorChecks
from xagent.web.connector_checks.fixtures import FixtureValue
from xagent.web.connector_checks.identity import ConnectorTarget, SessionPurpose
from xagent.web.connector_checks.results import RunResult
from xagent.web.services.mcp_connector_session import (
    ConnectorToolsOutcome,
    ConnectorToolsResult,
)

SUCCESS = {
    "content": [{"type": "text", "text": json.dumps({"status": "success"})}],
    "structured_content": None,
    "is_error": False,
}


@dataclass
class FakeTool:
    name: str
    result: Any = field(default_factory=lambda: SUCCESS)
    write_hint: MCPWriteHint = MCPWriteHint.UNDECLARED
    delay: float = 0.0
    error: Exception | None = None
    # Answers each call from its arguments instead of returning ``result``.
    respond: Callable[[dict[str, Any]], Any] | None = None
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mcp_tool = SimpleNamespace(name=self.name)

    async def run_json_async(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        if self.respond is not None:
            return self.respond(dict(args))
        return self.result


def listed(*tools: FakeTool) -> ConnectorToolsResult:
    load_result = MCPLoadResult(
        tools=cast(Any, tools), loaded_servers=("server",), failures=()
    )
    return ConnectorToolsResult(
        ConnectorToolsOutcome.OK, 1, "server", load_result=load_result
    )


def failed(outcome: ConnectorToolsOutcome, reason: str | None) -> ConnectorToolsResult:
    return ConnectorToolsResult(outcome, 1, "server", reason=reason)


Session = ConnectorToolsResult | Exception | Callable[[], Awaitable[Any]]


class FakeAdapter:
    def __init__(
        self,
        targets: Sequence[ConnectorTarget],
        sessions: Mapping[str, Session] | None = None,
        *,
        fixtures: Mapping[str, FixtureValue] | None = None,
        setup_error: Exception | None = None,
    ) -> None:
        self.targets = targets
        self.sessions = dict(sessions or {})
        self.fixtures = dict(fixtures or {})
        self.setup_error = setup_error
        self.opened: list[tuple[str, SessionPurpose]] = []

    @property
    def label(self) -> str:
        return "user"

    def enumerate_targets(self) -> Sequence[ConnectorTarget]:
        if self.setup_error is not None:
            raise self.setup_error
        return self.targets

    async def open_session(
        self, target: ConnectorTarget, *, purpose: SessionPurpose
    ) -> ConnectorToolsResult:
        self.opened.append((target.app_id, purpose))
        session = self.sessions[target.app_id]
        if isinstance(session, Exception):
            raise session
        if callable(session):
            return cast(ConnectorToolsResult, await session())
        return session

    def resolve_fixture(self, key: str) -> FixtureValue | None:
        return self.fixtures.get(key)


class FakePolicyAdapter(FakeAdapter):
    """A fake adapter whose identity has a write policy: ``verdicts`` maps a
    tool to the verdict it gives a restricted actor."""

    def __init__(
        self,
        targets: Sequence[ConnectorTarget],
        sessions: Mapping[str, Session] | None = None,
        *,
        verdicts: Mapping[str, str | None],
        policy_error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(targets, sessions, **kwargs)
        self.verdicts = dict(verdicts)
        self.policy_error = policy_error
        self.asked: list[tuple[str, str]] = []

    def write_policy_verdict(self, target: ConnectorTarget, tool: str) -> Any:
        self.asked.append((target.app_id, tool))
        if self.policy_error is not None:
            raise self.policy_error
        return self.verdicts.get(tool)


def targets(*app_ids: str) -> list[ConnectorTarget]:
    return [ConnectorTarget(app_id) for app_id in app_ids]


def declared(*checks: ConnectorChecks) -> Callable[[str], ConnectorChecks | None]:
    return {item.app_id: item for item in checks}.get


def rows(result: RunResult) -> list[tuple[Any, ...]]:
    """Each row as (connector, check, outcome, failure class, reason)."""
    return [
        (
            row["connector"],
            row["check"],
            row["outcome"],
            row["failure_class"],
            row["reason"],
        )
        for row in result.to_document()["rows"]
    ]
