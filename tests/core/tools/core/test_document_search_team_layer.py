"""Team-governed knowledge-base visibility: the team layer and its seam.

Covers ``_list_visible_collections`` resolving the team layer against the
*governing* agent's team instead of the runner's own team memberships, the
typed scope error surviving both re-wrap points, and the governing-team
values reaching the knowledge tools at build time.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from xagent.core.tools.core import document_search
from xagent.core.tools.core.knowledge_base_scope import KnowledgeBaseScopeError
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
    SearchPipelineResult,
)
from xagent.web.services import knowledge_base_team_scope as kb_scope

CREATOR = 100
MEMBER = 200
ADMIN = 400
TEAM = 1


def _collections(*collections: CollectionInfo) -> ListCollectionsResult:
    return ListCollectionsResult(
        status="success",
        collections=list(collections),
        total_count=len(collections),
        message="ok",
    )


def _kb(name: str, *, embeddings: int = 5, documents: int = 3) -> CollectionInfo:
    return CollectionInfo(name=name, embeddings=embeddings, documents=documents)


class _ListCollectionsSpy:
    """Routes ``list_collections(user_id=...)`` by user id and records calls."""

    def __init__(self, by_user: dict[int, list[CollectionInfo]]) -> None:
        self._by_user = by_user
        self.calls: list[tuple[Optional[int], Optional[bool]]] = []

    async def __call__(
        self,
        user_id: Optional[int] = None,
        is_admin: Optional[bool] = None,
        force_realtime: bool = False,
    ) -> ListCollectionsResult:
        self.calls.append((user_id, is_admin))
        return _collections(*self._by_user.get(user_id, []))

    def calls_for(self, user_id: int) -> int:
        return sum(1 for called_id, _ in self.calls if called_id == user_id)


class _SearchSpy:
    """Fake ``run_document_search`` recording which (collection, user_id,
    is_admin) triples were actually searched, and returning one hit for
    each so ``results`` stays non-empty."""

    def __init__(self) -> None:
        self.searched: list[tuple[str, Optional[int], bool]] = []

    def __call__(
        self,
        collection: str,
        query_text: str,
        config: dict,
        user_id: Optional[int],
        is_admin: bool,
    ) -> SearchPipelineResult:
        self.searched.append((collection, user_id, is_admin))
        return SearchPipelineResult(
            status="success",
            search_type="hybrid",
            results=[
                {
                    "doc_id": f"{collection}-doc",
                    "chunk_id": f"{collection}-chunk",
                    "text": f"hit from {collection}",
                    "score": 0.9,
                    "parse_hash": "hash",
                    "model_tag": "model",
                    "metadata": {},
                }
            ],
            result_count=1,
            warnings=[],
            message="ok",
            used_rerank=False,
        )


@pytest.fixture(autouse=True)
def _isolated_hooks():
    with kb_scope.snapshot_knowledge_base_team_hooks():
        kb_scope.set_knowledge_base_team_hooks()
        yield


def _install_team(monkeypatch: pytest.MonkeyPatch, teams: dict[int, list]) -> None:
    def _team_visibility(db: Any, *, team_id: int) -> list:
        return teams.get(team_id, [])

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_team_visibility)


def _install_collections(
    monkeypatch: pytest.MonkeyPatch, by_user: dict[int, list[CollectionInfo]]
) -> _ListCollectionsSpy:
    spy = _ListCollectionsSpy(by_user)
    monkeypatch.setattr(document_search, "list_collections", spy)
    return spy


def _install_search(monkeypatch: pytest.MonkeyPatch) -> _SearchSpy:
    spy = _SearchSpy()
    monkeypatch.setattr(document_search, "run_document_search", spy)
    return spy


async def _search(
    monkeypatch: pytest.MonkeyPatch,
    *,
    runner_id: int,
    is_admin: bool = False,
    team_id: Optional[int] = TEAM,
    creator_id: Optional[int] = CREATOR,
    declared: Optional[list[str]] = None,
    collections: Optional[list[str]] = None,
    allowed_collections: Optional[list[str]] = None,
) -> document_search.KnowledgeSearchResult:
    tool_args = document_search.KnowledgeSearchArgs(
        query="q",
        collections=collections or [],
        allowed_collections=allowed_collections,
    )
    return await document_search._search_knowledge_base_impl(
        tool_args,
        user_id=runner_id,
        is_admin=is_admin,
        governing_team_id=team_id,
        agent_creator_user_id=creator_id,
        declared_knowledge_bases=declared,
    )


# ---------------------------------------------------------------------------
# A name the governing team owns resolves to the team's storage tenant,
# for every runner.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "runner_id,is_admin",
    [
        (MEMBER, False),  # a team member
        (ADMIN, True),  # a platform admin -- admin status must not bypass this
    ],
)
async def test_kb_resolves_on_governing_team(
    monkeypatch: pytest.MonkeyPatch, runner_id: int, is_admin: bool
) -> None:
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="handbook",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(
        monkeypatch,
        {
            runner_id: [],
            CREATOR: [_kb("handbook")],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=runner_id,
        is_admin=is_admin,
        declared=["handbook"],
        collections=["handbook"],
    )

    assert search_spy.searched == [("handbook", CREATOR, False)]
    assert result.results


# ---------------------------------------------------------------------------
# No governing team => the team-keyed hook is never invoked, and no other
# tenant's collections are listed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_governing_team_never_touches_team_hook_or_creator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_calls: list[int] = []

    def _team_visibility(db: Any, *, team_id: int) -> list:
        team_calls.append(team_id)
        return []

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_team_visibility)
    spy = _install_collections(monkeypatch, {MEMBER: [_kb("kb1")]})
    _install_search(monkeypatch)

    await _search(
        monkeypatch,
        runner_id=MEMBER,
        team_id=None,
        creator_id=None,
        declared=["kb1"],
    )

    assert team_calls == []
    assert spy.calls_for(CREATOR) == 0


# ---------------------------------------------------------------------------
# Team hook not installed => legacy runner-keyed overlay, selection
# on the predicate, never on an empty return.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_team_hook_absent_falls_back_to_user_keyed_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Scoped narrowly to the TEAM LAYER only: with no team-keyed hook
    installed, ``_list_visible_collections`` must still resolve the legacy
    runner-keyed overlay when the (separate) user-keyed hook is installed.
    A governing team id is passed in deliberately, to show the selection is
    on the hook predicate rather than on the id being present.
    """
    legacy_access = kb_scope.KnowledgeBaseAccess(
        name="legacy-shared", storage_user_id=CREATOR, team_owned=True
    )
    kb_scope.set_knowledge_base_team_hooks(
        visibility=lambda db, user_id: [legacy_access]
    )
    _install_collections(monkeypatch, {MEMBER: [], CREATOR: [_kb("legacy-shared")]})

    result = await document_search._list_visible_collections(
        user_id=MEMBER, is_admin=False, governing_team_id=TEAM
    )

    names = {c.name for c in result.collections}
    assert "legacy-shared" in names


# ---------------------------------------------------------------------------
# Both a team-keyed hook and a runner-keyed hook installed at once: the
# governing branch must consult only the team-keyed one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_governing_branch_never_unions_with_runner_keyed_hook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two invariants pinned together, because either one can regress
    without the other noticing:

    1. Never a union. With both hooks installed, the governing branch must
       resolve only through the team-keyed hook. The team-keyed hook here
       answers ``[]`` for the governing team, while the runner-keyed hook
       answers a real collection ("runner-owned") for the same runner: if
       the governing branch ever merged in the runner-keyed hook's answer,
       "runner-owned" would appear in the result, and the runner-keyed
       hook's callable would be invoked at all -- both are asserted against.
    2. Selection is on the predicate, not on the answer. The team-keyed
       hook's empty answer must not be read as "no team-keyed hook, fall
       back to the runner-keyed one" -- the governing branch stays selected
       purely because ``team_knowledge_base_hook_installed()`` is true.
    """
    team_calls: list[int] = []
    runner_calls: list[int] = []

    def _team_visibility(db: Any, *, team_id: int) -> list:
        team_calls.append(team_id)
        return []

    def _runner_visibility(db: Any, user_id: int) -> list:
        runner_calls.append(user_id)
        return [
            kb_scope.KnowledgeBaseAccess(
                name="runner-owned", storage_user_id=CREATOR, team_owned=True
            )
        ]

    kb_scope.set_knowledge_base_team_hooks(
        team_visibility=_team_visibility, visibility=_runner_visibility
    )
    _install_collections(monkeypatch, {MEMBER: [], CREATOR: [_kb("runner-owned")]})

    result = await document_search._list_visible_collections(
        user_id=MEMBER, is_admin=False, governing_team_id=TEAM
    )

    names = {c.name for c in result.collections}
    assert "runner-owned" not in names
    assert team_calls == [TEAM]
    assert runner_calls == []


# ---------------------------------------------------------------------------
# The typed error survives the run-path re-wrap; the build-frame
# narrowings are defence in depth only.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_kb_scope_hook_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _broken_team_visibility(db: Any, *, team_id: int):
        raise RuntimeError("boom")

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_broken_team_visibility)
    _install_collections(monkeypatch, {MEMBER: []})
    _install_search(monkeypatch)

    with pytest.raises(KnowledgeBaseScopeError):
        await _search(monkeypatch, runner_id=MEMBER, declared=["handbook"])


@pytest.mark.asyncio
async def test_kb_scope_error_survives_tool_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defence in depth: even if a future change made a tool-build frame
    raise the typed error, the frame's own narrow catch must let it pass
    through unwrapped. Exercised directly against the narrowing clause's
    intended shape rather than the real (currently unreachable) call path.
    """
    from xagent.core.tools.adapters.vibe import knowledge_tools

    class _RaisingConfig:
        def get_allowed_collections(self):
            raise KnowledgeBaseScopeError("x", "boom")

        def get_user_id(self):
            return MEMBER

        def is_admin(self):
            return False

        def get_agent_creator_user_id(self):
            return None

        def get_declared_knowledge_bases(self):
            return None

    with pytest.raises(KnowledgeBaseScopeError):
        await knowledge_tools._create_knowledge_tools_impl(_RaisingConfig())


# ---------------------------------------------------------------------------
# user_id is None returns before any team resolution.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unauthenticated_caller_returns_before_team_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def _team_visibility(db: Any, *, team_id: int) -> list:
        calls.append(team_id)
        return []

    kb_scope.set_knowledge_base_team_hooks(team_visibility=_team_visibility)
    _install_collections(monkeypatch, {})

    # Must not raise TypeError from int(None), and must not touch the team hook.
    result = await document_search._list_visible_collections(
        user_id=None, is_admin=False, governing_team_id=TEAM
    )
    assert result.collections == []
    assert calls == []


# ---------------------------------------------------------------------------
# The save-time validation consumer (find_missing_knowledge_bases) stays
# runner-keyed.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_find_missing_knowledge_bases_is_runner_keyed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="handbook",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(monkeypatch, {MEMBER: [], CREATOR: [_kb("handbook")]})

    # find_missing_knowledge_bases takes no governing_team_id parameter at
    # all -- it stays a strictly two-argument (user_id, is_admin) call, so
    # it cannot see the governing team's rows even though one is installed.
    missing = await document_search._find_missing_knowledge_bases_impl(
        ["handbook"], user_id=MEMBER, is_admin=False
    )

    assert missing == ["handbook"]


# ---------------------------------------------------------------------------
# The knowledge tools read the governing-team values off the config, and
# there are exactly two ListKnowledgeBasesTool( construction sites.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_collections,expect_list_tool",
    [(None, True), (["kb1"], False)],
)
async def test_list_tool_built_only_without_declaration(
    monkeypatch: pytest.MonkeyPatch, allowed_collections, expect_list_tool
) -> None:
    from xagent.core.tools.adapters.vibe import knowledge_tools

    class _Config:
        def get_allowed_collections(self):
            return allowed_collections

        def get_user_id(self):
            return MEMBER

        def is_admin(self):
            return False

        def get_agent_creator_user_id(self):
            return CREATOR

        def get_declared_knowledge_bases(self):
            return allowed_collections

    tools = await knowledge_tools._create_knowledge_tools_impl(_Config())
    tool_names = {t.name for t in tools}
    assert ("list_knowledge_bases" in tool_names) is expect_list_tool


def test_list_knowledge_bases_tool_has_exactly_two_construction_sites() -> None:
    """Two sites, and the second one is exempt on purpose.

    The first is ``adapters/vibe/document_search.py``, which
    ``knowledge_tools.py`` calls and which carries the governing-team
    values. The second is ``web/api/websocket.py``, the agent-builder
    console: it constructs the tool directly with ``user_id`` and
    ``is_admin`` and nothing else, has no allowed-collections list, and runs
    for the person configuring an agent rather than inside any agent's run,
    so there is no governing team for it to resolve against and it keeps
    today's runner-keyed behaviour. That is why this count is asserted
    instead of asserting that every site threads the governing-team values:
    a third site appearing, or a refactor routing the builder console
    through ``knowledge_tools.py``, changes which of those two statements
    holds and should be re-decided rather than silently absorbed.
    """
    import pathlib
    import re
    import subprocess

    repo_root = pathlib.Path(__file__).resolve().parents[4]
    try:
        probe = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        pytest.skip("git is not available -- git grep counting is unavailable")
    if probe.returncode != 0 or probe.stdout.strip() != "true":
        pytest.skip("not inside a git checkout -- git grep counting is unavailable")

    output = subprocess.run(
        ["git", "grep", "-n", "ListKnowledgeBasesTool("],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    construction_sites = [
        line
        for line in output.splitlines()
        if "src/" in line and not re.search(r"class ListKnowledgeBasesTool\(", line)
    ]
    assert len(construction_sites) == 2, construction_sites
