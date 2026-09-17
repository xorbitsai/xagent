"""Team-governed knowledge-base resolution: the declared-name rule.

Covers a run whose governing agent is team-owned. A declared name the
governing team owns resolves to the team's copy for every runner; a
declared name the team does not own resolves to the creator's own personal
collection only when the runner *is* the creator, and to one of two
distinguishable, non-identity-leaking outcomes for every other runner.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest

from xagent.core.tools.core import document_search
from xagent.core.tools.core.RAG_tools.core.schemas import (
    CollectionInfo,
    ListCollectionsResult,
    SearchPipelineResult,
)
from xagent.web.services import knowledge_base_team_scope as kb_scope

CREATOR = 100
MEMBER = 200
NON_MEMBER = 300
TEAM = 1
OTHER_TEAM = 2


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


_THIRD_TENANT = 700


def _install_team_kb_off_creator_tenant_with_broken_creator_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> _SearchSpy:
    """The governing team owns "good", hosted on a tenant that is neither the
    runner's nor the creator's, and the creator's own listing raises. Resolving
    a team-owned name never touches the creator probe, so "good" is admitted
    while every name the probe would have classified is left unresolved."""
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="good",
        storage_user_id=_THIRD_TENANT,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})

    async def _list_collections_with_creator_failure(
        user_id=None, is_admin=None, force_realtime=False
    ):
        if user_id == CREATOR:
            raise RuntimeError("storage unavailable")
        if user_id == _THIRD_TENANT:
            return _collections(_kb("good"))
        return _collections()

    monkeypatch.setattr(
        document_search, "list_collections", _list_collections_with_creator_failure
    )
    return _install_search(monkeypatch)


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
# A same-named row of the runner's OTHER team never participates.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_cross_team_kb_borrowing(monkeypatch: pytest.MonkeyPatch) -> None:
    governing_team_kb = kb_scope.KnowledgeBaseAccess(
        name="other-doc",
        storage_user_id=999,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    other_team_kb = kb_scope.KnowledgeBaseAccess(
        name="handbook",
        storage_user_id=555,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [governing_team_kb], OTHER_TEAM: [other_team_kb]})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [],
            CREATOR: [],
            999: [_kb("other-doc")],  # T's real (differently-named) collection
            555: [_kb("handbook")],  # the OTHER team's storage tenant
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
    )

    assert search_spy.searched == []
    assert result.results == []
    assert "handbook" in result.summary


# ---------------------------------------------------------------------------
# The runner's own team membership never merges into the governing team's
# layer -- the module docstring's promise that a team-governed run resolves
# "never the runner's own team memberships, and never a union of the two"
# has no other pin.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_runner_own_team_membership_never_unions_with_governing_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MEMBER belongs to their own team (not the governing one), and that
    team owns ``my-team-kb`` through the runner-keyed visibility hook. The
    governing team owns ``gov-kb`` through the team-keyed hook. The agent
    declares both. Only ``gov-kb`` may resolve: ``my-team-kb`` must fail to
    resolve exactly as it would if MEMBER had no team at all -- never
    silently pass because a merge gave it ``ownership == "team"`` too.
    """
    gov_kb = kb_scope.KnowledgeBaseAccess(
        name="gov-kb",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    my_team_kb = kb_scope.KnowledgeBaseAccess(
        name="my-team-kb",
        storage_user_id=777,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )

    def _member_own_team_visibility(
        db: Any, user_id: int
    ) -> list[kb_scope.KnowledgeBaseAccess]:
        # The runner-keyed overlay (MEMBER's own team membership), never
        # consulted for a team-governed run's declared-name resolution.
        return [my_team_kb] if user_id == MEMBER else []

    def _governing_team_visibility(
        db: Any, *, team_id: int
    ) -> list[kb_scope.KnowledgeBaseAccess]:
        return [gov_kb] if team_id == TEAM else []

    kb_scope.set_knowledge_base_team_hooks(
        team_visibility=_governing_team_visibility,
        visibility=_member_own_team_visibility,
    )
    _install_collections(
        monkeypatch,
        {
            MEMBER: [],
            CREATOR: [_kb("gov-kb")],
            777: [_kb("my-team-kb")],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["gov-kb", "my-team-kb"],
    )

    assert search_spy.searched == [("gov-kb", CREATOR, False)]
    assert "my-team-kb" in result.summary


# ---------------------------------------------------------------------------
# The team's copy outranks the creator's own copy: even the creator gets
# T's copy over their own same-named personal collection.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_copy_wins_over_creator_personal_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="handbook",
        storage_user_id=999,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(
        monkeypatch,
        {
            CREATOR: [_kb("handbook")],  # creator's OWN copy, must lose
            999: [_kb("handbook")],  # the team's storage tenant
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=CREATOR,
        declared=["handbook"],
    )

    assert search_spy.searched == [("handbook", 999, False)]
    assert result.results


# ---------------------------------------------------------------------------
# The team does not own the declared name, and the runner IS the agent's
# creator: their own collection resolves unchanged. Applies equally to an
# interactive creator and to an external/widget end user whose passed-in
# runner id happens to equal the creator id.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creator_keeps_own_unshared_knowledge_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is an accepted-exposure ruling, not a gap: an external or
    anonymous runner is indistinguishable from the agent's creator at this
    layer, because no real end-user identity is carried to the resolution
    point. When the runner id passed in happens to equal the creator id --
    which is what an external/widget surface's identity substitution
    produces -- the creator's own collection resolves exactly as it would
    for the creator's own interactive session. No pin anywhere may assert
    that the declared name is instead rejected on that surface.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(monkeypatch, {CREATOR: [_kb("handbook")]})
    search_spy = _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=CREATOR, declared=["handbook"])

    assert search_spy.searched == [("handbook", CREATOR, False)]
    assert result.results


# ---------------------------------------------------------------------------
# The team does not own the declared name, the runner is not the creator,
# and the name IS the creator's own personal collection => the new
# sharing-gap message, no collection resolves. Parametrised over call style
# and partial failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_unshared_creator_kb_reports_sharing_gap(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    # T owns the two "good" names -- they resolve and are actually
    # searched. "private-notes" is the creator's own, never shared with T;
    # since the runner is not the creator this must degrade to the NEW
    # message, not a search.
    team_kbs = [
        kb_scope.KnowledgeBaseAccess(
            name=name,
            storage_user_id=CREATOR,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        )
        for name in ("good-one", "also-good")
    ]
    _install_team(monkeypatch, {TEAM: team_kbs})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [_kb("unrelated-own-doc")],
            CREATOR: [_kb("private-notes"), _kb("good-one"), _kb("also-good")],
        },
    )
    search_spy = _install_search(monkeypatch)

    declared = ["private-notes", "good-one", "also-good"]
    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=declared,
        collections=declared if explicit else None,
    )

    assert "private-notes" in result.summary
    assert "has not" in result.summary and "shared with the team" in result.summary
    searched_names = {name for name, owner, _ in search_spy.searched}
    # The two team-owned names were still searched, against T's storage
    # tenant -- a partial failure does not stop the rest of the agent's
    # knowledge bases from being searched normally.
    assert searched_names == {"good-one", "also-good"}
    assert all(owner == CREATOR for _, owner, _ in search_spy.searched)
    assert result.results
    assert not result.summary.startswith("Error:")


# ---------------------------------------------------------------------------
# Same shape, but the declared name is NOT the creator's collection either
# => the branch's generic "does not exist" outcome. A further variant:
# every declared name fails to resolve while the runner holds a same-named
# personal copy of each one.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_absent_kb_reports_does_not_exist(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(monkeypatch, {MEMBER: [_kb("unrelated-own-doc")], CREATOR: []})
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["ghost"],
        collections=["ghost"] if explicit else None,
    )

    assert result.results == []
    assert "Error:" in result.summary
    if explicit:
        assert "do not exist" in result.summary
    else:
        assert "None of the allowed collections exist" in result.summary
    # Negative control: the failed name is not the creator's collection
    # either, so the sharing-gap sentence must not appear.
    assert "shared with the team" not in result.summary


# ---------------------------------------------------------------------------
# The "Available collections:" clause is captured before the
# explicit-request narrowing, not after.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_narrowing_still_reports_what_the_verdict_loop_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent declares two names: "handbook", which the governing team
    owns and which the verdict loop therefore admits, and "ghost", which
    resolves to nothing. The model's explicit request names only "ghost" --
    so the narrowing step (applied after ``admitted_names`` is captured)
    empties ``to_search`` entirely, even though "handbook" was genuinely
    admitted. If ``admitted_names`` were captured after the narrowing
    instead of before, this fully-narrowed-away request would report
    "Available collections:" as empty, leaving the model with no way to
    learn what it could have asked for.
    """
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
        {MEMBER: [], CREATOR: [_kb("handbook")]},
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook", "ghost"],
        collections=["ghost"],
    )

    assert result.results == []
    assert "Error:" in result.summary and "do not exist" in result.summary
    assert "ghost" in result.summary
    available_clause = result.summary.split("Available collections:", 1)[1]
    assert "handbook" in available_clause


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_total_failure_never_reports_undeclared_team_kb_names(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """The governing team owns several knowledge bases the agent never
    declared. A non-member runner declares one unrelated name that fails
    to resolve entirely. The terminal "Available"/"Available collections"
    listing must only ever include names the agent itself declared --
    never the team's other, undeclared knowledge bases, which this runner
    has no other way to learn the names of. Before this pin, both terminal
    branches printed the full visible-union names, including the four
    undeclared ones below.
    """
    hidden_names = (
        "board-minutes",
        "layoff-plan-q3",
        "ma-target-list",
        "salary-bands",
    )
    hidden_kbs = [
        kb_scope.KnowledgeBaseAccess(
            name=name,
            storage_user_id=CREATOR,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        )
        for name in hidden_names
    ]
    _install_team(monkeypatch, {TEAM: hidden_kbs})
    _install_collections(
        monkeypatch,
        {
            NON_MEMBER: [],
            CREATOR: [_kb(name) for name in hidden_names],
        },
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=NON_MEMBER,
        declared=["hr-private"],
        collections=["hr-private"] if explicit else None,
    )

    assert result.results == []
    assert "Error:" in result.summary
    for hidden_name in hidden_names:
        assert hidden_name not in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_total_failure_of_a_single_unshared_name_still_reports_the_gap(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Every declared name fails to resolve (``to_search`` ends up empty),
    and the one declared name is the creator's own unshared collection.
    The terminal summary must keep its existing ``Error:`` prefix and
    empty result set (this is the total-failure path, not the
    partial-failure warnings channel), but it must still say the failed
    name belongs to the creator and was never shared -- not just that it
    "does not exist", which would be true of a name nobody owns and false
    of this one.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(
        monkeypatch,
        {MEMBER: [_kb("unrelated-own-doc")], CREATOR: [_kb("private-notes")]},
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["private-notes"],
        collections=["private-notes"] if explicit else None,
    )

    assert result.results == []
    assert result.summary.startswith("Error:")
    assert "has not" in result.summary and "shared with the team" in result.summary
    assert "private-notes" in result.summary
    # The sharing-gap sentence says the creator holds this collection; the
    # same summary must not also assert it is absent.
    assert "do not exist" not in result.summary
    assert "None of the allowed collections exist" not in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_mixed_terminal_failure_lists_each_name_once(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """Two declared names fail for different reasons and nothing resolves:
    "private-notes" is the creator's own unshared collection, "ghost" is
    nobody's. The absence sentence may name only "ghost" -- naming
    "private-notes" there would contradict the sharing-gap sentence in the
    same summary, and the default call style's blanket "None of the allowed
    collections exist" would assert the same falsehood about both.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(monkeypatch, {MEMBER: [], CREATOR: [_kb("private-notes")]})
    search_spy = _install_search(monkeypatch)

    declared = ["private-notes", "ghost"]
    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=declared,
        collections=declared if explicit else None,
    )

    assert search_spy.searched == []
    assert result.results == []
    assert result.summary.startswith("Error:")
    assert "has not" in result.summary and "shared with the team" in result.summary
    assert "None of the allowed collections exist" not in result.summary
    absence_clause = result.summary.split("do not exist:", 1)[1].split(".", 1)[0]
    assert "ghost" in absence_clause
    assert "private-notes" not in absence_clause


@pytest.mark.asyncio
async def test_all_unshared_terminal_still_reports_what_was_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The team owns "good", so the verdict loop admits it, but the model's
    explicit request names only the creator's unshared "private-notes" --
    the narrowing empties the search set. The summary must still say what
    the run could have searched, which is the whole reason the admitted set
    is captured before the narrowing.
    """
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="good",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(
        monkeypatch, {MEMBER: [], CREATOR: [_kb("good"), _kb("private-notes")]}
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["private-notes", "good"],
        collections=["private-notes"],
    )

    assert result.results == []
    assert result.summary.startswith("Error:")
    assert "shared with the team" in result.summary
    assert "do not exist" not in result.summary
    assert "Available collections: good" in result.summary


@pytest.mark.asyncio
async def test_other_teams_kb_hosted_on_creator_tenant_is_not_personal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A team the creator belongs to -- not the governing one -- owns
    "otherteam-kb", physically stored in the creator's own namespace. A raw
    listing of that namespace returns it next to the creator's real
    personal collection, so classifying off the listing alone calls a whole
    other team's knowledge base "a personal knowledge base belonging to
    this agent's creator" and aims the remediation at the wrong person.

    That same other team also owns "really-mine", which lives in a
    different member's namespace and shares its name with a collection the
    creator genuinely owns. Only the rows physically hosted on the
    creator's tenant may be discounted: a name-only exclusion would take
    the creator's own collection with it and report it as absent.
    """
    other_member_tenant = 500
    otherteam_kb = kb_scope.KnowledgeBaseAccess(
        name="otherteam-kb",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    same_name_on_another_tenant = kb_scope.KnowledgeBaseAccess(
        name="really-mine",
        storage_user_id=other_member_tenant,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )

    def _governing_team_visibility(db: Any, *, team_id: int) -> list:
        return []

    def _creator_own_team_visibility(db: Any, user_id: int) -> list:
        if user_id != CREATOR:
            return []
        return [otherteam_kb, same_name_on_another_tenant]

    kb_scope.set_knowledge_base_team_hooks(
        team_visibility=_governing_team_visibility,
        visibility=_creator_own_team_visibility,
    )
    _install_collections(
        monkeypatch,
        {MEMBER: [], CREATOR: [_kb("otherteam-kb"), _kb("really-mine")]},
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["otherteam-kb", "really-mine"],
    )

    assert search_spy.searched == []
    gap_clause = result.summary.split("Error:", 1)[1]
    assert "otherteam-kb is a personal knowledge base" not in gap_clause
    assert "otherteam-kb" in result.summary  # reported, as absent
    # The creator's genuine personal collection is still classified as one:
    # the team owns a row by that name too, but it is hosted elsewhere, so
    # discounting it here would describe a collection the creator really
    # does hold as absent.
    assert "really-mine is a personal knowledge base" in result.summary


@pytest.mark.asyncio
async def test_not_allowed_clause_lists_the_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clause tells the model what it may ask for, so it must be the
    agent's declaration -- not the declaration intersected with what is
    visible right now. "offsite-notes" is declared and currently resolves
    to nothing; dropping it from the clause tells the model it may not ask
    for a name the agent is configured with. The governing team's other,
    undeclared knowledge bases must not appear either.
    """
    team_kbs = [
        kb_scope.KnowledgeBaseAccess(
            name=name,
            storage_user_id=CREATOR,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        )
        for name in ("handbook", "board-minutes")
    ]
    _install_team(monkeypatch, {TEAM: team_kbs})
    _install_collections(
        monkeypatch,
        {MEMBER: [], CREATOR: [_kb("handbook"), _kb("board-minutes")]},
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook", "offsite-notes"],
        collections=["victim"],
    )

    assert search_spy.searched == []
    assert "not allowed" in result.summary
    clause = result.summary.split("Allowed collections:", 1)[1].strip()
    assert clause == "handbook, offsite-notes"
    assert "board-minutes" not in result.summary


@pytest.mark.asyncio
async def test_total_kb_failure_with_same_named_personal_copies_reports_generic_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The runner holds a same-named personal copy of every declared name,
    and the governing team owns none of them. A guard keyed on the raw
    requested/allowed names (instead of on the actual search set) would
    stay non-empty here -- the personal copies land in the visible
    union's ``available_names`` -- and fall through to the generic
    search-miss text instead of this branch's own terminal summary.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [_kb("handbook"), _kb("policies")],
            CREATOR: [],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch, runner_id=MEMBER, declared=["handbook", "policies"]
    )

    assert search_spy.searched == []
    assert result.results == []
    assert "Error:" in result.summary
    assert "None of the allowed collections exist" in result.summary
    assert "shared with the team" not in result.summary
    # The runner holds a personal "handbook" and "policies"; both are in the
    # pre-verdict visible union and neither is searchable. Reporting either
    # as available would tell the model the exact opposite of what the rule
    # just decided.
    #
    # Nothing was admitted, so the clause is dropped rather than rendered
    # with an empty list -- which is the stronger form of the same pin:
    # neither personal copy may be reported as available.
    assert "Available" not in result.summary
    # The declared names still appear -- in the "Allowed:" clause, which is
    # what the agent configured, not what resolved.
    assert "Allowed: handbook, policies" in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_terminal_summary_omits_the_availability_clause_when_nothing_was_admitted(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """The runner holds a same-named personal copy of the one declared
    name, and neither the governing team nor the creator holds it -- so the
    visible union is non-empty (the runner's own copy is in it) but nothing
    was admitted (the rule refuses to search the runner's own same-named
    copy). Before this pin the two uniform terminal sentences always
    rendered an "Available:" / "Available collections:" label followed by
    nothing; the label itself must now be omitted, byte for byte, and the
    "Allowed:" half of the default-style sentence must still render.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(
        monkeypatch,
        {MEMBER: [_kb("handbook")], CREATOR: []},
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        collections=["handbook"] if explicit else None,
    )

    if explicit:
        assert result.summary == (
            "Error: The following collections do not exist: handbook."
        )
    else:
        assert result.summary == (
            "Error: None of the allowed collections exist. Allowed: handbook."
        )
    assert "Available" not in result.summary


@pytest.mark.asyncio
@pytest.mark.parametrize("creator_holds", [False, True])
async def test_empty_visible_set_still_reports_a_verdict_per_declared_name(
    monkeypatch: pytest.MonkeyPatch, creator_holds: bool
) -> None:
    """The runner owns nothing and the governing team owns nothing, so the
    visible union is empty -- but the agent did declare a name. The generic
    "no knowledge bases exist on this platform" text would be wrong and
    would swallow the per-name verdict: this run has a specific answer
    (either the creator's unshared collection, or a name nobody holds), and
    the rule promises the response says which.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [],
            CREATOR: [_kb("handbook")] if creator_holds else [],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=MEMBER, declared=["handbook"])

    assert search_spy.searched == []
    assert result.results == []
    assert "No knowledge bases available" not in result.summary
    assert result.summary.startswith("Error:")
    assert "handbook" in result.summary
    if creator_holds:
        assert "has not" in result.summary and "shared with the team" in result.summary
    else:
        assert "None of the allowed collections exist" in result.summary
        assert "shared with the team" not in result.summary


# ---------------------------------------------------------------------------
# The "not allowed" gate, re-based conditionally on whether the run is
# team-governed with a declaration.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed_collections", [None, ["kb1"]])
async def test_team_governed_empty_declaration_keeps_legacy_not_allowed_gate(
    monkeypatch: pytest.MonkeyPatch, allowed_collections
) -> None:
    """The same gate, but on a team-governed run whose agent declares no
    knowledge bases at all, rather than on a run with no governing team.
    The two paths share one branch on purpose and must behave identically:
    applying the stored-declaration basis here regardless would either
    reject a model-named collection the agent never restricted in the
    first place, or -- as the empty ``authorised`` set would render it --
    reject with an ``Allowed collections:`` list that is always empty.
    """
    _install_team(monkeypatch, {TEAM: []})
    _install_collections(monkeypatch, {MEMBER: [_kb("kb1"), _kb("kb2")]})
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=[],
        collections=["kb2"],
        allowed_collections=allowed_collections,
    )

    if allowed_collections is None:
        assert result.results
    else:
        assert "Error:" in result.summary and "not allowed" in result.summary


# ---------------------------------------------------------------------------
# The model may narrow an explicit request to a subset of what the
# verdict loop admitted; only the narrowing half is pinned here.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explicit_request_narrows_the_search_to_what_it_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The governing team owns all three declared names, so the verdict
    loop admits all three -- but the model's explicit ``collections``
    argument names only one of them. Only that one collection should
    actually be searched, not all three admitted ones: this pins the
    narrowing half of "the model may narrow the search but never widen
    it," which the widening half (the "not allowed" gate above) does not
    cover.
    """
    team_kbs = [
        kb_scope.KnowledgeBaseAccess(
            name=name,
            storage_user_id=CREATOR,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        )
        for name in ("alpha", "beta", "gamma")
    ]
    _install_team(monkeypatch, {TEAM: team_kbs})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [],
            CREATOR: [_kb("alpha"), _kb("beta"), _kb("gamma")],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["alpha", "beta", "gamma"],
        collections=["beta"],
    )

    assert search_spy.searched == [("beta", CREATOR, False)]
    assert result.results


# ---------------------------------------------------------------------------
# Both report outcomes in one response, each listing only its own names, on
# both call styles.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_mixed_missing_and_unshared_names_report_separately(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="good",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(
        monkeypatch,
        {
            MEMBER: [_kb("unrelated-own-doc")],
            CREATOR: [_kb("unshared-one"), _kb("good")],
        },
    )
    search_spy = _install_search(monkeypatch)

    declared = ["unshared-one", "ghost-one", "good"]
    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=declared,
        collections=declared if explicit else None,
    )

    assert "unshared-one" in result.summary
    assert "has not" in result.summary and "shared with the team" in result.summary
    assert "ghost-one" in result.summary
    assert "could not be resolved" in result.summary
    assert result.results
    assert not result.summary.startswith("Error:")
    # The resolved name ("good") was actually searched, against T's
    # storage tenant -- the two failing names did not stop it.
    assert search_spy.searched == [("good", CREATOR, False)]
    # The note lists only the names that failed to resolve -- "good"
    # resolved successfully and must not appear alongside them.
    import re

    note_names = re.search(r"Note: (.+?) could not be resolved", result.summary)
    assert note_names is not None
    assert "good" not in note_names.group(1)


# ---------------------------------------------------------------------------
# The sharing-gap message never discloses the creator's identity.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sharing_gap_message_has_no_creator_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="good",
        storage_user_id=CREATOR,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    _install_collections(
        monkeypatch, {MEMBER: [], CREATOR: [_kb("secret"), _kb("good")]}
    )
    _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=MEMBER, declared=["secret", "good"])

    # The message actually rendered (not the unrelated early-empty-union
    # summary), and it still names no identity.
    assert "has not" in result.summary and "shared with the team" in result.summary
    assert str(CREATOR) not in result.summary
    assert (
        "creator" not in result.summary.lower()
        or "this agent's creator" in result.summary
    )
    assert "@" not in result.summary


# ---------------------------------------------------------------------------
# The creator-existence probe never runs for a name outside the agent's
# stored declaration. Two fixtures that discriminate a correct binding
# from a wrong one: the explicit call style and the default call style.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_creator_probe_ignores_undeclared_name_even_when_creator_really_owns_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The discriminating fixture: the creator genuinely owns
    "victim", but it is not in the agent's stored declaration. The model
    names it via BOTH ``collections`` and its own ``allowed_collections``
    (which ``setdefault`` would not override). The probe must never run,
    and the response must not distinguish this from the creator not
    owning "victim" at all.
    """
    _install_team(monkeypatch, {TEAM: []})
    spy = _install_collections(
        monkeypatch,
        {MEMBER: [_kb("unrelated-own-doc")], CREATOR: [_kb("victim")]},
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        collections=["victim"],
        allowed_collections=["victim"],
    )

    assert spy.calls_for(CREATOR) == 0
    assert result.results == []
    assert "Error:" in result.summary and "not allowed" in result.summary


@pytest.mark.asyncio
async def test_creator_probe_ignores_undeclared_name_even_when_creator_really_owns_it_default_style(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same discriminating shape as the fixture above, but on the default
    call style: the model supplies no explicit ``collections``, only its
    own ``allowed_collections=["victim"]`` -- so this exercises the verdict
    loop's own iteration source directly, not the "not allowed" gate (which
    only runs when the model names collections explicitly). If the loop
    iterated ``allowed_collections`` instead of the agent's stored
    declaration, "victim" would be visited and probed even though it was
    never declared.

    "handbook" (the agent's real, declared name) legitimately reaches the
    probe here -- the memoised call for it is not the leak. What must not
    happen is the outcome depending on whether the creator happens to own
    "victim": the two runs below (creator does / does not own "victim")
    must produce byte-identical responses, since "victim" is never
    declared and so must never be classified at all.
    """
    _install_team(monkeypatch, {TEAM: []})

    _install_collections(
        monkeypatch,
        {MEMBER: [_kb("unrelated-own-doc")], CREATOR: [_kb("victim")]},
    )
    _install_search(monkeypatch)
    with_victim = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        allowed_collections=["victim"],
    )

    _install_collections(monkeypatch, {MEMBER: [_kb("unrelated-own-doc")], CREATOR: []})
    _install_search(monkeypatch)
    without_victim = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        allowed_collections=["victim"],
    )

    assert with_victim.summary == without_victim.summary
    assert with_victim.results == without_victim.results
    assert "victim" not in with_victim.summary


@pytest.mark.asyncio
async def test_creator_probe_memoised_at_most_once_per_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three declared names all need classifying; the probe must be called
    at most once for the whole search, not once per name. The
    de-duplication of the stored declaration is exercised too, via a
    duplicated name in it.
    """
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="anchor",
        storage_user_id=999,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})
    spy = _install_collections(
        monkeypatch,
        {
            MEMBER: [_kb("unrelated-own-doc")],
            999: [_kb("anchor")],
            CREATOR: [],  # creator holds NONE of "one/two/three"
        },
    )
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["anchor", "one", "two", "three", "one"],  # "one" duplicated
    )

    assert spy.calls_for(CREATOR) <= 1
    # The duplicate is reported once, not twice, in the missing-names list.
    import re

    missing_list = re.search(r"Note: (.+?) could not be resolved", result.summary)
    assert missing_list is not None
    names = [n.strip() for n in missing_list.group(1).split(",")]
    assert names.count("one") == 1


# ---------------------------------------------------------------------------
# Team hook not installed => legacy runner-keyed overlay, selection
# on the predicate, never on an empty return.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_declared_name_rule_inert_until_hook_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The three-way gate: a governing team id and a non-empty declaration
    are not enough on their own -- the team-keyed hook must actually be
    installed, or the whole declared-name rule (verdict derivation, the
    re-based "not allowed" gate, and both new report messages) must stay
    off and the run must behave exactly as it does with no governing team
    at all.

    This is deliberately a *different* pin from the team-layer fallback
    test above: that one only checks which collections are visible; this
    one checks that the naming rule does not half-apply while the layer
    is still falling back. A change that lets the rule apply even though
    the socket is not installed would leave the team-layer pin green
    (visibility still falls back correctly) while this one goes red.
    """
    # No team_visibility hook installed at all -- team_knowledge_base_hook_installed()
    # is False. The runner has their own same-named personal collection,
    # which the declared-name rule (if it ran) would have to reject.
    _install_collections(monkeypatch, {MEMBER: [_kb("handbook")], CREATOR: []})
    search_spy = _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=MEMBER, declared=["handbook"])

    # Byte-for-byte legacy behaviour: the runner's own personal collection
    # searches normally, no sharing-gap or missing-name message, no rejection.
    assert search_spy.searched == [("handbook", MEMBER, False)]
    assert result.results
    assert "shared with the team" not in result.summary
    assert "could not be resolved" not in result.summary


@pytest.mark.asyncio
async def test_rule_stays_off_without_an_authenticated_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A governing team, an installed hook and a non-empty declaration are
    not enough on their own: an unauthenticated call (no runner id) never
    had a team layer resolved for it -- ``_list_visible_collections``
    returns before any team resolution when the runner id is absent -- so
    the declared-name rule must stay off rather than run a creator probe on
    nobody's behalf.
    """
    _install_team(monkeypatch, {TEAM: []})
    spy = _install_collections(monkeypatch, {None: [], CREATOR: [_kb("handbook")]})
    _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=None, declared=["handbook"])

    # No probe on nobody's behalf: the creator-existence lookup is never
    # requested, even though the creator genuinely holds "handbook".
    assert spy.calls_for(CREATOR) == 0
    assert "shared with the team" not in result.summary
    # Byte-for-byte the pre-rule behaviour for an unauthenticated call: the
    # inherited empty-visible-set guard, not a declared-name rule outcome.
    assert result.summary == (
        "No knowledge bases available. Please create a knowledge base and "
        "upload documents first."
    )


# ---------------------------------------------------------------------------
# The creator-collection lookup failing degrades to the generic
# outcome and never raises.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("explicit", [False, True])
async def test_creator_listing_failure_reports_unresolved_not_absent(
    monkeypatch: pytest.MonkeyPatch, explicit: bool
) -> None:
    """The creator-collection lookup raises. The search decision is
    unchanged -- nothing resolves, nothing is searched -- but the report
    must not claim the name does not exist: no lookup ever established
    that. The hedged wording is accurate whether or not the creator holds
    it, which is also what keeps the two outcomes indistinguishable. Holds
    on both call styles.
    """
    _install_team(monkeypatch, {TEAM: []})

    async def _broken_list_collections(
        user_id=None, is_admin=None, force_realtime=False
    ):
        if user_id == CREATOR:
            raise RuntimeError("storage unavailable")
        if user_id == MEMBER:
            return _collections(_kb("unrelated-own-doc"))
        return _collections()

    monkeypatch.setattr(document_search, "list_collections", _broken_list_collections)
    _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        collections=["handbook"] if explicit else None,
    )

    assert result.results == []
    assert "Error:" in result.summary
    assert "shared with the team" not in result.summary
    assert "handbook could not be resolved for this agent" in result.summary
    # The certainty wording is exactly what must not appear.
    assert "None of the allowed collections exist" not in result.summary
    assert "do not exist" not in result.summary


@pytest.mark.asyncio
async def test_creator_listing_error_status_reports_unresolved_not_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The creator-collection lookup does not raise this time -- it returns
    a result whose status reports an infrastructure failure, with an empty
    collection list, the way a real failure degrades in
    ``_list_collections_impl``. Read as a plain answer, that empty list
    says "the creator holds nothing", so the probe must treat it the same
    way it treats a raised failure: hedge, not assert an absence no lookup
    ever established.
    """
    _install_team(monkeypatch, {TEAM: []})

    async def _erroring_list_collections(
        user_id=None, is_admin=None, force_realtime=False
    ):
        if user_id == CREATOR:
            return ListCollectionsResult(
                status="error",
                collections=[],
                total_count=0,
                message="storage unavailable",
            )
        if user_id == MEMBER:
            return _collections(_kb("unrelated-own-doc"))
        return _collections()

    monkeypatch.setattr(document_search, "list_collections", _erroring_list_collections)
    _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=MEMBER, declared=["handbook"])

    assert result.results == []
    assert "handbook could not be resolved for this agent" in result.summary
    # The certainty wording is exactly what must not appear.
    assert "do not exist" not in result.summary
    assert "None of the allowed collections exist" not in result.summary


# ---------------------------------------------------------------------------
# The unresolvable terminal branch reports the same two things its sibling
# branches report: the names actually asked about, and what the run could
# still have asked for.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unresolvable_terminal_still_reports_what_was_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The agent declares two names: "good", which the governing team owns
    and hosts on a tenant other than the creator's, and "handbook", which
    needs the creator probe to classify. The model's explicit request names
    only "handbook", and the creator lookup raises. The verdict loop still
    admitted "good" before the narrowing -- resolving a team-owned name
    never touches the creator probe, so the probe's failure cannot take it
    down too -- and the unresolvable branch must say so, the same way the
    mixed branch already does.
    """
    search_spy = _install_team_kb_off_creator_tenant_with_broken_creator_lookup(
        monkeypatch
    )

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["good", "handbook"],
        collections=["handbook"],
    )

    assert search_spy.searched == []
    assert result.results == []
    assert result.summary == (
        "Error: handbook could not be resolved for this agent. "
        "Available collections: good."
    )


@pytest.mark.asyncio
async def test_partial_failure_hedges_when_the_creator_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same fixture as above, no explicit request: "good" is admitted and
    searched, "handbook" is left unclassified by the failed creator lookup.
    This is the partial-failure path, not the terminal one, and its note
    must carry the hedged wording rather than assert that "handbook" is
    absent -- no lookup established that.
    """
    search_spy = _install_team_kb_off_creator_tenant_with_broken_creator_lookup(
        monkeypatch
    )

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["good", "handbook"],
    )

    assert search_spy.searched == [("good", _THIRD_TENANT, False)]
    assert result.results
    assert not result.summary.startswith("Error:")
    assert (
        "Note: handbook could not be resolved for this agent. The agent's "
        "other knowledge bases were searched normally." in result.summary
    )
    assert "do not exist" not in result.summary
    assert "None of the allowed collections exist" not in result.summary
    assert "shared with the team" not in result.summary


@pytest.mark.asyncio
async def test_unresolvable_terminal_reports_only_the_requested_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three declared names all fail to classify because the creator lookup
    raises, and the model's explicit request narrows to just one of them.
    The unresolvable branch must report only the requested name -- the
    other two, which this call never asked about, must not appear.
    """
    _install_team(monkeypatch, {TEAM: []})

    async def _broken_list_collections(
        user_id=None, is_admin=None, force_realtime=False
    ):
        if user_id == CREATOR:
            raise RuntimeError("storage unavailable")
        return _collections()

    monkeypatch.setattr(document_search, "list_collections", _broken_list_collections)
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["alpha-notes", "beta-notes", "gamma-notes"],
        collections=["gamma-notes"],
    )

    assert search_spy.searched == []
    assert result.summary == "Error: gamma-notes could not be resolved for this agent."
    assert "alpha-notes" not in result.summary
    assert "beta-notes" not in result.summary


# ---------------------------------------------------------------------------
# Governing team set, but the agent declares no knowledge bases at all.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_team_governed_run_without_agent_config_uses_governing_team_layer(
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
    spy = _install_collections(
        monkeypatch,
        {
            MEMBER: [_kb("own-material")],
            CREATOR: [_kb("handbook")],
        },
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(monkeypatch, runner_id=MEMBER, declared=[])

    # The team layer is still visible, resolved against T, and gets
    # searched alongside the runner's own material -- an empty declaration
    # falls through to "search everything visible", and the visible set's
    # contents are still resolved on the governing team.
    searched_names = {name for name, _, _ in search_spy.searched}
    assert searched_names == {"own-material", "handbook"}

    # Holds trivially: exactly one creator-directed call, and it is the
    # team-ref resolution's owner scan inside _list_visible_collections --
    # not the creator-existence probe, which never runs for an empty
    # declaration.
    assert spy.calls_for(CREATOR) == 1
    assert "shared with the team" not in result.summary
    assert "could not be resolved" not in result.summary

    # The targeted-confirmation addition: the runner's own material is
    # still searched (the fallback, not an empty verdict-derived set).
    assert ("own-material", MEMBER, False) in search_spy.searched


@pytest.mark.asyncio
async def test_team_owned_undeclared_name_in_explicit_request_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The governing team owns "team-secret", and the runner is a member of
    that team, so it is visible to this run -- but the agent never declared
    it. Being team-owned is not authorisation: the gate is keyed on the
    agent's declaration, and a change that whitelists team-owned names
    would hand the model a knowledge base the agent was never configured
    with.
    """
    team_kbs = [
        kb_scope.KnowledgeBaseAccess(
            name=name,
            storage_user_id=CREATOR,
            team_owned=True,
            can_edit=False,
            can_delete=False,
        )
        for name in ("handbook", "team-secret")
    ]
    _install_team(monkeypatch, {TEAM: team_kbs})
    _install_collections(
        monkeypatch,
        {MEMBER: [], CREATOR: [_kb("handbook"), _kb("team-secret")]},
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=MEMBER,
        declared=["handbook"],
        collections=["team-secret"],
    )

    assert search_spy.searched == []
    assert result.results == []
    assert "not allowed" in result.summary
    clause = result.summary.split("Allowed collections:", 1)[1].strip()
    assert clause == "handbook"


@pytest.mark.asyncio
async def test_admin_governed_run_resolves_on_the_governing_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An admin's collection listing is platform-wide, so it carries every
    tenant's collections -- including the creator's unshared one and other
    users' entirely unrelated ones. The declared-name rule still applies:
    the team's copy is searched against the team's tenant with is_admin
    dropped, the creator's unshared collection is still refused, and the
    summary still names nothing the agent did not declare.
    """
    ADMIN = 400
    team_kb = kb_scope.KnowledgeBaseAccess(
        name="gov-kb",
        storage_user_id=999,
        team_owned=True,
        can_edit=False,
        can_delete=False,
    )
    _install_team(monkeypatch, {TEAM: [team_kb]})

    async def _platform_wide_list_collections(
        user_id=None, is_admin=None, force_realtime=False
    ):
        if is_admin:
            return _collections(
                _kb("gov-kb"), _kb("private-notes"), _kb("someone-elses-doc")
            )
        if user_id == 999:
            return _collections(_kb("gov-kb"))
        if user_id == CREATOR:
            return _collections(_kb("private-notes"))
        return _collections()

    monkeypatch.setattr(
        document_search, "list_collections", _platform_wide_list_collections
    )
    search_spy = _install_search(monkeypatch)

    result = await _search(
        monkeypatch,
        runner_id=ADMIN,
        is_admin=True,
        declared=["gov-kb", "private-notes"],
    )

    # The team's copy is searched against the team's tenant, and the run's
    # own admin flag does not travel into a team collection's search.
    assert search_spy.searched == [("gov-kb", 999, False)]
    # Visible platform-wide, still refused: the rule is not a visibility
    # filter.
    assert "private-notes" in result.summary
    assert "has not" in result.summary and "shared with the team" in result.summary
    assert "someone-elses-doc" not in result.summary
