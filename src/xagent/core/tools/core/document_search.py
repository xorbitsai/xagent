"""Core document search functionality for RAG pipelines."""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ....config import get_kb_search_timeout_seconds
from .knowledge_base_scope import KnowledgeBaseScopeError
from .RAG_tools.core.schemas import CollectionInfo, ListCollectionsResult
from .RAG_tools.management.collections import list_collections
from .RAG_tools.pipelines.document_search import run_document_search

logger = logging.getLogger(__name__)

# Prefix of the serialized READONLY_MODE warning that search_sparse raises
# unconditionally under readonly=True (see collection_handle.search_sparse).
# Matched against the string, not SearchWarning.code: the pipeline flattens
# warnings to f"{code}: {message}" in _serialize_warnings before they reach us,
# and the structured object is not plumbed this far. Change that format and this
# filter silently stops matching - the readonly notice reappears in summaries,
# which the readonly tests in test_document_search_collection_concurrency catch.
_READONLY_WARNING_PREFIX = "READONLY_MODE:"

# The fixed, unparaphrasable strings a team-governed search reports through.
# See the module's callers for when each one is used; the wording itself is
# a reviewed user-facing contract and must not be edited casually.
_UNSHARED_CREATOR_KB_MESSAGE = (
    "{name} is a personal knowledge base belonging to this agent's creator and has not "
    "been shared with the team. Ask the agent's owner to share it with the team to make "
    "it available here."
)
_PARTIAL_FAILURE_NOTE = (
    "Note: {names} could not be resolved for this agent. The agent's other knowledge "
    "bases were searched normally."
)
# The terminal counterpart of the note above, for the one case the certainty
# wording would be a falsehood: the creator-collection lookup did not
# complete -- it raised, or returned a result whose status reports failure --
# so nothing established that these names are absent, only that they could
# not be classified. Reuses the note's hedge; nothing was searched, so it does not
# borrow the note's second sentence.
_TERMINAL_UNRESOLVABLE_MESSAGE = "Error: {names} could not be resolved for this agent."

if TYPE_CHECKING:
    from .RAG_tools.kb import KBToolCompatibilityFacade


def _get_tool_compatibility_facade() -> "KBToolCompatibilityFacade":
    """Return the coordinator-owned tool compatibility facade."""
    from .RAG_tools.kb import get_kb_coordinator

    return get_kb_coordinator().tool_compatibility


async def _list_visible_collections(
    user_id: Optional[int],
    is_admin: bool,
    governing_team_id: Optional[int] = None,
) -> ListCollectionsResult:
    """Union personal collections with application-provided team overlays.

    For a team-governed run (``governing_team_id`` set and a team-keyed hook
    installed), the team layer is the *governing* team's own knowledge
    bases, resolved through ``resolve_team_knowledge_bases_or_raise`` --
    never the runner's own team memberships, and never a union of the two.
    Every other case falls back to the existing runner-keyed overlay through
    ``visible_team_knowledge_bases``, unchanged.

    This function stays a pure union either way: it decides *whose* team
    rows are visible, never which of the agent's declared names may search
    them -- that rule belongs to the callers that read the agent's stored
    configuration, not here, so a future consumer of this function cannot
    inherit the declared-name rule by accident.
    """
    result = await list_collections(user_id=user_id, is_admin=is_admin)
    # This is not an optimisation: the calls below do int(user_id), so
    # removing this guard would turn an unauthenticated caller into a
    # TypeError instead of this well-defined early return. It keeps its
    # position as the first check for exactly that reason -- the governing
    # team branch below still needs a real user_id.
    if user_id is None:
        return result

    from ....web.services.db_runtime import run_db_io_cancellation_safe
    from ....web.services.knowledge_base_team_scope import (
        has_knowledge_base_visibility_hook,
        resolve_team_knowledge_bases_or_raise,
        team_knowledge_base_hook_installed,
        visible_team_knowledge_bases,
    )

    if governing_team_id is not None and team_knowledge_base_hook_installed():
        # Selection is on the predicate above, never on an empty return: an
        # installed hook legitimately answers "this team owns nothing".
        #
        # Reached before the is_admin short-circuit below on purpose: for a
        # team-governed run, the team layer's source is the governing team,
        # not the admin's own team memberships. This changes only where the
        # team layer's rows come from -- the admin's platform-wide view in
        # ``result`` is not narrowed by it; matching names are re-stamped
        # with the team's ownership metadata below, the rest stay as-is.
        team_refs = await run_db_io_cancellation_safe(
            lambda: resolve_team_knowledge_bases_or_raise(
                None, team_id=governing_team_id, log_subject=user_id
            )
        )
    elif is_admin:
        return result
    elif not has_knowledge_base_visibility_hook():
        return result
    else:
        team_refs = await run_db_io_cancellation_safe(
            lambda: visible_team_knowledge_bases(None, int(user_id))
        )

    collections_by_name = {
        collection.name: collection for collection in result.collections
    }
    refs_by_owner: dict[int, list] = {}
    for ref in team_refs:
        refs_by_owner.setdefault(ref.storage_user_id, []).append(ref)
    for storage_user_id, refs in refs_by_owner.items():
        owner_result = await list_collections(user_id=storage_user_id, is_admin=False)
        owner_collections = {
            collection.name: collection for collection in owner_result.collections
        }
        for ref in refs:
            collection = owner_collections.get(ref.name)
            if collection is None:
                continue
            # The sole source of an ``ownership == "team"`` entry in the
            # merged map is this loop -- every consumer's "does the
            # governing team own this name" test (below, and in the two
            # impls further down) is only trustworthy because of that.
            collections_by_name[ref.name] = collection.model_copy(
                update={
                    "ownership": "team",
                    "storage_user_id": ref.storage_user_id,
                    "can_edit": ref.can_edit,
                    "can_delete": ref.can_delete,
                }
            )
    merged = list(collections_by_name.values())
    return result.model_copy(update={"collections": merged, "total_count": len(merged)})


class ListKnowledgeBasesArgs(BaseModel):
    """Arguments for listing knowledge bases."""

    allowed_collections: Optional[List[str]] = Field(
        default=None,
        description="Optional list of allowed collection names to filter. None means list all collections.",
    )


class ListKnowledgeBasesResult(BaseModel):
    knowledge_bases: List[Dict[str, Any]] = Field(
        description="List of available knowledge bases with statistics"
    )


class KnowledgeSearchArgs(BaseModel):
    query: str = Field(description="The search query or question")
    collections: List[str] = Field(
        default=[],
        description="Specific knowledge base collection names to search. Empty list searches the agent's default set: its own stored knowledge base list when the agent is owned by a team and that list is non-empty, otherwise allowed_collections when set, and every collection visible to this run when not.",
    )
    search_type: str = Field(
        default="hybrid",
        description="Search type: 'dense' (semantic), 'sparse' (keyword), or 'hybrid' (combined)",
    )
    top_k: int = Field(default=5, description="Maximum results per collection")
    min_score: float = Field(
        default=0.3, description="Minimum relevance score (0.0-1.0)"
    )
    embedding_model_id: Optional[str] = Field(
        default=None, description="Optional embedding model ID to use for searches"
    )
    rerank_model_id: Optional[str] = Field(
        default=None,
        description="Optional rerank model ID (registered in model hub) to rerank search results",
    )
    allowed_collections: Optional[List[str]] = Field(
        default=None,
        description="Optional list of allowed collection names. Used as default when collections is empty. Ignored when the agent is owned by a team and its stored knowledge base list is non-empty, which is then the authority.",
    )


class SearchResultItem(BaseModel):
    """Single search result with document information."""

    collection: str = Field(description="Knowledge base collection name")
    score: float = Field(description="Relevance score (0.0-1.0)")
    text: str = Field(description="Document text content")
    document_name: str = Field(default="", description="Original document filename")
    source_path: str = Field(default="", description="Full file path")
    doc_id: str = Field(default="", description="Internal document ID")
    chunk_id: str = Field(default="", description="Internal chunk ID")


class KnowledgeSearchResult(BaseModel):
    results: list[SearchResultItem] = Field(
        description="List of search results with document metadata"
    )
    summary: str = Field(
        default="", description="Human-readable summary of search results"
    )


async def list_knowledge_bases(
    tool_args: ListKnowledgeBasesArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
) -> ListKnowledgeBasesResult:
    """List all available knowledge bases through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().list_knowledge_bases(
        tool_args,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
    )


async def _list_knowledge_bases_impl(
    tool_args: ListKnowledgeBasesArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
) -> ListKnowledgeBasesResult:
    """List all available knowledge bases with their statistics.

    Args:
        tool_args: Args with optional allowed_collections filter
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges
        governing_team_id: The governing agent's owning team, if any. Only
            affects *which* collections are visible (the team layer resolves
            on this team instead of the runner's own team memberships); see
            ``_list_visible_collections``.

    Note:
        Unlike ``_search_knowledge_base_impl``, this function takes neither
        the governing agent's creator nor its stored knowledge-base
        declaration. It only reports what is available: it classifies no
        individual declared name, so an unresolved name simply does not
        appear in the returned list, with no separate message field, and it
        never triggers the creator-existence probe. Adding those two values
        here would give it nothing to do.

    Returns:
        ListKnowledgeBasesResult containing knowledge base information

    Raises:
        KnowledgeBaseScopeError: If the team-scope hook is installed but
            fails or returns a malformed answer.
        RuntimeError: If listing knowledge bases fails for any other reason.
    """
    try:
        result = await _list_visible_collections(
            user_id=user_id,
            is_admin=is_admin,
            governing_team_id=governing_team_id,
        )

        kb_list = []
        for collection in result.collections:
            # Filter by allowed_collections if specified
            if (
                tool_args.allowed_collections is not None
                and collection.name not in tool_args.allowed_collections
            ):
                continue

            kb_list.append(
                {
                    "name": collection.name,
                    "documents": collection.documents,
                    "embeddings": collection.embeddings,
                    "document_names": list(collection.document_names)
                    if collection.document_names
                    else [],
                }
            )

        return ListKnowledgeBasesResult(knowledge_bases=kb_list)

    except KnowledgeBaseScopeError:
        raise
    except Exception as e:
        logger.error(f"Failed to list knowledge bases: {e}", exc_info=True)
        raise RuntimeError(f"Failed to list knowledge bases: {e}") from e


async def find_missing_knowledge_bases(
    knowledge_bases: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    """Return missing KB names through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().find_missing_knowledge_bases(
        knowledge_bases,
        user_id=user_id,
        is_admin=is_admin,
    )


async def _find_missing_knowledge_bases_impl(
    knowledge_bases: List[str],
    user_id: Optional[int] = None,
    is_admin: bool = False,
) -> List[str]:
    """Return requested knowledge base names that are not visible to the user.

    Deliberately stays runner-keyed: this backs the agent-builder's
    save-time validation of an agent's own collection selection, which must
    answer against the runner's own visible set, not any governing team --
    an agent being edited is not yet the governing agent of any run.
    """
    requested = [name.strip() for name in knowledge_bases if name and name.strip()]
    if not requested:
        return []

    result = await _list_visible_collections(user_id=user_id, is_admin=is_admin)
    available = {collection.name for collection in result.collections}
    return [name for name in requested if name not in available]


async def search_knowledge_base(
    tool_args: KnowledgeSearchArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
    agent_creator_user_id: Optional[int] = None,
    declared_knowledge_bases: Optional[List[str]] = None,
) -> KnowledgeSearchResult:
    """Search across knowledge bases through the tool compatibility facade."""
    return await _get_tool_compatibility_facade().search_knowledge_base(
        tool_args,
        user_id=user_id,
        is_admin=is_admin,
        governing_team_id=governing_team_id,
        agent_creator_user_id=agent_creator_user_id,
        declared_knowledge_bases=declared_knowledge_bases,
    )


class _Unresolved:
    """Sentinel: a declared name the governing team does not own, and the
    runner is not the agent's creator.

    Distinguishes "needs the creator-existence probe" from "resolved to a
    collection" (any other object) and from "give up, report missing"
    (``None``), without conflating either with a real ``CollectionInfo``.
    """

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unresolved>"


_UNRESOLVED = _Unresolved()


def _resolve_declared_name(
    name: str,
    resolved: Dict[str, CollectionInfo],
    *,
    governing_team_id: Optional[int],
    agent_creator_user_id: Optional[int],
    runner_id: Optional[int],
    declared_names: "set[str] | frozenset[str]",
) -> "CollectionInfo | _Unresolved | None":
    """Governing team, then creator, then declared-name membership, in that
    order. Returns a collection, ``_UNRESOLVED`` (needs the
    creator-existence probe), or ``None`` (nothing to search).

    ``declared_names`` is the agent's STORED ``knowledge_bases`` list, never
    ``allowed_collections`` and never ``tool_args.collections`` -- both of
    those live on the model-supplied ``tool_args`` and can be overwritten by
    the model itself, so neither can be trusted as the boundary of what may
    be probed. See the ``name not in declared_names`` check below.
    """
    if governing_team_id is None:
        # No governing team: today's behaviour, untouched.
        return resolved.get(name)

    entry = resolved.get(name)
    if entry is not None and entry.ownership == "team":
        # A name the governing team owns is resolved here and never reaches
        # the creator check below. This test is trustworthy only because
        # ``_list_visible_collections`` guarantees the sole source of an
        # ``ownership == "team"`` entry is the governing team resolved
        # above -- never the runner's own team memberships.
        return entry

    if runner_id is not None and runner_id == agent_creator_user_id:
        # The team does not own this name, but the runner is the agent's
        # creator: their own personal collection (if any) still resolves,
        # exactly as it would with no governing team at all.
        return entry

    if name not in declared_names:
        # On the one caller today, this is always False: the caller's loop
        # iterates ``declared_names_for_this_run`` itself, so ``name`` is
        # already a member of ``declared_names`` (built from that same
        # list) by construction -- the iteration source is what actually
        # binds the probe to the agent's stored configuration here. This
        # check is a second guard for any future caller that classifies a
        # name from somewhere else: without it, a model-supplied name could
        # reach the probe below, and the two distinct outcomes it produces
        # would become an enumeration oracle over the creator's private
        # collection names.
        return None

    return _UNRESOLVED


async def _team_names_hosted_on(storage_user_id: int) -> set:
    """Names on ``storage_user_id``'s tenant that a team owns, not the user.

    A team's knowledge base lives in some member's storage namespace, so a
    raw ``list_collections`` for that member returns the team's rows
    alongside their own, and nothing on the row itself separates them:
    ``CollectionInfo.ownership`` is ``"personal"`` on every row
    ``list_collections`` produces -- the one place it becomes ``"team"`` is
    the overlay in ``_list_visible_collections``, which reads the very hook
    this function reads.

    Asked through the runner-keyed visibility hook, called with the
    tenant's own id, because it is the only seam that answers "which
    knowledge bases does a team own" for a team other than the governing
    one. The answer only ever narrows what may be called this user's own,
    and is never rendered: a name it removes is reported as absent instead
    of as the creator's, so the report can say less about the creator than
    before, never more.

    Bounded by what the seam can answer: it reports the teams this user is
    a member of *now*, so a team's row hosted on the tenant of someone who
    has since left that team still counts as personal. Closing that would
    need a "which team owns this name" question the hook contract does not
    have, and answering it would widen what a run can learn.
    """
    from ....web.services.db_runtime import run_db_io_cancellation_safe
    from ....web.services.knowledge_base_team_scope import (
        has_knowledge_base_visibility_hook,
        visible_team_knowledge_bases,
    )

    if not has_knowledge_base_visibility_hook():
        # Without the runner-keyed hook this question cannot be asked, so
        # every row on the tenant is treated as the user's own. Exact on
        # standalone xagent, where no team owns anything at all. On a
        # deployment that installed only the team-keyed hook it is not:
        # teams do exist there, and a team's row hosted on the creator's
        # tenant is classified as the creator's personal collection, so the
        # sharing-gap sentence can name a knowledge base another team owns.
        # Search decisions are unaffected -- the name is not searched under
        # either classification -- and the ask stops here rather than
        # falling back to the team-keyed hook, which answers only for the
        # governing team and so cannot answer this question.
        return set()
    refs = await run_db_io_cancellation_safe(
        lambda: visible_team_knowledge_bases(None, int(storage_user_id))
    )
    # Only the rows physically hosted on this tenant. A team row stored
    # elsewhere never appeared in this tenant's listing, and excluding its
    # name would drop a same-named collection this user really does own.
    return {ref.name for ref in refs if ref.storage_user_id == storage_user_id}


class _CreatorCollectionProbe:
    """Memoised lookup of the agent creator's own collection names.

    At most one creator lookup per search call -- one collection listing
    and one team-ownership question, both memoised together -- regardless
    of how many declared names need classifying. The lookup result is
    cached after the first call, including a failure (cached as
    "nothing"). A failure degrades to "not held" and never
    raises: a probe that cannot complete must not block the search, and
    must not be distinguishable from "the creator does not have this
    collection"; what that costs, and where the guarantee stops, is stated
    in the two paragraphs below.

    A failure is still not distinguishable per name -- it hits every name
    this run classifies alike -- so the enumeration oracle stays closed;
    what the caller does with ``lookup_failed`` is choose a report that does
    not assert an absence, not reveal one.

    On a deployment that does not install the save-time knowledge-base
    validator, this still leaves a costlier existence-only side channel
    open to anyone who can edit the agent, not only the creator: the stored
    ``knowledge_bases`` declaration this probe classifies against is
    itself editable by any same-team member, not just the creator --
    ``agent_store.get_owned_agent`` (via ``owned_agent_clause``) lets a
    non-admin team member update any team agent whose ``visibility ==
    'team'``, including its ``knowledge_bases`` list. A team member can
    therefore add a personal-collection-shaped name of their choosing to
    the declaration, run the agent, and read the sharing-gap sentence
    versus the generic "does not exist" outcome off the summary to learn
    whether the creator holds a same-named private collection -- at the
    cost of an agent edit per name probed, and revealing only that
    existence, never the collection's contents. The reference deployment
    does install that validator, and the edit this channel needs never
    lands there: an update whose declaration names a knowledge base the
    team does not own is rejected at save time (``agent_store`` ->
    ``validate_team_agent_knowledge_bases`` ->
    ``UnsharedKnowledgeBasesError``). With no validator installed that
    call returns an empty list and the edit is accepted, which is the
    configuration the paragraph above describes.
    """

    def __init__(self, agent_creator_user_id: Optional[int]) -> None:
        self._agent_creator_user_id = agent_creator_user_id
        self._names: Optional[set] = None
        # Whether the one lookup this probe makes failed -- raised, or
        # returned a result whose status reports failure. Read by the caller
        # to tell "the creator does not hold this name" apart from "nobody
        # asked" -- the failure degrades to "not held" for the search
        # decision (unchanged), but the report must not claim an absence
        # that was never established.
        self.lookup_failed = False

    async def holds(self, name: str) -> bool:
        if self._agent_creator_user_id is None:
            return False
        if self._names is None:
            try:
                creator_result = await list_collections(
                    user_id=self._agent_creator_user_id, is_admin=False
                )
                # Inside the same try on purpose: if the team lookup fails,
                # which of these rows are the creator's own is unknown, and
                # the caller must hedge rather than assert the creator
                # holds them. Falling back to the unfiltered listing would
                # restore exactly the misclassification this call exists to
                # remove.
                team_hosted = await _team_names_hosted_on(self._agent_creator_user_id)
            except Exception:
                # Deliberately blanket, the seam's typed
                # KnowledgeBaseScopeError included. Nothing inside this try
                # produces that type today: the listing reports failure
                # through its status instead of raising, and the
                # runner-keyed hook is called directly, without the wrapper
                # that mints it. If a typed producer is ever added here,
                # swallowing stays right for this one site: the probe's
                # answer chooses how a failure is worded and never what is
                # searched or by whom, and its contract is to degrade to
                # "not held" rather than abort a search that can still
                # succeed for the agent's other knowledge bases. Every
                # other handler of that type re-raises it; this one must
                # not, and a change of mind here is a change to the probe's
                # documented never-raise contract above.
                logger.warning(
                    "Failed to resolve creator's personal collections while "
                    "classifying an unshared knowledge base",
                    exc_info=True,
                )
                self._names = set()
                self.lookup_failed = True
            else:
                if creator_result.status == "success":
                    self._names = {
                        c.name for c in creator_result.collections
                    } - team_hosted
                else:
                    # The listing reports an infrastructure failure by
                    # returning status="error" with an empty collection list
                    # instead of raising (RAG_tools/management/collections).
                    # Read as a plain answer, that empty list says "the
                    # creator holds nothing", and the caller would assert
                    # the certain "does not exist" wording off a lookup that
                    # never completed. Same treatment as a raised failure:
                    # nothing is established, so the report must hedge.
                    logger.warning(
                        "Creator collection listing reported status=%s while "
                        "classifying an unshared knowledge base: %s",
                        creator_result.status,
                        creator_result.message,
                    )
                    self._names = set()
                    self.lookup_failed = True
        return name in self._names


def _render_unshared_names(names: List[str]) -> str:
    """One sentence per name, space-joined, in declaration order."""
    return " ".join(_UNSHARED_CREATOR_KB_MESSAGE.format(name=name) for name in names)


def _render_missing_names_note(names: List[str]) -> str:
    return _PARTIAL_FAILURE_NOTE.format(names=", ".join(names))


async def _search_knowledge_base_impl(
    tool_args: KnowledgeSearchArgs,
    user_id: Optional[int] = None,
    is_admin: bool = False,
    governing_team_id: Optional[int] = None,
    agent_creator_user_id: Optional[int] = None,
    declared_knowledge_bases: Optional[List[str]] = None,
) -> KnowledgeSearchResult:
    """Search across knowledge base collections.

    Args:
        tool_args: Search configuration including query, collections, and search parameters
        user_id: Optional user ID for multi-tenancy filtering
        is_admin: Whether the user has admin privileges
        governing_team_id: The governing agent's owning team, if any.
        agent_creator_user_id: The governing agent's creator, if any. Used
            to tell the agent's own creator apart from every other runner
            of a team-governed agent.
        declared_knowledge_bases: The governing agent's STORED
            ``knowledge_bases`` declaration, if any -- never
            ``tool_args.allowed_collections`` and never
            ``tool_args.collections``, both of which are model-authored on
            every path (a declared schema field the model can overwrite).
            Both chat call sites derive this and the tool's allowed
            collections from that one stored field, so in production the
            two are equal by construction; a test that pairs a declaration
            with different allowed collections is isolating forwarding, and
            this function's behaviour is defined for that pairing too.

    Returns:
        KnowledgeSearchResult with formatted search results

    Raises:
        KnowledgeBaseScopeError: If the team-scope hook is installed but
            fails or returns a malformed answer.
        RuntimeError: If search fails for any other reason.
    """
    try:
        # List all collections
        collections_result = await _list_visible_collections(
            user_id=user_id,
            is_admin=is_admin,
            governing_team_id=governing_team_id,
        )

        # The agent's stored declaration, de-duplicated with insertion order
        # kept (a stored declaration may legitimately contain a name twice --
        # nothing normalises Agent.knowledge_bases into a set -- and an
        # un-de-duplicated iteration would probe, and report, a duplicate
        # name twice).
        declared_names_for_this_run = list(
            dict.fromkeys(declared_knowledge_bases or [])
        )

        from ....web.services.knowledge_base_team_scope import (
            team_knowledge_base_hook_installed,
        )

        # Team-governed run WITH a declaration, on a deployment that has
        # actually installed the team-keyed hook: the declaration is the
        # authority for both what may be searched and what gets reported.
        # Gated on this four-way conjunction, not merely on
        # governing_team_id being set, for three independent reasons:
        #
        # - an unauthenticated call (``user_id`` is None) never had a team
        #   layer resolved for it at all: ``_list_visible_collections``
        #   returns before any team resolution when the runner id is absent,
        #   so the visible set it produced is the plain listing. Running the
        #   rule on top of that would classify every declared name through a
        #   creator probe made on nobody's behalf. Both chat call sites carry
        #   a real runner id today; this conjunct keeps it that way;
        # - an empty declaration (the case where agent_config carried no
        #   agent at all) has nothing to authorise or verdict-derive against,
        #   so the declared-name rule does not apply and the search falls
        #   through to the existing "search everything visible" branch
        #   below. The governing team's own knowledge bases stay in that
        #   visible set -- ``_list_visible_collections`` resolves the team
        #   layer onto the governing team whenever one is set and the hook
        #   is installed, independently of whether anything was declared --
        #   so an empty declaration still searches the runner's own
        #   material *and* the governing team's, the same as it always has;
        #   only which team the team layer resolves against changes. This
        #   conjunct is defensive rather than a live production case:
        #   knowledge_tools.py builds no knowledge tool at all when the
        #   agent's allowed collections are an empty list, and the two chat
        #   call sites derive that list and this declaration from the same
        #   expression, so a declaring-nothing agent never reaches a search;
        #   the remaining way in is an absent declaration, which on those
        #   call sites also means an absent governing team. Kept so a future
        #   caller that does supply a governing team without a declaration
        #   inherits today's behaviour instead of an empty search set;
        # - a deployment that has not installed the team-keyed hook must
        #   stay on today's behaviour byte for byte, the same way the team
        #   layer itself falls back in ``_list_visible_collections``. An
        #   upstream change must be safe to ship ahead of any downstream
        #   hook installation: selecting on "is a governing team id present"
        #   instead of "is the hook actually installed" would turn on the
        #   declared-name rule (and its two new report messages) for a
        #   deployment whose team layer still resolves the old, runner-keyed
        #   way -- a half-migrated state nothing asked for.
        #
        # This gate and the team layer's own gate in
        # ``_list_visible_collections`` are not the same predicate: they
        # share ``governing_team_id is not None and
        # team_knowledge_base_hook_installed()``, but this one adds two more
        # conjuncts (``user_id is not None`` and
        # ``declared_names_for_this_run``) that the team layer's gate does
        # not read. That is intentional, not a mismatch to close --
        # the team layer's job is only "resolve onto the governing team",
        # which an empty declaration does not change; this gate's job is
        # "does the declared-name rule apply at all", which an empty
        # declaration answers no to.
        declared_name_rule_applies = (
            user_id is not None
            and governing_team_id is not None
            and team_knowledge_base_hook_installed()
            and bool(declared_names_for_this_run)
        )

        # The inherited empty-visible-set guard, now skipped for exactly the
        # runs the declared-name rule owns the reporting of. With a governing
        # team, an installed hook and a non-empty declaration, every declared
        # name must still get its own verdict and its own sentence even when
        # nothing at all is visible: a team that owns nothing plus a runner
        # who owns nothing is a legitimate answer, and returning here would
        # make it indistinguishable from a platform with no knowledge bases.
        # Every other input -- no governing team, hook not installed, empty
        # declaration -- keeps this guard exactly as it is today.
        if not collections_result.collections and not declared_name_rule_applies:
            return KnowledgeSearchResult(
                results=[],
                summary="No knowledge bases available. Please create a knowledge base and upload documents first.",
            )

        # Determine which collections to search
        available_names = {c.name for c in collections_result.collections}

        # Debug: Log available collections for troubleshooting
        logger.info(
            f"📚 Available knowledge base collections: {sorted(available_names)}"
        )
        if tool_args.collections:
            logger.info(f"   - Requested collections: {tool_args.collections}")
        if tool_args.allowed_collections:
            logger.info(f"   - Allowed collections: {tool_args.allowed_collections}")

        # Collected by whichever branch below resolves the search set; a
        # partial failure warns without stopping the rest of the agent's
        # knowledge bases from being searched normally.
        partial_failure_notes = []

        if declared_name_rule_applies:
            resolved_by_name = {c.name: c for c in collections_result.collections}
            authorised = set(declared_names_for_this_run)

            explicit_request: Optional[set] = None
            if tool_args.collections:
                requested_set = set(tool_args.collections)
                # The re-based "not allowed" gate (wording unchanged): the
                # authority is the agent's stored declaration, not
                # tool_args.allowed_collections, because that field is
                # model-authored on every path and setdefault lets a
                # model-supplied value win outright over the agent's own
                # configured one.
                disallowed = requested_set - authorised
                if disallowed:
                    return KnowledgeSearchResult(
                        results=[],
                        summary=f"Error: The following collections are not allowed: {', '.join(sorted(disallowed))}. "
                        # The clause lists the declaration itself, not its
                        # intersection with the visible union: that union
                        # seeds the runner's own same-named personal
                        # collections, which the verdict loop below then
                        # refuses to search -- advertising them here as
                        # allowed and rejecting them one step later is the
                        # same contradiction the terminal branch avoids.
                        f"Allowed collections: {', '.join(sorted(authorised))}",
                    )
                explicit_request = requested_set
                logger.info(f"Searching specific collections: {sorted(requested_set)}")

            to_search: List[CollectionInfo] = []
            unshared_names: List[str] = []
            missing_names: List[str] = []
            unresolvable_names: List[str] = []
            creator_probe = _CreatorCollectionProbe(agent_creator_user_id)

            # One pass over the agent's stored declaration; every declared
            # name gets exactly one verdict, and the two report sets plus
            # the search set are all built from the verdicts -- never from
            # an intersection with available_names. That expression is what
            # would let the runner's own same-named personal collection back
            # in: available_names already contains it (the base union seeds
            # the runner's own collections), so a rule hung off its
            # complement would never see the very name it exists to reject.
            for name in declared_names_for_this_run:
                verdict = _resolve_declared_name(
                    name,
                    resolved_by_name,
                    governing_team_id=governing_team_id,
                    agent_creator_user_id=agent_creator_user_id,
                    runner_id=user_id,
                    declared_names=authorised,
                )
                if verdict is None:
                    missing_names.append(name)
                elif isinstance(verdict, _Unresolved):
                    if await creator_probe.holds(name):
                        unshared_names.append(name)
                    elif creator_probe.lookup_failed:
                        # Not "missing": the lookup that would have decided
                        # did not complete (raised, or reported failure
                        # through its status). The search outcome is the same either way
                        # (nothing resolves), only the report differs.
                        unresolvable_names.append(name)
                    else:
                        missing_names.append(name)
                else:
                    to_search.append(verdict)

            # Every name the verdict loop admitted, captured before the
            # explicit-request narrowing below. This -- never the visible
            # union ``available_names`` -- is what the terminal messages
            # report as available. A name only lands here after resolving to
            # a collection this run may actually search, so the runner's own
            # same-named personal copy (which the union does carry, and which
            # the rule exists to reject) can never be reported as available.
            admitted_names = {c.name for c in to_search}

            if explicit_request is not None:
                # The model may narrow the search to a subset of what the
                # verdicts admitted; it may never widen it past that set.
                to_search = [c for c in to_search if c.name in explicit_request]

            if not to_search:
                # Every declared name failed to resolve. Reached through
                # this guard, not one keyed on the raw requested/allowed
                # set: a guard keyed on the requested/allowed names alone
                # stays non-empty even when every one of them failed to
                # resolve to a real collection, as long as the runner
                # happens to hold a same-named personal copy of each one --
                # which would fall through to the generic search-miss text
                # below instead of the two outcomes this branch exists to
                # report.
                # The summary is derived from the verdicts, one report set
                # per verdict class, so no name is described twice and none
                # is described as both absent and held. "Available" is
                # always the verdict-admitted set, never
                # ``authorised & available_names``: that intersection
                # reports two sets of names it must not -- the governing
                # team's collections the agent never declared, and the
                # runner's own same-named personal copies, which seed the
                # visible union and are precisely what the rule refuses to
                # search. The admitted set is the one taken before the
                # narrowing above, so a run whose requested names all
                # failed still tells the model what it may ask for.
                reported_scope = (
                    explicit_request
                    if explicit_request is not None
                    else set(declared_names_for_this_run)
                )
                failed_unshared_names = [
                    name for name in unshared_names if name in reported_scope
                ]
                failed_missing_names = [
                    name for name in missing_names if name in reported_scope
                ]

                if unresolvable_names:
                    # The creator-collection lookup did not complete during
                    # this call, so every unresolved declared name is
                    # unclassified, not established absent. Reported as one
                    # undifferentiated outcome -- a lookup failure is not
                    # per-name, so this sentence still says nothing about
                    # which names the creator holds -- and scoped like the
                    # other two report sets, so an explicit request is
                    # answered about the names it asked for.
                    #
                    # The narrowed list cannot come out empty: with an
                    # explicit request, reaching this branch means every
                    # requested name was unresolved, because a requested
                    # name the verdict loop admitted would have survived the
                    # narrowing and left the search set non-empty; without
                    # one, the scope is the whole declaration.
                    #
                    # The other two report sets are necessarily empty here:
                    # the first failure caches the creator's name set as
                    # empty and marks the probe failed, so every later
                    # unresolved name takes this same classification, and a
                    # name reaches the missing set only on a run whose
                    # runner is the creator -- which never calls the probe
                    # at all.
                    #
                    # The admitted clause renders on the same condition as
                    # the mixed branch below, so a request narrowed away to
                    # nothing still says what could have been asked for.
                    failed_unresolvable_names = [
                        name for name in unresolvable_names if name in reported_scope
                    ]
                    unresolvable_segments = [
                        _TERMINAL_UNRESOLVABLE_MESSAGE.format(
                            names=", ".join(failed_unresolvable_names)
                        )
                    ]
                    if admitted_names:
                        unresolvable_segments.append(
                            "Available collections: "
                            f"{', '.join(sorted(admitted_names))}."
                        )
                    summary = " ".join(unresolvable_segments)
                elif not failed_unshared_names:
                    # Nothing in scope is the creator's own collection, so
                    # the two inherited terminal sentences are true as
                    # written and each renders byte for byte on its own
                    # call style. The availability clause they carry is
                    # dropped when the verdict loop admitted nothing, the
                    # same as the mixed branch below: its source is the
                    # admitted set, which this rule can empty where the
                    # visible union it replaced never could, and a label
                    # with an empty list after it tells the model nothing.
                    if explicit_request is not None:
                        summary = (
                            "Error: The following collections do not exist: "
                            f"{', '.join(sorted(explicit_request))}."
                        )
                        if admitted_names:
                            summary += (
                                " Available collections: "
                                f"{', '.join(sorted(admitted_names))}"
                            )
                    else:
                        summary = (
                            "Error: None of the allowed collections exist. "
                            f"Allowed: {', '.join(sorted(authorised))}."
                        )
                        if admitted_names:
                            summary += (
                                f" Available: {', '.join(sorted(admitted_names))}"
                            )
                else:
                    # At least one failed name is the creator's own
                    # collection, which does exist. Neither inherited
                    # sentence can carry that: one is a statement about
                    # every allowed collection, the other enumerates the
                    # whole request, and both would assert the absence of a
                    # name this same summary goes on to say the creator
                    # holds. The absence sentence therefore enumerates only
                    # the names actually classified missing and is dropped
                    # when there are none; the availability clause is
                    # dropped when nothing was admitted rather than
                    # rendering an empty list. Same shape on both call
                    # styles: each report set lists only its own names, and
                    # a name appears in exactly one of them.
                    segments: List[str] = []
                    if failed_missing_names:
                        segments.append(
                            "The following collections do not exist: "
                            f"{', '.join(sorted(failed_missing_names))}."
                        )
                    segments.append(_render_unshared_names(failed_unshared_names))
                    if admitted_names:
                        segments.append(
                            "Available collections: "
                            f"{', '.join(sorted(admitted_names))}."
                        )
                    summary = "Error: " + " ".join(segments)
                return KnowledgeSearchResult(results=[], summary=summary)

            collections_to_iterate = to_search
            if unshared_names:
                partial_failure_notes.append(_render_unshared_names(unshared_names))
            if missing_names or unresolvable_names:
                partial_failure_notes.append(
                    _render_missing_names_note(missing_names + unresolvable_names)
                )
        elif tool_args.collections:
            # No governing team, and the empty-declaration fallback, share
            # this branch: byte-for-byte today's behaviour, including its
            # allowed_collections-is-None skip below.
            requested_set = set(tool_args.collections)

            if tool_args.allowed_collections is not None:
                allowed_set = set(tool_args.allowed_collections)
                disallowed = requested_set - allowed_set

                if disallowed:
                    return KnowledgeSearchResult(
                        results=[],
                        summary=f"Error: The following collections are not allowed: {', '.join(sorted(disallowed))}. "
                        f"Allowed collections: {', '.join(sorted(allowed_set & available_names))}",
                    )

                collections_set = requested_set & allowed_set
            else:
                collections_set = requested_set

            invalid_names = collections_set - available_names
            if invalid_names:
                return KnowledgeSearchResult(
                    results=[],
                    summary=f"Error: The following collections do not exist: {', '.join(invalid_names)}. "
                    f"Available collections: {', '.join(sorted(available_names))}",
                )

            collections_to_iterate = [
                c for c in collections_result.collections if c.name in collections_set
            ]
            logger.info(f"Searching specific collections: {sorted(collections_set)}")
        elif tool_args.allowed_collections is not None:
            allowed_set = set(tool_args.allowed_collections)

            if not allowed_set:
                return KnowledgeSearchResult(
                    results=[],
                    summary="Knowledge base search is disabled for this agent (no knowledge bases configured).",
                )
            valid_collections = allowed_set & available_names

            if not valid_collections:
                return KnowledgeSearchResult(
                    results=[],
                    summary=f"Error: None of the allowed collections exist. "
                    f"Allowed: {', '.join(sorted(allowed_set))}. "
                    f"Available: {', '.join(sorted(available_names))}",
                )

            collections_to_iterate = [
                c for c in collections_result.collections if c.name in valid_collections
            ]
            logger.info(f"Searching allowed collections: {sorted(valid_collections)}")
        else:
            collections_to_iterate = collections_result.collections
            logger.info("Searching all collections")

        # Build base search config (per-collection overrides happen below)
        base_search_config = {
            "search_type": tool_args.search_type,
            "top_k": tool_args.top_k,
            "min_score": tool_args.min_score,
            "merge_results": True,
            # Retrieval must not *build indexes*: create_index() commits to the
            # LanceDB table, and searching collections concurrently can now race
            # that commit (see collection_manager's CommitConflict note).
            # Indexes are built during ingestion, which is where they belong.
            #
            # This suppresses both index paths, not just FTS: the readonly branch
            # of create_index returns before the dense/vector block and before the
            # FTS block (lancedb_stores.py). Only the sparse path reports it
            # (READONLY_MODE / FTS_INDEX_MISSING) - dense search hardcodes
            # warnings=[] on success, so a skipped dense index build is silent.
            #
            # Not a claim that the search writes nothing: resolving the embedding
            # model still stamps last_accessed_at per collection
            # (collection_manager.mark_collection_accessed). That row-overwrite is
            # pre-existing and guarded by a per-collection lock, so the N
            # concurrent searches this fan-out creates land on N distinct keys.
            "readonly": True,
        }

        if tool_args.embedding_model_id:
            base_search_config["embedding_model_id"] = tool_args.embedding_model_id

        # Search across collections and aggregate results
        all_results = []
        collection_errors: list[str] = []
        collection_warnings: list[str] = list(partial_failure_notes)
        total_searched = 0
        search_timeout_seconds = get_kb_search_timeout_seconds()

        async def _search_one(
            collection_info: Any,
        ) -> tuple[list[Dict[str, Any]], Optional[str], Optional[str], int]:
            """Search one collection off the event loop.

            Returns (results, error, warning, documents_searched); failures are
            returned rather than raised so one collection cannot fail the batch.
            """
            # Every attribute read lives inside the try below, so this really
            # cannot raise into the gather. _failure reads the name lazily, so
            # it stays usable even if that first read is what failed.
            collection_name = "<unknown>"

            def _failure(
                reason: str,
            ) -> tuple[list[Dict[str, Any]], Optional[str], Optional[str], int]:
                return [], f"{collection_name}: {reason}", None, 0

            try:
                collection_name = collection_info.name

                # Per-KB rerank resolution: explicit tool arg wins, otherwise
                # use the collection's bound rerank_model_id; when neither is
                # set, no rerank stage is added for this collection.
                search_config = dict(base_search_config)
                collection_rerank = getattr(collection_info, "rerank_model_id", None)
                effective_rerank = tool_args.rerank_model_id or collection_rerank
                if effective_rerank:
                    search_config["rerank_model_id"] = effective_rerank
                storage_user_id = getattr(collection_info, "storage_user_id", None)

                logger.info(
                    f"Searching collection '{collection_name}' for: {tool_args.query}"
                )

                # run_document_search is a blocking sync pipeline; running it
                # inline would pin the event loop for the whole retrieval.
                # The deadline covers queueing too, not just execution: it starts
                # when this coroutine is scheduled, so a saturated default
                # executor can burn it before run_document_search even starts.
                # And it frees this caller but not the worker - a timed-out
                # to_thread call keeps running in that shared executor.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_document_search,
                        collection=collection_name,
                        query_text=tool_args.query,
                        config=search_config,
                        user_id=storage_user_id
                        if storage_user_id is not None
                        else user_id,
                        is_admin=False
                        if getattr(collection_info, "ownership", "personal") == "team"
                        else is_admin,
                    ),
                    timeout=search_timeout_seconds,
                )

                if result.status not in {"success", "partial_success"}:
                    error_message = result.message or "; ".join(result.warnings)
                    logger.warning(
                        "Search pipeline returned status '%s' for collection '%s': %s",
                        result.status,
                        collection_name,
                        error_message,
                    )
                    return _failure(f"{error_message or 'search failed'}")

                # The pipeline's message on a non-success status is boilerplate
                # ("Hybrid search completed with warnings"), so the warning list
                # has to win or every real diagnostic is masked by it. The
                # readonly notice is self-inflicted by our own readonly=True and
                # fires on every search; FTS_INDEX_MISSING, the warning that
                # reports an actual consequence of it, is kept.
                warning: Optional[str] = None
                warning_message = "; ".join(
                    w
                    for w in result.warnings
                    if not w.startswith(_READONLY_WARNING_PREFIX)
                )
                if warning_message:
                    warning = f"{collection_name}: {warning_message}"

                if not result.results:
                    return [], None, warning, 0

                results = []
                for res in result.results:
                    res_dict = dict(res)
                    res_dict["collection"] = collection_name
                    results.append(res_dict)
                return results, None, warning, collection_info.documents

            except asyncio.TimeoutError:
                logger.warning(
                    "Search of collection '%s' exceeded %ss",
                    collection_name,
                    search_timeout_seconds,
                )
                return _failure(f"search timed out after {search_timeout_seconds}s")
            except Exception as e:
                logger.warning(f"Failed to search collection '{collection_name}': {e}")
                return _failure(str(e))

        # Skip collections with no embeddings before fanning out.
        searchable = []
        for collection_info in collections_to_iterate:
            if collection_info.embeddings == 0:
                logger.debug(
                    f"Skipping collection with no embeddings: {collection_info.name}"
                )
                continue
            searchable.append(collection_info)

        # Search every collection concurrently: total latency is bounded by the
        # slowest collection instead of the sum of all of them. gather preserves
        # input order, so aggregation order is unchanged.
        # ponytail: no concurrency cap. Holds for the bound-KB paths above, where
        # the agent's own configuration is the bound. It does NOT hold for the
        # final fallback branch: with neither `collections` nor
        # `allowed_collections` set, this fans out over every collection visible
        # to the caller, which _list_visible_collections unions across owners -
        # unbounded by anything the agent declared. Add a Semaphore when that
        # branch gets real traffic, or when fan-out width starves the shared
        # default executor that every asyncio.to_thread caller draws from.
        for results, error, warning, documents in await asyncio.gather(
            *(_search_one(collection_info) for collection_info in searchable)
        ):
            if error:
                collection_errors.append(error)
                continue
            if warning:
                collection_warnings.append(warning)
            if results:
                all_results.extend(results)
                total_searched += documents

        if not all_results:
            if collection_errors:
                summary = (
                    "Knowledge base search failed for one or more collections: "
                    + " | ".join(collection_errors)
                )
                if collection_warnings:
                    summary = (
                        summary + "\n\nWarnings: " + " | ".join(collection_warnings)
                    )
                return KnowledgeSearchResult(results=[], summary=summary)
            summary = (
                f"No relevant documents found in any knowledge base. "
                f"Searched {total_searched} documents across "
                f"{len(collections_result.collections)} collections. Query: {tool_args.query}"
            )
            if collection_warnings:
                summary = summary + "\n\nWarnings: " + " | ".join(collection_warnings)
            return KnowledgeSearchResult(results=[], summary=summary)

        # Format results (structured + summary)
        formatted_results, summary = _format_search_results(
            all_results, tool_args.query, total_searched
        )
        if collection_warnings:
            summary = summary + "\n\nWarnings: " + " | ".join(collection_warnings)
        if collection_errors:
            summary = summary + "\n\nErrors: " + " | ".join(collection_errors)

        return KnowledgeSearchResult(results=formatted_results, summary=summary)

    except KnowledgeBaseScopeError:
        raise
    except Exception as e:
        logger.error(f"Knowledge base search failed: {e}", exc_info=True)
        raise RuntimeError(f"Knowledge base search failed: {e}") from e


def _format_search_results(
    results: List[Dict[str, Any]], query: str, total_documents: int
) -> tuple[list[Dict[str, Any]], str]:
    """Format search results for LLM consumption.

    Returns:
        Tuple of (structured_results, summary_string)
    """
    formatted_results = []

    for result in results:
        collection = result.get("collection", "unknown")
        score = result.get("score", 0.0)
        text = result.get("text", "")
        metadata = result.get("metadata") or {}

        # Extract file information from metadata
        source_path = metadata.get("source", "")
        doc_id = metadata.get("doc_id", "")
        chunk_id = metadata.get("chunk_id", "")

        # Try to get document name from source_path
        document_name = ""
        if source_path:
            import os

            document_name = os.path.basename(source_path)

        # Create structured result
        structured_result = {
            "collection": collection,
            "score": score,
            "text": text,
            "document_name": document_name,
            "source_path": source_path,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
        }
        formatted_results.append(structured_result)

    # Create brief summary (token-efficient, no duplicate content)
    summary = f"Found {len(results)} relevant results from {total_documents} documents for query: '{query}'"

    return formatted_results, summary
