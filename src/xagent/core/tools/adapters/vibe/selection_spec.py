"""Declarative tool-selection spec for :class:`ToolFactory`.

Background:
    Before this spec, :func:`ToolFactory.create_all_tools` built the
    full set of registered tools and filtered them by name afterwards
    via ``config.get_allowed_tools()``. Callers that only needed a
    category-level filter (e.g. the WS chat path) had to pre-build the
    entire ~52 tool list just to read each tool's metadata.category
    and assemble a name list -- a redundant build that dominated
    per-task setup time (see issue #427).

    ``ToolSelectionSpec`` is a sealed ABC with three concrete
    subclasses (``_SpecAll`` / ``_SpecNone`` / ``_SpecByCategories``).
    Modes are explicit through subclass identity and
    :meth:`is_all` / :meth:`is_none` / :meth:`is_by_categories`
    predicates -- not the older "None vs frozenset() vs frozenset({...})"
    implicit signal that conflated the three states and caused
    legacy ``Agent.tool_categories=[]`` to be misread as "zero tools".

    Production callers MUST construct via
    :meth:`ToolSelectionSpec.from_raw`, the single normalizer over
    raw ORM / dict / SDK fields. Direct subclass instantiation is
    used by tests; production paths that bypass ``from_raw`` are
    flagged by a grep test.

Mode completeness:
    Each abstract method (``is_*`` / ``includes_*`` /
    ``compute_allowed_names``) must be implemented by every subclass.
    Missing an implementation is both a mypy error and a runtime
    ``TypeError`` at instantiation time. Adding a new ``includes_*``
    creator-dispatch method on the base forces every subclass to
    update -- no grep test required to police mode dispatch.

Backward compat:
    All three subclasses expose ``categories`` /``mcp_servers`` /
    ``published_agent_ids`` fields (the original spec shape).
    ``_SpecAll`` has them at None / ``_SpecNone`` at None plus empty
    ``categories``, so existing callsites that read ``spec.categories``
    directly keep working. New code should prefer the typed dispatch
    (``spec.is_by_categories()`` etc.).

This module deliberately has no dependencies on the rest of the
codebase so the spec can be imported by both the factory and the
individual tool creators without circular-import risk. The
``compute_allowed_names`` helper uses duck typing for tool metadata
access; no Tool type import required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional, Set

logger = logging.getLogger(__name__)


def normalize_mcp_server_name(name: str) -> str:
    """SSOT for normalizing an MCP / Custom-API server identity.

    Strips surrounding whitespace, folds spaces / hyphens to underscore,
    and case-folds, yielding a stable key for a server. Every site that
    derives a server identity must use this single transform:

      - :meth:`ToolSelectionSpec.from_raw` (parse ``mcp:<server>`` selector),
      - the per-server config filter in ``mcp_tools.create_mcp_tools``
        (server-config ``name``),
      - tool GENERATION (``mcp_adapter`` / ``api_tool_adapter``), which
        stamps the result onto ``metadata.source_server``.

    Because both the selector and the tool's ``source_server`` pass through
    here, server-scoped selection (``compute_allowed_names``) is a plain
    equality check on normalized keys -- a ``"mcp: Gmail"`` / ``"mcp:gmail"``
    selector reliably matches a ``"Gmail"`` server regardless of whitespace,
    case, or hyphen/space. The LLM-visible tool ``name`` keeps its original
    casing; only the structured ``source_server`` key is normalized.
    """
    return name.strip().replace(" ", "_").replace("-", "_").lower()


class ToolSelectionSpec(ABC):
    """Sealed type for tool selection.

    Three concrete subclasses, accessed through :meth:`from_raw`:
      - ``_SpecAll`` — legacy "未配置" / no restriction; build every
        default tool. Factory does not filter by name.
      - ``_SpecNone`` — explicit "zero tools"; factory returns ``[]``.
      - ``_SpecByCategories`` — filter by category, with optional
        ID-level scopes (mcp_servers / published_agent_ids) and
        ``name_allowlist`` (a pure name-level filter; workforce worker
        tool injection is one source).

    Mode completeness is enforced by ``@abstractmethod``: each
    subclass must implement every predicate / dispatch method.
    Missing one fails at instantiation time, not silently.

    Direct ``ToolSelectionSpec(...)`` construction is supported for
    backward compatibility: ``__new__`` inspects ``categories`` and
    dispatches to the matching subclass. Production code should
    prefer :meth:`from_raw` for the explicit normalizer contract;
    direct construction is mostly for tests + legacy callers.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "ToolSelectionSpec":
        """Dispatch ``ToolSelectionSpec(...)`` to the right subclass.

        Backward-compat shim for callers / tests that construct the
        old single-dataclass shape directly. New production code
        should prefer :meth:`from_raw`.

        Dispatch rules (match the legacy dataclass semantics):
          - ``categories=None`` (default) → ``_SpecAll``
          - ``categories=frozenset()``    → ``_SpecNone``
          - ``categories=frozenset({...})`` non-empty → ``_SpecByCategories``

        Subclass direct construction (``_SpecAll()``, ``_SpecNone()``,
        ``_SpecByCategories(...)``) bypasses this dispatch.
        """
        if cls is ToolSelectionSpec:
            categories = kwargs.get("categories")
            if categories is None:
                return _SpecAll.__new__(_SpecAll)
            if isinstance(categories, frozenset) and len(categories) == 0:
                return _SpecNone.__new__(_SpecNone)
            return _SpecByCategories.__new__(_SpecByCategories)
        return super().__new__(cls)

    # ── Mode predicates ────────────────────────────────────────────
    # Three mutually exclusive modes; exactly one is_*() returns True.

    @abstractmethod
    def is_all(self) -> bool:
        """Whether this is the ALL mode (build every default tool)."""

    @abstractmethod
    def is_none(self) -> bool:
        """Whether this is the NONE mode (factory returns ``[]``)."""

    @abstractmethod
    def is_by_categories(self) -> bool:
        """Whether this is the BY_CATEGORIES mode (filtered build)."""

    # ── Creator dispatch ──────────────────────────────────────────
    # ToolRegistry / individual creators consult these to decide
    # whether their work (DB queries / MCP init / etc) should run.

    @abstractmethod
    def includes_mcp(self) -> bool:
        """Whether the MCP creator should run."""

    @abstractmethod
    def includes_custom_api(self) -> bool:
        """Whether the Custom API creator should run.

        In BY_CATEGORIES mode Custom API tools are connector-scoped:
        only an explicit ``mcp:<server>`` selector should load the matching
        Custom API wrapper. Plain ``"other"`` must not bulk-enable every
        Custom API the user has configured.
        """

    @abstractmethod
    def includes_published_agent(self) -> bool:
        """Whether the Published Agent delegation creators should run."""

    @abstractmethod
    def scoped_mcp_servers(self) -> Optional[frozenset[str]]:
        """Pre-build MCP server restriction for the MCP creator.

        The MCP creator initializes server sessions (network I/O) before
        any tool name exists, so it filters at the config level. This
        method is the single source of that restriction, kept consistent
        with the parent/child rule ``compute_allowed_names`` applies
        post-build:

          - ``frozenset()`` — MCP is not selected; do not initialize any
            MCP server.
          - ``None`` — MCP is selected without a server restriction
            (the plain ``"mcp"`` parent is present, or ALL mode); initialize
            every MCP server.
          - non-empty — initialize only these normalized server keys.
        """

    # ── Final name-level filter ───────────────────────────────────

    @abstractmethod
    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        """Resolve the final allowed-tool-names set for this spec.

        Returns:
            ``None``       — caller keeps every tool in ``all_tools``
                             (ALL mode).
            ``frozenset()`` — caller returns ``[]`` (NONE mode).
            non-empty set  — caller filters ``all_tools`` to names
                             in the set (BY_CATEGORIES mode, plus
                             :attr:`name_allowlist` injection).

        The frozenset() vs None distinction is load-bearing:
        ``ToolFactory.create_all_tools`` reads this method and
        filters / short-circuits based on the three return types.
        """

    # ── Single normalizer ─────────────────────────────────────────

    @classmethod
    def from_raw(
        cls,
        *,
        tool_categories: Optional[List[str]] = None,
        published_agent_ids: Optional[List[int]] = None,
        name_allowlist: Optional[Set[str]] = None,
        explicit_none: bool = False,
        extras_only_when_unconfigured: bool = False,
    ) -> "ToolSelectionSpec":
        """Build a spec from raw ORM / dict / SDK fields.

        This is the **only** production entry point. Direct
        subclass construction is for tests; a grep test pins
        production code to use ``from_raw``.

        Empty / unset input semantics:
          - ``tool_categories=None`` → ``_SpecAll``
            (legacy "未配置" — build every default tool).
          - ``tool_categories=[]`` → ``_SpecNone``
            (caller explicitly selected zero tools — no tools injected).
          - ``explicit_none=True`` → ``_SpecNone`` regardless of
            ``tool_categories`` (reserved for future "zero tools"
            product UI).
          - ``extras_only_when_unconfigured=True`` with unset / empty
            categories → workforce manager runtime: ``published_agent_ids``
            declares the published-agent dispatch and ``name_allowlist``
            filters to the worker tool names (or ``_SpecNone`` when there
            is no ``published_agent_ids``). Lets an unconfigured manager
            delegate only to its workers without inheriting the full set.
          - Otherwise → ``_SpecByCategories`` after dropping non-assignable
            entries such as ``"other"``; if nothing assignable remains, NONE.

        ``name_allowlist`` is a pure name-level filter (only meaningful in
        BY_CATEGORIES; ALL already includes everything, NONE rejects
        everything). It does NOT trigger any creator -- dispatch is
        declared by ``categories`` / ``published_agent_ids``.
        """
        if explicit_none:
            return _SpecNone()

        names = frozenset(name_allowlist or set())

        if tool_categories is None or len(tool_categories) == 0:
            if extras_only_when_unconfigured:
                # Workforce manager runtime: ``published_agent_ids``
                # declares the dispatch (run the published-agent creator,
                # scoped to the worker agents); ``name_allowlist`` narrows
                # its output to the worker tool names. They are orthogonal.
                # ``name_allowlist`` alone is a pure filter -- with no
                # dispatch there is no creator output to filter, so that
                # collapses to NONE.
                pids = (
                    frozenset(published_agent_ids)
                    if published_agent_ids is not None
                    else None
                )
                if not pids:
                    return _SpecNone()
                return _SpecByCategories(
                    categories=frozenset(),
                    published_agent_ids=pids,
                    name_allowlist=names,
                )
            # ``None`` means truly unconfigured (legacy "all tools" behaviour).
            # An explicit empty list ``[]`` means the caller intentionally
            # selected zero tools — return NONE so no tools are injected.
            if tool_categories is not None:
                return _SpecNone()
            return _SpecAll()

        # ``tool_categories`` mixes two orthogonal shapes:
        #   - plain category names (``"basic"``, ``"file"``, ``"mcp"``);
        #     ``"other"`` is an internal fallback, not an assignable category
        #   - ``"mcp:<server>"`` — a specific MCP server (or the
        #     Custom-API tool fronting it)
        #
        # Keep them in separate fields, not one overloaded set:
        #   - plain entries  -> ``categories``
        #   - ``mcp:<server>`` -> ``mcp_servers`` ONLY
        #
        # Whether the MCP / Custom-API creators run is derived from
        # ``includes_mcp()`` / ``includes_custom_api()`` (which read
        # ``mcp_servers`` too); ``compute_allowed_names`` matches
        # ``mcp_servers`` against each tool's structured
        # ``metadata.source_server``. No support categories are injected and
        # no raw ``mcp:<server>`` string leaks into ``categories``.
        plain_cats: Set[str] = set()
        derived_mcp_servers: Set[str] = set()
        for entry in tool_categories:
            if isinstance(entry, str) and entry.startswith("mcp:"):
                server_name = normalize_mcp_server_name(entry.split(":", 1)[1])
                derived_mcp_servers.add(server_name)
            elif entry == "other":
                # Legacy-only entry, no longer assignable (see module
                # docstring). No built-in tool carries this category, so
                # dropping it just prunes stale data -- except a legacy
                # agent persisted with exactly ["other"] (no mcp_servers),
                # which loses its only tool access. Warn so this is
                # visible in logs; from_raw() has no agent id, so
                # per-agent traceability comes from the backfill
                # migration's logging instead.
                logger.warning(
                    "Dropping non-assignable 'other' tool category from "
                    "tool_categories=%r; if 'other' was the only entry, "
                    "this agent now gets zero tools instead of its "
                    "previous (leaky) Custom API access",
                    tool_categories,
                )
                continue
            else:
                plain_cats.add(entry)

        final_mcp_servers = (
            frozenset(derived_mcp_servers) if derived_mcp_servers else None
        )
        final_published_agent_ids = (
            frozenset(published_agent_ids) if published_agent_ids is not None else None
        )

        if not plain_cats and not final_mcp_servers and not final_published_agent_ids:
            return _SpecNone()

        return _SpecByCategories(
            categories=frozenset(plain_cats),
            mcp_servers=final_mcp_servers,
            published_agent_ids=final_published_agent_ids,
            name_allowlist=names,
        )

    # ── Backward-compat helper (kept from the original spec) ─────

    def includes_category(self, cat: str) -> bool:
        """Whether the given category passes the spec.

        ``ALL`` admits every category; ``NONE`` admits none; and
        ``BY_CATEGORIES`` admits only members of :attr:`categories`.
        Existing callers in ``factory.py`` registry-skip and
        creator-internal short-circuits keep using this.
        """
        if self.is_all():
            return True
        if self.is_none():
            return False
        # ``categories`` exists on _SpecByCategories; mypy follows it
        # through ``is_by_categories()`` narrowing in modern setups,
        # but the duck-typed attribute access is also safe here.
        categories: frozenset[str] = getattr(self, "categories", frozenset())
        return cat in categories


def should_load_mcp_server_configs(
    tool_selection_spec: Optional[ToolSelectionSpec],
) -> bool:
    """Whether web-task setup should load MCP server configs.

    ``includes_mcp()`` is broader creator/filter compatibility semantics:
    ALL mode returns true so legacy unrestricted factory paths keep admitting
    MCP tools when configs are already supplied. Web task setup has a narrower
    product meaning: pay the MCP config scan/session-init cost only when the
    user explicitly selected the MCP domain via categories.
    """
    if tool_selection_spec is None:
        return False
    if not tool_selection_spec.is_by_categories():
        return False
    return bool(tool_selection_spec.includes_mcp())


# ─────────────────────────────────────────────────────────────────
# Concrete subclasses. Production code MUST go through ``from_raw``.
# The leading underscore signals "internal" — direct construction is
# legal but flagged by a grep test in non-test paths.
# ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _SpecAll(ToolSelectionSpec):
    """ALL mode — legacy "未配置" / no restriction.

    Exposes ``categories`` / ``mcp_servers`` / ``published_agent_ids``
    at ``None`` for backward compat with callsites that read those
    attributes directly (e.g. the registry-level skip in
    ``factory.py:ToolRegistry``).
    """

    # Backward-compat fields (kept None to preserve existing
    # ``spec.categories is None`` truthiness in factory.py).
    categories: Optional[frozenset[str]] = None
    mcp_servers: Optional[frozenset[str]] = None
    published_agent_ids: Optional[frozenset[int]] = None

    def is_all(self) -> bool:
        return True

    def is_none(self) -> bool:
        return False

    def is_by_categories(self) -> bool:
        return False

    def includes_mcp(self) -> bool:
        # Backward-compat: legacy callers could write
        # ToolSelectionSpec(mcp_servers=frozenset()) to express
        # "no MCP tools" even without a categories filter. Honor
        # that here (the abstract method is still enforced -- this
        # is just the ALL subclass's concrete implementation).
        if self.mcp_servers is not None and len(self.mcp_servers) == 0:
            return False
        return True

    def includes_custom_api(self) -> bool:
        return True

    def includes_published_agent(self) -> bool:
        if self.published_agent_ids is not None and len(self.published_agent_ids) == 0:
            return False
        return True

    def scoped_mcp_servers(self) -> Optional[frozenset[str]]:
        # ALL mode: MCP selected without restriction -> initialize every
        # server. (``includes_mcp()`` already honors the legacy
        # explicit-empty ``mcp_servers`` exclude before the creator runs.)
        return None

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        # None signals "no name-level filter" -- factory keeps
        # every tool returned by the registry.
        return None


@dataclass(frozen=True)
class _SpecNone(ToolSelectionSpec):
    """NONE mode — explicit "zero tools" (no UI entry today, reserved).

    ``categories`` is an explicit empty frozenset so existing
    callsites that test ``spec.categories is not None`` see "set"
    and walk the "no intersection" branch (factory.py:ToolRegistry
    then skips every creator). This mirrors the original
    ``categories=frozenset()`` "explicit exclusion" semantics
    documented on the old dataclass.
    """

    categories: Optional[frozenset[str]] = field(default_factory=lambda: frozenset())
    mcp_servers: Optional[frozenset[str]] = None
    published_agent_ids: Optional[frozenset[int]] = None

    def is_all(self) -> bool:
        return False

    def is_none(self) -> bool:
        return True

    def is_by_categories(self) -> bool:
        return False

    def includes_mcp(self) -> bool:
        return False

    def includes_custom_api(self) -> bool:
        return False

    def includes_published_agent(self) -> bool:
        return False

    def scoped_mcp_servers(self) -> Optional[frozenset[str]]:
        # NONE mode: MCP not selected -> initialize no servers.
        return frozenset()

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        # Empty frozenset signals "filter to []" -- factory drops
        # every tool returned by the registry. Distinct from
        # ``None`` (ALL mode, keep everything).
        return frozenset()


@dataclass(frozen=True)
class _SpecByCategories(ToolSelectionSpec):
    """BY_CATEGORIES mode — filtered build.

    ``categories`` is normally non-empty. The one valid empty-category
    state is workforce manager injection: no ordinary categories, but
    explicit ``name_allowlist`` worker-agent tools.
    """

    categories: frozenset[str] = field(default_factory=frozenset)
    mcp_servers: Optional[frozenset[str]] = None
    published_agent_ids: Optional[frozenset[int]] = None
    # Workforce worker tool name injection. Only meaningful in
    # BY_CATEGORIES mode (in ALL the full set already includes
    # them; in NONE everything is rejected).
    # Extra tools admitted by exact name, unioned with the category
    # matches in ``compute_allowed_names``. A pure name-level filter fed
    # via ``from_raw(name_allowlist=...)`` (workforce passes its worker
    # tool names here). Does NOT trigger any creator -- dispatch is
    # declared by ``categories`` / ``published_agent_ids``. Only
    # meaningful in BY_CATEGORIES mode (ALL already includes everything;
    # NONE rejects everything).
    name_allowlist: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        # A by-categories spec must select *something*: a plain category,
        # a scoped MCP server (``mcp:<server>`` -> ``mcp_servers``), or an
        # explicit name in the allow-list. All three empty means "select
        # nothing", which should be expressed as _SpecNone / _SpecAll via
        # from_raw instead.
        # Only DISPATCH dimensions count as "selecting something":
        # categories, mcp_servers, published_agent_ids. ``name_allowlist``
        # is a pure filter (applied after creators run) -- a spec with only
        # name_allowlist and no dispatch produces nothing, which is NONE,
        # not a valid BY_CATEGORIES. So name_allowlist is excluded here.
        if (
            not self.categories
            and not self.mcp_servers
            and not self.published_agent_ids
        ):
            raise ValueError(
                "_SpecByCategories requires a non-empty selection in at "
                "least one DISPATCH dimension (categories, mcp_servers, or "
                "published_agent_ids). name_allowlist is a filter, not a "
                "selection. Use ToolSelectionSpec.from_raw() with empty / "
                "None categories to get _SpecAll, or explicit_none=True for "
                "_SpecNone."
            )

    def is_all(self) -> bool:
        return False

    def is_none(self) -> bool:
        return False

    def is_by_categories(self) -> bool:
        return True

    def includes_mcp(self) -> bool:
        # Explicit empty server set means "no MCP" (legacy
        # explicit-exclude shape). Otherwise the MCP creator runs when
        # the plain "mcp" category is selected (all MCP) or a specific
        # server was scoped via mcp:<server> (-> mcp_servers).
        if self.mcp_servers is not None and len(self.mcp_servers) == 0:
            return False
        return "mcp" in self.categories or bool(self.mcp_servers)

    def includes_custom_api(self) -> bool:
        # Custom API tools are exposed through connector scope only. A plain
        # "other" category used to bulk-load every Custom API for the user,
        # which let agents call APIs unrelated to the configured connector.
        # Filter happens in the creator via ``config.get_custom_api_configs``
        # and ``compute_allowed_names``'s structured ``source_server`` match
        # -- there is no spec-level custom_api id list (ids can't be mapped to
        # tool names in the spec layer).
        return bool(self.mcp_servers)

    def includes_published_agent(self) -> bool:
        # Pure dispatch decision: whether the published-agent creator runs
        # is declared by the "agent" category or by published_agent_ids --
        # NOT by name_allowlist. name_allowlist is a name-level filter
        # (applied after creators run), so letting it trigger this creator
        # would conflate filter with dispatch (issue #539). An explicit
        # empty published_agent_ids means "no delegation".
        if self.published_agent_ids is not None and len(self.published_agent_ids) == 0:
            return False
        return "agent" in self.categories or bool(self.published_agent_ids)

    def scoped_mcp_servers(self) -> Optional[frozenset[str]]:
        # Parent/child rule, identical to what compute_allowed_names applies
        # post-build: the plain "mcp" parent admits every server, so it means
        # "no restriction" (None). A server-only selection restricts to its
        # set. No MCP selected -> empty (initialize nothing). Mirrors
        # includes_mcp(): when that is False this returns frozenset().
        if "mcp" in self.categories:
            return None
        return self.mcp_servers or frozenset()

    def compute_allowed_names(self, all_tools: List[Any]) -> Optional[frozenset[str]]:
        """Filter ``all_tools`` by ``categories`` + ``mcp_servers``,
        then union ``name_allowlist``.

        Reads the orthogonal policy fields directly (no ``_user_picked``
        reconstruction):

          - a tool whose category ∈ ``categories`` is admitted (plain
            ``"mcp"`` admits all MCP tools, etc.; DB-backed Custom API
            tools are loaded only for scoped ``mcp:<server>`` connectors);
          - otherwise a tool whose ``metadata.source_server`` matches a
            scoped server in ``mcp_servers`` is admitted. ``source_server``
            is the normalized originating-server identity set once at
            generation (MCP adapter + Custom-API wrapper), so this is a
            structured equality match -- the spec never re-parses the tool
            name (``mcp_<server>_*`` / ``api_<server>_call``). A scoped
            ``mcp:<server>`` therefore admits both the server's MCP tools
            and its ``api_<server>_call`` wrapper, since both carry the same
            ``source_server``;
          - finally ``name_allowlist`` names are unioned in.

        Duck-typed access to ``tool.metadata`` keeps this module free of any
        Tool / AbstractBaseTool import.
        """
        norm_servers = {
            normalize_mcp_server_name(s) for s in (self.mcp_servers or frozenset())
        }
        names: Set[str] = set()
        for tool in all_tools:
            if not (hasattr(tool, "metadata") and hasattr(tool.metadata, "category")):
                continue
            tool_name = getattr(tool, "name", None)
            if not isinstance(tool_name, str):
                continue
            category = str(tool.metadata.category.value)

            # Plain category admit (categories holds only plain names).
            if category in self.categories:
                names.add(tool_name)
                continue

            # Server-scoped admit: the tool's structured originating-server
            # identity (set + normalized once at generation) equals a scoped
            # server. Covers MCP tools and their Custom-API wrapper uniformly,
            # with no tool-name re-parsing / case / strip / startswith-vs-==.
            # ``src`` is truthy-checked so a tool with no origin (None) or an
            # empty/whitespace-only server identity never matches a scope.
            src = getattr(tool.metadata, "source_server", None)
            if norm_servers and src and src in norm_servers:
                names.add(tool_name)
                continue

        # Union the exact-name allow-list (workforce injection +
        # generic name_allowlist; ``from_raw`` zeroes it for ALL / NONE).
        return frozenset(names | self.name_allowlist)
