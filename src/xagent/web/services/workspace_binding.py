"""Chat workspace binding: the CA sandbox mount set.

Two different concerns share one conceptual workspace root:

- what file tools are allowed to read/write (the *Actor*-logical view: the
  full workspace root under all ``ExecutionScope.workspace_segments``, plus
  the external directory allowlist). ``chat.py`` owns that view and builds
  it itself, through ``chat._build_allowed_external_dirs`` /
  ``WebToolConfig.workspace_config``; :func:`_build_external_allowlist`
  here recomputes the same allowlist purely as folding input, and the two
  are pinned equivalent by test (see
  ``tests/web/test_execution_scope_workspace_web.py``);
- what the sandbox container actually gets bind-mounted (the *CA* view: one
  mount root plus any genuinely separate extra mounts --
  ``ChatWorkspaceBinding.mount_intent``, which ``chat.py`` consumes
  directly when creating/reusing the task's sandbox). This module is that
  view's single construction point.

:func:`build_chat_workspace_binding` folds the CA mount candidates (the
computed mount root plus every allowlist entry) through
``SandboxMountIntent``'s covered/covering/disjoint classification so a
redundant nested mount never becomes a second bind:

- an allowlist entry the mount root already covers (equal to or a
  descendant of it) is dropped -- nothing is lost, the root's bind already
  exposes it;
- an allowlist entry that covers the mount root (a proper ancestor) absorbs
  it: the entry becomes the new root and the narrower original root is
  dropped, for the same reason.

The invariant this exists to enforce: when a scope isolates its external
dirs and narrows the mount to a prefix of ``workspace_segments`` (the
"CA root, Actor subtree" shape -- an org-level container shared by several
per-Actor scopes), the Actor's own subtree is *covered by* the CA mount root
and is dropped rather than surfacing as a second, Actor-specific bind. Two
Actors under the same CA then compute byte-identical mount intents and can
share one container -- keeping the Actor subtree as a separate bind would
make their desired configs diverge and is the root cause (#296) this
projection removes. That is a hard invariant rather than a best effort: an
Actor subtree that cannot fold into the CA root fails the build instead of
becoming an Actor-specific bind.

Containment and authorization compare each candidate's physical identity in the
backend path domain. The final mount intent deliberately retains a lexical
backend spelling for Docker-host translation; ``SandboxPathMapper`` derives the
matching physical guest target separately. Both views name the same directory,
including when an operator configured an ordinary symlink. What an escape from
the physical mount root means still depends on provenance -- see
:class:`_MountCandidate` and :func:`_fold_mount_paths`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Optional, Sequence

from ...config import get_external_upload_dirs, get_uploads_dir
from ...core.execution_scope import ExecutionScope
from ...core.workspace import scoped_user_root
from ...sandbox import (
    SandboxMountEscapeError,
    SandboxMountIntent,
    canonical_sandbox_path,
)
from ..sandbox_manager import backend_mount_path_views

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChatWorkspaceBinding:
    """Result of :func:`build_chat_workspace_binding`.

    ``mount_intent`` is the folded set actually handed to the sandbox
    manager. ``prepare_root`` is deliberately a separate field: the mount
    root computed *before* folding (``scoped_user_root`` at the scope's
    ``effective_mount_segments``), in the same canonical spelling the mount
    intent uses, so both name one directory. Folding can re-root
    ``mount_intent`` onto a covering ancestor already in the allowlist (see
    module docstring), but the directory the task's own files actually live
    in is always ``prepare_root`` -- that is the ``mkdir -p`` target, never
    ``mount_intent.mount_root`` directly, because in the re-rooted case the
    ancestor may already exist while the deeper subtree does not.
    """

    mount_intent: SandboxMountIntent
    prepare_root: str


# Process-level, once-per-kind guard for the known-limitation warnings below.
# Deliberately never cleared: the goal is one log line per limitation *kind*
# per process, enough to point at a producer, not a per-call trace.
_warned_limitation_kinds: set[str] = set()


def _warn_once(kind: str, message: str, *args: object) -> None:
    if kind in _warned_limitation_kinds:
        return
    _warned_limitation_kinds.add(kind)
    logger.warning(message, *args)


@dataclass(frozen=True)
class _MountCandidate:
    """One fold candidate plus the provenance that decides its fold policy.

    ``"scope"`` -- derived from this owner's workspace root at the scope's
    segments. Its containment in the mount root is a *precondition*, not an
    observation: the CA root is chosen so it covers this path, the directory
    tree it names is workspace-controlled, and its spelling varies per Actor.
    A per-Actor survivor in the folded set would break the #296 invariant
    that every Actor under one CA folds to a byte-identical intent, so this
    candidate must fold away (or absorb the root) or the build fails closed
    -- unless the deployment names the same mount point, see
    :func:`_fold_mount_paths`.

    ``"deployment"`` -- an operator-configured external directory
    (``XAGENT_EXTERNAL_UPLOAD_DIRS``). It is an independently declared mount
    in its own right, identical for every Actor in the process, so keeping
    its own bind neither breaks intent equality nor exposes anything the
    deployment did not name explicitly.
    """

    path: str
    origin: Literal["scope", "deployment"]


@dataclass(frozen=True)
class _NormalizedMountCandidate:
    """One candidate's translatable spelling and physical identity."""

    lexical_path: str
    physical_path: str
    origin: Literal["scope", "deployment"]


def _build_external_allowlist(
    owner_id: int, scope: Optional[ExecutionScope]
) -> tuple[_MountCandidate, ...]:
    """Mirror ``chat._build_allowed_external_dirs`` (``only_existing=False``).

    The user's upload dir is scope-narrowed only when
    ``isolate_external_dirs`` is set; deployment-level external upload dirs
    (``XAGENT_EXTERNAL_UPLOAD_DIRS``) are never user-root derived and are
    always included.

    ``path`` entries keep chat's exact spelling and order -- this mirrors an
    Actor-logical allowlist that is pinned equivalent to chat's own by test.
    Backend-domain normalization belongs to :func:`_fold_mount_paths`, which
    is where these values stop being an allowlist and become mount
    candidates; the ``origin`` each one carries is what lets that folding
    tell the two trust levels apart.
    """
    segments = (
        scope.workspace_segments
        if scope is not None and scope.isolate_external_dirs
        else ()
    )
    user_upload_dir = scoped_user_root(get_uploads_dir(), owner_id, segments)
    return (
        _MountCandidate(str(user_upload_dir), "scope"),
        *(_MountCandidate(str(d), "deployment") for d in get_external_upload_dirs()),
    )


def _canonical_mount_path(path: str) -> str:
    """Return one mount path's stable lexical backend spelling.

    Raw configuration may be relative, environment-expanded, ``~``-prefixed,
    or symlinked. The spelling stored in ``SandboxMountIntent`` must retain
    symlinks so ``SandboxPathMapper`` can translate it to the Docker-host
    storage root; physical identity is carried separately while folding.
    """
    views = backend_mount_path_views(path)
    return canonical_sandbox_path(str(views.lexical))


def canonical_workspace_base(owner_id: int, segments: Sequence[str] = ()) -> str:
    """Return one owner's stable lexical workspace root at ``segments``.

    ``TaskWorkspace`` resolves this spelling before file access. Keeping the
    lexical form here preserves the relative path needed for sibling-Docker
    host translation while both views still name the same directory.
    """
    return _canonical_mount_path(
        str(scoped_user_root(get_uploads_dir(), owner_id, tuple(segments)))
    )


def _path_relation(root: str, path: str) -> str:
    """``SandboxMountIntent``'s verdict inside one already-chosen domain."""
    probe = SandboxMountIntent(mount_root=root, extra_mounts=(path,))
    if probe.covered_extras:
        return "covered"
    if probe.covering_extras:
        return "covering"
    return "disjoint"


def _normalize_mount_candidate(candidate: _MountCandidate) -> _NormalizedMountCandidate:
    views = backend_mount_path_views(candidate.path)
    return _NormalizedMountCandidate(
        lexical_path=canonical_sandbox_path(str(views.lexical)),
        physical_path=canonical_sandbox_path(str(views.physical)),
        origin=candidate.origin,
    )


def _dedupe_mount_candidates(
    candidates: Sequence[_NormalizedMountCandidate],
) -> tuple[_NormalizedMountCandidate, ...]:
    """Choose one deterministic lexical spelling per physical directory.

    A deployment spelling wins over a scope-derived spelling because it is
    stable across every Actor sharing the sandbox and is the spelling the
    operator authorized. Ties use the lexical spelling so environment order
    cannot change the desired runtime spec.
    """
    selected: dict[str, _NormalizedMountCandidate] = {}
    for candidate in candidates:
        existing = selected.get(candidate.physical_path)
        if (
            existing is None
            or (candidate.origin == "deployment" and existing.origin == "scope")
            or (
                candidate.origin == existing.origin
                and candidate.lexical_path < existing.lexical_path
            )
        ):
            selected[candidate.physical_path] = candidate
    return tuple(selected[physical_path] for physical_path in sorted(selected))


def _fold_mount_paths(
    mount_root: str, candidates: Sequence[_MountCandidate]
) -> tuple[str, tuple[str, ...]]:
    """Collapse a mount root and allowlist candidates into one physical set.

    Each path keeps two views: folding and authorization use its physical
    identity, while the returned ``SandboxMountIntent`` retains one stable
    lexical backend spelling so sibling-mode host translation remains valid.

    Then, repeatedly, each candidate is classified against the current root
    by :func:`_path_relation`:

    - a candidate the root already covers (equal to it or a descendant) is
      redundant and dropped;
    - a candidate that covers the root (a proper ancestor) absorbs it: the
      candidate is promoted to root and the old root is dropped (it is now
      implied by the promoted one). Covering candidates are always a
      lexical chain -- all are prefixes of the same root, hence prefixes of
      each other -- so promoting the widest one (shortest path) is
      unambiguous and a single promotion reclassifies everything else
      against the new root.
    - anything left over is disjoint, and provenance decides its fate: a
      candidate whose identity the deployment named keeps its own mount,
      anything else fails the build closed. A deployment candidate is in
      that set by construction; a scope-derived one is in it exactly when an
      operator entry names the same mount point, under any spelling.

    A scope-derived candidate is normally the root, its descendant, or its
    ancestor because ``ExecutionScope`` enforces the corresponding segment
    prefix. A symlink may move its physical identity outside that tree. Such
    a path is rejected unless the deployment independently named that same
    physical directory; a symlink alias counts because aliases and their
    targets intentionally share one mount identity.

    Returns the final lexical root and surviving lexical candidates. Physical
    aliases are deduplicated before folding, preferring deployment spellings.
    """
    root_views = backend_mount_path_views(mount_root)
    root = _NormalizedMountCandidate(
        lexical_path=canonical_sandbox_path(str(root_views.lexical)),
        physical_path=canonical_sandbox_path(str(root_views.physical)),
        origin="scope",
    )
    normalized = tuple(_normalize_mount_candidate(c) for c in candidates)
    deployment_identities = {
        c.physical_path for c in normalized if c.origin == "deployment"
    }
    remaining = _dedupe_mount_candidates(normalized)

    while True:
        verdicts = [
            (c, _path_relation(root.physical_path, c.physical_path)) for c in remaining
        ]
        covering = [c for c, verdict in verdicts if verdict == "covering"]
        if not covering:
            # Judged only once the root has stopped moving: an earlier
            # promotion can widen the root enough to cover a candidate that
            # was disjoint from the narrower one.
            for candidate, verdict in verdicts:
                if (
                    verdict == "disjoint"
                    and candidate.physical_path not in deployment_identities
                ):
                    raise SandboxMountEscapeError(
                        f"Workspace path {candidate.lexical_path!r} (resolving to "
                        f"{candidate.physical_path!r}) is neither inside nor "
                        "a parent of sandbox mount root "
                        f"{root.lexical_path!r} (resolving to "
                        f"{root.physical_path!r}); refusing to bind it as a "
                        "separate mount"
                    )
            return root.lexical_path, tuple(
                c.lexical_path for c, verdict in verdicts if verdict != "covered"
            )
        new_root = min(
            covering,
            key=lambda c: (len(c.physical_path), c.physical_path, c.lexical_path),
        )
        remaining = tuple(c for c in remaining if c is not new_root)
        root = new_root


def build_chat_workspace_binding(
    owner_id: int, scope: Optional[ExecutionScope]
) -> ChatWorkspaceBinding:
    """Build the CA mount intent for a task's sandbox.

    Called from ``chat.py`` (task creation and agent reconstruction alike)
    to build ``mount_intent`` for the task's sandbox lease provider; the
    Actor-logical allowlist stays chat-owned, see the module docstring.

    Raises:
        SandboxMountEscapeError: This owner's workspace path escapes the
            mount root that has to cover it (see
            :func:`_fold_mount_paths`). A ``SandboxContractError``, so
            chat's task path fails the task instead of downgrading it to
            unsandboxed local execution.
    """
    mount_segments = scope.effective_mount_segments if scope is not None else ()

    external_allowlist = _build_external_allowlist(owner_id, scope)

    # The mkdir target and the root folding starts from are one path, and it
    # is in the same domain as the Actor-logical base chat hands the tools
    # (see :func:`canonical_workspace_base`).
    prepare_root = canonical_workspace_base(owner_id, mount_segments)
    folded_root, folded_extras = _fold_mount_paths(prepare_root, external_allowlist)

    # Known limitation (pending PR-2 scope authority): a suffix-less scope
    # shares the unscoped sandbox lifecycle key (``user:{owner}``) purely
    # from ``sandbox_key_suffix`` being absent, but ``isolate_external_dirs``
    # still narrows this call's own mount root away from the unscoped path.
    # The same lifecycle key then sees a different desired mount depending on
    # which scope happened to build it -- a config-equivalence hazard, not
    # something this projection can resolve on its own.
    if (
        scope is not None
        and scope.sandbox_key_suffix is None
        and scope.isolate_external_dirs
        and scope.workspace_segments
    ):
        _warn_once(
            "suffixless_isolate",
            "ExecutionScope has isolate_external_dirs=True and "
            "workspace_segments=%r but no sandbox_key_suffix: it shares the "
            "unscoped sandbox lifecycle key for owner %s while this call's "
            "own mount root (%s) diverges from the unscoped path. Known "
            "limitation pending PR-2 scope authority -- give this scope a "
            "sandbox_key_suffix.",
            scope.workspace_segments,
            owner_id,
            prepare_root,
        )

    # Known limitation (pending PR-2 scope authority): an explicit
    # sandbox_mount_segments prefix declares a narrower mount, but with
    # isolate_external_dirs left False the allowlist still carries the
    # unscoped user root, which always covers (and folds away) the
    # narrower mount root -- the requested narrowing has no physical effect.
    if (
        scope is not None
        and scope.sandbox_mount_segments is not None
        and not scope.isolate_external_dirs
    ):
        _warn_once(
            "mount_prefix_without_isolate",
            "ExecutionScope sets sandbox_mount_segments=%r without "
            "isolate_external_dirs: the narrower mount root is folded away "
            "into the unscoped user-root allowlist entry, so the requested "
            "mount narrowing has no effect. Known limitation pending PR-2 "
            "scope authority -- set isolate_external_dirs=True to keep the "
            "narrower mount.",
            scope.sandbox_mount_segments,
        )

    mount_intent = SandboxMountIntent(
        mount_root=folded_root, extra_mounts=folded_extras
    )
    return ChatWorkspaceBinding(mount_intent=mount_intent, prepare_root=prepare_root)
