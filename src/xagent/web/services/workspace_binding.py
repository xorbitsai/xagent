"""Chat workspace binding: the CA-physical sandbox mount intent.

Two different concerns share one conceptual workspace root:

- what file tools are allowed to read/write (the *Actor*-logical view: the
  full workspace root under all ``ExecutionScope.workspace_segments``, plus
  the external directory allowlist). ``chat.py`` owns that view and builds
  it itself, through ``chat._build_allowed_external_dirs`` /
  ``WebToolConfig.workspace_config``; :func:`_build_external_allowlist`
  here recomputes the same allowlist purely as folding input, and the two
  are pinned equivalent by test (see
  ``tests/web/test_execution_scope_workspace_web.py``);
- what the sandbox container actually gets bind-mounted (the *CA*-physical
  view: one mount root plus any genuinely separate extra mounts --
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

Because that classification is lexical, every candidate is absolutized,
canonicalized and symlink-resolved in the backend path domain first, and
what a disagreement between the two views means depends on where the
candidate came from -- its :class:`_MountCandidate` provenance, see
:func:`_fold_mount_paths`.
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
from ..sandbox_manager import absolute_backend_mount_path, resolve_backend_mount_path

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
    """One raw mount path in the identity domain the fold decides in.

    Absolutizes in the backend domain (raw configuration may be relative or
    ``~``-prefixed) and then canonicalizes, because the spelling that
    survives folding does not reach the backend as typed:
    ``SandboxMountIntent`` canonicalizes it. Folding on the pre-canonical
    spelling would therefore decide coverage, symlink resolution and
    authorization for a path other than the one actually mounted, and let
    two spellings of one mount point produce two different intents.
    """
    return canonical_sandbox_path(str(absolute_backend_mount_path(path)))


def canonical_workspace_base(owner_id: int, segments: Sequence[str] = ()) -> str:
    """One owner's workspace root at ``segments``, in the mount identity domain.

    The same directory has two consumers with two different normalizations:
    the sandbox binds it (canonicalized by ``SandboxMountIntent``, a lexical
    domain, because desired-vs-observed config comparison has to be lexical),
    while the file tools open it through ``TaskWorkspace``, which ``resolve()``s
    what it is handed (a physical domain, because files have to be found).

    ``config.get_uploads_dir`` rejects an uploads *root* whose two readings
    land in different directories, so for the root itself the remaining
    difference is the spelling rather than the destination. That does not
    extend to the segments composed onto it: a symlink planted inside the
    uploads tree still resolves elsewhere, and only the fold's resolved-view
    veto (:func:`_fold_relation`) speaks to that.

    The spelling matters on its own: the desired spec is compared byte-wise
    against what the backend reports, so a base dir carrying ``..`` or ``//``
    makes one workspace look like two configurations depending on which
    producer built it. Every producer of a backend workspace path composes it
    here so they all emit the reported form.
    """
    return _canonical_mount_path(
        str(scoped_user_root(get_uploads_dir(), owner_id, tuple(segments)))
    )


def _lexical_relation(root: str, path: str) -> str:
    """``SandboxMountIntent``'s verdict for one path against one root."""
    probe = SandboxMountIntent(mount_root=root, extra_mounts=(path,))
    if probe.covered_extras:
        return "covered"
    if probe.covering_extras:
        return "covering"
    return "disjoint"


def _fold_relation(root: str, resolved_root: str, path: str, resolved_path: str) -> str:
    """Fold verdict for one candidate: lexical, vetoed by the resolved view.

    ``SandboxMountIntent`` classifies lexically, which is blind to symlinks
    (its own docstring says so and requires callers to resolve first). Both
    folding directions are only sound when the two views agree, because
    each of them keeps the *unresolved* spelling as the surviving mount:

    - dropping a covered candidate assumes the root's bind already exposes
      it at its own path. A symlink lexically under the root but resolving
      outside it is exposed by nothing once the root is bind-mounted, so
      dropping it silently loses the mount;
    - promoting a covering candidate assumes mounting it still exposes the
      old root at the old root's path. A symlink that only *resolves* to an
      ancestor does not contain the old root at all.

    So a disagreement demotes the candidate to disjoint -- the verdict that
    grants a separate bind. What that means is provenance-dependent, and
    :func:`_fold_mount_paths` owns the decision: for a mount point the
    deployment named a separate bind is exactly the unfolded behavior and
    loses nothing, while for any other scope-derived one it is a rejected
    build.
    """
    lexical = _lexical_relation(root, path)
    if lexical != _lexical_relation(resolved_root, resolved_path):
        return "disjoint"
    return lexical


def _fold_mount_paths(
    mount_root: str, candidates: Sequence[_MountCandidate]
) -> tuple[str, tuple[str, ...]]:
    """Collapse a mount root and allowlist candidates into one physical set.

    Root and candidate paths first enter the identity domain the whole fold
    decides in (:func:`_canonical_mount_path`), which is
    ``SandboxMountIntent``'s own mount-path identity. Every step below --
    coverage, symlink resolution, authorization, promotion, the returned
    spelling -- therefore judges one mount point once, however it was typed.

    Then, repeatedly, each candidate is classified against the current root
    by :func:`_fold_relation`:

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

    A scope-derived candidate is always lexically the root, its descendant
    or its ancestor -- ``scoped_user_root`` composes both from the same user
    root, and ``ExecutionScope`` enforces that ``sandbox_mount_segments`` is
    a prefix of ``workspace_segments`` -- so the only way it reaches a
    disjoint verdict is a symlink inside the workspace tree that moves it
    out of the root it is required to share. Granting that its own bind
    would hand the backend a writable mount of whatever the symlink points
    at and give this Actor an intent no sibling Actor under the same CA
    produces, so it raises ``SandboxMountEscapeError`` instead. The
    deployment-named exception is why the identity domain matters: an
    operator who names a mount point takes responsibility for it whatever a
    scope also derives, but only for the mount point actually named -- a
    symlink *alias* of it is a different bind point and does not authorize
    its target. Canonicalization is lexical and keeps that distinction: it
    folds redundant spellings of one path together, it does not follow the
    symlink that makes an alias a path of its own.

    Returns the final root and the surviving candidates in candidate order,
    each canonical and never symlink-resolved -- the spelling the guest
    mount point keeps. ``SandboxMountIntent`` still owns their deduplication
    and ordering.
    """
    root = _canonical_mount_path(mount_root)
    remaining = tuple(
        _MountCandidate(_canonical_mount_path(c.path), c.origin) for c in candidates
    )
    deployment_identities = {c.path for c in remaining if c.origin == "deployment"}
    resolved = {c.path: resolve_backend_mount_path(c.path) for c in remaining}
    resolved_root = resolve_backend_mount_path(root)

    while True:
        verdicts = [
            (c, _fold_relation(root, resolved_root, c.path, resolved[c.path]))
            for c in remaining
        ]
        covering = [c for c, verdict in verdicts if verdict == "covering"]
        if not covering:
            # Judged only once the root has stopped moving: an earlier
            # promotion can widen the root enough to cover a candidate that
            # was disjoint from the narrower one.
            for candidate, verdict in verdicts:
                if (
                    verdict == "disjoint"
                    and candidate.path not in deployment_identities
                ):
                    raise SandboxMountEscapeError(
                        f"Workspace path {candidate.path!r} (resolving to "
                        f"{resolved[candidate.path]!r}) is neither inside nor "
                        f"a parent of sandbox mount root {root!r} (resolving "
                        f"to {resolved_root!r}); refusing to bind it as a "
                        "separate mount"
                    )
            return root, tuple(
                c.path for c, verdict in verdicts if verdict != "covered"
            )
        new_root = min(covering, key=lambda c: len(c.path))
        remaining = tuple(c for c in remaining if c.path != new_root.path)
        root, resolved_root = new_root.path, resolved[new_root.path]


def build_chat_workspace_binding(
    owner_id: int, scope: Optional[ExecutionScope]
) -> ChatWorkspaceBinding:
    """Build the CA-physical mount intent for a task's sandbox.

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
