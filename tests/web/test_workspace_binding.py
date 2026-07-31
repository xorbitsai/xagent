"""Golden tests for the chat workspace projection (#296).

``build_chat_workspace_binding`` is the sandbox mount intent's builder,
called from ``chat.py``. This module pins its ``mount_intent``/
``prepare_root`` output against a frozen reimplementation of the pre-folding
physical mount set (chat's ``_build_allowed_external_dirs`` +
``SandboxManager._workspace_mount_paths``'s "mount base_dir + every
allowed_external_dirs entry, deduplicated only by exact string" behavior),
so the folding this module applies stays a pinned, deliberate reduction of
that raw set rather than a silent behavior drift.

Six-row physical-set matrix (unscoped / scoped isolate=False / external CA
scoped / internal scoped / two known-limitation shapes), plus an Actor-path
invariant, an external-dir ancestor/descendant boundary check, and the
provenance split that decides what an unfoldable candidate means (a
deployment-configured dir keeps its own bind; a workspace path that escapes
the mount root covering it fails the build) together with the identity
domain that split is decided in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pytest

import xagent.web.services.workspace_binding as workspace_binding
from xagent.core.execution_scope import ExecutionScope
from xagent.core.workspace import scoped_user_root
from xagent.sandbox import SandboxContractError, SandboxMountEscapeError
from xagent.web.services.workspace_binding import (
    ChatWorkspaceBinding,
    build_chat_workspace_binding,
)

OWNER_ID = 42


@pytest.fixture(autouse=True)
def _uploads_dir(tmp_path, monkeypatch):
    """Point the builder's uploads dir at an isolated tmp tree."""
    monkeypatch.setattr(workspace_binding, "get_uploads_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _external_dirs(tmp_path, monkeypatch):
    """A single deployment-level external upload dir, disjoint from uploads.

    A sibling of ``tmp_path`` (not nested under it), matching real
    deployments where ``XAGENT_EXTERNAL_UPLOAD_DIRS`` points at a shared KB
    location outside the per-user uploads tree.
    """
    ext_dir = tmp_path.parent / f"{tmp_path.name}-shared-kb"
    monkeypatch.setattr(
        workspace_binding, "get_external_upload_dirs", lambda: [ext_dir]
    )
    return ext_dir


@pytest.fixture(autouse=True)
def _reset_warn_guard(monkeypatch):
    """Isolate the process-level once-per-kind warn guard per test."""
    monkeypatch.setattr(workspace_binding, "_warned_limitation_kinds", set())


def _legacy_today_paths(
    owner_id: int,
    scope: Optional[ExecutionScope],
    *,
    uploads_dir: Path,
    ext_dirs: list[Path],
) -> set[str]:
    """Frozen reimplementation of today's physical mount set.

    Mirrors chat.py's ``sandbox_workspace_config`` dict (``base_dir`` from
    ``scoped_user_root`` at ``effective_mount_segments``,
    ``allowed_external_dirs`` from ``_build_allowed_external_dirs``) fed
    through ``SandboxManager._workspace_mount_paths``, which mounts
    ``base_dir`` plus every ``allowed_external_dirs`` entry with no
    covered/covering folding -- only exact-string dedup (a plain ``set``).
    """
    mount_segments = scope.effective_mount_segments if scope is not None else ()
    base_dir = scoped_user_root(uploads_dir, owner_id, mount_segments)

    ext_segments = (
        scope.workspace_segments
        if scope is not None and scope.isolate_external_dirs
        else ()
    )
    user_upload_dir = scoped_user_root(uploads_dir, owner_id, ext_segments)
    allowed_external_dirs = [str(user_upload_dir)] + [str(d) for d in ext_dirs]

    return {str(base_dir)} | set(allowed_external_dirs)


def _new_physical_paths(binding: ChatWorkspaceBinding) -> set[str]:
    intent = binding.mount_intent
    assert intent.mount_root is not None
    return {intent.mount_root} | set(intent.extra_mounts)


class TestUnscopedRow:
    """Row a: scope=None -- byte-identical to today, no folding needed."""

    def test_physical_set_matches_today(self, _uploads_dir, _external_dirs):
        binding = build_chat_workspace_binding(OWNER_ID, None)
        old = _legacy_today_paths(
            OWNER_ID, None, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        assert _new_physical_paths(binding) == old

    def test_prepare_root_is_user_root(self, _uploads_dir):
        binding = build_chat_workspace_binding(OWNER_ID, None)
        assert binding.prepare_root == str(scoped_user_root(_uploads_dir, OWNER_ID, ()))


class TestScopedIsolateFalseRow:
    """Row b: scoped, isolate_external_dirs=False (default).

    The mount root (a scope subtree) is covered by an ancestor already in
    the allowlist (the unscoped user root, present because isolate=False
    keeps the shared, un-narrowed allowlist entry) -- that ancestor absorbs
    the mount, replacing it. The original mount root disappears from the
    physical set, but it stays fully reachable through the promoted
    ancestor's mount: this is the one documented, harmless reduction for
    this row (not a byte-exact match to today's raw, non-folding mount
    list).
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="tenantA", workspace_segments=("proj1",)
        )

    def test_covering_ancestor_absorbs_mount_root(self, _uploads_dir, _external_dirs):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        user_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ()))
        scoped_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("proj1",)))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {scoped_root, user_root, str(_external_dirs)}
        assert new == {user_root, str(_external_dirs)}
        assert old - new == {scoped_root}, (
            "only the absorbed (now-redundant) root drops"
        )
        assert new - old == set(), "folding never introduces a path absent from today"
        assert binding.mount_intent.mount_root == user_root

    def test_prepare_root_stays_the_unfolded_scope_subtree(self, _uploads_dir):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        assert binding.prepare_root == str(
            scoped_user_root(_uploads_dir, OWNER_ID, ("proj1",))
        )
        # mkdir target is the pre-fold root, distinct from the folded
        # mount_intent.mount_root (the promoted ancestor).
        assert binding.prepare_root != binding.mount_intent.mount_root

    def test_no_warning_emitted(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        assert caplog.records == []


class TestExternalCAScopedRow:
    """Row c: suffix + mount prefix + isolate=True -- the #296 fix itself.

    The Actor's own subtree (full workspace_segments, present in the
    allowlist because isolate=True) is covered by the CA mount root and is
    dropped. Unlike row b, the root itself is unchanged -- only a covered
    extra disappears. This is the sole row where the physical-set diff is
    the deliberate fix target, pinned exactly to the Actor subtree.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", "actor7"),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )

    def test_actor_child_is_the_only_diff_from_today(
        self, _uploads_dir, _external_dirs
    ):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        ca_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",)))
        actor_child = str(scoped_user_root(_uploads_dir, OWNER_ID, ("ca1", "actor7")))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {ca_root, actor_child, str(_external_dirs)}
        assert new == {ca_root, str(_external_dirs)}
        assert old - new == {actor_child}, (
            "the Actor subtree is exactly what disappears"
        )
        assert new - old == set()
        assert binding.mount_intent.mount_root == ca_root

    def test_prepare_root_is_the_ca_root(self, _uploads_dir):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        assert binding.prepare_root == str(
            scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        )
        assert binding.prepare_root == binding.mount_intent.mount_root

    def _sibling_scope(self, actor: str = "actor9") -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", actor),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )

    def test_two_actors_under_same_ca_fold_to_identical_intent(self, _uploads_dir):
        """The multi-Actor collision #296 is about: same CA, different Actor
        subtrees must fold to a byte-identical intent to share one container.
        """
        binding_a = build_chat_workspace_binding(OWNER_ID, self._scope())
        binding_b = build_chat_workspace_binding(OWNER_ID, self._sibling_scope())
        assert binding_a.mount_intent == binding_b.mount_intent

    def test_identical_intent_holds_with_real_actor_dirs_on_disk(self, _uploads_dir):
        """Same invariant with the Actor subtrees actually materialized, so
        the resolved view has real directories to answer with rather than
        falling back to the lexical spelling of a missing path."""
        ca_root = scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        (ca_root / "actor7").mkdir(parents=True)
        (ca_root / "actor9").mkdir(parents=True)

        binding_a = build_chat_workspace_binding(OWNER_ID, self._scope())
        binding_b = build_chat_workspace_binding(OWNER_ID, self._sibling_scope())
        assert binding_a.mount_intent == binding_b.mount_intent
        assert binding_a.mount_intent.mount_root == str(ca_root)
        assert str(ca_root / "actor7") not in binding_a.mount_intent.extra_mounts

    def test_identical_intent_holds_when_one_actor_dir_is_an_inside_symlink(
        self, _uploads_dir
    ):
        """An Actor subtree that is a symlink resolving back inside the CA
        root is still genuinely covered by the CA bind, so it must fold away
        exactly like a real directory -- otherwise the shared container
        splits again."""
        ca_root = scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        (ca_root / "actor7").mkdir(parents=True)
        (ca_root / "real-actor9").mkdir()
        (ca_root / "actor9").symlink_to(
            ca_root / "real-actor9", target_is_directory=True
        )

        binding_a = build_chat_workspace_binding(OWNER_ID, self._scope())
        binding_b = build_chat_workspace_binding(OWNER_ID, self._sibling_scope())
        assert binding_a.mount_intent == binding_b.mount_intent


class TestScopeDerivedEscapeFailsClosed:
    """A workspace path that escapes the mount root covering it is rejected.

    The scope-derived candidate's containment in the mount root is a
    precondition of this projection (the CA root is chosen to cover it), the
    tree it names is workspace-controlled, and its spelling varies per Actor.
    So an escape may not be demoted to a separate bind: that would hand the
    backend a writable mount of the symlink's target and give this Actor an
    intent no sibling Actor under the same CA produces, defeating #296's
    shared container. It fails the build instead.
    """

    def _ca_scope(self, actor: str) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", actor),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )

    def test_actor_subtree_escaping_the_ca_root_is_rejected(
        self, _uploads_dir, tmp_path
    ):
        """Covered direction: lexically under the CA root, resolving out."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside-actor"
        outside.mkdir()
        ca_root = scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        ca_root.mkdir(parents=True)
        (ca_root / "actor7").symlink_to(outside, target_is_directory=True)

        with pytest.raises(SandboxMountEscapeError) as excinfo:
            build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor7"))
        assert str(outside) in str(excinfo.value)

    def test_mount_root_escaping_the_user_root_is_rejected(
        self, _uploads_dir, tmp_path
    ):
        """Covering direction: with isolate_external_dirs=False the candidate
        is the unscoped user root, which must contain the narrower mount root
        it is about to absorb. A symlinked mount root pointing outside the
        user tree breaks that, so the promotion is rejected rather than
        binding both the alias and its parent."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside-proj"
        outside.mkdir()
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        user_root.mkdir(parents=True)
        (user_root / "proj1").symlink_to(outside, target_is_directory=True)

        scope = ExecutionScope(
            sandbox_key_suffix="tenantA", workspace_segments=("proj1",)
        )
        with pytest.raises(SandboxMountEscapeError):
            build_chat_workspace_binding(OWNER_ID, scope)

    def test_error_is_a_sandbox_contract_error(self, _uploads_dir, tmp_path):
        """chat.py classifies ``SandboxContractError`` as fail-closed: the
        task fails instead of silently running unsandboxed on the host."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside-actor"
        outside.mkdir()
        ca_root = scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        ca_root.mkdir(parents=True)
        (ca_root / "actor7").symlink_to(outside, target_is_directory=True)

        with pytest.raises(SandboxContractError):
            build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor7"))

    def test_same_shape_from_a_deployment_dir_still_gets_its_own_mount(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """Provenance is the whole difference: an operator-configured
        external dir with the identical escaping-symlink shape is an
        independently declared mount, identical for every Actor, so it keeps
        its own bind instead of failing the build."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside-kb"
        outside.mkdir()
        ca_root = scoped_user_root(_uploads_dir, OWNER_ID, ("ca1",))
        (ca_root / "actor7").mkdir(parents=True)
        link = ca_root / "shared-kb"
        link.symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [link]
        )

        binding = build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor7"))

        assert binding.mount_intent.mount_root == str(ca_root)
        assert binding.mount_intent.extra_mounts == (str(link),)


class TestDeploymentAuthorizationIdentityDomain:
    """A path can legitimately be both scope-derived and operator-configured.

    An unfoldable scope-derived candidate fails the build unless the
    deployment named the same mount point: an operator who names a path
    takes responsibility for it regardless of whether some Actor's scope
    also happens to derive it. "The same mount point" is
    ``canonical_sandbox_path`` -- the identity ``SandboxMountIntent`` itself
    uses -- not the configured spelling. That distinction is load-bearing
    because ``XAGENT_EXTERNAL_UPLOAD_DIRS`` takes raw operator strings and
    absolutizing them leaves ``..`` segments and a leading ``//`` in place:
    matching on the absolutized spelling instead would reject the Actor
    whose scope collides with a differently-spelled deployment entry while
    a sibling Actor -- whose own scope path never touches that entry --
    still gets it as a stable extra mount, which is exactly the per-Actor
    intent divergence this module exists to prevent.

    Parametrized over spellings rather than asserted once: the verdict for
    one mount point must not depend on how it was typed. Only spellings that
    survive ``Path`` are listed -- a trailing slash, a doubled inner slash or
    a ``.`` segment is already gone before production code sees the value.
    """

    def _ca_scope(self, actor: str) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-1",
            workspace_segments=("ca1", actor),
            sandbox_mount_segments=("ca1",),
            isolate_external_dirs=True,
        )

    def _escaping_actor7_tree(self, uploads_dir, tmp_path) -> Path:
        """CA root whose ``actor7`` subtree resolves out of it.

        That makes actor7's own scope-derived candidate unfoldable (lexically
        covered by the CA root, resolving outside it), so its fate is decided
        purely by whether the deployment named that mount point. actor9's
        scope path is an ordinary covered path and folds away, so its intent
        is the fold-only baseline the two builds must agree on.
        """
        outside = tmp_path.parent / f"{tmp_path.name}-outside-actor"
        outside.mkdir()
        ca_root = scoped_user_root(uploads_dir, OWNER_ID, ("ca1",))
        ca_root.mkdir(parents=True)
        (ca_root / "actor7").symlink_to(outside, target_is_directory=True)
        return ca_root

    @pytest.mark.parametrize(
        "spelling",
        [
            "{ca}/actor7",
            "{ca}/nonexistent/../actor7",
            "/{ca}/actor7",
        ],
    )
    def test_every_spelling_of_the_named_mount_point_authorizes_it(
        self, _uploads_dir, tmp_path, monkeypatch, spelling
    ):
        ca_root = self._escaping_actor7_tree(_uploads_dir, tmp_path)
        configured = spelling.format(ca=str(ca_root))
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [Path(configured)]
        )

        binding_a = build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor7"))
        binding_b = build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor9"))

        assert binding_a.mount_intent == binding_b.mount_intent
        assert binding_a.mount_intent.mount_root == str(ca_root)
        assert binding_a.mount_intent.extra_mounts == (str(ca_root / "actor7"),)

    def test_a_symlink_alias_of_the_path_does_not_authorize_it(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """The other edge of the same domain: the identity is lexical, so an
        operator entry that merely *resolves* to the escaping path is a
        different bind point and authorizes nothing. Naming an alias grants
        the alias its own mount; the scope path itself still fails closed.
        """
        ca_root = self._escaping_actor7_tree(_uploads_dir, tmp_path)
        alias = ca_root / "aliased-kb"
        alias.symlink_to(ca_root / "actor7", target_is_directory=True)
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [alias]
        )

        with pytest.raises(SandboxMountEscapeError):
            build_chat_workspace_binding(OWNER_ID, self._ca_scope("actor7"))

    def test_ancestor_spelled_non_canonically_still_absorbs_the_mount_root(
        self, _uploads_dir, monkeypatch
    ):
        """The covering direction shares the domain: a deployment ancestor
        spelled with a ``..`` segment is the same mount point as its
        canonical spelling, so it absorbs the mount root and the promoted
        root reaches the intent canonicalized.
        """
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        monkeypatch.setattr(
            workspace_binding,
            "get_external_upload_dirs",
            lambda: [Path(f"{user_root}/nonexistent/..")],
        )
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )

        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == ()

    def test_ancestor_spelled_through_a_symlink_and_dotdot_folds_identically(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """The two normalizations must not disagree on one mount point.

        ``..`` after a symlink is where lexical and resolved normalization
        part ways: ``<root>/link/..`` is lexically ``<root>``, while resolving
        the raw spelling lands wherever ``link`` points. Folding has to judge
        the path that will actually be mounted -- the canonical one -- so
        this spelling of an ancestor must absorb the mount root exactly as
        the plain spelling does, down to the runtime spec's mount list. Both
        directions matter: it is the covering side here, and the same domain
        split decides the escape rejection above.
        """
        # Nested one level down, so that resolving ``link/..`` lands on a
        # directory that is not an ancestor of the mount root either -- the
        # resolved view then contradicts the lexical one instead of agreeing
        # with it by accident.
        outside = tmp_path.parent / f"{tmp_path.name}-outside-root" / "nested"
        outside.mkdir(parents=True)
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        user_root.mkdir(parents=True)
        (user_root / "link").symlink_to(outside, target_is_directory=True)
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )

        monkeypatch.setattr(
            workspace_binding,
            "get_external_upload_dirs",
            lambda: [Path(f"{user_root}/link/..")],
        )

        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == ()


class TestInternalScopedRow:
    """Row d: suffix + mount=None (full segments) + isolate=True.

    base_dir and the isolate-narrowed allowlist entry are the *same* path
    already -- today's raw list already carries an exact-string duplicate
    that a plain ``set()`` collapses, so this row is byte-identical to
    today with no caveat.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="proj2",
            workspace_segments=("proj2",),
            isolate_external_dirs=True,
        )

    def test_physical_set_matches_today_exactly(self, _uploads_dir, _external_dirs):
        scope = self._scope()
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        workspace_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("proj2",)))

        old = _legacy_today_paths(
            OWNER_ID, scope, uploads_dir=_uploads_dir, ext_dirs=[_external_dirs]
        )
        new = _new_physical_paths(binding)

        assert old == {workspace_root, str(_external_dirs)}
        assert new == old
        assert binding.mount_intent.mount_root == workspace_root
        assert binding.prepare_root == workspace_root

    def test_no_warning_emitted(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        assert caplog.records == []


class TestSuffixlessIsolateRow:
    """Row e (known limitation): isolate=True with no sandbox_key_suffix.

    This scope shares the *unscoped* sandbox lifecycle key (no suffix) yet
    still computes a scoped, non-unscoped mount root -- the same container
    identity would see a different desired mount depending on which scope
    built it. Root cause: scope authority isn't fully closed until PR-2;
    this builder can only flag it, not fix it.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix=None,
            workspace_segments=("solo",),
            isolate_external_dirs=True,
        )

    def test_result_differs_from_unscoped(self, _uploads_dir, _external_dirs):
        scoped_binding = build_chat_workspace_binding(OWNER_ID, self._scope())
        unscoped_binding = build_chat_workspace_binding(OWNER_ID, None)

        scoped_root = str(scoped_user_root(_uploads_dir, OWNER_ID, ("solo",)))
        assert _new_physical_paths(scoped_binding) == {scoped_root, str(_external_dirs)}
        assert _new_physical_paths(scoped_binding) != _new_physical_paths(
            unscoped_binding
        )

    def test_emits_one_structured_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        matching = [r for r in caplog.records if "sandbox_key_suffix" in r.getMessage()]
        assert len(matching) == 1

    def test_warning_fires_only_once_per_process(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        scope = self._scope()
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(99, scope)
        matching = [r for r in caplog.records if "sandbox_key_suffix" in r.getMessage()]
        assert len(matching) == 1


class TestDivergentMountPrefixRow:
    """Row f (known limitation): explicit sandbox_mount_segments without
    isolate_external_dirs.

    The narrower mount root the caller asked for is folded away into the
    unscoped user-root allowlist entry (isolate=False keeps that entry
    un-narrowed), so the requested narrowing has zero physical effect --
    the result collapses to the same set as the plain unscoped row.
    """

    def _scope(self) -> ExecutionScope:
        return ExecutionScope(
            sandbox_key_suffix="ca-2",
            workspace_segments=("ca2", "actorX"),
            sandbox_mount_segments=("ca2",),
            isolate_external_dirs=False,
        )

    def test_narrowing_has_no_physical_effect(self, _uploads_dir, _external_dirs):
        binding = build_chat_workspace_binding(OWNER_ID, self._scope())
        unscoped_binding = build_chat_workspace_binding(OWNER_ID, None)
        assert _new_physical_paths(binding) == _new_physical_paths(unscoped_binding)

    def test_emits_one_structured_warning(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        build_chat_workspace_binding(OWNER_ID, self._scope())
        matching = [
            r for r in caplog.records if "sandbox_mount_segments" in r.getMessage()
        ]
        assert len(matching) == 1

    def test_warning_fires_only_once_per_process(self, caplog):
        caplog.set_level(logging.WARNING, logger=workspace_binding.logger.name)
        scope = self._scope()
        build_chat_workspace_binding(OWNER_ID, scope)
        build_chat_workspace_binding(OWNER_ID, scope)
        matching = [
            r for r in caplog.records if "sandbox_mount_segments" in r.getMessage()
        ]
        assert len(matching) == 1


class TestActorPathNeverEntersIntent:
    """Invariant: with isolate=True and a genuine mount/workspace split, the
    Actor's own (deeper) path never surfaces as a mount -- neither as an
    extra mount nor as the mount root -- at any workspace_segments depth.
    """

    @pytest.mark.parametrize(
        "workspace_segments,mount_segments",
        [
            (("ca", "actor"), ("ca",)),
            (("ca", "team", "actor"), ("ca",)),
            (("ca", "team", "actor"), ("ca", "team")),
        ],
    )
    def test_actor_subtree_excluded_at_every_depth(
        self, _uploads_dir, workspace_segments, mount_segments
    ):
        scope = ExecutionScope(
            sandbox_key_suffix="ca",
            workspace_segments=workspace_segments,
            sandbox_mount_segments=mount_segments,
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)
        actor_path = str(scoped_user_root(_uploads_dir, OWNER_ID, workspace_segments))
        mount_root_path = str(scoped_user_root(_uploads_dir, OWNER_ID, mount_segments))

        assert actor_path != mount_root_path, "fixture must exercise a genuine split"
        physical = _new_physical_paths(binding)
        assert actor_path not in physical
        assert binding.mount_intent.mount_root != actor_path


class TestExternalDirBoundary:
    """ext contains the mount root's ancestor or descendant (deployment-level
    external dir specifically -- isolate=True keeps the isolate-driven
    allowlist candidate equal to the mount root so it cannot itself act as
    the covering/covered path under test).
    """

    def test_ancestor_external_dir_becomes_new_root(self, _uploads_dir, monkeypatch):
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [user_root]
        )
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == ()

    def test_descendant_external_dir_is_dropped(self, _uploads_dir, monkeypatch):
        scoped_root = scoped_user_root(_uploads_dir, OWNER_ID, ("proj",))
        descendant = scoped_root / "kb"
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [descendant]
        )
        scope = ExecutionScope(
            sandbox_key_suffix="s",
            workspace_segments=("proj",),
            isolate_external_dirs=True,
        )
        binding = build_chat_workspace_binding(OWNER_ID, scope)

        assert binding.mount_intent.mount_root == str(scoped_root)
        assert binding.mount_intent.extra_mounts == ()


class TestBackendPathDomain:
    """Folding candidates are normalized, and symlink-checked, first.

    ``SandboxMountIntent`` compares paths lexically and rejects relative
    ones outright, so the builder owns both halves of that precondition:
    raw configuration spellings become absolute, canonical backend-domain
    paths, and a candidate whose lexical position disagrees with where it
    actually resolves is never folded away.
    """

    def test_relative_external_dir_is_absolutized(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """``XAGENT_EXTERNAL_UPLOAD_DIRS`` accepts any spelling naming an
        existing directory, cwd-relative included; the pre-projection path
        mapper absolutized those, so rejecting them here would fail every
        task on a deployment that configures one."""
        external = tmp_path.parent / f"{tmp_path.name}-relative-kb"
        external.mkdir()
        monkeypatch.chdir(external.parent)
        monkeypatch.setattr(
            workspace_binding,
            "get_external_upload_dirs",
            lambda: [Path(external.name)],
        )

        binding = build_chat_workspace_binding(OWNER_ID, None)

        assert binding.mount_intent.extra_mounts == (str(Path.cwd() / external.name),)

    def test_relative_uploads_dir_absolutizes_the_mount_root(
        self, tmp_path, monkeypatch
    ):
        """Same for ``XAGENT_UPLOADS_DIR``, which reaches the mount root."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(workspace_binding, "get_uploads_dir", lambda: Path("up"))
        monkeypatch.setattr(workspace_binding, "get_external_upload_dirs", lambda: [])

        binding = build_chat_workspace_binding(OWNER_ID, None)

        expected = str(Path.cwd() / "up" / f"user_{OWNER_ID}")
        assert binding.mount_intent.mount_root == expected
        assert binding.prepare_root == expected

    def test_uploads_dir_spelled_through_a_symlink_prepares_the_bound_root(
        self, tmp_path, monkeypatch
    ):
        """The mkdir target must be the directory that gets bind-mounted.

        ``prepare_root`` is the pre-fold root, deeper than the folded mount
        root but the same path domain: the mount intent canonicalizes what it
        binds, so an uploads dir spelled ``<base>/link/..`` binds ``<base>``
        while a raw-spelled preparation target would create the tree under
        ``link``'s target instead, leaving the real mount source empty.
        """
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside" / "nested"
        outside.mkdir(parents=True)
        (base / "link").symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr(
            workspace_binding, "get_uploads_dir", lambda: Path(f"{base}/link/..")
        )
        monkeypatch.setattr(workspace_binding, "get_external_upload_dirs", lambda: [])

        binding = build_chat_workspace_binding(OWNER_ID, None)

        assert binding.prepare_root == binding.mount_intent.mount_root
        assert binding.prepare_root == str(base / f"user_{OWNER_ID}")

    def test_symlink_escaping_the_root_keeps_its_own_mount(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """A candidate lexically under the mount root but resolving outside
        it is exposed by nothing once the root is bind-mounted, so folding
        it away would silently drop the mount."""
        outside = tmp_path.parent / f"{tmp_path.name}-outside-kb"
        outside.mkdir()
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        user_root.mkdir(parents=True)
        link = user_root / "shared-kb"
        link.symlink_to(outside, target_is_directory=True)
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [link]
        )

        binding = build_chat_workspace_binding(OWNER_ID, None)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == (str(link),)

    def test_symlink_staying_inside_the_root_still_folds_away(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """The veto is resolution-driven, not "any symlink is disjoint": a
        link resolving back inside the root is genuinely redundant and must
        still fold, or #296's shared container splits again."""
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        target = user_root / "real-kb"
        target.mkdir(parents=True)
        link = user_root / "linked-kb"
        link.symlink_to(target, target_is_directory=True)
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [link]
        )

        binding = build_chat_workspace_binding(OWNER_ID, None)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == ()

    def test_symlink_only_resolving_to_an_ancestor_is_not_promoted(
        self, _uploads_dir, tmp_path, monkeypatch
    ):
        """The covering direction needs the same veto: mounting a link that
        merely *resolves* to an ancestor does not expose the mount root at
        the mount root's own path, so it must not absorb it."""
        alias = tmp_path.parent / f"{tmp_path.name}-alias"
        alias.symlink_to(_uploads_dir, target_is_directory=True)
        user_root = scoped_user_root(_uploads_dir, OWNER_ID, ())
        user_root.mkdir(parents=True)
        monkeypatch.setattr(
            workspace_binding, "get_external_upload_dirs", lambda: [alias]
        )

        binding = build_chat_workspace_binding(OWNER_ID, None)

        assert binding.mount_intent.mount_root == str(user_root)
        assert binding.mount_intent.extra_mounts == (str(alias),)
