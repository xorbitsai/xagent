"""Guard: user-root paths and storage-key prefixes stay in one place.

Slice 3 of #757 centralized the ``user root + scope segments`` composition
in ``xagent.core.workspace.scoped_user_root`` and the storage-key layout in
``xagent.core.file_storage.keys``. The review of #757 found the user-root
format hand-built at roughly a dozen call sites — any one left unmigrated
writes or mounts unscoped paths while the rest of the system is scoped,
which is the "partially applied scope" bug class.

These tests grep the source tree so the format cannot fragment again: a
NEW hand-built ``user_{...}`` path literal or ``users/{...}`` key literal
fails the guard unless it goes through the entry points (or is added to
the reviewed allowlist below with a justification).
"""

import re
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "xagent"

# A quote immediately followed by ``user_{`` means an f-string composing a
# user-root path component from an id.
USER_ROOT_LITERAL = re.compile(r"""["']user_\{""")

# Reviewed occurrences that are NOT hand-built user-root path composition.
# Adding a new entry requires the same review: is this composing a path a
# scoped execution could reach? If so, use scoped_user_root() instead.
USER_ROOT_ALLOWLIST = {
    # The entry point itself — the single owner of the composition.
    "core/workspace.py",
    # Legacy fallback for lifecycle ids that fail the owner parse; the
    # parsed path goes through scoped_user_root() just above it.
    "web/sandbox_manager.py",
    # Matches a user segment inside already-persisted storage paths;
    # comparison, not composition.
    "web/api/kb.py",
    # Compares the first component of a workspace-relative path;
    # comparison, not composition.
    "web/services/task_execution.py",
}

# Same idea for the durable-storage key prefix (``users/{user_id}/...``).
STORAGE_KEY_LITERAL = re.compile(r"""["']users/\{|f"users/""")

STORAGE_KEY_ALLOWLIST = {
    # The canonical key builders — the single owner of the key layout.
    # ``get_user_file_storage`` binds ScopedFileStorage through
    # ``build_user_key_prefix`` from here, so factory.py no longer hand-builds
    # the ``users/{...}`` prefix and is not allowlisted.
    "core/file_storage/keys.py",
    "web/services/startup_file_storage_sync.py",
}


def _grep(pattern: re.Pattern) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        rel = path.relative_to(SRC_ROOT).as_posix()
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if pattern.search(line):
                hits.setdefault(rel, []).append(f"{rel}:{line_no}: {line.strip()}")
    return hits


def test_no_new_hand_built_user_root_literals() -> None:
    hits = _grep(USER_ROOT_LITERAL)
    offenders = {
        rel: lines for rel, lines in hits.items() if rel not in USER_ROOT_ALLOWLIST
    }
    assert not offenders, (
        "Hand-built user-root path literal(s) found outside "
        "xagent.core.workspace.scoped_user_root — compose the path through "
        "the entry point so ExecutionScope.workspace_segments cannot be "
        "partially applied:\n"
        + "\n".join(line for lines in offenders.values() for line in lines)
    )


def test_no_new_hand_built_storage_key_prefix_literals() -> None:
    hits = _grep(STORAGE_KEY_LITERAL)
    offenders = {
        rel: lines for rel, lines in hits.items() if rel not in STORAGE_KEY_ALLOWLIST
    }
    assert not offenders, (
        "Hand-built storage-key prefix literal(s) found outside "
        "xagent.core.file_storage.keys — build keys through the canonical "
        "builders so scope segments cannot be partially applied:\n"
        + "\n".join(line for lines in offenders.values() for line in lines)
    )


def test_guard_allowlist_entries_still_exist() -> None:
    """A stale allowlist entry means the occurrence moved — re-review it."""
    user_hits = _grep(USER_ROOT_LITERAL)
    key_hits = _grep(STORAGE_KEY_LITERAL)
    stale = (USER_ROOT_ALLOWLIST - set(user_hits)) | (
        STORAGE_KEY_ALLOWLIST - set(key_hits)
    )
    assert not stale, f"Allowlist entries with no remaining occurrence: {stale}"
