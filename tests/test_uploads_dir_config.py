"""The uploads root must have one physical meaning.

Every workspace, upload, knowledge-base and sandbox-mount path is composed
from ``get_uploads_dir()``, and its consumers normalize differently on
purpose: sandbox mount identity and desired-vs-observed spec comparison are
lexical, while ``TaskWorkspace``, the upload writers and ``files.py``'s
containment checks resolve symlinks. Those two readings agree on every
spelling except a symlink followed by ``..``, which splits one logical
directory into two real ones -- files then land where the sandbox never
mounted. Rejecting that input at the source is what lets each consumer keep
its own normalization.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from xagent.config import UploadsDirConfigurationError, get_uploads_dir


@pytest.fixture
def symlinked_tree(tmp_path: Path) -> Path:
    """``base/link`` points outside ``base``, one level down."""
    (tmp_path / "base").mkdir()
    (tmp_path / "outside" / "nested").mkdir(parents=True)
    (tmp_path / "base" / "link").symlink_to(
        tmp_path / "outside" / "nested", target_is_directory=True
    )
    return tmp_path


def test_symlink_followed_by_dotdot_is_rejected(symlinked_tree, monkeypatch):
    configured = f"{symlinked_tree / 'base'}/link/.."
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", configured)

    with pytest.raises(UploadsDirConfigurationError) as exc_info:
        get_uploads_dir()

    # The operator has to be able to see both readings to fix the value.
    message = str(exc_info.value)
    assert str(symlinked_tree / "base") in message
    assert os.path.realpath(symlinked_tree / "outside") in message


def test_the_web_dir_branch_is_validated_too(symlinked_tree, monkeypatch):
    """The check belongs to the value, not to one of its two producers.

    With ``XAGENT_UPLOADS_DIR`` unset the root is composed from
    ``XAGENT_WEB_DIR``, which is just as operator-configured, so the same
    ambiguity arrives by a different route.
    """
    monkeypatch.delenv("XAGENT_UPLOADS_DIR", raising=False)
    monkeypatch.setenv("XAGENT_WEB_DIR", f"{symlinked_tree / 'base'}/link/..")

    with pytest.raises(UploadsDirConfigurationError) as exc_info:
        get_uploads_dir()

    # Names the variable that is actually set, not the one that is not.
    assert "XAGENT_WEB_DIR" in str(exc_info.value)


def test_an_ordinary_symlink_is_accepted(tmp_path, monkeypatch):
    """Following a symlink is not a disagreement.

    Pointing the uploads dir at a symlink is ordinary deployment practice
    (a volume mounted elsewhere), and both readings still name one directory.
    Rejecting it would break far more deployments than it protects.
    """
    (tmp_path / "mnt" / "data").mkdir(parents=True)
    (tmp_path / "data").symlink_to(tmp_path / "mnt" / "data")
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", str(tmp_path / "data" / "uploads"))

    assert get_uploads_dir() == Path(tmp_path / "data" / "uploads")


@pytest.mark.parametrize(
    "spelling",
    [
        "{base}",
        "{base}/",
        "{base}//uploads",
        "{base}/nonexistent/..",
        "{base}/./uploads",
        "{base}/uploads-that-does-not-exist-yet",
    ],
)
def test_unambiguous_spellings_are_accepted(tmp_path, monkeypatch, spelling):
    """A ``..`` or ``//`` segment alone is not the problem.

    Those are why the mount side canonicalizes at all; they name the same
    directory under either reading, so the guard must not reject them.
    """
    (tmp_path / "base").mkdir()
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", spelling.format(base=tmp_path / "base"))

    get_uploads_dir()


def test_relative_and_home_spellings_are_accepted_as_what_was_checked(
    tmp_path, monkeypatch
):
    """Callers get the absolutized value, i.e. the one that was validated.

    A relative or ``~``-prefixed value is accepted -- configuration that
    predates any absolutization requirement still starts -- but it is resolved
    here rather than handed on unresolved: the Python execution tool calls
    ``os.chdir`` process-wide while a task runs, so a relative value returned
    verbatim could later name a directory this check never examined.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", "relative/uploads")
    assert get_uploads_dir() == tmp_path / "relative" / "uploads"

    monkeypatch.setenv("XAGENT_UPLOADS_DIR", "~/uploads")
    assert get_uploads_dir() == Path.home() / "uploads"


def test_a_later_chdir_cannot_move_the_validated_root(tmp_path, monkeypatch):
    """The returned root is fixed at validation, not re-resolved per caller."""
    (tmp_path / "elsewhere").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("XAGENT_UPLOADS_DIR", "relative/uploads")

    before = get_uploads_dir()
    os.chdir(tmp_path / "elsewhere")

    assert get_uploads_dir() == before


def test_unset_falls_back_to_the_packaged_directory(monkeypatch):
    monkeypatch.delenv("XAGENT_UPLOADS_DIR", raising=False)

    assert get_uploads_dir().name == "uploads"
