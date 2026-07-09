"""Shared adapter interface and parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path

from ..bundle import MigrationBundle, SkillItem


class SourceAdapter:
    """Read one source platform's footprint into a :class:`MigrationBundle`.

    Subclasses set :attr:`key` and implement :meth:`default_root` and
    :meth:`parse`. ``root`` overrides the auto-detected home directory (used by
    ``--source-dir`` and by tests pointing at fixtures).
    """

    key: str = ""

    def __init__(self, root: Path | None = None) -> None:
        self._root = Path(root).expanduser() if root is not None else None

    def default_root(self) -> Path:
        raise NotImplementedError

    @property
    def root(self) -> Path:
        return self._root if self._root is not None else self.default_root()

    def parse(self) -> MigrationBundle:
        raise NotImplementedError


def read_text(path: Path) -> str:
    """Read a text file, tolerating undecodable bytes, or ``""`` if absent."""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except (OSError, UnicodeError):
        return ""


def collect_skill_dirs(*roots: Path) -> dict[str, Path]:
    """Map skill name -> directory for every ``SKILL.md``-bearing subdir.

    Earlier roots win on name collision, matching the highest-precedence-first
    ordering the caller passes (workspace skills before shared skills).
    """
    found: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if not (child / "SKILL.md").is_file():
                continue
            found.setdefault(child.name, child)
    return found


def load_skill_dir(name: str, skill_dir: Path) -> SkillItem | None:
    """Load a skill directory into a :class:`SkillItem`, or ``None`` if unreadable."""
    files: dict[str, bytes] = {}
    for file_path in sorted(skill_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = str(file_path.relative_to(skill_dir)).replace("\\", "/")
        try:
            files[rel] = file_path.read_bytes()
        except OSError:
            continue
    if "SKILL.md" not in files:
        return None
    description = _skill_description(files["SKILL.md"])
    return SkillItem(
        name=name,
        files=files,
        source_path=str(skill_dir),
        description=description,
    )


def _skill_description(skill_md: bytes) -> str:
    """Best-effort one-line description from SKILL.md frontmatter or heading."""
    text = skill_md.decode("utf-8", errors="replace")
    match = re.search(r"^description:\s*(.+)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip().strip("\"'")
    return ""
