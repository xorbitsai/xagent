"""Shared production-source scan root for the interaction module's static
AST guards.

``_scan_root`` was two verbatim copies -- ``test_interaction_handoff_surface.py``
and ``test_task_interaction_anchor.py`` -- each backing that file's own
zero-direct-construction / zero-production-caller guard. One shared copy
here, imported by both, so the two guards cannot drift apart on what
"the production source tree" means.

Two blind spots this helper has by construction, disclosed rather than
closed:

(a) The exclusion is stem-keyed, not path-keyed. Any other file anywhere
    under this package that happened to carry the same filename would be
    dropped from the scan set alongside the module the caller meant to
    exclude, on filename alone.
(b) The scan root is ``xagent.__path__`` (``src/xagent``), not the
    repository root. Anything under the top-level ``scripts/`` directory,
    or any other tree outside the installed package, is invisible to every
    guard built on this helper -- positive controls included.
"""

from __future__ import annotations

from pathlib import Path

import xagent


def _scan_root(exclude_stem: str) -> list[Path]:
    root = Path(next(iter(xagent.__path__)))
    modules = [p for p in root.rglob("*.py") if p.stem != exclude_stem]
    assert modules, "production scan set is empty"
    return modules
