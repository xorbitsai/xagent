"""Sandbox runner utils."""

from __future__ import annotations

import os
import site


def ensure_user_bin_in_path() -> None:
    """Append pip user-installed bin directories to PATH. They are not included in PATH by default."""
    user_base = site.getuserbase()
    user_bin = os.path.join(user_base, "bin")
    current = os.environ.get("PATH", "")
    if user_bin not in current.split(os.pathsep):
        os.environ["PATH"] = f"{user_bin}{os.pathsep}{current}"
