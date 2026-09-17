"""Place the jieba dictionary lance's FTS tokenizer loads at runtime."""

from __future__ import annotations

import logging
import os
import shutil
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_once = threading.Lock()
_installed = False


def _default_model_home() -> Path:
    """Lance's local data directory, as the Rust ``dirs::data_local_dir`` resolves it."""
    if sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "lance" / "language_models"


def _language_model_home() -> Path:
    configured = os.environ.get("LANCE_LANGUAGE_MODEL_HOME")
    return Path(configured) if configured else _default_model_home()


def ensure_jieba_dictionary() -> bool:
    """Copy the PyPI jieba dictionary into lance's language-model home.

    Only the macOS wheel embeds a dictionary; on Linux both index creation and
    querying raise without this file, so it is placed on every connection path
    rather than only where the index is built.

    Returns:
        True once the dictionary is in place.
    """
    target = _language_model_home() / "jieba" / "default" / "dict.txt"

    try:
        import jieba

        source = Path(jieba.__file__).parent / "dict.txt"
        # Size, not existence: a truncated copy leaves lance with a dictionary
        # that silently loses words.
        if target.exists() and target.stat().st_size == source.stat().st_size:
            return True

        target.parent.mkdir(parents=True, exist_ok=True)
        # Concurrent workers race on this path, so the copy lands through an
        # atomic os.replace; a SIGKILL mid-copy is what strands a staging file.
        for stale in target.parent.glob("dict.txt.*.tmp"):
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                pass

        staged = target.with_name(f"dict.txt.{os.getpid()}.tmp")
        try:
            shutil.copyfile(source, staged)
            os.replace(staged, target)
        finally:
            staged.unlink(missing_ok=True)
        logger.info("Installed the jieba dictionary for LanceDB FTS at %s", target)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not install the jieba dictionary at %s (%s); LanceDB FTS index "
            "creation and full-text queries will fail until it exists",
            target,
            exc,
        )
        return False


def ensure_jieba_dictionary_once() -> None:
    """Install the dictionary once per process, retrying while it fails."""
    global _installed
    if _installed:
        return
    with _once:
        if _installed:
            return
        _installed = ensure_jieba_dictionary()
