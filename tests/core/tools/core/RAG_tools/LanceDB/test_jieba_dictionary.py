"""Tests for the jieba dictionary bootstrap."""

from __future__ import annotations

import builtins
import logging
import os
import shutil
import sys
from pathlib import Path

import jieba
import pytest

from xagent.core.tools.core.RAG_tools.LanceDB import jieba_dictionary
from xagent.core.tools.core.RAG_tools.LanceDB.jieba_dictionary import (
    ensure_jieba_dictionary,
    ensure_jieba_dictionary_once,
)

SOURCE = Path(jieba.__file__).parent / "dict.txt"


@pytest.fixture
def model_home(monkeypatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LANCE_LANGUAGE_MODEL_HOME", str(tmp_path))
    return tmp_path


def _target(model_home: Path) -> Path:
    return model_home / "jieba" / "default" / "dict.txt"


def _break_jieba_import(monkeypatch) -> None:
    real_import = builtins.__import__

    def no_jieba(name, *args, **kwargs):
        if name == "jieba":
            raise ImportError("no jieba here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_jieba)


def test_ensure_jieba_dictionary_copies_the_package_dictionary(model_home: Path):
    assert ensure_jieba_dictionary() is True

    target = _target(model_home)
    assert target.read_bytes() == SOURCE.read_bytes()
    # A leftover staging file would be read by lance as another dictionary.
    assert sorted(p.name for p in target.parent.iterdir()) == ["dict.txt"]


def test_ensure_jieba_dictionary_leaves_a_complete_file_alone(model_home: Path):
    target = _target(model_home)
    target.parent.mkdir(parents=True)
    shutil.copyfile(SOURCE, target)
    mtime = target.stat().st_mtime_ns

    assert ensure_jieba_dictionary() is True

    assert target.stat().st_mtime_ns == mtime


def test_ensure_jieba_dictionary_replaces_a_truncated_file(model_home: Path):
    target = _target(model_home)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")

    assert ensure_jieba_dictionary() is True

    assert target.read_bytes() == SOURCE.read_bytes()


def test_ensure_jieba_dictionary_warns_instead_of_raising(
    model_home: Path, monkeypatch, caplog
):
    _break_jieba_import(monkeypatch)

    with caplog.at_level(logging.WARNING):
        assert ensure_jieba_dictionary() is False

    assert not (model_home / "jieba").exists()
    assert "jieba" in caplog.text and "will fail" in caplog.text


def test_ensure_jieba_dictionary_removes_a_stale_staging_file(model_home: Path):
    """A SIGKILL mid-copy leaves one behind; lance would read it as a dictionary."""
    target = _target(model_home)
    target.parent.mkdir(parents=True)
    (target.parent / "dict.txt.99999.tmp").write_bytes(b"half a dictionary")

    assert ensure_jieba_dictionary() is True

    assert sorted(p.name for p in target.parent.iterdir()) == ["dict.txt"]


def test_ensure_jieba_dictionary_cleans_up_a_failed_copy(
    model_home: Path, monkeypatch, caplog
):
    def exploding_copyfile(src, dst):
        Path(dst).write_bytes(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(jieba_dictionary.shutil, "copyfile", exploding_copyfile)

    with caplog.at_level(logging.WARNING):
        assert ensure_jieba_dictionary() is False

    assert list(_target(model_home).parent.iterdir()) == []


@pytest.mark.skipif(
    sys.platform != "win32" and os.geteuid() == 0, reason="root ignores mode bits"
)
def test_ensure_jieba_dictionary_survives_a_read_only_home(
    monkeypatch, tmp_path: Path, caplog
):
    home = tmp_path / "ro"
    home.mkdir(mode=0o500)
    monkeypatch.setenv("LANCE_LANGUAGE_MODEL_HOME", str(home))

    with caplog.at_level(logging.WARNING):
        assert ensure_jieba_dictionary() is False

    assert "will fail" in caplog.text


def test_ensure_jieba_dictionary_once_retries_after_a_failure(
    model_home: Path, monkeypatch
):
    monkeypatch.setattr(jieba_dictionary, "_installed", False)
    with monkeypatch.context() as broken:
        _break_jieba_import(broken)
        ensure_jieba_dictionary_once()
    assert not _target(model_home).exists()

    ensure_jieba_dictionary_once()

    assert _target(model_home).read_bytes() == SOURCE.read_bytes()


def test_default_model_home_on_windows_uses_localappdata(monkeypatch):
    """Lance resolves it with dirs::data_local_dir, which is LOCALAPPDATA."""
    monkeypatch.delenv("LANCE_LANGUAGE_MODEL_HOME", raising=False)
    monkeypatch.setattr(jieba_dictionary.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\tester\AppData\Local")

    assert (
        jieba_dictionary._language_model_home()
        == Path(r"C:\Users\tester\AppData\Local") / "lance" / "language_models"
    )


def test_default_model_home_follows_the_platform_data_directory(monkeypatch):
    monkeypatch.delenv("LANCE_LANGUAGE_MODEL_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    home = jieba_dictionary._language_model_home()

    if sys.platform == "darwin":
        assert home == Path.home() / "Library/Application Support/lance/language_models"
    elif sys.platform != "win32":
        assert home == Path.home() / ".local/share/lance/language_models"
