"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import _normalize_skill_files, _safe_zip_to_files

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_zip(members: dict[str, bytes]) -> bytes:
    """Build an in-memory ZIP from a {filename: content} dict."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


SKILL_MD = b"# Test Skill\n\n## Description\nA test skill.\n"


# ── _normalize_skill_files ────────────────────────────────────────────────────


class TestNormalizeSkillFiles:
    def test_happy_path(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD})
        assert result == {"SKILL.md": SKILL_MD}

    def test_missing_skill_md_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"other.txt": b"data"})
        assert exc.value.status_code == 400
        assert "SKILL.md" in exc.value.detail

    def test_path_traversal_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "../escape.py": b"x"})
        assert exc.value.status_code == 400
        assert "traversal" in exc.value.detail.lower()

    def test_dotfile_raises(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, ".env": b"SECRET=1"})
        assert exc.value.status_code == 400
        assert "dot" in exc.value.detail.lower()

    def test_absolute_path_stripped(self):
        result = _normalize_skill_files({"/SKILL.md": SKILL_MD})
        assert "SKILL.md" in result

    def test_windows_separator_normalised(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD, "sub\\file.md": b"hi"})
        assert "sub/file.md" in result

    def test_size_cap_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "big.bin": big})
        assert exc.value.status_code == 413


# ── _safe_zip_to_files ────────────────────────────────────────────────────────


class TestSafeZipToFiles:
    def test_happy_path_flat(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "template.md": b"# Template"})
        result = _safe_zip_to_files(data)
        assert "SKILL.md" in result
        assert "template.md" in result

    def test_happy_path_nested(self):
        """ZIP with a top-level directory wrapper."""
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/extra.md": b"hi"})
        result = _safe_zip_to_files(data)
        assert "SKILL.md" in result
        assert "extra.md" in result

    def test_bad_zip_raises(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(b"not a zip")
        assert exc.value.status_code == 502

    def test_missing_skill_md_raises(self):
        data = _make_zip({"README.md": b"hello"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400
        assert "SKILL.md" in exc.value.detail

    def test_path_traversal_in_zip_raises(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"evil"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400

    def test_dotfile_in_zip_rejected_by_normalize(self):
        data = _make_zip({"SKILL.md": SKILL_MD, ".env": b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 400

    def test_oversized_member_raises(self):
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        big = b"x" * (_MAX_DOWNLOAD_BYTES + 1)
        data = _make_zip({"SKILL.md": SKILL_MD, "large.bin": big})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(data)
        assert exc.value.status_code == 413
