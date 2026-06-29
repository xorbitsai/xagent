"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import (
    _check_registry_security_gate,
    _normalize_skill_files,
    _safe_zip_to_files,
)

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


# ── _check_registry_security_gate ────────────────────────────────────────────


def _make_registry(display_name: str = "TestHub"):
    """Minimal registry stub with a ClawHub-compatible extract_scan_status."""
    from types import SimpleNamespace

    def extract_scan_status(raw_item):
        latest = raw_item.get("latestVersion") or {}
        security = latest.get("security") or {}
        return security.get("status") if isinstance(security, dict) else None

    return SimpleNamespace(
        display_name=display_name, extract_scan_status=extract_scan_status
    )


def _detail(*, scan_status=None, moderation_state=None):
    """Build a fake registry detail payload."""
    d = {}
    if scan_status is not None:
        d["latestVersion"] = {"security": {"status": scan_status}}
    if moderation_state is not None:
        d["moderation"] = {"moderationState": moderation_state}
    return d


class TestCheckRegistrySecurityGate:
    def test_malicious_scan_status_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(scan_status="malicious")
            )
        assert exc.value.status_code == 403
        assert "malicious" in exc.value.detail.lower()

    def test_quarantined_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(moderation_state="quarantined")
            )
        assert exc.value.status_code == 403
        assert "quarantined" in exc.value.detail.lower()

    def test_revoked_refused(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(), _detail(moderation_state="revoked")
            )
        assert exc.value.status_code == 403
        assert "revoked" in exc.value.detail.lower()

    def test_clean_scan_status_allowed(self):
        # Must not raise.
        _check_registry_security_gate(_make_registry(), _detail(scan_status="clean"))

    def test_suspicious_scan_status_allowed(self):
        # "suspicious" is a warning, not a hard block.
        _check_registry_security_gate(
            _make_registry(), _detail(scan_status="suspicious")
        )

    def test_no_security_data_allowed(self):
        # Missing keys → None scan status → gate passes.
        _check_registry_security_gate(_make_registry(), {})

    def test_both_signals_malicious_wins(self):
        with pytest.raises(HTTPException) as exc:
            _check_registry_security_gate(
                _make_registry(),
                _detail(scan_status="malicious", moderation_state="quarantined"),
            )
        assert exc.value.status_code == 403
        assert "malicious" in exc.value.detail.lower()
