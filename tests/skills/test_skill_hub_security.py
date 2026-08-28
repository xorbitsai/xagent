"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import zipfile

import pytest
from fastapi import HTTPException

from xagent.web.api.skill_hub import (
    _check_registry_security_gate,
    _normalize_skill_files,
    _safe_zip_extract,
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
        assert "hidden file" in exc.value.detail
        assert ".env" in exc.value.detail

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


# ── _safe_zip_extract ─────────────────────────────────────────────────────────


def _tamper_declared_size(zip_bytes: bytes, declared: int) -> bytes:
    """Rewrite the uncompressed-size field of the first member in both
    the local header and the central directory."""
    import struct

    data = bytearray(zip_bytes)
    local = data.find(b"PK\x03\x04")
    data[local + 22 : local + 26] = struct.pack("<I", declared)
    central = data.find(b"PK\x01\x02")
    data[central + 24 : central + 28] = struct.pack("<I", declared)
    return bytes(data)


def _deflate_zip_with_corrupt_member() -> bytes:
    """A DEFLATE member whose compressed stream has been overwritten."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("SKILL.md", b"B" * 20000)
    data = bytearray(buf.getvalue())
    local = data.find(b"PK\x03\x04")
    header_len = 30 + data[local + 26] + data[local + 28]
    for offset in range(header_len, header_len + 8):
        data[local + offset] = 0xAB
    return bytes(data)


class TestSafeZipExtract:
    def test_returns_nested_root_name(self):
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill/ref.md": b"hi"})
        files, root = _safe_zip_extract(data)
        assert root == "my-skill"
        assert set(files) == {"SKILL.md", "ref.md"}

    def test_returns_empty_root_for_flat_zip(self):
        data = _make_zip({"SKILL.md": SKILL_MD})
        files, root = _safe_zip_extract(data)
        assert root == ""
        assert "SKILL.md" in files

    def test_bad_zip_status_is_configurable(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip", bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_error_wording_is_source_neutral(self):
        # The extractor serves registry installs today and an upload route
        # next, so its messages must not name a specific registry.
        for build in (
            lambda: _safe_zip_extract(b"not a zip"),
            lambda: _safe_zip_extract(_make_zip({"README.md": b"no skill"})),
        ):
            with pytest.raises(HTTPException) as exc:
                build()
            assert "clawhub" not in exc.value.detail.lower()

    def test_decompression_bomb_is_refused_without_inflating_it(self):
        """A bomb must be rejected without its expansion ever being held.

        Reading the whole remaining budget in one call holds the result and
        zipfile's own growing buffer at the same time, so an ~800 KiB archive
        peaked at twice the 50 MiB budget. The declared size stops it before
        any inflation, and members are read in slices so the overshoot on a
        lying header is one chunk rather than the whole budget.
        """
        import tracemalloc

        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            for i in range(8):
                zf.writestr(f"bomb{i}.bin", b"\0" * (_MAX_DOWNLOAD_BYTES * 2))
        data = buf.getvalue()
        assert len(data) < 4 * 1024 * 1024  # tiny on the wire, huge inflated

        tracemalloc.start()
        try:
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(data)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert exc.value.status_code == 413
        # Generous bound: the point is "not a multiple of the budget".
        assert peak < _MAX_DOWNLOAD_BYTES / 2, f"peak {peak} bytes"

    def test_over_declared_member_is_refused_on_its_header(self):
        """An over-declared member is refused before it is inflated.

        The declared size is the cheap first gate. Trusting it costs an
        archive that under-states a small file nothing real -- writers emit
        truthful headers -- while inflating first to find out costs the
        whole budget per bomb, which is the shape an attacker can actually
        produce. The real byte count is still checked as it is read.
        """
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        data = _make_zip({"SKILL.md": SKILL_MD})
        lying = _tamper_declared_size(data, _MAX_DOWNLOAD_BYTES + 1)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(lying)
        assert exc.value.status_code == 413

    def test_honest_members_within_budget_are_read_whole(self):
        # The chunked read must reassemble a member larger than one chunk
        # byte-for-byte, not truncate it at the chunk boundary.
        from xagent.web.api.skill_hub import _ARCHIVE_CHUNK_BYTES

        body = bytes(range(256)) * ((_ARCHIVE_CHUNK_BYTES * 2) // 256 + 7)
        data = _make_zip({"SKILL.md": SKILL_MD, "big.bin": body})
        files, _root = _safe_zip_extract(data)
        assert files["big.bin"] == body


# ── archive root selection ───────────────────────────────────────────────────


class TestArchiveRootSelection:
    """The root is the shallowest SKILL.md, not the alphabetically first.

    A skill folder that ships its own ``Examples/SKILL.md`` sorts before the
    real root marker, so a plain sort imported the *example* as the skill —
    named "Examples", with the true root's files silently dropped and a 200
    returned. Ordinary shape, not an adversarial one.
    """

    def test_subfolder_does_not_beat_the_true_root(self):
        data = _make_zip(
            {
                "SKILL.md": SKILL_MD,
                "reference.md": b"reference material",
                "Examples/SKILL.md": b"---\ndescription: an example\n---\n# Ex\n",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == ""
        # The real skill wins and nothing is discarded.
        assert files["SKILL.md"] == SKILL_MD
        assert "reference.md" in files
        assert "Examples/SKILL.md" in files

    def test_wrapper_directory_still_resolves(self):
        data = _make_zip(
            {
                "pdf-tools/SKILL.md": SKILL_MD,
                "pdf-tools/ref.md": b"r",
                "pdf-tools/examples/SKILL.md": b"---\ndescription: e\n---\n# E\n",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md", "examples/SKILL.md", "ref.md"]

    def test_two_roots_at_the_same_depth_are_refused(self):
        data = _make_zip({"a/SKILL.md": SKILL_MD, "b/SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "multiple skills" in exc.value.detail

    def test_multiple_roots_reported_without_reflecting_the_whole_archive(self):
        """The rejection names a bounded sample, not every root it found.

        The names come from the archive, so an unbounded list would let a
        crafted upload reflect arbitrary attacker text back through the error.
        """
        long_name = "z" * 500
        data = _make_zip(
            {f"root{i}/SKILL.md": SKILL_MD for i in range(9)}
            | {f"{long_name}/SKILL.md": SKILL_MD}
        )
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert len(exc.value.detail) < 400
        assert long_name not in exc.value.detail
        assert "5 more" in exc.value.detail

    def test_cruft_is_not_a_root_candidate(self):
        """Cruft must be excluded before candidates are compared.

        At the *same* depth as the real skill it would otherwise look like a
        second skill and trigger a spurious "multiple skills" rejection — which
        is what a Finder-zipped folder actually produces. A deeper __MACOSX
        would be beaten by depth alone, so that shape proves nothing here.
        """
        data = _make_zip(
            {
                "pdf-tools/SKILL.md": SKILL_MD,
                "__MACOSX/SKILL.md": b"junk",
            }
        )
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert files["SKILL.md"] == SKILL_MD

    def test_cruft_alone_is_not_a_skill(self):
        data = _make_zip({"__MACOSX/SKILL.md": b"junk"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "no SKILL.md" in exc.value.detail


# ── unreadable archives ──────────────────────────────────────────────────────


class TestUnreadableArchives:
    """Anything zipfile raises on an untrusted archive must be a 4xx/5xx.

    Guarded at one boundary rather than by naming exception types, because the
    per-type list kept missing cases: a tampered end-of-central-directory
    offset raises ValueError from the constructor, well before any member read.
    """

    def test_tampered_eocd_offset(self):
        import struct

        raw = bytearray(_make_zip({"s/SKILL.md": SKILL_MD}))
        eocd = raw.rfind(b"PK\x05\x06")
        struct.pack_into("<I", raw, eocd + 16, 0xFFFFFFF0)
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(bytes(raw), bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_truncated_central_directory(self):
        raw = _make_zip({"s/SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(raw[: len(raw) // 2], bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_corrupt_deflate_member(self):
        # zipfile only detects the damage while reading the member, not at
        # open(): zlib.error subclasses neither BadZipFile nor RuntimeError,
        # so it used to escape the extractor as a 500.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_deflate_zip_with_corrupt_member(), bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_registry_source_keeps_its_status(self):
        # The boundary guard must not flatten the caller-specific status.
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip at all")
        assert exc.value.status_code == 502

    def test_corrupt_member_keeps_the_registry_status(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_to_files(_deflate_zip_with_corrupt_member())
        assert exc.value.status_code == 502

    def test_size_budget_still_reported_as_413(self):
        # Our own HTTPExceptions must pass through the guard unchanged, or
        # every rejection raised inside the try block collapses onto one code.
        from xagent.web.api.skill_hub import _MAX_DOWNLOAD_BYTES

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("big.bin", b"\0" * (_MAX_DOWNLOAD_BYTES + 1024))
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(buf.getvalue(), bad_zip_status=400)
        assert exc.value.status_code == 413

    def test_traversal_still_reported_as_400(self):
        # The other in-try rejection: 400 here and 413 above must stay
        # distinct even when the caller's bad-ZIP status is 502.
        data = _make_zip({"SKILL.md": SKILL_MD, "../escape.py": b"x"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400


# ── macOS archive cruft ──────────────────────────────────────────────────────


class TestArchiveCruft:
    """Zipping a folder in Finder sweeps in .DS_Store and resource forks.

    A skill folder that has been opened in Finder carries them, so refusing
    the whole archive over one would fail an entirely ordinary bundle —
    while real hidden files stay refused.
    """

    @pytest.mark.parametrize(
        "cruft",
        [
            "pdf-tools/.DS_Store",
            "pdf-tools/refs/.DS_Store",
            "__MACOSX/._pdf-tools",
            "pdf-tools/._reference.md",
            "pdf-tools/Thumbs.db",
            "pdf-tools/desktop.ini",
        ],
    )
    def test_cruft_is_dropped_not_rejected(self, cruft):
        data = _make_zip({"pdf-tools/SKILL.md": SKILL_MD, cruft: b"\x00\x01"})
        files, root = _safe_zip_extract(data)
        assert root == "pdf-tools"
        assert sorted(files) == ["SKILL.md"]

    @pytest.mark.parametrize("hidden", [".env", "sub/.env", "a/b/.secret"])
    def test_real_hidden_files_still_rejected(self, hidden):
        # The old check only looked at the first character of the whole path,
        # so ".env" was refused but "sub/.env" was not.
        data = _make_zip({"SKILL.md": SKILL_MD, hidden: b"SECRET=1"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "hidden file" in exc.value.detail


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("route", "expected_method"),
    [
        ("create", "create_skill"),
        ("update", "update_skill_file"),
        ("delete", "delete_skill"),
        ("registry_install", "create_skill"),
    ],
)
async def test_team_write_routes_adopt_the_central_provider_invoker(
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    expected_method: str,
) -> None:
    from dataclasses import fields
    from types import SimpleNamespace

    from xagent.skills.library import SkillScopeContext
    from xagent.web.api import skill_hub

    captured: list[tuple[str, object, dict]] = []
    scope = SkillScopeContext(user_id=7, metadata={"source_id": 11})

    class _Manager:
        async def get_skill(self, name: str):
            return {"name": name, "scope": "team", "path": ""}

    async def _central_invoker(provider, method, context, **kwargs):
        captured.append((method, context, kwargs))

    registry = SimpleNamespace(
        id="testhub",
        display_name="TestHub",
        get_skill=lambda slug: {},
        extract_scan_status=lambda detail: None,
        download_skill=lambda slug, version: (200, _make_zip({"SKILL.md": SKILL_MD})),
    )
    monkeypatch.setattr(skill_hub, "invoke_skill_write_provider", _central_invoker)
    monkeypatch.setattr(
        "xagent.skills.library.get_skill_write_provider", lambda: object()
    )

    async def _scoped_manager(*args):
        return _Manager()

    monkeypatch.setattr(skill_hub, "_get_scoped_manager", _scoped_manager)
    monkeypatch.setattr(skill_hub, "get_registry", lambda source: registry)

    request = SimpleNamespace()
    user = SimpleNamespace(id=7)
    if route == "create":
        await skill_hub.create_skill(
            skill_hub.CreateSkillRequest(
                name="writer", skill_md="# writer", scope="team"
            ),
            request,
            scope,
            object(),
            user,
        )
    elif route == "update":
        await skill_hub.edit_installed(
            "writer",
            skill_hub.EditSkillRequest(skill_md="# updated"),
            request,
            scope,
            object(),
            user,
        )
    elif route == "delete":
        await skill_hub.delete_installed("writer", request, scope, object(), user)
    else:
        await skill_hub.install_skill(
            "testhub",
            skill_hub.InstallSkillRequest(slug="writer", scope="team"),
            request,
            scope,
            object(),
            user,
        )

    assert len(captured) == 1
    method, context, kwargs = captured[0]
    assert method == expected_method
    assert {field.name for field in fields(context)} == {"user_id", "metadata"}
    assert context.user_id == scope.user_id
    assert context.metadata == scope.metadata
    assert kwargs["scope"] == "team"
