"""Security tests for skill_hub ZIP extraction and file normalisation guards."""

from __future__ import annotations

import io
import struct
import tracemalloc
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


# ── hostile-archive bounds (P2a) ──────────────────────────────────────────────


def _archive_with_padded_directory(*, entries: int, padding: int) -> bytes:
    """An archive whose central directory is inflated by per-entry extra fields.

    Names stay short and the count stays under the cap, so the only bound that
    can reject it is the one on bytes read during construction.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", SKILL_MD)
        for i in range(entries):
            info = zipfile.ZipInfo(f"f{i}.md")
            # 0xFFFF is an unassigned extra-field header id, so readers carry
            # the block along without interpreting it.
            info.extra = b"\xff\xff" + padding.to_bytes(2, "little") + b"\x00" * padding
            zf.writestr(info, b"x")
    return buf.getvalue()


def _zip64_archive_under_the_byte_cap(entries: int) -> bytes:
    """A genuine Zip64 archive whose directory stays under the byte cap.

    Assembled by hand: the stdlib only emits a Zip64 end record past 65,535
    entries, and such an archive's directory is far over the byte cap, so the
    byte bound answers it before the entry count is ever consulted. This shape
    -- short names, a small directory, a real Zip64 record made authoritative
    by pinning the legacy count to the sentinel -- is the one that exercises
    the Zip64 read path against the *entry* cap.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", SKILL_MD)
        for i in range(entries - 1):
            zf.writestr(str(i), b"")
    base = buf.getvalue()
    eocd_start = base.rfind(_EOCD_SIGNATURE)
    _sig, _d1, _d2, _eth, total, size_cd, offset_cd, _cl = struct.unpack(
        "<4s4H2LH", base[eocd_start : eocd_start + 22]
    )
    body = base[:eocd_start]
    eocd = bytearray(base[eocd_start:])
    zip64_eocd = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        total,
        total,
        size_cd,
        offset_cd,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, len(body), 1)
    struct.pack_into("<H", eocd, 10, 0xFFFF)
    return body + zip64_eocd + locator + bytes(eocd)


def _comment_boundary_archive() -> bytes:
    """An archive whose EOCD sits exactly 65536 bytes from EOF.

    CPython 3.11 searches ``filesize - 65536 - 22`` and 3.12+ searches
    ``filesize - 65535 - 22``, so this shape is findable on one and not the
    other. Any hand-written window had to pick a side and be wrong on the
    rest; the bounded reader never searches, so it is right on both.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("SKILL.md", SKILL_MD)
        for i in range(20000):
            zf.writestr(str(i), b"")
        zf.comment = b"C" * 65535
    return buf.getvalue() + b"X"


def _archive_with_entries(count: int, *, dirs: bool = False) -> bytes:
    """``count`` entries under one-character names, keeping the directory small.

    Short names are the point: an archive can then be far over the *entry* cap
    while its central directory stays under the *byte* cap, which is the shape
    that separates the two bounds.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", allowZip64=True) as zf:
        zf.writestr("SKILL.md", SKILL_MD)
        for i in range(count - 1):
            zf.writestr(str(i) + ("/" if dirs else ""), b"")
    return buf.getvalue()


def _with_eocd_count_decoy(zip_bytes: bytes, *, size_cd: int | None = None) -> bytes:
    """Rewrite the legacy EOCD counts to small, non-sentinel decoys.

    EOCD layout (``<4s4H2LH``): entries-this-disk at +8, entries-total at +10,
    central-directory size at +12.
    """
    raw = bytearray(zip_bytes)
    eocd = raw.rfind(_EOCD_SIGNATURE)
    struct.pack_into("<HH", raw, eocd + 8, 10, 10)
    if size_cd is not None:
        struct.pack_into("<L", raw, eocd + 12, size_cd)
    return bytes(raw)


# A self-extracting archive's stub: arbitrary bytes before the logical ZIP.
_STUB_PREFIX = b"MZ" + b"\x00" * 4094
_EOCD_SIGNATURE = b"PK\x05\x06"


class TestEntryCap:
    """The entry cap, and what it does and does not promise.

    It is applied to ``zf.infolist()`` after ``ZipFile`` has parsed the
    directory, so it bounds the per-entry work that follows -- one ORM insert
    per file, the walk over every member -- but *not* the construction itself.
    Bounding construction is a separate problem, tracked in #1941; these tests
    assert the cap that exists rather than one that does not.
    """

    @pytest.mark.parametrize(
        "label,build",
        [
            ("plain", lambda: _archive_with_entries(2001)),
            ("directories count too", lambda: _archive_with_entries(2001, dirs=True)),
            (
                "count decoyed in the EOCD",
                lambda: _with_eocd_count_decoy(_archive_with_entries(2001)),
            ),
            ("behind an SFX stub", lambda: _STUB_PREFIX + _archive_with_entries(2001)),
            (
                "zip64, directory under the old byte cap",
                lambda: _zip64_archive_under_the_byte_cap(2001),
            ),
        ],
    )
    def test_over_the_cap_is_refused(self, label, build):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(build(), bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "entries" in exc.value.detail

    @pytest.mark.parametrize(
        "label,build",
        [
            ("exactly at the cap", lambda: _archive_with_entries(2000)),
            (
                "ordinary bundle",
                lambda: _make_zip({"SKILL.md": SKILL_MD, "n.md": b"h"}),
            ),
            (
                "behind an SFX stub",
                lambda: _STUB_PREFIX + _make_zip({"SKILL.md": SKILL_MD, "n.md": b"h"}),
            ),
            ("zip64 at the cap", lambda: _zip64_archive_under_the_byte_cap(2000)),
        ],
    )
    def test_within_the_cap_is_accepted(self, label, build):
        files, _root = _safe_zip_extract(build(), bad_zip_status=400)
        assert files["SKILL.md"] == SKILL_MD

    def test_valid_comments_do_not_affect_the_count(self):
        """A comment is not entries, whatever it contains.

        Worth keeping even though nothing counts byte patterns any more: an
        earlier revision bounded the count by scanning the byte stream for
        central-header signatures, and rejected a valid 2,000-entry archive
        carrying a one-byte comment (the end-record scan overlaps the
        directory, so the same headers were counted twice) as well as a
        one-entry archive whose comment legally contained 2,001 copies of that
        signature.
        """
        for comment in (b"x", b"C" * 65535, b"PK\x01\x02" * 2001):
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("SKILL.md", SKILL_MD)
                for i in range(1999):
                    zf.writestr(str(i), b"")
                zf.comment = comment
            data = buf.getvalue()
            assert len(zipfile.ZipFile(io.BytesIO(data)).namelist()) == 2000
            files, _root = _safe_zip_extract(data, bad_zip_status=400)
            assert files["SKILL.md"] == SKILL_MD

    def test_signatures_inside_member_content_do_not_affect_the_count(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "decoy.bin": b"PK\x01\x02" * 100000})
        files, _root = _safe_zip_extract(data, bad_zip_status=400)
        assert len(files["decoy.bin"]) == 400000

    def test_non_zip_is_left_to_zipfile_to_reject(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(b"not a zip", bad_zip_status=400)
        assert exc.value.status_code == 400
        assert "readable ZIP" in exc.value.detail


class TestCanonicalizationAndDuplicates:
    """N2 — canonicalize before root selection; refuse duplicates."""

    def test_dot_slash_prefix_does_not_lose_the_root(self):
        """ "./SKILL.md" must count as depth 0, not lose to a nested root."""
        data = _make_zip(
            {"./SKILL.md": SKILL_MD, "nested/SKILL.md": b"# wrong", "./notes.md": b"n"}
        )
        files, root = _safe_zip_extract(data)
        assert files["SKILL.md"] == SKILL_MD
        assert "notes.md" in files
        assert root == ""

    def test_backslash_member_stays_inside_the_root(self):
        """A Windows-written path must not fall outside the containment check."""
        data = _make_zip({"my-skill/SKILL.md": SKILL_MD, "my-skill\\notes.md": b"kept"})
        files, root = _safe_zip_extract(data)
        assert root == "my-skill"
        # Before canonicalization moved ahead of root selection this file did
        # not start with "my-skill/" and was silently dropped, with a 200.
        assert files["notes.md"] == b"kept"

    def test_duplicate_canonical_paths_in_zip_rejected(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "./SKILL.md": b"# other"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400
        assert "two entries" in exc.value.detail

    def test_duplicate_via_backslash_rejected(self):
        data = _make_zip({"a/b.md": b"one", "a\\b.md": b"two", "SKILL.md": SKILL_MD})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data)
        assert exc.value.status_code == 400

    def test_duplicate_canonical_paths_in_normalizer_rejected(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files(
                {"SKILL.md": SKILL_MD, "docs/a.md": b"one", "docs//a.md": b"two"}
            )
        assert exc.value.status_code == 400
        assert "two entries" in exc.value.detail

    def test_redundant_separators_collapse(self):
        result = _normalize_skill_files({"SKILL.md": SKILL_MD, "a//./b.md": b"hi"})
        assert result["a/b.md"] == b"hi"

    @pytest.mark.parametrize("raw", ["///", ".", "./", "/", "\\\\"])
    def test_degenerate_paths_never_reach_the_bundle(self, raw):
        """A path that canonicalizes to nothing must be refused, not stored.

        Asserted on the resulting keys as well as the status. Without the
        check the empty string is written as a key and the bundle is returned
        with it -- a shape no consumer expects -- and a status-only assertion
        cannot see that, because a second degenerate entry would be caught by
        the duplicate check and a single one is simply accepted.
        """
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, raw: b"x"})
        assert exc.value.status_code == 400
        assert "empty file path" in exc.value.detail

    def test_a_bundle_never_carries_an_empty_key(self):
        """The invariant behind the check above, stated directly."""
        result = _normalize_skill_files({"SKILL.md": SKILL_MD, "docs/a.md": b"a"})
        assert all(key for key in result), result
        assert set(result) == {"SKILL.md", "docs/a.md"}

    def test_traversal_still_refused_not_resolved(self):
        """Canonicalization must not quietly resolve ".." into somewhere else."""
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, "a/../../x.md": b"e"})
        assert exc.value.status_code == 400
        assert "traversal" in exc.value.detail.lower()


class TestSkillMdContentBounds:
    """N7 / H4 / H5 — the shared preparation boundary's content contract."""

    @pytest.mark.parametrize("body", [b"", b"   ", b"\n\t \r\n"])
    def test_empty_skill_md_rejected(self, body):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": body})
        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    def test_empty_skill_md_rejected_from_zip(self):
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(_make_zip({"SKILL.md": b"  \n"}))
        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    def test_oversized_skill_md_rejected(self):
        from xagent.web.api.skill_hub import _MAX_SKILL_MD_CHARS

        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": b"x" * (_MAX_SKILL_MD_CHARS + 1)})
        assert exc.value.status_code == 400
        assert str(_MAX_SKILL_MD_CHARS) in exc.value.detail

    def test_oversized_skill_md_is_rejected_without_decoding_it(self):
        """Finding #3: the rejection must not cost a full decoded copy.

        Asserted on peak allocation, not on the status code. The character cap
        returns the same 400 either way, so a status-only test stays green
        with the byte bound deleted -- measured, 120 MiB instead of 80 MiB.
        """
        payload = b"a" * (40 * 1024 * 1024)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", payload)
        data = buf.getvalue()
        # A high-compression archive: tiny on the wire, huge once inflated.
        assert len(data) < 1024 * 1024

        tracemalloc.start()
        try:
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(data, bad_zip_status=400)
            _peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert exc.value.status_code == 400
        # The inflated member and the joined copy are unavoidable here; a
        # third full-size string is not. Threshold measured, not guessed: with
        # the byte bound the peak is ~80 MiB, without it 120 MiB, so it has to
        # sit between them. The first version of this used 3.5x (140 MiB) --
        # above *both* -- and could not fail.
        assert _peak < 2.5 * len(payload), (
            f"peak {_peak / 1048576:.1f} MiB suggests SKILL.md was decoded "
            "before the byte bound rejected it"
        )

    def test_skill_md_at_the_cap_is_accepted(self):
        from xagent.web.api.skill_hub import _MAX_SKILL_MD_CHARS

        result = _normalize_skill_files({"SKILL.md": b"x" * _MAX_SKILL_MD_CHARS})
        assert len(result["SKILL.md"]) == _MAX_SKILL_MD_CHARS

    def test_skill_md_cap_counts_characters_not_bytes(self):
        """Matches the Create/Edit schemas, which bound characters."""
        from xagent.web.api.skill_hub import _MAX_SKILL_MD_CHARS

        # 3 bytes per character in UTF-8: over the byte count, under the cap.
        body = ("€" * _MAX_SKILL_MD_CHARS).encode("utf-8")
        assert len(body) == 3 * _MAX_SKILL_MD_CHARS
        assert _normalize_skill_files({"SKILL.md": body})["SKILL.md"] == body

    def test_oversized_template_md_is_also_bounded(self):
        """The same byte bound applies to every parser-decoded key.

        ``template.md`` is decoded in full by ``parse_bundle`` exactly as
        ``SKILL.md`` is, so leaving it unbounded reproduced the allocation
        spike SKILL.md had just been fixed for -- 120 MiB peak for a 41 KiB
        archive.
        """
        payload = b"t" * (40 * 1024 * 1024)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("SKILL.md", SKILL_MD)
            zf.writestr("template.md", payload)
        data = buf.getvalue()
        assert len(data) < 1024 * 1024

        tracemalloc.start()
        try:
            with pytest.raises(HTTPException) as exc:
                _safe_zip_extract(data, bad_zip_status=400)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert exc.value.status_code == 400
        assert "template.md" in exc.value.detail
        assert peak < 2.5 * len(payload), f"peak {peak / 1048576:.1f} MiB"

    @pytest.mark.parametrize(
        "label,body",
        [
            ("ascii spaces", "   \t\n"),
            ("ideographic space U+3000", "\u3000" * 9),
            ("no-break space U+00A0", "\u00a0" * 9),
            ("zero-width space U+200B", "\u200b" * 9),
            ("zero-width non-joiner U+200C", "\u200c" * 9),
            ("word joiner U+2060", "\u2060" * 9),
            ("bare BOM", "\ufeff"),
            ("mixed invisible", "  \ufeff\u200b\u3000\n\t"),
        ],
    )
    def test_invisible_only_skill_md_rejected(self, label, body):
        """``bytes.strip()`` knew only ASCII, so all of these read as content.

        The zero-width ones are not Unicode whitespace either -- ``isspace()``
        is False for every one of them -- so ``str.strip()`` alone does not
        cover them, but a SKILL.md made of them carries no more meaning than
        an empty one.
        """
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": body.encode("utf-8")})
        assert exc.value.status_code == 400
        assert "empty" in exc.value.detail.lower()

    @pytest.mark.parametrize(
        "label,body",
        [
            ("ascii", "# Hi\n\n## Description\nd\n"),
            ("cjk", "# 技能\n\n## Description\nd\n"),
            ("content behind a BOM", "\ufeff# Hi\n\n## Description\nd\n"),
            ("emoji", "😀"),
        ],
    )
    def test_real_content_is_not_mistaken_for_blank(self, label, body):
        """The other direction: widening the strip must not eat real content."""
        raw = body.encode("utf-8")
        assert _normalize_skill_files({"SKILL.md": raw})["SKILL.md"] == raw

    def test_overlong_path_rejected(self):
        from xagent.web.api.skill_hub import _MAX_SKILL_FILE_PATH_CHARS

        long_path = "d/" + "n" * _MAX_SKILL_FILE_PATH_CHARS + ".md"
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files({"SKILL.md": SKILL_MD, long_path: b"x"})
        assert exc.value.status_code == 400
        assert str(_MAX_SKILL_FILE_PATH_CHARS) in exc.value.detail

    def test_path_at_the_cap_is_accepted(self):
        from xagent.web.api.skill_hub import _MAX_SKILL_FILE_PATH_CHARS

        path = "n" * (_MAX_SKILL_FILE_PATH_CHARS - 3) + ".md"
        assert len(path) == _MAX_SKILL_FILE_PATH_CHARS
        assert path in _normalize_skill_files({"SKILL.md": SKILL_MD, path: b"x"})


class TestParseGateIsWired:
    """The parser check must guard the shared write boundary, not just exist.

    Asserted through ``_normalize_skill_files`` -- the function every write
    path goes through -- rather than by calling the helper directly. A helper
    that is defined, tested and never called is what the previous round
    shipped: an unparsable bundle reached persistence and surfaced as a
    post-write 500 with an orphaned row instead of a pre-write 400.
    """

    def test_unparsable_template_is_refused_by_the_normalizer(self):
        with pytest.raises(HTTPException) as exc:
            _normalize_skill_files(
                {"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe bad"}
            )
        assert exc.value.status_code == 400
        assert "template.md" in exc.value.detail

    def test_unparsable_bundle_is_refused_before_extraction_returns(self):
        data = _make_zip({"SKILL.md": SKILL_MD, "template.md": b"\xff\xfe bad"})
        with pytest.raises(HTTPException) as exc:
            _safe_zip_extract(data, bad_zip_status=400)
        assert exc.value.status_code == 400

    def test_valid_bundle_still_passes_the_gate(self):
        result = _normalize_skill_files(
            {"SKILL.md": SKILL_MD, "assets/logo.png": bytes(range(200, 256))}
        )
        assert set(result) == {"SKILL.md", "assets/logo.png"}

    def test_every_builtin_skill_still_normalizes(self):
        """Calibration: the gate must not reject skills already shipping."""
        import pathlib

        roots = sorted(pathlib.Path("src/xagent/skills/builtin").glob("*/SKILL.md"))
        assert roots, "no builtin skills found -- fixture path is wrong"
        for skill_md in roots:
            directory = skill_md.parent
            files = {
                str(item.relative_to(directory)): item.read_bytes()
                for item in directory.rglob("*")
                if item.is_file()
            }
            assert _normalize_skill_files(files)["SKILL.md"]


class TestParseCulpritSelection:
    """N8 — blame only the files the parser actually decodes."""

    def test_invalid_template_is_named_not_a_valid_png(self):
        from xagent.web.api.skill_hub import _assert_bundle_parses

        png = b"\x89PNG\r\n\x1a\n" + bytes(range(200, 256))
        # The fixture only tests anything if it really is non-UTF-8.
        with pytest.raises(UnicodeDecodeError):
            png.decode("utf-8")
        with pytest.raises(HTTPException) as exc:
            _assert_bundle_parses(
                {
                    "SKILL.md": SKILL_MD,
                    "assets/logo.png": png,
                    "template.md": b"\xff\xfe bad",
                }
            )
        assert exc.value.status_code == 400
        assert "template.md" in exc.value.detail
        assert "logo.png" not in exc.value.detail

    def test_invalid_skill_md_is_named(self):
        from xagent.web.api.skill_hub import _assert_bundle_parses

        with pytest.raises(HTTPException) as exc:
            _assert_bundle_parses({"SKILL.md": b"\xff\xfe"})
        assert exc.value.status_code == 400
        assert "SKILL.md" in exc.value.detail

    def test_binary_attachments_alone_do_not_fail_the_parse(self):
        """A non-UTF-8 PNG is ordinary bundle content, not an error."""
        from xagent.web.api.skill_hub import _assert_bundle_parses

        _assert_bundle_parses(
            {"SKILL.md": SKILL_MD, "assets/logo.png": bytes(range(200, 256))}
        )


class TestParserKeySynchronization:
    """``_PARSER_DECODED_FILES`` must track what the parser actually reads.

    The tuple drives both the pre-decode byte bound and the UTF-8 culprit
    message, so a file the parser decodes without an entry here silently skips
    the size guard and is mis-named in errors.

    Asserted by *running* the parser against a recording mapping rather than
    by reading its source. An AST scan only sees literal ``files["x"]`` /
    ``files.get("x")`` shapes, so a refactor through a local variable, a
    helper, or a mapping alias would leave it green while the invariant broke.
    """

    def test_tuple_matches_every_key_the_parser_reads(self):
        from xagent.skills.parser import SkillParser
        from xagent.web.api.skill_hub import _PARSER_DECODED_FILES

        class _Recorder(dict):
            """A bundle that remembers which keys were asked for."""

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seen: list[str] = []

            def __getitem__(self, key):
                self.seen.append(key)
                return super().__getitem__(key)

            def __contains__(self, key):
                self.seen.append(key)
                return super().__contains__(key)

            def get(self, key, default=None):
                self.seen.append(key)
                return super().get(key, default)

        # Every declared file present, so the parser reaches all of them
        # rather than short-circuiting on a missing one.
        files = _Recorder({name: SKILL_MD for name in _PARSER_DECODED_FILES})
        SkillParser.parse_bundle(name="probe", files=files)

        # Only keys that name a bundle *file* matter; the parser may also look
        # up unrelated things. Compare against what it asked for.
        asked = [key for key in files.seen if isinstance(key, str)]
        assert asked, "the recorder saw no key lookups -- the probe is broken"
        assert set(asked) == set(_PARSER_DECODED_FILES), (
            f"SkillParser.parse_bundle reads {sorted(set(asked))} but "
            f"_PARSER_DECODED_FILES is {sorted(_PARSER_DECODED_FILES)}. A file "
            "the parser reads without an entry here skips the pre-decode size "
            "bound and is mis-named in UTF-8 errors."
        )

    def test_every_declared_file_is_actually_bounded(self):
        """The consumer side: each declared file gets the byte bound.

        Pairs with the invariant above -- that one keeps the tuple honest
        about the parser, this one keeps the bound honest about the tuple.
        """
        from xagent.web.api.skill_hub import (
            _MAX_SKILL_MD_CHARS,
            _MAX_UTF8_BYTES_PER_CHAR,
            _PARSER_DECODED_FILES,
            _assert_bundle_parses,
        )

        oversized = b"x" * (_MAX_UTF8_BYTES_PER_CHAR * _MAX_SKILL_MD_CHARS + 1)
        for name in _PARSER_DECODED_FILES:
            files = {"SKILL.md": SKILL_MD, name: oversized}
            with pytest.raises(HTTPException) as exc:
                _assert_bundle_parses(files)
            assert exc.value.status_code == 400
            assert name in exc.value.detail
