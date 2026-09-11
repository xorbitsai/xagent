import json
import os
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import onedrive


class MockResponse:
    def __init__(self, json_data=None, status_code=200, content=b"{}", url=None):
        self._json_data = json_data if json_data is not None else {}
        self.status_code = status_code
        self.content = content
        self.text = content.decode("utf-8", errors="replace")
        # Match the URL-bearing shape of real requests errors.
        self.url = url or "https://upload.example/session-default"

    def json(self):
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error: Error for url: {self.url}",
                response=self,
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AUTH_TOKEN", "test-graph-token")


@pytest.fixture(autouse=True)
def _upload_allowed_dirs_env(tmp_path, monkeypatch):
    """Scope onedrive_upload_file's read allowlist to an isolated per-test
    directory so tests aren't order-dependent on whatever the real working
    directory holds."""
    allowed_dir = tmp_path / "workspace"
    allowed_dir.mkdir()
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", str(allowed_dir))
    return allowed_dir


# ---------------------------------------------------------------------------
# onedrive_upload_file
# ---------------------------------------------------------------------------


def test_upload_file_sends_real_binary_content(monkeypatch, _upload_allowed_dirs_env):
    """Regression guard for the actual production bug: uploading an
    already-generated spreadsheet must send its real bytes with a real
    mimeType — not a text/plain placeholder string, which is all
    onedrive_upload_text_file's str content parameter can carry."""
    local_file = _upload_allowed_dirs_env / "Regional_Performance_Data-v3.xlsx"
    local_file.write_bytes(b"PK\x03\x04 fake xlsx bytes")

    mock_request = Mock(
        return_value=MockResponse(
            {
                "id": "item-1",
                "name": "Regional_Performance_Data-v3.xlsx",
                "file": {
                    "mimeType": (
                        "application/vnd.openxmlformats-officedocument"
                        ".spreadsheetml.sheet"
                    )
                },
            }
        )
    )
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert result["item"]["name"] == "Regional_Performance_Data-v3.xlsx"

    kwargs = mock_request.call_args.kwargs
    assert kwargs["method"] == "PUT"
    assert kwargs["url"].endswith(
        "/me/drive/root:/Regional_Performance_Data-v3.xlsx:/content"
    )
    assert kwargs["data"] == b"PK\x03\x04 fake xlsx bytes"
    assert kwargs["headers"]["Content-Type"] == (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert kwargs["timeout"] == onedrive._BINARY_UPLOAD_TIMEOUT_SECONDS


def test_upload_file_accepts_explicit_remote_path_and_mime_type(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "data.bin"
    local_file.write_bytes(b"\x00\x01\x02")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file),
            remote_path="Sandbox/custom.dat",
            mime_type="application/octet-stream",
        )
    )

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["url"].endswith("/me/drive/root:/Sandbox/custom.dat:/content")
    assert kwargs["headers"]["Content-Type"] == "application/octet-stream"


def test_upload_file_defaults_mime_type_when_unguessable(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "mystery_file_no_extension"
    local_file.write_bytes(b"some bytes")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    kwargs = mock_request.call_args.kwargs
    assert kwargs["headers"]["Content-Type"] == "application/octet-stream"


@pytest.mark.parametrize(
    "file_name,inner_type",
    [("report.pdf.gz", "application/pdf"), ("data.csv.gz", "text/csv")],
)
def test_upload_file_resolves_gzip_content_type_not_the_inner_type(
    monkeypatch, _upload_allowed_dirs_env, file_name, inner_type
):
    """Regression guard: mimetypes.guess_type("report.pdf.gz") returns
    ("application/pdf", "gzip") -- the type element describes the
    *decompressed* content, not what's actually going out over the wire.
    Blindly using that type element alone as Content-Type (discarding the
    encoding element) would tell any client trusting that header to parse
    a raw gzip stream as an uncompressed PDF (or CSV)."""
    local_file = _upload_allowed_dirs_env / file_name
    local_file.write_bytes(b"\x1f\x8b\x08\x00fake gzip bytes")

    # Confirm the premise directly against the real stdlib, independent of
    # whatever this host's mime.types happens to add on top.
    assert onedrive.mimetypes.guess_type(file_name) == (inner_type, "gzip")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["headers"]["Content-Type"] == (
        "application/gzip"
    )


@pytest.mark.parametrize(
    "extension,expected_mime_type",
    [
        (
            ".docx",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ),
        (".xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
        (
            ".pptx",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ),
        (".odt", "application/vnd.oasis.opendocument.text"),
    ],
)
def test_upload_file_resolves_ooxml_mime_type_without_relying_on_host_mime_db(
    monkeypatch, _upload_allowed_dirs_env, extension, expected_mime_type
):
    """Regression guard: stdlib mimetypes.guess_type() only recognizes OOXML/
    ODF extensions when a system mime.types file happens to be installed —
    verified directly (mimetypes.MimeTypes(filenames=()) returns (None, None)
    for all of these). A minimal/slim container image has no such file, so
    without _MIME_TYPE_OVERRIDES these would silently fall back to
    "application/octet-stream" instead of the correct, real mime type."""
    local_file = _upload_allowed_dirs_env / f"report{extension}"
    local_file.write_bytes(b"binary content")

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert (
        mock_request.call_args.kwargs["headers"]["Content-Type"] == expected_mime_type
    )


def test_upload_file_rejects_empty_or_root_remote_path(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: remote_path="/" previously reached _content_path
    (simple-PUT path) as an effectively empty target, raising a confusing
    "file_path is required" that names the wrong parameter, or reached
    _item_path (large-file path) silently building a request against the
    drive root itself instead of a named file."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file), remote_path="/"))

    assert result["status"] == "error"
    assert "remote_path" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize("remote_path", ["Documents/", "Documents/Reports/"])
def test_upload_file_rejects_trailing_slash_remote_path(
    monkeypatch, _upload_allowed_dirs_env, remote_path
):
    """Regression guard: a trailing "/" reads as "put it in this folder" --
    the obvious way an LLM caller would express that intent -- but
    _normalize_path strips it right off, so without this check the file
    would silently be uploaded as an item literally *named* "Documents"
    (or "Reports") at the parent location instead of placed inside that
    folder, with no error and nothing to signal the mistake."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(str(local_file), remote_path=remote_path)
    )

    assert result["status"] == "error"
    assert "filename" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_accepts_remote_path_with_folder_and_filename(
    monkeypatch, _upload_allowed_dirs_env
):
    """Complementary case: a remote_path that already includes a filename
    after the folder must still work normally."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock(return_value=MockResponse({"id": "f1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path="Documents/report.pdf"
        )
    )

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["url"].endswith(
        "/me/drive/root:/Documents/report.pdf:/content"
    )


@pytest.mark.parametrize(
    "traversal_remote_path",
    ["../../etc/passwd", "../../../../me/messages", "foo/../../bar", "./secret"],
)
def test_upload_file_rejects_dot_segments_in_remote_path(
    monkeypatch, _upload_allowed_dirs_env, traversal_remote_path
):
    """Regression guard for a confirmed request-forgery bug: requests'
    own URL preparation collapses ".." segments the same way a browser
    does (verified directly: 'root:/../../etc/x:/content' becomes
    '/me/etc/x:/content'), so an unvalidated remote_path could walk the
    actual HTTP request Graph receives entirely out of
    '/me/drive/root:/' and onto a different, unrelated Graph API endpoint
    under the same OAuth token -- not just the wrong file within Drive."""
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(local_file), remote_path=traversal_remote_path
        )
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_normalize_path_rejects_dot_segments_directly():
    """Direct regression guard on the shared choke point every path-
    building helper (_item_path/_children_path/_content_path) goes
    through, independent of which public tool calls it."""
    with pytest.raises(ValueError, match=r"\.\.|\bpath must not contain"):
        onedrive._normalize_path("../../etc/passwd")
    with pytest.raises(ValueError):
        onedrive._normalize_path("a/../b")
    with pytest.raises(ValueError):
        onedrive._normalize_path("./a")
    # A path with no dot-segments at all must be unaffected.
    assert onedrive._normalize_path("Documents/report.pdf") == "Documents/report.pdf"


@pytest.mark.parametrize(
    "call",
    [
        lambda: onedrive.onedrive_list_items(folder_path="../../etc"),
        lambda: onedrive.onedrive_get_item(path="../../etc/passwd"),
        lambda: onedrive.onedrive_get_file_content("../../etc/passwd"),
        lambda: onedrive.onedrive_create_folder("new-folder", parent_path="../../etc"),
        lambda: onedrive.onedrive_upload_text_file("../../etc/passwd", "x"),
    ],
    ids=[
        "onedrive_list_items",
        "onedrive_get_item",
        "onedrive_get_file_content",
        "onedrive_create_folder",
        "onedrive_upload_text_file",
    ],
)
def test_path_based_tools_reject_dot_segments_end_to_end(monkeypatch, call):
    """Regression guard: _normalize_path's dot-segment rejection is unit-
    tested directly above, and onedrive_upload_file's own remote_path is
    covered separately -- but that leaves the five *pre-existing* tools
    that also route a caller-supplied path through _item_path/
    _children_path/_content_path (and therefore _normalize_path)
    unexercised end-to-end. A future refactor that accidentally bypassed
    _normalize_path for one of these specific call sites (while the
    direct unit test above kept passing) would ship undetected without
    this."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(call())

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_upload_file_at_exact_boundary_uses_simple_put(
    monkeypatch, _upload_allowed_dirs_env
):
    """A file exactly at the size limit still uploads successfully."""
    local_file = _upload_allowed_dirs_env / "at_boundary.bin"
    local_file.write_bytes(b"\x00" * onedrive._SIMPLE_UPLOAD_MAX_BYTES)

    mock_request = Mock(return_value=MockResponse({"id": "item-1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)
    session_factory = Mock()
    monkeypatch.setattr(onedrive.requests, "Session", session_factory)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    assert mock_request.call_args.kwargs["method"] == "PUT"
    session_factory.assert_not_called()


@pytest.mark.parametrize(
    "file_path",
    [
        # Pre-existing binary-format coverage.
        "font.woff2",
        "font.woff",
        "font.ttf",
        "data.parquet",
        "cache.sqlite",
        "cache.sqlite3",
        "budget.numbers",
        "notes.pages",
        "vault.key",
        "extension.crx",
        "app.pub",
        "installer.dmg",
        "image.iso",
        "app.apk",
        "book.mobi",
        "app.jar",
        "Main.class",
        "installer.cab",
        "package.deb",
        "file.torrent",
        "keystore.p12",
        "cert.pfx",
        "movie.swf",
        # ML/data-science/VM-image/keystore formats a positive
        # known-binary-formats list kept missing across review rounds --
        # the specific gap that motivated switching to a default-deny,
        # known-*text*-formats allowlist instead (see
        # _KNOWN_TEXT_EXTENSIONS's own docstring). None of these need to be
        # individually enumerated anywhere for this test to pass -- that's
        # the point of the redesign: anything not affirmatively listed as
        # text is rejected, with no separate blocklist to keep patching.
        "model.safetensors",
        "weights.pkl",
        "array.npy",
        "array.npz",
        "model.onnx",
        "graph.pb",
        "module.pyc",
        "module.pyd",
        "pkg.whl",
        "pkg.egg",
        "blob.dat",
        "archive.xz",
        "archive.zst",
        "archive.lz4",
        "data.avro",
        "db.accdb",
        "df.feather",
        "table.arrow",
        "logs.orc",
        "disk.img",
        "disk.vmdk",
        "disk.qcow2",
        "store.jks",
        "model.sav",
        "mesh.stl",
        "weights.h5",
        "weights.hdf5",
        # Deliberately excluded from _KNOWN_TEXT_EXTENSIONS despite often
        # being PEM/text in practice -- see that set's own docstring on
        # why ".crt" specifically isn't given the same "text" judgment
        # call as ".bat"/".ts"/".scm"/".sc".
        "host.crt",
    ],
)
def test_upload_text_file_rejects_binary_extensions(monkeypatch, file_path):
    """Regression guard for the default-deny classifier: any extension not
    affirmatively listed in _KNOWN_TEXT_EXTENSIONS is rejected, regardless
    of whether it's a format anyone has thought to test mimetypes against.
    """
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "file_path",
    [
        "chart.svg",
        "app.ts",
        "deploy.bat",
        "session.scm",
        "worksheet.sc",
        "deploy.ps1",
        "schema.sql",
        "index.php",
        "main.dart",
        "paper.tex",
        "build.csh",
        "deploy.sh",
        "subs.srt",
        "schema.dtd",
        "layout.tpl",
    ],
)
def test_upload_text_file_allows_known_text_extensions_regardless_of_host_mimetypes(
    monkeypatch, file_path
):
    """Regression guard for the default-deny classifier's whole point:
    unlike the mimetype-driven designs this module cycled through in
    earlier rounds, _name_looks_binary never consults mimetypes.guess_type
    at all, so it can't be affected by whatever a given host's mime.types
    database happens to say. Proven directly here by stubbing
    mimetypes.guess_type to a value that would misclassify every one of
    these names if it were still consulted (a real binary-looking type
    with no text-safe signal at all) -- if the guard incorrectly fell back
    to mimetypes for any of these, this would catch it.

    ".ts"/".bat"/".scm"/".sc" are judgment calls (see
    _KNOWN_TEXT_EXTENSIONS's own docstring: each also names a real,
    unrelated binary format, but an agent-generated file with one of these
    names is overwhelmingly more likely to be genuine source text)."""
    monkeypatch.setattr(
        onedrive.mimetypes,
        "guess_type",
        lambda name: ("application/octet-stream", None),
    )
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "success"


def test_upload_text_file_allows_extensionless_names(monkeypatch):
    """Regression guard: a name with no extension at all (e.g. "Dockerfile",
    "README") isn't itself evidence of binary intent the way an
    unrecognized extension is -- _name_looks_binary's default-deny design
    only rejects a *recognized-as-suspicious* extension, not the absence of
    one."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    for name in ["Dockerfile", "README", "LICENSE"]:
        result = json.loads(onedrive.onedrive_upload_text_file(name, "some text"))
        assert result["status"] == "success", name


def test_upload_text_file_rejects_dotfile_shaped_binary_extension(monkeypatch):
    """Regression guard: a name that's *entirely* a leading dot plus
    extension (e.g. ".pdf") has an empty Path(...).suffix per pathlib's
    Unix-dotfile convention -- _split_stem_suffix exists specifically so
    this doesn't fall through _name_looks_binary as "extensionless" (and
    therefore accepted) the same way "Dockerfile" correctly does above."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(".pdf", "some text"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_upload_file_resolves_real_mime_type_for_ambiguous_extensions(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: _AMBIGUOUS_TEXT_EXTENSIONS forces ".ts"/".bat"/
    ".scm"/".sc"/".ps1" to be treated as non-binary by the
    onedrive_upload_text_file guard, but onedrive_upload_file uploads a
    real local file's real bytes -- a genuine ".ts" file is very commonly
    an actual MPEG transport-stream video chunk, not TypeScript source.
    An earlier version of this fix applied the override inside
    _guess_mime_type itself, which also corrupted this tool's real
    Content-Type resolution (sending "text/plain" for what Graph is told
    is a ".ts" file) whenever no explicit mime_type is passed.

    What mimetypes.guess_type() itself resolves ".ts" to is host-dependent
    -- confirmed directly: this file's own dev host resolves it to
    "video/mp2t", while a CI run on a different OS/Python resolved it to
    "text/vnd.trolltech.linguist" (Qt Linguist translation source, yet
    another real format that happens to share this extension) instead.
    Monkeypatching mimetypes.guess_type to a fixed value makes this test
    deterministic instead of depending on whichever mime.types database
    happens to be installed on whatever machine runs it -- what's actually
    under test is that _guess_mime_type's real answer reaches Content-Type
    unmodified, not what that real answer happens to be for ".ts"
    specifically."""
    monkeypatch.setattr(
        onedrive.mimetypes, "guess_type", lambda name: ("video/mp2t", None)
    )
    local_file = _upload_allowed_dirs_env / "segment001.ts"
    local_file.write_bytes(b"\x47" * 100)  # MPEG-TS sync byte, not text

    mock_request = Mock(return_value=MockResponse({"id": "f1"}))
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "success"
    sent_headers = mock_request.call_args.kwargs["headers"]
    assert sent_headers["Content-Type"] == "video/mp2t"


def test_upload_file_rejects_path_outside_allowed_directories(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    outside_file = tmp_path / "outside.pdf"
    outside_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(outside_file)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    # The absolute host path must never leak into the message the LLM sees.
    assert str(outside_file) not in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_sibling_directory_name_collision(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    """Regression guard: an allowed dir like "/workspace" must not
    accidentally admit a sibling "/workspace-other" just because it starts
    with the same string -- containment must be a real path-relative check
    (is_relative_to), not a naive string prefix comparison. Passing today,
    but pinned down directly so a future refactor to str.startswith()
    can't silently regress it while every other test stays green."""
    sibling_dir = tmp_path / (_upload_allowed_dirs_env.name + "-other")
    sibling_dir.mkdir()
    outside_file = sibling_dir / "secret.pdf"
    outside_file.write_bytes(b"content")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(outside_file)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_rejects_symlink_escaping_allowed_dir(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    """A symlink physically inside the allowed directory but pointing
    outside it must not grant access to its target -- resolve() follows
    the symlink to its real location before the containment check runs."""
    secret_dir = tmp_path / "secret"
    secret_dir.mkdir()
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("secret")
    link = _upload_allowed_dirs_env / "escape_link.txt"
    link.symlink_to(secret_file)

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(link)))

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_upload_file_allows_symlink_whose_target_is_inside_allowed_dir(
    monkeypatch, _upload_allowed_dirs_env
):
    """Complementary case to the escaping-symlink test above: a symlink
    that resolves to a target still *inside* the allowed directory must be
    accepted, not rejected outright -- guards against a future
    over-tightening (e.g. "reject any symlink at all") that would pass
    every existing test here while breaking a legitimate same-workspace
    symlink an agent's own tooling might create."""
    real_file = _upload_allowed_dirs_env / "real.pdf"
    real_file.write_bytes(b"content")
    link = _upload_allowed_dirs_env / "alias.pdf"
    link.symlink_to(real_file)

    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_file(str(link)))

    assert result["status"] == "success"


def test_upload_file_rejects_relative_traversal_outside_allowed_dir(
    monkeypatch, tmp_path, _upload_allowed_dirs_env
):
    (tmp_path / "secret.txt").write_text("secret")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(
        onedrive.onedrive_upload_file(
            str(_upload_allowed_dirs_env / ".." / "secret.txt")
        )
    )

    assert result["status"] == "error"
    assert "allowed upload directories" in result["message"]
    mock_request.assert_not_called()


def test_allowed_upload_dirs_falls_back_to_cwd_when_unset(monkeypatch):
    """onedrive.py now delegates to the shared allowed_dirs_from_env helper
    (mcp/utils.py) rather than keeping its own copy of this parsing
    logic -- see that helper's own docstring for why the copies drifted
    and produced a real bug before consolidation."""
    monkeypatch.delenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", raising=False)

    assert onedrive.allowed_dirs_from_env(onedrive._UPLOAD_ALLOWED_DIRS_ENV_VAR) == [
        onedrive.Path.cwd().resolve()
    ]


def test_allowed_upload_dirs_falls_back_to_cwd_when_malformed(monkeypatch):
    """Regression guard: a malformed-but-nonblank env value (e.g. a lone
    "," or " , ") must fall back to CWD the same way a fully-unset/blank
    value does, not silently resolve to an empty allowlist that rejects
    every upload with no diagnostic pointing at the env var as the cause."""
    monkeypatch.setenv("XAGENT_ONEDRIVE_FILE_ALLOWED_DIRS", " , ")

    assert onedrive.allowed_dirs_from_env(onedrive._UPLOAD_ALLOWED_DIRS_ENV_VAR) == [
        onedrive.Path.cwd().resolve()
    ]


def test_upload_file_rejects_missing_file(monkeypatch, _upload_allowed_dirs_env):
    missing_path = _upload_allowed_dirs_env / "does_not_exist.pdf"

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(missing_path)))

    assert result["status"] == "error"
    assert "not found" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_rejects_directory_with_a_distinct_message(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: a directory (or other non-regular file) used to
    raise the same "File not found" as a genuinely missing path, which
    reads as "retry, it'll show up" even though a directory never will."""
    a_directory = _upload_allowed_dirs_env / "not_a_file"
    a_directory.mkdir()

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(a_directory)))

    assert result["status"] == "error"
    assert "not a regular file" in result["message"].lower()
    mock_request.assert_not_called()


@pytest.mark.skipif(
    os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission bits aren't meaningful on Windows or when running as root",
)
def test_upload_file_does_not_leak_absolute_path_on_permission_error(
    monkeypatch, _upload_allowed_dirs_env
):
    """Regression guard: local_path.open("rb") previously had no OSError
    guard, so a permission-denied file (or a TOCTOU race between the
    allowlist check and the open) fell through to the generic
    `except Exception` and returned str(OSError) -- which embeds the
    absolute resolved host path (e.g. "[Errno 13] Permission denied:
    '/full/host/path'") -- straight to the caller/LLM, the same detail
    _resolve_upload_file_path's own error deliberately scrubs."""
    unreadable_file = _upload_allowed_dirs_env / "secret.pdf"
    unreadable_file.write_bytes(b"content")
    unreadable_file.chmod(0o000)
    try:
        mock_request = Mock()
        monkeypatch.setattr(onedrive.requests, "request", mock_request)

        result = json.loads(onedrive.onedrive_upload_file(str(unreadable_file)))

        assert result["status"] == "error"
        assert str(unreadable_file) not in result["message"]
        assert "secret.pdf" not in result["message"]
        mock_request.assert_not_called()
    finally:
        unreadable_file.chmod(0o644)


def test_upload_file_rejects_empty_file(monkeypatch, _upload_allowed_dirs_env):
    empty_file = _upload_allowed_dirs_env / "empty.pdf"
    empty_file.write_bytes(b"")

    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_file(str(empty_file)))

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    mock_request.assert_not_called()


def test_upload_file_returns_error_payload_on_api_failure(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "report.pdf"
    local_file.write_bytes(b"content")

    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(
            return_value=MockResponse(
                {"error": "boom"}, status_code=500, content=b'{"error": "boom"}'
            )
        ),
    )

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "boom" in result["message"]


# ---------------------------------------------------------------------------
# onedrive_upload_text_file — binary-content guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "file_path", ["report.pdf", "Documents/photo.PNG", "deck.pptx", "archive.zip"]
)
def test_upload_text_file_rejects_binary_looking_names(monkeypatch, file_path):
    """Regression guard for the actual production bug: onedrive_upload_text_file
    can only write text (content is utf-8 encoded), so a target path that
    looks like a binary format must be steered to onedrive_upload_file
    instead of silently getting a text/plain file with a misleading name."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "some text"))

    assert result["status"] == "error"
    assert "onedrive_upload_file" in result["message"]
    mock_request.assert_not_called()


@pytest.mark.parametrize(
    "file_path",
    [
        "notes.txt",
        "README.md",
        "config.json",
        "config.yaml",
        "config.yml",
        "data.csv",
        "script.py",
        "page.html",
        "server.log",
        "styles.css",
        "main.js",
        "data.xml",
        "notes",
    ],
)
def test_upload_text_file_allows_plain_text_names(monkeypatch, file_path):
    """Regression guard: the only broad "should be accepted" case used to
    be ".txt" alone, with every other positive case a narrow one-off added
    reactively after a specific extension was found broken in a prior
    round -- which is exactly how ".sh" shipped genuinely broken for a
    whole round before anyone tested it. This covers a broader set of
    everyday text formats an agent is likely to actually generate."""
    monkeypatch.setattr(
        onedrive.requests, "request", Mock(return_value=MockResponse({"id": "f1"}))
    )

    result = json.loads(onedrive.onedrive_upload_text_file(file_path, "hello world"))

    assert result["status"] == "success"


# ---------------------------------------------------------------------------
# item_id-based tools (onedrive_get_item, onedrive_rename_item,
# onedrive_delete_item) -- dot-segment rejection
# ---------------------------------------------------------------------------


def test_normalize_item_id_rejects_dot_segments_directly():
    """Regression guard: item_id-based endpoints are built as
    f"/me/drive/items/{quote(item_id, safe='')}" without ever going through
    _normalize_path/_item_path -- but quote() never percent-encodes a bare
    "." or ".." (they're in urllib's own "always safe" unreserved set), so
    without this explicit check an item_id of "." or ".." still reaches
    requests as a literal dot-segment and collapses the request path onto
    a different Graph endpoint under the same OAuth token, the same
    traversal class _normalize_path exists to stop for path-based tools."""
    for bad_id in [".", ".."]:
        with pytest.raises(ValueError, match="must not be"):
            onedrive._normalize_item_id(bad_id)


def test_normalize_item_id_rejects_empty_value():
    with pytest.raises(ValueError, match="required"):
        onedrive._normalize_item_id("   ")


@pytest.mark.parametrize(
    "call",
    [
        lambda item_id: onedrive.onedrive_get_item(item_id=item_id),
        lambda item_id: onedrive.onedrive_rename_item(item_id, "new-name.txt"),
        lambda item_id: onedrive.onedrive_delete_item(item_id),
    ],
    ids=["onedrive_get_item", "onedrive_rename_item", "onedrive_delete_item"],
)
@pytest.mark.parametrize("bad_id", [".", ".."])
def test_item_id_tools_reject_dot_segments(monkeypatch, call, bad_id):
    """Regression guard: each of the three item_id-based tools must refuse
    a "." or ".." item_id before ever making a request, not just when a
    Drive-relative path is used instead."""
    mock_request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", mock_request)

    result = json.loads(call(bad_id))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_item_by_item_id_still_works_for_a_normal_id(monkeypatch):
    monkeypatch.setattr(
        onedrive.requests,
        "request",
        Mock(return_value=MockResponse({"id": "abc123", "name": "report.pdf"})),
    )

    result = json.loads(onedrive.onedrive_get_item(item_id="abc123"))

    assert result["status"] == "success"
    assert result["item"]["id"] == "abc123"


def test_upload_file_one_byte_over_limit_rejected_before_network(
    monkeypatch, _upload_allowed_dirs_env
):
    local_file = _upload_allowed_dirs_env / "over_limit.bin"
    local_file.write_bytes(b"x" * (onedrive._SIMPLE_UPLOAD_MAX_BYTES + 1))
    request = Mock()
    monkeypatch.setattr(onedrive.requests, "request", request)
    session = Mock()
    monkeypatch.setattr(onedrive.requests, "Session", session)

    result = json.loads(onedrive.onedrive_upload_file(str(local_file)))

    assert result["status"] == "error"
    assert "4,000,000-byte (4 MB) limit" in result["message"]
    request.assert_not_called()
    session.assert_not_called()
