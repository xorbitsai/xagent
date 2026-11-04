"""Tests for the shared LibreOffice .pptx -> PDF conversion helper.

The helper is used by both the .pptx preview cache and the slide render tool,
so these tests pin the process contract (argv list, headless conversion,
per-call LibreOffice user profile, timeout, cancellation and cleanup) rather
than any caller's caching behavior.
"""

import asyncio
import json
import os
import pathlib
import signal
import sys
import time

import pytest


def _load_conversion_module():
    try:
        from xagent.core.tools.core import pptx_conversion
    except ImportError as exc:  # pragma: no cover - red phase only
        pytest.fail(f"shared pptx conversion module is missing: {exc}")
    return pptx_conversion


def _profile_dir_from_arguments(arguments: list[str]) -> pathlib.Path:
    """Return the user-installation directory named by the soffice argv."""
    for argument in arguments:
        if argument.startswith("-env:UserInstallation="):
            uri = argument.split("=", 1)[1]
            assert uri.startswith("file://"), uri
            return pathlib.Path(uri[len("file://") :])
    raise AssertionError(f"no -env:UserInstallation argument in {arguments!r}")


_simulated_processes = {}


@pytest.fixture(autouse=True)
def simulated_process_groups(monkeypatch, tmp_path):
    monkeypatch.setattr(_load_conversion_module().tempfile, "tempdir", str(tmp_path))
    kill_group = os.killpg

    def kill(group, signal):
        if group in _simulated_processes:
            _simulated_processes[group].kill()
        else:
            kill_group(group, signal)

    monkeypatch.setattr(os, "killpg", kill)
    yield
    _simulated_processes.clear()


class _RecordingProcess:
    """Fake asyncio subprocess that records kill/wait usage."""

    def __init__(self, *, returncode: int = 0, stderr: bytes = b""):
        self.returncode: int | None = None
        self.final_returncode = returncode
        self.stderr = stderr
        self.pid = id(self)
        _simulated_processes[self.pid] = self
        self.killed = 0
        self.wait_calls = 0

    def kill(self) -> None:
        self.killed += 1

    async def wait(self) -> int:
        self.wait_calls += 1
        self.returncode = -9 if self.killed else self.final_returncode
        return self.returncode


class _HangingProcess(_RecordingProcess):
    """Fake subprocess whose wait() remains pending until termination."""

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.killed:
            self.returncode = -9
            return -9
        self.started.set()
        # Bounded stand-in: the real per-call timeout used by the tests is much
        # shorter, and a mutant that drops the timeout must not stall the suite.
        await asyncio.sleep(5)
        return 0


def _install_process_factory(monkeypatch, conversion, factory):
    calls: list[dict] = []

    async def create_subprocess_exec(*arguments, **kwargs):
        calls.append({"arguments": list(arguments), "kwargs": dict(kwargs)})
        process = factory(calls)
        kwargs["stderr"].write(process.stderr)
        kwargs["stderr"].flush()
        return process

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )
    return calls


def test_default_contract_constants_are_shared():
    conversion = _load_conversion_module()

    assert conversion.DEFAULT_PPTX_CONVERSION_TIMEOUT_SECONDS == 60
    assert conversion.DEFAULT_SOFFICE_BINARY == "soffice"


def test_expected_pdf_path_uses_the_exact_source_stem(tmp_path):
    conversion = _load_conversion_module()

    assert conversion.expected_pdf_path(pathlib.Path("/d/deck.pptx"), tmp_path) == (
        tmp_path / "deck.pdf"
    )
    assert conversion.expected_pdf_path(pathlib.Path("/d/a.b.pptx"), tmp_path) == (
        tmp_path / "a.b.pdf"
    )


def test_build_arguments_use_parameter_list_without_shell_quoting(tmp_path):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck with spaces.pptx"
    output_dir = tmp_path / "out"
    profile_dir = tmp_path / "profile"

    arguments = conversion.build_soffice_arguments(pptx_path, output_dir, profile_dir)

    assert arguments == [
        "soffice",
        "--headless",
        f"-env:UserInstallation={profile_dir.as_uri()}",
        "--convert-to",
        "pdf",
        "--outdir",
        str(output_dir),
        str(pptx_path),
    ]
    assert isinstance(arguments, list)
    assert profile_dir.as_uri().startswith("file://")


@pytest.mark.asyncio
async def test_conversion_passes_headless_convert_to_and_per_call_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    observed: dict[str, object] = {}

    process = _RecordingProcess()

    async def wait():
        observed["profile_dir"] = _profile_dir_from_arguments(calls[0]["arguments"])
        observed["profile_existed_during_run"] = observed["profile_dir"].is_dir()
        (output_dir / "deck.pdf").write_bytes(b"%PDF-1.4 conversion")
        process.returncode = 0
        return 0

    process.wait = wait  # type: ignore[method-assign]

    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    pdf_path = await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert pdf_path == output_dir / "deck.pdf"
    assert pdf_path.read_bytes() == b"%PDF-1.4 conversion"
    arguments = calls[0]["arguments"]
    assert arguments[0] == "soffice"
    assert arguments[1] == "--headless"
    assert arguments[3] == "--convert-to"
    assert arguments[4] == "pdf"
    assert arguments[5] == "--outdir"
    assert arguments[6] == str(output_dir)
    assert arguments[7] == str(pptx_path)
    assert calls[0]["kwargs"]["stdout"] == asyncio.subprocess.DEVNULL
    assert hasattr(calls[0]["kwargs"]["stderr"], "fileno")
    # The user profile is a per-call directory, not the PDF output directory.
    assert observed["profile_existed_during_run"] is True
    assert observed["profile_dir"] != output_dir
    assert not observed["profile_dir"].exists()
    assert output_dir.is_dir()


@pytest.mark.asyncio
async def test_missing_soffice_raises_a_clear_dependency_error(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()

    async def create_subprocess_exec(*_arguments, **_kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'soffice'")

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )

    with pytest.raises(conversion.PptxConverterUnavailableError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    message = str(fault.value)
    assert "soffice" in message
    assert "PATH" in message
    assert "deck.pptx" in message
    assert list(output_dir.iterdir()) == []
    leftovers = list(tmp_path.glob("xagent-pptx-profile-*"))
    assert leftovers == []


@pytest.mark.asyncio
async def test_non_zero_exit_reports_stderr_and_cleans_the_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    observed: dict[str, object] = {}

    process = _RecordingProcess(returncode=1, stderr=b"libreoffice exploded")

    async def wait():
        observed["profile_dir"] = _profile_dir_from_arguments(calls[0]["arguments"])
        process.returncode = 1
        return 1

    process.wait = wait  # type: ignore[method-assign]
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    message = str(fault.value)
    assert "deck.pptx" in message
    assert "libreoffice exploded" in message
    assert "1" in message
    assert not observed["profile_dir"].exists()


@pytest.mark.asyncio
async def test_success_without_the_expected_pdf_is_a_failure(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _RecordingProcess()
    _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert "deck.pdf" in str(fault.value)


@pytest.mark.asyncio
async def test_a_different_pdf_is_not_silently_accepted(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _RecordingProcess()

    async def wait():
        (output_dir / "some-other.pdf").write_bytes(b"%PDF-1.4 stray")
        process.returncode = 0
        return 0

    process.wait = wait  # type: ignore[method-assign]
    _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionFailedError) as fault:
        await conversion.convert_pptx_to_pdf(pptx_path, output_dir)

    assert "deck.pdf" in str(fault.value)


@pytest.mark.asyncio
async def test_timeout_kills_and_awaits_the_process_and_cleans_the_profile(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _HangingProcess()
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    with pytest.raises(conversion.PptxConversionTimeoutError) as fault:
        await conversion.convert_pptx_to_pdf(
            pptx_path, output_dir, timeout_seconds=0.05
        )

    assert "0.05" in str(fault.value)
    assert "deck.pptx" in str(fault.value)
    assert process.killed == 1
    assert process.wait_calls == 2
    profile_dir = _profile_dir_from_arguments(calls[0]["arguments"])
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_cancellation_kills_and_awaits_the_process_then_cleans_up(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    pptx_path = tmp_path / "deck.pptx"
    pptx_path.write_bytes(b"pptx bytes")
    output_dir = tmp_path / "pdf-out"
    output_dir.mkdir()
    process = _HangingProcess()
    calls = _install_process_factory(monkeypatch, conversion, lambda _calls: process)

    task = asyncio.create_task(conversion.convert_pptx_to_pdf(pptx_path, output_dir))
    await asyncio.wait_for(process.started.wait(), timeout=5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.killed == 1
    assert process.wait_calls == 2
    profile_dir = _profile_dir_from_arguments(calls[0]["arguments"])
    assert not profile_dir.exists()


@pytest.mark.asyncio
async def test_concurrent_conversions_get_isolated_profiles(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    output_one = tmp_path / "out-one"
    output_two = tmp_path / "out-two"
    output_one.mkdir()
    output_two.mkdir()
    first = tmp_path / "first.pptx"
    second = tmp_path / "second.pptx"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    calls: list[dict] = []
    release = asyncio.Event()
    arrived = asyncio.Event()

    class _Process(_RecordingProcess):
        def __init__(self, output_dir: pathlib.Path, name: str):
            super().__init__()
            self.output_dir = output_dir
            self.name = name

        async def wait(self):
            self.returncode = 0
            arrived.set()
            await release.wait()
            self.output_dir.joinpath(self.name).write_bytes(b"%PDF-1.4")
            return 0

    async def create_subprocess_exec(*arguments, **kwargs):
        calls.append({"arguments": list(arguments), "kwargs": dict(kwargs)})
        output_dir = pathlib.Path(arguments[arguments.index("--outdir") + 1])
        return _Process(output_dir, pathlib.Path(arguments[-1]).stem + ".pdf")

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )

    both = asyncio.gather(
        conversion.convert_pptx_to_pdf(first, output_one),
        conversion.convert_pptx_to_pdf(second, output_two),
    )
    await asyncio.wait_for(arrived.wait(), timeout=5)
    release.set()
    results = await asyncio.wait_for(both, timeout=10)

    assert [path.name for path in results] == ["first.pdf", "second.pdf"]
    profiles = [_profile_dir_from_arguments(call["arguments"]) for call in calls]
    outdirs = [
        pathlib.Path(call["arguments"][call["arguments"].index("--outdir") + 1])
        for call in calls
    ]
    assert len({str(path) for path in profiles}) == 2
    assert len({str(path) for path in outdirs}) == 2
    for profile in profiles:
        assert not profile.exists()
    assert (output_one / "first.pdf").is_file()
    assert (output_two / "second.pdf").is_file()


@pytest.mark.asyncio
async def test_concurrent_conversions_are_capped_process_wide(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    limit = conversion.MAX_CONCURRENT_CONVERSIONS
    release = asyncio.Event()
    full = asyncio.Event()
    running = 0
    peak = 0

    class _Process(_RecordingProcess):
        def __init__(self, pdf_path: pathlib.Path):
            super().__init__()
            self.pdf_path = pdf_path

        async def wait(self):
            nonlocal running
            await release.wait()
            self.pdf_path.write_bytes(b"%PDF-1.4")
            running -= 1
            self.returncode = 0
            return 0

    async def create_subprocess_exec(*arguments, **kwargs):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        if running == limit:
            full.set()
        output_dir = pathlib.Path(arguments[arguments.index("--outdir") + 1])
        return _Process(output_dir / (pathlib.Path(arguments[-1]).stem + ".pdf"))

    monkeypatch.setattr(
        conversion.asyncio, "create_subprocess_exec", create_subprocess_exec
    )
    conversions = []
    for index in range(limit + 1):
        source = tmp_path / f"deck-{index}.pptx"
        source.write_bytes(b"pptx")
        output_dir = tmp_path / f"out-{index}"
        output_dir.mkdir()
        conversions.append(
            asyncio.create_task(conversion.convert_pptx_to_pdf(source, output_dir))
        )

    await asyncio.wait_for(full.wait(), timeout=5)
    await asyncio.sleep(0)
    assert peak == limit
    release.set()
    results = await asyncio.wait_for(asyncio.gather(*conversions), timeout=10)

    assert peak == limit
    assert [path.name for path in results] == [
        f"deck-{index}.pdf" for index in range(limit + 1)
    ]


_FAKE_CONVERTER_SCRIPT = r"""
import json
import os
import subprocess
import sys
import time

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
os.write(int(os.environ["CONVERTER_READY_FD"]), json.dumps([os.getpid(), child.pid]).encode())
os.close(int(os.environ["CONVERTER_READY_FD"]))
time.sleep(60)
"""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["timeout", "cancel"])
async def test_converter_group_teardown_is_bounded_with_inherited_pipes(
    tmp_path, monkeypatch, mode
):
    conversion = _load_conversion_module()
    source = tmp_path / (mode + ".pptx")
    source.write_bytes(b"synthetic presentation")
    binary = tmp_path / "soffice"
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    binary.write_text("#!" + sys.executable + "\n" + _FAKE_CONVERTER_SCRIPT)
    binary.chmod(0o700)
    output = tmp_path / "pdf"
    output.mkdir()
    profiles = []
    create_process = asyncio.create_subprocess_exec
    reader_fd, writer_fd = os.pipe()
    loop = asyncio.get_running_loop()
    ready = loop.create_future()
    process_ready = asyncio.Event()

    def read_ready():
        loop.remove_reader(reader_fd)
        ready.set_result(json.loads(os.read(reader_fd, 1024)))

    loop.add_reader(reader_fd, read_ready)

    async def spawn(*arguments, **kwargs):
        profiles.append(_profile_dir_from_arguments(arguments))
        process = await create_process(
            *arguments,
            **kwargs,
            pass_fds=(writer_fd,),
            env={**os.environ, "CONVERTER_READY_FD": str(writer_fd)},
        )
        await asyncio.shield(ready)
        loop.call_soon(process_ready.set)
        return process

    monkeypatch.setattr(conversion.asyncio, "create_subprocess_exec", spawn)
    task = asyncio.create_task(
        conversion.convert_pptx_to_pdf(
            source,
            output,
            timeout_seconds=0.8 if mode == "timeout" else 30,
        )
    )
    pids = []
    try:
        pids = await asyncio.wait_for(ready, 5)
        await asyncio.wait_for(process_ready.wait(), 5)
        if mode == "cancel":
            task.cancel()
        done, pending = await asyncio.wait({task}, timeout=3)
        assert not pending, "conversion teardown exceeded three seconds"
        expected = (
            conversion.PptxConversionTimeoutError
            if mode == "timeout"
            else asyncio.CancelledError
        )
        with pytest.raises(expected):
            task.result()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            try:
                os.kill(pids[1], 0)
            except ProcessLookupError:
                break
            await asyncio.sleep(0.01)
        with pytest.raises(ProcessLookupError):
            os.kill(pids[1], 0)
        assert profiles and all(not path.exists() for path in profiles)
    finally:
        loop.remove_reader(reader_fd)
        os.close(reader_fd)
        os.close(writer_fd)
        for pid in pids:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def test_hidden_slide_export_is_an_explicit_impress_filter_option(tmp_path):
    import inspect

    conversion = _load_conversion_module()
    assert (
        "include_hidden_slides"
        in inspect.signature(conversion.build_soffice_arguments).parameters
    )
    default = conversion.build_soffice_arguments(
        tmp_path / "deck.pptx", tmp_path, tmp_path
    )
    assert default[default.index("--convert-to") + 1] == "pdf"
    enabled = conversion.build_soffice_arguments(
        tmp_path / "deck.pptx", tmp_path, tmp_path, include_hidden_slides=True
    )
    extension, filter_name, options = enabled[enabled.index("--convert-to") + 1].split(
        ":", 2
    )
    assert (extension, filter_name) == ("pdf", "impress_pdf_Export")
    assert json.loads(options) == {
        "ExportHiddenSlides": {"type": "boolean", "value": "true"}
    }


@pytest.mark.asyncio
async def test_success_uses_file_stderr_without_killing_group(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    source = tmp_path / "deck.pptx"
    source.write_bytes(b"presentation")
    (tmp_path / "deck.pdf").write_bytes(b"%PDF-1.4")
    process = _RecordingProcess()

    async def wait():
        process.returncode = 0
        return 0

    process.wait = wait
    calls = _install_process_factory(monkeypatch, conversion, lambda _: process)
    await conversion.convert_pptx_to_pdf(source, tmp_path)
    assert calls[0]["kwargs"]["stdout"] == asyncio.subprocess.DEVNULL
    stderr = calls[0]["kwargs"]["stderr"]
    assert hasattr(stderr, "fileno")
    assert pathlib.Path(stderr.name).parent.name.startswith("xagent-pptx-profile-")
    assert process.killed == 0


@pytest.mark.asyncio
async def test_permission_denied_during_group_cleanup_preserves_timeout(
    tmp_path, monkeypatch
):
    conversion = _load_conversion_module()
    process = _HangingProcess()

    async def wait():
        await asyncio.sleep(5)
        return 0

    process.wait = wait
    _install_process_factory(monkeypatch, conversion, lambda _: process)

    def denied(*args):
        raise PermissionError("group cannot be signalled")

    monkeypatch.setattr(os, "killpg", denied)
    with pytest.raises(conversion.PptxConversionTimeoutError):
        await conversion.convert_pptx_to_pdf(
            tmp_path / "deck.pptx", tmp_path, timeout_seconds=0.01
        )


@pytest.mark.asyncio
async def test_normal_leader_exit_does_not_wait_for_descendant(tmp_path, monkeypatch):
    conversion = _load_conversion_module()
    source = tmp_path / "deck.pptx"
    source.write_bytes(b"presentation")
    ready_path = tmp_path / "child.pid"
    binary = tmp_path / "soffice"
    binary.write_text(
        "#!" + sys.executable + "\n" + "import pathlib, subprocess, sys\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"pathlib.Path({str(ready_path)!r}).write_text(str(child.pid))\n"
        f"pathlib.Path({str(tmp_path / 'deck.pdf')!r}).write_bytes(b'%PDF-1.4')\n"
    )
    binary.chmod(0o700)
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(conversion.tempfile, "tempdir", str(tmp_path))
    child_pid = None
    try:
        result = await asyncio.wait_for(
            conversion.convert_pptx_to_pdf(source, tmp_path), 3
        )
        child_pid = int(ready_path.read_text())
        assert result == tmp_path / "deck.pdf"
        assert not list(tmp_path.glob("xagent-pptx-profile-*"))
        os.kill(child_pid, 0)
    finally:
        if child_pid is None and ready_path.exists():
            child_pid = int(ready_path.read_text())
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
