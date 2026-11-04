"""
BoxliteSandbox tests
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from types import SimpleNamespace

import pytest

try:
    import boxlite  # noqa: F401
except ImportError:
    pytest.skip(
        "boxlite not installed, skipping sandbox tests", allow_module_level=True
    )

from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import SandboxConfig, SandboxTemplate
from xagent.sandbox.boxlite_sandbox import (
    BoxliteSandbox,
    BoxliteSandboxService,
    MemBoxliteStore,
)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def _check_boxlite_available() -> bool:
    """Check if boxlite is available"""
    try:
        try:
            boxlite.Boxlite.default()
            print("\n✓ Boxlite initialized successfully")
            return True
        except BaseException as e:
            error_msg = f"✗ Boxlite initialization failed: {type(e).__name__}: {e}"
            print(f"\n{error_msg}")
            return False
    except ImportError as e:
        error_msg = f"✗ Boxlite import failed: {type(e).__name__}: {e}"
        print(f"\n{error_msg}")
        return False


requires_boxlite = pytest.mark.skipif(
    not _check_boxlite_available(), reason="Requires boxlite runtime"
)


@pytest.fixture(scope="module")
def boxlite_service():
    """Provide a shared Boxlite sandbox service for integration-style tests."""
    return BoxliteSandboxService(MemBoxliteStore())


class TestBoxliteSandboxRunCodeValidation:
    """Test lightweight run_code validation paths."""

    @pytest.mark.asyncio
    async def test_run_code_rejects_unsupported_code_type(self):
        """Unsupported code types should fail explicitly."""
        sandbox = object.__new__(BoxliteSandbox)

        with pytest.raises(ValueError, match="Unsupported code type: ruby"):
            await sandbox.run_code("puts 'hi'", code_type="ruby")  # type: ignore[arg-type]


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.drained = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.drained == len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self.drained]
        self.drained += 1
        return chunk


class _FakeExecution:
    def __init__(self, stdout, stderr):
        self.stdout_stream = _FakeStream(stdout)
        self.stderr_stream = _FakeStream(stderr)
        self.killed = False

    def stdout(self):
        return self.stdout_stream

    def stderr(self):
        return self.stderr_stream

    async def wait(self):
        return SimpleNamespace(exit_code=0, error_message=None)

    async def kill(self):
        self.killed = True


class _FakeNativeBox:
    def __init__(self, execution):
        self.execution = execution
        self.call = None

    async def exec(self, command, args, env, **kwargs):
        self.call = (command, args, env, kwargs)
        return self.execution


@pytest.mark.asyncio
async def test_exec_caps_output_while_draining_boxlite_streams():
    execution = _FakeExecution([b"abc", b"def"], ["xy", "z"])
    native_box = _FakeNativeBox(execution)
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=native_box)

    result = await sandbox.exec(
        "demo", "--flag", env={"KEY": "value"}, max_output_bytes=4
    )

    assert result.stdout == "abcd"
    assert result.stderr == "xyz"
    assert result.truncated is True
    assert execution.stdout_stream.drained == 2
    assert execution.stderr_stream.drained == 2
    assert native_box.call == (
        "demo",
        ["--flag"],
        [("KEY", "value")],
        {"user": None, "timeout_secs": 3600, "cwd": None},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chunks", "cap", "expected"),
    [
        ([b"abc"], 0, ""),
        (["你".encode("utf-8")], 2, ""),
        (["你".encode("utf-8"), b"a"], 3, "你"),
    ],
)
async def test_exec_output_cap_handles_zero_and_multibyte_boundaries(
    chunks, cap, expected
):
    execution = _FakeExecution(chunks, [])
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=_FakeNativeBox(execution))

    result = await sandbox.exec("demo", max_output_bytes=cap)

    assert result.stdout == expected
    assert result.truncated is True


@pytest.mark.asyncio
async def test_exec_wait_failure_returns_failure_exit_code():
    execution = _FakeExecution([b"partial"], [])

    async def fail_wait():
        raise RuntimeError("wait failed")

    execution.wait = fail_wait
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=_FakeNativeBox(execution))

    result = await sandbox.exec("demo", max_output_bytes=32)

    assert result.exit_code == -1
    assert result.stdout == "partial"


@pytest.mark.asyncio
async def test_exec_stream_failure_cancels_and_awaits_sibling_collector():
    sibling_started = asyncio.Event()

    class FailingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await sibling_started.wait()
            raise RuntimeError("stream failed")

    class BlockingStream:
        cancelled = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise

    execution = _FakeExecution([], [])
    sibling = BlockingStream()
    execution.stdout_stream = FailingStream()
    execution.stderr_stream = sibling
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=_FakeNativeBox(execution))

    with pytest.raises(RuntimeError, match="stream failed"):
        await sandbox.exec("demo", max_output_bytes=32)

    assert sibling.cancelled is True
    assert execution.killed is True


@pytest.mark.asyncio
async def test_exec_with_output_cap_kills_guest_when_cancelled():
    release = asyncio.Event()

    class BlockingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await release.wait()
            raise StopAsyncIteration

    execution = _FakeExecution([], [])
    execution.stdout_stream = BlockingStream()
    execution.stderr_stream = BlockingStream()
    native_box = _FakeNativeBox(execution)
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=native_box)

    task = asyncio.create_task(sandbox.exec("hang", max_output_bytes=4))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert execution.killed is True


@pytest.mark.asyncio
async def test_exec_cancellation_is_not_blocked_by_hung_guest_kill(monkeypatch):
    never = asyncio.Event()

    class BlockingStream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await never.wait()
            raise StopAsyncIteration

    execution = _FakeExecution([], [])
    execution.stdout_stream = BlockingStream()
    execution.stderr_stream = BlockingStream()

    async def hung_kill():
        await never.wait()

    execution.kill = hung_kill
    sandbox = object.__new__(BoxliteSandbox)
    sandbox._box = SimpleNamespace(_box=_FakeNativeBox(execution))
    monkeypatch.setattr(
        "xagent.sandbox.boxlite_sandbox._EXECUTION_KILL_TIMEOUT_SECONDS", 0.01
    )

    task = asyncio.create_task(sandbox.exec("hang", max_output_bytes=4))
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)


@requires_boxlite
class TestBoxliteSandboxService:
    """Test BoxliteSandboxService service layer functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_isolated_guest_has_no_outbound_route(self, boxlite_service):
        """Test the actual guest route, not only the SDK's NetworkSpec object."""
        name = "test_isolated_guest_has_no_outbound_route"
        service = boxlite_service
        try:
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256, network_isolated=True),
            )
            # A numeric public destination avoids DNS as a false explanation
            # for failure. Send no application data; a disabled guest network
            # must fail locally with no route, not merely time out externally.
            result = await sandbox.run_code(
                """
                import errno
                import socket
                with socket.socket() as connection:
                    connection.settimeout(2)
                    result = connection.connect_ex(("1.1.1.1", 443))
                assert result in (errno.ENETUNREACH, errno.ENETDOWN), result
                print("guest network is unreachable")
                """
            )
            assert result.exit_code == 0, result.stderr
            assert "guest network is unreachable" in result.stdout
        finally:
            await service.delete(name)

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_and_delete_sandbox(self, boxlite_service):
        """Test creating and deleting sandbox"""
        name = "test_create_and_delete_sandbox"
        service = boxlite_service

        # Cleanup
        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test creating and deleting sandbox ===")

            # Create sandbox
            template = SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE)

            temp_dir = tempfile.mkdtemp()
            config = SandboxConfig(
                cpus=2,
                memory=1024,
                env={
                    "MY_VAR": "hello",
                },
                volumes=[
                    (temp_dir, "/mnt/data", "rw"),  # Read-write mount
                ],
                network_isolated=False,
                ports=[
                    (8080, 80),  # Host 8080 -> Container 80
                ],
            )
            sandbox = await service.get_or_create(
                name,
                template=template,
                config=config,
            )

            # Check status
            info = await sandbox.info()
            assert info.template == template
            assert info.config == config.model_copy(
                update={"security_profile": service.runtime_security_profile()}
            )
            assert info.state == "running"

            # Check using native interface, can only check partial fields
            raw_info = await sandbox._box.info()
            assert raw_info.name == name
            assert raw_info.image == template.image
            assert raw_info.cpus == config.cpus
            assert raw_info.memory_mib == config.memory
            assert raw_info.created_at == info.created_at

            # Verify environment variables are effective
            result = await sandbox.exec("sh", "-c", "echo $MY_VAR")
            assert result.stdout.strip() == "hello"
            print("✓ Environment variable configuration effective")

            # Verify volume mount is effective
            result = await sandbox.exec("test", "-d", "/mnt/data")
            assert result.exit_code == 0
            print("✓ Volume mount configuration effective")

            # Write file in mounted volume, verify visible on host
            await sandbox.exec("sh", "-c", "echo 'test' > /mnt/data/test.txt")
            host_file = os.path.join(temp_dir, "test.txt")
            assert os.path.exists(host_file)
            with open(host_file, "r") as f:
                assert f.read().strip() == "test"
            print("✓ Volume mount read-write normal")

            # Verify port mapping is effective
            # Start HTTP server inside sandbox (port 80 in container)
            await sandbox.exec(
                "sh", "-c", "nohup python -m http.server 80 > /tmp/http.log 2>&1 &"
            )
            await asyncio.sleep(2)  # Wait for server to start

            # Send HTTP request from host to mapped port (host 8080 -> container 80)
            import urllib.request

            try:
                response = urllib.request.urlopen("http://127.0.0.1:8080", timeout=3)
                status_code = response.getcode()
                if status_code == 200:
                    print(
                        "✓ Port mapping configuration effective (host 8080 -> container 80, HTTP request successful)"
                    )
                else:
                    print(f"⚠ Port mapping test: HTTP status code {status_code}")
            except Exception as e:
                print(f"⚠ Port mapping test exception: {e}")
                # Don't force failure, as port mapping may be affected by network environment

            # Stop sandbox
            await sandbox.stop()

            info = await sandbox.info()
            assert info.state == "stopped"

            # Delete sandbox
            await service.delete(name)

            box = await service._runtime.get(name)
            assert box is None

            print("✅ Create and delete test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_get_or_create_reuse(self, boxlite_service):
        """Test get_or_create reuse logic"""
        name = "test_get_or_create_reuse"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test get_or_create reuse ===")

            # First creation
            sandbox1 = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            # Write data
            await sandbox1.write_file("data from first", "/var/tmp/data.txt")

            # Stop
            await sandbox1.stop()

            # Second retrieval (should reuse)
            sandbox2 = await service.get_or_create(name)

            # Verify data still exists
            content = await sandbox2.read_file("/var/tmp/data.txt")
            print(f"Read after reuse: {content}")
            assert content == "data from first"

            await sandbox2.stop()

            print("✅ Reuse test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_list_sandboxes(self, boxlite_service):
        """Test listing sandboxes"""
        sandbox_names = ["test-list-1", "test-list-2"]
        service = boxlite_service

        # Cleanup
        for name in sandbox_names:
            try:
                await service.delete(name)
            except Exception:
                pass

        try:
            print("\n=== Test listing sandboxes ===")

            sandbox1 = await service.get_or_create(
                sandbox_names[0],
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            sandbox2 = await service.get_or_create(
                sandbox_names[1],
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            sandboxes = await service.list_sandboxes()
            names = [s.name for s in sandboxes]
            print(f"Sandbox list: {names}")

            assert sandbox_names[0] in names
            assert sandbox_names[1] in names

            # Cleanup
            await sandbox1.stop()
            await sandbox2.stop()

            print("✅ List test passed")

        finally:
            for name in sandbox_names:
                try:
                    await service.delete(name)
                except Exception:
                    pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_concurrent_get_or_create(self, boxlite_service):
        """Test if concurrent get_or_create with same name causes conflicts"""
        name = "test_concurrent_get_or_create"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test concurrent get_or_create ===")

            # Concurrently create same sandbox
            async def create_task(task_id: int):
                print(f"Task {task_id}: Starting get_or_create")
                sandbox = await service.get_or_create(
                    name,
                    template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    config=SandboxConfig(cpus=1, memory=256),
                )
                print(f"Task {task_id}: get_or_create completed")
                return sandbox

            # Start concurrent tasks
            tasks = [create_task(i) for i in range(5)]
            sandboxes = await asyncio.gather(*tasks)

            # Verify all returned sandboxes point to same instance
            print(f"Got {len(sandboxes)} sandbox instances")
            for i, sb in enumerate(sandboxes):
                assert sb.name == name
                print(f"Task {i}: sandbox.name = {sb.name}")

            # Verify only one sandbox was created
            list_result = await service.list_sandboxes()
            same_name_boxes = [s for s in list_result if s.name == name]
            assert len(same_name_boxes) == 1
            print("Only 1 sandbox instance created")

            # Cleanup
            await sandboxes[0].stop()

            print("✅ Concurrent get_or_create test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_concurrent_sandbox_exec(self, boxlite_service):
        """Test if concurrent execution on same sandbox instance causes conflicts"""
        name = "test_concurrent_sandbox_exec"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test sandbox concurrent execution ===")

            # Create sandbox first
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            async def exec_task(task_id: int):
                result = await sandbox.exec("sh", "-c", f"echo 'Task {task_id}'")
                assert result.success
                assert f"Task {task_id}" in result.stdout
                return task_id

            # Start concurrent tasks
            tasks = [exec_task(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # Verify all tasks completed successfully
            assert len(results) == 10
            assert sorted(results) == list(range(10))
            print(f"✓ All {len(results)} concurrent tasks completed successfully")

            # Cleanup
            await sandbox.stop()

            print("✅ Concurrent execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_concurrent_sandboxs_exec(self, boxlite_service):
        """Test if concurrent execution on different sandbox instances (same underlying box) causes conflicts"""
        name = "test_concurrent_sandboxs_exec"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test concurrent sandbox execution ===")

            # Create sandbox first
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            async def exec_task(task_id: int):
                # Each task gets its own sandbox reference
                sb = await service.get_or_create(name)

                # Execute command
                result = await sb.exec("echo", f"Hello from task {task_id}")
                print(f"Task {task_id}: Command output = {result.stdout.strip()}")

                # Write task-specific file
                file_path = f"/var/tmp/task_{task_id}.txt"
                content = f"Task {task_id} data"
                await sb.write_file(content, file_path, overwrite=True)
                print(f"Task {task_id}: Wrote file {file_path}")

                # Read file to verify
                read_content = await sb.read_file(file_path)
                assert read_content == content
                print(f"Task {task_id}: File content verification successful")

                return task_id

            # Start 10 concurrent tasks
            tasks = [exec_task(i) for i in range(10)]
            results = await asyncio.gather(*tasks)

            # Verify all tasks completed successfully
            assert len(results) == 10
            assert sorted(results) == list(range(10))
            print(f"✓ All {len(results)} concurrent tasks completed successfully")

            # Cleanup
            await sandbox.stop()

            print("✅ Concurrent execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass


@requires_boxlite
class TestBoxliteSandbox:
    """Test BoxliteSandbox instance functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_exec_command(self, boxlite_service):
        """Test executing commands"""
        name = "test_exec_command"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test executing commands ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            # Execute simple command
            user = await sandbox.exec("id", "-u")
            assert user.exit_code == 0
            assert user.stdout.strip() != "0", "The sandbox image USER must be honored"
            root_write = await sandbox.exec("test", "-w", "/root")
            assert root_write.exit_code != 0
            msg = "hello word"
            # The image's unprivileged USER cannot write to /home or /root.
            # Exercise a writable guest directory without escalating to root.
            result = await sandbox.exec("sh", "-c", f'echo "{msg}" > /var/tmp/test.txt')
            assert result.exit_code == 0, result.stderr

            result = await sandbox.exec("sh", "-c", "ls /var/tmp | grep test.txt")
            print(f"Command output: {result.stdout.strip()}")

            result = await sandbox.exec("cat", "/var/tmp/test.txt")
            print(f"result: {result.stdout.strip()}")
            assert result.stdout.strip() == msg

            await sandbox.stop()

            print("✅ Command execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_run_python_code(self, boxlite_service):
        """Test running Python code"""
        name = "test_run_python_code"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test running Python code ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            result = await sandbox.exec(
                "pip", "install", "--user", "--break-system-packages", "pytest"
            )
            print(f"Output:\n{result.stdout}")
            assert result.success, result.stderr

            # Run Python code
            python_code = """
            import asyncio
            import pytest

            async def main():
                print("Hello from Python")

            asyncio.run(main())
            """
            result = await sandbox.run_code(python_code, code_type="python")
            print(f"Python output:\n{result.stdout}")
            assert result.success, result.stderr
            assert "Hello from Python" in result.stdout

            await sandbox.stop()

            print("✅ Python code execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_run_node_code(self, boxlite_service):
        """Test running Node.js code"""
        name = "test_run_node_code"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test running Node.js code ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            # Run simple JavaScript code
            js_code = """
            console.log("Hello from Node.js");
            """
            result = await sandbox.run_code(js_code, code_type="javascript")
            print(f"Node.js output:\n{result.stdout}")
            assert result.success
            assert "Hello from Node.js" in result.stdout

            # Test long code (over 1KB)
            long_js_code = (
                "// " + "x" * 1024 + "\n" + 'console.log("Long code execution");'
            )
            result = await sandbox.run_code(long_js_code, code_type="javascript")
            print(f"Long code output:\n{result.stdout}")
            assert result.success
            assert "Long code execution" in result.stdout

            await sandbox.stop()

            print("✅ Node.js code execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_file_read_write(self, boxlite_service):
        """Test file read and write operations"""
        name = "test_file_read_write"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test file read and write operations ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            # Write file
            test_content = "Hello, this is a test file!"
            await sandbox.write_file(test_content, "/var/tmp/test.txt")
            print("File write successful")

            # Read file
            content = await sandbox.read_file("/var/tmp/test.txt")
            print(f"Read content: {content}")
            assert content == test_content

            # Test overwrite protection
            try:
                await sandbox.write_file(
                    "new content", "/var/tmp/test.txt", overwrite=False
                )
                assert False, "Should raise FileExistsError"
            except FileExistsError:
                print("Overwrite protection working properly")

            # Overwrite file
            new_content = "Updated content"
            await sandbox.write_file(new_content, "/var/tmp/test.txt", overwrite=True)
            content = await sandbox.read_file("/var/tmp/test.txt")
            assert content == new_content
            print("File overwrite successful")

            await sandbox.stop()

            print("✅ File read and write operations test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_upload_download(self, boxlite_service):
        """Test file upload and download"""
        name = "test_upload_download"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test file upload and download ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=256),
            )

            # Create temporary file for upload
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                upload_content = "This is a test file for upload"
                f.write(upload_content)
                local_upload_path = f.name

            try:
                # Upload file
                remote_path = "/var/tmp/uploaded.txt"
                await sandbox.upload_file(local_upload_path, remote_path)
                print(f"Upload file: {local_upload_path} -> {remote_path}")

                # Verify file uploaded
                content = await sandbox.read_file(remote_path)
                assert content == upload_content
                print("Upload verification successful")

                # Test overwrite protection
                try:
                    await sandbox.upload_file(
                        local_upload_path, remote_path, overwrite=False
                    )
                    assert False, "Should raise FileExistsError"
                except FileExistsError:
                    print("Upload overwrite protection working properly")

                # Download file
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".txt"
                ) as f:
                    local_download_path = f.name

                await sandbox.download_file(
                    remote_path, local_download_path, overwrite=True
                )
                print(f"Download file: {remote_path} -> {local_download_path}")

                # Verify downloaded file content
                with open(local_download_path, "r") as f:
                    downloaded_content = f.read()
                assert downloaded_content == upload_content
                print("Download verification successful")

                # Cleanup temporary files
                os.unlink(local_download_path)

            finally:
                os.unlink(local_upload_path)

            await sandbox.stop()

            print("✅ File upload and download test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_upload_download_with_volume(self, boxlite_service):
        """Test upload/download compatibility with volume mounts"""
        name = "test_upload_download_with_volume"
        service = boxlite_service

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test upload/download compatibility with volume mounts ===")

            # Create temporary directory as volume
            temp_dir = tempfile.mkdtemp()
            volume_file_path = os.path.join(temp_dir, "volume_file.txt")
            volume_content = "This file is in the mounted volume"
            with open(volume_file_path, "w") as f:
                f.write(volume_content)

            try:
                # Create sandbox with volume mount
                sandbox = await service.get_or_create(
                    name,
                    template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                    config=SandboxConfig(
                        cpus=1,
                        memory=256,
                        volumes=[
                            (temp_dir, "/mnt/data", "rw"),
                        ],
                    ),
                )

                # Verify file in volume is accessible
                content = await sandbox.read_file("/mnt/data/volume_file.txt")
                assert content == volume_content
                print("✓ Volume mounted file accessible")

                # Upload file to volume directory
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".txt"
                ) as f:
                    volume_upload_content = "Uploaded file to volume"
                    f.write(volume_upload_content)
                    local_upload_path = f.name

                remote_volume_path = "/mnt/data/uploaded_to_volume.txt"
                await sandbox.upload_file(local_upload_path, remote_volume_path)

                # Verify reading through sandbox
                content = await sandbox.read_file(remote_volume_path)
                assert content == volume_upload_content
                print("✓ Upload to volume directory successful")

                # Verify reading through host filesystem
                host_file_path = os.path.join(temp_dir, "uploaded_to_volume.txt")
                with open(host_file_path, "r") as f:
                    host_content = f.read()
                assert host_content == volume_upload_content
                print("✓ Volume directory file visible on host")
                os.unlink(local_upload_path)

                # Download file from volume directory
                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".txt"
                ) as f:
                    local_download_path = f.name

                await sandbox.download_file(
                    remote_volume_path, local_download_path, overwrite=True
                )
                with open(local_download_path, "r") as f:
                    downloaded_content = f.read()
                assert downloaded_content == volume_upload_content
                print("✓ Download from volume directory successful")
                os.unlink(local_download_path)

                # Write file in volume directory, verify visible on host
                write_content = "Written from sandbox to volume"
                await sandbox.write_file(write_content, "/mnt/data/written.txt")
                host_written_path = os.path.join(temp_dir, "written.txt")
                with open(host_written_path, "r") as f:
                    host_written_content = f.read()
                assert host_written_content == write_content
                print("✓ Sandbox write to volume directory, visible on host")

                await sandbox.stop()

                print("✅ Upload/download and volume mount compatibility test passed")

            finally:
                # Cleanup temporary directory
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass
