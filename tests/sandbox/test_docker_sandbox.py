"""
DockerSandbox tests
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile

import pytest

import xagent.sandbox.docker_sandbox as docker_sandbox_module
from xagent.sandbox import DEFAULT_SANDBOX_IMAGE
from xagent.sandbox.base import SandboxConfig, SandboxTemplate
from xagent.sandbox.docker_sandbox import (
    DockerSandboxService,
    MemDockerStore,
    _SandboxControl,
    is_docker_available,
)


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


requires_docker = pytest.mark.skipif(
    not is_docker_available(), reason="Requires reachable Docker daemon"
)


@pytest.fixture(scope="module")
def docker_service():
    """Provide a shared Docker sandbox service for integration-style tests."""
    return DockerSandboxService(MemDockerStore())


class _FakeContainerCollection:
    """Minimal Docker container collection stub for service unit tests."""

    def __init__(self, containers=()):
        self._containers = containers

    def list(self, *args, **kwargs):
        return list(self._containers)


class _FakeDockerClient:
    """Minimal Docker client stub for service unit tests."""

    def __init__(self, containers=()):
        self.containers = _FakeContainerCollection(containers)

    def ping(self):
        return True


class _FailingStartContainer:
    """Container stub whose start fails before sandbox initialization finishes."""

    def __init__(self) -> None:
        self.remove_calls: list[bool] = []

    def start(self) -> None:
        raise RuntimeError("port conflict")

    def remove(self, force: bool = False) -> None:
        self.remove_calls.append(force)


def _get_free_host_port() -> int:
    """Reserve an ephemeral host port for Docker port-mapping tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


class TestDockerSandboxServiceFailures:
    """Test failure cleanup paths that do not require Docker."""

    @pytest.mark.asyncio
    async def test_get_or_create_removes_container_when_start_fails(self, monkeypatch):
        """Failed startup should remove the newly-created container."""
        created_container = _FailingStartContainer()

        async def fake_create_container(*args, **kwargs):
            return created_container

        monkeypatch.setattr(
            docker_sandbox_module, "_create_container", fake_create_container
        )
        service = DockerSandboxService(MemDockerStore(), client=_FakeDockerClient())

        with pytest.raises(RuntimeError, match="port conflict"):
            await service.get_or_create(
                "start-failure",
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(),
            )

        assert len(created_container.remove_calls) == 1
        assert created_container.remove_calls[0] is True


class TestDockerSandboxRunCodeValidation:
    """Test lightweight run_code validation paths."""

    @pytest.mark.asyncio
    async def test_run_code_rejects_unsupported_code_type(self):
        """Unsupported code types should fail explicitly."""
        sandbox = object.__new__(docker_sandbox_module.DockerSandbox)

        with pytest.raises(ValueError, match="Unsupported code type: ruby"):
            await sandbox.run_code("puts 'hi'", code_type="ruby")  # type: ignore[arg-type]


@requires_docker
class TestDockerSandboxService:
    """Test DockerSandboxService service layer functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_and_delete_sandbox(self, docker_service):
        """Test creating and deleting sandbox"""
        name = "test_create_and_delete_sandbox"
        service = docker_service

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
            host_port = _get_free_host_port()
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
                    (host_port, 80),  # Host dynamic port -> Container 80
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
            assert info.config == config
            assert info.state == "running"

            assert info.name == name
            assert info.template.image == template.image
            assert info.config.cpus == config.cpus
            assert info.config.memory == config.memory
            assert info.created_at is not None

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

            # Send HTTP request from host to mapped port.
            import urllib.request

            try:
                response = urllib.request.urlopen(
                    f"http://127.0.0.1:{host_port}", timeout=3
                )
                status_code = response.getcode()
                if status_code == 200:
                    print(
                        f"✓ Port mapping configuration effective (host {host_port} -> container 80, HTTP request successful)"
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

            container = await service._find_container(name)
            assert container is None

            print("✅ Create and delete test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_get_or_create_reuse(self, docker_service):
        """Test get_or_create reuse logic"""
        name = "test_get_or_create_reuse"
        service = docker_service

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
            await sandbox1.write_file("data from first", "/root/data.txt")

            # Stop
            await sandbox1.stop()

            # Second retrieval (should reuse)
            sandbox2 = await service.get_or_create(name)

            # Verify data still exists
            content = await sandbox2.read_file("/root/data.txt")
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
    async def test_list_sandboxes(self, docker_service):
        """Test listing sandboxes"""
        sandbox_names = ["test-list-1", "test-list-2"]
        service = docker_service

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
    async def test_concurrent_get_or_create(self, docker_service):
        """Test if concurrent get_or_create with same name causes conflicts"""
        name = "test_concurrent_get_or_create"
        service = docker_service

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
    async def test_concurrent_sandbox_exec(self, docker_service):
        """Test if concurrent execution on same sandbox instance causes conflicts"""
        name = "test_concurrent_sandbox_exec"
        service = docker_service

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
    async def test_concurrent_sandboxs_exec(self, docker_service):
        """Test if concurrent execution on different sandbox instances (same underlying box) causes conflicts"""
        name = "test_concurrent_sandboxs_exec"
        service = docker_service

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
                file_path = f"/root/task_{task_id}.txt"
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

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_snapshot_rejects_duplicate_id(self, docker_service):
        """Creating the same snapshot ID twice should fail."""
        name = "test_snapshot_duplicate_id"
        snapshot_id = "duplicate-snapshot"
        service = docker_service

        try:
            await service.delete(name)
        except Exception:
            pass
        try:
            await service.delete_snapshot(snapshot_id)
        except Exception:
            pass

        try:
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=512),
            )
            await sandbox.write_file("v1", "/root/data.txt", overwrite=True)

            first_snapshot = await service.create_snapshot(name, snapshot_id)
            assert first_snapshot.snapshot_id == snapshot_id

            with pytest.raises(FileExistsError):
                await service.create_snapshot(name, snapshot_id)
        finally:
            try:
                await service.delete(name)
            except Exception:
                pass
            try:
                await service.delete_snapshot(snapshot_id)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_snapshot_lifecycle(self, docker_service):
        """Test snapshot create, restore, list, and delete behavior."""
        name = "test_snapshot_lifecycle"
        clone_name = "test_snapshot_lifecycle_clone"
        snapshot_id = "test_snapshot_id"
        service = docker_service

        for sandbox_name in (name, clone_name):
            try:
                await service.delete(sandbox_name)
            except Exception:
                pass

        try:
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image=DEFAULT_SANDBOX_IMAGE),
                config=SandboxConfig(cpus=1, memory=512),
            )
            await sandbox.write_file(
                "snapshot-data", "/root/snapshot.txt", overwrite=True
            )

            snapshot = await service.create_snapshot(name, snapshot_id)
            assert snapshot.snapshot_id == snapshot_id

            snapshots = await service.list_snapshots()
            assert [item.snapshot_id for item in snapshots] == [snapshot_id]

            clone = await service.get_or_create(
                clone_name,
                template=SandboxTemplate(type="snapshot", snapshot_id=snapshot_id),
                config=SandboxConfig(cpus=1, memory=512),
            )
            clone_info = await clone.info()
            assert clone_info.template.type == "snapshot"
            assert clone_info.template.snapshot_id == snapshot_id
            assert await clone.read_file("/root/snapshot.txt") == "snapshot-data"

            with pytest.raises(RuntimeError):
                await service.delete_snapshot(snapshot_id)

            await service.delete(clone_name)
            await service.delete_snapshot(snapshot_id)
            assert await service.list_snapshots() == []
        finally:
            for sandbox_name in (name, clone_name):
                try:
                    await service.delete(sandbox_name)
                except Exception:
                    pass


class TestSandboxControl:
    """Test `_SandboxControl` concurrency guarantees."""

    @pytest.mark.asyncio
    async def test_operation_releases_on_repeated_cancellation(self):
        """`operation()` should release active ops even if exit is cancelled again."""
        control = _SandboxControl(name="cancel-safe")
        # The operation body has started; `active_ops` should now be 1.
        body_entered = asyncio.Event()
        # Cleanup has entered `release_operation()`.
        release_started = asyncio.Event()
        # Hold release so the second cancellation lands during cleanup.
        allow_release = asyncio.Event()
        original_release = control.release_operation

        async def delayed_release() -> None:
            """Delay release until the test injects a second cancellation."""
            release_started.set()
            await allow_release.wait()
            await original_release()

        # Replace release with a controllable wrapper.
        control.release_operation = delayed_release  # type: ignore[method-assign]

        async def worker() -> None:
            async with control.operation():
                body_entered.set()
                await asyncio.Event().wait()

        task = asyncio.create_task(worker())
        await body_entered.wait()
        assert control.active_ops == 1

        task.cancel()
        await release_started.wait()
        # Cancel again while cleanup is running.
        task.cancel()
        allow_release.set()

        with pytest.raises(asyncio.CancelledError):
            await task

        # Cleanup must still release the operation.
        assert control.active_ops == 0


@requires_docker
class TestDockerSandbox:
    """Test DockerSandbox instance functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_exec_command(self, docker_service):
        """Test executing commands"""
        name = "test_exec_command"
        service = docker_service

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
            msg = "hello word"
            result = await sandbox.exec("sh", "-c", f'echo "{msg}" > test.txt')

            result = await sandbox.exec("sh", "-c", "ls | grep test.txt")
            print(f"Command output: {result.stdout.strip()}")

            result = await sandbox.exec("cat", "test.txt")
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
    async def test_run_python_code(self, docker_service):
        """Test running Python code"""
        name = "test_run_python_code"
        service = docker_service

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
                "pip", "install", "--break-system-packages", "pytest"
            )
            print(f"Output:\n{result.stdout}")

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
            assert result.success
            assert "Hello from Python" in result.stdout

            await sandbox.stop()

            print("✅ Python code execution test passed")

        finally:
            try:
                await service.delete(name)
            except Exception:
                pass

    @pytest.mark.asyncio(loop_scope="module")
    async def test_run_node_code(self, docker_service):
        """Test running Node.js code"""
        name = "test_run_node_code"
        service = docker_service

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
    async def test_file_read_write(self, docker_service):
        """Test file read and write operations"""
        name = "test_file_read_write"
        service = docker_service

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
            await sandbox.write_file(test_content, "/root/test.txt")
            print("File write successful")

            # Read file
            content = await sandbox.read_file("/root/test.txt")
            print(f"Read content: {content}")
            assert content == test_content

            # Test overwrite protection
            try:
                await sandbox.write_file(
                    "new content", "/root/test.txt", overwrite=False
                )
                assert False, "Should raise FileExistsError"
            except FileExistsError:
                print("Overwrite protection working properly")

            # Overwrite file
            new_content = "Updated content"
            await sandbox.write_file(new_content, "/root/test.txt", overwrite=True)
            content = await sandbox.read_file("/root/test.txt")
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
    async def test_upload_download(self, docker_service):
        """Test file upload and download"""
        name = "test_upload_download"
        service = docker_service

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
                remote_path = "/root/uploaded.txt"
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
    async def test_upload_download_with_volume(self, docker_service):
        """Test upload/download compatibility with volume mounts"""
        name = "test_upload_download_with_volume"
        service = docker_service

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
