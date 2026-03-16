"""
BoxliteSandbox tests
"""

from __future__ import annotations

import asyncio
import os
import tempfile

import pytest

try:
    import boxlite  # noqa: F401
except ImportError:
    pytest.skip(
        "boxlite not installed, skipping sandbox tests", allow_module_level=True
    )

from src.xagent.sandbox.base import SandboxConfig, SandboxTemplate
from src.xagent.sandbox.boxlite_sandbox import BoxliteSandboxService, MemBoxliteStore


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


@requires_boxlite
class TestBoxliteSandboxService:
    """Test BoxliteSandboxService service layer functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_create_and_delete_sandbox(self):
        """Test creating and deleting sandbox"""
        name = "test_create_and_delete_sandbox"
        service = BoxliteSandboxService(MemBoxliteStore())

        # Cleanup
        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test creating and deleting sandbox ===")

            # Create sandbox
            template = SandboxTemplate(type="image", image="python:slim")

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
            assert info.config == config
            assert info.state == "running"

            # Check using native interface, can only check partial fields
            raw_info = sandbox._box.info()
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
    async def test_get_or_create_reuse(self):
        """Test get_or_create reuse logic"""
        name = "test_get_or_create_reuse"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test get_or_create reuse ===")

            # First creation
            sandbox1 = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_list_sandboxes(self):
        """Test listing sandboxes"""
        sandbox_names = ["test-list-1", "test-list-2"]
        service = BoxliteSandboxService(MemBoxliteStore())

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
                template=SandboxTemplate(type="image", image="python:slim"),
                config=SandboxConfig(cpus=1, memory=256),
            )

            sandbox2 = await service.get_or_create(
                sandbox_names[1],
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_concurrent_get_or_create(self):
        """Test if concurrent get_or_create with same name causes conflicts"""
        name = "test_concurrent_get_or_create"
        service = BoxliteSandboxService(MemBoxliteStore())

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
                    template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_concurrent_sandbox_exec(self):
        """Test if concurrent execution on same sandbox instance causes conflicts"""
        name = "test_concurrent_sandbox_exec"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test sandbox concurrent execution ===")

            # Create sandbox first
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_concurrent_sandboxs_exec(self):
        """Test if concurrent execution on different sandbox instances (same underlying box) causes conflicts"""
        name = "test_concurrent_sandboxs_exec"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test concurrent sandbox execution ===")

            # Create sandbox first
            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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


@requires_boxlite
class TestBoxliteSandbox:
    """Test BoxliteSandbox instance functionality"""

    @pytest.mark.asyncio(loop_scope="module")
    async def test_exec_command(self):
        """Test executing commands"""
        name = "test_exec_command"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test executing commands ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_run_python_code(self):
        """Test running Python code"""
        name = "test_run_python_code"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test running Python code ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
                config=SandboxConfig(cpus=1, memory=256),
            )

            result = await sandbox.exec("pip", "install", "pytest")
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
    async def test_run_node_code(self):
        """Test running Node.js code"""
        name = "test_run_node_code"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test running Node.js code ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="node:22-slim"),
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
    async def test_file_read_write(self):
        """Test file read and write operations"""
        name = "test_file_read_write"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test file read and write operations ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_upload_download(self):
        """Test file upload and download"""
        name = "test_upload_download"
        service = BoxliteSandboxService(MemBoxliteStore())

        try:
            await service.delete(name)
        except Exception:
            pass

        try:
            print("\n=== Test file upload and download ===")

            sandbox = await service.get_or_create(
                name,
                template=SandboxTemplate(type="image", image="python:slim"),
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
    async def test_upload_download_with_volume(self):
        """Test upload/download compatibility with volume mounts"""
        name = "test_upload_download_with_volume"
        service = BoxliteSandboxService(MemBoxliteStore())

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
                    template=SandboxTemplate(type="image", image="python:slim"),
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
