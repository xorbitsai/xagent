"""
Sandbox management in application layer.
"""

import asyncio
import logging
import os
import threading
from typing import Optional

from ..sandbox import DEFAULT_SANDBOX_IMAGE, SandboxService
from ..sandbox.base import Sandbox, SandboxConfig, SandboxTemplate
from .config import UPLOADS_DIR

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Manages sandbox instances.
    """

    def __init__(self, service: SandboxService):
        """
        Initialize sandbox manager.

        Args:
            service: SandboxService instance for creating sandboxes
        """
        self._service: SandboxService = service
        self._cache: dict[str, Sandbox] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    @staticmethod
    def make_sandbox_name(lifecycle_type: str, lifecycle_id: str) -> str:
        """Build a sandbox name from lifecycle type and id."""
        return f"{lifecycle_type}::{lifecycle_id}"

    @staticmethod
    def parse_sandbox_name(name: str) -> tuple[str, str]:
        """Parse a sandbox name into (lifecycle_type, lifecycle_id).

        Raises:
            ValueError: Invalid sandbox name format.
        """
        parts = name.split("::", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid sandbox name format: {name!r}")
        return parts[0], parts[1]

    def _get_sandbox_config(self) -> tuple[str, int, int]:
        sandbox_image = os.getenv("SANDBOX_IMAGE", DEFAULT_SANDBOX_IMAGE)
        try:
            sandbox_cpus = int(os.getenv("SANDBOX_CPUS", "1"))
        except ValueError:
            logger.warning("Invalid SANDBOX_CPUS value, using default")
            sandbox_cpus = 1
        try:
            sandbox_memory = int(os.getenv("SANDBOX_MEMORY", "512"))
        except ValueError:
            logger.warning("Invalid SANDBOX_MEMORY value, using default")
            sandbox_memory = 512
        return sandbox_image, sandbox_cpus, sandbox_memory

    def _make_volumes(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
        *,
        ensure_dir: bool,
    ) -> Optional[list[tuple[str, str, str]]]:
        """
        Build volume param.

        Only supports user lifecycle type at the moment.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
            ensure_dir: When True, create the host directory
        """
        volumes: Optional[list[tuple[str, str, str]]] = None
        if lifecycle_type == "user":
            user_workspace = str((UPLOADS_DIR / f"user_{lifecycle_id}").resolve())
            if ensure_dir:
                os.makedirs(user_workspace, exist_ok=True)
            # Use the same absolute path for both host and sandbox.
            volumes = [(user_workspace, user_workspace, "rw")]
        return volumes

    async def get_or_create_sandbox(
        self,
        lifecycle_type: str,
        lifecycle_id: str,
    ) -> Sandbox:
        """
        Get or create a sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id

        Returns:
            Sandbox instance
        """
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        if sandbox_name in self._cache:
            return self._cache[sandbox_name]

        # Acquire per-name lock to prevent concurrent creation
        async with self._locks_guard:
            if sandbox_name not in self._locks:
                self._locks[sandbox_name] = asyncio.Lock()
            lock = self._locks[sandbox_name]

        async with lock:
            # Double-check after acquiring lock
            if sandbox_name in self._cache:
                return self._cache[sandbox_name]

            # TODO: Determine template and config based on user configuration
            sandbox_image, sandbox_cpus, sandbox_memory = self._get_sandbox_config()

            template = SandboxTemplate(type="image", image=sandbox_image)

            volumes = self._make_volumes(lifecycle_type, lifecycle_id, ensure_dir=True)
            config = SandboxConfig(
                cpus=sandbox_cpus,
                memory=sandbox_memory,
                volumes=volumes,
            )

            logger.debug(f"Getting or creating sandbox for: {sandbox_name}")
            sandbox = await self._service.get_or_create(
                sandbox_name,
                template=template,
                config=config,
            )

            # Package and upload xagent code
            from ..core.tools.adapters.vibe.sandboxed_tool.sandboxed_tool_wrapper import (
                upload_code_to_sandbox,
            )

            await upload_code_to_sandbox(sandbox)
            self._cache[sandbox_name] = sandbox
            return sandbox

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_name = self.make_sandbox_name(lifecycle_type, lifecycle_id)
        try:
            await self._service.delete(sandbox_name)
            logger.debug(f"Sandbox deleted: {sandbox_name}")
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_name}: {e}")
        finally:
            # Always evict from cache — even on failure the instance
            # may be in an unknown state and should be recreated.
            self._cache.pop(sandbox_name, None)
            self._locks.pop(sandbox_name, None)

    async def warmup(self) -> None:
        """
        Warmup default image.
        """
        sandbox_image, sandbox_cpus, sandbox_memory = self._get_sandbox_config()
        warmup_name = "__warmup__"
        try:
            template = SandboxTemplate(type="image", image=sandbox_image)
            config = SandboxConfig()
            async with await self._service.get_or_create(
                warmup_name, template=template, config=config
            ) as _:
                pass
            await self._service.delete(warmup_name)
            logger.info(f"Sandbox image warmup completed: {sandbox_image}")
        except Exception as e:
            logger.error(f"Failed to warmup sandbox image: {e}")

    async def cleanup(self) -> None:
        """Stop all running sandboxes.

        Delete sandboxes whose config (image, cpus, memory, volumes)
        differs from the current environment so they get recreated
        with the correct settings next time.

        Note:
            If ``UPLOADS_DIR`` (via ``XAGENT_UPLOADS_DIR`` env var) changes
            between deployments, all user sandboxes will be detected as
            having stale volume mounts and will be deleted for recreation.
        """
        try:
            sandboxes = await self._service.list_sandboxes()
            if not sandboxes:
                logger.info("No sandboxes to clean up")
                return

            sandbox_image, sandbox_cpus, sandbox_memory = self._get_sandbox_config()

            for sb in sandboxes:
                try:
                    lifecycle_type, lifecycle_id = None, None
                    try:
                        lifecycle_type, lifecycle_id = self.parse_sandbox_name(sb.name)
                    except ValueError:
                        # Not a normal managed sandbox name, stop
                        if sb.state == "running":
                            box = await self._service.get_or_create(
                                sb.name, template=sb.template, config=sb.config
                            )
                            await box.stop()
                            logger.debug(f"Stopped sandbox: {sb.name}")
                        continue

                    # Delete sandbox if config changed (force recreate on next start)
                    image_changed = sb.template.image != sandbox_image
                    cpus_changed = sb.config.cpus != sandbox_cpus
                    memory_changed = sb.config.memory != sandbox_memory

                    # Check if volumes changed (e.g. UPLOADS_DIR path changed)
                    volumes_changed = False
                    expected_volumes = self._make_volumes(
                        lifecycle_type, lifecycle_id, ensure_dir=False
                    )
                    if sb.config.volumes != expected_volumes:
                        volumes_changed = True

                    if (
                        image_changed
                        or cpus_changed
                        or memory_changed
                        or volumes_changed
                    ):
                        changes = []
                        if image_changed:
                            changes.append(
                                f"image: {sb.template.image} -> {sandbox_image}"
                            )
                        if cpus_changed:
                            changes.append(f"cpus: {sb.config.cpus} -> {sandbox_cpus}")
                        if memory_changed:
                            changes.append(
                                f"memory: {sb.config.memory} -> {sandbox_memory}"
                            )
                        if volumes_changed:
                            changes.append(
                                f"volumes: {sb.config.volumes} -> {expected_volumes}"
                            )
                        logger.info(
                            f"Config changed for sandbox [{sb.name}]: "
                            f"{', '.join(changes)}, deleting"
                        )
                        await self._service.delete(sb.name)
                        continue

                    # Stop running sandboxes with matching image
                    if sb.state == "running":
                        box = await self._service.get_or_create(
                            sb.name, template=sb.template, config=sb.config
                        )
                        await box.stop()
                        logger.debug(f"Stopped sandbox: {sb.name}")
                except Exception as e:
                    logger.error(f"Failed to handle sandbox {sb.name}: {e}")

            self._cache.clear()
            self._locks.clear()
            logger.info("Sandbox cleanup completed")
        except Exception as e:
            logger.error(f"Failed to cleanup sandboxes: {e}")


# Global sandbox manager instance
_sandbox_manager: Optional[SandboxManager] = None
_sandbox_manager_lock = threading.Lock()
_sandbox_manager_initialized = False


def _create_sandbox_service() -> Optional[SandboxService]:
    """
    Create sandbox service based on environment configuration.

    Environment variables:
    - SANDBOX_ENABLED: Enable/disable sandbox (default: true)
    - SANDBOX_IMPLEMENTATION: Implementation type (default: boxlite)
      - boxlite: Use Boxlite sandbox
    - BOXLITE_HOME_DIR: Boxlite home directory (optional)

    Returns:
        SandboxService instance or None if disabled
    """
    # Check if sandbox is enabled
    sandbox_enabled = os.getenv("SANDBOX_ENABLED", "false").lower() == "true"
    if not sandbox_enabled:
        logger.info("Sandbox is disabled via SANDBOX_ENABLED environment variable")
        return None

    # Get implementation type
    implementation = os.getenv("SANDBOX_IMPLEMENTATION", "boxlite")

    if implementation == "boxlite":
        return _create_boxlite_service()
    else:
        logger.warning(
            f"Unknown sandbox implementation: {implementation}, falling back to boxlite"
        )
        return _create_boxlite_service()


def _create_boxlite_service() -> Optional[SandboxService]:
    """Create Boxlite sandbox service."""
    try:
        from ..sandbox import BoxliteSandboxService
    except ImportError:
        logger.error("boxlite is not installed.")
        return None

    from .sandbox_store import DBBoxliteStore

    store = DBBoxliteStore()
    # Get home directory
    home_dir = os.getenv("BOXLITE_HOME_DIR")

    service = None
    try:
        service = BoxliteSandboxService(store=store, home_dir=home_dir)
        logger.info(
            f"Created Boxlite sandbox service (home_dir={home_dir or 'default'})"
        )
    except Exception as e:
        logger.error(f"Failed to create Boxlite sandbox service: {e}")

    return service


def get_sandbox_manager() -> Optional[SandboxManager]:
    """
    Get or create global sandbox manager instance.

    Thread-safe singleton pattern with double-checked locking.

    Returns:
        SandboxManager instance or None if sandbox is disabled
    """
    global _sandbox_manager, _sandbox_manager_initialized

    # Fast path: already initialized (either successfully or service was None)
    if _sandbox_manager_initialized:
        return _sandbox_manager

    # Slow path: need to initialize
    with _sandbox_manager_lock:
        # Double-check after acquiring lock
        if _sandbox_manager_initialized:
            return _sandbox_manager

        # Get sandbox service
        service = _create_sandbox_service()
        if service is None:
            _sandbox_manager_initialized = True
            return None

        # Create sandbox manager
        _sandbox_manager = SandboxManager(service)
        _sandbox_manager_initialized = True
        logger.info("Created global sandbox manager")

        return _sandbox_manager
