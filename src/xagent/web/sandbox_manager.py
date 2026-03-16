"""
Sandbox management in application layer.
"""

import logging
import os
import threading
from typing import Optional

from ..sandbox import SandboxService
from ..sandbox.base import Sandbox, SandboxConfig, SandboxTemplate

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
        # TODO: Determine template and config based on user configuration
        sandbox_image = os.getenv("SANDBOX_IMAGE", "python:slim")
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

        template = SandboxTemplate(type="image", image=sandbox_image)
        config = SandboxConfig(cpus=sandbox_cpus, memory=sandbox_memory)

        # Create sandbox with task-specific name
        sandbox_name = f"{lifecycle_type}::{lifecycle_id}"

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
        return sandbox

    async def delete_sandbox(self, lifecycle_type: str, lifecycle_id: str) -> None:
        """
        Delete sandbox.

        Args:
            lifecycle_type: e.g. task|user
            lifecycle_id: e.g. task_id|user_id
        """
        sandbox_name = f"{lifecycle_type}::{lifecycle_id}"
        try:
            await self._service.delete(sandbox_name)
            logger.debug(f"Sandbox deleted: {sandbox_name}")
        except Exception as e:
            logger.error(f"Failed to delete sandbox {sandbox_name}: {e}")

    async def warmup(self) -> None:
        """
        Warmup default image.
        """
        sandbox_image = os.getenv("SANDBOX_IMAGE", "python:slim")
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

    async def shutdown_all(self) -> None:
        """
        Stop all running sandboxes. Called on app shutdown.
        """
        try:
            sandboxes = await self._service.list_sandboxes()
            if not sandboxes:
                logger.info("No sandboxes to clean up")
                return

            running = [sb for sb in sandboxes if sb.state == "running"]
            if not running:
                logger.info("No running sandboxes to stop")
                return

            logger.info(f"Stopping {len(running)} running sandbox(es)...")
            for sb in running:
                try:
                    box = await self._service.get_or_create(
                        sb.name, template=sb.template, config=sb.config
                    )
                    await box.stop()
                    logger.debug(f"Stopped sandbox: {sb.name}")
                except Exception as e:
                    logger.error(f"Failed to stop sandbox {sb.name}: {e}")

            logger.info("All running sandboxes stopped")
        except Exception as e:
            logger.error(f"Failed to shut down sandboxes: {e}")


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
