"""
Base interfaces and data classes for execution services.

Defines the unified abstraction for execution services with support for
multiple isolation strategies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class IsolationType(Enum):
    """Isolation type enumeration.

    Defines the level of isolation for code execution.
    """

    PROCESS = "process"  # Process-level isolation (using xoscar)
    SANDBOX = "sandbox"  # Sandbox isolation (future work)


class ServiceStatus(Enum):
    """Service status enumeration.

    Tracks the lifecycle state of a service.
    """

    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Execution result dataclass.

    Unified result format for all execution types (Python, JavaScript, command).
    """

    success: bool
    output: str
    error: str = ""
    return_code: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None
    memory_used_mb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the execution result
        """
        return {
            "success": self.success,
            "output": self.output,
            "error": self.error,
            "return_code": self.return_code,
            "metadata": self.metadata,
            "execution_time": self.execution_time,
            "memory_used_mb": self.memory_used_mb,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExecutionResult":
        """Create from dictionary format.

        Args:
            data: Dictionary representation

        Returns:
            ExecutionResult instance
        """
        return cls(
            success=data.get("success", False),
            output=data.get("output", ""),
            error=data.get("error", ""),
            return_code=data.get("return_code", 0),
            metadata=data.get("metadata", {}),
            execution_time=data.get("execution_time"),
            memory_used_mb=data.get("memory_used_mb"),
        )


@dataclass
class ServiceInfo:
    """Service information dataclass.

    Contains status, resource information, and metrics for a service.
    """

    name: str
    status: ServiceStatus
    resource_info: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary format.

        Returns:
            Dictionary representation of the service info
        """
        return {
            "name": self.name,
            "status": self.status.value,
            "resource_info": self.resource_info,
            "metrics": self.metrics,
        }


class BaseService(ABC):
    """Base service abstract class.

    Defines the unified lifecycle management interface for all services.
    All services (ProcessService, SandboxService, etc.) must inherit from this.
    """

    def __init__(self) -> None:
        """Initialize base service."""
        self._status = ServiceStatus.STOPPED

    @property
    @abstractmethod
    def service_name(self) -> str:
        """Service name.

        Returns:
            Unique service name identifier
        """
        pass

    @property
    def status(self) -> ServiceStatus:
        """Get service status.

        Returns:
            Current service status
        """
        return self._status

    @abstractmethod
    async def start(self) -> None:
        """Start service.

        Initialize resources, establish connections, etc.
        """
        pass

    @abstractmethod
    async def stop(self) -> None:
        """Stop service.

        Release resources, close connections, etc.
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Health check.

        Returns:
            True if service is healthy, False otherwise
        """
        pass

    @abstractmethod
    def get_info(self) -> ServiceInfo:
        """Get service information.

        Returns:
            ServiceInfo containing status, resource info, and metrics
        """
        pass

    async def restart(self) -> None:
        """Restart service.

        Stop and then start the service.
        """
        await self.stop()
        await self.start()
