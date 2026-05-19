"""
Execution module for process-level isolation.

Provides dynamic process execution - creates a new process for each execution.
"""

from .service.base import (
    BaseService,
    ExecutionResult,
    IsolationType,
    ServiceInfo,
    ServiceStatus,
)
from .service.manager import (
    clear_process_service,
    get_process_service,
    set_process_service,
)
from .service.process import ProcessService

__all__ = [
    "BaseService",
    "ExecutionResult",
    "IsolationType",
    "ServiceInfo",
    "ServiceStatus",
    "ProcessService",
    "get_process_service",
    "set_process_service",
    "clear_process_service",
]
