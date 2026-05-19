"""
Service layer for execution management.

Provides lifecycle management for execution services including process isolation.
"""

from .base import (
    BaseService,
    ExecutionResult,
    IsolationType,
    ServiceInfo,
    ServiceStatus,
)
from .manager import (
    clear_process_service,
    get_process_service,
    set_process_service,
)
from .process import ProcessService

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
