"""
Service manager for execution services.

Provides global access to execution services from anywhere in the application.
"""

import logging
from typing import Optional

from .process import ProcessService

logger = logging.getLogger(__name__)


# Global service instance
_process_service: Optional[ProcessService] = None


def get_process_service() -> Optional[ProcessService]:
    """Get the global ProcessService instance.

    Returns:
        ProcessService instance or None if not initialized
    """
    return _process_service


def set_process_service(service: ProcessService) -> None:
    """Set the global ProcessService instance.

    Args:
        service: ProcessService instance to set as global
    """
    global _process_service
    _process_service = service


def clear_process_service() -> None:
    """Clear the global ProcessService instance."""
    global _process_service
    _process_service = None
