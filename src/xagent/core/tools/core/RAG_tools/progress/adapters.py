"""Adapters for integrating different parsers with progress callbacks."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional

from .tracker import ProgressCallback

logger = logging.getLogger(__name__)


class DeepDocProgressAdapter:
    """Adapter to convert DeepDoc's callback format to our ProgressCallback interface.

    DeepDoc uses callback(progress_float, message_string) format.
    We convert this to our status-based callback format.
    """

    def __init__(self, callback: ProgressCallback):
        self.callback = callback
        self.last_status: Optional[str] = None

    def get_callback(self) -> Callable[[float, Optional[str]], None]:
        """Get a callback function compatible with DeepDoc's format.

        Returns:
            A function that can be passed to DeepDoc's parse_into_bboxes()
        """

        def deepdoc_callback(progress: float, message: Optional[str] = None) -> None:
            """Convert DeepDoc callback to our format.

            DeepDoc provides:
            - progress: float (0.0-1.0, e.g., 0.4, 0.63, 0.83, 0.92)
            - message: Optional[str] (e.g., "OCR finished (2.34s)", "Layout analysis (1.23s)")

            Note:
                DeepDoc's internal OCR stage sometimes invokes callback with only
                one argument (progress). We therefore keep `message` optional and
                fall back to a stable status label.

            We extract the meaningful status information.
            """
            try:
                # Some DeepDoc paths emit only progress without a message.
                message_text = message or "OCR processing"

                # Extract the core status message from DeepDoc's format
                # Examples:
                # "OCR finished (2.34s)" -> "OCR finished"
                # "Layout analysis (1.23s)" -> "Layout analysis"
                # "Table analysis (0.89s)" -> "Table analysis"
                # "Text merged (1.45s)" -> "Text merged"

                if " (" in message_text and ")" in message_text:
                    # Remove timing information
                    status = message_text.split(" (")[0]
                else:
                    status = message_text

                # Only report status changes to avoid spam
                if status != self.last_status:
                    self.callback.on_status_update(status)
                    self.last_status = status

            except Exception as e:
                logger.warning("Failed to process DeepDoc progress callback: %s", e)
                # Fallback: report the raw message
                self.callback.on_status_update(message or "OCR processing")

        return deepdoc_callback


class FallbackProgressAdapter:
    """Adapter for parsers that don't support progress callbacks.

    Provides reasonable status updates for parsers without native progress support.
    """

    def __init__(self, callback: ProgressCallback):
        self.callback = callback

    def report_status(
        self, status: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report a status update."""
        self.callback.on_status_update(status, details)


def create_progress_adapter(parser_type: str, callback: ProgressCallback) -> Any:
    """Factory function to create appropriate progress adapter for different parsers.

    Args:
        parser_type: Type of parser ("deepdoc", "pypdf", "pdfplumber", etc.)
        callback: Our progress callback interface

    Returns:
        Adapter instance appropriate for the parser type
    """
    if parser_type == "deepdoc":
        return DeepDocProgressAdapter(callback)
    else:
        # For other parsers, return a fallback adapter
        return FallbackProgressAdapter(callback)
