"""
Base class for Xinference model clients.

Provides common functionality for ASR, TTS, and other Xinference-based models.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Protocol

try:
    from xinference.client.restful.restful_client import (
        RESTfulClient as XinferenceClient,
    )
except ImportError:
    from xinference_client import RESTfulClient as XinferenceClient  # noqa: F401

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Protocol for xinference model handle."""

    def close(self) -> None: ...


class BaseXinferenceModel:
    """
    Base class for Xinference model clients.

    Provides common functionality for client initialization, session management,
    and resource cleanup.
    """

    def __init__(
        self,
        model: str,
        model_uid: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize Xinference model client.

        Args:
            model: Model name (e.g., "whisper-base", "chat-tts")
            model_uid: Unique model UID in Xinference (if model is already launched)
            base_url: Xinference server base URL (e.g., "http://localhost:9997")
            api_key: Optional API key for authentication
        """
        self.model = model
        self._model_uid = model_uid or model
        self.base_url = (base_url or "http://localhost:9997").rstrip("/")
        self.api_key = api_key

        # Initialize the Xinference client (lazy initialization)
        self._client: Optional[Any] = None  # AsyncClient
        self._model_handle: Optional[ModelProtocol] = None

    async def _get_session(self) -> Any:  # AsyncClient
        """Get or create async Xinference client."""
        if self._client is None:
            try:
                # Try to import from local xinference package first
                from xinference.client.restful.async_restful_client import (
                    AsyncClient,
                )
            except ImportError:
                # Fallback to xinference_client package
                from xinference_client.client.restful.async_restful_client import (  # type: ignore
                    AsyncClient,
                )

            self._client = AsyncClient(base_url=self.base_url, api_key=self.api_key)
        return self._client

    async def _ensure_model_handle(self) -> Any:  # AsyncModelProtocol
        """Ensure the model handle is initialized."""
        if self._model_handle is None:
            client = await self._get_session()
            # Get the model handle (assumes model is already launched on the server)
            self._model_handle = await client.get_model(self._model_uid)
        return self._model_handle

    def close(self) -> None:
        """Close the Xinference client and cleanup resources (sync version)."""
        if self._model_handle is not None:
            try:
                self._model_handle.close()
            except Exception:
                pass
            self._model_handle = None

        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    async def aclose(self) -> None:
        """Close the Xinference client and cleanup resources (async version)."""
        self.close()

    def __enter__(self) -> "BaseXinferenceModel":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit - cleanup resources."""
        self.close()
