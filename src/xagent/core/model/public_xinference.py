"""Public-only sessions at the Xinference SDK's client and model boundaries."""

from typing import Any

import aiohttp
from xinference_client import RESTfulClient
from xinference_client.client.restful.async_restful_client import (
    AsyncClient,
    AsyncRESTfulChatModelHandle,
)

from .network_policy import ModelNetworkPolicy
from .public_http import create_public_model_aiohttp_session
from .public_requests import Session


class PublicXinferenceClient(RESTfulClient):  # type: ignore[no-any-unimported]  # SDK is untyped
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.session: Any = None
        ModelNetworkPolicy().destination(base_url)
        try:
            super().__init__(base_url=base_url, api_key=api_key)
        except BaseException:
            if self.session is not None:
                self.session.close()
            raise
        if not isinstance(self.session, Session):
            self.close()
            raise RuntimeError("Unsupported Xinference SDK session contract")

    def _check_cluster_authenticated(self) -> None:
        # The SDK constructor calls this hook before its first HTTP request.
        self.session.close()
        self.session = Session()
        super()._check_cluster_authenticated()

    def get_model(self, model_uid: str) -> Any:
        handle = super().get_model(model_uid)
        handle.session.close()
        handle.session = Session()
        return handle


class PublicAsyncXinferenceClient(AsyncClient):  # type: ignore[no-any-unimported]  # SDK is untyped
    def __init__(self, base_url: str, api_key: str | None = None) -> None:
        self.session: Any = None
        ModelNetworkPolicy().destination(base_url)
        # The SDK's constructor creates an unrestricted aiohttp session and
        # performs a synchronous Requests auth probe. Initialize its tested
        # 2.x client state here instead, preserving the chat adapter's existing
        # explicit-key/no-blocking-probe behavior. Bound to the tested 2.x SDK;
        # real-SDK regressions cover discovery and every returned model handle.
        self.base_url = base_url
        self._headers = (
            {"Authorization": f"Bearer {api_key}"} if api_key is not None else {}
        )
        self._cluster_authed = api_key is not None
        self.timeout = aiohttp.ClientTimeout(total=1800)
        self.session = create_public_model_aiohttp_session(timeout=self.timeout)

    async def get_model(self, model_uid: str) -> Any:
        handle = await super().get_model(model_uid)
        return await self._guard_model_handle(handle)

    async def get_chat_model(self, model_uid: str) -> tuple[Any, bool]:
        """Discover the chat handle and its routing support in one request."""
        description = await self.describe_model(model_uid)
        if description.get("model_type") != "LLM" or "chat" not in description.get(
            "model_ability", []
        ):
            raise ValueError(f"Model {model_uid} does not support chat")
        handle = AsyncRESTfulChatModelHandle(
            model_uid, self.base_url, auth_headers=self._headers
        )
        return (
            await self._guard_model_handle(handle),
            description.get("supports_chat_session_affinity") is True,
        )

    async def _guard_model_handle(self, handle: Any) -> Any:
        await handle.session.close()
        handle.session = create_public_model_aiohttp_session(timeout=handle.timeout)
        return handle
