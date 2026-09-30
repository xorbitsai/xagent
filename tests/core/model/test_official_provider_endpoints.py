"""Official endpoint construction and credential field declarations.

Covers the shared provider contract: Azure's resource-derived official
endpoint, per-provider credential field declarations, the single
resolution path model construction uses, and what the base-URL model
configuration pages are offered from it.
"""

import pytest

from xagent.core.model.model import ChatModelConfig
from xagent.core.model.providers import (
    AZURE_OPENAI_ENDPOINT_SUFFIX,
    ENDPOINT_KIND_AZURE_RESOURCE,
    azure_resource_endpoint_for_resource_name,
    get_supported_provider_metadata,
    official_endpoint_for_provider,
    provider_credential_fields,
    provider_endpoint_kind,
)
from xagent.web.services.model_list_service import get_supported_providers


def _metadata_entry(provider_id: str) -> dict | None:
    for entry in get_supported_provider_metadata():
        if entry["id"] == provider_id:
            return entry
    return None


class TestAzureMetadata:
    def test_azure_openai_is_a_registered_llm_provider(self):
        entry = _metadata_entry("azure_openai")
        assert entry is not None
        assert entry["requires_base_url"] is False
        assert "llm" in entry["category"]
        # No openai_compatible marker: the adapter's dedicated azure_openai
        # branch must stay reachable.
        assert "compatibility" not in entry
        assert provider_endpoint_kind("azure_openai") == ENDPOINT_KIND_AZURE_RESOURCE

    def test_azure_declares_its_two_generic_fields(self):
        assert provider_credential_fields("azure_openai") == [
            {
                "name": "resource_name",
                "label": "Resource name",
                "kind": "plain",
                "required": True,
            },
            {"name": "api_key", "label": "API key", "kind": "secret", "required": True},
        ]

    def test_other_providers_get_the_single_key_default(self):
        default = [
            {"name": "api_key", "label": "API key", "kind": "secret", "required": True}
        ]
        assert provider_credential_fields("deepseek") == default
        assert provider_credential_fields("openrouter") == default
        assert provider_credential_fields("openai-compatible") == default
        assert provider_credential_fields("not-a-provider") == default

    @pytest.mark.parametrize("provider", ["xinference", " Xinference "])
    def test_xinference_declares_an_optional_key(self, provider):
        assert provider_credential_fields(provider) == [
            {"name": "api_key", "label": "API key", "kind": "secret", "required": False}
        ]

    @pytest.mark.parametrize("provider", ["azure_openai", "deepseek", "not-a-provider"])
    def test_credential_fields_are_fresh_copies(self, provider):
        fields = provider_credential_fields(provider)
        expected = [dict(field) for field in fields]
        fields[0]["label"] = "changed"
        fields.append({"name": "extra"})
        assert provider_credential_fields(provider) == expected

    def test_metadata_entries_are_deep_copies(self):
        entry = _metadata_entry("azure_openai")
        assert entry is not None
        entry["credential_fields"][0]["label"] = "changed"
        entry["credential_fields"].append({"name": "extra"})
        entry["category"].append("image")
        again = _metadata_entry("azure_openai")
        assert again is not None
        assert again["credential_fields"] == provider_credential_fields("azure_openai")
        assert again["credential_fields"][0]["label"] == "Resource name"
        assert again["category"] == ["llm"]


class TestAzureResourceEndpoint:
    def test_the_endpoint_is_built_from_a_validated_identifier(self):
        assert azure_resource_endpoint_for_resource_name("my-team") == (
            f"https://my-team{AZURE_OPENAI_ENDPOINT_SUFFIX}"
        )
        # Case is normalized; the host is always the official Microsoft domain.
        assert azure_resource_endpoint_for_resource_name("MyTeam-01") == (
            f"https://myteam-01{AZURE_OPENAI_ENDPOINT_SUFFIX}"
        )

    @pytest.mark.parametrize(
        "bad",
        [
            "",
            "a",
            "-bad",
            "bad-",
            "bad_name",
            "has space",
            "x" * 65,
            "bad.",
            "evil.example.com",
            "my-team.openai.azure.com.evil.example",
            "https://my-team.openai.azure.com",
            "my-team:443",
        ],
    )
    def test_no_url_or_invalid_identifier_is_accepted(self, bad):
        with pytest.raises(ValueError, match="resource_name"):
            azure_resource_endpoint_for_resource_name(bad)


class TestOfficialEndpointResolution:
    def test_azure_resolves_through_the_constructed_endpoint(self):
        endpoint = official_endpoint_for_provider(
            "azure_openai", {"resource_name": "my-team", "api_key": "k"}
        )
        assert endpoint == f"https://my-team{AZURE_OPENAI_ENDPOINT_SUFFIX}"

    def test_azure_requires_the_identifier_before_any_call(self):
        with pytest.raises(ValueError, match="resource_name"):
            official_endpoint_for_provider("azure_openai", {"api_key": "k"})
        with pytest.raises(ValueError, match="resource_name"):
            official_endpoint_for_provider("azure_openai")

    def test_other_providers_keep_the_registry_resolution(self):
        assert (
            official_endpoint_for_provider("openrouter")
            == "https://openrouter.ai/api/v1"
        )

    @pytest.mark.parametrize(
        "provider",
        ["claude", "gemini", "openai-compatible", "xinference", "not-a-provider"],
    )
    def test_none_when_the_registry_has_no_endpoint(self, provider):
        assert official_endpoint_for_provider(provider) is None

    def test_a_scoped_env_override_replaces_the_default(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://deepseek.example/v1")
        assert official_endpoint_for_provider("deepseek") == (
            "https://deepseek.example/v1"
        )
        monkeypatch.delenv("DEEPSEEK_BASE_URL")
        assert official_endpoint_for_provider("deepseek") == "https://api.deepseek.com"

    def test_the_azure_adapter_branch_stays_reachable(self):
        from xagent.core.model.chat.basic.adapter import create_base_llm
        from xagent.core.model.chat.basic.azure_openai import AzureOpenAILLM

        config = ChatModelConfig(
            id="my-deployment",
            model_provider="azure_openai",
            model_name="my-deployment",
            api_key="k",
            base_url=official_endpoint_for_provider(
                "azure_openai", {"resource_name": "my-team"}
            ),
            abilities=["chat", "tool_calling"],
        )
        llm = create_base_llm(config)
        assert isinstance(llm._inner, AzureOpenAILLM)
        assert (
            llm._inner.azure_endpoint
            == f"https://my-team{AZURE_OPENAI_ENDPOINT_SUFFIX}"
        )


class TestNativeProviderList:
    def test_credential_field_endpoints_are_not_offered_to_native_pages(self):
        native = {provider["id"] for provider in get_supported_providers()}
        shared = {provider["id"] for provider in get_supported_provider_metadata()}
        # The shared contract keeps Azure for hosts that collect its fields...
        assert "azure_openai" in shared
        # ...while the base-URL-only native pages do not offer it.
        assert "azure_openai" not in native
        assert native == shared - {"azure_openai"}

    def test_routing_hints_stay_on_native_pages_only(self):
        native = {p["id"]: p for p in get_supported_providers()}
        shared = {p["id"]: p for p in get_supported_provider_metadata()}
        # The exact description these pages showed before the hint moved out.
        assert native["openrouter"]["description"] == (
            "OpenRouter aggregator: reach Claude, Gemini, GPT, DeepSeek, GLM, "
            "and more through one OpenAI-compatible key. Use model 'auto' to let "
            "xrouter-llm pick the cheapest capable model per prompt."
        )
        assert "routing_hint" not in native["openrouter"]
        assert "auto" not in shared["openrouter"]["description"]
        assert shared["openrouter"]["routing_hint"].startswith("Use model 'auto'")

    def test_native_pages_get_only_the_fields_they_already_had(self):
        page_fields = {
            "id",
            "name",
            "description",
            "requires_base_url",
            "compatibility",
            "default_base_url",
            "category",
        }
        for provider in get_supported_providers():
            assert set(provider) <= page_fields, provider["id"]
        # The declarations stay available to hosts that read the registry.
        shared = {p["id"]: p for p in get_supported_provider_metadata()}
        assert shared["xinference"]["credential_fields"][0]["required"] is False
