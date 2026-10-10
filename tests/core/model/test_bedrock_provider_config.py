import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.model import ChatModelConfig
from xagent.core.model.providers import (
    get_supported_provider_metadata,
    provider_credential_fields,
    validate_bedrock_settings,
)
from xagent.core.model.storage.db.adapter import SQLAlchemyModelHub
from xagent.core.model.storage.db.db_models import create_model_table


def test_validate_bedrock_api_key_settings() -> None:
    validate_bedrock_settings(
        region="us-east-1",
        auth_mode="api_key",
        api_key="bedrock-token",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )


def test_bedrock_provider_metadata_declares_settings_contract() -> None:
    assert any(
        provider["id"] == "bedrock" for provider in get_supported_provider_metadata()
    )
    assert provider_credential_fields("bedrock") == [
        {
            "name": "api_key",
            "label": "Bedrock API key",
            "kind": "secret",
            "required": False,
        },
        {
            "name": "bedrock_region",
            "label": "AWS region",
            "kind": "plain",
            "required": True,
        },
        {
            "name": "bedrock_auth_mode",
            "label": "Authentication mode",
            "kind": "plain",
            "required": True,
        },
    ]


def test_validate_bedrock_credentials_chain_settings() -> None:
    validate_bedrock_settings(
        region="eu-west-1",
        auth_mode="credentials_chain",
        api_key=None,
        endpoint_url=None,
    )


def test_validate_bedrock_credentials_chain_rejects_explicit_api_key() -> None:
    with pytest.raises(ValueError, match="cannot include an API key"):
        validate_bedrock_settings(
            region="us-east-1",
            auth_mode="credentials_chain",
            api_key="must-not-be-retained",
            endpoint_url=None,
        )


def test_credentials_chain_maps_to_runtime_aws_credentials_mode() -> None:
    llm = create_base_llm(
        ChatModelConfig(
            id="bedrock-chain",
            model_provider="bedrock",
            model_name="amazon.nova-lite-v1:0",
            api_key="",
            bedrock_region="us-east-1",
            bedrock_auth_mode="credentials_chain",
        )
    )

    assert llm._inner.auth_mode == "aws_credentials"


def test_standalone_model_hub_round_trips_bedrock_settings(monkeypatch) -> None:
    monkeypatch.setenv("ENCRYPTION_KEY", "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc=")
    engine = create_engine("sqlite:///:memory:")
    base = declarative_base()
    model = create_model_table(base)
    base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    hub = SQLAlchemyModelHub(session, model)

    hub.store(
        ChatModelConfig(
            id="bedrock-profile",
            model_provider="bedrock",
            model_name=(
                "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example"
            ),
            api_key="bedrock-token",
            base_url="https://bedrock-runtime.us-west-2.amazonaws.com",
            bedrock_region="us-west-2",
            bedrock_auth_mode="api_key",
        )
    )

    loaded = hub.load("bedrock-profile")
    listed = hub.list()["bedrock-profile"]
    assert isinstance(loaded, ChatModelConfig)
    assert loaded.bedrock_region == listed.bedrock_region == "us-west-2"
    assert loaded.bedrock_auth_mode == listed.bedrock_auth_mode == "api_key"
    assert loaded.model_name == (
        "arn:aws:bedrock:us-west-2:123456789012:inference-profile/example"
    )
    session.close()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("region", "", "valid AWS region"),
        ("auth_mode", "ambient", "authentication mode"),
        ("api_key", "", "API key is required"),
        (
            "endpoint_url",
            "https://bedrock-mantle.us-east-1.api.aws/v1",
            "do not support Converse",
        ),
    ],
)
def test_validate_bedrock_rejects_invalid_settings(
    field: str, value: str, message: str
) -> None:
    settings = {
        "region": "us-east-1",
        "auth_mode": "api_key",
        "api_key": "bedrock-token",
        "endpoint_url": None,
    }
    settings[field] = value
    with pytest.raises(ValueError, match=message):
        validate_bedrock_settings(**settings)
