import pytest

from xagent.core.model.providers import validate_bedrock_settings


def test_validate_bedrock_api_key_settings() -> None:
    validate_bedrock_settings(
        region="us-east-1",
        auth_mode="api_key",
        api_key="bedrock-token",
        endpoint_url="https://bedrock-runtime.us-east-1.amazonaws.com",
    )


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
