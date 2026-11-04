import pytest

from xagent.core.tools.adapters.vibe.connector_runtime import (
    CONNECTOR_TYPE_CUSTOM_API,
    CONNECTOR_TYPE_MCP,
    ERROR_DELEGATED_AUTHORIZATION_FAILED,
    ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
    ERROR_RUNTIME_SECRET_UNAVAILABLE,
    REDACTED_RUNTIME_SECRET,
    ConnectorRef,
    ConnectorRuntimeError,
    redact_runtime_sensitive_payload,
    redact_runtime_value,
    validate_runtime_config_declaration,
    validate_runtime_source_key,
)


def test_connector_ref_round_trips_wire_shape() -> None:
    ref = ConnectorRef.from_wire(
        {"connector_type": CONNECTOR_TYPE_MCP, "connector_id": 123}
    )

    assert ref.connector_type == CONNECTOR_TYPE_MCP
    assert ref.connector_id == 123
    assert ref.storage_key == "mcp:123"
    assert ref.to_wire() == {"connector_type": "mcp", "connector_id": 123}


@pytest.mark.parametrize(
    "value",
    [
        None,
        "mcp:123",
        {"connector_type": "mcp", "connector_id": "123"},
        {"connector_type": "unknown", "connector_id": 123},
        {"connector_type": CONNECTOR_TYPE_CUSTOM_API, "connector_id": 0},
        {
            "connector_type": CONNECTOR_TYPE_CUSTOM_API,
            "connector_id": 1,
            "name": "ambiguous",
        },
    ],
)
def test_connector_ref_rejects_ambiguous_or_invalid_wire_shape(value) -> None:
    with pytest.raises(ValueError):
        ConnectorRef.from_wire(value)


@pytest.mark.parametrize("key", ["account_id", "tenant-123", "user_42", "A1"])
def test_validate_runtime_source_key_accepts_simple_keys(key: str) -> None:
    assert validate_runtime_source_key(key) == key


@pytest.mark.parametrize("key", ["", "shiftcare.com/account_id", "a.b", "space key"])
def test_validate_runtime_source_key_rejects_keys_with_parse_ambiguity(
    key: str,
) -> None:
    with pytest.raises(ValueError):
        validate_runtime_source_key(key)


def test_connector_runtime_error_is_public_safe_and_contains_connector_ref() -> None:
    ref = ConnectorRef(connector_type=CONNECTOR_TYPE_MCP, connector_id=7)

    error = ConnectorRuntimeError(
        ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
        "MCP OAuth authorization is unavailable",
        connector_ref=ref,
        details={"reason": "missing_grant"},
        status_code=401,
    )

    assert str(error) == (
        "mcp_oauth_authorization_failed: MCP OAuth authorization is unavailable"
    )
    assert error.status_code == 401
    assert error.to_public_error() == {
        "code": ERROR_MCP_OAUTH_AUTHORIZATION_FAILED,
        "message": "MCP OAuth authorization is unavailable",
        "details": {
            "reason": "missing_grant",
            "connector_ref": {"connector_type": "mcp", "connector_id": 7},
        },
    }


def test_delegated_and_managed_oauth_errors_are_distinct() -> None:
    assert ERROR_DELEGATED_AUTHORIZATION_FAILED == "delegated_authorization_failed"
    assert ERROR_MCP_OAUTH_AUTHORIZATION_FAILED == "mcp_oauth_authorization_failed"
    assert ERROR_DELEGATED_AUTHORIZATION_FAILED != ERROR_MCP_OAUTH_AUTHORIZATION_FAILED


def test_redact_runtime_value_does_not_preserve_secret_material() -> None:
    value = {
        "authorization": "Bearer secret-token",
        "nested": {"resource_owner_key": "person-1"},
    }

    redacted = redact_runtime_value(value)

    assert redacted == {
        "authorization": REDACTED_RUNTIME_SECRET,
        "nested": REDACTED_RUNTIME_SECRET,
    }
    assert "secret-token" not in repr(redacted)
    assert "person-1" not in repr(redacted)


def test_redact_runtime_sensitive_payload_recursively_redacts_credentials() -> None:
    value = {
        "headers": {"Authorization": "Bearer secret-token", "X-Safe": "ok"},
        "credentials": {
            "apikey": "compact-api-key-secret",
            "aws_secret_access_key": "aws-secret-access-key",
            "service_access_key": "service-access-key-secret",
            "secret_key": "secret-key-secret",
            "user_api_key": "api-key-secret",
            "api_key_v1": "versioned-api-key-secret",
            "private_key": "private-key-secret",
        },
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {"authorization": "Bearer tenant-token"},
            "auth_selector": {"resource_owner_key": "xagent:user:owner"},
        },
        "usage": {
            "prompt_tokens": 11,
            "total_tokens": 42,
            "token_type": "Bearer",
        },
        "oauth": {
            "access_token": "access-secret",
            "client_secret": "client-secret",
            "token": "generic-token-secret",
        },
        "tuple_body": ({"refresh_token": "tuple-refresh-secret"}, {"safe": "value"}),
        "body": [{"refresh_token": "refresh-secret"}, {"safe": "value"}],
    }

    redacted = redact_runtime_sensitive_payload(value)

    assert redacted == {
        "headers": {
            "Authorization": REDACTED_RUNTIME_SECRET,
            "X-Safe": "ok",
        },
        "credentials": {
            "apikey": REDACTED_RUNTIME_SECRET,
            "aws_secret_access_key": REDACTED_RUNTIME_SECRET,
            "service_access_key": REDACTED_RUNTIME_SECRET,
            "secret_key": REDACTED_RUNTIME_SECRET,
            "user_api_key": REDACTED_RUNTIME_SECRET,
            "api_key_v1": REDACTED_RUNTIME_SECRET,
            "private_key": REDACTED_RUNTIME_SECRET,
        },
        "connector_runtime": {
            "context": {"account_id": "6185"},
            "secrets": {"authorization": REDACTED_RUNTIME_SECRET},
            "auth_selector": {"resource_owner_key": REDACTED_RUNTIME_SECRET},
        },
        "usage": {
            "prompt_tokens": 11,
            "total_tokens": 42,
            "token_type": "Bearer",
        },
        "oauth": {
            "access_token": REDACTED_RUNTIME_SECRET,
            "client_secret": REDACTED_RUNTIME_SECRET,
            "token": REDACTED_RUNTIME_SECRET,
        },
        "tuple_body": (
            {"refresh_token": REDACTED_RUNTIME_SECRET},
            {"safe": "value"},
        ),
        "body": [
            {"refresh_token": REDACTED_RUNTIME_SECRET},
            {"safe": "value"},
        ],
    }
    assert "secret-token" not in repr(redacted)
    assert "compact-api-key-secret" not in repr(redacted)
    assert "aws-secret-access-key" not in repr(redacted)
    assert "service-access-key-secret" not in repr(redacted)
    assert "secret-key-secret" not in repr(redacted)
    assert "api-key-secret" not in repr(redacted)
    assert "versioned-api-key-secret" not in repr(redacted)
    assert "private-key-secret" not in repr(redacted)
    assert "tenant-token" not in repr(redacted)
    assert "xagent:user:owner" not in repr(redacted)
    assert "access-secret" not in repr(redacted)
    assert "client-secret" not in repr(redacted)
    assert "generic-token-secret" not in repr(redacted)
    assert "tuple-refresh-secret" not in repr(redacted)


def test_runtime_config_validation_rejects_mcp_context_to_transport_header() -> None:
    with pytest.raises(ValueError, match="context cannot bind"):
        validate_runtime_config_declaration(
            connector_type="mcp",
            runtime_input_schema={
                "context": {"account_id": {"type": "string", "required": True}}
            },
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {
                        "target_type": "transport_headers",
                        "key": "X-Account-ID",
                    },
                }
            ],
            allow_delegated_authorization=False,
        )


def test_runtime_config_validation_accepts_custom_api_context_header() -> None:
    validate_runtime_config_declaration(
        connector_type="custom_api",
        runtime_input_schema={
            "context": {"account_id": {"type": "string", "required": True}}
        },
        runtime_bindings=[
            {
                "source": {"input_type": "context", "key": "account_id"},
                "target": {"target_type": "headers", "key": "X-Account-ID"},
            }
        ],
        allow_delegated_authorization=False,
    )


def test_runtime_config_validation_rejects_object_header_binding() -> None:
    with pytest.raises(ValueError, match="object runtime values cannot bind"):
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema={"context": {"actor": {"type": "object"}}},
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "actor"},
                    "target": {"target_type": "headers", "key": "X-Actor"},
                }
            ],
            allow_delegated_authorization=False,
        )


def test_runtime_config_validation_requires_delegated_flag_for_authorization() -> None:
    with pytest.raises(ValueError, match="requires delegated authorization"):
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema={"secrets": {"authorization": {"type": "string"}}},
            runtime_bindings=[
                {
                    "source": {"input_type": "secrets", "key": "authorization"},
                    "target": {"target_type": "headers", "key": "Authorization"},
                }
            ],
            allow_delegated_authorization=False,
        )


def test_runtime_config_validation_rejects_static_header_conflict() -> None:
    with pytest.raises(ValueError, match="conflicts with static header"):
        validate_runtime_config_declaration(
            connector_type="custom_api",
            runtime_input_schema={"context": {"account_id": {"type": "string"}}},
            runtime_bindings=[
                {
                    "source": {"input_type": "context", "key": "account_id"},
                    "target": {"target_type": "headers", "key": "X-Account-ID"},
                }
            ],
            allow_delegated_authorization=False,
            static_headers={"x-account-id": "static"},
        )


@pytest.mark.asyncio
async def test_tool_registry_does_not_swallow_connector_runtime_error() -> None:
    from xagent.core.tools.adapters.vibe.factory import ToolRegistry

    saved_creators = list(ToolRegistry._tool_creators)
    saved_imported = ToolRegistry._modules_imported
    ToolRegistry._tool_creators = []
    ToolRegistry._modules_imported = True

    async def _creator(_config):
        raise ConnectorRuntimeError(
            ERROR_RUNTIME_SECRET_UNAVAILABLE,
            "Required runtime secret is unavailable.",
            details={"reason": "store_lost"},
        )

    try:
        ToolRegistry.register(_creator, categories={"mcp"})
        with pytest.raises(ConnectorRuntimeError) as exc_info:
            await ToolRegistry.create_registered_tools(object())
    finally:
        ToolRegistry._tool_creators = saved_creators
        ToolRegistry._modules_imported = saved_imported

    assert exc_info.value.code == ERROR_RUNTIME_SECRET_UNAVAILABLE
