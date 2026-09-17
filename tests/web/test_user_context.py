"""Structural and behavior pins for MCP user-context utilities."""

from xagent.web import user_context


def test_user_context_module_removes_request_auth_helpers_and_preserves_utilities(
    monkeypatch,
) -> None:
    """Only explicit context and access utilities remain at this boundary."""
    assert not hasattr(user_context, "get_user_context_from_request")
    assert not hasattr(user_context, "create_user_context")

    monkeypatch.setenv("XAGENT_USER_ID", "original-user")
    context = user_context.UserContext("tool-user")
    with context.set_context():
        assert context.get_current_user() == "tool-user"
    assert context.get_current_user() == "original-user"

    assert user_context.validate_user_access(None, ["system"]) is True
    assert user_context.validate_user_access("tool-user", None) is True
    assert user_context.validate_user_access("tool-user", ["other-user"]) is False
