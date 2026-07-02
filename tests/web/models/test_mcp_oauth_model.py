from xagent.web.models.mcp_oauth import MCPUserOAuthToken


def test_table_and_columns():
    t = MCPUserOAuthToken.__table__
    assert t.name == "mcp_user_oauth_tokens"
    cols = set(t.columns.keys())
    assert {
        "id", "user_id", "mcpserver_id", "access_token", "refresh_token",
        "expires_at", "token_type", "scope", "status", "pkce_verifier",
        "state", "created_at", "updated_at",
    } <= cols


def test_unique_user_server_constraint():
    uniques = [
        c for c in MCPUserOAuthToken.__table__.constraints
        if c.__class__.__name__ == "UniqueConstraint"
    ]
    cols = {tuple(sorted(col.name for col in u.columns)) for u in uniques}
    assert ("mcpserver_id", "user_id") in cols
