from datetime import datetime, timezone

from xagent.web.models.user_channel import UserChannel
from xagent.web.schemas.user_channel import UserChannelResponse


def test_user_channel_constructor_accepts_config_dict() -> None:
    channel = UserChannel(
        user_id=1,
        channel_type="telegram",
        channel_name="Telegram Bot",
        config={"bot_token": "plain-token"},
        is_active=True,
    )

    assert channel.config["bot_token"] == "plain-token"
    assert channel._config["bot_token"] != "plain-token"


def test_user_channel_encrypts_slack_app_token() -> None:
    channel = UserChannel(
        user_id=1,
        channel_type="slack",
        channel_name="Slack Bot",
        config={
            "bot_token": "xoxb-plain",
            "app_token": "xapp-plain",
        },
        is_active=True,
    )

    assert channel.config["bot_token"] == "xoxb-plain"
    assert channel.config["app_token"] == "xapp-plain"
    assert channel._config["bot_token"] != "xoxb-plain"
    assert channel._config["app_token"] != "xapp-plain"


def test_oauth_channel_response_redacts_workspace_tokens() -> None:
    response = UserChannelResponse(
        id=1,
        user_id=2,
        channel_type="slack",
        channel_name="Acme",
        config={
            "installation_mode": "oauth",
            "team_id": "T1",
            "bot_token": "xoxb-secret",
        },
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    serialized = response.model_dump()
    assert "bot_token" not in serialized["config"]
    assert serialized["config"]["bot_token_configured"] is True
    assert serialized["config"]["team_id"] == "T1"


def test_channel_response_redacts_secrets_for_every_installation_mode() -> None:
    response = UserChannelResponse(
        id=1,
        user_id=2,
        channel_type="slack",
        channel_name="Manual Slack",
        config={
            "installation_mode": "manual",
            "bot_token": "xoxb-secret",
            "app_token": "xapp-secret",
            "allowed_users": None,
        },
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    serialized = response.model_dump()
    assert "bot_token" not in serialized["config"]
    assert "app_token" not in serialized["config"]
    assert serialized["config"]["bot_token_configured"] is True
    assert serialized["config"]["app_token_configured"] is True


def test_channel_response_redacts_feishu_app_secret() -> None:
    response = UserChannelResponse(
        id=3,
        user_id=2,
        channel_type="feishu",
        channel_name="Feishu Bot",
        config={"app_id": "cli_123", "app_secret": "feishu-secret"},
        is_active=True,
        created_at=datetime.now(timezone.utc),
    )

    serialized = response.model_dump()
    assert "app_secret" not in serialized["config"]
    assert serialized["config"]["app_secret_configured"] is True
    assert serialized["config"]["app_id"] == "cli_123"
