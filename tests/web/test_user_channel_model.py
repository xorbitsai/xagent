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
