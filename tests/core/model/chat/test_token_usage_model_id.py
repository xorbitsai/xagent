"""Token-usage details carry model_id, disambiguating identically-named models."""

from xagent.core.model.chat.token_context import TokenContextManager, add_token_usage


def test_add_token_usage_records_model_id():
    with TokenContextManager() as mgr:
        add_token_usage(
            input_tokens=10,
            output_tokens=5,
            model="deepseek-v4-flash",
            model_id="platform-ds",
            call_type="chat",
        )
        details = mgr.get_usage().details

    by_type = {d["type"]: d for d in details}
    assert by_type["input"]["model_id"] == "platform-ds"
    assert by_type["input"]["model"] == "deepseek-v4-flash"
    assert by_type["output"]["model_id"] == "platform-ds"


def test_model_id_defaults_empty_when_absent():
    with TokenContextManager() as mgr:
        add_token_usage(input_tokens=3, model="m", call_type="chat")
        details = mgr.get_usage().details

    assert details[0]["model_id"] == ""
