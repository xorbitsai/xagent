"""Token-usage details carry model_id, disambiguating identically-named models."""

from xagent.core.model import ChatModelConfig
from xagent.core.model.chat.basic.adapter import create_base_llm
from xagent.core.model.chat.token_context import TokenContextManager, add_token_usage


def test_create_base_llm_stamps_model_id():
    """create_base_llm stamps ChatModelConfig.id so adapters report it."""
    config = ChatModelConfig(
        id="deepseek-plat-abc",
        model_provider="deepseek",
        model_name="deepseek-v4-flash",
        api_key="test-api-key",
    )
    llm = create_base_llm(config)
    # The retry wrapper delegates attribute access to the inner LLM.
    assert llm._inner.model_id == "deepseek-plat-abc"
    assert llm.model_id == "deepseek-plat-abc"


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
    assert details[0]["cached_tokens"] == 0


def test_add_token_usage_records_cached_tokens():
    with TokenContextManager() as mgr:
        add_token_usage(
            input_tokens=100,
            model="m",
            model_id="platform/x",
            call_type="chat",
            cached_input_tokens=40,
        )
        details = mgr.get_usage().details

    inp = next(d for d in details if d["type"] == "input")
    assert inp["tokens"] == 100
    assert inp["cached_tokens"] == 40


def test_cached_input_tokens_helper():
    from xagent.core.model.chat.basic.openai import _cached_input_tokens

    class DeepSeekUsage:
        prompt_cache_hit_tokens = 30

    class Details:
        cached_tokens = 25

    class OpenAIUsage:
        prompt_tokens_details = Details()

    class NoCache:
        pass

    assert _cached_input_tokens(DeepSeekUsage()) == 30  # deepseek field
    assert _cached_input_tokens(OpenAIUsage()) == 25  # openai/dashscope field
    assert _cached_input_tokens(NoCache()) == 0
    assert _cached_input_tokens(None) == 0
