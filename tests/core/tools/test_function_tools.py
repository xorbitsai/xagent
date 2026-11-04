import pytest
from pydantic import BaseModel, ValidationError

from xagent.core.tools.adapters.vibe.base import ToolVisibility
from xagent.core.tools.adapters.vibe.function import FunctionTool

# ========== Test Functions ==========


class GreetOutput(BaseModel):
    greeting: str


def greet(name: str) -> GreetOutput:
    return GreetOutput(greeting=f"Hello, {name}!")


async def multiply(a: int, b: int) -> dict:
    return {"result": a * b}


# ========== Tests ==========


def test_sync_function_tool():
    tool = FunctionTool(greet)

    assert tool.name == "greet"
    assert tool.is_async() is False

    args_model = tool.args_type()
    ret_model = tool.return_type()

    parsed = args_model(name="Alice")
    assert parsed.name == "Alice"

    result = tool.run_json_sync({"name": "Alice"})
    assert result == {"greeting": "Hello, Alice!"}

    # Check return model parsing
    parsed_ret = ret_model(**result)
    if hasattr(parsed_ret, "greeting"):
        assert parsed_ret.greeting == "Hello, Alice!"
    elif hasattr(parsed_ret, "root"):
        assert parsed_ret.root["greeting"] == "Hello, Alice!"
    else:
        raise AssertionError("Unexpected return model structure")

    # Check metadata
    metadata = tool.metadata
    assert metadata.name == "greet"
    assert metadata.visibility == ToolVisibility.PRIVATE
    assert metadata.has_state is False


@pytest.mark.asyncio
async def test_async_function_tool():
    tool = FunctionTool(multiply, name="multiply", tags=["math"])

    assert tool.is_async() is True

    result = await tool.run_json_async({"a": 3, "b": 5})
    assert result == {"result": 15}

    args_model = tool.args_type()
    assert issubclass(args_model, BaseModel)
    parsed = args_model(a=2, b=4)
    assert parsed.model_dump() == {"a": 2, "b": 4}

    ret_model = tool.return_type()
    assert issubclass(ret_model, BaseModel)
    parsed_ret = ret_model(**result)
    assert parsed_ret.root["result"] == 15

    metadata = tool.metadata
    assert metadata.tags == ["math"]
    assert metadata.has_state is False


def test_invalid_sync_args():
    tool = FunctionTool(greet)
    with pytest.raises(ValidationError):
        tool.run_json_sync({"wrong": 123})


@pytest.mark.asyncio
async def test_invalid_async_args():
    tool = FunctionTool(multiply)
    with pytest.raises(ValidationError):
        await tool.run_json_async({"a": "bad", "b": 2})
