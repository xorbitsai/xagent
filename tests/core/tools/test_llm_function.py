"""Test llm() function in Python executor"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from xagent.core.tools.core.python_executor import PythonExecutorCore


class TestLLMFunction:
    """Test llm() function injected into Python execution environment"""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM instance"""
        mock_llm = MagicMock()

        async def mock_chat(messages):
            await asyncio.sleep(0.01)
            return {"content": "Test LLM response"}

        mock_llm.chat = mock_chat
        return mock_llm

    @pytest.fixture
    def executor(self, mock_llm):
        """Create executor with mock LLM"""
        return PythonExecutorCore(llm=mock_llm)

    def test_llm_function_available(self, executor):
        """Test that llm() function is available in execution environment"""
        code = "result = llm('test'); print('llm available')"
        result = executor.execute_code(code)
        assert result["success"]
        assert "llm available" in result["output"]

    def test_llm_basic_call(self, executor):
        """Test basic llm() function call"""
        code = 'result = llm("What is 2+2?"); print(result)'
        result = executor.execute_code(code)
        assert result["success"]
        assert "Test LLM response" in result["output"]

    def test_llm_with_system_prompt(self, executor):
        """Test llm() with system prompt parameter"""
        code = """result = llm("Translate to English: Bonjour", system_prompt="You are a translator")
print(result)"""
        result = executor.execute_code(code)
        assert result["success"]
        assert "Test LLM response" in result["output"]

    def test_llm_in_thread_pool(self, mock_llm):
        """Test llm() function when called from ThreadPoolExecutor (actual usage scenario)"""

        def run_in_thread():
            code = 'result = llm("Test in thread"); print(result)'
            executor = PythonExecutorCore(llm=mock_llm)
            return executor.execute_code(code)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_thread)
            result = future.result(timeout=5)
            assert result["success"]
            assert "Test LLM response" in result["output"]

    def test_llm_multiple_calls(self, executor):
        """Test multiple sequential llm() calls"""
        code = """r1 = llm("First")
r2 = llm("Second")
print(f"{r1} - {r2}")"""
        result = executor.execute_code(code)
        assert result["success"]
        assert "Test LLM response" in result["output"]

    def test_llm_no_llm_available(self):
        """Test llm() when no LLM instance is available"""
        executor = PythonExecutorCore(llm=None)
        code = 'result = llm("test")'
        result = executor.execute_code(code)
        assert not result["success"]
        # When no LLM, llm function is not injected, so it's a NameError
        assert "llm" in result["error"] or "not defined" in result["error"]

    def test_llm_error_propagation(self, mock_llm):
        """Test that LLM errors are properly propagated"""

        async def failing_chat(messages):
            raise ValueError("LLM API error")

        mock_llm.chat = failing_chat
        executor = PythonExecutorCore(llm=mock_llm)

        code = 'result = llm("test")'
        result = executor.execute_code(code)
        assert not result["success"]
        assert "LLM API error" in result["error"]
