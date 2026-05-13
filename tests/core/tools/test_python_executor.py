"""
Tests for PythonExecutor tool
"""

import sys

import pytest

from xagent.core.tools.adapters.vibe.python_executor import (
    PythonExecutorArgs,
    PythonExecutorResult,
    PythonExecutorTool,
)


def _is_module_available(module_name):
    """Helper function to check if a module is available"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


@pytest.fixture
def python_executor():
    """Create PythonExecutorTool instance for testing"""
    return PythonExecutorTool()


class TestPythonExecutorTool:
    """Test cases for PythonExecutorTool"""

    def test_tool_properties(self, python_executor):
        """Test basic tool properties"""
        assert python_executor.name == "python_executor"
        assert "python" in python_executor.tags
        assert python_executor.args_type() == PythonExecutorArgs
        assert python_executor.return_type() == PythonExecutorResult

    def test_simple_calculation(self, python_executor):
        """Test simple mathematical calculation"""
        result = python_executor.run_json_sync(
            {"code": "result = 2 + 3\nprint(result)", "capture_output": True}
        )

        assert result["success"] is True
        assert "5" in result["output"]
        assert result["error"] == ""

    def test_variable_creation_display(self, python_executor):
        """Test that variables are displayed when no output is produced"""
        result = python_executor.run_json_sync(
            {"code": "x = 10\ny = 'hello'\nz = [1, 2, 3]", "capture_output": True}
        )

        assert result["success"] is True
        assert "Variables created:" in result["output"]
        assert "x = 10" in result["output"]
        assert "y = 'hello'" in result["output"]
        assert "z = [1, 2, 3]" in result["output"]

    def test_import_visible_inside_comprehension(self, python_executor):
        """Test imported modules are visible inside nested scopes."""
        result = python_executor.run_json_sync(
            {
                "code": (
                    "import random\n"
                    "values = [random.randint(1, 10) for _ in range(3)]\n"
                    "print(len(values), all(1 <= value <= 10 for value in values))"
                ),
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "3 True" in result["output"]
        assert result["error"] == ""

    def test_print_output(self, python_executor):
        """Test code with print statements"""
        result = python_executor.run_json_sync(
            {
                "code": "print('Hello, World!')\nprint('Second line')",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Hello, World!" in result["output"]
        assert "Second line" in result["output"]
        assert result["error"] == ""

    def test_syntax_error(self, python_executor):
        """Test handling of syntax errors"""
        result = python_executor.run_json_sync(
            {"code": "if True\n    print('missing colon')", "capture_output": True}
        )

        assert result["success"] is False
        assert result["output"] == ""
        assert "Syntax Error" in result["error"]

    def test_runtime_error(self, python_executor):
        """Test handling of runtime errors"""
        result = python_executor.run_json_sync(
            {"code": "x = 1 / 0", "capture_output": True}
        )

        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_import_error_handling(self, python_executor):
        """Test handling of import errors"""
        result = python_executor.run_json_sync(
            {"code": "import nonexistent_module", "capture_output": True}
        )

        assert result["success"] is False
        assert "ModuleNotFoundError" in result["error"]

    def test_builtin_functions_available(self, python_executor):
        """Test that common builtin functions are available"""
        result = python_executor.run_json_sync(
            {
                "code": """
numbers = [1, 2, 3, 4, 5]
print(f"Length: {len(numbers)}")
print(f"Sum: {sum(numbers)}")
print(f"Max: {max(numbers)}")
print(f"Min: {min(numbers)}")
print(f"Sorted: {sorted(numbers, reverse=True)}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Length: 5" in result["output"]
        assert "Sum: 15" in result["output"]
        assert "Max: 5" in result["output"]
        assert "Min: 1" in result["output"]

    def test_math_module_available(self, python_executor):
        """Test that math module is available"""
        result = python_executor.run_json_sync(
            {
                "code": """
import math
print(f"Pi: {math.pi}")
print(f"Square root of 16: {math.sqrt(16)}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Pi:" in result["output"]
        assert "Square root of 16: 4" in result["output"]

    def test_json_module_available(self, python_executor):
        """Test that json module is available"""
        result = python_executor.run_json_sync(
            {
                "code": """
import json
data = {"name": "test", "value": 42}
json_str = json.dumps(data)
print(json_str)
parsed = json.loads(json_str)
print(parsed["name"])
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert '"name": "test"' in result["output"]
        assert "test" in result["output"]

    def test_datetime_module_available(self, python_executor):
        """Test that datetime module is available"""
        result = python_executor.run_json_sync(
            {
                "code": """
import datetime
now = datetime.datetime.now()
print(f"Year: {now.year}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Year:" in result["output"]

    def test_re_module_available(self, python_executor):
        """Test that re module is available"""
        result = python_executor.run_json_sync(
            {
                "code": """
import re
pattern = r'\\d+'
text = "I have 123 apples"
match = re.search(pattern, text)
if match:
    print(f"Found number: {match.group()}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Found number: 123" in result["output"]

    @pytest.mark.skipif(not _is_module_available("numpy"), reason="numpy not available")
    def test_numpy_available_if_installed(self, python_executor):
        """Test that numpy is available if installed"""
        result = python_executor.run_json_sync(
            {
                "code": """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Mean: {np.mean(arr)}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Array:" in result["output"]
        assert "Mean:" in result["output"]

    @pytest.mark.skipif(
        not _is_module_available("pandas"), reason="pandas not available"
    )
    def test_pandas_available_if_installed(self, python_executor):
        """Test that pandas is available if installed"""
        result = python_executor.run_json_sync(
            {
                "code": """
import pandas as pd
df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
print(df)
print(f"Shape: {df.shape}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Shape: (3, 2)" in result["output"]

    def test_output_capture_disabled(self, python_executor):
        """Test behavior when output capture is disabled"""
        result = python_executor.run_json_sync(
            {
                "code": "print('This should not be captured')\nx = 42",
                "capture_output": False,
            }
        )

        assert result["success"] is True
        # With capture_output=False, variables should still be shown when no print output
        assert (
            "Variables created:" in result["output"]
            or "Code executed successfully (no output)" in result["output"]
        )

    def test_stderr_capture(self, python_executor):
        """Test that stderr is captured"""
        result = python_executor.run_json_sync(
            {
                "code": """
import sys
print("stdout message")
print("stderr message", file=sys.stderr)
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "stdout message" in result["output"]

    def test_multiline_code_execution(self, python_executor):
        """Test execution of complex multiline code"""
        code = """
numbers = []
for i in range(5):
    numbers.append(i * 2)

total = sum(numbers)
print(f"Numbers: {numbers}")
print(f"Total: {total}")
"""
        result = python_executor.run_json_sync({"code": code, "capture_output": True})

        assert result["success"] is True
        assert "Numbers: [0, 2, 4, 6, 8]" in result["output"]
        assert "Total: 20" in result["output"]

    def test_exception_with_traceback(self, python_executor):
        """Test that exceptions include full traceback"""
        result = python_executor.run_json_sync(
            {"code": "result = 1 / 0", "capture_output": True}
        )

        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_private_variables_not_displayed(self, python_executor):
        """Test that private variables (starting with _) are not displayed"""
        result = python_executor.run_json_sync(
            {"code": "_private = 'secret'\npublic = 'visible'", "capture_output": True}
        )

        assert result["success"] is True
        assert "public = 'visible'" in result["output"]
        assert "_private" not in result["output"]

    @pytest.mark.asyncio
    async def test_async_execution_same_as_sync(self, python_executor):
        """Test that async execution produces same results as sync"""
        code = "result = 10 * 5\nprint(result)"

        sync_result = python_executor.run_json_sync(
            {"code": code, "capture_output": True}
        )

        async_result = await python_executor.run_json_async(
            {"code": code, "capture_output": True}
        )

        assert sync_result == async_result

    def test_args_validation(self):
        """Test PythonExecutorArgs validation"""
        # Valid args with defaults
        args = PythonExecutorArgs(code="print('test')")
        assert args.code == "print('test')"
        assert args.capture_output is True  # default

        # Custom args
        args = PythonExecutorArgs(code="x = 1", capture_output=False)
        assert args.code == "x = 1"
        assert args.capture_output is False

    def test_result_model(self):
        """Test PythonExecutorResult model"""
        # Success result
        result = PythonExecutorResult(success=True, output="test output", error="")
        assert result.success is True
        assert result.output == "test output"
        assert result.error == ""

        # Error result
        result = PythonExecutorResult(success=False, output="", error="Some error")
        assert result.success is False
        assert result.output == ""
        assert result.error == "Some error"

    def test_large_output_handling(self, python_executor):
        """Test handling of large output"""
        result = python_executor.run_json_sync(
            {
                "code": """
# Generate large output
for i in range(100):
    print(f"Line {i}: {'x' * 50}")
""",
                "capture_output": True,
            }
        )

        assert result["success"] is True
        assert "Line 0:" in result["output"]
        assert "Line 99:" in result["output"]

    def test_stdout_stderr_restoration(self, python_executor):
        """Test that stdout and stderr are properly restored after execution"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Execute some code
        python_executor.run_json_sync({"code": "print('test')", "capture_output": True})

        # Check that original streams are restored
        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr
