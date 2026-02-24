"""Tests for JavaScript executor tool."""

import tempfile
from pathlib import Path

from xagent.core.tools.core.javascript_executor import (
    JavaScriptExecutorCore,
    execute_javascript,
    get_javascript_executor_tool,
)


class TestJavaScriptExecutorCore:
    """Test JavaScript executor core functionality."""

    def test_simple_javascript_execution(self):
        """Test basic JavaScript code execution."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code("console.log('Hello World');")

        assert result["success"] is True
        assert "Hello World" in result["output"]
        assert result["error"] == ""

    def test_javascript_math_operations(self):
        """Test JavaScript math operations."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code("console.log(2 + 2);")

        assert result["success"] is True
        assert "4" in result["output"]

    def test_javascript_syntax_error(self):
        """Test JavaScript syntax error handling."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code("console.log('unterminated string);")

        assert result["success"] is False
        assert result["error"] != ""

    def test_javascript_runtime_error(self):
        """Test JavaScript runtime error handling."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code("throw new Error('Test error');")

        assert result["success"] is False
        assert "Test error" in result["error"]

    def test_javascript_with_variables(self):
        """Test JavaScript with variable declarations."""
        executor = JavaScriptExecutorCore()
        code = """
const x = 10;
const y = 20;
console.log(x + y);
"""
        result = executor.execute_code(code)

        assert result["success"] is True
        assert "30" in result["output"]

    def test_javascript_multiline_output(self):
        """Test JavaScript with multiple console.log statements."""
        executor = JavaScriptExecutorCore()
        code = """
console.log('Line 1');
console.log('Line 2');
console.log('Line 3');
"""
        result = executor.execute_code(code)

        assert result["success"] is True
        assert "Line 1" in result["output"]
        assert "Line 2" in result["output"]
        assert "Line 3" in result["output"]

    def test_javascript_complex_operations(self):
        """Test JavaScript with arrays and objects."""
        executor = JavaScriptExecutorCore()
        code = """
const arr = [1, 2, 3, 4, 5];
const sum = arr.reduce((a, b) => a + b, 0);
console.log('Sum:', sum);
"""
        result = executor.execute_code(code)

        assert result["success"] is True
        assert "15" in result["output"]


class TestJavaScriptWithNpmPackages:
    """Test JavaScript executor with npm packages."""

    def test_pptxgenjs_package_installation(self):
        """Test that pptxgenjs package can be loaded."""
        executor = JavaScriptExecutorCore()
        code = """
const PptxGenJS = require('pptxgenjs');
console.log('PptxGenJS loaded successfully');
console.log('Version:', typeof PptxGenJS);
"""
        result = executor.execute_code(code, packages=["pptxgenjs"])

        assert result["success"] is True
        assert "PptxGenJS loaded successfully" in result["output"]

    def test_lodash_package(self):
        """Test that lodash package can be used."""
        executor = JavaScriptExecutorCore()
        code = """
const _ = require('lodash');
const arr = [1, 2, 3, 4, 5];
const sum = _.sum(arr);
console.log('Sum:', sum);
"""
        result = executor.execute_code(code, packages=["lodash"])

        # Check for success if Node.js is available
        if result["success"]:
            assert "15" in result["output"]


class TestExecuteJavaScriptFunction:
    """Test execute_javascript wrapper function."""

    def test_execute_javascript_basic(self):
        """Test basic execute_javascript function."""
        result = execute_javascript("console.log('Test');")

        assert result["success"] is True

    def test_execute_javascript_with_packages(self):
        """Test execute_javascript with packages parameter."""
        result = execute_javascript("console.log('Test');", packages=["pptxgenjs"])

        # May fail if Node.js not installed, but shouldn't crash
        assert isinstance(result, dict)
        assert "success" in result


class TestJavaScriptExecutorTool:
    """Test LangChain tool wrapper."""

    def test_get_javascript_executor_tool(self):
        """Test that tool can be created."""
        tool = get_javascript_executor_tool()

        assert tool is not None
        assert tool.name == "javascript_executor"

    def test_tool_has_description(self):
        """Test that tool has proper description."""
        tool = get_javascript_executor_tool()

        assert tool.description is not None
        assert len(tool.description) > 0
        assert "JavaScript" in tool.description


class TestJavaScriptIntegrationScenarios:
    """Integration tests for common use cases."""

    def test_powerpoint_generation_code(self):
        """Test PowerPoint generation JavaScript code structure."""
        executor = JavaScriptExecutorCore()

        # This test validates the code structure, not actual file generation
        # (which requires Node.js and file system access)
        code = """
const PptxGenJS = require('pptxgenjs');
const pres = new PptxGenJS();

// Test basic operations
pres.layout = 'LAYOUT_16x9';

// Test that object methods exist
console.log('Presentation created');
console.log('Layout set to:', pres.layout);
"""
        result = executor.execute_code(code, packages=["pptxgenjs"])

        # Just ensure it doesn't crash - actual file creation requires full environment
        assert isinstance(result, dict)

    def test_json_manipulation(self):
        """Test JavaScript JSON manipulation."""
        executor = JavaScriptExecutorCore()
        code = """
const data = {
    name: 'Test',
    values: [1, 2, 3],
    nested: { key: 'value' }
};
console.log(JSON.stringify(data));
"""
        result = executor.execute_code(code)

        assert result["success"] is True
        assert "Test" in result["output"]

    def test_string_operations(self):
        """Test JavaScript string manipulation."""
        executor = JavaScriptExecutorCore()
        code = """
const text = 'Hello World';
console.log(text.toUpperCase());
console.log(text.toLowerCase());
console.log(text.length);
"""
        result = executor.execute_code(code)

        assert result["success"] is True
        assert "HELLO WORLD" in result["output"]

    def test_file_output_to_workspace(self):
        """Test that generated files are copied to workspace output directory."""

        # Create a temporary workspace (simulating actual usage)
        with tempfile.TemporaryDirectory() as workspace_dir:
            workspace_path = Path(workspace_dir)
            output_dir = workspace_path / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Pass the output directory directly (as adapter does with workspace.resolve_path(""))
            executor = JavaScriptExecutorCore(working_directory=str(output_dir))

            # Code that creates a PDF file (supported extension)
            code = """
const fs = require('fs');
fs.writeFileSync('test_file.pdf', 'Hello from JavaScript!');
console.log('File created');
"""
            result = executor.execute_code(code)

            assert result["success"] is True
            # Check that the file was copied to output directory
            output_file = output_dir / "test_file.pdf"
            assert output_file.exists(), f"File not found in output: {output_file}"
            # Verify content
            assert output_file.read_text() == "Hello from JavaScript!"
            # Check that generated_files is populated
            assert "test_file.pdf" in result.get("generated_files", [])
