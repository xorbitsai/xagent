"""Tests for JavaScript executor tool."""

import tempfile
from pathlib import Path

import pytest

from xagent.core.tools.adapters.vibe.javascript_executor import JavaScriptExecutorTool
from xagent.core.tools.artifacts import GENERATED_ARTIFACT_EXTENSIONS
from xagent.core.tools.core.javascript_executor import (
    GENERATED_FILE_PATTERNS,
    JavaScriptExecutorCore,
    execute_javascript,
    get_javascript_executor_tool,
)
from xagent.core.workspace import TaskWorkspace


class TestJavaScriptExecutorCore:
    """Test JavaScript executor core functionality."""

    def test_generated_file_patterns_match_supported_artifact_extensions(self):
        assert GENERATED_FILE_PATTERNS == tuple(
            f"*{extension}" for extension in sorted(GENERATED_ARTIFACT_EXTENSIONS)
        )

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


class TestPptxgenjsWarningInterception:
    """The JS wrapper converts pptxgenjs warn-only failures into hard fails.

    pptxgenjs's ``addTable``/``addImage``/etc. log to stdout with
    ``"<method>: <description>"`` when an argument is malformed, then
    keep going — leaving a broken ``.pptx`` on disk while node still
    exits 0. When the caller requested ``pptxgenjs`` we intercept
    ``console.log`` in the wrapper: set a wrapper-scoped flag, throw,
    and after the user-code ``try`` block re-check the flag and
    ``process.exit(1)`` so a user-level ``try/catch`` around the
    pptxgenjs call cannot suppress the failure.

    The interceptor is GATED on `packages` containing pptxgenjs so a
    script that doesn't request it can't be misclassified by a
    pptxgenjs-shaped log line.
    """

    def test_pptxgenjs_warning_in_user_code_triggers_failure(self):
        """A script that just ``console.log``s the warning text is enough —
        the wrapper turns it into a throw without needing real pptxgenjs."""
        executor = JavaScriptExecutorCore()
        code = (
            "console.log("
            "'addTable: tableRows has a bad row. "
            "A row should be an array of cells.'"
            ");"
        )
        result = executor.execute_code(code, packages=["pptxgenjs"])

        assert result["success"] is False
        assert "pptxgenjs reported a validation problem" in result["error"]
        assert "addTable: tableRows has a bad row" in result["error"]

    def test_other_pptxgenjs_method_prefixes_also_trigger_failure(self):
        """addText/addImage/addShape/writeFile prefixes all match."""
        for prefix in ("addText", "addImage", "addShape", "writeFile"):
            executor = JavaScriptExecutorCore()
            result = executor.execute_code(
                f"console.log('{prefix}: something wrong');",
                packages=["pptxgenjs"],
            )
            assert result["success"] is False, prefix
            assert prefix in result["error"], prefix

    def test_generic_prefixes_no_longer_trigger_failure(self):
        """rogercloud comment 1: bare ``write:``/``stream:`` are off the
        allow-list now — ordinary user logs must not collide."""
        for line in ("write: complete", "stream: ready"):
            executor = JavaScriptExecutorCore()
            result = executor.execute_code(
                f"console.log('{line}');", packages=["pptxgenjs"]
            )
            assert result["success"] is True, line
            assert line in result["output"], line

    def test_unrelated_colon_lines_do_not_trigger_failure(self):
        """User code logging `foo: bar` style strings stays a clean success."""
        for line in (
            "Sum: 42",
            "error: oops in my data",
            "DEBUG: starting",
            "info: 200 OK",
        ):
            executor = JavaScriptExecutorCore()
            result = executor.execute_code(f"console.log('{line}');")
            assert result["success"] is True, line
            assert line in result["output"], line

    def test_method_prefix_requires_colon_space_separator(self):
        """`addTable:nodescription` (no space) is not matched."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code(
            "console.log('addTable:no-space');", packages=["pptxgenjs"]
        )
        assert result["success"] is True
        assert "addTable:no-space" in result["output"]

    def test_clean_run_unchanged(self):
        """Sanity: code with no pptxgenjs-shaped logs still succeeds."""
        executor = JavaScriptExecutorCore()
        result = executor.execute_code("console.log('all good');")
        assert result["success"] is True
        assert "all good" in result["output"]
        assert result["error"] == ""

    def test_interception_gated_on_pptxgenjs_request(self):
        """rogercloud comment 1: when the caller didn't ask for pptxgenjs,
        even an exact ``addTable: ...`` line must NOT be flagged. The
        detector belongs only to executions that actually load the library.
        """
        executor = JavaScriptExecutorCore()
        result = executor.execute_code(
            "console.log('addTable: tableRows has a bad row.');"
        )
        assert result["success"] is True
        assert "addTable: tableRows has a bad row." in result["output"]

    def test_user_try_catch_cannot_suppress_failure(self):
        """rogercloud comment 2: a user-level try/catch around the
        pptxgenjs call must NOT be able to swallow the validation
        failure. The wrapper-scoped flag check runs after the user try
        block and exits the process from outside user code's reach.
        """
        executor = JavaScriptExecutorCore()
        # Simulate pptxgenjs's behaviour: the offending method only
        # ``console.log``s and returns normally. Wrap the call in a
        # broad try/catch so the injected throw is consumed.
        code = (
            "try {"
            "  console.log('addTable: tableRows has a bad row.');"
            "  console.log('about to write the malformed pptx');"
            "} catch (e) {"
            "  console.log('user swallowed the throw and kept going');"
            "}"
        )
        result = executor.execute_code(code, packages=["pptxgenjs"])

        # The wrapper-level flag check still fires after the user try
        # block returns, so the tool reports failure even though user
        # code didn't propagate the throw.
        assert result["success"] is False
        assert "pptxgenjs reported a validation problem" in result["error"]
        assert "addTable: tableRows has a bad row" in result["error"]


# Parametrized end-to-end coverage of every failure mode the executor is
# expected to surface, plus a positive control and a false-positive guard.
# Each case is a real subprocess run; the pptxgenjs cases install the npm
# package on first run and are therefore slow (~10s each on a cold cache).
_ERROR_SIGNAL_CASES = [
    pytest.param(
        'console.log("unterminated;',
        None,
        False,
        None,  # node's wrapped error.message varies by version; presence of
        # *any* error is enough
        id="syntax_error",
    ),
    pytest.param(
        'throw new Error("boom");',
        None,
        False,
        "boom",
        id="explicit_throw",
    ),
    pytest.param(
        "console.log(somethingUndefined);",
        None,
        False,
        "not defined",
        id="undefined_reference",
    ),
    pytest.param(
        # Reproduces demo task/1100: pptxgenjs prints "addTable: tableRows
        # has a bad row..." to stdout and continues. Node exits 0, but the
        # generated .pptx is malformed. The executor must convert this
        # warning into success=False so the agent retries.
        'const P=require("pptxgenjs");'
        "const p=new P();"
        'p.addSlide().addTable([["ok"],{text:"C"}],{x:1,y:1,w:5});'
        'p.writeFile({fileName:"t.pptx"});'
        'console.log("done");',
        ["pptxgenjs"],
        False,
        "tableRows has a bad",
        id="pptxgenjs_warn_only_demo_bug",
    ),
    pytest.param(
        # Hard-error path: pptxgenjs rejects a flat row array immediately.
        'const P=require("pptxgenjs");'
        "const p=new P();"
        'p.addSlide().addTable(["A","B"],{x:1,y:1,w:5});'
        'p.writeFile({fileName:"t.pptx"});',
        ["pptxgenjs"],
        False,
        "'rows' should be",
        id="pptxgenjs_hard_fail_flat_rows",
    ),
    pytest.param(
        # Positive control: a normal pptxgenjs run produces a .pptx and
        # the executor reports success=True with no error.
        'const P=require("pptxgenjs");'
        "const p=new P();"
        'p.addSlide().addText("Hi",{x:1,y:1,fontSize:32});'
        'p.writeFile({fileName:"good.pptx"});'
        'console.log("all good");',
        ["pptxgenjs"],
        True,
        None,
        id="pptxgenjs_clean_run",
    ),
    pytest.param(
        # False-positive guard: user code that happens to log strings
        # containing colons must NOT be misclassified as a library warning.
        'console.log("Sum: 42");'
        'console.log("error: oops in my data");'
        'console.log("DEBUG: starting");',
        None,
        True,
        None,
        id="user_log_with_colon_no_false_positive",
    ),
]


@pytest.mark.parametrize(
    "code,packages,expected_success,error_marker", _ERROR_SIGNAL_CASES
)
def test_error_signal_reaches_agent(code, packages, expected_success, error_marker):
    """Every kind of JS failure must surface to the agent as success=False
    with a useful error message; the executor must not silently mask a
    library warning that left malformed output behind, and must not flag
    a clean run as failure.

    Mirrors the manual verification matrix written up in the PR
    description for fix(js_executor): surface pptxgenjs validation
    warnings as soft failures.
    """
    executor = JavaScriptExecutorCore()
    result = executor.execute_code(code, packages=packages)

    assert result["success"] is expected_success, (
        f"expected success={expected_success}, got {result['success']!r}; "
        f"error={result.get('error')!r}, output={result.get('output')!r}"
    )
    if error_marker is not None:
        err = (result.get("error") or "").lower()
        assert error_marker.lower() in err, (
            f"error marker {error_marker!r} not in error={result.get('error')!r}"
        )


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
            assert result.get("generated_files", []) == []
            assert "Generated files:" not in result["output"]


class TestJavaScriptExecutorToolArtifacts:
    """Test workspace-bound JavaScript executor generated file artifacts."""

    def test_generated_pptx_file_returns_inline_artifact(self, tmp_path):
        workspace = TaskWorkspace("test_js_pptx", str(tmp_path))
        executor = JavaScriptExecutorTool(workspace=workspace)

        result = executor.run_json_sync(
            {
                "code": "const fs = require('fs'); fs.writeFileSync('slides.pptx', 'pptx');",
            }
        )

        assert result["success"] is True
        assert result["generated_files"] == ["slides.pptx"]
        assert result["file_refs"][0]["filename"] == "slides.pptx"
        assert result["file_refs"][0]["file_id"]
        assert result["artifacts"] == [
            {
                "type": "presentation",
                "file_id": result["file_refs"][0]["file_id"],
                "filename": "slides.pptx",
                "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "display": "inline",
                "validation": result["file_refs"][0]["validation"],
            }
        ]

    def test_stale_generated_files_are_not_returned_as_inline_artifacts(self, tmp_path):
        workspace = TaskWorkspace("test_js_stale_artifacts", str(tmp_path))
        executor = JavaScriptExecutorTool(workspace=workspace)

        first_result = executor.run_json_sync(
            {
                "code": "const fs = require('fs'); fs.writeFileSync('old.pdf', 'old');",
            }
        )
        second_result = executor.run_json_sync(
            {
                "code": "const fs = require('fs'); fs.writeFileSync('new.pdf', 'new');",
            }
        )

        assert first_result["success"] is True
        assert second_result["success"] is True
        assert second_result["generated_files"] == ["new.pdf"]
        assert [file_ref["filename"] for file_ref in second_result["file_refs"]] == [
            "new.pdf"
        ]
        assert [artifact["filename"] for artifact in second_result["artifacts"]] == [
            "new.pdf"
        ]
        assert "new.pdf" in second_result["output"]
        assert "old.pdf" not in second_result["output"]
