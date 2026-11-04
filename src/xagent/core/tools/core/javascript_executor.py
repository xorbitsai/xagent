"""
JavaScript Code Execution Tool

Executes JavaScript code using Node.js runtime.
Supports npm packages for extended functionality.
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from ..artifacts import GENERATED_ARTIFACT_EXTENSIONS

logger = logging.getLogger(__name__)

GENERATED_FILE_PATTERNS = tuple(
    f"*{extension}" for extension in sorted(GENERATED_ARTIFACT_EXTENSIONS)
)


# Known pptxgenjs API method names whose "methodName: <desc>" console.log
# output indicates a validation failure that did NOT throw. Defined as a
# tuple so additions are obvious in code review; the JS regex is built
# from it so the two stay in sync. The list is intentionally narrow —
# only real pptxgenjs methods that the library is known to emit
# `<method>: <description>` warnings from. Generic-sounding names like
# bare `write` or `stream` are deliberately excluded so unrelated user
# logs (`console.log("write: complete")`) can't trip the detector.
_PPTXGENJS_WARN_METHODS = (
    "addText",
    "addTable",
    "addImage",
    "addShape",
    "addChart",
    "addMedia",
    "addNotes",
    "addSlide",
    "addSection",
    "addBackgroundImage",
    "writeFile",
)
_PPTXGENJS_WARN_JS_REGEX = f"/^(?:{'|'.join(_PPTXGENJS_WARN_METHODS)}): /"


def _requests_pptxgenjs(packages: Optional[list[str]]) -> bool:
    """True iff the caller asked for the pptxgenjs npm package.

    The validation-error interception only runs when this is true —
    otherwise a user script that happens to log a pptxgenjs-shaped
    string (``console.log("addTable: oh no")``) would be misclassified
    as a library warning, returning success=False against the agent's
    expectation.
    """
    if not packages:
        return False
    return any((pkg or "").strip().lower() == "pptxgenjs" for pkg in packages)


def _wrap_user_code(code: str, *, intercept_pptxgenjs: bool) -> str:
    """Build the JS source actually fed to ``node`` for *code*.

    Responsibilities:

    1. Buffer ``console.log`` so it doesn't interleave with our own
       diagnostic output. Logs are flushed back to stdout when the
       script finishes normally.
    2. (Only when ``intercept_pptxgenjs`` is true.) Intercept the
       pptxgenjs "warn only, keep going" failure path — pptxgenjs
       calls ``console.log("addTable: tableRows has a bad row...")``
       and then continues, leaving a malformed ``.pptx`` on disk
       while node still exits 0.

       We record a wrapper-scoped flag the moment a matching message
       is seen, AND throw inside the override so an unguarded pptxgenjs
       call bails before ``writeFile`` runs. After the user-code
       ``try`` block we re-check the flag — that way a user
       ``try/catch`` around the pptxgenjs call can swallow the injected
       throw and still not suppress the failure, because the flag check
       runs outside user code's reach. Either path lands the failure
       on the framework's normal hard-error path (non-zero exit,
       populated stderr) so the agent gets a clear error signal and
       can retry with corrected code.

    The flag-after-try design comes directly from rogercloud's review
    on PR #442: relying on the in-throw mechanism alone is unsafe —
    user code can catch and continue past it.
    """
    intercept_flag = "true" if intercept_pptxgenjs else "false"
    return f"""
const __INTERCEPT_PPTXGENJS = {intercept_flag};
const __PPTXGENJS_WARN_RE = {_PPTXGENJS_WARN_JS_REGEX};
const __logs = [];
let __pptxgenjsValidationFailure = null;
const __originalLog = console.log;
console.log = (...args) => {{
    const __msg = args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' ');
    if (__INTERCEPT_PPTXGENJS && __PPTXGENJS_WARN_RE.test(__msg)) {{
        if (__pptxgenjsValidationFailure === null) {{
            __pptxgenjsValidationFailure = __msg;
        }}
        // Throw so an unguarded pptxgenjs call site bails before
        // writeFile creates a malformed .pptx. If user code happens
        // to wrap the call in try/catch, this throw gets swallowed —
        // but the flag set just above still survives, and the
        // wrapper-level check after the try block exits the process.
        throw new Error(
            'pptxgenjs reported a validation problem and the generated '
            + 'output is likely malformed even though the call did not '
            + 'throw. Fix the offending arguments and retry. Warning: '
            + __msg
        );
    }}
    __logs.push(__msg);
}};

let __userCodeError = null;
try {{
{code}
}} catch (error) {{
    // Don't exit yet: even if user code's own try/catch consumed our
    // injected throw, the wrapper-scoped flag is the source of truth
    // for pptxgenjs failures. Record any unrelated user-code error
    // here and fall through to the post-try checks below.
    __userCodeError = error;
}}

console.log = __originalLog;

if (__pptxgenjsValidationFailure !== null) {{
    // Wrapper-owned failure: runs outside user code's reach so a
    // user-level try/catch cannot suppress it.
    console.error(
        'pptxgenjs reported a validation problem and the generated '
        + 'output is likely malformed even though the call did not '
        + 'throw. Fix the offending arguments and retry. Warning: '
        + __pptxgenjsValidationFailure
    );
    process.exit(1);
}}

if (__userCodeError !== null) {{
    console.error(__userCodeError.message);
    process.exit(1);
}}

__logs.forEach(__line => console.log(__line));
"""


class JavaScriptExecutorCore:
    """JavaScript executor using Node.js"""

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize the JavaScript executor.

        Args:
            working_directory: Directory to use as working directory during execution
        """
        self.working_directory = working_directory
        self.timeout = 30  # seconds

    def execute_code(
        self,
        code: str,
        packages: Optional[list[str]] = None,
        capture_output: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute JavaScript code and return result.

        Args:
            code: JavaScript code to execute
            packages: Optional list of npm packages to install (e.g., ['pptxgenjs', 'axios'])
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with success status, output, and error information
        """
        from pathlib import Path

        try:
            # Determine execution directory
            if self.working_directory:
                # Execute directly in workspace output directory
                exec_dir = Path(self.working_directory)
                exec_dir.mkdir(parents=True, exist_ok=True)

                # Use temp directory only for node_modules
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    return self._execute_with_workspace(
                        code, packages, capture_output, exec_dir, temp_path
                    )
            else:
                # No workspace, use temp directory for everything
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    return self._execute_in_temp(
                        code, packages, capture_output, temp_path
                    )

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Execution timed out"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_in_temp(
        self,
        code: str,
        packages: Optional[list[str]],
        capture_output: bool,
        temp_path: Path,
    ) -> Dict[str, Any]:
        """Execute in temp directory (no workspace)"""
        # Create package.json
        deps = self._get_deps(packages)

        package_json = temp_path / "package.json"
        if deps:
            import json

            package_json.write_text(
                json.dumps({"dependencies": deps}), encoding="utf-8"
            )

        # Create the JS script — wrap user code so pptxgenjs warn-only
        # failures get converted to throws (see _wrap_user_code). The
        # interceptor only fires when the caller actually requested
        # pptxgenjs so unrelated logs (`console.log("write: complete")`)
        # don't trip the detector.
        script_file = temp_path / "script.js"
        script_file.write_text(
            _wrap_user_code(code, intercept_pptxgenjs=_requests_pptxgenjs(packages)),
            encoding="utf-8",
        )

        # Install dependencies if needed
        if deps:
            result = subprocess.run(
                ["npm", "install", "--silent", "--no-audit", "--no-fund"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(f"npm install failed: {result.stderr}")

        # Execute the script
        result = subprocess.run(
            ["node", "script.js"],
            cwd=temp_path,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            return {
                "success": False,
                "error": error_msg,
                "output": result.stdout,
            }

        return {
            "success": True,
            "output": result.stdout or "Code executed successfully (no output)",
            "error": "",
        }

    def _execute_with_workspace(
        self,
        code: str,
        packages: Optional[list[str]],
        capture_output: bool,
        exec_dir: Path,
        temp_path: Path,
    ) -> Dict[str, Any]:
        """Execute in workspace output directory with node_modules in temp"""
        # Create package.json in temp directory
        deps = self._get_deps(packages)

        package_json = temp_path / "package.json"
        if deps:
            import json

            package_json.write_text(
                json.dumps({"dependencies": deps}), encoding="utf-8"
            )

        # Create the JS script in execution directory (so files are created
        # there). The wrapper buffers console.log AND, only when the caller
        # requested pptxgenjs, converts its warn-only validation failures
        # into thrown errors plus a wrapper-scoped flag check that user
        # try/catch can't suppress (see _wrap_user_code). When
        # capture_output is False the caller has opted out of buffering
        # and gets the raw code.
        script_file = exec_dir / "script.js"
        wrapped_code = (
            _wrap_user_code(code, intercept_pptxgenjs=_requests_pptxgenjs(packages))
            if capture_output
            else code
        )
        script_file.write_text(wrapped_code, encoding="utf-8")

        # Install dependencies in temp directory
        if deps:
            result = subprocess.run(
                ["npm", "install", "--silent", "--no-audit", "--no-fund"],
                cwd=temp_path,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                logger.warning(f"npm install failed: {result.stderr}")

        # Execute the script in workspace output directory
        # Set NODE_PATH to include temp directory's node_modules
        env = os.environ.copy()
        node_modules_path = temp_path / "node_modules"
        if node_modules_path.exists():
            env["NODE_PATH"] = str(node_modules_path)

        result = subprocess.run(
            ["node", "script.js"],
            cwd=exec_dir,  # Execute in workspace output directory
            capture_output=True,
            text=True,
            timeout=self.timeout,
            env=env,
        )

        if result.returncode != 0:
            error_msg = result.stderr or "Unknown error"
            return {
                "success": False,
                "error": error_msg,
                "output": result.stdout,
            }

        return {
            "success": True,
            "output": result.stdout or "Code executed successfully (no output)",
            "error": "",
            "generated_files": [],
        }

    def _get_deps(self, packages: Optional[list[str]]) -> Dict[str, str]:
        """Get dependency map for packages"""
        deps = {}
        if packages:
            for pkg in packages:
                if pkg == "pptxgenjs":
                    deps[pkg] = "^4.0.1"
                elif pkg == "axios":
                    deps[pkg] = "^1.6.0"
                elif pkg == "lodash":
                    deps[pkg] = "^4.17.21"
                else:
                    deps[pkg] = "latest"
        return deps


def execute_javascript(
    code: str,
    packages: Optional[list[str]] = None,
    workspace: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Execute JavaScript code with optional npm packages.

    Args:
        code: JavaScript code string to execute
        packages: Optional list of npm packages (e.g., ['pptxgenjs'])
        workspace: Optional workspace for working directory

    Returns:
        Dictionary with success status and output

    Example:
        >>> execute_javascript(\"\"\"
        ... const PptxGenJS = require('pptxgenjs');
        ... const pres = new PptxGenJS();
        ... pres.addText('Hello', { x: 1, y: 1 });
        ... pres.writeFile({ fileName: 'test.pptx' });
        ... console.log('Success');
        ... \"\"\", packages=['pptxgenjs'])
    """
    working_dir = None
    if workspace:
        working_dir = str(workspace.output_dir)

    executor = JavaScriptExecutorCore(working_directory=working_dir)
    return executor.execute_code(code, packages=packages)


def get_javascript_executor_tool(_info: Optional[dict[str, str]] = None) -> Any:
    """
    Get JavaScript executor tool for LangChain integration.

    Args:
        _info: Optional tool info (unused)

    Returns:
        LangChain tool instance
    """
    from langchain_core.tools import tool

    @tool
    def javascript_executor(
        code: str, packages: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute JavaScript code using Node.js runtime.

        Supports npm packages for extended functionality like pptxgenjs for PowerPoint generation.

        Args:
            code: JavaScript code to execute
            packages: Comma-separated list of npm packages (e.g., 'pptxgenjs,axios')

        Returns:
            Dictionary with execution result

        Examples:
            # Generate PowerPoint
            javascript_executor(\"\"\"
            const PptxGenJS = require('pptxgenjs');
            const pres = new PptxGenJS();
            pres.addText('Hello World', { x: 1, y: 1, fontSize: 32 });
            pres.writeFile({ fileName: 'output.pptx' });
            \"\"\", packages='pptxgenjs')

            # Simple calculation
            javascript_executor('console.log(2 + 2);')
        """
        pkg_list = None
        if packages:
            pkg_list = [p.strip() for p in packages.split(",")]

        return execute_javascript(code, packages=pkg_list)

    return javascript_executor
