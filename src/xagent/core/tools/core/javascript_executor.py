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

logger = logging.getLogger(__name__)


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

        # Create the JS script
        script_file = temp_path / "script.js"
        script_file.write_text(code, encoding="utf-8")

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

        # Create the JS script in execution directory (so files are created there)
        script_file = exec_dir / "script.js"

        # Wrap code to capture output
        wrapped_code = code
        if capture_output:
            wrapped_code = f"""
const logs = [];
const originalLog = console.log;
console.log = (...args) => {{
    logs.push(args.map(a => typeof a === 'object' ? JSON.stringify(a) : String(a)).join(' '));
}};

try {{
{code}
}} catch (error) {{
    console.error(error.message);
    process.exit(1);
}}

console.log = originalLog;
logs.forEach(log => console.log(log));
"""

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

        # Find generated files (they're already in the right place)
        generated_files = []
        for ext in ["*.pptx", "*.png", "*.jpg", "*.jpeg", "*.gif", "*.pdf"]:
            for file in exec_dir.glob(ext):
                # Only count files created during this execution (not script.js)
                if file.name != "script.js":
                    generated_files.append(file.name)

        output = result.stdout or "Code executed successfully (no output)"
        if generated_files:
            file_info = f"\n\nGenerated files: {', '.join(generated_files)}"
            output += file_info

        return {
            "success": True,
            "output": output,
            "error": "",
            "generated_files": generated_files,
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
