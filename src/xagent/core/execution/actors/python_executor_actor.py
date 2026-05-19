"""
Python executor actor.

Executes Python code in isolated process using xoscar actor framework.
"""

import contextlib
import io
import os
import traceback
from typing import Optional

from .base_executor_actor import BaseExecutorActor


class PythonExecutorActor(BaseExecutorActor):
    """Python executor actor.

    Executes Python code in an isolated process with configurable timeout.
    """

    async def execute(
        self,
        code: str,
        workspace: Optional[str] = None,
        timeout: int = 300,
    ) -> dict:
        """Execute Python code.

        Args:
            code: Python code to execute
            workspace: Working directory path (optional)
            timeout: Execution timeout in seconds (default: 300)

        Returns:
            Execution result dictionary with keys:
                - success: bool
                - output: str
                - error: str
                - return_code: int
                - metadata: dict
                - execution_time: float
        """

        def _execute() -> dict:
            """Internal execution function."""
            # Change to workspace if specified
            old_cwd = None
            if workspace:
                old_cwd = os.getcwd()
                try:
                    os.chdir(workspace)
                except FileNotFoundError:
                    return {
                        "output": "",
                        "error": f"Workspace directory not found: {workspace}",
                        "return_code": 1,
                    }

            try:
                # Capture stdout and stderr
                stdout_capture = io.StringIO()
                stderr_capture = io.StringIO()

                with (
                    contextlib.redirect_stdout(stdout_capture),
                    contextlib.redirect_stderr(stderr_capture),
                ):
                    # Execute code
                    exec_globals = {
                        "__name__": "__main__",
                        "__builtins__": __builtins__,
                    }
                    exec(code, exec_globals)

                # Get output
                stdout_value = stdout_capture.getvalue()
                stderr_value = stderr_capture.getvalue()

                return {
                    "output": stdout_value,
                    "error": stderr_value,
                    "return_code": 0,
                    "metadata": {},
                }

            except Exception as e:
                error_message = f"{type(e).__name__}: {str(e)}"
                error_traceback = traceback.format_exc()
                return {
                    "output": "",
                    "error": f"{error_message}\n{error_traceback}",
                    "return_code": 1,
                    "metadata": {"exception_type": type(e).__name__},
                }

            finally:
                # Restore working directory
                if old_cwd is not None:
                    try:
                        os.chdir(old_cwd)
                    except Exception:
                        pass

        return self._execute_with_tracking(_execute)
