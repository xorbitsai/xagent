"""
Pure Python Code Execution Tool
Standalone Python execution functionality without framework dependencies
"""

import ast
import io
import logging
import os
import sys
import traceback
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Optional imports
plt: Any = None
matplotlib_module: Any = None

try:
    import matplotlib as matplotlib_module
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError:
    pass


class PythonExecutorCore:
    """Pure Python executor without framework dependencies"""

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize the Python executor.

        Args:
            working_directory: Directory to use as working directory during execution
        """
        self.working_directory = working_directory

    def execute_code(self, code: str, capture_output: bool = True) -> Dict[str, Any]:
        """
        Execute Python code and return result.

        Args:
            code: Python code to execute
            capture_output: Whether to capture stdout/stderr

        Returns:
            Dictionary with success status, output, and error information
        """
        try:
            # Validate syntax first
            ast.parse(code)

            # Prepare execution environment
            output_buffer = io.StringIO()
            error_buffer = io.StringIO()

            # Create a safe globals environment
            safe_globals = self._create_safe_globals()
            local_vars: Dict[str, Any] = {}

            old_cwd = None
            if self.working_directory:
                old_cwd = os.getcwd()
                logger.info(
                    f"PythonExecutor: Changing working directory from {old_cwd} to {self.working_directory}"
                )
                os.chdir(self.working_directory)

            if capture_output:
                old_stdout = sys.stdout
                old_stderr = sys.stderr
                sys.stdout = output_buffer
                sys.stderr = error_buffer

            try:
                # Execute the code
                exec(code, safe_globals, local_vars)

                output = ""
                if capture_output:
                    output = output_buffer.getvalue()

                # If no output but there are variables, show them
                if not output and local_vars:
                    visible_vars = {
                        k: v for k, v in local_vars.items() if not k.startswith("_")
                    }
                    if visible_vars:
                        output = "Variables created:\n"
                        for name, value in visible_vars.items():
                            output += f"{name} = {repr(value)}\n"

                return {
                    "success": True,
                    "output": output or "Code executed successfully (no output)",
                    "error": "",
                }

            except Exception:
                error_msg = traceback.format_exc()
                stderr_content = error_buffer.getvalue() if capture_output else ""

                return {
                    "success": False,
                    "output": output_buffer.getvalue() if capture_output else "",
                    "error": f"{error_msg}\n{stderr_content}".strip(),
                }

            finally:
                if old_cwd is not None:
                    logger.info(
                        f"PythonExecutor: Restoring working directory to {old_cwd}"
                    )
                    os.chdir(old_cwd)

                if capture_output:
                    sys.stdout = old_stdout
                    sys.stderr = old_stderr

        except SyntaxError as e:
            return {"success": False, "output": "", "error": f"Syntax Error: {str(e)}"}
        except Exception as e:
            return {"success": False, "output": "", "error": f"Error: {str(e)}"}

    def _create_safe_globals(self) -> Dict[str, Any]:
        """Create a safe globals environment with common imports"""
        safe_globals = {
            "__builtins__": __builtins__,
            "print": print,
            "len": len,
            "range": range,
            "str": str,
            "int": int,
            "float": float,
            "list": list,
            "dict": dict,
            "set": set,
            "tuple": tuple,
            "abs": abs,
            "max": max,
            "min": min,
            "sum": sum,
            "sorted": sorted,
            "reversed": reversed,
            "enumerate": enumerate,
            "zip": zip,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
        }

        # Add common modules
        try:
            import datetime
            import json
            import math
            import re

            # Set matplotlib backend to non-interactive
            os.environ["MPLBACKEND"] = "Agg"

            safe_globals.update(
                {
                    "math": math,
                    "json": json,
                    "datetime": datetime,
                    "re": re,
                    "os": os,
                }
            )
        except ImportError:
            pass

        # Add numpy if available
        try:
            import numpy as np

            safe_globals["np"] = np
            safe_globals["numpy"] = np
        except ImportError:
            pass

        # Add pandas if available
        if pd is not None:
            safe_globals["pd"] = pd
            safe_globals["pandas"] = pd

        # Add matplotlib with safe configuration
        if matplotlib_module is not None and plt is not None:
            matplotlib_module.use("Agg")
            safe_globals["matplotlib"] = matplotlib_module
            safe_globals["plt"] = plt

        return safe_globals


# Convenience function for direct usage
def execute_python_code(
    code: str, capture_output: bool = True, working_directory: Optional[str] = None
) -> Dict[str, Any]:
    """
    Execute Python code and return result.

    Args:
        code: Python code to execute
        capture_output: Whether to capture stdout/stderr
        working_directory: Directory to use as working directory

    Returns:
        Dictionary with execution results
    """
    executor = PythonExecutorCore(working_directory)
    return executor.execute_code(code, capture_output)
