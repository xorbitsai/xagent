"""
Command Line Executor Tool

Execute shell commands and scripts with proper controls.
"""

import logging
import os
import subprocess
import tempfile
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CommandExecutorCore:
    """Shell command executor with execution controls"""

    def __init__(self, working_directory: Optional[str] = None):
        """
        Initialize the command executor.

        Args:
            working_directory: Directory to use as working directory during execution
        """
        self.working_directory = working_directory
        self.timeout = 300  # 5 minutes default

    def execute_command(
        self,
        command: str,
        timeout: Optional[int] = None,
        capture_output: bool = True,
        shell: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute shell command and return result.

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds (default: 300)
            capture_output: Whether to capture stdout/stderr
            shell: Whether to use shell (allows pipes, redirects, etc.)

        Returns:
            Dictionary with success status, output, and error information
        """
        timeout = timeout or self.timeout

        old_cwd = None
        if self.working_directory:
            old_cwd = os.getcwd()
            logger.info(
                f"CommandExecutor: Changing working directory from {old_cwd} to {self.working_directory}"
            )
            os.chdir(self.working_directory)

        try:
            logger.info(f"CommandExecutor: Executing command: {command}")

            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
            )

            return {
                "success": result.returncode == 0,
                "output": result.stdout if capture_output else "",
                "error": result.stderr if capture_output else "",
                "return_code": result.returncode,
            }

        except subprocess.TimeoutExpired:
            logger.warning(
                f"CommandExecutor: Command timed out after {timeout} seconds"
            )
            return {
                "success": False,
                "output": "",
                "error": f"Command timed out after {timeout} seconds",
                "return_code": -1,
            }
        except Exception as e:
            logger.error(f"CommandExecutor: Execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "return_code": -1,
            }
        finally:
            if old_cwd is not None:
                logger.info(
                    f"CommandExecutor: Restoring working directory to {old_cwd}"
                )
                os.chdir(old_cwd)

    def execute_script(
        self,
        script_content: str,
        interpreter: str = "bash",
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute script content with specified interpreter.

        Args:
            script_content: Script content to execute
            interpreter: Interpreter to use (bash, python, node, sh, etc.)
            timeout: Execution timeout in seconds

        Returns:
            Dictionary with execution result
        """
        timeout = timeout or self.timeout

        try:
            logger.info(
                f"CommandExecutor: Executing script with interpreter: {interpreter}"
            )

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{interpreter}", delete=False
            ) as f:
                f.write(script_content)
                script_path = f.name

            try:
                os.chmod(script_path, 0o755)
                command = f"{interpreter} {script_path}"
                return self.execute_command(command, timeout=timeout)
            finally:
                os.unlink(script_path)

        except Exception as e:
            logger.error(f"CommandExecutor: Script execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Script execution error: {str(e)}",
                "return_code": -1,
            }


# Convenience functions for direct usage
def execute_command(
    command: str,
    working_directory: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a shell command.

    Args:
        command: Shell command to execute
        working_directory: Directory to use as working directory
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with execution result
    """
    executor = CommandExecutorCore(working_directory)
    return executor.execute_command(command, timeout=timeout)


def execute_script(
    script_content: str,
    interpreter: str = "bash",
    working_directory: Optional[str] = None,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute script content.

    Args:
        script_content: Script content to execute
        interpreter: Interpreter to use (bash, python, node, etc.)
        working_directory: Directory to use as working directory
        timeout: Execution timeout in seconds

    Returns:
        Dictionary with execution result
    """
    executor = CommandExecutorCore(working_directory)
    return executor.execute_script(script_content, interpreter, timeout)


def get_command_executor_tool(_info: Optional[dict[str, str]] = None) -> Any:
    """
    Get command executor tool for LangChain integration.

    Args:
        _info: Optional tool info (may contain 'workspace' key with workspace object)

    Returns:
        LangChain tool instance
    """
    from langchain_core.tools import tool

    @tool
    def command_executor(command: str, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute shell commands and scripts.

        Supports any shell command including:
        - System commands (ls, cat, grep, etc.)
        - Script execution (./script.sh, python script.py, etc.)
        - Pipes and redirects (cat file.txt | grep pattern)
        - Complex commands with multiple operations

        Args:
            command: Shell command to execute
            timeout: Execution timeout in seconds (default: 300)

        Returns:
            Dictionary with execution result including:
            - success: Boolean indicating if command succeeded
            - output: Standard output from the command
            - error: Standard error from the command (if any)
            - return_code: Process exit code

        Examples:
            # List files in current directory
            command_executor("ls -la")

            # Search for a pattern in files
            command_executor("grep -r 'pattern' /path/to/dir")

            # Run a shell script
            command_executor("./deploy.sh")

            # Use pipes to chain commands
            command_executor("cat data.csv | grep error | wc -l")

            # Install npm packages
            command_executor("npm install")

            # Run Python script
            command_executor("python script.py --arg value")
        """
        # Get working directory from info if provided
        working_dir = None
        if _info and "workspace" in _info:
            workspace = _info["workspace"]
            if hasattr(workspace, "path"):
                working_dir = workspace.path

        executor = CommandExecutorCore(working_dir)
        return executor.execute_command(command, timeout=timeout)

    return command_executor
