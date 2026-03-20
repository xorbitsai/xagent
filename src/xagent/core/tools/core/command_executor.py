"""
Command Line Executor Tool

Execute shell commands and scripts with proper controls.
"""

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Constants
# Maximum output size to prevent memory exhaustion (10 MB)
MAX_OUTPUT_SIZE = 10 * 1024 * 1024

# Timeout return code constant
TIMEOUT_EXIT_CODE = -999


def _validate_timeout(timeout: Optional[int], default_timeout: int) -> int:
    """
    Validate and normalize timeout value.

    Args:
        timeout: Timeout in seconds
        default_timeout: Default timeout to use if timeout is None

    Returns:
        Validated timeout value

    Raises:
        ValueError: If timeout is invalid
    """
    if timeout is not None:
        if timeout <= 0:
            raise ValueError(f"timeout must be positive, got: {timeout}")
        return timeout
    return default_timeout


def _sanitize_command_for_logging(command: Any, max_length: int = 200) -> str:
    """
    Sanitize command for logging to avoid exposing sensitive data.

    Args:
        command: The command to sanitize (str or list)
        max_length: Maximum length of command to log

    Returns:
        Sanitized command string
    """
    # Convert list command to string for logging
    if isinstance(command, list):
        command_str = " ".join(str(x) for x in command)
    else:
        command_str = str(command)

    # Truncate long commands
    if len(command_str) > max_length:
        return command_str[:max_length] + "... [TRUNCATED]"

    # Redact potential sensitive patterns
    sensitive_patterns = [
        (
            r"(Bearer|Authorization|Token|API[_-]?KEY|PASSWORD|PASSWD|SECRET)[=\s][^\s]+",
            "REDACTED",
        ),
        (r"--password[=\s][^\s]+", "--password=REDACTED"),
        (r"-p\s+[^\s]+", "-p REDACTED"),
    ]

    for pattern, replacement in sensitive_patterns:
        command_str = re.sub(pattern, replacement, command_str, flags=re.IGNORECASE)

    return command_str


def _validate_working_directory(working_directory: Optional[str]) -> None:
    """
    Validate working directory before use.

    Args:
        working_directory: Directory path to validate

    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
        PermissionError: If directory is not accessible
    """
    if not working_directory:
        return

    work_dir = Path(working_directory)

    if not work_dir.exists():
        raise FileNotFoundError(
            f"Working directory does not exist: {working_directory}"
        )

    if not work_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {working_directory}")

    if not os.access(working_directory, os.X_OK):
        raise PermissionError(
            f"No execute permission for directory: {working_directory}"
        )


def _sanitize_interpreter_suffix(interpreter: str) -> str:
    """
    Sanitize interpreter name for use as temp file suffix.

    Args:
        interpreter: Interpreter name (e.g., 'bash', 'python3.11')

    Returns:
        Sanitized interpreter name suitable for file suffix
    """
    # Take first part (before any space) and remove dots
    safe_name = interpreter.split()[0].replace(".", "").replace("-", "_")
    return safe_name if safe_name else "tmp"


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

        Raises:
            ValueError: If timeout is invalid
            FileNotFoundError: If working directory doesn't exist
            NotADirectoryError: If working directory path is not a directory
            PermissionError: If working directory is not accessible
        """
        timeout = _validate_timeout(timeout, self.timeout)
        _validate_working_directory(self.working_directory)

        # Sanitize command for logging
        safe_command = _sanitize_command_for_logging(command)
        logger.info(f"CommandExecutor: Executing: {safe_command}")

        if self.working_directory:
            logger.info(
                f"CommandExecutor: Using working directory: {self.working_directory}"
            )

        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                timeout=timeout,
                cwd=self.working_directory,  # Use cwd parameter instead of os.chdir()
            )

            output = result.stdout if capture_output else ""
            error = result.stderr if capture_output else ""

            # Truncate output if it exceeds maximum size
            if capture_output:
                if len(output) > MAX_OUTPUT_SIZE:
                    output = output[:MAX_OUTPUT_SIZE] + "\n[OUTPUT TRUNCATED]"
                if len(error) > MAX_OUTPUT_SIZE:
                    error = error[:MAX_OUTPUT_SIZE] + "\n[ERROR TRUNCATED]"

            return {
                "success": result.returncode == 0,
                "output": output,
                "error": error,
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
                "return_code": TIMEOUT_EXIT_CODE,
            }
        except Exception as e:
            logger.error(f"CommandExecutor: Execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Execution error: {str(e)}",
                "return_code": TIMEOUT_EXIT_CODE,
            }

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

        Raises:
            ValueError: If timeout is invalid
        """
        timeout = _validate_timeout(timeout, self.timeout)

        try:
            logger.info(
                f"CommandExecutor: Executing script with interpreter: {interpreter}"
            )

            # Sanitize interpreter for temp file suffix
            safe_suffix = _sanitize_interpreter_suffix(interpreter)

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=f".{safe_suffix}", delete=False
            ) as f:
                f.write(script_content)
                script_path = f.name

            try:
                os.chmod(script_path, 0o755)
                command = f"{interpreter} {script_path}"
                return self.execute_command(command, timeout=timeout)
            finally:
                try:
                    os.unlink(script_path)
                except OSError:
                    pass  # Temp file cleanup failed, but command already ran

        except Exception as e:
            logger.error(f"CommandExecutor: Script execution error: {str(e)}")
            return {
                "success": False,
                "output": "",
                "error": f"Script execution error: {str(e)}",
                "return_code": TIMEOUT_EXIT_CODE,
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


def get_command_executor_tool(_info: Optional[dict[str, Any]] = None) -> Any:
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
            # Use resolve_path method for consistency with adapter
            if hasattr(workspace, "resolve_path"):
                working_dir = str(workspace.resolve_path(""))
            elif hasattr(workspace, "path"):
                working_dir = workspace.path

        executor = CommandExecutorCore(working_dir)
        return executor.execute_command(command, timeout=timeout)

    return command_executor
