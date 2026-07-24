"""Stable error codes and the single SSH domain exception."""

from __future__ import annotations

from enum import Enum
from typing import Any


class SshErrorCode(str, Enum):
    """Stable, public error codes. Values are a contract; do not rename."""

    TARGET_NOT_FOUND = "ssh_target_not_found"
    TARGET_DISABLED = "ssh_target_disabled"
    OPERATION_NOT_ALLOWED = "ssh_operation_not_allowed"
    HOST_KEY_MISMATCH = "ssh_host_key_mismatch"
    EGRESS_DENIED = "ssh_egress_denied"
    APPROVAL_REQUIRED = "ssh_approval_required"
    CONNECTION_FAILED = "ssh_connection_failed"
    COMMAND_TIMEOUT = "ssh_command_timeout"
    SECRET_UNAVAILABLE = "ssh_secret_unavailable"
    SANDBOX_UNAVAILABLE = "ssh_sandbox_unavailable"
    CONCURRENT_UPDATE = "ssh_concurrent_update"


class SshError(Exception):
    """SSH domain error carrying a stable code.

    The message and context are user/log facing and MUST NOT contain private
    keys, passphrases, ciphertext, decrypted material, or sandbox secret paths.
    """

    def __init__(
        self,
        code: SshErrorCode,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.context: dict[str, Any] = context if context is not None else {}
        self.cause = cause
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code.value,
            "message": str(self),
            "context": self.context,
        }
