"""Stable, machine-readable error codes for workforce-run failures.

The workforce run service is shared across channels (internal web, /v1
SDK, widget, ...). Each channel needs to map a failure to its own error
envelope, but the failures were historically raised as bare
``HTTPException``s carrying only free-text ``detail`` -- forcing callers
to substring-match prose to tell cases apart, which silently breaks when
a message is reworded.

:class:`WorkforceRunError` attaches a stable ``code`` to the exception so
callers switch on that instead. It remains an ``HTTPException`` subclass,
so channels that don't care about the code (the JWT web endpoints, which
surface it through FastAPI's default handler) keep working unchanged.
"""

from fastapi import HTTPException


class WorkforceRunErrorCode:
    """Stable discriminants carried by :class:`WorkforceRunError`.

    These are service-layer codes, deliberately decoupled from any one
    channel's wire codes (e.g. the /v1 ``V1ErrorCode`` enum) -- each
    channel maps these to its own envelope.
    """

    ARCHIVED = "workforce_archived"
    NOT_ACTIVE = "workforce_not_active"
    IDEMPOTENCY_CONFLICT = "idempotency_conflict"
    INVALID_EXECUTION_MODE = "invalid_execution_mode"
    INVALID_IDEMPOTENCY_KEY = "invalid_idempotency_key"
    FILE_NOT_FOUND = "file_not_found"


class WorkforceRunError(HTTPException):
    """``HTTPException`` carrying a stable machine ``code``.

    Downstream channel adapters switch on ``code`` rather than matching
    the human-readable ``detail`` string.
    """

    def __init__(self, status_code: int, detail: str, *, code: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.code = code
