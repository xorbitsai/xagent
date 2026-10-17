"""Application JWT primitives backed by PyJWT, never python-jose.

Keep a small import boundary so authentication owners share the same exception
family. Rejection classification only inspects independently verified claims.
"""

from typing import Any

import jwt as _jwt
from jwt import ExpiredSignatureError as ExpiredSignatureError
from jwt import PyJWTError
from jwt import encode as encode
from jwt.types import Options

JWTError = PyJWTError


class InvalidTemporalClaims(PyJWTError):
    """Verified claims contain a NumericDate that cannot be converted."""


def decode(
    token: str | bytes,
    key: Any,
    algorithms: list[str],
    options: Options | None = None,
) -> dict[str, Any]:
    """Retain PyJWT validation and expose stable malformed-date classification.

    Do not inspect undocumented exception contexts or trust unverified claims.
    Only decode/issued-at errors take the slower path: independently verify the same
    signature and non-temporal claims before inspecting NumericDate conversion.
    Expired/not-yet-valid tokens retain PyJWT's original rejection and ordering.
    """
    try:
        return _jwt.decode(token, key, algorithms=algorithms, options=options)
    except _jwt.InvalidSignatureError:
        raise
    except (_jwt.DecodeError, _jwt.InvalidIssuedAtError) as error:
        enabled: Options = options.copy() if options is not None else {}
        if not enabled.get("verify_signature", True):
            raise
        temporal_disabled: Options = enabled.copy()
        temporal_disabled["verify_exp"] = False
        temporal_disabled["verify_nbf"] = False
        temporal_disabled["verify_iat"] = False
        try:
            claims = _jwt.decode(
                token,
                key,
                algorithms=algorithms,
                options=temporal_disabled,
            )
        except PyJWTError:
            raise error from None
        # PyJWT checks iat, nbf, exp in that order. A library-handled ValueError
        # is a generic invalid token, not our typed conversion-failure contract.
        for claim in ("iat", "nbf", "exp"):
            if not enabled.get(f"verify_{claim}", True) or claim not in claims:
                continue
            try:
                int(claims[claim])
            except ValueError:
                break
            except (TypeError, OverflowError):
                raise InvalidTemporalClaims("Invalid temporal claim") from None
        raise
