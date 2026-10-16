"""Application JWT primitives backed by PyJWT, never python-jose.

Keep a small import boundary so authentication owners share the same exception
family. Unverified claims are diagnostic input only, never authentication.
"""

from typing import Any

import jwt as _jwt
from jwt import ExpiredSignatureError as ExpiredSignatureError
from jwt import PyJWTError
from jwt import decode as decode
from jwt import encode as encode
from jwt import get_unverified_header as get_unverified_header

JWTError = PyJWTError


def get_unverified_claims(token: str) -> dict[str, Any]:
    """Decode diagnostic claims without trusting them or following key URLs."""
    return _jwt.decode(
        token,
        options={
            "verify_signature": False,
            "verify_exp": False,
            "verify_nbf": False,
            "verify_iat": False,
            "verify_aud": False,
            "verify_iss": False,
            "verify_sub": False,
            "verify_jti": False,
        },
    )
