"""Timeout configuration for LLM streaming calls."""

import os
from dataclasses import dataclass


@dataclass
class TimeoutConfig:
    """LLM timeout configuration

    Attributes:
        first_token_timeout: First token timeout (TTFT) - Time to wait for first token (seconds)
        token_interval_timeout: Token interval timeout (TPOT) - Maximum interval between tokens (seconds)
    """

    first_token_timeout: float = float(
        os.getenv(
            "XAGENT_LLM_FIRST_TOKEN_TIMEOUT",
            "300",  # Default 5 minutes
        )
    )

    token_interval_timeout: float = float(
        os.getenv(
            "XAGENT_LLM_TOKEN_INTERVAL_TIMEOUT",
            "60",  # Default 1 minute
        )
    )
