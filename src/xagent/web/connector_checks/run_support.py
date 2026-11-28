"""Shared setup for connector checks and sweeps."""

from __future__ import annotations

import logging
import math
import time
from datetime import timedelta
from typing import Any

from .fixtures import FixtureMissing
from .identity import ConnectorTarget, IdentityAdapter, IdentitySetupError


class ResolverFailed(Exception):
    """The adapter's fixture resolver raised; carries nothing from the error."""


class SafeResolver:
    """Wraps the adapter's resolver so a raise becomes ``ResolverFailed``."""

    def __init__(self, adapter: IdentityAdapter, logger: logging.Logger) -> None:
        self._adapter = adapter
        self._logger = logger

    def resolve_fixture(self, key: str) -> Any:
        try:
            return self._adapter.resolve_fixture(key)
        except FixtureMissing:
            # A missing key all the same; reported under the requested key,
            # which is a validated code, never under one the adapter chose.
            raise FixtureMissing(key) from None
        except Exception as error:
            self._logger.warning(
                "Resolving fixture %s raised %s", key, type(error).__name__
            )
            raise ResolverFailed() from None


class RunDeadline:
    """The run's deadline, and whether it has cut anything."""

    def __init__(self, total_timeout: float) -> None:
        self._started = time.monotonic()
        self._deadline = self._started + total_timeout
        self.cut = False

    def elapsed(self) -> timedelta:
        return timedelta(seconds=time.monotonic() - self._started)

    def remaining(self) -> float:
        return self._deadline - time.monotonic()

    def expired(self) -> bool:
        if not self.cut and self.remaining() <= 0:
            self.cut = True
        return self.cut


def validate_timeout(timeout: float) -> None:
    # A bool is an int; NaN breaks the event loop's timer ordering, and
    # infinity never expires.
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("a timeout must be a finite number of seconds above 0")


def enumerate_targets(
    adapter: IdentityAdapter, logger: logging.Logger
) -> tuple[list[ConnectorTarget], str | None]:
    """Return targets in adapter order, or a sanitized setup failure code."""
    try:
        targets = list(adapter.enumerate_targets())
    except IdentitySetupError as error:
        return [], error.code
    except Exception as error:
        logger.warning("Enumerating connectors raised %s", type(error).__name__)
        return [], "enumeration:exception"
    if not targets:
        return [], "enumeration:no_connectors"
    if len({target.connector for target in targets}) != len(targets):
        return [], "enumeration:duplicate_connector"
    return targets, None
