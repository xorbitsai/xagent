"""Run the smoke V2 checks as a local user and print the run result.

Usage: ``python -m xagent.web.connector_checks --user NAME [--connector LABEL]...
[--check CHECK_ID]... [--environment LABEL] [--check-timeout SECONDS]
[--total-timeout SECONDS]``, run in the backend container with the running
service's environment, the same way as ``oauth_readiness``::

    docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \\
        python -m xagent.web.connector_checks --user monitor

It runs the engine (``engine.run_checks``) under the ``user`` identity
(``user_identity.py``): the user must be named in
``XAGENT_CONNECTOR_PROBE_USERS``, and any other user is refused before a
connector starts. It takes selectors only (the user, connector labels and
check ids), never a tool name or an argument.

Stdout carries exactly one JSON document, the run result
(``result.schema.json``), and logs go to stderr. During the run, file
descriptor 1 and ``sys.stdout`` point at stderr, so nothing a connector, a
tool, a library or a child process writes can reach the document's stream.
A run that cannot be set up still prints its document, with a ``failure``:
the adapter's and the engine's codes, ``setup:database`` when the database
cannot be opened, or ``run:exception`` when the engine raises. Only invalid
arguments print no document.

The exit status is 0 when the run is complete and no row failed, 1 otherwise
(a ``skip`` or ``unprobeable`` row does not fail the run), and 2 for invalid
arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import math
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime, timezone
from typing import Any

from .engine import DEFAULT_TOTAL_TIMEOUT_SECONDS, run_checks
from .results import ENVIRONMENT_PATTERN, Outcome, RunResult
from .user_identity import USER_IDENTITY_LABEL, UserIdentityAdapter

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    with _stdout_to_stderr():
        result = asyncio.run(_run(args))
    print(json.dumps(result.to_document()), flush=True)
    failed = any(row.outcome is Outcome.FAIL for row in result.rows)
    return 0 if result.complete and not failed else 1


async def _run(args: argparse.Namespace) -> RunResult:
    started_at = datetime.now(timezone.utc)

    def failure(code: str) -> RunResult:
        return RunResult(
            identity=USER_IDENTITY_LABEL,
            environment=args.environment,
            started_at=started_at,
            finished_at=max(started_at, datetime.now(timezone.utc)),
            complete=False,
            rows=(),
            failure=code,
        )

    try:
        with _database() as db:
            adapter = UserIdentityAdapter(db, args.user)
            try:
                return await run_checks(
                    adapter,
                    environment=args.environment,
                    connectors=_flatten(args.connector),
                    checks=_flatten(args.check),
                    check_timeout=args.check_timeout,
                    total_timeout=args.total_timeout,
                )
            except Exception as error:
                # The class name only: the text can carry a value.
                logger.error("The engine raised %s", type(error).__name__)
                return failure("run:exception")
    except Exception as error:
        # The class name only: a driver error can carry the database URL.
        logger.error("The database could not be opened: %s", type(error).__name__)
        return failure("setup:database")


@contextlib.contextmanager
def _database() -> Iterator[Any]:
    """A session on the service's database. Bind only: container startup
    owns migrations and seeding."""
    from ..models.database import configure_db, get_engine, get_session_local

    configure_db()
    engine = get_engine()
    try:
        with get_session_local()() as db:
            yield db
    finally:
        engine.dispose()


@contextlib.contextmanager
def _stdout_to_stderr() -> Iterator[None]:
    """Point file descriptor 1 and ``sys.stdout`` at stderr until the block
    exits, so only the document reaches stdout."""
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            yield
    finally:
        # Written while descriptor 1 is still stderr.
        sys.stdout.flush()
        os.dup2(saved, 1)
        os.close(saved)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m xagent.web.connector_checks",
        description="Run the smoke V2 checks of a local user's connected catalog "
        "connectors and print the run result as one JSON document.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--user",
        required=True,
        help="the user to run as: a username, or an email address compared "
        "case-insensitively; must be named in XAGENT_CONNECTOR_PROBE_USERS",
    )
    parser.add_argument(
        "--connector",
        action="append",
        type=_selector_names,
        metavar="LABEL",
        help="run only these connectors, by their label in the result; "
        "repeatable or comma-separated",
    )
    parser.add_argument(
        "--check",
        action="append",
        type=_selector_names,
        metavar="CHECK_ID",
        help="run only these checks, by id (gmail.l3_probe); repeatable or "
        "comma-separated",
    )
    parser.add_argument(
        "--environment",
        type=_environment,
        help="the environment label the result carries (stage)",
    )
    parser.add_argument(
        "--check-timeout",
        type=_seconds,
        help="each tool call's bound in seconds; defaults to the L3 probe call's",
    )
    parser.add_argument(
        "--total-timeout",
        type=_seconds,
        default=DEFAULT_TOTAL_TIMEOUT_SECONDS,
        help="the whole run's bound in seconds (default: %(default)g)",
    )
    return parser


def _flatten(values: Iterable[frozenset[str]] | None) -> frozenset[str] | None:
    """One set from a repeatable selector; None when it was not given."""
    if values is None:
        return None
    return frozenset(name for names in values for name in names)


def _selector_names(value: str) -> frozenset[str]:
    return frozenset(item.strip() for item in value.split(",") if item.strip())


def _environment(value: str) -> str:
    if not ENVIRONMENT_PATTERN.fullmatch(value):
        raise argparse.ArgumentTypeError("must be a lowercase label such as stage")
    return value


def _seconds(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError:
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a number of seconds above 0")
    return seconds
