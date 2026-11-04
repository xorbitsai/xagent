"""Post-deploy readiness check for builtin OAuth clients.

Usage: ``python -m xagent.web.oauth_readiness [--exclude PROVIDER]...``, run in
the backend container with the running service's environment. In the stock
image that means through its entrypoint, which loads ``ENCRYPTION_KEY`` from
``XAGENT_ENCRYPTION_KEY_FILE``::

    docker compose exec -T backend /opt/xagent/deploy/entrypoint.sh \
        python -m xagent.web.oauth_readiness

Without a usable key the command prints one ``FAIL encryption-key`` line and
checks nothing: the server refuses to start in production without one, so
this process would not be seeing the server's environment.

The authorization launch only resolves a provider's client ID, so a missing
client secret first fails in the callback, after the user has consented at
the vendor. This command resolves both fields for every builtin OAuth provider
through the runtime resolver (the ``oauth_providers`` row first, then the
``<PROVIDER>_CLIENT_ID`` / ``<PROVIDER>_CLIENT_SECRET`` environment variable)
and prints one line per provider. It reports only where each field resolved
from, never a value or part of one.

It proves that a value is present, not that the vendor accepts it. A row
encrypted under a different key is reported as ``unreadable``: the runtime
would send the ciphertext as the value, and the environment variable cannot
take over because the row is not blank.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TextIO

from sqlalchemy.orm import Session

from ..core.utils.encryption import (
    EncryptionDecodeError,
    decrypt_value_strict,
    get_cipher,
)

FieldSource = Literal["row", "env", "missing", "unreadable"]
_READY_SOURCES = frozenset({"row", "env"})


@dataclass(frozen=True)
class ProviderReadiness:
    provider: str
    has_row: bool
    client_id: FieldSource
    client_secret: FieldSource

    @property
    def ready(self) -> bool:
        return (
            self.has_row
            and self.client_id in _READY_SOURCES
            and self.client_secret in _READY_SOURCES
        )


def _field_source(provider: str, stored: str | None, env_suffix: str) -> FieldSource:
    from .api.auth import _resolve_oauth_secret

    from_row = False
    if stored:
        try:
            from_row = bool(decrypt_value_strict(stored))
        except EncryptionDecodeError:
            return "unreadable"
    if not _resolve_oauth_secret(provider, stored, env_suffix):
        return "missing"
    return "row" if from_row else "env"


def check_provider(provider: str, row: Any | None) -> ProviderReadiness:
    """Resolve one provider's client ID and secret the way a connect does."""
    return ProviderReadiness(
        provider=provider,
        has_row=row is not None,
        client_id=_field_source(provider, getattr(row, "client_id", None), "CLIENT_ID"),
        client_secret=_field_source(
            provider, getattr(row, "client_secret", None), "CLIENT_SECRET"
        ),
    )


def check_providers(
    rows: Iterable[Any], *, builtin_providers: Iterable[str], exclude: set[str]
) -> list[ProviderReadiness]:
    """Check every builtin provider and every ``oauth_providers`` row.

    Raises ``ValueError`` when ``exclude`` names a provider that is in
    neither set, so a misspelled exclusion fails instead of excluding nothing.
    """
    rows_by_provider = {str(row.provider_name): row for row in rows}
    providers = list(dict.fromkeys([*builtin_providers, *sorted(rows_by_provider)]))
    unknown = sorted(exclude - set(providers))
    if unknown:
        raise ValueError(f"unknown provider to exclude: {', '.join(unknown)}")
    return [
        check_provider(provider, rows_by_provider.get(provider))
        for provider in providers
        if provider not in exclude
    ]


def format_result(result: ProviderReadiness) -> str:
    status = "PASS" if result.ready else "FAIL"
    if not result.has_row:
        detail = "no provider row"
    else:
        detail = f"client_id={result.client_id} client_secret={result.client_secret}"
    return f"{status} oauth-client:{result.provider}: {detail}"


def _load_provider_rows() -> list[Any]:
    from .models.database import create_database_engine
    from .models.oauth_provider import OAuthProvider

    engine = create_database_engine(read_only=True)
    try:
        with Session(engine) as db:
            rows = db.query(OAuthProvider).all()
            db.expunge_all()
            return rows
    finally:
        engine.dispose()


def _parse_exclude(values: Iterable[str] | None) -> set[str]:
    return {
        item.strip()
        for value in values or ()
        for item in value.split(",")
        if item.strip()
    }


def main(argv: Sequence[str] | None = None, *, out: TextIO | None = None) -> int:
    out = sys.stdout if out is None else out
    parser = argparse.ArgumentParser(
        prog="python -m xagent.web.oauth_readiness",
        description="Check that every builtin OAuth provider has a client ID "
        "and client secret configured. Prints no values.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        metavar="PROVIDER",
        help="OAuth provider name (e.g. google), not a catalog app ID as in "
        "connector_smoke's --exclude, that this deployment leaves unconfigured "
        "on purpose; repeatable or comma-separated",
    )
    args = parser.parse_args(argv)

    from .builtin_mcp_registry import get_builtin_oauth_provider_rows

    builtin_providers = [
        str(row["provider_name"]) for row in get_builtin_oauth_provider_rows()
    ]
    try:
        get_cipher()
    except ValueError as exc:
        print(f"FAIL encryption-key: {type(exc).__name__}", file=out)
        return 1
    try:
        rows = _load_provider_rows()
    except Exception as exc:
        # The class name only: a driver error can carry the database host.
        print(f"FAIL database: {type(exc).__name__}", file=out)
        return 1
    try:
        results = check_providers(
            rows,
            builtin_providers=builtin_providers,
            exclude=_parse_exclude(args.exclude),
        )
    except ValueError as exc:
        parser.error(str(exc))

    failed = sum(not result.ready for result in results)
    for result in results:
        print(format_result(result), file=out)
    print(f"{len(results)} checks, {failed} failed", file=out)
    return 1 if failed else 0


if __name__ == "__main__":
    from dotenv import load_dotenv

    # Same as the server's own entry point (xagent.web.__main__).
    load_dotenv()
    sys.exit(main())
