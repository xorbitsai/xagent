"""Audit or migrate active Gmail Pub/Sub endpoints to the S2S API base.

Run an audit first:

    python -m xagent.web.reconcile_gmail_push_endpoints

Apply the reported changes after the regional ingress is reachable:

    python -m xagent.web.reconcile_gmail_push_endpoints --execute

This command changes only each referenced active watch's Pub/Sub push endpoint,
matching OIDC audience, and stored ``push_audience``. It does not register a new
Gmail watch, so the existing history cursor and watch expiration are preserved.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Sequence

from .models.database import configure_db, get_session_local, init_db
from .services.gmail_provisioning import (
    GmailProvisioningError,
    GmailPushEndpointReconciliation,
    reconcile_gmail_push_endpoints,
)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit or migrate active Gmail Pub/Sub push endpoints to "
            "XAGENT_S2S_API_BASE_URL."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply changes; without this flag the command is read-only.",
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    """Run the reconciliation and return a process-compatible status code."""
    args = _parse_args(argv)
    if args.execute:
        init_db()
    else:
        configure_db(read_only=True)
    db = get_session_local()()
    try:
        try:
            result = reconcile_gmail_push_endpoints(db, execute=args.execute)
        except (GmailProvisioningError, ValueError) as exc:
            result = GmailPushEndpointReconciliation(
                scanned=0,
                changed=0,
                unchanged=0,
                skipped=0,
                failed=1,
                errors=(str(exc),),
            )
    finally:
        db.close()

    payload = {
        "mode": "execute" if args.execute else "audit",
        **asdict(result),
    }
    print(json.dumps(payload, sort_keys=True))
    return 1 if result.failed else 0


def main() -> None:
    """CLI entry point."""
    raise SystemExit(run())


if __name__ == "__main__":
    main()
