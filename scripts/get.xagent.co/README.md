# get.xagent.co

Cloudflare Worker that serves [`scripts/install.sh`](../install.sh) so users can install Xagent with:

```bash
curl -fsSL https://get.xagent.co | sh
```

The Worker serves the installer pinned to the **latest GitHub release tag**, so the public one-liner only ever runs a shipped, immutable version. It **fails closed**: if the release can't be resolved, or that tag doesn't contain the script, it returns `502` instead of falling back to a floating ref like `main` — a public `curl | sh` endpoint must never serve unreleased code.

> The endpoint only works once a release that includes `scripts/install.sh` exists. Cut a release after this lands to bring it online.

## Deploy

Requires [`wrangler`](https://developers.cloudflare.com/workers/wrangler/) and access to the Cloudflare account that owns the `xagent.co` zone.

```bash
cd scripts/get.xagent.co
wrangler deploy
```

Then map `get.xagent.co` to this Worker (the `routes` entry in `wrangler.toml` does this once the zone is on the account).

## Notes

- The Worker only serves the script; it runs no user code.
- Edge-caches the resolved script for 5 minutes (`CACHE_TTL_SECONDS`).
- To publish a change to the installer: merge it to `main`, then cut a release — the endpoint serves the latest release tag, so changes go live only once released.
