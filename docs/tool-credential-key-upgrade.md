# Tool credential key upgrades

Production requires both `ENCRYPTION_KEY` (Fernet for model API keys, channel
configuration, OAuth/MCP credentials and runtime values) and
`XAGENT_SECRET_ENCRYPTION_KEY` (32+ secret characters for tool credentials).
`SECRET_KEY` remains an alias for the latter. Preserve the exact old value if
it already meets the requirement: whitespace is part of historical derivation.
Web, standalone workers, combined launchers and Celery worker/beat validate
encryption at startup. Database initialization binds the configured database,
then checks persisted ciphertext before any schema migration or seeding;
workers also check before accepting work. An unreadable durable credential in
`models`, `user_channels`, `oauth_providers`, `mcp_servers`, `user_mcpservers`,
`actor_mcp_server_connections`, `custom_apis`, `mcp_oauth_clients`,
`mcp_oauth_grants`, `agent_triggers` (`secret_encrypted`),
`global_memory_embedding_authority`, `tool_configs` or `user_tool_configs`
aborts startup with a table/row/field location and this guide, never a silent
environment/default fallback. Failed database initialization clears its engine
and session factory so subsequent lazy worker tasks must retry admission.
At runtime, an unreadable stored SQL connection is skipped individually without
substituting its same-name environment connection; other connections remain usable.
Unreadable `user_oauth` tokens and transient `mcp_oauth_flow_states`,
`task_runtime_secrets` and `trigger_runs` values instead warn so account
reauthorization and administrator recovery remain available. Their ciphertext
is preserved; the affected connection, flow or run must be reauthorized/restarted,
not treated as an empty credential or replaced by a default. The offline rotation
CLI remains strict and rolls back on any unreadable encrypted field; restore the
source key to rotate that data. No data is automatically discarded, and no
plaintext or ciphertext is logged.

The backend image generates a separate persistent tool secret at
`/run/xagent-secrets/tool-credentials.secret` when neither tool-key variable is
configured. Back up the secrets volume together with the database. A generated
key does not decrypt data written under an older default/key.

Before upgrading a deployment that used `xagent-dev-key` or a short key:

1. Back up the database and all original keys. Stop every web/worker/Celery
   writer; do not run mixed-key replicas.
2. Configure the new strong `XAGENT_SECRET_ENCRYPTION_KEY` consistently on all
   services. Known public placeholders are rejected even if 32+ characters long.
   For Docker, remove/unset an old short `SECRET_KEY` alias (and an old short
   primary variable) before initializing and reusing the persisted tool-key file;
   the entrypoint preserves explicit values rather than replacing weak keys.
3. In an offline container/process with the same database configuration, set
   `XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY` to the exact old tool key (explicitly
   `xagent-dev-key` only if that was really used), then run
   `python -m xagent.web.tool_credentials_key_migration`.
4. The command reads raw existing `config` columns without requiring the latest
   ORM schema or a per-user table that older versions did not have. It
   re-encrypts global tool credentials and per-user SQL connection
   ciphertext in one transaction, preserves unrelated fields, skips values
   already readable with the new key, and rolls back if any value cannot be
   decrypted. It never prints keys or plaintext. Unset the migration variable.
5. Start all services using the new key and verify existing connections. If
   migration fails, keep services stopped, restore the backup/configuration or
   correct the old key; do not overwrite original ciphertext.

## Upgrading the published v0.7.x ENCRYPTION_KEY

Older examples shipped a public development Fernet key; installs without an
explicit key could also use it implicitly. Do not simply replace it and restart:
existing model/channel/OAuth data needs re-encryption first. There is deliberately
no production fallback to a published key.
Equivalent Base64 spellings of that public key are rejected too, including
standard-Base64 alphabet variants, extra padding and embedded whitespace.
The exact historical value is `DEV_FALLBACK_ENCRYPTION_KEY` in `xagent.config`;
use that source only if it is the key your deployment actually used.

`python -m xagent.web.encryption_key_migration` reads the exact old Fernet key from
`XAGENT_ENCRYPTION_MIGRATION_KEY` and the explicit private destination from
`ENCRYPTION_KEY`. It reads raw columns (bypassing ORM encryption hooks), rotates
recognized encrypted values in one transaction and rolls back on any unreadable
entry. Already migrated fields are skipped; unrelated plaintext/JSON is preserved.
Both startup checks and rotation follow the writers' exact encrypted JSON paths:
`trigger_runs.payload_snapshot.encrypted_payload`, the three channel token fields,
MCP sensitive auth fields, and direct string values in encrypted environment maps.
Webhook event metadata and other plaintext fields are never treated as ciphertext,
even when their values have a Fernet-like shape or decrypt with an old key.
It covers the core model, channel, OAuth provider/account, MCP/custom API, trigger,
runtime-secret and global embedding-authority columns. The global authority's
key-derived verifier is recomputed. Verify memory admission after restart too.
Additional embedding/plugin databases or external credential files are not part
of this core database catalog and need their own rotation procedure.

### Concrete Docker Compose sequence

Back up the database, `.env` and `xagent_secrets` first. Stop **all** web, Agent,
Celery and other writers, including hosts outside this Compose project. Configure
the new explicit private `ENCRYPTION_KEY` in `.env` and retain an existing strong
tool key, or initialize the separate persistent tool key through the entrypoint.
Use a backend image containing these migration commands.

```bash
docker compose stop backend worker scheduler

# Enter the exact OLD Fernet key without placing it in shell history/argv.
read -r -s XAGENT_ENCRYPTION_MIGRATION_KEY
export XAGENT_ENCRYPTION_MIGRATION_KEY
docker compose run --rm --no-deps -e XAGENT_ENCRYPTION_MIGRATION_KEY \
  backend python -m xagent.web.encryption_key_migration
unset XAGENT_ENCRYPTION_MIGRATION_KEY

# Only when the tool key also changes (for example the old xagent-dev-key).
read -r -s XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY
export XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY
docker compose run --rm --no-deps -e XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY \
  backend python -m xagent.web.tool_credentials_key_migration
unset XAGENT_TOOL_CREDENTIALS_MIGRATION_KEY

# Run only after both required commands succeed.
docker compose up -d backend worker scheduler
```

Keep PostgreSQL available during the offline commands. Never restart writers
between the two migrations or run mixed-key replicas. On failure, keep them
stopped and correct the source key or restore the backup/configuration together.
Transient OAuth authorization flows carrying old encrypted state must restart;
drain those flows and allow their ten-minute lifetime to expire during the upgrade.
