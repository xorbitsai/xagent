# Tool credential key upgrades

Production requires both `ENCRYPTION_KEY` (Fernet for model API keys, channel
configuration, OAuth/MCP credentials and runtime values) and
`XAGENT_SECRET_ENCRYPTION_KEY` (32+ secret characters for tool credentials).
`SECRET_KEY` remains an alias for the latter. Preserve the exact old value if
it already meets the requirement: whitespace is part of historical derivation.
Web, standalone workers, combined launchers and Celery worker/beat validate
encryption at startup. Database initialization binds the configured database,
then checks persisted ciphertext before any schema migration or seeding;
workers also check before accepting work. An unreadable stored
credential aborts startup with a table/row/field location and this upgrade guide,
never a silent environment/default fallback. No plaintext or ciphertext is logged.

The backend image generates a separate persistent tool secret at
`/run/xagent-secrets/tool-credentials.secret` when neither tool-key variable is
configured. Back up the secrets volume together with the database. A generated
key does not decrypt data written under an older default/key.

Before upgrading a deployment that used `xagent-dev-key` or a short key:

1. Back up the database and all original keys. Stop every web/worker/Celery
   writer; do not run mixed-key replicas.
2. Configure the new strong `XAGENT_SECRET_ENCRYPTION_KEY` consistently on all
   services. For Docker, initialize and reuse the persisted tool-key file.
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

`python -m xagent.web.encryption_key_migration` reads the exact old Fernet key from
`XAGENT_ENCRYPTION_MIGRATION_KEY` and the explicit private destination from
`ENCRYPTION_KEY`. It reads raw columns (bypassing ORM encryption hooks), rotates
recognized encrypted values in one transaction and rolls back on any unreadable
entry. Already migrated fields are skipped; unrelated plaintext/JSON is preserved.
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
