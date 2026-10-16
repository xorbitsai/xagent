# Tool credential key upgrades

Production requires both `ENCRYPTION_KEY` (Fernet for OAuth/runtime values) and
`XAGENT_SECRET_ENCRYPTION_KEY` (32+ secret characters for tool credentials).
`SECRET_KEY` remains an alias for the latter. Preserve the exact old value if
it already meets the requirement: whitespace is part of historical derivation.
Web, standalone workers and combined launchers validate encryption at startup.

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
4. The command re-encrypts global tool credentials and per-user SQL connection
   ciphertext in one transaction, preserves unrelated fields, skips values
   already readable with the new key, and rolls back if any value cannot be
   decrypted. It never prints keys or plaintext. Unset the migration variable.
5. Start all services using the new key and verify existing connections. If
   migration fails, keep services stopped, restore the backup/configuration or
   correct the old key; do not overwrite original ciphertext.

This does not rotate OAuth/runtime `ENCRYPTION_KEY`. There is deliberately no
automatic production fallback to the published development tool key.
