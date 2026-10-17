# Legacy file access compatibility

Public preview/download now accepts registered UUID file capabilities only.
The public `/uploads` mount exposes passive agent-logo images, not user files.

`/preview/{legacy_path}` is intentionally Bearer-authenticated and owner-only,
including for administrators. Admin access to the authenticated file-ID APIs
is unchanged. Plain browser navigations and image tags cannot attach a Bearer
header, so old `/preview/...` URLs and unmapped `file:<path>` inline references
will no longer render anonymously, even for their owner. Changing their URL to
the authenticated API alone would not fix this browser-header limitation.

Clients migrating old messages must resolve/register legacy references under
the owner's authenticated request and use the returned file ID (or fetch the
authenticated preview into a blob). Existing frontend file mappings continue
to work. Do not restore cross-user path resolution on a public endpoint or put
the login JWT in a URL to preserve old links.
