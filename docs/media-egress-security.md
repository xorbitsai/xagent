# Provider media egress hardening (CASA 15)

Image/video result downloads are untrusted outbound targets, even when their
URLs came from a model provider. The downloader now:

- Accepts only absolute HTTP(S) network URLs without userinfo.
- Rejects private/loopback/link-local/multicast/reserved targets, including
  mapped IPv6 literals and mixed public/private DNS answers.
- Returns validated DNS results directly to aiohttp's connection resolver.
  There is no independent hostname lookup after a safety preflight. DNS caching
  is disabled and the existing no-environment-proxy behavior is retained;
  normal hostname TLS verification remains enabled. Media TLS keeps aiohttp's
  stdlib `ssl.create_default_context()` trust-store behavior, including platform
  default CA paths and `SSL_CERT_FILE` / `SSL_CERT_DIR` overrides; it does not
  switch to HTTPX's certifi-only defaults. The context is cached by those paths;
  restart workers after replacing CA contents at an unchanged path.
- Follows at most five redirects, checking every target before dispatch and
  refusing HTTPS-to-HTTP downgrades. Hosts are checked after IDNA/UTS46
  normalization. Ambiguous decimal/short IPv4 literals are rejected; other
  numeric spellings (such as hexadecimal hostnames) still pass through DNS
  address validation rather than bypassing it.
- Enforces a total time budget, declared and streamed decoded-body limits,
  and rejects empty downloads. Streams to a private temporary file, replacing
  the destination only on success and cleaning up on failure.
  Downloads now use mode 0600 instead of the previous umask-derived mode;
  deployments sharing files across UIDs must arrange
  an explicit sharing policy rather than widening permissions for all downloads.
- Restricts local image results to the task workspace, including symlink
  resolution, matching the existing video boundary. In-workspace files and
  existing data-URL workflows remain supported.

The default remote limits are 64 MiB for images and 512 MiB for videos; change
`XAGENT_IMAGE_DOWNLOAD_MAX_BYTES` / `XAGENT_VIDEO_DOWNLOAD_MAX_BYTES` with positive
byte counts or size suffixes such as `64M` when needed. These settings do not allow private addresses.
Private/local HTTP result hosting now fails closed; private model servers must
return base64/data URLs, workspace files or a reviewed public result endpoint
instead. Configured model API endpoints are
unchanged, so this does not unilaterally remove legitimate private model access.
Xinference image generation/editing now requests `b64_json` by default because
its `url` response may contain a server-local filesystem path. Explicit URL
responses must still satisfy the workspace/public-network result boundary.
Xinference video results are still URL-based when supplied that way by the
provider; private/co-located HTTP result servers are not exempt. Return inline
base64/data content, a file inside the task workspace, or a public result URL.
There is no private-network fallback or automatic video format conversion.
When a provider succeeds but saving its result fails, image generation/editing
and video generation return `saved_to_workspace: false` and a redacted,
non-empty `download_error`. Missing-local-file and filename-bearing OS errors
use path-free messages. Provider success alone does not mean a local artifact
was created.

This patch does not claim global SSRF closure. General-purpose HTTP tools,
configured MCP/model connections, proxy-side DNS policies and worker egress
firewalls need their own trust/allowlist boundaries. Do not apply a global
private-network exception to public URL downloads to work around a failure.
Image edit-input downloads in provider adapters remain a separate hardening
follow-up; this result-download boundary does not make them safe automatically.
