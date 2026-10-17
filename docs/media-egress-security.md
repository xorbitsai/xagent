# Provider media egress hardening (CASA 15)

Image/video result downloads are untrusted outbound targets, even when their
URLs came from a model provider. The downloader now:

- Accepts only absolute HTTP(S) network URLs without userinfo.
- Rejects private/loopback/link-local/multicast/reserved targets, including
  mapped IPv6 literals and mixed public/private DNS answers.
- Returns validated DNS results directly to aiohttp's connection resolver.
  There is no independent hostname lookup after a safety preflight. DNS caching
  and environment proxies are disabled; normal hostname TLS verification and
  configured CA bundles remain in use.
- Follows at most five redirects, checking every target before dispatch.
- Enforces a total time budget, declared and streamed decoded-body limits,
  and rejects empty downloads. Streams to a private temporary file, replacing
  the destination only on success and cleaning up on failure.
- Restricts local image results to the task workspace, including symlink
  resolution, matching the existing video boundary. In-workspace files and
  existing data-URL workflows remain supported.

The default remote limits are 64 MiB for images and 512 MiB for videos; change
`XAGENT_IMAGE_DOWNLOAD_MAX_BYTES` / `XAGENT_VIDEO_DOWNLOAD_MAX_BYTES` with positive
integer byte counts when needed. These settings do not allow private addresses.
Private/local HTTP result hosting now fails closed; use workspace files or a
reviewed public result endpoint instead. Configured model API endpoints are
unchanged, so this does not unilaterally remove legitimate private model access.

This patch does not claim global SSRF closure. General-purpose HTTP tools,
configured MCP/model connections, proxy-side DNS policies and worker egress
firewalls need their own trust/allowlist boundaries. Do not apply a global
private-network exception to public URL downloads to work around a failure.
SaaS needs a reviewed core-pin update, rebuild and staging acceptance; no
production network settings or hosts were modified.
