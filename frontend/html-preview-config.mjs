// Shared by the browser runner resolver and the build-time Next CSP.
export const AUTO_PREVIEW_HOSTS = new Map([
  ["cloud.xagent.co", "preview.xagent.co"],
  ["stage.cloud.xagent.co", "preview-stage.xagent.co"],
  ["dev.cloud.xagent.co", "preview-dev.xagent.co"],
])

/** @param {string | undefined} configuredOrigin */
export function htmlPreviewOrigins(configuredOrigin) {
  const configured = configuredOrigin?.trim()
  if (!configured) return []
  if (configured === "auto") return [...AUTO_PREVIEW_HOSTS.values()].map(host => `https://${host}`)
  try {
    // Keep operator configuration in canonical URL form. WHATWG URL parsing
    // otherwise silently rewrites encoded dots, backslashes and short IPv4
    // spellings differently from the packaged Python CSP builder.
    if (/[\\%]/.test(configured)) return []
    const url = new URL(configured)
    // URL parsing alone permits e.g. semicolons in hostnames, unsafe as CSP
    // source expressions. Accept only canonical DNS/IPv4/IPv6 host spellings.
    if (!/^(?:[a-z0-9.-]+|\[[0-9a-f:]+\])$/i.test(url.hostname)) return []
    const authority = configured.match(/^http:\/\/([^/?#]+)/i)?.[1]?.toLowerCase()
    const loopback = url.hostname === "localhost"
      ? /^localhost(?::\d+)?$/.test(authority ?? "")
      : url.hostname === "127.0.0.1"
        ? /^127\.0\.0\.1(?::\d+)?$/.test(authority ?? "")
        : url.hostname === "[::1]"
          ? /^\[[0-9a-f:]+\](?::\d+)?$/i.test(authority ?? "")
          : false
    if (url.username || url.password) return []
    if (url.protocol !== "https:" && !(url.protocol === "http:" && loopback)) return []
    return [url.origin]
  } catch {
    return []
  }
}

/**
 * @param {string | undefined} configuredOrigin
 * @param {string | undefined} [browserOrigin]
 */
export function resolveHtmlPreviewRunnerUrl(configuredOrigin, browserOrigin) {
  const configured = configuredOrigin?.trim()
  if (configured === "auto") {
    try {
      const host = AUTO_PREVIEW_HOSTS.get(new URL(browserOrigin ?? "").hostname)
      return host ? `https://${host}/render` : null
    } catch {
      return null
    }
  }
  const origin = htmlPreviewOrigins(configured)[0]
  return origin ? `${origin}/render` : null
}
