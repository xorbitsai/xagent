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
    const url = new URL(configured)
    // URL parsing alone permits e.g. semicolons in hostnames, unsafe as CSP
    // source expressions. Accept only canonical DNS/IPv4/IPv6 host spellings.
    if (!/^(?:[a-z0-9.-]+|\[[0-9a-f:]+\])$/i.test(url.hostname)) return []
    const loopback = ["localhost", "127.0.0.1", "[::1]"].includes(url.hostname)
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
