// Keep the independently served Next frontend covered too. srcDoc previews
// inherit this policy: inline/HTTPS scripts are an explicit compatibility
// exception, not a claim of strict CSP or full CASA #14 closure.
export function browserSecurityHeaders({ embeddable = false, development = false } = {}) {
  const directives = [
    "default-src 'self'",
    "base-uri 'none'",
    "object-src 'none'",
    "form-action 'self'",
    `script-src 'self' 'unsafe-inline' https:${development ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline' https:",
    `img-src 'self' data: blob: https:${development ? " http:" : ""}`,
    `media-src 'self' data: blob: https:${development ? " http:" : ""}`,
    "font-src 'self' data: https:",
    `connect-src 'self' https: wss:${development ? " http: ws:" : ""}`,
    "frame-src 'self' blob: https://docs.google.com",
    "worker-src 'self' blob:",
  ]
  if (!embeddable) directives.push("frame-ancestors 'none'")
  return [
    { key: "Content-Security-Policy", value: directives.join("; ") },
    { key: "X-Content-Type-Options", value: "nosniff" },
    { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
    ...(!embeddable ? [{ key: "X-Frame-Options", value: "DENY" }] : []),
  ]
}
