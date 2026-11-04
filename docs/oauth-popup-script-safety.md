# OAuth popup script encoding

OAuth success callbacks embed provider identity, connector ID and the return
origin in an inline script. JSON quoting alone does not prevent `</script>`
from terminating that element, so all three expressions use the shared
`script_safe_json` encoder. The Slack popup uses the same encoder.

The encoder escapes HTML-sensitive characters without changing the decoded
JSON value. Tests exercise real callback persistence with hostile provider
identity and connector IDs, ordinary strings, mixed-case script terminators,
HTML comments, Unicode and the return-origin expression. Slack regression
coverage remains in place.

This is a targeted XSS fix, not complete CASA #14 closure. It does not add a
site-wide CSP, audit every preview renderer or establish deployment evidence.
Deploy the updated core package on every API replica; SaaS must advance its
core submodule before this fix is included in its deployment.
