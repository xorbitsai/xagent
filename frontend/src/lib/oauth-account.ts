/**
 * The account identifier step for builtin OAuth providers whose endpoints
 * live on the customer's own account host (e.g. Zendesk's
 * `{subdomain}.zendesk.com`). The backend declares the requirement on the
 * catalog app as `oauth_account_input` (src/xagent/web/oauth_account_hosts.py)
 * and takes the label as the login route's `account` parameter.
 */

/** What the connect UI shows when asking for the account label. Mirrors
 * OAuthAccountHost.input_metadata() on the backend. */
export interface OAuthAccountInput {
  label: string
  placeholder: string
  help: string
  /** The host suffix the label is joined to, with its leading dot, e.g.
   * ".zendesk.com". Shown next to the input. */
  suffix: string
}

// One DNS label, the same pattern as the backend's _ACCOUNT_LABEL_PATTERN.
// The backend stays authoritative; this only gives fast feedback.
const ACCOUNT_LABEL_PATTERN = /^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$/
const URL_SCHEME_PATTERN = /^[a-z][a-z0-9+.-]*:\/\//

/**
 * Returns the account label to send as `account`, or null when the input is
 * not a valid label.
 *
 * Trims and lowercases like the backend. A pasted address whose host is
 * exactly one label on the declared suffix (`https://acme.zendesk.com/...`)
 * is reduced to that label; anything else must already be a bare label.
 */
export function normalizeOAuthAccount(raw: string, suffix: string): string | null {
  let value = raw.trim().toLowerCase()
  const trimmedSuffix = suffix.trim().toLowerCase()
  if (trimmedSuffix) {
    // Anchor on a label boundary so "acmezendesk.com" never matches.
    const hostSuffix = trimmedSuffix.startsWith(".") ? trimmedSuffix : `.${trimmedSuffix}`
    const host = value.replace(URL_SCHEME_PATTERN, "").split(/[/?#]/, 1)[0]
    if (host.length > hostSuffix.length && host.endsWith(hostSuffix)) {
      value = host.slice(0, -hostSuffix.length)
    }
  }
  return ACCOUNT_LABEL_PATTERN.test(value) ? value : null
}
