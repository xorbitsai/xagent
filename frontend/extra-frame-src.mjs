// Host-supplied frame-src additions, e.g. a third-party widget iframe that a
// custom build embeds in its pages. Kept in parity with
// src/xagent/web/extra_frame_src.py.
//
// Entries are matched as plain strings, not parsed as URLs: only
// `https://<host>` and `https://*.<host>` with a canonical lowercase DNS host
// pass, so ports, paths, credentials, IP literals, escapes and CSP delimiters
// cannot reach the policy. A wildcard needs at least two labels after `*.`.
const LABEL = "[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
const TOP_LABEL = "[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?"
const ENTRY = new RegExp(`^https://(?:\\*\\.(?:${LABEL}\\.)+|(?:${LABEL}\\.)*)${TOP_LABEL}$`)

/**
 * Valid entries of a space-separated list, in order and without duplicates.
 * Invalid entries are dropped one by one.
 *
 * @param {string | undefined} configured
 */
export function extraFrameSrcOrigins(configured) {
  const entries = (configured ?? "").split(/[\t\n\f\r ]+/)
  return [...new Set(entries.filter(entry => ENTRY.test(entry)))]
}
