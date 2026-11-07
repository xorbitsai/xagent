export interface UserDisplayIdentity {
  email?: string | null
  username?: string | null
  first_name?: string | null
}

/** A short greeting name, separate from the account's email label. */
export function userFirstName(
  identity: UserDisplayIdentity | null | undefined,
  fallback: string,
): string {
  const givenName = identity?.first_name?.trim()
  if (givenName && !givenName.includes("@")) return givenName
  const username = identity?.username?.trim()
  const isInternalUsername = /^acct_[a-f0-9]{32}$/.test(username ?? "")
  if (username && !username.includes("@") && !isInternalUsername) {
    return username.split(/\s+/)[0]
  }
  const email = identity?.email?.trim() || (username?.includes("@") ? username : "")
  const localName = email.split("@")[0].split(/[._+\-\s]+/)[0]
  return localName || fallback
}

/**
 * Return the human-facing account label without changing account identity.
 *
 * Email is the product label because SaaS password registrations use opaque
 * usernames. The username fallback keeps legacy email-less service accounts
 * distinguishable while callers provide a final context-specific fallback for
 * deleted or otherwise incomplete records.
 */
export function userDisplayLabel(
  identity: UserDisplayIdentity | null | undefined,
  fallback: string,
): string {
  return identity?.email?.trim() || identity?.username?.trim() || fallback
}
