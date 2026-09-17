export interface UserDisplayIdentity {
  email?: string | null
  username?: string | null
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
