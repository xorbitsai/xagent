import { apiRequest } from "@/lib/api-wrapper"
import { readAuthCache, readAuthSessionSnapshot, recordIdleSessionIfCurrent } from "@/lib/auth-cache"
import { getApiUrl } from "@/lib/utils"

/**
 * Sends one explicit activity report for the given browser login. Resolves
 * false only when the report should be retried (transport or server error).
 * An idle-expired login is handled by apiRequest and never retried.
 */
export async function reportIdleActivity(sessionId: string): Promise<boolean> {
  const captured = readAuthSessionSnapshot()
  if (captured.sessionId !== sessionId || !captured.accessToken) return true
  try {
    const response = await apiRequest(`${getApiUrl()}/api/auth/activity`, { method: "POST" }, captured)
    if (response.ok) {
      recordIdleSessionIfCurrent(captured, await response.json().catch(() => null))
      return true
    }
    return response.status < 500
  } catch {
    return false
  }
}

/**
 * Asks the server whether the login is still valid, without counting as
 * activity. An idle-expired login is signed out by apiRequest; a server
 * error leaves the session alone.
 */
export async function revalidateIdleSession(sessionId: string): Promise<void> {
  const captured = readAuthSessionSnapshot()
  const username = readAuthCache()?.user.username
  if (captured.sessionId !== sessionId || !captured.accessToken) return
  try {
    const response = await apiRequest(
      `${getApiUrl()}/api/auth/verify`,
      username ? { headers: { "X-Username": username } } : {},
      captured,
    )
    if (response.ok) recordIdleSessionIfCurrent(captured, await response.json().catch(() => null))
  } catch {
    // Unknown is not expired: the next deadline check or server response decides.
  }
}
