import { apiRequest } from "@/lib/api-wrapper"
import { readAuthSessionSnapshot, recordIdleSessionIfCurrent } from "@/lib/auth-cache"
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
