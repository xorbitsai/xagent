"use client"

import { useEffect } from "react"

import { markIdleActivityReportedIfCurrent, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { isUserActivityEvent, startIdleActivityReporter } from "@/lib/idle-activity"
import { reportIdleActivity } from "@/lib/idle-session-client"

/**
 * Browser side of the server idle-session policy for one login: reports
 * deliberate user input. Renders nothing for servers without idle support.
 */
export function IdleSessionManager({ sessionId }: { sessionId: string | null }) {
  useEffect(() => {
    if (!sessionId) return
    const reporter = startIdleActivityReporter({
      target: window,
      isUserEvent: isUserActivityEvent,
      getRecord: () => {
        const record = readCurrentIdleSession()
        return record?.sessionId === sessionId ? record : null
      },
      markReported: () => {
        const current = readAuthSessionSnapshot()
        if (current.sessionId === sessionId) markIdleActivityReportedIfCurrent(current)
      },
      report: () => reportIdleActivity(sessionId),
    })
    return () => reporter.stop()
  }, [sessionId])
  return null
}
