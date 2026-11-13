"use client"

import { useEffect } from "react"

import { markIdleActivityReportedIfCurrent, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { isUserActivityEvent, startIdleActivityReporter } from "@/lib/idle-activity"
import { AUTH_REVALIDATE_EVENT, idleRemainingMs } from "@/lib/idle-session"
import { reportIdleActivity, revalidateIdleSession } from "@/lib/idle-session-client"

/** How often the deadline is checked; also the gap that reveals a sleeping device or a clock change. */
const DEADLINE_TICK_MS = 5_000
const CLOCK_JUMP_TOLERANCE_MS = 30_000
/** Minimum gap between two server checks, so a server outage does not turn into a request loop. */
const REVALIDATE_MIN_GAP_MS = 15_000

/**
 * Browser side of the server idle-session policy for one login: reports
 * deliberate user input, and asks the server (verify, which is not activity)
 * when the deadline seems to have passed, after the device wakes, comes
 * back online or regains the user's attention, and when a socket says the
 * session ended. It never signs the user out on its own clock; the server's
 * answer does. Renders nothing for servers without idle support.
 */
export function IdleSessionManager({ sessionId }: { sessionId: string | null }) {
  useEffect(() => {
    if (!sessionId) return
    const getRecord = () => {
      const record = readCurrentIdleSession()
      return record?.sessionId === sessionId ? record : null
    }
    const reporter = startIdleActivityReporter({
      target: window,
      isUserEvent: isUserActivityEvent,
      getRecord,
      markReported: () => {
        const current = readAuthSessionSnapshot()
        if (current.sessionId === sessionId) markIdleActivityReportedIfCurrent(current)
      },
      report: () => reportIdleActivity(sessionId),
    })

    let inFlight = false
    let lastCheckAt = Number.NEGATIVE_INFINITY
    const revalidate = (requireEnforced: boolean) => {
      const record = getRecord()
      if (!record || (requireEnforced && !record.enforced)) return
      if (inFlight || Date.now() - lastCheckAt < REVALIDATE_MIN_GAP_MS) return
      inFlight = true
      lastCheckAt = Date.now()
      void revalidateIdleSession(sessionId).finally(() => { inFlight = false })
    }
    const checkDeadline = () => {
      const record = getRecord()
      if (record?.enforced && idleRemainingMs(record) <= 0) revalidate(true)
    }
    let lastTick = Date.now()
    const tick = () => {
      const now = Date.now()
      const jumped = now < lastTick || now - lastTick > DEADLINE_TICK_MS + CLOCK_JUMP_TOLERANCE_MS
      lastTick = now
      if (jumped) revalidate(true)
      else checkDeadline()
    }
    const onWake = () => {
      if (document.visibilityState === "hidden") return
      revalidate(true)
    }
    // A closed socket is a server statement about this login, enforced or not.
    const onRevalidate = () => revalidate(false)

    checkDeadline()
    const interval = setInterval(tick, DEADLINE_TICK_MS)
    document.addEventListener("visibilitychange", onWake)
    window.addEventListener("focus", onWake)
    window.addEventListener("online", onWake)
    window.addEventListener(AUTH_REVALIDATE_EVENT, onRevalidate)
    return () => {
      reporter.stop()
      clearInterval(interval)
      document.removeEventListener("visibilitychange", onWake)
      window.removeEventListener("focus", onWake)
      window.removeEventListener("online", onWake)
      window.removeEventListener(AUTH_REVALIDATE_EVENT, onRevalidate)
    }
  }, [sessionId])
  return null
}
