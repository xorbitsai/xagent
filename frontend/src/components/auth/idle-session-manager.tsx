"use client"

import React, { useEffect, useRef, useState } from "react"

import {
  AlertDialog, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { useI18n } from "@/contexts/i18n-context"

import { markIdleActivityReportedIfCurrent, readAuthSessionSnapshot, readCurrentIdleSession } from "@/lib/auth-cache"
import { isUserActivityEvent, startIdleActivityReporter, type IdleActivityReporter } from "@/lib/idle-activity"
import { AUTH_REVALIDATE_EVENT, idleRemainingMs, type IdleSessionRecord } from "@/lib/idle-session"
import { reportIdleActivity, revalidateIdleSession } from "@/lib/idle-session-client"

/** How often the deadline is checked; also the gap that reveals a sleeping device or a clock change. */
const DEADLINE_TICK_MS = 1_000
const CLOCK_JUMP_TOLERANCE_MS = 30_000
/** Minimum gap between two server checks, so a server outage does not turn into a request loop. */
const REVALIDATE_MIN_GAP_MS = 15_000
/**
 * The server rounds idle_expires_in down, so the local deadline can be up to a
 * second before the server's. Waiting this long past it means the first check
 * lands after the server deadline instead of being answered "still valid" and
 * then held back by the gap above.
 */
const DEADLINE_GRACE_MS = 1_500
/** Warning lead time: a UI constant, not policy. Halved thresholds keep very short test policies usable. */
const WARNING_LEAD_MS = 2 * 60 * 1000

function warningLeadMs(record: IdleSessionRecord): number {
  return Math.min(WARNING_LEAD_MS, record.idleTimeoutSeconds * 1000 / 2)
}

function warningSecondsLeft(record: IdleSessionRecord | null): number | null {
  if (!record?.enforced) return null
  const remaining = idleRemainingMs(record)
  // At the deadline the warning stays at 0:00 until the server answers: the
  // check runs a little later, and if it fails (offline, 5xx) the page must
  // not look normal while the login has most likely expired.
  return remaining <= warningLeadMs(record) ? Math.max(0, Math.ceil(remaining / 1000)) : null
}

function formatCountdown(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
}

/**
 * Browser side of the server idle-session policy for one login: reports
 * deliberate user input, and asks the server (verify, which is not activity)
 * when the deadline seems to have passed, when the device wakes, comes
 * back online or regains the user's attention near the deadline, and when a
 * socket says the session ended. It never signs the user out on its own clock; the server's
 * answer does. Shows an accessible warning before the deadline. Renders
 * nothing for servers without idle support.
 */
export function IdleSessionManager({ sessionId, onSignOut }: { sessionId: string | null; onSignOut: () => unknown }) {
  const { t } = useI18n()
  const [secondsLeft, setSecondsLeft] = useState<number | null>(null)
  const reporterRef = useRef<IdleActivityReporter | null>(null)

  useEffect(() => {
    setSecondsLeft(null)
    if (!sessionId) return
    const getRecord = () => {
      const record = readCurrentIdleSession()
      return record?.sessionId === sessionId ? record : null
    }
    const reporter = startIdleActivityReporter({
      target: window,
      // Explicit on purpose: tests replace this module export to treat jsdom's
      // untrusted events as user input.
      isUserEvent: isUserActivityEvent,
      getRecord,
      markReported: async () => {
        const current = readAuthSessionSnapshot()
        if (current.sessionId === sessionId) await markIdleActivityReportedIfCurrent(current)
      },
      report: isStillFresh => reportIdleActivity(sessionId, isStillFresh),
    })
    reporterRef.current = reporter

    let inFlight = false
    let lastCheckAt = Number.NEGATIVE_INFINITY
    // A socket notice that arrived while a check was running or too soon
    // after one; the next tick checks once the gap allows it.
    let socketNoticePending = false
    // The clock moved backwards while the tab was hidden: the local deadline
    // now looks further away than it is, so the next wake asks the server
    // even far from the deadline.
    let rolledBackWhileHidden = false
    const revalidate = (requireEnforced: boolean) => {
      const record = getRecord()
      if (!record || (requireEnforced && !record.enforced)) return
      // A negative gap means the clock moved backwards: the gap no longer
      // says anything, so a fresh server answer is allowed.
      const sinceLastCheck = Date.now() - lastCheckAt
      if (inFlight || (sinceLastCheck >= 0 && sinceLastCheck < REVALIDATE_MIN_GAP_MS)) {
        if (!requireEnforced) socketNoticePending = true
        return
      }
      socketNoticePending = false
      rolledBackWhileHidden = false
      inFlight = true
      lastCheckAt = Date.now()
      void revalidateIdleSession(sessionId).finally(() => { inFlight = false })
    }
    const checkDeadline = () => {
      const record = getRecord()
      setSecondsLeft(warningSecondsLeft(record))
      if (record?.enforced && idleRemainingMs(record) <= -DEADLINE_GRACE_MS) revalidate(true)
    }
    let lastTick = Date.now()
    const tick = () => {
      const now = Date.now()
      const rolledBack = now < lastTick
      const jumped = rolledBack || now - lastTick > DEADLINE_TICK_MS + CLOCK_JUMP_TOLERANCE_MS
      lastTick = now
      // Hidden tabs have throttled timers, so a long forward gap there is
      // expected and already counted by the wall-clock countdown. A backward
      // jump is remembered for the next wake.
      const hidden = document.visibilityState === "hidden"
      if (jumped && hidden && rolledBack) rolledBackWhileHidden = true
      if (jumped && !hidden) revalidate(true)
      else checkDeadline()
      if (socketNoticePending) revalidate(false)
    }
    const onWake = () => {
      if (document.visibilityState === "hidden") return
      // The countdown uses wall-clock time, so it already counts a sleep. Far
      // from the deadline (an ordinary user's one-year threshold, say) the
      // server has nothing new to say, so only ask near or past it.
      const record = getRecord()
      if (!record) return
      if (!rolledBackWhileHidden && idleRemainingMs(record) > warningLeadMs(record)) return
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
      reporterRef.current = null
      clearInterval(interval)
      document.removeEventListener("visibilitychange", onWake)
      window.removeEventListener("focus", onWake)
      window.removeEventListener("online", onWake)
      window.removeEventListener(AUTH_REVALIDATE_EVENT, onRevalidate)
    }
  }, [sessionId])

  if (secondsLeft === null) return null
  const continueSession = () => {
    void reporterRef.current?.reportNow()
  }
  return (
    <AlertDialog open>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("session_idle.warning_title")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("session_idle.warning_description", { time: formatCountdown(secondsLeft) })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <Button variant="outline" onClick={() => { void onSignOut() }}>{t("session_idle.sign_out")}</Button>
          <Button autoFocus onClick={continueSession}>{t("session_idle.continue")}</Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
