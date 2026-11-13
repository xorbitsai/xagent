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
/** Warning lead time: a UI constant, not policy. Halved thresholds keep very short test policies usable. */
const WARNING_LEAD_MS = 2 * 60 * 1000

function warningSecondsLeft(record: IdleSessionRecord | null): number | null {
  if (!record?.enforced) return null
  const remaining = idleRemainingMs(record)
  const lead = Math.min(WARNING_LEAD_MS, record.idleTimeoutSeconds * 1000 / 2)
  return remaining > 0 && remaining <= lead ? Math.ceil(remaining / 1000) : null
}

function formatCountdown(seconds: number): string {
  return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
}

/**
 * Browser side of the server idle-session policy for one login: reports
 * deliberate user input, and asks the server (verify, which is not activity)
 * when the deadline seems to have passed, after the device wakes, comes
 * back online or regains the user's attention, and when a socket says the
 * session ended. It never signs the user out on its own clock; the server's
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
      isUserEvent: isUserActivityEvent,
      getRecord,
      markReported: () => {
        const current = readAuthSessionSnapshot()
        if (current.sessionId === sessionId) markIdleActivityReportedIfCurrent(current)
      },
      report: () => reportIdleActivity(sessionId),
    })
    reporterRef.current = reporter

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
      setSecondsLeft(warningSecondsLeft(record))
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
