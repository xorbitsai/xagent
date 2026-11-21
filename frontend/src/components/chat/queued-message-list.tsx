"use client"

import React, { useId, useState } from "react"
import { ChevronDown, ChevronRight, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { SessionMessageQueueControls } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { isAwaitingAnswer, type QueuedMessage } from "@/contexts/session-message-queue"
import { cn } from "@/lib/utils"

const HELD_KEYS = {
  turn_failed: "messageQueue.held.turnFailed",
  turn_paused: "messageQueue.held.turnInterrupted",
  send_failed: "messageQueue.held.sendFailed",
  earlier_send_failed: "messageQueue.held.earlierSendFailed",
  send_now_unknown: "messageQueue.held.sendNowUnknown",
} as const

const NOTICE_CLASS =
  "rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:bg-amber-950 dark:text-amber-100"

function QueuedRow({ item, onRemove }: { item: QueuedMessage; onRemove: () => void }) {
  const { t } = useI18n()
  const [expanded, setExpanded] = useState(false)
  const textId = useId()
  const Chevron = expanded ? ChevronDown : ChevronRight
  return (
    <li className="flex items-start gap-2 text-sm">
      <button
        type="button"
        className="mt-0.5 shrink-0 text-muted-foreground"
        aria-expanded={expanded}
        aria-describedby={textId}
        aria-label={t(expanded ? "messageQueue.collapse" : "messageQueue.expand")}
        onClick={() => setExpanded(value => !value)}
      >
        <Chevron className="h-4 w-4" />
      </button>
      {/* Visitor text is shown as plain text, never parsed. */}
      <span id={textId} className={cn("min-w-0 flex-1", expanded ? "whitespace-pre-wrap break-words" : "truncate")}>
        {item.text}
      </span>
      <span
        aria-live="polite"
        className={cn("shrink-0 text-xs", item.status === "failed" ? "text-destructive" : "text-muted-foreground")}
      >
        {t(item.status === "waiting" && item.next ? "messageQueue.status.next" : `messageQueue.status.${item.status}`)}
      </span>
      <button
        type="button"
        className="shrink-0 text-muted-foreground disabled:opacity-50"
        aria-label={t("messageQueue.remove")}
        aria-describedby={textId}
        disabled={item.status === "sending"}
        onClick={onRemove}
      >
        <X className="h-4 w-4" />
      </button>
    </li>
  )
}

// The messages held for the turn in progress, with what the visitor can do
// about them. `canContinue` is false while the queue cannot send.
export function QueuedMessageList({
  queue,
  canContinue,
}: {
  queue: SessionMessageQueueControls
  canContinue: boolean
}) {
  const { t } = useI18n()
  const unavailableId = useId()
  const { state } = queue
  if (state.items.length === 0) return null
  return (
    <div className="mb-2 space-y-2">
      {state.phase.kind === "held" && (
        <div className={cn("space-y-2", NOTICE_CLASS)}>
          <p role="status">{t(HELD_KEYS[state.phase.reason])}</p>
          {!canContinue && <p id={unavailableId}>{t("messageQueue.continueUnavailable")}</p>}
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              size="sm"
              variant="outline"
              disabled={!canContinue}
              aria-describedby={canContinue ? undefined : unavailableId}
              onClick={queue.continueSending}
            >
              {t("messageQueue.continue")}
            </Button>
            <Button type="button" size="sm" variant="outline" onClick={queue.clear}>
              {t("messageQueue.clear")}
            </Button>
          </div>
        </div>
      )}
      {isAwaitingAnswer(state) && (
        <div role="status" className={NOTICE_CLASS}>{t("messageQueue.waitingAnswer")}</div>
      )}
      <ul aria-label={t("messageQueue.listLabel")} className="max-h-40 space-y-1 overflow-y-auto rounded-md border bg-card/60 px-3 py-2">
        {state.items.map(item => (
          <QueuedRow
            key={item.seq}
            item={item}
            onRemove={() => queue.remove(item.clientMessageId)}
          />
        ))}
      </ul>
    </div>
  )
}
