"use client"

import React, { useCallback, useEffect, useRef, useState } from "react"
import { ArrowLeft, Loader2 } from "lucide-react"
import { ChatInput } from "@/components/chat/ChatInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { UserMessageContent } from "@/components/chat/user-message-content"
import type { SessionConversationAttachment, useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import {
  readWidgetConversation,
  type WidgetConversationMessage,
  type WidgetSessionHistoryAuth,
  type WidgetSessionHistoryErrorKind,
} from "@/lib/widget-session-history"
import { PHASE_ROUTE } from "./use-session-open-message"
import { readErrorKind, usePagedHistory } from "./use-paged-history"

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

// The check before a hand-over failed. The view shows its own line for it, so
// the composer must not raise a second notice.
export class HistoryPreflightError extends Error {
  readonly notificationHandled = true

  constructor(kind: WidgetSessionHistoryErrorKind) {
    super(kind)
    this.name = "HistoryPreflightError"
  }
}

export interface WidgetHistoryViewProps {
  origin: string
  // Called for every request, so a request always uses the token policy of the
  // conversation phase the session is in at that moment.
  getAuth: () => WidgetSessionHistoryAuth
  // What to say when a request is rejected as unauthorized and may not be
  // refreshed in the current phase.
  unauthorizedNotice: "refreshToLoad" | "tryLater"
  taskId: number
  title: string
  // The page closes this view while a hand-over is in flight, so neither of
  // these needs to account for one.
  sessionPhase: SessionConversationPhase
  sessionAttachment: SessionConversationAttachment
  onBack: () => void
  // Called once the source has been checked, with the visitor's message. The
  // page takes it from here and this view goes away.
  onContinue: (message: string, sourceTaskId: number) => void
}

// A source that is gone or switched off cannot be continued, whatever the
// visitor types, so its composer is taken away.
const endsTheView = (kind: WidgetSessionHistoryErrorKind | null): boolean => kind === "not_found" || kind === "disabled"

const messageContent = (message: WidgetConversationMessage): React.ReactNode => {
  if (message.role !== "user" || message.attachments.length === 0) return message.content
  // File names only: no id reaches the DOM and a file cannot be opened from
  // here.
  return (
    <UserMessageContent
      message={message.content}
      files={message.attachments.map((attachment) => ({
        name: attachment.filename,
        type: attachment.mimeType ?? undefined,
        size: attachment.fileSize ?? undefined,
      }))}
    />
  )
}

// Reads one earlier conversation. It never touches the conversation the
// visitor is in: its only effect is the check before a hand-over, which reads
// one message and changes nothing.
export function WidgetHistoryView({
  origin,
  getAuth,
  unauthorizedNotice,
  taskId,
  title,
  sessionPhase,
  sessionAttachment,
  onBack,
  onContinue,
}: WidgetHistoryViewProps) {
  const { t } = useI18n()
  const [sendFailure, setSendFailure] = useState<WidgetSessionHistoryErrorKind | null>(null)
  const [checking, setChecking] = useState(false)
  const backRef = useRef<HTMLButtonElement>(null)
  const mountedRef = useRef(false)
  // The check before a hand-over is cancelled when the view closes, so a
  // late rejection cannot go on to refresh the token with the auth of the
  // phase the check started in.
  const checkControllerRef = useRef<AbortController | null>(null)
  const sourceRef = useRef({ origin, getAuth, taskId })

  useEffect(() => {
    sourceRef.current = { origin, getAuth, taskId }
  }, [origin, getAuth, taskId])

  const fetchPage = useCallback((cursor: string | null, signal: AbortSignal) => {
    const source = sourceRef.current
    return readWidgetConversation(source.origin, source.getAuth(), source.taskId, { cursor }, signal).then(
      (page) => ({ items: page.messages, nextCursor: page.nextCursor }),
    )
  }, [])
  const {
    items: messages,
    nextCursor,
    firstPage,
    morePages,
    errorKind: readFailure,
    load,
  } = usePagedHistory(fetchPage)

  useEffect(() => {
    mountedRef.current = true
    backRef.current?.focus()
    return () => {
      mountedRef.current = false
      checkControllerRef.current?.abort()
    }
  }, [])

  const failureLine = (kind: WidgetSessionHistoryErrorKind): string => (
    kind === "not_found"
      ? t("widgetSession.history.notFound")
      : kind === "disabled"
        ? t("widgetSession.history.disabled")
        : kind === "unauthorized"
          ? t(`widgetSession.history.${unauthorizedNotice}`)
          : t("widgetSession.history.loadFailed")
  )
  // A reload is the only way past "refreshToLoad": a retry or a check before a
  // hand-over would only be rejected again.
  const reloadOnly = (kind: WidgetSessionHistoryErrorKind | null): boolean => (
    kind === "unauthorized" && unauthorizedNotice === "refreshToLoad"
  )

  const route = PHASE_ROUTE[sessionPhase]
  const blockedNotice = sessionAttachment === "unsupported"
    ? "sendUnavailable"
    : sessionAttachment !== "attached" || (route !== "send" && route !== "reset")
      ? "waitForCurrent"
      : null
  const readFailed = firstPage === "failed" || morePages === "failed"
  const sendLocked = reloadOnly(sendFailure) || (readFailed && reloadOnly(readFailure))

  const submit = async (message: string) => {
    setSendFailure(null)
    setChecking(true)
    const controller = new AbortController()
    checkControllerRef.current = controller
    try {
      const source = sourceRef.current
      await readWidgetConversation(source.origin, source.getAuth(), source.taskId, { limit: 1 }, controller.signal)
    } catch (error) {
      if (!mountedRef.current) return
      const kind = readErrorKind(error)
      setSendFailure(kind)
      setChecking(false)
      throw new HistoryPreflightError(kind)
    }
    if (!mountedRef.current) return
    setChecking(false)
    onContinue(message, taskId)
  }

  const composerGone = endsTheView(sendFailure) || (firstPage === "failed" && endsTheView(readFailure))
  const retry = reloadOnly(readFailure) ? null : (
    <button
      type="button"
      onClick={() => load(firstPage === "failed" ? null : nextCursor)}
      className="mt-2 rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {t("widgetSession.history.retry")}
    </button>
  )

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label={title}>
      <div className="flex flex-none items-center gap-2 border-b px-3 py-2">
        <button
          ref={backRef}
          type="button"
          onClick={onBack}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-sm text-muted-foreground hover:bg-muted/50 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          <ArrowLeft className="h-4 w-4" aria-hidden="true" />
          {t("widgetSession.history.back")}
        </button>
        <h2 className="truncate text-sm font-semibold">{title}</h2>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {firstPage === "loading" ? (
          <div className="flex items-center gap-2 text-sm text-muted-foreground" role="status">
            <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" />
            <span className="sr-only">{t("common.loading")}</span>
          </div>
        ) : firstPage === "failed" ? (
          <div className="text-sm">
            <p className="text-destructive" role="alert">{failureLine(readFailure)}</p>
            {endsTheView(readFailure) ? null : retry}
          </div>
        ) : messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t("widgetSession.history.viewEmpty")}</p>
        ) : (
          <>
            <div className="flex flex-col gap-4">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  role={message.role}
                  content={messageContent(message)}
                  rawContent={message.role === "assistant" ? message.content : undefined}
                  showProcessView={false}
                  timestamp={message.createdAt}
                />
              ))}
            </div>
            {morePages === "failed" ? (
              <div className="mt-3 text-sm">
                <p className="text-destructive" role="alert">{failureLine(readFailure)}</p>
                {retry}
              </div>
            ) : nextCursor !== null ? (
              <button
                type="button"
                disabled={morePages === "loading"}
                onClick={() => load(nextCursor)}
                className="mt-3 rounded-md border px-3 py-1.5 text-sm hover:bg-muted/50 disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {t("widgetSession.history.loadMore")}
              </button>
            ) : null}
          </>
        )}
      </div>
      <div className="flex-none border-t p-3">
        {sendFailure ? (
          <p className="mb-2 text-xs text-destructive" role="alert">
            {sendFailure === "load_failed"
              ? t("widgetSession.startMessageFailed")
              : failureLine(sendFailure)}
          </p>
        ) : null}
        {composerGone ? null : (
          <ChatInput
            onSend={(message) => submit(message)}
            isLoading={blockedNotice !== null || checking || sendLocked}
            hideFileUpload={true}
            filesDisabled={true}
            hideConfig={true}
            readOnlyConfig={true}
            compact={true}
            voiceInputEnabled={false}
          />
        )}
        {blockedNotice ? (
          <p className="mt-2 text-xs text-muted-foreground" role="status">
            {t(`widgetSession.history.${blockedNotice}`)}
          </p>
        ) : null}
      </div>
    </section>
  )
}
