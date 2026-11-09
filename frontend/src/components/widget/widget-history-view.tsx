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
  WidgetSessionHistoryError,
  type WidgetConversationMessage,
  type WidgetSessionHistoryAuth,
  type WidgetSessionHistoryErrorKind,
} from "@/lib/widget-session-history"
import { PHASE_ROUTE } from "./use-session-open-message"

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

export type HistoryPreflightErrorKind =
  | "source_not_found"
  | "history_disabled"
  | "unauthorized"
  | "preflight_failed"

// Why the check before a hand-over failed. The view shows its own line for it,
// so the composer must not raise a second notice.
export class HistoryPreflightError extends Error {
  readonly kind: HistoryPreflightErrorKind
  readonly notificationHandled = true

  constructor(kind: HistoryPreflightErrorKind) {
    super(kind)
    this.name = "HistoryPreflightError"
    this.kind = kind
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
  sessionPhase: SessionConversationPhase
  sessionAttachment: SessionConversationAttachment
  // True while a message from an earlier hand-over is still being sent.
  continueBusy: boolean
  onBack: () => void
  // Called once the source has been checked, with the visitor's message. The
  // page takes it from here and this view goes away.
  onContinue: (message: string, sourceTaskId: number) => void
}

type FailureLine = "notFound" | "disabled" | "refreshToLoad" | "tryLater" | "loadFailed"

const PREFLIGHT_KIND: Record<WidgetSessionHistoryErrorKind, HistoryPreflightErrorKind> = {
  not_found: "source_not_found",
  disabled: "history_disabled",
  unauthorized: "unauthorized",
  load_failed: "preflight_failed",
}

const readErrorKind = (error: unknown): WidgetSessionHistoryErrorKind =>
  error instanceof WidgetSessionHistoryError ? error.kind : "load_failed"

// A source that is gone or switched off cannot be continued, whatever the
// visitor types, so its composer is taken away.
const endsTheView = (line: FailureLine | "startMessageFailed" | null): boolean => line === "notFound" || line === "disabled"

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
  continueBusy,
  onBack,
  onContinue,
}: WidgetHistoryViewProps) {
  const { t } = useI18n()
  const [messages, setMessages] = useState<WidgetConversationMessage[]>([])
  const [nextCursor, setNextCursor] = useState<string | null>(null)
  const [firstPage, setFirstPage] = useState<"loading" | "ready" | "failed">("loading")
  const [morePages, setMorePages] = useState<"idle" | "loading" | "failed">("idle")
  const [readFailure, setReadFailure] = useState<FailureLine>("loadFailed")
  const [sendFailure, setSendFailure] = useState<FailureLine | "startMessageFailed" | null>(null)
  const [checking, setChecking] = useState(false)
  const backRef = useRef<HTMLButtonElement>(null)
  const mountedRef = useRef(false)
  const latestRequestRef = useRef(0)
  const controllerRef = useRef<AbortController | null>(null)
  const sourceRef = useRef({ origin, getAuth, taskId, unauthorizedNotice })

  useEffect(() => {
    sourceRef.current = { origin, getAuth, taskId, unauthorizedNotice }
  }, [origin, getAuth, taskId, unauthorizedNotice])

  const failureFor = useCallback((kind: WidgetSessionHistoryErrorKind): FailureLine => {
    if (kind === "not_found") return "notFound"
    if (kind === "disabled") return "disabled"
    if (kind === "unauthorized") return sourceRef.current.unauthorizedNotice
    return "loadFailed"
  }, [])

  const load = useCallback((cursor: string | null) => {
    controllerRef.current?.abort()
    const controller = new AbortController()
    controllerRef.current = controller
    latestRequestRef.current += 1
    const request = latestRequestRef.current
    if (cursor === null) setFirstPage("loading")
    else setMorePages("loading")
    const source = sourceRef.current
    readWidgetConversation(
      source.origin,
      source.getAuth(),
      source.taskId,
      { cursor },
      controller.signal,
    ).then(
      (page) => {
        if (request !== latestRequestRef.current) return
        setMessages((current) => (cursor === null ? page.messages : [...current, ...page.messages]))
        setNextCursor(page.nextCursor)
        setFirstPage("ready")
        setMorePages("idle")
      },
      (error: unknown) => {
        if (request !== latestRequestRef.current) return
        setReadFailure(failureFor(readErrorKind(error)))
        if (cursor === null) setFirstPage("failed")
        else setMorePages("failed")
      },
    )
  }, [failureFor])

  useEffect(() => {
    mountedRef.current = true
    backRef.current?.focus()
    load(null)
    return () => {
      mountedRef.current = false
      latestRequestRef.current += 1
      controllerRef.current?.abort()
    }
  }, [load])

  const route = PHASE_ROUTE[sessionPhase]
  const blockedNotice = sessionAttachment === "unsupported"
    ? "sendUnavailable"
    : sessionAttachment !== "attached" || continueBusy || (route !== "send" && route !== "reset")
      ? "waitForCurrent"
      : null

  const submit = async (message: string) => {
    setSendFailure(null)
    setChecking(true)
    try {
      const source = sourceRef.current
      await readWidgetConversation(source.origin, source.getAuth(), source.taskId, { limit: 1 })
    } catch (error) {
      if (!mountedRef.current) return
      const kind = readErrorKind(error)
      setSendFailure(kind === "load_failed" ? "startMessageFailed" : failureFor(kind))
      setChecking(false)
      throw new HistoryPreflightError(PREFLIGHT_KIND[kind])
    }
    if (!mountedRef.current) return
    setChecking(false)
    onContinue(message, taskId)
  }

  const composerGone = endsTheView(sendFailure) || (firstPage === "failed" && endsTheView(readFailure))
  // A reload is the only way past "refreshToLoad", so a retry would only fail again.
  const retry = readFailure === "refreshToLoad" ? null : (
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
          </div>
        ) : firstPage === "failed" ? (
          <div className="text-sm">
            <p className="text-destructive" role="alert">{t(`widgetSession.history.${readFailure}`)}</p>
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
                <p className="text-destructive" role="alert">{t(`widgetSession.history.${readFailure}`)}</p>
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
            {sendFailure === "startMessageFailed"
              ? t("widgetSession.startMessageFailed")
              : t(`widgetSession.history.${sendFailure}`)}
          </p>
        ) : null}
        {composerGone ? null : (
          <ChatInput
            onSend={(message) => submit(message)}
            isLoading={blockedNotice !== null || checking}
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
