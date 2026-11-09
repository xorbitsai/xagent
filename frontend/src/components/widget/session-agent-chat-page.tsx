"use client"

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Loader2 } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { WidgetChromeControls } from "@/components/widget/widget-chrome-controls"
import {
  AppProvider,
  type AppProviderTransportConfig,
  type SessionAttachmentsConfig,
  useApp,
} from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { useWidgetInitialMessage } from "@/hooks/use-widget-initial-message"
import { useWidgetOpenMessage } from "@/hooks/use-widget-open-message"
import type { WebSocketConnection } from "@/hooks/use-websocket"
import { readSendErrorCode } from "@/components/chat/clarification-delivery"
import {
  clientErrorTranslationKey,
  type ClientErrorCode,
} from "@/lib/client-errors"
import type { WidgetSessionHistoryAuth } from "@/lib/widget-session-history"
import {
  toAttachments,
  useSessionAttachments,
  type SessionUploadControls,
} from "./use-session-attachments"
import {
  buildWidgetSessionWebSocketUrl,
  useWidgetSession,
  type WidgetSessionAgent,
  type WidgetSessionStatus,
} from "./use-widget-session"
import { useHistoryContinue, type HistoryContinueRequest } from "./use-history-continue"
import { useSessionOpenMessage } from "./use-session-open-message"
import { WidgetHistoryList } from "./widget-history-list"
import { WidgetHistoryView } from "./widget-history-view"

const SESSION_PROTOCOL = "xagent-session-v1"
// Asks the session server to send a connection_capabilities frame, then
// heartbeat frames. This page requires a server that honors it. A server that
// ignores the marker sends no capabilities frame: the connection runs without
// dead-connection detection, and sessionConversationAttachment stays
// "pending", so neither data-initial-message (see useWidgetInitialMessage
// below) nor a message from XagentWidget.open() is ever sent.
const SESSION_LIVENESS_PROTOCOL = "xagent-session-liveness-v1"
// Asks the session server to announce when attach, including any resume replay, has finished.
const SESSION_ATTACH_PROTOCOL = "xagent-session-attach-v1"
type SessionTerminalPresentation = "expired" | "unavailable"

const SESSION_TERMINAL_PRESENTATION: Readonly<Record<string, SessionTerminalPresentation>> = {
  session_expired: "expired",
  grant_expired: "expired",
  grant_already_used: "expired",
  reconnect_invalid: "unavailable",
  identity_mismatch: "unavailable",
  ws_4408: "expired",
  ws_4403: "unavailable",
  unexpected_error: "unavailable",
}

const getSessionTerminalPresentation = (code: string | null): SessionTerminalPresentation => (
  code ? SESSION_TERMINAL_PRESENTATION[code] ?? "unavailable" : "unavailable"
)

// What the area below the header shows. The chat stays mounted underneath the
// other two, so a half-typed message is still there on the way back.
type HistoryViewState =
  | { kind: "chat" }
  | { kind: "history_list" }
  | { kind: "history_view"; taskId: number; title: string }

const isNotificationHandled = (error: unknown): boolean =>
  typeof error === "object" && error !== null
  && (error as { notificationHandled?: unknown }).notificationHandled === true

interface SessionConversationContentProps {
  status: WidgetSessionStatus
  agent: WidgetSessionAgent | null
  terminalCode: string | null
  isAbsoluteExpiryWarningVisible: boolean
  // Lifted to the never-unmounting SessionAgentChatPage below: this
  // component's own "active" JSX branch unmounts on a degraded/reconnect
  // transition (see the status check further down), which would otherwise
  // silently reset an accepted expand back to the collapsed label/icon while
  // the panel itself is still actually expanded.
  isExpanded: boolean
  onExpandedChange: (expanded: boolean) => void
  // Present only while the session server advertises file uploads.
  attachments?: SessionAttachmentsConfig
  uploads: SessionUploadControls
  // Whether the session server offers earlier conversations to this session.
  historyAvailable: boolean
  // The token of the session the page holds right now, or null while it has none.
  readCurrentToken: () => string | null
}

function SessionConversationContent({
  status,
  agent,
  terminalCode,
  isAbsoluteExpiryWarningVisible,
  isExpanded,
  onExpandedChange,
  attachments,
  uploads,
  historyAvailable,
  readCurrentToken,
}: SessionConversationContentProps) {
  const {
    state,
    filesDisabled,
    voiceInputEnabled,
    isConnected,
    sendMessage,
    startNewConversation,
    isConversationResetPending,
    isMessageDeliveryPending,
    isSessionInteractionLocked,
    sessionConversationState,
    sessionConversationAttachment,
  } = useApp()
  const { t } = useI18n()
  const conversationEstablished = state.taskId !== null
  const {
    startFiles,
    onStartFilesChange,
    sendStartMessage,
    panelFiles,
    onPanelFilesChange,
    sendWithAttachments,
    statuses: attachmentStatuses,
    isBusy: isAttachmentBusy,
    retryAttachment,
  } = useSessionAttachments({
    attachments,
    uploads,
    status,
    conversationEstablished,
    sendMessage,
    sessionConversationState,
    isSessionInteractionLocked,
    isConversationResetPending,
  })
  const [resetError, setResetError] = useState(false)
  // The failure is stored as a code, not as text, so the line is translated
  // at render time and follows a locale switch. It is only a note on the
  // start page; the toast raised by the composer is the failure signal.
  // `continued` marks a failure of a send that carried an earlier conversation
  // as its source.
  const [startMessageFailure, setStartMessageFailure] =
    useState<{ code: ClientErrorCode | null; continued?: true } | null>(null)
  const [view, setView] = useState<HistoryViewState>({ kind: "chat" })
  // The message a visitor wrote over an earlier conversation, waiting for the
  // hand-over driver below to send it. A newer one replaces an older one
  // still here.
  const [continueRequest, setContinueRequest] = useState<HistoryContinueRequest | null>(null)
  const continueSeqRef = useRef(0)
  const [continueNotice, setContinueNotice] = useState<"not_sent" | null>(null)

  // The failure line belongs to the current start-page visit only: whenever
  // the conversation becomes established or returns to the start page, the
  // stored failure is dropped, so a failure that arrived while a conversation
  // was established never shows on a later start page. The drop happens while
  // rendering rather than in an effect: React re-runs this component with the
  // cleared failure before committing, so the stale line under role="alert"
  // never reaches the DOM, not even for a single commit.
  const [lastSeenConversationEstablished, setLastSeenConversationEstablished] =
    useState(conversationEstablished)
  if (lastSeenConversationEstablished !== conversationEstablished) {
    setLastSeenConversationEstablished(conversationEstablished)
    setStartMessageFailure(null)
    setContinueNotice(null)
  }

  const handleStartMessage = useCallback(async (
    message: string,
    files: File[],
    config?: Record<string, unknown>,
  ) => {
    setStartMessageFailure(null)
    try {
      await sendStartMessage(message, files, config)
    } catch (error) {
      // An error that is already announced, or that the visitor caused, leaves
      // no failure line behind.
      if (!isNotificationHandled(error)) {
        setStartMessageFailure({
          code: readSendErrorCode(error),
          ...(typeof config?.sourceTaskId === "number" ? { continued: true as const } : {}),
        })
      }
      throw error
    }
  }, [sendStartMessage])

  const handleStartNewConversation = useCallback(async () => {
    setResetError(false)
    try {
      await startNewConversation()
    } catch {
      setResetError(true)
    }
  }, [startNewConversation])

  // The same failure line as the button above, and the failure is passed on
  // so the open message driver drops its message.
  const startNewConversationForOpen = useCallback(async () => {
    setResetError(false)
    try {
      await startNewConversation()
    } catch (error) {
      setResetError(true)
      throw error
    }
  }, [startNewConversation])

  const sendContinueMessage = useCallback(
    (message: string, sourceTaskId: number) => handleStartMessage(message, [], { sourceTaskId }),
    [handleStartMessage],
  )
  const takeContinue = useCallback((seq: number) => {
    setContinueRequest((current) => (current?.seq === seq ? null : current))
  }, [])
  const handleContinue = useCallback((message: string, sourceTaskId: number) => {
    continueSeqRef.current += 1
    setView({ kind: "chat" })
    setContinueNotice(null)
    setContinueRequest({ seq: continueSeqRef.current, message, sourceTaskId })
  }, [])

  // The hooks below run in this order. The host page's open message waits for
  // a hand-over (it is given `continueBusy`), the hand-over never waits for it,
  // and the host page's initial message is spent by a hand-over; all three
  // read `continueBusy` during render, so which effect runs first decides
  // nothing. What this guarantees is that no message enters the wrong
  // conversation, not that none is lost: if the host page's initial message
  // arrives in the very render that places a hand-over, both frames go out and
  // the session server refuses the second one. A refused hand-over always
  // tells the visitor: on the start page through the failure line there, and
  // once a conversation is open, where that line is not shown, through the
  // "not sent" line.
  const openMessage = useWidgetOpenMessage({ enabled: true })
  const { busy: continueBusy } = useHistoryContinue({
    request: continueRequest,
    take: takeContinue,
    status,
    hasAgent: agent !== null,
    sessionConversationAttachment,
    sessionConversationState,
    isMessageDeliveryPending,
    isConversationResetPending,
    isAttachmentBusy,
    startNewConversation: startNewConversationForOpen,
    sendContinueMessage,
    onOutcome: (outcome) => {
      if (
        outcome === "superseded"
        || outcome === "unsupported"
        || (outcome === "send_failed" && conversationEstablished)
      ) {
        setContinueNotice("not_sent")
      } else if (outcome === "sent") {
        setContinueNotice(null)
      }
    },
  })

  useWidgetInitialMessage({
    // An open socket does not yet tell whether the server is restoring a
    // conversation, because its replay arrives after the open. Attach
    // completion follows that replay, so the page knows from there on. A
    // server whose connection_capabilities frame does not list attach
    // completion ("unsupported") leaves nothing more to wait for, so the page
    // asks once that frame arrives. The page requires a server that sends
    // connection_capabilities: without it the value stays "pending" and the
    // message is never requested. No timer stands in for either frame: an
    // attach-capable server whose frames are slow would then be taken for
    // "unsupported", and the page could ask before a restored conversation
    // arrives.
    // Both "attached" and "unsupported" describe the current open socket only.
    ready: status === "active" && agent !== null
      && sessionConversationAttachment !== "pending",
    // A first send queued during the initial connect has no task yet. A
    // message being handed over from an earlier conversation counts too: the
    // visitor has already started their own conversation, so the host page's
    // message is dropped for good, even if the hand-over then fails.
    hasConversation: state.taskId !== null || isMessageDeliveryPending || continueBusy,
    onMessage: (message) => {
      setView({ kind: "chat" })
      return handleStartMessage(message, [])
    },
  })

  const sendStartMessageForOpen = useCallback(
    (message: string) => handleStartMessage(message, []),
    [handleStartMessage],
  )

  useSessionOpenMessage({
    request: openMessage.request,
    take: openMessage.take,
    status,
    hasAgent: agent !== null,
    sessionConversationAttachment,
    sessionConversationState,
    isMessageDeliveryPending: isMessageDeliveryPending || continueBusy,
    isConversationResetPending,
    isAttachmentBusy,
    startNewConversation: startNewConversationForOpen,
    sendStartMessage: sendStartMessageForOpen,
  })

  // History reads use the token policy of the phase the conversation is in at
  // the moment of each request: before a conversation is replaced, the token
  // is refreshed when needed and retried once after a rejection; while one is
  // being replaced, a refresh would rebuild the connection and cost the
  // visitor the new conversation, so the current token is used as it is.
  const phaseRef = useRef(sessionConversationState)
  useEffect(() => {
    phaseRef.current = sessionConversationState
  }, [sessionConversationState])
  const getHistoryAuth = useCallback((): WidgetSessionHistoryAuth => {
    const phase = phaseRef.current
    if (phase === "unbound" || phase === "bound") {
      return {
        getToken: () => uploads.ensureFreshSession().then((session) => session.token),
        refreshToken: (rejectedToken) => (
          uploads.refreshSession(rejectedToken).then((session) => session.token)
        ),
      }
    }
    return {
      getToken: () => {
        const token = readCurrentToken()
        return token === null
          ? Promise.reject(new Error("The session has no token"))
          : Promise.resolve(token)
      },
    }
  }, [uploads, readCurrentToken])

  // Anything under way that sends or resets, including a file upload for a
  // message, or a host page message still waiting, takes the visitor back to
  // the chat and keeps the history entry away. A reload-required session does
  // the same: its notice only exists on the chat. The render in which an open
  // message has been taken and its reset confirmed, but its send has not
  // started, is the one render in which none of these is set; the send starts
  // in that render's effect and the next render closes whatever was opened
  // meanwhile.
  const historyBlocked = openMessage.request !== null
    || isConversationResetPending
    || isMessageDeliveryPending
    || isAttachmentBusy
    || continueBusy
  if (view.kind !== "chat" && (isSessionInteractionLocked || historyBlocked)) {
    setView({ kind: "chat" })
  }

  if (status === "waiting" || (status === "refreshing" && !agent)) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="flex items-center gap-2 text-sm text-muted-foreground">
          <Loader2 className="h-5 w-5 animate-spin" />
          {status === "waiting"
            ? t("widgetChat.status.initializing")
            : t("widgetChat.status.connecting")}
        </div>
      </div>
    )
  }

  if (status === "degraded" || status === "terminal" || !agent) {
    const isExpired = status !== "degraded"
      && getSessionTerminalPresentation(terminalCode) === "expired"
    return (
      <div className="flex h-screen items-center justify-center bg-background p-6">
        <div className="max-w-md text-center">
          <h1 className="text-lg font-semibold">
            {isExpired
              ? t("widgetSession.expired.title")
              : t("widgetSession.unavailable.title")}
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            {isExpired
              ? t("widgetSession.expired.description")
              : t("widgetSession.unavailable.description")}
          </p>
        </div>
      </div>
    )
  }

  const reloadRequired = isSessionInteractionLocked
  const showHistoryEntry = historyAvailable && !reloadRequired && !historyBlocked
  const resetDisabled =
    reloadRequired
    || isConversationResetPending
    || isMessageDeliveryPending
    || isAttachmentBusy

  return (
    <div className="flex h-screen flex-col bg-background">
      <header className="flex-none border-b bg-card p-4 text-card-foreground shadow-sm">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center overflow-hidden rounded-full bg-primary/10 text-primary">
            {agent.logoUrl ? (
              // Session metadata may name an arbitrary remote logo host.
              // eslint-disable-next-line @next/next/no-img-element
              <img
                src={agent.logoUrl}
                alt={agent.name}
                className="h-full w-full object-cover"
              />
            ) : (
              <div className="h-5 w-5 rounded-full bg-primary/20" />
            )}
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-sm font-semibold">
              {agent.name}
            </h1>
            <p className="text-xs text-muted-foreground">
              {status === "active" && isConnected
                ? t("widgetChat.status.online")
                : t("widgetChat.status.connecting")}
            </p>
          </div>
          <WidgetChromeControls
            newConversation={conversationEstablished ? {
              label: isConversationResetPending
                ? t("widgetSession.resetting")
                : t("widgetSession.startNewConversation"),
              onClick: () => {
                setView({ kind: "chat" })
                void handleStartNewConversation()
              },
              disabled: resetDisabled,
              pending: isConversationResetPending,
            } : undefined}
            history={showHistoryEntry ? {
              label: t("widgetSession.history.menuLabel"),
              onClick: () => setView({ kind: "history_list" }),
            } : undefined}
            expanded={isExpanded}
            onExpandedChange={onExpandedChange}
          />
        </div>
        {isAbsoluteExpiryWarningVisible ? (
          <p
            className="mt-3 rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-800 dark:bg-amber-950/30 dark:text-amber-200"
            role="status"
          >
            {t("widgetSession.expiryWarning")}
          </p>
        ) : null}
        {resetError && !reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.resetFailed")}
          </p>
        ) : null}
        {startMessageFailure && !reloadRequired && !conversationEstablished ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {startMessageFailure.continued && startMessageFailure.code === "task_unavailable"
              ? t("widgetSession.history.continueUnavailable")
              : startMessageFailure.code
                ? t(clientErrorTranslationKey(startMessageFailure.code))
                : t("widgetSession.startMessageFailed")}
          </p>
        ) : null}
        {continueNotice && !reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.history.continueNotSent")}
          </p>
        ) : null}
        {reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.reloadRequired")}
          </p>
        ) : null}
      </header>

      <div className="relative min-h-0 flex-1">
        {/* The chat stays mounted while history is open, only hidden: a message
            being typed, a reply still streaming and the scroll position all
            carry on underneath. visibility keeps the layout and takes the
            hidden part out of focus and out of the accessibility tree. */}
        <div
          className="h-full"
          style={view.kind === "chat" ? undefined : { visibility: "hidden" }}
          aria-hidden={view.kind === "chat" ? undefined : true}
        >
          {conversationEstablished ? (
            <TaskConversationPanel
              mode="page"
              showTaskActions={false}
              showTokenUsage={false}
              showDagPreview={false}
              showTaskFiles={false}
              hideFileUpload={!attachments}
              hideConfig={true}
              compactInput={true}
              deferFileUpload={true}
              autoFocusInput={view.kind === "chat"}
              onSend={attachments ? sendWithAttachments : undefined}
              attachments={attachments}
              attachmentStatuses={attachments ? attachmentStatuses : undefined}
              onRetryAttachment={attachments ? retryAttachment : undefined}
              attachmentsBusy={isAttachmentBusy}
              attachmentFiles={attachments ? panelFiles : []}
              onAttachmentFilesChange={attachments ? onPanelFilesChange : undefined}
            />
          ) : (
            <div className="h-full overflow-y-auto">
              <main className="container mx-auto max-w-4xl px-4 py-8">
                <ChatStartScreen
                  title={agent.name}
                  description={agent.description}
                  prompts={agent.suggestedPrompts}
                  onSend={handleStartMessage}
                  sendPromptOnClick={true}
                  isSending={
                    state.isProcessing
                    || isConversationResetPending
                    || isMessageDeliveryPending
                    || reloadRequired
                    || isAttachmentBusy
                    || continueBusy
                  }
                  files={attachments ? startFiles : []}
                  onFilesChange={attachments ? onStartFilesChange : () => undefined}
                  attachments={attachments}
                  attachmentStatuses={attachments ? attachmentStatuses : undefined}
                  onRetryAttachment={attachments ? retryAttachment : undefined}
                  attachmentsBusy={isAttachmentBusy}
                  readOnlyConfig={true}
                  hideConfig={true}
                  compactInput={true}
                  deferFileUpload={true}
                  filesDisabled={filesDisabled}
                  voiceInputEnabled={voiceInputEnabled}
                  autoFocus={view.kind === "chat"}
                  inputMinHeightClass="min-h-[44px]"
                />
              </main>
            </div>
          )}
        </div>
        {view.kind === "chat" ? null : (
          <div className="absolute inset-0 flex flex-col bg-background">
            {view.kind === "history_list" ? (
              <WidgetHistoryList
                origin={window.location.origin}
                getAuth={getHistoryAuth}
                unauthorizedNotice={sessionConversationState === "replacement_ready" ? "refreshToLoad" : "tryLater"}
                currentTaskId={state.taskId}
                onBack={() => setView({ kind: "chat" })}
                onOpenCurrent={() => setView({ kind: "chat" })}
                onOpen={(taskId, title) => setView({ kind: "history_view", taskId, title })}
              />
            ) : (
              <WidgetHistoryView
                key={view.taskId}
                origin={window.location.origin}
                getAuth={getHistoryAuth}
                unauthorizedNotice={sessionConversationState === "replacement_ready" ? "refreshToLoad" : "tryLater"}
                taskId={view.taskId}
                title={view.title}
                sessionPhase={sessionConversationState}
                sessionAttachment={sessionConversationAttachment}
                continueBusy={continueBusy}
                onBack={() => setView({ kind: "history_list" })}
                onContinue={handleContinue}
              />
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export function SessionAgentChatPage() {
  const bridge = useWidgetSession()
  const [isExpanded, setIsExpanded] = useState(false)

  // The advertised capabilities follow every session_update. While the bridge
  // refreshes the token it has no session, and the composer keeps the last
  // advertisement so the visitor's files do not disappear for the refresh.
  const advertisedUploads = bridge.session?.capabilities?.fileUploads
  const [heldUploads, setHeldUploads] = useState(advertisedUploads)
  if (bridge.session && heldUploads !== advertisedUploads) {
    setHeldUploads(advertisedUploads)
  }
  const fileUploads = bridge.session
    ? advertisedUploads
    : bridge.status === "refreshing" ? heldUploads : undefined
  const attachments = useMemo(() => toAttachments(fileUploads), [fileUploads])

  // Held across a token refresh in the same way, so the history entry does not
  // flicker out and back while the bridge has no session.
  const advertisedHistory = bridge.session?.capabilities?.conversationHistory === true
  const [heldHistory, setHeldHistory] = useState(advertisedHistory)
  if (bridge.session && heldHistory !== advertisedHistory) {
    setHeldHistory(advertisedHistory)
  }
  const historyAvailable = bridge.session
    ? advertisedHistory
    : bridge.status === "refreshing" ? heldHistory : false
  const sessionTokenRef = useRef<string | null>(null)
  useEffect(() => {
    sessionTokenRef.current = bridge.session?.token ?? null
  }, [bridge.session])
  const readCurrentToken = useCallback(() => sessionTokenRef.current, [])
  const uploads = useMemo<SessionUploadControls>(() => ({
    ensureFreshSession: bridge.ensureFreshSession,
    refreshSession: bridge.refreshSession,
    waitForOpenConnection: bridge.waitForOpenConnection,
  }), [bridge.ensureFreshSession, bridge.refreshSession, bridge.waitForOpenConnection])

  const connection = useMemo<WebSocketConnection | null>(() => {
    if (bridge.status !== "active" || !bridge.session) return null

    return {
      identity: `widget-session:${bridge.session.generation}`,
      url: buildWidgetSessionWebSocketUrl(window.location.origin),
      protocols: [
        SESSION_PROTOCOL,
        `xagent-session-token.${bridge.session.token}`,
        SESSION_LIVENESS_PROTOCOL,
        SESSION_ATTACH_PROTOCOL,
      ],
      expectedProtocol: SESSION_PROTOCOL,
      chatTaskIdMode: "omit",
      taskBindingMode: "session-subprotocol",
      credentialOwner: { kind: "external" },
    }
  }, [bridge.session, bridge.status])

  const transport = useMemo<AppProviderTransportConfig>(() => ({
    legacyErrorProse: "untrusted",
    capabilities: {
      // This page only renders for the embedded widget's session-resume
      // route, so an in-tab navigation always abandons the visitor's iframe.
      linksOpenInNewTab: "enabled",
      taskStop: "enabled",
    },
    session: {
      connection,
      onConnectionClose: bridge.handleConnectionClose,
      onConnectionFailure: bridge.handleConnectionFailure,
      onConnectionOpen: bridge.handleConnectionOpen,
      allowTasklessChat: true,
      supportsConversationReset: true,
      history: "none",
      // Uploading is the separate `attachments` option: "disabled" here keeps
      // previews, downloads and file mentions closed.
      files: "disabled",
      ...(attachments ? { attachments } : {}),
      agentCards: "disabled",
      voice: "disabled",
      taskControls: "disabled",
    },
  }), [
    attachments,
    bridge.handleConnectionClose,
    bridge.handleConnectionFailure,
    bridge.handleConnectionOpen,
    connection,
  ])

  return (
    <AppProvider transport={transport}>
      <SessionConversationContent
        status={bridge.status}
        agent={bridge.agent}
        terminalCode={bridge.terminalCode}
        isAbsoluteExpiryWarningVisible={
          bridge.isAbsoluteExpiryWarningVisible
        }
        isExpanded={isExpanded}
        onExpandedChange={setIsExpanded}
        attachments={attachments}
        uploads={uploads}
        historyAvailable={historyAvailable}
        readCurrentToken={readCurrentToken}
      />
    </AppProvider>
  )
}
