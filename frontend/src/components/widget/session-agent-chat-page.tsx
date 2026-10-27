"use client"

import React, { useCallback, useMemo, useState } from "react"
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
import type { WebSocketConnection } from "@/hooks/use-websocket"
import { readSendErrorCode } from "@/components/chat/clarification-delivery"
import {
  clientErrorTranslationKey,
  type ClientErrorCode,
} from "@/lib/client-errors"
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

const SESSION_PROTOCOL = "xagent-session-v1"
// Asks the session server to confirm and send heartbeat frames. A server that
// does not know the marker ignores it, and the connection then runs without
// dead-connection detection.
const SESSION_LIVENESS_PROTOCOL = "xagent-session-liveness-v1"
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
  const [startMessageFailure, setStartMessageFailure] =
    useState<{ code: ClientErrorCode | null } | null>(null)

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
        setStartMessageFailure({ code: readSendErrorCode(error) })
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

  useWidgetInitialMessage({
    ready: status === "active" && agent !== null && isConnected,
    // A first send queued during the initial connect has no task yet.
    hasConversation: state.taskId !== null || isMessageDeliveryPending,
    onMessage: (message) => handleStartMessage(message, []),
  })

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
              onClick: () => void handleStartNewConversation(),
              disabled: resetDisabled,
              pending: isConversationResetPending,
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
            {startMessageFailure.code
              ? t(clientErrorTranslationKey(startMessageFailure.code))
              : t("widgetSession.startMessageFailed")}
          </p>
        ) : null}
        {reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.reloadRequired")}
          </p>
        ) : null}
      </header>

      <div className="min-h-0 flex-1">
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
                isSending={
                  state.isProcessing
                  || isConversationResetPending
                  || isMessageDeliveryPending
                  || reloadRequired
                  || isAttachmentBusy
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
                autoFocus={true}
                inputMinHeightClass="min-h-[44px]"
              />
            </main>
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
      />
    </AppProvider>
  )
}
