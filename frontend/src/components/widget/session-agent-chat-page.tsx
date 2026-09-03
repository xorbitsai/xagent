"use client"

import React, { useCallback, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { WidgetChromeControls } from "@/components/widget/widget-chrome-controls"
import {
  AppProvider,
  type AppProviderTransportConfig,
  useApp,
} from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import type { WebSocketConnection } from "@/hooks/use-websocket"
import {
  buildWidgetSessionWebSocketUrl,
  useWidgetSession,
  type WidgetSessionAgent,
  type WidgetSessionStatus,
} from "./use-widget-session"

const SESSION_PROTOCOL = "xagent-session-v1"
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
}

function SessionConversationContent({
  status,
  agent,
  terminalCode,
  isAbsoluteExpiryWarningVisible,
  isExpanded,
  onExpandedChange,
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
  const [resetError, setResetError] = useState(false)
  const [startMessageError, setStartMessageError] = useState(false)

  const handleStartMessage = useCallback(async (
    message: string,
    files: File[],
    config?: Record<string, unknown>,
  ) => {
    setStartMessageError(false)
    try {
      await sendMessage(message, config, files)
    } catch (error) {
      setStartMessageError(true)
      throw error
    }
  }, [sendMessage])

  const handleStartNewConversation = useCallback(async () => {
    setResetError(false)
    try {
      await startNewConversation()
    } catch {
      setResetError(true)
    }
  }, [startNewConversation])

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

  const hasEstablishedConversation = state.taskId !== null
  const reloadRequired = isSessionInteractionLocked
  const resetDisabled =
    reloadRequired || isConversationResetPending || isMessageDeliveryPending

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
            newConversation={hasEstablishedConversation ? {
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
        {startMessageError && !reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.startMessageFailed")}
          </p>
        ) : null}
        {reloadRequired ? (
          <p className="mt-2 text-xs text-destructive" role="alert">
            {t("widgetSession.reloadRequired")}
          </p>
        ) : null}
      </header>

      <div className="min-h-0 flex-1">
        {hasEstablishedConversation ? (
          <TaskConversationPanel
            mode="page"
            showTaskActions={false}
            showTokenUsage={false}
            showDagPreview={false}
            showTaskFiles={false}
            hideFileUpload={true}
            hideConfig={true}
            compactInput={true}
            deferFileUpload={true}
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
                }
                files={[]}
                onFilesChange={() => undefined}
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

  const connection = useMemo<WebSocketConnection | null>(() => {
    if (bridge.status !== "active" || !bridge.session) return null

    return {
      identity: `widget-session:${bridge.session.generation}`,
      url: buildWidgetSessionWebSocketUrl(window.location.origin),
      protocols: [
        SESSION_PROTOCOL,
        `xagent-session-token.${bridge.session.token}`,
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
    },
    session: {
      connection,
      onConnectionClose: bridge.handleConnectionClose,
      onConnectionFailure: bridge.handleConnectionFailure,
      onConnectionOpen: bridge.handleConnectionOpen,
      allowTasklessChat: true,
      supportsConversationReset: true,
      history: "none",
      files: "disabled",
      agentCards: "disabled",
      voice: "disabled",
      taskControls: "disabled",
    },
  }), [
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
      />
    </AppProvider>
  )
}
