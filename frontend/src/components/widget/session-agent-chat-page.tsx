"use client"

import React, { useCallback, useMemo, useRef, useState } from "react"
import { Loader2 } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { WidgetVoiceConversation } from "@/components/chat/widget-voice-conversation"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { WidgetChromeControls } from "@/components/widget/widget-chrome-controls"
import {
  AppProvider,
  type AppProviderTransportConfig,
  type SessionAttachmentsConfig,
  useApp,
} from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { VoiceCallStoreProvider } from "@/contexts/voice-call-store"
import { useWidgetInitialMessage } from "@/hooks/use-widget-initial-message"
import { useWidgetOpenMessage } from "@/hooks/use-widget-open-message"
import { classifyWidgetVoiceEvent, WIDGET_VOICE_BUSY_TASK_STATUSES, widgetVoiceAnswerRequestId } from "@/hooks/use-widget-voice-call"
import type { WebSocketConnection } from "@/hooks/use-websocket"
import { readSendErrorCode } from "@/components/chat/clarification-delivery"
import {
  clientErrorTranslationKey,
  type ClientErrorCode,
} from "@/lib/client-errors"
import type { VoiceAudioTransport } from "@/lib/voice-audio"
import type { WidgetSessionHistoryAuth } from "@/lib/widget-session-history"
import { pageMayUseMicrophone } from "@/lib/widget-microphone"
import {
  toAttachments,
  useSessionAttachments,
  type SessionUploadControls,
} from "./use-session-attachments"
import {
  buildWidgetSessionVoiceAsrUrl,
  buildWidgetSessionWebSocketUrl,
  WIDGET_SESSION_VOICE_TTS_PATH,
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

type SessionConversationPhase = ReturnType<typeof useApp>["sessionConversationState"]

// Whether a refused session token may be replaced in this conversation phase.
// A replacement rebuilds the chat connection: while a conversation is being
// replaced, or once the page needs a reload, that would cost the visitor the
// new conversation, so the token the page holds is used as it is.
const tokenReplaceableIn = (phase: SessionConversationPhase): boolean =>
  phase === "unbound" || phase === "bound"

// What one open call bar has had refused of its session tokens, each with the
// single replacement asked for it. A retry stays in the same call; closing the
// bar and calling again starts a new record.
interface VoiceCallTokenUse {
  replaced: Map<string, Promise<string>>
}

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
  // transition (see the status check further down). widget.js re-reports the
  // size when the header remounts, but keeping it here means the label is
  // right from the first frame, including with an older widget.js that sends
  // no such report.
  isExpanded: boolean
  onExpandedChange: (expanded: boolean) => void
  // Present only while the session server advertises file uploads.
  attachments?: SessionAttachmentsConfig
  uploads: SessionUploadControls
  // Whether the session server offers earlier conversations to this session.
  historyAvailable: boolean
  // The token of the session the page holds right now, or null while it has none.
  readCurrentToken: () => string | null
  // Whether a voice call is offered to this session (voice=1, the page may use the microphone, and the server's voice capability).
  voiceOffered: boolean
  // Ends the session as expired when the voice recognition socket is closed with 4408.
  onSessionExpired: () => void
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
  voiceOffered,
  onSessionExpired,
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
    isSendInFlight,
    canStopTask,
    stopTask,
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

  const { busy: openBusy } = useSessionOpenMessage({
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
  // the moment of each request (tokenReplaceableIn): before a conversation is
  // replaced, the token is refreshed when needed and retried once after a
  // rejection; while one is being replaced, a refresh would rebuild the
  // connection and cost the visitor the new conversation, so the current token
  // is used as it is. A voice call's refused tokens follow the same rule.
  // The phase is recorded while rendering, not in an effect: a list or view
  // mounting in this commit reads it from its own effect, and a child's
  // effects run before this component's.
  const phaseRef = useRef(sessionConversationState)
  phaseRef.current = sessionConversationState
  const getHistoryAuth = useCallback((): WidgetSessionHistoryAuth => {
    if (tokenReplaceableIn(phaseRef.current)) {
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
  // message, or a host page message from its arrival until it has ended,
  // takes the visitor back to the chat and keeps the history entry away. A
  // reload-required session does the same: its notice only exists on the chat.
  const historyBlocked = openBusy
    || isConversationResetPending
    || isMessageDeliveryPending
    || isAttachmentBusy
    || continueBusy
  // An active session that no longer offers history closes the list or the
  // view too. A session that is refreshing keeps its last offer, and one that
  // is down shows no chat at all, so neither closes what is open: it is shown
  // again once the session is back.
  const historyWithdrawn = status === "active" && !historyAvailable
  if (view.kind !== "chat" && (isSessionInteractionLocked || historyBlocked || historyWithdrawn)) {
    setView({ kind: "chat" })
  }
  const unauthorizedHistoryNotice = sessionConversationState === "replacement_ready" ? "refreshToLoad" : "tryLater"

  const voiceCallRef = useRef<VoiceCallTokenUse>({ replaced: new Map() })
  // Replaces a token the session voice endpoints refused, only in a phase that
  // allows it and at most once for that token in this call: a second refusal of
  // the same token, as when two sentences are refused together or a retry
  // offers it again, shares that replacement. Resolves with the new token; null
  // when the phase allows none. The phase is read when the refusal arrives.
  const replaceRefused = useCallback((call: VoiceCallTokenUse, refused: string): Promise<string> | null => {
    if (!tokenReplaceableIn(phaseRef.current)) return null
    const asked = call.replaced.get(refused)
    if (asked) return asked
    const renewed = uploads.refreshSession(refused).then((session) => session.token)
    renewed.catch(() => undefined)
    call.replaced.set(refused, renewed)
    return renewed
  }, [uploads])
  // The transport is fixed when a call starts, so the phase and the token are
  // read again at each request or handshake, never held here.
  const voiceTransport = useMemo<VoiceAudioTransport | undefined>(() => {
    if (!voiceOffered) return undefined
    const currentToken = () => {
      const token = readCurrentToken()
      if (token === null) throw new Error("The session has no token")
      return token
    }
    const synthesize = (token: string, text: string, signal: AbortSignal) => fetch(
      `${window.location.origin}${WIDGET_SESSION_VOICE_TTS_PATH}`,
      {
        method: "POST",
        signal,
        headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      },
    )
    // The token for a synthesis request. While a refresh is under way the page
    // holds none: the request waits for the next session's chat socket to open,
    // asking for nothing itself, up to the bridge's deadline. A sentence given
    // up meanwhile (the visitor talked over it or hung up) stops waiting at
    // once.
    const tokenForSynthesis = async (signal: AbortSignal): Promise<string> => {
      const held = readCurrentToken()
      if (held !== null) return held
      const abandoned = () => new DOMException("Aborted", "AbortError")
      if (signal.aborted) throw abandoned()
      let onAbort!: () => void
      const aborted = new Promise<never>((_, reject) => {
        onAbort = () => reject(abandoned())
        signal.addEventListener("abort", onAbort, { once: true })
      })
      try {
        await Promise.race([uploads.waitForOpenConnection(), aborted])
      } finally {
        signal.removeEventListener("abort", onAbort)
      }
      return currentToken()
    }
    return {
      // The token offered the way the chat socket offers it; the server binds
      // the connection to the session, so a later token change leaves it open.
      openAsrSocket: () => {
        const call = voiceCallRef.current
        const token = currentToken()
        const socket = new WebSocket(
          buildWidgetSessionVoiceAsrUrl(window.location.origin),
          [SESSION_PROTOCOL, `xagent-session-token.${token}`],
        )
        // What a close says about the session, not the call, is acted on here;
        // classify below only reads what it means for the call. A socket the
        // call closes itself closes with 1000 or 1005, so these codes come
        // from the server. 4408: the session has expired, and the page ends it
        // as a chat socket closed with 4408 does. 4401: the handshake token
        // was refused, and it is replaced where replaceRefused allows, so the
        // visitor's retry offers the new one.
        socket.addEventListener("close", (event) => {
          if (event.code === 4408) onSessionExpired()
          else if (event.code === 4401) void replaceRefused(call, token)
        })
        return socket
      },
      startFrame: { type: "start", format: "pcm", sample_rate: 16000 },
      // The token the page holds at the time of the request, or the next one
      // when it holds none. Only a 401 is retried, once, with the token
      // replaceRefused gives; a 401 it gives none for, a second 401, or a
      // failed replacement is handed back as the refusal. A sentence already
      // abandoned (the call ended, or the visitor talked over it) neither
      // replaces the token, which would rebuild the chat socket, nor retries:
      // its 401 is handed back as it is.
      requestTts: async (text, signal) => {
        const call = voiceCallRef.current
        const token = await tokenForSynthesis(signal)
        const response = await synthesize(token, text, signal)
        if (response.status !== 401 || signal.aborted) return response
        const renewal = replaceRefused(call, token)
        if (!renewal) return response
        let renewed: string
        try {
          renewed = await renewal
        } catch {
          return response
        }
        if (signal.aborted) return response
        await response.body?.cancel()
        return synthesize(renewed, text, signal)
      },
      classify: classifyWidgetVoiceEvent,
    }
  }, [voiceOffered, readCurrentToken, replaceRefused, uploads, onSessionExpired])

  // A call opened by the visitor's click. Without a transport no call is
  // offered, and an open call is dropped so a later offer never starts one
  // without a click. History hides the chat and the call bar with it, so
  // whatever opens it also hangs up: the visitor could not reach the hang-up
  // control while the microphone stays live, and coming back to the chat
  // starts no call until the visitor asks for one again.
  const [voiceOpen, setVoiceOpen] = useState(false)
  if ((!voiceTransport || view.kind !== "chat") && voiceOpen) setVoiceOpen(false)
  // Recorded while rendering, as phaseRef is: a click reads the bar as last
  // rendered. Pressing start while a call is open changes nothing.
  const voiceOpenRef = useRef(voiceOpen)
  voiceOpenRef.current = voiceOpen
  const openVoiceCall = useCallback(() => {
    if (!voiceOpenRef.current) voiceCallRef.current = { replaced: new Map() }
    setVoiceOpen(true)
  }, [])
  const voiceTaskStatus = state.currentTask?.status || ""
  // What holds a typed send on the start page: a reply or a send under way, a
  // reset, a typed message's files still uploading, or a message handed over
  // from history. Spoken words wait for the same, whichever view the call was
  // opened from.
  const sendHeld = state.isProcessing
    || isConversationResetPending
    || isMessageDeliveryPending
    || isAttachmentBusy
    || continueBusy
  // Spoken words also wait while the chat socket is rebuilt (a token refresh),
  // while a send is in flight and while the task runs; the call goes on
  // meanwhile. It ends when the session is down or the page needs a reload.
  const voiceBusy = sendHeld || !isConnected || isSendInFlight
    || WIDGET_VOICE_BUSY_TASK_STATUSES.includes(voiceTaskStatus)
  const voiceDisabled = status === "degraded" || status === "terminal" || isSessionInteractionLocked
  // Spoken words take the path typed ones take: the start page's send before
  // a conversation exists, the panel's send after, with the call's client
  // message id and the request id of the question this task is waiting on.
  const sendSpoken = useCallback((text: string, clientMessageId: string) => {
    const requestId = widgetVoiceAnswerRequestId(state.currentTask, state.taskId)
    const config = { clientMessageId, ...(requestId ? { metadata: { request_id: requestId } } : {}) }
    return conversationEstablished
      ? sendWithAttachments(text, config, [])
      : handleStartMessage(text, [], config)
  }, [conversationEstablished, handleStartMessage, sendWithAttachments, state.currentTask, state.taskId])
  // Talking over a reply stops the run when the stop control could.
  const interruptSpoken = useCallback(() => {
    if (!canStopTask) return false
    stopTask()
    return true
  }, [canStopTask, stopTask])
  const voiceOffer = voiceTransport
    ? { onVoiceConversation: openVoiceCall, voiceConversationSource: "provided" as const }
    : {}
  // The call lives above every view of the page, so it outlasts a switch
  // between the start page and the panel and sees the session go down.
  const withVoiceCall = (node: React.ReactNode) => (
    <VoiceCallStoreProvider>
      {voiceTransport ? (
        <WidgetVoiceConversation
          transport={voiceTransport}
          taskId={state.taskId}
          messages={state.messages}
          busy={voiceBusy}
          waiting={voiceTaskStatus === "waiting_for_user"}
          disabled={voiceDisabled}
          onSend={sendSpoken}
          onInterrupt={interruptSpoken}
          open={voiceOpen}
          onOpenChange={setVoiceOpen}
        />
      ) : null}
      {node}
    </VoiceCallStoreProvider>
  )

  if (status === "waiting" || (status === "refreshing" && !agent)) {
    return withVoiceCall(
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
    return withVoiceCall(
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

  return withVoiceCall(
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
              // A message handed over from history holds the composer as the
              // start screen's isSending does: a reply sent here meanwhile
              // would go into the conversation the hand-over then resets.
              attachmentsBusy={isAttachmentBusy || continueBusy}
              attachmentFiles={attachments ? panelFiles : []}
              onAttachmentFilesChange={attachments ? onPanelFilesChange : undefined}
              {...(voiceTransport ? { onProvidedVoiceConversation: openVoiceCall } : {})}
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
                  isSending={sendHeld || reloadRequired}
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
                  {...voiceOffer}
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
                unauthorizedNotice={unauthorizedHistoryNotice}
                currentTaskId={state.taskId}
                onBack={() => setView({ kind: "chat" })}
                onOpen={(taskId, title) => setView({ kind: "history_view", taskId, title })}
              />
            ) : (
              <WidgetHistoryView
                key={view.taskId}
                origin={window.location.origin}
                getAuth={getHistoryAuth}
                unauthorizedNotice={unauthorizedHistoryNotice}
                taskId={view.taskId}
                title={view.title}
                sessionPhase={sessionConversationState}
                sessionAttachment={sessionConversationAttachment}
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
  // Recorded while rendering for the same reason as the phase in
  // SessionConversationContent: a history list or view that mounts in this
  // commit asks for the token from an effect that runs before this one's.
  const sessionTokenRef = useRef<string | null>(null)
  sessionTokenRef.current = bridge.session?.token ?? null
  const readCurrentToken = useCallback(() => sessionTokenRef.current, [])

  // The page runs in the host page's iframe only. widget.js adds voice=1
  // wherever it delegated the microphone, which is every page that did not set
  // data-voice="false". voice=1 is an unsigned hint that any page can add to
  // the URL itself and is not a permission: whether voice calls are offered is
  // decided by the server (the voice capability of the session payload), and
  // the voice endpoints check again on every request.
  const [voiceRequested] = useState(
    // This first render also runs on the server, where there is no window:
    // voice is then not requested, and the browser computes it on hydration.
    () => typeof window !== "undefined"
      && new URLSearchParams(window.location.search).get("voice") === "1"
      && pageMayUseMicrophone(),
  )
  // Held across a refresh like the capabilities above, and kept once the
  // session is down, so an open call sees the session end instead of
  // disappearing with the flag.
  const advertisedVoice = bridge.session?.capabilities?.voice === true
  const [heldVoice, setHeldVoice] = useState(advertisedVoice)
  if (bridge.session && heldVoice !== advertisedVoice) {
    setHeldVoice(advertisedVoice)
  }
  const voiceOffered = voiceRequested && (bridge.session ? advertisedVoice : heldVoice)
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
      messageQueue: "enabled",
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
        voiceOffered={voiceOffered}
        onSessionExpired={bridge.handleSessionExpired}
      />
    </AppProvider>
  )
}
