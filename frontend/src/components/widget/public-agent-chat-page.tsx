"use client"

import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Loader2, MessageSquarePlus } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { AppProvider, useApp, type AppProviderTransportConfig } from "@/contexts/app-context-chat"
import { usePublicFileAccessPolicy } from "@/contexts/file-access-context"
import { useI18n } from "@/contexts/i18n-context"
import { uploadPublicChatFile } from "@/lib/public-chat-file-upload"
import { normalizeTaskStatus } from "@/lib/task-status"
import {
  getApiUrl,
} from "@/lib/utils"

interface PublicAgentChatPageProps {
  authMode: "widget" | "share"
  routeToken: string
  guestId?: string | null
  searchAgentId?: number | null
  embedTicket?: string | null
  widgetKey?: string | null
}

type PublicAuthResult = {
  access_token: string
  agent_id?: number | null
  agent_name?: string | null
  agent_logo?: string | null
  agent_description?: string | null
  suggested_prompts?: string[] | null
  // Set instead of agent_id when the share token exposes a workforce.
  workforce_id?: number | null
}

interface PublicConversationContentProps {
  authMode: "widget" | "share"
  routeToken: string
  normalizedGuestId?: string | null
  accessToken: string
  agentId: number | null
  workforceId: number | null
  agentName: string | null
  agentLogo: string | null
  agentDescription: string | null
  suggestedPrompts: string[]
  onAuthInvalidated?: () => void
}

// WS-close reasons that mean "this session isn't usable as-is" rather than a
// transport failure — used to distinguish a recoverable auth/isolation denial
// (recoverable by dropping the stale task/token and starting fresh) from a
// generic connection drop (must not wipe the session). #973. NOT the
// non-recoverable causes: "Share link is unavailable" is emitted when the owner
// disabled the link, unpublished the agent/workforce, or the channel
// mismatches, so treating it as recoverable would trigger a pointless clear +
// re-auth round-trip that still lands on the terminal error.
//   - "Task not found or access denied": the per-guest isolation denial — a
//     returning visitor's persisted taskId belongs to another (or a pre-#973)
//     guest — or a deleted task; both recover the same way. The backend
//     deliberately reuses its not-found detail for guest mismatches so task
//     ids can't be enumerated (see _require_share_guest_owns_task). Backend
//     HTTPException.detail surfaced as event.reason on a 4003.
//   - "Access denied": use-websocket.ts's fallback when a 4003 carries no
//     reason.
//   - "Invalid share token": the persisted guest JWT no longer validates
//     server-side (e.g. a mid-session secret rotation; plain expiry is already
//     caught proactively at mount by isShareTokenExpired). Recovery here drops
//     only the stale *task-id* key and lands the visitor on the start screen;
//     the token/auth-blob key is cleared separately by onAuthInvalidated when
//     the first send hits a task-create 401, which then re-auths. Without this,
//     the WS-resume path would strand the visitor with no send to trigger that.
const SHARE_ACCESS_DENIED_REASONS = new Set([
  "Task not found or access denied",
  "Access denied",
  "Invalid share token",
])

// localStorage.removeItem can throw in restricted contexts (private mode /
// sandboxed iframe). Every share-recovery path that clears a key must survive
// that so the in-memory session reset still proceeds. #973.
const safeRemoveItem = (key: string) => {
  try {
    localStorage.removeItem(key)
  } catch {
    // Non-fatal: the caller's state reset recovers the session regardless.
  }
}

// Decode a JWT's payload segment WITHOUT verifying its signature — a read-only
// liveness/identity pre-filter for client-side routing only. Every security
// decision (signature, expiry-on-use, isolation) stays server-side. Returns
// null on any malformed input so callers fall back to their safe default. #973.
const decodeJwtClaims = (token: string): Record<string, unknown> | null => {
  try {
    const payload = token.split(".")[1]
    if (!payload) {
      return null
    }
    const claims = JSON.parse(atob(payload.replace(/-/g, "+").replace(/_/g, "/")))
    return typeof claims === "object" && claims !== null ? claims : null
  } catch {
    return null
  }
}

// A persisted guest JWT past its exp (30-day TTL) is dead on arrival: reusing
// it fails every request, and over the WS-resume path it strands the visitor on
// a session that can never re-auth on its own. Treat an expired token as absent
// so the mount re-auths fresh, instead of relying on a failed connect to
// recover. Any parse failure falls through to the server (which fails closed).
const isShareTokenExpired = (token: string): boolean => {
  const exp = decodeJwtClaims(token)?.exp
  return typeof exp === "number" && exp * 1000 <= Date.now()
}

// The server-minted guest_id (#973) that scopes every piece of per-guest client
// state. It lives only inside the signed guest JWT, so derive it here to scope
// the persisted task-id key: a task-id minted under guest A can then never be
// read back under guest B's token (two-tab race, post-expiry re-auth, or a
// pre-#973 legacy token), which would otherwise leave that task permanently
// unreachable. Falls back to a stable bucket on any decode failure.
const shareGuestIdFromToken = (token: string): string => {
  const guestId = decodeJwtClaims(token)?.guest_id
  return typeof guestId === "string" && guestId ? guestId : "anonymous"
}

type PublicMessageConfig = Record<string, unknown>

function PublicConversationContent({
  authMode,
  routeToken,
  normalizedGuestId,
  accessToken,
  agentId,
  workforceId,
  agentName,
  agentLogo,
  agentDescription,
  suggestedPrompts,
  onAuthInvalidated,
}: PublicConversationContentProps) {
  const {
    state,
    dispatch,
    sendMessage,
    setTaskId,
    connectionError,
    voiceInputEnabled,
  } = useApp()
  const { t } = useI18n()
  const [createTaskError, setCreateTaskError] = useState<string | null>(null)
  const [draftMessage, setDraftMessage] = useState("")
  const [draftFiles, setDraftFiles] = useState<File[]>([])
  const [isBootstrappingTask, setIsBootstrappingTask] = useState(false)
  const [hasResolvedStoredTask, setHasResolvedStoredTask] = useState(false)
  const storageKey = authMode === "share"
    // Scope the persisted task-id by guest_id (decoded from the guest JWT) so a
    // task minted under one guest is never read back under another's token — a
    // two-tab race, a post-expiry re-auth, or a pre-#973 legacy token would
    // otherwise leave that task permanently unreachable. #973.
    ? `${authMode}_task_${routeToken}_${agentId ?? "anonymous"}_${shareGuestIdFromToken(accessToken)}`
    // Include the owner id so two different widgets embedded on the same site
    // (which share a guest id) don't collide on one stored task. A workforce
    // widget has no agentId, so fall back to its workforceId.
    : `${authMode}_task_${agentId ?? (workforceId ? `wf${workforceId}` : "anonymous")}_${normalizedGuestId ?? "anonymous"}`
  const publicApiPrefix = authMode === "share" ? "/api/share" : "/api/widget"

  useEffect(() => {
    setHasResolvedStoredTask(false)
    const savedTaskId = localStorage.getItem(storageKey)
    if (!savedTaskId) {
      setTaskId(null, { navigate: false })
      setHasResolvedStoredTask(true)
      return
    }

    const parsedTaskId = parseInt(savedTaskId, 10)
    if (Number.isNaN(parsedTaskId)) {
      setTaskId(null, { navigate: false })
      setHasResolvedStoredTask(true)
      return
    }

    setTaskId(parsedTaskId, { navigate: false })
    setHasResolvedStoredTask(true)
  }, [setTaskId, storageKey])

  useEffect(() => {
    if (!hasResolvedStoredTask) {
      return
    }

    if (state.taskId) {
      localStorage.setItem(storageKey, state.taskId.toString())
      return
    }

    safeRemoveItem(storageKey)
  }, [hasResolvedStoredTask, state.taskId, storageKey])

  // Backstop recovery for a share session the server rejects mid-flight. The
  // guest-scoped storageKey above already makes a *foreign* taskId structurally
  // unreadable, so this covers the residual case where an otherwise-valid
  // session is denied at the WS layer — e.g. "Invalid share token" from a
  // mid-session secret rotation. The backend accepts the socket before its
  // auth check, so this close arrives as a real 4003 with its reason intact
  // (a pre-accept close would collapse to a bare 403 the browser reports as
  // code 1006). Drop the stale taskId and fall back to the start screen instead
  // of stranding the visitor on an error. Scoped to access-denied reasons so a
  // transient transport drop never wipes a live session. #973.
  useEffect(() => {
    if (authMode !== "share" || !connectionError || !state.taskId) {
      return
    }
    if (!SHARE_ACCESS_DENIED_REASONS.has(connectionError.message)) {
      return
    }
    safeRemoveItem(storageKey)
    setTaskId(null, { navigate: false })
  }, [authMode, connectionError, state.taskId, storageKey, setTaskId])

  // Ending a conversation is purely client-side: drop the persisted id and the
  // active taskId, and the visitor is back on the start screen; the next
  // message creates a fresh task through handleSend. #1039
  const handleNewConversation = useCallback(() => {
    safeRemoveItem(storageKey)
    setTaskId(null, { navigate: false })
    // Nulling taskId closes the socket, so no terminal WS event will ever
    // reset these. Left stale mid-run, isProcessing keeps the start screen's
    // composer disabled forever, currentTask pins the header on
    // "Connecting...", and isHistoryLoading (cleared by an onConnect timer
    // that may never have been scheduled) pins it on "Initializing".
    dispatch({ type: "SET_PROCESSING", payload: false })
    dispatch({ type: "SET_CURRENT_TASK", payload: null })
    dispatch({ type: "SET_HISTORY_LOADING", payload: false })
    setDraftMessage("")
    setDraftFiles([])
    setCreateTaskError(null)
  }, [dispatch, storageKey, setTaskId])

  const handleSend = useCallback(async (
    message: string,
    config?: PublicMessageConfig,
    files?: File[],
  ) => {
    if (state.taskId) {
      await sendMessage(message, config, files)
      return
    }

    // For a workforce share the first turn starts inside task creation, which
    // rejects an empty message server-side (400) — AFTER any files uploaded
    // above would already be orphaned. Guard the empty case here so files are
    // never uploaded for a turn that cannot start.
    if (workforceId && !message.trim()) {
      return
    }

    setIsBootstrappingTask(true)
    try {
      const taskPayload: Record<string, string | number | string[]> = {
        title: message,
        description: message,
      }
      if (agentId) {
        taskPayload.agent_id = agentId
      }

      setCreateTaskError(null)

      // Workforce shares start their first turn inside task creation, so any
      // opening-message attachments must be uploaded (task-lessly — no task
      // exists yet) and threaded in as file ids BEFORE the run begins;
      // otherwise the first turn never sees them.
      if (workforceId && files?.length) {
        const uploaded = await Promise.all(files.map((file) => uploadPublicChatFile({
          url: `${getApiUrl()}${publicApiPrefix}/files/upload`,
          accessToken,
          file,
          taskType: "task",
          fallbackError: t("files.uploadFailed"),
        })))
        taskPayload.files = uploaded.map((item) => item.file_id)
      }

      const response = await fetch(`${getApiUrl()}${publicApiPrefix}/chat/task/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${accessToken}`,
        },
        body: JSON.stringify(taskPayload),
      })

      if (!response.ok) {
        // A share task-create carries no task id, so 401/403 here means the
        // guest token itself is no longer valid (rotated/disabled link, or a
        // legacy token rejected post-#973). Drop the persisted token and force
        // a fresh auth rather than leaving the visitor on a dead session.
        if (authMode === "share" && (response.status === 401 || response.status === 403)) {
          safeRemoveItem(storageKey)
          onAuthInvalidated?.()
        }
        const errorData = await response.json().catch(() => null)
        const errorMessage = errorData?.detail || t("widgetChat.messages.error_init")
        setCreateTaskError(errorMessage)
        throw new Error(errorMessage)
      }

      const taskData = await response.json()
      const newTaskId = taskData.task_id
      if (typeof newTaskId !== "number") {
        throw new Error("Task creation failed")
      }

      setTaskId(newTaskId, { navigate: false })
      dispatch({
        type: "SET_CURRENT_TASK",
        payload: {
          id: newTaskId.toString(),
          title: taskData.title || message,
          status: normalizeTaskStatus(taskData.status) || "pending",
          description: taskData.description || message,
          createdAt: taskData.created_at || new Date().toISOString(),
          updatedAt:
            taskData.updated_at
            || taskData.created_at
            || new Date().toISOString(),
          agentId: taskData.agent_id ?? agentId ?? undefined,
          agentName: taskData.agent_name || agentName || undefined,
          agentLogoUrl: taskData.agent_logo_url || agentLogo || undefined,
        },
      })

      if (!workforceId) {
        await sendMessage(message, { ...config, targetTaskId: newTaskId }, files)
      }
      // Workforce share sessions already started their first turn (with the
      // files threaded in above) inside task creation — the connection
      // replays it from history, so re-sending over the websocket would
      // duplicate the turn.
      setDraftMessage("")
      setDraftFiles([])
    } catch (error) {
      setIsBootstrappingTask(false)
      throw error
    }
  }, [accessToken, agentId, agentLogo, agentName, authMode, dispatch, onAuthInvalidated, publicApiPrefix, sendMessage, setTaskId, state.taskId, storageKey, t, workforceId])

  useEffect(() => {
    if (state.taskId || createTaskError) {
      setIsBootstrappingTask(false)
    }
  }, [createTaskError, state.taskId])

  const resolvedAgentName = state.currentTask?.agentName || agentName || t("widgetChat.title")
  const resolvedAgentLogo = state.currentTask?.agentLogoUrl || agentLogo || null
  const shouldShowStartScreen = !state.taskId && hasResolvedStoredTask
  const showStatus = !createTaskError
  const statusText = state.isHistoryLoading
    ? t("widgetChat.status.initializing")
    : state.currentTask?.status === "running" || state.isProcessing || isBootstrappingTask
      ? t("widgetChat.status.connecting")
      : t("widgetChat.status.online")

  return (
    <div className="h-screen flex flex-col bg-background">
      <div className="flex-none p-4 border-b bg-card text-card-foreground shadow-sm z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary overflow-hidden">
            {resolvedAgentLogo ? (
              <img
                src={resolvedAgentLogo.startsWith("http") ? resolvedAgentLogo : `${getApiUrl()}${resolvedAgentLogo.startsWith("/") ? "" : "/"}${resolvedAgentLogo}`}
                alt={resolvedAgentName}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-5 h-5 rounded-full bg-primary/20" />
            )}
          </div>
          <div>
            <h1 className="text-sm font-semibold">{resolvedAgentName}</h1>
            {showStatus ? (
              <p className="text-xs text-muted-foreground">{statusText}</p>
            ) : (
              <p className="text-xs text-destructive">{createTaskError}</p>
            )}
          </div>
          {state.taskId && (
            <button
              type="button"
              onClick={handleNewConversation}
              title={t("widgetChat.newConversation")}
              aria-label={t("widgetChat.newConversation")}
              className="ml-auto p-2 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted/50 transition-colors"
            >
              <MessageSquarePlus className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {!hasResolvedStoredTask && !state.taskId ? (
          <div className="flex h-full items-center justify-center">
            <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
          </div>
        ) : shouldShowStartScreen ? (
          <div className="h-full overflow-y-auto">
            <main className="container max-w-4xl mx-auto px-4 py-8">
              <ChatStartScreen
                title={resolvedAgentName}
                description={agentDescription || undefined}
                prompts={suggestedPrompts}
                onSend={(message, files, config) => handleSend(message, config, files)}
                isSending={isBootstrappingTask || state.isProcessing}
                inputValue={draftMessage}
                onInputChange={setDraftMessage}
                files={draftFiles}
                onFilesChange={setDraftFiles}
                readOnlyConfig={true}
                hideConfig={true}
                compactInput={true}
                deferFileUpload={true}
                voiceInputEnabled={voiceInputEnabled}
                autoFocus={true}
                inputMinHeightClass="min-h-[44px]"
              />
            </main>
          </div>
        ) : (
          <TaskConversationPanel
            mode="page"
            showTaskActions={false}
            showTokenUsage={false}
            showDagPreview={false}
            showTaskFiles={false}
            // A visitor on a customer's site gets the answer, not the run: no
            // reasoning, no tool arguments, no raw tool output. Share links
            // keep the trace — #1041 scopes the hiding to the widget only.
            showProcessView={authMode === "share"}
            hideFileUpload={false}
            hideConfig={true}
            compactInput={true}
            deferFileUpload={true}
            onSend={handleSend}
          />
        )}
      </div>
    </div>
  )
}

export function PublicAgentChatPage({
  authMode,
  routeToken,
  guestId,
  searchAgentId = null,
  embedTicket = null,
  widgetKey = null,
}: PublicAgentChatPageProps) {
  const { t } = useI18n()
  const normalizedGuestId = authMode === "widget" ? (guestId || "anonymous") : null
  const [isInitializing, setIsInitializing] = useState(true)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [authResult, setAuthResult] = useState<PublicAuthResult | null>(null)
  // Bumped to force a fresh /api/share/auth when a persisted guest token turns
  // out to be invalid (see onAuthInvalidated below).
  const [reauthNonce, setReauthNonce] = useState(0)

  // The guest token is minted server-side and carries the per-guest isolation
  // credential (#973). Persist it per share link and REUSE it across reloads
  // instead of re-authing every mount: re-authing would mint a new guest_id
  // each time, so a returning visitor's own tasks would fail the per-guest
  // check. Widget keeps its old behavior (its guest_id is client-supplied).
  const shareAuthStorageKey = authMode === "share" ? `share_auth_${routeToken}` : null

  const onAuthInvalidated = useCallback(() => {
    if (shareAuthStorageKey) {
      safeRemoveItem(shareAuthStorageKey)
    }
    setAuthResult(null)
    setIsInitializing(true)
    setReauthNonce((n) => n + 1)
  }, [shareAuthStorageKey])

  useEffect(() => {
    const persistShareAuth = (data: PublicAuthResult) => {
      if (!shareAuthStorageKey) {
        return
      }
      try {
        localStorage.setItem(shareAuthStorageKey, JSON.stringify(data))
      } catch {
        // Non-fatal: without persistence the visitor simply re-auths (and gets
        // a new guest session) on the next reload.
      }
    }

    const readPersistedShareAuth = (): PublicAuthResult | null => {
      if (!shareAuthStorageKey) {
        return null
      }
      try {
        const raw = localStorage.getItem(shareAuthStorageKey)
        if (!raw) {
          return null
        }
        const parsed: unknown = JSON.parse(raw)
        // Reject anything that isn't a well-shaped auth blob so a corrupt or
        // cross-version localStorage entry falls back to a clean re-auth rather
        // than flowing malformed values downstream. agent_id/workforce_id are
        // display/routing hints (never the isolation credential — that lives in
        // the signed guest JWT), but keep their optional-number contract intact.
        const isNullableNumber = (value: unknown) =>
          value === undefined || value === null || typeof value === "number"
        if (
          !parsed
          || typeof parsed !== "object"
          || typeof (parsed as PublicAuthResult).access_token !== "string"
          || !(parsed as PublicAuthResult).access_token
          || !isNullableNumber((parsed as PublicAuthResult).agent_id)
          || !isNullableNumber((parsed as PublicAuthResult).workforce_id)
        ) {
          return null
        }
        // Drop an expired token here so the mount re-auths fresh instead of
        // reusing a dead credential (which the WS-resume path can't recover
        // from on its own). #973.
        if (isShareTokenExpired((parsed as PublicAuthResult).access_token)) {
          return null
        }
        return parsed as PublicAuthResult
      } catch {
        return null
      }
    }

    const initPublicChat = async () => {
      try {
        const persisted = readPersistedShareAuth()
        if (persisted) {
          setAuthResult(persisted)
          setErrorMessage(null)
          return
        }

        const authPath = authMode === "share" ? "/api/share/auth" : "/api/widget/auth"
        const authPayload = authMode === "share"
          ? { share_token: routeToken }
          : {
              guest_id: normalizedGuestId,
              agent_id: searchAgentId,
              embed_ticket: embedTicket || undefined,
              // Direct visits (no embed ticket) authenticate with the widget
              // key carried in the opened URL.
              widget_key: embedTicket ? undefined : widgetKey || undefined,
            }

        const authResponse = await fetch(`${getApiUrl()}${authPath}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(authPayload),
        })

        if (!authResponse.ok) {
          const errorData = await authResponse.json().catch(() => null)
          throw new Error(errorData?.detail || "Widget authentication failed")
        }

        const authData = await authResponse.json()
        persistShareAuth(authData)
        setAuthResult(authData)
        setErrorMessage(null)
      } catch (error) {
        console.error(error)
        setErrorMessage((error as Error).message || t("widgetChat.messages.error_init"))
      } finally {
        setIsInitializing(false)
      }
    }

    initPublicChat()
  }, [authMode, embedTicket, widgetKey, normalizedGuestId, routeToken, searchAgentId, shareAuthStorageKey, reauthNonce, t])

  const publicAccessToken = authResult?.access_token ?? ""
  const fileAccess = usePublicFileAccessPolicy(publicAccessToken)

  const transport = useMemo<AppProviderTransportConfig>(() => ({
    capabilities: {
      agentCards: "disabled",
      voice: "disabled",
    },
    buildWebSocketUrl: ({ baseUrl, taskId, token }) =>
      `${baseUrl}/${authMode === "share" ? "api/share" : "api/widget"}/chat/ws/${taskId}${token ? `?token=${token}` : ""}`,
    fileAccess,
    uploadFiles: (files, params) =>
      Promise.all(files.map((file) =>
        uploadPublicChatFile({
          url: `${getApiUrl()}/${authMode === "share" ? "api/share" : "api/widget"}/files/upload`,
          accessToken: publicAccessToken,
          file,
          taskType: params.taskType,
          taskId: params.taskId,
          fallbackError: t("files.uploadFailed"),
        }),
      )),
  }), [authMode, fileAccess, publicAccessToken, t])

  if (isInitializing) {
    return (
      <div className="h-screen flex items-center justify-center bg-background">
        <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!authResult || errorMessage) {
    return (
      <div className="h-screen flex items-center justify-center bg-background p-6">
        <div className="max-w-md text-center text-sm text-muted-foreground">
          {errorMessage || t("widgetChat.messages.error_init")}
        </div>
      </div>
    )
  }

  const resolvedAuthResult = authResult

  return (
    <AppProvider token={publicAccessToken} transport={transport}>
      <PublicConversationContent
        authMode={authMode}
        routeToken={routeToken}
        normalizedGuestId={normalizedGuestId}
        accessToken={resolvedAuthResult.access_token}
        agentId={resolvedAuthResult.agent_id ?? searchAgentId ?? null}
        workforceId={resolvedAuthResult.workforce_id ?? null}
        agentName={resolvedAuthResult.agent_name ?? null}
        agentLogo={resolvedAuthResult.agent_logo ?? null}
        agentDescription={resolvedAuthResult.agent_description ?? null}
        suggestedPrompts={resolvedAuthResult.suggested_prompts ?? []}
        onAuthInvalidated={onAuthInvalidated}
      />
    </AppProvider>
  )
}
