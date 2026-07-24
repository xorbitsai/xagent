"use client"

import React, { useCallback, useEffect, useMemo, useState } from "react"
import { Loader2 } from "lucide-react"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { AppProvider, useApp, type AppProviderTransportConfig } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { uploadPublicChatFile } from "@/lib/public-chat-file-upload"
import { normalizeTaskStatus } from "@/lib/task-status"
import {
  getApiUrl,
  getFilePublicDownloadUrl,
  getFilePublicPreviewUrl,
  setPublicAccessToken,
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
}: PublicConversationContentProps) {
  const { state, dispatch, sendMessage, setTaskId } = useApp()
  const { t } = useI18n()
  const [createTaskError, setCreateTaskError] = useState<string | null>(null)
  const [draftMessage, setDraftMessage] = useState("")
  const [draftFiles, setDraftFiles] = useState<File[]>([])
  const [isBootstrappingTask, setIsBootstrappingTask] = useState(false)
  const [hasResolvedStoredTask, setHasResolvedStoredTask] = useState(false)
  const storageKey = authMode === "share"
    ? `${authMode}_task_${routeToken}_${agentId ?? "anonymous"}`
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

    localStorage.removeItem(storageKey)
  }, [hasResolvedStoredTask, state.taskId, storageKey])

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
  }, [accessToken, agentId, agentLogo, agentName, dispatch, publicApiPrefix, sendMessage, setTaskId, state.taskId, t, workforceId])

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

  useEffect(() => {
    const initPublicChat = async () => {
      try {
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
        setAuthResult(authData)
        setPublicAccessToken(authData.access_token ?? null)
        setErrorMessage(null)
      } catch (error) {
        console.error(error)
        setErrorMessage((error as Error).message || t("widgetChat.messages.error_init"))
        setPublicAccessToken(null)
      } finally {
        setIsInitializing(false)
      }
    }

    initPublicChat()
    return () => setPublicAccessToken(null)
  }, [authMode, embedTicket, widgetKey, normalizedGuestId, routeToken, searchAgentId, t])

  const publicAccessToken = authResult?.access_token ?? ""

  const transport = useMemo<AppProviderTransportConfig>(() => ({
    buildWebSocketUrl: ({ baseUrl, taskId, token }) =>
      `${baseUrl}/${authMode === "share" ? "api/share" : "api/widget"}/chat/ws/${taskId}${token ? `?token=${token}` : ""}`,
    buildFilePreviewUrl: ({ baseUrl, fileId }) =>
      getFilePublicPreviewUrl(fileId, baseUrl),
    buildFileDownloadUrl: ({ baseUrl, fileId }) =>
      getFilePublicDownloadUrl(fileId, baseUrl),
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
  }), [authMode, publicAccessToken, t])

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
      />
    </AppProvider>
  )
}
