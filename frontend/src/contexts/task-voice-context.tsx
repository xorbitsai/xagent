"use client"

import React, { createContext, useContext, useLayoutEffect, useRef, useState } from "react"
import { usePathname, useSearchParams } from "next/navigation"
import { useApp } from "@/contexts/app-context-chat"
import { useAuth } from "@/contexts/auth-context"
import { VoiceConversation } from "@/components/chat/voice-conversation"
import { isJsonRecord } from "@/lib/api-wrapper"

type VoiceConfig = Record<string, unknown>
const TaskVoiceContext = createContext<((config: VoiceConfig) => void) | undefined>(undefined)
export const useTaskVoice = () => useContext(TaskVoiceContext)

/** Keep a home-started call alive while its first utterance creates a task. */
export function TaskVoiceProvider({ children }: { children: React.ReactNode }) {
  const { state, sendMessage, pauseTask, isConnected, isMessageDeliveryPending,
    isConversationResetPending, isSessionInteractionLocked, voiceInputEnabled, taskControlsEnabled } = useApp()
  const pathname = usePathname()
  const search = useSearchParams().toString()
  const auth = useAuth()
  const login = JSON.stringify([auth.user?.id, auth.session?.sessionId])
  const [open, setOpen] = useState(false)
  const [creating, setCreating] = useState(false)
  const owner = useRef<{ path: string; search: string; login: string; config: VoiceConfig; createdTaskId: number | null } | null>(null)

  const close = () => { owner.current = null; setOpen(false); setCreating(false) }
  useLayoutEffect(() => () => { owner.current = null }, [])
  useLayoutEffect(() => {
    const active = owner.current
    if (!active) return
    if (active.login !== login) { close(); return }
    if (active.path === pathname && active.search === search) {
      if (creating && isConnected && active.createdTaskId === state.taskId && pathname === `/task/${active.createdTaskId}`) setCreating(false)
      return
    }
    if (active.createdTaskId !== null && active.path === "/task" && state.taskId === active.createdTaskId && pathname === `/task/${active.createdTaskId}`) {
      active.path = pathname
      active.search = search
      if (isConnected) setCreating(false)
      return
    }
    close()
    // Navigation to a different draft/task invalidates pending create ownership.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pathname, search, state.taskId, login, isConnected, creating])

  return <TaskVoiceContext.Provider value={config => {
    if (!voiceInputEnabled || !taskControlsEnabled) return
    owner.current = { path: pathname, search, login, config, createdTaskId: null }
    setOpen(true)
  }}>
    {children}
    <VoiceConversation
      open={open} onOpenChange={value => { if (!value) close() }}
      allowTaskAdoption={creating && owner.current?.createdTaskId === state.taskId}
      taskId={state.taskId} messages={state.messages}
      busy={state.isProcessing || isMessageDeliveryPending || ["running", "pending", "pause_requested", "resume_requested"].includes(state.currentTask?.status || "")}
      waiting={state.currentTask?.status === "waiting_for_user"}
      disabled={!voiceInputEnabled || !taskControlsEnabled || isConversationResetPending || isSessionInteractionLocked || (state.taskId !== null && !creating && !isConnected)}
      onInterrupt={() => {
        if (state.isProcessing || state.currentTask?.status === "running") { pauseTask(); return true }
        return false
      }}
      onSend={async (text, clientMessageId) => {
        const active = owner.current
        if (!active) return
        const startsTask = state.taskId === null
        setCreating(startsTask)
        let delivered = false
        try {
          await sendMessage(text, {
            ...active.config, clientMessageId,
            ...(state.currentTask?.status === "waiting_for_user" && state.currentTask.waitingRequestId
              ? { metadata: { ...(isJsonRecord(active.config.metadata) ? active.config.metadata : {}), request_id: state.currentTask.waitingRequestId } } : {}),
          }, [], {
            canAdoptCreatedTask: () => owner.current === active,
            onCreatedTask: id => { if (owner.current === active) active.createdTaskId = id },
          })
          delivered = true
        } finally {
          if (owner.current === active && (!delivered || !startsTask || active.createdTaskId === null)) {
            // The created task can attach before Next commits its route change.
            // Retain that one permitted route transition until the layout sees it.
            setCreating(false)
          }
        }
      }}
    />
  </TaskVoiceContext.Provider>
}
