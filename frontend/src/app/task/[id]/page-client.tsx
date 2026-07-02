"use client"

import { Suspense, useEffect } from "react"
import { ArrowLeft, Loader2 } from "lucide-react"
import { useParams, useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { TaskConversationPanel } from "@/components/task/task-conversation-panel"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { getApiUrl } from "@/lib/utils"

function TaskDetailContent() {
  const { state, setTaskId, closeFilePreview } = useApp()
  const { t } = useI18n()
  const params = useParams()
  const router = useRouter()
  const taskIdFromUrl = params.id

  useEffect(() => {
    if (taskIdFromUrl && typeof taskIdFromUrl === "string") {
      const taskIdNum = parseInt(taskIdFromUrl, 10)
      if (!Number.isNaN(taskIdNum) && taskIdNum !== state.taskId) {
        setTaskId(taskIdNum)
      }
    }
  }, [taskIdFromUrl, setTaskId, state.taskId])

  useEffect(() => {
    return () => {
      closeFilePreview()
    }
  }, [closeFilePreview])

  return (
    <div className="h-full flex flex-col bg-background">
      {state.currentTask?.agentId && (
        <div className="flex-none flex items-center gap-3 px-4 py-3 border-b bg-background/95 backdrop-blur z-50 sticky top-0">
          <Button
            variant="ghost"
            size="icon"
            className="rounded-full hover:bg-muted flex-shrink-0"
            onClick={() => {
              const agentId = state.currentTask?.agentId
              router.push(agentId ? `/agent/${agentId}` : "/task")
            }}
            title={t("common.back")}
          >
            <ArrowLeft className="w-5 h-5" />
          </Button>
          {(state.currentTask?.agentName || state.currentTask?.agentLogoUrl) && (
            <div className="flex items-center gap-3 overflow-hidden">
              {state.currentTask?.agentLogoUrl ? (
                <img
                  src={state.currentTask.agentLogoUrl.startsWith('http') ? state.currentTask.agentLogoUrl : `${getApiUrl()}${state.currentTask.agentLogoUrl}`}
                  alt={state.currentTask.agentName || t("agent.logo")}
                  className="w-8 h-8 object-cover rounded-sm flex-shrink-0"
                />
              ) : null}
              {state.currentTask?.agentName ? (
                <span className="text-lg font-bold text-foreground truncate">{state.currentTask.agentName}</span>
              ) : null}
            </div>
          )}
        </div>
      )}
      <div className="flex-1 min-h-0 relative">
        <TaskConversationPanel mode="page" />
      </div>
    </div>
  )
}

export default function TaskDetailPage() {
  return (
    <Suspense fallback={<div className="flex items-center justify-center h-full"><Loader2 className="w-8 h-8 animate-spin" /></div>}>
      <TaskDetailContent />
    </Suspense>
  )
}
