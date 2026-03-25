"use client"

import React, { useState, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Bot } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"
import { ChatStartScreen } from "@/components/chat/ChatStartScreen"
import { toast } from "sonner"

function getModelDetailUrl(modelId: string | number): string {
  return `${getApiUrl()}/api/models/by-id/${encodeURIComponent(String(modelId))}`
}

interface Agent {
  id: number
  name: string
  description: string
  logo_url: string | null
  instructions: string | null
  execution_mode: string
  suggested_prompts: string[]
  models?: {
    general?: number
    small_fast?: number
    visual?: number
    compact?: number
  }
}

export default function AgentChatPage() {
  const { t } = useI18n()
  const { dispatch, setPendingMessage, setTaskId } = useApp()
  const params = useParams()
  const router = useRouter()
  const agentId = params.id as string

  const [agent, setAgent] = useState<Agent | null>(null)
  const [loading, setLoading] = useState(true)
  const [agentModelId, setAgentModelId] = useState<string>("")
  const [isSending, setIsSending] = useState(false)
  const [inputValue, setInputValue] = useState("")
  const [files, setFiles] = useState<File[]>([])

  // Load agent
  useEffect(() => {
    const fetchAgent = async () => {
      try {
        setLoading(true)
        const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`)
        if (response.ok) {
          const data = await response.json()
          setAgent(data)

          // Fetch model identifier if agent has general model configured
          if (data.models?.general) {
            try {
              const modelResponse = await apiRequest(getModelDetailUrl(data.models.general))
              if (modelResponse.ok) {
                const modelData = await modelResponse.json()
                setAgentModelId(modelData.model_id || modelData.name || "")
              }
            } catch (err) {
              console.error("Failed to load model name:", err)
            }
          }
        } else {
          toast.error(t('builds.list.chat.notFound'))
        }
      } catch (err) {
        console.error("Failed to load agent:", err)
        toast.error(t('builds.list.chat.failed'))
      } finally {
        setLoading(false)
      }
    }

    fetchAgent()
  }, [agentId])

  const handleSendMessage = async (content: string, filesToSend: File[]) => {
    setIsSending(true)
    setInputValue("")

    try {
      // Create task with agent_id
      // Backend will automatically fetch agent's model configuration from database
      const taskResponse = await apiRequest(`${getApiUrl()}/api/chat/task/create`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: content,
          description: content,
          agent_id: parseInt(agentId),
        }),
      })

      if (taskResponse.ok) {
        const taskData = await taskResponse.json()
        const taskId = taskData.id || taskData.task_id

        if (taskId) {
          dispatch({ type: "TRIGGER_TASK_UPDATE" })
          // Set pending message and files for WebSocket to send upon connection
          setPendingMessage({
            message: content,
            files: filesToSend,
            targetTaskId: typeof taskId === 'string' ? parseInt(taskId) : taskId
          })

          // Use setTaskId to trigger context update and redirect
          setTaskId(typeof taskId === 'string' ? parseInt(taskId) : taskId)
          return
        }
      } else {
        const errorData = await taskResponse.json()
        toast.error(t('builds.list.chat.error', { message: errorData.detail || t('builds.list.chat.sendFailed') }))
      }
    } catch (err) {
      console.error("Failed to send message:", err)
      toast.error(t('builds.list.chat.sendFailed'))
    } finally {
      setIsSending(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center">
          <Bot className="h-12 w-12 mx-auto mb-4 animate-pulse text-muted-foreground" />
          <p className="text-muted-foreground">{t('builds.list.chat.loading')}</p>
        </div>
      </div>
    )
  }

  return (
    <>
      {!agent ? <div className="flex items-center justify-center min-h-screen">
        <div className="max-w-md w-full text-center space-y-6">
          {/* Icon */}
          <div className="flex justify-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
              <Bot className="h-6 w-6 text-muted-foreground" />
            </div>
          </div>

          {/* Title */}
          <div className="space-y-2">
            <h2 className="text-lg font-semibold">
              {t('builds.list.chat.notFound')}
            </h2>
            <p className="text-sm text-muted-foreground">
              {t('builds.list.chat.notFoundDescription')}
            </p>
          </div>

          {/* Action */}
          <Button
            className="w-full"
            onClick={() => router.push("/build")}
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t('builds.list.header.create')}
          </Button>
        </div>
      </div> : <div className="h-screen bg-background flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto">
          <main className="container max-w-4xl mx-auto px-4 py-8">
            <ChatStartScreen
              title={agent.name}
              description={agent.description || undefined}
              icon={agent.logo_url ? `${getApiUrl()}${agent.logo_url}` : <Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />}
              prompts={agent.suggested_prompts}
              onSend={(msg, filesToSend) => handleSendMessage(msg, filesToSend)}
              isSending={isSending}
              inputValue={inputValue}
              onInputChange={setInputValue}
              files={files}
              onFilesChange={setFiles}
              readOnlyConfig={true}
              taskConfig={{ model: agentModelId }}
            />
          </main>
        </div>
      </div>}
    </>
  )
}
