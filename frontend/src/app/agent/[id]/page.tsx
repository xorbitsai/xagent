"use client"

import React, { useState, useEffect, useRef } from "react"
import { useParams, useRouter } from "next/navigation"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { ArrowLeft, Bot, Sparkles } from "lucide-react"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { ChatInput } from "@/components/chat/ChatInput"
import { useAuth } from "@/contexts/auth-context"
import { useI18n } from "@/contexts/i18n-context"
import { useApp } from "@/contexts/app-context-chat"

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

interface Message {
  role: "user" | "assistant"
  content: string
  id?: string
  timestamp?: number
}

export default function AgentChatPage() {
  const { token } = useAuth()
  const { t } = useI18n()
  const { dispatch } = useApp()
  const params = useParams()
  const router = useRouter()
  const agentId = params.id as string

  const [agent, setAgent] = useState<Agent | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [agentModelName, setAgentModelName] = useState<string>("")

  const [messages, setMessages] = useState<Message[]>([])
  const [isSending, setIsSending] = useState(false)

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [inputValue, setInputValue] = useState("")

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Load agent
  useEffect(() => {
    const fetchAgent = async () => {
      try {
        setLoading(true)
        const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`)
        if (response.ok) {
          const data = await response.json()
          setAgent(data)

          // Fetch model name if agent has general model configured
          if (data.models?.general) {
            try {
              const modelResponse = await apiRequest(`${getApiUrl()}/api/models/${data.models.general}`)
              if (modelResponse.ok) {
                const modelData = await modelResponse.json()
                setAgentModelName(modelData.model_id || modelData.name || "")
              }
            } catch (err) {
              console.error("Failed to load model name:", err)
            }
          }
        } else {
          setError(t('builds.list.chat.notFound'))
        }
      } catch (err) {
        console.error("Failed to load agent:", err)
        setError(t('builds.list.chat.failed'))
      } finally {
        setLoading(false)
      }
    }

    fetchAgent()
  }, [agentId])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      role: "user",
      content,
      id: `user-${Date.now()}`,
      timestamp: Date.now(),
    }
    setMessages(prev => [...prev, userMessage])
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
          router.push(`/task/${taskId}`)
          return
        }

        // Fallback if no task ID returned
        const assistantMessage: Message = {
          role: "assistant",
          content: t('builds.list.chat.taskCreated'),
          id: `assistant-${Date.now()}`,
          timestamp: Date.now(),
        }
        setMessages(prev => [...prev, assistantMessage])
      } else {
        const errorData = await taskResponse.json()
        const errorMessage: Message = {
          role: "assistant",
          content: t('builds.list.chat.error', { message: errorData.detail || t('builds.list.chat.sendFailed') }),
          id: `assistant-${Date.now()}`,
          timestamp: Date.now(),
        }
        setMessages(prev => [...prev, errorMessage])
      }
    } catch (err) {
      console.error("Failed to send message:", err)
      const errorMessage: Message = {
        role: "assistant",
        content: t('builds.list.chat.sendFailed'),
        id: `assistant-${Date.now()}`,
        timestamp: Date.now(),
      }
      setMessages(prev => [...prev, errorMessage])
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

  if (error || !agent) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="text-center text-destructive">
          <p>{error || t('builds.list.chat.notFound')}</p>
          <Button variant="outline" className="mt-4" onClick={() => router.push("/build")}>
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t('builds.list.header.create')}
          </Button>
        </div>
      </div>
    )
  }

  const hasMessages = messages.length > 0

  return (
    <div className="h-screen bg-background flex flex-col">
      {/* 消息滚动区域 */}
      <div className="flex-1 overflow-y-auto">
        <main className="container max-w-4xl mx-auto px-4 py-8 relative z-0">
          <div className="space-y-6 pb-4">
            {hasMessages ? (
              <>
                {messages.map((msg) => (
                  <ChatMessage
                    key={msg.id}
                    role={msg.role}
                    content={msg.content}
                    traceEvents={[]}
                    showProcessView={false}
                  />
                ))}
                {isSending && !messages[messages.length - 1]?.content?.includes('Task created') && (
                  <ChatMessage
                    role="assistant"
                    content={null}
                    traceEvents={[]}
                    showProcessView={false}
                    isVirtual
                  />
                )}
              </>
            ) : (
              /* 空状态 */
              <div className="flex flex-col items-center justify-center min-h-[80vh] py-16 text-center">
                <div className="relative mb-6">
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-[hsl(var(--gradient-from))]/20 to-[hsl(var(--gradient-to))]/10 flex items-center justify-center animate-float">
                    {agent.logo_url ? (
                      <img
                        src={`${getApiUrl()}${agent.logo_url}`}
                        alt={agent.name}
                        className="w-10 h-10 rounded-lg object-cover"
                      />
                    ) : (
                      <Bot className="w-10 h-10 text-[hsl(var(--gradient-from))]" />
                    )}
                  </div>
                  <div className="absolute -inset-4 rounded-3xl bg-gradient-to-br from-primary/5 via-accent/5 to-transparent blur-xl -z-10" />
                </div>
                <h2 className="text-2xl font-bold mb-2 gradient-text">
                  {agent.name}
                </h2>
                {agent.description && (
                  <p className="text-sm text-muted-foreground/70 mb-8 max-w-md">{agent.description}</p>
                )}

                <div className="mt-8 w-full max-w-4xl mx-auto space-y-8">
                  <ChatInput
                    onSend={handleSendMessage}
                    isLoading={isSending}
                    files={[]}
                    onFilesChange={() => {}}
                    showModeToggle={false}
                    inputValue={inputValue}
                    onInputChange={setInputValue}
                    readOnlyConfig={true}
                    taskConfig={{ model: agentModelName }}
                  />

                  {/* 建议提示词 */}
                  {agent.suggested_prompts && agent.suggested_prompts.length > 0 && (
                    <div className="space-y-4">
                      <div className="flex items-center gap-2 text-sm text-muted-foreground/80 px-1">
                        <Sparkles className="w-4 h-4" />
                        <span>{t('chatPage.page.startWith')}</span>
                      </div>
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        {agent.suggested_prompts.map((prompt, index) => (
                          <div
                            key={index}
                            onClick={() => setInputValue(prompt)}
                            className="group relative p-4 h-24 rounded-xl border border-border/40 bg-card/30 hover:bg-card hover:border-primary/50 cursor-pointer transition-all duration-300 hover:-translate-y-1 hover:shadow-lg flex flex-col justify-center text-left"
                          >
                            <p className="text-sm text-foreground/90 line-clamp-2">{prompt}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </main>
      </div>

      {/* 固定在底部的输入框 */}
      {hasMessages && (
        <div className="flex-shrink-0 z-10 glass pb-6">
          <div className="container max-w-4xl mx-auto px-4">
            <ChatInput
              onSend={handleSendMessage}
              isLoading={isSending}
              files={[]}
              onFilesChange={() => {}}
              showModeToggle={false}
              inputValue={inputValue}
              onInputChange={setInputValue}
              readOnlyConfig={true}
              taskConfig={{ model: agentModelName }}
            />
          </div>
        </div>
      )}
    </div>
  )
}
