import React, { useState, useRef, useEffect, useCallback } from "react"
import { Bot } from "lucide-react"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { ChatInput } from "@/components/chat/ChatInput"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAuth } from "@/contexts/auth-context"
import { getApiUrl } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"

interface Message {
  role: "user" | "assistant" | "system"
  content: string
  traceEvents?: any[]
  timestamp?: number
}

export interface AgentConfig {
  id?: number | string
  name: string
  description: string
  instructions: string
  executionMode: string
  suggestedPrompts: string[]
  modelConfig?: {
    general: number | null
    small_fast: number | null
    visual: number | null
    compact: number | null
  }
  selectedKbs?: string[]
  selectedSkills?: string[]
  selectedToolCategories?: string[]
}

interface AgentBuilderChatProps {
  agentConfig: AgentConfig
  onUpdateConfig: (config: Partial<AgentConfig>) => void
  availableOptions?: any
  initialPrompt?: string | null
}

export function AgentBuilderChat({ agentConfig, onUpdateConfig, availableOptions, initialPrompt }: AgentBuilderChatProps) {
  const { t } = useI18n()
  const { token } = useAuth()
  const [messages, setMessages] = useState<Message[]>([])
  const [hasSentInitial, setHasSentInitial] = useState(false)

  // Set initial message on mount to avoid hydration mismatch and get translation
  useEffect(() => {
    setMessages(prev => {
      if (prev.length > 0) return prev;
      return [
        {
          role: "assistant",
          content: t("builds.configForm.chat.initialMessage", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" }),
          timestamp: Date.now()
        }
      ]
    })
  }, [t])

  const [isLoading, setIsLoading] = useState(false)
  const scrollRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [])

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      const scrollElement = scrollRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollElement) {
        scrollElement.scrollTop = scrollElement.scrollHeight
      } else {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight
      }
    }
  }, [messages])

  const handleSendMessage = useCallback((text: string) => {
    if (!text.trim() || isLoading) return

    const newMessages: Message[] = [...messages, { role: "user", content: text, timestamp: Date.now() }]
    setMessages(newMessages)
    setIsLoading(true)

    // Add empty assistant message for streaming
    setMessages(prev => [...prev, { role: "assistant", content: "", traceEvents: [], timestamp: Date.now() }])

    let currentReply = ""

    const sendPayload = (ws: WebSocket) => {
      const { modelConfig, ...restConfig } = agentConfig
      ws.send(JSON.stringify({
        message: text,
        ...restConfig,
        models: modelConfig || {}
      }))
    }

    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        // Reuse existing connection
        sendPayload(wsRef.current)
      } else {
        // Create new connection if none exists or it was closed
        const wsUrl = getApiUrl().replace(/^http/, "ws") + `/ws/build/chat?token=${token}`
        const ws = new WebSocket(wsUrl)
        wsRef.current = ws

        ws.onopen = () => {
          sendPayload(ws)
        }

        ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)

            if (data.type === "trace_event") {
              // Update the last message (assistant) with the new trace event
              setMessages(prev => {
                const updated = [...prev]
                const lastMsg = updated[updated.length - 1]
                if (lastMsg && lastMsg.role === 'assistant') {
                  updated[updated.length - 1] = {
                    ...lastMsg,
                    traceEvents: [...(lastMsg.traceEvents || []), data]
                  }
                }
                return updated
              })

              if (data.event_type === "ai_message") {
                if (data.data?.message_type === "reasoning") {
                  // Do not update the main message content for reasoning.
                  // TraceEventRenderer will handle displaying it in the execution logs.
                } else {
                  currentReply = data.data.content || ""

                  const displayReply = currentReply.replace(/```json[\s\S]*?(```|$)/gi, "").trim()

                  setMessages(prev => {
                    const updated = [...prev]
                    updated[updated.length - 1].content = displayReply
                    return updated
                  })
                }
              } else if (data.event_type === "tool_execution_start") {
                // Update state to indicate tool is running if needed
                console.log("Tool execution started:", data.data)
              } else if (data.event_type === "tool_execution_end") {
                // Tool finished
                console.log("Tool execution ended:", data.data)
                if (data.data && (data.data.tool_name === "create_agent" || data.data.tool_name === "update_agent") && data.data.tool_args && typeof data.data.tool_args === 'object') {
                  // Extract configuration updates from tool_args and agent_id from result
                  const toolArgs = data.data.tool_args;
                  const result = data.data.result || {};

                  const configUpdates: Partial<AgentConfig> = {};
                  if (toolArgs.name) configUpdates.name = toolArgs.name;
                  if (toolArgs.description) configUpdates.description = toolArgs.description;
                  if (toolArgs.instructions) configUpdates.instructions = toolArgs.instructions;
                  if (result.status === "success" && result.agent_id) {
                    configUpdates.id = result.agent_id;
                  }
                  if (Object.keys(configUpdates).length > 0) {
                    onUpdateConfig(configUpdates);
                  }

                  // Update URL if agent was created
                  if (result.status === "success" && result.agent_id) {
                    const currentUrl = window.location.pathname;
                    if (currentUrl === '/build/new' || currentUrl === '/build') {
                      window.history.pushState({}, '', `/build/${result.agent_id}`);
                    }
                  }
                }
              }
            } else if (data.type === "task_completed") {
              setIsLoading(false)

              // The backend no longer sends config_updates in task_completed.
              // We handle it in tool_execution_end.

              const finalContent = data.result || currentReply;
              const cleanReply = finalContent.replace(/```json[\s\S]*?(```|$)/gi, "").trim()
              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1].content = cleanReply || t("builds.configForm.chat.defaultReply") || "I have updated the configuration based on your request."
                return updated
              })

              currentReply = ""
            } else if (data.type === "error" || data.type === "task_error") {
              setIsLoading(false)
              toast.error(data.message || data.error || t("builds.configForm.chat.errorCommunicate", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" }))
              ws.close()
            }
          } catch (e) {
            console.error("Error parsing WebSocket message:", e)
          }
        }

        ws.onerror = (error) => {
          console.error("WebSocket error:", error)
          setIsLoading(false)
          toast.error(t("builds.configForm.chat.errorConnection", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" }) || "Connection error. Please try again.")
        }

        ws.onclose = () => {
          setIsLoading(false)
          wsRef.current = null
        }
      }
    } catch (error) {
      console.error(error)
      toast.error(t("builds.configForm.chat.errorInit") || "Failed to initialize connection.")
      setIsLoading(false)
    }
  }, [messages, isLoading, token, agentConfig, onUpdateConfig])

  // Handle initial prompt from URL
  useEffect(() => {
    if (initialPrompt && !hasSentInitial && token && messages.length > 0 && !isLoading) {
      setHasSentInitial(true)
      handleSendMessage(initialPrompt)
    }
  }, [initialPrompt, hasSentInitial, token, messages.length, isLoading, handleSendMessage])

  const handleStop = () => {
    if (wsRef.current) {
      wsRef.current.close()
      setIsLoading(false)
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 h-full bg-muted/10 border-r">
      <div className="flex items-center gap-2 px-4 py-3 border-b bg-background">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
          <Bot className="h-5 w-5" />
        </div>
        <div>
          <h3 className="font-semibold text-sm">{t("builds.configForm.chat.title", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" })}</h3>
          <p className="text-xs text-muted-foreground">{t("builds.configForm.chat.subtitle")}</p>
        </div>
      </div>

      <ScrollArea className="flex-1 min-h-0 p-4" ref={scrollRef}>
        <div className="flex flex-col gap-4 pb-4">
          {messages.map((msg, index) => (
            <ChatMessage
              key={index}
              role={msg.role}
              content={msg.content}
              traceEvents={msg.traceEvents}
              showProcessView={true}
              timestamp={msg.timestamp}
            />
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 bg-background border-t">
        <ChatInput
          onSend={(text) => handleSendMessage(text)}
          isLoading={isLoading}
          hideConfig={true}
          hideFileUpload={true}
          compact={true}
        />
      </div>
    </div>
  )
}
