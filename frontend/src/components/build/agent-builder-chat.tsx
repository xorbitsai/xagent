import React, { useState, useRef, useEffect, useCallback } from "react"
import { Bot, Sparkles } from "lucide-react"
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
}

export interface AgentConfig {
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
  availableOptions?: {
    models: { id: number, name: string }[]
    knowledgeBases: { name: string }[]
    skills: { name: string }[]
    toolCategories: string[]
  }
}

export function AgentBuilderChat({ agentConfig, onUpdateConfig, availableOptions }: AgentBuilderChatProps) {
  const { t } = useI18n()
  const { token } = useAuth()
  const [messages, setMessages] = useState<Message[]>([])

  // Set initial message on mount to avoid hydration mismatch and get translation
  useEffect(() => {
    setMessages([
      {
        role: "assistant",
        content: t("builds.configForm.chat.initialMessage") || "Hello! I am your XAgent Assistant. Describe what kind of agent you want to create, and I'll help you configure it."
      }
    ])
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

    const newMessages: Message[] = [...messages, { role: "user", content: text }]
    setMessages(newMessages)
    setIsLoading(true)

    // Add empty assistant message for streaming
    setMessages(prev => [...prev, { role: "assistant", content: "" }])

    let currentReply = ""

    const sendPayload = (ws: WebSocket) => {
      ws.send(JSON.stringify({
        messages: newMessages.map(m => ({ role: m.role, content: m.content })),
        current_config: agentConfig,
        available_options: availableOptions
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

            if (data.type === "message_delta") {
              currentReply += data.delta

              // Clean up partial or complete JSON blocks during streaming
              const displayReply = currentReply.replace(/```json[\s\S]*?(```|$)/gi, "").trim()

              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1].content = displayReply
                return updated
              })
            } else if (data.type === "message_end") {
              setIsLoading(false)

              if (data.config_updates && Object.keys(data.config_updates).length > 0) {
                onUpdateConfig(data.config_updates)
              }

              // Clean up the JSON block from the final message text to make it look clean
              const cleanReply = currentReply.replace(/```json[\s\S]*?(```|$)/gi, "").trim()
              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1].content = cleanReply || t("builds.configForm.chat.defaultReply") || "I have updated the configuration based on your request."
                return updated
              })

              // Reset reply state for the next message on the same connection
              currentReply = ""
            } else if (data.type === "error") {
              setIsLoading(false)
              toast.error(data.message || t("builds.configForm.chat.errorCommunicate") || "Failed to communicate with XAgent Assistant.")
              ws.close()
            }
          } catch (e) {
            console.error("Error parsing WebSocket message:", e)
          }
        }

        ws.onerror = (error) => {
          console.error("WebSocket error:", error)
          setIsLoading(false)
          toast.error(t("builds.configForm.chat.errorConnection") || "Connection error. Please try again.")
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
          <h3 className="font-semibold text-sm">{t("builds.configForm.chat.title")}</h3>
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
