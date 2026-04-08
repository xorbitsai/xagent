"use client"

import React, { useState, useEffect, useRef } from "react"
import { useParams, useSearchParams } from "next/navigation"
import { Bot, Send, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { getApiUrl, getWsUrl } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"
import { MarkdownRenderer } from "@/components/ui/markdown-renderer"

export default function WidgetChatPage() {
  const { t } = useI18n()
  const params = useParams()
  const searchParams = useSearchParams()
  const token = params.token as string
  const guestId = searchParams.get("guest_id") || "anonymous"
  const agentId = searchParams.get("agent_id")

  const [messages, setMessages] = useState<{ role: "user" | "assistant", content: string }[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isInitializing, setIsInitializing] = useState(true)
  const [isConnecting, setIsConnecting] = useState(false)
  const [accessToken, setAccessToken] = useState<string | null>(null)
  const [taskId, setTaskId] = useState<number | null>(null)
  const [agentName, setAgentName] = useState<string | null>(null)
  const [agentLogo, setAgentLogo] = useState<string | null>(null)
  const [hasError, setHasError] = useState(false)
  const [isWaitingForResponse, setIsWaitingForResponse] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Initialize widget auth ONLY
  useEffect(() => {
    const initWidget = async () => {
      try {
        // 1. Authenticate and get guest token
        const authRes = await fetch(`${getApiUrl()}/api/widget/auth`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            token,
            guest_id: guestId,
            agent_id: agentId ? parseInt(agentId) : null
          })
        })
        if (!authRes.ok) {
          const errorData = await authRes.json().catch(() => null);
          throw new Error(errorData?.detail || "Widget authentication failed")
        }
        const authData = await authRes.json()
        setAccessToken(authData.access_token)
        if (authData.agent_name) {
          setAgentName(authData.agent_name)
        }
        if (authData.agent_logo) {
          setAgentLogo(authData.agent_logo)
        }
        setHasError(false)

        // Try to load existing task ID from localStorage
        const storageKey = `widget_task_${token}_${guestId}`
        const savedTaskId = localStorage.getItem(storageKey)

        if (savedTaskId) {
          const parsedTaskId = parseInt(savedTaskId, 10)
          setTaskId(parsedTaskId)
          // We don't add welcome message if we have history
          setMessages([])

          // Connect to existing task
          connectWs(parsedTaskId, authData.access_token).then(ws => {
            wsRef.current = ws
          }).catch(err => {
            console.error("Failed to connect to existing task:", err)
            // If connection fails, clear stored task ID and start fresh
            localStorage.removeItem(storageKey)
            setTaskId(null)
            setMessages([{ role: "assistant", content: t("widgetChat.messages.welcome") }])
          })
        } else {
          setMessages([{ role: "assistant", content: t("widgetChat.messages.welcome") }])
        }
      } catch (err) {
        console.error(err)
        setHasError(true)
        setMessages([{ role: "assistant", content: (err as Error).message || t("widgetChat.messages.error_init") }])
      } finally {
        setIsInitializing(false)
      }
    }

    initWidget()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, guestId])

  const connectWs = (currentTaskId: number, currentToken: string): Promise<WebSocket> => {
    return new Promise((resolve, reject) => {
      const wsUrl = `${getWsUrl()}/api/widget/chat/ws/${currentTaskId}?token=${currentToken}`
      const ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        setIsConnecting(false)
        resolve(ws)
      }

      ws.onerror = (error) => {
        if (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.CLOSED) {
          reject(error)
        }
      }

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          if (data.type === "trace_event" && (data.event_type === "task_completed" || data.event_type === "task_completion" || data.event_type === "react_task_end" || data.event_type === "task_end_react")) {
            setIsWaitingForResponse(false)
            let result = data.data.result || data.data.output || data.data.message

            // For task_completion, result might be an object with output field
            if (data.event_type === "task_completion" && result) {
              const resultContent = result.content || result;
              if (typeof resultContent === 'string') {
                try {
                  const parsed = JSON.parse(resultContent);
                  if (parsed && parsed.output) {
                    result = parsed.output;
                  } else {
                    result = resultContent;
                  }
                } catch {
                  result = resultContent;
                }
              } else if (typeof resultContent === 'object' && resultContent.output) {
                result = resultContent.output;
              }
            }

            if (result && typeof result === 'string') {
              setMessages(prev => {
                const lastMessage = prev[prev.length - 1]
                if (lastMessage?.role === "assistant" && lastMessage.content === result) {
                  return prev
                }
                return [...prev, { role: "assistant", content: result }]
              })
            } else if (result && typeof result === 'object' && result.output) {
              setMessages(prev => {
                const lastMessage = prev[prev.length - 1]
                if (lastMessage?.role === "assistant" && lastMessage.content === result.output) {
                  return prev
                }
                return [...prev, { role: "assistant", content: result.output }]
              })
            }
          } else if (data.type === "trace_event" && data.event_type === "agent_error") {
            setIsWaitingForResponse(false)
            setMessages(prev => [...prev, { role: "assistant", content: `${t("widgetChat.messages.error_prefix")} ${data.data.message}` }])
          } else if (data.type === "trace_event" && data.event_type === "user_message") {
            const messageContent = data.data?.message || data.data?.content || ""
            if (messageContent) {
              setMessages(prev => {
                // Avoid duplicating the user message we just optimistically added
                const lastMessage = prev[prev.length - 1]
                if (lastMessage?.role === "user" && lastMessage.content === messageContent) {
                  return prev
                }
                return [...prev, { role: "user", content: messageContent }]
              })
            }
          }
        } catch (e) {
          console.error("Failed to parse WS message", e)
        }
      }

      ws.onclose = () => {
        if (wsRef.current === ws) {
          setIsConnecting(true)
          setTimeout(() => {
            if (wsRef.current === ws) {
              connectWs(currentTaskId, currentToken).then(newWs => {
                wsRef.current = newWs
              }).catch(console.error)
            }
          }, 3000)
        }
      }
    })
  }

  // Cleanup WebSocket on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        const ws = wsRef.current
        wsRef.current = null
        ws.close()
      }
    }
  }, [])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isInitializing || isConnecting || isWaitingForResponse || !accessToken) return

    const messageToSend = inputValue.trim()
    setInputValue("")
    setMessages(prev => [...prev, { role: "user", content: messageToSend }])
    setIsWaitingForResponse(true)

    let currentWs = wsRef.current

    try {
      if (!taskId || !currentWs || currentWs.readyState !== WebSocket.OPEN) {
        setIsConnecting(true)

        let currentTaskId = taskId

        // 1. Create task if we don't have one
        if (!currentTaskId) {
          const taskTitle = messageToSend.substring(0, 50) + (messageToSend.length > 50 ? '...' : '')
          const taskPayload: Record<string, string | number> = { title: taskTitle, description: messageToSend }
          if (agentId) {
            taskPayload.agent_id = parseInt(agentId)
          }

          const taskRes = await fetch(`${getApiUrl()}/api/widget/chat/task/create`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              "Authorization": `Bearer ${accessToken}`
            },
            body: JSON.stringify(taskPayload)
          })
          if (!taskRes.ok) throw new Error("Task creation failed")
          const taskData = await taskRes.json()
          currentTaskId = taskData.task_id
          if (currentTaskId) {
            setTaskId(currentTaskId)
            localStorage.setItem(`widget_task_${token}_${guestId}`, currentTaskId.toString())
          }
        }

        // 2. Connect WS if not connected
        if (!currentWs || currentWs.readyState !== WebSocket.OPEN) {
          currentWs = await connectWs(currentTaskId!, accessToken)
          wsRef.current = currentWs
        }
      }

      // 3. Send message
      currentWs.send(JSON.stringify({
        type: "chat",
        message: messageToSend,
        context: {}
      }))

    } catch (err) {
      console.error("Failed to send message:", err)
      setMessages(prev => [...prev, { role: "assistant", content: t("widgetChat.messages.error_init") }])
      setIsConnecting(false)
      setIsWaitingForResponse(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-background w-full">
      {/* Header */}
      <div className="flex-none p-4 border-b bg-card text-card-foreground shadow-sm z-10">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary/10 text-primary overflow-hidden">
              {agentLogo ? (
                <img
                  src={agentLogo.startsWith('http') ? agentLogo : `${getApiUrl()}${agentLogo.startsWith('/') ? '' : '/'}${agentLogo}`}
                  alt={agentName || "Agent Logo"}
                  className="w-full h-full object-cover"
                />
              ) : (
                <Bot className="w-5 h-5" />
              )}
            </div>
            <div>
              <h1 className="text-sm font-semibold">{agentName || t("widgetChat.title")}</h1>
              {!hasError && (
                <p className="text-xs text-muted-foreground">
                  {isInitializing ? t("widgetChat.status.initializing") : isConnecting ? t("widgetChat.status.connecting") : t("widgetChat.status.online")}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex flex-col max-w-[85%] ${msg.role === "user" ? "ml-auto items-end" : "mr-auto items-start"}`}
          >
            <div
              className={`px-4 py-2 rounded-2xl ${msg.role === "user"
                ? "bg-primary text-primary-foreground rounded-tr-sm"
                : "bg-muted text-foreground rounded-tl-sm"
                }`}
            >
              {msg.role === "assistant" ? (
                <MarkdownRenderer className="text-sm [&>p:first-child]:mt-0 [&>p:last-child]:mb-0" content={msg.content} />
              ) : (
                <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
              )}
            </div>
          </div>
        ))}
        {isWaitingForResponse && (
          <div className="flex flex-col max-w-[85%] mr-auto items-start">
            <div className="px-4 py-3 bg-muted text-foreground rounded-2xl rounded-tl-sm flex items-center gap-1.5 h-[36px]">
              <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
              <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
              <div className="w-1.5 h-1.5 bg-foreground/40 rounded-full animate-bounce"></div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="flex-none p-4 bg-background border-t">
        <div className="relative flex items-end gap-2 bg-muted/50 p-2 rounded-xl border focus-within:ring-1 focus-within:ring-primary/50 transition-all">
          <Textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                handleSendMessage()
              }
            }}
            placeholder={t("widgetChat.input.placeholder")}
            className="min-h-[44px] max-h-[120px] resize-none border-0 bg-transparent focus-visible:ring-0 p-3 shadow-none text-sm"
            rows={1}
            disabled={isInitializing || isConnecting || isWaitingForResponse}
          />
          <Button
            size="icon"
            className="h-9 w-9 shrink-0 rounded-lg mb-1 mr-1"
            onClick={handleSendMessage}
            disabled={!inputValue.trim() || isInitializing || isConnecting || isWaitingForResponse}
          >
            {isConnecting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </div>
        <div className="mt-2 text-center">
          <span className="text-[10px] text-muted-foreground opacity-50">{t("widgetChat.footer.powered_by", { appName: process.env.NEXT_PUBLIC_APP_NAME || "Xagent" })}</span>
        </div>
      </div>
    </div>
  )
}
