import React, { useState, useRef, useEffect, useCallback } from "react"
import { Bot } from "lucide-react"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { ChatInput } from "@/components/chat/ChatInput"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAuth } from "@/contexts/auth-context"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"

import { Interaction } from "@/contexts/app-context-chat"

import { FileAttachment } from "@/components/file/file-attachment"

interface Message {
  role: "user" | "assistant" | "system"
  content: string | React.ReactNode
  traceEvents?: any[]
  timestamp?: number
  interactions?: Interaction[]
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
  toolCategories?: string[]
}

export function AgentBuilderChat({ agentConfig, onUpdateConfig, availableOptions, initialPrompt, toolCategories = [] }: AgentBuilderChatProps) {
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

  const handleSendMessage = useCallback(async (text: string, files?: File[], metadata?: any) => {
    if ((!text.trim() && (!files || files.length === 0)) || isLoading) return

    let displayMessage: string | React.ReactNode = text || t("chatPage.clarification.uploadedFiles")
    if (files && files.length > 0) {
      displayMessage = (
        <div className="space-y-2">
          <div className="whitespace-pre-wrap max-h-60 overflow-y-auto">{text || t("chatPage.clarification.uploadedFiles")}</div>
          <FileAttachment
            files={files.map(f => ({ name: f.name, type: f.type, size: f.size, path: '' }))}
            variant="user-message"
          />
        </div>
      )
    }

    const newMessages: Message[] = [...messages, { role: "user", content: displayMessage, timestamp: Date.now() }]
    setMessages(newMessages)
    setIsLoading(true)

    // Add empty assistant message for streaming
    setMessages(prev => [...prev, { role: "assistant", content: "", traceEvents: [], timestamp: Date.now() }])

    let currentReply = ""
    let finalMessage = text;

    if (files && files.length > 0) {
      try {
        // Create a more meaningful name from the first file, falling back to a generic name
        let baseName = files[0].name.split('.')[0].replace(/[^a-zA-Z0-9_-]/g, '_');
        if (!baseName || baseName.length < 2) baseName = 'documents';
        const collectionName = `${baseName}_${Math.floor(Date.now() / 1000)}`;

        for (const file of files) {
          const formData = new FormData();
          formData.append('file', file);
          formData.append('collection', collectionName);

          const response = await apiRequest(`${getApiUrl()}/api/kb/ingest`, {
            method: 'POST',
            body: formData,
          });
          if (!response.ok) {
            throw new Error(`Upload failed: ${response.statusText}`);
          }
        }

        finalMessage += `\n\n[System Note: The user has uploaded ${files.length} file(s). I have automatically created a knowledge base named "${collectionName}" with these files. You MUST NOT use list_knowledge_bases to check it, just ASSUME it exists. If the agent does not exist yet (no agent ID), use create_agent to build it and include "${collectionName}" in the knowledge_bases list. If it exists, use update_agent.]`;
      } catch (err) {
        console.error("Failed to upload files to KB", err);
        toast.error("Failed to upload files to create knowledge base");
        setIsLoading(false);
        setMessages(prev => prev.slice(0, -1)); // Remove the empty assistant message
        return;
      }
    } else if (metadata?.url) {
      // Extract the URL from the structured metadata instead of matching raw text
      const url = metadata.url;
      finalMessage += `\n\n[System Note: The user has provided the website URL: ${url}. Please IMMEDIATELY use the \`create_knowledge_base_from_url\` tool to ingest it, then create/update the agent with the new knowledge base. Do not ask for the URL again.]`;
    }

    const sendPayload = (ws: WebSocket) => {
      // Add selected MCP servers back into tool_categories
      const finalToolCategories = [...(agentConfig.selectedToolCategories || [])];
      toolCategories.forEach(server => {
        finalToolCategories.push(`mcp:${server}`);
      });

      const { modelConfig, selectedToolCategories, ...restConfig } = agentConfig
      ws.send(JSON.stringify({
        message: finalMessage,
        ...restConfig,
        tool_categories: finalToolCategories,
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

                  let displayReply = currentReply.replace(/```json[\s\S]*?(```|$)/gi, "").trim()
                  let interactions = undefined;

                  // Check if currentReply is directly a JSON object
                  try {
                    const parsed = JSON.parse(currentReply);
                    if (parsed.type === 'chat' && parsed.chat?.interactions) {
                      displayReply = parsed.chat.message || "";
                      interactions = parsed.chat.interactions;
                    }
                  } catch (e) {
                    // Check if there is a JSON block for clarification form
                    const jsonMatch = currentReply.match(/```json\s*([\s\S]*?)\s*```/);
                    if (jsonMatch) {
                      try {
                        const parsed = JSON.parse(jsonMatch[1]);
                        if (parsed.type === 'chat' && parsed.chat?.interactions) {
                          interactions = parsed.chat.interactions;
                          if (parsed.chat.message && !displayReply) {
                            displayReply = parsed.chat.message;
                          }
                        }
                      } catch (e) {
                        // ignore parse errors
                      }
                    }
                  }

                  setMessages(prev => {
                    const updated = [...prev]
                    updated[updated.length - 1].content = displayReply
                    if (interactions) {
                      updated[updated.length - 1].interactions = interactions;
                    }
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
                  if (toolArgs.knowledge_bases) {
                    const kbs = Array.isArray(toolArgs.knowledge_bases) ? toolArgs.knowledge_bases : [toolArgs.knowledge_bases];
                    configUpdates.selectedKbs = kbs.map((kb: any) => typeof kb === 'string' ? kb : kb.name || kb.value).filter(Boolean);
                  }
                  if (toolArgs.skills) {
                    const skills = Array.isArray(toolArgs.skills) ? toolArgs.skills : [toolArgs.skills];
                    configUpdates.selectedSkills = skills.map((skill: any) => typeof skill === 'string' ? skill : skill.name || skill.value).filter(Boolean);
                  }
                  if (toolArgs.tool_categories) {
                    const tcs = Array.isArray(toolArgs.tool_categories) ? toolArgs.tool_categories : [toolArgs.tool_categories];
                    configUpdates.selectedToolCategories = tcs.map((tc: any) => typeof tc === 'string' ? tc : tc.name || tc.category || tc.value).filter(Boolean);
                  }
                  if (toolArgs.suggested_prompts) {
                    const sp = Array.isArray(toolArgs.suggested_prompts) ? toolArgs.suggested_prompts : [toolArgs.suggested_prompts];
                    configUpdates.suggestedPrompts = sp.map((p: any) => typeof p === 'string' ? p : p.value || p.prompt).filter(Boolean);
                  }
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
              let cleanReply = finalContent.replace(/```json[\s\S]*?(```|$)/gi, "").trim()
              let interactions = undefined;

              // Check if finalContent is directly a JSON object
              try {
                const parsed = JSON.parse(finalContent);
                if (parsed.type === 'chat' && parsed.chat?.interactions) {
                  cleanReply = parsed.chat.message || "";
                  interactions = parsed.chat.interactions;
                }
              } catch (e) {
                // If it's not direct JSON, check for markdown JSON blocks
                const jsonMatch = finalContent.match(/```json\s*([\s\S]*?)\s*```/);
                if (jsonMatch) {
                  try {
                    const parsed = JSON.parse(jsonMatch[1]);
                    if (parsed.type === 'chat' && parsed.chat?.interactions) {
                      interactions = parsed.chat.interactions;
                      if (parsed.chat.message && !cleanReply) {
                        cleanReply = parsed.chat.message;
                      }
                    }
                  } catch (e) {
                    // ignore parse errors
                  }
                }
              }

              setMessages(prev => {
                const updated = [...prev]
                updated[updated.length - 1].content = cleanReply || t("builds.configForm.chat.defaultReply") || "I have updated the configuration based on your request."
                if (interactions) {
                  updated[updated.length - 1].interactions = interactions;
                }
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
              interactions={msg.interactions}
              onSendInteraction={(text, files, meta) => handleSendMessage(text, files, meta)}
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
