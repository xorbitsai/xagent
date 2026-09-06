import React, { useState, useRef, useEffect, useCallback } from "react"
import { Bot } from "lucide-react"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { ChatInput } from "@/components/chat/ChatInput"
import { ScrollArea } from "@/components/ui/scroll-area"
import { useAuth } from "@/contexts/auth-context"
import { getApiUrl, getUploadApiUrl } from "@/lib/utils"
import { apiRequest, getUploadErrorMessage, isJsonRecord, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { normalizeTaskCompletedMessage } from "@/lib/task-completion"
import {
  normalizeTraceProcessStatus,
  type TraceProcessStatus,
} from "@/lib/trace-process-status"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "@/components/ui/sonner"
import { getBrandingFromEnv } from "@/lib/branding"
import { normalizeUploadFileIds } from "@/lib/upload-file-ids"
import { expectsUserResponse, getMessageSurface, isMessageDisplayEventType } from "@/lib/message-surface"
import { isFinalAnswerStreamEventType } from "@/lib/streaming-final-answer"

import { Interaction } from "@/contexts/app-context-chat"

import { FileAttachment } from "@/components/file/file-attachment"

interface Message {
  role: "user" | "assistant" | "system"
  content: string | React.ReactNode
  traceEvents?: any[]
  timestamp?: number
  interactions?: Interaction[]
  processStatus?: TraceProcessStatus
  isPending?: boolean
  isFinalAnswer?: boolean
}

const findLastAssistantIndex = (
  messages: Message[],
  predicate: (message: Message) => boolean = () => true,
  afterIndex = -1,
): number => {
  for (let i = messages.length - 1; i > afterIndex; i--) {
    if (messages[i].role === "assistant" && predicate(messages[i])) return i
  }
  return -1
}

const findLastUserIndex = (messages: Message[]): number => {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "user") return i
  }
  return -1
}

const updateLastAssistantMessage = (
  messages: Message[],
  update: (message: Message) => Message
): Message[] => {
  const pendingIndex = findLastAssistantIndex(messages, message => message.isPending === true)
  const index = pendingIndex >= 0 ? pendingIndex : findLastAssistantIndex(messages)
  if (index >= 0) {
    const updated = [...messages]
    updated[index] = update(messages[index])
    return updated
  }
  return messages
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

interface BuildChatPayload {
  message: string
  id?: number | string
  name: string
  description: string
  instructions: string
  executionMode: string
  suggestedPrompts: string[]
  selectedKbs?: string[]
  selectedSkills?: string[]
  tool_categories: string[]
  models: {
    general?: number | null
    small_fast?: number | null
    visual?: number | null
    compact?: number | null
  }
  files?: { file_id: string; name: string; size: number; type: string }[]
}

type UploadedBuildFile = {
  file_id: string
  name: string
  size: number
  type: string
}

interface AgentBuilderChatProps {
  agentConfig: AgentConfig
  onUpdateConfig: (config: Partial<AgentConfig>) => void
  availableOptions?: any
  toolCategories?: string[]
}

export function AgentBuilderChat({ agentConfig, onUpdateConfig, availableOptions, toolCategories = [] }: AgentBuilderChatProps) {
  const { t } = useI18n()
  const { token } = useAuth()
  const [messages, setMessages] = useState<Message[]>([])
  const [files, setFiles] = useState<File[]>([])
  const branding = getBrandingFromEnv()

  // Set initial message on mount to avoid hydration mismatch and get translation
  useEffect(() => {
    setMessages(prev => {
      if (prev.length > 0) return prev;
      return [
        {
          role: "assistant",
          content: t("builds.configForm.chat.initialMessage", { appName: branding.appName }),
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
    if ((!text.trim() && (!files || files.length === 0)) || isLoading) return false

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
    setMessages(prev => [...prev, {
      role: "assistant",
      content: "",
      traceEvents: [],
      timestamp: Date.now(),
      isPending: true,
    }])

    let currentReply = ""
    let finalMessage = text;
    let uploadedFileIds: UploadedBuildFile[] = [];

    if (files && files.length > 0) {
      try {
        const filesToUpload = files.filter((file) => typeof (file as File & { file_id?: string }).file_id !== "string")
        uploadedFileIds = files
          .map((file) => {
            const fileId = (file as File & { file_id?: string }).file_id
            if (typeof fileId !== "string") return null
            return {
              file_id: fileId,
              name: file.name,
              size: file.size,
              type: file.type || "",
            }
          })
          .filter((file): file is UploadedBuildFile => file !== null)

        if (filesToUpload.length > 0) {
          const formData = new FormData();
          filesToUpload.forEach(f => formData.append('files', f));
          formData.append('task_type', 'task');

          const uploadResponse = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
            method: 'POST',
            body: formData,
          });
          const parsed = await parseApiResponse(uploadResponse);
          if (!uploadResponse.ok) {
            throw new Error(getUploadErrorMessage(uploadResponse, parsed, {
              generic: "Failed to upload files",
              ...UPLOAD_ERROR_MESSAGES,
            }));
          }
          const uploadData = parsed.data;
          if (isJsonRecord(uploadData) && uploadData.success && Array.isArray(uploadData.files)) {
            const normalizedFileIds = normalizeUploadFileIds(
              uploadData.files.map((file) => isJsonRecord(file) ? file.file_id : undefined),
              filesToUpload.length,
            )
            if (!normalizedFileIds) {
              throw new Error("Failed to upload files")
            }
            uploadedFileIds.push(
              ...uploadData.files.map((f: any, index) => ({
                file_id: normalizedFileIds[index],
                name: f.filename || '',
                size: f.file_size || 0,
                type: f.mime_type || '',
              }))
            );
          }
        }

        const normalizedFileIds = normalizeUploadFileIds(
          uploadedFileIds.map((file) => file.file_id),
          files.length,
        )
        if (!normalizedFileIds) {
          throw new Error("Failed to upload files")
        }
        uploadedFileIds = uploadedFileIds.map((file, index) => ({
          ...file,
          file_id: normalizedFileIds[index],
        }))
      } catch (err) {
        console.error("Failed to upload files", err);
        toast.error(err instanceof Error ? err.message : "Failed to upload files");
        setIsLoading(false);
        setMessages(prev => prev.slice(0, -1));
        return false;
      }
    } else if (metadata?.url) {
      const url = metadata.url;
      finalMessage += `\n\n[System Note: The user has provided the website URL: ${url}. Do not ask for the URL again. Before deciding whether to create a new knowledge base, you MUST first call \`list_knowledge_bases\` to check whether a relevant knowledge base for this website/domain already exists. Only if no relevant knowledge base exists should you call \`create_knowledge_base_from_url\`, then create/update the agent with that knowledge base.]`;
    }

    const sendPayload = (ws: WebSocket) => {
      const finalToolCategories = [...(agentConfig.selectedToolCategories || [])];
      toolCategories.forEach(server => {
        finalToolCategories.push(`mcp:${server}`);
      });

      const { modelConfig, selectedToolCategories, ...restConfig } = agentConfig
      const payload: BuildChatPayload = {
        message: finalMessage,
        ...restConfig,
        tool_categories: finalToolCategories,
        models: modelConfig || {}
      }
      if (uploadedFileIds.length > 0) {
        payload.files = uploadedFileIds;
      }
      ws.send(JSON.stringify(payload))
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

            if (isFinalAnswerStreamEventType(data.type)) {
              if (data.type === "final_answer_error") {
                setIsLoading(false)
                setMessages(prev => updateLastAssistantMessage(prev, message => ({
                  ...message,
                  isPending: false,
                  processStatus: "failed",
                })))
                return
              }

              if (data.type === "final_answer_start") {
                currentReply = ""
                return
              }
              if (data.type === "final_answer_delta") {
                currentReply += typeof data.delta === "string" ? data.delta : ""
              } else if (data.type === "final_answer_end") {
                currentReply = typeof data.content === "string" ? data.content : currentReply
              }

              if (currentReply) {
                setMessages(prev => {
                  const updated = [...prev]
                  const lastUserIndex = findLastUserIndex(updated)
                  let index = findLastAssistantIndex(
                    updated,
                    message => message.isPending === true,
                    lastUserIndex,
                  )
                  if (index < 0) {
                    index = findLastAssistantIndex(
                      updated,
                      message => message.isFinalAnswer === true,
                      lastUserIndex,
                    )
                  }
                  if (index < 0) {
                    updated.push({
                      role: "assistant",
                      content: "",
                      traceEvents: [],
                      timestamp: Date.now(),
                      isPending: true,
                    })
                    index = updated.length - 1
                  }
                  updated[index] = {
                    ...updated[index],
                    content: currentReply,
                    isFinalAnswer: true,
                    isPending: data.type !== "final_answer_end",
                  }
                  return updated
                })
              }
            } else if (data.type === "trace_event") {
              const messageSurface = isMessageDisplayEventType(data.event_type)
                ? getMessageSurface(data.event_type, data.data)
                : null
              if (messageSurface === "ignore") {
                return
              }
              // Chat-surface events are attached to the bubble they create
              // below. Other events stay on the active execution bubble.
              if (messageSurface !== "chat") {
                setMessages(prev => {
                  return updateLastAssistantMessage(prev, lastMsg => ({
                    ...lastMsg,
                    traceEvents: [...(lastMsg.traceEvents || []), data]
                  }))
                })
              }

              if (data.event_type === "ai_message" && messageSurface === "chat") {
                if (data.data?.message_type === "reasoning") {
                  // Do not update the main message content for reasoning.
                  // TraceEventRenderer will handle displaying it in the execution logs.
                } else {
                  currentReply = data.data.content || ""

                  let displayReply = currentReply.replace(/```json[\s\S]*?(```|$)/gi, "").trim()
                  let interactions = undefined;

                  // First check if data has structured chat_response (new backend format)
                  if (data.data?.chat_response?.interactions) {
                    interactions = data.data.chat_response.interactions;
                    if (data.data.chat_response.message) {
                      displayReply = data.data.chat_response.message;
                    }
                  } else {
                    // Fallback to checking if currentReply is directly a JSON object
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
                  }

                  setMessages(prev => {
                    const updated = [...prev]
                    let index = findLastAssistantIndex(
                      updated,
                      message => message.isPending === true,
                    )
                    if (index < 0) {
                      updated.push({
                        role: "assistant",
                        content: "",
                        traceEvents: [],
                        timestamp: Date.now(),
                        isPending: true,
                      })
                      index = updated.length - 1
                    }
                    updated[index] = {
                      ...updated[index],
                      content: displayReply,
                      interactions: interactions || updated[index].interactions,
                      traceEvents: [...(updated[index].traceEvents || []), data],
                      isPending: false,
                      isFinalAnswer: true,
                    }
                    return updated
                  })
                }
              } else if (isMessageDisplayEventType(data.event_type) && messageSurface === "chat") {
                const displayReply = data.data?.message || data.data?.content || ""
                const interactions = data.data?.metadata?.interactions
                if (!displayReply) {
                  return
                }
                const waitsForUser = expectsUserResponse(data.event_type, data.data)
                if (waitsForUser) {
                  setIsLoading(false)
                }
                setMessages(prev => {
                  const updated = [...prev]
                  let index = findLastAssistantIndex(
                    updated,
                    message => message.isPending === true,
                  )
                  if (index < 0) {
                    updated.push({
                      role: "assistant",
                      content: "",
                      traceEvents: [],
                      timestamp: Date.now(),
                      isPending: true,
                    })
                    index = updated.length - 1
                  }
                  const target = updated[index]
                  updated[index] = {
                    ...target,
                    content: displayReply,
                    interactions: Array.isArray(interactions) ? interactions : target.interactions,
                    traceEvents: [...(target.traceEvents || []), data],
                    isPending: false,
                    isFinalAnswer: false,
                  }
                  if (!waitsForUser) {
                    updated.push({
                      role: "assistant",
                      content: "",
                      traceEvents: [],
                      timestamp: Date.now(),
                      isPending: true,
                    })
                  }
                  return updated
                })
              } else if (data.event_type === "tool_execution_start") {
                // Update state to indicate tool is running if needed
                console.log("Tool execution started:", data.data)
              } else if (data.event_type === "tool_execution_end") {
                // Tool finished
                console.log("Tool execution ended:", data.data)
                if (data.data && (data.data.tool_name === "create_agent" || data.data.tool_name === "update_agent")) {
                  // Extract configuration updates from tool_args/tool_params and agent_id from result
                  const toolArgs = data.data.tool_args || data.data.tool_params;

                  if (toolArgs && typeof toolArgs === 'object') {
                    const result = data.data.result || {};

                    if (result.status === "success") {
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
                      if (result.agent_id) {
                        configUpdates.id = result.agent_id;
                      }
                      if (Object.keys(configUpdates).length > 0) {
                        onUpdateConfig(configUpdates);
                      }

                      // Update URL if agent was created
                      if (result.agent_id) {
                        const currentUrl = window.location.pathname;
                        if (currentUrl === '/build/new' || currentUrl === '/build') {
                          window.history.replaceState(null, '', `/build/${result.agent_id}`);
                        }
                      }
                    }
                  }
                }
              }
            } else if (data.type === "task_completed") {
              setIsLoading(false)
              const taskCompletion = normalizeTaskCompletedMessage(data)

              // The backend no longer sends config_updates in task_completed.
              // We handle it in tool_execution_end.

              let finalContent = taskCompletion.result && typeof taskCompletion.result === 'object' ? (taskCompletion.result as any).content : taskCompletion.result;
              finalContent = finalContent || currentReply;

              let cleanReply = typeof finalContent === 'string' ? finalContent.replace(/```json[\s\S]*?(```|$)/gi, "").trim() : "";
              let interactions = undefined;
              const resultRecord = taskCompletion.result && typeof taskCompletion.result === 'object'
                ? taskCompletion.result as any
                : null;

              // Check if we have chat_response structure
              if (taskCompletion.chatResponse && typeof taskCompletion.chatResponse === 'object') {
                interactions = (taskCompletion.chatResponse as any).interactions;
                if ((taskCompletion.chatResponse as any).message) {
                  cleanReply = (taskCompletion.chatResponse as any).message;
                }
              } else if (resultRecord?.chat_response) {
                interactions = resultRecord.chat_response.interactions;
                if (resultRecord.chat_response.message) {
                  cleanReply = resultRecord.chat_response.message;
                }
              }

              // Fallback to checking finalContent if interactions is still undefined
              if (!interactions && typeof finalContent === 'string') {
                try {
                  const parsed = JSON.parse(finalContent);
                  if (parsed.type === 'chat' && parsed.chat?.interactions) {
                    cleanReply = parsed.chat.message || cleanReply;
                    interactions = parsed.chat.interactions;
                  }
                } catch (e) {
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
                      // ignore
                    }
                  }
                }
              }

              setMessages(prev => {
                const updated = [...prev]
                const hasFinalContent = typeof cleanReply === "string" && cleanReply.length > 0
                const lastUserIndex = findLastUserIndex(updated)
                const finalIndex = findLastAssistantIndex(
                  updated,
                  message => message.isFinalAnswer === true,
                  lastUserIndex,
                )
                if (finalIndex >= 0) {
                  updated[finalIndex] = {
                    ...updated[finalIndex],
                    content: hasFinalContent ? cleanReply : updated[finalIndex].content,
                    interactions: interactions || updated[finalIndex].interactions,
                    processStatus: taskCompletion.status,
                    isPending: false,
                  }
                  return updated.filter(message => !(
                    message.role === "assistant" &&
                    message.isPending === true &&
                    message.content === ""
                  ))
                }

                if (hasFinalContent) {
                  return updateLastAssistantMessage(updated, message => ({
                    ...message,
                    content: cleanReply,
                    interactions: interactions || message.interactions,
                    processStatus: taskCompletion.status,
                    isPending: false,
                    isFinalAnswer: true,
                  }))
                }

                let meaningfulIndex = -1
                for (let i = updated.length - 1; i > lastUserIndex; i--) {
                  if (updated[i].role === "assistant" && updated[i].content !== "") {
                    meaningfulIndex = i
                    break
                  }
                }
                if (meaningfulIndex >= 0) {
                  updated[meaningfulIndex] = {
                    ...updated[meaningfulIndex],
                    processStatus: taskCompletion.status,
                  }
                  return updated.filter(message => !(
                    message.role === "assistant" &&
                    message.isPending === true &&
                    message.content === ""
                  ))
                }

                return updateLastAssistantMessage(updated, message => ({
                  ...message,
                  content: t("builds.configForm.chat.defaultReply") || "I have updated the configuration based on your request.",
                  interactions: interactions || message.interactions,
                  processStatus: taskCompletion.status,
                  isPending: false,
                }))
              })

              currentReply = ""
            } else if (data.type === "error" || data.type === "task_error") {
              setIsLoading(false)
              const processStatus =
                normalizeTraceProcessStatus(data.task?.status) ||
                normalizeTraceProcessStatus(data.status) ||
                "failed"
              setMessages(prev => {
                return updateLastAssistantMessage(prev, message => ({
                  ...message,
                  processStatus,
                }))
              })
              toast.error(data.message || data.error || t("builds.configForm.chat.errorCommunicate", { appName: branding.appName }))
              ws.close()
            }
          } catch (e) {
            console.error("Error parsing WebSocket message:", e)
          }
        }

        ws.onerror = (error) => {
          console.error("WebSocket error:", error)
          setIsLoading(false)
          toast.error(t("builds.configForm.chat.errorConnection", { appName: branding.appName }))
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
      return false
    }
    return true
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
          <h3 className="font-semibold text-sm">{t("builds.configForm.chat.title", { appName: branding.appName })}</h3>
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
              processStatus={msg.processStatus}
              timestamp={msg.timestamp}
              interactions={msg.interactions}
              onSendInteraction={async (text, files, meta) => {
                const didSend = await handleSendMessage(text, files, meta)
                if (!didSend) {
                  throw new Error("Failed to send interaction")
                }
              }}
            />
          ))}
        </div>
      </ScrollArea>

      <div className="p-4 bg-background border-t">
        <ChatInput
          onSend={async (text) => {
            const didSend = await handleSendMessage(text, files)
            if (didSend) {
              setFiles([])
            }
          }}
          isLoading={isLoading}
          hideConfig={true}
          compact={true}
          files={files}
          onFilesChange={setFiles}
        />
      </div>
    </div>
  )
}
