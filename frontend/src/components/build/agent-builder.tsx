"use client"

import React, { useState, useEffect, useRef, useMemo } from "react"
import { ResizableSplitLayout } from "@/components/layout/resizable-split-layout"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ChatInput } from "@/components/chat/ChatInput"
import { ChatMessage } from "@/components/chat/ChatMessage"
import { apiRequest } from "@/lib/api-wrapper"
import { getApiUrl, getWsUrl } from "@/lib/utils"
import { PlusCircle, MessageSquare, Upload, Download, Info } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"
import { useAuth } from "@/contexts/auth-context"
import { FileAttachment } from "@/components/file-attachment"
import { MultiSelect } from "@/components/ui/multi-select"
import { Select } from "@/components/ui/select"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { useRouter, useSearchParams } from "next/navigation"

interface KnowledgeBase {
  name: string
  [key: string]: any
}

interface Skill {
  name: string
  description?: string
  when_to_use?: string
  tags?: string[]
  [key: string]: any
}

interface Tool {
  name: string
  description: string
  type: string
  category: string
  enabled: boolean
  [key: string]: any
}

interface Model {
  id: number
  model_id: string
  model_name: string
  model_provider: string
  category: string
}

interface UserDefaultModel {
  id: number
  config_type: string
  model: {
    id: number
    model_id: string
    model_name: string
    model_provider: string
  }
}

interface AgentModelConfig {
  general: number | null
  small_fast: number | null
  visual: number | null
  compact: number | null
}

interface Message {
  role: "user" | "assistant" | "system"
  content: string | React.ReactNode
  traceEvents?: any[]
  timestamp?: number
}

interface AgentBuilderProps {
  agentId?: string
}

export function AgentBuilder({ agentId }: AgentBuilderProps) {
  const { t, locale } = useI18n()
  const { token } = useAuth()
  const router = useRouter()
  const searchParams = useSearchParams()
  const templateId = searchParams.get("template")
  const isEditMode = !!agentId

  // Config State
  const [name, setName] = useState("")
  const [description, setDescription] = useState("")
  const [instructions, setInstructions] = useState("")
  const [executionMode, setExecutionMode] = useState("react") // "simple", "react", "graph"
  const [suggestedPrompts, setSuggestedPrompts] = useState<string[]>([])
  const [modelConfig, setModelConfig] = useState<AgentModelConfig>({
    general: null,
    small_fast: null,
    visual: null,
    compact: null,
  })
  const [selectedKbs, setSelectedKbs] = useState<string[]>([])
  const [selectedSkills, setSelectedSkills] = useState<string[]>([])
  const [selectedToolCategories, setSelectedToolCategories] = useState<string[]>([])
  const [logoFile, setLogoFile] = useState<File | null>(null)
  const [logoUrl, setLogoUrl] = useState<string | null>(null)  // Existing logo URL
  const [isCreating, setIsCreating] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [loadingAgent, setLoadingAgent] = useState(false)
  const [originalData, setOriginalData] = useState<any>(null)

  // Create Success Dialog State
  const [showSuccessDialog, setShowSuccessDialog] = useState(false)
  const [createdAgent, setCreatedAgent] = useState<any>(null)

  // Data State
  const [models, setModels] = useState<Model[]>([])
  const [kbs, setKbs] = useState<KnowledgeBase[]>([])
  const [skills, setSkills] = useState<Skill[]>([])
  const [tools, setTools] = useState<Tool[]>([])

  // Chat State
  const [messages, setMessages] = useState<Message[]>([])

  useEffect(() => {
    setMessages([{
      role: "assistant",
      content: t("builds.preview.initialMessage")
    }])
  }, [t])

  const [isChatLoading, setIsChatLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [files, setFiles] = useState<File[]>([])

  // WebSocket for preview
  const [wsConnected, setWsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const previewStepsRef = useRef<any[]>([])
  const traceEventsRef = useRef<any[]>([])
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5

  // Setup WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      if (!token) {
        console.log('⏳ Waiting for token to connect to WS...')
        return
      }

      // Clear any existing reconnect timeout
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
        reconnectTimeoutRef.current = null
      }

      const baseUrl = getWsUrl()
      const wsUrl = `${baseUrl}/ws/build/preview?token=${token}`
      console.log('🔌 Connecting to Build Preview WS:', wsUrl)

      try {
        const ws = new WebSocket(wsUrl)

        ws.onopen = () => {
          console.log('✅ Build preview WebSocket connected')
          setWsConnected(true)
          wsRef.current = ws
          reconnectAttemptsRef.current = 0
        }

        ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data)
            console.log('Build preview WebSocket message:', message)

            // Handle different message types
            if (message.type === 'preview_started') {
              setIsChatLoading(true)
              previewStepsRef.current = []
              traceEventsRef.current = []
              // Add a placeholder message for the assistant response
              setMessages(prev => [...prev, {
                role: "assistant",
                content: "",
                traceEvents: [],
                timestamp: Date.now()
              }])
            } else if (message.type === 'trace_event') {
              // Collect trace events and steps
              traceEventsRef.current.push(message)
              if (message.event_type === 'dag_step_start' || message.event_type === 'dag_step_end') {
                previewStepsRef.current.push(message)
              }
              // Update the last message (assistant) with the new trace event
              setMessages(prev => {
                const newMessages = [...prev]
                const lastMsg = newMessages[newMessages.length - 1]
                if (lastMsg && lastMsg.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMsg,
                    traceEvents: [...(lastMsg.traceEvents || []), message]
                  }
                  return newMessages
                }
                return prev
              })
            } else if (message.type === 'task_completed') {
              setIsChatLoading(false)
              setMessages(prev => {
                const newMessages = [...prev]
                const lastMsg = newMessages[newMessages.length - 1]
                if (lastMsg && lastMsg.role === 'assistant') {
                  newMessages[newMessages.length - 1] = {
                    ...lastMsg,
                    content: message.result || message.output || "Preview completed"
                  }
                  return newMessages
                }
                return prev
              })
            } else if (message.type === 'task_error') {
              setIsChatLoading(false)
              setMessages(prev => [...prev, {
                role: "assistant",
                content: `Error: ${message.error}`
              }])
            }
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error)
          }
        }

        ws.onerror = (error) => {
          console.error('Build preview WebSocket error:', error)
          // Don't set connected false here, let onclose handle it
        }

        ws.onclose = (event) => {
          console.log('Build preview WebSocket closed', event.code, event.reason)
          setWsConnected(false)
          wsRef.current = null

          // Don't reconnect if component unmounted or token changed (handled by cleanup)
          // Retry logic
          if (reconnectAttemptsRef.current < maxReconnectAttempts) {
            reconnectAttemptsRef.current++
            const delay = Math.min(1000 * reconnectAttemptsRef.current, 5000)
            console.log(`🔄 Reconnecting in ${delay}ms... (Attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts})`)
            reconnectTimeoutRef.current = setTimeout(connectWebSocket, delay)
          } else {
            console.log('❌ Max reconnect attempts reached')
          }
        }
      } catch (error) {
        console.error('Failed to create WebSocket:', error)
        // Retry immediately if creation failed
        if (reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          reconnectTimeoutRef.current = setTimeout(connectWebSocket, 1000)
        }
      }
    }

    connectWebSocket()

    return () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current)
      }
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [token])

  // Fetch Data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [kbRes, skillsRes, toolsRes, modelsRes, userDefaultsRes] = await Promise.all([
          apiRequest(`${getApiUrl()}/api/kb/collections`),
          apiRequest(`${getApiUrl()}/api/skills/`),
          apiRequest(`${getApiUrl()}/api/tools/available`),
          apiRequest(`${getApiUrl()}/api/models/?category=llm`),
          apiRequest(`${getApiUrl()}/api/models/user-default`)
        ])

        if (kbRes.ok) {
          const kbData = await kbRes.json()
          setKbs(kbData.collections || [])
        }

        if (skillsRes.ok) {
          const skillsData = await skillsRes.json()
          console.log("Skills API response:", skillsData)
          setSkills(skillsData || [])
        } else {
          console.error("Skills API failed:", skillsRes.status, await skillsRes.text())
        }

        if (toolsRes.ok) {
          const toolsData = await toolsRes.json()
          // Filter only enabled tools
          setTools((toolsData.tools || []).filter((t: Tool) => t.enabled))
        }

        let availableModels: Model[] = []
        if (modelsRes.ok) {
          availableModels = await modelsRes.json()
          setModels(availableModels || [])
        }

        if (userDefaultsRes.ok) {
          const userDefaults = await userDefaultsRes.json()

          // Set model config based on user defaults (only for new agent)
          if (!isEditMode) {
            const config: AgentModelConfig = {
              general: null,
              small_fast: null,
              visual: null,
              compact: null,
            }

            for (const m of userDefaults) {
              if (m.config_type === 'general') config.general = m.model.id
              else if (m.config_type === 'small_fast') config.small_fast = m.model.id
              else if (m.config_type === 'visual') config.visual = m.model.id
              else if (m.config_type === 'compact') config.compact = m.model.id
            }

            // Fallback: If no general model set, pick first available LLM
            if (!config.general && availableModels.length > 0) {
              // models endpoint was called with ?category=llm so these should be LLMs
              const firstLlm = availableModels[0]
              if (firstLlm) {
                config.general = firstLlm.id
              }
            }

            setModelConfig(config)
          }
        }
      } catch (error) {
        console.error("Failed to fetch data:", error)
      }
    }

    fetchData()
  }, [])

  // Load agent data in edit mode
  useEffect(() => {
    if (!isEditMode || !agentId) return

    const loadAgent = async () => {
      try {
        setLoadingAgent(true)
        const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}`)
        if (response.ok) {
          const agent = await response.json()
          setOriginalData(agent)
          setName(agent.name || "")
          setDescription(agent.description || "")
          setInstructions(agent.instructions || "")
          setExecutionMode(agent.execution_mode || "graph")
          setSuggestedPrompts(agent.suggested_prompts || [])
          setSelectedKbs(agent.knowledge_bases || [])
          setSelectedSkills(agent.skills || [])
          setSelectedToolCategories(agent.tool_categories || [])
          setLogoUrl(agent.logo_url || null)

          // Load models
          if (agent.models) {
            setModelConfig({
              general: agent.models.general || null,
              small_fast: agent.models.small_fast || null,
              visual: agent.models.visual || null,
              compact: agent.models.compact || null,
            })
          }
        }
      } catch (error) {
        console.error("Failed to load agent:", error)
      } finally {
        setLoadingAgent(false)
      }
    }

    loadAgent()
  }, [isEditMode, agentId])

  // Load template data when template parameter is present
  useEffect(() => {
    if (!templateId || isEditMode) return

    const loadTemplate = async () => {
      try {
        setLoadingAgent(true)
        const response = await apiRequest(
          `${getApiUrl()}/api/templates/${templateId}`
        )
        if (response.ok) {
          const template = await response.json()
          setName(template.name || "")
          setDescription(template.description || "")
          setInstructions(template.agent_config?.instructions || "")
          setSelectedSkills(template.agent_config?.skills || [])
          setSelectedToolCategories(template.agent_config?.tool_categories || [])
        }
      } catch (error) {
        console.error("Failed to load template:", error)
      } finally {
        setLoadingAgent(false)
      }
    }

    loadTemplate()
  }, [templateId, isEditMode, locale])

  // Convert kbs to MultiSelect options
  const kbOptions = kbs.map((kb) => ({
    value: kb.name,
    label: kb.name,
  }))

  // Convert skills to MultiSelect options
  const skillOptions = skills.map((skill) => ({
    value: skill.name,
    label: skill.name,
    description: skill.description || skill.when_to_use || undefined,
  }))

  const modelOptions = [
    { value: "", label: "--" },
    ...models.map((model) => ({
      value: model.id.toString(),
      label: model.model_name,
    }))
  ]

  // Group tools by category for category selection
  const toolCategories = Array.from(
    new Set(tools.map(t => t.category))
  ).sort()

  const toolCategoryOptions = toolCategories.map(category => {
    const toolsInCategory = tools.filter(t => t.category === category)
    return {
      value: category,
      label: getCategoryLabel(category),
      description: `${toolsInCategory.length} tools: ${toolsInCategory.map(t => t.name).join(', ')}`
    }
  })

  // Helper function for category labels
  function getCategoryLabel(category: string): string {
    const labels: Record<string, string> = {
      'basic': t('builds.configForm.tools.categories.basic'),
      'file': t('builds.configForm.tools.categories.file'),
      'vision': t('builds.configForm.tools.categories.vision'),
      'image': t('builds.configForm.tools.categories.image'),
      'knowledge': t('builds.configForm.tools.categories.knowledge'),
      'mcp': t('builds.configForm.tools.categories.mcp'),
      'browser': t('builds.configForm.tools.categories.browser'),
      'ppt': t('builds.configForm.tools.categories.ppt'),
      'office': t('builds.configForm.tools.categories.office'),
      'special_image': t('builds.configForm.tools.categories.specialImage'),
    }
    return labels[category] || category
  }

  // Auto-scroll chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  const [previewState, setPreviewState] = useState<{
    isOpen: boolean;
    fileUrl?: string;
    fileName?: string;
    fileType?: string;
  }>({ isOpen: false });

  const handlePreviewFile = (url: string, name: string, type: string) => {
    setPreviewState({
      isOpen: true,
      fileUrl: url,
      fileName: name,
      fileType: type
    });
  };

  const handleDownloadFile = () => {
    if (!previewState.fileUrl || !previewState.fileName) return;
    const a = document.createElement('a');
    a.href = previewState.fileUrl;
    a.download = previewState.fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  const handleSendMessage = async (content: string, _config?: any) => {
    // Construct UI message with files if present
    let uiContent: React.ReactNode = content
    if (files.length > 0) {
       // Create object URLs for local preview
       const fileInfos = files.map(f => ({
         name: f.name,
         size: f.size,
         type: f.type,
         path: URL.createObjectURL(f)
       }));

       uiContent = (
         <div className="space-y-2">
           <div>{content}</div>
           <FileAttachment
             files={fileInfos}
             variant="user-message"
             onPreview={(file) => {
               if (file.path) {
                 handlePreviewFile(file.path, file.name, file.type);
               }
             }}
           />
         </div>
       )
     }

    setMessages(prev => [...prev, { role: "user", content: uiContent, timestamp: Date.now() }])
    setIsChatLoading(true)

    try {
      // Check if general model is selected
      if (!modelConfig.general) {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: t("builds.preview.errors.noModel")
        }])
        setIsChatLoading(false)
        return
      }

      // Check WebSocket connection
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setMessages(prev => [...prev, {
          role: "assistant",
          content: "⚠️ WebSocket not connected. The system is attempting to reconnect. Please wait a moment and try again."
        }])
        setIsChatLoading(false)
        return
      }

      // Process files if any
      let processedFiles: any[] = []
      if (files.length > 0) {
        processedFiles = await Promise.all(files.map(async (file) => ({
          name: file.name,
          type: file.type,
          content: await fileToBase64(file),
          size: file.size
        })))
      }

      // Ensure message is not empty for backend
      let backendMessage = content
      if (!backendMessage.trim() && processedFiles.length > 0) {
        backendMessage = `Uploaded files: ${processedFiles.map(f => f.name).join(', ')}`
      }

      // Send preview request via WebSocket
      wsRef.current.send(JSON.stringify({
        type: "preview",
        agent_id: agentId && typeof agentId === 'string' ? parseInt(agentId) : null,  // Exclude this agent from agent tools if published
        instructions,
        execution_mode: executionMode,
        models: modelConfig,
        knowledge_bases: selectedKbs,
        skills: selectedSkills,
        tool_categories: selectedToolCategories,
        message: backendMessage,
        files: processedFiles
      }))

      setFiles([])

    } catch (error) {
      console.error("Preview failed:", error)
      setMessages(prev => [...prev, {
        role: "assistant",
        content: t("builds.preview.errors.requestFailed")
      }])
      setIsChatLoading(false)
    }
  }

  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setLogoFile(e.target.files[0])
    }
  }

  const fileToBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result as string
        resolve(result)
      }
      reader.onerror = reject
      reader.readAsDataURL(file)
    })
  }

  const isDirty = useMemo(() => {
    if (!originalData) return false

    // Helper to normalize arrays for comparison
    const normalize = (arr: any[]) => [...(arr || [])].sort().join(',')

    // Helper to normalize prompts (filter empty)
    const normalizePrompts = (arr: string[]) =>
      [...(arr || [])].filter(p => p.trim()).sort().join(',')

    // Compare basic fields
    if (name !== (originalData.name || "")) return true
    if ((description || "") !== (originalData.description || "")) return true
    if ((instructions || "") !== (originalData.instructions || "")) return true
    if (executionMode !== (originalData.execution_mode || "graph")) return true

    // Compare logo
    if (logoFile) return true

    // Compare arrays
    if (normalizePrompts(suggestedPrompts) !== normalizePrompts(originalData.suggested_prompts)) return true
    if (normalize(selectedKbs) !== normalize(originalData.knowledge_bases)) return true
    if (normalize(selectedSkills) !== normalize(originalData.skills)) return true
    if (normalize(selectedToolCategories) !== normalize(originalData.tool_categories)) return true

    // Compare models
    const origModels = originalData.models || {}
    if ((modelConfig.general || null) !== (origModels.general || null)) return true
    if ((modelConfig.small_fast || null) !== (origModels.small_fast || null)) return true
    if ((modelConfig.visual || null) !== (origModels.visual || null)) return true
    if ((modelConfig.compact || null) !== (origModels.compact || null)) return true

    return false
  }, [name, description, instructions, executionMode, logoFile, suggestedPrompts, selectedKbs, selectedSkills, selectedToolCategories, modelConfig, originalData])

  const handleCreate = async () => {
    // Clear previous error
    setErrorMessage(null)

    // Validation
    if (!name.trim()) {
      setErrorMessage(t("builds.editor.validation.nameRequired"))
      return
    }

    if (!modelConfig.general) {
      setErrorMessage(t("builds.editor.validation.modelRequired"))
      return
    }

    setIsCreating(true)

    try {
      // Convert logo to base64 if provided
      let logo_base64: string | undefined
      if (logoFile) {
        logo_base64 = await fileToBase64(logoFile)
      }

      const url = isEditMode && agentId
        ? `${getApiUrl()}/api/agents/${agentId}`
        : `${getApiUrl()}/api/agents`

      const method = isEditMode ? "PUT" : "POST"

      const response = await apiRequest(url, {
        method,
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          name: name.trim(),
          description: description.trim() || undefined,
          instructions: instructions.trim() || undefined,
          execution_mode: executionMode,
          suggested_prompts: suggestedPrompts.filter(p => p.trim()),
          models: modelConfig,
          knowledge_bases: selectedKbs,
          skills: selectedSkills,
          tool_categories: selectedToolCategories,
          logo_base64,
        }),
      })

      if (response.ok) {
        if (isEditMode) {
          const trimmedName = name.trim()
          const trimmedDesc = description.trim()
          const trimmedInstr = instructions.trim()
          const trimmedPrompts = suggestedPrompts.filter(p => p.trim())

          // Update local state to match saved data
          setName(trimmedName)
          setDescription(trimmedDesc)
          setInstructions(trimmedInstr)
          setSuggestedPrompts(trimmedPrompts)

          // Update original data to reflect saved state
          setOriginalData({
            ...originalData,
            name: trimmedName,
            description: trimmedDesc || undefined,
            instructions: trimmedInstr || undefined,
            execution_mode: executionMode,
            suggested_prompts: trimmedPrompts,
            models: modelConfig,
            knowledge_bases: selectedKbs,
            skills: selectedSkills,
            tool_categories: selectedToolCategories,
          })
          setLogoFile(null)
          // Optional: Reload agent to get updated logo URL if needed, but avoiding it keeps it fast
        } else {
          const newAgent = await response.json()
          setCreatedAgent(newAgent)
          setShowSuccessDialog(true)

          // Silently update URL to include ID so refreshing works
          // We don't want to trigger a full navigation that might close the dialog or reset state if not handled carefully
          // But since we are setting state, a replace might be fine.
          // Let's use history API to be safe and avoid component remount
          window.history.pushState({}, '', `/build/${newAgent.id}`)

          // Also update internal state so "Edit Mode" logic kicks in effectively if we were to re-render
          // Note: agentId comes from searchParams which won't update until router.push/replace
          // But for the dialog purpose, we have what we need.
        }
      } else {
        const error = await response.json()
        setErrorMessage(error.detail || t("builds.editor.error.unknown"))
      }
    } catch (error) {
      console.error("Failed to save agent:", error)
      setErrorMessage(t("builds.editor.error.unknown"))
    } finally {
      setIsCreating(false)
    }
  }

  const handlePublish = async () => {
    if (!agentId) return

    setLoadingAgent(true)
    setErrorMessage(null)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/publish`, {
        method: "POST",
      })

      if (response.ok) {
        // Optional: show success message
        // Update local state to reflect published status
        setOriginalData({
          ...originalData,
          status: "published"
        })
        router.push("/build")
      } else {
        const error = await response.json()
        setErrorMessage(error.detail || "Failed to publish agent")
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
      setErrorMessage(t("builds.editor.error.unknown"))
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleUnpublish = async () => {
    if (!agentId) return

    setLoadingAgent(true)
    setErrorMessage(null)

    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${agentId}/unpublish`, {
        method: "POST",
      })

      if (response.ok) {
        // Update local state to reflect unpublished status
        setOriginalData({
          ...originalData,
          status: "draft"
        })
        router.push("/build")
      } else {
        const error = await response.json()
        setErrorMessage(error.detail || "Failed to unpublish agent")
      }
    } catch (error) {
      console.error("Failed to unpublish agent:", error)
      setErrorMessage(t("builds.editor.error.unknown"))
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleDialogPublish = async () => {
    if (!createdAgent?.id) return

    setLoadingAgent(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/agents/${createdAgent.id}/publish`, {
        method: "POST",
      })

      if (response.ok) {
        setShowSuccessDialog(false)
        router.push("/build")
      } else {
        const error = await response.json()
        // Show error in dialog? or main error message
        // For now, let's just close and show main error if possible, or alert
        alert(error.detail || "Failed to publish agent")
      }
    } catch (error) {
      console.error("Failed to publish agent:", error)
      alert("Failed to publish agent")
    } finally {
      setLoadingAgent(false)
    }
  }

  const handleDialogClose = () => {
    setShowSuccessDialog(false)
    // Refresh to enter full edit mode with proper context
    window.location.reload()
  }

  const LeftPanel = (
    <div className="p-6 space-y-8 min-h-full bg-card/50">
      <div className="space-y-6">
        {/* Logo Upload */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.logo.label")}</Label>
          <div className="flex items-center gap-4">
            <div
              className="h-16 w-16 rounded-lg border border-dashed border-muted-foreground/50 flex items-center justify-center bg-background overflow-hidden cursor-pointer hover:bg-muted/50 transition-colors"
              onClick={() => fileInputRef.current?.click()}
            >
              {logoFile ? (
                <img src={URL.createObjectURL(logoFile)} alt="Logo" className="h-full w-full object-cover" />
              ) : logoUrl ? (
                <img src={`${getApiUrl()}${logoUrl}`} alt="Logo" className="h-full w-full object-cover" />
              ) : (
                <Upload className="h-6 w-6 text-muted-foreground" />
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              className="hidden"
              ref={fileInputRef}
              onChange={handleLogoUpload}
            />
          </div>
        </div>

        {/* Name */}
        <div className="space-y-2">
          <Label htmlFor="name">{t("builds.configForm.name.label")}</Label>
          <Input
            id="name"
            placeholder={t("builds.configForm.name.placeholder")}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>

        {/* Description */}
        <div className="space-y-2">
          <Label htmlFor="description">{t("builds.configForm.description.label")}</Label>
          <Textarea
            id="description"
            placeholder={t("builds.configForm.description.placeholder")}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        {/* Instructions */}
        <div className="space-y-2">
          <Label htmlFor="instructions">{t("builds.configForm.instructions.label")}</Label>
          <Textarea
            id="instructions"
            placeholder={t("builds.configForm.instructions.placeholder")}
            className="min-h-[150px] font-mono text-sm"
            value={instructions}
            onChange={(e) => setInstructions(e.target.value)}
          />
        </div>

        {/* Execution Mode */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.executionMode.label")}</Label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              className={`px-3 py-2 text-sm border rounded-md transition-colors ${
                executionMode === "react"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-background hover:bg-accent"
              }`}
              onClick={() => setExecutionMode("react")}
            >
              <div className="font-medium">{t("builds.configForm.executionMode.react.title")}</div>
              <div className="text-xs opacity-80">{t("builds.configForm.executionMode.react.description")}</div>
            </button>
            <button
              type="button"
              className={`px-3 py-2 text-sm border rounded-md transition-colors ${
                executionMode === "graph"
                  ? "bg-primary text-primary-foreground border-primary"
                  : "bg-background hover:bg-accent"
              }`}
              onClick={() => setExecutionMode("graph")}
            >
              <div className="font-medium">{t("builds.configForm.executionMode.graph.title")}</div>
              <div className="text-xs opacity-80">{t("builds.configForm.executionMode.graph.description")}</div>
            </button>
          </div>
        </div>

        {/* Model Selection - 4 types */}
        <div className="space-y-4">
          <Label>{t("builds.configForm.model.label")}</Label>
          {models.length > 0 ? (
            <div className="grid grid-cols-2 gap-3">
              {/* General Model */}
              <div className="space-y-1">
                <div className="flex items-center gap-1.5">
                  <Label className="text-xs text-muted-foreground">
                    {t("builds.configForm.model.types.general")}
                  </Label>
                  <TooltipProvider>
                    <Tooltip delayDuration={300}>
                      <TooltipTrigger asChild>
                        <div className="cursor-default">
                          <Info className="h-3 w-3 text-muted-foreground/70 hover:text-muted-foreground" />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-[200px]">{t("builds.configForm.model.tips.general")}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Select
                  value={modelConfig.general?.toString() || ""}
                  onValueChange={(value) => setModelConfig(prev => ({
                    ...prev,
                    general: value ? Number(value) : null
                  }))}
                  options={modelOptions}
                  placeholder="--"
                />
              </div>

              {/* Small & Fast Model */}
              <div className="space-y-1">
                <div className="flex items-center gap-1.5">
                  <Label className="text-xs text-muted-foreground">
                    {t("builds.configForm.model.types.smallFast")}
                  </Label>
                  <TooltipProvider>
                    <Tooltip delayDuration={300}>
                      <TooltipTrigger asChild>
                        <div className="cursor-default">
                          <Info className="h-3 w-3 text-muted-foreground/70 hover:text-muted-foreground" />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-[200px]">{t("builds.configForm.model.tips.smallFast")}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Select
                  value={modelConfig.small_fast?.toString() || ""}
                  onValueChange={(value) => setModelConfig(prev => ({
                    ...prev,
                    small_fast: value ? Number(value) : null
                  }))}
                  options={modelOptions}
                  placeholder="--"
                />
              </div>

              {/* Visual Model */}
              <div className="space-y-1">
                <div className="flex items-center gap-1.5">
                  <Label className="text-xs text-muted-foreground">
                    {t("builds.configForm.model.types.visual")}
                  </Label>
                  <TooltipProvider>
                    <Tooltip delayDuration={300}>
                      <TooltipTrigger asChild>
                        <div className="cursor-default">
                          <Info className="h-3 w-3 text-muted-foreground/70 hover:text-muted-foreground" />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-[200px]">{t("builds.configForm.model.tips.visual")}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Select
                  value={modelConfig.visual?.toString() || ""}
                  onValueChange={(value) => setModelConfig(prev => ({
                    ...prev,
                    visual: value ? Number(value) : null
                  }))}
                  options={modelOptions}
                  placeholder="--"
                />
              </div>

              {/* Compact Model */}
              <div className="space-y-1">
                <div className="flex items-center gap-1.5">
                  <Label className="text-xs text-muted-foreground">
                    {t("builds.configForm.model.types.compact")}
                  </Label>
                  <TooltipProvider>
                    <Tooltip delayDuration={300}>
                      <TooltipTrigger asChild>
                        <div className="cursor-default">
                          <Info className="h-3 w-3 text-muted-foreground/70 hover:text-muted-foreground" />
                        </div>
                      </TooltipTrigger>
                      <TooltipContent>
                        <p className="max-w-[200px]">{t("builds.configForm.model.tips.compact")}</p>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
                <Select
                  value={modelConfig.compact?.toString() || ""}
                  onValueChange={(value) => setModelConfig(prev => ({
                    ...prev,
                    compact: value ? Number(value) : null
                  }))}
                  options={modelOptions}
                  placeholder="--"
                />
              </div>
            </div>
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.model.noData")}
            </div>
          )}
        </div>

        {/* Knowledge Base - Multi Select */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.knowledgeBase.label")}</Label>
          {kbs.length > 0 ? (
            <MultiSelect
              values={selectedKbs}
              onValuesChange={setSelectedKbs}
              options={kbOptions}
              placeholder={t("builds.configForm.knowledgeBase.placeholder")}
            />
          ) : (
            <Button
              variant="outline"
              className="w-full justify-start text-muted-foreground"
              onClick={() => router.push('/kb')}
            >
              <PlusCircle className="mr-2 h-4 w-4" />
              {t("builds.configForm.knowledgeBase.create")}
            </Button>
          )}
        </div>

        {/* Skills - Multi Select */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.skills.label")}</Label>
          {skills.length > 0 ? (
            <MultiSelect
              values={selectedSkills}
              onValuesChange={setSelectedSkills}
              options={skillOptions}
              placeholder={t("builds.configForm.skills.placeholder")}
            />
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.skills.noData")}
            </div>
          )}
        </div>

        {/* Tools - Multi Select by Category */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.tools.label")}</Label>
          {toolCategories.length > 0 ? (
            <MultiSelect
              values={selectedToolCategories}
              onValuesChange={setSelectedToolCategories}
              options={toolCategoryOptions}
              placeholder={t("builds.configForm.tools.placeholder")}
            />
          ) : (
            <div className="text-sm text-muted-foreground">
              {t("builds.configForm.tools.noData")}
            </div>
          )}
          {selectedToolCategories.length > 0 && (
            <div className="text-xs text-muted-foreground">
              {t("builds.configForm.tools.selectedCount", {
                count: selectedToolCategories.length,
                tools: tools.filter(t => selectedToolCategories.includes(t.category)).length
              })}
            </div>
          )}
        </div>

        {/* Suggested Prompts */}
        <div className="space-y-2">
          <Label>{t("builds.configForm.suggestedPrompts.label")}</Label>
          <div className="text-xs text-muted-foreground mb-2">
            {t("builds.configForm.suggestedPrompts.description")}
          </div>
          <div className="space-y-2">
            {suggestedPrompts.map((prompt, index) => (
              <div key={index} className="flex gap-2 items-start">
                <Input
                  value={prompt}
                  onChange={(e) => {
                    const newPrompts = [...suggestedPrompts]
                    newPrompts[index] = e.target.value
                    setSuggestedPrompts(newPrompts)
                  }}
                  placeholder={t("builds.configForm.suggestedPrompts.placeholder", { index: index + 1 })}
                  className="flex-1"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  onClick={() => {
                    const newPrompts = suggestedPrompts.filter((_, i) => i !== index)
                    setSuggestedPrompts(newPrompts)
                  }}
                >
                  {t("builds.configForm.suggestedPrompts.delete")}
                </Button>
              </div>
            ))}
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => setSuggestedPrompts([...suggestedPrompts, ""])}
            >
              {t("builds.configForm.suggestedPrompts.add")}
            </Button>
          </div>
        </div>
      </div>
    </div>
  )

  const RightPanel = (
    <div className="flex flex-col h-full bg-background border-l">
      {/* Header */}
      <div className="h-14 border-b flex items-center px-4 gap-2 bg-card/30">
        <MessageSquare className="h-5 w-5 text-muted-foreground" />
        <span className="font-medium">{t("builds.preview.title")}</span>
        <div className="ml-auto flex items-center gap-2">
          <div
            className={`w-2.5 h-2.5 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-500'}`}
            title={wsConnected ? 'Connected' : 'Disconnected'}
          />
          <span className="text-xs text-muted-foreground">
            {wsConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-hidden relative">
        <ScrollArea className="h-full px-4 py-4">
          <div className="space-y-4 max-w-3xl mx-auto">
            {messages.map((msg, index) => (
              <ChatMessage
                key={index}
                role={msg.role}
                content={msg.content}
                traceEvents={msg.traceEvents}
                showProcessView={true}
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>
      </div>

      {/* Input */}
      <div className="p-4 border-t bg-card/30 mb-8">
        <div className="max-w-3xl mx-auto">
          <ChatInput
            onSend={handleSendMessage}
            isLoading={isChatLoading}
            hideConfig={true}
            files={files}
            onFilesChange={setFiles}
          />
        </div>
      </div>
    </div>
  )

  return (
    <div className="flex flex-col h-[100vh]">
      {/* Header */}
      <div className="border-b flex justify-between items-center p-8">
        <div>
          <h1 className="text-3xl font-bold mb-1">{t("builds.editor.header.title")}</h1>
          <p className="text-muted-foreground">{t("builds.editor.header.subtitle")}</p>
        </div>
        <div className="flex items-center gap-4">
          {errorMessage && (
            <div className="text-sm text-destructive">{errorMessage}</div>
          )}
          <Button onClick={isEditMode && !isDirty ? (originalData?.status === "published" ? handleUnpublish : handlePublish) : handleCreate} disabled={isCreating || loadingAgent}>
            {isCreating
              ? (isEditMode ? t("builds.editor.header.updating") : t("builds.editor.header.creating"))
              : isEditMode
              ? (isDirty ? t("builds.editor.header.update") : (originalData?.status === "published" ? t("builds.editor.header.unpublish") : t("builds.editor.header.publish")))
              : t("builds.editor.header.create")
            }
          </Button>
        </div>
      </div>

      <div className="flex-1 min-h-0">
        <ResizableSplitLayout
          leftPanel={LeftPanel}
          rightPanel={RightPanel}
          initialLeftWidth={50}
          minLeftWidth={30}
          maxLeftWidth={70}
        />
      </div>
      {/* File Preview Drawer */}
      <Sheet open={previewState.isOpen} onOpenChange={(open) => setPreviewState(prev => ({ ...prev, isOpen: open }))}>
        <SheetContent className="!max-w-[1200px] w-[90vw] sm:w-[800px] md:w-[900px] lg:w-[1000px] flex flex-col p-0 gap-0">
          <div className="flex flex-col gap-1.5 p-4 flex-shrink-0 bg-background/80 backdrop-blur-sm border-b">
            <div className="flex items-center justify-between">
              <SheetTitle className="flex items-center gap-2">
                {previewState.fileName}
              </SheetTitle>
              <div className="flex items-center gap-2 mr-8">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleDownloadFile}
                  className="h-8 w-8 p-0"
                  title={t("files.previewDialog.buttons.download")}
                >
                  <Download className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
          <div className="flex-1 overflow-hidden flex flex-col min-h-0 bg-muted/30 p-4">
             {previewState.fileUrl && (
               <div className="w-full h-full flex items-center justify-center bg-background rounded-lg border overflow-auto">
                 {previewState.fileType?.startsWith('image/') ? (
                   <img
                     src={previewState.fileUrl}
                     alt={previewState.fileName}
                     className="max-w-full max-h-full object-contain"
                   />
                 ) : (previewState.fileType?.includes('pdf') || previewState.fileName?.endsWith('.pdf')) ? (
                   <iframe
                     src={previewState.fileUrl}
                     className="w-full h-full border-0"
                     title={previewState.fileName}
                   />
                 ) : (
                   <div className="text-center p-8">
                     <p className="text-muted-foreground mb-4">{t("files.previewDialog.noPreview") || "No preview available for this file type."}</p>
                     <Button onClick={handleDownloadFile} variant="outline">
                       <Download className="mr-2 h-4 w-4" />
                       {t("files.previewDialog.buttons.download")}
                     </Button>
                   </div>
                 )}
               </div>
             )}
          </div>
        </SheetContent>
      </Sheet>

      {/* Success Dialog */}
      <Dialog open={showSuccessDialog} onOpenChange={handleDialogClose}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{t("builds.editor.success.created")}</DialogTitle>
            <DialogDescription>
              {t("builds.editor.success.createdDesc", { name: createdAgent?.name })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button variant="outline" onClick={handleDialogClose}>
              {t("common.cancel") || "Close"}
            </Button>
            <Button onClick={handleDialogPublish}>
              {t("builds.editor.header.publish")}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
