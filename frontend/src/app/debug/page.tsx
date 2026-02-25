"use client"

import { useSearchParams } from "next/navigation"
import { Suspense } from "react"
import { AppProvider, useApp } from "@/contexts/app-context"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Wifi, WifiOff, AlertCircle, ArrowLeft, Send, Pause, Play, Zap, Workflow, Target } from "lucide-react"
import Link from "next/link"
import { useEffect, useState, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import { RightPanel } from "@/components/layout/right-panel"
import { ThreeColumnLayout } from "@/components/layout/three-column-layout"
import { LeftPanel } from "@/components/layout/left-panel"
import { CenterPanel } from "@/components/layout/center-panel"
import { FilePreviewDialog } from "@/components/file-preview-dialog"
import { AgentInput } from "@/components/agent-input"
import { ReplayControls } from "@/components/replay/replay-controls"
import { VibeModeSelector, VibeModeConfig } from "@/components/vibe-mode-selector"
import { ConfigDialog } from "@/components/config-dialog"
import { ModelInfoDisplay } from "@/components/model-info-display"
import { cn, getApiUrl, getAuthHeaders } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useAuth } from "@/contexts/auth-context"
import dagre from "dagre"
import { RotateCcw, LayoutDashboard,LayoutPanelLeft, Settings } from "lucide-react"
import { useI18n } from "@/contexts/i18n-context"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  timestamp: string
  status?: "pending" | "running" | "completed" | "failed"
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused"
  description: string
  createdAt: string
  updatedAt: string
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
  vibeMode?: "task" | "process"
}

interface DAGExecution {
  phase: "planning" | "executing" | "completed" | "failed"
  current_plan: Record<string, unknown>
  created_at: string | number
  updated_at: string | number
}

interface DAGNode {
  id: string
  type: string
  position: { x: number; y: number }
  data: {
    label: string
    status: "pending" | "running" | "completed" | "failed" | "skipped"
    description?: string
    tool_names?: string[]
    started_at?: string | number
    completed_at?: string | number
    result?: unknown
    conditional_branches?: Record<string, string>
    required_branch?: string | null
    is_conditional?: boolean
  }
}

interface DAGEdge {
  id: string
  source: string
  target: string
  data: {
    label?: string
  }
}

interface AgentConfig {
  model: string
  smallFastModel?: string
  visualModel?: string
  compactModel?: string
  memorySimilarityThreshold?: number
}

function AgentContent() {
  const searchParams = useSearchParams()
  const taskId = searchParams.get('id')
  const { state, sendMessage, isConnected, connectionError, setTaskId, executeTask, pauseTask, resumeTask, dispatch, requestStatus, openFilePreview, closeFilePreview, startReplay, stopReplay, setReplayPlaying, setReplaySpeed, setReplayProgress } = useApp()
  const { t } = useI18n()
  const [inputMessage, setInputMessage] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const [processModeFiles, setProcessModeFiles] = useState<File[]>([])
  const [agentConfig, setAgentConfig] = useState<AgentConfig>({
    model: "",
    smallFastModel: undefined,
    visualModel: undefined,
    compactModel: undefined,
    memorySimilarityThreshold: 1.5
  })
  const [hasSubmitted, setHasSubmitted] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const [dagLayout, setDagLayout] = useState<'TB' | 'LR'>('TB')
  const [vibeModeConfig, setVibeModeConfig] = useState<VibeModeConfig>({
    mode: "task",
    processDescription: "",
    examples: []
  })

  // Load default configuration if not already set
  useEffect(() => {
    const loadDefaultConfigurationIfNeeded = async () => {
      // Only load if no model is configured
      if (!agentConfig.model) {
        try {
          const apiUrl = getApiUrl()

          // Fetch user default models
          const defaultResponse = await apiRequest(`${apiUrl}/api/models/user-default`, {
            headers: {}
          })

          let defaultModels: Record<string, any> = {}
          if (defaultResponse.ok) {
            const defaults = await defaultResponse.json()
            if (Array.isArray(defaults)) {
              defaults.forEach((defaultConfig: any) => {
                if (defaultConfig && defaultConfig.config_type && defaultConfig.model) {
                  defaultModels[defaultConfig.config_type] = defaultConfig.model
                }
              })
            }
          }

          // Fetch all models to get fallback model details
          const modelsResponse = await apiRequest(`${apiUrl}/api/models/?category=llm`, {
            headers: {}
          })
          if (!modelsResponse.ok) return

          const data = await modelsResponse.json()
          if (data.length > 0) {
            const defaultModel = defaultModels.general || data.find((m: any) => m.is_default) || data[0]
            setAgentConfig(prev => ({
              ...prev,
              model: prev.model || defaultModel.model_id,
              smallFastModel: prev.smallFastModel || defaultModels.small_fast?.model_id,
              visualModel: prev.visualModel || defaultModels.visual?.model_id,
              compactModel: prev.compactModel || defaultModels.compact?.model_id
            }))

                      }
        } catch (error) {
          console.error('Failed to auto-load default configuration:', error)
        }
      }
    }

    loadDefaultConfigurationIfNeeded()
  }, []) // Only run once on mount

  // Check if we should show initial state (no messages, no task, and no taskId in URL)
  const showInitialState = !hasSubmitted && state.messages.length === 0 && !state.currentTask && !taskId


  // Listen for file preview events
  useEffect(() => {
    const handleFilePreviewEvent = (event: CustomEvent) => {
      const { filePath, fileName, allFiles, currentIndex } = event.detail
      if (allFiles && Array.isArray(allFiles)) {
        openFilePreview(filePath, fileName, allFiles, currentIndex || 0)
      } else {
        openFilePreview(filePath, fileName)
      }
    }

    window.addEventListener('openFilePreview', handleFilePreviewEvent as EventListener)

    return () => {
      window.removeEventListener('openFilePreview', handleFilePreviewEvent as EventListener)
    }
  }, [openFilePreview])


  const { user, token } = useAuth()

  // Load default config from backend on mount
  useEffect(() => {
    const loadDefaultConfig = async () => {
      try {
        const response = await apiRequest(`${getApiUrl()}/api/models/`, {
          headers: getAuthHeaders(token)
        })
        if (response.ok) {
          const data = await response.json()
          // Find default model (first one or marked as default)
          const defaultModel = data.models?.[0]?.id || ""
          setAgentConfig({
            model: defaultModel,
            smallFastModel: undefined,
            visualModel: undefined,
            compactModel: undefined,
            memorySimilarityThreshold: 1.0
          })
        }
      } catch (error) {
        console.error('Failed to load default config:', error)
      }
    }

    loadDefaultConfig()
  }, [])

  // Create dagre graph for dynamic layout
  const dagreGraph = new dagre.graphlib.Graph()

  // Analyze graph structure to determine optimal layout
  const calculateOptimalLayout = (): 'TB' | 'LR' => {
    if (state.steps.length === 0) return 'TB'

    // Calculate width and depth of the graph
    const nodeLevels = new Map<string, number>()
    const processedNodes = new Set<string>()

    // Assign levels to nodes (topological sort)
    const assignLevels = () => {
      const queue: string[] = []

      // Find root nodes (no dependencies)
      state.steps.forEach(step => {
        if (step.dependencies.length === 0) {
          nodeLevels.set(step.id, 0)
          queue.push(step.id)
        }
      })

      // Process nodes in BFS order
      while (queue.length > 0) {
        const nodeId = queue.shift()!
        processedNodes.add(nodeId)

        // Find all nodes that depend on this node
        const dependents = state.steps.filter(step =>
          step.dependencies.includes(nodeId)
        )

        dependents.forEach(dependent => {
          // Check if all dependencies of this dependent have been processed
          const allDepsProcessed = dependent.dependencies.every(dep =>
            processedNodes.has(dep)
          )

          if (allDepsProcessed) {
            const level = Math.max(
              ...dependent.dependencies.map(dep => nodeLevels.get(dep) || 0)
            ) + 1
            nodeLevels.set(dependent.id, level)
            queue.push(dependent.id)
          }
        })
      }
    }

    assignLevels()

    // Calculate max width (nodes at same level) and max depth
    const levelCounts = new Map<number, number>()
    nodeLevels.forEach(level => {
      levelCounts.set(level, (levelCounts.get(level) || 0) + 1)
    })

    const maxWidth = Math.max(...levelCounts.values())
    const maxDepth = Math.max(...nodeLevels.values())

    // Choose layout based on aspect ratio
    // If width > depth, use horizontal layout (LR)
    // If depth > width, use vertical layout (TB)
    return maxWidth > maxDepth ? 'LR' : 'TB'
  }

  // Update layout when graph changes
  useEffect(() => {
    if (state.steps.length > 0) {
      const optimalLayout = calculateOptimalLayout()
      setDagLayout(optimalLayout)
    }
  }, [state.steps])

  dagreGraph.setGraph({
    rankdir: dagLayout, // Dynamic layout direction
    nodesep: dagLayout === 'TB' ? 180 : 120,  // Adjust spacing based on layout
    ranksep: dagLayout === 'TB' ? 180 : 120,
    marginx: 50,
    marginy: 50
  })
  dagreGraph.setDefaultEdgeLabel(() => '')

  // Add nodes to dagre graph (only with valid IDs)
  const validSteps = state.steps.filter(step => {
    const isValid = step.id && typeof step.id === 'string' && step.id.trim() !== ''
    if (!isValid) {
      console.warn('Skipping step with invalid ID:', step)
    }
    return isValid
  })

  validSteps.forEach((step) => {
    try {
      dagreGraph.setNode(step.id, {
        width: 180,
        height: 60
      })
    } catch (error) {
      console.error('Error adding node to dagre:', step.id, error)
    }
  })

  // Add edges to dagre graph based on dependencies (only with valid IDs)
  validSteps.forEach((step) => {
    if (!step.dependencies || !Array.isArray(step.dependencies)) {
      console.warn('Step has invalid dependencies:', step)
      return
    }

    step.dependencies.forEach(depId => {
      // Skip dependencies with empty or invalid IDs
      if (!depId || typeof depId !== 'string' || depId.trim() === '') {
        console.warn('Skipping dependency with invalid ID:', depId)
        return
      }

      // Only add edge if both nodes exist and have valid IDs
      const depStep = validSteps.find(s => s.id === depId)
      if (depStep) {
        try {
          dagreGraph.setEdge(depId, step.id, {})
        } catch (error) {
          console.error('Error adding edge to dagre:', `${depId} -> ${step.id}`, error)
        }
      } else {
        console.warn('Dependency not found in valid steps:', depId)
      }
    })
  })

  // Apply dagre layout
  let dagreLayoutSuccessful = true
  try {
    dagre.layout(dagreGraph)
  } catch (error) {
    console.error('Dagre layout failed:', error)
    dagreLayoutSuccessful = false
  }

  // Convert steps to DAG nodes using dagre positions or fallback layout
  const dagNodes: DAGNode[] = state.steps.map((step, index) => {
    let node, safeNode

    // Skip steps with invalid IDs - use fallback positioning
    if (!step.id || typeof step.id !== 'string' || step.id.trim() === '') {
      console.warn('Skipping step with invalid ID:', step)
      safeNode = { x: (index % 3) * 200, y: Math.floor(index / 3) * 100 }
    } else if (dagreLayoutSuccessful) {
      // Use dagre layout if it was successful
      try {
        node = dagreGraph.node(step.id)
        // Ensure node is an object, not a string (prevents 'points' property error)
        safeNode = typeof node === 'object' && node !== null ? node : { x: (index % 3) * 200, y: Math.floor(index / 3) * 100 }
      } catch (error) {
        console.error(`Error getting node for ${step.id}:`, error)
        // Fallback position based on index
        safeNode = { x: (index % 3) * 200, y: Math.floor(index / 3) * 100 }
      }
    } else {
      // Use fallback layout if dagre failed
      safeNode = { x: (index % 3) * 200, y: Math.floor(index / 3) * 100 }
    }

    return {
      id: step.id || `step-${index}`, // Ensure always has a valid ID
      type: "default",
      position: { x: (safeNode.x || 0) - 90, y: (safeNode.y || 0) - 30 }, // Center the node
      data: {
        label: step.name || `Step ${index + 1}`, // Ensure always has a label
        status: step.status,
        description: step.description,
        tool_names: step.tool_names,
        started_at: step.started_at,
        completed_at: step.completed_at,
        result: step.result_data,
        conditional_branches: step.conditional_branches,
        required_branch: step.required_branch,
        is_conditional: step.is_conditional,
      }
    }
  })

  // Create edges based on dependencies with better validation
  const dagEdges: DAGEdge[] = []
  const validNodeIds = new Set(validSteps.map(s => s.id))


  // Only create edges if we have valid node IDs and dagre layout was successful
  if (dagreLayoutSuccessful) {
    validSteps.forEach((step) => {
      if (!step.dependencies || !Array.isArray(step.dependencies)) {
        console.warn('Step has invalid dependencies for edge creation:', step)
        return
      }

      step.dependencies.forEach(depId => {
        // Skip dependencies with invalid IDs
        if (!depId || typeof depId !== 'string' || depId.trim() === '') {
          console.warn('Skipping dependency with invalid ID for edge creation:', depId)
          return
        }

        // Only create edges if both source and target nodes have valid IDs
        if (validNodeIds.has(depId) && validNodeIds.has(step.id)) {
          const edge = {
            id: `${depId}-${step.id}`,
            source: depId,
            target: step.id,
            data: {}
          }
          dagEdges.push(edge)
        } else {
          console.warn('Skipping edge creation - missing nodes:', { source: depId, target: step.id, validNodeIds: Array.from(validNodeIds) })
        }
      })
    })
  } else {
    console.warn('Skipping edge creation due to dagre layout failure')
  }



  useEffect(() => {
    if (taskId) {
      const taskIdNum = parseInt(taskId, 10)
      if (!isNaN(taskIdNum)) {
        setTaskId(taskIdNum)
        // WebSocket will automatically load historical data when connected
      } else {
      }
    } else {
    }
  }, [taskId, setTaskId, searchParams])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [state.messages])

  const handleSendMessage = (files?: File[]) => {
    // Use the passed files if available, otherwise use the appropriate component state
    const filesToSend = files || (vibeModeConfig.mode === "process" ? processModeFiles : selectedFiles)

    // For task mode, require input message
    // For process mode, require process description
    const canSubmit = vibeModeConfig.mode === "task"
      ? (inputMessage.trim() || (filesToSend && filesToSend.length > 0))
      : (vibeModeConfig.processDescription || (filesToSend && filesToSend.length > 0))

    if (canSubmit) {
      // Include vibe mode config in the agent config
      const configWithVibeMode = {
        ...agentConfig,
        vibeMode: vibeModeConfig
      }
      // For process mode, use process description as message
      // For task mode, use user input
      const messageToSend = vibeModeConfig.mode === "process"
        ? (vibeModeConfig.processDescription || "")
        : inputMessage.trim()
      ;(sendMessage as any)(messageToSend, configWithVibeMode, filesToSend)
      setInputMessage("")
      setSelectedFiles([])
      setProcessModeFiles([])
      if (!hasSubmitted) {
        setHasSubmitted(true)
      }
    }
  }


  const handleBuild = () => {
    if (state.currentTask) {
      const vibeMode = (state.currentTask as any)?.vibeMode || 'task'

      // Only allow building in process mode
      if (vibeMode !== 'process') {
        return
      }

      // Navigate to build page with task ID
      window.location.href = `/build?taskId=${state.currentTask.id}`
    }
  }

  const handleConfigChange = (newConfig: AgentConfig) => {
    setAgentConfig(newConfig)
  }

  // Replay control methods
  const handleStartReplay = () => {
    if (!state.currentTask) return

    // Clear all existing content and replay cache
    dispatch({ type: "CLEAR_MESSAGES" })
    dispatch({ type: "SET_STEPS", payload: [] })
    dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
    dispatch({ type: "SET_DAG_EXECUTION", payload: null })
    dispatch({ type: "CLEAR_REPLAY_CACHE" })

    // Try to trigger historical data reload without disconnecting
    if (state.taskId) {
      // Set replay mode - new events will be processed with time delays
      dispatch({ type: "SET_REPLAY_TASK_ID", payload: state.taskId })
      setReplayPlaying(true)

      // Clear duplicate message cache
      ;(window as any).clearDuplicateMessageCache?.()

      // Request historical data
      setTimeout(() => {
        requestStatus()
      }, 100)
    }
  }

  const handlePauseReplay = () => {
    setReplayPlaying(false)
  }

  const handleStopReplay = () => {
    setReplayPlaying(false)
    stopReplay()
    // Also clear the displayed content to reset to original task state
    dispatch({ type: "CLEAR_MESSAGES" })
    dispatch({ type: "SET_STEPS", payload: [] })
    dispatch({ type: "SET_TRACE_EVENTS", payload: [] })
    dispatch({ type: "SET_DAG_EXECUTION", payload: null })
  }

  const handleSpeedChange = (speed: number) => {
    setReplaySpeed(speed)
  }

  const handleProgressChange = (progress: number) => {
    // Progress control not implemented in this simplified version
  }

  const { formatTime, formatDuration } = require('@/lib/time-utils')
  const formatTimestamp = (timestamp: string | number) => formatTime(timestamp)

  const formatTraceData = (data: any) => {
    if (typeof data === 'string') {
      return data
    }
    try {
      return JSON.stringify(data, null, 2)
    } catch {
      return String(data)
    }
  }

  return (
    <div className="h-full flex flex-col bg-background">
      {/* Header */}
      <header className="h-14 border-b border-border bg-card/50 backdrop-blur-sm">
        <div className="flex items-center justify-between h-full px-4">
          <div className="flex items-center gap-4">
            <Link
              href="/debug/vibe"
              className="p-2 text-muted-foreground hover:text-foreground hover:bg-accent rounded-md transition-colors"
              title={t('agent.header.backTitle')}
            >
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <h1 className="text-lg font-semibold text-foreground">{t('agent.title')}</h1>
            <div className="flex items-center gap-2">
              {taskId && (
                <Badge variant="outline" className="text-xs border-border">
                  {t('agent.taskId', { id: taskId })}
                </Badge>
              )}
              {state.currentTask && (
                <Badge
                  variant="outline"
                  className={`text-xs ${
                    ((state.currentTask as any)?.vibeMode || 'task') === 'process'
                      ? 'bg-purple-500/10 text-purple-400 border-purple-500/30'
                      : 'bg-blue-500/10 text-blue-400 border-blue-500/30'
                  }`}
                >
                  {((state.currentTask as any)?.vibeMode || 'task') === 'process' ? (
                    <>
                      <Workflow className="h-3 w-3 mr-1" />
                      {t('agent.header.badge.process')}
                    </>
                  ) : (
                    <>
                      <Target className="h-3 w-3 mr-1" />
                      {t('agent.header.badge.task')}
                    </>
                  )}
                </Badge>
              )}
              {state.currentTask && (
                <Badge variant={state.currentTask.status === 'running' ? 'default' : 'secondary'} className="text-xs">
                  {state.currentTask.status === 'running' && '🚀 '}
                  {t(`agent.status.${state.currentTask.status}`)}
                </Badge>
              )}
              {hasSubmitted && (
                <>
                  {isConnected ? (
                    <Badge variant="secondary" className="text-xs">
                      <Wifi className="h-3 w-3 mr-1" />
                      {t('agent.header.connection.connected')}
                    </Badge>
                  ) : (
                    <Badge variant="destructive" className="text-xs">
                      <WifiOff className="h-3 w-3 mr-1" />
                      {t('agent.header.connection.disconnected')}
                    </Badge>
                  )}
                </>
              )}
            </div>
          </div>

          {connectionError && (
            <Alert className="py-2 px-3 border-destructive/20 bg-destructive/10">
              <AlertCircle className="h-4 w-4 text-destructive" />
              <AlertDescription className="text-xs text-destructive-foreground">
                {t('agent.header.connection.errorPrefix', { message: connectionError.message })}
              </AlertDescription>
            </Alert>
          )}

          <div className="flex items-center gap-3">
            {/* Replay Controls */}
            {state.currentTask && state.currentTask.status === 'completed' && (
              <ReplayControls
                isPlaying={state.isReplaying}
                playbackSpeed={state.replaySpeed}
                onPlay={handleStartReplay}
                onPause={handlePauseReplay}
                onStop={handleStopReplay}
                onSpeedChange={handleSpeedChange}
              />
            )}

            {/* Build Button - only enabled for process mode */}
            {state.currentTask && state.currentTask.status === 'completed' && (
              <Button
                onClick={handleBuild}
                disabled={((state.currentTask as any)?.vibeMode || 'task') !== 'process'}
                className="bg-green-600 hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <Zap className="h-4 w-4 mr-2" />
                {((state.currentTask as any)?.vibeMode || 'task') === 'process'
                  ? t('agent.header.buildButton')
                  : t('agent.build.onlyProcessModeShort')}
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden w-full relative">
        {showInitialState ? (
          /* Initial State - Centered Input */
          <div className="absolute inset-0 flex items-center justify-center bg-background transition-all duration-700 ease-out">
            {/* Blurred background elements */}
            <div className="absolute inset-0">
              <div className="absolute top-10 left-10 w-32 h-32 bg-primary/5 rounded-full blur-3xl"></div>
              <div className="absolute bottom-10 right-10 w-40 h-40 bg-blue-500/5 rounded-full blur-3xl"></div>
              <div className="absolute top-1/2 left-1/4 w-24 h-24 bg-green-500/5 rounded-full blur-2xl"></div>
            </div>

            {/* Centered Input Container */}
            <div className="relative z-10 w-full max-w-2xl px-8 animate-in fade-in-50 zoom-in-95 duration-500">
              <div className="text-center mb-8 animate-in slide-in-from-bottom-4 duration-700 delay-150">
                <h1 className="text-3xl font-bold text-foreground mb-2">{t('agent.initial.title')}</h1>
                <p className="text-muted-foreground">{t('agent.initial.subtitle')}</p>
              </div>

              <div className="relative bg-card/80 backdrop-blur-sm border border-border rounded-2xl p-6 shadow-lg animate-in slide-in-from-bottom-4 duration-700 delay-300">
                <div className="space-y-6">
                  {/* VIBE Mode Selector */}
                  <VibeModeSelector
                    config={vibeModeConfig}
                    onChange={setVibeModeConfig}
                    selectedFiles={processModeFiles}
                    onFilesChange={setProcessModeFiles}
                  />

                  {/* Task Input - only show for task mode */}
                  {vibeModeConfig.mode === "task" && (
                    <AgentInput
                      value={inputMessage}
                      onChange={(value) => setInputMessage(value)}
                      onSend={handleSendMessage}
                      placeholder={t('agent.input.placeholder.default')}
                      agentConfig={agentConfig}
                      onConfigChange={handleConfigChange}
                      rows={6}
                      variant="expanded"
                      showStatus={false}
                      selectedFiles={selectedFiles}
                      setSelectedFiles={setSelectedFiles}
                    />
                  )}

                  {/* Process Mode - Model Config and Execute Button in same row */}
                  {vibeModeConfig.mode === "process" && (
                    <div className="flex items-center justify-between">
                      {/* Model Configuration - left */}
                      <ConfigDialog
                        onConfigChange={handleConfigChange}
                        currentConfig={agentConfig}
                        trigger={
                          <div className="flex items-center gap-2 text-sm text-muted-foreground">
                            <ModelInfoDisplay
                              currentTask={null}
                              onConfigChange={undefined}
                            />
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 px-2 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
                              title={t('agent.config.title')}
                            >
                              <Settings className="h-3.5 w-3.5" />
                            </Button>
                          </div>
                        }
                      />

                      {/* Execute Button - right */}
                      <Button
                        onClick={() => handleSendMessage()}
                        disabled={!vibeModeConfig.processDescription && processModeFiles.length === 0}
                        size="lg"
                        className="px-8"
                      >
                        {t('agent.process.execute')}
                      </Button>
                    </div>
                  )}

                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">
                      {vibeModeConfig.mode === "task"
                        ? t('agent.hints.taskMode')
                        : t('agent.hints.processMode')}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          /* Active State - Three Column Layout */
          <div className={`w-full h-full transition-all duration-500 animate-in fade-in duration-700`}>
            <ThreeColumnLayout
              leftPanel={
                <LeftPanel
                  onSendMessage={async (message: string, config?: AgentConfig, files?: File[]) => {
                    await (sendMessage as any)(message, config, files)
                  }}
                  onPauseTask={pauseTask}
                  onResumeTask={resumeTask}
                  messages={state.messages}
                  currentTask={state.currentTask}
                  isProcessing={state.isProcessing}
                  agentConfig={agentConfig}
                  onConfigChange={handleConfigChange}
                />
              }
              centerPanel={
                <CenterPanel
                  dagExecution={state.dagExecution}
                  dagNodes={dagNodes}
                  dagEdges={dagEdges}
                  dagLayout={dagLayout}
                  onLayoutChange={setDagLayout}
                  onNodeClick={(node) => {
                    // Node click handler
                  }}
                  onRefresh={() => {
                    requestStatus()
                  }}
                  isPlanning={dagNodes.length === 0 && state.dagExecution?.phase === "planning"}
                  hasError={dagNodes.length === 0 && (state.dagExecution?.phase === "failed" || state.currentTask?.status === "failed")}
                  currentTaskStatus={state.currentTask?.status}
                  onFileClick={openFilePreview}
                />
              }
              rightPanel={
                <RightPanel
                  steps={state.steps}
                  traceEvents={state.traceEvents}
                  selectedStepId={state.selectedStepId || undefined}
                  onStepSelect={(stepId) => {
                    dispatch({ type: "SELECT_STEP", payload: stepId })
                  }}
                  onPauseStep={(stepId) => {
                    // Handle step pause if needed
                  }}
                  onResumeStep={(stepId) => {
                    // Handle step resume if needed
                  }}
                  onRetryStep={(stepId) => {
                    // Handle step retry if needed
                  }}
                />
              }
            />
          </div>
        )}
      </div>

      {/* File Preview Modal */}
      <FilePreviewDialog
        open={state.filePreview.isOpen}
        onOpenChange={(open) => {
          if (!open) closeFilePreview()
        }}
      />
    </div>
  )
}

function AgentPageWrapper() {
  const { token } = useAuth()
  return (
    <AppProvider token={token || undefined}>
      <Suspense fallback={<div>Loading...</div>}>
        <AgentContent />
      </Suspense>
    </AppProvider>
  )
}

export default AgentPageWrapper
