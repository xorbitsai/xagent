"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"
import { Send, Upload, Pause, Play, Paperclip, XCircle, Settings } from "lucide-react"
import { cn, getApiUrl } from "@/lib/utils"
import { ConfigDialog } from "@/components/config-dialog"
import { ModelInfoDisplay } from "@/components/model-info-display"
import { useI18n } from "@/contexts/i18n-context"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string | React.ReactNode
  timestamp: string | number
  status?: "pending" | "running" | "completed" | "failed"
  isResult?: boolean
  isFileOutput?: boolean
}

interface Task {
  id: string
  title: string
  status: "pending" | "running" | "completed" | "failed" | "paused" | "waiting_for_user"
  description: string
  createdAt: string | number
  updatedAt: string | number
  modelName?: string
  smallFastModelName?: string
  visualModelName?: string
}

interface AgentConfig {
  model: string
  smallFastModel?: string
  visualModel?: string
  compactModel?: string
  memorySimilarityThreshold?: number
}

interface AgentInputProps {
  value: string
  onChange: (value: string) => void
  onSend: (files?: File[]) => void | Promise<void>
  onKeyDown?: (e: React.KeyboardEvent) => void
  onCompositionStart?: () => void
  onCompositionEnd?: () => void
  placeholder?: string
  disabled?: boolean
  isProcessing?: boolean
  agentConfig?: AgentConfig
  onConfigChange?: (config: AgentConfig) => void
  currentTask?: Task | null
  onPauseTask?: () => void
  onResumeTask?: () => void
  className?: string
  rows?: number
  showStatus?: boolean
  variant?: "compact" | "expanded"
  selectedFiles?: File[]
  setSelectedFiles?: (files: File[]) => void
}

export function AgentInput({
  value,
  onChange,
  onSend,
  onKeyDown,
  onCompositionStart,
  onCompositionEnd,
  placeholder,
  disabled = false,
  isProcessing = false,
  agentConfig,
  onConfigChange,
  currentTask,
  onPauseTask,
  onResumeTask,
  className = "",
  rows = 3,
  showStatus = true,
  variant = "compact",
  selectedFiles: externalSelectedFiles,
  setSelectedFiles: externalSetSelectedFiles
}: AgentInputProps) {
  // Use external file state if provided, otherwise use internal state
  const [internalSelectedFiles, setInternalSelectedFiles] = useState<File[]>([])
  const files = externalSelectedFiles || internalSelectedFiles

  const { t } = useI18n()

  const setFiles = useCallback((newFiles: File[] | ((prev: File[]) => File[])) => {
    if (typeof newFiles === 'function') {
      if (externalSetSelectedFiles) {
        externalSetSelectedFiles(newFiles(externalSelectedFiles || []))
      } else {
        setInternalSelectedFiles(newFiles(internalSelectedFiles))
      }
    } else {
      if (externalSetSelectedFiles) {
        externalSetSelectedFiles(newFiles)
      } else {
        setInternalSelectedFiles(newFiles)
      }
    }
  }, [externalSetSelectedFiles, externalSelectedFiles, internalSelectedFiles, setInternalSelectedFiles])

  const [isComposing, setIsComposing] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const isSubmittingRef = useRef(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const hasDraft = value.trim().length > 0 || files.length > 0

  const clearSubmitting = useCallback(() => {
    isSubmittingRef.current = false
    setIsSubmitting(false)
  }, [])

  useEffect(() => {
    if (isSubmittingRef.current && !hasDraft) {
      clearSubmitting()
    }
  }, [clearSubmitting, hasDraft])

  // Debug: Log currentTask and onPauseTask
  useEffect(() => {
    console.log('🔍 AgentInput Debug:', {
      hasCurrentTask: !!currentTask,
      currentTaskStatus: currentTask?.status,
      hasOnPauseTask: !!onPauseTask,
      hasOnResumeTask: !!onResumeTask,
      isProcessing,
      disabled
    })
  }, [currentTask, onPauseTask, onResumeTask, isProcessing, disabled])

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getStatusBadge = (status: Task["status"]) => {
    const variants = {
      pending: "secondary",
      running: "default",
      completed: "default",
      failed: "destructive",
      paused: "secondary",
      waiting_for_user: "secondary"
    } as const

    const labels = {
      pending: t('agent.input.status.labels.pending'),
      running: t('agent.input.status.labels.running'),
      completed: t('agent.input.status.labels.completed'),
      failed: t('agent.input.status.labels.failed'),
      paused: t('agent.input.status.labels.paused'),
      waiting_for_user: t('agent.input.status.labels.waitingForUser')
    }

    const customStyles = {
      pending: "bg-muted/50 text-muted-foreground border-border",
      running: "bg-primary/10 text-primary border-primary/20",
      completed: "bg-green-500/10 text-green-500 border-green-500/20",
      failed: "bg-destructive/10 text-destructive border-destructive/20",
      paused: "bg-secondary text-secondary-foreground border-border",
      waiting_for_user: "bg-secondary text-secondary-foreground border-border"
    }

    return (
      <Badge
        variant={variants[status]}
        className={`text-xs border ${customStyles[status]}`}
      >
        {labels[status]}
      </Badge>
    )
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files
    if (selectedFiles) {
      setFiles((prev: File[]) => [...prev, ...Array.from(selectedFiles)])
    }
  }

  const handleFileUpload = async () => {
    // Files will be handled by WebSocket message directly
    console.log('Files will be sent via WebSocket message')
  }

  const removeFile = (index: number) => {
    setFiles((prev: File[]) => prev.filter((_, i) => i !== index))
  }

  const handleSend = async () => {
    if (isSubmittingRef.current || !hasDraft) return

    console.log('🚀 AgentInput handleSend called:', {
      value: value.trim(),
      selectedFiles: files.map(f => f.name),
      hasFiles: files.length > 0
    })

    console.log('📤 AgentInput calling onSend with files:', files.map(f => f.name))
    try {
      isSubmittingRef.current = true
      setIsSubmitting(true)
      await onSend(files.length > 0 ? files : undefined)
      setFiles([])
    } catch (error) {
      clearSubmitting()
      throw error
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey && !isComposing) {
      e.preventDefault()
      handleSend()
    }
    onKeyDown?.(e)
  }

  const handlePaste = (e: React.ClipboardEvent) => {
    const items = Array.from(e.clipboardData.items)
    const imageItems = items.filter(item => item.type.startsWith('image/'))

    if (imageItems.length > 0) {
      e.preventDefault()

      imageItems.forEach(item => {
        const file = item.getAsFile()
        if (file) {
          // Create a new file with a proper name
          const timestamp = new Date().getTime()
          const extension = item.type.split('/')[1] || 'png'
          const namedFile = new File([file], `pasted-image-${timestamp}.${extension}`, {
            type: item.type,
            lastModified: Date.now()
          })
          setFiles((prev: File[]) => [...prev, namedFile])
        }
      })
    }
  }

  // Allow input in active-control states so the user can guide a running task.
  const allowsLiveGuidanceInput =
    currentTask?.status === 'running' ||
    currentTask?.status === 'paused' ||
    currentTask?.status === 'waiting_for_user'
  const isPaused = currentTask?.status === 'paused' || currentTask?.status === 'waiting_for_user'
  const isSendDisabled = !hasDraft || disabled || isSubmitting || (!allowsLiveGuidanceInput && isProcessing)

  // Dynamic placeholder
  const dynamicPlaceholder = isPaused
    ? t('agent.input.placeholder.continueExecution')
    : placeholder

  return (
    <div className={cn("space-y-3", className)}>
      {/* Selected Files */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {files.map((file, index) => (
            <div key={index} className="flex items-center gap-1 px-2 py-1 bg-muted/50 rounded-md border border-border">
              <Paperclip className="h-3 w-3 text-muted-foreground" />
              <span className="text-xs truncate max-w-[120px]">{file.name}</span>
              <span className="text-xs text-muted-foreground">{formatFileSize(file.size)}</span>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => removeFile(index)}
                className="h-4 w-4 p-0 text-muted-foreground hover:text-destructive"
              >
                <XCircle className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>
      )}

      {/* Input Container */}
      <div>

        <Textarea
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyPress}
          onPaste={handlePaste}
          onCompositionStart={() => {
            setIsComposing(true)
            onCompositionStart?.()
          }}
          onCompositionEnd={() => {
            setIsComposing(false)
            onCompositionEnd?.()
          }}
          placeholder={dynamicPlaceholder ?? t('agent.input.placeholder.default')}
          className={cn(
            "resize-none bg-background border-border focus:border-primary",
            variant === "expanded"
              ? "text-base"
              : "",
            variant === "expanded" && "text-base"
          )}
          rows={rows}
          disabled={disabled || (!allowsLiveGuidanceInput && isProcessing)}
          autoFocus={variant === "expanded"}
        />

        {/* Action Buttons Row */}
        <div className="flex items-center justify-between gap-2 mt-2">
          {/* Left side - Config Button */}
          {onConfigChange && (
            <ConfigDialog
              onConfigChange={onConfigChange}
              currentConfig={agentConfig}
              trigger={
                <div className="flex items-center gap-2">
                  <ModelInfoDisplay
                    currentTask={currentTask}
                    onConfigChange={undefined} // Don't pass onConfigChange to avoid double handling
                  />
                  {!currentTask && (
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
                      title={t('agent.input.actions.config')}
                    >
                      <Settings className="h-3.5 w-3.5" />
                    </Button>
                  )}
                </div>
              }
            />
          )}

          {/* Right side - Action Buttons */}
          <div className="flex gap-1">
            {/* File Upload Button */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileSelect}
              multiple
              className="hidden"
              disabled={disabled || (!allowsLiveGuidanceInput && isProcessing)}
            />
            <Button
              variant="ghost"
              size="sm"
              onClick={() => fileInputRef.current?.click()}
              disabled={disabled || (!allowsLiveGuidanceInput && isProcessing)}
              className="h-7 w-7 p-0 text-muted-foreground hover:text-foreground hover:bg-muted rounded-md"
              title={t('agent.input.actions.uploadFile')}
            >
              <Upload className="h-3.5 w-3.5" />
            </Button>

            {/* Send Button */}
            <Button
              onClick={handleSend}
              disabled={isSendDisabled}
              size="sm"
              className="h-7 w-7 p-0 bg-primary hover:bg-primary/90 rounded-md"
              title={files.length > 0 ? t('agent.input.actions.sendWithFiles') : t('agent.input.actions.sendMessage')}
            >
              <Send className="h-3.5 w-3.5" />
            </Button>

            {/* Pause/Resume Button */}
            {currentTask && currentTask.status === 'running' && !hasDraft && onPauseTask && (
              <Button
                onClick={onPauseTask}
                disabled={disabled}
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0 text-destructive hover:text-destructive hover:bg-destructive/10 rounded-md"
                title={t('agent.input.actions.pauseTask')}
              >
                <Pause className="h-3.5 w-3.5" />
              </Button>
            )}

            {currentTask && currentTask.status === 'paused' && onResumeTask && (
              <Button
                onClick={onResumeTask}
                disabled={disabled}
                size="sm"
                variant="ghost"
                className="h-7 w-7 p-0 text-green-500 hover:text-green-600 hover:bg-green-500/10 rounded-md"
                title={t('agent.input.actions.resumeTask')}
              >
                <Play className="h-3.5 w-3.5" />
              </Button>
            )}
          </div>
        </div>
      </div>

      {/* Status Indicators */}
      {showStatus && (
        <div className="flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            {files.length > 0 && (
              <span>{t('agent.input.status.selectedFiles', { count: files.length })}</span>
            )}
            {currentTask && (
              <span>{t('agent.input.status.taskStatusPrefix')}{getStatusBadge(currentTask.status)}</span>
            )}
          </div>
          <div>
            {isProcessing && <span>{t('agent.input.status.processing')}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
