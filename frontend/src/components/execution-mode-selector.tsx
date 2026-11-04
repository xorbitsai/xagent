"use client"

import { useState, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Textarea } from "@/components/ui/textarea"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Plus, Trash2, Zap, Workflow, Upload, XCircle, Paperclip, Sparkles, Brain } from "lucide-react"
import { cn } from "@/lib/utils"
import { useI18n } from "@/contexts/i18n-context"

export type AgentPattern = "flash" | "balanced" | "think"

export interface ExecutionModeConfig {
  mode: AgentPattern
  processDescription?: string
  examples?: Array<{ input: string; output: string }>
}

interface ExecutionModeSelectorProps {
  config: ExecutionModeConfig
  onChange: (config: ExecutionModeConfig) => void
  disabled?: boolean
  selectedFiles?: File[]
  onFilesChange?: (files: File[]) => void
}

interface ExampleItem {
  input: string
  output: string
}

export function ExecutionModeSelector({ config, onChange, disabled, selectedFiles: externalSelectedFiles, onFilesChange }: ExecutionModeSelectorProps) {
  const [examples, setExamples] = useState<ExampleItem[]>(
    config.examples || [
      { input: "", output: "" }
    ]
  )

  const { t } = useI18n()

  // File management
  const [internalSelectedFiles, setInternalSelectedFiles] = useState<File[]>([])
  const files = externalSelectedFiles || internalSelectedFiles
  const fileInputRef = useRef<HTMLInputElement>(null)

  const setFiles = useCallback((newFiles: File[] | ((prev: File[]) => File[])) => {
    if (typeof newFiles === 'function') {
      const updatedFiles = newFiles(externalSelectedFiles || internalSelectedFiles)
      if (onFilesChange) {
        onFilesChange(updatedFiles)
      } else {
        setInternalSelectedFiles(updatedFiles)
      }
    } else {
      if (onFilesChange) {
        onFilesChange(newFiles)
      } else {
        setInternalSelectedFiles(newFiles)
      }
    }
  }, [onFilesChange, externalSelectedFiles, internalSelectedFiles])

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = event.target.files
    if (selectedFiles) {
      setFiles((prev: File[]) => [...prev, ...Array.from(selectedFiles)])
    }
  }

  const removeFile = (index: number) => {
    setFiles((prev: File[]) => prev.filter((_, i) => i !== index))
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

  const handleModeChange = (mode: AgentPattern) => {
    onChange({ ...config, mode })
  }

  const handleProcessDescriptionChange = (value: string) => {
    onChange({ ...config, processDescription: value })
  }

  const handleExampleChange = (index: number, field: "input" | "output", value: string) => {
    const newExamples = [...examples]
    newExamples[index][field] = value
    setExamples(newExamples)
    onChange({ ...config, examples: newExamples.filter(ex => ex.input || ex.output) })
  }

  const addExample = () => {
    const newExamples = [...examples, { input: "", output: "" }]
    setExamples(newExamples)
  }

  const removeExample = (index: number) => {
    if (examples.length === 1) {
      // If it's the only example, just clear it instead of removing
      const newExamples = [{ input: "", output: "" }]
      setExamples(newExamples)
      onChange({ ...config, examples: [] })
    } else {
      const newExamples = examples.filter((_, i) => i !== index)
      setExamples(newExamples)
      onChange({ ...config, examples: newExamples.filter(ex => ex.input || ex.output) })
    }
  }

  return (
    <div className="space-y-4">
      {/* Mode Selection Tabs */}
      <div className="grid grid-cols-3 gap-2">
        <Button
          type="button"
          variant={config.mode === "flash" ? "default" : "outline"}
          onClick={() => handleModeChange("flash")}
          disabled={disabled}
          className="flex-col gap-1 h-auto py-3"
        >
          <Zap className="h-4 w-4" />
          <span className="text-xs font-medium">{t('agent.vibeMode.tabs.flash')}</span>
          <span className="text-[10px] text-muted-foreground">{t('agent.vibeMode.tabs.flashSubtitle')}</span>
        </Button>
        <Button
          type="button"
          variant={config.mode === "balanced" ? "default" : "outline"}
          onClick={() => handleModeChange("balanced")}
          disabled={disabled}
          className="flex-col gap-1 h-auto py-3"
        >
          <Sparkles className="h-4 w-4" />
          <span className="text-xs font-medium">{t('agent.vibeMode.tabs.balanced')}</span>
          <span className="text-[10px] text-muted-foreground">{t('agent.vibeMode.tabs.balancedSubtitle')}</span>
        </Button>
        <Button
          type="button"
          variant={config.mode === "think" ? "default" : "outline"}
          onClick={() => handleModeChange("think")}
          disabled={disabled}
          className="flex-col gap-1 h-auto py-3"
        >
          <Brain className="h-4 w-4" />
          <span className="text-xs font-medium">{t('agent.vibeMode.tabs.think')}</span>
          <span className="text-[10px] text-muted-foreground">{t('agent.vibeMode.tabs.thinkSubtitle')}</span>
        </Button>
      </div>

      {/* Mode Description */}
      {config.mode === "flash" && (
        <Card className="p-4 bg-muted/30 border-border">
          <div className="text-sm space-y-2">
            <div className="font-semibold text-foreground">{t('agent.vibeMode.descriptions.flash.title')}</div>
            <div className="text-muted-foreground">
              {t('agent.vibeMode.descriptions.flash.text')}
            </div>
          </div>
        </Card>
      )}

      {config.mode === "balanced" && (
        <Card className="p-4 bg-muted/30 border-border">
          <div className="text-sm space-y-2">
            <div className="font-semibold text-foreground">{t('agent.vibeMode.descriptions.balanced.title')}</div>
            <div className="text-muted-foreground">
              {t('agent.vibeMode.descriptions.balanced.text')}
            </div>
          </div>
        </Card>
      )}

      {config.mode === "think" && (
        <div className="space-y-4">
          <Card className="p-4 bg-muted/30 border-border">
            <div className="text-sm space-y-2">
              <div className="font-semibold text-foreground">{t('agent.vibeMode.descriptions.think.title')}</div>
              <div className="text-muted-foreground">
                {t('agent.vibeMode.descriptions.think.text')}
              </div>
            </div>
          </Card>

          {/* Process Description */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-foreground">
                {t('agent.vibeMode.form.processDescription.label')} <span className="text-destructive">*</span>
              </label>
              {/* File Upload Button */}
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileSelect}
                multiple
                className="hidden"
                disabled={disabled}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => fileInputRef.current?.click()}
                disabled={disabled}
                className="h-7 px-2"
              >
                <Upload className="h-3 w-3 mr-1" />
                {t('agent.input.actions.uploadFile')}
              </Button>
            </div>

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
            <Textarea
              placeholder={t('agent.vibeMode.form.processDescription.placeholder')}
              value={config.processDescription || ""}
              onChange={(e) => handleProcessDescriptionChange(e.target.value)}
              onPaste={handlePaste}
              disabled={disabled}
              rows={6}
              className="bg-background border-border text-foreground text-sm"
            />
          </div>

          {/* Input/Output Examples */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-foreground">
                {t('agent.vibeMode.form.examples.label')} <span className="text-muted-foreground">{t('agent.vibeMode.form.examples.optional')}</span>
              </label>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={addExample}
                disabled={disabled}
                className="h-7 px-2"
              >
                <Plus className="h-3 w-3 mr-1" />
                {t('agent.vibeMode.form.examples.add')}
              </Button>
            </div>

            <div className="space-y-2">
              {examples.map((example, index) => (
                <Card key={index} className="p-3 bg-card border-border">
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="text-xs">
                        {t('agent.vibeMode.form.examples.badge', { index: index + 1 })}
                      </Badge>
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        onClick={() => removeExample(index)}
                        disabled={disabled}
                        className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                      >
                        <Trash2 className="h-3 w-3" />
                      </Button>
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-muted-foreground">{t('agent.vibeMode.form.examples.input.label')}</label>
                      <Input
                        placeholder={t('agent.vibeMode.form.examples.input.placeholder')}
                        value={example.input}
                        onChange={(e) => handleExampleChange(index, "input", e.target.value)}
                        disabled={disabled}
                        className="bg-background border-border text-foreground text-xs h-8"
                      />
                    </div>

                    <div className="space-y-1">
                      <label className="text-xs text-muted-foreground">{t('agent.vibeMode.form.examples.output.label')}</label>
                      <Textarea
                        placeholder={t('agent.vibeMode.form.examples.output.placeholder')}
                        value={example.output}
                        onChange={(e) => handleExampleChange(index, "output", e.target.value)}
                        disabled={disabled}
                        rows={1}
                        className="bg-background border-border text-foreground text-xs h-8 resize-none py-1"
                      />
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

// Backward compatibility: export old names as aliases
export const VibeModeSelector = ExecutionModeSelector
export type VibeModeConfig = ExecutionModeConfig
