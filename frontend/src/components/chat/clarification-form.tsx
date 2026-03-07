import React, { useState, useEffect, useRef } from "react"
import { Interaction } from "@/contexts/app-context-chat"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Switch } from "@/components/ui/switch"
import { MultiSelect } from "@/components/ui/multi-select"
import { Select } from "@/components/ui/select"
import { useApp } from "@/contexts/app-context-chat"
import { useI18n } from "@/contexts/i18n-context"
import { toast } from "sonner"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { ChevronDown, ChevronRight, MessageSquare, Upload, File as FileIcon, X } from "lucide-react"

interface ClarificationFormProps {
  message?: string
  interactions: Interaction[]
  timeout?: number
  messageId?: string
  expiresAt?: string
}

export function ClarificationForm({ interactions, timeout, messageId, expiresAt }: ClarificationFormProps) {
  const { state, sendMessage } = useApp()
  const { t } = useI18n()
  const [formState, setFormState] = useState<Record<string, any>>({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [isOpen, setIsOpen] = useState(true)

  const isTaskRunning = state.currentTask?.status === "running"

  // Check if this is the last message
  const isLastMessage = React.useMemo(() => {
    if (!messageId || !state.messages.length) return true
    const lastMsg = state.messages[state.messages.length - 1]
    return lastMsg.id === messageId
  }, [state.messages, messageId])

  const [timeLeft, setTimeLeft] = useState<number | null>(() => {
    if (expiresAt) {
      const now = new Date().getTime()
      const end = new Date(expiresAt).getTime()
      return Math.max(0, Math.floor((end - now) / 1000))
    }
    return timeout || null
  })
  const hasAutoSubmitted = useRef(false)

  useEffect(() => {
    if (!isLastMessage) {
      setTimeLeft(null)
      return
    }

    // Determine initial time left and start timer
    let initialTimeLeft: number | null = null
    if (expiresAt) {
      const now = new Date().getTime()
      const end = new Date(expiresAt).getTime()
      initialTimeLeft = Math.max(0, Math.floor((end - now) / 1000))
    } else if (timeout !== undefined && timeout !== null && timeout > 0) {
      initialTimeLeft = timeout
    }

    if (initialTimeLeft === null) {
      setTimeLeft(null)
      return
    }

    // If already expired (and using expiresAt), trigger auto-continue immediately
    if (expiresAt && initialTimeLeft <= 0) {
      setTimeLeft(0)
      // Use setTimeout to ensure state updates complete before triggering
      setTimeout(() => {
        if (!hasAutoSubmitted.current && !isSubmitting && !isTaskRunning) {
          handleAutoContinue()
        }
      }, 0)
      return
    }

    setTimeLeft(initialTimeLeft)

    const timer = setInterval(() => {
      if (expiresAt) {
        // Recalculate based on current time
        const now = new Date().getTime()
        const end = new Date(expiresAt).getTime()
        const diff = Math.max(0, Math.floor((end - now) / 1000))

        setTimeLeft(diff)

        if (diff <= 0) {
          clearInterval(timer)
        }
      } else {
        // Legacy decrement
        setTimeLeft((prev) => {
          if (prev === null) return null
          if (prev <= 1) {
            clearInterval(timer)
            return 0
          }
          return prev - 1
        })
      }
    }, 1000)

    return () => clearInterval(timer)
  }, [timeout, expiresAt, isLastMessage])

  // Auto-trigger continue when timeLeft reaches 0
  useEffect(() => {
    if (timeLeft === 0 && isLastMessage && !hasAutoSubmitted.current && !isSubmitting && !isTaskRunning) {
      handleAutoContinue()
    }
  }, [timeLeft, isLastMessage, isSubmitting, isTaskRunning])

  const handleAutoContinue = async () => {
    if (hasAutoSubmitted.current || isSubmitting || isTaskRunning) return
    hasAutoSubmitted.current = true

    try {
      setIsSubmitting(true)
      await sendMessage(t("chatPage.clarification.continue"), { force: true })
      setIsOpen(false)
      // Hide the continue button by clearing the timeout
      setTimeLeft(null)
    } catch (error) {
      console.error("Failed to auto-continue", error)
    } finally {
      setIsSubmitting(false)
    }
  }

  const handleInputChange = (field: string, value: any) => {
    setFormState((prev) => ({ ...prev, [field]: value }))
  }

  const handleSubmit = async () => {
    // Construct the message
    const lines = interactions.map(interaction => {
      const value = formState[interaction.field]

      // Skip empty values unless it's a boolean (confirm) which might be false
      if (value === undefined || value === null || (typeof value === "string" && value.trim() === "") || (Array.isArray(value) && value.length === 0)) {
          // If it's a confirm type, default to false if undefined? Or maybe it's required?
          if (interaction.type === "confirm" && value === undefined) {
              return { field: interaction.field, label: interaction.label || interaction.field, value: t("chatPage.clarification.no"), isFile: false }
          }
          return null
      }

      let displayValue = value
      let isFile = false

      if (interaction.type === "select_multiple" && Array.isArray(value)) {
        const labels = value.map(v => interaction.options?.find(o => o.value === v)?.label || v)
        displayValue = labels.join(", ")
      } else if (interaction.type === "select_one") {
         const label = interaction.options?.find(o => o.value === value)?.label || value
         displayValue = label
      } else if (interaction.type === "confirm") {
        displayValue = value ? t("chatPage.clarification.yes") : t("chatPage.clarification.no")
      } else if (interaction.type === "file_upload") {
         isFile = true
      }

      return { field: interaction.field, label: interaction.label || interaction.field, value: isFile ? value : displayValue, isFile }
    }).filter(Boolean) as any[]

    if (lines.length === 0) {
        toast.error(t("chatPage.clarification.required"))
        return
    }

    // Separate files and text
    const textParts = lines.filter(l => !l.isFile).map(l => `${l.label}: ${l.value}`)
    const fileParts = lines.filter(l => l.isFile)

    const textMessage = textParts.join("\n")
    const files: File[] = []

    fileParts.forEach(part => {
        if (part.value instanceof FileList) {
            for (let i = 0; i < part.value.length; i++) {
                files.push(part.value[i])
            }
        } else if (Array.isArray(part.value)) {
             // Assuming array of Files
             part.value.forEach((f: any) => {
                 if (f instanceof File) files.push(f)
             })
        }
    })

    try {
        setIsSubmitting(true)
        // If textMessage is empty but we have files, send a generic message?
        const finalMessage = textMessage || (files.length > 0 ? t("chatPage.clarification.uploadedFiles") : t("chatPage.clarification.confirmed"))

        await sendMessage(finalMessage, { force: true }, files)
        setIsOpen(false) // Collapse after submission
    } catch (error) {
        console.error("Failed to send clarification response", error)
        toast.error(t("chatPage.clarification.sendError"))
    } finally {
        setIsSubmitting(false)
    }
  }

  const renderField = (interaction: Interaction) => {
    const value = formState[interaction.field]

    switch (interaction.type) {
      case "text_input":
        return interaction.multiline ? (
          <Textarea
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        ) : (
          <Input
            placeholder={interaction.placeholder}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "number_input":
        return (
          <Input
            type="number"
            placeholder={interaction.placeholder}
            min={interaction.min}
            max={interaction.max}
            value={value || ""}
            onChange={(e) => handleInputChange(interaction.field, e.target.value)}
          />
        )

      case "select_one":
        return (
          <Select
            value={value}
            onValueChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={t("chatPage.clarification.selectOption")}
          />
        )

      case "select_multiple":
        return (
          <MultiSelect
            values={value || []}
            onValuesChange={(v) => handleInputChange(interaction.field, v)}
            options={interaction.options || []}
            placeholder={interaction.placeholder || t("chatPage.clarification.selectOptions")}
          />
        )

      case "file_upload":
        const fileValue = formState[interaction.field]
        const files: File[] = []
        if (fileValue instanceof FileList) {
          for (let i = 0; i < fileValue.length; i++) {
            files.push(fileValue[i])
          }
        } else if (Array.isArray(fileValue)) {
          files.push(...fileValue)
        }

        const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
          if (e.target.files && e.target.files.length > 0) {
            const newFiles = Array.from(e.target.files)
            // Always allow multiple files
            handleInputChange(interaction.field, [...files, ...newFiles])
          }
          // Reset input value to allow selecting same file again
          e.target.value = ''
        }

        const removeFile = (index: number) => {
          const newFiles = [...files]
          newFiles.splice(index, 1)
          handleInputChange(interaction.field, newFiles)
        }

        return (
          <div className="grid w-full gap-2">
            {files.length > 0 && (
              <div className="grid gap-2">
                {files.map((file, index) => (
                  <div key={index} className="flex items-center gap-2 rounded-md border p-2 text-sm bg-muted/50">
                    <FileIcon className="h-4 w-4 text-muted-foreground shrink-0" />
                    <span className="flex-1 truncate">{file.name}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={() => removeFile(index)}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}

            {/* Always show upload area to allow adding more files */}
            <div className="relative group cursor-pointer">
              <div className="flex h-24 w-full flex-col items-center justify-center gap-2 rounded-md border border-dashed bg-muted/30 hover:bg-muted/50 transition-colors">
                <Upload className="h-6 w-6 text-muted-foreground group-hover:scale-110 transition-transform" />
                <span className="text-xs text-muted-foreground">{t("chatPage.fileUpload.hintDragClick")}</span>
              </div>
              <Input
                type="file"
                className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                accept={Array.isArray(interaction.accept) ? interaction.accept.join(",") : interaction.accept}
                multiple={interaction.multiple ?? true}
                onChange={handleFileChange}
              />
            </div>
            <div className="text-xs text-muted-foreground">
              {t("chatPage.clarification.acceptedFormats")}: {Array.isArray(interaction.accept) ? interaction.accept.join(", ") : interaction.accept || t("chatPage.clarification.any")}
            </div>
          </div>
        )

      case "confirm":
        return (
          <div className="flex items-center space-x-2">
            <Switch
              id={interaction.field}
              checked={!!value}
              onCheckedChange={(checked) => handleInputChange(interaction.field, checked)}
            />
            <Label htmlFor={interaction.field}>{t("chatPage.clarification.yes")}</Label>
          </div>
        )

      default:
        return <div className="text-destructive text-sm">{t("chatPage.clarification.unsupportedType", { type: interaction.type })}</div>
    }
  }

  return (
    <Collapsible
      open={isOpen}
      onOpenChange={setIsOpen}
      className="w-full space-y-2 rounded-lg border bg-card text-card-foreground shadow-sm my-2"
    >
      <CollapsibleTrigger asChild>
        <div className="flex items-center justify-between p-4 bg-muted/80 cursor-pointer hover:bg-muted/60 transition-colors">
          <div className="flex items-center gap-2 font-semibold">
            <MessageSquare className="h-4 w-4" />
            <span className="text-sm">{t("chatPage.clarification.title")}</span>
          </div>
          <div>
            {isOpen ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
          </div>
        </div>
      </CollapsibleTrigger>

      <CollapsibleContent className="space-y-4 p-4">
        <div className="space-y-4">
          {interactions.map((interaction, index) => (
            <div key={`${interaction.field}-${index}`} className="space-y-2">
              <Label className="text-sm font-medium">
                {interaction.label || interaction.field}
                {interaction.type === "confirm" ? "" : ":"}
              </Label>

              {renderField(interaction)}
            </div>
          ))}
        </div>

        <div className="pt-2 flex gap-2">
          {timeLeft !== null && timeLeft > 0 && (
            <Button
              variant="outline"
              size="sm"
              onClick={handleAutoContinue}
              disabled={isSubmitting || isTaskRunning}
            >
              {t("chatPage.clarification.continue")} ({timeLeft}s)
            </Button>
          )}
          <Button className="flex-1" size="sm" onClick={handleSubmit} disabled={isSubmitting || isTaskRunning}>
            {isSubmitting ? t("chatPage.clarification.submitting") : t("chatPage.clarification.submit")}
          </Button>
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
