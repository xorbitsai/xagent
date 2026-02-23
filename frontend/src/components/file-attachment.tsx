"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  FileText,
  Download,
  Eye,
  Image,
  File,
  Music,
  Video,
  Archive,
  FileCode,
  FileSpreadsheet,
  X
} from "lucide-react"
import { cn, getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"

interface FileInfo {
  name: string
  size: number
  type: string
  path?: string
}

interface FileAttachmentProps {
  files: FileInfo[]
  className?: string
  showPreview?: boolean
  variant?: 'default' | 'user-message'
  onPreview?: (file: FileInfo) => void
}

export function FileAttachment({ files, className, showPreview = true, variant = 'default', onPreview }: FileAttachmentProps) {
  const [previewFile, setPreviewFile] = useState<FileInfo | null>(null)
  const [previewContent, setPreviewContent] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const { t } = useI18n()

  const getFileIcon = (fileType: string) => {
    if (fileType.startsWith('image/')) {
      return <Image className="h-4 w-4" />
    } else if (fileType.startsWith('video/')) {
      return <Video className="h-4 w-4" />
    } else if (fileType.startsWith('audio/')) {
      return <Music className="h-4 w-4" />
    } else if (fileType.includes('pdf')) {
      return <FileText className="h-4 w-4" />
    } else if (fileType.includes('word') || fileType.includes('document')) {
      return <FileText className="h-4 w-4" />
    } else if (fileType.includes('sheet') || fileType.includes('excel')) {
      return <FileSpreadsheet className="h-4 w-4" />
    } else if (fileType.includes('presentation') || fileType.includes('powerpoint')) {
      return <FileText className="h-4 w-4" />
    } else if (fileType.includes('zip') || fileType.includes('rar') || fileType.includes('tar')) {
      return <Archive className="h-4 w-4" />
    } else if (fileType.includes('json') || fileType.includes('xml') || fileType.includes('csv')) {
      return <FileCode className="h-4 w-4" />
    } else {
      return <File className="h-4 w-4" />
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileTypeLabel = (file: FileInfo) => {
    const ext = file.name.includes('.') ? file.name.split('.').pop()?.toLowerCase() : ''
    if (ext) {
      return ext
    }

    if (file.type.startsWith('image/')) return 'image'
    if (file.type.startsWith('text/')) return 'txt'
    if (file.type.includes('pdf')) return 'pdf'

    return t('files.attachment.typeUnknown')
  }

  const handlePreview = async (file: FileInfo) => {
    if (!file.path) return

    if (onPreview) {
      onPreview(file)
      return
    }

    setIsLoading(true)
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(file.path)}`)
      if (response.ok) {
        // For image files, use arrayBuffer to get binary data
        // For text files, use text() for proper encoding
        let fileContent
        if (file.type.startsWith('image/')) {
          const arrayBuffer = await response.arrayBuffer()
          // Convert binary data to base64 using chunks to avoid stack overflow
          const chunkSize = 16384; // 16KB chunks
          const bytes = new Uint8Array(arrayBuffer)
          let binary = ''

          for (let i = 0; i < bytes.length; i += chunkSize) {
            const chunk = bytes.slice(i, i + chunkSize)
            binary += String.fromCharCode.apply(null, Array.from(chunk))
          }
          fileContent = btoa(binary)
        } else {
          // For text files, use text() for proper encoding
          fileContent = await response.text()
        }

        setPreviewContent(fileContent)
        setPreviewFile(file)
      }
    } catch (error) {
      console.error('Failed to preview file:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const handleDownload = async (file: FileInfo) => {
    if (!file.path) return

    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(file.path)}`)
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = file.name
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      }
    } catch (error) {
      console.error('Failed to download file:', error)
    }
  }

  if (files.length === 0) return null

  return (
    <div className={cn("space-y-2 rounded-lg border bg-card/50 p-3", className)}>
      <div className="flex flex-wrap gap-2">
        {files.map((file, index) => (
          <Card
            key={index}
            className={cn(
              "p-2 transition-colors",
              variant === 'user-message'
                ? "bg-accent/50 border-accent/80 hover:bg-accent/70"
                : "border-border bg-card/50"
            )}
          >
            <CardContent className="p-0 flex items-center gap-2 min-w-0">
              <div className={cn(
                "transition-colors",
                variant === 'user-message' ? "text-foreground" : "text-muted-foreground"
              )}>
                {getFileIcon(file.type)}
              </div>
              <div className="flex flex-col min-w-0">
                <span className={cn(
                  "text-sm font-medium truncate",
                  variant === 'user-message'
                    ? "text-foreground max-w-[120px]"
                    : "text-foreground max-w-[200px]"
                )}>
                  {file.name}
                </span>
                <div className="flex items-center gap-1 flex-wrap">
                  <Badge
                    variant={variant === 'user-message' ? "secondary" : "outline"}
                    className={cn(
                      "text-xs shrink-0 px-0",
                      variant === 'user-message'
                        ? "bg-accent/30 text-foreground border-accent/50"
                        : ""
                    )}
                  >
                    {formatFileSize(file.size)}
                  </Badge>
                  <Badge
                    variant={variant === 'user-message' ? "secondary" : "outline"}
                    className={cn(
                      "text-xs shrink-0 pl-0",
                      variant === 'user-message'
                        ? "bg-accent/30 text-foreground border-accent/50"
                        : ""
                    )}
                  >
                    {getFileTypeLabel(file)}
                  </Badge>
                </div>
              </div>
              {showPreview && (
                <div className="flex gap-1">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handlePreview(file)}
                    disabled={isLoading || !file.path}
                    className={cn(
                      "h-6 w-6 p-0",
                      variant === 'user-message'
                        ? "text-foreground hover:text-foreground hover:bg-accent/70"
                        : ""
                    )}
                  >
                    <Eye className="h-3 w-3" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDownload(file)}
                    disabled={!file.path}
                    className={cn(
                      "h-6 w-6 p-0",
                      variant === 'user-message'
                        ? "text-foreground hover:text-foreground hover:bg-accent/70"
                        : ""
                    )}
                  >
                    <Download className="h-3 w-3" />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        ))}
      </div>

      {/* Preview Modal */}
      {previewFile && previewContent && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-4xl max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <div className="flex items-center gap-2">
                {getFileIcon(previewFile.type)}
                <span className="font-medium">{previewFile.name}</span>
                <Badge variant="secondary">{formatFileSize(previewFile.size)}</Badge>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => {
                  setPreviewFile(null)
                  setPreviewContent(null)
                }}
                aria-label={t('common.cancel')}
                title={t('common.cancel')}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
            <div className="flex-1 overflow-auto p-4">
              {previewFile.type.startsWith('image/') ? (
                <img
                  src={`data:${previewFile.type};base64,${previewContent}`}
                  alt={previewFile.name}
                  className="max-w-full max-h-full mx-auto"
                />
              ) : (
                <pre className="whitespace-pre-wrap text-sm bg-muted p-4 rounded overflow-auto max-h-full">
                  {previewContent}
                </pre>
              )}
            </div>
            <div className="p-4 border-t flex justify-end">
              <Button
                variant="outline"
                onClick={() => handleDownload(previewFile)}
                title={t('files.attachment.preview.download')}
              >
                <Download className="h-4 w-4 mr-2" />
                {t('files.attachment.preview.download')}
              </Button>
            </div>
          </Card>
        </div>
      )}
    </div>
  )
}
