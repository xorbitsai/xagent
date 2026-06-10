"use client"

import React from "react"
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { getApiUrl, getUploadApiUrl } from "@/lib/utils"
import { apiRequest, getApiErrorMessage, getUploadErrorMessage, parseApiResponse, UPLOAD_ERROR_MESSAGES } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { StandaloneFilePreviewDialog } from "@/components/file/standalone-file-preview-dialog"
import { SearchInput } from "@/components/ui/search-input"
import { ConfirmDialog } from "@/components/ui/confirm-dialog"
import { toast } from "@/components/ui/sonner"
import {
  Upload,
  FileText,
  Image as ImageIcon,
  Video,
  Archive,
  Download,
  Trash2,
  FileCode,
  FileJson,
  FileSpreadsheet,
  Folder,
  LayoutGrid,
  Eye,
  MessageSquare,
  ChevronLeft,
  ChevronRight
} from "lucide-react"
import { cn } from "@/lib/utils"

interface FileItem {
  file_id: string
  filename: string
  file_size: number
  modified_time: number
  file_type?: string
  task_id?: number | null
  workspace_id?: string
  relative_path?: string
}

interface TaskFilterItem {
  task_id: number
  title: string
  file_count?: number
}

export function FilesPage() {
  const pageSize = 20
  const [files, setFiles] = useState<FileItem[]>([])
  const [tasks, setTasks] = useState<TaskFilterItem[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [debouncedSearchQuery, setDebouncedSearchQuery] = useState("")
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [page, setPage] = useState(1)
  const [pages, setPages] = useState(1)
  const [totalFiles, setTotalFiles] = useState(0)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const latestFilesRequestRef = useRef(0)

  const { t } = useI18n()

  const formatRelativeTime = (timestamp: number): string => {
    const now = Date.now()
    const diff = now - timestamp * 1000 // timestamp is in seconds

    const minute = 60 * 1000
    const hour = 60 * minute
    const day = 24 * hour
    const month = 30 * day
    const year = 365 * day

    if (diff < minute) return t('files.time.justNow')
    if (diff < hour) return t('files.time.minsAgo', { count: Math.floor(diff / minute) })
    if (diff < day) return t('files.time.hoursAgo', { count: Math.floor(diff / hour) })
    if (diff < month) return t('files.time.daysAgo', { count: Math.floor(diff / day) })
    if (diff < year) return t('files.time.monthsAgo', { count: Math.floor(diff / month) })
    return t('files.time.yearsAgo', { count: Math.floor(diff / year) })
  }

  // File preview state
  const [previewFile, setPreviewFile] = useState<{ fileId: string; fileName: string } | null>(null)
  const [isPreviewOpen, setIsPreviewOpen] = useState(false)

  useEffect(() => {
    loadTasks()
  }, [])

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setDebouncedSearchQuery(searchQuery)
      setPage(1)
    }, 300)

    return () => window.clearTimeout(timer)
  }, [searchQuery])

  useEffect(() => {
    loadFiles(page, debouncedSearchQuery, selectedCategory)
  }, [page, debouncedSearchQuery, selectedCategory])

  const loadTasks = async () => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/tasks`)
      if (response.ok) {
        const data = await response.json()
        setTasks(Array.isArray(data?.tasks) ? data.tasks : [])
      }
    } catch (error) {
      console.error('Failed to load tasks:', error)
    }
  }

  const loadFiles = async (targetPage = page, targetSearch = debouncedSearchQuery, targetCategory = selectedCategory) => {
    const requestId = latestFilesRequestRef.current + 1
    latestFilesRequestRef.current = requestId
    try {
      setIsLoading(true)
      const params = new URLSearchParams({
        page: targetPage.toString(),
        size: pageSize.toString(),
      })
      const normalizedSearch = targetSearch.trim()
      if (normalizedSearch) {
        params.set('search', normalizedSearch)
      }
      if (targetCategory === 'uploads') {
        params.set('uploads_only', 'true')
      } else if (targetCategory.startsWith('task-')) {
        params.set('task_id', targetCategory.split('-')[1])
      }

      const response = await apiRequest(`${getApiUrl()}/api/files/list?${params.toString()}`)
      if (response.ok) {
        const data = await response.json()
        if (requestId === latestFilesRequestRef.current && data && data.files) {
          setFiles(data.files)
          setSelectedFiles(prev => prev.filter(fileId => data.files.some((file: FileItem) => file.file_id === fileId)))
          setTotalFiles(typeof data.total_count === 'number' ? data.total_count : data.files.length)
          setPage(typeof data.page === 'number' ? data.page : targetPage)
          setPages(typeof data.pages === 'number' && data.pages > 0 ? data.pages : 1)
        }
      }
    } catch (error) {
      if (requestId === latestFilesRequestRef.current) {
        console.error('Failed to load files:', error)
      }
    } finally {
      if (requestId === latestFilesRequestRef.current) {
        setIsLoading(false)
      }
    }
  }

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files
    if (!files || files.length === 0) return

    setUploading(true)

    try {
      const formData = new FormData()
      Array.from(files).forEach(file => {
        formData.append('files', file)
      })
      formData.append('task_type', 'general')
      formData.append('message', '')

      const response = await apiRequest(`${getUploadApiUrl()}/api/files/upload`, {
        method: 'POST',
        body: formData
      })

      const parsed = await parseApiResponse(response)

      if (response.ok && parsed.data) {
        await loadTasks()
        if (page === 1) {
          await loadFiles(1)
        } else {
          setPage(1)
        }
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      } else {
        throw new Error(getUploadErrorMessage(response, parsed, {
          generic: t('files.actions.upload') || 'Upload failed',
          ...UPLOAD_ERROR_MESSAGES,
        }))
      }
    } catch (error) {
      console.error('Upload failed:', error)
      toast.error(error instanceof Error ? error.message : (t('files.actions.upload') || 'Upload failed'))
    } finally {
      setUploading(false)
    }
  }

  const [confirmDialog, setConfirmDialog] = useState<{ isOpen: boolean, type: 'single' | 'multiple', file?: FileItem }>({ isOpen: false, type: 'single' })

  const [isDeletingFile, setIsDeletingFile] = useState(false)

  const deleteFile = async (file: FileItem) => {
    setConfirmDialog({ isOpen: true, type: 'single', file })
  }

  const downloadFile = async (file: FileItem) => {
    try {
      const response = await apiRequest(`${getApiUrl()}/api/files/download/${encodeURIComponent(file.file_id)}`)

      if (!response.ok) {
        throw new Error(`Download failed: ${response.statusText}`)
      }

      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url
      link.download = file.filename
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(url)
    } catch (error) {
      console.error('Failed to download file:', error)
    }
  }

  const handlePreviewFile = (file: FileItem) => {
    setPreviewFile({
      fileId: file.file_id,
      fileName: file.filename
    })
    setIsPreviewOpen(true)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getFileIcon = (filename: string) => {
    const ext = filename.split('.').pop()?.toLowerCase() || ''

    if (['py', 'js', 'ts', 'tsx', 'jsx', 'java', 'c', 'cpp', 'go', 'rs'].includes(ext)) {
      return <FileCode className="h-4 w-4 text-blue-500" />
    }
    if (['json', 'yaml', 'yml', 'xml'].includes(ext)) {
      return <FileJson className="h-4 w-4 text-orange-500" />
    }
    if (['csv', 'xls', 'xlsx'].includes(ext)) {
      return <FileSpreadsheet className="h-4 w-4 text-green-500" />
    }
    if (['jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'].includes(ext)) {
      return <ImageIcon className="h-4 w-4 text-purple-500" />
    }
    if (['mp4', 'avi', 'mov', 'mkv'].includes(ext)) {
      return <Video className="h-4 w-4 text-red-500" />
    }
    if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) {
      return <Archive className="h-4 w-4 text-yellow-500" />
    }
    return <FileText className="h-4 w-4 text-slate-500" />
  }

  const toggleFileSelection = (fileId: string) => {
    setSelectedFiles(prev =>
      prev.includes(fileId)
        ? prev.filter(f => f !== fileId)
        : [...prev, fileId]
    )
  }

  const deleteSelectedFiles = async () => {
    if (selectedFiles.length === 0) return
    setConfirmDialog({ isOpen: true, type: 'multiple' })
  }

  const handleConfirmDelete = async () => {
    setIsDeletingFile(true)
    try {
      if (confirmDialog.type === 'single' && confirmDialog.file) {
        const file = confirmDialog.file
        const response = await apiRequest(`${getApiUrl()}/api/files/${encodeURIComponent(file.file_id)}`, {
          method: 'DELETE'
        })

        if (response.ok) {
          setSelectedFiles(prev => prev.filter(f => f !== file.file_id))
          await loadFiles(page)
          await loadTasks()
        } else {
          const parsed = await parseApiResponse(response)
          toast.error(getApiErrorMessage(
            response,
            parsed,
            t('common.deleteFailed') || "Failed to delete file",
          ))
        }
      } else if (confirmDialog.type === 'multiple') {
        let errorMessage: string | null = null
        for (const fileId of selectedFiles) {
          const fileToDelete = files.find(f => f.file_id === fileId)
          if (fileToDelete) {
            const response = await apiRequest(`${getApiUrl()}/api/files/${encodeURIComponent(fileToDelete.file_id)}`, {
              method: 'DELETE'
            })
            if (response.ok) {
              setFiles(prev => prev.filter(f => f.file_id !== fileToDelete.file_id))
              setTotalFiles(prev => Math.max(prev - 1, 0))
            } else {
              const parsed = await parseApiResponse(response)
              errorMessage = errorMessage || getApiErrorMessage(
                response,
                parsed,
                t('common.deleteFailed'),
              )
            }
          }
        }
        setSelectedFiles([])
        await loadFiles(page)
        await loadTasks()
        if (errorMessage) {
          toast.error(errorMessage)
        }
      }
    } catch (error) {
      console.error('Failed to delete file(s):', error)
      toast.error(error instanceof Error ? error.message : t('common.deleteFailed'))
    } finally {
      setIsDeletingFile(false)
      setConfirmDialog({ isOpen: false, type: 'single' })
    }
  }

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Header (Title + Actions) */}
      <div className="border-b flex flex-col sm:flex-row justify-between items-start sm:items-center p-4 sm:p-8 gap-4">
        <div className="space-y-1 w-full sm:w-auto">
          <h1 className="text-2xl sm:text-3xl font-bold mb-1">{t('files.header.title')}</h1>
          <p className="text-sm sm:text-base text-muted-foreground">{t('files.header.description')}</p>
        </div>

        <div className="flex items-center gap-3 w-full sm:w-auto">
          <SearchInput
            placeholder={t('files.search.placeholder')}
            value={searchQuery}
            onChange={(value) => {
              setSearchQuery(value)
            }}
            containerClassName="flex-1 sm:w-64"
            className="h-9 w-full bg-background"
          />

          <input
            ref={fileInputRef}
            type="file"
            multiple
            onChange={handleFileUpload}
            className="hidden"
          />
          <Button
            size="sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="shrink-0"
          >
            <Upload className="h-4 w-4 sm:mr-2" />
            <span className="hidden sm:inline">{uploading ? t('files.actions.uploading') : t('files.actions.upload')}</span>
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <aside className="w-64 border-r bg-muted/10 flex-shrink-0 flex flex-col">
          <div className="p-6">
            <div className="space-y-6">
              {/* Folders Section */}
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-3 mb-3">
                  {t('files.sidebar.folders')}
                </h3>
                <Button
                  variant={selectedCategory === 'all' ? 'secondary' : 'ghost'}
                  className={cn("w-full justify-start", selectedCategory === 'all' && "bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-300")}
                  onClick={() => {
                    setSelectedCategory('all')
                    setPage(1)
                  }}
                >
                  <LayoutGrid className="h-4 w-4 mr-2" />
                  {t('files.sidebar.allFiles')}
                </Button>
              </div>

              {/* Tasks Section */}
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-3 mb-3">
                  {t('files.sidebar.tasks')}
                </h3>
                <div className="space-y-1 max-h-[300px] overflow-y-auto pr-1">
                  {tasks.length > 0 ? (
                    tasks.map((task) => (
                      <Button
                        key={task.task_id}
                        variant={selectedCategory === `task-${task.task_id}` ? 'secondary' : 'ghost'}
                        className={cn(
                          "w-full justify-start text-muted-foreground hover:text-foreground",
                          selectedCategory === `task-${task.task_id}` && "bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-300"
                        )}
                        onClick={() => {
                          setSelectedCategory(`task-${task.task_id}`)
                          setPage(1)
                        }}
                        title={task.title}
                      >
                        <MessageSquare className="h-4 w-4 mr-2" />
                        <span className="truncate">
                          {task.title}
                        </span>
                      </Button>
                    ))
                  ) : (
                    <div className="px-3 text-xs text-muted-foreground">
                      {t('files.sidebar.noTasks')}
                    </div>
                  )}
                </div>
              </div>

              {/* System Section */}
              <div className="space-y-2">
                <h3 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider px-3 mb-3">
                  {t('files.sidebar.system')}
                </h3>
                <Button
                  variant={selectedCategory === 'uploads' ? 'secondary' : 'ghost'}
                  className={cn(
                    "w-full justify-start text-muted-foreground hover:text-foreground",
                    selectedCategory === 'uploads' && "bg-blue-100 text-blue-700 hover:bg-blue-200 dark:bg-blue-900/30 dark:text-blue-300"
                  )}
                  onClick={() => {
                    setSelectedCategory('uploads')
                    setPage(1)
                  }}
                >
                  <Folder className="h-4 w-4 mr-2" />
                  {t('files.sidebar.userUploads')}
                </Button>
              </div>
            </div>
          </div>
        </aside>

        {/* Main Content */}
        <main className="flex-1 flex flex-col overflow-hidden bg-background">

          {/* Breadcrumb / Title Bar */}
          <div className="px-8 py-4 flex items-center justify-between text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <span className="font-medium text-foreground">{t('files.breadcrumb.files')}</span>
              <span>&gt;</span>
              <span className="font-medium text-foreground">
                {selectedCategory === 'all'
                  ? t('files.sidebar.allFiles')
                  : selectedCategory === 'uploads'
                    ? t('files.sidebar.userUploads')
                    : selectedCategory.startsWith('task-')
                      ? tasks.find(task => `task-${task.task_id}` === selectedCategory)?.title || t('files.breadcrumb.unknownTask')
                      : t('files.breadcrumb.unknownCategory')}
              </span>
            </div>

            {selectedFiles.length > 0 && (
              <div className="flex items-center gap-2">
                <Badge variant="secondary">
                  {t('files.selection.selected', { count: selectedFiles.length })}
                </Badge>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-8 text-destructive hover:text-destructive"
                  onClick={deleteSelectedFiles}
                >
                  <Trash2 className="h-4 w-4 mr-2" />
                  {t('files.actions.delete')}
                </Button>
              </div>
            )}
          </div>

          {/* File List Table */}
          <div className="flex-1 overflow-auto px-8 pb-8">
            <div className="border rounded-lg bg-card shadow-sm">
              {/* Table Header */}
              <div className="grid grid-cols-12 gap-4 p-4 border-b text-xs font-medium text-muted-foreground uppercase tracking-wider bg-muted/30">
                <div className="col-span-5 pl-2">{t('files.table.name')}</div>
                <div className="col-span-2">{t('files.table.type')}</div>
                <div className="col-span-2">{t('files.table.size')}</div>
                <div className="col-span-3">{t('files.table.dateModified')}</div>
              </div>

              {/* Table Body */}
              {isLoading ? (
                <div className="p-12 text-center text-muted-foreground">
                  {t('files.table.empty.loading')}
                </div>
              ) : files.length === 0 ? (
                <div className="p-12 text-center text-muted-foreground">
                  {searchQuery ? t('files.table.empty.noMatch') : t('files.table.empty.noFiles')}
                </div>
              ) : (
                <div className="divide-y">
                  {files.map((file) => (
                    <div
                      key={file.file_id}
                      className="grid grid-cols-12 gap-4 p-4 hover:bg-muted/50 transition-colors items-center group text-sm"
                    >
                      <div className="col-span-5 flex items-center gap-3 min-w-0">
                        {/* Checkbox only visible on hover or selected */}
                        <div className="w-5 flex justify-center">
                          <input
                            type="checkbox"
                            checked={selectedFiles.includes(file.file_id)}
                            onChange={() => toggleFileSelection(file.file_id)}
                            className={cn(
                              "rounded border-gray-300 text-primary focus:ring-primary accent-primary h-4 w-4 transition-opacity",
                              selectedFiles.includes(file.file_id) ? "opacity-100" : "opacity-0 group-hover:opacity-100"
                            )}
                          />
                        </div>

                        <div className="flex-shrink-0">
                          {getFileIcon(file.filename)}
                        </div>
                        <span className="font-medium truncate text-foreground select-text" title={file.filename}>
                          {file.filename}
                        </span>
                      </div>

                      <div className="col-span-2 text-muted-foreground uppercase text-xs">
                        {file.filename.split('.').pop() || '-'}
                      </div>

                      <div className="col-span-2 text-muted-foreground text-xs">
                        {formatFileSize(file.file_size)}
                      </div>

                      <div className="col-span-3 text-muted-foreground text-xs flex items-center justify-between">
                        <span>{formatRelativeTime(file.modified_time)}</span>

                        {/* Actions */}
                        <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity justify-end">
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                            onClick={() => handlePreviewFile(file)}
                            title={t('files.actions.preview')}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                            onClick={() => downloadFile(file)}
                            title={t('files.actions.download')}
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                            onClick={() => deleteFile(file)}
                            title={t('files.actions.delete')}
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
            {pages > 1 && !isLoading && (
              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-muted-foreground">
                  {t('files.pagination.showing', {
                    start: totalFiles === 0 ? 0 : (page - 1) * pageSize + 1,
                    end: Math.min(page * pageSize, totalFiles),
                    total: totalFiles,
                  })}
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(current => current - 1)}
                    disabled={page <= 1}
                  >
                    <ChevronLeft className="mr-1 h-4 w-4" />
                    {t('files.pagination.prev')}
                  </Button>
                  <span className="text-sm text-muted-foreground">
                    {t('files.pagination.page', { page, pages })}
                  </span>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPage(current => current + 1)}
                    disabled={page >= pages}
                  >
                    {t('files.pagination.next')}
                    <ChevronRight className="ml-1 h-4 w-4" />
                  </Button>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* File Preview Dialog */}
      {previewFile && (
        <StandaloneFilePreviewDialog
          open={isPreviewOpen}
          onOpenChange={setIsPreviewOpen}
          fileId={previewFile.fileId}
          fileName={previewFile.fileName}
        />
      )}

      <ConfirmDialog
        isOpen={confirmDialog.isOpen}
        onOpenChange={(open) => setConfirmDialog(prev => ({ ...prev, isOpen: open }))}
        onConfirm={handleConfirmDelete}
        isLoading={isDeletingFile}
        description={confirmDialog.type === 'single'
          ? t('files.delete.confirmSingle', { name: confirmDialog.file?.filename || '' })
          : t('files.delete.confirmMultiple', { count: selectedFiles.length })}
      />
    </div>
  )
}
