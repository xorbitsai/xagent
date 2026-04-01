"use client"

import { useEffect, useState } from "react"
import { useApp } from "@/contexts/app-context-chat"
import { getApiUrl } from "@/lib/utils"
import { apiRequest } from "@/lib/api-wrapper"
import { useI18n } from "@/contexts/i18n-context"
import { FileViewer } from "@/components/file/file-viewer"

interface FilePreviewContentProps {
  open: boolean
}

export function FilePreviewContent({ open }: FilePreviewContentProps) {
  const { state, dispatch } = useApp()
  const { filePreview } = state
  const { t } = useI18n()

  // Load file content when the preview is open within container
  useEffect(() => {
    if (open && filePreview.fileId && !filePreview.content && !filePreview.error) {
      const loadFileContent = async () => {
        try {
          const apiUrl = getApiUrl()

          // PPTX files are converted to PDF by backend, treat as PDF
          const isPptx = filePreview.fileName.match(/\.pptx$/i)
          const isPdf = isPptx || filePreview.fileName.match(/\.pdf$/i)
          const isDocx = filePreview.fileName.match(/\.docx$/i)

          const url = `${apiUrl}/api/files/preview/${filePreview.fileId}`

          const response = await apiRequest(url, {
            cache: 'no-cache',
            headers: {
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache'
            }
          })

          if (response.ok) {
            let fileContent

            // Get MIME type from response headers (more reliable than file extension)
            const contentType = response.headers.get('content-type') || ''
            const mimeType = contentType.split(';')[0].trim()

            // Determine file type based on MIME type instead of file extension
            const isImage = mimeType.startsWith('image/')
            const isPdf = mimeType.startsWith('application/pdf') || mimeType === 'application/pdf'
            const isDocx = mimeType === 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
            const isTextType = mimeType.startsWith('text/') || mimeType === 'application/json' || mimeType === 'application/xml' || mimeType === 'application/javascript'

            console.log('File preview debug:', {
              fileName: filePreview.fileName,
              mimeType,
              isImage,
              isDocx,
              isPdf,
              isTextType,
              contentType: response.headers.get('content-type')
            })

            if (!isTextType) {
              const arrayBuffer = await response.arrayBuffer()

              // Use modern, efficient base64 conversion
              const bytes = new Uint8Array(arrayBuffer)
              const binaryString = Array.from(bytes, (byte) => String.fromCharCode(byte)).join('')
              fileContent = btoa(binaryString)

              console.log('Base64 conversion completed:', {
                mimeType,
                originalSize: arrayBuffer.byteLength,
                base64Size: fileContent.length
              })
            } else {
              fileContent = await response.text()
            }

            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: fileContent, mimeType, error: null }
            })
          } else {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: t('files.previewDialog.errors.loadFailed') }
            })
          }
        } catch (error) {
          if ((error as any)?.name === 'TypeError' && (error as any)?.message?.includes('Failed to fetch')) {
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: t('files.previewDialog.errors.cors') }
            })
          } else {
            const msg = (error as any)?.message || t('common.errors.unknown')
            dispatch({
              type: "SET_FILE_PREVIEW_CONTENT",
              payload: { content: "", mimeType: undefined, error: t('files.previewDialog.errors.networkErrorWithMsg', { msg }) }
            })
          }
        }
      }

      loadFileContent()
    }
  }, [open, filePreview.fileId, filePreview.content, filePreview.error, dispatch, t, filePreview.fileName])

  return (
    <div className="w-full h-full flex flex-col">
      <div className="flex-1 overflow-hidden flex flex-col min-h-0 h-full">
        <FileViewer
          fileName={filePreview.fileName}
          fileId={filePreview.fileId}
          content={filePreview.content}
          mimeType={filePreview.mimeType}
          isLoading={filePreview.isLoading}
          error={filePreview.error}
          viewMode={filePreview.viewMode}
        />
      </div>
    </div>
  )
}
