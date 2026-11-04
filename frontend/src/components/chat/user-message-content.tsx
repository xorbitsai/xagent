import React from "react"

import { FileChip } from "./FileChip"

export type UserMessageFile = {
  name: string
  type?: string
  size?: number
  file_id?: string
  path?: string
}

type UserMessageContentProps = {
  message: string
  files: UserMessageFile[]
  onPreview?: (
    file: UserMessageFile,
    files: UserMessageFile[],
    index: number
  ) => void
}

const MARKDOWN_FILE_REF_RE = /!?\[[^\]\n]*\]\(file:(?:\/\/)?([^)\n]+)\)/g

export const sanitizeUserMessageFiles = (files: unknown): UserMessageFile[] => {
  if (!Array.isArray(files)) {
    return []
  }

  return files.filter((file): file is UserMessageFile => {
    if (file === null || typeof file !== "object") {
      return false
    }
    const candidate = file as Record<string, unknown>
    return (
      typeof candidate.name === "string" &&
      (candidate.type === undefined || typeof candidate.type === "string") &&
      (candidate.size === undefined || typeof candidate.size === "number") &&
      (candidate.file_id === undefined ||
        typeof candidate.file_id === "string") &&
      (candidate.path === undefined || typeof candidate.path === "string")
    )
  })
}

const decodeFileId = (value: string): string => {
  try {
    return decodeURIComponent(value)
  } catch {
    return value
  }
}

export const stripAttachedFileRefs = (
  message: unknown,
  files: unknown
): string => {
  if (typeof message !== "string") {
    return ""
  }

  const validFiles = sanitizeUserMessageFiles(files)
  const attachedIds = new Set(
    validFiles
      .map((file) => file.file_id?.trim())
      .filter((fileId): fileId is string => Boolean(fileId))
  )

  if (attachedIds.size === 0) {
    return message
  }

  return message
    .replace(MARKDOWN_FILE_REF_RE, (match, encodedFileId: string) => {
      const fileId = decodeFileId(encodedFileId).trim()
      return attachedIds.has(fileId) ? "" : match
    })
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim()
}

export function UserMessageContent({
  message,
  files,
  onPreview,
}: UserMessageContentProps) {
  const validFiles = sanitizeUserMessageFiles(files)
  const displayMessage = stripAttachedFileRefs(message, validFiles)

  return (
    <div className="whitespace-pre-wrap max-h-60 overflow-y-auto">
      {displayMessage}
      {validFiles.map((file, index) => {
        const path = file.file_id || file.path || file.name
        const canPreview = Boolean(file.file_id && onPreview)
        return (
          <FileChip
            key={`${path}-${index}`}
            path={path}
            filename={file.name}
            mimeType={file.type}
            onClick={
              canPreview
                ? () => onPreview?.(file, validFiles, index)
                : undefined
            }
          />
        )
      })}
    </div>
  )
}
