import React from "react"
import type { LucideIcon, LucideProps } from "lucide-react"
import {
  Archive,
  File,
  FileCode,
  FileSpreadsheet,
  FileText,
  Image,
  Music,
  Presentation,
  Video,
} from "lucide-react"

export type FileVisualKind =
  | "image"
  | "video"
  | "audio"
  | "spreadsheet"
  | "presentation"
  | "archive"
  | "code"
  | "document"
  | "file"

const EXTENSION_KIND: Record<string, FileVisualKind> = {
  jpg: "image",
  jpeg: "image",
  png: "image",
  gif: "image",
  webp: "image",
  svg: "image",
  heic: "image",
  heif: "image",
  mp4: "video",
  avi: "video",
  mov: "video",
  mkv: "video",
  webm: "video",
  mp3: "audio",
  wav: "audio",
  ogg: "audio",
  opus: "audio",
  flac: "audio",
  m4a: "audio",
  aac: "audio",
  csv: "spreadsheet",
  xls: "spreadsheet",
  xlsx: "spreadsheet",
  ods: "spreadsheet",
  ppt: "presentation",
  pptx: "presentation",
  odp: "presentation",
  zip: "archive",
  rar: "archive",
  "7z": "archive",
  tar: "archive",
  gz: "archive",
  bz2: "archive",
  py: "code",
  js: "code",
  jsx: "code",
  ts: "code",
  tsx: "code",
  java: "code",
  c: "code",
  cpp: "code",
  go: "code",
  rs: "code",
  json: "code",
  yaml: "code",
  yml: "code",
  xml: "code",
  html: "code",
  css: "code",
  sh: "code",
  pdf: "document",
  doc: "document",
  docx: "document",
  odt: "document",
  rtf: "document",
  txt: "document",
  md: "document",
}

const ICON_BY_KIND: Record<FileVisualKind, LucideIcon> = {
  image: Image,
  video: Video,
  audio: Music,
  spreadsheet: FileSpreadsheet,
  presentation: Presentation,
  archive: Archive,
  code: FileCode,
  document: FileText,
  file: File,
}

export const getFileVisualKind = (
  filename?: string,
  mimeType?: string
): FileVisualKind => {
  const mime = mimeType?.toLowerCase() || ""

  if (mime.startsWith("image/")) return "image"
  if (mime.startsWith("video/")) return "video"
  if (mime.startsWith("audio/")) return "audio"
  // Keep specific OpenXML MIME checks above the generic "document" branch.
  if (mime.includes("spreadsheet") || mime.includes("excel") || mime === "text/csv") {
    return "spreadsheet"
  }
  if (mime.includes("presentation") || mime.includes("powerpoint")) {
    return "presentation"
  }
  if (/\b(zip|rar|7z|tar|gzip|bzip2)\b/.test(mime)) return "archive"
  if (
    mime.includes("json") ||
    mime.includes("xml") ||
    mime.includes("javascript")
  ) {
    return "code"
  }
  if (
    mime.startsWith("text/") ||
    mime.includes("pdf") ||
    mime.includes("word") ||
    mime.includes("document")
  ) {
    return "document"
  }

  const extension = filename?.split(".").pop()?.toLowerCase()
  return (extension && EXTENSION_KIND[extension]) || "file"
}

type FileTypeIconProps = LucideProps & {
  filename?: string
  mimeType?: string
}

export function FileTypeIcon({
  filename,
  mimeType,
  ...iconProps
}: FileTypeIconProps) {
  const kind = getFileVisualKind(filename, mimeType)
  const Icon = ICON_BY_KIND[kind]
  return <Icon {...iconProps} data-file-kind={kind} />
}
