const TEXT_PREVIEW_EXTENSIONS = new Set([
  "c",
  "cfg",
  "conf",
  "cpp",
  "css",
  "csv",
  "go",
  "h",
  "hpp",
  "htm",
  "html",
  "ini",
  "java",
  "js",
  "json",
  "jsx",
  "log",
  "md",
  "markdown",
  "mjs",
  "py",
  "r",
  "rs",
  "sh",
  "sql",
  "srt",
  "toml",
  "ts",
  "tsv",
  "tsx",
  "txt",
  "vtt",
  "xml",
  "yaml",
  "yml",
])

export function isTextPreviewFile(fileName?: string, mimeType = ""): boolean {
  const normalizedMimeType = mimeType.split(";")[0].trim().toLowerCase()
  if (
    normalizedMimeType.startsWith("text/") ||
    normalizedMimeType === "application/json" ||
    normalizedMimeType === "application/javascript" ||
    normalizedMimeType === "application/xml"
  ) {
    return true
  }

  if (!fileName) return false
  const baseName = fileName.split(/[\\/]/).pop() || fileName
  const extension = baseName.split(".").pop()?.toLowerCase() || ""
  return TEXT_PREVIEW_EXTENSIONS.has(extension)
}
