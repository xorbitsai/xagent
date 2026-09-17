import { unified } from "unified"
import { parseEntities } from "parse-entities"
import remarkGfm from "remark-gfm"
import remarkParse from "remark-parse"

const FILE_NAME_KEYS = new Set(["filename", "file_name", "name"])
const FILE_DESCRIPTOR_KEYS = new Set(["mime_type", "type"])
const GENERIC_IDENTITY_AND_LOCATION_KEYS = new Set(["id", "url", "href", "uri"])
const FILE_RECORD_COLLECTION_KEYS = new Set(["artifacts", "documents", "files"])
const KNOWN_FILE_DESCRIPTOR_VALUES = new Set([
  "audio", "document", "file", "image", "presentation", "spreadsheet", "video",
])
const LOCAL_FILE_LOCATION_KEYS = new Set([
  "absolute_path", "file_path", "image_path", "local_path", "output_dir", "output_path",
  "audio_path", "backup_path", "base_dir", "current_path", "full_path", "json_path",
  "marked_image_path", "relative_path", "source_path", "storage_path", "transcription_path",
  "translation_path", "uploads_directory", "video_path", "workspace_dir",
])
const AMBIGUOUS_FILE_LOCATION_KEYS = new Set(["path", "directory"])
const LOCAL_PATH_CANDIDATE_START_PATTERN = String.raw`(?:/|~[\\/]|\.{1,2}[\\/]|[a-z]:[\\/]|\\{1,2}[^\\/]+[\\/]|(?:artifacts?|output|uploads?|workspace)(?:[\\/]|$))`
const EXPLICIT_LOCAL_PATH_RE = /^(?:~[\\/]|\.{1,2}[\\/]|[a-z]:[\\/]|\\{1,2}[^\\/]+[\\/]|(?:artifacts?|output|uploads?|workspace)(?:[\\/]|$))/i
const LOCAL_POSIX_ROOT_RE = /^\/(?:app|data|etc|home|mnt|opt|private|raw|root|sandbox|srv|tmp|users|var|workspace)(?:[\\/]|$)/i
// Shell redirects and prose separators are token boundaries. Relative paths
// are recognized only under the explicit artifact/output/upload/workspace
// roots above; arbitrary slash-delimited prose deliberately remains outside
// this narrow grammar.
const UNQUOTED_PATH_PREFIX_PATTERN = "(^|[\\s\"'([{=,;<>|:&!])"
const UNQUOTED_PATH_SUFFIX_PATTERN = "[^\\s\"'`<>),;]*"
const BARE_FILE_URI_RE = /\bfile:[^\s<>"'`]+/gi
const MANAGED_FILE_PATH_PREFIX_RE = /^\/api\/files(?:\/|$)/
const MANAGED_FILE_ROUTE_RE = /^\/api\/files(?:\/public)?\/(?:preview-pdf|preview|download)(?:\/|$)/
const MAX_MANAGED_PATH_DECODES = 2
const CIRCULAR_PRESENTATION_SENTINEL = "[Circular]"

const isRecord = (value: unknown): value is Record<string, unknown> => (
  typeof value === "object" && value !== null && !Array.isArray(value)
)

const normalizeKey = (key: string): string => (
  key.replace(/([a-z0-9])([A-Z])/g, "$1_$2").toLowerCase()
)

const isFileIdKey = (key: string): boolean => {
  const normalized = normalizeKey(key)
  return normalized === "file_id" || normalized.endsWith("_file_id")
}

const isStrongFileNameKey = (key: string): boolean => {
  const normalized = normalizeKey(key)
  return normalized === "filename" || normalized === "file_name"
}

const isFileAccessKey = (key: string): boolean => (
  /(^|_)(preview|download|signed|file)_url$/.test(normalizeKey(key))
)

const hasStrongFileIdentity = (value: Record<string, unknown>): boolean => {
  const entries = Object.entries(value)
  const keys = entries.map(([key]) => normalizeKey(key))
  if (keys.some(isFileIdKey) || keys.some(isStrongFileNameKey)) return true
  const hasName = keys.some((key) => FILE_NAME_KEYS.has(key))
  const descriptor = entries.find(([key]) => FILE_DESCRIPTOR_KEYS.has(normalizeKey(key)))?.[1]
  return hasName && typeof descriptor === "string" && (
    descriptor.includes("/") || KNOWN_FILE_DESCRIPTOR_VALUES.has(descriptor.toLowerCase())
  )
}

const hasFileCopyEvidence = (value: Record<string, unknown>): boolean => {
  const keys = new Set(Object.keys(value).map(normalizeKey))
  return keys.has("source") && keys.has("destination") && typeof value.extracted === "boolean"
}

const isLocalFileLocationKey = (
  key: string,
  owner: Record<string, unknown>,
  fileRecordContext = false,
): boolean => {
  const normalized = normalizeKey(key)
  if (LOCAL_FILE_LOCATION_KEYS.has(normalized)) return true
  if (AMBIGUOUS_FILE_LOCATION_KEYS.has(normalized)) return fileRecordContext || hasStrongFileIdentity(owner)
  if (normalized === "html_src") return hasStrongFileIdentity(owner)
  return (normalized === "source" || normalized === "destination") && hasFileCopyEvidence(owner)
}

const basename = (value: string): string | null => value.split(/[\\/]/).filter(Boolean).at(-1) || null
const normalizeKnownLocalRoot = (value: string): string => {
  const withoutTrailingSeparators = value.replace(/[\\/]+$/, "")
  return withoutTrailingSeparators || value.slice(0, 1)
}

const rawFileLabel = (value: Record<string, unknown>): string | null => {
  for (const key of ["filename", "file_name", "fileName", "name"]) {
    const candidate = value[key]
    if (typeof candidate === "string" && candidate.trim()) return basename(candidate) || candidate
  }
  return null
}

const collectKnownLocalPaths = (value: unknown): Map<string, string> => {
  const paths = new Map<string, string>()
  const visiting = new WeakSet<object>()
  const visit = (item: unknown, fileRecordContext = false): void => {
    if (Array.isArray(item)) {
      if (visiting.has(item)) return
      visiting.add(item)
      try {
        item.forEach((child) => visit(child, fileRecordContext))
      } finally {
        visiting.delete(item)
      }
      return
    }
    if (!isRecord(item)) return
    if (visiting.has(item)) return
    visiting.add(item)
    try {
      const label = rawFileLabel(item)
      for (const [key, child] of Object.entries(item)) {
        if (isLocalFileLocationKey(key, item, fileRecordContext) && typeof child === "string" && child.trim()) {
          const fallback = basename(child)
          if (label || fallback) {
            const root = normalizeKnownLocalRoot(child.trim())
            paths.set(root, label ?? fallback ?? root)
          }
        }
        visit(child, fileRecordContext || FILE_RECORD_COLLECTION_KEYS.has(normalizeKey(key)))
      }
    } finally {
      visiting.delete(item)
    }
  }
  visit(value)
  return paths
}

const escapeRegex = (value: string): string => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")

const replaceKnownLocalPaths = (value: string, knownPaths: Map<string, string>): string => {
  const entries = [...knownPaths.entries()].sort(([left], [right]) => right.length - left.length)
  if (entries.length === 0) return value
  const roots = entries.map(([path]) => escapeRegex(path)).join("|")
  // Consume an entire descendant token so a known root cannot leave its private
  // suffix visible. The longest root wins because alternatives are sorted first.
  const knownRootToken = new RegExp(
    UNQUOTED_PATH_PREFIX_PATTERN + "(" + roots + ")(?:[\\\\\\/][^\\s\\\"'`<>(),;]*)*(?=$|[\\s\\\"'`,.;:!?<>()])",
    "g",
  )
  return value.replace(knownRootToken, (_match, prefix: string, path: string) => {
    const matched = entries.find(([root]) => path === root)
    return `${prefix}${matched?.[1] ?? basename(path) ?? path}`
  })
}

const isRecognizedLocalPath = (value: string): boolean => {
  const path = value.trim().replace(/[.,;:!?]+$/, "")
  return (
    EXPLICIT_LOCAL_PATH_RE.test(path)
    || LOCAL_POSIX_ROOT_RE.test(path)
  )
}

const LOCAL_PATH_TOKEN_RE = new RegExp(
  UNQUOTED_PATH_PREFIX_PATTERN + "(" + LOCAL_PATH_CANDIDATE_START_PATTERN + UNQUOTED_PATH_SUFFIX_PATTERN + ")",
  "gi",
)

type MarkdownPosition = {
  start: { offset?: number }
  end: { offset?: number }
}

type MarkdownNode = {
  type: string
  value?: string
  url?: string
  title?: string | null
  alt?: string | null
  identifier?: string
  children?: MarkdownNode[]
  position?: MarkdownPosition
}

type TextReplacement = {
  start: number
  end: number
  value: string
}

const markdownParser = unified().use(remarkParse).use(remarkGfm)

const sanitizePlainPresentationText = (
  value: string,
  redactUnquotedPaths: boolean,
): string => {
  const withoutFileUris = value.replace(BARE_FILE_URI_RE, (candidate) => {
    let trailing = ""
    let target = candidate
    while (/[.,;:!?]$/.test(target)) {
      trailing = target.slice(-1) + trailing
      target = target.slice(0, -1)
    }
    for (const [opening, closing] of [["(", ")"], ["[", "]"], ["{", "}"]] as const) {
      const openingCount = [...target].filter((character) => character === opening).length
      let closingCount = [...target].filter((character) => character === closing).length
      while (target.endsWith(closing) && closingCount > openingCount) {
        trailing = closing + trailing
        target = target.slice(0, -1)
        closingCount -= 1
      }
    }
    return `file${trailing}`
  })
  const sanitized = withoutFileUris.replace(/`([^`\n]+)`/g, (match, candidate: string) => {
    const path = candidate.trim()
    if (!isRecognizedLocalPath(path)) return match
    return basename(path) || path
  })
  return redactUnquotedPaths
    ? sanitized.replace(LOCAL_PATH_TOKEN_RE, (
      _match,
      prefix: string,
      path: string,
    ) => (
      isRecognizedLocalPath(path)
        ? `${prefix}${basename(path) || path}`
        : _match
    ))
    : sanitized
}

const markdownNodeText = (node: MarkdownNode): string => {
  if (typeof node.value === "string") return node.value
  if (typeof node.alt === "string") return node.alt
  return node.children?.map(markdownNodeText).join("") ?? ""
}

const markdownOffsets = (node: MarkdownNode): { start: number; end: number } | null => {
  const start = node.position?.start.offset
  const end = node.position?.end.offset
  return typeof start === "number" && typeof end === "number" ? { start, end } : null
}

const decodedTargetCandidates = (target: string): string[] => {
  const candidates: string[] = []
  let candidate = target
  for (let decodeCount = 0; decodeCount <= MAX_MANAGED_PATH_DECODES; decodeCount += 1) {
    candidates.push(candidate)
    if (!candidate.includes("%") || decodeCount === MAX_MANAGED_PATH_DECODES) break
    // Decode each valid triplet independently. A malformed escape later in
    // the target must not hide an encoded local or managed-route prefix.
    const decoded = candidate.replace(/%([0-9a-f]{2})/gi, (_match, hex: string) => (
      String.fromCharCode(Number.parseInt(hex, 16))
    ))
    if (decoded === candidate) break
    candidate = decoded
  }
  return candidates
}

const isKnownLocalTarget = (
  target: string,
  knownPaths: Map<string, string>,
): boolean => {
  const normalizedTarget = target.replace(/^\/+/, "/")
  return [...knownPaths.keys()].some((rawRoot) => {
    const root = normalizeKnownLocalRoot(rawRoot)
    if (root === "/") return normalizedTarget.startsWith("/")
    return (
      normalizedTarget === root
      || normalizedTarget.startsWith(`${root}/`)
      || normalizedTarget.startsWith(`${root}\\`)
    )
  })
}

const isInertFileTarget = (
  target: string,
  knownPaths: Map<string, string>,
): boolean => decodedTargetCandidates(target).some((candidate) => {
  const normalizedCandidate = candidate.replace(/^\/+/, "/")
  return (
    /^file:/i.test(candidate)
    || isManagedFileUrl(candidate)
    || isRecognizedLocalPath(normalizedCandidate)
    || isKnownLocalTarget(normalizedCandidate, knownPaths)
  )
})

const escapeMarkdownLabel = (value: string): string => value.replace(/([\\[\]])/g, "\\$1")
const escapeMarkdownTitle = (value: string): string => value.replace(/([\\"])/g, "\\$1")
const escapeMarkdownPlainText = (value: string): string => (
  value.replace(/([\\`*{}\[\]#+!_|<>])/g, "\\$1")
)
const formatSafeInlineCode = (value: string): string => {
  const longestBacktickRun = Math.max(
    0,
    ...([...value.matchAll(/`+/g)].map((match) => match[0].length)),
  )
  const delimiter = "`".repeat(Math.max(1, longestBacktickRun + 1))
  return `${delimiter} ${value} ${delimiter}`
}
const formatSafeCodeBlock = (value: string): string => {
  const longestTildeRun = Math.max(
    0,
    ...([...value.matchAll(/~+/g)].map((match) => match[0].length)),
  )
  const fence = "~".repeat(Math.max(3, longestTildeRun + 1))
  return `${fence}\n${value}\n${fence}`
}
const formatMarkdownTarget = (value: string): string => (
  /[\s()]/.test(value) ? `<${value.replace(/>/g, "%3E")}>` : value
)

const applyTextReplacements = (source: string, replacements: TextReplacement[]): string => {
  const sorted = replacements.sort((left, right) => right.start - left.start || right.end - left.end)
  let result = source
  let lastStart = source.length + 1
  for (const replacement of sorted) {
    if (replacement.end > lastStart) continue
    result = result.slice(0, replacement.start) + replacement.value + result.slice(replacement.end)
    lastStart = replacement.start
  }
  return result
}

const sanitizeFilesDisabledPresentationTextWithPaths = (
  value: string,
  redactUnquotedPaths: boolean,
  knownPaths: Map<string, string>,
): string => {
  try {
    const tree = markdownParser.runSync(markdownParser.parse(value)) as MarkdownNode
    const definitions = new Map<string, MarkdownNode>()
    const collectDefinitions = (node: MarkdownNode): void => {
      if (node.type === "definition" && node.identifier) {
        definitions.set(node.identifier.toLowerCase(), node)
      }
      node.children?.forEach(collectDefinitions)
    }
    collectDefinitions(tree)

    const replacements: TextReplacement[] = []
    const replaceNode = (node: MarkdownNode, replacement: string): boolean => {
      const offsets = markdownOffsets(node)
      if (!offsets) return false
      replacements.push({ ...offsets, value: replacement })
      return true
    }

    const visit = (node: MarkdownNode): void => {
      if (node.type === "definition") {
        if (node.url && isInertFileTarget(node.url, knownPaths)) {
          replaceNode(node, "")
          return
        }
        if (node.title) {
          const safeTitle = sanitizePlainPresentationText(node.title, redactUnquotedPaths)
          if (safeTitle !== node.title && node.identifier && node.url) {
            replaceNode(
              node,
              `[${escapeMarkdownLabel(node.identifier)}]: <${node.url}> "${escapeMarkdownTitle(safeTitle)}"`,
            )
            return
          }
        }
      }

      if (node.type === "link" || node.type === "image") {
        const target = node.url ?? ""
        if (isInertFileTarget(target, knownPaths)) {
          // markdownNodeText is alt-first for images, then link/image
          // children text -- but an empty label/alt (e.g. a media
          // reference emitted as ``[](file:id "generated_video.mp4")``,
          // see file_reference_output_service.py) leaves nothing there.
          // Falling back to the title before "file" keeps that filename
          // visible instead of degrading to the literal word "file".
          const rawLabel = markdownNodeText(node) || node.title || ""
          const label = sanitizePlainPresentationText(rawLabel, redactUnquotedPaths) || "file"
          replaceNode(node, label)
          return
        }
        if (node.type === "link" && node.title) {
          const safeTitle = sanitizePlainPresentationText(node.title, redactUnquotedPaths)
          if (safeTitle !== node.title) {
            const safeLabel = sanitizePlainPresentationText(markdownNodeText(node), redactUnquotedPaths)
            replaceNode(
              node,
              `[${escapeMarkdownLabel(safeLabel)}](${formatMarkdownTarget(target)} "${escapeMarkdownTitle(safeTitle)}")`,
            )
            return
          }
        }
        if (node.type === "image") {
          const safeAlt = sanitizePlainPresentationText(node.alt ?? "", redactUnquotedPaths)
          const safeTitle = node.title
            ? sanitizePlainPresentationText(node.title, redactUnquotedPaths)
            : null
          if (safeAlt !== (node.alt ?? "") || safeTitle !== node.title) {
            replaceNode(
              node,
              `![${escapeMarkdownLabel(safeAlt)}](${formatMarkdownTarget(target)}${safeTitle === null ? "" : ` "${escapeMarkdownTitle(safeTitle)}"`})`,
            )
            return
          }
        }
      }

      if (node.type === "linkReference" || node.type === "imageReference") {
        const definition = node.identifier
          ? definitions.get(node.identifier.toLowerCase())
          : undefined
        if (definition?.url && isInertFileTarget(definition.url, knownPaths)) {
          const label = sanitizePlainPresentationText(markdownNodeText(node), redactUnquotedPaths) || "file"
          replaceNode(node, label)
          return
        }
      }

      if (typeof node.value === "string" && node.type === "text") {
        const offsets = markdownOffsets(node)
        if (offsets) {
          const raw = value.slice(offsets.start, offsets.end)
          const safeValue = sanitizePlainPresentationText(node.value, redactUnquotedPaths)
          if (safeValue !== node.value) {
            replacements.push({
              ...offsets,
              value: raw === node.value ? safeValue : escapeMarkdownPlainText(safeValue),
            })
          } else {
            const safeRaw = sanitizePlainPresentationText(raw, redactUnquotedPaths)
            if (safeRaw !== raw) replacements.push({ ...offsets, value: safeRaw })
          }
        }
      } else if (typeof node.value === "string" && node.type === "inlineCode") {
        const semantic = parseEntities(node.value)
        const safe = sanitizePlainPresentationText(semantic, redactUnquotedPaths)
        if (safe !== semantic) {
          const needsLiteralContainer = (
            /[\\`*{}\[\]#+!_|<>]/.test(safe)
            || /\bhttps?:\/\//i.test(safe)
          )
          replaceNode(node, needsLiteralContainer ? formatSafeInlineCode(safe) : safe)
        }
      } else if (typeof node.value === "string" && node.type === "code") {
        const semantic = parseEntities(node.value)
        const safe = sanitizePlainPresentationText(semantic, redactUnquotedPaths)
        if (safe !== semantic) replaceNode(node, formatSafeCodeBlock(safe))
      } else if (typeof node.value === "string" && node.type === "html") {
        const semantic = parseEntities(node.value)
        const safe = sanitizePlainPresentationText(semantic, redactUnquotedPaths)
        if (safe !== semantic) replaceNode(node, escapeMarkdownPlainText(safe))
      }

      node.children?.forEach(visit)
    }
    visit(tree)
    return applyTextReplacements(value, replacements)
  } catch {
    return sanitizePlainPresentationText(value, redactUnquotedPaths)
  }
}

export function sanitizeFilesDisabledPresentationText(value: string, redactUnquotedPaths = true): string {
  return sanitizeFilesDisabledPresentationTextWithPaths(
    value,
    redactUnquotedPaths,
    new Map(),
  )
}

const projectWithPaths = (
  value: unknown,
  knownPaths: Map<string, string>,
  fileRecordContext = false,
  memo = new WeakMap<object, Map<boolean, unknown>>(),
  visiting = new WeakSet<object>(),
): unknown => {
  if (typeof value === "string") {
    return replaceKnownLocalPaths(
      sanitizeFilesDisabledPresentationTextWithPaths(value, true, knownPaths),
      knownPaths,
    )
  }
  if (Array.isArray(value)) {
    const cached = memo.get(value)
    if (cached?.has(fileRecordContext)) return cached.get(fileRecordContext)
    if (visiting.has(value)) return CIRCULAR_PRESENTATION_SENTINEL
    visiting.add(value)
    try {
      const projected = value.map((item) => projectWithPaths(
        item,
        knownPaths,
        fileRecordContext,
        memo,
        visiting,
      ))
      const nextCached = cached ?? new Map<boolean, unknown>()
      nextCached.set(fileRecordContext, projected)
      memo.set(value, nextCached)
      return projected
    } finally {
      visiting.delete(value)
    }
  }
  if (!isRecord(value)) return value
  const cached = memo.get(value)
  if (cached?.has(fileRecordContext)) return cached.get(fileRecordContext)
  if (visiting.has(value)) return CIRCULAR_PRESENTATION_SENTINEL
  visiting.add(value)
  try {
    const strongFileIdentity = hasStrongFileIdentity(value)
    const hasLocalFileEvidence = Object.entries(value).some(([key]) => (
      isLocalFileLocationKey(key, value, fileRecordContext)
    ))
    const projected = Object.fromEntries(Object.entries(value).flatMap(([key, child]) => {
      const normalized = normalizeKey(key)
      if (isFileIdKey(normalized) || isFileAccessKey(normalized) || isLocalFileLocationKey(normalized, value, fileRecordContext)) return []
      if (strongFileIdentity && GENERIC_IDENTITY_AND_LOCATION_KEYS.has(normalized)) return []
      if ((isStrongFileNameKey(normalized) || (strongFileIdentity && normalized === "name")) && typeof child === "string") {
        return [[key, basename(child) ?? child]]
      }
      return [[key, projectWithPaths(
        child,
        knownPaths,
        fileRecordContext || strongFileIdentity || hasLocalFileEvidence || FILE_RECORD_COLLECTION_KEYS.has(normalized),
        memo,
        visiting,
      )]]
    }))
    const nextCached = cached ?? new Map<boolean, unknown>()
    nextCached.set(fileRecordContext, projected)
    memo.set(value, nextCached)
    return projected
  } finally {
    visiting.delete(value)
  }
}

export function projectFilesDisabledPresentation(value: unknown): unknown {
  return projectWithPaths(value, collectKnownLocalPaths(value))
}

export function projectFilesDisabledToolResultPresentation(value: unknown): unknown {
  const projected = projectFilesDisabledPresentation(value)
  if (isRecord(projected)) return projected.output ?? projected.message ?? projected
  return projected
}

export function getFilesDisabledPresentationFileLabel(value: unknown): string | null {
  if (isRecord(value)) {
    const label = rawFileLabel(value)
    if (label) return label
    for (const [key, child] of Object.entries(value)) {
      if (isLocalFileLocationKey(key, value, true) && typeof child === "string") return basename(child)
    }
  }
  const projected = projectFilesDisabledPresentation(value)
  if (!isRecord(projected)) return null
  return rawFileLabel(projected)
}

const hasPresentationEvidence = (value: unknown, visiting = new WeakSet<object>()): boolean => {
  if (Array.isArray(value)) {
    if (visiting.has(value)) return false
    visiting.add(value)
    try {
      return value.some((item) => hasPresentationEvidence(item, visiting))
    } finally {
      visiting.delete(value)
    }
  }
  if (!isRecord(value)) return false
  if (visiting.has(value)) return false
  visiting.add(value)
  try {
    const strongIdentity = hasStrongFileIdentity(value)
    return Object.entries(value).some(([key, child]) => (
      isFileIdKey(key)
      || isFileAccessKey(key)
      || isLocalFileLocationKey(key, value)
      || (strongIdentity && GENERIC_IDENTITY_AND_LOCATION_KEYS.has(normalizeKey(key)))
      || hasPresentationEvidence(child, visiting)
    ))
  } finally {
    visiting.delete(value)
  }
}

export function serializeFilesDisabledPresentation(value: unknown): string {
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value)
      if (hasPresentationEvidence(parsed)) return JSON.stringify(projectFilesDisabledPresentation(parsed), null, 2)
    } catch {
      // Plain presentation text deliberately remains text.
    }
    return sanitizeFilesDisabledPresentationText(value)
  }
  try {
    const serialized = JSON.stringify(projectFilesDisabledPresentation(value), null, 2)
    return serialized ?? String(value)
  } catch {
    return String(value)
  }
}

/** Identifies only app-managed file endpoints, including relative API paths. */
export function isManagedFileUrl(value: string): boolean {
  try {
    const url = new URL(value, "http://xagent.invalid")
    const candidates = decodedTargetCandidates(url.pathname)

    for (const pathname of candidates) {
      const normalizedPathname = pathname.replace(/^\/+/, "/")
      if (MANAGED_FILE_ROUTE_RE.test(normalizedPathname)) return true
    }
    const finalPathname = candidates.at(-1) ?? url.pathname
    const normalizedFinalPathname = finalPathname.replace(/^\/+/, "/")
    return (
      MANAGED_FILE_PATH_PREFIX_RE.test(normalizedFinalPathname)
      && finalPathname.includes("%")
    )
  } catch {
    return false
  }
}
