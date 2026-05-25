import { getFilePublicPreviewUrl } from '@/lib/utils'

export type InlineFilePreviewKind =
  | 'image'
  | 'presentation'
  | 'document'
  | 'spreadsheet'
  | 'file'

export type PreviewableInlineFileKind = Exclude<InlineFilePreviewKind, 'file'>

export type InlineFilePreviewSource = {
  fileId?: string
  previewUrl?: string
  filename?: string
  mimeType?: string
  type?: string
}

export type PreviewUrlTrust = {
  domain?: string
  isExternal: boolean
  isTrusted: boolean
}

const PREVIEWABLE_KINDS = new Set<PreviewableInlineFileKind>([
  'image',
  'presentation',
  'document',
  'spreadsheet',
])

// Only OOXML presentations are browser-previewable: PptxPreviewRenderer
// uses pptxviewjs, which only supports .pptx. Legacy binary .ppt (mime
// ``application/vnd.ms-powerpoint``) is intentionally NOT in this set —
// it falls through to ``'file'`` so callers render a download link
// instead of mounting an unsupported renderer.
const PRESENTATION_MIME_TYPES = new Set<string>([
  'application/vnd.openxmlformats-officedocument.presentationml.presentation',
])

const SPREADSHEET_MIME_TYPES = new Set<string>([
  'application/vnd.ms-excel',
  'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
  'text/csv',
])

export const isPreviewableInlineFileKind = (
  kind: InlineFilePreviewKind
): kind is PreviewableInlineFileKind =>
  PREVIEWABLE_KINDS.has(kind as PreviewableInlineFileKind)

export const getInlineFilePreviewKind = (
  source: InlineFilePreviewSource
): InlineFilePreviewKind => {
  const type = source.type?.toLowerCase() || ''
  const filename = source.filename?.toLowerCase() || ''
  const mimeType = source.mimeType?.toLowerCase() || ''

  if (type === 'image') return 'image'
  if (type === 'presentation') {
    // An explicit ``type: 'presentation'`` artifact must still be
    // cross-checked: pptxviewjs only supports OOXML .pptx, so a
    // producer that emits ``type: 'presentation'`` for a legacy .ppt
    // (or any non-OOXML payload) must NOT reach PptxPreviewRenderer.
    // If the filename/mime contradicts the type — or simply isn't
    // identifiably .pptx — fall through to the generic 'file' kind so
    // the UI renders a download link instead. The .pptx-only
    // contract is now enforced here, regardless of caller.
    const looksLikePptx =
      mimeType ===
        'application/vnd.openxmlformats-officedocument.presentationml.presentation' ||
      mimeType.includes('presentationml') ||
      filename.endsWith('.pptx')
    const looksLikeLegacyPpt =
      mimeType === 'application/vnd.ms-powerpoint' ||
      (filename.endsWith('.ppt') && !filename.endsWith('.pptx'))
    if (looksLikeLegacyPpt) return 'file'
    if (looksLikePptx) return 'presentation'
    // No filename / mime info to verify either way: assume .pptx (this
    // was the historical behavior for ``type: 'presentation'`` and
    // matches what producers emit for previewable artifacts).
    if (!filename && !mimeType) return 'presentation'
    return 'file'
  }
  if (type === 'document') return 'document'
  if (type === 'spreadsheet') return 'spreadsheet'

  if (mimeType.startsWith('image/')) return 'image'
  if (PRESENTATION_MIME_TYPES.has(mimeType) || mimeType.includes('presentationml')) {
    return 'presentation'
  }
  if (mimeType.includes('wordprocessingml')) return 'document'
  if (SPREADSHEET_MIME_TYPES.has(mimeType) || mimeType.includes('spreadsheetml')) {
    return 'spreadsheet'
  }

  if (/\.(jpg|jpeg|png|gif|webp|svg)$/.test(filename)) return 'image'
  // Only OOXML .pptx is previewable inline — see PRESENTATION_MIME_TYPES
  // comment. Legacy .ppt falls through to the generic 'file' kind.
  if (filename.endsWith('.pptx')) return 'presentation'
  if (filename.endsWith('.docx')) return 'document'
  if (/\.(csv|xls|xlsx)$/.test(filename)) return 'spreadsheet'

  return 'file'
}

export const getInlineFilePreviewUrl = (
  source: InlineFilePreviewSource,
  apiUrl: string
): string => {
  if (source.fileId) return getFilePublicPreviewUrl(source.fileId, apiUrl)
  if (source.previewUrl) {
    if (/^https?:\/\//.test(source.previewUrl)) return source.previewUrl
    return `${apiUrl}${source.previewUrl.startsWith('/') ? '' : '/'}${source.previewUrl}`
  }
  return ''
}

// The "Open" affordance must hand the user the *original* artifact —
// /api/files/public/preview is intended for inline rendering (e.g. the
// PptxPreviewRenderer feeds raw bytes into pptxviewjs) and on some
// deployments returns a derived preview payload instead of the source
// file. Routing the open link through /api/files/public/download
// guarantees the original bytes plus a ``Content-Disposition:
// attachment; filename=...`` header, so the browser saves the file
// under its real name instead of a bare file id.
//
// This MUST stay on the public/* route, not /api/files/download. The
// auth'd download route depends on ``get_current_user`` and the
// frontend only adds the bearer token through ``apiRequest``, never
// through plain anchor navigation. Routing ``<a href>`` clicks (and
// middle-click / right-click "open in new tab" / "copy link") through
// the auth'd route would 401 every navigation that doesn't pass
// through a JS click handler — including the no-callback
// ``InlineFilePreview`` surfaces (widget / standalone markdown).
//
// External (cross-origin) sources have no managed download endpoint
// and fall back to the source's own previewUrl.
export const getInlineFileDownloadUrl = (
  source: InlineFilePreviewSource,
  apiUrl: string
): string => {
  if (source.fileId) {
    return `${apiUrl}/api/files/public/download/${encodeURIComponent(source.fileId)}`
  }
  return getInlineFilePreviewUrl(source, apiUrl)
}

export const getPreviewUrlTrust = (
  source: InlineFilePreviewSource,
  apiUrl: string
): PreviewUrlTrust => {
  if (!source.previewUrl || source.fileId) {
    return { isExternal: false, isTrusted: true }
  }

  if (source.previewUrl.startsWith('/api/files/')) {
    return { isExternal: false, isTrusted: true }
  }

  let previewUrl: URL
  let apiOrigin: string
  try {
    previewUrl = new URL(source.previewUrl, apiUrl)
    apiOrigin = new URL(apiUrl).origin
  } catch {
    return { isExternal: true, isTrusted: false }
  }

  if (
    previewUrl.origin === apiOrigin &&
    previewUrl.pathname.startsWith('/api/files/')
  ) {
    return { isExternal: false, isTrusted: true }
  }

  return {
    domain: previewUrl.hostname,
    isExternal: true,
    isTrusted: false,
  }
}

export const arrayBufferToBase64 = (buffer: ArrayBuffer): string => {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  const chunkSize = 0x8000

  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize))
  }

  return btoa(binary)
}

export const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
