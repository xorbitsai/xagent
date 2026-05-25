import { describe, expect, it } from 'vitest'

import {
  getInlineFileDownloadUrl,
  getInlineFilePreviewKind,
  getInlineFilePreviewUrl,
  getPreviewUrlTrust,
} from './inline-file-preview-utils'

describe('inline-file-preview-utils', () => {
  it('prefers explicit artifact type when filename/mime agree', () => {
    expect(
      getInlineFilePreviewKind({
        type: 'presentation',
        filename: 'unknown.bin',
        mimeType:
          'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      })
    ).toBe('presentation')
  })

  it('falls back to mime type and filename extension when resolving preview kind', () => {
    expect(
      getInlineFilePreviewKind({
        filename: 'report',
        mimeType:
          'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      })
    ).toBe('document')
    expect(getInlineFilePreviewKind({ filename: 'data.xlsx' })).toBe('spreadsheet')
    expect(getInlineFilePreviewKind({ filename: 'chart.png' })).toBe('image')
  })

  it('classifies .pptx (OOXML) as inline-previewable presentation', () => {
    // PptxPreviewRenderer (pptxviewjs) renders .pptx in-browser, so the
    // shared classifier flags it as 'presentation'.
    expect(getInlineFilePreviewKind({ filename: 'deck.pptx' })).toBe('presentation')
    expect(
      getInlineFilePreviewKind({
        filename: 'unknown',
        mimeType:
          'application/vnd.openxmlformats-officedocument.presentationml.presentation',
      })
    ).toBe('presentation')
  })

  it('classifies legacy .ppt as a non-previewable file', () => {
    // pptxviewjs only supports OOXML .pptx; the legacy binary .ppt
    // (mime ``application/vnd.ms-powerpoint``) must NOT reach the
    // PptxPreviewRenderer mount path. The shared classifier returns
    // 'file' so callers render a download/file link instead.
    expect(getInlineFilePreviewKind({ filename: 'old-deck.ppt' })).toBe('file')
    expect(
      getInlineFilePreviewKind({
        filename: 'unknown',
        mimeType: 'application/vnd.ms-powerpoint',
      })
    ).toBe('file')
  })

  it('does not bypass the .pptx-only boundary via explicit type', () => {
    // Producer-emitted ``type: 'presentation'`` is no longer an
    // unconditional pass — the classifier still cross-checks
    // filename/mime to confirm the payload is OOXML .pptx. A legacy
    // .ppt arriving with ``type: 'presentation'`` (which the old
    // backend-side ``artifact_type_for_filename`` produced) must fall
    // through to 'file' so PptxPreviewRenderer never tries to render
    // a format pptxviewjs cannot parse.
    expect(
      getInlineFilePreviewKind({
        type: 'presentation',
        filename: 'old-deck.ppt',
      })
    ).toBe('file')
    expect(
      getInlineFilePreviewKind({
        type: 'presentation',
        mimeType: 'application/vnd.ms-powerpoint',
      })
    ).toBe('file')
    // But a matching ``type: 'presentation'`` + .pptx artifact still
    // resolves to the previewable kind.
    expect(
      getInlineFilePreviewKind({
        type: 'presentation',
        filename: 'deck.pptx',
      })
    ).toBe('presentation')
    // No filename/mime hint: keep the historical lenient behavior so
    // existing callers that only set ``type`` (and assume .pptx) keep
    // working.
    expect(getInlineFilePreviewKind({ type: 'presentation' })).toBe('presentation')
  })

  it('builds public preview URLs from file ids and preserves absolute preview URLs', () => {
    expect(
      getInlineFilePreviewUrl(
        { fileId: 'slides-file-id', filename: 'slides.pptx' },
        'http://api.local'
      )
    ).toBe('http://api.local/api/files/public/preview/slides-file-id')

    expect(
      getInlineFilePreviewUrl(
        {
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        },
        'http://api.local'
      )
    ).toBe('https://cdn.example.com/report.docx')
  })

  it('prefers file-id preview URLs over external preview URLs', () => {
    expect(
      getInlineFilePreviewUrl(
        {
          fileId: 'doc-file-id',
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        },
        'http://api.local'
      )
    ).toBe('http://api.local/api/files/public/preview/doc-file-id')
  })

  it('classifies file-id and API preview URLs as trusted', () => {
    expect(
      getPreviewUrlTrust(
        { fileId: 'slides-file-id', filename: 'slides.pptx' },
        'http://api.local'
      )
    ).toEqual({ isExternal: false, isTrusted: true })

    expect(
      getPreviewUrlTrust(
        { previewUrl: '/api/files/public/preview/slides-file-id' },
        'http://api.local'
      )
    ).toEqual({ isExternal: false, isTrusted: true })

    expect(
      getPreviewUrlTrust(
        { previewUrl: 'http://api.local/api/files/public/preview/slides-file-id' },
        'http://api.local'
      )
    ).toEqual({ isExternal: false, isTrusted: true })
  })

  it('classifies cross-origin preview URLs as external and untrusted', () => {
    expect(
      getPreviewUrlTrust(
        { previewUrl: 'https://cdn.example.com/report.docx' },
        'http://api.local'
      )
    ).toEqual({
      domain: 'cdn.example.com',
      isExternal: true,
      isTrusted: false,
    })
  })

  it('routes managed file ids through the public download endpoint for open links', () => {
    // The "Open" affordance must hand the user the source artifact, not
    // the inline-preview payload (which on some deployments is a derived
    // PDF), so file-id sources resolve to /api/files/public/download.
    // The public/* route is required because plain ``<a href>`` clicks
    // (and middle-click / right-click "open in new tab" / "copy link")
    // don't carry the bearer token that the auth'd /api/files/download
    // route requires.
    expect(
      getInlineFileDownloadUrl(
        { fileId: 'slides-file-id', filename: 'slides.pptx' },
        'http://api.local'
      )
    ).toBe('http://api.local/api/files/public/download/slides-file-id')
  })

  it('falls back to the preview URL for external sources without a file id', () => {
    // Cross-origin previewUrl sources have no managed download endpoint —
    // the only sane "Open" target is the previewUrl itself.
    expect(
      getInlineFileDownloadUrl(
        {
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        },
        'http://api.local'
      )
    ).toBe('https://cdn.example.com/report.docx')
  })

  it('prefers file-id download URLs over external preview URLs', () => {
    expect(
      getInlineFileDownloadUrl(
        {
          fileId: 'doc-file-id',
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        },
        'http://api.local'
      )
    ).toBe('http://api.local/api/files/public/download/doc-file-id')
  })

  it('returns an empty string when no file id or preview URL is available', () => {
    expect(getInlineFileDownloadUrl({ filename: 'anon.bin' }, 'http://api.local')).toBe('')
  })
})
