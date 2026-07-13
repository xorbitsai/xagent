/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/utils', () => ({
  cn: (...classes: Array<string | undefined | false>) => classes.filter(Boolean).join(' '),
  getApiUrl: () => 'http://api.local',
  getFilePublicPreviewUrl: (fileId: string, apiUrl = 'http://api.local') =>
    `${apiUrl}/api/files/public/preview/${encodeURIComponent(fileId)}`,
  getFilePublicDownloadUrl: (fileId: string, apiUrl = 'http://api.local') =>
    `${apiUrl}/api/files/public/download/${encodeURIComponent(fileId)}`,
}))

vi.mock('@/lib/api-wrapper', () => ({
  apiRequest: apiRequestMock,
}))

vi.mock('@/components/file/docx-preview-renderer', () => ({
  DocxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="docx-preview">{base64Content}</div>
  ),
}))

vi.mock('@/components/file/excel-preview-renderer', () => ({
  ExcelPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="excel-preview">{base64Content}</div>
  ),
}))

vi.mock('@/components/file/pptx-preview-renderer', () => ({
  PptxPreviewRenderer: ({ base64Content }: { base64Content: string }) => (
    <div data-testid="pptx-preview">{base64Content}</div>
  ),
}))

import { InlineFilePreview } from './inline-file-preview'

describe('InlineFilePreview', () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('falls back to authenticated preview for uuid image file ids', async () => {
    const blob = new Blob(['image-bytes'], { type: 'image/png' })
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => blob,
    })

    render(
      <InlineFilePreview
        source={{
          type: 'image',
          fileId: '550e8400-e29b-41d4-a716-446655440000',
          filename: 'linkedin-visual.png',
        }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/550e8400-e29b-41d4-a716-446655440000',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const image = screen.getByAltText('linkedin-visual.png')
    await waitFor(() => {
      expect(image.getAttribute('src')).toMatch(/^blob:/)
    })
  })

  it('extracts uuid from file paths that include a filename suffix', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })

    render(
      <InlineFilePreview
        source={{
          type: 'image',
          fileId: '550e8400-e29b-41d4-a716-446655440000/linkedin.png',
          filename: 'linkedin.png',
        }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/550e8400-e29b-41d4-a716-446655440000',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })
  })

  it('loads image previews through authenticated preview when file id is present', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })

    render(
      <InlineFilePreview
        source={{ type: 'image', fileId: 'image-file-id', filename: 'plot.png' }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/image-file-id',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const image = await screen.findByAltText('plot.png')
    expect(image.getAttribute('src')).toMatch(/^blob:/)
  })

  it('passes resolved file id to onFileClick for uuid paths with filename suffix', () => {
    const handleFileClick = vi.fn()

    render(
      <InlineFilePreview
        source={{
          type: 'presentation',
          fileId: '550e8400-e29b-41d4-a716-446655440000/slides.pptx',
          filename: 'slides.pptx',
        }}
        onFileClick={handleFileClick}
      />
    )

    fireEvent.click(screen.getByText('Open'))

    expect(handleFileClick).toHaveBeenCalledWith(
      '550e8400-e29b-41d4-a716-446655440000',
      'slides.pptx'
    )
  })

  it('does not request public preview before authenticated image fallback', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })

    render(
      <InlineFilePreview
        source={{
          type: 'image',
          fileId: '550e8400-e29b-41d4-a716-446655440000',
          filename: 'plot.png',
        }}
      />
    )

    expect(screen.queryByAltText('plot.png')).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/550e8400-e29b-41d4-a716-446655440000',
      expect.anything()
    )

    await waitFor(() => {
      expect(screen.getByAltText('plot.png').getAttribute('src')).toMatch(/^blob:/)
    })
  })

  it('renders image previews from file ids', async () => {
    apiRequestMock.mockResolvedValue({
      ok: false,
    })

    render(
      <InlineFilePreview
        source={{ type: 'image', fileId: 'image-file-id', filename: 'plot.png' }}
      />
    )

    const image = await screen.findByAltText('plot.png')
    await waitFor(() => {
      expect(image).toHaveAttribute(
        'src',
        'http://api.local/api/files/public/preview/image-file-id'
      )
    })
  })

  it('loads managed audio files through authenticated preview', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['audio-bytes'], { type: 'audio/mpeg' }),
    })

    render(
      <InlineFilePreview
        source={{ type: 'audio', fileId: 'audio-file-id', filename: 'podcast.mp3' }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/audio-file-id',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const audio = await screen.findByLabelText('podcast.mp3')
    expect(audio.tagName.toLowerCase()).toBe('audio')
    expect(audio.getAttribute('src')).toMatch(/^blob:/)
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('href')).toMatch(
      /^blob:/
    )
  })

  it('falls back to the public audio preview when authenticated loading fails', async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 401 })

    render(
      <InlineFilePreview
        source={{ type: 'audio', fileId: 'audio-file-id', filename: 'podcast.mp3' }}
      />
    )

    const audio = await screen.findByLabelText('podcast.mp3')
    expect(audio).toHaveAttribute(
      'src',
      'http://api.local/api/files/public/preview/audio-file-id'
    )
    expect(screen.getByRole('link', { name: 'Open' })).toHaveAttribute(
      'href',
      'http://api.local/api/files/public/preview/audio-file-id'
    )
  })

  it('loads legacy workspace audio paths through authenticated preview', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['audio-bytes'], { type: 'audio/mpeg' }),
    })

    render(
      <InlineFilePreview
        source={{
          type: 'audio',
          fileId: 'output/xagent_061_podcast.mp3',
          filename: 'xagent_061_podcast.mp3',
        }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/output%2Fxagent_061_podcast.mp3',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const audio = await screen.findByLabelText('xagent_061_podcast.mp3')
    await waitFor(() => {
      expect(audio.getAttribute('src')).toMatch(/^blob:/)
    })
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('href')).toMatch(
      /^blob:/
    )
  })

  it('mounts PptxPreviewRenderer immediately with fileId without eager byte fetch', () => {
    // PDF-first path: when a managed fileId is available, InlineFilePreview
    // skips the eager /api/files/public/preview bytes download and mounts
    // PptxPreviewRenderer directly with the fileId.  The renderer then probes
    // /api/files/preview-pdf/{fileId} (LibreOffice PDF) first and only
    // lazy-fetches raw bytes if that 503s.  This avoids paying the full PPTX
    // download + base64 memory cost for large decks when LibreOffice is
    // available.
    render(
      <InlineFilePreview
        source={{
          type: 'presentation',
          fileId: 'slides-file-id',
          filename: 'slides.pptx',
        }}
      />
    )

    // Renderer is mounted synchronously — no async wait needed.
    expect(screen.getByTestId('pptx-preview')).toBeInTheDocument()
    // No eager byte fetch — the renderer lazy-fetches on its own if needed.
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/slides-file-id',
      expect.anything()
    )
  })

  it('opens inline previews through the file preview callback when available', () => {
    const handleFileClick = vi.fn()

    render(
      <InlineFilePreview
        source={{
          type: 'presentation',
          fileId: 'slides-file-id',
          filename: 'slides.pptx',
        }}
        onFileClick={handleFileClick}
      />
    )

    fireEvent.click(screen.getByText('Open'))

    expect(handleFileClick).toHaveBeenCalledWith('slides-file-id', 'slides.pptx')
  })

  it('uses the public download URL as the inline preview open link href', () => {
    // The "Open" link must route through /api/files/public/download, not
    // /api/files/public/preview: preview is for inline rendering (and on
    // some deployments returns a derived payload), while public/download
    // serves the source bytes with a ``Content-Disposition: attachment;
    // filename=...`` header so a save lands as the real filename rather
    // than the bare file id. The public/* route is required because
    // plain ``<a href>`` navigation (and middle/right-click open-in-tab
    // / copy-link) doesn't carry a bearer token.
    const handleFileClick = vi.fn()

    render(
      <InlineFilePreview
        source={{
          type: 'presentation',
          fileId: 'slides-file-id',
          filename: 'slides.pptx',
        }}
        onFileClick={handleFileClick}
      />
    )

    const openLink = screen.getByRole('link', { name: 'Open' })
    expect(openLink).toHaveAttribute(
      'href',
      'http://api.local/api/files/public/download/slides-file-id'
    )

    fireEvent.click(openLink)
    expect(handleFileClick).toHaveBeenCalledWith('slides-file-id', 'slides.pptx')
  })

  it('loads document previews through the document renderer', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })

    render(
      <InlineFilePreview
        source={{ type: 'document', fileId: 'doc-file-id', filename: 'report.docx' }}
      />
    )

    expect(await screen.findByTestId('docx-preview')).toHaveTextContent('QUI=')
    expect(apiRequestMock).toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/doc-file-id',
      expect.objectContaining({ cache: 'no-cache' })
    )
  })

  it('loads spreadsheet previews through the spreadsheet renderer', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([88, 89]).buffer,
    })

    render(
      <InlineFilePreview
        source={{
          type: 'spreadsheet',
          fileId: 'sheet-file-id',
          filename: 'data.xlsx',
        }}
      />
    )

    expect(await screen.findByTestId('excel-preview')).toHaveTextContent('WFk=')
  })

  it('uses localized text for preview load failures', async () => {
    apiRequestMock.mockResolvedValue({ ok: false })

    render(
      <InlineFilePreview
        source={{ type: 'document', fileId: 'doc-file-id', filename: 'report.docx' }}
      />
    )

    expect(await screen.findByText('Failed to load preview.')).toBeInTheDocument()
    expect(screen.queryByText('Localized load failure')).not.toBeInTheDocument()
  })

  it('uses the public download URL as the non-previewable file link href', () => {
    // Non-previewable artifacts (zip, etc.) collapse the file card into
    // a single download link — same reasoning as the inline-preview Open
    // link: route through /api/files/public/download so the save
    // filename is the source name, not the file id, AND so middle/
    // right-click open-in-tab / copy-link still works without a token.
    const handleFileClick = vi.fn()

    render(
      <InlineFilePreview
        source={{ type: 'file', fileId: 'archive-file-id', filename: 'archive.zip' }}
        onFileClick={handleFileClick}
      />
    )

    const link = screen.getByRole('link', { name: 'archive.zip' })
    expect(link).toHaveAttribute(
      'href',
      'http://api.local/api/files/public/download/archive-file-id'
    )

    fireEvent.click(link)
    expect(handleFileClick).toHaveBeenCalledWith('archive-file-id', 'archive.zip')
  })

  it('does not automatically load cross-origin document preview URLs', () => {
    render(
      <InlineFilePreview
        source={{
          type: 'document',
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        }}
      />
    )

    expect(screen.getByText('cdn.example.com')).toBeInTheDocument()
    expect(
      screen.getByRole('link', { name: 'report.docx cdn.example.com Open' })
    ).toHaveAttribute(
      'href',
      'https://cdn.example.com/report.docx'
    )
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByTestId('docx-preview')).not.toBeInTheDocument()
  })

  it('uses file-id previews when a source also has an external preview URL', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })

    render(
      <InlineFilePreview
        source={{
          type: 'document',
          fileId: 'doc-file-id',
          previewUrl: 'https://cdn.example.com/report.docx',
          filename: 'report.docx',
        }}
      />
    )

    expect(await screen.findByTestId('docx-preview')).toBeInTheDocument()
    expect(apiRequestMock).toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/doc-file-id',
      expect.objectContaining({ cache: 'no-cache' })
    )
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      'https://cdn.example.com/report.docx',
      expect.anything()
    )
  })

  it('uses authenticated preview for images with fileId even when previewUrl is set', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })

    render(
      <InlineFilePreview
        source={{
          type: 'image',
          fileId: 'image-file-id',
          previewUrl: 'https://cdn.example.com/plot.png',
          filename: 'plot.png',
        }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/image-file-id',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const image = await screen.findByAltText('plot.png')
    expect(image.getAttribute('src')).toMatch(/^blob:/)
  })

  it('revokes blob URLs when unmounting during authenticated image fetch', async () => {
    const createObjectUrlSpy = vi.spyOn(URL, 'createObjectURL')
    const revokeObjectUrlSpy = vi.spyOn(URL, 'revokeObjectURL')
    let resolveBlob: ((blob: Blob) => void) | undefined

    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: () =>
        new Promise<Blob>((resolve) => {
          resolveBlob = resolve
        }),
    })

    const { unmount } = render(
      <InlineFilePreview
        source={{
          type: 'image',
          fileId: 'image-file-id',
          filename: 'plot.png',
        }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalled()
    })

    unmount()
    resolveBlob?.(new Blob(['image-bytes'], { type: 'image/png' }))

    await waitFor(() => {
      expect(createObjectUrlSpy).not.toHaveBeenCalled()
    })
    expect(revokeObjectUrlSpy).not.toHaveBeenCalled()

    createObjectUrlSpy.mockRestore()
    revokeObjectUrlSpy.mockRestore()
  })

  it('does not automatically render cross-origin image preview URLs', () => {
    render(
      <InlineFilePreview
        source={{
          type: 'image',
          previewUrl: 'https://cdn.example.com/plot.png',
          filename: 'plot.png',
        }}
      />
    )

    expect(screen.getByText('cdn.example.com')).toBeInTheDocument()
    expect(screen.queryByAltText('plot.png')).not.toBeInTheDocument()
  })
})
