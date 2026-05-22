/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/utils', () => ({
  cn: (...classes: Array<string | undefined | false>) => classes.filter(Boolean).join(' '),
  getApiUrl: () => 'http://api.local',
  getFilePublicPreviewUrl: (fileId: string, apiUrl = 'http://api.local') =>
    `${apiUrl}/api/files/public/preview/${encodeURIComponent(fileId)}`,
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

vi.mock('@/contexts/i18n-context', () => ({
  useI18n: () => ({
    t: (key: string) => {
      if (key === 'files.previewDialog.buttons.open') return 'Open'
      if (key === 'files.previewDialog.errors.loadFailed') return 'Failed to load preview.'
      if (key === 'markdownRenderer.loadAgentDetailsFailed') return key
      return key
    },
  }),
}))

import { MarkdownRenderer } from '../markdown-renderer'

describe('MarkdownRenderer', () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it('renders inline math with KaTeX without leaving dollar delimiters', () => {
    const content = 'The equation is $x^2 + y^2 = 1$.'
    render(<MarkdownRenderer content={content} />)

    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBeGreaterThan(0)
    expect(screen.queryByText(/\$x\^2 \+ y\^2 = 1\$/)).toBeNull()
  })

  it('does not treat $PATH inside code block as math', () => {
    const content = '```bash\necho $PATH\n```'
    render(<MarkdownRenderer content={content} />)

    const pre = screen.getByText(/echo \$PATH/)
    expect(pre).toBeInTheDocument()
    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBe(0)
  })

  it('does not treat $HOME inside inline code as math', () => {
    const content = 'Use `echo $HOME` to see your home dir.'
    render(<MarkdownRenderer content={content} />)

    const code = screen.getByText('echo $HOME')
    expect(code.tagName.toLowerCase()).toBe('code')
    const mathElements = document.querySelectorAll('.katex')
    expect(mathElements.length).toBe(0)
  })

  it('handles file: links with onFileClick callback', () => {
    const handleFileClick = vi.fn()
    const content = '[open file](file:/tmp/test.txt)'

    render(<MarkdownRenderer content={content} onFileClick={handleFileClick} />)

    const link = screen.getByText('open file')
    fireEvent.click(link)

    expect(handleFileClick).toHaveBeenCalledTimes(1)
    expect(handleFileClick).toHaveBeenCalledWith('/tmp/test.txt', 'open file')
  })

  it('renders pptx file links as inline previews', async () => {
    // Arbitrary sentinel bytes; base64 encodes to "AQI=". See
    // inline-file-preview.test.tsx for why we avoid "PK" here.
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([0x01, 0x02]).buffer,
    })

    const content = '[example_presentation.pptx](file:99fb81ab-b995-4976-be18-21b02f748768)'
    render(<MarkdownRenderer content={content} />)

    // Browsers can't render raw .pptx in an iframe, and the backend's
    // /api/files/public/preview endpoint now returns the raw bytes, so
    // pptx inline previews are funnelled through PptxPreviewRenderer
    // (canvas-based, pptxviewjs) — same fetch+base64 pattern as docx/xlsx.
    expect(await screen.findByTestId('pptx-preview')).toHaveTextContent('AQI=')
    expect(apiRequestMock).toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/99fb81ab-b995-4976-be18-21b02f748768',
      expect.objectContaining({ cache: 'no-cache' })
    )
    expect(screen.queryByText('example_presentation.pptx')?.tagName.toLowerCase()).not.toBe('a')
  })

  it('opens pptx inline preview links with onFileClick when provided', () => {
    const handleFileClick = vi.fn()
    const content = '[example_presentation.pptx](file:pptx-file-id)'

    render(<MarkdownRenderer content={content} onFileClick={handleFileClick} />)

    fireEvent.click(screen.getByText('Open'))

    expect(handleFileClick).toHaveBeenCalledWith(
      'pptx-file-id',
      'example_presentation.pptx'
    )
  })

  it('renders docx file links with the document preview renderer', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })
    const content = '[report.docx](file:doc-file-id)'

    render(<MarkdownRenderer content={content} />)

    expect(await screen.findByTestId('docx-preview')).toHaveTextContent('QUI=')
    expect(apiRequestMock).toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/doc-file-id',
      expect.objectContaining({ cache: 'no-cache' })
    )
  })

  it('renders xlsx file links with the spreadsheet preview renderer', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([88, 89]).buffer,
    })
    const content = '[data.xlsx](file:sheet-file-id)'

    render(<MarkdownRenderer content={content} />)

    expect(await screen.findByTestId('excel-preview')).toHaveTextContent('WFk=')
  })

  it('preserves standard relative markdown links and images', () => {
    const content = '[relative doc](../doc.md)\n\n![relative image](./a.png)'
    render(<MarkdownRenderer content={content} />)

    const link = screen.getByText('relative doc')
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute('href', '../doc.md')

    const image = screen.getByAltText('relative image')
    expect(image).toBeInTheDocument()
    expect(image).toHaveAttribute('src', './a.png')
  })

  it('uses authenticated preview fallback for non-uuid file: images', async () => {
    apiRequestMock.mockResolvedValue({ ok: false })
    const content = '![final image](file:output/screenshot.png)'
    render(<MarkdownRenderer content={content} />)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/output%2Fscreenshot.png',
        expect.objectContaining({
          cache: 'no-cache',
          headers: expect.objectContaining({
            'Cache-Control': 'no-cache',
            Pragma: 'no-cache',
          }),
        })
      )
    })
  })

  it('does not run authenticated fallback for uuid file: images', async () => {
    const content = '![uuid image](file:550e8400-e29b-41d4-a716-446655440000)'
    render(<MarkdownRenderer content={content} />)

    await waitFor(() => {
      const image = screen.getByAltText('uuid image')
      expect(image).toBeInTheDocument()
    })

    expect(apiRequestMock).not.toHaveBeenCalled()
  })
})
