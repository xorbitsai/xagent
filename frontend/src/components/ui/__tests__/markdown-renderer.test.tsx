/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor, cleanup } from '@testing-library/react'

const apiRequestMock = vi.hoisted(() => vi.fn())

vi.mock('@/lib/utils', () => ({
  getApiUrl: () => 'http://api.local',
}))

vi.mock('@/lib/api-wrapper', () => ({
  apiRequest: apiRequestMock,
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
