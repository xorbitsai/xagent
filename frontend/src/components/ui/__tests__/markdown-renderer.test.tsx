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
  PptxPreviewRenderer: ({
    base64Content,
    fileId,
  }: {
    base64Content?: string
    fileId?: string
  }) => <div data-testid="pptx-preview">{base64Content ?? fileId ?? ''}</div>,
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

import { JsonRenderer, MarkdownRenderer } from '../markdown-renderer'
import {
  FileAccessProvider,
  defaultFileAccessPolicy,
} from '@/contexts/file-access-context'
import {
  AgentCardPresentationCapability,
  LinksOpenInNewTabCapability,
} from '@/contexts/presentation-capabilities'
import {
  getFilesDisabledPresentationFileLabel,
  projectFilesDisabledPresentation,
  sanitizeFilesDisabledPresentationText,
} from '@/lib/files-disabled-presentation'

describe('MarkdownRenderer', () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
  })

  it('retains snake-case filenames as the safe label for file records', () => {
    expect(getFilesDisabledPresentationFileLabel({
      file_path: '/private/report.pdf',
      file_name: 'report.pdf',
    })).toBe('report.pdf')
  })

  it('preserves unrelated backtick URLs while inertizing file references', () => {
    expect(sanitizeFilesDisabledPresentationText(
      'Call `https://api.example/tasks/42` then [open report](file:secret-id).',
    )).toBe(
      'Call `https://api.example/tasks/42` then open report.',
    )
  })

  it('removes producer-shaped local path fields without erasing sibling business identity', () => {
    expect(projectFilesDisabledPresentation({
      success: true,
      id: 'workspace-id',
      url: 'https://api.example/workspaces/workspace-id',
      workspace_dir: '/private/workspaces/workspace-id',
      output_dir: '/private/workspaces/workspace-id/output',
      message: [
        'Workspace /private/workspaces/workspace-id',
        'writes to /private/workspaces/workspace-id/output',
      ].join(' '),
      files: [{ path: 'SKILL.md', size: 1234 }],
    })).toEqual({
      success: true,
      id: 'workspace-id',
      url: 'https://api.example/workspaces/workspace-id',
      message: 'Workspace workspace-id writes to output',
      files: [{ size: 1234 }],
    })
  })

  it('preserves connector business paths that have no file evidence', () => {
    const connectorResult = {
      request_path: '/v1/shifts',
      method: 'GET',
      shift: {
        path: '/care/shift/42',
        directory: 'north-region',
        status: 'open',
      },
      sizedShift: {
        path: '/care/shift/43',
        size: 20,
        status: 'open',
      },
      preview: {
        id: 'invoice-42',
        preview_url: 'https://api.example/invoices/42/preview',
        url: 'https://api.example/invoices/42',
        status: 'ready',
      },
      edge: {
        source: 'care-service',
        destination: 'billing-service',
        size: 20,
      },
    }

    expect(projectFilesDisabledPresentation(connectorResult)).toEqual({
      ...connectorResult,
      preview: {
        id: 'invoice-42',
        url: 'https://api.example/invoices/42',
        status: 'ready',
      },
    })
  })

  it('removes local path fields emitted by current workspace, media, and document tools', () => {
    const projected = projectFilesDisabledPresentation({
      audio_path: '/private/output/audio.mp3',
      videoPath: '/private/output/video.mp4',
      translation_path: '/private/output/translation.json',
      transcriptionPath: '/private/output/transcription.txt',
      full_path: '/private/uploads/report.pdf',
      sourcePath: '/private/source/report.pdf',
      json_path: '/private/chunks/report.json',
      uploadsDirectory: '/private/uploads',
      current_path: '/private/workspace/current',
      markedImagePath: '/private/output/marked.png',
      copiedFile: {
        source: '/private/skills/example/SKILL.md',
        destination: 'output/SKILL.md',
        size: 1234,
        extracted: false,
      },
      safe: 'keep this',
    })

    expect(projected).toEqual({
      copiedFile: {
        size: 1234,
        extracted: false,
      },
      safe: 'keep this',
    })
  })

  it('removes all file identities and the HTML source from prepared asset results', () => {
    expect(projectFilesDisabledPresentation({
      success: true,
      source_file_id: 'source-file-id',
      assetFileID: 'asset-file-id',
      html_src: 'assets/chart.png',
      file_id: 'registered-file-id',
      filename: 'chart.png',
      mime_type: 'image/png',
      preview_url: 'https://files.example/preview/registered-file-id',
      file_ref: {
        file_id: 'nested-file-id',
        filename: 'chart.png',
        relative_path: 'output/assets/chart.png',
        preview_url: 'https://files.example/preview/nested-file-id',
      },
    })).toEqual({
      success: true,
      filename: 'chart.png',
      mime_type: 'image/png',
      file_ref: {
        filename: 'chart.png',
      },
    })
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

  it.each([
    '$15/month or $25/month.',
    'Budget: $10–$25.',
    'Price ratio: $15/$25.',
    'A$5.6M ≈ US$3.7M',
    'S$15 vs S$25',
    'Prices: $ 15 and $ 25.',
    '$0.50, $1,200.00 and $-25',
  ])('preserves currency as ordinary text: %s', (content) => {
    const { container } = render(<MarkdownRenderer content={content} />)

    expect(container.querySelector('p')?.textContent).toBe(content)
    expect(container.querySelector('.katex')).toBeNull()
  })

  it('preserves bold plan names and prices in the SIMBA comparison', () => {
    const { container } = render(<MarkdownRenderer content={[
      '**SIMBA $15:** 600GB, budget $10–$25.',
      'A$5.6M ≈ US$3.7M',
    ].join('\n\n')} />)

    expect(screen.getByText('SIMBA $15:').tagName).toBe('STRONG')
    expect(container.querySelector('p')?.textContent).toBe('SIMBA $15: 600GB, budget $10–$25.')
    expect(screen.getByText('A$5.6M ≈ US$3.7M')).toBeInTheDocument()
    expect(container.querySelector('.katex')).toBeNull()
  })

  it('preserves price cells, bold labels, and source links in a comparison table', () => {
    const { container } = render(<MarkdownRenderer content={[
      '| Plan | Price | Source |',
      '| --- | --- | --- |',
      '| **SIMBA** | $15–$25 | [Plans $15–$25](https://example.com/plans?min=$15&max=$25) |',
      '| Other | A$5.6M ≈ US$3.7M | — |',
    ].join('\n')} />)

    expect(screen.getAllByRole('row')).toHaveLength(3)
    expect(screen.getAllByRole('cell')).toHaveLength(6)
    expect(screen.getByRole('cell', { name: '$15–$25' })).toBeInTheDocument()
    expect(screen.getByRole('cell', { name: 'A$5.6M ≈ US$3.7M' })).toBeInTheDocument()
    expect(screen.getByText('SIMBA').tagName).toBe('STRONG')
    expect(screen.getByRole('link', { name: 'Plans $15–$25' })).toHaveAttribute(
      'href', 'https://example.com/plans?min=$15&max=$25',
    )
    expect(container.querySelector('.katex')).toBeNull()
  })

  it('keeps an unfinished second price from swallowing the first during rerenders', () => {
    const { container, rerender } = render(<MarkdownRenderer content="" />)

    for (const content of ['SIMBA $15, budget $', 'SIMBA $15, budget $10', 'SIMBA $15, budget $10–$25.']) {
      rerender(<MarkdownRenderer content={content} />)
      expect(container.querySelector('p')?.textContent).toBe(content)
      expect(container.querySelector('.katex')).toBeNull()
    }
  })

  it('wraps each table in its own keyboard-accessible scroll region', () => {
    render(<MarkdownRenderer content={[
      '| Long plan description | Price |', '| :--- | ---: |', '| SIMBA | $15 |',
      '', 'Another comparison:', '',
      '| Country | Status |', '| --- | --- |', '| AU | Paused |',
    ].join('\n')} />)
    const tables = screen.getAllByRole('table')
    expect(tables).toHaveLength(2)
    for (const table of tables) {
      expect(table.parentElement).toHaveClass('markdown-table-scroll')
      expect(table.parentElement).toHaveAttribute('tabindex', '0')
      expect(table.parentElement).toHaveAttribute('role', 'region')
      expect(table.parentElement).toHaveAttribute('aria-label', 'markdownRenderer.tableScrollLabel')
    }
    expect(screen.getByRole('columnheader', { name: 'Price' })).toHaveStyle({ textAlign: 'right' })
    expect(screen.getByRole('cell', { name: '$15' })).toHaveStyle({ textAlign: 'right' })
  })

  it('preserves escaped pipes and code inside cells without moving values between columns', () => {
    render(<MarkdownRenderer content={[
      '| Campaign | Country | Status | Spend |', '| --- | --- | --- | --- |',
      String.raw`| Jack \| TEST | AU | Paused | $1,909.81 |`,
      String.raw`| \`a\|b\` | SG | Active | $15–$25 |`.replaceAll('\\`', '`'),
    ].join('\n')} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual([
      'Jack | TEST', 'AU', 'Paused', '$1,909.81', 'a|b', 'SG', 'Active', '$15–$25',
    ])
  })

  it('keeps the table and scroll wrapper stable while a streamed row grows', () => {
    const header = '| Plan | Price |\n| --- | --- |\n'
    const { rerender } = render(<MarkdownRenderer content={header + '| SIMBA |'} />)
    const table = screen.getByRole('table')
    const wrapper = table.parentElement
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['SIMBA', ''])
    for (const row of ['| SIMBA |', '| SIMBA | $', '| SIMBA | $15', '| SIMBA | $15–$25 |']) {
      rerender(<MarkdownRenderer content={header + row} />)
      expect(screen.getByRole('table')).toBe(table)
      expect(table.parentElement).toBe(wrapper)
    }
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['SIMBA', '$15–$25'])
  })

  it.each([
    '| SIMBA | $15 | never discard this |',
    '| Jack | TEST | $1,909.81 |',
    '| `a|b` | $15 |',
    '| <img src=x onerror=alert(1)> | $15 | private value |',
  ])('preserves the exact source of a table with mismatched cells: %s', (row) => {
    const source = ['| Plan | Price |', '| --- | --- |', row].join('\n')
    const { container } = render(<MarkdownRenderer content={source} />)
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    expect(container.querySelector('pre code')?.textContent).toBe(source + '\n')
    expect(container.querySelector('img')).toBeNull()
  })

  it('preserves nested mismatched tables without changing surrounding content', () => {
    const source = '> | A | B |\n> | --- | --- |\n> | one | two | three |'
    const { container } = render(<MarkdownRenderer content={[
      'Before **table**.', '', source, '',
      '- Nested list:', '', '  | A | B |', '  | --- | --- |', '  | x | y | z |', '',
      '| Valid | Table |', '| --- | --- |', '| yes | $15 |', '', 'After table.',
    ].join('\n')} />)
    expect(container.querySelector('blockquote pre code')?.textContent).toBe(source.slice(2) + '\n')
    expect(container.querySelector('li pre code')?.textContent).toBe('| A | B |\n  | --- | --- |\n  | x | y | z |\n')
    expect(screen.getAllByRole('table')).toHaveLength(1)
    expect(screen.getByText('After table.')).toBeInTheDocument()
    expect(container.querySelector('strong')?.textContent).toBe('table')
  })

  it('preserves surplus cells from CRLF input and renders valid CRLF tables', () => {
    const header = '| A | B |\r\n| --- | --- |\r\n'
    const { container, rerender } = render(<MarkdownRenderer content={header + '| x | y | z |'} />)
    expect(container.querySelector('pre code')?.textContent).toBe(header + '| x | y | z |\n')
    rerender(<MarkdownRenderer content={header + '| x | y |'} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['x', 'y'])
  })

  it('switches safely between incomplete, valid and overflowing streamed rows', () => {
    const header = '| Campaign | Spend |\n| --- | --- |\n'
    const { container, rerender } = render(<MarkdownRenderer content={header + '| Jack |'} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['Jack', ''])
    expect(container.querySelector('pre')).toBeNull()
    rerender(<MarkdownRenderer content={header + '| Jack | $15 |'} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['Jack', '$15'])
    const overflow = header + '| Jack | TEST | $15 |'
    rerender(<MarkdownRenderer content={overflow} />)
    expect(container.querySelector('pre code')?.textContent).toBe(overflow + '\n')
    expect(screen.queryByRole('table')).not.toBeInTheDocument()
    rerender(<MarkdownRenderer content={header + String.raw`| Jack \| TEST | $15 |`} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['Jack | TEST', '$15'])
  })

  it('retains explicitly empty cells and leaves table-shaped code examples alone', () => {
    const { container } = render(<MarkdownRenderer content={[
      '| A | B |', '| --- | --- |', '| one | |', '| | two |', '',
      '```markdown', '| A | B |', '| --- | --- |', '| x | y | z |', '```',
    ].join('\n')} />)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(['one', '', '', 'two'])
    expect(container.querySelector('pre code')?.textContent).toBe('| A | B |\n| --- | --- |\n| x | y | z |\n')
  })

  it.each([
    ['| A | B |\n| --- | --- |', 2, []],
    ['| A |\n| --- |\n| one |', 1, ['one']],
    ['| A |\n| --- |', 1, []],
    ['| A | B | C |\n| --- | --- | --- |\n| one |\n| two | three |', 3, ['one', '', '', 'two', 'three', '']],
  ])('keeps lossless GFM tables rendered: %s', (source, columns, cells) => {
    const { container } = render(<MarkdownRenderer content={source} />)
    expect(screen.getByRole('table')).toBeInTheDocument()
    expect(screen.getAllByRole('columnheader')).toHaveLength(columns)
    expect(screen.queryAllByRole('cell').map((cell) => cell.textContent)).toEqual(cells)
    expect(container.querySelector('pre')).toBeNull()
  })

  it('replays the plan comparison table captured from the original TC6 task', () => {
    // Verbatim table excerpt from local benchmark task 880, assistant message
    // 2555. This is a rendering fixture, not a claim about current plan prices.
    const source = [
      '| Provider | Representative plans | Main proposition | Competitive strength | Weakness |',
      '|---|---:|---|---|---|',
      '| **SIMBA** | $10–$25 per 30 days | 500–800GB local data, large APAC/global roaming bundles, IDD and local-call benefits | Best international value; highly generous roaming; low prices | Plan architecture is complicated; very large headline data quotas can look less credible or less relevant to ordinary users |',
      '| **Singtel hi!** | About $15–$40 per 4 weeks, plus senior options | Large local data bundles, Singtel 5G+ network, roaming and IDD, app-based top-up | Strongest brand and perceived network reliability; broad retail and service ecosystem | Usually weaker raw value than SIMBA, Maxx or eight; higher prices for comparable heavy-data users |',
      '| **eight** | About $8–$18 per 30 days | 488–688GB headline data, 4G/5G tiers, roaming, IDD and local benefits | Strong data value plus automatic rollover; rollover balance can build up to eight months of current-plan entitlement | More complex naming and benefit structure; customers may struggle to understand what is local data versus roaming data |',
      '| **Maxx** | About $7.90–$12 per month | 290–500GB plans, 4G/5G options, Singapore/Malaysia or wider Asian usage, IDD and roaming | Very sharp price points; simple proposition; backed by M1’s network | Less generous than SIMBA/eight at the top end; no data rollover in the surfaced comparison; physical SIM availability may be less flexible than eSIM-led rivals |',
    ].join('\n')
    const { container } = render(<MarkdownRenderer content={source} />)
    expect(screen.getAllByRole('columnheader')).toHaveLength(5)
    expect(screen.getAllByRole('row')).toHaveLength(5)
    expect(screen.getAllByRole('cell').map((cell) => cell.textContent)).toEqual(
      source.split('\n').slice(2).flatMap((row) => row.split('|').slice(1, -1).map(
        (cell) => cell.trim().replaceAll('**', ''),
      )),
    )
    expect(screen.getByRole('table').parentElement).toHaveClass('markdown-table-scroll')
    expect(container.querySelector('.katex')).toBeNull()
  })

  it('renders numeric formulas alongside currency without merging their delimiters', () => {
    const { container } = render(
      <MarkdownRenderer content="Pay $15 or $25; the equation is $2 + 2 = 4$, then $x^2$." />,
    )

    expect(container.querySelector('p')?.textContent).toMatch(/^Pay \$15 or \$25; the equation is /)
    expect(container.querySelectorAll('.katex')).toHaveLength(2)
    expect(Array.from(container.querySelectorAll('annotation')).map((node) => node.textContent)).toEqual([
      '2 + 2 = 4', 'x^2',
    ])
  })

  it.each([
    ['Number: $10$.', '10'],
    ['The $x$-axis.', 'x'],
    ['Coordinate ($x$).', 'x'],
    ['At end $x$', 'x'],
    ['Inline $$ x + y $$ math.', 'x + y'],
    ['Inline $$\tx\t$$ math.', '\tx\t'],
    ['$$\nE = mc^2\n$$', 'E = mc^2'],
    ['Formula: $x_1 + \\frac{1}{2}$.', 'x_1 + \\frac{1}{2}'],
  ])('preserves explicit math delimiters: %s', (content, formula) => {
    const { container } = render(<MarkdownRenderer content={content} />)

    expect(container.querySelectorAll('.katex')).toHaveLength(1)
    expect(container.querySelector('annotation')?.textContent).toBe(formula)
  })

  it.each(['$ x$', '$x $', '$x$2', '$\tx\t$', '$\u00a0x\u00a0$'])(
    'requires tight single-dollar delimiters and a non-digit after the closer: %s',
    (content) => {
      const { container } = render(<MarkdownRenderer content={content} />)

      expect(container.querySelector('p')?.textContent).toBe(content)
      expect(container.querySelector('.katex')).toBeNull()
    },
  )

  it.each([
    [String.raw`It costs \$20 (USD\$).`, 'It costs $20 (USD$).'],
    [String.raw`Cost \$5, currency USD\$`, 'Cost $5, currency USD$'],
    [String.raw`\$X+\$Y`, '$X+$Y'],
    [String.raw`\$10\$`, '$10$'],
  ])('supports explicit literal dollars in ambiguous prose: %s', (content, expected) => {
    const { container } = render(<MarkdownRenderer content={content} />)

    expect(container.querySelector('p')?.textContent).toBe(expected)
    expect(container.querySelector('.katex')).toBeNull()
  })

  it('preserves existing escaped-dollar and code behavior', () => {
    const { container } = render(<MarkdownRenderer content={[
      String.raw`Prices: \$15 and \$25.`,
      '`$15 and $25`',
      '```text\n$15 and $25\n```',
    ].join('\n\n')} />)

    expect(screen.getByText('Prices: $15 and $25.')).toBeInTheDocument()
    expect(Array.from(container.querySelectorAll('code')).map((node) => node.textContent?.trim())).toEqual([
      '$15 and $25', '$15 and $25',
    ])
    expect(container.querySelector('.katex')).toBeNull()
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

  it('passes resolved file id to onFileClick for non-previewable file links', () => {
    const handleFileClick = vi.fn()
    const content = '[archive.zip](file:550e8400-e29b-41d4-a716-446655440000/archive.zip)'

    render(<MarkdownRenderer content={content} onFileClick={handleFileClick} />)

    fireEvent.click(screen.getByText('archive.zip'))

    expect(handleFileClick).toHaveBeenCalledWith(
      '550e8400-e29b-41d4-a716-446655440000',
      'archive.zip'
    )
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

  it('passes the model label, not the title, to onFileClick for a non-previewable file link', () => {
    // "notes.txt" isn't a previewable kind, so resolvePreviewableFileLink
    // returns null and this falls to the plain-anchor branch below --
    // the filename handed to onFileClick must still be linkText-first,
    // matching every other display path in this component.
    const handleFileClick = vi.fn()
    const content = '[Open Notes](file:some-id "notes.txt")'

    render(<MarkdownRenderer content={content} onFileClick={handleFileClick} />)

    fireEvent.click(screen.getByText('Open Notes'))

    expect(handleFileClick).toHaveBeenCalledWith('some-id', 'Open Notes')
  })

  it('renders file links and images as inert text when files are disabled', () => {
    const handleFileClick = vi.fn()
    const content = [
      '[report.docx](file:doc-file-id)',
      '![generated image](file:output/generated.png)',
    ].join('\n\n')

    const { container } = render(
      <MarkdownRenderer
        content={content}
        filesDisabled
        onFileClick={handleFileClick}
      />
    )

    expect(screen.getByText('report.docx')).not.toHaveAttribute('href')
    expect(screen.getByText('generated image')).not.toHaveAttribute('src')
    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.queryByTestId('docx-preview')).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container.innerHTML).not.toContain('doc-file-id')
    expect(container.innerHTML).not.toContain('output/generated.png')

    fireEvent.click(screen.getByText('report.docx'))
    fireEvent.click(screen.getByText('generated image'))
    expect(handleFileClick).not.toHaveBeenCalled()
  })

  it('disables agent cards whenever files are disabled, including an explicit card capability', () => {
    render(
      <MarkdownRenderer
        filesDisabled
        agentCardsEnabled
        content="[Specialist](agent://42)"
      />,
    )

    expect(screen.getByText('Specialist')).not.toHaveAttribute('data-agent-id')
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it('keeps managed app URLs inert and avoids agent-card REST calls when their capabilities are disabled', () => {
    const { container } = render(
      <AgentCardPresentationCapability.Provider value={false}>
        <MarkdownRenderer
          filesDisabled
          content={[
            '[preview](/api/files/public/preview/secret-file)',
            '![download](https://app.example/api/files/download/secret-file)',
            '[encoded preview](/api/files/public/%70review/encoded-secret)',
            '![encoded download](https://app.example/api/files/%64ownload/encoded-secret)',
            '[encoded slash](/api/files/preview%2Fslash-secret)',
            '![double encoded](https://app.example/api/files/%2570review/double-secret)',
            '[encoded api](/%61pi/files/public/preview/api-secret)',
            '![encoded files](https://app.example/api/%66iles/public/preview/files-secret)',
            '[encoded boundary](/api/files%2Fdownload/boundary-secret)',
            '![double encoded api](/%2561pi/files/public/preview/double-api-secret)',
            '[encoded leading slash](/%2Fapi%2Ffiles%2Fpublic%2Fpreview%2Fslash-api-secret)',
            '[malformed](/api/files/%E0%A4%A)',
            '[local target](/private/tenant/local-secret.txt)',
            '![relative target](workspace/tenant/relative-secret.png)',
            '[Specialist](agent://42)',
          ].join('\n\n')}
        />
      </AgentCardPresentationCapability.Provider>,
    )

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(screen.getByText('Specialist')).not.toHaveAttribute('data-agent-id')
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container.innerHTML).not.toContain('secret-file')
    expect(container.innerHTML).not.toContain('encoded-secret')
    expect(container.innerHTML).not.toContain('slash-secret')
    expect(container.innerHTML).not.toContain('double-secret')
    expect(container.innerHTML).not.toContain('api-secret')
    expect(container.innerHTML).not.toContain('files-secret')
    expect(container.innerHTML).not.toContain('boundary-secret')
    expect(container.innerHTML).not.toContain('double-api-secret')
    expect(container.innerHTML).not.toContain('slash-api-secret')
    expect(container.innerHTML).not.toContain('local-secret')
    expect(container.innerHTML).not.toContain('relative-secret')
    expect(container.innerHTML).not.toContain('%E0%A4%A')
  })

  it('sanitizes Markdown labels and titles while preserving safe external targets', () => {
    const { container } = render(
      <MarkdownRenderer
        filesDisabled
        content={[
          '[/private/labels/report.txt](https://example.com/report "/tmp/titles/report.txt")',
          '![/workspace/images/chart.png](https://example.com/chart.png "/sandbox/titles/chart.png")',
        ].join('\n\n')}
      />,
    )

    expect(screen.getByRole('link', { name: 'report.txt' })).toHaveAttribute('href', 'https://example.com/report')
    expect(screen.getByRole('link', { name: 'report.txt' })).toHaveAttribute('title', 'report.txt')
    expect(screen.getByRole('img', { name: 'chart.png' })).toHaveAttribute('src', 'https://example.com/chart.png')
    expect(screen.getByRole('img', { name: 'chart.png' })).toHaveAttribute('title', 'chart.png')
    expect(container.innerHTML).not.toContain('/private/labels')
    expect(container.innerHTML).not.toContain('/tmp/titles')
    expect(container.innerHTML).not.toContain('/workspace/images')
    expect(container.innerHTML).not.toContain('/sandbox/titles')
  })

  it('preserves free-text and Markdown business routes when files are disabled', () => {
    render(
      <MarkdownRenderer
        filesDisabled
        content={'Call /v1/shifts, /v1/openapi.json, /care/shift/42, or /care/export.csv.\n\n[Shift API](https://api.example/v1/shifts)'}
      />,
    )

    expect(screen.getByText(/Call \/v1\/shifts, \/v1\/openapi\.json, \/care\/shift\/42, or \/care\/export\.csv/)).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Shift API' })).toHaveAttribute(
      'href',
      'https://api.example/v1/shifts',
    )
  })

  it('projects files-disabled root Markdown before parsing and keeps ordinary slash text intact', () => {
    const { container } = render(
      <MarkdownRenderer
        filesDisabled
        content={'Assistant wrote /private/reports/secret.txt from /app/src, /opt/xagent, /var/tmp, /srv/app, and /etc/passwd; keep `and/or`.\n\n[managed](file:secret-id)'}
      />,
    )

    expect(container).toHaveTextContent('Assistant wrote secret.txt from src, xagent, tmp, app, and passwd; keep and/or.')
    expect(container.innerHTML).not.toContain('/private/reports/secret.txt')
    expect(container.innerHTML).not.toContain('/app/src')
    expect(container.innerHTML).not.toContain('/opt/xagent')
    expect(container.innerHTML).not.toContain('/var/tmp')
    expect(container.innerHTML).not.toContain('/srv/app')
    expect(container.innerHTML).not.toContain('/etc/passwd')
    expect(container.innerHTML).not.toContain('secret-id')
  })

  it('renders parser-valid file reference variants as inert labels without identifiers', () => {
    const { container } = render(
      <MarkdownRenderer
        filesDisabled
        content={[
          '[reference label][artifact]',
          '',
          '[artifact]: file:reference-secret',
          '',
          '[nested **safe** label](file:nested-secret)',
          '',
          '[balanced](file:tenant(private)/balanced-secret)',
          '',
          'Bare file:bare(secret)/secret-id, file:bare-secret, and <file:autolink-secret>.',
        ].join('\n')}
      />,
    )

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('reference label')
    expect(container).toHaveTextContent('nested safe label')
    expect(container).toHaveTextContent('balanced')
    expect(container).toHaveTextContent('Bare file, file, and file.')
    expect(container.innerHTML).not.toContain('reference-secret')
    expect(container.innerHTML).not.toContain('nested-secret')
    expect(container.innerHTML).not.toContain('balanced-secret')
    expect(container.innerHTML).not.toContain('bare-secret')
    expect(container.innerHTML).not.toContain('secret-id')
    expect(container.innerHTML).not.toContain('autolink-secret')
  })

  it('keeps entity-encoded text and percent-encoded local targets inside the same inert boundary', () => {
    const { container } = render(
      <MarkdownRenderer
        filesDisabled
        content={[
          'Saved /private&#47;tenant&#47;entity-secret.txt and file&#58;entity-file-secret.',
          '',
          '[encoded root](/%70rivate/tenant/link-secret.txt)',
          '',
          '![encoded slash](/private%2Ftenant%2Fimage-secret.png)',
          '',
          '[malformed encoded root](/%70rivate/%E0%A4%A/malformed-secret.txt)',
          '',
          '![invalid encoded root](/%70rivate/%ZZ/invalid-secret.png)',
        ].join('\n')}
      />,
    )

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('Saved entity-secret.txt and file.')
    expect(container).toHaveTextContent('encoded root')
    expect(container).toHaveTextContent('encoded slash')
    expect(container).toHaveTextContent('malformed encoded root')
    expect(container).toHaveTextContent('invalid encoded root')
    expect(container.innerHTML).not.toContain('/private')
    expect(container.innerHTML).not.toContain('entity-file-secret')
    expect(container.innerHTML).not.toContain('link-secret')
    expect(container.innerHTML).not.toContain('image-secret')
    expect(container.innerHTML).not.toContain('malformed-secret')
    expect(container.innerHTML).not.toContain('invalid-secret')
  })

  it('keeps entity-decoded code delimiters from activating links or images', () => {
    const { container } = render(
      <MarkdownRenderer
        filesDisabled
        content={[
          '`/private/inline-secret.txt &#96; ![track](https://evil.example/pixel)`',
          '',
          '```text',
          '/private/fenced-secret.txt &#96;&#96;&#96;',
          '![track](https://evil.example/fenced-pixel)',
          '```',
        ].join('\n')}
      />,
    )

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(container.innerHTML).not.toContain('/private/')
    expect(container).toHaveTextContent('inline-secret.txt')
    expect(container).toHaveTextContent('fenced-secret.txt')
  })

  it('keeps custom trailing-root file targets inert in structured presentation', () => {
    const { container } = render(
      <JsonRenderer
        filesDisabled
        data={{
          files: [{ file_name: 'report.pdf', output_dir: '/custom/output/' }],
          message: [
            '[open](/custom/output/nested/tenant-secret.pdf)',
            '![image](/custom/output/nested/tenant-secret.png)',
          ].join(' '),
        }}
      />,
    )

    expect(screen.queryByRole('link')).not.toBeInTheDocument()
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
    expect(container).toHaveTextContent('open image')
    expect(container.innerHTML).not.toContain('tenant-secret')
    expect(container.innerHTML).not.toContain('/custom/output')
  })

  it('keeps nested JSON markdown file metadata inert when files are disabled', () => {
    const handleFileClick = vi.fn()
    const data = JSON.stringify({
      goal: 'Create a report',
      output: '[nested report.docx](file:nested-doc-id)',
    })

    const { container } = render(
      <JsonRenderer
        data={data}
        filesDisabled
        onFileClick={handleFileClick}
      />
    )

    expect(screen.getByText('nested report.docx')).not.toHaveAttribute('href')
    expect(screen.queryByTestId('docx-preview')).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(container.innerHTML).not.toContain('nested-doc-id')
    fireEvent.click(screen.getByText('nested report.docx'))
    expect(handleFileClick).not.toHaveBeenCalled()
  })

  it('projects nested file metadata out of JSON DOM while preserving unrelated values', () => {
    const data = {
      requestUrl: 'https://api.example/tasks/42',
      artifacts: [
        {
          file_id: 'nested-file-id',
          id: 'generic-file-record-id',
          file_path: '/private/reports/secret.pdf',
          storage_path: '/workspace/storage/nested-file-id/report.pdf',
          absolute_path: '/sandbox/absolute/report.pdf',
          image_path: '/sandbox/images/report.png',
          local_path: '/sandbox/local/report.pdf',
          output_dir: '/sandbox/output',
          output_path: '/sandbox/output/report.pdf',
          preview_url: 'https://files.example/preview/secret',
          downloadUrl: 'https://files.example/download/secret',
          signed_url: 'https://files.example/signed/secret',
          signedURL: 'https://files.example/signed/acronym-secret',
          url: 'https://files.example/generic-secret-url',
          filename: 'report.pdf',
          mime_type: 'application/pdf',
          text: '[open report](file:nested-file-id)',
        },
        {
          fileId: 'camel-file-id',
          path: '/private/camel.txt',
          storagePath: '/workspace/storage/camel-file-id/camel.txt',
          absolutePath: '/sandbox/camel/absolute.txt',
          imagePath: '/sandbox/camel/image.png',
          localPath: '/sandbox/camel/local.txt',
          outputDir: '/sandbox/camel/output',
          outputPath: '/sandbox/camel/output/camel.txt',
          previewUrl: 'https://files.example/preview/camel',
          previewURL: 'https://files.example/preview/acronym-camel',
          download_url: 'https://files.example/download/camel',
          downloadURL: 'https://files.example/download/acronym-camel',
          fileName: 'camel.txt',
          type: 'text/plain',
        },
      ],
      nested: { message: 'keep this' },
      request: { id: 'unrelated-request-id', url: 'https://api.example/requests/42' },
      rawToolPaths: [
        {
          absolute_path: '/raw/absolute/result.txt',
          text: 'safe absolute result',
        },
        {
          imagePath: '/raw/images/result.png',
          text: 'safe image result',
        },
        {
          outputDir: '/raw/output',
          text: 'safe output result',
        },
      ],
    }

    const { container } = render(<JsonRenderer data={data} filesDisabled />)

    expect(container).toHaveTextContent('report.pdf')
    expect(container).toHaveTextContent('camel.txt')
    expect(container).toHaveTextContent('open report')
    expect(container).toHaveTextContent('https://api.example/tasks/42')
    expect(container).toHaveTextContent('keep this')
    expect(container).toHaveTextContent('unrelated-request-id')
    expect(container).toHaveTextContent('https://api.example/requests/42')
    expect(container).toHaveTextContent('safe absolute result')
    expect(container).toHaveTextContent('safe image result')
    expect(container).toHaveTextContent('safe output result')
    expect(container.innerHTML).not.toContain('nested-file-id')
    expect(container.innerHTML).not.toContain('generic-file-record-id')
    expect(container.innerHTML).not.toContain('/private/reports/secret.pdf')
    expect(container.innerHTML).not.toContain('/workspace/storage/nested-file-id/report.pdf')
    expect(container.innerHTML).not.toContain('/sandbox/absolute/report.pdf')
    expect(container.innerHTML).not.toContain('/sandbox/images/report.png')
    expect(container.innerHTML).not.toContain('/sandbox/local/report.pdf')
    expect(container.innerHTML).not.toContain('/sandbox/output')
    expect(container.innerHTML).not.toContain('https://files.example/preview/secret')
    expect(container.innerHTML).not.toContain('https://files.example/download/secret')
    expect(container.innerHTML).not.toContain('https://files.example/signed/secret')
    expect(container.innerHTML).not.toContain('https://files.example/signed/acronym-secret')
    expect(container.innerHTML).not.toContain('https://files.example/generic-secret-url')
    expect(container.innerHTML).not.toContain('camel-file-id')
    expect(container.innerHTML).not.toContain('/private/camel.txt')
    expect(container.innerHTML).not.toContain('/workspace/storage/camel-file-id/camel.txt')
    expect(container.innerHTML).not.toContain('/sandbox/camel/absolute.txt')
    expect(container.innerHTML).not.toContain('/sandbox/camel/image.png')
    expect(container.innerHTML).not.toContain('/sandbox/camel/local.txt')
    expect(container.innerHTML).not.toContain('/sandbox/camel/output')
    expect(container.innerHTML).not.toContain('/raw/absolute/result.txt')
    expect(container.innerHTML).not.toContain('/raw/images/result.png')
    expect(container.innerHTML).not.toContain('/raw/output')
    expect(container.innerHTML).not.toContain('https://files.example/preview/camel')
    expect(container.innerHTML).not.toContain('https://files.example/preview/acronym-camel')
    expect(container.innerHTML).not.toContain('https://files.example/download/camel')
    expect(container.innerHTML).not.toContain('https://files.example/download/acronym-camel')
  })

  it('renders pptx file links as inline previews', async () => {
    const content = '[example_presentation.pptx](file:99fb81ab-b995-4976-be18-21b02f748768)'
    render(<MarkdownRenderer content={content} />)

    // Managed fileId path: mount PptxPreviewRenderer immediately and let it
    // probe the PDF endpoint first instead of eagerly downloading raw bytes.
    expect(await screen.findByTestId('pptx-preview')).toHaveTextContent(
      '99fb81ab-b995-4976-be18-21b02f748768'
    )
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      'http://api.local/api/files/public/preview/99fb81ab-b995-4976-be18-21b02f748768',
      expect.anything()
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

  it('preserves standard relative markdown links and images, and does not open them in a new tab by default', () => {
    const content = '[relative doc](../doc.md)\n\n![relative image](./a.png)'
    render(<MarkdownRenderer content={content} />)

    const link = screen.getByText('relative doc')
    expect(link).toBeInTheDocument()
    expect(link).toHaveAttribute('href', '../doc.md')
    expect(link).not.toHaveAttribute('target')
    expect(link).not.toHaveAttribute('rel')

    const image = screen.getByAltText('relative image')
    expect(image).toBeInTheDocument()
    expect(image).toHaveAttribute('src', './a.png')
  })

  it('opens ordinary links in a new tab only when the LinksOpenInNewTab capability is enabled', () => {
    render(
      <LinksOpenInNewTabCapability.Provider value={true}>
        <MarkdownRenderer content="[Google](https://www.google.com)" />
      </LinksOpenInNewTabCapability.Provider>,
    )

    const link = screen.getByRole('link', { name: 'Google' })
    expect(link).toHaveAttribute('target', '_blank')
    expect(link).toHaveAttribute('rel', 'noopener noreferrer')
  })

  it('keeps in-page anchor links and mailto links in the current tab even when the capability is enabled', () => {
    const content = [
      '[Jump to section](#section)',
      '[Email us](mailto:hello@example.com)',
    ].join('\n\n')
    render(
      <LinksOpenInNewTabCapability.Provider value={true}>
        <MarkdownRenderer content={content} />
      </LinksOpenInNewTabCapability.Provider>,
    )

    for (const name of ['Jump to section', 'Email us']) {
      const link = screen.getByRole('link', { name })
      expect(link).not.toHaveAttribute('target')
      expect(link).not.toHaveAttribute('rel')
    }
  })

  it('neutralizes javascript: hrefs before the new-tab check ever sees them', () => {
    render(<MarkdownRenderer content="[click me](javascript:alert(1))" />)

    const link = screen.getByText('click me')
    expect(link).not.toHaveAttribute('href')
    expect(link).not.toHaveAttribute('target')
    expect(link).not.toHaveAttribute('rel')
  })

  it('does not add target/rel to a link with an empty href', () => {
    render(<MarkdownRenderer content="[x]()" />)

    const link = screen.getByText('x')
    expect(link).not.toHaveAttribute('href')
    expect(link).not.toHaveAttribute('target')
    expect(link).not.toHaveAttribute('rel')
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

  it('corrects legacy-path image markdown for mp3 files to a playable audio blob', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['audio-bytes'], { type: 'audio/mpeg' }),
    })
    const content =
      '![xagent_061_podcast.mp3](file:output/xagent_061_podcast.mp3)'
    render(<MarkdownRenderer content={content} />)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/output%2Fxagent_061_podcast.mp3',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })
    const audio = await screen.findByLabelText('xagent_061_podcast.mp3')
    expect(audio.getAttribute('src')).toMatch(/^blob:/)
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('href')).toMatch(
      /^blob:/
    )
    expect(screen.queryByRole('img')).not.toBeInTheDocument()
  })

  it('keeps a playing audio element mounted when surrounding page callbacks update', async () => {
    // This test is about DOM node identity across rerenders, not about
    // streaming-ticket mechanics -- opt out of the mint attempt explicitly
    // (rather than relying on the mocked response lacking .json(), which
    // would make the mint throw internally and fall through to the blob
    // fetch for an incidental reason unrelated to what this test verifies).
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['audio-bytes'], { type: 'audio/mpeg' }),
    })
    const content = '![podcast.mp3](file:output/podcast.mp3)'
    // Stable across both renders: a fresh object literal per render would
    // change useResolvedMediaUrl's fileAccess dependency identity and
    // reset it to its loading state, unmounting/remounting the <audio>
    // element for a reason unrelated to what this test verifies.
    const testPolicy = { ...defaultFileAccessPolicy, getStreamingUrl: undefined }
    const { rerender } = render(
      <FileAccessProvider policy={testPolicy}>
        <MarkdownRenderer content={content} onFileClick={vi.fn()} />
      </FileAccessProvider>
    )

    const audioBeforeUpdate = await screen.findByLabelText('podcast.mp3')
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(1))

    // Trace and task events update the surrounding chat message and create new
    // callback props. The media DOM node must survive that rerender so the
    // browser keeps its currentTime and playing state.
    rerender(
      <FileAccessProvider policy={testPolicy}>
        <MarkdownRenderer content={content} onFileClick={vi.fn()} />
      </FileAccessProvider>
    )

    expect(await screen.findByLabelText('podcast.mp3')).toBe(audioBeforeUpdate)
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it('prefers link label over generic file id when determining preview kind', async () => {
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

  it('renders video file links as an inline video preview', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })
    const content = '[puppy_drinking.mp4](file:550e8400-e29b-41d4-a716-446655440000)'
    render(<MarkdownRenderer content={content} />)

    const video = await screen.findByLabelText('puppy_drinking.mp4')
    expect(video.tagName.toLowerCase()).toBe('video')
    await waitFor(() => {
      expect(video.getAttribute('src')).toMatch(/^blob:/)
    })
  })

  it('renders image-syntax video references as an inline video preview', async () => {
    // The image renderer resolves the video kind from alt/title text
    // instead of falling back to a broken <img> for ![...] refs pointing
    // at media files.
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })
    const content = '![puppy_drinking.mp4](file:550e8400-e29b-41d4-a716-446655440000)'
    render(<MarkdownRenderer content={content} />)

    const video = await screen.findByLabelText('puppy_drinking.mp4')
    expect(video.tagName.toLowerCase()).toBe('video')
  })

  it('detects video via the link title while displaying the model label', async () => {
    // The backend (file_reference_output_service.reconcile_assistant_file_
    // references) adds the real filename as the link *title* rather than
    // overwriting the label, so a localized label survives while the
    // frontend still detects the media type from the title.
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })
    const content =
      '[下载视频（MP4）](file:550e8400-e29b-41d4-a716-446655440000 "puppy_drinking.mp4")'
    render(<MarkdownRenderer content={content} />)

    const video = await screen.findByLabelText('下载视频（MP4）')
    expect(video.tagName.toLowerCase()).toBe('video')
    expect(screen.queryByLabelText('puppy_drinking.mp4')).not.toBeInTheDocument()
    expect(screen.getByText('下载视频（MP4）')).toBeInTheDocument()
  })

  it('falls back to the label for detection when a pre-existing title does not reveal the type', async () => {
    // The backend only injects/overwrites a title when the label doesn't
    // already reveal the media type (label_reveals_type in
    // file_reference_output_service.py) -- so it leaves this exact
    // combination (a label that already says .mp4, alongside an
    // unrelated, non-descriptive pre-existing title) untouched. Detection
    // must mirror that: falling back to the label when the title alone
    // doesn't classify, instead of trusting a title that was never meant
    // to be a detection hint in the first place.
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })
    const content =
      '[report.mp4](file:550e8400-e29b-41d4-a716-446655440000 "See the notes")'
    render(<MarkdownRenderer content={content} />)

    const video = await screen.findByLabelText('report.mp4')
    expect(video.tagName.toLowerCase()).toBe('video')
  })

  it('detects video via the image title while displaying the model alt text', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })
    const content =
      '![预览视频](file:550e8400-e29b-41d4-a716-446655440000 "puppy_drinking.mp4")'
    render(<MarkdownRenderer content={content} />)

    const video = await screen.findByLabelText('预览视频')
    expect(video.tagName.toLowerCase()).toBe('video')
  })

  it('shows the model alt text, not the backend-injected title, when files are disabled', () => {
    // This exercises MarkdownRenderer's tree-level pre-sanitizer
    // (sanitizeFilesDisabledPresentationText), NOT MarkdownImage's own
    // internal `if (filesDisabled) return <span>...</span>` branch for a
    // `file:` src: the pre-sanitizer converts a `![alt](file:... "title")`
    // node to plain text before ReactMarkdown ever builds an <img> node, so
    // that branch is unreachable through this entry point and this test
    // cannot exercise it (confirmed: reverting that branch's fix does not
    // change this test's outcome). It's still real, useful coverage of a
    // real code path -- the pre-sanitizer must itself prefer alt over
    // title when converting the node to a label -- just not what its
    // original name/comment claimed to guard. The component-level branch
    // is kept as defense-in-depth (see its own comment in
    // markdown-renderer.tsx) since falling through to InlineFilePreview
    // would render an interactive preview despite files being disabled.
    render(
      <MarkdownRenderer
        filesDisabled
        content='![预览视频](file:550e8400-e29b-41d4-a716-446655440000 "generated_video.mp4")'
      />,
    )

    expect(screen.getByText('预览视频')).toBeInTheDocument()
    expect(screen.queryByText('generated_video.mp4')).not.toBeInTheDocument()
  })

  it('displays the model label, not the title, for a non-media titled file link', async () => {
    // The title is only ever a media-type detection hint; even once it
    // wins detection (the label here reveals no type of its own), the
    // model's own label must still be what's shown as the previewed
    // file's name, not the title. Uses a .docx title rather than .pdf
    // deliberately: .pdf is never previewable via either title or label,
    // so a test using it can't distinguish "display correctly stayed
    // label-first" from "nothing preview-related could ever happen here
    // regardless of precedence" -- this needs a title that genuinely wins
    // detection (mounting the real office-preview header) to be
    // load-bearing against a title-first display regression.
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })
    const content = '[下载报告](file:doc-file-id "annual_report_2024.docx")'
    render(<MarkdownRenderer content={content} />)

    expect(await screen.findByTestId('docx-preview')).toBeInTheDocument()
    expect(screen.getByText('下载报告')).toBeInTheDocument()
    expect(screen.queryByText('annual_report_2024.docx')).not.toBeInTheDocument()
  })

  it('lets a title that classifies differently than the label win detection', async () => {
    // resolvePreviewableFileLink tries title first and stops at the first
    // candidate that classifies to any previewable kind -- it does not
    // compare title's and visibleText's kinds and prefer whichever the
    // label reveals. The backend's own gate never produces this shape (it
    // only writes a title when the label doesn't already reveal a type),
    // so this only arises for a hand-authored or otherwise unusual
    // reference; pinning it here documents the actual (title-wins)
    // behavior rather than the label-aware guarantee the code comment used
    // to (incorrectly) claim.
    apiRequestMock.mockResolvedValue({
      ok: true,
      arrayBuffer: async () => new Uint8Array([65, 66]).buffer,
    })
    const content = '[report.mp4](file:doc-file-id "notes.docx")'

    render(<MarkdownRenderer content={content} />)

    expect(await screen.findByTestId('docx-preview')).toBeInTheDocument()
    expect(screen.queryByLabelText('report.mp4')).not.toBeInTheDocument()
  })

  it('renders file links as image previews when the path has an image extension', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })
    const content = '[LinkedIn visual](file:550e8400-e29b-41d4-a716-446655440000/linkedin.png)'
    render(<MarkdownRenderer content={content} />)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/550e8400-e29b-41d4-a716-446655440000',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const image = screen.getByAltText('LinkedIn visual')
    await waitFor(() => {
      expect(image.getAttribute('src')).toMatch(/^blob:/)
    })
  })

  it('uses authenticated preview fallback for uuid file: images', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
    })
    const content = '![uuid image](file:550e8400-e29b-41d4-a716-446655440000)'
    render(<MarkdownRenderer content={content} />)

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/550e8400-e29b-41d4-a716-446655440000',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const image = screen.getByAltText('uuid image')
    await waitFor(() => {
      expect(image.getAttribute('src')).toMatch(/^blob:/)
    })
  })
})
