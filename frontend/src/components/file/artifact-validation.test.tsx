/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { ArtifactValidation } from './artifact-validation'
import { FileAccessProvider, defaultFileAccessPolicy, createPublicFileAccessPolicy } from '@/contexts/file-access-context'
import { I18nProvider } from '@/contexts/i18n-context'
import { InlineFilePreview } from './inline-file-preview'

const reportHeaders = { 'content-type': 'application/vnd.xagent.validation+json' }
const report = (status: string, message = '', supported = true) => new Response(JSON.stringify({ status, supported, sha256: '0'.repeat(64), checks: [{ status, message }] }), { headers: reportHeaders })
const wrap = (children: React.ReactNode, request = vi.fn(), extra = {}) => <I18nProvider>
  <FileAccessProvider policy={{ ...defaultFileAccessPolicy, request, ...extra }}>{children}</FileAccessProvider>
</I18nProvider>

afterEach(() => { cleanup(); vi.unstubAllGlobals(); vi.useRealTimers(); localStorage.removeItem('app_locale') })

describe('ArtifactValidation', () => {
  it.each(['valid', 'invalid', 'unchecked'] as const)('shows %s without hiding repair/download access', async status => {
    const request = vi.fn().mockResolvedValue(report(status, status === 'invalid' ? 'Corrupt package' : ''))
    const { container } = render(wrap(<ArtifactValidation fileId="one"><a href="/download">artifact</a></ArtifactValidation>, request))
    expect(container.querySelector('[data-artifact-validation="checking"]')).toBeTruthy()
    await waitFor(() => expect(container.querySelector(`[data-artifact-validation="${status}"]`)).toBeTruthy())
    expect(screen.getByRole('link', { name: 'artifact' })).toHaveAttribute('href', '/download')
    expect(request).toHaveBeenCalledWith('/api/files/preview/one?validation_only=true', expect.objectContaining({ cache: 'no-store', signal: expect.any(AbortSignal) }))
  })

  it('rechecks the same file id after repair without reusing a cached pass', async () => {
    const request = vi.fn().mockResolvedValueOnce(report('valid')).mockResolvedValueOnce(report('invalid', 'broken'))
    const { container } = render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, request))
    await screen.findByText('Format readable · content not verified')
    fireEvent.click(screen.getByRole('button', { name: 'Recheck' }))
    expect(container.querySelector('[data-artifact-validation="checking"]')).toBeTruthy()
    await screen.findByText('File validation failed · repair required')
    expect(request).toHaveBeenCalledTimes(2)
  })

  it.each([
    new Response('', { status: 403 }), new Response('', { status: 500 }), report('made-up'), new Response('not json'),
    new Response('{"status":"valid"}', { headers: reportHeaders }),
    new Response('{"status":"valid","checks":[{"status":"valid"}]}', { headers: reportHeaders }),
    new Response(JSON.stringify({ status: 'valid', sha256: '0'.repeat(64), checks: [{ status: 'unchecked' }] }), { headers: reportHeaders }),
    new Response(JSON.stringify({ status: 'valid', sha256: '0'.repeat(64), checks: [{ status: 'valid' }] }), { headers: { 'content-type': 'application/json' } }),
  ])('does not treat a failed/malformed request as a pass', async response => {
    render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, vi.fn().mockResolvedValue(response)))
    await screen.findByText('Unable to request file validation · try again')
    expect(screen.queryByText('File not checked')).toBeNull()
  })

  it('ignores an old response after the file changes', async () => {
    let finish!: (response: Response) => void
    const request = vi.fn().mockImplementationOnce(() => new Promise(resolve => { finish = resolve })).mockResolvedValueOnce(report('invalid'))
    const policy = { ...defaultFileAccessPolicy, request }
    const view = (fileId: string) => <I18nProvider><FileAccessProvider policy={policy}><ArtifactValidation fileId={fileId}>file</ArtifactValidation></FileAccessProvider></I18nProvider>
    const { rerender } = render(view('one'))
    rerender(view('two'))
    await screen.findByText('File validation failed · repair required')
    await act(async () => { finish(report('valid')) })
    expect(screen.queryByText('Format readable · content not verified')).toBeNull()
    expect(request.mock.calls[0][1].signal.aborted).toBe(true)
  })

  it('does not request validation for policies without the capability', () => {
    const request = vi.fn()
    render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, request, { validationUrl: undefined }))
    expect(request).not.toHaveBeenCalled()
    expect(screen.queryByRole('status')).toBeNull()
  })

  it('uses the public scoped token and strips ambient authorization', async () => {
    const fetchMock = vi.fn().mockResolvedValue(report('valid'))
    vi.stubGlobal('fetch', fetchMock)
    const policy = createPublicFileAccessPolicy('guest-token')
    render(<I18nProvider><FileAccessProvider policy={policy}><ArtifactValidation fileId="one">file</ArtifactValidation></FileAccessProvider></I18nProvider>)
    await screen.findByText('Format readable · content not verified')
    expect(fetchMock.mock.calls[0][0]).toContain('token=guest-token&validation_only=true')
    expect(fetchMock.mock.calls[0][1].headers.has('Authorization')).toBe(false)
  })

  it('hides unsupported-format controls using the server capability, retaining the file', async () => {
    const request = vi.fn().mockResolvedValue(report('unchecked', 'No validator is installed for this format.', false))
    render(wrap(<InlineFilePreview source={{ fileId: '123e4567-e89b-12d3-a456-426614174000', filename: 'notes.txt' }} />, request))
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull())
    expect(request).toHaveBeenCalledTimes(1)
    expect(screen.queryByRole('button', { name: 'Recheck' })).toBeNull()
    expect(screen.getByRole('link')).toHaveTextContent('notes.txt')
  })

  it('distinguishes a network error and permits a successful retry', async () => {
    const request = vi.fn().mockRejectedValueOnce(new TypeError('offline')).mockResolvedValueOnce(report('valid'))
    render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, request))
    await screen.findByText('Unable to request file validation · try again')
    fireEvent.click(screen.getByRole('button', { name: 'Recheck' }))
    await screen.findByText('Format readable · content not verified')
  })

  it('reports a client timeout as a request error, not a completed unchecked report', async () => {
    vi.useFakeTimers()
    const request = vi.fn().mockImplementation((_url, options) => new Promise((_resolve, reject) => {
      options.signal.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')))
    }))
    render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, request))
    await act(async () => { await vi.advanceTimersByTimeAsync(20_001) })
    expect(screen.getByText('Unable to request file validation · try again')).toBeInTheDocument()
    expect(request.mock.calls[0][1].signal.aborted).toBe(true)
  })

  it('uses translated machine-status labels instead of raw English parser diagnostics', async () => {
    localStorage.setItem('app_locale', 'zh')
    const request = vi.fn().mockResolvedValue(report('invalid', 'PDF header is missing.'))
    render(wrap(<ArtifactValidation fileId="one">file</ArtifactValidation>, request))
    await screen.findByText('文件校验失败 · 需要修复')
    expect(screen.queryByText('PDF header is missing.')).toBeNull()
    expect(screen.getByRole('button', { name: '重新检查' })).toBeInTheDocument()
  })
})
