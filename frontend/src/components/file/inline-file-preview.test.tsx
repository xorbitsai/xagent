/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react'
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock('@/components/ui/sonner', () => ({
  toast: { error: toastErrorMock },
}))

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

import {
  InlineFilePreview,
  __mintStreamingUrlForTests,
  __resetStreamingUrlCacheForTests,
} from './inline-file-preview'
import {
  FileAccessProvider,
  createPublicFileAccessPolicy,
  defaultFileAccessPolicy,
} from '@/contexts/file-access-context'

describe('InlineFilePreview', () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    // Several cases below reuse the same fileId; without this, an earlier
    // case's minted (or attempted) streaming ticket would leak into a
    // later case that expects its own mock to be exercised.
    __resetStreamingUrlCacheForTests()
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
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

  it('uses the provider-scoped public credential for image and audio elements', async () => {
    // With the public policy the tokened URL is handed to the media element
    // directly — no blob fetch is made (preserving HTTP range requests for
    // media playback) and the authenticated apiRequest path is never touched.
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    render(
      <FileAccessProvider policy={createPublicFileAccessPolicy('guest-a')}>
        <>
          <InlineFilePreview
            source={{ type: 'image', fileId: 'image-id', filename: 'image.png' }}
          />
          <InlineFilePreview
            source={{ type: 'audio', fileId: 'audio-id', filename: 'audio.mp3' }}
          />
        </>
      </FileAccessProvider>,
    )

    const image = await screen.findByAltText('image.png')
    expect(image.getAttribute('src')).toBe(
      'http://api.local/api/files/public/preview/image-id?token=guest-a',
    )
    const audio = await screen.findByTitle('audio.mp3')
    expect(audio.tagName).toBe('AUDIO')
    expect(audio.getAttribute('src')).toBe(
      'http://api.local/api/files/public/preview/audio-id?token=guest-a',
    )
    expect(fetchMock).not.toHaveBeenCalled()
    expect(apiRequestMock).not.toHaveBeenCalled()
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

  it('loads managed video files through authenticated preview', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await waitFor(() => {
      expect(apiRequestMock).toHaveBeenCalledWith(
        'http://api.local/api/files/preview/video-file-id',
        expect.objectContaining({ cache: 'no-cache' })
      )
    })

    const video = await screen.findByLabelText('clip.mp4')
    expect(video.tagName.toLowerCase()).toBe('video')
    expect(video.getAttribute('src')).toMatch(/^blob:/)
    expect(screen.getByRole('link', { name: 'Open' }).getAttribute('href')).toMatch(
      /^blob:/
    )
  })

  it('streams video directly from the tokened URL under the public policy', async () => {
    // Range requests only work when the media element loads the URL itself;
    // a blob fetch would force the full download before playback starts.
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)

    render(
      <FileAccessProvider policy={createPublicFileAccessPolicy('guest-a')}>
        <InlineFilePreview
          source={{ type: 'video', fileId: 'video-id', filename: 'clip.mp4' }}
        />
      </FileAccessProvider>,
    )

    const video = await screen.findByLabelText('clip.mp4')
    expect(video.tagName.toLowerCase()).toBe('video')
    expect(video.getAttribute('src')).toBe(
      'http://api.local/api/files/public/preview/video-id?token=guest-a',
    )
    expect(fetchMock).not.toHaveBeenCalled()
    expect(apiRequestMock).not.toHaveBeenCalled()
  })

  it('lets a click on "Open" navigate natively under the public policy instead of the popup-blocker-dependent tab dance', async () => {
    // Regression test (F2): the public policy never sets isStreamingUrl
    // (it has no getStreamingUrl), so openUrl is already the safe,
    // already-tokened direct URL -- opening a blank tab and redirecting it
    // via JS exists only to smuggle a fresh URL past a 403-prone static
    // href while streaming, which doesn't apply here. Hijacking this click
    // anyway would make "Open" popup-blocker-dependent for no reason.
    vi.stubGlobal('fetch', vi.fn())
    const openSpy = vi.spyOn(window, 'open')

    render(
      <FileAccessProvider policy={createPublicFileAccessPolicy('guest-a')}>
        <InlineFilePreview
          source={{ type: 'video', fileId: 'video-id', filename: 'clip.mp4' }}
        />
      </FileAccessProvider>,
    )
    await screen.findByLabelText('clip.mp4')

    const clickEvent = new MouseEvent('click', { bubbles: true, cancelable: true })
    fireEvent(screen.getByRole('link', { name: 'Open' }), clickEvent)

    expect(openSpy).not.toHaveBeenCalled()
    expect(clickEvent.defaultPrevented).toBe(false)

    openSpy.mockRestore()
  })

  it('streams video directly from a minted ticket under the default policy', async () => {
    // The default policy's getStreamingUrl mints a ticket via a dedicated
    // endpoint; when that succeeds the media element loads the ticketed
    // URL directly instead of blob-fetching the whole file.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    expect(video.getAttribute('src')).toBe(
      'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket'
    )
    expect(apiRequestMock).toHaveBeenCalledWith(
      'http://api.local/api/files/stream-tickets/video-file-id',
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it('keeps the public preview URL as the static href when there is no click handler', async () => {
    // The static href stays the credential-free public URL (never the
    // ticketed one, which is a replayable credential that must never be
    // exposed via a link a user can put in the address bar, browser
    // history, or copy/paste) -- this is the middle-click/right-click
    // "open in new tab" escape hatch. The next test covers what actually
    // happens on a primary click, where this URL would 403 on an
    // access-controlled task.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await screen.findByLabelText('clip.mp4')
    expect(screen.getByRole('link', { name: 'Open' })).toHaveAttribute(
      'href',
      'http://api.local/api/files/public/preview/video-file-id'
    )
  })

  it('resolves a fresh authenticated blob for "Open" on a primary click when there is no dialog and streaming is active', async () => {
    // Regression test (F2): the public preview URL used as the static href
    // above 403s ("Public file access token required") on an
    // access-controlled task for surfaces with no in-app dialog (a
    // read-only transcript/log viewer, the skill-hub docs viewer, etc.),
    // and pre-PR the Open href was the blob URL and worked everywhere. A
    // primary click must resolve to a URL that will actually load instead
    // of navigating to that 403-prone one.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })
    const fakeTab = { location: { href: '' }, closed: false, opener: 'set-by-code' as unknown }
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(fakeTab as unknown as Window)

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await screen.findByLabelText('clip.mp4')
    fireEvent.click(screen.getByRole('link', { name: 'Open' }))

    expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank')
    expect(fakeTab.opener).toBeNull()
    await waitFor(() => {
      expect(fakeTab.location.href).toMatch(/^blob:/)
    })

    openSpy.mockRestore()
  })

  it('falls back to the public preview URL for "Open" when the on-demand blob fetch fails', async () => {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return { ok: false, status: 401 }
    })
    const fakeTab = { location: { href: '' }, closed: false, opener: 'set-by-code' as unknown }
    vi.spyOn(window, 'open').mockReturnValue(fakeTab as unknown as Window)

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await screen.findByLabelText('clip.mp4')
    fireEvent.click(screen.getByRole('link', { name: 'Open' }))

    await waitFor(() => {
      expect(fakeTab.location.href).toBe(
        'http://api.local/api/files/public/preview/video-file-id'
      )
    })

    vi.restoreAllMocks()
  })

  it('keeps every "Open"-minted blob alive across repeated clicks, revoking only on unmount', async () => {
    // Regression test (N-AP9): a previous version revoked the prior
    // Open-tab's blob URL on the *next* click. A still-open first tab's
    // media keeps issuing Range requests against its own blob URL as the
    // user seeks/reloads it, so revoking it out from under that tab broke
    // playback there the moment a second "Open" click happened (e.g. on a
    // different attachment's tab, or the same one again).
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })
    const revokeSpy = vi.spyOn(URL, 'revokeObjectURL')
    const fakeTab1 = { location: { href: '' }, closed: false, opener: 'x' as unknown }
    const fakeTab2 = { location: { href: '' }, closed: false, opener: 'x' as unknown }
    const openSpy = vi
      .spyOn(window, 'open')
      .mockReturnValueOnce(fakeTab1 as unknown as Window)
      .mockReturnValueOnce(fakeTab2 as unknown as Window)

    const { unmount } = render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )
    await screen.findByLabelText('clip.mp4')
    const openLink = screen.getByRole('link', { name: 'Open' })

    fireEvent.click(openLink)
    await waitFor(() => expect(fakeTab1.location.href).toMatch(/^blob:/))
    fireEvent.click(openLink)
    await waitFor(() => expect(fakeTab2.location.href).toMatch(/^blob:/))

    // Neither blob revoked yet -- both fake tabs are still "open".
    expect(revokeSpy).not.toHaveBeenCalled()

    unmount()

    expect(revokeSpy).toHaveBeenCalledTimes(2)
    expect(revokeSpy).toHaveBeenCalledWith(fakeTab1.location.href)
    expect(revokeSpy).toHaveBeenCalledWith(fakeTab2.location.href)

    openSpy.mockRestore()
    revokeSpy.mockRestore()
  })

  it('resolves a fresh authenticated blob for "Open" on a middle click when there is no dialog and streaming is active', async () => {
    // Regression test (N-D1): middle-click/ctrl-click fire a native
    // 'auxclick' event that bypasses onClick entirely, so without an
    // onAuxClick handler wired to the same recovery path, a middle click
    // would navigate straight to the static href -- the 403-prone public
    // preview URL while streaming is active -- even though a primary click
    // on the exact same link works.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })
    const fakeTab = { location: { href: '' }, closed: false, opener: 'set-by-code' as unknown }
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(fakeTab as unknown as Window)

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await screen.findByLabelText('clip.mp4')
    // testing-library 10.4's built-in fireEvent map has no auxClick helper
    // (added later), so the native event is dispatched directly -- a
    // middle-click (button 1) is exactly what fires 'auxclick'.
    fireEvent(
      screen.getByRole('link', { name: 'Open' }),
      new MouseEvent('auxclick', { bubbles: true, cancelable: true, button: 1 })
    )

    expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank')
    expect(fakeTab.opener).toBeNull()
    await waitFor(() => {
      expect(fakeTab.location.href).toMatch(/^blob:/)
    })

    openSpy.mockRestore()
  })

  it('leaves a right-click on "Open" alone instead of hijacking the context menu', async () => {
    // auxclick fires for ANY non-primary button -- right-click (button 2)
    // included. The middle-click recovery above must not swallow it: the
    // user wants the browser context menu, not a surprise tab.
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        path: '/api/files/preview/video-file-id?ticket=signed-ticket',
      }),
    })
    const openSpy = vi.spyOn(window, 'open')

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )
    await screen.findByLabelText('clip.mp4')

    const rightClick = new MouseEvent('auxclick', {
      bubbles: true,
      cancelable: true,
      button: 2,
    })
    fireEvent(screen.getByRole('link', { name: 'Open' }), rightClick)

    expect(openSpy).not.toHaveBeenCalled()
    // preventDefault must not have run either -- that's what would suppress
    // the browser's own handling around the gesture.
    expect(rightClick.defaultPrevented).toBe(false)

    openSpy.mockRestore()
  })

  it('notifies instead of resolving a URL when the "Open" popup is blocked', async () => {
    // Regression test (N-AP8): window.open returning null (popup blocker,
    // some webviews) used to be silently swallowed -- preventDefault had
    // already run, so nothing loaded and the user got no feedback at all.
    apiRequestMock.mockResolvedValue({
      ok: true,
      json: async () => ({
        path: '/api/files/preview/video-file-id?ticket=signed-ticket',
      }),
    })
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(null)

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await screen.findByLabelText('clip.mp4')
    fireEvent.click(screen.getByRole('link', { name: 'Open' }))

    expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank')
    expect(toastErrorMock).toHaveBeenCalled()
    // The blob/mint fetch must never even be attempted once there's no tab
    // left to redirect -- only the initial ticket mint call happened.
    expect(apiRequestMock).toHaveBeenCalledTimes(1)

    openSpy.mockRestore()
  })

  it('routes "Open" through onFileClick instead of the public preview URL when a dialog is available', async () => {
    // For the default in-app policy, the credential-free fallback above is
    // the tokenless public preview route, which 403s ("Public file access
    // token required") for any task with access control configured --
    // every standard agent chat. All in-app chat surfaces pass onFileClick,
    // so "Open" must route through it (same as the image/generic-file
    // branches of InlineFilePreview already do) instead of ever landing on
    // that URL.
    const handleFileClick = vi.fn()
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
        onFileClick={handleFileClick}
      />
    )

    await screen.findByLabelText('clip.mp4')
    fireEvent.click(screen.getByRole('link', { name: 'Open' }))

    expect(handleFileClick).toHaveBeenCalledWith('video-file-id', 'clip.mp4')
  })

  it('resolves a fresh authenticated blob for "Open" on a middle click even when a dialog is available', async () => {
    // Regression test (N-D1 residual): a dialog answers "open in-app" for a
    // primary click, but a middle-click's intent is always "open in a new
    // tab" -- onFileClick has no answer for that. Without onAuxClick wired
    // on this branch too, a middle click fell through to the native href
    // (the 403-prone public preview URL while streaming) on every standard
    // in-app chat surface, even though the exact same primary click worked.
    const handleFileClick = vi.fn()
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })
    const fakeTab = { location: { href: '' }, closed: false, opener: 'x' as unknown }
    const openSpy = vi.spyOn(window, 'open').mockReturnValue(fakeTab as unknown as Window)

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
        onFileClick={handleFileClick}
      />
    )

    await screen.findByLabelText('clip.mp4')
    fireEvent(
      screen.getByRole('link', { name: 'Open' }),
      new MouseEvent('auxclick', { bubbles: true, cancelable: true, button: 1 })
    )

    expect(openSpy).toHaveBeenCalledWith('about:blank', '_blank')
    expect(handleFileClick).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(fakeTab.location.href).toMatch(/^blob:/)
    })

    openSpy.mockRestore()
  })

  it('reuses a minted streaming ticket across remounts of the same file', async () => {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    const { unmount } = render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )
    await screen.findByLabelText('clip.mp4')
    unmount()

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )
    await screen.findByLabelText('clip.mp4')

    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it('shares one in-flight mint across concurrent mounts of the same file', async () => {
    // The same attachment can render twice at once (e.g. the same file
    // referenced from two visible messages, or a trace-event artifact
    // duplicated into the transcript). Both mounts must share one mint
    // round trip rather than each paying their own.
    let resolveMint: ((value: unknown) => void) | undefined
    apiRequestMock.mockImplementation((url: string) => {
      if (url.includes('/stream-tickets/')) {
        return new Promise((resolve) => {
          resolveMint = resolve
        })
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <>
        <InlineFilePreview
          source={{ type: 'video', fileId: 'shared-file-id', filename: 'clip.mp4' }}
        />
        <InlineFilePreview
          source={{ type: 'video', fileId: 'shared-file-id', filename: 'clip.mp4' }}
        />
      </>
    )

    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(1))

    resolveMint?.({
      ok: true,
      json: async () => ({
        path: '/api/files/preview/shared-file-id?ticket=signed-ticket',
      }),
    })

    await waitFor(() => {
      expect(screen.getAllByLabelText('clip.mp4')).toHaveLength(2)
    })
    for (const video of screen.getAllByLabelText('clip.mp4')) {
      expect(video.getAttribute('src')).toBe(
        'http://api.local/api/files/preview/shared-file-id?ticket=signed-ticket'
      )
    }
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it('retries minting on the next mount after a failed mint instead of reusing the rejection', async () => {
    let callCount = 0
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        callCount += 1
        if (callCount === 1) return { ok: false, status: 500 }
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/retry-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    const { unmount } = render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'retry-file-id', filename: 'clip.mp4' }}
      />
    )
    // First mount: mint fails, falls back to the direct public URL (the
    // default policy's requiresBlobFetch is true, but blob-fetch itself
    // isn't mocked as reachable here, so this assertion only needs the
    // player to end up off the streaming path -- confirmed by the second
    // mount actually re-minting below rather than reusing a cached failure).
    await waitFor(() => expect(callCount).toBe(1))
    unmount()

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'retry-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    expect(video.getAttribute('src')).toBe(
      'http://api.local/api/files/preview/retry-file-id?ticket=signed-ticket'
    )
    expect(callCount).toBe(2)
  })

  it('does not mint a redundant ticket per waiter when several concurrent callers race on one failed mint', async () => {
    // Regression test (N-AP12): only the caller that originated a cached
    // mint promise ever notices *that* promise's own failure and evicts it
    // (in its own try/catch, a different code path from this test's
    // target). Every OTHER concurrent waiter shares that failed promise
    // and used to independently mint its own replacement instead of
    // reusing whichever replacement won first -- N waiters on one failed
    // mint paid N-1 redundant tickets. Exercises mintStreamingUrl's
    // dedup/retry logic directly (via the test-only export) for precise
    // control over the race, rather than through React's own effect
    // timing across several mounted components.
    let mintCallCount = 0
    let rejectFirstMint: ((error: Error) => void) | undefined
    const fileAccess = {
      ...defaultFileAccessPolicy,
      getStreamingUrl: (): Promise<string> => {
        mintCallCount += 1
        if (mintCallCount === 1) {
          return new Promise((_resolve, reject) => {
            rejectFirstMint = reject
          })
        }
        return Promise.resolve(`https://example.test/ticket-${mintCallCount}`)
      },
    }

    const results = Promise.all([
      __mintStreamingUrlForTests(fileAccess, 'shared-file-id').catch((error) => error),
      __mintStreamingUrlForTests(fileAccess, 'shared-file-id').catch((error) => error),
      __mintStreamingUrlForTests(fileAccess, 'shared-file-id').catch((error) => error),
    ])

    await waitFor(() => expect(mintCallCount).toBe(1))
    rejectFirstMint?.(new Error('mint failed'))

    const [first, second, third] = await results

    // The originating call's own promise always rejects -- it never
    // retries itself, only evicts. Exactly one shared retry mint must
    // cover both other waiters (mintCallCount === 2, not 3).
    expect(first).toBeInstanceOf(Error)
    expect(second).toBe('https://example.test/ticket-2')
    expect(third).toBe('https://example.test/ticket-2')
    expect(mintCallCount).toBe(2)
  })

  it('falls back to blob fetch when a minted ticket fails to actually load', async () => {
    // Minting never checks file ownership -- only redemption does -- so a
    // ticket can mint successfully for a file the caller can't actually
    // read; the media element's own fetch then fails at redemption, and
    // playback must still recover via the blob path rather than being
    // left on a dead ticketed src.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const streamedVideo = await screen.findByLabelText('clip.mp4')
    expect(streamedVideo.getAttribute('src')).toBe(
      'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket'
    )

    fireEvent.error(streamedVideo)

    await waitFor(() => {
      expect(screen.getByLabelText('clip.mp4').getAttribute('src')).toMatch(/^blob:/)
    })
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()
  })

  it('recovers via a fresh ticket when one expires mid-playback, resuming near the same currentTime', async () => {
    // Regression test (F1, tightened per round-5 review N-AP1): a network/
    // src error after data has already loaded once (e.g. the ticket's
    // short TTL expiring mid-session, so the browser's next Range request
    // 401/403s) used to be silently swallowed by the "keep the player
    // mounted, let the element handle it" guard meant for mid-playback
    // decode hiccups -- leaving playback stalled with no recovery for the
    // rest of the mount. It must instead re-mint a fresh ticket for the
    // same fileId (minting is cheap and doesn't re-check ownership) and
    // resume near where the ticketed stream left off rather than
    // restarting from 0 -- NOT fall straight to a full-file blob download,
    // which would silently reintroduce the exact #1201 symptom this PR
    // exists to remove for any video longer than the ticket TTL.
    let mintCount = 0
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        mintCount += 1
        return {
          ok: true,
          json: async () => ({
            path: `/api/files/preview/video-file-id?ticket=signed-ticket-${mintCount}`,
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const streamedVideo = await screen.findByLabelText('clip.mp4')
    expect(streamedVideo.getAttribute('src')).toBe(
      'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket-1'
    )

    fireEvent.loadedData(streamedVideo)
    Object.defineProperty(streamedVideo, 'currentTime', {
      value: 42,
      writable: true,
      configurable: true,
    })
    // code 2 === MediaError.MEDIA_ERR_NETWORK; jsdom has no MediaError
    // global to construct a real instance from (see the matching comment
    // in inline-file-preview.tsx), so the plain numeric code is faked here
    // the same way production code reads it.
    Object.defineProperty(streamedVideo, 'error', {
      value: { code: 2 },
      configurable: true,
    })
    fireEvent.error(streamedVideo)

    const recoveredVideo = await waitFor(() => {
      const element = screen.getByLabelText('clip.mp4')
      expect(element.getAttribute('src')).toBe(
        'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket-2'
      )
      return element as HTMLVideoElement
    })
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()

    // A real browser resets currentTime to 0 when a new src loads; set that
    // up explicitly so the assertion below only passes if the seek-on-load
    // logic actually ran, not because the old value was left untouched.
    recoveredVideo.currentTime = 0
    fireEvent.loadedData(recoveredVideo)
    expect(recoveredVideo.currentTime).toBe(42)
  })

  it('falls back to the blob path when re-minting also fails after a ticket expires mid-playback', async () => {
    // Companion to the recovery test above: if the re-mint itself fails
    // (e.g. access was actually revoked, not just the ticket expired), the
    // player must still fall back to the blob/direct strategy via
    // reportLoadFailure rather than being left on a dead ticketed src.
    let mintCount = 0
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        mintCount += 1
        if (mintCount > 1) {
          return { ok: false, status: 403 }
        }
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket-1',
          }),
        }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const streamedVideo = await screen.findByLabelText('clip.mp4')
    fireEvent.loadedData(streamedVideo)
    Object.defineProperty(streamedVideo, 'error', {
      value: { code: 2 },
      configurable: true,
    })
    fireEvent.error(streamedVideo)

    await waitFor(() => {
      expect(screen.getByLabelText('clip.mp4').getAttribute('src')).toMatch(/^blob:/)
    })
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()
  })

  it('keeps the player mounted for a decode-hiccup error after loading, even while streaming', async () => {
    // Contrast with the recovery test above: an error with no network-error
    // code (or none at all) after data has loaded is the ordinary
    // mid-playback hiccup case and must NOT trigger a re-mint/fallback --
    // the element handles those itself. This holds regardless of whether
    // the current src is the ticketed one.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return {
          ok: true,
          json: async () => ({
            path: '/api/files/preview/video-file-id?ticket=signed-ticket',
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const streamedVideo = await screen.findByLabelText('clip.mp4')
    fireEvent.loadedData(streamedVideo)
    Object.defineProperty(streamedVideo, 'error', {
      value: { code: 3 }, // MEDIA_ERR_DECODE, not a network failure
      configurable: true,
    })
    fireEvent.error(streamedVideo)

    expect(screen.getByLabelText('clip.mp4').getAttribute('src')).toBe(
      'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket'
    )
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()
  })

  it('also recovers via a fresh ticket for a MEDIA_ERR_SRC_NOT_SUPPORTED error while streaming', async () => {
    // Regression test (N-AP2): browsers are inconsistent about which
    // MediaError code an expired ticket's mid-stream 401/403 surfaces as --
    // some report MEDIA_ERR_NETWORK (2), others MEDIA_ERR_SRC_NOT_SUPPORTED
    // (4) because the failed fetch never yields a decodable resource.
    // Gating recovery on code 2 alone would silently swallow the 4 case the
    // exact same way the pre-F1 bug swallowed all of them.
    let mintCount = 0
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        mintCount += 1
        return {
          ok: true,
          json: async () => ({
            path: `/api/files/preview/video-file-id?ticket=signed-ticket-${mintCount}`,
          }),
        }
      }
      throw new Error(`unexpected blob fetch for ${url}`)
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const streamedVideo = await screen.findByLabelText('clip.mp4')
    fireEvent.loadedData(streamedVideo)
    Object.defineProperty(streamedVideo, 'error', {
      value: { code: 4 }, // MEDIA_ERR_SRC_NOT_SUPPORTED
      configurable: true,
    })
    fireEvent.error(streamedVideo)

    await waitFor(() => {
      expect(screen.getByLabelText('clip.mp4').getAttribute('src')).toBe(
        'http://api.local/api/files/preview/video-file-id?ticket=signed-ticket-2'
      )
    })
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()
  })

  it('does not get stuck loading when unmounted while a ticket mint is in flight', async () => {
    const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})
    let resolveMint: ((value: { ok: boolean; json: () => Promise<unknown> }) => void) | undefined
    apiRequestMock.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveMint = resolve
        })
    )

    const { unmount } = render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    await waitFor(() => expect(apiRequestMock).toHaveBeenCalled())
    unmount()

    // Resolving after unmount must not update state on an unmounted
    // component (React would log an error) or throw -- the isCancelled
    // guard in useResolvedMediaUrl's effect cleanup is what prevents this.
    resolveMint?.({
      ok: true,
      json: async () => ({
        path: '/api/files/preview/video-file-id?ticket=signed-ticket',
      }),
    })
    await Promise.resolve()
    await Promise.resolve()

    expect(consoleErrorSpy).not.toHaveBeenCalled()
    consoleErrorSpy.mockRestore()
  })

  it('falls back to blob fetch when ticket minting fails', async () => {
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        return { ok: false, status: 500 }
      }
      return {
        ok: true,
        blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
      }
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    await waitFor(() => {
      expect(video.getAttribute('src')).toMatch(/^blob:/)
    })
    expect(apiRequestMock).toHaveBeenCalledTimes(2)
  })

  it('does not attempt to mint a streaming ticket for images', async () => {
    // Images have no seek/progressive-loading need, so the extra ticket
    // round trip is pure overhead -- only audio/video opt into it.
    apiRequestMock.mockImplementation(async (url: string) => {
      if (url.includes('/stream-tickets/')) {
        throw new Error('images must not request a streaming ticket')
      }
      return {
        ok: true,
        blob: async () => new Blob(['image-bytes'], { type: 'image/png' }),
      }
    })

    render(
      <InlineFilePreview
        source={{ type: 'image', fileId: 'image-file-id', filename: 'chart.png' }}
      />
    )

    const image = await screen.findByAltText('chart.png')
    await waitFor(() => {
      expect(image.getAttribute('src')).toMatch(/^blob:/)
    })
    expect(apiRequestMock).toHaveBeenCalledTimes(1)
  })

  it('falls back to the public video preview when authenticated loading fails', async () => {
    apiRequestMock.mockResolvedValue({ ok: false, status: 401 })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    expect(video).toHaveAttribute(
      'src',
      'http://api.local/api/files/public/preview/video-file-id'
    )
    expect(screen.getByRole('link', { name: 'Open' })).toHaveAttribute(
      'href',
      'http://api.local/api/files/public/preview/video-file-id'
    )
  })

  it('revokes the video blob URL on unmount', async () => {
    const revokeObjectUrlSpy = vi.spyOn(URL, 'revokeObjectURL')
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })

    const { unmount } = render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    const blobUrl = video.getAttribute('src') || ''
    expect(blobUrl).toMatch(/^blob:/)

    unmount()
    expect(revokeObjectUrlSpy).toHaveBeenCalledWith(blobUrl)

    revokeObjectUrlSpy.mockRestore()
  })

  it('shows the load error text when the video fails on both paths', async () => {
    // The hook already fell back to the public preview URL after the
    // authenticated fetch failed; an error event from the element itself
    // means both paths are exhausted.
    apiRequestMock.mockResolvedValue({ ok: false, status: 403 })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    fireEvent.error(video)

    expect(await screen.findByText('Failed to load preview.')).toBeInTheDocument()
    expect(screen.queryByLabelText('clip.mp4')).not.toBeInTheDocument()
    // The header keeps the filename and Open link so the file stays reachable.
    expect(screen.getByText('clip.mp4')).toBeInTheDocument()
    expect(screen.getByRole('link', { name: 'Open' })).toBeInTheDocument()
  })

  it('keeps the player mounted when an error follows successful loading', async () => {
    // A mid-playback decode hiccup fires the same native error event as a
    // load failure; once data has loaded the element surfaces the problem
    // itself and must not be replaced by the terminal error state.
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/mp4' }),
    })

    render(
      <InlineFilePreview
        source={{ type: 'video', fileId: 'video-file-id', filename: 'clip.mp4' }}
      />
    )

    const video = await screen.findByLabelText('clip.mp4')
    fireEvent.loadedData(video)
    fireEvent.error(video)

    expect(screen.getByLabelText('clip.mp4')).toBeInTheDocument()
    expect(screen.queryByText('Failed to load preview.')).not.toBeInTheDocument()
  })

  it('renders a video link with a filename extension as an inline video preview', async () => {
    apiRequestMock.mockResolvedValue({
      ok: true,
      blob: async () => new Blob(['video-bytes'], { type: 'video/webm' }),
    })

    render(
      <InlineFilePreview source={{ fileId: 'abc-123', filename: 'clip.webm' }} />
    )

    const video = await screen.findByLabelText('clip.webm')
    expect(video.tagName.toLowerCase()).toBe('video')
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
