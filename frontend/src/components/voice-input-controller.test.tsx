import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    cn: (...classes: Array<string | false | null | undefined>) =>
      classes.filter(Boolean).join(" "),
    getApiUrl: () => "http://api.local",
    getUploadApiUrl: () => "http://upload.local",
  }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: {
    error: toastErrorMock,
  },
}))

vi.mock("lucide-react", () => {
  const Icon = (props: React.SVGProps<SVGSVGElement>) => <svg {...props} />
  return {
    Loader2: Icon,
    Mic: Icon,
    Square: Icon,
  }
})

import { VoiceInputController } from "./voice-input-controller"

function jsonResponse(data: unknown): Response {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function setVisibleRect(
  element: HTMLElement,
  rect: Partial<DOMRect> = {}
): void {
  const visibleRect = {
    bottom: 50,
    height: 40,
    left: 10,
    right: 210,
    top: 10,
    width: 200,
    x: 10,
    y: 10,
    ...rect,
  }
  element.getBoundingClientRect = () =>
    ({
      ...visibleRect,
      toJSON: () => ({}),
    }) as DOMRect
}

class MediaRecorderMock {
  static isTypeSupported = vi.fn(() => true)

  mimeType: string
  ondataavailable: ((event: BlobEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null
  onstop: ((event: Event) => void) | null = null
  state: RecordingState = "inactive"

  constructor(_stream: MediaStream, options?: MediaRecorderOptions) {
    this.mimeType = options?.mimeType || "audio/webm"
  }

  start(): void {
    this.state = "recording"
  }

  stop(): void {
    this.state = "inactive"
    const blob = new Blob(["audio"], { type: this.mimeType })
    this.ondataavailable?.({ data: blob } as BlobEvent)
    this.onstop?.(new Event("stop"))
  }
}

describe("VoiceInputController", () => {
  beforeEach(() => {
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(jsonResponse([{ abilities: ["asr"] }]))
      }
      if (url === "http://upload.local/api/models/speech/transcribe") {
        return Promise.resolve(jsonResponse({ text: "voice text" }))
      }
      return Promise.reject(new Error(`unexpected request: ${url}`))
    })

    Object.defineProperty(globalThis, "MediaRecorder", {
      configurable: true,
      value: MediaRecorderMock,
    })
    Object.defineProperty(window, "MediaRecorder", {
      configurable: true,
      value: MediaRecorderMock,
    })
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: {
        getUserMedia: vi.fn().mockResolvedValue({
          getTracks: () => [{ stop: vi.fn() }],
        } as unknown as MediaStream),
      },
    })
  })

  afterEach(() => {
    cleanup()
  })

  it("inserts transcription into the field active when recording started", async () => {
    render(
      <>
        <input aria-label="first prompt" data-voice-input="true" />
        <input aria-label="second prompt" data-voice-input="true" />
        <VoiceInputController />
      </>
    )

    const first = screen.getByLabelText("first prompt") as HTMLInputElement
    const second = screen.getByLabelText("second prompt") as HTMLInputElement
    setVisibleRect(first)
    setVisibleRect(second)

    first.focus()
    fireEvent.focusIn(first)
    fireEvent.click(await screen.findByLabelText("voiceInput.start"))

    await screen.findByLabelText("voiceInput.stop")
    second.focus()
    fireEvent.focusIn(second)
    fireEvent.click(screen.getByLabelText("voiceInput.stop"))

    await waitFor(() => {
      expect(first).toHaveValue("voice text")
    })
    expect(second).toHaveValue("")
  })

  it("shows an error when voice recording is unsupported", async () => {
    render(
      <>
        <input aria-label="prompt" data-voice-input="true" />
        <VoiceInputController />
      </>
    )

    const input = screen.getByLabelText("prompt")
    setVisibleRect(input)
    input.focus()
    fireEvent.focusIn(input)

    Object.defineProperty(globalThis, "MediaRecorder", {
      configurable: true,
      value: undefined,
    })
    Object.defineProperty(window, "MediaRecorder", {
      configurable: true,
      value: undefined,
    })
    Object.defineProperty(navigator, "mediaDevices", {
      configurable: true,
      value: undefined,
    })

    fireEvent.click(await screen.findByLabelText("voiceInput.start"))

    expect(toastErrorMock).toHaveBeenCalledWith("voiceInput.errors.unsupported")
  })

  it("does not enable arbitrary contentEditable regions without opt-in", () => {
    render(
      <>
        <div contentEditable suppressContentEditableWarning>
          private draft
        </div>
        <VoiceInputController />
      </>
    )

    const editable = screen.getByText("private draft")
    setVisibleRect(editable)
    editable.focus()
    fireEvent.focusIn(editable)

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })

  it("does not enable regular form fields without opt-in", () => {
    render(
      <>
        <input aria-label="agent name" />
        <VoiceInputController />
      </>
    )

    const input = screen.getByLabelText("agent name")
    setVisibleRect(input)
    input.focus()
    fireEvent.focusIn(input)

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })

  it("positions contentEditable controls against the enclosing input shell", async () => {
    render(
      <>
        <div data-testid="input-shell">
          <div
            contentEditable
            data-voice-input="true"
            role="textbox"
            suppressContentEditableWarning
          />
        </div>
        <VoiceInputController />
      </>
    )

    const shell = screen.getByTestId("input-shell")
    const editor = screen.getByRole("textbox")
    Object.defineProperty(editor, "isContentEditable", {
      configurable: true,
      value: true,
    })
    setVisibleRect(shell, {
      bottom: 160,
      height: 140,
      left: 40,
      right: 440,
      top: 20,
      width: 400,
      x: 40,
      y: 20,
    })
    setVisibleRect(editor, {
      bottom: 82,
      height: 40,
      left: 60,
      right: 130,
      top: 42,
      width: 70,
      x: 60,
      y: 42,
    })

    editor.focus()
    fireEvent.focusIn(editor)

    const button = await screen.findByLabelText("voiceInput.start")
    expect(button).toHaveStyle({ left: "336px", top: "104px" })
  })

  it("shows textarea controls on hover against the enclosing input shell", async () => {
    render(
      <>
        <div data-testid="input-shell">
          <textarea aria-label="task prompt" data-voice-input="true" />
        </div>
        <VoiceInputController />
      </>
    )

    const shell = screen.getByTestId("input-shell")
    const textarea = screen.getByLabelText("task prompt")
    setVisibleRect(shell, {
      bottom: 160,
      height: 140,
      left: 40,
      right: 440,
      top: 20,
      width: 400,
      x: 40,
      y: 20,
    })
    setVisibleRect(textarea, {
      bottom: 82,
      height: 40,
      left: 60,
      right: 130,
      top: 42,
      width: 70,
      x: 60,
      y: 42,
    })

    fireEvent.pointerOver(textarea)

    const button = await screen.findByLabelText("voiceInput.start")
    expect(button).toHaveStyle({ left: "336px", top: "104px" })
  })

  it("does not float controls over large framed textareas", () => {
    render(
      <>
        <div data-testid="input-shell">
          <textarea aria-label="large prompt" />
        </div>
        <VoiceInputController />
      </>
    )

    const shell = screen.getByTestId("input-shell")
    const textarea = screen.getByLabelText("large prompt")
    setVisibleRect(shell, {
      bottom: 180,
      height: 160,
      left: 40,
      right: 440,
      top: 20,
      width: 400,
      x: 40,
      y: 20,
    })
    setVisibleRect(textarea, {
      bottom: 160,
      height: 120,
      left: 60,
      right: 420,
      top: 40,
      width: 360,
      x: 60,
      y: 40,
    })

    fireEvent.pointerOver(textarea)

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })

  it("does not float controls over textareas with nearby action buttons", () => {
    render(
      <>
        <div data-testid="input-shell">
          <textarea aria-label="action prompt" />
          <button type="button">Send</button>
        </div>
        <VoiceInputController />
      </>
    )

    const shell = screen.getByTestId("input-shell")
    const textarea = screen.getByLabelText("action prompt")
    const button = screen.getByText("Send")
    setVisibleRect(shell, {
      bottom: 90,
      height: 70,
      left: 40,
      right: 440,
      top: 20,
      width: 400,
      x: 40,
      y: 20,
    })
    setVisibleRect(textarea, {
      bottom: 70,
      height: 40,
      left: 60,
      right: 360,
      top: 30,
      width: 300,
      x: 60,
      y: 30,
    })
    setVisibleRect(button, {
      bottom: 72,
      height: 32,
      left: 372,
      right: 428,
      top: 40,
      width: 56,
      x: 372,
      y: 40,
    })

    fireEvent.pointerOver(textarea)

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })
})
