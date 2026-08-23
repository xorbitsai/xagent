import React, { Suspense, startTransition } from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { createRoot } from "react-dom/client"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const openFilePreviewMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const resetMentionMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const authUserMock = vi.hoisted(() => ({
  current: { id: "1", is_admin: true } as { id: string; is_admin: boolean },
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper"
  )
  return {
    ...actual,
    apiRequest: apiRequestMock,
  }
})

vi.mock("@/lib/utils", () => ({
  cn: (...classes: Array<string | false | null | undefined>) =>
    classes.filter(Boolean).join(" "),
  generateClientMessageId: vi.fn()
    .mockReturnValueOnce("client-message-1")
    .mockReturnValueOnce("client-message-2")
    .mockReturnValue("client-message-next"),
  getApiUrl: () => "http://api.local",
  getUploadApiUrl: () => "http://upload.local",
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    openFilePreview: openFilePreviewMock,
  }),
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: authUserMock.current }),
}))

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: unknown }) => trigger,
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: routerPushMock,
  }),
}))

vi.mock("@/hooks/use-file-mention", () => ({
  useFileMention: () => ({
    checkTrigger: vi.fn(),
    dropdownPosition: null,
    fileList: [],
    filteredFiles: [],
    handleKeyDown: vi.fn(() => false),
    insertFile: vi.fn(),
    isLoadingFiles: false,
    resetMention: resetMentionMock,
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
  },
}))

import { ChatInput } from "./ChatInput"

const emptyJsonResponse = () =>
  new Response(JSON.stringify([]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })

describe("ChatInput", () => {
  beforeEach(() => {
    authUserMock.current = { id: "1", is_admin: true }
    apiRequestMock.mockReset()
    apiRequestMock.mockImplementation(() => Promise.resolve(emptyJsonResponse()))
    openFilePreviewMock.mockReset()
    routerPushMock.mockReset()
    resetMentionMock.mockReset()
    toastErrorMock.mockReset()
  })

  afterEach(() => {
    cleanup()
    Reflect.deleteProperty(document, "execCommand")
    vi.unstubAllGlobals()
  })

  it("requires a model when submitting generic chat", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(screen.getByText("chatPage.input.noModelAlert")).toBeInTheDocument()
    })
    expect(onSend).not.toHaveBeenCalled()
  })

  it("binds a new task to a selected local browser window", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/computer/local-browser/readiness") {
        return Promise.resolve(
          new Response(JSON.stringify({
            ready: true,
            connected: true,
            attached: true,
            application: "Google Chrome",
            title: "GitHub",
            windows: [
              {
                pid: 100,
                window_id: 20,
                application: "Google Chrome",
                title: "GitHub",
              },
            ],
            permissions: {},
            issues: [],
            message: "",
          }), { status: 200, headers: { "Content-Type": "application/json" } })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect this page"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1" }}
      />
    )

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"))
    fireEvent.click(await screen.findByText("GitHub"))

    expect(screen.queryByText("chatPage.input.localBrowser.chipLabel")).not.toBeInTheDocument()
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument()
    const form = container.querySelector("form") as HTMLFormElement
    expect(form).toHaveClass("rounded-2xl")
    expect(form).not.toHaveClass("rounded-tl-none")
    fireEvent.submit(form)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "inspect this page",
        expect.objectContaining({
          runtimeExtensions: {
            local_browser: {
              pid: 100,
              window_id: 20,
              application: "Google Chrome",
              title: "GitHub",
            },
          },
        }),
      )
    })
  })

  it("does not submit a stale local browser target after the feature is hidden", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/computer/local-browser/readiness") {
        return Promise.resolve(
          new Response(JSON.stringify({
            ready: true,
            application: "Google Chrome",
            windows: [{
              pid: 100,
              window_id: 20,
              application: "Google Chrome",
              title: "GitHub",
            }],
            issues: [],
            message: "",
          }), { status: 200, headers: { "Content-Type": "application/json" } }),
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })
    const onSend = vi.fn()
    const commonProps = {
      hideFileUpload: true,
      inputValue: "inspect this page",
      onInputChange: vi.fn(),
      onSend,
      taskConfig: { model: "model-1" },
    }
    const { container, rerender } = render(<ChatInput {...commonProps} />)

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"))
    fireEvent.click(await screen.findByText("GitHub"))
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument()

    rerender(<ChatInput {...commonProps} readOnlyConfig />)
    expect(screen.queryByLabelText("Google Chrome · GitHub")).not.toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledTimes(1))
    expect(onSend.mock.calls[0][1]).not.toHaveProperty("runtimeExtensions")
  })

  it("clears a selected target that disappears from refreshed readiness", async () => {
    let readinessCalls = 0
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/computer/local-browser/readiness") {
        readinessCalls += 1
        return Promise.resolve(
          new Response(JSON.stringify({
            ready: true,
            application: "Google Chrome",
            windows: readinessCalls === 1 ? [{
              pid: 100,
              window_id: 20,
              application: "Google Chrome",
              title: "GitHub",
            }] : [{
              pid: 100,
              window_id: 21,
              application: "Google Chrome",
              title: "Xagent",
            }],
            issues: [],
            message: "",
          }), { status: 200, headers: { "Content-Type": "application/json" } }),
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })
    render(
      <ChatInput
        hideFileUpload
        inputValue="inspect this page"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        taskConfig={{ model: "model-1" }}
      />,
    )

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"))
    fireEvent.click(await screen.findByText("GitHub"))
    expect(screen.getByLabelText("Google Chrome · GitHub")).toBeInTheDocument()

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(
      screen.getAllByText("chatPage.input.localBrowser.label").at(-1) as HTMLElement,
    )

    await waitFor(() => {
      expect(screen.queryByLabelText("Google Chrome · GitHub")).not.toBeInTheDocument()
    })
  })

  it("disambiguates browser windows with the same application and title", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/computer/local-browser/readiness") {
        return Promise.resolve(
          new Response(JSON.stringify({
            ready: true,
            application: "Google Chrome",
            windows: [
              {
                pid: 100,
                window_id: 20,
                application: "Google Chrome",
                title: "GitHub",
              },
              {
                pid: 100,
                window_id: 21,
                application: "Google Chrome",
                title: "GitHub",
              },
            ],
            issues: [],
            message: "",
          }), { status: 200, headers: { "Content-Type": "application/json" } }),
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    render(
      <ChatInput
        hideFileUpload
        inputValue="inspect"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        taskConfig={{ model: "model-1" }}
      />,
    )

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"))

    expect(await screen.findAllByText("GitHub")).toHaveLength(2)
    expect(
      screen.getAllByText("chatPage.input.localBrowser.windowIdentifier"),
    ).toHaveLength(2)
  })

  it("does not advertise local browser to a non-admin", () => {
    authUserMock.current = { id: "2", is_admin: false }

    render(
      <ChatInput
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        taskConfig={{ model: "model-1" }}
      />
    )

    expect(screen.queryByLabelText("chatPage.input.actions.add")).not.toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalledWith(
      "http://api.local/api/computer/local-browser/readiness",
      expect.anything(),
    )
  })

  it("aborts an in-flight readiness request when the picker unmounts", async () => {
    let readinessSignal: AbortSignal | undefined
    apiRequestMock.mockImplementation((url: string, init?: RequestInit) => {
      if (url === "http://api.local/api/computer/local-browser/readiness") {
        readinessSignal = init?.signal ?? undefined
        return new Promise<Response>(() => {})
      }
      return Promise.resolve(emptyJsonResponse())
    })
    const { unmount } = render(
      <ChatInput
        hideFileUpload
        inputValue="inspect"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        taskConfig={{ model: "model-1" }}
      />
    )

    fireEvent.click(screen.getByLabelText("chatPage.input.actions.add"))
    fireEvent.click(screen.getByText("chatPage.input.localBrowser.label"))
    await waitFor(() => expect(readinessSignal).toBeDefined())

    unmount()

    expect(readinessSignal?.aborted).toBe(true)
  })

  it("suppresses preseeded files and restored file previews when files are disabled", () => {
    const onSend = vi.fn()
    const onFilesChange = vi.fn()
    const uploadFile = vi.fn()
    const file = new File(["secret"], "secret.txt", { type: "text/plain" })
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn(),
    })
    const { container } = render(
      <ChatInput
        files={[file]}
        filesDisabled
        hideConfig
        inputValue="[secret.txt](file:file-secret)"
        onFilesChange={onFilesChange}
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
        uploadFile={uploadFile}
      />
    )

    expect(container.querySelector('input[type="file"]')).toBeNull()
    expect(container.querySelector(".file-chip-preview")).toBeNull()
    const editor = screen.getByRole("textbox")
    expect(editor).toHaveTextContent("secret.txt")
    expect(editor).not.toHaveTextContent("file-secret")
    expect(editor.innerHTML).not.toContain("file-secret")

    editor.innerHTML = '<span class="file-chip-preview" data-file-path="file-secret">secret</span>'
    fireEvent.click(editor.querySelector(".file-chip-preview") as HTMLElement)

    fireEvent.drop(container.querySelector("form") as HTMLFormElement, {
      dataTransfer: { types: ["Files"] },
    })
    fireEvent.paste(editor, {
      clipboardData: {
        items: [{
          kind: "file",
          type: "text/plain",
          getAsFile: () => file,
        }],
        getData: () => "",
      },
    })

    expect(openFilePreviewMock).not.toHaveBeenCalled()
    expect(onFilesChange).not.toHaveBeenCalled()
    expect(uploadFile).not.toHaveBeenCalled()
    expect(onSend).not.toHaveBeenCalled()
    expect(
      apiRequestMock.mock.calls.some(
        ([url]) => typeof url === "string" && url.includes("/api/files/"),
      )
    ).toBe(false)
  })

  it("keeps restored file preview semantics when only upload UI is hidden", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="[report.txt](file:file-report)"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        readOnlyConfig
      />
    )

    const chip = container.querySelector(".file-chip-preview")
    expect(chip).not.toBeNull()
    fireEvent.click(chip as HTMLElement)
    expect(openFilePreviewMock).toHaveBeenCalledWith(
      "file-report",
      "file-report",
      [{ fileName: "file-report", fileId: "file-report" }],
    )
  })

  it("restores a draft file chip without displaying its canonical id and preserves that id on send", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="[report.txt](file:canonical-file-id)"
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
      />
    )

    expect(screen.getByText("report.txt")).toBeInTheDocument()
    expect(screen.queryByText("canonical-file-id")).not.toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "[report.txt](file:canonical-file-id)",
        expect.any(Object),
      )
    })
  })

  it("keeps a controlled first-message draft after an async send rejection", async () => {
    const onInputChange = vi.fn()
    const onSend = vi.fn().mockRejectedValue(new Error("delivery failed"))
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="retry this"
        onInputChange={onInputChange}
        onSend={onSend}
        readOnlyConfig
      />,
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)
    await waitFor(() => expect(onSend).toHaveBeenCalled())
    expect(onInputChange).not.toHaveBeenCalledWith("")
  })

  it("allows selected agent submissions without a local model", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={onSend}
        readOnlyConfig
        selectedAgents={[{ id: 42, name: "Shared Agent" }]}
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "hello",
        expect.objectContaining({ model: "" })
      )
    })
    expect(resetMentionMock).toHaveBeenCalledTimes(1)
    expect(screen.queryByText("chatPage.input.noModelAlert")).not.toBeInTheDocument()
  })

  it("does not show pause for uppercase terminal task status", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="next request"
        isLoading
        onInputChange={vi.fn()}
        onPause={vi.fn()}
        onSend={vi.fn()}
        taskStatus="FAILED"
      />
    )

    expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).not.toBeDisabled()
  })

  it("renders voice input as an inline toolbar action when ASR is configured", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(await screen.findByLabelText("voiceInput.start")).toBeInTheDocument()
  })

  it("does not load or render voice controls when voice input is disabled", async () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled={false}
      />
    )

    await Promise.resolve()

    expect(apiRequestMock).not.toHaveBeenCalled()
    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })

  it("stops active recording without transcribing when voice input is disabled", async () => {
    const trackStop = vi.fn()
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: trackStop }],
    })
    const recorders: Array<{
      state: "inactive" | "recording"
      stop: ReturnType<typeof vi.fn>
    }> = []

    class FakeMediaRecorder {
      static isTypeSupported = vi.fn(() => true)
      mimeType = "audio/webm"
      state: "inactive" | "recording" = "inactive"
      ondataavailable: ((event: { data: Blob }) => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null
      stop = vi.fn(() => {
        this.state = "inactive"
        this.ondataavailable?.({ data: new Blob(["audio"], { type: this.mimeType }) })
        this.onstop?.()
      })

      constructor() {
        recorders.push(this)
      }

      start() {
        this.state = "recording"
      }
    }

    vi.stubGlobal("MediaRecorder", FakeMediaRecorder)
    vi.stubGlobal("navigator", {
      ...navigator,
      mediaDevices: { getUserMedia },
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    const { rerender } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )

    fireEvent.click(await screen.findByLabelText("voiceInput.start"))
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledWith({ audio: true }))
    expect(recorders).toHaveLength(1)

    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled={false}
      />
    )

    await waitFor(() => expect(recorders[0].stop).toHaveBeenCalledTimes(1))
    expect(trackStop).toHaveBeenCalledTimes(1)
    expect(
      apiRequestMock.mock.calls.some(
        ([url]) => typeof url === "string" && url.includes("/api/models/speech/transcribe"),
      ),
    ).toBe(false)
  })

  it("does not invalidate a committed recorder for an abandoned disabled render", async () => {
    const trackStop = vi.fn()
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: trackStop }],
    })
    const recorders: Array<{
      state: "inactive" | "recording"
      stop: ReturnType<typeof vi.fn>
    }> = []
    let suspendedRenderStarted = false
    const neverResolves = new Promise<never>(() => undefined)

    class FakeMediaRecorder {
      static isTypeSupported = vi.fn(() => true)
      mimeType = "audio/webm"
      state: "inactive" | "recording" = "inactive"
      ondataavailable: ((event: { data: Blob }) => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null
      stop = vi.fn(() => {
        this.state = "inactive"
        this.ondataavailable?.({ data: new Blob(["audio"], { type: this.mimeType }) })
        this.onstop?.()
      })

      constructor() {
        recorders.push(this)
      }

      start() {
        this.state = "recording"
      }
    }

    const NeverSettles = () => {
      suspendedRenderStarted = true
      throw neverResolves
    }
    const VoiceInputHarness = ({
      enabled,
      suspend,
    }: {
      enabled: boolean
      suspend: boolean
    }) => (
      <Suspense fallback={null}>
        <ChatInput
          hideConfig
          hideFileUpload
          inputValue=""
          onInputChange={vi.fn()}
          onSend={vi.fn()}
          voiceInputEnabled={enabled}
        />
        {suspend && <NeverSettles />}
      </Suspense>
    )

    vi.stubGlobal("MediaRecorder", FakeMediaRecorder)
    vi.stubGlobal("navigator", {
      ...navigator,
      mediaDevices: { getUserMedia },
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)
    try {
      await act(async () => {
        root.render(<VoiceInputHarness enabled suspend={false} />)
      })

      fireEvent.click(await screen.findByLabelText("voiceInput.start"))
      await waitFor(() => expect(getUserMedia).toHaveBeenCalledWith({ audio: true }))
      expect(recorders).toHaveLength(1)

      await act(async () => {
        startTransition(() => {
          root.render(<VoiceInputHarness enabled={false} suspend />)
        })
        await Promise.resolve()
      })
      expect(suspendedRenderStarted).toBe(true)

      await act(async () => {
        root.render(<VoiceInputHarness enabled suspend={false} />)
      })

      expect(
        apiRequestMock.mock.calls.some(
          ([url]) => typeof url === "string" && url.includes("/api/models/speech/transcribe"),
        ),
      ).toBe(false)
      fireEvent.click(screen.getByLabelText("voiceInput.stop"))

      await waitFor(() => expect(trackStop).toHaveBeenCalledTimes(1))
    } finally {
      await act(async () => {
        root.unmount()
      })
      container.remove()
    }
  })

  it("stops a stale media stream when voice capability is restored before getUserMedia resolves", async () => {
    let resolveStream!: (stream: { getTracks: () => Array<{ stop: ReturnType<typeof vi.fn> }> }) => void
    const staleTrackStop = vi.fn()
    const staleStream = {
      getTracks: () => [{ stop: staleTrackStop }],
    }
    const getUserMedia = vi.fn(() => new Promise<typeof staleStream>((resolve) => {
      resolveStream = resolve
    }))
    const recorders: unknown[] = []

    class FakeMediaRecorder {
      static isTypeSupported = vi.fn(() => true)
      mimeType = "audio/webm"
      state: "inactive" | "recording" = "inactive"

      constructor() {
        recorders.push(this)
      }

      start() {
        this.state = "recording"
      }

      stop() {
        this.state = "inactive"
      }
    }

    vi.stubGlobal("MediaRecorder", FakeMediaRecorder)
    vi.stubGlobal("navigator", {
      ...navigator,
      mediaDevices: { getUserMedia },
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })

    const { rerender } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )

    fireEvent.click(await screen.findByLabelText("voiceInput.start"))
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledWith({ audio: true }))

    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled={false}
      />
    )
    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )

    resolveStream(staleStream)

    await waitFor(() => expect(staleTrackStop).toHaveBeenCalledTimes(1))
    expect(recorders).toHaveLength(0)
    expect(
      apiRequestMock.mock.calls.some(
        ([url]) => typeof url === "string" && url.includes("/api/models/speech/transcribe"),
      ),
    ).toBe(false)
  })

  it("does not apply a stale transcription after voice capability is restored", async () => {
    let resolveTranscription!: (response: Response) => void
    const transcription = new Promise<Response>((resolve) => {
      resolveTranscription = resolve
    })
    const getUserMedia = vi.fn().mockResolvedValue({
      getTracks: () => [{ stop: vi.fn() }],
    })
    const onInputChange = vi.fn()

    class FakeMediaRecorder {
      static isTypeSupported = vi.fn(() => true)
      mimeType = "audio/webm"
      state: "inactive" | "recording" = "inactive"
      ondataavailable: ((event: { data: Blob }) => void) | null = null
      onstop: (() => void) | null = null
      onerror: (() => void) | null = null

      start() {
        this.state = "recording"
      }

      stop() {
        this.state = "inactive"
        this.ondataavailable?.({ data: new Blob(["audio"], { type: this.mimeType }) })
        this.onstop?.()
      }
    }

    vi.stubGlobal("MediaRecorder", FakeMediaRecorder)
    vi.stubGlobal("navigator", {
      ...navigator,
      mediaDevices: { getUserMedia },
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(
          new Response(JSON.stringify([{ abilities: ["asr"] }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      if (url === "http://upload.local/api/models/speech/transcribe") {
        return transcription
      }
      return Promise.resolve(emptyJsonResponse())
    })

    const { rerender } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={onInputChange}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )
    const editor = screen.getByRole("textbox")
    Object.defineProperty(document, "execCommand", {
      configurable: true,
      value: vi.fn((_command: string, _showUi: boolean, value: string) => {
        editor.appendChild(document.createTextNode(value))
        return true
      }),
    })

    fireEvent.click(await screen.findByLabelText("voiceInput.start"))
    await waitFor(() => expect(getUserMedia).toHaveBeenCalledWith({ audio: true }))
    fireEvent.click(screen.getByLabelText("voiceInput.stop"))
    await waitFor(() => {
      expect(
        apiRequestMock.mock.calls.some(
          ([url]) => url === "http://upload.local/api/models/speech/transcribe",
        ),
      ).toBe(true)
    })

    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={onInputChange}
        onSend={vi.fn()}
        voiceInputEnabled={false}
      />
    )
    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={onInputChange}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )

    resolveTranscription(
      new Response(JSON.stringify({ text: "stale transcription" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    )

    await new Promise<void>((resolve) => setTimeout(resolve, 0))
    expect(editor).not.toHaveTextContent("stale transcription")
    expect(onInputChange).not.toHaveBeenCalledWith("stale transcription")
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("does not restore stale ASR availability after voice input is disabled", async () => {
    let resolveInitialModels!: (models: Array<{ abilities: string[] }>) => void
    const initialModels = new Promise<Array<{ abilities: string[] }>>((resolve) => {
      resolveInitialModels = resolve
    })
    let availabilityRequests = 0
    apiRequestMock.mockImplementation((url: string) => {
      if (url !== "http://api.local/api/models/?category=speech&limit=1000") {
        return Promise.resolve(emptyJsonResponse())
      }
      availabilityRequests += 1
      return availabilityRequests === 1
        ? { ok: true, json: () => initialModels }
        : new Promise<Response>(() => undefined)
    })

    const { rerender } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )
    await waitFor(() => expect(availabilityRequests).toBe(1))

    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled={false}
      />
    )
    resolveInitialModels([{ abilities: ["asr"] }])
    await new Promise<void>((resolve) => setTimeout(resolve, 0))

    rerender(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        voiceInputEnabled
      />
    )

    expect(screen.queryByLabelText("voiceInput.start")).not.toBeInTheDocument()
  })

  it("allows live guidance while a task is running", async () => {
    const onSend = vi.fn()
    const onPause = vi.fn()
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="please focus on the API contract"
        isLoading
        onInputChange={vi.fn()}
        onPause={onPause}
        onSend={onSend}
        taskStatus="running"
      />
    )

    expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).not.toBeDisabled()

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "please focus on the API contract",
        expect.objectContaining({ model: "" })
      )
    })
  })

  it("keeps the draft when durable delivery is rejected", async () => {
    const onInputChange = vi.fn()
    const onSend = vi.fn().mockRejectedValue(new Error("Message was rejected"))
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="keep this draft"
        onInputChange={onInputChange}
        onSend={onSend}
        readOnlyConfig
      />
    )

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    await waitFor(() => {
      expect(onInputChange).not.toHaveBeenCalledWith("")
    })
    expect(onSend.mock.calls[0][1]).toEqual(
      expect.objectContaining({ clientMessageId: expect.any(String) })
    )
  })

  const mockDefaultModel = () => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url === "http://api.local/api/models/?category=llm") {
        return Promise.resolve(
          new Response(JSON.stringify([{ model_id: "model-1", is_default: true }]), {
            status: 200,
            headers: { "Content-Type": "application/json" },
          })
        )
      }
      return Promise.resolve(emptyJsonResponse())
    })
  }

  const executionModeLabel = (mode: string) =>
    `builds.configForm.executionMode.${mode}.title`
  const UNSET_MODE_LABEL = "builds.configForm.executionMode.unset"

  it("sends the picked execution mode for a new standalone task", async () => {
    mockDefaultModel()
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="plan a trip"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    const trigger = await screen.findByText(UNSET_MODE_LABEL)
    fireEvent.click(trigger)
    fireEvent.click(screen.getByText(executionModeLabel("think")))
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "plan a trip",
        expect.objectContaining({ executionMode: { mode: "think" } })
      )
    })
  })

  it("sends an explicitly picked auto mode", async () => {
    mockDefaultModel()
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="plan a trip"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    fireEvent.click(await screen.findByText(UNSET_MODE_LABEL))
    fireEvent.click(screen.getByText(executionModeLabel("auto")))
    expect(screen.getByText(executionModeLabel("auto"))).toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "plan a trip",
        expect.objectContaining({ executionMode: { mode: "auto" } })
      )
    })
  })

  it("omits the execution mode when the picker was never touched", async () => {
    mockDefaultModel()
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="plan a trip"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    await screen.findByText(UNSET_MODE_LABEL)
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    expect(onSend.mock.calls[0][1].executionMode).toBeUndefined()
  })

  it("hides the execution mode picker for a template-resolved task", async () => {
    mockDefaultModel()
    render(
      <ChatInput
        hideFileUpload
        inputValue="summarize this doc"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        selectedTemplate={{ id: "doc-summarizer", name: "Doc Summarizer" }}
      />
    )

    expect(screen.queryByText(UNSET_MODE_LABEL)).not.toBeInTheDocument()
  })

  it("drops a pick made before the picker was hidden", async () => {
    mockDefaultModel()
    const onSend = vi.fn()
    const props = {
      hideFileUpload: true,
      inputValue: "plan a trip",
      onInputChange: vi.fn(),
      onSend,
    }
    const { container, rerender } = render(<ChatInput {...props} />)

    fireEvent.click(await screen.findByText(UNSET_MODE_LABEL))
    fireEvent.click(screen.getByText(executionModeLabel("think")))
    expect(screen.getByText(executionModeLabel("think"))).toBeInTheDocument()

    rerender(
      <ChatInput {...props} selectedTemplate={{ id: "doc-summarizer", name: "Doc Summarizer" }} />
    )
    rerender(<ChatInput {...props} />)

    expect(screen.getByText(UNSET_MODE_LABEL)).toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    expect(onSend.mock.calls[0][1].executionMode).toBeUndefined()
  })

  it("closes the execution mode menu after a send", async () => {
    mockDefaultModel()
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="plan a trip"
        onInputChange={vi.fn()}
        onSend={onSend}
      />
    )

    fireEvent.click(await screen.findByText(UNSET_MODE_LABEL))
    expect(screen.getByText(executionModeLabel("think"))).toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => expect(onSend).toHaveBeenCalledOnce())
    await waitFor(() => {
      expect(screen.queryByText(executionModeLabel("think"))).not.toBeInTheDocument()
    })
  })

  it("hides the execution mode picker when the composer config is read-only", async () => {
    mockDefaultModel()
    render(
      <ChatInput
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        readOnlyConfig
      />
    )

    expect(screen.queryByText(UNSET_MODE_LABEL)).not.toBeInTheDocument()
  })

  it("hides the execution mode picker when composer config is hidden", async () => {
    mockDefaultModel()
    render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="hello"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(screen.queryByText(UNSET_MODE_LABEL)).not.toBeInTheDocument()
  })

  it("hides the execution mode picker for an existing task and keeps its mode", async () => {
    const onSend = vi.fn()
    const { container } = render(
      <ChatInput
        hideFileUpload
        inputValue="continue"
        onInputChange={vi.fn()}
        onSend={onSend}
        taskConfig={{ model: "model-1", executionMode: "flash" }}
      />
    )

    expect(
      screen.queryByText("builds.configForm.executionMode.auto.title")
    ).not.toBeInTheDocument()
    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "continue",
        expect.objectContaining({ executionMode: { mode: "flash" } })
      )
    })
  })

  it("keeps generic loading input disabled without a live task status", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue="wait"
        isLoading
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })

  it("shows pause for a running task when there is no draft to send", () => {
    const { container } = render(
      <ChatInput
        hideConfig
        hideFileUpload
        inputValue=""
        isLoading
        onInputChange={vi.fn()}
        onPause={vi.fn()}
        onSend={vi.fn()}
        taskStatus="running"
      />
    )

    expect(screen.getByTitle("agent.input.actions.pauseTask")).toBeInTheDocument()
    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })

  it("keeps pause hidden while running draft files are still uploading", async () => {
    const onPause = vi.fn()
    const uploadFile = vi.fn(() => new Promise<{ file_id: string }>(() => {}))

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])

      return (
        <ChatInput
          hideConfig
          inputValue=""
          files={files}
          isLoading
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onPause={onPause}
          onSend={vi.fn()}
          taskStatus="running"
          uploadFile={uploadFile}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const file = new File(["draft"], "draft.txt", { type: "text/plain" })

    fireEvent.change(fileInput, { target: { files: [file] } })

    await waitFor(() => {
      expect(uploadFile).toHaveBeenCalledWith(
        file,
        expect.objectContaining({ taskType: "task" })
      )
    })
    await waitFor(() => {
      expect(screen.queryByTitle("agent.input.actions.pauseTask")).not.toBeInTheDocument()
    })
    expect(container.querySelector('button[type="submit"]')).toBeDisabled()
  })

  it("serializes production attachment requests in repeated-selection order", async () => {
    const completions: Array<() => void> = []
    const requestedFiles: File[] = []
    let activeUploads = 0
    let maximumActiveUploads = 0
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url !== "http://upload.local/api/files/upload") {
        return Promise.resolve(emptyJsonResponse())
      }

      const body = options?.body as FormData
      const requestedFile = body.get("file") as File
      requestedFiles.push(requestedFile)
      activeUploads += 1
      maximumActiveUploads = Math.max(maximumActiveUploads, activeUploads)
      return new Promise<Response>(resolve => {
        completions.push(() => {
          activeUploads -= 1
          resolve(new Response(
            JSON.stringify({ success: true, file_id: `id-${requestedFile.name}` }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ))
        })
      })
    })

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const files = [
      new File(["one"], "one.txt", { type: "text/plain" }),
      new File(["two"], "two.txt", { type: "text/plain" }),
      new File(["three"], "three.txt", { type: "text/plain" }),
    ]

    fireEvent.change(fileInput, { target: { files: files.slice(0, 2) } })
    fireEvent.change(fileInput, { target: { files: files.slice(2) } })

    await waitFor(() => expect(requestedFiles).toEqual([files[0]]))

    for (let index = 0; index < files.length; index += 1) {
      await act(async () => completions[index]?.())
      if (index + 1 < files.length) {
        await waitFor(() => expect(requestedFiles).toHaveLength(index + 2))
        expect(requestedFiles[index + 1]).toBe(files[index + 1])
      }
    }

    expect(requestedFiles).toEqual(files)
    expect(maximumActiveUploads).toBe(1)
  })

  it("aborts an active production upload and promotes the next queued file", async () => {
    const uploadSignals: AbortSignal[] = []
    let completeSecondUpload: (() => void) | undefined
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url !== "http://upload.local/api/files/upload") {
        return Promise.resolve(emptyJsonResponse())
      }

      const signal = options?.signal as AbortSignal
      uploadSignals.push(signal)
      if (uploadSignals.length === 1) {
        return new Promise<Response>((_resolve, reject) => {
          signal.addEventListener("abort", () => {
            reject(signal.reason ?? new DOMException("Upload cancelled", "AbortError"))
          }, { once: true })
        })
      }
      return new Promise<Response>(resolve => {
        completeSecondUpload = () => resolve(new Response(
          JSON.stringify({ success: true, file_id: "second-id" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ))
      })
    })

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const first = new File(["one"], "one.txt")
    const second = new File(["two"], "two.txt")

    fireEvent.change(fileInput, { target: { files: [first, second] } })
    await waitFor(() => expect(uploadSignals).toHaveLength(1))

    fireEvent.click(screen.getAllByTitle("common.cancel")[0])

    expect(uploadSignals[0].aborted).toBe(true)
    await waitFor(() => expect(uploadSignals).toHaveLength(2))
    expect(uploadSignals[1].aborted).toBe(false)
    await act(async () => completeSecondUpload?.())
    await waitFor(() => expect(screen.queryByText("one.txt")).not.toBeInTheDocument())
    expect(screen.getByText("two.txt")).toBeInTheDocument()
  })

  it("aborts a hung production upload at its deadline and promotes the queue", async () => {
    vi.useFakeTimers()
    try {
      const uploadSignals: AbortSignal[] = []
      apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
        if (url !== "http://upload.local/api/files/upload") {
          return Promise.resolve(emptyJsonResponse())
        }

        const signal = options?.signal as AbortSignal
        uploadSignals.push(signal)
        if (uploadSignals.length === 1) {
          return new Promise<Response>((_resolve, reject) => {
            signal.addEventListener("abort", () => {
              reject(signal.reason ?? new DOMException("Upload timed out", "TimeoutError"))
            }, { once: true })
          })
        }
        return Promise.resolve(new Response(
          JSON.stringify({ success: true, file_id: "second-id" }),
          { status: 200, headers: { "Content-Type": "application/json" } },
        ))
      })

      function Harness() {
        const [files, setFiles] = React.useState<File[]>([])
        return (
          <ChatInput
            files={files}
            hideConfig
            inputValue=""
            onFilesChange={setFiles}
            onInputChange={vi.fn()}
            onSend={vi.fn()}
          />
        )
      }

      const { container } = render(<Harness />)
      const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
      const first = new File(["one"], "one.txt")
      const second = new File(["two"], "two.txt")

      fireEvent.change(fileInput, { target: { files: [first, second] } })
      await act(async () => undefined)
      expect(uploadSignals).toHaveLength(1)

      await act(async () => {
        await vi.advanceTimersByTimeAsync(15 * 60 * 1000)
      })

      expect(uploadSignals[0].aborted).toBe(true)
      expect(uploadSignals).toHaveLength(2)
      expect(screen.queryByText("one.txt")).not.toBeInTheDocument()
      expect(screen.getByText("two.txt")).toBeInTheDocument()
      expect(toastErrorMock).toHaveBeenCalledWith(
        "files.uploadFailed",
        expect.objectContaining({ duration: 8000 }),
      )
    } finally {
      vi.useRealTimers()
    }
  })

  it("cancels a queued equal-name file without aborting the active upload", async () => {
    let uploadCallCount = 0
    let completeFirstUpload: (() => void) | undefined
    const uploadSignals: AbortSignal[] = []
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url !== "http://upload.local/api/files/upload") {
        return Promise.resolve(emptyJsonResponse())
      }

      uploadCallCount += 1
      const signal = options?.signal as AbortSignal
      uploadSignals.push(signal)
      if (uploadCallCount === 1) {
        return new Promise<Response>((resolve, reject) => {
          completeFirstUpload = () => resolve(new Response(
            JSON.stringify({ success: true, file_id: "first-id" }),
            { status: 200, headers: { "Content-Type": "application/json" } },
          ))
          signal.addEventListener("abort", () => {
            reject(new DOMException("Upload cancelled", "AbortError"))
          }, { once: true })
        })
      }
      return Promise.resolve(new Response(
        JSON.stringify({ success: true, file_id: "unexpected-id" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      ))
    })

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    const lastModified = 1_700_000_000_000
    const activeFile = new File(["first"], "duplicate.txt", { lastModified })
    const queuedFile = new File(["second"], "duplicate.txt", { lastModified })

    fireEvent.change(fileInput, { target: { files: [activeFile, queuedFile] } })
    await waitFor(() => expect(uploadCallCount).toBe(1))

    fireEvent.click(screen.getAllByTitle("common.cancel")[1])

    expect(uploadSignals[0].aborted).toBe(false)
    await act(async () => completeFirstUpload?.())
    await waitFor(() => expect(screen.getAllByText("duplicate.txt")).toHaveLength(1))
    expect(uploadCallCount).toBe(1)
  })

  it("removes exactly one occurrence when the same File object is attached twice", () => {
    const duplicate = new File(["duplicate"], "duplicate.txt")
    const onFilesChange = vi.fn()

    render(
      <ChatInput
        files={[duplicate, duplicate]}
        hideConfig
        inputValue=""
        onFilesChange={onFilesChange}
        onInputChange={vi.fn()}
        onSend={vi.fn()}
      />
    )

    fireEvent.click(screen.getAllByTitle("common.remove")[0])

    expect(onFilesChange).toHaveBeenCalledOnce()
    expect(onFilesChange).toHaveBeenCalledWith([duplicate])
  })

  it("removes the rendered attachment by identity after a failure updates the live list", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const failed = new File(["bad"], "failed.txt")
    const target = new File(["target"], "target.txt")
    const sibling = new File(["sibling"], "sibling.txt")
    const uploadFile = vi.fn((file: File) => {
      if (file === failed) return Promise.reject(new Error("storage unavailable"))
      return Promise.resolve({ file_id: `id-${file.name}` })
    })
    let clickedStaleTarget = false
    let renderedFilesAtFailure: string[] = []
    let filesAfterRemoval: File[] | undefined

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      const handleFilesChange = (nextFiles: File[]) => {
        if (!clickedStaleTarget && nextFiles.length === 2 && !nextFiles.includes(failed)) {
          clickedStaleTarget = true
          renderedFilesAtFailure = screen
            .getAllByTitle("common.remove")
            .map(button => button.parentElement?.textContent || "")
          const targetChip = screen.getByText("target.txt").parentElement
          fireEvent.click(targetChip?.querySelector("button") as HTMLButtonElement)
          return
        }
        if (clickedStaleTarget) filesAfterRemoval = nextFiles
        setFiles(nextFiles)
      }

      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={handleFilesChange}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
          uploadFile={uploadFile}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [failed, target, sibling] } })

    await waitFor(() => expect(clickedStaleTarget).toBe(true))
    expect(renderedFilesAtFailure).toEqual(["failed.txt", "target.txt", "sibling.txt"])
    expect(filesAfterRemoval).toEqual([sibling])
    await waitFor(() => {
      expect(screen.queryByText("target.txt")).not.toBeInTheDocument()
      expect(screen.queryByText("failed.txt")).not.toBeInTheDocument()
      expect(screen.getByText("sibling.txt")).toBeInTheDocument()
    })
    consoleError.mockRestore()
  })

  it("keeps successful attachments when a queued peer fails", async () => {
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const successfulOne = new File(["one"], "one.txt")
    const failed = new File(["bad"], "failed.txt")
    const successfulTwo = new File(["two"], "two.txt")
    const uploadFile = vi.fn((file: File) => {
      if (file === failed) return Promise.reject(new Error("storage unavailable"))
      return Promise.resolve({ file_id: `id-${file.name}` })
    })

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
          uploadFile={uploadFile}
        />
      )
    }

    const { container } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(fileInput, {
      target: { files: [successfulOne, failed, successfulTwo] },
    })

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith(
      "storage unavailable",
      expect.objectContaining({ duration: 8000 }),
    ))
    expect(screen.getByText("one.txt")).toBeInTheDocument()
    expect(screen.queryByText("failed.txt")).not.toBeInTheDocument()
    expect(screen.getByText("two.txt")).toBeInTheDocument()
    expect(consoleError).toHaveBeenCalledWith(
      "Error uploading file:",
      expect.objectContaining({ message: "storage unavailable" }),
    )
    consoleError.mockRestore()
  })

  it("aborts the active upload and drops queued uploads on unmount", async () => {
    let uploadCallCount = 0
    let activeSignal: AbortSignal | undefined
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url !== "http://upload.local/api/files/upload") {
        return Promise.resolve(emptyJsonResponse())
      }
      uploadCallCount += 1
      activeSignal = options?.signal ?? undefined
      return new Promise<Response>((_resolve, reject) => {
        activeSignal?.addEventListener("abort", () => {
          reject(new DOMException("Upload cancelled", "AbortError"))
        }, { once: true })
      })
    })

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
        />
      )
    }

    const { container, unmount } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(fileInput, {
      target: {
        files: [new File(["one"], "one.txt"), new File(["two"], "two.txt")],
      },
    })
    await waitFor(() => expect(uploadCallCount).toBe(1))

    unmount()

    expect(activeSignal?.aborted).toBe(true)
    await act(async () => undefined)
    expect(uploadCallCount).toBe(1)
  })

  it("does not publish a custom upload result after unmount", async () => {
    let completeUpload: ((result: { file_id: string }) => void) | undefined
    const uploadFile = vi.fn(() => new Promise<{ file_id: string }>(resolve => {
      completeUpload = resolve
    }))
    const file = new File(["late"], "late.txt")

    function Harness() {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <ChatInput
          files={files}
          hideConfig
          inputValue=""
          onFilesChange={setFiles}
          onInputChange={vi.fn()}
          onSend={vi.fn()}
          uploadFile={uploadFile}
        />
      )
    }

    const { container, unmount } = render(<Harness />)
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
    fireEvent.change(fileInput, { target: { files: [file] } })
    await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1))

    unmount()
    await act(async () => completeUpload?.({ file_id: "late-id" }))

    expect((file as File & { file_id?: string }).file_id).toBeUndefined()
  })

  it("ignores filesDisabled from an abandoned render", async () => {
    let completeUpload: ((result: { file_id: string }) => void) | undefined
    let suspendedRenderStarted = false
    const neverResolves = new Promise<never>(() => undefined)
    const uploadFile = vi.fn(() => new Promise<{ file_id: string }>(resolve => {
      completeUpload = resolve
    }))
    const file = new File(["kept"], "kept.txt")

    const NeverSettles = () => {
      suspendedRenderStarted = true
      throw neverResolves
    }
    function Harness({ disabled, suspend }: { disabled: boolean; suspend: boolean }) {
      const [files, setFiles] = React.useState<File[]>([])
      return (
        <Suspense fallback={null}>
          <ChatInput
            files={files}
            filesDisabled={disabled}
            hideConfig
            inputValue=""
            onFilesChange={setFiles}
            onInputChange={vi.fn()}
            onSend={vi.fn()}
            uploadFile={uploadFile}
          />
          {suspend && <NeverSettles />}
        </Suspense>
      )
    }

    const container = document.createElement("div")
    document.body.appendChild(container)
    const root = createRoot(container)
    try {
      await act(async () => {
        root.render(<Harness disabled={false} suspend={false} />)
      })
      const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement
      fireEvent.change(fileInput, { target: { files: [file] } })
      await waitFor(() => expect(uploadFile).toHaveBeenCalledTimes(1))

      await act(async () => {
        startTransition(() => {
          root.render(<Harness disabled suspend />)
        })
        await Promise.resolve()
      })
      expect(suspendedRenderStarted).toBe(true)

      await act(async () => completeUpload?.({ file_id: "kept-id" }))

      expect((file as File & { file_id?: string }).file_id).toBe("kept-id")
      await waitFor(() => expect(screen.getByTitle("common.remove")).toBeInTheDocument())
    } finally {
      await act(async () => root.unmount())
      container.remove()
    }
  })
})
