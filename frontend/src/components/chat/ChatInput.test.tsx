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
    // selectedAgents no longer renders a "Using @name" chip (that
    // mechanism was removed as unreachable dead code) - only drives the
    // no-model guard above. A regression reintroducing it would leave
    // this test green without an explicit check.
    expect(screen.queryByText("@Shared Agent")).not.toBeInTheDocument()
  })

  it("clears the read-only model display instead of showing a stale lead's model while a new one's config is still loading", () => {
    const commonProps = {
      hideFileUpload: true,
      inputValue: "hello",
      onInputChange: vi.fn(),
      onSend: vi.fn(),
      readOnlyConfig: true,
    }
    const { rerender } = render(<ChatInput {...commonProps} taskConfig={{ model: "agent-a-model" }} />)

    expect(screen.getByText("agent-a-model")).toBeInTheDocument()

    // Caller switched leads - the new lead's own config fetch resets
    // taskConfig to undefined while still in flight (readOnlyConfig stays
    // true throughout).
    rerender(<ChatInput {...commonProps} taskConfig={undefined} />)

    expect(screen.queryByText("agent-a-model")).not.toBeInTheDocument()
    expect(screen.getByText("chatPage.input.noModel")).toBeInTheDocument()
  })

  it("submits the live taskConfig model, not a stale value inherited by the internal mirror", async () => {
    // The internal `agentConfig` mirror inherits the previous model when a
    // new taskConfig doesn't specify one (`taskConfig.model || prev.model`
    // in the syncing effect) - correct for genuine partial updates to the
    // SAME lead, but submission (and the read-only model badge, which reads
    // the same live-`taskConfig` value) must still prefer the live prop over
    // that mirror once a config is meant to be authoritative (readOnlyConfig),
    // so a real timing gap between a lead switch and its own effect catching
    // up can't submit - or display - an unrelated previous lead's model.
    const onSend = vi.fn()
    const commonProps = {
      hideFileUpload: true,
      inputValue: "hello",
      onInputChange: vi.fn(),
      onSend,
      readOnlyConfig: true,
    }
    const { container, rerender } = render(
      <ChatInput {...commonProps} taskConfig={{ model: "agent-a-model", executionMode: "think" }} />
    )
    expect(screen.getByText("agent-a-model")).toBeInTheDocument()

    // New lead's taskConfig arrives without its own model yet (e.g. only
    // executionMode resolved so far) - the internal mirror would inherit the
    // previous lead's model rather than clearing it, but the live-read badge
    // must not show that stale value either.
    rerender(<ChatInput {...commonProps} taskConfig={{ executionMode: "flash" }} />)
    expect(screen.queryByText("agent-a-model")).not.toBeInTheDocument()
    expect(screen.getByText("chatPage.input.noModel")).toBeInTheDocument()

    fireEvent.submit(container.querySelector("form") as HTMLFormElement)

    await waitFor(() => {
      expect(onSend).toHaveBeenCalledWith(
        "hello",
        expect.objectContaining({ model: "" })
      )
    })
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

  it("hides the execution mode picker for a read-only (agent-resolved) task", async () => {
    mockDefaultModel()
    render(
      <ChatInput
        hideFileUpload
        inputValue="summarize this doc"
        onInputChange={vi.fn()}
        onSend={vi.fn()}
        readOnlyConfig
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

    rerender(<ChatInput {...props} readOnlyConfig />)
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
})
