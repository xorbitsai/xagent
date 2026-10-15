/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getWorkforceMock = vi.hoisted(() => vi.fn())
const getWorkforceRunMock = vi.hoisted(() => vi.fn())
const getWorkforceAgentExecutionMock = vi.hoisted(() => vi.fn())
const listWorkforceRunsMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const paramsMock = vi.hoisted(() => ({ id: "42" as string | string[] | undefined }))
const searchParamsMock = vi.hoisted(() => new URLSearchParams())
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) => {
    if (!vars) return key
    return Object.entries(vars).reduce(
      (value, [name, replacement]) => value.replace(`{${name}}`, String(replacement)),
      key,
    )
  },
)
const appState = vi.hoisted(() => ({
  currentTask: null as null | { id: string; status: string },
  traceEvents: [] as Array<Record<string, unknown>>,
  steps: [] as Array<Record<string, unknown>>,
  dagExecution: null as unknown,
  isHistoryLoading: false,
  messages: [] as Array<Record<string, unknown>>,
  filePreview: { isOpen: false, fileId: "", fileName: "", viewMode: "preview" },
}))

vi.mock("next/navigation", () => ({
  useParams: () => paramsMock,
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => searchParamsMock,
}))

vi.mock("next/link", () => ({
  default: ({ children, href, ...props }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en-US", t: translateMock, tDynamic: translateMock }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: vi.fn(),
    setTaskId: setTaskIdMock,
    closeFilePreview: closeFilePreviewMock,
    dispatch: dispatchMock,
    openFilePreview: vi.fn(),
    getFileDownloadUrl: (fileId: string) => `/api/files/${fileId}/download`,
    state: appState,
  }),
  isTerminalTaskStatus: (status: string | null | undefined) => status === "completed" || status === "failed",
}))

vi.mock("@/contexts/auth-context", () => ({
  useAuth: () => ({ user: { id: "1", is_admin: false } }),
}))

vi.mock("@/components/config-dialog", () => ({
  ConfigDialog: ({ trigger }: { trigger: unknown }) => trigger,
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
    resetMention: vi.fn(),
    selectedFileIndex: 0,
    showFilePicker: false,
  }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    hasAsrModel: false,
    startRecording: vi.fn(),
    status: "idle",
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: () => <div data-testid="task-conversation-panel" />,
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: () => <div data-testid="file-preview" />,
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: () => <div data-testid="file-actions" />,
}))

vi.mock("@/lib/workforces-api", () => ({
  getWorkforce: getWorkforceMock,
  getWorkforceRun: getWorkforceRunMock,
  getWorkforceAgentExecution: getWorkforceAgentExecutionMock,
  listWorkforceRuns: listWorkforceRunsMock,
  runWorkforce: runWorkforceMock,
}))

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>("@/lib/api-wrapper")
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("sonner", () => ({ toast: { error: toastErrorMock, success: vi.fn() } }))

import WorkforceRunPage from "./page-client"
import type { WorkforceDetail, WorkforceRunHistoryItem } from "@/types/workforce"

const workforceDetail: WorkforceDetail = {
  id: 42,
  name: "Launch Workforce",
  description: null,
  status: "active",
  manager: { id: 7, name: "Manager Agent", description: null, logo_url: null, status: "published" },
  workers: [],
  canvas_layout: null,
  scope_type: "user",
  scope_id: "1",
  owner_user_id: 1,
  created_at: null,
  updated_at: null,
}

const pastRun: WorkforceRunHistoryItem = {
  id: 9,
  task_id: 99,
  status: "completed",
  is_preview: false,
  task_title: "Past run title",
  message: "draft plan",
  created_at: "2026-07-19T10:00:00Z",
  completed_at: "2026-07-19T10:03:00Z",
}

const runResponse = { workforce_run_id: 5, task_id: 99, status: "running", redirect_url: "/task/99" }

function emptyJsonResponse(): Response {
  return new Response(JSON.stringify([]), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function uploadResponse(fileId: string): Response {
  return new Response(JSON.stringify({ success: true, file_id: fileId }), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  })
}

function routeUploadResponse(fileId: string) {
  apiRequestMock.mockImplementation((url: string) =>
    typeof url === "string" && url.includes("/api/files/upload")
      ? Promise.resolve(uploadResponse(fileId))
      : Promise.resolve(emptyJsonResponse()),
  )
}

async function renderRunPage() {
  getWorkforceMock.mockResolvedValueOnce(workforceDetail)
  const view = render(<WorkforceRunPage />)
  await screen.findByText("workforces.run.placeholder")
  return view
}

function getComposer(container: HTMLElement) {
  const editor = screen.getByRole("textbox")
  const submit = container.querySelector('form button[type="submit"]') as HTMLButtonElement
  return { editor, submit }
}

function typeMessage(editor: HTMLElement, text: string) {
  editor.innerHTML = text
  fireEvent.input(editor)
}

async function attachFile(container: HTMLElement, name = "brief.txt", fileId = "file-123") {
  routeUploadResponse(fileId)
  return attachFiles(container, [name])
}

async function attachFiles(container: HTMLElement, names: string[]) {
  const files = names.map((name) => new File(["hello"], name, { type: "text/plain", lastModified: 1 }))
  fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement, {
    target: { files },
  })
  for (const name of names) await screen.findByText(name)
  return files
}

async function openNewRun() {
  fireEvent.click(screen.getByRole("button", { name: "workforces.runs.title" }))
  fireEvent.click(await screen.findByRole("button", { name: "workforces.run.newRun" }))
}

async function sendFirstMessage(container: HTMLElement, text = "Plan launch") {
  const { editor, submit } = getComposer(container)
  typeMessage(editor, text)
  await waitFor(() => expect(submit).not.toBeDisabled())
  fireEvent.click(submit)
}

describe("workforce run first-message attachments", () => {
  beforeEach(() => {
    getWorkforceMock.mockReset()
    getWorkforceRunMock.mockReset()
    getWorkforceAgentExecutionMock.mockReset()
    listWorkforceRunsMock.mockReset().mockResolvedValue({ items: [], total: 0, page: 1, size: 20, pages: 0 })
    runWorkforceMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    setTaskIdMock.mockReset()
    closeFilePreviewMock.mockReset()
    dispatchMock.mockReset()
    apiRequestMock.mockReset().mockImplementation(() => Promise.resolve(emptyJsonResponse()))
    toastErrorMock.mockReset()
    paramsMock.id = "42"
    searchParamsMock.delete("run")
    appState.currentTask = null
    appState.filePreview = { isOpen: false, fileId: "", fileName: "", viewMode: "preview" }
  })

  afterEach(() => cleanup())

  it("exposes a file upload control before the run starts", async () => {
    const { container } = await renderRunPage()

    expect(container.querySelector('input[type="file"]')).not.toBeNull()
  })

  it("sends the first message with an uploaded file id", async () => {
    runWorkforceMock.mockResolvedValueOnce(runResponse)
    const { container } = await renderRunPage()
    await attachFile(container)

    await sendFirstMessage(container)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Plan launch",
        files: ["file-123"],
        is_visible: false,
      })
    })
  })

  it("sends the first message without files as an empty list", async () => {
    runWorkforceMock.mockResolvedValueOnce(runResponse)
    const { container } = await renderRunPage()

    await sendFirstMessage(container)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Plan launch",
        files: [],
        is_visible: false,
      })
    })
  })

  it("keeps the attached file and the draft when the first run fails", async () => {
    runWorkforceMock.mockRejectedValueOnce(new Error("run failed"))
    const { container } = await renderRunPage()
    await attachFile(container)

    await sendFirstMessage(container)

    await waitFor(() => expect(runWorkforceMock).toHaveBeenCalled())
    expect(await screen.findByText("brief.txt")).toBeInTheDocument()
    expect(screen.getByRole("textbox")).toHaveTextContent("Plan launch")
    expect(screen.queryByTestId("task-conversation-panel")).not.toBeInTheDocument()
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(getComposer(container).submit).not.toBeDisabled())
  })

  it("asks for a message instead of running with attachments only", async () => {
    const { container } = await renderRunPage()
    await attachFile(container)
    const { submit } = getComposer(container)
    await waitFor(() => expect(submit).not.toBeDisabled())

    fireEvent.click(submit)

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("workforces.run.messageRequired"))
    expect(toastErrorMock).toHaveBeenCalledTimes(1)
    expect(runWorkforceMock).not.toHaveBeenCalled()
    expect(screen.getByText("brief.txt")).toBeInTheDocument()
  })

  it("clears the attachment after a successful first send", async () => {
    runWorkforceMock.mockResolvedValueOnce(runResponse)
    const { container } = await renderRunPage()
    await attachFile(container)

    await sendFirstMessage(container)
    await waitFor(() => expect(runWorkforceMock).toHaveBeenCalledTimes(1))

    await openNewRun()

    expect(screen.queryByText("brief.txt")).not.toBeInTheDocument()
    runWorkforceMock.mockResolvedValueOnce(runResponse)
    await sendFirstMessage(container, "Second run")
    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenLastCalledWith("42", {
        message: "Second run",
        files: [],
        is_visible: false,
      })
    })
  })

  it("clears the attachment when a past run is opened before starting a new run", async () => {
    listWorkforceRunsMock.mockResolvedValueOnce({ items: [pastRun], total: 1, page: 1, size: 20, pages: 1 })
    const { container } = await renderRunPage()
    await attachFile(container)

    fireEvent.click(screen.getByRole("button", { name: "workforces.runs.title" }))
    fireEvent.click(await screen.findByRole("button", { name: /Past run title/ }))
    await screen.findByTestId("task-conversation-panel")

    fireEvent.click(screen.getByRole("button", { name: "workforces.run.newRun" }))

    expect(screen.queryByText("brief.txt")).not.toBeInTheDocument()
    runWorkforceMock.mockResolvedValueOnce(runResponse)
    await sendFirstMessage(container, "Fresh run")
    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Fresh run",
        files: [],
        is_visible: false,
      })
    })
  })

  it("keeps the submit disabled and ignores new uploads while the first run is pending", async () => {
    let resolveRun: (value: unknown) => void = () => {}
    runWorkforceMock.mockImplementationOnce(() => new Promise((resolve) => { resolveRun = resolve }))
    const { container } = await renderRunPage()

    await sendFirstMessage(container)
    const { submit } = getComposer(container)
    await waitFor(() => expect(runWorkforceMock).toHaveBeenCalledTimes(1))

    expect(submit).toBeDisabled()
    fireEvent.change(container.querySelector('input[type="file"]') as HTMLInputElement, {
      target: { files: [new File(["late"], "late.txt", { type: "text/plain", lastModified: 1 })] },
    })
    expect(screen.queryByText("late.txt")).not.toBeInTheDocument()

    resolveRun(runResponse)
    await waitFor(() => expect(screen.queryByTestId("task-conversation-panel")).toBeInTheDocument())
  })

  it("ignores an upload that resolves after the pre-run attachments were cleared", async () => {
    listWorkforceRunsMock.mockResolvedValueOnce({ items: [pastRun], total: 1, page: 1, size: 20, pages: 1 })
    const pending: Array<{ name: string; resolve: (response: Response) => void }> = []
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (typeof url === "string" && url.includes("/api/files/upload")) {
        const file = (options?.body as FormData).get("file") as File
        return new Promise<Response>((resolve) => pending.push({ name: file.name, resolve }))
      }
      return Promise.resolve(emptyJsonResponse())
    })

    const { container } = await renderRunPage()
    await attachFiles(container, ["ok.txt", "fail.txt"])

    fireEvent.click(screen.getByRole("button", { name: "workforces.runs.title" }))
    fireEvent.click(await screen.findByRole("button", { name: /Past run title/ }))
    await screen.findByTestId("task-conversation-panel")
    fireEvent.click(screen.getByRole("button", { name: "workforces.run.newRun" }))
    await waitFor(() => expect(screen.getByRole("textbox")).toBeInTheDocument())

    // The old composer's partial-failure update ([ok.txt]) must not repopulate
    // the list the run switch already cleared.
    for (const item of pending) {
      item.resolve(item.name === "ok.txt"
        ? uploadResponse("file-ok")
        : new Response(JSON.stringify({ success: false, file_id: null }), {
            status: 500,
            headers: { "Content-Type": "application/json" },
          }))
    }

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalled())
    expect(screen.queryByText("ok.txt")).not.toBeInTheDocument()
  })
})
