/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getWorkforceMock = vi.hoisted(() => vi.fn())
const getWorkforceAgentExecutionMock = vi.hoisted(() => vi.fn())
const getWorkforceRunMock = vi.hoisted(() => vi.fn())
const listAgentOptionsMock = vi.hoisted(() => vi.fn().mockResolvedValue([]))
const listWorkforcesMock = vi.hoisted(() => vi.fn())
const listWorkforceRunsMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const addWorkforceAgentMock = vi.hoisted(() => vi.fn())
const archiveWorkforceMock = vi.hoisted(() => vi.fn())
const publishWorkforceMock = vi.hoisted(() => vi.fn())
const removeWorkforceAgentMock = vi.hoisted(() => vi.fn())
const unpublishWorkforceMock = vi.hoisted(() => vi.fn())
const updateWorkforceMock = vi.hoisted(() => vi.fn())
const updateWorkforceAgentMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const closeFilePreviewMock = vi.hoisted(() => vi.fn())
const dispatchMock = vi.hoisted(() => vi.fn())
const apiRequestMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const workforceAppState = vi.hoisted(() => ({
  currentTask: null as null | { id: string; status: string },
  traceEvents: [] as Array<Record<string, unknown>>,
  filePreview: {
    isOpen: false,
    fileId: "",
    fileName: "",
    viewMode: "preview" as "preview" | "code",
  },
}))
const paramsMock = vi.hoisted(() => ({ id: "42" as string | string[] | undefined }))
const searchParamsMock = vi.hoisted(() => new URLSearchParams())
const translateMock = vi.hoisted(
  () => (key: string, vars?: Record<string, string | number>) => {
    if (!vars) return key
    return Object.entries(vars).reduce(
      (value, [name, replacement]) =>
        value.replace(`{${name}}`, String(replacement)),
      key,
    )
  },
)

vi.mock("next/navigation", () => ({
  useParams: () => paramsMock,
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => searchParamsMock,
}))

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en-US",
    t: translateMock,
    tDynamic: (key: string) => translateMock(key),
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: vi.fn(),
    setTaskId: setTaskIdMock,
    closeFilePreview: closeFilePreviewMock,
    dispatch: dispatchMock,
    getFileDownloadUrl: (fileId: string) => `/api/files/${fileId}/download`,
    state: workforceAppState,
  }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: ({
    onSend,
    onAgentExecutionClick,
    showTaskActions,
    showTokenUsage,
    showTaskFiles,
  }: {
    onSend?: (message: string) => void
    onAgentExecutionClick?: (execution: {
      workerTaskId: string
      agentName: string
      status: "completed"
    }) => void
    showTaskActions?: boolean
    showTokenUsage?: boolean
    showTaskFiles?: boolean
  }) => (
    <div
      data-testid="task-conversation-panel"
      data-show-task-actions={String(Boolean(showTaskActions))}
      data-show-token-usage={String(Boolean(showTokenUsage))}
      data-show-task-files={String(Boolean(showTaskFiles))}
    >
      <button onClick={() => onSend?.("test message")}>Send Test</button>
      <button onClick={() => onAgentExecutionClick?.({
        workerTaskId: "agent_17_run",
        agentName: "Editor Agent",
        status: "completed",
      })}>View Editor execution</button>
      <button onClick={() => onAgentExecutionClick?.({
        workerTaskId: "agent_18_run",
        agentName: "QA Agent",
        status: "completed",
      })}>View QA execution</button>
    </div>
  ),
}))

vi.mock("@/components/file/file-preview-content", () => ({
  FilePreviewContent: () => <div data-testid="workforce-file-preview" />,
}))

vi.mock("@/components/file/file-preview-action-buttons", () => ({
  FilePreviewActionButtons: ({ onDownload }: { onDownload: () => void }) => (
    <button type="button" data-testid="workforce-file-actions" onClick={onDownload}>
      Download file
    </button>
  ),
}))

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/workforces-api", () => ({
  addWorkforceAgent: addWorkforceAgentMock,
  archiveWorkforce: archiveWorkforceMock,
  getWorkforce: getWorkforceMock,
  getWorkforceAgentExecution: getWorkforceAgentExecutionMock,
  getWorkforceRun: getWorkforceRunMock,
  listAgentOptions: listAgentOptionsMock,
  listWorkforceRuns: listWorkforceRunsMock,
  listWorkforces: listWorkforcesMock,
  publishWorkforce: publishWorkforceMock,
  removeWorkforceAgent: removeWorkforceAgentMock,
  runWorkforce: runWorkforceMock,
  unpublishWorkforce: unpublishWorkforceMock,
  updateWorkforce: updateWorkforceMock,
  updateWorkforceAgent: updateWorkforceAgentMock,
}))

vi.mock("sonner", () => ({
  toast: {
    error: toastErrorMock,
    success: vi.fn(),
  },
}))

vi.mock("@/components/ui/select", () => ({
  Select: ({
    value,
    onValueChange,
    options,
    placeholder,
    disabled,
  }: {
    value?: string
    onValueChange: (value: string) => void
    options?: Array<{ value: string; label: string }>
    placeholder?: string
    disabled?: boolean
  }) => (
    <select
      aria-label={placeholder}
      value={value || ""}
      disabled={disabled}
      onChange={(event) => onValueChange(event.target.value)}
    >
      <option value="">{placeholder}</option>
      {(options || []).map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  ),
}))

vi.mock("@/components/ui/switch", () => ({
  Switch: ({
    checked,
    disabled,
    onCheckedChange,
  }: {
    checked?: boolean
    disabled?: boolean
    onCheckedChange?: (checked: boolean) => void
  }) => (
    <input
      aria-label="switch"
      type="checkbox"
      checked={Boolean(checked)}
      disabled={disabled}
      onChange={(event) => onCheckedChange?.(event.target.checked)}
    />
  ),
}))

import WorkforcesPage from "./page"
import WorkforceDetailPage from "./[id]/page"
import WorkforceRunPage from "./[id]/run/page"
import {
  getAgentExecutionConclusion,
  mergeAgentExecutionTraceEvents,
} from "./[id]/run/page-client"
import { getNavigationGroupsForUser } from "@/components/layout/sidebar"
import type { WorkforceDetail, WorkforceListResponse } from "@/types/workforce"

const workforceDetail: WorkforceDetail = {
  id: 42,
  name: "Launch Workforce",
  description: null,
  status: "active",
  manager: {
    id: 7,
    name: "Manager Agent",
    description: null,
    logo_url: null,
    status: "published",
  },
  workers: [],
  canvas_layout: null,
  scope_type: "user",
  scope_id: "1",
  owner_user_id: 1,
  created_at: null,
  updated_at: null,
}

const listResponse: WorkforceListResponse = {
  items: [
    {
      id: 42,
      name: "Launch Workforce",
      description: "Coordinate launch work",
      status: "active",
      manager: {
        id: 7,
        name: "Manager Agent",
        logo_url: null,
      },
      worker_count: 2,
      last_run: {
        id: 9,
        task_id: 99,
        status: "completed",
        created_at: null,
      },
      created_at: null,
      updated_at: "2026-05-27T00:00:00Z",
    },
  ],
  total: 1,
  page: 1,
  size: 10,
  pages: 1,
}

async function selectRunFromHistory(name: RegExp): Promise<void> {
  fireEvent.click(
    await screen.findByRole("button", { name: "workforces.runs.title" }),
  )
  fireEvent.click(await screen.findByRole("button", { name }))
}

describe("workforce route entry points", () => {
  beforeEach(() => {
    getWorkforceMock.mockReset()
    getWorkforceAgentExecutionMock.mockReset()
    getWorkforceRunMock.mockReset()
    listAgentOptionsMock.mockReset().mockResolvedValue([])
    listWorkforceRunsMock.mockReset().mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      size: 20,
      pages: 0,
    })
    listWorkforcesMock.mockReset()
    runWorkforceMock.mockReset()
    addWorkforceAgentMock.mockReset()
    archiveWorkforceMock.mockReset()
    publishWorkforceMock.mockReset()
    removeWorkforceAgentMock.mockReset()
    unpublishWorkforceMock.mockReset()
    updateWorkforceMock.mockReset()
    updateWorkforceAgentMock.mockReset()
    routerPushMock.mockReset()
    routerReplaceMock.mockReset()
    setTaskIdMock.mockReset()
    closeFilePreviewMock.mockReset()
    dispatchMock.mockReset()
    apiRequestMock.mockReset()
    toastErrorMock.mockReset()
    workforceAppState.currentTask = null
    workforceAppState.traceEvents = []
    workforceAppState.filePreview = {
      isOpen: false,
      fileId: "",
      fileName: "",
      viewMode: "preview",
    }
    paramsMock.id = "42"
    searchParamsMock.delete("run")
    window.history.replaceState(null, "", "/workforces/42/run")
  })

  afterEach(() => {
    cleanup()
  })

  it("adds the visible sidebar entry for workforces", () => {
    const agentDevelopment = getNavigationGroupsForUser(null)[0]

    expect(agentDevelopment.items).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          href: "/workforces",
          nameKey: "nav.workforces",
        }),
      ]),
    )
  })

  it("merges streaming child traces into the selected Agent execution", () => {
    const merged = mergeAgentExecutionTraceEvents(
      [{
        event_id: "persisted-start",
        event_type: "react_task_start",
        data: {
          source: "xagent-agent-tool-child",
          worker_task_id: "agent_17_live",
        },
      }],
      [{
        event_id: "live-progress",
        event_type: "agent_progress",
        data: {
          source: "xagent-agent-tool-child",
          worker_task_id: "agent_17_live",
          message: "Generating scene 4",
        },
      }, {
        event_id: "other-worker",
        event_type: "agent_progress",
        data: {
          source: "xagent-agent-tool-child",
          worker_task_id: "agent_18_other",
        },
      }],
      "agent_17_live",
    )

    expect(merged.map((event) => event.event_id)).toEqual([
      "persisted-start",
      "live-progress",
    ])
  })

  it("ignores malformed streaming child trace payloads", () => {
    expect(() => mergeAgentExecutionTraceEvents(
      [],
      [
        null,
        42,
        "not-an-event",
        { event_id: "null-data", data: null },
        { event_id: "string-data", data: "not-an-object" },
        { event_id: "array-data", data: [] },
      ],
      "agent_17_live",
    )).not.toThrow()

    expect(mergeAgentExecutionTraceEvents(
      [],
      [{ event_id: "malformed", data: "not-an-object" }],
      "agent_17_live",
    )).toEqual([])
  })

  it("keeps the latest Agent execution when an older request resolves late", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [{
        id: 15,
        task_id: 760,
        status: "completed",
        is_preview: false,
        task_title: "What If Studio run",
        message: "Generate the film",
        created_at: "2026-07-17T16:32:08Z",
        completed_at: "2026-07-17T16:33:08Z",
      }],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })

    let resolveEditor!: (value: Record<string, unknown>) => void
    let resolveQa!: (value: Record<string, unknown>) => void
    getWorkforceAgentExecutionMock
      .mockReturnValueOnce(new Promise((resolve) => { resolveEditor = resolve }))
      .mockReturnValueOnce(new Promise((resolve) => { resolveQa = resolve }))

    render(<WorkforceRunPage />)

    await selectRunFromHistory(/What If Studio run/)
    fireEvent.click(screen.getByRole("button", { name: "View Editor execution" }))
    fireEvent.click(screen.getByRole("button", { name: "View QA execution" }))

    await act(async () => {
      resolveQa({
        task_id: 760,
        worker_task_id: "agent_18_run",
        agent_name: "QA Agent",
        status: "completed",
        trace_events: [{
          event_type: "task_completion",
          data: { result: { content: "QA is the latest result." } },
        }],
      })
    })
    expect(await screen.findByText("QA is the latest result.")).toBeInTheDocument()

    await act(async () => {
      resolveEditor({
        task_id: 760,
        worker_task_id: "agent_17_run",
        agent_name: "Editor Agent",
        status: "completed",
        trace_events: [{
          event_type: "task_completion",
          data: { result: { content: "Stale Editor result." } },
        }],
      })
    })

    expect(screen.getByText("QA is the latest result.")).toBeInTheDocument()
    expect(screen.queryByText("Stale Editor result.")).not.toBeInTheDocument()
  })

  it("keeps the latest retry for the same Agent execution", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [{
        id: 15,
        task_id: 760,
        status: "completed",
        is_preview: false,
        task_title: "What If Studio run",
        message: "Generate the film",
        created_at: "2026-07-17T16:32:08Z",
        completed_at: "2026-07-17T16:33:08Z",
      }],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })

    let resolveOlder!: (value: Record<string, unknown>) => void
    let resolveNewer!: (value: Record<string, unknown>) => void
    getWorkforceAgentExecutionMock
      .mockReturnValueOnce(new Promise((resolve) => { resolveOlder = resolve }))
      .mockReturnValueOnce(new Promise((resolve) => { resolveNewer = resolve }))

    render(<WorkforceRunPage />)

    await selectRunFromHistory(/What If Studio run/)
    const editorButton = screen.getByRole("button", { name: "View Editor execution" })
    fireEvent.click(editorButton)
    fireEvent.click(editorButton)

    await act(async () => {
      resolveNewer({
        task_id: 760,
        worker_task_id: "agent_17_run",
        agent_name: "Editor Agent",
        status: "completed",
        trace_events: [{
          event_type: "task_completion",
          data: { result: { content: "Newest Editor result." } },
        }],
      })
    })
    expect(await screen.findByText("Newest Editor result.")).toBeInTheDocument()

    await act(async () => {
      resolveOlder({
        task_id: 760,
        worker_task_id: "agent_17_run",
        agent_name: "Editor Agent",
        status: "completed",
        trace_events: [{
          event_type: "task_completion",
          data: { result: { content: "Older Editor result." } },
        }],
      })
    })

    expect(screen.getByText("Newest Editor result.")).toBeInTheDocument()
    expect(screen.queryByText("Older Editor result.")).not.toBeInTheDocument()
  })

  it("extracts a delegated Agent conclusion from terminal trace events", () => {
    expect(getAgentExecutionConclusion([{
      event_type: "react_task_end",
      data: { result: { output: "Fallback conclusion" } },
    }, {
      event_type: "ai_message",
      data: { content: "Agent's final conclusion" },
    }, {
      event_type: "task_completion",
      data: { result: { content: "Canonical completion" } },
    }])).toBe("Canonical completion")
  })

  it("renders the workforce list with PR7 route links", async () => {
    listWorkforcesMock.mockResolvedValueOnce(listResponse)

    render(<WorkforcesPage />)

    expect(await screen.findByText("Launch Workforce")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: /workforces.actions.new/ })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: /workforces.actions.run/ })).toHaveAttribute(
      "href",
      "/workforces/42/run",
    )
    expect(screen.getByRole("link", { name: /workforces.actions.edit/ })).toHaveAttribute(
      "href",
      "/workforces/42",
    )
  })

  it("keeps list run actions disabled for draft workforces", async () => {
    listWorkforcesMock.mockResolvedValueOnce({
      ...listResponse,
      items: [
        {
          ...listResponse.items[0],
          id: 43,
          name: "Draft Workforce",
          status: "draft",
          last_run: null,
        },
      ],
    })

    render(<WorkforcesPage />)

    expect(await screen.findByText("Draft Workforce")).toBeInTheDocument()
    expect(screen.queryByRole("link", { name: /workforces.actions.run/ })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: /workforces.actions.run/ })).toBeDisabled()
  })

  it("uses the shared Runs list on the workforce detail page", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [
        {
          id: 9,
          task_id: 99,
          status: "completed",
          is_preview: false,
          task_title: "Launch Workforce: draft plan",
          message: "draft plan",
          created_at: "2026-07-19T10:00:00Z",
          completed_at: "2026-07-19T10:03:00Z",
        },
      ],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })

    render(<WorkforceDetailPage />)

    fireEvent.click(
      await screen.findByRole("button", { name: /workforces.runs.title/ }),
    )
    fireEvent.click(await screen.findByText("Launch Workforce: draft plan"))

    expect(listWorkforceRunsMock).toHaveBeenCalledWith("42", { page: 1, size: 20 })
    expect(routerPushMock).toHaveBeenCalledWith("/workforces/42/run?run=9")
  })

  it("runs an active workforce and opens the created task", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    runWorkforceMock.mockResolvedValueOnce({
      workforce_run_id: 5,
      task_id: 99,
      status: "running",
      redirect_url: "/task/99",
    })

    const { container } = render(<WorkforceRunPage />)

    expect(await screen.findByText("Launch Workforce")).toBeInTheDocument()

    const textarea = screen.getByPlaceholderText("workforces.run.placeholder")
    fireEvent.change(textarea, { target: { value: " Draft launch plan " } })

    // Wait for the submit button to become enabled after state update
    await waitFor(() => {
      const submitBtn = container.querySelector("textarea + button:not([disabled])")
      expect(submitBtn).not.toBeNull()
    })

    const submitBtn = container.querySelector("textarea + button:not([disabled])")
    fireEvent.click(submitBtn!)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        files: [],
        is_visible: false,
        message: "Draft launch plan",
      })
    })
    expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false })
    expect(screen.getByTestId("task-conversation-panel")).toHaveAttribute(
      "data-show-task-actions",
      "true",
    )
    expect(screen.getByTestId("task-conversation-panel")).toHaveAttribute(
      "data-show-token-usage",
      "true",
    )
    expect(screen.getByTestId("task-conversation-panel")).toHaveAttribute(
      "data-show-task-files",
      "true",
    )
  })

  it("opens a historical run from the shared Runs popover", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [
        {
          id: 15,
          task_id: 760,
          status: "completed",
          is_preview: false,
          task_title: "What If Studio: Fix the subtitles",
          message: "Fix the subtitles",
          created_at: "2026-07-17T16:32:08Z",
          completed_at: "2026-07-17T16:33:08Z",
        },
      ],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })

    render(<WorkforceRunPage />)

    await selectRunFromHistory(/What If Studio: Fix the subtitles/)

    expect(setTaskIdMock).toHaveBeenCalledWith(760, { navigate: false })
    expect(screen.getByTestId("task-conversation-panel")).toBeInTheDocument()
  })

  it("restores a historical task from the shared run query", async () => {
    searchParamsMock.set("run", "15")
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    getWorkforceRunMock.mockResolvedValueOnce({
      id: 15,
      task_id: 760,
      status: "completed",
      is_preview: false,
      task_title: "What If Studio: Fix the subtitles",
      message: "Fix the subtitles",
      created_at: "2026-07-17T16:32:08Z",
      completed_at: "2026-07-17T16:33:08Z",
    })

    render(<WorkforceRunPage />)

    await waitFor(() => {
      expect(getWorkforceRunMock).toHaveBeenCalledWith("42", "15")
      expect(setTaskIdMock).toHaveBeenCalledWith(760, { navigate: false })
    })
    expect(screen.getByTestId("task-conversation-panel")).toBeInTheDocument()
  })

  it("switches Flow, Agent, and file content inside one persistent inspector", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [{
        id: 15,
        task_id: 760,
        status: "completed",
        is_preview: false,
        task_title: "What If Studio run",
        message: "Generate the film",
        created_at: "2026-07-17T16:32:08Z",
        completed_at: "2026-07-17T16:33:08Z",
      }],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })
    getWorkforceAgentExecutionMock.mockResolvedValueOnce({
      task_id: 760,
      worker_task_id: "agent_17_run",
      agent_id: 17,
      agent_name: "Editor Agent",
      worker_alias: "Editor",
      status: "completed",
      trace_events: [{
        event_id: "worker-start",
        event_type: "react_task_start",
        timestamp: Date.now(),
        data: { step_name: "Edit video" },
      }, {
        event_id: "worker-completion",
        event_type: "task_completion",
        timestamp: Date.now() + 1,
        data: { result: { content: "The final film is ready for delivery." } },
      }],
    })

    const view = render(<WorkforceRunPage />)

    await selectRunFromHistory(/What If Studio run/)
    expect(getWorkforceAgentExecutionMock).not.toHaveBeenCalled()
    const conversationPanel = screen.getByTestId("task-conversation-panel")

    fireEvent.click(screen.getByRole("button", { name: "workforces.canvas.title" }))
    expect(screen.getByTestId("workforce-run-inspector")).toHaveAttribute("data-mode", "flow")
    expect(screen.getByTestId("workforce-run-inspector").parentElement).toHaveStyle({ width: "35%" })
    expect(screen.getByTestId("task-conversation-panel")).toBe(conversationPanel)

    fireEvent.click(screen.getByRole("button", { name: "View Editor execution" }))

    await waitFor(() => {
      expect(getWorkforceAgentExecutionMock).toHaveBeenCalledWith(
        "42",
        760,
        "agent_17_run",
      )
    })
    expect(await screen.findByText("Editor")).toBeInTheDocument()
    expect(screen.getByText("The final film is ready for delivery.")).toBeInTheDocument()
    expect(screen.getByTestId("workforce-run-inspector")).toHaveAttribute("data-mode", "agent")
    expect(screen.getByTestId("workforce-run-inspector").parentElement).toHaveStyle({ width: "35%" })

    workforceAppState.filePreview = {
      isOpen: true,
      fileId: "final-film.mp4",
      fileName: "final-film.mp4",
      viewMode: "preview",
    }
    view.rerender(<WorkforceRunPage />)

    await waitFor(() => {
      expect(screen.getByTestId("workforce-run-inspector")).toHaveAttribute("data-mode", "file")
    })
    expect(screen.getByTestId("workforce-run-inspector").parentElement).toHaveStyle({ width: "50%" })
    expect(screen.getByTestId("workforce-file-preview")).toBeInTheDocument()
    expect(screen.queryByText("Editor")).not.toBeInTheDocument()
  })

  it("shows an error toast when a file download fails", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    apiRequestMock.mockResolvedValueOnce({ ok: false, statusText: "Gone" })
    workforceAppState.filePreview = {
      isOpen: true,
      fileId: "missing-file",
      fileName: "missing.mp4",
      viewMode: "preview",
    }

    render(<WorkforceRunPage />)

    fireEvent.click(await screen.findByRole("button", { name: "Download file" }))

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith("workforces.errors.download")
    })
  })

  it("marks an orphaned Agent execution interrupted after its parent run stops", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValueOnce({
      items: [{
        id: 15,
        task_id: 760,
        status: "completed",
        is_preview: false,
        task_title: "What If Studio run",
        message: "Generate the film",
        created_at: "2026-07-17T16:32:08Z",
        completed_at: "2026-07-17T16:33:08Z",
      }],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })
    getWorkforceAgentExecutionMock.mockResolvedValueOnce({
      task_id: 760,
      worker_task_id: "agent_17_run",
      agent_id: 17,
      agent_name: "Editor Agent",
      worker_alias: "Editor",
      status: "running",
      trace_events: [],
    })
    workforceAppState.currentTask = { id: "760", status: "completed" }

    render(<WorkforceRunPage />)

    await selectRunFromHistory(/What If Studio run/)
    fireEvent.click(screen.getByRole("button", { name: "View Editor execution" }))

    expect(await screen.findByText("workforces.status.interrupted")).toBeInTheDocument()
  })

  it("tests a draft workforce from the editor preview", async () => {
    getWorkforceMock.mockResolvedValueOnce({
      ...workforceDetail,
      status: "draft",
    })
    runWorkforceMock.mockResolvedValueOnce({
      workforce_run_id: 6,
      task_id: 100,
      status: "running",
      redirect_url: "/task/100",
    })

    render(<WorkforceDetailPage />)

    fireEvent.click(await screen.findByRole("button", { name: "Send Test" }))

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        files: [],
        is_preview: true,
        is_visible: false,
        message: "test message",
      })
    })
  })

  it("shows run history in the detail page runs tab and opens a run", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listWorkforceRunsMock.mockResolvedValue({
      items: [
        {
          id: 9,
          task_id: 99,
          status: "completed",
          is_preview: false,
          task_title: "Launch Workforce: draft plan",
          message: "draft plan",
          created_at: "2026-07-19T10:00:00Z",
          completed_at: "2026-07-19T10:03:00Z",
        },
      ],
      total: 1,
      page: 1,
      size: 20,
      pages: 1,
    })

    render(<WorkforceDetailPage />)

    fireEvent.click(await screen.findByRole("button", { name: /workforces.runs.title/ }))

    const runRow = await screen.findByText("Launch Workforce: draft plan")
    expect(listWorkforceRunsMock).toHaveBeenCalledWith("42", { page: 1, size: 20 })

    fireEvent.click(runRow)
    expect(routerPushMock).toHaveBeenCalledWith("/workforces/42/run?run=9")
  })

  it("opens a past run on the run page via the run query param", async () => {
    searchParamsMock.set("run", "9")
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    getWorkforceRunMock.mockResolvedValueOnce({
      id: 9,
      task_id: 99,
      status: "completed",
      is_preview: false,
      task_title: "Launch Workforce: draft plan",
      message: "draft plan",
      created_at: "2026-07-19T10:00:00Z",
      completed_at: "2026-07-19T10:03:00Z",
    })

    render(<WorkforceRunPage />)

    await waitFor(() => {
      expect(getWorkforceRunMock).toHaveBeenCalledWith("42", "9")
    })
    await waitFor(() => {
      expect(setTaskIdMock).toHaveBeenCalledWith(99, { navigate: false })
    })
    // openRun must keep ?run= authoritative regardless of the opening path.
    expect(routerReplaceMock).toHaveBeenCalledWith("/workforces/42/run?run=9", {
      scroll: false,
    })
    expect(await screen.findByTestId("task-conversation-panel")).toBeInTheDocument()
  })

  it("keeps the current manager visible when it is hidden from agent options", async () => {
    getWorkforceMock.mockResolvedValueOnce(workforceDetail)
    listAgentOptionsMock.mockResolvedValueOnce([
      {
        id: 8,
        name: "Worker Agent",
        description: null,
        logo_url: null,
        status: "published",
      },
    ])

    render(<WorkforceDetailPage />)

    expect((await screen.findAllByText("Manager Agent")).length).toBeGreaterThan(0)
    fireEvent.click(screen.getByText("workforces.actions.change"))
    expect((await screen.findByText("Worker Agent")).closest("button")).toBeInTheDocument()
    expect(screen.getAllByText("Manager Agent").length).toBeGreaterThan(0)
  })

  it("preserves worker sort order when saving member details", async () => {
    const worker = {
      id: 100,
      agent: {
        id: 8,
        name: "Worker Agent",
        description: null,
        logo_url: null,
        status: "published",
      },
      alias: "Researcher",
      assignment_instructions: "Research launch tasks",
      source_type: "existing" as const,
      template_id: null,
      enabled: true,
      sort_order: 3,
      canvas_position: null,
      created_at: null,
      updated_at: null,
    }
    getWorkforceMock.mockResolvedValueOnce({ ...workforceDetail, workers: [worker] })
    listAgentOptionsMock.mockResolvedValueOnce([])
    updateWorkforceAgentMock.mockResolvedValueOnce(worker)

    render(<WorkforceDetailPage />)

    fireEvent.click(await screen.findByText("Researcher"))
    const dialog = await screen.findByRole("dialog")
    fireEvent.change(within(dialog).getByDisplayValue("Research launch tasks"), {
      target: { value: "Write the launch report" },
    })
    fireEvent.click(within(dialog).getByText("common.done"))

    await waitFor(() => {
      expect(updateWorkforceAgentMock).toHaveBeenCalledWith(
        "42",
        100,
        expect.objectContaining({
          assignment_instructions: "Write the launch report",
          sort_order: 3,
        }),
      )
    })
  })

  it("keeps the detail page visible while refreshing after adding a worker", async () => {
    const agentOptions = [
      {
        id: 8,
        name: "Worker Agent",
        description: null,
        logo_url: null,
        status: "published",
      },
    ]
    let resolveReload: (value: WorkforceDetail) => void
    const reload = new Promise<WorkforceDetail>((resolve) => {
      resolveReload = resolve
    })
    getWorkforceMock
      .mockResolvedValueOnce(workforceDetail)
      .mockReturnValueOnce(reload)
    listAgentOptionsMock
      .mockResolvedValueOnce(agentOptions)
      .mockResolvedValueOnce(agentOptions)
    addWorkforceAgentMock.mockResolvedValueOnce({ id: 101 })

    render(<WorkforceDetailPage />)

    expect((await screen.findAllByText("Launch Workforce")).length).toBeGreaterThan(0)
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getByDisplayValue("Launch Workforce"), {
      target: { value: "Unsaved Workforce Name" },
    })
    fireEvent.click(screen.getByText("workforces.actions.addAgent"))
    fireEvent.click((await screen.findByText("Worker Agent")).closest("button")!)

    await waitFor(() => {
      expect(addWorkforceAgentMock).toHaveBeenCalledWith("42", {
        agent_id: 8,
        alias: undefined,
        assignment_instructions: "Worker Agent",
        enabled: true,
        sort_order: 1,
        source_type: "existing",
      })
    })
    expect(screen.queryByText("workforces.loading.detail")).not.toBeInTheDocument()
    expect(screen.getByDisplayValue("Unsaved Workforce Name")).toBeInTheDocument()

    resolveReload!({
      ...workforceDetail,
      workers: [
        {
          id: 100,
          agent: agentOptions[0],
          alias: null,
          assignment_instructions: "Research launch tasks",
          source_type: "existing",
          template_id: null,
          enabled: true,
          sort_order: 1,
          canvas_position: null,
          created_at: null,
          updated_at: null,
        },
      ],
    })
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    })
    expect(await screen.findByText("Worker Agent")).toBeInTheDocument()
  })

})
