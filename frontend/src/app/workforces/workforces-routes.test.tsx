/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const getWorkforceMock = vi.hoisted(() => vi.fn())
const listAgentOptionsMock = vi.hoisted(() => vi.fn().mockResolvedValue([]))
const listWorkforcesMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const addWorkforceAgentMock = vi.hoisted(() => vi.fn())
const archiveWorkforceMock = vi.hoisted(() => vi.fn())
const publishWorkforceMock = vi.hoisted(() => vi.fn())
const removeWorkforceAgentMock = vi.hoisted(() => vi.fn())
const unpublishWorkforceMock = vi.hoisted(() => vi.fn())
const updateWorkforceMock = vi.hoisted(() => vi.fn())
const updateWorkforceAgentMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const paramsMock = vi.hoisted(() => ({ id: "42" as string | string[] | undefined }))
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
  useRouter: () => ({ push: routerPushMock }),
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
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: vi.fn(),
    setTaskId: vi.fn(),
    closeFilePreview: vi.fn(),
    dispatch: vi.fn(),
  }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: ({ onSend }: { onSend?: (message: string) => void }) => (
    <div data-testid="task-conversation-panel">
      <button onClick={() => onSend?.("test message")}>Send Test</button>
    </div>
  ),
}))

vi.mock("@/lib/workforces-api", () => ({
  addWorkforceAgent: addWorkforceAgentMock,
  archiveWorkforce: archiveWorkforceMock,
  getWorkforce: getWorkforceMock,
  listAgentOptions: listAgentOptionsMock,
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
    error: vi.fn(),
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
  manager_instructions: null,
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

describe("workforce route entry points", () => {
  beforeEach(() => {
    getWorkforceMock.mockReset()
    listAgentOptionsMock.mockReset().mockResolvedValue([])
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
    paramsMock.id = "42"
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

  it("keeps list run actions disabled for non-active workforces", async () => {
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

  it("runs a workforce and redirects to the created task", async () => {
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
      const submitBtn = container.querySelector(".rounded-2xl button:not([disabled])")
      expect(submitBtn).not.toBeNull()
    })

    const submitBtn = container.querySelector(".rounded-2xl button:not([disabled])")
    fireEvent.click(submitBtn!)

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith("42", {
        message: "Draft launch plan",
      })
    })
    expect(routerPushMock).toHaveBeenCalledWith("/task/99")
  })

  it("shows a run disabled reason for non-active workforces", async () => {
    getWorkforceMock.mockResolvedValueOnce({
      ...workforceDetail,
      status: "draft",
    })

    render(<WorkforceRunPage />)

    expect(await screen.findByText("Launch Workforce")).toBeInTheDocument()
    expect(screen.getByText("workforces.run.inactiveDisabled")).toBeInTheDocument()
    expect(screen.queryByText("workforces.actions.runWorkforce")).not.toBeInTheDocument()
    expect(runWorkforceMock).not.toHaveBeenCalled()
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

    expect(await screen.findByRole("heading", { name: "Launch Workforce" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "Manager Agent" })).toHaveAttribute(
      "value",
      "7",
    )
  })

  it("allows clearing and replacing worker sort order before saving", async () => {
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
    updateWorkforceAgentMock.mockResolvedValueOnce({ ...worker, sort_order: 12 })

    render(<WorkforceDetailPage />)

    const sortInput = (await screen.findByDisplayValue("3")) as HTMLInputElement
    fireEvent.change(sortInput, { target: { value: "" } })
    expect(sortInput.value).toBe("")
    fireEvent.change(sortInput, { target: { value: "12" } })
    fireEvent.click(screen.getByText("workforces.actions.saveWorker"))

    await waitFor(() => {
      expect(updateWorkforceAgentMock).toHaveBeenCalledWith(
        "42",
        100,
        expect.objectContaining({ sort_order: 12 }),
      )
    })
  })

  it("keeps worker sort order integer-only when saving", async () => {
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

    const sortInput = (await screen.findByDisplayValue("3")) as HTMLInputElement
    fireEvent.change(sortInput, { target: { value: "1.5" } })
    fireEvent.click(screen.getByText("workforces.actions.saveWorker"))

    await waitFor(() => {
      expect(updateWorkforceAgentMock).toHaveBeenCalledWith(
        "42",
        100,
        expect.objectContaining({ sort_order: 3 }),
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

    const { container } = render(<WorkforceDetailPage />)

    expect(await screen.findByRole("heading", { name: "Launch Workforce" })).toBeInTheDocument()
    fireEvent.change(screen.getByDisplayValue("Launch Workforce"), {
      target: { value: "Unsaved Workforce Name" },
    })
    fireEvent.change(screen.getByLabelText("workforces.workers.chooseAgent"), {
      target: { value: "8" },
    })
    const textareas = container.querySelectorAll("textarea")
    fireEvent.change(textareas[textareas.length - 1], {
      target: { value: "Research launch tasks" },
    })
    fireEvent.click(screen.getByText("workforces.actions.addWorker"))

    await waitFor(() => {
      expect(addWorkforceAgentMock).toHaveBeenCalled()
    })
    expect(screen.queryByText("workforces.loading.detail")).not.toBeInTheDocument()
    expect(screen.getByRole("heading", { name: "Launch Workforce" })).toBeInTheDocument()

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
    await screen.findByText("workforces.messages.workerAdded")
    expect(screen.getByDisplayValue("Unsaved Workforce Name")).toBeInTheDocument()
  })

})
