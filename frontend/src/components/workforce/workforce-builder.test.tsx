/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const createWorkforceMock = vi.hoisted(() => vi.fn())
const listAgentOptionsMock = vi.hoisted(() => vi.fn())
const runWorkforcePreviewMock = vi.hoisted(() => vi.fn())
const getWorkforceMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
const routerReplaceMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const searchParamsMock = vi.hoisted(() => new URLSearchParams())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("next/navigation", () => ({
  useParams: () => ({}),
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => searchParamsMock,
}))

vi.mock("next/link", () => ({
  default: ({ children, href }: { children: React.ReactNode; href: string }) => <a href={href}>{children}</a>,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en-US", t: translateMock }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: vi.fn(),
    setTaskId: vi.fn(),
    closeFilePreview: vi.fn(),
    dispatch: vi.fn(),
    state: { currentTask: null, traceEvents: [], filePreview: { isOpen: false, fileId: "", fileName: "", viewMode: "preview" } },
  }),
}))

vi.mock("@/components/task/task-conversation-panel", () => ({
  TaskConversationPanel: ({ onSend }: { onSend?: (message: string) => void }) => (
    <div data-testid="task-conversation-panel">
      <button onClick={() => onSend?.("test message")}>Send Test</button>
    </div>
  ),
}))

vi.mock("@/components/build/agent-triggers-dialog", () => ({
  AgentTriggersDialog: () => null,
}))

vi.mock("@/lib/workforces-api", () => ({
  createWorkforce: createWorkforceMock,
  listAgentOptions: listAgentOptionsMock,
  runWorkforcePreview: runWorkforcePreviewMock,
  getWorkforce: getWorkforceMock,
  runWorkforce: runWorkforceMock,
}))

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import { WorkforceBuilder } from "./workforce-builder"

describe("WorkforceBuilder — create mode (no workforceId)", () => {
  beforeEach(() => {
    createWorkforceMock.mockReset()
    listAgentOptionsMock.mockReset().mockResolvedValue([
      {
        id: 7,
        name: "Project Coordinator",
        description: "Coordinates the workforce",
        logo_url: null,
        status: "published",
      },
      {
        id: 8,
        name: "Web Researcher",
        description: "Gathers the web",
        logo_url: null,
        status: "published",
      },
      {
        id: 9,
        name: "Silent Analyst",
        description: null,
        logo_url: null,
        status: "published",
      },
    ])
    routerReplaceMock.mockReset()
    runWorkforcePreviewMock.mockReset()
    getWorkforceMock.mockReset()
    runWorkforceMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("renders the same builder shell as the detail page — Configure/Canvas tabs and an Unsaved badge, no upfront form", async () => {
    render(<WorkforceBuilder />)

    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    expect(screen.getByText("workforces.create.unsavedBadge")).toBeInTheDocument()
    expect(screen.getByText("workforces.detail.configure")).toBeInTheDocument()
    expect(screen.getByText("workforces.canvas.title")).toBeInTheDocument()
    expect(screen.queryByText("workforces.runs.title")).not.toBeInTheDocument()

    // Canvas tab shows the empty-state placeholders immediately.
    fireEvent.click(screen.getByText("workforces.canvas.title"))
    expect(screen.getByText("workforces.canvas.chooseLead.title")).toBeInTheDocument()
    expect(screen.getByText("workforces.canvas.addFirstAgent.title")).toBeInTheDocument()
  })

  it("unlocks the test panel once a manager and a worker are picked, without saving the workforce", async () => {
    runWorkforcePreviewMock.mockResolvedValueOnce({
      workforce_run_id: 1,
      task_id: 42,
      status: "running",
      redirect_url: "/task/42",
    })

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    expect(screen.getByText("workforces.run.createToTest")).toBeInTheDocument()
    expect(screen.queryByTestId("task-conversation-panel")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("workforces.canvas.title"))
    fireEvent.click(screen.getByText("workforces.canvas.chooseLead.title"))
    fireEvent.click(await screen.findByText("Project Coordinator"))

    // Manager alone isn't enough — still needs a worker to delegate to.
    expect(screen.getByText("workforces.run.createToTest")).toBeInTheDocument()

    fireEvent.click(screen.getByText("workforces.canvas.addFirstAgent.title"))
    fireEvent.click(await screen.findByText("Web Researcher"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    expect(await screen.findByTestId("task-conversation-panel")).toBeInTheDocument()
    expect(createWorkforceMock).not.toHaveBeenCalled()

    fireEvent.click(screen.getByText("Send Test"))

    await waitFor(() => {
      expect(runWorkforcePreviewMock).toHaveBeenCalledWith({
        name: undefined,
        description: undefined,
        manager_agent_id: 7,
        workers: [{ agent_id: 8, alias: undefined, assignment_instructions: "Gathers the web" }],
        message: "test message",
        files: [],
      })
    })
    expect(createWorkforceMock).not.toHaveBeenCalled()
  })

  it("picks a lead and a worker via the canvas, then creates the workforce and replaces the URL", async () => {
    const created = { id: 55, name: "Launch Team", status: "draft" }
    createWorkforceMock.mockResolvedValueOnce(created)

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.canvas.title"))

    fireEvent.click(screen.getByText("workforces.canvas.chooseLead.title"))
    fireEvent.click(await screen.findByText("Project Coordinator"))

    fireEvent.click(screen.getByText("workforces.canvas.addFirstAgent.title"))
    fireEvent.click(await screen.findByText("Web Researcher"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    // Fill the name in the Configure tab (also editable in create mode).
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    const nameInput = screen.getAllByRole("textbox")[0]
    fireEvent.change(nameInput, { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())
    fireEvent.click(createButton)

    await waitFor(() => {
      expect(createWorkforceMock).toHaveBeenCalledWith({
        name: "Launch Team",
        description: undefined,
        manager_agent_id: 7,
        workers: [expect.objectContaining({ agent_id: 8, assignment_instructions: "Gathers the web" })],
      })
    })
    expect(routerReplaceMock).toHaveBeenCalledWith("/workforces/55")
  })

  it("starts a fresh run against the saved workforce after Create, instead of continuing the stale pre-save preview", async () => {
    const created = { id: 55, name: "Launch Team", status: "draft" }
    createWorkforceMock.mockResolvedValueOnce(created)
    runWorkforcePreviewMock.mockResolvedValueOnce({
      workforce_run_id: 1,
      task_id: 42,
      status: "running",
      redirect_url: "/task/42",
    })
    getWorkforceMock.mockResolvedValue({
      id: 55,
      name: "Launch Team",
      description: null,
      status: "draft",
      manager: {
        id: 7,
        name: "Project Coordinator",
        description: "Coordinates the workforce",
        logo_url: null,
        status: "published",
      },
      workers: [
        {
          id: 1,
          agent: {
            id: 8,
            name: "Web Researcher",
            description: "Gathers the web",
            logo_url: null,
            status: "published",
          },
          alias: null,
          assignment_instructions: "Gathers the web",
          source_type: "existing",
          template_id: null,
          enabled: true,
          sort_order: 1,
          canvas_position: null,
          created_at: null,
          updated_at: null,
        },
      ],
      canvas_layout: null,
      scope_type: "user",
      scope_id: "1",
      owner_user_id: 1,
      created_at: null,
      updated_at: null,
    })
    runWorkforceMock.mockResolvedValueOnce({
      workforce_run_id: 2,
      task_id: 99,
      status: "running",
      redirect_url: "/task/99",
    })

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.canvas.title"))
    fireEvent.click(screen.getByText("workforces.canvas.chooseLead.title"))
    fireEvent.click(await screen.findByText("Project Coordinator"))
    fireEvent.click(screen.getByText("workforces.canvas.addFirstAgent.title"))
    fireEvent.click(await screen.findByText("Web Researcher"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    // Test the draft before saving -- opens an ephemeral preview run (task 42).
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => expect(runWorkforcePreviewMock).toHaveBeenCalledOnce())

    // Fill the name and save.
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    const nameInput = screen.getAllByRole("textbox")[0]
    fireEvent.change(nameInput, { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())
    fireEvent.click(createButton)

    await waitFor(() => expect(getWorkforceMock).toHaveBeenCalledWith("55"))

    // Sending another test message after saving must start a fresh run
    // against the saved workforce (via runWorkforce), not continue chatting
    // into the stale, frozen pre-save preview run (task 42).
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith(
        "55",
        expect.objectContaining({ message: "test message" }),
      )
    })
    expect(runWorkforcePreviewMock).toHaveBeenCalledOnce()
  })

  it("does not count a worker's auto-filled name/id placeholder as a written delegation rule in Get Started", async () => {
    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.canvas.title"))
    fireEvent.click(screen.getByText("workforces.canvas.chooseLead.title"))
    fireEvent.click(await screen.findByText("Project Coordinator"))

    // Silent Analyst has no description, so the one-click add falls all the
    // way back to the agent's bare name as assignment_instructions.
    fireEvent.click(screen.getByText("workforces.canvas.addFirstAgent.title"))
    fireEvent.click(await screen.findByText("Silent Analyst"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    // name + lead + agents done; delegation must NOT count the placeholder.
    expect(await screen.findByText("3/6")).toBeInTheDocument()
    expect(screen.queryByText("4/6")).not.toBeInTheDocument()
  })
})
