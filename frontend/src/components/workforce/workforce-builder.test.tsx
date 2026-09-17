/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
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
const sendMessageMock = vi.hoisted(() => vi.fn())

// Spied rather than mocked via vi.mock: handleCreate now updates the address
// bar via the native History API instead of router.replace/push, precisely
// to avoid the cross-route-segment remount that a Next.js navigation would
// trigger (PR review round 8, finding #1 REOPENED). Real jsdom replaceState
// would also work, but no-opping it keeps this test's window.location
// stable regardless of run order. replaceState, not pushState (PR review
// round 9, MINOR-1): this is a one-time create->save transition, not a new
// history entry the user would ever want Back to land on.
const historyReplaceStateSpy = vi.spyOn(window.history, "replaceState").mockImplementation(() => {})

vi.mock("next/navigation", () => ({
  useParams: () => ({}),
  useRouter: () => ({ push: routerPushMock, replace: routerReplaceMock }),
  useSearchParams: () => searchParamsMock,
}))

vi.mock("next/link", () => ({
  default: ({ children, href, onClick }: { children: React.ReactNode; href: string; onClick?: (e: React.MouseEvent) => void }) => (
    <a href={href} onClick={onClick}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ locale: "en-US", t: translateMock }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    sendMessage: sendMessageMock,
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
    sendMessageMock.mockReset().mockResolvedValue(undefined)
    historyReplaceStateSpy.mockClear()
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
        // PR review round 7, finding #4: the picker no longer defaults a new
        // member's instructions to the agent's own description.
        workers: [{ agent_id: 8, alias: undefined, assignment_instructions: "Web Researcher" }],
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
        // PR review round 7, finding #4: no longer defaults to the agent's
        // own description.
        workers: [expect.objectContaining({ agent_id: 8, assignment_instructions: "Web Researcher" })],
      })
    })
    // History API, not router.replace/push (PR review round 8, finding #1
    // REOPENED) -- see historyReplaceStateSpy's comment above.
    expect(historyReplaceStateSpy).toHaveBeenCalledWith({}, "", "/workforces/55")
    expect(routerReplaceMock).not.toHaveBeenCalled()
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

  it("does not unmount the test panel on the create->save transition (PR review round 7 finding #1, reopened round 8)", async () => {
    // Regression test, twice over. Round 7's fix (a ref-suppressed silent
    // load()) only helps if THIS component instance survives Create -- but
    // /workforces/new and /workforces/[id] are separate route segments with
    // no shared layout, so a real router.replace/push there tears down this
    // whole instance (taking the ref with it) and mounts a fresh one under
    // [id], reopening the exact bug (PR review round 8, finding #1). A mocked
    // router can't simulate that teardown, which is exactly why round 7's
    // version of this test was a false green -- so this version additionally
    // asserts router.replace/push are never called at all for this
    // transition (the fix updates the URL via the native History API
    // instead, verified via historyReplaceStateSpy below), which is the one
    // invariant a mocked router *can* prove: if this ever regresses back to
    // router.replace/push, this assertion catches it even though the mock
    // can't simulate the resulting unmount.
    //
    // The rest of this test (the deferred getWorkforce + node-identity check)
    // still guards the same-instance mechanism round 7 fixed: setLocalId used
    // to let the [localId] effect fire a non-silent load(), setting
    // loading=true and hitting the top-level `isEditMode && loading` early
    // return -- unmounting the tree until the redundant getWorkforce call
    // resolved. getWorkforce is held pending here so that window is directly
    // observable instead of racing real timers/microtasks.
    const created = { id: 55, name: "Launch Team", status: "draft" }
    createWorkforceMock.mockResolvedValueOnce(created)
    let resolveGetWorkforce: (value: unknown) => void = () => {}
    getWorkforceMock.mockImplementation(
      () => new Promise((resolve) => { resolveGetWorkforce = resolve }),
    )

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

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    const nameInput = screen.getAllByRole("textbox")[0]
    fireEvent.change(nameInput, { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const panelBeforeCreate = await screen.findByTestId("task-conversation-panel")

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())

    await act(async () => {
      fireEvent.click(createButton)
      // Flush handleCreate's continuation (setWorkforce/setLocalId) and the
      // [localId] effect's synchronous portion -- if load() isn't silent,
      // setLoading(true) commits here, before getWorkforce's promise (held
      // pending above) ever resolves.
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(getWorkforceMock).toHaveBeenCalledWith("55")
    expect(screen.queryByText("workforces.loading.detail")).not.toBeInTheDocument()
    expect(screen.getByTestId("task-conversation-panel")).toBe(panelBeforeCreate)
    expect(historyReplaceStateSpy).toHaveBeenCalledWith({}, "", "/workforces/55")
    expect(routerReplaceMock).not.toHaveBeenCalled()
    expect(routerPushMock).not.toHaveBeenCalled()

    await act(async () => {
      resolveGetWorkforce({
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
    })

    expect(screen.getByTestId("task-conversation-panel")).toBe(panelBeforeCreate)
  })

  it("does not mark the test step done for a config invalidated while a continuation send is in flight (PR review round 7, finding #2)", async () => {
    // Regression test: the !taskId branch re-checks previewGenerationRef
    // after its await before touching state, but the else branch (continuing
    // an already-pinned preview run via sendMessage) had no equivalent
    // check, so an invalidatePreviewRun() firing mid-flight (e.g. editing
    // the draft while a test message is in the air) still marked the Get
    // Started "test" step done for the now-discarded configuration.
    runWorkforcePreviewMock.mockResolvedValueOnce({
      workforce_run_id: 1,
      task_id: 42,
      status: "running",
      redirect_url: "/task/42",
    })
    let resolveSend: () => void = () => {}
    sendMessageMock.mockImplementationOnce(
      () => new Promise<void>((resolve) => { resolveSend = resolve }),
    )

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

    // First send pins the preview run (task 42) and marks "test" done.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => expect(runWorkforcePreviewMock).toHaveBeenCalledOnce())

    // Expand the Get Started checklist so the "test" step's done state is
    // visible via GetStartedChecklist's step rows, not just the badge count.
    fireEvent.click(screen.getByText("workforces.getStarted.title"))
    expect(screen.getByText("workforces.getStarted.steps.test")).toHaveClass("text-foreground")

    // Second send reuses task 42 via the else/continuation branch. Its
    // sendMessage call won't resolve until we say so below.
    fireEvent.click(screen.getByText("Send Test"))

    // Edit the still-unsaved draft while that send is in flight --
    // invalidatePreviewRun() resets hasSentTestMessage synchronously.
    fireEvent.click(screen.getByText("workforces.actions.addAgent"))
    fireEvent.click(await screen.findByText("Silent Analyst"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })
    expect(screen.getByText("workforces.getStarted.steps.test")).toHaveClass("text-muted-foreground")

    // Now let the stale send resolve. Without the fix, its unconditional
    // setHasSentTestMessage(true) would re-mark "test" done for the config
    // that was just invalidated.
    await act(async () => {
      resolveSend()
      await Promise.resolve()
    })

    expect(screen.getByText("workforces.getStarted.steps.test")).toHaveClass("text-muted-foreground")
  })

  it("does not permanently block Create after a failed attempt (creatingRef resets so a retry can proceed)", async () => {
    // Regression coverage for the creatingRef guard added alongside the
    // `saving` state check (PR review round 7, finding #7 -- see
    // handleCreate): it must reset in the finally block on every path, not
    // just success, or one failed Create would silently and permanently
    // disable every future click without the `saving`/`!canCreate` UI ever
    // reflecting why.
    createWorkforceMock.mockRejectedValueOnce(new Error("boom"))
    createWorkforceMock.mockResolvedValueOnce({ id: 55, name: "Launch Team", status: "draft" })

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

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    const nameInput = screen.getAllByRole("textbox")[0]
    fireEvent.change(nameInput, { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())

    fireEvent.click(createButton)
    await waitFor(() => expect(createWorkforceMock).toHaveBeenCalledOnce())
    await waitFor(() => expect(createButton).not.toBeDisabled())

    fireEvent.click(createButton)
    await waitFor(() => expect(createWorkforceMock).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(historyReplaceStateSpy).toHaveBeenCalledWith({}, "", "/workforces/55"))
  })

  it("starts a fresh preview run instead of continuing a stale one when the unsaved draft is edited", async () => {
    runWorkforcePreviewMock
      .mockResolvedValueOnce({
        workforce_run_id: 1,
        task_id: 42,
        status: "running",
        redirect_url: "/task/42",
      })
      .mockResolvedValueOnce({
        workforce_run_id: 2,
        task_id: 43,
        status: "running",
        redirect_url: "/task/43",
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

    // Test the unsaved draft -- pins an ephemeral preview run (task 42).
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => expect(runWorkforcePreviewMock).toHaveBeenCalledTimes(1))

    // Edit the still-unsaved draft (add another worker) before saving.
    fireEvent.click(screen.getByText("workforces.actions.addAgent"))
    fireEvent.click(await screen.findByText("Silent Analyst"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    // A later test message must start a brand-new preview run, proving the
    // edit invalidated the one pinned before it -- not silently continue
    // chatting into task 42 with the now-stale two-worker config.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => {
      expect(runWorkforcePreviewMock).toHaveBeenCalledTimes(2)
    })
  })

  it("does not fire a second concurrent preview-creation request when a draft edit invalidates the first one mid-flight", async () => {
    let resolveFirstPreview: (value: unknown) => void = () => {}
    runWorkforcePreviewMock
      .mockImplementationOnce(() => new Promise((resolve) => { resolveFirstPreview = resolve }))
      .mockResolvedValueOnce({
        workforce_run_id: 2,
        task_id: 43,
        status: "running",
        redirect_url: "/task/43",
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

    // Start a test send whose network call won't resolve until we say so.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => expect(runWorkforcePreviewMock).toHaveBeenCalledTimes(1))

    // Edit the still-unsaved draft while the first request is in flight --
    // invalidatePreviewRun resets previewTaskIdRef synchronously, which used
    // to also clear the "-1 means a creation request is pending" guard.
    fireEvent.click(screen.getByText("workforces.actions.addAgent"))
    fireEvent.click(await screen.findByText("Silent Analyst"))
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.addMemberTitle")).not.toBeInTheDocument()
    })

    // Sending again before the first request settles must NOT fire a second
    // concurrent runWorkforcePreview call -- it must be a no-op until the
    // first one resolves.
    fireEvent.click(screen.getByText("Send Test"))
    await Promise.resolve()
    expect(runWorkforcePreviewMock).toHaveBeenCalledTimes(1)

    // Let the first (now-stale) request resolve and flush its continuation.
    await act(async () => {
      resolveFirstPreview({
        workforce_run_id: 1,
        task_id: 42,
        status: "running",
        redirect_url: "/task/42",
      })
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // The guard must release afterward -- a further send starts a fresh run.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => {
      expect(runWorkforcePreviewMock).toHaveBeenCalledTimes(2)
    })
  })

  it("does not let a preview response that resolves after Create clobber the reset with the stale task", async () => {
    let resolvePreview: (value: unknown) => void = () => {}
    runWorkforcePreviewMock.mockImplementationOnce(
      () => new Promise((resolve) => { resolvePreview = resolve }),
    )

    const created = { id: 55, name: "Launch Team", status: "draft" }
    createWorkforceMock.mockResolvedValueOnce(created)
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

    // Start a test send whose network call won't resolve until we say so.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => expect(runWorkforcePreviewMock).toHaveBeenCalledOnce())

    // While it's in flight, fill the name and save -- this resets the
    // preview state synchronously, before the pending call above resolves.
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())
    fireEvent.click(createButton)
    await waitFor(() => expect(getWorkforceMock).toHaveBeenCalledWith("55"))

    // Now let the stale in-flight preview call resolve, and flush its
    // continuation to completion before checking what state it left behind
    // -- otherwise the click below could race the continuation instead of
    // strictly following it.
    await act(async () => {
      resolvePreview({
        workforce_run_id: 1,
        task_id: 42,
        status: "running",
        redirect_url: "/task/42",
      })
      await Promise.resolve()
      await Promise.resolve()
      await Promise.resolve()
    })

    // A subsequent test message must still start a fresh run against the
    // saved workforce -- proving the stale resolution above didn't
    // resurrect task 42 as the active preview task.
    fireEvent.click(screen.getByText("Send Test"))
    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith(
        "55",
        expect.objectContaining({ message: "test message" }),
      )
    })
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

  it("confirms before switching to Canvas mid-edit, and only discards the edit if the user agrees", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })

    // Declining the confirm must leave the edit form open on Configure --
    // membersTitle only renders on the Configure panel, never on Canvas.
    confirmSpy.mockReturnValueOnce(false)
    fireEvent.click(screen.getByText("workforces.canvas.title"))
    expect(confirmSpy).toHaveBeenCalledOnce()
    expect(screen.getByDisplayValue("Draft name")).toBeInTheDocument()
    expect(screen.getByText("workforces.detail.membersTitle")).toBeInTheDocument()

    // Agreeing discards the in-progress edit and switches to Canvas.
    confirmSpy.mockReturnValueOnce(true)
    fireEvent.click(screen.getByText("workforces.canvas.title"))
    expect(confirmSpy).toHaveBeenCalledTimes(2)
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.membersTitle")).not.toBeInTheDocument()
    })

    confirmSpy.mockRestore()
  })

  it("does not confirm when switching views without an in-progress details edit", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("workforces.canvas.title"))

    expect(confirmSpy).not.toHaveBeenCalled()
    await waitFor(() => {
      expect(screen.queryByText("workforces.detail.membersTitle")).not.toBeInTheDocument()
    })

    confirmSpy.mockRestore()
  })

  it("confirms before following the back link away from an in-progress details edit", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })

    const backLink = screen.getByRole("link")

    // Declining leaves the navigation cancelled (fireEvent.click returns
    // false when the event's default was prevented) and the edit intact.
    confirmSpy.mockReturnValueOnce(false)
    expect(fireEvent.click(backLink)).toBe(false)
    expect(screen.getByDisplayValue("Draft name")).toBeInTheDocument()

    // Agreeing lets the navigation proceed (default not prevented).
    confirmSpy.mockReturnValueOnce(true)
    expect(fireEvent.click(backLink)).toBe(true)

    confirmSpy.mockRestore()
  })

  it("does not confirm on the back link when the create-mode draft is still untouched", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    const backLink = screen.getByRole("link")
    expect(fireEvent.click(backLink)).toBe(true)
    expect(confirmSpy).not.toHaveBeenCalled()

    confirmSpy.mockRestore()
  })

  it("confirms before following the back link away from an unsaved create-mode draft", async () => {
    // Regression test: nothing in a create-mode draft (name, manager,
    // workers) hits the API until Create is clicked, so navigating away
    // before then used to silently discard it with no warning at all.
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    // Commit a draft name via Configure -> Edit -> Save. This is local
    // create-mode draft state, not an in-progress details *edit* - wait for
    // the save (async, even in create mode) to resolve so isEditingDetails
    // is back to false and doesn't short-circuit the check this test wants
    // to exercise.
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })
    fireEvent.click(screen.getByText("common.save"))
    await waitFor(() => {
      expect(screen.queryByText("common.save")).not.toBeInTheDocument()
    })
    // The DOM update above reflects the config panel's own local
    // editingDetails state; it reports up to the parent's isEditingDetails
    // via a passive effect that can run a tick later. Flush it explicitly
    // so this test isn't racing that propagation.
    await act(async () => {})

    const backLink = screen.getByRole("link")

    confirmSpy.mockReturnValueOnce(false)
    expect(fireEvent.click(backLink)).toBe(false)
    expect(confirmSpy).toHaveBeenCalledWith("workforces.create.discardDraftConfirm")

    confirmSpy.mockReturnValueOnce(true)
    expect(fireEvent.click(backLink)).toBe(true)

    confirmSpy.mockRestore()
  })

  it("registers a beforeunload guard once the create-mode draft has content", async () => {
    const addEventListenerSpy = vi.spyOn(window, "addEventListener")
    const removeEventListenerSpy = vi.spyOn(window, "removeEventListener")

    const { unmount } = render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    expect(addEventListenerSpy).not.toHaveBeenCalledWith("beforeunload", expect.any(Function))

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })
    fireEvent.click(screen.getByText("common.save"))

    await waitFor(() => {
      expect(addEventListenerSpy).toHaveBeenCalledWith("beforeunload", expect.any(Function))
    })

    unmount()
    expect(removeEventListenerSpy).toHaveBeenCalledWith("beforeunload", expect.any(Function))

    addEventListenerSpy.mockRestore()
    removeEventListenerSpy.mockRestore()
  })

  it("registers a beforeunload guard for an in-progress details edit, even with no committed draft yet", async () => {
    // Regression test: hasUnsavedDraft alone (committed draftName/manager/
    // workers) used to gate the beforeunload listener, so typing a name and
    // hitting refresh *before* clicking Save was silently unprotected, even
    // though the in-app back-link guard already covers that exact case via
    // confirmDiscardDetailsEdit.
    const addEventListenerSpy = vi.spyOn(window, "addEventListener")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    expect(addEventListenerSpy).not.toHaveBeenCalledWith("beforeunload", expect.any(Function))

    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })
    // Deliberately not clicking Save - the edit is still in progress.

    await waitFor(() => {
      expect(addEventListenerSpy).toHaveBeenCalledWith("beforeunload", expect.any(Function))
    })

    addEventListenerSpy.mockRestore()
  })

  it("shows only one confirm dialog when both an in-progress details edit and a committed draft exist", async () => {
    // Regression test: chaining confirmDiscardDetailsEdit() and
    // confirmDiscardDraft() with || used to be able to pop two sequential
    // native confirm() dialogs for one click.
    const confirmSpy = vi.spyOn(window, "confirm")

    render(<WorkforceBuilder />)
    await waitFor(() => expect(listAgentOptionsMock).toHaveBeenCalledOnce())

    // Commit a draft name first (hasUnsavedDraft becomes true)...
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })
    fireEvent.click(screen.getByText("common.save"))
    await waitFor(() => {
      expect(screen.queryByText("common.save")).not.toBeInTheDocument()
    })
    await act(async () => {})

    // ...then start a second, never-saved edit (isEditingDetails becomes
    // true again, on top of the already-committed draft).
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Second edit" } })

    const backLink = screen.getByRole("link")
    confirmSpy.mockReturnValueOnce(true)
    fireEvent.click(backLink)

    expect(confirmSpy).toHaveBeenCalledOnce()
    expect(confirmSpy).toHaveBeenCalledWith("workforces.detail.discardEditConfirm")

    confirmSpy.mockRestore()
  })

  it("confirms before creating the workforce while a details edit is in progress", async () => {
    const confirmSpy = vi.spyOn(window, "confirm")
    const created = { id: 55, name: "Draft name", status: "draft" }
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

    // Commit an initial name so Create is enabled, independent of the
    // still-unsaved second edit started below.
    fireEvent.click(screen.getByText("workforces.detail.configure"))
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Launch Team" } })
    fireEvent.click(screen.getByText("common.save"))

    const createButton = screen.getByText("workforces.actions.createTeam")
    await waitFor(() => expect(createButton).not.toBeDisabled())

    // Start a second, never-saved edit.
    fireEvent.click(screen.getByText("common.edit"))
    fireEvent.change(screen.getAllByRole("textbox")[0], { target: { value: "Draft name" } })

    // Declining must not create the workforce at all.
    confirmSpy.mockReturnValueOnce(false)
    fireEvent.click(createButton)
    expect(confirmSpy).toHaveBeenCalledOnce()
    expect(createWorkforceMock).not.toHaveBeenCalled()

    // Agreeing proceeds with Create as usual.
    confirmSpy.mockReturnValueOnce(true)
    fireEvent.click(createButton)
    await waitFor(() => expect(createWorkforceMock).toHaveBeenCalledOnce())

    confirmSpy.mockRestore()
  })
})
