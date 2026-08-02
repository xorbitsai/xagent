/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const translateMock = vi.hoisted(() => (key: string) => key)
const fitViewSpy = vi.hoisted(() => vi.fn())

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

// Only useReactFlow's fitView is wrapped -- @xyflow/react's own initial
// fitView (from the `fitView` prop) still runs against its real internal
// store, so this only observes the imperative re-fit our own effect issues.
vi.mock("@xyflow/react", async () => {
  const actual = await vi.importActual<typeof import("@xyflow/react")>("@xyflow/react")
  return {
    ...actual,
    useReactFlow: () => ({ ...actual.useReactFlow(), fitView: fitViewSpy }),
  }
})

import { WorkforceCanvas } from "./workforce-canvas"
import type { WorkforceAgentSummary, WorkforceWorker } from "@/types/workforce"
import type { WorkforceEditDialogsState } from "./workforce-edit-dialogs"

const manager: WorkforceAgentSummary = {
  id: 7,
  name: "Project Coordinator",
  description: "Coordinates the workforce",
  logo_url: null,
  status: "published",
}

const worker: WorkforceWorker = {
  id: 100,
  agent: {
    id: 8,
    name: "Web Researcher",
    description: "Gathers the web and synthesizes findings",
    logo_url: null,
    status: "published",
  },
  alias: null,
  assignment_instructions: "Research launch tasks",
  source_type: "existing",
  template_id: null,
  enabled: true,
  sort_order: 1,
  canvas_position: null,
  created_at: null,
  updated_at: null,
}

function makeDialogs(overrides: Partial<WorkforceEditDialogsState> = {}): WorkforceEditDialogsState {
  return {
    changeLeadOpen: false,
    setChangeLeadOpen: vi.fn(),
    addMemberOpen: false,
    setAddMemberOpen: vi.fn(),
    selectedWorker: null,
    openMemberDetail: vi.fn(),
    closeMemberDetail: vi.fn(),
    memberAlias: "",
    setMemberAlias: vi.fn(),
    memberInstructions: "",
    setMemberInstructions: vi.fn(),
    handleChangeLead: vi.fn(),
    handleAddMember: vi.fn(),
    handleSaveMember: vi.fn(),
    handleRemoveMember: vi.fn(),
    availableForLead: [],
    availableForMember: [],
    ...overrides,
  }
}

function buildCanvasElement(props: Partial<React.ComponentProps<typeof WorkforceCanvas>>) {
  const dialogs = props.dialogs ?? makeDialogs()
  const onSaveDetails = props.onSaveDetails ?? vi.fn().mockResolvedValue(undefined)
  return {
    element: (
      <WorkforceCanvas
        name={props.name ?? "Launch Workforce"}
        description={props.description ?? ""}
        onSaveDetails={onSaveDetails}
        manager={"manager" in props ? props.manager! : manager}
        workers={props.workers ?? [worker]}
        isArchived={props.isArchived ?? false}
        saving={props.saving ?? false}
        dialogs={dialogs}
        getStartedSteps={props.getStartedSteps ?? []}
        getStartedCollapsed={props.getStartedCollapsed ?? false}
        onToggleGetStarted={props.onToggleGetStarted ?? vi.fn()}
      />
    ),
    dialogs,
    onSaveDetails,
  }
}

function renderCanvas(props: Partial<React.ComponentProps<typeof WorkforceCanvas>> = {}) {
  const { element, dialogs, onSaveDetails } = buildCanvasElement(props)
  const { rerender } = render(element)
  return {
    dialogs,
    onSaveDetails,
    rerender: (nextProps: Partial<React.ComponentProps<typeof WorkforceCanvas>>) =>
      rerender(buildCanvasElement({ ...props, ...nextProps, dialogs, onSaveDetails }).element),
  }
}

describe("WorkforceCanvas", () => {
  afterEach(() => {
    cleanup()
    fitViewSpy.mockClear()
  })

  it("re-fits the canvas when the Get Started checklist is expanded or collapsed after mount", () => {
    const { rerender } = renderCanvas({ getStartedCollapsed: true })
    expect(fitViewSpy).not.toHaveBeenCalled()

    rerender({ getStartedCollapsed: false })
    expect(fitViewSpy).toHaveBeenCalledTimes(1)

    rerender({ getStartedCollapsed: true })
    expect(fitViewSpy).toHaveBeenCalledTimes(2)
  })

  it("clicking the manager node opens the change-lead dialog", () => {
    const { dialogs } = renderCanvas()

    fireEvent.click(screen.getByText("Project Coordinator"))

    expect(dialogs.setChangeLeadOpen).toHaveBeenCalledWith(true)
  })

  it("clicking a worker node opens its member detail dialog", () => {
    const { dialogs } = renderCanvas()

    fireEvent.click(screen.getByText("Web Researcher"))

    expect(dialogs.openMemberDetail).toHaveBeenCalledWith(worker)
  })

  it("clicking the add-agent node opens the add-member dialog", () => {
    const { dialogs } = renderCanvas()

    fireEvent.click(screen.getByText("workforces.actions.addAgent"))

    expect(dialogs.setAddMemberOpen).toHaveBeenCalledWith(true)
  })

  it("does not render the add-agent node and ignores clicks when archived", () => {
    const { dialogs } = renderCanvas({ isArchived: true })

    expect(screen.queryByText("workforces.actions.addAgent")).not.toBeInTheDocument()

    fireEvent.click(screen.getByText("Project Coordinator"))
    fireEvent.click(screen.getByText("Web Researcher"))

    expect(dialogs.setChangeLeadOpen).not.toHaveBeenCalled()
    expect(dialogs.openMemberDetail).not.toHaveBeenCalled()
  })

  it("shows a choose-lead placeholder and an add-first-agent node when nothing is set yet", () => {
    const { dialogs } = renderCanvas({ manager: null, workers: [] })

    expect(screen.getByText("workforces.canvas.chooseLead.title")).toBeInTheDocument()
    expect(screen.getByText("workforces.canvas.addFirstAgent.title")).toBeInTheDocument()

    fireEvent.click(screen.getByText("workforces.canvas.chooseLead.title"))
    expect(dialogs.setChangeLeadOpen).toHaveBeenCalledWith(true)
  })

  it("renders the workforce details as an editable node and saves on blur", () => {
    const { onSaveDetails } = renderCanvas({ name: "Launch Workforce", description: "Coordinate launch work" })

    const nameInput = screen.getByDisplayValue("Launch Workforce")
    fireEvent.change(nameInput, { target: { value: "Renamed Workforce" } })
    fireEvent.blur(nameInput)

    expect(onSaveDetails).toHaveBeenCalledWith({ name: "Renamed Workforce", description: "Coordinate launch work" })
  })

  it("reverts an emptied name on blur instead of saving it", () => {
    const { onSaveDetails } = renderCanvas({ name: "Launch Workforce", description: "Coordinate launch work" })

    const nameInput = screen.getByDisplayValue("Launch Workforce")
    fireEvent.change(nameInput, { target: { value: "   " } })
    fireEvent.blur(nameInput)

    expect(onSaveDetails).not.toHaveBeenCalled()
    expect(screen.getByDisplayValue("Launch Workforce")).toBeInTheDocument()
  })

  it("disables the details node's name/description fields when archived (PR review round 8, F-NEW-2)", () => {
    // Regression test: every other interactive canvas node (manager, worker,
    // add-worker) was already gated on isArchived; the details node's own
    // Input/Textarea were the one exception, letting a user edit an archived
    // workforce's name/description on the canvas even though the backend
    // rejects the resulting PATCH with a 409.
    renderCanvas({
      name: "Launch Workforce",
      description: "Coordinate launch work",
      isArchived: true,
    })

    expect(screen.getByDisplayValue("Launch Workforce")).toBeDisabled()
    expect(screen.getByDisplayValue("Coordinate launch work")).toBeDisabled()
  })

  it("disables the details node's name/description fields while a save is in flight (PR review round 8, F-NEW-3)", () => {
    // Regression test: WorkforceCanvas used to be invoked with no `saving`
    // prop at all, unlike its Configure-panel sibling (which disables its
    // equivalent fields via `disabled={saving}`). Without this, a save
    // response landing after the user typed further changes could silently
    // clobber those keystrokes via the data.name/data.description resync
    // effects -- disabling the fields during the window closes that off by
    // construction, since a disabled field can't receive further input.
    renderCanvas({
      name: "Launch Workforce",
      description: "Coordinate launch work",
      saving: true,
    })

    expect(screen.getByDisplayValue("Launch Workforce")).toBeDisabled()
    expect(screen.getByDisplayValue("Coordinate launch work")).toBeDisabled()
  })

  it("re-enables the details node's fields once saving finishes", () => {
    const { rerender } = renderCanvas({
      name: "Launch Workforce",
      description: "Coordinate launch work",
      saving: true,
    })
    expect(screen.getByDisplayValue("Launch Workforce")).toBeDisabled()

    rerender({ saving: false })

    expect(screen.getByDisplayValue("Launch Workforce")).not.toBeDisabled()
  })

  it("does not fire a premature save when an unrelated mutation disables the field mid-edit (self-review finding after round 8)", () => {
    // Regression test: `saving` is one flag shared by every mutation handler
    // in workforce-builder.tsx (handleAddWorker, handlePublish, etc.), not
    // just handleSaveDetails. A field becoming `disabled` while it still has
    // focus forces the browser to blur it -- and this node commits on blur.
    // Without a disabled-guard in commit(), an unrelated action saving
    // elsewhere in the builder while the user is still mid-edit here would
    // force a blur and silently save the half-typed value.
    const { onSaveDetails, rerender } = renderCanvas({
      name: "Launch Workforce",
      description: "Coordinate launch work",
      saving: false,
    })

    const nameInput = screen.getByDisplayValue("Launch Workforce")
    fireEvent.change(nameInput, { target: { value: "Mid-edit unsaved title" } })

    // An unrelated mutation elsewhere flips the shared `saving` flag.
    rerender({ saving: true })
    expect(nameInput).toBeDisabled()

    // The forced blur a real browser would fire on disabling a focused
    // field, simulated directly since jsdom does not reproduce that native
    // behavior automatically.
    fireEvent.blur(nameInput)

    expect(onSaveDetails).not.toHaveBeenCalled()
    // The unsaved edit isn't lost -- it's still sitting in local state,
    // simply not disabled anymore once the unrelated save finishes.
    rerender({ saving: false })
    expect(screen.getByDisplayValue("Mid-edit unsaved title")).toBeInTheDocument()
  })
})
