/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

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

function renderCanvas(props: Partial<React.ComponentProps<typeof WorkforceCanvas>> = {}) {
  const dialogs = props.dialogs ?? makeDialogs()
  const onSaveDetails = props.onSaveDetails ?? vi.fn().mockResolvedValue(undefined)
  render(
    <WorkforceCanvas
      name={props.name ?? "Launch Workforce"}
      description={props.description ?? ""}
      onSaveDetails={onSaveDetails}
      manager={"manager" in props ? props.manager! : manager}
      workers={props.workers ?? [worker]}
      isArchived={props.isArchived ?? false}
      dialogs={dialogs}
      getStartedSteps={props.getStartedSteps ?? []}
      getStartedCollapsed={props.getStartedCollapsed ?? false}
      onToggleGetStarted={props.onToggleGetStarted ?? vi.fn()}
    />,
  )
  return { dialogs, onSaveDetails }
}

describe("WorkforceCanvas", () => {
  afterEach(() => {
    cleanup()
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
})
