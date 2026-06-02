import React from "react"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const createWorkforceMock = vi.hoisted(() => vi.fn())
const createWorkforceFromPromptMock = vi.hoisted(() => vi.fn())
const listAgentOptionsMock = vi.hoisted(() => vi.fn())
const runWorkforceMock = vi.hoisted(() => vi.fn())
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

vi.mock("@/lib/workforces-api", () => ({
  createWorkforce: createWorkforceMock,
  createWorkforceFromPrompt: createWorkforceFromPromptMock,
  listAgentOptions: listAgentOptionsMock,
  runWorkforce: runWorkforceMock,
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    locale: "en-US",
    t: translateMock,
  }),
}))

vi.mock("@/components/ui/scroll-area", () => {
  const ScrollArea = React.forwardRef<HTMLDivElement, React.HTMLAttributes<HTMLDivElement>>(
    ({ children, ...props }, ref) => (
      <div ref={ref} data-radix-scroll-area-viewport {...props}>
        {children}
      </div>
    ),
  )
  ScrollArea.displayName = "MockScrollArea"
  return { ScrollArea }
})

vi.mock("@/components/ui/select", () => ({
  Select: ({
    value,
    onValueChange,
    options,
    placeholder,
  }: {
    value?: string
    onValueChange: (value: string) => void
    options?: Array<{ value: string; label: string }>
    placeholder?: string
  }) => (
    <select
      aria-label={placeholder}
      value={value || ""}
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
    onCheckedChange,
  }: {
    checked?: boolean
    onCheckedChange?: (checked: boolean) => void
  }) => (
    <input
      aria-label="switch"
      type="checkbox"
      checked={Boolean(checked)}
      onChange={(event) => onCheckedChange?.(event.target.checked)}
    />
  ),
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

import {
  ProposedPatchCard,
  ReviewStep,
  WorkforceBuilderChat,
  WorkforcePromptCreator,
  WorkforceSummary,
  WorkforceTestPanel,
  WorkforceWizard,
} from "."
import type { WorkforceBuilderPatch, WorkforceDetail } from "@/types/workforce"

const workforceDetail: WorkforceDetail = {
  id: 42,
  name: "Launch Workforce",
  description: null,
  status: "draft",
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

describe("workforce frontend core components", () => {
  beforeEach(() => {
    createWorkforceMock.mockReset()
    createWorkforceFromPromptMock.mockReset()
    listAgentOptionsMock.mockReset()
    runWorkforceMock.mockReset()
  })

  afterEach(() => {
    cleanup()
  })

  it("submits builder chat messages without routing concerns", async () => {
    const onSubmit = vi.fn()
    render(
      <WorkforceBuilderChat
        messages={[
          {
            id: 1,
            role: "assistant",
            content: "Current plan",
            status: "stored",
            proposed_patch: null,
            created_at: "2026-05-26T00:00:00Z",
          },
        ]}
        onSubmit={onSubmit}
      />,
    )

    fireEvent.change(screen.getByPlaceholderText("workforces.builder.messagePlaceholder"), {
      target: { value: " Add a researcher " },
    })
    fireEvent.click(screen.getByText("workforces.actions.proposeChanges"))

    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalledWith("Add a researcher")
    })
  })

  it("keeps builder chat read-only when the route disables editing", () => {
    const onSubmit = vi.fn()
    render(
      <WorkforceBuilderChat
        messages={[]}
        readOnly
        readOnlyReason="workforces.builder.archivedReadOnly"
        onSubmit={onSubmit}
      />,
    )

    expect(screen.getByPlaceholderText("workforces.builder.messagePlaceholder")).toBeDisabled()
    expect(screen.getByText("workforces.builder.archivedReadOnly")).toBeInTheDocument()
    expect(screen.getByText("workforces.actions.readOnly")).toBeDisabled()
    fireEvent.click(screen.getByText("workforces.actions.readOnly"))
    expect(onSubmit).not.toHaveBeenCalled()
  })

  it("applies the exact stored builder patch", async () => {
    const patch: WorkforceBuilderPatch = {
      summary: "Add worker",
      operations: [{ op: "add_existing_worker", agent_id: 8 }],
      warnings: [],
    }
    const onApply = vi.fn()

    render(
      <ProposedPatchCard
        patch={patch}
        messageId={12}
        onApply={onApply}
      />,
    )

    fireEvent.click(screen.getByText("workforces.actions.applyChanges"))

    expect(onApply).toHaveBeenCalledWith(12, patch)
  })

  it("does not apply builder patches in read-only mode", () => {
    const patch: WorkforceBuilderPatch = {
      summary: "Add worker",
      operations: [{ op: "add_existing_worker", agent_id: 8 }],
      warnings: [],
    }
    const onApply = vi.fn()

    render(
      <ProposedPatchCard
        patch={patch}
        messageId={12}
        readOnly
        onApply={onApply}
      />,
    )

    expect(screen.getByText("workforces.actions.readOnly")).toBeDisabled()
    fireEvent.click(screen.getByText("workforces.actions.readOnly"))
    expect(onApply).not.toHaveBeenCalled()
  })

  it("creates from prompt through callbacks instead of hardcoded routes", async () => {
    createWorkforceFromPromptMock.mockResolvedValueOnce(workforceDetail)
    const onCreated = vi.fn()

    render(
      <WorkforcePromptCreator
        onCreated={onCreated}
      />,
    )

    fireEvent.change(screen.getByPlaceholderText("workforces.create.prompt.placeholder"), {
      target: { value: "Build a launch workforce" },
    })
    fireEvent.click(screen.getByText("workforces.create.prompt.generate"))

    await waitFor(() => {
      expect(createWorkforceFromPromptMock).toHaveBeenCalledWith({
        prompt: "Build a launch workforce",
      })
    })
    expect(onCreated).toHaveBeenCalledWith(workforceDetail)
  })

  it("runs a workforce and returns the run result to the route layer", async () => {
    const runResult = {
      workforce_run_id: 2,
      task_id: 9,
      status: "running",
      redirect_url: "/task/9",
    }
    runWorkforceMock.mockResolvedValueOnce(runResult)
    const onRunCreated = vi.fn()

    render(<WorkforceTestPanel workforceId={42} onRunCreated={onRunCreated} />)

    fireEvent.change(screen.getByPlaceholderText("workforces.run.placeholder"), {
      target: { value: " Draft a launch brief " },
    })
    fireEvent.click(screen.getByText("workforces.actions.runWorkforce"))

    await waitFor(() => {
      expect(runWorkforceMock).toHaveBeenCalledWith(42, {
        message: "Draft a launch brief",
      })
    })
    expect(onRunCreated).toHaveBeenCalledWith(runResult)
  })

  it("shows a disabled reason and does not run when the route disables execution", () => {
    const onRunCreated = vi.fn()

    render(
      <WorkforceTestPanel
        workforceId={42}
        disabled
        disabledReason="workforces.run.inactiveDisabled"
        onRunCreated={onRunCreated}
      />,
    )

    expect(screen.getByText("workforces.run.inactiveDisabled")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("workforces.run.placeholder")).toBeDisabled()
    expect(screen.getByText("workforces.actions.runWorkforce")).toBeDisabled()
    fireEvent.click(screen.getByText("workforces.actions.runWorkforce"))
    expect(runWorkforceMock).not.toHaveBeenCalled()
    expect(onRunCreated).not.toHaveBeenCalled()
  })

  it("does not link readonly workforce agents to the editor", () => {
    render(
      <ReviewStep
        name="Launch Workforce"
        description=""
        managerAgentId="7"
        managerInstructions=""
        agents={[
          {
            id: 7,
            name: "Shared Manager",
            description: null,
            logo_url: null,
            status: "published",
            readonly: true,
            can_edit: false,
          },
          {
            id: 8,
            name: "Owned Worker",
            description: null,
            logo_url: null,
            status: "published",
            readonly: false,
            can_edit: true,
          },
        ]}
        workers={[
          {
            source_type: "existing",
            agent_id: 8,
            alias: "",
            assignment_instructions: "Research competitors",
            enabled: true,
            sort_order: 1,
          },
        ]}
      />,
    )

    expect(screen.queryByText("workforces.actions.openAgentEditor")).not.toBeInTheDocument()
    expect(screen.getAllByText("workforces.actions.readOnly")).toHaveLength(1)
  })

  it("hides summary editor links for readonly workforce agents", () => {
    render(
      <WorkforceSummary
        workforce={{
          ...workforceDetail,
          manager: {
            ...workforceDetail.manager,
            readonly: true,
            can_edit: false,
          },
          workers: [
            {
              id: 1,
              agent: {
                id: 8,
                name: "Shared Worker",
                description: null,
                logo_url: null,
                status: "published",
                readonly: true,
                can_edit: false,
              },
              alias: null,
              assignment_instructions: "Research competitors",
              source_type: "existing",
              template_id: null,
              enabled: true,
              sort_order: 1,
              canvas_position: null,
              created_at: null,
              updated_at: null,
            },
          ],
        }}
      />,
    )

    expect(screen.queryByText("workforces.actions.openAgentEditor")).not.toBeInTheDocument()
    expect(screen.queryByText("workforces.actions.editAgent")).not.toBeInTheDocument()
    expect(screen.getAllByText("workforces.actions.readOnly")).toHaveLength(2)
  })

  it("filters manager and worker choices to published agents and creates PR5 payloads", async () => {
    listAgentOptionsMock.mockResolvedValueOnce([
      {
        id: 7,
        name: "Manager Agent",
        description: null,
        logo_url: null,
        status: "published",
      },
      {
        id: 8,
        name: "Research Agent",
        description: "Research",
        logo_url: null,
        status: "published",
      },
      {
        id: 9,
        name: "Draft Agent",
        description: null,
        logo_url: null,
        status: "draft",
      },
    ])
    createWorkforceMock.mockResolvedValueOnce(workforceDetail)
    const onCreated = vi.fn()

    render(<WorkforceWizard onCreated={onCreated} />)

    await screen.findByRole("region", { name: "Progress Stepper" })

    fireEvent.change(screen.getByPlaceholderText("workforces.create.placeholders.name"), {
      target: { value: "Launch Workforce" },
    })
    fireEvent.change(
      screen.getByPlaceholderText("workforces.create.placeholders.managerInstructions"),
      {
        target: { value: "Coordinate work" },
      },
    )
    fireEvent.change(screen.getByLabelText("workforces.create.manager.placeholder"), {
      target: { value: "7" },
    })
    fireEvent.click(screen.getByText("common.next"))

    expect(screen.queryByRole("option", { name: "Draft Agent" })).not.toBeInTheDocument()
    expect(screen.queryByRole("option", { name: "Manager Agent" })).not.toBeInTheDocument()
    fireEvent.change(screen.getByLabelText("workforces.workers.chooseAgent"), {
      target: { value: "8" },
    })
    fireEvent.change(screen.getByPlaceholderText("workforces.workers.instructionsPlaceholder"), {
      target: { value: "Research competitors" },
    })
    fireEvent.click(screen.getByText("workforces.actions.addWorker"))
    fireEvent.click(screen.getByText("common.next"))
    fireEvent.click(screen.getByText("workforces.actions.create"))

    await waitFor(() => {
      expect(createWorkforceMock).toHaveBeenCalledWith({
        name: "Launch Workforce",
        description: undefined,
        manager_agent_id: 7,
        manager_instructions: "Coordinate work",
        workers: [
          {
            source_type: "existing",
            agent_id: 8,
            alias: undefined,
            assignment_instructions: "Research competitors",
            enabled: true,
            sort_order: 1,
            canvas_position: undefined,
          },
        ],
      })
    })
    expect(JSON.stringify(createWorkforceMock.mock.calls[0][0])).not.toContain("status")
    expect(onCreated).toHaveBeenCalledWith(workforceDetail)
  })

  it("blocks submit when a changed manager is already configured as a worker", async () => {
    listAgentOptionsMock.mockResolvedValueOnce([
      {
        id: 7,
        name: "Manager Agent",
        description: null,
        logo_url: null,
        status: "published",
      },
      {
        id: 8,
        name: "Research Agent",
        description: "Research",
        logo_url: null,
        status: "published",
      },
    ])

    render(<WorkforceWizard onCreated={vi.fn()} />)

    await screen.findByRole("region", { name: "Progress Stepper" })

    fireEvent.change(screen.getByPlaceholderText("workforces.create.placeholders.name"), {
      target: { value: "Launch Workforce" },
    })
    fireEvent.change(screen.getByLabelText("workforces.create.manager.placeholder"), {
      target: { value: "7" },
    })
    fireEvent.click(screen.getByText("common.next"))

    fireEvent.change(screen.getByLabelText("workforces.workers.chooseAgent"), {
      target: { value: "8" },
    })
    fireEvent.change(screen.getByPlaceholderText("workforces.workers.instructionsPlaceholder"), {
      target: { value: "Research competitors" },
    })
    fireEvent.click(screen.getByText("workforces.actions.addWorker"))

    fireEvent.click(screen.getByText("common.back"))
    fireEvent.change(screen.getByLabelText("workforces.create.manager.placeholder"), {
      target: { value: "8" },
    })

    expect(
      screen.getByText("workforces.review.warnings.managerCannotBeWorker"),
    ).toBeInTheDocument()

    fireEvent.click(screen.getByText("common.next"))
    expect(screen.getByText("common.next")).toBeDisabled()
    expect(createWorkforceMock).not.toHaveBeenCalled()
  })
})
