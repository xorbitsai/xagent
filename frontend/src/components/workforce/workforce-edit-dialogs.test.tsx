/// <reference types="@testing-library/jest-dom/vitest" />

import React from "react"
import { act, cleanup, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const apiRequestMock = vi.hoisted(() => vi.fn())
const translateMock = vi.hoisted(() => (key: string) => key)

vi.mock("@/lib/api-wrapper", () => ({
  apiRequest: apiRequestMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return { ...actual, getApiUrl: () => "https://api.test" }
})

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: translateMock }),
}))

vi.mock("sonner", () => ({
  toast: { error: vi.fn(), success: vi.fn() },
}))

import { AgentPickerDialog, useWorkforceEditDialogs } from "./workforce-edit-dialogs"
import type { WorkforceAgentOption, WorkforceWorker } from "@/types/workforce"

const manager = {
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

const availableAgent: WorkforceAgentOption = {
  id: 42,
  name: "Copy Editor",
  description: "Polishes marketing copy for external publication",
  logo_url: null,
  status: "published",
}

function renderDialogs(overrides: Partial<Parameters<typeof useWorkforceEditDialogs>[0]> = {}) {
  return renderHook(() =>
    useWorkforceEditDialogs({
      manager,
      workers: [worker],
      agents: [availableAgent],
      onChangeLead: vi.fn(),
      onAddWorker: vi.fn(),
      onSaveWorker: vi.fn(),
      onRemoveWorker: vi.fn(),
      ...overrides,
    }),
  )
}

afterEach(() => {
  vi.restoreAllMocks()
})

describe("useWorkforceEditDialogs — handleAddMember", () => {
  it("never derives instructions from the agent's own description (PR review round 7, finding #4)", async () => {
    // Regression test: one click on the agent picker used to copy the
    // agent's public-facing description verbatim into the new member's
    // operational system instructions, with no user review. The picker
    // still forwards a `description` argument for signature compatibility,
    // but it must never end up sourced from agent.description.
    const onAddWorker = vi.fn().mockResolvedValue(undefined)
    const { result } = renderDialogs({ onAddWorker })

    await act(async () => {
      await result.current.handleAddMember(availableAgent.id)
    })

    expect(onAddWorker).toHaveBeenCalledWith(availableAgent.id, availableAgent.name)
    expect(onAddWorker).not.toHaveBeenCalledWith(availableAgent.id, availableAgent.description)
  })

  it("falls back to the agent id when the agent has no resolvable name", async () => {
    const onAddWorker = vi.fn().mockResolvedValue(undefined)
    const { result } = renderDialogs({ agents: [], onAddWorker })

    await act(async () => {
      await result.current.handleAddMember(999)
    })

    expect(onAddWorker).toHaveBeenCalledWith(999, "999")
  })
})

describe("useWorkforceEditDialogs — save/remove failure handling", () => {
  it("handleSaveMember swallows a rejected onSaveWorker instead of leaving an unhandled rejection (PR review round 7, finding #8)", async () => {
    const onSaveWorker = vi.fn().mockRejectedValue(new Error("save failed"))
    const { result } = renderDialogs({ onSaveWorker })

    act(() => {
      result.current.openMemberDetail(worker)
    })

    await expect(
      act(async () => {
        await result.current.handleSaveMember()
      }),
    ).resolves.not.toThrow()

    expect(onSaveWorker).toHaveBeenCalledOnce()
  })

  it("handleRemoveMember swallows a rejected onRemoveWorker instead of leaving an unhandled rejection (PR review round 7, finding #8)", async () => {
    const onRemoveWorker = vi.fn().mockRejectedValue(new Error("remove failed"))
    const { result } = renderDialogs({ onRemoveWorker })

    await expect(
      act(async () => {
        await result.current.handleRemoveMember(worker)
      }),
    ).resolves.not.toThrow()

    expect(onRemoveWorker).toHaveBeenCalledOnce()
  })

  it("handleSaveMember still closes the member detail on success", async () => {
    const onSaveWorker = vi.fn().mockResolvedValue(undefined)
    const { result } = renderDialogs({ onSaveWorker })

    act(() => {
      result.current.openMemberDetail(worker)
    })
    expect(result.current.selectedWorker).toEqual(worker)

    await act(async () => {
      await result.current.handleSaveMember()
    })

    expect(result.current.selectedWorker).toBeNull()
  })
})

describe("AgentPickerDialog — create-and-add from a built-in template", () => {
  afterEach(() => {
    cleanup()
    apiRequestMock.mockReset()
  })

  it("passes the freshly created agent's own name, not a stale-list lookup that would fall through to the bare agent id", async () => {
    // Regression test: this same fix's handleAddMember (in the describe
    // block above) stopped falling back to agent?.description, but a
    // template-created agent's onSelectAgent(newAgent.id) call used to rely
    // on that same fallback chain finding the agent in the parent's `agents`
    // list -- which this brand-new agent, created moments earlier by this
    // exact flow, is never in yet (that list is only refreshed on a
    // separate, later reload). Without passing the name explicitly here,
    // handleAddMember's chain bottoms out at String(agentId): a bare numeric
    // id silently saved as the new member's live system instructions.
    apiRequestMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [{ id: "tmpl-1", name: "Researcher Template", description: "A template" }],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 104, name: "New Researcher", description: "Some public blurb" }),
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })

    const onSelectAgent = vi.fn().mockResolvedValue(undefined)

    render(
      <AgentPickerDialog
        open={true}
        onOpenChange={vi.fn()}
        title="Add member"
        agents={[]}
        onSelectAgent={onSelectAgent}
        locale="en-US"
      />,
    )

    fireEvent.click(screen.getByText("workforces.workers.tabTemplates"))
    fireEvent.click(await screen.findByText("Researcher Template"))

    const nameInput = await screen.findByPlaceholderText("workforces.templates.agentNamePlaceholder")
    fireEvent.change(nameInput, { target: { value: "New Researcher" } })
    fireEvent.click(screen.getByText("workforces.templates.createAndAdd"))

    await waitFor(() => expect(onSelectAgent).toHaveBeenCalledOnce())
    expect(onSelectAgent).toHaveBeenCalledWith(104, "New Researcher")
  })

  it("reports the freshly created agent via onAgentCreated so a create-mode parent's stale agents list gets it (PR review round 9, NEW-F2)", async () => {
    // Regression test: create mode fetches `agents` once on mount and never
    // refreshes it, so a template-created agent used to be invisible to any
    // by-id lookup against that list (toFakeWorker, manager resolution) --
    // silently rendering a blank member card or leaving "Choose a lead"
    // shown even though the manager was actually set.
    apiRequestMock
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [{ id: "tmpl-1", name: "Researcher Template", description: "A template" }],
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: 104, name: "New Researcher", description: "Some public blurb" }),
      })
      .mockResolvedValueOnce({ ok: true, json: async () => ({}) })

    const onSelectAgent = vi.fn().mockResolvedValue(undefined)
    const onAgentCreated = vi.fn()

    render(
      <AgentPickerDialog
        open={true}
        onOpenChange={vi.fn()}
        title="Add member"
        agents={[]}
        onSelectAgent={onSelectAgent}
        locale="en-US"
        onAgentCreated={onAgentCreated}
      />,
    )

    fireEvent.click(screen.getByText("workforces.workers.tabTemplates"))
    fireEvent.click(await screen.findByText("Researcher Template"))

    const nameInput = await screen.findByPlaceholderText("workforces.templates.agentNamePlaceholder")
    fireEvent.change(nameInput, { target: { value: "New Researcher" } })
    fireEvent.click(screen.getByText("workforces.templates.createAndAdd"))

    await waitFor(() => expect(onSelectAgent).toHaveBeenCalledOnce())
    expect(onAgentCreated).toHaveBeenCalledWith({
      id: 104,
      name: "New Researcher",
      description: "Some public blurb",
      logo_url: null,
      status: "published",
    })
  })
})
