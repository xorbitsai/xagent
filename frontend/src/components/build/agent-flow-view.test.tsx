import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

import { AgentFlowView, type AgentFlowViewProps } from "./agent-flow-view"

function makeProps(overrides: Partial<AgentFlowViewProps> = {}): AgentFlowViewProps {
  return {
    name: "Test Agent",
    modelLabel: "GPT Test",
    executionMode: "balanced",
    instructions: "1. First step\n2. Second step",
    onInstructionsChange: vi.fn(),
    kbSelected: [],
    kbOptions: [{ value: "kb1", label: "KB One" }],
    onKbChange: vi.fn(),
    skillsSelected: [],
    skillOptions: [],
    onSkillsChange: vi.fn(),
    toolsSelected: [],
    toolOptions: [],
    onToolsChange: vi.fn(),
    connectorNames: [],
    onOpenConnectors: vi.fn(),
    triggerRows: [],
    onOpenTriggers: vi.fn(),
    promptCount: 0,
    ...overrides,
  }
}

afterEach(cleanup)

describe("AgentFlowView", () => {
  it("renders the input, agent, and output nodes with the agent name and plan steps", () => {
    render(<AgentFlowView {...makeProps()} />)

    expect(screen.getByText("builds.editor.flow.input.eyebrow")).toBeInTheDocument()
    expect(screen.getByText("builds.editor.flow.agent.eyebrow")).toBeInTheDocument()
    expect(screen.getByText("builds.editor.flow.output.eyebrow")).toBeInTheDocument()
    expect(screen.getByText("Test Agent")).toBeInTheDocument()
    expect(screen.getByText("GPT Test")).toBeInTheDocument()
    expect(screen.getByText("First step")).toBeInTheDocument()
    expect(screen.getByText("Second step")).toBeInTheDocument()
  })

  it("shows the empty-plan hint when instructions have no steps", () => {
    render(<AgentFlowView {...makeProps({ instructions: "" })} />)
    expect(screen.getByText("builds.editor.flow.agent.emptyPlan")).toBeInTheDocument()
  })

  it("renders trigger rows in the input node", () => {
    render(
      <AgentFlowView
        {...makeProps({
          triggerRows: [{ key: "webhook", label: "Webhook Trigger", description: "Fires on POST" }],
        })}
      />,
    )
    expect(screen.getByText("Webhook Trigger")).toBeInTheDocument()
    expect(screen.getByText("Fires on POST")).toBeInTheDocument()
  })

  it("commits an inline step edit on blur", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("First step"))
    const input = screen.getByDisplayValue("First step")
    fireEvent.change(input, { target: { value: "Edited step" } })
    fireEvent.blur(input)

    expect(props.onInstructionsChange).toHaveBeenCalledWith("1. Edited step\n2. Second step")
  })

  it("discards an inline step edit on Escape without committing", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("First step"))
    const input = screen.getByDisplayValue("First step")
    fireEvent.change(input, { target: { value: "Discarded edit" } })
    fireEvent.keyDown(input, { key: "Escape" })

    expect(props.onInstructionsChange).not.toHaveBeenCalled()
    expect(screen.queryByDisplayValue("Discarded edit")).not.toBeInTheDocument()
    expect(screen.getByText("First step")).toBeInTheDocument()
  })

  it("does not swallow the next commit after an Escape discard", () => {
    // Regression guard: Escape sets ignoreBlurRef; if the flag were left
    // stale, the following edit's blur-commit would be silently dropped.
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("First step"))
    // Escape blurs the input itself, which unmounts it and resets the flag
    fireEvent.keyDown(screen.getByDisplayValue("First step"), { key: "Escape" })

    fireEvent.click(screen.getByText("Second step"))
    const input = screen.getByDisplayValue("Second step")
    fireEvent.change(input, { target: { value: "Edited after escape" } })
    fireEvent.blur(input)

    expect(props.onInstructionsChange).toHaveBeenCalledWith("1. First step\n2. Edited after escape")
  })

  it("moves a step down via the hover action button", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getAllByTitle("builds.editor.flow.agent.moveDown")[0])

    expect(props.onInstructionsChange).toHaveBeenCalledWith("1. Second step\n2. First step")
  })

  it("removes a step via the hover action button", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getAllByTitle("builds.editor.flow.agent.removeStep")[0])

    expect(props.onInstructionsChange).toHaveBeenCalledWith("1. Second step")
  })

  it("appends a placeholder step via Add Step", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("builds.editor.flow.agent.addStep"))

    expect(props.onInstructionsChange).toHaveBeenCalledWith(
      "1. First step\n2. Second step\n3. builds.editor.flow.agent.newStep",
    )
  })

  it("opens the triggers dialog from the input node button", () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("builds.editor.flow.input.trigger"))

    expect(props.onOpenTriggers).toHaveBeenCalled()
  })

  it("opens the capability picker on card click and toggles a selection", async () => {
    const props = makeProps()
    render(<AgentFlowView {...props} />)

    fireEvent.click(screen.getByText("builds.editor.flow.caps.knowledgeBase"))
    fireEvent.click(await screen.findByText("KB One"))

    expect(props.onKbChange).toHaveBeenCalledWith(["kb1"])
  })

  describe("readOnly mode", () => {
    it("hides Add Step and all structural step controls", () => {
      render(<AgentFlowView {...makeProps({ readOnly: true })} />)

      expect(screen.queryByText("builds.editor.flow.agent.addStep")).not.toBeInTheDocument()
      expect(screen.queryAllByTitle("builds.editor.flow.agent.moveUp")).toHaveLength(0)
      expect(screen.queryAllByTitle("builds.editor.flow.agent.moveDown")).toHaveLength(0)
      expect(screen.queryAllByTitle("builds.editor.flow.agent.removeStep")).toHaveLength(0)
    })

    it("does not enter edit mode when a step is clicked", () => {
      const props = makeProps({ readOnly: true })
      render(<AgentFlowView {...props} />)

      fireEvent.click(screen.getByText("First step"))

      expect(screen.queryByDisplayValue("First step")).not.toBeInTheDocument()
      expect(props.onInstructionsChange).not.toHaveBeenCalled()
    })
  })
})
