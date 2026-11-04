import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string, vars?: Record<string, string | number>) =>
      vars?.name ? `${key}:${vars.name}` : key,
  }),
}))

vi.mock("@/components/workforce", () => ({
  WorkforceStatusBadge: ({ status }: { status: string }) => (
    <span>{`workforces.status.${status}`}</span>
  ),
}))

import { AgentDeleteDialog } from "./agent-delete-dialog"
import type { AgentDeleteConflictDetail } from "@/lib/agent-delete"

const conflict: AgentDeleteConflictDetail = {
  code: "agent_in_use_by_workforce",
  message: "Agent is referenced by workforces",
  references: [
    {
      workforce_id: 7,
      name: "Editable Draft",
      status: "draft",
      roles: ["manager"],
      can_edit: true,
      can_discard: true,
    },
    {
      workforce_id: 8,
      name: "Read-only Active",
      status: "active",
      roles: ["worker"],
      can_edit: false,
      can_discard: false,
    },
  ],
  has_hidden_references: true,
}

describe("AgentDeleteDialog", () => {
  afterEach(() => cleanup())

  it("renders visible references, routes, roles, and hidden-reference copy", () => {
    render(
      <AgentDeleteDialog
        target={{ id: 42, name: "Research Agent" }}
        conflict={conflict}
        pendingAction={null}
        onOpenChange={vi.fn()}
        onConfirmDelete={vi.fn()}
        onDiscardWorkforce={vi.fn()}
      />,
    )

    expect(screen.getByText("Editable Draft")).toBeInTheDocument()
    expect(screen.getByText("Read-only Active")).toBeInTheDocument()
    expect(screen.getByText("builds.list.deleteDialog.roles.manager")).toBeInTheDocument()
    expect(screen.getByText("builds.list.deleteDialog.roles.worker")).toBeInTheDocument()
    expect(screen.getByText("workforces.actions.readOnly")).toBeInTheDocument()
    expect(screen.getByText("builds.list.deleteDialog.hiddenReferences")).toBeInTheDocument()

    const links = screen.getAllByRole("link")
    expect(links[0]).toHaveAttribute("href", "/workforces/7")
    expect(links[0]).toHaveAttribute("target", "_blank")
    expect(links[1]).toHaveAttribute("href", "/workforces/8")
  })

  it("requires two clicks before discarding an eligible draft", () => {
    const onDiscardWorkforce = vi.fn()
    render(
      <AgentDeleteDialog
        target={{ id: 42, name: "Research Agent" }}
        conflict={conflict}
        pendingAction={null}
        onOpenChange={vi.fn()}
        onConfirmDelete={vi.fn()}
        onDiscardWorkforce={onDiscardWorkforce}
      />,
    )

    expect(
      screen.queryByRole("button", {
        name: "builds.list.deleteDialog.discardDraft:Read-only Active",
      }),
    ).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Editable Draft",
    }))
    expect(onDiscardWorkforce).not.toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.confirmDiscardDraft:Editable Draft",
    }))
    expect(onDiscardWorkforce).toHaveBeenCalledWith(conflict.references[0])

    expect(screen.getByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Editable Draft",
    })).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.discardDraft:Editable Draft",
    }))
    expect(onDiscardWorkforce).toHaveBeenCalledTimes(1)
  })

  it("survives translation extensions wrapping text nodes when the spinner toggles", () => {
    const props = {
      target: { id: 42, name: "Research Agent" },
      conflict,
      onOpenChange: vi.fn(),
      onConfirmDelete: vi.fn(),
      onDiscardWorkforce: vi.fn(),
    }
    const { rerender } = render(
      <AgentDeleteDialog {...props} pendingAction={null} />,
    )

    // Browser translation extensions (Chrome auto-translate, immersive
    // translate, ...) replace bare text nodes with <font> wrappers. React's
    // insertBefore then fails if a conditional sibling (the Loader2 spinner)
    // is anchored on a bare text node.
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT)
    const textNodes: Text[] = []
    while (walker.nextNode()) textNodes.push(walker.currentNode as Text)
    for (const textNode of textNodes) {
      const font = document.createElement("font")
      textNode.parentNode?.insertBefore(font, textNode)
      font.appendChild(textNode)
    }

    expect(() =>
      rerender(
        <AgentDeleteDialog {...props} pendingAction={{ kind: "delete" }} />,
      ),
    ).not.toThrow()
  })

  it("keeps explicit retry available after the visible blockers are cleared", () => {
    const onConfirmDelete = vi.fn()
    render(
      <AgentDeleteDialog
        target={{ id: 42, name: "Research Agent" }}
        conflict={{ ...conflict, references: [], has_hidden_references: false }}
        pendingAction={null}
        onOpenChange={vi.fn()}
        onConfirmDelete={onConfirmDelete}
        onDiscardWorkforce={vi.fn()}
      />,
    )

    expect(screen.getByText("builds.list.deleteDialog.readyToRetry")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", {
      name: "builds.list.deleteDialog.retryDelete",
    }))
    expect(onConfirmDelete).toHaveBeenCalledTimes(1)
  })
})
