import React from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { PersonaAvatar } from "./persona-avatar"

afterEach(cleanup)

describe("PersonaAvatar", () => {
  it("renders a named image by default when an avatar is set", () => {
    render(<PersonaAvatar persona={{ name: "Maya", avatar: "/avatars/maya.png" }} sizeClassName="h-8 w-8" />)

    expect(screen.getByRole("img", { name: "Maya" })).toHaveAttribute("src", "/avatars/maya.png")
  })

  it("falls back to an initial when no avatar is set", () => {
    render(<PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" />)

    expect(screen.queryByRole("img")).toBeNull()
    expect(screen.getByText("M")).toBeTruthy()
  })

  it("hides the image from the accessibility tree when decorative, without removing it from the DOM", () => {
    const { container } = render(
      <PersonaAvatar persona={{ name: "Maya", avatar: "/avatars/maya.png" }} sizeClassName="h-8 w-8" decorative />
    )

    expect(screen.queryByRole("img", { name: "Maya" })).toBeNull()
    expect(container.querySelector("img")).toHaveAttribute("src", "/avatars/maya.png")
  })

  it("hides the fallback initial from the accessibility tree when decorative", () => {
    const { container } = render(<PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" decorative />)

    expect(container.querySelector('[aria-hidden="true"]')).toHaveTextContent("M")
  })

  // Pins a PR review test-coverage gap: the `style` pass-through prop
  // (added for the onboarding page's category-colored ring) was never
  // referenced by this file's own tests.
  it("passes the style prop through to the image", () => {
    render(
      <PersonaAvatar
        persona={{ name: "Maya", avatar: "/avatars/maya.png" }}
        sizeClassName="h-8 w-8"
        style={{ boxShadow: "0 0 0 2px red" }}
      />
    )

    expect(screen.getByRole("img", { name: "Maya" })).toHaveStyle({ boxShadow: "0 0 0 2px red" })
  })

  it("passes the style prop through to the fallback initial", () => {
    render(
      <PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" style={{ boxShadow: "0 0 0 2px red" }} />
    )

    expect(screen.getByText("M")).toHaveStyle({ boxShadow: "0 0 0 2px red" })
  })

  it("pulses an image avatar only while thinking", () => {
    const persona = { name: "Maya", avatar: "/avatars/maya.png" }
    const { rerender } = render(<PersonaAvatar persona={persona} sizeClassName="h-8 w-8" thinking />)

    expect(screen.getByRole("img", { name: "Maya" })).toHaveClass("motion-safe:animate-pulse")

    rerender(<PersonaAvatar persona={persona} sizeClassName="h-8 w-8" />)

    expect(screen.getByRole("img", { name: "Maya" })).not.toHaveClass("motion-safe:animate-pulse")
  })

  it("pulses the fallback initial only while thinking", () => {
    const { rerender } = render(<PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" thinking />)

    expect(screen.getByText("M")).toHaveClass("motion-safe:animate-pulse")

    rerender(<PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" />)

    expect(screen.getByText("M")).not.toHaveClass("motion-safe:animate-pulse")
  })
})

describe("PersonaAvatar for an agent", () => {
  it("draws the same avatar whatever agent it stands for", () => {
    const { container, rerender } = render(<PersonaAvatar persona={{ name: "Maya" }} sizeClassName="h-8 w-8" />)
    const plain = container.innerHTML

    rerender(<PersonaAvatar persona={{ name: "Maya", agentId: 7 }} sizeClassName="h-8 w-8" />)

    expect(container.innerHTML).toBe(plain)
  })
})
