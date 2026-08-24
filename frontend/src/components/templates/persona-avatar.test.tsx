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
})
