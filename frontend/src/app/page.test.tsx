import React from "react"
import { cleanup, render, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

const routerReplaceMock = vi.hoisted(() => vi.fn())

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: routerReplaceMock }),
}))

import Home from "./page"

describe("Home route", () => {
  afterEach(() => {
    routerReplaceMock.mockReset()
    cleanup()
  })

  it("replaces the root route with the task page without rendering the removed home page", async () => {
    const { container } = render(<Home />)

    expect(container).toBeEmptyDOMElement()
    await waitFor(() => expect(routerReplaceMock).toHaveBeenCalledWith("/task"))
    expect(routerReplaceMock).toHaveBeenCalledTimes(1)
  })
})
