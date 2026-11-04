import { describe, expect, it } from "vitest"

describe("vitest browser storage setup", () => {
  it("provides localStorage to tests that exercise browser auth code", () => {
    localStorage.setItem("auth_token", "access-token")

    expect(localStorage.getItem("auth_token")).toBe("access-token")
  })

  it("starts each test with an empty localStorage", () => {
    expect(localStorage.getItem("auth_token")).toBeNull()
  })
})
