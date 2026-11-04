import { describe, expect, it } from "vitest"

import { userDisplayLabel } from "./user-display"

describe("userDisplayLabel", () => {
  it("prefers a normalized email over the internal username", () => {
    expect(
      userDisplayLabel(
        { email: " account@example.com ", username: "acct_0123456789abcdef0123456789abcdef" },
        "missing",
      ),
    ).toBe("account@example.com")
  })

  it("falls back through username to the caller label", () => {
    expect(userDisplayLabel({ email: " ", username: " legacy-user " }, "missing")).toBe(
      "legacy-user",
    )
    expect(userDisplayLabel({ email: null, username: null }, "missing")).toBe("missing")
  })
})
