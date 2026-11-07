import { describe, expect, it } from "vitest"

import { userDisplayLabel, userFirstName } from "./user-display"

describe("userFirstName", () => {
  it("preserves the provider's compound given name", () => {
    expect(userFirstName({ first_name: " Mary Jane ", username: "alias@gmail.com" }, "there")).toBe("Mary Jane")
  })

  it("falls back to a short name without exposing email addresses or internal usernames", () => {
    expect(userFirstName({ username: "cheffrey.imutan@gmail.com" }, "there")).toBe("cheffrey")
    expect(userFirstName({ username: "Ada Lovelace" }, "there")).toBe("Ada")
    expect(userFirstName({ username: "acct_0123456789abcdef0123456789abcdef", email: "ada@example.com" }, "there")).toBe("ada")
    expect(userFirstName({ username: "acct_0123456789abcdef0123456789abcdef" }, "there")).toBe("there")
    expect(userFirstName({ first_name: "alias@gmail.com", username: "alias@gmail.com" }, "there")).toBe("alias")
    expect(userFirstName(null, "there")).toBe("there")
  })
})

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
