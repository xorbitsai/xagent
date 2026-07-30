import { describe, expect, it } from "vitest"
import { resolveTranslation } from "@/i18n/translations"

import { authMutationUnavailableTranslationKey, isAuthPublicPath, isExternalRoutePath } from "./auth-pages"

describe("auth public paths", () => {
  it("allows the OIDC callback route through the auth guard", () => {
    expect(isAuthPublicPath("/auth/oidc/callback")).toBe(true)
  })

  it("classifies widget and share routes as external provider boundaries", () => {
    expect(isExternalRoutePath("/widget/chat/session")).toBe(true)
    expect(isExternalRoutePath("/share/public-token")).toBe(true)
    expect(isExternalRoutePath("/settings")).toBe(false)
    expect(isExternalRoutePath("/widgets")).toBe(false)
    expect(isExternalRoutePath("/widget-admin")).toBe(false)
    expect(isExternalRoutePath("/share-settings")).toBe(false)
  })

  it.each([
    ["storage_unavailable", "Your browser is blocking local storage. Enable storage and try again.", "浏览器阻止了本地存储。请启用本地存储后重试。"],
    ["coordination_unavailable", "Your browser does not support the secure sign-in features this application requires.", "您的浏览器不支持此应用所需的安全登录功能。"],
    ["operation_failed", "Your sign-in session could not be updated. Please try again.", "无法更新您的登录会话，请重试。"],
  ] as const)("maps %s to a real localized availability key", (reason, english, chinese) => {
    const key = authMutationUnavailableTranslationKey(reason)
    expect(resolveTranslation("en", key)).toBe(english)
    expect(resolveTranslation("zh", key)).toBe(chinese)
  })
})
