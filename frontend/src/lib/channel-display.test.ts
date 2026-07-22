import { describe, expect, it } from "vitest"

import { getChannelTooltip, getCompactChannelName } from "./channel-display"

describe("channel display", () => {
  it("handles missing channel names", () => {
    expect(getCompactChannelName(undefined, "telegram")).toBe("")
    expect(getCompactChannelName(null, "feishu")).toBe("")
    expect(getCompactChannelName("", "telegram")).toBe("")
    expect(getChannelTooltip(undefined, "telegram")).toBe("")
    expect(getChannelTooltip(null, "feishu")).toBe("")
    expect(getChannelTooltip("", "telegram")).toBe("")
  })

  it("replaces a redundant Telegram suffix with the platform icon context", () => {
    expect(getCompactChannelName("Xagent Telegram", "telegram")).toBe("Xagent")
  })

  it("keeps the distinguishing part of a Telegram channel name", () => {
    expect(getCompactChannelName("Telegram Support", "TELEGRAM")).toBe("Support")
    expect(getCompactChannelName("Xagent - Telegram", "telegram")).toBe("Xagent")
    expect(getCompactChannelName("Telegram: Support", "telegram")).toBe("Support")
  })

  it("preserves separators in the distinguishing channel name", () => {
    expect(getCompactChannelName("Prod-EU Telegram", "telegram")).toBe("Prod-EU")
    expect(getCompactChannelName("Team|A Telegram", "telegram")).toBe("Team|A")
  })

  it("keeps a platform-only name instead of producing an empty label", () => {
    expect(getCompactChannelName("Telegram", "telegram")).toBe("Telegram")
  })

  it("supports Feishu and Lark channel aliases", () => {
    expect(getCompactChannelName("Xagent Feishu", "feishu")).toBe("Xagent")
    expect(getCompactChannelName("Lark Support", "feishu")).toBe("Support")
    expect(getCompactChannelName("生产环境飞书", "feishu")).toBe("生产环境")
  })

  it("does not strip a CJK alias from the middle of a larger token", () => {
    expect(getCompactChannelName("超级飞书商店", "feishu")).toBe("超级飞书商店")
  })

  it("does not infer the channel type from its name", () => {
    expect(getCompactChannelName("Xagent Telegram")).toBe("Xagent Telegram")
    expect(getCompactChannelName("TelegramBot", "telegram")).toBe("TelegramBot")
  })

  it("preserves the complete channel identity in the tooltip", () => {
    expect(getChannelTooltip("Xagent Telegram", "telegram")).toBe(
      "Telegram · Xagent Telegram",
    )
    expect(getChannelTooltip("Xagent Feishu", "feishu")).toBe(
      "Feishu · Xagent Feishu",
    )
  })
})
