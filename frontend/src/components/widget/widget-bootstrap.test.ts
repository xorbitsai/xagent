import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(
  widgetScriptPath,
  "utf8",
)
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

function runWidget(attributes: Record<string, string>) {
  const script = document.createElement("script")
  script.src = "https://chat.example/widget.js"
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", {
    configurable: true,
    value: script,
  })

  window.eval(`${widgetScript}\n//# sourceURL=${widgetScriptUrl}`)
}

describe("widget bootstrap", () => {
  const fetchMock = vi.fn()
  let currentScriptDescriptor: PropertyDescriptor | undefined

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    localStorage.setItem("xagent_guest_id", "guest-fixed")
    vi.stubGlobal("fetch", fetchMock)
    fetchMock.mockReset()
  })

  afterEach(() => {
    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("fails closed without a widget key on the default token channel", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget({})

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector(".xagent-widget-container")).toBeNull()
    expect(document.head.querySelector("style")).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("Missing data-widget-key"))
  })

  it("generates and persists a guest id for a first-time visitor", async () => {
    localStorage.removeItem("xagent_guest_id")
    vi.spyOn(Math, "random")
      .mockReturnValueOnce(0.123456)
      .mockReturnValueOnce(0.654321)
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ticket: "ticket/one",
      agent_id: 17,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    runWidget({ "data-widget-key": "widget-secret" })

    const guestId = localStorage.getItem("xagent_guest_id")
    expect(guestId).toBe("guest_4fzyo82mvyqnk000qvgin")
    await vi.waitFor(() => {
      expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
        `https://chat.example/widget/chat/default?guest_id=${guestId}&agent_id=17&embed_ticket=ticket%2Fone`,
      )
    })
  })

  it("loads an embed-ticket iframe URL without exposing the widget key", async () => {
    fetchMock.mockResolvedValueOnce(new Response(JSON.stringify({
      ticket: "ticket/one",
      agent_id: 17,
    }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    }))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
        "https://chat.example/widget/chat/default?guest_id=guest-fixed&agent_id=17&embed_ticket=ticket%2Fone",
      )
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      "https://chat.example/api/widget/embed-ticket",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ widget_key: "widget-secret" }),
      },
    )
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src)
      .not.toContain("widget-secret")
  })

  it("does not navigate the iframe on a non-OK embed-ticket response", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(new Response("forbidden", { status: 403 }))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("embed authorization failed"))
    })
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.getAttribute("src")).toBeNull()
  })

  it("does not navigate the iframe on an embed-ticket network failure", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockRejectedValueOnce(new Error("network unavailable"))

    runWidget({ "data-widget-key": "widget-secret" })

    await vi.waitFor(() => {
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("embed authorization request failed"))
    })
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.getAttribute("src")).toBeNull()
  })

  it("keeps the deprecated non-default token channel available without a ticket", () => {
    runWidget({ "data-token": "legacy-token" })

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")?.src).toBe(
      "https://chat.example/widget/chat/legacy-token?guest_id=guest-fixed",
    )
  })
})
