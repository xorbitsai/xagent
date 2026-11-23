import React from "react"
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/contexts/i18n-context", () => ({ useI18n: () => ({ t: (key: string) => key }) }))

vi.mock("@/lib/browser-location", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/browser-location")>()),
  replaceBrowserLocation: vi.fn(),
}))

import { replaceBrowserLocation } from "@/lib/browser-location"
import { LinkSessionPage } from "./link-session-page"
import type { LinkWorkerLike } from "./link-session-host"

const CODE = `lk_${"D".repeat(43)}`
const MINUTE = 60_000
const HOUR = 60 * MINUTE

type Listener = (event: { data?: unknown; preventDefault?: () => void }) => void

class FakeWorker implements LinkWorkerLike {
  posted: Array<Record<string, unknown>> = []
  terminated = false
  private listeners = new Map<string, Listener[]>()

  postMessage(message: unknown) {
    this.posted.push(message as Record<string, unknown>)
  }

  terminate() {
    this.terminated = true
  }

  addEventListener(type: "message" | "error" | "messageerror", listener: Listener) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), listener])
  }

  emit(type: "message" | "error" | "messageerror", data?: unknown) {
    for (const listener of this.listeners.get(type) ?? []) listener({ data, preventDefault: () => undefined })
  }

  count(type: string) {
    return this.posted.filter((m) => m.type === type).length
  }
}

function sessionFields() {
  const now = Date.now()
  return {
    session_token: "st_value",
    session_token_expires_at: new Date(now + 15 * MINUTE).toISOString(),
    absolute_expires_at: new Date(now + 8 * HOUR).toISOString(),
    agent: { id: 1, name: "Agent" },
  }
}

interface Mounted {
  worker: FakeWorker
  navigate: ReturnType<typeof vi.fn>
  createWorker: ReturnType<typeof vi.fn>
  unmount: () => void
}

function mount(options: { strict?: boolean } = {}): Mounted {
  const worker = new FakeWorker()
  const navigate = vi.fn()
  const createWorker = vi.fn(() => worker)
  const tree = <LinkSessionPage environment={{ createWorker, navigate }} />
  const { unmount } = render(options.strict ? <React.StrictMode>{tree}</React.StrictMode> : tree)
  return { worker, navigate, createWorker, unmount }
}

function setLocation(pathAndQuery: string, hash = `#code=${CODE}`) {
  window.history.replaceState(null, "", `${pathAndQuery}${hash}`)
}

function activate(m: Mounted) {
  act(() => {
    m.worker.emit("message", { type: "hello" })
    m.worker.emit("message", { type: "redeeming" })
    m.worker.emit("message", { type: "session", session: sessionFields() })
  })
}

function iframeEl(): HTMLIFrameElement | null {
  return document.querySelector("iframe")
}

function iframeSays(type: string, extra: Record<string, unknown> = {}, source?: Window | null) {
  const frame = iframeEl()
  act(() => {
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type, ...extra },
      origin: window.location.origin,
      source: source === undefined ? frame?.contentWindow ?? null : source,
    }))
  })
}

beforeEach(() => {
  vi.useFakeTimers()
  setLocation("/widget/chat/link")
})

afterEach(() => {
  cleanup()
  vi.useRealTimers()
  vi.unstubAllGlobals()
  window.history.replaceState(null, "", "/")
})

describe("link session page", () => {
  it("shows a loading state and the close button, with no iframe, until the first session arrives", () => {
    const m = mount()
    expect(iframeEl()).toBeNull()
    expect(screen.getByRole("status")).toHaveTextContent("widgetLink.loading")
    expect(screen.getByRole("button", { name: "widgetLink.close" })).toBeInTheDocument()
    act(() => {
      m.worker.emit("message", { type: "hello" })
      m.worker.emit("message", { type: "redeeming" })
    })
    expect(iframeEl()).toBeNull()
    activate(m)
    expect(iframeEl()).not.toBeNull()
  })

  it("erases the fragment, keeps the query, and starts the Worker with the code", () => {
    setLocation("/widget/chat/link?timezone=Australia%2FSydney")
    const m = mount()
    expect(window.location.hash).toBe("")
    expect(window.location.pathname + window.location.search).toBe("/widget/chat/link?timezone=Australia%2FSydney")
    expect(m.worker.posted).toEqual([{ type: "start", code: CODE }])
  })

  it("ends with link_invalid when the page opens without a code", () => {
    setLocation("/widget/chat/link", "")
    const m = mount()
    expect(m.createWorker).not.toHaveBeenCalled()
    expect(m.navigate).toHaveBeenCalledWith("/widget/chat/link/ended?reason=link_invalid")
  })

  it("leaves the page with a history-replacing navigation by default", () => {
    const worker = new FakeWorker()
    render(<LinkSessionPage environment={{ createWorker: () => worker }} />)
    act(() => {
      worker.emit("message", { type: "terminal", reason: "network" })
    })
    expect(replaceBrowserLocation).toHaveBeenCalledWith("/widget/chat/link/ended?reason=network")
  })

  it("builds the chat iframe with the timezone, no voice and only clipboard-write", () => {
    setLocation("/widget/chat/link?timezone=Australia%2FSydney&voice=1")
    const m = mount()
    activate(m)
    const frame = iframeEl() as HTMLIFrameElement
    expect(frame.getAttribute("src")).toBe("/widget/chat/session?timezone=Australia%2FSydney")
    expect(frame.getAttribute("src")).not.toContain("voice")
    expect(frame.getAttribute("allow")).toBe("clipboard-write")
    expect(frame.getAttribute("allow")).not.toContain("microphone")
  })

  it("leaves a safe-area margin around the iframe", () => {
    const m = mount()
    activate(m)
    const style = (iframeEl() as HTMLIFrameElement).getAttribute("style") ?? ""
    for (const side of ["top", "right", "bottom", "left"]) {
      expect(style).toContain(`env(safe-area-inset-${side})`)
    }
  })

  it("delivers the session to the iframe once it reports ready", () => {
    const m = mount()
    activate(m)
    const frame = iframeEl() as HTMLIFrameElement
    const posted = vi.fn()
    Object.defineProperty(frame, "contentWindow", { configurable: true, value: { postMessage: posted } })
    iframeSays("ready", {}, frame.contentWindow)
    expect(posted).toHaveBeenCalledTimes(1)
    const [message, target] = posted.mock.calls[0] as [Record<string, unknown>, string]
    expect(message).toMatchObject({ xagent: true, v: 1, type: "session_update", session_token: "st_value" })
    expect(target).toBe(window.location.origin)
  })

  it("ignores messages from another window or another origin", () => {
    const m = mount()
    activate(m)
    iframeSays("reconnect_request", {}, window)
    iframeSays("reconnect_request", {}, null)
    const frame = iframeEl() as HTMLIFrameElement
    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        data: { xagent: true, v: 1, type: "reconnect_request" },
        origin: "https://evil.example",
        source: frame.contentWindow,
      }))
    })
    expect(m.worker.count("renew")).toBe(0)
    iframeSays("reconnect_request")
    expect(m.worker.count("renew")).toBe(1)
  })

  it("ends with session_ended when the page is restored from the back-forward cache", () => {
    const m = mount()
    activate(m)
    const event = new Event("pageshow")
    Object.defineProperty(event, "persisted", { value: true })
    act(() => {
      window.dispatchEvent(event)
    })
    expect(m.worker.terminated).toBe(true)
    expect(m.navigate).toHaveBeenCalledWith("/widget/chat/link/ended?reason=session_ended")
  })

  it("wakes the Worker when the network returns or the page becomes visible", () => {
    const m = mount()
    activate(m)
    act(() => {
      window.dispatchEvent(new Event("online"))
    })
    expect(m.worker.count("wake")).toBe(1)
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(m.worker.count("wake")).toBe(2)
    vi.spyOn(document, "visibilityState", "get").mockReturnValue("hidden")
    act(() => {
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(m.worker.count("wake")).toBe(2)
  })

  it("starts exactly once under React strict mode", () => {
    const m = mount({ strict: true })
    expect(m.createWorker).toHaveBeenCalledTimes(1)
    expect(m.worker.count("start")).toBe(1)
    expect(m.navigate).not.toHaveBeenCalled()
    act(() => {
      vi.advanceTimersByTime(1)
    })
    expect(m.worker.terminated).toBe(false)
    activate(m)
    expect(iframeEl()).not.toBeNull()
  })

  it("stops the Worker and the timers when the page unmounts", () => {
    const m = mount()
    activate(m)
    m.unmount()
    act(() => {
      vi.advanceTimersByTime(9 * HOUR)
    })
    expect(m.worker.terminated).toBe(true)
    expect(m.navigate).not.toHaveBeenCalled()
  })
})

describe("link session page close button", () => {
  it("is visible at first, hides on widget_chrome_ready and returns on widget_chrome_not_ready", () => {
    const m = mount()
    activate(m)
    expect(screen.getByRole("button", { name: "widgetLink.close" })).toBeInTheDocument()
    iframeSays("widget_chrome_ready")
    expect(screen.queryByRole("button", { name: "widgetLink.close" })).toBeNull()
    iframeSays("widget_chrome_not_ready")
    expect(screen.getByRole("button", { name: "widgetLink.close" })).toBeInTheDocument()
  })

  it("is not affected by a ready message from another window", () => {
    const m = mount()
    activate(m)
    iframeSays("widget_chrome_ready", {}, window)
    expect(screen.getByRole("button", { name: "widgetLink.close" })).toBeInTheDocument()
  })

  it("ends with closed when clicked", () => {
    const m = mount()
    activate(m)
    fireEvent.click(screen.getByRole("button", { name: "widgetLink.close" }))
    expect(m.navigate).toHaveBeenCalledTimes(1)
    expect(m.navigate).toHaveBeenCalledWith("/widget/chat/link/ended?reason=closed")
  })

  it("is usable while the page is still waiting for the first session", () => {
    const m = mount()
    fireEvent.click(screen.getByRole("button", { name: "widgetLink.close" }))
    expect(m.navigate).toHaveBeenCalledWith("/widget/chat/link/ended?reason=closed")
    expect(m.worker.terminated).toBe(true)
  })

  it("floats at the top right and stays clear of the safe area", () => {
    mount()
    const style = screen.getByRole("button", { name: "widgetLink.close" }).getAttribute("style") ?? ""
    expect(style).toContain("env(safe-area-inset-top)")
    expect(style).toContain("env(safe-area-inset-right)")
  })
})

describe("link session page after the session ended", () => {
  it.each([
    ["no link code", () => {
      setLocation("/widget/chat/link", "")
      mount()
    }],
    ["Worker terminal before a session", () => {
      const m = mount()
      act(() => m.worker.emit("message", { type: "terminal", reason: "network" }))
    }],
    ["Worker terminal after a session", () => {
      const m = mount()
      activate(m)
      act(() => m.worker.emit("message", { type: "terminal", reason: "network" }))
    }],
  ])("shows the ended text without a spinner or a close button (%s)", (_name, end) => {
    end()
    expect(iframeEl()).toBeNull()
    expect(screen.queryByText("widgetLink.loading")).toBeNull()
    const status = screen.getByRole("status")
    expect(status).toHaveTextContent("widgetLink.ended.generic.title")
    expect(status).toHaveTextContent("widgetLink.ended.generic.description")
    expect(screen.queryByRole("button", { name: "widgetLink.close" })).toBeNull()
  })

  it("replaces the close button with the ended text once it is clicked", () => {
    const m = mount()
    activate(m)
    fireEvent.click(screen.getByRole("button", { name: "widgetLink.close" }))
    expect(m.navigate).toHaveBeenCalledTimes(1)
    expect(m.navigate).toHaveBeenCalledWith("/widget/chat/link/ended?reason=closed")
    expect(screen.queryByRole("button", { name: "widgetLink.close" })).toBeNull()
    expect(screen.getByRole("status")).toHaveTextContent("widgetLink.ended.generic.title")
  })
})

describe("link session page browser storage", () => {
  it("does not read or write storage while connecting, renewing and ending", () => {
    const localSet = vi.spyOn(Storage.prototype, "setItem")
    const localGet = vi.spyOn(Storage.prototype, "getItem")
    const sessionSet = vi.spyOn(window.sessionStorage, "setItem")
    const cookieSet = vi.fn()
    Object.defineProperty(document, "cookie", { configurable: true, get: () => "", set: cookieSet })
    const idbOpen = vi.fn()
    const cachesOpen = vi.fn()
    vi.stubGlobal("indexedDB", { open: idbOpen })
    vi.stubGlobal("caches", { open: cachesOpen })
    try {
      const m = mount()
      activate(m)
      iframeSays("ready")
      iframeSays("reconnect_request")
      act(() => {
        m.worker.emit("message", { type: "session", session: sessionFields() })
        m.worker.emit("message", { type: "terminal", reason: "network" })
      })
      expect(localSet).not.toHaveBeenCalled()
      expect(localGet).not.toHaveBeenCalled()
      expect(sessionSet).not.toHaveBeenCalled()
      expect(cookieSet).not.toHaveBeenCalled()
      expect(idbOpen).not.toHaveBeenCalled()
      expect(cachesOpen).not.toHaveBeenCalled()
    } finally {
      Reflect.deleteProperty(document, "cookie")
    }
  })
})
