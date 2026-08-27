import { readFileSync } from "node:fs"
import { resolve } from "node:path"
import { pathToFileURL } from "node:url"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useWidgetSession } from "./use-widget-session"

const widgetScriptPath = resolve(process.cwd(), "public/widget.js")
const widgetScript = readFileSync(widgetScriptPath, "utf8")
const widgetScriptUrl = pathToFileURL(widgetScriptPath).href

export const HOST = "https://chat.example"
export const GRANT = "eyJhbGciOiJIUzI1NiJ9.grant-one.sig"
const EXCHANGE_URL = `${HOST}/v1/external/chat/sessions`
const RECONNECT_URL = `${HOST}/v1/external/chat/sessions/reconnect`

const fetchMock = vi.fn()

function runWidget(attributes: Record<string, string>) {
  const script = document.createElement("script")
  script.src = `${HOST}/widget.js`
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", { configurable: true, value: script })
  window.eval(`${widgetScript}\n//# sourceURL=${widgetScriptUrl}`)
  return script
}

function iframeEl(): HTMLIFrameElement | null {
  return document.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")
}

function panelEl(): Element | null {
  return document.querySelector(".xagent-widget-panel")
}

function fabEl(): HTMLButtonElement | null {
  return document.querySelector<HTMLButtonElement>(".xagent-widget-fab")
}

const CLOSED_KEY = "xagent_widget_closed"

function spyOnIframePostMessage(frame = iframeEl()) {
  if (!frame?.contentWindow) throw new Error("iframe not mounted")
  return vi.spyOn(frame.contentWindow, "postMessage")
}

function fromSpecificIframe(
  frame: HTMLIFrameElement | null,
  type: string,
  extra: Record<string, unknown> = {},
) {
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type, ...extra },
    origin: HOST,
    source: frame?.contentWindow as Window,
  }))
}

function fromIframe(type: string, extra: Record<string, unknown> = {}) {
  fromSpecificIframe(iframeEl(), type, extra)
}

function bridgeWidgetSessionHook(frame: HTMLIFrameElement) {
  const parent = {
    postMessage: vi.fn((message: Record<string, unknown>) => {
      act(() => {
        window.dispatchEvent(new MessageEvent("message", {
          data: message,
          origin: HOST,
          source: frame.contentWindow as Window,
        }))
      })
    }),
  }
  Object.defineProperty(window, "parent", { configurable: true, value: parent })
  const postToIframe = vi.spyOn(frame.contentWindow!, "postMessage").mockImplementation((message) => {
    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        data: message,
        origin: HOST,
        source: parent as unknown as MessageEventSource,
      }))
    })
  })
  return { parent, postToIframe }
}

function jsonResponse(status: number, body: unknown, headers: Record<string, string> = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json", ...headers },
  })
}

function directJsonResponse(status: number, data: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers(),
    json: () => Promise.resolve(data),
  } as Response
}

function errorResponse(status: number, code: string) {
  return jsonResponse(status, { error: { code, message: "nope" } })
}

// vi.waitFor's condition here (fetchMock call count) goes true the instant
// fetch() is invoked, which happens synchronously during runWidget(). That
// resolves the waitFor promise before the mocked Response's real .json()
// body-read (an inherently async, multi-microtask-tick operation) has settled.
// A macrotask flush lets every already-scheduled microtask (the fetch/json/
// applySession chain) drain before we simulate the iframe's "ready" signal.
function flushAsync() {
  return new Promise<void>((resolve) => setTimeout(resolve, 0))
}

function firePageShow(persisted: boolean) {
  const event = new Event("pageshow") as PageTransitionEvent & { persisted: boolean }
  Object.defineProperty(event, "persisted", { value: persisted })
  window.dispatchEvent(event)
}

function firePageHide(persisted: boolean) {
  const event = new Event("pagehide") as PageTransitionEvent & { persisted: boolean }
  Object.defineProperty(event, "persisted", { value: persisted })
  window.dispatchEvent(event)
}

function firePageRestore() {
  firePageHide(true)
  firePageShow(true)
}

function exchangeBody(overrides: Record<string, unknown> = {}) {
  return {
    session_token: "st_first",
    session_token_expires_at: new Date(Date.now() + 15 * 60_000).toISOString(),
    reconnect_token: "rt_first",
    session: {
      absolute_expires_at: new Date(Date.now() + 8 * 3_600_000).toISOString(),
      agent: {
        id: 42,
        name: "Care Assistant",
        description: "Rosters and shifts",
        logo_url: `${HOST}/logo.png`,
        suggested_prompts: ["What is my next shift?"],
      },
    },
    ...overrides,
  }
}

describe("widget session mode", () => {
  let currentScriptDescriptor: PropertyDescriptor | undefined
  let parentDescriptor: PropertyDescriptor | undefined
  // runWidget()'s attach() registers message and bfcache lifecycle listeners
  // directly on the shared jsdom `window` (vitest reuses one window per test
  // file). Without tracking and removing them, a controller from an earlier
  // test stays subscribed and reacts to a later test's fromIframe()/
  // firePageShow() dispatches, corrupting that later test's fetch-call
  // counts. We wrap addEventListener for the duration of each test to record
  // every (type, listener) pair the production code registers on window, and
  // remove them all in afterEach.
  let windowListeners: Array<[string, EventListenerOrEventListenerObject]> = []
  let realAddEventListener: typeof window.addEventListener

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    parentDescriptor = Object.getOwnPropertyDescriptor(window, "parent")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    localStorage.clear()
    // The grant dedupe registry lives on window and survives between tests in a file.
    Reflect.deleteProperty(window as unknown as Record<string, unknown>, "__xagentWidgetGrants")
    vi.stubGlobal("fetch", fetchMock)
    fetchMock.mockReset()

    windowListeners = []
    realAddEventListener = window.addEventListener.bind(window)
    vi.spyOn(window, "addEventListener").mockImplementation((type, listener, options) => {
      windowListeners.push([type, listener as EventListenerOrEventListenerObject])
      realAddEventListener(type, listener, options)
    })
  })

  afterEach(async () => {
    for (const container of document.querySelectorAll(".xagent-widget-container")) {
      container.remove()
    }
    // Let each controller's MutationObserver run its production teardown
    // before Vitest removes the jsdom globals.
    await Promise.resolve()
    await Promise.resolve()

    for (const [type, listener] of windowListeners) {
      window.removeEventListener(type, listener)
    }
    windowListeners = []

    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    if (parentDescriptor) {
      Object.defineProperty(window, "parent", parentDescriptor)
    }
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it("harness boots the guest path unchanged", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ticket: "t", agent_id: 17 }))
    runWidget({ "data-widget-key": "widget-secret" })
    await vi.waitFor(() => {
      expect(iframeEl()?.src).toContain("/widget/chat/default")
    })
  })

  it("passes data-timezone to the session iframe", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT, "data-timezone": "Australia/Perth" })

    expect(iframeEl()?.src).toBe(`${HOST}/widget/chat/session?timezone=Australia%2FPerth`)
  })

  it("appends data-timezone to the guest iframe URL that already has a query", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ticket: "t", agent_id: 17 }))

    runWidget({ "data-widget-key": "widget-secret", "data-timezone": "Australia/Perth" })

    await vi.waitFor(() => {
      expect(iframeEl()?.src).toContain("&timezone=Australia%2FPerth")
    })
    expect(iframeEl()?.src).toContain("?guest_id=")
  })

  it("leaves the iframe URL untouched for a blank data-timezone", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT, "data-timezone": "   " })

    expect(iframeEl()?.src).toBe(`${HOST}/widget/chat/session`)
  })

  it("still loads the session iframe when data-timezone is malformed", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    // A lone surrogate makes encodeURIComponent throw; the widget must fall
    // back to a plain iframe URL rather than never assigning src.
    runWidget({ "data-encrypted-context": GRANT, "data-timezone": "\uD800" })

    expect(iframeEl()?.src).toBe(`${HOST}/widget/chat/session`)
  })

  it("still loads the guest iframe when data-timezone is malformed", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { ticket: "t", agent_id: 17 }))

    runWidget({ "data-widget-key": "widget-secret", "data-timezone": "\uD800" })

    await vi.waitFor(() => {
      expect(iframeEl()?.src).toContain("/widget/chat/default")
    })
    expect(iframeEl()?.src).not.toContain("timezone=")
  })

  it("navigates the iframe to the session URL and exchanges the grant immediately", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    const observeSpy = vi.spyOn(MutationObserver.prototype, "observe")

    runWidget({ "data-encrypted-context": GRANT })

    expect(iframeEl()?.src).toBe(`${HOST}/widget/chat/session`)
    // Two MutationObservers now exist in session mode: the widget's own
    // panel-removal teardown (armed before mode.attach runs) and this
    // controller's iframe-connectivity one, both on document.documentElement
    // with the same options -- so "was called with these args" alone no
    // longer pins this controller's own observer specifically. Assert every
    // observe() call has the expected target/options, so a regression in
    // either one (e.g. this controller's own observer losing `subtree`, or
    // reverting to document.body -- which misses a host framework replacing
    // <body> wholesale on navigation) still fails here instead of being
    // masked by the other happening to be correct.
    expect(observeSpy).toHaveBeenCalledTimes(2)
    observeSpy.mock.calls.forEach(([target, options]) => {
      expect(target).toBe(document.documentElement)
      expect(options).toEqual({ childList: true, subtree: true })
    })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(fetchMock).toHaveBeenCalledWith(EXCHANGE_URL, expect.objectContaining({
      method: "POST",
      credentials: "omit",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ encrypted_context: GRANT }),
    }))
  })

  it("consumes the response body returned by the session exchange", async () => {
    const response = jsonResponse(200, exchangeBody())
    fetchMock.mockResolvedValueOnce(response)

    runWidget({ "data-encrypted-context": GRANT })

    await vi.waitFor(() => {
      expect(response.bodyUsed).toBe(true)
    })
  })

  it("auto-opens once the initial grant exchange succeeds, for a visitor who last left it open", async () => {
    localStorage.setItem(CLOSED_KEY, "false")
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT })

    // Session mode sets iframe.src synchronously and unconditionally --
    // unlike guest mode, that alone doesn't mean the grant is usable, so
    // this has to wait for the exchange itself to resolve.
    await vi.waitFor(() => {
      expect(panelEl()).toHaveClass("open")
    })
  })

  it("does not auto-open when the initial grant exchange fails, even for a visitor who left it open", async () => {
    // A returning visitor's grant can have expired, been consumed, or the
    // exchange can just fail -- auto-opening ahead of that would show the
    // degraded/terminal screen unprompted, on every reload.
    localStorage.setItem(CLOSED_KEY, "false")
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(401, "grant_expired"))

    runWidget({ "data-encrypted-context": GRANT })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[grant_expired]"),
    ))
    expect(panelEl()).not.toHaveClass("open")
  })

  it("does not re-open a panel the visitor has since closed when a later reconnect succeeds", async () => {
    localStorage.setItem(CLOSED_KEY, "false")
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    // The initial exchange auto-opened it (per the test above); confirm that
    // before the visitor closes it themselves, so a later false-negative here
    // can't be mistaken for this test's own point.
    expect(panelEl()).toHaveClass("open")

    fabEl()?.click()
    expect(panelEl()).not.toHaveClass("open")

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await flushAsync()

    expect(panelEl()).not.toHaveClass("open")
  })

  it("sends the opaque grant value verbatim after checking that it is not blank", () => {
    const rawGrant = `  ${GRANT}\n`
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": rawGrant })

    expect(fetchMock).toHaveBeenCalledWith(EXCHANGE_URL, expect.objectContaining({
      body: JSON.stringify({ encrypted_context: rawGrant }),
    }))
  })

  it("fails closed on an empty grant attribute with no DOM and no network", () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

    runWidget({ "data-encrypted-context": "   " })

    expect(fetchMock).not.toHaveBeenCalled()
    expect(document.querySelector(".xagent-widget-container")).toBeNull()
    expect(document.head.querySelector("style")).toBeNull()
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[grant_malformed]"))
  })

  it.each(["data-widget-key", "data-token"])(
    "fails closed when %s coexists with the grant",
    (legacyAttribute) => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)

      const script = runWidget({ "data-encrypted-context": GRANT, [legacyAttribute]: "legacy" })

      expect(fetchMock).not.toHaveBeenCalled()
      expect(document.querySelector(".xagent-widget-container")).toBeNull()
      expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[attribute_conflict]"))
      expect(script.hasAttribute("data-encrypted-context")).toBe(false)
    },
  )

  it("keeps cosmetic attributes working in session mode", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT, "data-button-color": "rgb(1, 2, 3)" })

    expect(document.head.querySelector("style")?.innerHTML).toContain("rgb(1, 2, 3)")
  })

  it("never writes a guest id in session mode", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": GRANT })

    expect(localStorage.getItem("xagent_guest_id")).toBeNull()
  })

  it("removes the grant attribute from the DOM once it is read", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    const script = runWidget({ "data-encrypted-context": GRANT })

    expect(script.hasAttribute("data-encrypted-context")).toBe(false)
  })

  it("deduplicates the same grant while keeping different grants and message channels isolated", async () => {
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      const body = JSON.parse(init.body as string)
      const suffix = body.encrypted_context === GRANT ? "first" : "second"
      return Promise.resolve(jsonResponse(200, exchangeBody({
        session_token: `st_${suffix}`,
        reconnect_token: `rt_${suffix}`,
      })))
    })

    runWidget({ "data-encrypted-context": GRANT })
    const duplicateScript = runWidget({ "data-encrypted-context": GRANT })

    expect(document.querySelectorAll(".xagent-widget-container")).toHaveLength(1)
    expect(warnSpy).toHaveBeenCalledWith(expect.stringMatching(
      /single-use.*fresh grant.*\[duplicate_init\]/,
    ))
    expect(duplicateScript.hasAttribute("data-encrypted-context")).toBe(false)

    runWidget({ "data-encrypted-context": `${GRANT}-other` })

    const frames = Array.from(document.querySelectorAll<HTMLIFrameElement>(".xagent-widget-iframe"))
    expect(frames).toHaveLength(2)
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await flushAsync()
    expect(fetchMock.mock.calls.map((call) => JSON.parse(call[1].body))).toEqual([
      { encrypted_context: GRANT },
      { encrypted_context: `${GRANT}-other` },
    ])

    const firstPost = spyOnIframePostMessage(frames[0])
    const secondPost = spyOnIframePostMessage(frames[1])
    fromSpecificIframe(frames[1], "ready")

    expect(firstPost).not.toHaveBeenCalled()
    expect(secondPost).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
  })

  it("uses the exact 32-bit FNV-1a digest for grant dedupe keys", () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))

    runWidget({ "data-encrypted-context": "hello" })

    expect(
      (window as unknown as { __xagentWidgetGrants: Record<string, boolean> })
        .__xagentWidgetGrants,
    ).toEqual({ g4f9f2cab: true })
  })

  it("pushes session_update to the iframe once it announces ready", async () => {
    const body = exchangeBody()
    const absoluteExpiresAt = body.session.absolute_expires_at
    fetchMock.mockResolvedValueOnce(jsonResponse(200, body))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(1)
    const [message, targetOrigin] = post.mock.calls[0]
    expect(targetOrigin).toBe(HOST)
    expect(message).toMatchObject({
      xagent: true,
      v: 1,
      type: "session_update",
      session_token: "st_first",
      absolute_expires_at: absoluteExpiresAt,
      agent: { id: 42, name: "Care Assistant" },
    })
    expect(message).not.toHaveProperty("reconnect_token")
  })

  it("holds only the latest state until ready arrives", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await flushAsync()

    expect(post).not.toHaveBeenCalled()
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_second" })
  })

  it("re-sends the current state on every ready", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")
    fromIframe("ready")

    expect(post).toHaveBeenCalledTimes(2)
    expect(post.mock.calls[1][0]).toMatchObject({ type: "session_update", session_token: "st_first" })
    expect(post.mock.calls[1][0].session_delivery_id).toBe(post.mock.calls[0][0].session_delivery_id)
  })

  it("admits three logical recoveries and terminalizes the fourth before fetch", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second", reconnect_token: "rt_second" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_third", reconnect_token: "rt_third" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fourth", reconnect_token: "rt_fourth" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")

    for (let recovery = 0; recovery < 3; recovery += 1) {
      fromIframe("reconnect_request", { reason: "ws_closed" })
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(recovery + 2))
      await flushAsync()
    }
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await flushAsync()

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("keeps each admitted logical recovery on its own four-request HTTP budget", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second", reconnect_token: "rt_second" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_third", reconnect_token: "rt_third" })))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(7_000)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(5)
  })

  it("starts stability only from the exact correlated open and makes duplicate opens idempotent", async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    const deliveryId = post.mock.calls[0][0].session_delivery_id

    await vi.advanceTimersByTimeAsync(14_999)
    fromIframe("session_connection_open", { session_delivery_id: deliveryId })
    fromIframe("session_connection_open", { session_delivery_id: deliveryId })
    await vi.advanceTimersByTimeAsync(14_999)
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it.each([
    [14_999, 2],
    [15_000, 3],
  ])(
    "after %i ms of correlated open, admits exactly %i recoveries from the remaining rapid budget",
    async (stableMs, admittedRecoveries) => {
      vi.useFakeTimers()
      fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(200, exchangeBody())))
      runWidget({ "data-encrypted-context": GRANT })
      const post = spyOnIframePostMessage()
      await vi.advanceTimersByTimeAsync(0)
      fromIframe("ready")

      fromIframe("reconnect_request", { reason: "ws_closed" })
      await vi.advanceTimersByTimeAsync(0)
      const deliveryId = post.mock.calls
        .filter(([message]) => message.type === "session_update")
        .at(-1)?.[0].session_delivery_id
      fromIframe("session_connection_open", { session_delivery_id: deliveryId })
      await vi.advanceTimersByTimeAsync(stableMs)

      for (let recovery = 0; recovery < admittedRecoveries; recovery += 1) {
        fromIframe("reconnect_request", { reason: "ws_closed" })
        await vi.advanceTimersByTimeAsync(0)
      }
      const requestsAfterAdmissions = fetchMock.mock.calls.length
      fromIframe("reconnect_request", { reason: "ws_closed" })
      await vi.advanceTimersByTimeAsync(0)

      expect(requestsAfterAdmissions).toBe(admittedRecoveries + 2)
      expect(fetchMock).toHaveBeenCalledTimes(requestsAfterAdmissions)
      expect(post).toHaveBeenCalledWith(
        expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
        HOST,
      )
    },
  )

  it("arms only current delivery B and treats duplicate B opens as idempotent before and after completion", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second", reconnect_token: "rt_second" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_third", reconnect_token: "rt_third" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fourth", reconnect_token: "rt_fourth" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fifth", reconnect_token: "rt_fifth" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    const deliveryA = post.mock.calls.at(-1)?.[0].session_delivery_id

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    const deliveryB = post.mock.calls.at(-1)?.[0].session_delivery_id
    expect(deliveryB).not.toBe(deliveryA)

    fromIframe("session_connection_open", { session_delivery_id: deliveryA })
    expect(vi.getTimerCount()).toBe(0)
    fromIframe("session_connection_open", { session_delivery_id: deliveryB })
    expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(10_000)
    fromIframe("session_connection_open", { session_delivery_id: deliveryB })
    expect(vi.getTimerCount()).toBe(1)
    await vi.advanceTimersByTimeAsync(5_000)
    expect(vi.getTimerCount()).toBe(0)
    fromIframe("ready")
    await vi.advanceTimersByTimeAsync(0)
    expect(vi.getTimerCount()).toBe(0)
    fromIframe("session_connection_open", { session_delivery_id: deliveryB })
    expect(vi.getTimerCount()).toBe(0)

    for (let recovery = 0; recovery < 3; recovery += 1) {
      fromIframe("reconnect_request", { reason: "ws_closed" })
      await vi.advanceTimersByTimeAsync(0)
    }
    expect(fetchMock).toHaveBeenCalledTimes(5)
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("a recovery at 14,999 ms invalidates the old stability callback before it can reset rapid count", async () => {
    vi.useFakeTimers()
    let resolvePreBoundary: (response: Response) => void = () => undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolvePreBoundary = resolve
      }))
      .mockImplementation(() => Promise.resolve(jsonResponse(200, exchangeBody())))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    const deliveryId = post.mock.calls
      .filter(([message]) => message.type === "session_update")
      .at(-1)?.[0].session_delivery_id
    fromIframe("session_connection_open", { session_delivery_id: deliveryId })
    await vi.advanceTimersByTimeAsync(14_999)

    fromIframe("reconnect_request", { reason: "ws_closed" })
    expect(fetchMock).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(1)
    resolvePreBoundary(jsonResponse(200, exchangeBody({
      session_token: "st_third",
      reconnect_token: "rt_third",
    })))
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(4)

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("mints a fresh delivery epoch for a healthy bfcache restore and resets only at 15 seconds", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second", reconnect_token: "rt_second" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_third", reconnect_token: "rt_third" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fourth", reconnect_token: "rt_fourth" })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fifth", reconnect_token: "rt_fifth" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await vi.advanceTimersByTimeAsync(0)
    const oldId = post.mock.calls.at(-1)?.[0].session_delivery_id
    fromIframe("session_connection_open", { session_delivery_id: oldId })
    firePageHide(true)
    firePageShow(true)
    const newId = post.mock.calls.at(-1)?.[0].session_delivery_id
    expect(newId).not.toBe(oldId)

    for (const session_delivery_id of [undefined, " ", 1, true, {}, []]) {
      fromIframe("session_connection_open", { session_delivery_id })
    }
    fromIframe("session_connection_open", { session_delivery_id: oldId })
    fromIframe("session_connection_open", { session_delivery_id: newId })
    await vi.advanceTimersByTimeAsync(14_999)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(1)

    for (let recovery = 0; recovery < 3; recovery += 1) {
      fromIframe("reconnect_request", { reason: "ws_closed" })
      await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(recovery + 3))
      await vi.advanceTimersByTimeAsync(0)
    }
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("keeps the resumed recovery owner when the canceled wrapper settles before a duplicate signal", async () => {
    vi.useFakeTimers()
    let frozenSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
        session_token_expires_at: new Date(Date.now() + 61_000).toISOString(),
      })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_third",
        reconnect_token: "rt_third",
        session_token_expires_at: new Date(Date.now() + 61_000).toISOString(),
      })))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        frozenSignal = init.signal as AbortSignal
        return new Promise<Response>((_resolve, reject) => {
          frozenSignal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")), { once: true })
        })
      })
      .mockImplementationOnce(() => new Promise<Response>(() => undefined))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(4))

    await vi.advanceTimersByTimeAsync(2_000)
    firePageHide(true)
    firePageShow(true)
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(5))
    await vi.advanceTimersByTimeAsync(0)
    expect(frozenSignal?.aborted).toBe(true)

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(post).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal" }),
      HOST,
    )
  })

  it("ignores messages from a foreign origin, a foreign source, or a foreign shape", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "ready" },
      origin: "https://evil.example",
      source: iframeEl()?.contentWindow as Window,
    }))
    window.dispatchEvent(new MessageEvent("message", {
      data: { xagent: true, v: 1, type: "ready" },
      origin: HOST,
      source: window,
    }))
    fromIframe("ready", { v: 2 })
    window.dispatchEvent(new MessageEvent("message", {
      data: { type: "ready" },
      origin: HOST,
      source: iframeEl()?.contentWindow as Window,
    }))

    expect(post).not.toHaveBeenCalled()
  })

  it("goes terminal and tells the frame when the grant is rejected outright", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(401, "signature_invalid"))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[signature_invalid]"),
    ))
    expect(fetchMock).toHaveBeenCalledTimes(1)  // integrity class never retries or reconnects
    fromIframe("ready")
    expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "signature_invalid" })
  })

  it("goes terminal on a stale grant when no reconnect token is held", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(401, "grant_already_used"))
    runWidget({ "data-encrypted-context": GRANT })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[grant_already_used]"),
    ))
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("logs the registered exchange diagnostic reason without forwarding it to the iframe", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(403, {
      error: {
        code: "agent_not_granted",
        reason: "origin_not_allowed",
        message: "remote diagnostic must not be logged",
      },
    }))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      "Xagent Widget: chat unavailable [agent_not_granted/origin_not_allowed] (HTTP 403).",
    ))
    fromIframe("ready")

    expect(post.mock.calls[0][0]).toEqual({
      xagent: true,
      v: 1,
      type: "session_terminal",
      code: "agent_not_granted",
    })
  })

  it("scopes a registered diagnostic reason to its owning terminal code", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(401, {
      error: {
        code: "session_expired",
        reason: "origin_not_allowed",
      },
    }))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy.mock.calls).toEqual([[
      "Xagent Widget: chat unavailable [session_expired] (HTTP 401).",
    ]]))
    fromIframe("ready")

    expect(post.mock.calls[0][0]).toEqual({
      xagent: true,
      v: 1,
      type: "session_terminal",
      code: "session_expired",
    })
  })

  it("ignores an inherited synthetic session failure code", async () => {
    const syntheticCodeDescriptor = Object.getOwnPropertyDescriptor(Object.prototype, "syntheticCode")
    Object.defineProperty(Object.prototype, "syntheticCode", {
      configurable: true,
      value: "network_unavailable",
    })
    try {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
      fetchMock.mockResolvedValueOnce(jsonResponse(403, {
        error: {
          code: "agent_not_granted",
          reason: "origin_not_allowed",
        },
      }))
      runWidget({ "data-encrypted-context": GRANT })
      const post = spyOnIframePostMessage()

      await vi.waitFor(() => expect(errorSpy.mock.calls).toEqual([[
        "Xagent Widget: chat unavailable [agent_not_granted/origin_not_allowed] (HTTP 403).",
      ]]))
      fromIframe("ready")

      expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "agent_not_granted" })
    } finally {
      if (syntheticCodeDescriptor) {
        Object.defineProperty(Object.prototype, "syntheticCode", syntheticCodeDescriptor)
      } else {
        Reflect.deleteProperty(Object.prototype, "syntheticCode")
      }
    }
  })

  it("keeps a registered diagnostic intact when Object.prototype has a toString tag", async () => {
    const toStringTagDescriptor = Object.getOwnPropertyDescriptor(Object.prototype, Symbol.toStringTag)
    Object.defineProperty(Object.prototype, Symbol.toStringTag, {
      configurable: true,
      value: "polluted",
    })
    try {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
      fetchMock.mockResolvedValueOnce(jsonResponse(403, {
        error: {
          code: "agent_not_granted",
          reason: "origin_not_allowed",
        },
      }))
      runWidget({ "data-encrypted-context": GRANT })
      const post = spyOnIframePostMessage()

      await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
        "Xagent Widget: chat unavailable [agent_not_granted/origin_not_allowed] (HTTP 403).",
      ))
      fromIframe("ready")

      expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "agent_not_granted" })
    } finally {
      if (toStringTagDescriptor) {
        Object.defineProperty(Object.prototype, Symbol.toStringTag, toStringTagDescriptor)
      } else {
        Reflect.deleteProperty(Object.prototype, Symbol.toStringTag)
      }
    }
  })

  it("keeps the existing exchange diagnostic unchanged when the reason is absent", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(403, "agent_not_granted"))
    runWidget({ "data-encrypted-context": GRANT })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      "Xagent Widget: chat unavailable [agent_not_granted] (HTTP 403).",
    ))
  })

  it("logs the registered reconnect diagnostic reason without forwarding it to the iframe", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(403, {
        error: {
          code: "agent_not_granted",
          reason: "origin_not_allowed",
          message: "remote diagnostic must not be logged",
        },
      }))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await flushAsync()
    fromIframe("ready")
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      "Xagent Widget: chat unavailable [agent_not_granted/origin_not_allowed] (HTTP 403).",
    ))
    expect(post).toHaveBeenCalledWith({
      xagent: true,
      v: 1,
      type: "session_terminal",
      code: "agent_not_granted",
    }, HOST)
  })

  it.each([
    ["root array", () => ["agent_not_granted", "origin_not_allowed"], "unexpected_error"],
    ["missing error", () => ({}), "unexpected_error"],
    ["null error", () => ({ error: null }), "unexpected_error"],
    ["error array", () => ({ error: ["agent_not_granted", "origin_not_allowed"] }), "unexpected_error"],
    ["non-string code", () => ({ error: { code: 42, reason: "origin_not_allowed", message: "sentinel" } }), "unexpected_error"],
    ["non-string code array", () => ({ error: { code: ["agent_not_granted"], reason: "origin_not_allowed", message: "sentinel" } }), "unexpected_error"],
    ["missing reason", () => ({ error: { code: "agent_not_granted", message: "sentinel" } }), "agent_not_granted"],
    ["null reason", () => ({ error: { code: "agent_not_granted", reason: null, message: "sentinel" } }), "agent_not_granted"],
    ["non-string reason", () => ({ error: { code: "agent_not_granted", reason: 42, message: "sentinel" } }), "agent_not_granted"],
    ["non-string reason array", () => ({ error: { code: "agent_not_granted", reason: ["origin_not_allowed"], message: "sentinel" } }), "agent_not_granted"],
    ["unregistered reason", () => ({ error: { code: "agent_not_granted", reason: "future_reason", message: "sentinel" } }), "agent_not_granted"],
    ["prototype-key reason", () => ({ error: { code: "agent_not_granted", reason: "toString", message: "sentinel" } }), "agent_not_granted"],
  ])("suppresses an untrusted %s diagnostic component from a JSON response", async (_case, makeData, expectedCode) => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(403, makeData()))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      `Xagent Widget: chat unavailable [${expectedCode}] (HTTP 403).`,
    ))
    expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining("origin_not_allowed"))
    expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining("sentinel"))
    fromIframe("ready")

    expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: expectedCode })
  })

  it("fails closed on null-prototype response objects", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const error = Object.assign(Object.create(null), {
      code: "agent_not_granted",
      reason: "origin_not_allowed",
    })
    const data = Object.assign(Object.create(null), { error })
    fetchMock.mockResolvedValueOnce(directJsonResponse(403, data))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      "Xagent Widget: chat unavailable [unexpected_error] (HTTP 403).",
    ))
    expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining("origin_not_allowed"))
    fromIframe("ready")

    expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "unexpected_error" })
  })

  it.each([
    [
      "data.error",
      "error",
      { code: "agent_not_granted", reason: "origin_not_allowed" },
      () => ({}),
      "unexpected_error",
    ],
    [
      "error.code",
      "code",
      "agent_not_granted",
      () => ({ error: { reason: "origin_not_allowed" } }),
      "unexpected_error",
    ],
    [
      "error.reason",
      "reason",
      "origin_not_allowed",
      () => ({ error: { code: "agent_not_granted" } }),
      "agent_not_granted",
    ],
  ])("suppresses an inherited %s diagnostic component from Object.prototype", async (
    _boundary,
    property,
    inheritedValue,
    makeData,
    expectedCode,
  ) => {
    const descriptor = Object.getOwnPropertyDescriptor(Object.prototype, property)
    Object.defineProperty(Object.prototype, property, {
      configurable: true,
      writable: true,
      value: inheritedValue,
    })
    try {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
      fetchMock.mockResolvedValueOnce(jsonResponse(403, makeData()))
      runWidget({ "data-encrypted-context": GRANT })
      const post = spyOnIframePostMessage()

      await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
        `Xagent Widget: chat unavailable [${expectedCode}] (HTTP 403).`,
      ))
      expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining("origin_not_allowed"))
      fromIframe("ready")

      expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: expectedCode })
    } finally {
      if (descriptor) {
        Object.defineProperty(Object.prototype, property, descriptor)
      } else {
        Reflect.deleteProperty(Object.prototype, property)
      }
    }
  })

  it("does not retry a coded 4xx and reports unexpected_error for an uncoded one", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(new Response("<html>payload too large</html>", { status: 413 }))
    runWidget({ "data-encrypted-context": GRANT })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[unexpected_error] (HTTP 413)"),
    ))
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it.each([
    ["grant_malformed", 400],
    ["encryption_required", 400],
    ["agent_not_granted", 403],
    ["agent_not_available", 409],
    ["widget_disabled", 409],
    ["invalid_input", 422],
    ["invalid_runtime_context", 422],
  ])("goes terminal on %s without retrying", async (code, status) => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(status as number, code as string))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(`[${code}] (HTTP ${status})`),
    ))
    expect(fetchMock).toHaveBeenCalledTimes(1)
    fromIframe("ready")
    expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code })
  })

  it.each(["future_code_v2", "toString"])(
    "fails closed on the unrecognized error code %s",
    async (unknownCode) => {
      const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
      fetchMock.mockResolvedValueOnce(errorResponse(403, unknownCode))
      runWidget({ "data-encrypted-context": GRANT })
      const post = spyOnIframePostMessage()

      await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining("[unexpected_error] (HTTP 403)"),
      ))
      expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining(`[${unknownCode}]`))
      fromIframe("ready")
      expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "unexpected_error" })
    },
  )

  it("fails closed when a successful response is missing required session fields", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(200, { session_token: "st_only" }))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await flushAsync()
    fromIframe("ready")

    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[unexpected_error] (HTTP 200)"))
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "unexpected_error" }),
      HOST,
    )
    expect(post).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update" }),
      HOST,
    )
  })

  it("publishes degraded between retryable parent attempts and resumes the same phase on success", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_recovered" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    await vi.advanceTimersByTimeAsync(0)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_degraded", code: "network_unavailable" }),
      HOST,
    )

    await vi.advanceTimersByTimeAsync(1_000)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_recovered" }),
      HOST,
    )
  })

  it("retries an uncoded 5xx three times with 1s/2s/4s backoff, then terminalizes network_unavailable", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(
      new Response("<html>bad gateway</html>", { status: 502 }),
    ))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()

    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1000)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(2000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(4000)
    expect(fetchMock).toHaveBeenCalledTimes(4)

    await vi.advanceTimersByTimeAsync(8000)
    expect(fetchMock).toHaveBeenCalledTimes(4)  // four attempts total, never a fifth
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable] (HTTP 502)"))
    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
    await vi.advanceTimersByTimeAsync(30_000)
    expect(fetchMock).toHaveBeenCalledTimes(4)
  })

  it("retries an unknown coded 5xx before reporting network_unavailable", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(errorResponse(503, "future_server_code")))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(7000)

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable] (HTTP 503)"))
    expect(errorSpy).not.toHaveBeenCalledWith(expect.stringContaining("[future_server_code]"))
  })

  it("keeps unknown-5xx retry warnings and terminal frames code-only", async () => {
    vi.useFakeTimers()
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(503, {
      error: {
        code: "future_server_code",
        reason: "origin_not_allowed",
        message: "retry diagnostic must not be logged",
      },
    })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    await vi.advanceTimersByTimeAsync(7_000)

    expect(warnSpy.mock.calls).toEqual([
      ["Xagent Widget: session recovery is retrying [network_unavailable]."],
      ["Xagent Widget: session recovery is retrying [network_unavailable]."],
      ["Xagent Widget: session recovery is retrying [network_unavailable]."],
    ])
    expect(errorSpy).toHaveBeenCalledWith(
      "Xagent Widget: chat unavailable [network_unavailable] (HTTP 503).",
    )
    expect(post.mock.calls.map(([message]) => message)).toEqual([
      { xagent: true, v: 1, type: "session_degraded", code: "network_unavailable" },
      { xagent: true, v: 1, type: "session_degraded", code: "network_unavailable" },
      { xagent: true, v: 1, type: "session_degraded", code: "network_unavailable" },
      { xagent: true, v: 1, type: "session_terminal", code: "network_unavailable" },
    ])
  })

  it("retries a server-supplied network_unavailable 5xx as an unknown transport failure", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(errorResponse(503, "network_unavailable")))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(7_000)

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable] (HTTP 503)"))
  })

  it("honors a known error code on 5xx without status-based retry", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(errorResponse(503, "widget_disabled")))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(7000)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[widget_disabled] (HTTP 503)"))
  })

  it("fails closed on an uncoded 429 instead of guessing rate_limited", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(
      new Response("<html>too many requests</html>", { status: 429 }),
    ))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(3000)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[unexpected_error] (HTTP 429)"))
  })

  it("treats a rejected fetch as the network class", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"))
    runWidget({ "data-encrypted-context": GRANT })

    await vi.advanceTimersByTimeAsync(7000)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable]"))
  })

  it("aborts an attempt after 5s and keeps the whole budget inside the 30s idempotency window", async () => {
    vi.useFakeTimers()
    const signals: AbortSignal[] = []
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      signals.push(init.signal as AbortSignal)
      return new Promise((_resolve, reject) => {
        (init.signal as AbortSignal).addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      })
    })
    runWidget({ "data-encrypted-context": GRANT })

    await vi.advanceTimersByTimeAsync(5000)
    expect(signals[0].aborted).toBe(true)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    // attempt 2 at 6s, attempt 3 at 13s, attempt 4 at 22s, all done by 27s
    await vi.advanceTimersByTimeAsync(22_000)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(vi.getTimerCount()).toBe(0)
  })

  it("keeps the 5s deadline active while the response body is being read", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const signals: AbortSignal[] = []
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      const signal = init.signal as AbortSignal
      signals.push(signal)
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          signal.addEventListener("abort", () => {
            controller.error(new DOMException("aborted", "AbortError"))
          }, { once: true })
        },
      })
      return Promise.resolve(new Response(body, {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }))
    })
    runWidget({ "data-encrypted-context": GRANT })

    await vi.advanceTimersByTimeAsync(27_000)

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(signals).toHaveLength(4)
    expect(signals.every((signal) => signal.aborted)).toBe(true)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable]"))
    expect(vi.getTimerCount()).toBe(0)
  })

  it("honors Retry-After on 429 at most twice", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(
      jsonResponse(429, { error: { code: "rate_limited" } }, { "Retry-After": "3" }),
    ))
    runWidget({ "data-encrypted-context": GRANT })

    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(3000)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    await vi.advanceTimersByTimeAsync(3000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(3000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[rate_limited] (HTTP 429)"))
  })

  it("keeps rate-limit retry warnings and degraded frames code-only", async () => {
    vi.useFakeTimers()
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(429, {
      error: {
        code: "rate_limited",
        reason: "origin_not_allowed",
        message: "retry diagnostic must not be logged",
      },
    }, { "Retry-After": "1" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    await vi.advanceTimersByTimeAsync(0)

    expect(warnSpy).toHaveBeenCalledWith(
      "Xagent Widget: session recovery is retrying [rate_limited].",
    )
    expect(post).toHaveBeenCalledWith(
      { xagent: true, v: 1, type: "session_degraded", code: "rate_limited" },
      HOST,
    )
  })

  it("honors an HTTP-date Retry-After value", async () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-27T00:00:00Z"))
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(
      429,
      { error: { code: "rate_limited" } },
      { "Retry-After": "Mon, 27 Jul 2026 00:00:03 GMT" },
    )))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(2999)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it.each([
    ["a negative value", "-1"],
    ["a malformed value", "not-a-date"],
    ["a missing value", undefined],
  ])("uses the retry fallback for %s", async (_case, retryAfter) => {
    vi.useFakeTimers()
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(
      429,
      { error: { code: "rate_limited" } },
      retryAfter ? { "Retry-After": retryAfter } : {},
    )))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(999)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(1)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("counts 429 and transport failures against one four-request exchange budget", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    const signals: AbortSignal[] = []
    fetchMock
      .mockResolvedValueOnce(jsonResponse(
        429,
        { error: { code: "rate_limited" } },
        { "Retry-After": "3" },
      ))
      .mockResolvedValueOnce(jsonResponse(
        429,
        { error: { code: "rate_limited" } },
        { "Retry-After": "3" },
      ))
      .mockImplementation((_url: string, init: RequestInit) => {
        const signal = init.signal as AbortSignal
        signals.push(signal)
        return new Promise((_resolve, reject) => {
          signal.addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      })

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(30_000)

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(signals).toHaveLength(2)
    expect(signals.every((signal) => signal.aborted)).toBe(true)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable]"))
    expect(vi.getTimerCount()).toBe(0)
  })

  it("does not schedule a 429 retry beyond the exchange deadline", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(
      429,
      { error: { code: "rate_limited" } },
      { "Retry-After": "31" },
    )))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(30_000)

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[rate_limited] (HTTP 429)"))
    expect(vi.getTimerCount()).toBe(0)
  })

  it("coalesces concurrent reconnect requests into a single call and one broadcast", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })
    fromIframe("reconnect_request", { reason: "ws_closed" })
    fromIframe("reconnect_request", { reason: "token_expired" })
    await vi.waitFor(() => expect(post).toHaveBeenCalled())

    expect(fetchMock).toHaveBeenCalledTimes(2)  // one exchange + one reconnect
    expect(fetchMock.mock.calls[1][0]).toBe(RECONNECT_URL)
    expect(JSON.parse(fetchMock.mock.calls[1][1].body)).toEqual({
      reconnect_token: "rt_first",
      encrypted_context: GRANT,
    })
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_second" })
  })

  it("restarts a frozen stale-session reconnect with the held reconnect token", async () => {
    let frozenReconnectSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token_expires_at: new Date(Date.now() + 30_000).toISOString(),
      })))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        frozenReconnectSignal = init.signal as AbortSignal
        return new Promise(() => undefined)
      })
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_recovered",
        reconnect_token: "rt_recovered",
      })))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    const post = spyOnIframePostMessage()
    fromIframe("ready")
    post.mockClear()

    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(3))

    expect(frozenReconnectSignal?.aborted).toBe(true)
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      EXCHANGE_URL,
      RECONNECT_URL,
      RECONNECT_URL,
    ])
    expect(JSON.parse(fetchMock.mock.calls[2][1].body)).toEqual({
      reconnect_token: "rt_first",
      encrypted_context: GRANT,
    })
    await vi.waitFor(() => expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_recovered" }),
      HOST,
    ))
  })

  it("keeps reconnect requests joined to the replacement exchange across bfcache", async () => {
    let resolveFirstExchange: (value: Response) => void = () => undefined
    let resolveSecondExchange: (value: Response) => void = () => undefined
    let firstExchangeSignal: AbortSignal | undefined
    fetchMock
      .mockImplementationOnce((_url: string, init: RequestInit) => new Promise<Response>((resolve) => {
        resolveFirstExchange = resolve
        firstExchangeSignal = init.signal as AbortSignal
      }))
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveSecondExchange = resolve
      }))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    fromIframe("reconnect_request", { reason: "ws_closed" })
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(firstExchangeSignal?.aborted).toBe(true)

    resolveFirstExchange(jsonResponse(200, exchangeBody({
      session_token: "st_old",
      reconnect_token: "rt_old",
    })))
    await flushAsync()
    fromIframe("reconnect_request", { reason: "token_expired" })

    resolveSecondExchange(jsonResponse(200, exchangeBody({
      session_token: "st_new",
      reconnect_token: "rt_new",
    })))
    await flushAsync()
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("keeps the last healthy session without replaying a frozen reconnect", async () => {
    vi.useFakeTimers()
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    let frozenReconnectSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        frozenReconnectSignal = init.signal as AbortSignal
        return new Promise(() => undefined)
      })

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    const post = spyOnIframePostMessage()
    fromIframe("ready")
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(0)
    firePageRestore()
    await vi.advanceTimersByTimeAsync(7_000)

    expect(frozenReconnectSignal?.aborted).toBe(true)
    expect(fetchMock.mock.calls.map((call) => call[0])).toEqual([
      EXCHANGE_URL,
      RECONNECT_URL,
    ])
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_first" }),
      HOST,
    )
    expect(post).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_degraded" }),
      HOST,
    )
  })

  it("never fires a deferred reconnect once the exchange it waited on goes terminal", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(409, "widget_disabled"))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    // Make the frame ready while the exchange is still pending so both the
    // terminal transition and a deferred reconnect continuation could flush.
    fromIframe("ready")
    // reconnect_request arrives while the exchange is still in flight (its
    // response hasn't been parsed yet), so it coalesces via singleFlight and
    // defers its own network call until the exchange settles.
    fromIframe("reconnect_request", { reason: "ws_closed" })

    // The exchange settles into a terminal error.
    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[widget_disabled]"),
    ))
    await vi.waitFor(() => expect(post).toHaveBeenCalled())
    // Give the deferred reconnect's `wait.then` callback a turn to run: it
    // only wakes up after singleFlight clears state.inflight.exchange, which
    // happens one more microtask turn after handleResult/goTerminal return.
    await Promise.resolve()
    await Promise.resolve()

    // The deferred reconnect must re-check the terminal latch before firing
    // its network call and bail out instead — no second (reconnect) fetch,
    // ever, and every broadcast the frame receives is the terminal one.
    expect(fetchMock).toHaveBeenCalledTimes(1)
    for (const call of post.mock.calls) {
      expect(call[0]).toMatchObject({ type: "session_terminal", code: "widget_disabled" })
    }
    expect(post).toHaveBeenCalledTimes(1)
  })

  it("answers reconnect_request from a latched terminal state without any network call", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(409, "widget_disabled"))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    // Wait for the exchange to actually finish failing (terminal latch set),
    // not just for fetch() to have been called — the response body still
    // needs to be parsed asynchronously before goTerminal() runs.
    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining("[widget_disabled]"),
    ))
    fromIframe("ready")
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })
    fromIframe("reconnect_request", { reason: "ws_closed" })

    expect(fetchMock).toHaveBeenCalledTimes(1)
    expect(post).toHaveBeenCalledTimes(2)
    expect(post.mock.calls[0][0]).toMatchObject({ type: "session_terminal", code: "widget_disabled" })
  })

  it("reconnects before handing the frame a token with under 60s left", async () => {
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token_expires_at: new Date(Date.now() + 30_000).toISOString(),
      })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_fresh" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fromIframe("ready")
    await vi.waitFor(() => expect(post).toHaveBeenCalled())

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe(RECONNECT_URL)
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_fresh" })
  })

  it.each([
    ["reconnect_invalid", 401],
    ["session_expired", 401],
    ["identity_mismatch", 403],
  ])("goes terminal on %s from the reconnect endpoint without retrying", async (code, status) => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(errorResponse(status as number, code as string))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    await flushAsync()
    fromIframe("ready")
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })

    await vi.waitFor(() => expect(errorSpy).toHaveBeenCalledWith(
      expect.stringContaining(`[${code}] (HTTP ${status})`),
    ))
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe(RECONNECT_URL)

    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code }),
      HOST,
    )
  })

  it("uses the shared four-attempt retry policy on the reconnect endpoint", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    fetchMock.mockImplementation(() => Promise.resolve(
      new Response("<html>bad gateway</html>", { status: 502 }),
    ))
    fromIframe("reconnect_request", { reason: "ws_closed" })

    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe(RECONNECT_URL)
    await vi.advanceTimersByTimeAsync(1000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(2000)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    await vi.advanceTimersByTimeAsync(4000)
    expect(fetchMock).toHaveBeenCalledTimes(5)

    await vi.advanceTimersByTimeAsync(8000)
    expect(fetchMock).toHaveBeenCalledTimes(5)  // exchange + four reconnect attempts
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[network_unavailable] (HTTP 502)"))
  })

  it("terminalizes a restored reconnect after its fourth request was already dispatched", async () => {
    vi.useFakeTimers()
    let fourthSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token_expires_at: new Date(Date.now() + 30_000).toISOString(),
      })))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        fourthSignal = init.signal as AbortSignal
        return new Promise<Response>(() => undefined)
      })
      .mockResolvedValue(jsonResponse(502, { error: { code: "upstream_unavailable" } }))

    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    await vi.advanceTimersByTimeAsync(7_000)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)

    firePageRestore()
    await vi.advanceTimersByTimeAsync(0)

    expect(fourthSignal?.aborted).toBe(true)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("continues a restored reconnect with only its remaining attempts", async () => {
    vi.useFakeTimers()
    let secondSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token_expires_at: new Date(Date.now() + 30_000).toISOString(),
      })))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        secondSignal = init.signal as AbortSignal
        return new Promise<Response>((_resolve, reject) => {
          ;(init.signal as AbortSignal).addEventListener(
            "abort",
            () => reject(new DOMException("aborted", "AbortError")),
            { once: true },
          )
        })
      })
      .mockResolvedValue(jsonResponse(502, { error: { code: "upstream_unavailable" } }))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    await vi.advanceTimersByTimeAsync(1_000)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(2)

    firePageRestore()
    await vi.advanceTimersByTimeAsync(7_000)

    expect(secondSignal?.aborted).toBe(true)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)
  })

  it("does not restart reconnect after a bfcache suspension passes its original deadline", async () => {
    vi.useFakeTimers()
    const startedAt = new Date("2026-07-28T00:00:00.000Z")
    vi.setSystemTime(startedAt)
    const reconnectSignals: AbortSignal[] = []
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token_expires_at: new Date(startedAt.getTime() + 30_000).toISOString(),
      })))
      .mockImplementation((_url: string, init: RequestInit) => {
        reconnectSignals.push(init.signal as AbortSignal)
        return new Promise<Response>(() => undefined)
      })

    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(1)

    firePageHide(true)
    vi.setSystemTime(new Date(startedAt.getTime() + 31_000))
    firePageShow(true)
    await vi.advanceTimersByTimeAsync(0)

    expect(reconnectSignals[0]?.aborted).toBe(true)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(1)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("starts a fresh retry lineage after success rotates the reconnect token", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_second",
        reconnect_token: "rt_second",
      })))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({
        session_token: "st_third",
        reconnect_token: "rt_third",
      })))

    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(7_000)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)

    fromIframe("reconnect_request", { reason: "token_expired" })
    await vi.advanceTimersByTimeAsync(0)

    const reconnectCalls = fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)
    expect(reconnectCalls).toHaveLength(5)
    expect(JSON.parse(reconnectCalls[4][1].body)).toMatchObject({ reconnect_token: "rt_second" })
  })

  it("allows two rate-limit retries inside the shared reconnect budget", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)

    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(
      429,
      { error: { code: "rate_limited" } },
      { "Retry-After": "1" },
    )))
    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(3_000)

    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[rate_limited] (HTTP 429)"))
    expect(vi.getTimerCount()).toBe(0)
  })

  it("ignores a late duplicate exchange response after the session moved on", async () => {
    // Fake timers must be installed before runWidget() schedules postJson's
    // SESSION_TIMEOUT_MS abort timer, or advanceTimersByTimeAsync below has
    // nothing to advance and the real timer never fires within the test.
    vi.useFakeTimers()
    let resolveLate: (value: Response) => void = () => undefined
    fetchMock
      // The first attempt hangs until the client gives up: postJson's own
      // AbortController fires at SESSION_TIMEOUT_MS, and this mock (like the
      // "aborts an attempt" test above) must react to that signal the way a
      // real fetch would, or withRetry never sees this attempt settle and
      // never starts the retry that's supposed to win the race.
      .mockImplementationOnce((_url: string, init: RequestInit) => new Promise<Response>((resolve, reject) => {
        resolveLate = resolve
        ;(init.signal as AbortSignal).addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")))
      }))
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_real" })))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    // The first attempt times out and the retry wins.
    await vi.advanceTimersByTimeAsync(6000)
    await vi.waitFor(() => expect(post).toHaveBeenCalled())
    vi.useRealTimers()
    post.mockClear()

    resolveLate(jsonResponse(200, exchangeBody({ session_token: "st_stale", reconnect_token: "rt_stale" })))
    await Promise.resolve()

    expect(post).not.toHaveBeenCalled()
  })

  it("cancels a frozen exchange before starting bfcache recovery", async () => {
    let resolveFirst: (value: Response) => void = () => undefined
    let firstSignal: AbortSignal | undefined
    fetchMock.mockImplementationOnce((_url: string, init: RequestInit) => new Promise<Response>((resolve) => {
      resolveFirst = resolve
      firstSignal = init.signal as AbortSignal
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second" })))
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    expect(firstSignal?.aborted).toBe(true)

    const post = spyOnIframePostMessage()
    await flushAsync()
    fromIframe("ready")
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_second" })
    post.mockClear()

    resolveFirst(jsonResponse(200, exchangeBody({ session_token: "st_stale", reconnect_token: "rt_stale" })))
    await flushAsync()

    expect(post).not.toHaveBeenCalled()
    fromIframe("ready")
    expect(post).toHaveBeenCalledTimes(1)
    expect(post.mock.calls[0][0]).toMatchObject({ session_token: "st_second" })
  })

  it("terminalizes a restored exchange after its fourth request was already dispatched", async () => {
    vi.useFakeTimers()
    let fourthSignal: AbortSignal | undefined
    fetchMock
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        fourthSignal = init.signal as AbortSignal
        return new Promise<Response>(() => undefined)
      })
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    await vi.advanceTimersByTimeAsync(7_000)
    expect(fetchMock).toHaveBeenCalledTimes(4)

    firePageRestore()
    await vi.advanceTimersByTimeAsync(0)

    expect(fourthSignal?.aborted).toBe(true)
    expect(fetchMock).toHaveBeenCalledTimes(4)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "network_unavailable" }),
      HOST,
    )
  })

  it("joins the initial exchange for reconnect requests before a reconnect token exists", async () => {
    let resolveExchange: (value: Response) => void = () => undefined
    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveExchange = resolve
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fromIframe("reconnect_request", { reason: "ws_closed" })
    resolveExchange(jsonResponse(200, exchangeBody()))
    await flushAsync()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("keeps the iframe nonterminal through a parent-owned 25-second recovery and resumes on the fourth response", async () => {
    vi.useFakeTimers()
    let resolveFourth: (value: Response) => void = () => undefined
    const rejectOnAbort = (_url: string, init: RequestInit) => new Promise<Response>((_resolve, reject) => {
      ;(init.signal as AbortSignal).addEventListener("abort", () => {
        reject(new DOMException("aborted", "AbortError"))
      }, { once: true })
    })
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockImplementationOnce(rejectOnAbort)
      .mockImplementationOnce(rejectOnAbort)
      .mockImplementationOnce(rejectOnAbort)
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveFourth = resolve
      }))
    runWidget({ "data-encrypted-context": GRANT })
    const frame = iframeEl()
    if (!frame) throw new Error("iframe not mounted")
    const bridge = bridgeWidgetSessionHook(frame)
    const { result } = renderHook(() => useWidgetSession())

    await vi.advanceTimersByTimeAsync(0)
    expect(result.current.status).toBe("active")

    act(() => result.current.requestReconnect("ws_closed"))
    expect(bridge.parent.postMessage.mock.calls.filter(
      ([message]) => message.type === "reconnect_request",
    )).toHaveLength(1)

    await vi.advanceTimersByTimeAsync(22_000)
    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)
    expect(result.current.status).toBe("degraded")
    expect(result.current.terminalCode).toBeNull()
    expect(bridge.parent.postMessage.mock.calls.filter(
      ([message]) => message.type === "reconnect_request",
    )).toHaveLength(1)

    await vi.advanceTimersByTimeAsync(3_000)
    expect(result.current.status).toBe("degraded")
    expect(result.current.terminalCode).toBeNull()
    fromIframe("ready")
    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(result.current.status).toBe("degraded")
    expect(result.current.terminalCode).toBeNull()

    resolveFourth(jsonResponse(200, exchangeBody({ session_token: "st_fourth" })))
    await vi.advanceTimersByTimeAsync(0)

    expect(result.current.status).toBe("active")
    expect(result.current.session?.token).toBe("st_fourth")
    expect(result.current.terminalCode).toBeNull()
  })

  it("delivers exactly one parent terminal result to the iframe after recovery exhaustion", async () => {
    vi.useFakeTimers()
    fetchMock
      .mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
      .mockResolvedValueOnce(jsonResponse(502, { error: { code: "upstream_unavailable" } }))
    runWidget({ "data-encrypted-context": GRANT })
    const frame = iframeEl()
    if (!frame) throw new Error("iframe not mounted")
    const bridge = bridgeWidgetSessionHook(frame)
    const { result } = renderHook(() => useWidgetSession())

    await vi.advanceTimersByTimeAsync(0)
    act(() => result.current.requestReconnect("ws_closed"))
    await vi.advanceTimersByTimeAsync(31_000)

    expect(fetchMock).toHaveBeenCalledTimes(5)
    expect(fetchMock.mock.calls.filter(([url]) => url === RECONNECT_URL)).toHaveLength(4)
    expect(result.current.status).toBe("terminal")
    expect(result.current.terminalCode).toBe("network_unavailable")
    expect(bridge.postToIframe.mock.calls.filter(
      ([message]) => (message as Record<string, unknown>).type === "session_terminal",
    )).toHaveLength(1)
  })

  it("ignores a superseded exchange failure after the replacement exchange succeeds", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    let resolveFirst: (value: Response) => void = () => undefined
    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveFirst = resolve
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second" })))
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    const post = spyOnIframePostMessage()
    await flushAsync()
    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
    post.mockClear()

    resolveFirst(errorResponse(409, "agent_not_available"))
    await flushAsync()

    expect(errorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("[agent_not_available]"),
    )
    expect(post).not.toHaveBeenCalled()
    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
  })

  it("ignores a superseded exchange failure before the replacement exchange succeeds", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    let resolveFirst: (value: Response) => void = () => undefined
    let resolveSecond: (value: Response) => void = () => undefined
    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveFirst = resolve
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveSecond = resolve
    }))
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    const post = spyOnIframePostMessage()
    fromIframe("ready")
    resolveFirst(errorResponse(409, "agent_not_available"))
    await flushAsync()

    expect(errorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("[agent_not_available]"),
    )
    expect(post).not.toHaveBeenCalled()

    resolveSecond(jsonResponse(200, exchangeBody({ session_token: "st_second" })))
    await flushAsync()
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
  })

  it("publishes the replacement exchange without parking a reconnect operation", async () => {
    let resolveFirst: (value: Response) => void = () => undefined
    let resolveSecond: (value: Response) => void = () => undefined
    fetchMock
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveFirst = resolve
      }))
      .mockImplementationOnce(() => new Promise<Response>((resolve) => {
        resolveSecond = resolve
      }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    const post = spyOnIframePostMessage()
    fromIframe("ready")

    resolveFirst(errorResponse(409, "agent_not_available"))
    await flushAsync()
    fromIframe("reconnect_request", { reason: "ws_closed" })
    expect(fetchMock).toHaveBeenCalledTimes(2)

    resolveSecond(jsonResponse(200, exchangeBody({
      session_token: "st_second",
      reconnect_token: "rt_second",
    })))
    await flushAsync()

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
  })

  it("does not retry a network failure from a superseded exchange", async () => {
    vi.useFakeTimers()
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    let rejectFirst: (reason: unknown) => void = () => undefined
    fetchMock.mockImplementationOnce(() => new Promise<Response>((_resolve, reject) => {
      rejectFirst = reject
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody({ session_token: "st_second" })))
    firePageRestore()
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(2)

    const post = spyOnIframePostMessage()
    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
    post.mockClear()

    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"))
    rejectFirst(new TypeError("Failed to fetch"))
    await vi.advanceTimersByTimeAsync(7000)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(errorSpy).not.toHaveBeenCalledWith(
      expect.stringContaining("[network_unavailable]"),
    )
    expect(post).not.toHaveBeenCalled()
    fromIframe("ready")
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second" }),
      HOST,
    )
  })

  it("ignores a canceled exchange even when it resolves before its replacement", async () => {
    let resolveFirst: (value: Response) => void = () => undefined
    let resolveSecond: (value: Response) => void = () => undefined
    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveFirst = resolve
    }))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => {
      resolveSecond = resolve
    }))
    firePageRestore()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))

    const post = spyOnIframePostMessage()
    resolveFirst(jsonResponse(200, exchangeBody({ session_token: "st_first_winner" })))
    await flushAsync()
    fromIframe("ready")
    expect(post).not.toHaveBeenCalled()

    resolveSecond(jsonResponse(200, exchangeBody({ session_token: "st_second_late" })))
    await flushAsync()

    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update", session_token: "st_second_late" }),
      HOST,
    )
  })

  it("fails closed when reconnect still returns an already-stale token", async () => {
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(200, exchangeBody({
      session_token_expires_at: new Date(Date.now() + 10_000).toISOString(),
    }))))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    fromIframe("ready")
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
    await flushAsync()

    expect(errorSpy).toHaveBeenCalledWith(expect.stringContaining("[unexpected_error] (HTTP 200)"))
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_terminal", code: "unexpected_error" }),
      HOST,
    )
    expect(post).not.toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_update" }),
      HOST,
    )
  })

  it("re-runs the load flow when a bfcache restore finds no session", async () => {
    vi.useFakeTimers()
    // The exchange never settles: the page was frozen mid-flight.
    fetchMock.mockImplementationOnce(() => new Promise(() => undefined))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    firePageRestore()
    await vi.advanceTimersByTimeAsync(0)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(fetchMock.mock.calls[1][0]).toBe(EXCHANGE_URL)
  })

  it("processes one persisted pagehide only once across repeated pageshow events", async () => {
    vi.useFakeTimers()
    let recoverySignal: AbortSignal | undefined
    fetchMock
      .mockImplementationOnce(() => new Promise(() => undefined))
      .mockImplementationOnce((_url: string, init: RequestInit) => {
        recoverySignal = init.signal as AbortSignal
        return new Promise(() => undefined)
      })
    runWidget({ "data-encrypted-context": GRANT })
    await vi.advanceTimersByTimeAsync(0)

    firePageRestore()
    await vi.advanceTimersByTimeAsync(0)
    expect(fetchMock).toHaveBeenCalledTimes(2)

    firePageShow(true)
    await vi.advanceTimersByTimeAsync(0)

    expect(fetchMock).toHaveBeenCalledTimes(2)
    expect(recoverySignal?.aborted).toBe(false)
  })

  it("does nothing on a bfcache restore with a healthy session", async () => {
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // vi.waitFor's condition above (fetchMock call count) goes true the
    // instant fetch() is invoked, before the mocked Response's .json() body
    // read (and the applySession() it feeds) has actually settled — see the
    // flushAsync() comment near the top of this file for the same race. A
    // macrotask flush here lets state.session actually be populated before
    // firePageShow, or this test would spuriously re-trigger the load flow.
    await flushAsync()

    firePageRestore()
    await Promise.resolve()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("does nothing on a bfcache restore after a terminal outcome", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockResolvedValueOnce(errorResponse(403, "agent_not_granted"))
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
    // Same race as above: let goTerminal() actually latch state.terminalCode
    // before firePageShow, or this test would spuriously re-trigger the load
    // flow while the terminal outcome is still mid-flight.
    await flushAsync()

    firePageRestore()
    await Promise.resolve()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("keeps retryable rate limits inside the active parent phase", async () => {
    vi.useFakeTimers()
    vi.spyOn(console, "error").mockImplementation(() => undefined)
    fetchMock.mockImplementation(() => Promise.resolve(jsonResponse(
      429,
      { error: { code: "rate_limited" } },
      { "Retry-After": "1" },
    )))
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    fromIframe("ready")
    await vi.advanceTimersByTimeAsync(3_000)

    expect(fetchMock).toHaveBeenCalledTimes(3)
    expect(post).toHaveBeenCalledWith(
      expect.objectContaining({ type: "session_degraded", code: "rate_limited" }),
      HOST,
    )
    post.mockClear()

    fromIframe("reconnect_request", { reason: "ws_closed" })
    await vi.advanceTimersByTimeAsync(1_000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
    await vi.advanceTimersByTimeAsync(30_000)
    expect(fetchMock).toHaveBeenCalledTimes(3)
  })

  it("does nothing on a normal (non-persisted) pageshow", async () => {
    fetchMock.mockImplementationOnce(() => new Promise(() => undefined))
    runWidget({ "data-encrypted-context": GRANT })

    firePageShow(false)
    await Promise.resolve()

    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("tears down listeners and pending requests when the session iframe leaves the DOM", async () => {
    let requestSignal: AbortSignal | undefined
    const removeListenerSpy = vi.spyOn(window, "removeEventListener")
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      requestSignal = init.signal as AbortSignal
      return new Promise(() => undefined)
    })
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    document.querySelector(".xagent-widget-container")?.remove()
    await vi.waitFor(() => expect(requestSignal?.aborted).toBe(true))
    expect(removeListenerSpy).toHaveBeenCalledWith("message", expect.any(Function))
    expect(removeListenerSpy).toHaveBeenCalledWith("pageshow", expect.any(Function))
    expect(removeListenerSpy).toHaveBeenCalledWith("pagehide", expect.any(Function))

    firePageRestore()
    await flushAsync()
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("tears down when the iframe is removed from its still-mounted widget subtree", async () => {
    let requestSignal: AbortSignal | undefined
    fetchMock.mockImplementation((_url: string, init: RequestInit) => {
      requestSignal = init.signal as AbortSignal
      return new Promise(() => undefined)
    })
    runWidget({ "data-encrypted-context": GRANT })
    await vi.waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))

    iframeEl()?.remove()
    await vi.waitFor(() => expect(requestSignal?.aborted).toBe(true))
  })

  it("cancels the delivery-stability timer when the session iframe is detached", async () => {
    vi.useFakeTimers()
    fetchMock.mockResolvedValueOnce(jsonResponse(200, exchangeBody()))
    const clearTimeoutSpy = vi.spyOn(window, "clearTimeout")
    runWidget({ "data-encrypted-context": GRANT })
    const post = spyOnIframePostMessage()
    await vi.advanceTimersByTimeAsync(0)
    fromIframe("ready")
    const deliveryId = post.mock.calls.at(-1)?.[0].session_delivery_id
    fromIframe("session_connection_open", { session_delivery_id: deliveryId })
    const clearsBeforeDetach = clearTimeoutSpy.mock.calls.length

    iframeEl()?.remove()
    await Promise.resolve()
    await Promise.resolve()

    expect(clearTimeoutSpy.mock.calls.length).toBeGreaterThan(clearsBeforeDetach)
  })
})
