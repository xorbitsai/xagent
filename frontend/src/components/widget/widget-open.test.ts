import { afterEach, beforeEach, describe, expect, it, vi, type MockInstance } from "vitest"
import {
  MAX_INITIAL_MESSAGE_LENGTH,
  listenForOpenMessages,
  listenForWidgetExpandState,
} from "@/lib/widget-parent-message"

import { evaluateWidgetScript, widgetScript } from "@/test-utils/widget-script"

const HOST = "https://chat.example"
const GRANT = "eyJhbGciOiJIUzI1NiJ9.open-grant.sig"

const W_NO_INSTANCE = "Xagent Widget: XagentWidget.open() was called, but no chat widget is active on this page."
const W_BAD_OPTIONS = "Xagent Widget: XagentWidget.open() expects an options object; opening the chat without sending a message."
const W_BAD_MESSAGE = "Xagent Widget: XagentWidget.open() message must be a non-empty string of at most 2000 characters; opening the chat without sending it."
const unknownOptionsWarning = (keys: string) => (
  `Xagent Widget: XagentWidget.open() ignored unknown option(s): ${keys}.`
)
const W_BAD_EXPANDED = "Xagent Widget: XagentWidget.open() expanded must be true or false; ignoring it."
const DUPLICATE_GRANT_WARNING = "Xagent Widget: this grant is single-use; mint a fresh grant before remounting the widget [duplicate_init]."

type Instance = {
  container: HTMLElement
  panel: HTMLElement
  fab: HTMLButtonElement
  iframe: HTMLIFrameElement
  post: MockInstance<Window["postMessage"]>
}

type OpenTarget = { open: (options?: unknown) => void; isLive: () => boolean }

// Forwards exactly the arguments it was given, so "no options" stays distinct
// from an explicit undefined.
function open(...args: unknown[]): unknown {
  const api = (window as unknown as { XagentWidget: { open: (...args: unknown[]) => unknown } }).XagentWidget
  return api.open(...args)
}

function openTargets(): OpenTarget[] {
  return (window as unknown as { __xagentWidgetOpenTargets: OpenTarget[] }).__xagentWidgetOpenTargets
}

function containers() {
  return document.querySelectorAll<HTMLElement>(".xagent-widget-container")
}

// Runs one copy of widget.js; returns the instance it rendered, or null when
// this copy failed to initialize and rendered nothing.
function mount(attributes: Record<string, string>): Instance | null {
  const before = containers().length
  const script = document.createElement("script")
  script.src = `${HOST}/widget.js`
  for (const [name, value] of Object.entries(attributes)) {
    script.setAttribute(name, value)
  }
  document.body.appendChild(script)
  Object.defineProperty(document, "currentScript", { configurable: true, value: script })
  evaluateWidgetScript()
  const all = containers()
  if (all.length === before) return null
  const container = all[all.length - 1]
  const iframe = container.querySelector<HTMLIFrameElement>(".xagent-widget-iframe")!
  return {
    container,
    panel: container.querySelector<HTMLElement>(".xagent-widget-panel")!,
    fab: container.querySelector<HTMLButtonElement>(".xagent-widget-fab")!,
    iframe,
    post: vi.spyOn(iframe.contentWindow!, "postMessage"),
  }
}

function mountGuest(key = "widget-secret", extra: Record<string, string> = {}): Instance {
  const instance = mount({ "data-widget-key": key, ...extra })
  if (!instance) throw new Error("guest widget did not mount")
  return instance
}

function fromIframe(
  instance: Instance,
  type: string,
  extra: Record<string, unknown> = {},
  { origin = HOST, source = instance.iframe.contentWindow as Window | null }: {
    origin?: string
    source?: Window | null
  } = {},
) {
  window.dispatchEvent(new MessageEvent("message", {
    data: { xagent: true, v: 1, type, ...extra },
    origin,
    source,
  }))
}

function deliveriesOfType(instance: Instance, type: string) {
  return instance.post.mock.calls
    .filter(([data]) => (data as { type?: string }).type === type)
    .map(([data, target]) => {
      expect(target).toBe(HOST)
      return data
    })
}

const openDeliveries = (instance: Instance) => deliveriesOfType(instance, "open_message")
const expandStates = (instance: Instance) => deliveriesOfType(instance, "widget_expand_state")
const expandState = (expanded: boolean) => ({ xagent: true, v: 1, type: "widget_expand_state", expanded })
const delivery = (id: string, value: string) => ({ xagent: true, v: 1, type: "open_message", id, value })

function firePageTransition(type: "pagehide" | "pageshow") {
  const event = new Event(type) as PageTransitionEvent & { persisted: boolean }
  Object.defineProperty(event, "persisted", { value: true })
  window.dispatchEvent(event)
}

async function teardown(instance: Instance) {
  instance.container.remove()
  // panelRemovalObserver runs the teardown a microtask after the removal.
  await Promise.resolve()
}

describe("XagentWidget.open", () => {
  let currentScriptDescriptor: PropertyDescriptor | undefined
  let warn: MockInstance<typeof console.warn>
  const fetchMock = vi.fn()

  beforeEach(() => {
    currentScriptDescriptor = Object.getOwnPropertyDescriptor(document, "currentScript")
    document.head.innerHTML = ""
    document.body.innerHTML = ""
    localStorage.clear()
    localStorage.setItem("xagent_guest_id", "guest-fixed")
    fetchMock.mockReset()
    // Neither the guest ticket nor the Session exchange settles: open() never
    // depends on the iframe having loaded.
    fetchMock.mockReturnValue(new Promise(() => {}))
    vi.stubGlobal("fetch", fetchMock)
    warn = vi.spyOn(console, "warn").mockImplementation(() => {})
    vi.spyOn(console, "error").mockImplementation(() => {})
  })

  afterEach(async () => {
    for (const container of containers()) {
      container.remove()
    }
    // Let every teardown observer run before the jsdom globals go away.
    await Promise.resolve()
    await Promise.resolve()
    for (const name of ["XagentWidget", "__xagentWidgetOpenTargets", "__xagentWidgetGrants"]) {
      Reflect.deleteProperty(window as unknown as Record<string, unknown>, name)
    }
    if (currentScriptDescriptor) {
      Object.defineProperty(document, "currentScript", currentScriptDescriptor)
    } else {
      Reflect.deleteProperty(document, "currentScript")
    }
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  describe("defines XagentWidget even when this instance cannot initialize", () => {
    it.each<{ name: string; attributes: Record<string, string> }>([
      { name: "a guest embed without data-widget-key", attributes: {} },
      { name: "a Session embed with an empty grant", attributes: { "data-encrypted-context": "  " } },
    ])("$name", ({ attributes }) => {
      expect(mount(attributes)).toBeNull()
      expect(typeof (window as unknown as { XagentWidget: { open: unknown } }).XagentWidget.open).toBe("function")

      open({ message: "hi" })

      expect(warn.mock.calls).toEqual([[W_NO_INSTANCE]])
      expect(document.querySelector(".xagent-widget-panel")).toBeNull()
    })

    it("a Session grant reused after its first instance was removed", async () => {
      const first = mount({ "data-encrypted-context": GRANT })!
      await teardown(first)

      expect(mount({ "data-encrypted-context": GRANT })).toBeNull()
      expect(warn.mock.calls).toEqual([[DUPLICATE_GRANT_WARNING]])
      warn.mockClear()

      open({ message: "hi" })

      expect(warn.mock.calls).toEqual([[W_NO_INSTANCE]])
      expect(document.querySelector(".xagent-widget-panel.open")).toBeNull()
    })

    it("a Session grant reused while its first instance is still live", () => {
      const first = mount({ "data-encrypted-context": GRANT })!
      fromIframe(first, "open_message_request")

      expect(mount({ "data-encrypted-context": GRANT })).toBeNull()
      warn.mockClear()

      open({ message: "hi" })

      expect(openDeliveries(first)).toEqual([delivery("1", "hi")])
      expect(warn).not.toHaveBeenCalled()
    })
  })

  describe("validates open() options like data-initial-message", () => {
    const MAX = MAX_INITIAL_MESSAGE_LENGTH
    it.each([
      { name: "no options", args: [], sent: null, warnings: [] },
      { name: "an empty object", args: [{}], sent: null, warnings: [] },
      { name: "an undefined message", args: [{ message: undefined }], sent: null, warnings: [] },
      { name: "a padded message", args: [{ message: "  hi  " }], sent: "hi", warnings: [] },
      { name: "a message at the length cap", args: [{ message: "x".repeat(MAX) }], sent: "x".repeat(MAX), warnings: [] },
      { name: "a message over the length cap", args: [{ message: "x".repeat(MAX + 1) }], sent: null, warnings: [[W_BAD_MESSAGE]] },
      { name: "a blank message", args: [{ message: "   " }], sent: null, warnings: [[W_BAD_MESSAGE]] },
      { name: "an empty message", args: [{ message: "" }], sent: null, warnings: [[W_BAD_MESSAGE]] },
      { name: "a numeric message", args: [{ message: 42 }], sent: null, warnings: [[W_BAD_MESSAGE]] },
      { name: "a null message", args: [{ message: null }], sent: null, warnings: [[W_BAD_MESSAGE]] },
      { name: "null options", args: [null], sent: null, warnings: [[W_BAD_OPTIONS]] },
      { name: "string options", args: ["hi"], sent: null, warnings: [[W_BAD_OPTIONS]] },
      { name: "numeric options", args: [42], sent: null, warnings: [[W_BAD_OPTIONS]] },
      { name: "array options", args: [["hi"]], sent: null, warnings: [[W_BAD_OPTIONS]] },
      { name: "function options", args: [() => {}], sent: null, warnings: [[W_BAD_OPTIONS]] },
      {
        name: "an unknown option beside a message",
        args: [{ message: "hi", conversation: "current" }],
        sent: "hi",
        warnings: [[unknownOptionsWarning("conversation")]],
      },
      {
        name: "an expanded option beside an unknown one",
        args: [{ expanded: true, conversation: "current" }],
        sent: null,
        warnings: [[unknownOptionsWarning("conversation")]],
      },
      {
        name: "a misspelled message option",
        args: [{ mesage: "hi" }], // codespell:ignore mesage
        sent: null,
        warnings: [[unknownOptionsWarning("mesage")]], // codespell:ignore mesage
      },
    ])("$name", ({ args, sent, warnings }) => {
      const instance = mountGuest()
      fromIframe(instance, "open_message_request")

      const result = open(...args)

      expect(result).toBeUndefined()
      expect(instance.panel).toHaveClass("open")
      expect(openDeliveries(instance)).toEqual(sent === null ? [] : [delivery("1", sent)])
      expect(warn.mock.calls).toEqual(warnings)
    })

    it.each([
      {
        name: "a throwing message getter",
        options: { get message() { throw new Error("getter") } },
        error: "getter",
      },
      {
        name: "a Proxy whose keys cannot be listed",
        options: new Proxy({}, { ownKeys() { throw new Error("keys") } }),
        error: "keys",
      },
    ])("passes $name through before opening anything", ({ options, error }) => {
      const instance = mountGuest()
      fromIframe(instance, "open_message_request")

      expect(() => open(options)).toThrow(error)

      expect(instance.panel).not.toHaveClass("open")
      expect(openDeliveries(instance)).toEqual([])
      expect(warn).not.toHaveBeenCalled()
    })

    it("knows exactly the options these rows cover", () => {
      const declared = widgetScript.match(/var OPEN_OPTION_KEYS = (\[[^\]]*\]);/)
      expect(declared).not.toBeNull()
      expect(JSON.parse(declared![1].replace(/'/g, '"'))).toEqual(["message", "expanded"])
    })
  })

  it.each([
    { name: "closed", openFirst: false },
    { name: "already open", openFirst: true },
  ])("keeps an open panel open (panel $name)", ({ openFirst }) => {
    const instance = mountGuest()
    if (openFirst) instance.fab.click()

    open({ message: "hi" })

    expect(instance.panel).toHaveClass("open")
    expect(instance.fab).toHaveAttribute("aria-label", "Minimize chat")
  })

  describe("open(message) voids data-initial-message", () => {
    const initialDeliveries = (instance: Instance) => deliveriesOfType(instance, "initial_message")

    it("before the panel was ever opened", () => {
      const instance = mountGuest("widget-secret", { "data-initial-message": "hello" })
      fromIframe(instance, "initial_message_request")

      open({ message: "hi" })
      fromIframe(instance, "initial_message_request")

      expect(initialDeliveries(instance)).toEqual([])
    })

    it("after it was delivered but before its ack", () => {
      const instance = mountGuest("widget-secret", { "data-initial-message": "hello" })
      instance.fab.click()
      fromIframe(instance, "initial_message_request")
      expect(initialDeliveries(instance)).toHaveLength(1)

      open({ message: "hi" })
      fromIframe(instance, "initial_message_request")

      expect(initialDeliveries(instance)).toHaveLength(1)
    })

    it("but open() without a message leaves it in place", () => {
      const instance = mountGuest("widget-secret", { "data-initial-message": "hello" })

      open()
      fromIframe(instance, "initial_message_request")

      expect(initialDeliveries(instance)).toEqual([
        { xagent: true, v: 1, type: "initial_message", value: "hello" },
      ])
    })
  })

  it("keeps one pending open message until its own id is acknowledged", () => {
    const instance = mountGuest()

    open({ message: "A" })
    open({ message: "B" })
    expect(openDeliveries(instance)).toEqual([])

    fromIframe(instance, "open_message_request")
    expect(openDeliveries(instance)).toEqual([delivery("2", "B")])

    fromIframe(instance, "open_message_ack", { id: "1" })
    fromIframe(instance, "open_message_request")
    expect(openDeliveries(instance)).toEqual([delivery("2", "B"), delivery("2", "B")])

    fromIframe(instance, "open_message_ack", { id: "2" })
    fromIframe(instance, "open_message_request")
    expect(openDeliveries(instance)).toEqual([delivery("2", "B"), delivery("2", "B")])
  })

  it("delivers at once after a request and again to a reloaded iframe", () => {
    const instance = mountGuest()
    fromIframe(instance, "open_message_request")

    open({ message: "A" })
    expect(openDeliveries(instance)).toEqual([delivery("1", "A")])

    // One request keeps every later call deliverable at once.
    fromIframe(instance, "open_message_ack", { id: "1" })
    open({ message: "B" })
    expect(openDeliveries(instance)).toEqual([delivery("1", "A"), delivery("2", "B")])

    // A reloaded iframe asks again without having acknowledged.
    fromIframe(instance, "open_message_request")
    expect(openDeliveries(instance)).toEqual([delivery("1", "A"), delivery("2", "B"), delivery("2", "B")])

    fromIframe(instance, "open_message_ack", { id: "2" })
    fromIframe(instance, "open_message_request")
    expect(openDeliveries(instance)).toHaveLength(3)
  })

  describe("routes open() to the newest live instance", () => {
    it("and falls back as instances are torn down", async () => {
      const older = mountGuest("key-one")
      const newer = mountGuest("key-two")
      fromIframe(older, "open_message_request")
      fromIframe(newer, "open_message_request")

      open({ message: "m1" })
      expect(openDeliveries(newer)).toEqual([delivery("1", "m1")])
      expect(openDeliveries(older)).toEqual([])

      await teardown(newer)
      // The torn-down instance's slot is gone with it: asking again gets nothing.
      fromIframe(newer, "open_message_request")
      expect(openDeliveries(newer)).toHaveLength(1)

      open({ message: "m2" })
      expect(openDeliveries(older)).toEqual([delivery("1", "m2")])

      await teardown(older)
      open({ message: "m3" })
      expect(warn.mock.calls).toEqual([[W_NO_INSTANCE]])
    })

    it("as soon as the newer panel leaves the page", () => {
      const older = mountGuest("key-one")
      const newer = mountGuest("key-two")
      fromIframe(older, "open_message_request")
      fromIframe(newer, "open_message_request")

      // Before the removal observer has run.
      newer.container.remove()
      open({ message: "m1" })

      expect(openDeliveries(older)).toEqual([delivery("1", "m1")])
      expect(openDeliveries(newer)).toEqual([])
    })

    it("and treats a target list the host page overwrote as empty", async () => {
      const instance = mountGuest()
      fromIframe(instance, "open_message_request")
      ;(window as unknown as Record<string, unknown>).__xagentWidgetOpenTargets = "overwritten"

      open({ message: "hi" })
      await teardown(instance)

      expect(warn.mock.calls).toEqual([[W_NO_INSTANCE]])
      expect(openDeliveries(instance)).toEqual([])
      expect(openTargets()).toEqual([])
    })

    it("and an instance's own entry stays inert once torn down", async () => {
      const instance = mountGuest()
      fromIframe(instance, "open_message_request")
      const entry = openTargets()[openTargets().length - 1]

      await teardown(instance)
      entry.open({ message: "late" })

      expect(openTargets()).toEqual([])
      expect(instance.panel).not.toHaveClass("open")
      expect(openDeliveries(instance)).toEqual([])
    })
  })

  it.each([
    { name: "another origin", origin: "https://evil.example", fromWindow: false },
    { name: "another window", origin: HOST, fromWindow: true },
  ])("ignores acknowledgements from $name", ({ origin, fromWindow }) => {
    const instance = mountGuest()
    fromIframe(instance, "open_message_request")
    open({ message: "A" })

    fromIframe(instance, "open_message_ack", { id: "1" }, {
      origin,
      source: fromWindow ? window : instance.iframe.contentWindow,
    })
    fromIframe(instance, "open_message_request")

    expect(openDeliveries(instance)).toEqual([delivery("1", "A"), delivery("1", "A")])
  })

  describe("round-trips an open message and its acknowledgement between the real widget.js and the iframe listener", () => {
    it.each([
      { name: "a short message", message: "hi" },
      { name: "a message at the shared length cap", message: "x".repeat(MAX_INITIAL_MESSAGE_LENGTH) },
    ])("$name", ({ message }) => {
      const instance = mountGuest()
      const parentPost = vi.fn()
      vi.stubGlobal("parent", { postMessage: parentPost })
      const onMessage = vi.fn()
      const stop = listenForOpenMessages(onMessage)
      // Every message below is the real output of the other side, passed on
      // unchanged: a renamed type or field on either side breaks the trip.
      const toWidget = (data: unknown) => window.dispatchEvent(new MessageEvent("message", {
        data,
        origin: HOST,
        source: instance.iframe.contentWindow,
      }))
      const [[request]] = parentPost.mock.calls

      toWidget(request)
      open({ message })
      expect(instance.post).toHaveBeenCalledTimes(1)
      const [[delivered]] = instance.post.mock.calls
      window.dispatchEvent(new MessageEvent("message", {
        data: delivered,
        source: window.parent as unknown as MessageEventSource,
      }))

      expect(onMessage.mock.calls).toEqual([[message, (delivered as { id: string }).id]])
      expect(parentPost).toHaveBeenCalledTimes(2)
      const [, [acknowledgement]] = parentPost.mock.calls
      toWidget(acknowledgement)
      toWidget(request)

      expect(instance.post).toHaveBeenCalledTimes(1)
      stop()
    })
  })

  describe("sizes the panel from open({ expanded })", () => {
    let originalInnerWidth: number
    beforeEach(() => {
      originalInnerWidth = window.innerWidth
    })
    afterEach(() => {
      Object.defineProperty(window, "innerWidth", { configurable: true, value: originalInnerWidth })
    })

    function setViewport(viewport: "desktop" | "mobile") {
      Object.defineProperty(window, "innerWidth", { configurable: true, value: viewport === "mobile" ? 400 : 1024 })
    }

    type Start = "closed, normal" | "closed, expanded" | "open, normal" | "open, expanded"
    // Reaches the starting state through the visitor's own controls only.
    function startFrom(instance: Instance, start: Start) {
      if (start.endsWith("expanded")) {
        instance.fab.click()
        fromIframe(instance, "widget_expand")
        expect(instance.panel).toHaveClass("expanded")
        if (start.startsWith("closed")) instance.fab.click()
      } else if (start.startsWith("open")) {
        instance.fab.click()
      }
      expect(instance.panel.classList.contains("open")).toBe(start.startsWith("open"))
      instance.post.mockClear()
      warn.mockClear()
    }

    const desktopCells = ([true, false] as const).flatMap((expanded) => (
      (["closed, normal", "closed, expanded", "open, normal", "open, expanded"] as const)
        .map((start) => ({ expanded, start, viewport: "desktop" as const }))
    ))
    const mobileCells = ([true, false] as const).flatMap((expanded) => (
      (["closed, normal", "open, normal"] as const)
        .map((start) => ({ expanded, start, viewport: "mobile" as const }))
    ))

    it.each([...desktopCells, ...mobileCells])(
      "expanded: $expanded from $start on $viewport",
      ({ expanded, start, viewport }) => {
        localStorage.setItem("xagent_widget_width", "500")
        setViewport(viewport)
        const instance = mountGuest()
        startFrom(instance, start)

        open({ expanded })

        const wantExpanded = expanded && viewport === "desktop"
        expect(instance.panel).toHaveClass("open")
        expect(instance.panel.classList.contains("expanded")).toBe(wantExpanded)
        const states = expandStates(instance)
        expect(states).toHaveLength(1)
        expect(states[states.length - 1]).toEqual(expandState(instance.panel.classList.contains("expanded")))
        if (viewport === "desktop") {
          // Collapsing restores the stored width; expanding leaves width to the CSS rule.
          expect(instance.panel.style.width).toBe(wantExpanded ? "" : "500px")
        }
        expect(warn).not.toHaveBeenCalled()
      },
    )

    it.each([
      { name: "the string \"true\"", value: "true" },
      { name: "1", value: 1 },
      { name: "0", value: 0 },
      { name: "null", value: null },
      { name: "an object", value: {} },
      { name: "an array", value: [] },
    ])("ignores $name with one fixed warning, still opening and sending the message", ({ value }) => {
      const instance = mountGuest()
      fromIframe(instance, "open_message_request")

      open({ message: "hi", expanded: value })

      expect(warn.mock.calls).toEqual([[W_BAD_EXPANDED]])
      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).not.toHaveClass("expanded")
      expect(openDeliveries(instance)).toEqual([delivery("1", "hi")])
      expect(expandStates(instance)).toEqual([])
    })

    it("treats an undefined expanded as not given, without a warning", () => {
      const instance = mountGuest()

      open({ expanded: undefined })

      expect(warn).not.toHaveBeenCalled()
      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).not.toHaveClass("expanded")
      expect(expandStates(instance)).toEqual([])
    })

    it.each([
      { name: "the string \"true\"", value: "true", warnings: [[W_BAD_EXPANDED]] },
      { name: "1", value: 1, warnings: [[W_BAD_EXPANDED]] },
      { name: "null", value: null, warnings: [[W_BAD_EXPANDED]] },
      { name: "undefined", value: undefined, warnings: [] },
    ])("keeps an open, expanded panel as it is for $name", ({ value, warnings }) => {
      const instance = mountGuest()
      startFrom(instance, "open, expanded")

      open({ expanded: value })

      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).toHaveClass("expanded")
      expect(expandStates(instance)).toEqual([])
      expect(warn.mock.calls).toEqual(warnings)
    })

    describe.each([
      { start: "open, normal" as const },
      { start: "open, expanded" as const },
    ])("leaves the size alone without expanded ($start)", ({ start }) => {
      it.each([
        { name: "open()", args: [] },
        { name: "open({ message })", args: [{ message: "hi" }] },
      ])("$name", ({ args }) => {
        const instance = mountGuest()
        startFrom(instance, start)

        open(...args)

        expect(instance.panel).toHaveClass("open")
        expect(instance.panel.classList.contains("expanded")).toBe(start === "open, expanded")
        expect(expandStates(instance)).toEqual([])
      })
    })

    it("passes a throwing expanded getter through before anything changes", () => {
      const instance = mountGuest("widget-secret", { "data-expanded": "true" })
      const options = { conversation: "current" }
      Object.defineProperty(options, "expanded", {
        enumerable: true,
        get() { throw new Error("expanded getter") },
      })

      expect(() => open(options)).toThrow("expanded getter")

      expect(warn.mock.calls).toEqual([[unknownOptionsWarning("conversation")]])
      expect(instance.panel).not.toHaveClass("open")
      expect(instance.post).not.toHaveBeenCalled()

      // The first-open size rule is still unspent.
      instance.fab.click()
      expect(instance.panel).toHaveClass("expanded")
    })

    it("lets an explicit expanded: false on the first open spend data-expanded", () => {
      const instance = mountGuest("widget-secret", { "data-expanded": "true" })

      open({ expanded: false })
      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).not.toHaveClass("expanded")
      expect(expandStates(instance)).toEqual([expandState(false)])

      instance.fab.click()
      instance.fab.click()
      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).not.toHaveClass("expanded")
      expect(expandStates(instance)).toEqual([expandState(false)])
    })

    it.each([
      { name: "the FAB", args: null, message: null, warnings: [] },
      { name: "open()", args: [], message: null, warnings: [] },
      { name: "open({ message })", args: [{ message: "hi" }], message: "hi", warnings: [] },
      { name: "open({ expanded: \"true\" })", args: [{ expanded: "true" }], message: null, warnings: [[W_BAD_EXPANDED]] },
    ])("applies data-expanded to a first open through $name", ({ args, message, warnings }) => {
      const instance = mountGuest("widget-secret", { "data-expanded": "true" })
      fromIframe(instance, "open_message_request")

      if (args === null) instance.fab.click()
      else open(...args)

      expect(instance.panel).toHaveClass("open")
      expect(instance.panel).toHaveClass("expanded")
      expect(expandStates(instance)).toEqual([expandState(true)])
      expect(openDeliveries(instance)).toEqual(message === null ? [] : [delivery("1", message)])
      expect(warn.mock.calls).toEqual(warnings)
    })

    it("tells the real iframe listener about the size the host page set", () => {
      const instance = mountGuest()
      vi.stubGlobal("parent", { postMessage: vi.fn() })
      const onState = vi.fn()
      const stop = listenForWidgetExpandState(onState)

      open({ expanded: true })
      // The real output of widget.js, passed on unchanged.
      expect(instance.post).toHaveBeenCalledTimes(1)
      const [[delivered]] = instance.post.mock.calls
      window.dispatchEvent(new MessageEvent("message", {
        data: delivered,
        source: window.parent as unknown as MessageEventSource,
      }))

      expect(onState.mock.calls).toEqual([[true]])
      stop()
    })
  })

  describe("keeps a pending open message across a persisted pagehide and pageshow", () => {
    it.each<{ mode: string; attributes: Record<string, string>; requestedBefore: boolean }>([
      { mode: "guest", attributes: { "data-widget-key": "widget-secret" }, requestedBefore: false },
      { mode: "guest", attributes: { "data-widget-key": "widget-secret" }, requestedBefore: true },
      { mode: "Session", attributes: { "data-encrypted-context": GRANT }, requestedBefore: false },
      { mode: "Session", attributes: { "data-encrypted-context": GRANT }, requestedBefore: true },
    ])("$mode mode, iframe asked before the freeze: $requestedBefore", ({ attributes, requestedBefore }) => {
      const instance = mount(attributes)!
      if (requestedBefore) fromIframe(instance, "open_message_request")
      open({ message: "A" })
      expect(openDeliveries(instance)).toEqual(requestedBefore ? [delivery("1", "A")] : [])

      firePageTransition("pagehide")
      firePageTransition("pageshow")
      // The restore itself posts nothing again.
      expect(openDeliveries(instance)).toEqual(requestedBefore ? [delivery("1", "A")] : [])

      fromIframe(instance, "open_message_request")
      const afterRequest = requestedBefore ? 2 : 1
      expect(openDeliveries(instance)).toEqual(Array(afterRequest).fill(delivery("1", "A")))

      fromIframe(instance, "open_message_ack", { id: "1" })
      fromIframe(instance, "open_message_request")
      expect(openDeliveries(instance)).toHaveLength(afterRequest)
    })
  })
})
