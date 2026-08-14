import React, { StrictMode } from "react"
import { readFileSync } from "node:fs"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { HomeGetStartedDestinationOverrides } from "@/lib/page-extension-contracts"

const apiRequestMock = vi.hoisted(() => vi.fn())
const resolveTaskLlmSelectionMock = vi.hoisted(() => vi.fn())
const homeExtensionRenderMock = vi.hoisted(() => vi.fn())
const setPendingMessageMock = vi.hoisted(() => vi.fn())
const setTaskIdMock = vi.hoisted(() => vi.fn())
const routerPushMock = vi.hoisted(() => vi.fn())
const toastErrorMock = vi.hoisted(() => vi.fn())
const localeMock = vi.hoisted(() => ({ value: "en" as "en" | "zh" }))
const resolveAgentLogoUrlMock = vi.hoisted(() => vi.fn())
const formatDisplayDateMock = vi.hoisted(() => vi.fn())
const homeGetStartedDestinationOverridesMock = vi.hoisted(() => (
  {} as HomeGetStartedDestinationOverrides
))

async function createHomeExtensionMock() {
  const ReactModule = await vi.importActual<typeof import("react")>("react")
  const HomePageExtension = ReactModule.memo(() => {
    ReactModule.useState(null)
    homeExtensionRenderMock()
    return ReactModule.createElement("div", { "data-testid": "home-extension" })
  })
  return { HomePageExtension, homeGetStartedDestinationOverrides: homeGetStartedDestinationOverridesMock }
}

vi.mock("@/lib/api-wrapper", async () => {
  const actual = await vi.importActual<typeof import("@/lib/api-wrapper")>(
    "@/lib/api-wrapper",
  )
  return { ...actual, apiRequest: apiRequestMock }
})

vi.mock("@/lib/models", () => ({
  resolveTaskLlmSelection: resolveTaskLlmSelectionMock,
}))

vi.mock("@/lib/utils", async () => {
  const actual = await vi.importActual<typeof import("@/lib/utils")>("@/lib/utils")
  return {
    ...actual,
    getApiUrl: () => "http://api.local",
    resolveAgentLogoUrl: (...args: Parameters<typeof actual.resolveAgentLogoUrl>) => {
      resolveAgentLogoUrlMock(...args)
      return actual.resolveAgentLogoUrl(...args)
    },
  }
})

vi.mock("@/lib/time-utils", () => ({
  formatDisplayDate: (...args: [unknown, "en" | "zh", Intl.DateTimeFormatOptions]) => {
    formatDisplayDateMock(...args)
    return typeof args[0] === "string" && args[0].startsWith("2024-")
      ? `formatted:${args[0]}`
      : ""
  },
}))

vi.mock("@/components/ui/sonner", () => ({
  toast: { error: toastErrorMock },
}))

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: routerPushMock }),
}))

vi.mock("next/link", () => ({
  default: ({
    children,
    href,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { href: string }) => (
    <a href={href} {...props}>{children}</a>
  ),
}))

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({
    t: (key: string) => key,
    locale: localeMock.value,
  }),
}))

vi.mock("@/contexts/app-context-chat", () => ({
  useApp: () => ({
    setTaskId: setTaskIdMock,
    setPendingMessage: setPendingMessageMock,
  }),
}))

vi.mock("@/lib/branding", () => ({
  getBrandingFromEnv: () => ({
    appName: "Xagent",
    whiteLogoPath: "/logo-white.png",
  }),
}))

vi.mock("@/components/voice-input-controller", () => ({
  useVoiceInputControls: () => ({
    status: "idle",
    hasAsrModel: true,
    startRecording: vi.fn(),
    stopRecording: vi.fn(),
  }),
}))

vi.mock("@/components/welcome-modal", () => ({
  WelcomeModal: () => null,
}))

vi.mock("@/lib/home-page-extension", createHomeExtensionMock)

import Home from "./page"

const successfulSelection = {
  kind: "success" as const,
  llmIds: ["general", null, null, null] as [string, null, null, null],
}

function jsonResponse(data: unknown, init?: ResponseInit) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "Content-Type": "application/json" },
    ...init,
  })
}

function unreadableResponse() {
  const response = new Response("unreadable")
  Object.defineProperty(response, "text", {
    value: vi.fn().mockRejectedValue(new Error("body unavailable")),
  })
  return response
}

function deferred<T>() {
  let resolve!: (value: T | PromiseLike<T>) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

function input(): HTMLTextAreaElement {
  return screen.getByPlaceholderText("home.hero.searchPlaceholder")
}

function submitButton(): HTMLButtonElement {
  const button = input().parentElement?.querySelector("button:not([aria-label])")
  if (!(button instanceof HTMLButtonElement)) throw new Error("Home submit button not found")
  return button
}

function typePrompt(value: string) {
  fireEvent.input(input(), { target: { value } })
}

function submitWithEnter() {
  fireEvent.keyDown(input(), { key: "Enter" })
}

// Reproduces only voice-input-controller.tsx's native-setter write + bubbled
// input/change event dispatch (setNativeValue + dispatchInputEvents): the native
// HTMLTextAreaElement.prototype value setter (bypassing React's tracked instance
// setter) followed by the same bubbled input/change events production dispatches
// after a transcription lands, rather than testing-library's fireEvent convenience
// helpers. This does NOT cover insertTranscribedText's focus/caret-splice/
// setSelectionRange/fragment-data behavior — caret handling is untested here.
function simulateVoiceTranscription(target: HTMLTextAreaElement, text: string) {
  const descriptor = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")
  act(() => {
    descriptor?.set?.call(target, text)
    try {
      target.dispatchEvent(new InputEvent("input", { bubbles: true, data: text, inputType: "insertText" }))
    } catch {
      target.dispatchEvent(new Event("input", { bubbles: true }))
    }
    target.dispatchEvent(new Event("change", { bubbles: true }))
  })
}

function taskCore(taskId = 7) {
  return {
    task_id: taskId,
    title: "created task",
    status: "running",
    created_at: "2026-01-01T00:00:00Z",
  }
}

function templateCard(id: string, overrides: Record<string, unknown> = {}) {
  return {
    id,
    name: `Template ${id}`,
    category: "Automation",
    description: `Description ${id}`,
    features: [`Feature ${id}`],
    connections: [{ name: `Connection ${id}`, logo: null }],
    setup_time: "5 min",
    likes: 2,
    used_count: 3,
    ...overrides,
  }
}

function recentTask(taskId: number, overrides: Record<string, unknown> = {}) {
  return {
    task_id: taskId,
    title: `Task ${taskId}`,
    created_at: "2026-01-01T00:00:00Z",
    agent_name: "Agent",
    agent_logo_url: null,
    ...overrides,
  }
}

function omitField(value: Record<string, unknown>, field: string) {
  const copy = { ...value }
  delete copy[field]
  return copy
}

function sourceSlice(source: string, start: string, end: string) {
  const startIndex = source.indexOf(start)
  const endIndex = source.indexOf(end, startIndex + start.length)
  if (startIndex < 0 || endIndex < 0) throw new Error(`Missing source slice: ${start} -> ${end}`)
  return source.slice(startIndex, endIndex)
}

function getStartedCard(title: string) {
  const heading = screen.getByRole("heading", { name: title })
  const card = heading.closest('[data-slot="card"]')
  if (!(card instanceof HTMLDivElement)) throw new Error(`Get Started card not found: ${title}`)
  const wrapper = card.parentElement
  if (!(wrapper instanceof HTMLDivElement || wrapper instanceof HTMLAnchorElement)) {
    throw new Error(`Get Started wrapper not found: ${title}`)
  }
  return { card, wrapper }
}

function expectInertGetStartedCard(title: string) {
  const { card, wrapper } = getStartedCard(title)
  const wrapperAndDescendants = [wrapper, ...Array.from(wrapper.querySelectorAll<HTMLElement | SVGElement>("*"))]
  const inertInteractionSelector = [
    "a",
    "button",
    "input",
    "select",
    "textarea",
    "[tabindex]:not([tabindex='-1'])",
    "[role='button']",
    "[role='checkbox']",
    "[role='combobox']",
    "[role='link']",
    "[role='menuitem']",
    "[role='menuitemcheckbox']",
    "[role='menuitemradio']",
    "[role='option']",
    "[role='radio']",
    "[role='searchbox']",
    "[role='slider']",
    "[role='spinbutton']",
    "[role='switch']",
    "[role='tab']",
    "[role='treeitem']",
    "[contenteditable]",
    "summary",
    "details",
    "iframe",
    "object",
    "embed",
    "video[controls]",
    "audio[controls]",
    "[draggable='true']",
  ].join(", ")
  expect(wrapper.tagName).toBe("DIV")
  expect(wrapper).not.toHaveAttribute("role")
  expect(wrapper).not.toHaveAttribute("tabindex")
  expect(wrapper.tabIndex).toBe(-1)
  for (const className of [
    "rounded-xl",
    "outline-none",
    "group",
    "cursor-pointer",
    "hover:shadow-md",
    "focus-visible:outline-none",
    "focus-visible:ring-2",
    "focus-visible:ring-ring",
    "focus-visible:ring-offset-2",
    "focus-visible:ring-offset-background",
  ]) {
    expect(wrapper).not.toHaveClass(className)
  }
  for (const className of ["group", "cursor-pointer", "hover:shadow-md"]) {
    expect(card).not.toHaveClass(className)
  }
  expect(wrapperAndDescendants.filter((element) => element.matches(inertInteractionSelector))).toHaveLength(0)
  const wrapperAndDescendantClassTokens = wrapperAndDescendants
    .flatMap((element) => (element.getAttribute("class") ?? "").split(/\s+/).filter(Boolean))
  expect(wrapperAndDescendantClassTokens.some((className) => (
    className.includes("hover")
    || className.includes("focus-visible")
    || className.includes("group")
    || className.includes("cursor-")
    || /(?:^|:)!?\[cursor:[^\]]+\]$/.test(className)
  ))).toBe(false)
  expect(wrapperAndDescendants.some((element) => (element.getAttribute("style") ?? "").trim() !== "")).toBe(false)
  expect(wrapperAndDescendants.some((element) => (element.getAttribute("cursor") ?? "").trim() !== "")).toBe(false)
}

function expectLinkedGetStartedCard(title: string, href: string) {
  const { card, wrapper } = getStartedCard(title)
  expect(wrapper).toBeInstanceOf(HTMLAnchorElement)
  expect(wrapper).toHaveAttribute("href", href)
  expect(wrapper).toHaveAttribute("target", "_blank")
  expect(wrapper).toHaveAttribute("rel", "noopener noreferrer")
  expect(wrapper).not.toHaveAttribute("tabindex")
  expect(wrapper.tabIndex).toBe(0)
  wrapper.focus()
  expect(wrapper).toHaveFocus()
  expect(wrapper).toHaveClass(
    "focus-visible:outline-none",
    "focus-visible:ring-2",
    "focus-visible:ring-ring",
    "focus-visible:ring-offset-2",
    "focus-visible:ring-offset-background",
    "rounded-xl",
  )
  expect(card).toHaveClass("group", "cursor-pointer", "hover:shadow-md")
  expect(card.querySelector("h3")).toHaveClass("group-hover:text-primary")
  if (title === "home.getStarted.guides.title") {
    expect(card.querySelector("svg")?.parentElement).toHaveClass("group-hover:scale-110")
  }
}

function restoreIntersectionObserver(descriptor: PropertyDescriptor | undefined) {
  if (descriptor) {
    Object.defineProperty(globalThis, "IntersectionObserver", descriptor)
  } else {
    delete (globalThis as { IntersectionObserver?: typeof IntersectionObserver }).IntersectionObserver
  }
}

function templateUrl(locale = localeMock.value) {
  return `http://api.local/api/templates/?lang=${locale}`
}

const recentTasksUrl = "http://api.local/api/chat/tasks?page=1&per_page=5"

describe("Home", () => {
  let consoleErrorMock: ReturnType<typeof vi.spyOn>

  function expectNoTaskCreatePublication() {
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(routerPushMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  }

  beforeEach(() => {
    delete homeGetStartedDestinationOverridesMock.video
    delete homeGetStartedDestinationOverridesMock.docs
    delete homeGetStartedDestinationOverridesMock.guides
    delete homeGetStartedDestinationOverridesMock.whatsNew
    apiRequestMock.mockReset()
    resolveTaskLlmSelectionMock.mockReset()
    homeExtensionRenderMock.mockClear()
    setPendingMessageMock.mockReset()
    setTaskIdMock.mockReset()
    routerPushMock.mockReset()
    toastErrorMock.mockReset()
    resolveAgentLogoUrlMock.mockReset()
    formatDisplayDateMock.mockReset()
    consoleErrorMock = vi.spyOn(console, "error").mockImplementation(() => undefined)
    resolveTaskLlmSelectionMock.mockResolvedValue(successfulSelection)
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) {
        return Promise.resolve(jsonResponse([]))
      }
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") {
        return Promise.resolve(jsonResponse({ tasks: [] }))
      }
      throw new Error(`Unhandled apiRequest: ${url}`)
    })
  })

  afterEach(() => {
    consoleErrorMock.mockRestore()
    cleanup()
  })

  it("renders a hook-bearing configured extension as one component", async () => {
    render(<Home />)

    expect(screen.getByRole("button", { name: "voiceInput.start" })).toBeInTheDocument()
    const extension = await screen.findByTestId("home-extension")
    expect(extension).toBeInTheDocument()
    expect(extension.parentElement).toHaveAttribute("data-slot", "home-page-extension")
    expect(extension.parentElement).toHaveClass("shrink-0")
    expect(screen.getAllByTestId("home-extension")).toHaveLength(1)
    expect(homeExtensionRenderMock).toHaveBeenCalled()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledTimes(2))
  })

  it("renders the shipped default extension in an inert canonical slot", async () => {
    vi.doUnmock("@/lib/home-page-extension")
    vi.resetModules()
    try {
      const { default: DefaultHome } = await import("./page")
      const defaultExtension = await import("@/lib/home-page-extension")
      const { container } = render(<DefaultHome />)
      const slot = container.querySelector('[data-slot="home-page-extension"]')

      expect(slot).toBeInTheDocument()
      expect(slot).toHaveClass("shrink-0", { exact: true })
      expect(slot).not.toHaveAttribute("style")
      expect(slot).toBeEmptyDOMElement()
      expect(
        (defaultExtension as { homeGetStartedDestinationOverrides?: unknown })
          .homeGetStartedDestinationOverrides,
      ).toEqual({})
    } finally {
      vi.doMock("@/lib/home-page-extension", createHomeExtensionMock)
      vi.resetModules()
    }
    const restoredMock = await import("@/lib/home-page-extension")
    expect(
      (restoredMock as { homeGetStartedDestinationOverrides?: unknown })
        .homeGetStartedDestinationOverrides,
    ).toBe(homeGetStartedDestinationOverridesMock)
    const { default: RestoredHome } = await import("./page")
    render(<RestoredHome />)
    expect(await screen.findByTestId("home-extension")).toBeInTheDocument()
    expect(homeExtensionRenderMock).toHaveBeenCalledTimes(1)
  })

  it("renders every canonical Get Started destination when a replacement extension omits homeGetStartedDestinationOverrides entirely (A1 old-surface compatibility)", async () => {
    vi.doUnmock("@/lib/home-page-extension")
    vi.resetModules()
    try {
      vi.doMock("@/lib/home-page-extension", async () => {
        const ReactModule = await vi.importActual<typeof import("react")>("react")
        return {
          HomePageExtension: ReactModule.memo(() => (
            ReactModule.createElement("div", { "data-testid": "old-surface-extension" })
          )),
          // The old replacement surface genuinely has no such export; vitest's
          // mock proxy throws on access of a key absent from the factory's
          // return value (a stricter guard than real module namespace access),
          // so this is declared with an undefined value to reproduce the real
          // "export absent" shape without tripping that guard.
          homeGetStartedDestinationOverrides: undefined,
        }
      })
      vi.resetModules()
      const { default: OldSurfaceHome } = await import("./page")
      render(<OldSurfaceHome />)

      expect(await screen.findByTestId("old-surface-extension")).toBeInTheDocument()
      expectLinkedGetStartedCard("home.getStarted.docs.title", "https://docs.xagent.co/api-reference/introduction")
      expectLinkedGetStartedCard("home.getStarted.guides.title", "https://docs.xagent.co/models/overview")
      expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")
    } finally {
      vi.doMock("@/lib/home-page-extension", createHomeExtensionMock)
      vi.resetModules()
    }
    const restoredMock = await import("@/lib/home-page-extension")
    expect(
      (restoredMock as { homeGetStartedDestinationOverrides?: unknown })
        .homeGetStartedDestinationOverrides,
    ).toBe(homeGetStartedDestinationOverridesMock)
    const { default: RestoredHome } = await import("./page")
    render(<RestoredHome />)
    expect(await screen.findByTestId("home-extension")).toBeInTheDocument()
  })

  it("resolves canonical and distinct configured Get Started destinations per key", () => {
    render(<Home />)

    expectLinkedGetStartedCard("home.getStarted.docs.title", "https://docs.xagent.co/api-reference/introduction")
    expectLinkedGetStartedCard("home.getStarted.guides.title", "https://docs.xagent.co/models/overview")
    expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")
    expectInertGetStartedCard("home.getStarted.video.title")

    cleanup()
    homeGetStartedDestinationOverridesMock.docs = "/docs with internal space"
    homeGetStartedDestinationOverridesMock.guides = "custom-guide:destination"
    homeGetStartedDestinationOverridesMock.whatsNew = "  /whats-new  "
    render(<Home />)

    expectLinkedGetStartedCard("home.getStarted.docs.title", "/docs with internal space")
    expectLinkedGetStartedCard("home.getStarted.guides.title", "custom-guide:destination")
    expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "  /whats-new  ")
    expect(getStartedCard("home.getStarted.video.title").wrapper.querySelector("a")).toBeNull()
  })

  it("resolves a configured video destination while keeping the inline tutorial video and canonical siblings", () => {
    homeGetStartedDestinationOverridesMock.video = "https://help.xagent.co/user-guide/demo-videos.html"
    render(<Home />)

    expectLinkedGetStartedCard("home.getStarted.video.title", "https://help.xagent.co/user-guide/demo-videos.html")
    const linkedVideo = getStartedCard("home.getStarted.video.title").card.querySelector("video")
    if (!(linkedVideo instanceof HTMLVideoElement)) throw new Error("Tutorial video was not eagerly loaded")
    expect(linkedVideo).toHaveAttribute("src", "/videos/Tutorial.mp4")
    expectLinkedGetStartedCard("home.getStarted.docs.title", "https://docs.xagent.co/api-reference/introduction")
    expectLinkedGetStartedCard("home.getStarted.guides.title", "https://docs.xagent.co/models/overview")
    expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")

    cleanup()
    for (const invalid of [null, "", "   "]) {
      ;(homeGetStartedDestinationOverridesMock as Record<string, unknown>).video = invalid
      const { unmount } = render(<Home />)
      expectInertGetStartedCard("home.getStarted.video.title")
      const inertVideo = getStartedCard("home.getStarted.video.title").card.querySelector("video")
      if (!(inertVideo instanceof HTMLVideoElement)) throw new Error("Tutorial video was not eagerly loaded")
      unmount()
    }
  })

  it("rejects pointer, hover, focus, native, inline-style, SVG, and media-control affordances injected into an inert card", () => {
    render(<Home />)
    const { card, wrapper } = getStartedCard("home.getStarted.video.title")
    const video = card.querySelector("video")
    if (!(video instanceof HTMLVideoElement)) throw new Error("Tutorial video was not eagerly loaded")
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg")
    card.append(svg)
    expectInertGetStartedCard("home.getStarted.video.title")
    try {
      card.classList.add("cursor-pointer")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("cursor-pointer")

      card.classList.add("cursor-[pointer]")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("cursor-[pointer]")

      card.classList.add("[cursor:pointer]")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("[cursor:pointer]")

      card.classList.add("dark:![cursor:pointer]")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("dark:![cursor:pointer]")

      card.classList.add("hover:shadow-lg")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("hover:shadow-lg")

      card.classList.add("dark:hover:shadow-lg")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("dark:hover:shadow-lg")

      card.classList.add("[&:hover]:shadow-lg")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("[&:hover]:shadow-lg")

      card.classList.add("group-hover/get-started:shadow-lg")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      card.classList.remove("group-hover/get-started:shadow-lg")

      wrapper.classList.add("focus-visible:ring-2")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      wrapper.classList.remove("focus-visible:ring-2")

      wrapper.setAttribute("style", "cursor: url(/pointer.cur), pointer")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      wrapper.removeAttribute("style")

      wrapper.setAttribute("contenteditable", "true")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      wrapper.removeAttribute("contenteditable")

      wrapper.setAttribute("draggable", "true")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      wrapper.removeAttribute("draggable")

      svg.setAttribute("class", "hover:shadow-lg")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      svg.removeAttribute("class")

      svg.setAttribute("cursor", "pointer")
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
      svg.removeAttribute("cursor")

      video.controls = true
      expect(() => expectInertGetStartedCard("home.getStarted.video.title")).toThrow()
    } finally {
      card.classList.remove("cursor-pointer")
      card.classList.remove("cursor-[pointer]")
      card.classList.remove("[cursor:pointer]")
      card.classList.remove("dark:![cursor:pointer]")
      card.classList.remove("hover:shadow-lg")
      card.classList.remove("dark:hover:shadow-lg")
      card.classList.remove("[&:hover]:shadow-lg")
      card.classList.remove("group-hover/get-started:shadow-lg")
      wrapper.classList.remove("focus-visible:ring-2")
      wrapper.removeAttribute("style")
      wrapper.removeAttribute("contenteditable")
      wrapper.removeAttribute("draggable")
      svg.remove()
      video.controls = false
    }
  })

  it("keeps null, blank, and runtime-invalid configured destinations inert without affecting siblings", () => {
    const invalidValues: unknown[] = [
      1,
      true,
      {},
      [],
      new String("boxed"),
      (globalThis as { BigInt: (value: number) => bigint }).BigInt(1),
      Symbol("destination"),
      () => "/not-used",
    ]

    homeGetStartedDestinationOverridesMock.docs = null
    homeGetStartedDestinationOverridesMock.guides = null
    homeGetStartedDestinationOverridesMock.whatsNew = null
    const nullView = render(<Home />)
    expectInertGetStartedCard("home.getStarted.docs.title")
    expectInertGetStartedCard("home.getStarted.guides.title")
    expectInertGetStartedCard("home.getStarted.whatsNew.title")
    expectInertGetStartedCard("home.getStarted.video.title")
    nullView.unmount()

    for (const invalid of [null, "", " \t\n", "\u00a0", ...invalidValues]) {
      for (const key of ["docs", "guides", "whatsNew"] as const) {
        delete homeGetStartedDestinationOverridesMock.docs
        delete homeGetStartedDestinationOverridesMock.guides
        delete homeGetStartedDestinationOverridesMock.whatsNew
        if (key === "docs") homeGetStartedDestinationOverridesMock.guides = "/valid-guides"
        if (key === "guides") homeGetStartedDestinationOverridesMock.docs = "/valid-docs"
        if (key === "whatsNew") homeGetStartedDestinationOverridesMock.docs = "/valid-docs"
        ;(homeGetStartedDestinationOverridesMock as Record<string, unknown>)[key] = invalid
        const { unmount } = render(<Home />)

        if (key === "docs") {
          expectInertGetStartedCard("home.getStarted.docs.title")
          expectLinkedGetStartedCard("home.getStarted.guides.title", "/valid-guides")
          expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")
        } else if (key === "guides") {
          expectLinkedGetStartedCard("home.getStarted.docs.title", "/valid-docs")
          expectInertGetStartedCard("home.getStarted.guides.title")
          expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")
        } else {
          expectLinkedGetStartedCard("home.getStarted.docs.title", "/valid-docs")
          expectLinkedGetStartedCard("home.getStarted.guides.title", "https://docs.xagent.co/models/overview")
          expectInertGetStartedCard("home.getStarted.whatsNew.title")
        }
        expectInertGetStartedCard("home.getStarted.video.title")
        unmount()
      }
    }
  })

  it("resolves a mixed custom, null, and undefined destination row independently", () => {
    homeGetStartedDestinationOverridesMock.docs = "custom:docs"
    homeGetStartedDestinationOverridesMock.guides = null
    delete homeGetStartedDestinationOverridesMock.whatsNew
    render(<Home />)

    expectLinkedGetStartedCard("home.getStarted.docs.title", "custom:docs")
    expectInertGetStartedCard("home.getStarted.guides.title")
    expectLinkedGetStartedCard("home.getStarted.whatsNew.title", "https://docs.xagent.co/release-notes")
    expect([
      "home.getStarted.video.title",
      "home.getStarted.docs.title",
      "home.getStarted.guides.title",
      "home.getStarted.whatsNew.title",
    ].filter((title) => getStartedCard(title).wrapper instanceof HTMLAnchorElement)).toHaveLength(2)
  })

  it("observes both video cards with exact options and loads each only after its own intersection", async () => {
    const originalIntersectionObserverDescriptor = Object.getOwnPropertyDescriptor(globalThis, "IntersectionObserver")
    const observe = vi.fn()
    const disconnect = vi.fn()
    let callback: IntersectionObserverCallback | undefined
    let options: IntersectionObserverInit | undefined
    let view: ReturnType<typeof render> | undefined
    class FakeIntersectionObserver {
      root = null
      rootMargin = ""
      thresholds: readonly number[] = []
      constructor(nextCallback: IntersectionObserverCallback, nextOptions?: IntersectionObserverInit) {
        callback = nextCallback
        options = nextOptions
      }
      observe = observe
      unobserve = vi.fn()
      disconnect = disconnect
      takeRecords = () => [] as IntersectionObserverEntry[]
    }

    Object.defineProperty(globalThis, "IntersectionObserver", {
      configurable: true,
      writable: true,
      value: FakeIntersectionObserver,
    })
    homeGetStartedDestinationOverridesMock.docs = null
    try {
      view = render(<Home />)
      const videoTargets = Array.from(document.querySelectorAll<HTMLElement>("[data-get-started-video='true']"))
      expectInertGetStartedCard("home.getStarted.docs.title")
      expect(options).toEqual({ rootMargin: "200px 0px", threshold: 0.1 })
      expect(observe).toHaveBeenCalledTimes(2)
      expect(observe.mock.calls[0][0]).toBe(videoTargets[0])
      expect(observe.mock.calls[1][0]).toBe(videoTargets[1])
      expect(videoTargets.map((target) => target.dataset.videoIndex)).toEqual(["0", "1"])
      expect(videoTargets[0].querySelector("video")).toBeNull()
      expect(videoTargets[1].querySelector("video")).toBeNull()
      expect(videoTargets[0].querySelector("svg")).toBeInTheDocument()
      expect(videoTargets[1].querySelector("svg")).toBeInTheDocument()
      expect(document.querySelectorAll("video")).toHaveLength(0)
      if (!callback) throw new Error("IntersectionObserver callback was not registered")

      await act(async () => callback?.([
        { isIntersecting: false, target: videoTargets[0] },
        { isIntersecting: false, target: videoTargets[1] },
      ] as unknown as IntersectionObserverEntry[], {} as IntersectionObserver))
      expect(document.querySelectorAll("video")).toHaveLength(0)
      expect(videoTargets[0].querySelector("video")).toBeNull()
      expect(videoTargets[1].querySelector("video")).toBeNull()
      expect(videoTargets[0].querySelector("svg")).toBeInTheDocument()
      expect(videoTargets[1].querySelector("svg")).toBeInTheDocument()

      await act(async () => callback?.([
        { isIntersecting: true, target: videoTargets[1] },
      ] as unknown as IntersectionObserverEntry[], {} as IntersectionObserver))
      expect(document.querySelector('video[src="/videos/Documentation.mp4"]')).toBeInTheDocument()
      expect(document.querySelector('video[src="/videos/Tutorial.mp4"]')).toBeNull()

      await act(async () => callback?.([
        { isIntersecting: true, target: videoTargets[0] },
      ] as unknown as IntersectionObserverEntry[], {} as IntersectionObserver))
      expect(document.querySelector('video[src="/videos/Tutorial.mp4"]')).toBeInTheDocument()
      expect(document.querySelector('video[src="/videos/Documentation.mp4"]')).toBeInTheDocument()
      view.unmount()
      view = undefined
      expect(disconnect).toHaveBeenCalledTimes(1)
    } finally {
      view?.unmount()
      restoreIntersectionObserver(originalIntersectionObserverDescriptor)
      expect(Object.getOwnPropertyDescriptor(globalThis, "IntersectionObserver")).toEqual(originalIntersectionObserverDescriptor)
    }
  })

  it("continues observing disabled destinations and eagerly loads both videos without IntersectionObserver", () => {
    const originalIntersectionObserverDescriptor = Object.getOwnPropertyDescriptor(globalThis, "IntersectionObserver")
    homeGetStartedDestinationOverridesMock.docs = null
    let view: ReturnType<typeof render> | undefined
    try {
      Object.defineProperty(globalThis, "IntersectionObserver", {
        configurable: true,
        writable: true,
        value: undefined,
      })
      view = render(<Home />)
      expectInertGetStartedCard("home.getStarted.docs.title")
      expect(document.querySelector('video[src="/videos/Tutorial.mp4"]')).toBeInTheDocument()
      expect(document.querySelector('video[src="/videos/Documentation.mp4"]')).toBeInTheDocument()
      view.unmount()
      view = undefined
    } finally {
      view?.unmount()
      restoreIntersectionObserver(originalIntersectionObserverDescriptor)
      expect(Object.getOwnPropertyDescriptor(globalThis, "IntersectionObserver")).toEqual(originalIntersectionObserverDescriptor)
    }
  })

  it("keeps the Home replacement contract, resolver, and interaction owner non-vacuously source-locked", () => {
    const contractsSource = readFileSync("src/lib/page-extension-contracts.ts", "utf8")
    const extensionSource = readFileSync("src/lib/home-page-extension.tsx", "utf8")
    const pageSource = readFileSync("src/app/page.tsx", "utf8")
    const contractInterface = sourceSlice(
      contractsSource,
      "export interface HomeGetStartedDestinationOverrides",
      "// The page guarantees a stable Provider lifetime and agentId join key.",
    )
    const resolver = sourceSlice(pageSource, "function resolveHomeGetStartedDestination(", "export default function Home()")
    const cardRender = sourceSlice(pageSource, "{[", "          {/* Build agents with templates */}")

    expect(contractInterface).toMatch(/video\?: string \| null/)
    expect(contractInterface).toMatch(/docs\?: string \| null/)
    expect(contractInterface).toMatch(/guides\?: string \| null/)
    expect(contractInterface).toMatch(/whatsNew\?: string \| null/)
    expect(contractInterface.match(/\?: string \| null/g)).toHaveLength(4)
    expect(contractInterface.match(/^\s+\w+\??:/gm)).toHaveLength(4)
    expect(contractInterface).not.toContain("tutorial")
    expect(extensionSource).toMatch(/export const homeGetStartedDestinationOverrides: HomeGetStartedDestinationOverrides = \{\}/)
    expect(pageSource).toMatch(/import \* as homePageExtensionModule from "@\/lib\/home-page-extension";/)
    expect(pageSource).toMatch(
      /const homeGetStartedDestinationOverrides: HomeGetStartedDestinationOverrides =\s*\(homePageExtensionModule as \{ homeGetStartedDestinationOverrides\?: HomeGetStartedDestinationOverrides \}\)\s*\.homeGetStartedDestinationOverrides \?\? \{\}/,
    )
    expect(pageSource).toMatch(/const defaultHomeGetStartedDestinations: Record<keyof HomeGetStartedDestinationOverrides, string \| null> = \{/)
    expect(resolver).toContain("configured === undefined")
    expect(resolver).toContain("typeof configured !== \"string\"")
    expect(resolver).toContain("configured.trim().length === 0")
    expect(resolver).toContain("return configured")
    expect(resolver).not.toMatch(/\?\?|return configured\.trim\(\)/)
    expect(cardRender).toContain("const isLinked = typeof card.link === \"string\"")
    expect(cardRender).toContain("isLinked &&")
    expect(cardRender).toContain("focus-visible:ring-2")
    expect(cardRender).not.toMatch(/\bon[A-Z][A-Za-z]*|\btabIndex\b|\brole\b|\bcontrols\b|\bstyle\b|\bcursor\s*=|\bcontentEditable\b|\bsuppressContentEditableWarning\b|\bdraggable\b|\{\s*\.\.\./)
    const destinationCalls = Array.from(cardRender.matchAll(
      /resolveHomeGetStartedDestination\(homeGetStartedDestinationOverrides\.(video|docs|guides|whatsNew), defaultHomeGetStartedDestinations\.\1\)/g,
    )).map((match) => match[1])
    expect(destinationCalls).toEqual(["video", "docs", "guides", "whatsNew"])
    expect(cardRender.match(/resolveHomeGetStartedDestination\(/g)).toHaveLength(4)
    expect(cardRender.match(/defaultHomeGetStartedDestinations\./g)).toHaveLength(4)
    expect(cardRender).not.toMatch(/https:\/\/docs\.xagent\.co\//)
  })

  it("uses the shared resolver, real task body parser, and ordered successful commit", async () => {
    const events: string[] = []
    setPendingMessageMock.mockImplementation(() => events.push("pending"))
    setTaskIdMock.mockImplementation(() => {
      events.push("taskId")
      expect(input()).toHaveValue("  hello\n  world  ")
      expect(input().style.height).toBe("56px")
    })
    apiRequestMock.mockImplementation((url: string, options?: RequestInit) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") {
        expect(options?.method).toBe("POST")
        expect(JSON.parse(String(options?.body))).toEqual({
          title: "hello world",
          description: "hello\n  world",
          llm_ids: ["general", null, null, null],
        })
        return Promise.resolve(jsonResponse(taskCore(9)))
      }
      throw new Error(`Unexpected apiRequest: ${url}`)
    })

    render(<Home />)
    typePrompt("  hello\n  world  ")
    input().style.height = "56px"
    submitWithEnter()

    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(9))
    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(events).toEqual(["pending", "taskId"])
    expect(setPendingMessageMock).toHaveBeenCalledWith({
      message: "hello\n  world",
      files: [],
      targetTaskId: 9,
    })
    expect(input().value).toBe("")
    expect(input().style.height).toBe("auto")
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  it("keeps no_model distinct from an operational resolver failure", async () => {
    resolveTaskLlmSelectionMock.mockResolvedValueOnce({ kind: "no_model" })
    render(<Home />)
    const noModelPrompt = "  no model draft  "
    typePrompt(noModelPrompt)
    input().style.height = "72px"
    const noModelStyle = input().getAttribute("style")
    submitWithEnter()
    expect(await screen.findByText("chatPage.input.noModelAlert")).toBeInTheDocument()
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(input().value).toBe(noModelPrompt)
    expect(input()).toHaveAttribute("style", noModelStyle)

    cleanup()
    resolveTaskLlmSelectionMock.mockResolvedValueOnce({
      kind: "operational_error",
      error: new Error("resolver failure"),
    })
    render(<Home />)
    const operationalErrorPrompt = "  operational error draft  "
    typePrompt(operationalErrorPrompt)
    input().style.height = "64px"
    const operationalErrorStyle = input().getAttribute("style")
    submitWithEnter()
    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe(operationalErrorPrompt)
    expect(input()).toHaveAttribute("style", operationalErrorStyle)
  })

  it("uses the ref token as a same-act Enter/click latch", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    render(<Home />)
    typePrompt("prompt")

    await act(async () => {
      submitWithEnter()
      fireEvent.click(submitButton())
    })

    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)
    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
  })

  it("does not POST or publish effects after unmount during model resolution", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    view.unmount()
    await act(async () => selection.resolve(successfulSelection))

    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("silences a rejected resolver after Home unmount", async () => {
    const selection = deferred<typeof successfulSelection>()
    resolveTaskLlmSelectionMock.mockReturnValueOnce(selection.promise)
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(1)

    view.unmount()
    await act(async () => selection.reject(new Error("resolver unavailable")))

    expect(apiRequestMock).not.toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything())
    expectNoTaskCreatePublication()
  })

  it("silences a rejected create transport after Home unmount", async () => {
    const response = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return response.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    view.unmount()
    await act(async () => response.reject(new Error("transport unavailable")))

    expectNoTaskCreatePublication()
  })

  it("silences parseApiResponse's empty-data fallback after Response.text() rejects post-unmount", async () => {
    const body = deferred<string>()
    const taskResponse = new Response("unreadable")
    const text = vi.fn(() => body.promise)
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(text).toHaveBeenCalledTimes(1))

    view.unmount()
    await act(async () => {
      body.reject(new Error("body unavailable"))
      await new Promise((resolve) => setTimeout(resolve, 20))
    })

    expectNoTaskCreatePublication()
  })

  it("does not parse a response that arrives after unmount", async () => {
    const response = deferred<Response>()
    const taskResponse = new Response(JSON.stringify(taskCore()))
    const text = vi.fn(taskResponse.text.bind(taskResponse))
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return response.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    view.unmount()
    await act(async () => response.resolve(taskResponse))

    expect(text).not.toHaveBeenCalled()
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("allows parsing to finish after unmount but publishes no effects", async () => {
    const body = deferred<string>()
    const taskResponse = new Response("")
    const text = vi.fn(() => body.promise)
    Object.defineProperty(taskResponse, "text", { value: text })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    const view = render(<Home />)
    typePrompt("prompt")
    submitWithEnter()
    await waitFor(() => expect(text).toHaveBeenCalledTimes(1))

    view.unmount()
    await act(async () => body.resolve(JSON.stringify(taskCore())))

    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).not.toHaveBeenCalled()
  })

  it("reports a mounted create transport rejection and preserves the draft and height", async () => {
    const response = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return response.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    const rejectedPrompt = "prompt"
    typePrompt(rejectedPrompt)
    input().style.height = "80px"
    const rejectedStyle = input().getAttribute("style")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    await act(async () => response.reject(new Error("transport unavailable")))

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(routerPushMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe(rejectedPrompt)
    expect(input()).toHaveAttribute("style", rejectedStyle)
  })

  it.each([
    ["non-OK", jsonResponse(taskCore(), { status: 500 })],
    ["empty", new Response("")],
    ["malformed", new Response("{")],
    ["invalid core", jsonResponse({ id: 7 })],
  ])("keeps task actions at zero and reports current %s create failure", async (_name, taskResponse) => {
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    const failurePrompt = "prompt"
    typePrompt(failurePrompt)
    input().style.height = "88px"
    const failureStyle = input().getAttribute("style")
    submitWithEnter()

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe(failurePrompt)
    expect(input()).toHaveAttribute("style", failureStyle)
  })

  it("reports a current unreadable task body as one operational failure", async () => {
    const taskResponse = new Response("unreadable")
    Object.defineProperty(taskResponse, "text", {
      value: vi.fn().mockRejectedValue(new Error("body unavailable")),
    })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(taskResponse)
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    const unreadablePrompt = "prompt"
    typePrompt(unreadablePrompt)
    input().style.height = "92px"
    const unreadableStyle = input().getAttribute("style")
    submitWithEnter()

    await waitFor(() => expect(toastErrorMock).toHaveBeenCalledWith("common.errors.taskFailed"))
    expect(setPendingMessageMock).not.toHaveBeenCalled()
    expect(setTaskIdMock).not.toHaveBeenCalled()
    expect(input().value).toBe(unreadablePrompt)
    expect(input()).toHaveAttribute("style", unreadableStyle)
  })

  it("never overwrites a newer B draft or its height after A succeeds or fails", async () => {
    const result = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return result.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    input().style.height = "72px"

    await act(async () => result.resolve(jsonResponse(taskCore())))
    expect(input().value).toBe("B")
    expect(input().style.height).toBe("72px")

    cleanup()
    const failed = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return failed.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    input().style.height = "72px"
    await act(async () => failed.resolve(jsonResponse(taskCore(), { status: 500 })))

    expect(input().value).toBe("B")
    expect(input().style.height).toBe("72px")
  })

  it("preserves A after an ABA edit even when the old attempt succeeds", async () => {
    const result = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return result.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))
    typePrompt("B")
    typePrompt("A")
    input().style.height = "64px"

    await act(async () => result.resolve(jsonResponse(taskCore())))

    expect(input().value).toBe("A")
    expect(input().style.height).toBe("64px")
  })

  it("treats a native voice transcription write as an edit that blocks a stale successful clear (A3)", async () => {
    const result = deferred<Response>()
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return result.promise
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("A")
    submitWithEnter()
    await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith("http://api.local/api/chat/task/create", expect.anything()))

    simulateVoiceTranscription(input(), "A transcribed by voice")
    expect(input().value).toBe("A transcribed by voice")

    await act(async () => result.resolve(jsonResponse(taskCore())))

    expect(setPendingMessageMock).toHaveBeenCalledTimes(1)
    expect(setTaskIdMock).toHaveBeenCalledWith(7)
    expect(input().value).toBe("A transcribed by voice")
  })

  it("does not classify a synchronous commit collaborator throw as an operational create failure", async () => {
    setTaskIdMock.mockImplementation(() => { throw new Error("commit failed") })
    apiRequestMock.mockImplementation((url: string) => {
      if (url.startsWith("http://api.local/api/templates/")) return Promise.resolve(jsonResponse([]))
      if (url === "http://api.local/api/chat/tasks?page=1&per_page=5") return Promise.resolve(jsonResponse({ tasks: [] }))
      if (url === "http://api.local/api/chat/task/create") return Promise.resolve(jsonResponse(taskCore()))
      throw new Error(`Unexpected apiRequest: ${url}`)
    })
    render(<Home />)
    typePrompt("prompt")
    submitWithEnter()

    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    expect(setPendingMessageMock).toHaveBeenCalledTimes(1)
    expect(toastErrorMock).not.toHaveBeenCalled()
    expect(consoleErrorMock).toHaveBeenCalled()
    expect(input().value).toBe("prompt")

    const resolverCallsBeforeRetry = resolveTaskLlmSelectionMock.mock.calls.length
    setTaskIdMock.mockReset()
    await waitFor(() => expect(submitButton()).toBeEnabled())
    fireEvent.click(submitButton())
    await waitFor(() => expect(resolveTaskLlmSelectionMock).toHaveBeenCalledTimes(resolverCallsBeforeRetry + 1))
    await waitFor(() => expect(setTaskIdMock).toHaveBeenCalledWith(7))
    expect(toastErrorMock).not.toHaveBeenCalled()
  })

  describe("independent Home loaders", () => {
    beforeEach(() => {
      localeMock.value = "en"
    })

    it("keeps recent tasks when the template transport fails and reports the template owner", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.reject(new Error("templates down"))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(1)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Task 1")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
      expect(screen.queryByText("Template 1")).not.toBeInTheDocument()
    })

    it("keeps templates when the recent-task transport fails and reports the recent owner", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("one")]))
        if (url === recentTasksUrl) return Promise.reject(new Error("tasks down"))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Template one")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
      expect(screen.queryByText("Task 1")).not.toBeInTheDocument()
    })

    it.each([
      ["empty", new Response("")],
      ["malformed", new Response("{")],
      ["body-reader rejection with parser empty fallback", unreadableResponse()],
      ["JSON primitive", jsonResponse("not a template collection")],
      ["wrong shape", jsonResponse({ templates: [] })],
    ])("treats current template %s data as its own failure", async (_name, response) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(response)
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(2)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Task 2")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
    })

    it.each([
      ["empty", new Response("")],
      ["malformed", new Response("{")],
      ["body-reader rejection with parser empty fallback", unreadableResponse()],
      ["JSON primitive", jsonResponse(7)],
      ["unsupported array", jsonResponse([recentTask(3)])],
      ["wrong object", jsonResponse({ pagination: {} })],
    ])("treats current recent %s data as its own failure", async (_name, response) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("kept")]))
        if (url === recentTasksUrl) return Promise.resolve(response)
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Template kept")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
    })

    it("rejects non-OK loader responses before body parsing and clears only old templates", async () => {
      const oldTemplate = deferred<Response>()
      const currentBody = vi.fn(() => Promise.resolve(JSON.stringify([templateCard("bad")])) )
      let templateRequest = 0
      apiRequestMock.mockImplementation((url: string) => {
        if (url.startsWith("http://api.local/api/templates/")) {
          templateRequest += 1
          if (templateRequest === 1) return oldTemplate.promise
          const response = new Response("valid but unavailable", { status: 503 })
          Object.defineProperty(response, "text", { value: currentBody })
          return Promise.resolve(response)
        }
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(4)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      await act(async () => oldTemplate.resolve(jsonResponse([templateCard("old")])))
      expect(await screen.findByText("Template old")).toBeInTheDocument()

      localeMock.value = "zh"
      view.rerender(<Home />)

      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
      expect(screen.queryByText("Template old")).not.toBeInTheDocument()
      expect(screen.getByText("Task 4")).toBeInTheDocument()
      expect(currentBody).not.toHaveBeenCalled()
    })

    it("rejects a current non-OK recent response before parsing without affecting templates", async () => {
      const recentBody = vi.fn(() => Promise.resolve(JSON.stringify({ tasks: [recentTask(40)] })))
      const recentResponse = new Response("valid but unavailable", { status: 503 })
      Object.defineProperty(recentResponse, "text", { value: recentBody })
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("kept")]))
        if (url === recentTasksUrl) return Promise.resolve(recentResponse)
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Template kept")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
      expect(recentBody).not.toHaveBeenCalled()
      expect(screen.queryByText("Task 40")).not.toBeInTheDocument()
    })

    it("validates every template before retaining the ordered first three and navigation IDs", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([
          templateCard("one"), templateCard("two"), templateCard("three"), templateCard("four"),
        ]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [] }))
        if (url === "http://api.local/api/templates/two/use") return Promise.resolve(jsonResponse({}))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Template one")).toBeInTheDocument()
      expect(screen.getByText("Template two")).toBeInTheDocument()
      expect(screen.getByText("Template three")).toBeInTheDocument()
      expect(screen.queryByText("Template four")).not.toBeInTheDocument()
      fireEvent.click(screen.getAllByRole("button", { name: "home.templates.useTemplate" })[1])
      await waitFor(() => expect(routerPushMock).toHaveBeenCalledWith("/build/new?template=two"))
    })

    it("renders every copied template and recent-task value from distinctive wire fields", async () => {
      const createdAt = "2024-05-06T07:08:09Z"
      const formattedDate = `formatted:${createdAt}`
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([
          templateCard("distinct-template-id", {
            name: "Distinct Template Name",
            category: "Distinct Template Category",
            description: "Distinct Template Description",
            features: [],
            connections: [{ name: "Distinct Connection Name", logo: "https://assets.local/distinct-connection.png" }],
            setup_time: "37 distinctive minutes",
            likes: 7654321,
            used_count: 7654322,
          }),
          templateCard("feature-template-id", {
            name: "Feature Template Name",
            features: ["Distinct Template Feature"],
          }),
        ]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(8765432, {
          title: "Distinct Recent Title",
          created_at: createdAt,
          agent_name: "Distinct Recent Agent",
          agent_logo_url: "https://assets.local/distinct-agent.png",
        })] }))
        if (url === "http://api.local/api/templates/distinct-template-id/use") return Promise.resolve(jsonResponse({}))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Distinct Template Name")).toBeInTheDocument()
      expect(screen.getByText("Distinct Template Category")).toBeInTheDocument()
      expect(screen.getByText("Distinct Template Description")).toBeInTheDocument()
      expect(screen.getByText("Distinct Template Feature")).toBeInTheDocument()
      expect(screen.getByText("37 distinctive minutes")).toBeInTheDocument()
      expect(screen.getByText("7654321")).toBeInTheDocument()
      expect(screen.getByText("7654322")).toBeInTheDocument()
      expect(screen.getByRole("img", { name: "Distinct Connection Name" })).toHaveAttribute(
        "src", "https://assets.local/distinct-connection.png",
      )
      fireEvent.click(screen.getAllByRole("button", { name: "home.templates.useTemplate" })[0])
      await waitFor(() => expect(routerPushMock).toHaveBeenCalledWith("/build/new?template=distinct-template-id"))

      const recentLink = screen.getByRole("link", { name: /Distinct Recent Title/ })
      expect(recentLink).toHaveAttribute("href", "/task/8765432")
      expect(recentLink).toHaveTextContent("Distinct Recent Agent")
      expect(recentLink).toHaveTextContent(formattedDate)
      expect(screen.getByRole("img", { name: "Agent" })).toHaveAttribute(
        "src", "https://assets.local/distinct-agent.png",
      )
    })

    it("rejects a malformed fourth template instead of slicing before complete validation", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([
          templateCard("one"), templateCard("two"), templateCard("three"), templateCard("bad", { likes: 1.5 }),
        ]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(5)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)

      expect(await screen.findByText("Task 5")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
      expect(screen.queryByText("Template one")).not.toBeInTheDocument()
    })

    it.each([
      ["id", { id: 1 }], ["name", { name: 1 }], ["category", { category: 1 }],
      ["description", { description: 1 }], ["setup_time", { setup_time: 1 }],
      ["features array", { features: "feature" }], ["feature member", { features: [1] }],
      ["connections array", { connections: {} }], ["connection record", { connections: [null] }],
      ["connection name", { connections: [{ name: 1, logo: null }] }],
      ["connection logo", { connections: [{ name: "ok", logo: 1 }] }],
      ["likes non-number", { likes: "1" }], ["likes fraction", { likes: 1.5 }],
      ["likes unsafe", { likes: Number.MAX_SAFE_INTEGER + 1 }],
      ["used count non-number", { used_count: "1" }], ["used count fraction", { used_count: 1.5 }],
      ["used count unsafe", { used_count: Number.MAX_SAFE_INTEGER + 1 }],
    ])("rejects malformed consumed template %s", async (_name, patch) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("bad", patch)]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
    })

    it.each([
      ["missing id", omitField(templateCard("bad"), "id")], ["null id", templateCard("bad", { id: null })],
      ["missing name", omitField(templateCard("bad"), "name")], ["null name", templateCard("bad", { name: null })],
      ["missing category", omitField(templateCard("bad"), "category")], ["null category", templateCard("bad", { category: null })],
      ["missing description", omitField(templateCard("bad"), "description")], ["null description", templateCard("bad", { description: null })],
      ["missing features", omitField(templateCard("bad"), "features")], ["null features", templateCard("bad", { features: null })],
      ["missing connections", omitField(templateCard("bad"), "connections")], ["null connections", templateCard("bad", { connections: null })],
      ["missing setup_time", omitField(templateCard("bad"), "setup_time")], ["null setup_time", templateCard("bad", { setup_time: null })],
      ["missing likes", omitField(templateCard("bad"), "likes")], ["null likes", templateCard("bad", { likes: null })],
      ["missing used_count", omitField(templateCard("bad"), "used_count")], ["null used_count", templateCard("bad", { used_count: null })],
      ["missing connection name", templateCard("bad", { connections: [omitField({ name: "connection", logo: null }, "name")] })],
      ["null connection name", templateCard("bad", { connections: [{ name: null, logo: null }] })],
      ["missing connection logo", templateCard("bad", { connections: [omitField({ name: "connection", logo: null }, "logo")] })],
    ])("rejects producer-required template field when %s", async (_name, record) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([record]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(51)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findByText("Task 51")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch templates:", expect.any(Error)))
    })

    it("accepts blank values for every consumed template string, null and blank logos, negative safe counters, and ignores broader fields", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("", {
          name: "", category: "", description: "", setup_time: "", features: [""],
          connections: [{ name: "", logo: "" }], likes: -2, used_count: -3,
          featured: "wrong", sample_prompts: "wrong", tags: "wrong", author: 1, version: null, views: "wrong", is_liked: "wrong",
        }), templateCard("null-logo", { connections: [{ name: "", logo: null }] })]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findAllByRole("button", { name: "home.templates.useTemplate" })).toHaveLength(2)
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it.each([
      ["numeric string ID", { task_id: "7" }], ["zero ID", { task_id: 0 }], ["negative ID", { task_id: -1 }],
      ["fraction ID", { task_id: 1.5 }], ["unsafe ID", { task_id: Number.MAX_SAFE_INTEGER + 1 }],
      ["non-string title", { title: 1 }], ["invalid date", { created_at: 1 }],
      ["invalid name", { agent_name: null }], ["invalid logo", { agent_logo_url: 1 }],
    ])("rejects malformed consumed recent-task %s", async (_name, patch) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("kept")]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(6, patch)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findByText("Template kept")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
    })

    it.each([
      ["missing task_id", omitField(recentTask(52), "task_id")],
      ["null task_id", recentTask(52, { task_id: null })],
      ["missing title", omitField(recentTask(52), "title")],
      ["null title", recentTask(52, { title: null })],
    ])("rejects producer-required recent-task field when %s", async (_name, record) => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("kept-required")]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [record] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findByText("Template kept-required")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
    })

    it("rejects a null tasks property while templates remain visible", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([templateCard("kept-null-tasks")]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: null }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findByText("Template kept-null-tasks")).toBeInTheDocument()
      await waitFor(() => expect(consoleErrorMock).toHaveBeenCalledWith("Failed to fetch recent tasks:", expect.any(Error)))
    })

    it("accepts producer-omitted Agent fields and renders each existing fallback", async () => {
      const withoutAgentName = omitField(
        recentTask(11, { agent_logo_url: "/assets/agent-11.png" }),
        "agent_name",
      )
      const withoutAgentLogo = omitField(
        recentTask(12, { agent_name: "Named Agent" }),
        "agent_logo_url",
      )
      const withoutAgent = omitField(
        omitField(recentTask(13), "agent_name"),
        "agent_logo_url",
      )
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([]))
        if (url === recentTasksUrl) {
          return Promise.resolve(jsonResponse({
            tasks: [withoutAgentName, withoutAgentLogo, withoutAgent],
          }))
        }
        throw new Error(`Unexpected apiRequest: ${url}`)
      })

      render(<Home />)

      const withoutNameLink = await screen.findByRole("link", { name: /Task 11/ })
      expect(withoutNameLink).toHaveTextContent("home.recent.defaultAgent")
      expect(withoutNameLink.querySelector("img")).toHaveAttribute(
        "src",
        "http://api.local/assets/agent-11.png",
      )

      const withoutLogoLink = screen.getByRole("link", { name: /Task 12/ })
      expect(withoutLogoLink).toHaveTextContent("Named Agent")
      expect(withoutLogoLink.querySelector("img")).not.toBeInTheDocument()
      expect(withoutLogoLink.firstElementChild?.firstElementChild?.querySelector("svg")).toBeInTheDocument()

      const withoutAgentLink = screen.getByRole("link", { name: /Task 13/ })
      expect(withoutAgentLink).toHaveTextContent("home.recent.defaultAgent")
      expect(withoutAgentLink.querySelector("img")).not.toBeInTheDocument()
      expect(withoutAgentLink.firstElementChild?.firstElementChild?.querySelector("svg")).toBeInTheDocument()
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("accepts recent blank/null/missing display fields and ignores unconsumed task and pagination data", async () => {
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({
          tasks: [
            recentTask(7, { title: "", created_at: "not a date", agent_name: "", agent_logo_url: "", status: 1, model_id: {}, total_tokens: "wrong" }),
            recentTask(8, { created_at: null }),
            recentTask(9, { created_at: "" }),
            { task_id: 10, title: "Task 10", agent_name: "Agent", agent_logo_url: null },
          ],
          pagination: "wrong",
        }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<Home />)
      expect(await screen.findByRole("link", { name: /home\.recent\.untitledTask/ })).toHaveAttribute("href", "/task/7")
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("resolves each Recent Task Agent logo once and keeps date metadata structurally complete", async () => {
      const validCreatedAt = "2024-05-06T07:08:09Z"
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(jsonResponse([]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({
          tasks: [
            recentTask(71, {
              created_at: validCreatedAt,
              agent_name: "Resolved Agent",
              agent_logo_url: "/logos/resolved.png",
            }),
            recentTask(72, {
              created_at: "not-a-date",
              agent_name: "Fallback Agent",
              agent_logo_url: "javascript:alert(1)",
            }),
          ],
        }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })

      render(<Home />)

      const validLink = await screen.findByRole("link", { name: /Task 71/ })
      const invalidLink = screen.getByRole("link", { name: /Task 72/ })
      expect(validLink.querySelector("img")).toHaveAttribute(
        "src",
        "http://api.local/logos/resolved.png",
      )
      const invalidAvatar = invalidLink.querySelector('div[class*="w-12"][class*="h-12"]')
      expect(invalidAvatar).not.toBeNull()
      expect(invalidAvatar?.querySelectorAll("img")).toHaveLength(0)
      expect(invalidAvatar?.querySelectorAll("svg")).toHaveLength(1)
      const invalidBot = invalidAvatar?.querySelector("svg")
      expect(invalidBot).not.toHaveClass("lucide-chevron-right")
      expect(invalidBot?.querySelector('rect[width="18"][height="10"]')).toBeInTheDocument()
      const validMetadata = validLink.querySelector("p.text-muted-foreground.font-medium")
      const invalidMetadata = invalidLink.querySelector("p.text-muted-foreground.font-medium")
      expect(validMetadata?.textContent).toBe(`Resolved Agent • formatted:${validCreatedAt}`)
      expect(invalidMetadata?.textContent).toBe("Fallback Agent")
      expect(invalidLink).not.toHaveTextContent("Invalid Date")
      expect(resolveAgentLogoUrlMock).toHaveBeenCalledTimes(2)
      expect(resolveAgentLogoUrlMock).toHaveBeenNthCalledWith(
        1,
        "/logos/resolved.png",
        "http://api.local",
      )
      expect(resolveAgentLogoUrlMock).toHaveBeenNthCalledWith(
        2,
        "javascript:alert(1)",
        "http://api.local",
      )
      expect(formatDisplayDateMock).toHaveBeenCalledTimes(2)
      expect(formatDisplayDateMock).toHaveBeenNthCalledWith(1, validCreatedAt, "en", {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      })
      expect(formatDisplayDateMock).toHaveBeenNthCalledWith(2, "not-a-date", "en", {
        month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
      })
    })

    it("does not refetch recents on locale change and suppresses an old locale completion", async () => {
      const en = deferred<Response>()
      const zh = deferred<Response>()
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl("en")) return en.promise
        if (url === templateUrl("zh")) return zh.promise
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [recentTask(8)] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      await waitFor(() => expect(apiRequestMock).toHaveBeenCalledWith(recentTasksUrl))
      const recentCalls = apiRequestMock.mock.calls.filter(([url]) => url === recentTasksUrl).length
      localeMock.value = "zh"
      view.rerender(<Home />)
      await act(async () => en.resolve(jsonResponse([templateCard("old")])))
      expect(screen.queryByText("Template old")).not.toBeInTheDocument()
      await act(async () => zh.resolve(jsonResponse([templateCard("new")])))
      expect(await screen.findByText("Template new")).toBeInTheDocument()
      expect(apiRequestMock.mock.calls.filter(([url]) => url === recentTasksUrl)).toHaveLength(recentCalls)
    })

    it("keeps current-locale templates when an old valid body finishes parsing last", async () => {
      const oldBody = deferred<string>()
      const oldResponse = new Response("old")
      const oldText = vi.fn(() => oldBody.promise)
      Object.defineProperty(oldResponse, "text", { value: oldText })
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl("en")) return Promise.resolve(oldResponse)
        if (url === templateUrl("zh")) return Promise.resolve(jsonResponse([templateCard("current-zh")]))
        if (url === recentTasksUrl) return Promise.resolve(jsonResponse({ tasks: [] }))
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      await waitFor(() => expect(oldText).toHaveBeenCalledTimes(1))

      localeMock.value = "zh"
      view.rerender(<Home />)
      expect(await screen.findByText("Template current-zh")).toBeInTheDocument()

      await act(async () => oldBody.resolve(JSON.stringify([templateCard("obsolete-en")])))
      expect(screen.getByText("Template current-zh")).toBeInTheDocument()
      expect(screen.queryByText("Template obsolete-en")).not.toBeInTheDocument()
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("does not parse or diagnose either source after unmount, including parser fallback", async () => {
      const templates = deferred<Response>()
      const tasks = deferred<Response>()
      const templateResponse = new Response("template")
      const recentResponse = new Response("recent")
      const templateText = vi.fn(templateResponse.text.bind(templateResponse))
      const recentText = vi.fn(() => Promise.reject(new Error("body unavailable")))
      Object.defineProperty(templateResponse, "text", { value: templateText })
      Object.defineProperty(recentResponse, "text", { value: recentText })
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return templates.promise
        if (url === recentTasksUrl) return tasks.promise
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      view.unmount()
      await act(async () => {
        templates.resolve(templateResponse)
        tasks.resolve(recentResponse)
      })
      expect(templateText).not.toHaveBeenCalled()
      expect(recentText).not.toHaveBeenCalled()
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("silences both loader transport rejections after unmount", async () => {
      const templates = deferred<Response>()
      const tasks = deferred<Response>()
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return templates.promise
        if (url === recentTasksUrl) return tasks.promise
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      view.unmount()
      await act(async () => {
        templates.reject(new Error("templates unavailable"))
        tasks.reject(new Error("tasks unavailable"))
      })
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("allows parser fallback to settle after cleanup without publishing either loader", async () => {
      const templateBody = deferred<string>()
      const recentBody = deferred<string>()
      const templateResponse = new Response("template")
      const recentResponse = new Response("recent")
      const templateText = vi.fn(() => templateBody.promise)
      const recentText = vi.fn(() => recentBody.promise)
      Object.defineProperty(templateResponse, "text", { value: templateText })
      Object.defineProperty(recentResponse, "text", { value: recentText })
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return Promise.resolve(templateResponse)
        if (url === recentTasksUrl) return Promise.resolve(recentResponse)
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      const view = render(<Home />)
      await waitFor(() => {
        expect(templateText).toHaveBeenCalledTimes(1)
        expect(recentText).toHaveBeenCalledTimes(1)
      })
      view.unmount()
      await act(async () => {
        templateBody.resolve("{")
        recentBody.reject(new Error("reader unavailable"))
      })
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("is StrictMode-safe for obsolete completion and rejection", async () => {
      const firstTemplates = deferred<Response>()
      const secondTemplates = deferred<Response>()
      const firstRecent = deferred<Response>()
      const secondRecent = deferred<Response>()
      let templateCall = 0
      let recentCall = 0
      apiRequestMock.mockImplementation((url: string) => {
        if (url === templateUrl()) return (++templateCall === 1 ? firstTemplates : secondTemplates).promise
        if (url === recentTasksUrl) return (++recentCall === 1 ? firstRecent : secondRecent).promise
        throw new Error(`Unexpected apiRequest: ${url}`)
      })
      render(<StrictMode><Home /></StrictMode>)
      await act(async () => {
        firstTemplates.resolve(jsonResponse([templateCard("obsolete")]))
        firstRecent.reject(new Error("obsolete recent"))
        secondTemplates.resolve(jsonResponse([templateCard("active")]))
        secondRecent.resolve(jsonResponse({ tasks: [recentTask(9)] }))
      })
      expect(await screen.findByText("Template active")).toBeInTheDocument()
      expect(screen.queryByText("Template obsolete")).not.toBeInTheDocument()
      expect(screen.getByText("Task 9")).toBeInTheDocument()
      expect(consoleErrorMock).not.toHaveBeenCalled()
    })

    it("keeps exact copied Home projections and ordered loader ownership fences in page source", () => {
      const source = readFileSync("src/app/page.tsx", "utf8")
      const templateDecoder = sourceSlice(
        source,
        "function decodeHomeTemplateCard(value: unknown)",
        "function decodeHomeTemplates(value: unknown)",
      )
      const recentDecoder = sourceSlice(
        source,
        "function decodeRecentTask(value: unknown)",
        "function decodeRecentTasks(value: unknown)",
      )
      const templateLoader = sourceSlice(
        source,
        "let active = true;",
        "    void fetchTemplates();",
      )
      const recentLoader = sourceSlice(
        source,
        "const fetchRecentTasks = async () => {",
        "    void fetchRecentTasks();",
      )

      expect(source).toMatch(/interface HomeTemplateCard[\s\S]*id: string[\s\S]*used_count: number/)
      expect(source).toMatch(/interface RecentTask[\s\S]*task_id: number[\s\S]*agent_logo_url\?: string \| null/)
      expect(source).not.toMatch(/templateGenerationRef/)
      expect(templateLoader).toContain(
        "const isCurrent = () => active;",
      )
      expect(templateLoader).toMatch(
        /const response = await apiRequest\(`\$\{getApiUrl\(\)\}\/api\/templates\/\?lang=\$\{locale\}`\);\s*if \(!isCurrent\(\)\) return;\s*if \(!response\.ok\)[\s\S]*?const parsed = await parseApiResponse\(response\);\s*if \(!isCurrent\(\)\) return;\s*const decoded = decodeHomeTemplates\(parsed\.data\);/,
      )
      expect(recentLoader).toMatch(
        /const response = await apiRequest\(`\$\{getApiUrl\(\)\}\/api\/chat\/tasks\?page=1&per_page=5`\);\s*if \(!active\) return;\s*if \(!response\.ok\)[\s\S]*?const parsed = await parseApiResponse\(response\);\s*if \(!active\) return;\s*const decoded = decodeRecentTasks\(parsed\.data\);/,
      )

      expect(templateDecoder.match(/return \{/g)).toHaveLength(1)
      expect(templateDecoder).toMatch(
        /connections\.push\(\{ name: connection\.name, logo: connection\.logo \}\);[\s\S]*?return \{\s*id: value\.id,\s*name: value\.name,\s*category: value\.category,\s*description: value\.description,\s*features: \[\.\.\.value\.features\],\s*connections,\s*setup_time: value\.setup_time,\s*likes: value\.likes,\s*used_count: value\.used_count,[\s\S]*?type: typeof value\.type === "string" \? value\.type : "agent",\s*\};/,
      )
      expect(recentDecoder.match(/return \{/g)).toHaveLength(1)
      expect(recentDecoder).toMatch(
        /return \{\s*task_id: value\.task_id,\s*title: value\.title,\s*created_at: value\.created_at,\s*agent_name: value\.agent_name,\s*agent_logo_url: value\.agent_logo_url,\s*\};/,
      )
      expect(templateDecoder).not.toMatch(/\.\.\.value\s*[,}]|\.\.\.connection\s*[,}]|Object\.assign/)
      expect(recentDecoder).not.toMatch(/\.\.\.value\s*[,}]|Object\.assign/)
      expect(`${templateDecoder}\n${recentDecoder}`).not.toMatch(
        /as HomeTemplateCard|as RecentTask|return value;/,
      )
      const recentRender = sourceSlice(
        source,
        "{recentTasks.map((task) => {",
        "              </div>\n            </>",
      )
      expect(recentRender).toContain("const resolvedLogoUrl = resolveAgentLogoUrl(task.agent_logo_url, getApiUrl());")
      expect(recentRender.match(/resolveAgentLogoUrl\(/g)).toHaveLength(1)
      expect(recentRender).toContain("const displayDate = formatDisplayDate(task.created_at, locale, {")
      expect(recentRender).toContain("{resolvedLogoUrl ? (")
      expect(recentRender.match(/\{displayDate \? ` • \$\{displayDate\}` : ""\}/g)).toHaveLength(1)
      expect(recentRender).not.toMatch(/startsWith\(["']http|new Date\(|toLocaleDateString|\$\{getApiUrl\(\)\}\$\{task\.agent_logo_url\}/)
      expect(source).not.toMatch(/Promise\.all\(\[\s*apiRequest\(`\$\{getApiUrl\(\)\}\/api\/templates/)
    })
  })
})
