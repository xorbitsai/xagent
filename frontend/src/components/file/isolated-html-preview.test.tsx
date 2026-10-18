/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { resolveHtmlPreviewRunnerUrl } from "../../../html-preview-config.mjs"
import { IsolatedHtmlPreview } from "./isolated-html-preview"

const env = "NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN"
const ready = { type: "xagent-html-preview-ready", version: 1 }
const dom = (globalThis as unknown as { jsdom: { reconfigure: (options: { url: string }) => void } }).jsdom

function send(source: MessageEventSource | null, data: unknown = ready, origin = "null") {
  act(() => { window.dispatchEvent(new MessageEvent("message", { source, data, origin })) })
}

describe("resolveHtmlPreviewRunnerUrl", () => {
  it.each([
    ["https://cloud.xagent.co", "https://preview.xagent.co/render"],
    ["https://stage.cloud.xagent.co", "https://preview-stage.xagent.co/render"],
    ["https://dev.cloud.xagent.co", "https://preview-dev.xagent.co/render"],
  ])("maps %s to %s", (appOrigin, expected) => {
    expect(resolveHtmlPreviewRunnerUrl("auto", appOrigin)).toBe(expected)
  })

  it.each([undefined, "", "auto", "garbage", "http://preview.example.test",
    "https://user:pass@preview.example.test", "javascript:alert(1)"])("rejects %s without a usable browser origin", value => {
    expect(resolveHtmlPreviewRunnerUrl(value)).toBeNull()
  })

  it("falls back for unknown or malformed auto origins", () => {
    expect(resolveHtmlPreviewRunnerUrl("auto", "https://self-hosted.example.test")).toBeNull()
    expect(resolveHtmlPreviewRunnerUrl("auto", "invalid")).toBeNull()
  })

  it.each([
    [" https://preview.example.test/path?ignored=1#fragment ", "https://preview.example.test/render"],
    ["https://preview.example.test/", "https://preview.example.test/render"],
    ["http://localhost:8787", "http://localhost:8787/render"],
    ["http://127.0.0.1:8787/", "http://127.0.0.1:8787/render"],
    ["http://[::1]:8787/path", "http://[::1]:8787/render"],
  ])("canonicalizes %s", (configured, expected) => {
    expect(resolveHtmlPreviewRunnerUrl(configured)).toBe(expected)
  })
})

describe("IsolatedHtmlPreview", () => {
  beforeEach(() => {
    vi.stubEnv(env, "")
    vi.useFakeTimers()
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    vi.unstubAllEnvs()
    vi.useRealTimers()
    dom.reconfigure({ url: "http://localhost:3000" })
  })

  it.each(["", "invalid", "http://untrusted.example.test"])("keeps the opaque srcdoc fallback for %s", value => {
    vi.stubEnv(env, value)
    render(<IsolatedHtmlPreview html="<script>window.preview = true</script>" title="demo" />)
    expect(screen.getByTitle("demo")).toHaveAttribute("srcdoc", "<script>window.preview = true</script>")
    expect(screen.getByTitle("demo")).toHaveAttribute("sandbox", "allow-scripts")
    expect(screen.getByTitle("demo")).not.toHaveAttribute("src")
  })

  it("does not navigate for empty HTML", () => {
    vi.stubEnv(env, "https://preview.example.test")
    render(<IsolatedHtmlPreview html="" title="empty" />)
    expect(screen.queryByTitle("empty")).toBeNull()
    expect(vi.getTimerCount()).toBe(0)
  })

  it.each([
    ["https://cloud.xagent.co", "https://preview.xagent.co/render"],
    ["https://stage.cloud.xagent.co", "https://preview-stage.xagent.co/render"],
    ["https://dev.cloud.xagent.co", "https://preview-dev.xagent.co/render"],
  ])("resolves auto on %s in the component", (origin, expected) => {
    dom.reconfigure({ url: origin })
    vi.stubEnv(env, "auto")
    render(<IsolatedHtmlPreview html="<p>preview</p>" title="auto" />)
    expect(screen.getByTitle("auto")).toHaveAttribute("src", expected)
  })

  it("delivers once only for a valid v1 ready from the exact runner window", () => {
    vi.stubEnv(env, "https://preview.example.test")
    const html = '<script src="https://cdn.example.test/app.js"></script>'
    render(<IsolatedHtmlPreview html={html} title="remote" />)
    const frame = screen.getByTitle("remote") as HTMLIFrameElement
    expect(frame).toHaveAttribute("src", "https://preview.example.test/render")
    expect(frame).not.toHaveAttribute("srcdoc")
    expect(frame).toHaveAttribute("sandbox", "allow-scripts")
    expect(frame).toHaveAttribute("referrerpolicy", "no-referrer")
    expect(frame).toHaveAttribute("aria-busy", "true")
    const post = vi.spyOn(frame.contentWindow!, "postMessage")
    send(window)
    send(frame.contentWindow, ready, "https://preview.example.test")
    for (const data of [null, "ready", {}, { type: "wrong", version: 1 },
      { type: ready.type, version: 2 }, { type: ready.type }]) send(frame.contentWindow, data)
    expect(post).not.toHaveBeenCalled()
    send(frame.contentWindow)
    expect(post).toHaveBeenCalledOnce()
    expect(post).toHaveBeenCalledWith({ type: "xagent-html-preview-render", version: 1, html }, "*")
    send(frame.contentWindow)
    expect(post).toHaveBeenCalledOnce()
    expect(frame).toHaveAttribute("aria-busy", "false")
    act(() => vi.advanceTimersByTime(10_000))
    expect(screen.getByTitle("remote")).toBe(frame)
    expect(vi.getTimerCount()).toBe(0)
  })

  it("falls back safely when the runner does not become ready", () => {
    vi.stubEnv(env, "https://preview.example.test")
    render(<IsolatedHtmlPreview html="<p>fallback</p>" title="fallback" />)
    const oldFrame = screen.getByTitle("fallback")
    const oldWindow = (oldFrame as HTMLIFrameElement).contentWindow!
    const oldPost = vi.spyOn(oldWindow, "postMessage")
    act(() => vi.advanceTimersByTime(8_000))
    const frame = screen.getByTitle("fallback")
    expect(frame).not.toBe(oldFrame)
    expect(frame).toHaveAttribute("srcdoc", "<p>fallback</p>")
    expect(frame).toHaveAttribute("sandbox", "allow-scripts")
    expect(frame).not.toHaveAttribute("src")
    expect(vi.getTimerCount()).toBe(0)
    send(oldWindow)
    expect(oldPost).not.toHaveBeenCalled()
  })

  it("uses srcdoc for auto mode on an unknown app host", () => {
    dom.reconfigure({ url: "https://customer.example.test" })
    vi.stubEnv(env, "auto")
    render(<IsolatedHtmlPreview html="<p>fallback</p>" title="unknown" />)
    expect(screen.getByTitle("unknown")).toHaveAttribute("srcdoc", "<p>fallback</p>")
  })

  it("uses a fresh browsing context and never delivers new HTML to an outgoing document", () => {
    vi.stubEnv(env, "https://preview.example.test")
    const { rerender } = render(<IsolatedHtmlPreview html="<p>old</p>" title="switch" />)
    const oldFrame = screen.getByTitle("switch") as HTMLIFrameElement
    const oldWindow = oldFrame.contentWindow!
    const oldPost = vi.spyOn(oldWindow, "postMessage")
    rerender(<IsolatedHtmlPreview html="<p>new</p>" title="switch" />)
    const frame = screen.getByTitle("switch") as HTMLIFrameElement
    expect(frame).not.toBe(oldFrame)
    const post = vi.spyOn(frame.contentWindow!, "postMessage")
    send(oldWindow)
    expect(oldPost).not.toHaveBeenCalled()
    expect(post).not.toHaveBeenCalled()
    send(frame.contentWindow)
    expect(post).toHaveBeenCalledOnce()
    expect(post).toHaveBeenCalledWith({
      type: "xagent-html-preview-render", version: 1, html: "<p>new</p>",
    }, "*")
  })

  it("cleans up its listener and timeout on unmount", () => {
    vi.stubEnv(env, "https://preview.example.test")
    const add = vi.spyOn(window, "addEventListener")
    const remove = vi.spyOn(window, "removeEventListener")
    const { unmount } = render(<IsolatedHtmlPreview html="<p>preview</p>" title="cleanup" />)
    const frameWindow = (screen.getByTitle("cleanup") as HTMLIFrameElement).contentWindow!
    const post = vi.spyOn(frameWindow, "postMessage")
    const listener = add.mock.calls.find(([type]) => type === "message")![1]
    unmount()
    expect(remove).toHaveBeenCalledWith("message", listener)
    expect(vi.getTimerCount()).toBe(0)
    send(frameWindow)
    expect(post).not.toHaveBeenCalled()
  })
})
