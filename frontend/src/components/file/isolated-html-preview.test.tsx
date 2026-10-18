/// <reference types="@testing-library/jest-dom/vitest" />
import React from "react"
import { act, cleanup, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import {
  IsolatedHtmlPreview,
  resolveHtmlPreviewRunnerUrl,
} from "./isolated-html-preview"

describe("resolveHtmlPreviewRunnerUrl", () => {
  it.each([
    ["https://cloud.xagent.co", "https://preview.xagent.co/render"],
    ["https://stage.cloud.xagent.co", "https://preview.stage.xagent.co/render"],
    ["https://dev.cloud.xagent.co", "https://preview.dev.xagent.co/render"],
  ])("maps the SaaS app origin %s to %s", (appOrigin, expected) => {
    expect(resolveHtmlPreviewRunnerUrl("auto", appOrigin)).toBe(expected)
  })

  it("fails closed to srcDoc for unknown auto origins and unsafe explicit origins", () => {
    expect(resolveHtmlPreviewRunnerUrl("auto", "https://self-hosted.example.test")).toBeNull()
    expect(resolveHtmlPreviewRunnerUrl("http://preview.example.test")).toBeNull()
    expect(resolveHtmlPreviewRunnerUrl("https://user:pass@preview.example.test")).toBeNull()
  })

  it("allows HTTPS origins and loopback HTTP for local development", () => {
    expect(resolveHtmlPreviewRunnerUrl("https://preview.example.test/path?ignored=1"))
      .toBe("https://preview.example.test/render")
    expect(resolveHtmlPreviewRunnerUrl("http://localhost:8787"))
      .toBe("http://localhost:8787/render")
  })
})

describe("IsolatedHtmlPreview", () => {
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("keeps a scripts-only opaque srcDoc fallback without a configured runner", () => {
    render(<IsolatedHtmlPreview html="<script>window.preview = true</script>" title="demo" />)

    const iframe = screen.getByTitle("demo")
    expect(iframe).toHaveAttribute("srcdoc", "<script>window.preview = true</script>")
    expect(iframe).toHaveAttribute("sandbox", "allow-scripts")
    expect(iframe).not.toHaveAttribute("sandbox", expect.stringContaining("allow-same-origin"))
  })

  it("delivers HTML once to the exact isolated runner window", () => {
    render(
      <IsolatedHtmlPreview
        html={'<script src="https://cdn.example.test/app.js"></script>'}
        title="remote script"
        previewOrigin="https://preview.example.test"
      />,
    )

    const iframe = screen.getByTitle("remote script") as HTMLIFrameElement
    expect(iframe).not.toHaveAttribute("srcdoc")
    expect(iframe.src).toMatch(/^https:\/\/preview\.example\.test\/render\?v=[0-9a-f]+$/)
    const postMessage = vi.spyOn(iframe.contentWindow!, "postMessage")

    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        source: window,
        data: { type: "xagent-html-preview-ready", version: 1 },
      }))
    })
    expect(postMessage).not.toHaveBeenCalled()

    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        source: iframe.contentWindow,
        data: { type: "xagent-html-preview-ready", version: 1 },
      }))
    })
    expect(postMessage).toHaveBeenCalledOnce()
    expect(postMessage).toHaveBeenCalledWith({
      type: "xagent-html-preview-render",
      version: 1,
      html: '<script src="https://cdn.example.test/app.js"></script>',
    }, "*")

    act(() => {
      window.dispatchEvent(new MessageEvent("message", {
        source: iframe.contentWindow,
        data: { type: "xagent-html-preview-ready", version: 1 },
      }))
    })
    expect(postMessage).toHaveBeenCalledOnce()
  })
})
