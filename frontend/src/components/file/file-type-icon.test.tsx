import React from "react"
import { cleanup, render } from "@testing-library/react"
import { afterEach, describe, expect, it } from "vitest"

import { FileTypeIcon, getFileVisualKind } from "./file-type-icon"

afterEach(cleanup)

describe("getFileVisualKind", () => {
  it("prefers MIME types when they are available", () => {
    expect(getFileVisualKind("opaque-name", "image/jpeg")).toBe("image")
    expect(getFileVisualKind("opaque-name", "audio/mpeg")).toBe("audio")
  })

  it("checks specific OpenXML MIME types before generic documents", () => {
    expect(
      getFileVisualKind(
        "opaque-name",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
      )
    ).toBe("spreadsheet")
    expect(
      getFileVisualKind(
        "opaque-name",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation"
      )
    ).toBe("presentation")
  })

  it("falls back to filename extensions", () => {
    expect(getFileVisualKind("report.xlsx")).toBe("spreadsheet")
    expect(getFileVisualKind("archive.tar.gz")).toBe("archive")
    expect(getFileVisualKind("script.py")).toBe("code")
  })

  it("keeps unknown files generic", () => {
    expect(getFileVisualKind("README")).toBe("file")
  })
})

describe("FileTypeIcon", () => {
  it("renders the matching Lucide icon", () => {
    const { container } = render(
      <FileTypeIcon filename="photo.jpg" mimeType="image/jpeg" />
    )

    expect(container.querySelector('[data-file-kind="image"]')).toBeInTheDocument()
  })
})
