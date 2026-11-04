import { describe, expect, it } from "vitest"

import { sanitizeGooglePickerDocuments } from "./google-picker"

describe("sanitizeGooglePickerDocuments", () => {
  it("keeps valid picker documents and numeric sizes", () => {
    expect(sanitizeGooglePickerDocuments([
      {
        id: "file-1",
        name: "Deck",
        mimeType: "application/vnd.google-apps.presentation",
        resourceKey: "resource-key",
        sizeBytes: 1024,
      },
    ])).toEqual([
      {
        id: "file-1",
        name: "Deck",
        mimeType: "application/vnd.google-apps.presentation",
        resourceKey: "resource-key",
        sizeBytes: 1024,
      },
    ])
  })

  it("drops malformed entries and unsafe field types", () => {
    expect(sanitizeGooglePickerDocuments([
      null,
      { name: "missing id" },
      { id: "", sizeBytes: "1024" },
      { id: "file-2", sizeBytes: "1024", resourceKey: 7 },
      { id: "file-3", sizeBytes: Number.NaN },
    ])).toEqual([
      {
        id: "file-2",
        name: undefined,
        mimeType: undefined,
        resourceKey: undefined,
        sizeBytes: undefined,
      },
      {
        id: "file-3",
        name: undefined,
        mimeType: undefined,
        resourceKey: undefined,
        sizeBytes: undefined,
      },
    ])
  })
})
