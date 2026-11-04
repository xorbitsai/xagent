import { describe, expect, it } from "vitest"

import { normalizeUploadFileIds } from "./upload-file-ids"

describe("normalizeUploadFileIds", () => {
  it("returns trimmed unique identifiers when every upload result is valid", () => {
    expect(normalizeUploadFileIds([" file-1 ", "file-2"], 2)).toEqual([
      "file-1",
      "file-2",
    ])
  })

  it.each([
    { ids: [""], count: 1 },
    { ids: ["   "], count: 1 },
    { ids: ["file-1", " file-1 "], count: 2 },
    { ids: ["file-1"], count: 2 },
    { ids: ["file-1", null], count: 2 },
  ])("rejects malformed upload identifiers: $ids", ({ ids, count }) => {
    expect(normalizeUploadFileIds(ids, count)).toBeNull()
  })
})
