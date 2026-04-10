import { describe, expect, it } from "vitest"

import { buildDeleteDocumentUrl, getCollectionDocuments, getDeleteErrorMessage } from "./knowledge-base-detail-helpers"

describe("knowledge-base-detail helpers", () => {
  it("prefers richer document metadata over legacy names", () => {
    expect(
      getCollectionDocuments({
        document_names: ["report.pdf"],
        document_metadata: [{ filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" }],
      })
    ).toEqual([{ filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" }])
  })

  it("falls back to legacy document_names when metadata is absent", () => {
    expect(
      getCollectionDocuments({
        document_names: ["legacy.txt", "nested/path.md"],
      })
    ).toEqual([{ filename: "legacy.txt" }, { filename: "nested/path.md" }])
  })

  it("preserves legacy document_names when metadata is partial", () => {
    expect(
      getCollectionDocuments({
        document_names: ["report.pdf", "legacy.txt", "extra.md"],
        document_metadata: [{ filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" }],
      })
    ).toEqual([
      { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
      { filename: "legacy.txt" },
      { filename: "extra.md" },
    ])
  })

  it("keeps same-filename metadata entries when their identifiers differ", () => {
    expect(
      getCollectionDocuments({
        document_names: ["report.pdf"],
        document_metadata: [
          { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
          { filename: " report.pdf ", file_id: "file-456", doc_id: "doc-123" },
          { filename: "report.pdf", file_id: "file-123", doc_id: "doc-789" },
        ],
      })
    ).toEqual([
      { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
      { filename: "report.pdf", file_id: "file-456", doc_id: "doc-123" },
      { filename: "report.pdf", file_id: "file-123", doc_id: "doc-789" },
    ])
  })

  it("dedupes exact metadata duplicates before considering legacy names", () => {
    expect(
      getCollectionDocuments({
        document_names: ["report.pdf", "legacy.txt", "legacy.txt"],
        document_metadata: [
          { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
          { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
        ],
      })
    ).toEqual([
      { filename: "report.pdf", file_id: "file-123", doc_id: "doc-123" },
      { filename: "legacy.txt" },
    ])
  })

  it("builds delete urls with file_id first, then doc_id, then filename only", () => {
    expect(
      buildDeleteDocumentUrl("http://api.local", "demo", {
        filename: "report.pdf",
        file_id: "file-123",
        doc_id: "doc-123",
      })
    ).toBe("http://api.local/api/kb/collections/demo/documents/report.pdf?file_id=file-123")

    expect(
      buildDeleteDocumentUrl("http://api.local", "demo", {
        filename: "page.html",
        doc_id: "doc-9",
      })
    ).toBe("http://api.local/api/kb/collections/demo/documents/page.html?doc_id=doc-9")

    expect(
      buildDeleteDocumentUrl("http://api.local", "demo", {
        filename: "legacy.txt",
      })
    ).toBe("http://api.local/api/kb/collections/demo/documents/legacy.txt")
  })

  it("extracts the most useful delete error message defensively", () => {
    expect(getDeleteErrorMessage({ detail: "ambiguous" }, "fallback")).toBe("ambiguous")
    expect(getDeleteErrorMessage({ message: "failed" }, "fallback")).toBe("failed")
    expect(getDeleteErrorMessage({ errors: ["first error"] }, "fallback")).toBe("first error")
    expect(getDeleteErrorMessage(null, "fallback")).toBe("fallback")
  })
})
