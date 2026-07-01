import { describe, expect, it, vi } from "vitest"

import {
  buildDeleteDocumentUrl,
  fetchCollectionDocumentStatuses,
  getCollectionDocuments,
  getDeleteErrorMessage,
  isTerminalDocumentStatus,
  mergeDocumentStatuses,
} from "./knowledge-base-detail-helpers"

function jsonResponse(body: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  } as unknown as Response
}

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

  it("classifies terminal vs non-terminal document statuses", () => {
    expect(isTerminalDocumentStatus("success")).toBe(true)
    expect(isTerminalDocumentStatus("failed")).toBe(true)
    expect(isTerminalDocumentStatus("cancelled")).toBe(true)
    expect(isTerminalDocumentStatus("running")).toBe(false)
    expect(isTerminalDocumentStatus("uploading")).toBe(false)
    expect(isTerminalDocumentStatus("pending")).toBe(false)
  })
})

describe("fetchCollectionDocumentStatuses", () => {
  it("normalizes backend rows and defaults unknown statuses to success", async () => {
    const requester = vi.fn().mockResolvedValue(
      jsonResponse({
        documents: [
          { filename: "a.pdf", file_id: "f1", doc_id: "d1", status: "running", can_delete: false },
          { filename: "b.pdf", doc_id: "d2", status: "weird-value" },
          { filename: "", status: "success" },
        ],
      })
    )

    const rows = await fetchCollectionDocumentStatuses("http://api.local", "demo", requester)
    expect(requester).toHaveBeenCalledWith("http://api.local/api/kb/collections/demo/documents")
    expect(rows).toEqual([
      { filename: "a.pdf", file_id: "f1", doc_id: "d1", status: "running", message: undefined, updated_at: undefined, can_delete: false },
      { filename: "b.pdf", file_id: undefined, doc_id: "d2", status: "success", message: undefined, updated_at: undefined, can_delete: true },
    ])
  })

  it("returns null on auth/not-found so callers can stop polling", async () => {
    for (const status of [401, 403, 404]) {
      const requester = vi.fn().mockResolvedValue(jsonResponse({}, status))
      expect(await fetchCollectionDocumentStatuses("http://api.local", "demo", requester)).toBeNull()
    }
  })

  it("throws on transient failures so callers can retain last rows", async () => {
    const requester = vi.fn().mockResolvedValue(jsonResponse({}, 500))
    await expect(
      fetchCollectionDocumentStatuses("http://api.local", "demo", requester)
    ).rejects.toThrow()
  })
})

describe("mergeDocumentStatuses", () => {
  it("prepends optimistic uploading rows for files without a backend row", () => {
    const merged = mergeDocumentStatuses(
      [{ filename: "done.pdf", status: "success", can_delete: true }],
      ["new.pdf", "done.pdf"]
    )
    expect(merged).toEqual([
      { filename: "new.pdf", status: "uploading", can_delete: false },
      { filename: "done.pdf", status: "success", can_delete: true },
    ])
  })

  it("does not duplicate an optimistic row when the backend row uses a nested path", () => {
    const merged = mergeDocumentStatuses(
      [{ filename: "user_1/demo/report.pdf", status: "running", can_delete: false }],
      ["report.pdf"]
    )
    expect(merged).toHaveLength(1)
    expect(merged[0].status).toBe("running")
  })
})
