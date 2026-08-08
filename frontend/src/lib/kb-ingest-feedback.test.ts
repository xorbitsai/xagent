import { describe, expect, it } from "vitest"

import {
  buildKnowledgeBaseErrorResult,
  getKnowledgeBaseErrorToastContent,
  normalizeKnowledgeBaseIngestionResult,
} from "./kb-ingest-feedback"

const COPY = {
  genericTitle: "上传失败",
  nameUnavailableTitle: "该知识库名称不可用",
  nameUnavailableDescription: "请换一个知识库名称后重试。",
  conflictTitle: "该知识库的状态已变化",
  conflictDescription: "请刷新页面后重试；若仍然失败，该名称可能已归他人所有。",
  embeddingTitle: "知识库导入失败：未配置可用的嵌入模型",
  embeddingDescription: "请先配置默认嵌入模型，或选择一个可用的嵌入模型后重试。",
  rollbackTitle: "知识库导入失败，清理未完全完成",
  rollbackDescription: "系统已尝试回滚本次导入，请查看处理结果中的详细错误。",
}

describe("getKnowledgeBaseErrorToastContent", () => {
  it("maps embedding configuration errors to a concise actionable toast", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Model 'text-embedding-v4' not found in hub and no environment configuration available for embedding.",
      COPY,
      { nameEntry: "user-entered" }
    )

    expect(result).toEqual({
      title: COPY.embeddingTitle,
      description: COPY.embeddingDescription,
    })
  })

  it("maps a name-taken 409 to actionable copy without leaking the raw detail", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Knowledge base name unavailable: test. Please choose a different name.",
      COPY,
      { status: 409, nameEntry: "user-entered" }
    )

    expect(result).toEqual({
      title: COPY.nameUnavailableTitle,
      description: COPY.nameUnavailableDescription,
    })
  })

  it("classifies a 409 by status alone, whatever wording the backend sends", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "some entirely different sentence the backend may switch to",
      COPY,
      { status: 409, nameEntry: "user-entered" }
    )

    expect(result).toEqual({
      title: COPY.nameUnavailableTitle,
      description: COPY.nameUnavailableDescription,
    })
  })

  it("does not treat the conflict wording as a conflict without a 409", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Knowledge base name unavailable: test. Please choose a different name.",
      COPY,
      { status: 500, nameEntry: "user-entered" }
    )

    expect(result.title).toBe(COPY.genericTitle)
  })

  it("never puts the detail of an ambiguous 409 in front of the user", () => {
    // Rename answers 409 for lock contention and for storage collisions whose
    // detail names an internal user id, so "opaque" must yield neutral copy.
    const leaky =
      "Failed to rename collection: target physical directory already exists "
      + "for user_3. A collection named 'x' already has physical files."
    const result = getKnowledgeBaseErrorToastContent(leaky, COPY, {
      status: 409,
      nameEntry: "none",
    })

    expect(result).toEqual({
      title: COPY.conflictTitle,
      description: COPY.conflictDescription,
    })
    expect(JSON.stringify(result)).not.toContain("user_3")
    expect(JSON.stringify(result)).not.toContain("physical")
  })

  it("does not offer the rename advice when the caller cannot confirm a name clash", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Knowledge base name unavailable: demo. Please choose a different name.",
      COPY,
      { status: 409, nameEntry: "none" }
    )

    expect(result.title).toBe(COPY.conflictTitle)
  })

  it("keeps rollback toasts concise while preserving rollback context", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Failed to fully roll back ingest for demo/file.txt: delete failed. Original ingestion error: Model 'text-embedding-v4' not found in hub and no environment configuration available for embedding.",
      COPY,
      { nameEntry: "user-entered" }
    )

    expect(result).toEqual({
      title: COPY.embeddingTitle,
      description: `${COPY.embeddingDescription} ${COPY.rollbackDescription}`,
    })
  })

  it("falls back to a generic title with a truncated description for unknown errors", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "A very long upload failure happened while processing the document and there are many more technical details that should not become the toast title for end users.",
      COPY,
      { nameEntry: "user-entered" }
    )

    expect(result.title).toBe(COPY.genericTitle)
    expect(result.description).toContain("A very long upload failure happened")
  })
})

describe("buildKnowledgeBaseErrorResult", () => {
  it("builds a synthetic error result for processing details panels", () => {
    expect(
      buildKnowledgeBaseErrorResult(
        "demo",
        "Failed to upload file",
        "resolve_embedding_adapter",
        "failed.xlsx"
      )
    ).toEqual({
      collection: "demo",
      document_count: 0,
      chunks_count: 0,
      status: "error",
      message: "Failed to upload file",
      failed_step: "resolve_embedding_adapter",
      file_name: "failed.xlsx",
    })
  })
})

describe("normalizeKnowledgeBaseIngestionResult", () => {
  it("maps backend ingest fields into the display shape used by result panels", () => {
    expect(
      normalizeKnowledgeBaseIngestionResult(
        {
          status: "success",
          message: "done",
          doc_id: "doc-123",
          chunk_count: 8,
          embedding_count: 8,
          vector_count: 8,
        },
        {
          collection: "demo",
          fileName: "report.pdf",
        }
      )
    ).toEqual({
      collection: "demo",
      document_count: 1,
      chunks_count: 8,
      status: "success",
      message: "done",
      failed_step: undefined,
      file_name: "report.pdf",
      parses_completed: 1,
      doc_id: "doc-123",
      embedding_count: 8,
      vector_count: 8,
      embeddings_created: undefined,
      error: undefined,
    })
  })

  it("falls back to embedding-derived counts when vector_count is absent", () => {
    expect(
      normalizeKnowledgeBaseIngestionResult(
        {
          status: "success",
          message: "done",
          doc_id: "doc-456",
          chunk_count: 3,
          embeddings_created: 5,
        },
        {
          collection: "demo",
          fileName: "report.pdf",
        }
      )
    ).toMatchObject({
      collection: "demo",
      document_count: 1,
      chunks_count: 3,
      parses_completed: 1,
      file_name: "report.pdf",
      vector_count: 5,
    })
  })
})
