import { describe, expect, it } from "vitest"

import {
  buildKnowledgeBaseErrorResult,
  getKnowledgeBaseErrorToastContent,
  normalizeKnowledgeBaseIngestionResult,
} from "./kb-ingest-feedback"

const COPY = {
  genericTitle: "上传失败",
  embeddingTitle: "知识库导入失败：未配置可用的嵌入模型",
  embeddingDescription: "请先配置默认嵌入模型，或选择一个可用的嵌入模型后重试。",
  rollbackTitle: "知识库导入失败，清理未完全完成",
  rollbackDescription: "系统已尝试回滚本次导入，请查看处理结果中的详细错误。",
}

describe("getKnowledgeBaseErrorToastContent", () => {
  it("maps embedding configuration errors to a concise actionable toast", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Model 'text-embedding-v4' not found in hub and no environment configuration available for embedding.",
      COPY
    )

    expect(result).toEqual({
      title: COPY.embeddingTitle,
      description: COPY.embeddingDescription,
    })
  })

  it("keeps rollback toasts concise while preserving rollback context", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "Failed to fully roll back ingest for demo/file.txt: delete failed. Original ingestion error: Model 'text-embedding-v4' not found in hub and no environment configuration available for embedding.",
      COPY
    )

    expect(result).toEqual({
      title: COPY.embeddingTitle,
      description: `${COPY.embeddingDescription} ${COPY.rollbackDescription}`,
    })
  })

  it("falls back to a generic title with a truncated description for unknown errors", () => {
    const result = getKnowledgeBaseErrorToastContent(
      "A very long upload failure happened while processing the document and there are many more technical details that should not become the toast title for end users.",
      COPY
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
