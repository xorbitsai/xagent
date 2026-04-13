import { describe, expect, it } from "vitest"

import { normalizeIngestionConfigForFilename } from "./ingestion-form"

describe("normalizeIngestionConfigForFilename", () => {
  it("falls back to default parser for non-PDF files when a PDF-only parser is selected", () => {
    const config = {
      parse_method: "pypdf",
      chunk_strategy: "recursive",
      chunk_size: 1000,
      chunk_overlap: 200,
      separators: "",
      embedding_model_id: "embed-1",
      embedding_batch_size: 10,
      max_retries: 3,
      retry_delay: 1,
    }

    expect(normalizeIngestionConfigForFilename(config, "sheet.xlsx").parse_method).toBe("default")
    expect(normalizeIngestionConfigForFilename(config, "notes.csv").parse_method).toBe("default")
  })

  it("preserves the selected parser for PDF files", () => {
    const config = {
      parse_method: "pypdf",
      chunk_strategy: "recursive",
      chunk_size: 1000,
      chunk_overlap: 200,
      separators: "",
      embedding_model_id: "embed-1",
      embedding_batch_size: 10,
      max_retries: 3,
      retry_delay: 1,
    }

    expect(normalizeIngestionConfigForFilename(config, "manual.pdf").parse_method).toBe("pypdf")
  })
})
