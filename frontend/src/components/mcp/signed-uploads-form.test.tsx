import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { MCPServerFormData } from "./custom-api-form"
import {
  SignedUploadsForm,
  signedUploadRowsFromValue,
  signedUploadsFromRows,
} from "./signed-uploads-form"

vi.mock("@/contexts/i18n-context", () => ({
  useI18n: () => ({ t: (key: string) => key }),
}))

const KEY = "tools.mcp.signedUploads"

const REPLACEMENT = {
  tool_name: "attach_document",
  file_argument: "document_file_id",
  issuing_tool_name: "request_upload",
  allowed_upload_hosts: ["uploads.example.test"],
  upload_path_prefix: "/staged/",
  content_types: { ".pdf": "application/pdf" },
}
const ISSUER = {
  issuing_tool_name: "request_upload",
  allowed_upload_hosts: ["uploads.example.test"],
  upload_path_prefix: "/staged/",
  content_types: { ".pdf": "application/pdf" },
}
const LINK = { tool_name: "attach_picture", link_argument: "image_url" }

function counter() {
  let n = 0
  return () => `id-${n++}`
}

function without(entry: Record<string, unknown>, key: string) {
  const copy = { ...entry }
  delete copy[key]
  return copy
}

function modes(value: unknown) {
  return signedUploadRowsFromValue(value, counter()).map((row) => row.mode)
}

describe("signedUploadRowsFromValue", () => {
  it.each([
    ["replacement keys", REPLACEMENT, "replacement"],
    ["issuer keys", ISSUER, "issuer"],
    ["link keys", LINK, "link"],
    ["replacement missing content_types", without(REPLACEMENT, "content_types"), "replacement"],
    ["replacement missing the file argument", without(REPLACEMENT, "file_argument"), "replacement"],
    ["issuer keys plus tool_name", { ...ISSUER, tool_name: "attach_document" }, "replacement"],
    ["an empty object", {}, "replacement"],
    ["tool_name alone", { tool_name: "attach_picture" }, "replacement"],
    ["a string", "attach", "replacement"],
    ["null", null, "replacement"],
    ["a number", 3, "replacement"],
    ["an array", [LINK], "replacement"],
    ["link with an extra key", { ...LINK, extra: "x" }, "unrecognized"],
    ["an unknown key", { ...ISSUER, note: "x" }, "unrecognized"],
    ["issuer keys plus link_argument", { ...ISSUER, link_argument: "image_url" }, "unrecognized"],
    ["replacement keys plus link_argument", { ...REPLACEMENT, link_argument: "image_url" }, "unrecognized"],
    ["file_argument with link_argument", { file_argument: "f", link_argument: "u" }, "unrecognized"],
  ])("reads $0", (_name, entry, mode) => {
    expect(modes([entry])).toEqual([mode])
  })

  it("reads a non-object entry as an empty row that is dropped on save", () => {
    const rows = signedUploadRowsFromValue(["x", null, 3, []], counter())
    expect(rows.map((row) => row.toolName + row.fileArgument)).toEqual(["", "", "", ""])
    expect(signedUploadsFromRows(rows)).toEqual([])
  })

  it("reads a link row's fields and empties non-strings", () => {
    const [row] = signedUploadRowsFromValue([LINK], counter())
    expect(row.toolName).toBe("attach_picture")
    expect(row.linkArgument).toBe("image_url")
    const [bad] = signedUploadRowsFromValue(
      [{ tool_name: 3, link_argument: null }],
      counter(),
    )
    expect(bad.mode).toBe("link")
    expect(bad.toolName).toBe("")
    expect(bad.linkArgument).toBe("")
  })

  it("reads mistyped replacement fields as empty", () => {
    const [row] = signedUploadRowsFromValue(
      [{ ...REPLACEMENT, file_argument: 5, allowed_upload_hosts: "x", content_types: [] }],
      counter(),
    )
    expect(row.mode).toBe("replacement")
    expect(row.fileArgument).toBe("")
    expect(row.hostsText).toBe("")
    expect(row.contentTypes).toEqual([])
  })

  it("keeps a deep copy of an unrecognized entry", () => {
    const entry = { tool_name: "t", nested: { a: [1, 2] } }
    const [row] = signedUploadRowsFromValue([entry], counter())
    expect(row.mode).toBe("unrecognized")
    expect(row.raw).toEqual(entry)
    expect(row.raw).not.toBe(entry)
    expect((row.raw as typeof entry).nested).not.toBe(entry.nested)
  })

  it("reads a value that is not a list as no rows", () => {
    expect(signedUploadRowsFromValue(null, counter())).toEqual([])
    expect(signedUploadRowsFromValue({}, counter())).toEqual([])
  })
})

describe("signedUploadsFromRows", () => {
  it.each([
    ["all three forms", [REPLACEMENT, ISSUER, LINK]],
    ["link first", [LINK, REPLACEMENT]],
    [
      "unrecognized entries in between",
      [LINK, { tool_name: " spaced ", extra: true }, { note: [1, { a: 2 }] }, REPLACEMENT, ISSUER],
    ],
    ["an empty list", []],
  ])("round-trips %s exactly", (_name, value) => {
    const rows = signedUploadRowsFromValue(value, counter())
    expect(signedUploadsFromRows(rows)).toEqual(value)
  })

  it("trims a link entry's values", () => {
    const [row] = signedUploadRowsFromValue([LINK], counter())
    const out = signedUploadsFromRows([
      { ...row, toolName: "  attach_picture ", linkArgument: " image_url  " },
    ])
    expect(out).toEqual([LINK])
    expect(Object.keys(out[0])).toEqual(["tool_name", "link_argument"])
  })

  it("drops a link row with both values empty and keeps one with a single value", () => {
    const [row] = signedUploadRowsFromValue([LINK], counter())
    expect(signedUploadsFromRows([{ ...row, toolName: " ", linkArgument: "" }])).toEqual([])
    expect(signedUploadsFromRows([{ ...row, toolName: "t", linkArgument: "" }])).toEqual([
      { tool_name: "t", link_argument: "" },
    ])
  })

  it("writes an unrecognized entry untrimmed and in place", () => {
    const odd = { tool_name: " a ", z: 1 }
    const rows = signedUploadRowsFromValue([LINK, odd, ISSUER], counter())
    const out = signedUploadsFromRows(rows)
    expect(out[1]).toEqual(odd)
    expect(out).toHaveLength(3)
  })
})

function renderForm(signed_uploads: unknown) {
  const seen: MCPServerFormData[] = []
  const initial = {
    name: "records",
    transport: "streamable_http",
    description: "",
    config: { url: "https://mcp.example.com/mcp" },
    can_edit_global: true,
    signed_uploads,
    signed_uploads_error: null,
  } as MCPServerFormData
  function Harness() {
    const [formData, setFormData] = React.useState<MCPServerFormData>(initial)
    React.useEffect(() => {
      seen.push(formData)
    }, [formData])
    return <SignedUploadsForm formData={formData} setFormData={setFormData} />
  }
  render(<Harness />)
  return { latest: () => seen[seen.length - 1] }
}

describe("SignedUploadsForm link entries", () => {
  afterEach(() => cleanup())

  it("shows only the tool name and link argument for a link entry", () => {
    renderForm([LINK])
    expect(screen.getByLabelText(`${KEY}.toolName`)).toHaveValue("attach_picture")
    expect(screen.getByLabelText(`${KEY}.linkArgument`)).toHaveValue("image_url")
    expect(screen.queryByLabelText(`${KEY}.fileArgument`)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(`${KEY}.issuingToolName`)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(`${KEY}.allowedUploadHosts`)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(`${KEY}.uploadPathPrefix`)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(`${KEY}.extension`)).not.toBeInTheDocument()
    expect(screen.getByText(`${KEY}.modeLinkHint`)).toBeInTheDocument()
    expect(screen.getByText("attach_picture")).toBeInTheDocument()
  })

  it("edits a link entry into exactly its two keys", () => {
    const { latest } = renderForm([LINK])
    fireEvent.change(screen.getByLabelText(`${KEY}.linkArgument`), {
      target: { value: " picture_url " },
    })
    expect(latest().signed_uploads).toEqual([
      { tool_name: "attach_picture", link_argument: "picture_url" },
    ])
  })

  it("switches a replacement row to link and back, dropping the other mode's fields", () => {
    const { latest } = renderForm([REPLACEMENT])
    fireEvent.click(screen.getByRole("radio", { name: `${KEY}.modeLink` }))
    // The tool name stays; every field only the other modes have is gone.
    expect(latest().signed_uploads).toEqual([
      { tool_name: "attach_document", link_argument: "" },
    ])
    fireEvent.change(screen.getByLabelText(`${KEY}.linkArgument`), {
      target: { value: "image_url" },
    })
    expect(latest().signed_uploads).toEqual([
      { tool_name: "attach_document", link_argument: "image_url" },
    ])
    fireEvent.click(screen.getByRole("radio", { name: `${KEY}.modeReplacement` }))
    expect(screen.getByLabelText(`${KEY}.toolName`)).toHaveValue("attach_document")
    expect(screen.getByLabelText(`${KEY}.fileArgument`)).toHaveValue("")
    expect(screen.getByLabelText(`${KEY}.allowedUploadHosts`)).toHaveValue("")
    fireEvent.click(screen.getByRole("radio", { name: `${KEY}.modeLink` }))
    expect(screen.getByLabelText(`${KEY}.linkArgument`)).toHaveValue("")
  })

  it("shows an unrecognized entry read-only and writes it back unchanged", () => {
    const odd = { tool_name: "attach", link_argument: "u", extra: 1 }
    const { latest } = renderForm([LINK, odd])
    expect(screen.getByText(`${KEY}.unrecognizedEntry`)).toBeInTheDocument()
    const shown = screen.getByLabelText(`${KEY}.unrecognizedEntryValue`)
    expect(shown).toHaveAttribute("readonly")
    expect((shown as HTMLTextAreaElement).value).toBe(JSON.stringify(odd, null, 2))
    // Editing the other row keeps the unrecognized one as stored.
    fireEvent.change(screen.getByLabelText(`${KEY}.linkArgument`), {
      target: { value: "picture_url" },
    })
    expect(latest().signed_uploads).toEqual([
      { tool_name: "attach_picture", link_argument: "picture_url" },
      odd,
    ])
  })

  it("removes an unrecognized entry", () => {
    const { latest } = renderForm([{ nope: 1 }])
    fireEvent.click(screen.getByRole("button", { name: `${KEY}.removeTool` }))
    expect(latest().signed_uploads).toEqual([])
  })
})
