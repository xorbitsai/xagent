import { describe, expect, it } from "vitest"

import {
  isManagedFileUrl,
  projectFilesDisabledPresentation,
  sanitizeFilesDisabledPresentationText,
  serializeFilesDisabledPresentation,
} from "./files-disabled-presentation"

describe("files-disabled presentation", () => {
  it("recognizes managed preview and download URLs without classifying unrelated URLs", () => {
    expect(isManagedFileUrl("/api/files/public/preview/file-id")).toBe(true)
    expect(isManagedFileUrl("https://app.example/api/files/download/file-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/public/%70review/file-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/%64ownload/file-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/preview%2Ffile-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/%2570review/file-id")).toBe(true)
    expect(isManagedFileUrl("/%61pi/files/public/preview/file-id")).toBe(true)
    expect(isManagedFileUrl("/api/%66iles/public/preview/file-id")).toBe(true)
    expect(isManagedFileUrl("/api/files%2Fdownload/file-id")).toBe(true)
    expect(isManagedFileUrl("/%2561pi/files/public/preview/file-id")).toBe(true)
    expect(isManagedFileUrl("/%2Fapi%2Ffiles%2Fpublic%2Fpreview%2Ffile-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/preview-pdf/presentation-id")).toBe(true)
    expect(isManagedFileUrl("https://app.example/api/files/preview-pdf/presentation-id")).toBe(true)
    expect(isManagedFileUrl("/api/files/%E0%A4%A")).toBe(true)
    expect(isManagedFileUrl("/%61pi/files/public/%70review/%ZZ/secret")).toBe(true)
    expect(isManagedFileUrl("/api/files/metadata/file-id")).toBe(false)
    expect(isManagedFileUrl("https://example.com/guide/downloads")).toBe(false)
    expect(isManagedFileUrl("https://example.com/%70review/file-id")).toBe(false)
  })

  it("only treats deliberate local-path grammar as a file reference", () => {
    expect(sanitizeFilesDisabledPresentationText("Keep `and/or` unchanged.")).toBe(
      "Keep `and/or` unchanged.",
    )
    expect(sanitizeFilesDisabledPresentationText("Use `/private/reports/secret.txt`.")).toBe(
      "Use secret.txt.",
    )
    expect(sanitizeFilesDisabledPresentationText("Use `C:\\private\\secret.txt`.")).toBe(
      "Use secret.txt.",
    )
  })

  it("redacts recognized unquoted local paths without touching URLs or ordinary slash text", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Saved /private/reports/secret.txt beside ~/notes.txt, ./draft.md, ../parent.txt, C:\\private\\secret.txt, and \\\\server\\share\\secret.txt; keep https://example.com/a/b and and/or.",
    )).toBe(
      "Saved secret.txt beside notes.txt, draft.md, parent.txt, secret.txt, and secret.txt; keep https://example.com/a/b and and/or.",
    )
  })

  it("redacts recognized local paths after punctuation and assignment boundaries", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Open (/private/tenant/secret.txt), path=/private/tenant/config.json, workspace/draft.md, artifacts/chart.png, output/report.pdf, and uploads/input.csv; keep https://example.com/a/b and `and/or`.",
    )).toBe(
      "Open (secret.txt), path=config.json, draft.md, chart.png, report.pdf, and input.csv; keep https://example.com/a/b and `and/or`.",
    )
  })

  it("redacts local paths after shell redirects and prose delimiters", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Write >/private/out.txt, 2>/private/error.txt, </private/input.txt, |/private/pipe.txt, :/private/colon.txt, &/private/and.txt, and !/private/bang.txt.",
    )).toBe(
      "Write >out.txt, 2>error.txt, <input.txt, |pipe.txt, :colon.txt, &and.txt, and !bang.txt.",
    )
  })

  it("redacts local paths after compact prose labels without changing URLs or ordinary slash text", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Saved to:/private/tenant/reported.txt; Label:/private/tenant/one.txt and Label: /private/tenant/two.txt; keep custom+tool:/private/tenant/custom.txt, http:/private/tenant/http.txt, and https:/private/tenant/https.txt. Keep http://example.com/a/b, https://example.com/a/b, 1/2, and/or, and prose/path.",
    )).toBe(
      "Saved to:reported.txt; Label:one.txt and Label: two.txt; keep custom+tool:custom.txt, http:http.txt, and https:https.txt. Keep http://example.com/a/b, https://example.com/a/b, 1/2, and/or, and prose/path.",
    )
  })

  it("preserves ordinary colon ratios", () => {
    expect(sanitizeFilesDisabledPresentationText("Render at 16:9 and compare 16/9 with 1/2.")).toBe(
      "Render at 16:9 and compare 16/9 with 1/2.",
    )
  })

  it("preserves business routes while redacting local roots and file-like absolute paths", () => {
    expect(sanitizeFilesDisabledPresentationText([
      "Keep /v1/shifts, /v1/openapi.json, /care/shift/42, and /care/export.csv.",
      "Hide /private/a.txt, /tmp/b.txt, /workspace/c.txt, /sandbox/d.txt, /home/user/e.txt,",
      "/Users/alex/f.txt, /mnt/share/g.txt, /root/h.txt, /raw/i.txt, /data/j.txt,",
      "/app/src, /opt/xagent, /var/tmp, /srv/app, /etc/passwd, and /srv/reports/final.pdf.",
    ].join(" "))).toBe([
      "Keep /v1/shifts, /v1/openapi.json, /care/shift/42, and /care/export.csv.",
      "Hide a.txt, b.txt, c.txt, d.txt, e.txt,",
      "f.txt, g.txt, h.txt, i.txt, j.txt,",
      "src, xagent, tmp, app, passwd, and final.pdf.",
    ].join(" "))
  })

  it("sanitizes Markdown labels and titles independently from safe external targets", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "[/private/labels/report.txt](https://example.com/report \"/tmp/titles/report.txt\") and ![/workspace/images/chart.png](https://example.com/chart.png '/sandbox/titles/chart.png')",
    )).toBe(
      "[report.txt](https://example.com/report \"report.txt\") and ![chart.png](https://example.com/chart.png \"chart.png\")",
    )
  })

  it("projects every Markdown file-reference form through the same parser grammar", () => {
    expect(sanitizeFilesDisabledPresentationText([
      "[safe][artifact]",
      "",
      "[artifact]: file:tenant-reference-secret",
      "",
      "[nested **safe** label](file:tenant-nested-secret)",
      "",
      "[balanced](file:tenant(private)/balanced-secret)",
      "",
      "Bare file:tenant-bare(secret)/secret-id, file:tenant-bare-secret, and <file:tenant-autolink-secret>.",
    ].join("\n"))).toBe([
      "safe",
      "",
      "",
      "",
      "nested safe label",
      "",
      "balanced",
      "",
      "Bare file, file, and file.",
    ].join("\n"))
  })

  it("falls back to the link title, not the literal word file, when the label is empty", () => {
    // reconcile_assistant_file_references (file_reference_output_service.py)
    // emits media references with an empty label and the real filename in
    // the title -- e.g. [](file:id "generated_video.mp4") -- so the title
    // must still surface here even though there's no label/alt text at all.
    expect(sanitizeFilesDisabledPresentationText(
      '[](file:tenant-secret "generated_video.mp4")',
    )).toBe("generated_video.mp4")
    expect(sanitizeFilesDisabledPresentationText(
      '![](file:tenant-secret "generated_video.mp4")',
    )).toBe("generated_video.mp4")
  })

  it("preserves dotted business routes outside explicit local runtime roots", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Keep /docs/index.html, /customers/acme.com, /api/users/profile.json, and /v1/openapi.json.",
    )).toBe(
      "Keep /docs/index.html, /customers/acme.com, /api/users/profile.json, and /v1/openapi.json.",
    )
  })

  it("sanitizes parser-decoded character references instead of their raw source spelling", () => {
    expect(sanitizeFilesDisabledPresentationText(
      "Saved /private&#47;tenant&#47;secret.txt and file&#58;tenant-secret.",
    )).toBe(
      "Saved secret.txt and file.",
    )
  })

  it("sanitizes entity-encoded file references in raw HTML and code nodes", () => {
    expect(sanitizeFilesDisabledPresentationText([
      '<span data-path="/private&#47;tenant&#47;html-secret.txt">result</span>',
      '<img src="file&#58;html-file-secret">',
      "`/private&#47;tenant&#47;inline-secret.txt`",
      "```text",
      "/private&#47;tenant&#47;fenced-secret.txt",
      "```",
    ].join("\n"))).toBe([
      '\\<span data-path="html-secret.txt"\\>result</span>',
      '\\<img src="file"\\>',
      "inline-secret.txt",
      "~~~",
      "fenced-secret.txt",
      "~~~",
    ].join("\n"))
  })

  it("keeps entity-decoded code delimiters inside an inert presentation", () => {
    const projected = sanitizeFilesDisabledPresentationText([
      "`/private/inline-secret.txt &#96; ![track](https://evil.example/pixel)`",
      "```text",
      "/private/fenced-secret.txt &#96;&#96;&#96;",
      "![track](https://evil.example/fenced-pixel)",
      "```",
    ].join("\n"))

    expect(projected).not.toContain("/private/")
    expect(projected).toContain("![track](https://evil.example/pixel)")
    expect(projected).toContain("~~~")
    expect(projected).toContain("![track](https://evil.example/fenced-pixel)")
  })

  it("decodes local Markdown targets before classifying them as inert", () => {
    expect(sanitizeFilesDisabledPresentationText([
      "[encoded root](/%70rivate/tenant/secret.txt)",
      "![encoded slash](/private%2Ftenant%2Fsecret.png)",
      "[malformed suffix](/%70rivate/%E0%A4%A/tenant-secret.txt)",
      "![invalid suffix](/%70rivate/%ZZ/tenant-secret.png)",
      "[reference][encoded]",
      "",
      "[encoded]: /private%2Ftenant/reference-secret.txt",
    ].join("\n"))).toBe([
      "encoded root",
      "encoded slash",
      "malformed suffix",
      "invalid suffix",
      "reference",
      "",
      "",
    ].join("\n"))
  })

  it("basename-normalizes strong file-record name fields", () => {
    expect(projectFilesDisabledPresentation({
      artifacts: [
        { filename: "/private/tenant/secret.txt" },
        { file_name: "workspace/reports/summary.pdf" },
      ],
    })).toEqual({
      artifacts: [
        { filename: "secret.txt" },
        { file_name: "summary.pdf" },
      ],
    })
  })

  it("basename-normalizes descriptor-qualified name fields without rewriting business names", () => {
    expect(projectFilesDisabledPresentation({
      posix: { name: "/private/tenant/secret.txt", type: "file" },
      windows: { name: "C:\\private\\tenant\\secret.txt", mime_type: "application/octet-stream" },
      relative: { name: "reports/summary.pdf", type: "document" },
      business: { name: "Sales/West", type: "region" },
    })).toEqual({
      posix: { name: "secret.txt", type: "file" },
      windows: { name: "secret.txt", mime_type: "application/octet-stream" },
      relative: { name: "summary.pdf", type: "document" },
      business: { name: "Sales/West", type: "region" },
    })
  })

  it("redacts trace-style strings when projecting object presentation", () => {
    expect(projectFilesDisabledPresentation({
      message: "Open (/private/tenant/secret.txt) with path=/private/tenant/config.json and output/report.pdf",
    })).toEqual({
      message: "Open (secret.txt) with path=config.json and report.pdf",
    })
  })

  it("sanitizes displayable structured strings by default while preserving business routes", () => {
    expect(projectFilesDisabledPresentation({
      summary: "Summary /private/tenant/summary.txt",
      detail: "Detail /tmp/tenant/detail.txt",
      note: "Note /workspace/tenant/note.txt",
      stdout: "stdout: /sandbox/tenant/stdout.log",
      stderr: "stderr: /data/tenant/stderr.log",
      request_path: "/v1/shifts",
      route: "/care/shift/42",
    })).toEqual({
      summary: "Summary summary.txt",
      detail: "Detail detail.txt",
      note: "Note note.txt",
      stdout: "stdout: stdout.log",
      stderr: "stderr: stderr.log",
      request_path: "/v1/shifts",
      route: "/care/shift/42",
    })
  })

  it("replaces complete known-root descendants using the longest root", () => {
    expect(projectFilesDisabledPresentation({
      files: [{
        file_name: "report.pdf",
        output_dir: "/private/reports",
      }],
      message: "Created /private/reports/nested/secret.pdf",
    })).toEqual({
      files: [{ file_name: "report.pdf" }],
      message: "Created secret.pdf",
    })
  })

  it("redacts known local roots after scheme-like prefixes", () => {
    expect(projectFilesDisabledPresentation({
      files: [{ file_name: "report.pdf", output_dir: "/custom/output" }],
      message: "Open urn:/custom/output/nested/secret.txt and custom+tool:/custom/output/tool.txt",
    })).toEqual({
      files: [{ file_name: "report.pdf" }],
      message: "Open urn:report.pdf and custom+tool:report.pdf",
    })
  })

  it("inertizes known-root Markdown targets before substituting display labels", () => {
    expect(projectFilesDisabledPresentation({
      files: [{
        file_name: "report.pdf",
        output_dir: "/private/reports",
      }],
      message: [
        "[open](/private/reports/nested/secret.pdf)",
        "![image](/private/reports/nested/secret.png)",
      ].join(" "),
    })).toEqual({
      files: [{ file_name: "report.pdf" }],
      message: "open image",
    })
  })

  it("normalizes trailing separators on custom known roots before classifying targets", () => {
    expect(projectFilesDisabledPresentation({
      files: [{
        file_name: "report.pdf",
        output_dir: "/custom/output/",
      }, {
        file_name: "relative.txt",
        output_dir: "custom-output/",
      }],
      message: [
        "[open](/custom/output/nested/tenant-secret.pdf)",
        "![image](/custom/output/nested/tenant-secret.png)",
        "[relative][artifact]",
        "",
        "[artifact]: custom-output/nested/tenant-secret.txt",
      ].join("\n"),
    })).toEqual({
      files: [{ file_name: "report.pdf" }, { file_name: "relative.txt" }],
      message: [
        "open",
        "image",
        "relative",
        "",
        "",
      ].join("\n"),
    })
  })

  it("treats a declared POSIX filesystem root as the ancestor of absolute targets", () => {
    expect(projectFilesDisabledPresentation({
      files: [{ file_name: "root-output.pdf", output_dir: "/" }],
      message: "[open](/custom/tenant-secret.pdf) ![image](/custom/tenant-secret.png)",
    })).toEqual({
      files: [{ file_name: "root-output.pdf" }],
      message: "open image",
    })
  })

  it("does not apply a known-root label outside its root boundary", () => {
    expect(projectFilesDisabledPresentation({
      files: [{ file_name: "report.pdf", output_dir: "/private/reports" }],
      message: "/private/reports2/keep.txt and /private/reports/nested/secret.txt",
    })).toEqual({
      files: [{ file_name: "report.pdf" }],
      message: "keep.txt and secret.txt",
    })
  })

  it("keeps unrelated JSON-looking text byte-for-byte unchanged", () => {
    const value = '{"status":"ok","path":"and/or"}'
    expect(serializeFilesDisabledPresentation(value)).toBe(value)
  })

  it("reflects mutations to the source on every pure projection", () => {
    const value = { artifacts: [{ file_id: "file-id", file_name: "report.txt" }] }
    expect(projectFilesDisabledPresentation(value)).toEqual({ artifacts: [{ file_name: "report.txt" }] })
    value.artifacts[0].file_name = "updated.txt"
    expect(projectFilesDisabledPresentation(value)).toEqual({ artifacts: [{ file_name: "updated.txt" }] })
  })

  it("projects cyclic records through the serializer without retaining file locations", () => {
    const selfObject: Record<string, unknown> = {
      file_path: "/private/tenant/object-secret.txt",
      label: "object",
    }
    selfObject.self = selfObject

    const selfArray: unknown[] = [
      { output_dir: "/private/tenant/array-output", label: "array" },
    ]
    selfArray.push(selfArray)

    const left: Record<string, unknown> = {
      source_path: "/private/tenant/left-secret.txt",
      label: "left",
    }
    const right: Record<string, unknown> = { label: "right", next: left }
    left.next = right

    const serialized = serializeFilesDisabledPresentation({ selfObject, selfArray, left })

    expect(serialized).not.toContain("/private/")
    expect(JSON.parse(serialized)).toEqual({
      selfObject: { label: "object", self: "[Circular]" },
      selfArray: [{ label: "array" }, "[Circular]"],
      left: { label: "left", next: { label: "right", next: "[Circular]" } },
    })
  })

  it("preserves generic link and src fields and rejects uppercase non-routes", () => {
    const value = {
      link: "https://business.example/invoices/42",
      src: "https://cdn.example/image.png",
      route: "/api/files/PREVIEW/does-not-exist",
    }

    expect(projectFilesDisabledPresentation(value)).toEqual(value)
    expect(isManagedFileUrl(value.route)).toBe(false)
  })
})
