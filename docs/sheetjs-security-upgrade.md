# SheetJS security upgrade

The frontend reads user-supplied spreadsheets. The npm registry's `xlsx`
0.18.5 is affected by CVE-2023-30533 (prototype pollution) and CVE-2024-22363
(regular expression denial of service). Reading untrusted workbooks is in
scope; escaping the generated HTML alone does not fix these parser issues.

Use SheetJS CE 0.20.3 from the exact publisher-owned tarball documented in
[the official installation guide](https://docs.sheetjs.com/docs/getting-started/installation/nodejs/).
The lockfile pins its SHA-512 integrity. Clean `npm ci` must be able to access
`cdn.sheetjs.com`; do not fall back to the stale npm registry release.

The new reader decodes XML entities and its HTML exporter escapes hyperlink
attributes. The preview therefore passes through the original link target
instead of manually escaping it again. Rich text, formulas and the final
HTML still pass through the existing sanitization policy. Tests cover real
workbook bytes, unsafe hyperlinks, attribute injection, literal entity text,
merged cells, cached formulas and CSV input.

`npm run test:file-previews` runs all file-component tests as a required frontend
CI step, including spreadsheet security regressions; the manifest
contract rejects removal of this step. Some shared previews also run in the
widget lane; the separate lane covers the remaining file components too.
KaTeX regressions run with the Markdown/widget tests.

The follow-up also pins KaTeX 0.18.2 and overrides its transitive copies so
`rehype-katex` and `micromark-extension-math` do not retain vulnerable 0.16.x
installations. GHSA-238p-pmpm-9mq7 requires an existing polluted prototype or
attacker-controlled renderer-options prototype; it is not a claim that KaTeX
itself causes prototype pollution. A regression supplies an inherited `trust`
option and checks that untrusted links/images remain disabled with the fixed
renderer. Existing formula and Markdown compatibility tests remain.
The override intentionally crosses the consumers' current ^0.16 dependency
ranges; retain it until all transitive consumers declare and resolve a fixed
KaTeX range, then remove it after rerunning the compatibility tests.

This upgrade covers SheetJS and KaTeX, not every dependency finding.
Both Docker image builds and local `npm ci` require outbound HTTPS access to
`cdn.sheetjs.com` in addition to the npm registry. Restricted builders must
allow that publisher endpoint (or use an integrity-preserving internal mirror);
never substitute the stale npm registry `xlsx` release. Rebuild deployed assets
after updating the lockfile.
