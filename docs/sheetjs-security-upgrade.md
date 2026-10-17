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

`npm run test:file-previews` runs all file-preview tests as a required frontend
CI step; the manifest contract rejects removal of this step.

This only addresses the SheetJS portion of CASA #5. Other frontend and Python
dependency findings remain separate work. SaaS must update its core submodule
and rebuild the deployed image to include this upgrade.
