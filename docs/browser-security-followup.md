# Browser rendering follow-up (CASA 14)

## Fixed boundaries

File-chip HTML used by chat, file mentions and the agent builder now escapes
raw file paths, IDs, names and CSS-class values in both attributes and text.
Canonical/legacy file tokens are parsed from raw input before escaping, so
special characters round-trip once and backticks cannot reparse generated HTML.
These changes complement, not replace, core PR #3179's OAuth popup encoding and
PR #3180's spreadsheet/math parser fixes.

The frontend now receives a compatible enforced CSP in both delivery modes:

- The FastAPI static-export server adds the policy to actual HTML files,
  dynamic-route shells and HTML 404 responses, not RSC/JS assets.
- The independent Next server sets headers through its own configuration.
  Export builds deliberately do not rely on Next headers, which export ignores.
- Regular pages deny embedding; Widget pages retain third-party embedding.
- Base URLs and plugins are blocked, form destinations restricted, and
  production eval disabled. Ordinary API/upload responses deny script execution;
  the existing OAuth callback pages retain inline popup feedback without eval.

## Deliberately not called strict CSP

Generated HTML previews use a scripts-only, opaque-origin srcDoc sandbox. Such
frames inherit their embedding document's CSP. A strict parent script nonce/hash
policy would block existing generated inline scripts and external chart/CDN
scripts even inside that sandbox. This patch therefore explicitly retains
inline/HTTPS scripts in the page baseline; it is **not full CASA 14 closure**.

Completing strict script CSP requires a separately reviewed preview delivery
boundary (for example, an isolated preview origin), plus a compatible strategy
for static Next output. Do not solve this by enabling `allow-same-origin`, by
globally restoring eval, or by silently breaking interactive previews. The
current policy can also intentionally block nested third-party iframes and
generated eval-based code; staging must inventory any legitimate uses.
Production intentionally blocks HTTP images/media (including mixed-content
assets), `<base>`-dependent relative preview URLs, and eval-based in-DOM
framework builds such as Alpine/Vue. Use HTTPS assets, explicit resource URLs,
and CSP-compatible precompiled framework builds; these are compatibility
changes, not a guarantee that every generated preview continues to work.

Release acceptance must use real browsers (not jsdom) to open PDF and video
previews and the Google Drive Picker under the enforced production policy.
The Picker frame is restricted to https://docs.google.com; PDF/video bytes are
decoded locally into revocable blob URLs, without broadening frame/connect
permissions to data URLs.

Release acceptance must cover login/OAuth popups, chat file mentions, builder
instructions, spreadsheets/Markdown, interactive HTML previews, standalone and
packaged frontend delivery, and a Widget embedded from another origin. SaaS
must update its reviewed core pin and rebuild; this change does not update
Cloudflare/ingress headers or claim deployment acceptance.
