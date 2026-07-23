---
name: html-deck-editorial
description: |
  Generate a single-file HTML presentation deck with magazine-grade editorial
  styling. Use only when the user explicitly asks for an "html deck",
  "editorial slides", "magazine style presentation", "beautiful slides", "美观
  PPT", or "网页版幻灯片" — i.e. a slide-style visual deliverable AND a
  beautiful, designer-looking result is expected. Do not use this for generic
  visual design requests such as ad creatives, posters, banners, social media
  graphics, or image generation; use `static-visual-design` for designed static
  graphics and the image-generation tools for standalone image generation.
  Output is one self-contained .html file (inline CSS + JS, no build, no npm),
  printable to PDF via the browser, with keyboard navigation. Prefer this over
  native .pptx when the user wants visual quality over Office compatibility.
when_to_use: |
  Use for the explicit slide-style HTML requests described above when editorial
  polish matters more than Office compatibility. Prefer over `pptx-editorial`
  only when HTML output is requested.
tags:
  - presentation
  - html
  - deck
  - editorial
---

# HTML Editorial Deck

You will generate **one self-contained .html file** that presents the user's
content as a magazine-style slide deck. Write the file to the workspace using
the available file-writing tool (e.g. `file_tool` write_file or
`workspace_file_tool`), then return its path.

## Scope gate

Apply the rest of this skill only if the latest user request explicitly asks
for slides, a deck, a presentation, or an HTML slide deliverable. Do not infer a
deck merely because the user wants polished visual content.

If this skill was loaded for an ad creative, poster, banner, social media image,
or other designed static graphic, stop following this skill and use
`static-visual-design` instead. For a standalone illustration or photo, use the
available image-generation tool directly.

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** If the prompt is Chinese (中文), ALL slide
   content (kickers, titles, body, captions) must be in Chinese. NEVER copy
   English template phrases like `RESEARCH BRIEF` / `THE NEXT PARADIGM` into
   a Chinese deck — translate them (`研究简报` / `下一个范式`). Same applies
   to other languages.

1. **One file only.** All custom CSS + JS inline. No external `<script src=>`,
   no images you don't generate inline (use CSS color blocks or SVG).
   **One exception:** a single Google Fonts `<link rel="stylesheet" href="https://fonts.googleapis.com/...">`
   is allowed (and required) for the two typography families below — there
   is no other external resource permitted.
2. **Pick exactly ONE palette** from the 5 below. **Never mix hex values across
   palettes.** Never invent new hex values.
3. **Use only the 2 font families per palette.** No custom fonts.
4. **Forbidden visual elements** (output will look generic if you violate):
   `drop-shadow`, `box-shadow`, `border-radius > 2px`, `gradient backgrounds`,
   `blur`, `emoji as decoration`, `clipart`, `stock-photo placeholders`,
   `bevel`, `glassmorphism`.
5. **Real content only.** No `lorem ipsum`, no `[Title]` placeholders, no
   fabricated statistics. If the user didn't give data for a slot, leave that
   slot out — don't fill with junk.
6. **Slide count is driven by content.** Short content: 6–12 slides. Long
   content: 15–25. Do not pad. Do not omit user material.

## 🎨 Palettes — pick ONE, use only its 4 hex values

Each palette: `ink` (text + dark surfaces), `paper` (background), `paper-tint`
(subtle alt background), `ink-tint` (subtle alt text / dividers).

- **Monocle (default / business / tech)**
  ink `#0a0a0b` · paper `#f1efea` · paper-tint `#e8e5de` · ink-tint `#18181a`
- **Indigo Porcelain (data / research)**
  ink `#0a1f3d` · paper `#f1f3f5` · paper-tint `#e4e8ec` · ink-tint `#152a4a`
- **Forest Ink (sustainability / culture)**
  ink `#1a2e1f` · paper `#f5f1e8` · paper-tint `#ece7da` · ink-tint `#253d2c`
- **Kraft Paper (humanities / literature)**
  ink `#2a1e13` · paper `#eedfc7` · paper-tint `#e0d0b6` · ink-tint `#3a2a1d`
- **Dune (art / design / fashion)**
  ink `#1f1a14` · paper `#f0e6d2` · paper-tint `#e3d7bf` · ink-tint `#2d2620`

## ✒️ Typography

- **Display** (titles, big numbers): `'Playfair Display', 'Noto Serif SC', serif`
- **Body** (paragraphs, captions, UI): `'Inter', 'Noto Sans SC', sans-serif`
- Load via Google Fonts `<link>` is the **only** allowed external resource.
- `kicker` (small uppercase labels above titles): 11px, `letter-spacing: 0.12em`,
  `text-transform: uppercase`, color = `ink-tint`.
- Slide titles: 5–10vw display serif, line-height 1.05.
- Body: 16–18px, line-height 1.6.
- `folio` (page number bottom right): `01 / 12` style, Inter 11px.

## 📐 10 layouts — reuse freely, pick by content shape

| ID | Name | When to use |
|---|---|---|
| L01 | Hero Cover | First slide. Centered display title + kicker + lead paragraph + bottom meta row (author / date) |
| L02 | Act Divider | Section break. Kicker + 8.5vw display headline + single supporting line. Reverse colors (ink bg, paper text) for emphasis |
| L03 | Big Numbers Grid | 3×2 stat cards: small label + large number (display serif) + caption |
| L04 | Quote + Image | Left: kicker + headline + body + callout. Right: 16:10 visual block (CSS color block or inline SVG) |
| L05 | Image Grid | 3×2 or 3×1 visual blocks, **all same height** (use `26vh` or `22vh`, never mix) |
| L06 | Pipeline / Flow | Horizontal numbered steps: `№X` + step title + 1-line description |
| L07 | Hero Question | One full-screen question at 7vw, semantic line breaks, surroundings empty |
| L08 | Big Quote | 5.8vw display-serif quotation + translation (if any) + attribution + date |
| L09 | Before / After | 1:1 split. Left column `opacity: 0.55` (before). Right column full opacity (after) |
| L10 | Mixed Media | 8:4 split. Left: kicker / headline / body / callout. Right: 3:4 vertical visual block |

## 🖼️ Visual blocks (in place of images)

To keep the deck self-contained without external image assets, generate visual
blocks using:
- CSS color blocks with `paper-tint` / `ink-tint` background
- Inline SVG: geometric shapes (circles, lines, rectangles) using palette colors
- For data: hand-drawn-feeling bar/line charts inline SVG with palette colors

Never use external image URLs, never use Unsplash placeholders.

## ⌨️ Interaction (must include)

- Keyboard: `←` previous slide, `→` next slide, `Home` first, `End` last
- URL hash sync: `#slide-3` jumps to slide 3
- Mouse: click on right half of viewport = next, left half = previous
- Top progress bar (1px tall, ink color, filled to current slide ratio)

## 📝 Output checklist

Before writing the file, mentally verify:
- [ ] Exactly one palette, all hex values match
- [ ] Exactly 2 fonts (display + body), both loaded via single Google Fonts link
- [ ] No forbidden visual elements (shadows, gradients, emoji decoration, …)
- [ ] Slide count appropriate to content density (6–25 typical)
- [ ] Cover (L01) on slide 1
- [ ] Folio + progress bar present
- [ ] Keyboard nav + hash sync wired
- [ ] All content from the user's input, no fabricated data

Then write the file to the workspace, name it `deck.html` (or
`<topic>-deck.html` if the user gave a topic).

### 📎 Deliver as a clickable chip in chat

File-producing tools return the registered file reference directly. After
writing the file, **read the tool's response** for the `markdown_link`
field (or for `file_refs[].markdown_link` when there are multiple files)
and use that string verbatim as the first line of your final answer.

✅ **CORRECT**: copy the exact `markdown_link` string from the successful
file-writing tool result. The opaque file ID must originate in that result.

❌ **WRONG**: constructing a Markdown link yourself, guessing a UUID, reusing
an illustrative/example UUID, calling `get_file_info` to invent or "fetch" a
file ID, or wrapping the chip in a code fence. A syntactically valid `file:`
link is not evidence that a file exists. If the current task has no successful
file-writing result containing that link, the deck is not complete and you
must not present one.

After the chip line, report: number of slides, palette chosen, and a 1-line
summary of layout choices (e.g. `L01 cover → L02 divider → 3× L03 stats → …`).
