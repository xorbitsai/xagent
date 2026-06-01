---
name: pptx-editorial
description: |
  Generate a native PowerPoint (.pptx) file with magazine-grade editorial
  styling — pitch decks, investor presentations (路演 PPT), board reviews,
  executive memos, conference talks, product launches. Output is a single
  .pptx file generated via pptxgenjs through the JavaScript executor runtime, openable
  in PowerPoint / Keynote / Google Slides. Uses five named editorial palettes
  (Monocle, Indigo Porcelain, Forest Ink, Kraft Paper, Dune) with strict
  typography rules (Georgia display + Calibri body) and 10 numbered layouts
  (L01 Cover, L02 Act Divider, L03 Big Numbers, ..., L10 Closing).
  Prefer this skill over the builtin presentation-generator whenever the
  user wants a polished editorial look, names one of the five palettes,
  asks for a "美观 PPT" / "投资人路演 PPT" / "pitch deck", or otherwise
  signals that visual quality matters beyond a generic deck.
  Use `html-deck-editorial` instead only when the user explicitly wants HTML
  output (more visual freedom but not Office-compatible).
when_to_use: |
  When the user wants a polished native .pptx deliverable (pitch deck, 路演 PPT,
  board review, executive memo) where visual quality matters and the file must
  open in PowerPoint / Keynote / Google Slides. Use `html-deck-editorial`
  instead when the user explicitly wants HTML output.
tags:
  - presentation
  - pptx
  - powerpoint
  - deck
  - editorial
---

# Editorial PPTX Deck

You will generate one `.pptx` file by writing a JavaScript program that uses
**pptxgenjs** and runs it via the `execute_javascript_code` tool. Save the file
to the workspace (use the output dir the tool returns), then report the path
and a 1-line layout summary.

## ⚠️ Hard rules — NO exceptions

0. **MATCH THE USER'S LANGUAGE.** This is rule zero, the most violated rule.
   If the user prompt is in Chinese (中文), EVERY string in the deck — kickers,
   slide titles, body text, captions, kicker labels, folio prefixes, sample
   names, captions — must be in Chinese. **NEVER** copy English template
   phrases like `THE PROBLEM`, `FEATURE STORY`, `RESEARCH BRIEF`,
   `THE BOTTOM LINE`, `BUILT BY BUILDERS` into a Chinese deck. Translate
   them: e.g. `THE PROBLEM` → `问题所在`, `THE SOLUTION` → `解决方案`,
   `MARKET OPPORTUNITY` → `市场机会`, `THE TEAM` → `团队`, `USE OF FUNDS` →
   `资金用途`, `THE BOTTOM LINE` → `一句话总结`, `RESEARCH BRIEF` →
   `研究简报`, `CASE STUDY` → `案例研究`. Person names in a Chinese deck
   should be Chinese names (e.g. 张伟 · 陈琳), not transliterated English
   names. Folio reads `01 / 12` in either language (digits are universal).
   Same applies to other languages: match the prompt.
1. **One palette only.** Pick one of the 5 palettes below, use only its 4 hex
   values. Never invent new hex.
2. **Two fonts only.** Display = `Georgia` (serif, present on all OSes).
   Body = `Calibri` (sans, Office default). No custom fonts — `.pptx`
   recipients won't have them and the deck will fall back to Times.
   **For Chinese decks**: Georgia + Calibri both render Chinese via system
   fallback fonts (PingFang on macOS, Microsoft YaHei on Windows) — that's
   fine, do not switch to custom Chinese fonts.
2a. **🚫 NEVER apply `charSpacing` to CJK / non-Latin text.** The trick of
    "tracked-out caps" only works for Latin letters (`A B C D`). On Chinese,
    Japanese, Korean characters — which are already full-square glyphs —
    `charSpacing: 10` (0.10em) renders as `部　署　周　期` with huge gaps
    between every character. This looks broken on EVERY rendering platform.
    Rule: if the text contains ANY CJK character, set `charSpacing: 0`
    (or omit the property entirely). Only apply `charSpacing: 8-15` when
    the entire string is uppercase Latin (e.g. `RESEARCH BRIEF`, `PART 02`).
    For Chinese kickers, just use bold + small caps look via 11pt bold ink-tint
    text WITHOUT any character spacing.
3. **Forbidden visual elements** (PowerPoint defaults to these — you must
   actively avoid):
   - drop shadow / glow / reflection / soft edge / bevel / 3-D rotation
   - gradient backgrounds
   - rounded rectangles (use `rectRadius: 0`)
   - clipart, emoji as decoration, stock-photo placeholders
   - Comic Sans, Times New Roman default, Arial Black
   - smart-art with default styling
   - rainbow chart palettes (chart uses ink + paper-tint only)
4. **Real content only.** No lorem ipsum, no `[Title here]` placeholders,
   no fabricated statistics. If a slot has no user data, drop the slot.
   - **Team slides especially**: if the user did not provide team names,
     titles, or bios, DO NOT invent them. Don't make up `陈明远 · 前阿里
     云智能平台总监 · 清华大学计算机硕士` — that's fabrication and looks
     fake to anyone in the industry. Either ask the user for team info
     OR drop the team slide entirely and note "团队信息待补充".
   - Same for fake financial metrics (MRR / ARR / growth rate), fake
     customer names, fake competitive comparisons. If you don't have it,
     don't ship it.
5. **Slide count is driven by content.** 6–12 short, 15–25 long. No padding.
6. **Fill the slide.** The 13.33 × 7.5 inch canvas is large. Do NOT cram
   everything into the left third. Use `align: 'center'` on hero text, place
   the body block across columns 1–11 (x: 0.7 to 12.6), use the full vertical
   range top-to-bottom. A slide that looks 60%+ empty is broken.
7. **Sizing for Chinese text**: Chinese characters take MORE horizontal
   space per char than Latin. Rules of thumb:
   - Hero title at fontSize 40-48: give the textbox at least `w: 12` inches
     (the full content width). Title at `w: 6` will overflow / wrap badly.
   - 1 Chinese character at 40pt ≈ 0.55 inch wide. Plan accordingly.
   - For section titles (24-28pt) keep `w` ≥ 8 inches.
   - For body text (14-16pt) `w: 11+` is safe for full sentences.
8. **No overlapping shapes.** When you place a dark section header / banner
   on a slide, leave at least 0.3 inch of vertical clearance before any
   text shape. Banners at `y: 1.2, h: 0.8` should not have a title block
   starting at `y: 1.5` — that overlaps and clips text.

## 🎨 Palettes — pick ONE

Each: `ink` (text + dark surfaces), `paper` (slide background), `paper-tint`
(card background / chart fill), `ink-tint` (kicker + dividers).

- **Monocle** (default / business / tech)
  ink `0A0A0B` · paper `F1EFEA` · paper-tint `E8E5DE` · ink-tint `18181A`
- **Indigo Porcelain** (data / research)
  ink `0A1F3D` · paper `F1F3F5` · paper-tint `E4E8EC` · ink-tint `152A4A`
- **Forest Ink** (sustainability / culture)
  ink `1A2E1F` · paper `F5F1E8` · paper-tint `ECE7DA` · ink-tint `253D2C`
- **Kraft Paper** (humanities / literature)
  ink `2A1E13` · paper `EEDFC7` · paper-tint `E0D0B6` · ink-tint `3A2A1D`
- **Dune** (art / design / fashion)
  ink `1F1A14` · paper `F0E6D2` · paper-tint `E3D7BF` · ink-tint `2D2620`

(pptxgenjs takes hex without the `#` prefix.)

## ✒️ Typography rules (use exactly these sizes)

- **Slide title (display)**: Georgia, 48pt, `ink` color, `bold: false`
- **Body**: Calibri, 16pt, `ink`, line spacing 1.4
- **Kicker** (small uppercase label above title): Calibri, 10pt, `ink-tint`,
  `bold: true`, **`charSpacing: 10`**
  ⚠️ pptxgenjs `charSpacing` units are **hundredths of an em** (relative to
  font size), NOT 1/100 pt. CSS standard for "tracked-out caps" is
  `letter-spacing: 0.08em–0.12em` → use **`charSpacing: 8` to `12`**.
  - `charSpacing: 10` = 0.10em (DEFAULT for kicker — tight, readable)
  - `charSpacing: 15` = 0.15em (max for kicker)
  - `charSpacing: 30` = 0.30em (already visibly too wide — letters drift apart)
  - `charSpacing: 50` = 0.5em = very airy, **don't use anywhere**
  - `charSpacing: 100` = 1em = one full character gap (DON'T USE)
  - `charSpacing: 200`+ = letters split into separate columns (DESTROYED)
  - **HARD CAP: never use `charSpacing > 15`.** No exceptions for any text
    element — bigger values always look broken in PPT.
- **Folio** (`01 / 12` bottom-right): Calibri, 10pt, `ink-tint`
- **Big number** (L03): Georgia, 80pt, `ink`
- **Big quote** (L08): Georgia italic, 36pt, `ink`

## 🏷️ Kicker semantics — READ CAREFULLY

The **kicker is a STANDALONE section label**, not a fragment of the title.
It's the magazine equivalent of a department tag (NYT-style).

✅ **Correct kickers** (short, 1–4 words, all caps, semantic label):
- `FEATURE STORY`
- `THE NEXT PARADIGM`
- `RESEARCH BRIEF`
- `POINT 02`
- `CASE STUDY`
- `THE BOTTOM LINE`
- `INDUSTRY ANALYSIS`

❌ **NEVER do this** (splitting the title across kicker + display):
- Kicker `THE RISE OF REMOTE-FIRST` + Title `Engineering Teams` — WRONG.
  The title must read as one complete phrase.

✅ **Correct version of above**:
- Kicker `RESEARCH BRIEF` (or `THE FUTURE OF WORK`)
- Title `The Rise of Remote-First Engineering Teams` (complete, one block)

## 🔣 Punctuation rules

- Meta rows and bylines: separate items with **middle dot `·`** (Unicode U+00B7),
  NOT hyphen `-` or em-dash `—`.
  Example: `Remote Work Research · May 2026` (NOT `Remote Work Research - May 2026`)
- Citations: `— Author Name, Role · Source, Year` (em-dash before name)
- En-dash `–` for number ranges (`2019–2024`), hyphen `-` only inside compound
  words (`remote-first`).

## 📐 10 layouts

Each slide should set background to `paper` (or `ink` for L02 inverted).
Standard slide is 13.33 × 7.5 inches (16:9).

| ID | Name | Layout |
|---|---|---|
| L01 | Hero Cover | Center: kicker (top 1.5") + display title (centered 3.5") + lead body (5.5") + meta row "Author · Date" (bottom 6.5") |
| L02 | Act Divider | Background = `ink`, text = `paper`. Kicker + 60pt display title centered |
| L03 | Big Numbers Grid | 3 cards in row: each = paper-tint background (no shadow!), kicker top, 80pt big number, 14pt caption |
| L04 | Quote + Visual | Left half (kicker + title + body + callout) · Right half (color block, paper-tint with single accent line) |
| L05 | Image Grid | 2×2 or 3×2 paper-tint blocks (same dims), kicker label above each |
| L06 | Pipeline / Flow | 3–5 numbered columns: large № (Georgia 40pt) + step title + 1-line desc |
| L07 | Hero Question | One full-slide question, Georgia 54pt, centered, 1.3 line height |
| L08 | Big Quote | 36pt Georgia italic quote (5 lines max), 14pt attribution below "— Name, Role · Source, Year" |
| L09 | Before / After | 50/50 split. Left column 55% opacity (before). Right column 100% (after). Use kicker `BEFORE` / `AFTER` |
| L10 | Closing | Like L01 but kicker = `THE BOTTOM LINE`. End with a takeaway sentence, no folio on this slide |

## 🛠️ pptxgenjs implementation pattern

⚠️ **Top-level `await` is FORBIDDEN.** The JavaScript executor runtime runs the
script as `node script.js` with NO `package.json` in the exec dir, so
top-level `await` makes Node guess ESM mode and `require()` then throws
`require is not defined in ES module scope`. **Wrap everything in an async
IIFE** so the await lives inside a function:

```javascript
const pptxgen = require('pptxgenjs');

(async () => {
  const pres = new pptxgen();
  pres.layout = 'LAYOUT_WIDE';  // 13.33 × 7.5

// Define palette ONCE at top, reference everywhere
const palette = { ink: '0A0A0B', paper: 'F1EFEA', paperTint: 'E8E5DE', inkTint: '18181A' };
const fonts = { display: 'Georgia', body: 'Calibri' };

// L01 Cover (centered hero — fills the slide)
// NOTE on language: if the user prompt is Chinese, kicker / title / meta MUST be Chinese.
// Examples below are English for illustration; substitute your language.
const s1 = pres.addSlide();
s1.background = { color: palette.paper };
// Kicker — small, centered, only ~10 chars tracked-out
s1.addText('RESEARCH BRIEF', {
  x: 0.7, y: 2.4, w: 12, h: 0.4,
  fontFace: fonts.body, fontSize: 11, color: palette.inkTint,
  bold: true, charSpacing: 10, align: 'center',
});
// Display title — centered, wide enough to wrap on 2 lines if needed
s1.addText('Why AI Agents Will Change Software Development', {
  x: 0.7, y: 2.9, w: 12, h: 1.8,
  fontFace: fonts.display, fontSize: 48, color: palette.ink, align: 'center',
});
// Lead body (optional one-liner subtitle)
s1.addText('A 2026 research brief on agent infrastructure, market shape, and the shift to production-grade systems.', {
  x: 1.5, y: 4.8, w: 10.3, h: 1.0,
  fontFace: fonts.body, fontSize: 14, color: palette.ink, align: 'center',
});
// Meta row — bottom center
s1.addText('Xagent Team · May 2026', {
  x: 0.7, y: 6.6, w: 12, h: 0.3,
  fontFace: fonts.body, fontSize: 11, color: palette.inkTint, align: 'center',
});
// Folio bottom-right
s1.addText('01 / 06', { x: 11.5, y: 7.0, w: 1.5, h: 0.3,
  fontFace: fonts.body, fontSize: 10, color: palette.inkTint, align: 'right' });

  // Save — `await` is INSIDE the async IIFE, this is fine
  await pres.writeFile({ fileName: 'editorial-deck.pptx' });
  console.log('Done: editorial-deck.pptx');
})();  // ← critical: close + invoke the async IIFE
```

⚠️ **About the layout numbers above**: the cover places hero text in the
*center* of the slide, not the left edge. Common mistake from earlier
templates was `x: 0.7, y: 1.8` with no `align: 'center'` — that pins title
to the upper-left and leaves ~60% of the slide empty. Don't repeat that.

⚠️ **Don't forget the `packages` arg.** When calling `execute_javascript_code`,
pass `packages: "pptxgenjs"` (string) so the executor `npm install`s it
before running. Without it the script crashes with `Cannot find module
'pptxgenjs'`.

pptxgenjs is **not** preinstalled in the sandbox — supplying it via
`packages` is what makes it available for the run. Always pass it
through `packages`, even on repeated runs in the same task.

## 📝 Output checklist (verify before running)

- [ ] **JS is wrapped in `(async () => { ... })()` IIFE** — no top-level `await`
      (Node will pick ESM mode and `require()` will throw).
- [ ] **`packages: "pptxgenjs"` is passed to `execute_javascript_code`** so
      the executor installs it before running.
- [ ] **LANGUAGE matches the user's prompt** (Chinese prompt → Chinese deck;
      NO English template phrases like `THE PROBLEM` leaking through). Check
      every `addText` call before running.
- [ ] **Hero/cover text is centered (`align: 'center'`) and uses the full
      slide width** (`w: 12+`), not pinned to `x: 0.7` with no alignment.
- [ ] Palette: exactly ONE chosen, all 4 hex values match
- [ ] Fonts: ONLY Georgia + Calibri (no Playfair Display etc — those won't render in PPT)
- [ ] No `shadow:`, no `glow:`, no `gradient:`, no `rectRadius:` > 0
- [ ] **`charSpacing` ≤ 15** for English uppercase kickers (units are 1/100 em).
      For **Chinese / CJK** text: `charSpacing: 0` ALWAYS (no exceptions).
      Anything > 0 on CJK looks broken with huge gaps between every character.
- [ ] **Kicker is a SEPARATE section label** (e.g. `RESEARCH BRIEF` for EN
      decks, `研究简报` for ZH decks), NOT a fragment of the slide title.
      Title is one complete phrase by itself.
- [ ] Meta rows use middle dot `·` between items (not `-` or `—`)
- [ ] Slide count 6–25 driven by user content
- [ ] L01 cover on slide 1, L10 closing on last slide
- [ ] Folio "NN / TT" on every slide except L10
- [ ] All numbers / quotes from user content (or clearly attributed sources)
- [ ] **No fabricated team / customer / financial data**. If user didn't
      give team names, drop the team slide (or use "团队信息待补充").
- [ ] **Chinese hero titles use `w: 12`** (not `w: 6`). Chinese chars are
      ~0.55" each at 40pt — narrow textboxes overflow / wrap broken.
- [ ] **No overlapping shapes** (banner blocks vs. title blocks). Allow at
      least 0.3" vertical clearance between adjacent y-positioned shapes.
- [ ] **Final answer FIRST LINE is `[filename](file:UUID)`** as bare markdown
      (not in code fences, not as "file_id: UUID" text). Otherwise the
      chip won't render — the user can't click anything.

Run via `execute_javascript_code` (with `packages: "pptxgenjs"`), then:

1. **Read the tool's response.** When the JS run succeeds and produces a
   .pptx in the workspace, the tool result includes a `markdown_link`
   field (e.g. when there's one output file) or a `file_refs` array
   whose entries each carry `file_id` + `filename` + `markdown_link`.
   Use the returned `markdown_link` string verbatim.
2. In your final answer, **the first line MUST be that chip link**
   (literal markdown syntax, NOT wrapped in backticks, NOT presented as
   "here's the UUID"):

   ✅ **CORRECT** (renders as clickable chip):
   ```
   [editorial-deck.pptx](file:20fae785-3823-4906-b385-d0e8a7807dc8)
   ```
   That is, the message body literally starts with `[filename](file:UUID)`
   on its own line — NOT inside ``` code fences. The exact UUID comes
   from the tool response; never fabricate one.

   ❌ **WRONG — common failure mode that renders as plain text**:
   ```
   已获取文件信息：
   - 文件名: `editorial-deck.pptx`
   - file_id: `20fae785-3823-4906-b385-d0e8a7807dc8`
   ```
   This shows the UUID as decoration and the user can't click anything.
   The chat UI looks for `[name](file:UUID)` markdown specifically.

   ❌ **ALSO WRONG**: putting the chip link inside a code block:
   <pre>
   ```
   [editorial-deck.pptx](file:20fae785-...)
   ```
   </pre>
   Code blocks suppress markdown — the link won't render as a chip.

   ❌ **ALSO WRONG**: calling `get_file_info(...)` to "fetch" the
   file_id — its `FileInfo` return shape does not include `file_id`.
   The chip reference is in the executor's own response, not a
   separate metadata call.

3. After the chip link line, report:
   - Palette chosen
   - Layout sequence (e.g. `L01 cover → L03 stats → L06 pipeline → L08 quote → L10 closing`)
   - Page count and file size
