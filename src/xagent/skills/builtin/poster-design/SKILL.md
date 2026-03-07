---

name: poster-designer
description: Unified poster generation skill with hard routing. Produces a single final poster image using mandatory HTML-based layout for information-heavy posters and image-first generation only for strictly visual posters.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Poster Designer

## Overview

**Poster Designer** is a unified poster generation skill that always outputs **one final poster image**, while internally selecting the appropriate generation strategy.

Unlike soft, heuristic-based tools, this skill enforces **hard routing rules** to ensure that information-heavy posters are generated using deterministic layout rather than direct image generation.

The skill supports two internal strategies:

* **Layout-first (HTML-based)** for posters that convey structured information
* **Image-first (direct image generation)** for posters that are purely visual and symbolic

Strategy selection is **final and non-negotiable** once routing conditions are met.

---

## When to Use

Use this skill whenever the desired output is:

* A poster / long banner / vertical visual
* A single image intended for publishing or long-term display

Typical use cases include:

* Open-source milestone announcements
* Community or project notices
* Recruiting posters
* Product or platform announcements
* Technical infographics

---

## Output Contract

This skill always produces:

* **Exactly one final poster image**

The skill must **never** output:

* HTML or CSS
* Image-generation prompts
* Design explanations
* Intermediate artifacts

Only the final image is returned.

---

## Strategy Selection (Hard Routing)

Poster Designer applies **mandatory routing rules** when selecting the generation strategy.

These rules override upstream planner preferences and must not be bypassed.

---

## Mandatory Layout-first Conditions

The **Layout-first (HTML-based) strategy MUST be used** if **any** of the following conditions are satisfied:

* The poster is described as an **announcement**, **notice**, or **formal milestone**
* The poster is intended for **long-term display**, including but not limited to:

  * README files
  * Official websites
  * Documentation pages
  * Community announcements
* The request explicitly mentions or implies:

  * structured information
  * clear hierarchy
  * readability
  * sections, paragraphs, or multiple text blocks
* The poster contains more than one semantic text role, such as:

  * a title
  * a numeric highlight
  * explanatory or descriptive text
  * acknowledgements or credits
  * footer or meta information

When **any** of these conditions are met:

* **Direct image generation MUST NOT be used**
* The task must be treated as a **layout-driven artifact**
* HTML-based composition is **required** before producing the final image

---

## Layout-first Strategy (HTML-based)

### Characteristics

* Fixed-width layout (default: **1080px**)
* Content-adaptive height (no hard-coded height)
* Deterministic typography, spacing, and alignment
* Clear reading order and visual hierarchy
* Final poster image generated via full-container screenshot

### Critical Screenshot Requirement

**When using browser screenshot tools, the viewport width MUST be explicitly set to match the layout width (default: 1080px), and the output filename MUST clearly indicate the poster's purpose.**

### Intent

This strategy prioritizes:

* Text clarity and legibility
* Information hierarchy
* Long-term usability and stability

It is the **default strategy** for any poster that functions as an information artifact rather than a purely visual symbol.

---

## Image-first Strategy (Direct Generation)

### Usage Constraints

The **Image-first strategy is ONLY permitted** when **all** of the following conditions are satisfied:

* The poster contains **a single headline or a single numeric highlight**
* Supporting text is minimal and decorative (no paragraphs or sections)
* The request prioritizes **atmosphere, emotion, or visual impact** over information clarity
* The poster resembles a **keynote KV**, **launch visual**, or **symbolic milestone image**

If **any** of these conditions are not met, Image-first generation **must not be used**.

### Characteristics

* Composition-driven design
* Cinematic lighting and atmosphere
* Text treated as a graphic element rather than structured content

---

## Visual Design Principles

Regardless of strategy, Poster Designer enforces the following principles:

* **Structure before decoration**
* **Clarity before effects**
* **Semantic visuals over abstract decoration**

The poster must clearly communicate:

* What the poster represents
* Why it matters
* What the primary visual anchor is

---

## Semantic Visual Anchors

Poster Designer introduces **domain-specific visual anchors** instead of generic backgrounds.

Examples include:

* Docker milestones → containers, image layers, registry-to-node pull networks
* Recruiting → structured sections, engineering environments, technical diagrams
* Product platforms → architecture diagrams, UI-like compositions
* Community milestones → nodes, constellations, global networks

Purely abstract backgrounds without semantic meaning are insufficient.

---

## Typography & Text Handling

* Clear hierarchy between headline, supporting text, and meta information
* Text quantity adapts to the selected strategy
* Text must remain legible at poster scale
* Decorative effects must never reduce readability

---

## Image Usage

* Images are purposeful, not decorative fillers
* Background imagery must preserve text readability
* Mid-poster visuals must reinforce meaning
* No competing focal points are allowed

---

## Element Overlap & Visual Clarity Check

Before final screenshot, verify:

1. **No critical content is obscured**
   * Main title, key numbers, primary CTAs must be fully visible
   * Background imagery must not compete with text readability
   * Text placed over images should only be done when the obscured image area is non-essential

2. **Layer discipline**
   * Fixed-position elements must not cover center content
   * Decorative elements should not obscure semantic content
   * High z-index elements (tooltips, popups, overlays) should be user-triggered, not persistent

3. **Visual hierarchy**
   * Primary visual anchor should be immediately apparent
   * No competing focal points that confuse the viewer
   * Clear separation between foreground content and background decoration

4. **Text-on-image safety**
   * When text is placed over images, ensure sufficient contrast
   * Avoid placing text over areas with high visual detail that would reduce legibility
   * Solid or gradient overlays may be used to improve text readability

If any of these checks fail, the poster must be adjusted before taking the final screenshot.

---

## Quality Bar

A successful output must be suitable for:

* GitHub README banners
* Docker Hub or open-source announcements
* Conference or meetup visuals
* Company or community social posts

If the output resembles:

* A dashboard or analytics screenshot
* A generic AI marketing image
* An abstract background with loosely placed text

Then the skill has failed.

---

## Example Invocation (Conceptual)

"Create a formal milestone announcement poster for Xinference celebrating **5,000,000 Docker Hub downloads**. The poster must include a title, a prominent numeric highlight, explanatory text, acknowledgements, and footer information suitable for long-term display on README and official websites."

---

## Summary

**Poster Designer** is an intent-aware, rule-enforced poster generation skill.

It guarantees that information-heavy posters are generated using deterministic layout, while reserving direct image generation strictly for minimal, visual-only use cases.
