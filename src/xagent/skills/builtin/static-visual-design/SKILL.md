---
name: static-visual-design
description: |
  Create polished commercial and brand-facing static visual designs as complete
  PNG or JPEG assets. Use only for advertising creatives, campaign posters,
  promotional social posts, event or announcement cards, banners, and placement
  variants where art direction, typography, hierarchy, brand fidelity, and
  visual quality matter.
when_to_use: |
  Use only for marketing, campaign, event, or brand communication. Do not use
  for educational infographics, technical diagrams, concept explainers, charts,
  data visualizations, or standalone illustrations or photos.
---

# Static Visual Design

## Scope

Use this skill when the deliverable is marketing, promotional, campaign, event,
or brand-facing material. Explanatory work — concept images, comparison
graphics, educational infographics, technical diagrams, charts, data
visualizations — belongs to the general image-generation workflow instead, even
when a polished PNG is requested. Logo invention, video, ad-account management,
standalone illustration with no designed layout, and copy-only requests are all
out of scope.

Produce the finished visual with `generate_image` and `edit_image`. Let the
image model solve composition, type, atmosphere, and graphic language together.
Reserve HTML/CSS plus browser screenshots for when the user explicitly asks for
an editable HTML deliverable.

## Read the art direction reference

Before defining directions or rendering, read
`references/static-ad-art-direction.md` with `read_skill_doc`. It carries the
communication structures, layout specification, depth stack, copy budget,
render-prompt form, and review checklist this skill relies on.

## Establish the brief

Settle the minimum useful brief before generating: communication goal and
audience; primary message, supporting copy, CTA, and any required disclaimer;
output channel with its aspect ratio; and the logo, QR, product, or brand-guide
assets available.

Preserve copy the user marks as exact. Where the user supplies facts but not
final wording, write concise campaign copy from those facts. Every price,
date, milestone, performance claim, offer mechanic, eligibility rule, URL, and
piece of legal copy must come from the user or a sanctioned source below —
keep the visual to the claims you were actually given.

## Develop directions

For an open-ended brief, develop two or three directions that differ in
proposition, focal subject, and structure, then render each one. When the user
asks for exactly one final asset, compare directions internally and render the
strongest. The reference explains how to invent a direction's visual device and
how far apart a set should sit.

## Brand and identity assets

Inspect any reference images this task provides with `understand_media`, and
pass the useful ones into generation so the result belongs to the intended
visual world.

For work naming a real brand, resolve the identity before rendering finals. Two
sources need no permission: an asset supplied in this task — including one
attached in an earlier turn, recoverable with `list_all_user_files` — and the
current task workspace. When neither has the logo, stop before rendering finals
and ask the user how to proceed — offer to take the asset from them, to reserve
clean space where the logo will go, or to build the concept unbranded — and let
them choose. Going to look for it yourself is a third path that belongs to the
user, not to you: take it only when they tell you to, because a plausible
search result is not proof of authenticity, and one question costs a turn while
an unverified mark costs the asset.

The user's other tasks and earlier outputs are never a source: a file found by
listing them proves nothing about whose brand it shows. Neither is a reference
image, a competitor's material, or your own memory of what the brand looks
like — a logo you reconstruct is a logo you invented.

Attach an official logo through the image tool's `images` argument (or use
`edit_image`); naming it in the prompt text does not attach it. A generated or
edited recreation is never the final logo. QR codes, certification marks,
sponsor marks, and UI screenshots cannot survive generative rendering
pixel-for-pixel — when exact reproduction matters, say so and ask the user to
arrange deterministic post-processing.

Unless the user asked for an unbranded concept, a brand-specific final requires
a verified logo. Asking how to proceed without one is a complete answer to a
blocked brief, not a failure to work around: it is the outcome this skill wants,
and no amount of searching, reconstructing, or typesetting a substitute is a
better one. Render a reserved-space or unbranded draft only once the user has
chosen it, and label it a concept draft when you hand it back.

## Generate

Each `generate_image` call renders one finished placement on one continuous
canvas, one direction at a time. Exclude contact sheets, option grids, split
frames, presentation mockups, repeated layouts, and duplicated headlines, and
keep words like "variations" or "option A/B" out of the prompt.

Send only the exact text that should appear on the canvas — never alternative
copy, strategy labels, markdown, or rationale.

Pick each aspect ratio from the actual placement: social campaigns default to
4:5 feed and 9:16 story, print and out-of-home follow their placement's real
dimensions, and 1:1 applies when a channel requires it. When the user has not
named channels, either ask or cover the likely placements and say which asset
serves which. Compose natively for the ratio you chose.

Write a concept-specific prompt for every direction, following the render-prompt
form in the reference.

## Inspect

Inspect every candidate with `understand_media`. Reject: misspelled or invented
text, wrong numbers, dates, CTA, offer, or disclaimer; unclear hierarchy or
illegible thumbnail-size type; clipped or overlapping essentials; several ads
merged into one image; fake, duplicated, or distorted logos; fake QR codes,
watermarks, and unrelated lettering. A successful tool call is not evidence the
asset is finished.

## Repair budget

A **required asset** is one direction at one placement that the brief or this
skill calls for. Anything beyond that count is optional.

**Coverage** — producing the first candidate for a required asset that has none
— is never budgeted. Coverage means the asset is absent, not that an existing
candidate is bad.

A **repair** is any render call on an asset that already has a candidate: at
most two per asset and four per run. Calls on optional assets, re-renders of a
direction already delivered, and anything you would call a variant or retry all
cost a repair.

Regenerate from the design specification when the organizing idea, focal
subject, hierarchy, or canvas structure is wrong; use `edit_image` for a
localized defect on an otherwise strong candidate. When a limit is reached,
deliver that asset's best candidate. Counters do not carry across plan steps —
an asset that failed inspection in an earlier step is delivered with its defects
named, not retried on a fresh budget.

## Finish

Coverage is unconditional: every required asset must exist as a successful tool
result before you finish. One thing overrides it — a brand-specific brief with
no verified logo ends in the question above, with nothing rendered, and that
counts as finished. Quality is what the budget releases — when the budget
is spent and an asset still fails inspection, deliver its best candidate, name
the defect concretely (which text is misspelled, which element is clipped, which
logo is not authentic), and mark the answer's outcome `partial` so the user can
decide whether to spend another round.

Return only PNG or JPEG files that were actually created. Lead with the files,
then give one concise line per asset identifying its communication angle and
dimensions.
