---
name: static-visual-design
description: |
  Design a finished ad, poster, banner, social post, or invitation as one
  PNG/JPEG with layout, headline, and brand styling rendered in, or adapt one
  into another placement size or aspect ratio.
when_to_use: |
  Any marketing, promotion, campaign, event, or brand-facing image, including
  ones naming a real brand. Not for explanatory diagrams, charts, infographics,
  or plain illustrations.
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

One creative lead defines and compares the whole set sequentially, in one pass.
Do not split ideation across independent agents or parallel plan nodes —
independent ideation converges on the same obvious brand cues and varies only the
decoration. Parallel execution is available only after every brief and
specification is locked, and each parallel executor then holds one distinct
proposition it may not reinterpret.

Order the work, and represent that order in any execution plan: brand and
reference acquisition is a shared prerequisite, creative direction depends on it,
and every render depends on the locked specification. Never plan a render to run
alongside acquisition.

Searching for identity assets is never a planned step, sequential or otherwise.
A plan step carries no authorization, so a step that exists will run — and
acquisition reporting no verified logo is the trigger for the question, not for a
search. Plan the question instead; a search only ever follows the user answering
it by asking for one. Leave asset availability out of step definitions too:
whether a verified logo exists is an outcome of acquisition, so how the brand
gets represented stays open until that step runs.

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

Separate stable identity cues from temporary campaign styling. Stable cues
recur across several recent official materials: the logo, colour relationships,
typography character, product imagery, graphic proportions. A gradient, metallic
treatment, bevel, glow, ribbon, or confetti field appearing in one old banner is
that campaign's styling, not the brand's identity — do not promote it into a
permanent cue. Preserve recognition through the stable cues while modernizing
hierarchy, whitespace, type discipline, and the number of competing effects.

Unless the user asked for an unbranded concept, a brand-specific final requires
a verified logo. Ask with `ask_user_question` so the turn ends waiting for the
user and resumes on their choice; that is a complete answer to a blocked brief,
not a failure to work around, and no amount of searching, reconstructing, or
typesetting a substitute is a better one. Render a reserved-space or unbranded
draft only once the user has chosen it, and label it a concept draft when you
hand it back.

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

A render that omits a required brand asset is a rejection too. Passing a verified
logo as a reference does not prove the image model placed it, so confirm the mark
is present and faithful in the result. Re-render within the repair budget; if the
budget runs out with the asset still missing, name the omission in the hand-back
rather than presenting the render as branded.

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

Where both definitions fit one call, coverage wins. A required placement with no
candidate of its own is free coverage even when the same direction was already
delivered at another placement — the 4:5 feed asset does not make the 9:16 story
asset a repair. Only a second render of that same placement costs one.

Regenerate from the design specification when the organizing idea, focal
subject, hierarchy, or canvas structure is wrong; use `edit_image` for a
localized defect on an otherwise strong candidate. When a limit is reached,
deliver that asset's best candidate. Counters do not carry across plan steps —
an asset that failed inspection in an earlier step is delivered with its defects
named, not retried on a fresh budget.

## Finish

Coverage is unconditional: every required asset must exist as a successful tool
result before you finish. One thing overrides it — a brand-specific *final* with
no verified logo. Before the user has chosen, that turn ends in the question
above with nothing rendered, and that counts as finished. After they choose a
reserved-space or unbranded route, render it, inspect it, and hand it back as a
concept draft; coverage then applies to the drafts they chose, and what stays
unfinished is the branded final, not this turn. Quality is what the budget
releases — when the budget
is spent and an asset still fails inspection, deliver its best candidate, name
the defect concretely (which text is misspelled, which element is clipped, which
logo is not authentic), and mark the answer's outcome `partial` so the user can
decide whether to spend another round.

A spent budget with complete coverage is terminal. The defects belong in the
answer text where the user reads them, and `missing_verification` stays empty —
naming a defect there reopens the loop the budget just closed, and another render
or replan is not the correct response to it. Call `final_answer` with
`outcome=partial` and stop.

Return only PNG or JPEG files that were actually created. When the run ends in a
question there are none, and the question is the whole answer — do not list or
describe files that do not exist. Otherwise lead with the files, then give one
concise line per asset identifying its communication angle and dimensions.
