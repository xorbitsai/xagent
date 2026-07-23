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

## Stay within the commercial-creative scope

Use this skill only when the requested deliverable communicates marketing,
promotional, campaign, event, or brand-facing material. A request to explain a
concept with an image, comparison graphic, educational infographic, technical
diagram, chart, or data visualization is outside this skill even when the user
asks for a polished PNG or JPEG. Use the general image-generation workflow or a
more specific explanatory-visual skill instead.

Do not use this skill to invent a logo, create video, manage ad accounts,
generate a standalone illustration or photo with no designed commercial
layout, or answer copy-only requests.

Produce the finished visual with image generation and editing. Let the image
model solve the composition, type treatment, atmosphere, and graphic language
together instead of reducing the work to a generated background plus an HTML
layout.

For advertising creatives, campaign posters, promotional social posts,
announcement graphics, and other commercially designed static visuals, read
`references/static-ad-art-direction.md` with `read_skill_doc` before defining
directions or rendering. Use its communication structures, layout specification,
one-canvas contract, and rejection rules. Do not treat the reference as a menu
of decorative styles.

## Establish the brief

Identify the minimum useful brief before generating:

- communication goal, audience, and intended response;
- primary message, supporting copy, CTA, and required disclaimer;
- output channel and target aspect ratio or dimensions;
- available logo, QR code, product, brand-guide, campaign, and reference assets.

When available, ground creative direction in real brand materials, product
facts, customer language, prior winning creative, and performance evidence.
Treat those inputs as evidence, not as permission to invent adjacent claims.

Distinguish exact supplied copy from facts that still need copywriting. Preserve
copy the user marks as exact. When the user supplies facts but not final wording,
write concise, idiomatic campaign copy instead of mechanically restating the
brief. Avoid defensive or generic claims such as "you cannot be wrong," vague
superlatives, and urgency unsupported by the offer.

Do not create a separate brief document unless the user asks for one. Infer
low-risk creative choices, but never invent prices, milestones, performance
claims, offer mechanics, eligibility, dates, URLs, legal copy, or brand rules.
Verify time-sensitive claims when necessary. If exact campaign terms are absent,
keep the visual to the claims the user actually supplied.

## Develop campaign directions before rendering

For an open-ended brand campaign, poster, promotion, or advertising request,
turn the brief into two or three genuinely different communication angles
before committing to a visual. Useful angle families include milestone pride or
social proof, offer-led value, product benefit, emotional identity, and urgency,
but choose only those supported by the brief.

Keep this initial direction-setting in one coherent planning pass. One creative
lead should define and compare the full set of directions; do not delegate
open-ended ideation to independent agents or parallel nodes. Independent
ideation tends to converge on the same obvious brand cues while changing only
surface decoration. Parallel execution is useful only after the briefs are
locked: each executor receives a distinct proposition, structure, focal device,
production finish, and explicit exclusions, and must not reinterpret the other
directions.

Represent this order in the execution plan. Brand/reference acquisition is a
shared prerequisite, creative direction and design specification depend on that
grounding, and every render depends on the locked specification. Do not search
for identity assets in parallel with artifact generation. Inspection depends on
the render results. Independent renders may run in parallel only after all
shared inputs are resolved.

Never bake asset-availability conclusions into planned work before the
acquisition step has run. Do not write instructions such as "use the brand name
in typography since no logo asset is available" into a step definition; whether
a verified logo exists is an outcome of asset resolution, and how the brand is
represented must stay open until that outcome is known.

Give every direction one single-minded proposition, one visual device, and one
structural approach. The proposition explains why the audience should care;
the visual device creates the memorable image; the structure controls how the
message is scanned. Evaluate directions for brand fit, stopping power,
glance-level clarity, offer comprehension, and factual safety. Do not let a
creative become two disconnected ads stacked in one canvas.

Invent each direction's visual device deliberately instead of jumping from an
angle name to a layout. Use generative moves — dramatize the verb in the
proposition, give the proof number physical form in a real place, literalize
the offer as an object or event, stage a recognizable cultural moment, shift
scale, or cast a brand asset as the protagonist — and sketch more device
candidates than you have direction slots before keeping the strongest. An
angle name like "milestone pride" is not a device; "the milestone staged as a
rally the viewer is invited to join" is. The art-direction reference expands
these moves.

Do not create directions by accumulating a universal blacklist of subjects or
styles. No device is inherently right or wrong: a person, product, phone,
mascot, gradient, confetti field, typographic composition, or empty color field
can all work when it performs a clear role in the proposition. Exclude an
element only when it conflicts with the chosen direction, brand evidence, or
message—not because it appeared in a generic list.

When client taste is unknown, build a creative-risk ladder instead of betting
everything on one aesthetic:

1. **Brand-safe evolution** — immediately recognizable, with a cleaner and more
   disciplined use of familiar brand cues.
2. **Contemporary reinterpretation** — preserves identity while changing the
   composition, image language, or type system meaningfully.
3. **Bold exploration** — uses a more ownable campaign metaphor or unexpected
   art direction while remaining factually and strategically on brief.

Do not make every option safe or every option experimental. The ladder gives a
client a comfortable choice and reveals how much change they will accept.

Do not interpret the singular nouns "an ad," "a poster," or "a social post" as
an instruction to explore only one direction. When the creative direction is
open, render two or three candidates so the user can choose. When the user
explicitly requests exactly one final asset, still compare possible directions
and render the strongest one without exposing unnecessary internal deliberation.

## Choose a designed structure

Do not default every request to a stock portrait or product mockup beside large
headline text. Select a structure that serves the message, such as a dominant
stat, a single hero scene, a product or object spotlight with callouts, a
before/after contrast, a problem/solution transition, an editorial typographic
composition, a proof or quote card, or an information-rich event poster. Treat
these as a varied vocabulary, not fixed templates.

Make the headline and image divide the communication work. The image should add
proof, tension, emotion, scale, contrast, or surprise instead of literally
captioning the headline. Prefer one ownable campaign device over a collage of
unrelated symbols. Match information density to the placement: feed ads and
banners usually need a short hook and one support line; event and information
posters may carry more detail if the hierarchy remains obvious.

Treat the visual device as a mechanism, not a medium label. Photography,
illustration, objects, characters, expressive type, spatial relationships, and
material treatments are all valid, but the direction must state what the
device makes the audience notice or understand. “Bold typography,” “brand
gradient,” “large number,” and “premium styling” are treatments, not complete
ideas on their own.

Before rendering, turn each direction into a compact design specification:
canvas and placement, communication structure, focal subject, visual-weight
distribution, three-level information hierarchy, scan path, the depth stack
(environment, atmosphere, focal subject, supporting graphics, typography,
brand — each layer decided, not defaulted to empty), typography character and
line count, color roles, production finish, reference responsibilities, and
explicit exclusions. If the scan order is not
clear in the specification, the direction is not ready to generate. Keep the
specification compact enough to survive intact into the render prompt. Do not
create a separate directions or strategy artifact unless the user asks for it,
and do not make reading a required reference document its own deliverable step.

## Use brand and reference assets intentionally

Inspect relevant uploaded or workspace images with `understand_media`. Pass
useful product, campaign, style, or layout references to image generation or
editing so the result belongs to the intended visual world.

For work naming a real brand, resolve the brand identity before rendering final
candidates. Look in user uploads and the task workspace. If no verified asset is
available, ask the user for it. A visually plausible search result is not proof
that a logo is authentic.

Treat identity-critical assets differently:

- Use an official supplied logo as the source of truth. Include it as a
  generation reference whenever the image tool supports references so palette,
  brand language, proportions, and reserved placement influence the whole
  design. Pass the actual path, URL, or file_id through the image tool's
  `images` argument (or use `edit_image` directly); naming the asset only in the
  prompt does not attach it. Still do not trust a generated or edited recreation
  as the final logo.
- Treat QR codes, certification marks, sponsor marks, UI screenshots, and other
  exact assets as non-generative inputs. This runtime does not provide
  deterministic compositing, and generative rendering cannot preserve these
  assets pixel-for-pixel. If the final placement requires exact reproduction,
  explain the limitation and ask the user to arrange deterministic
  post-processing; never claim the generated candidate is an exact final.
- Unless the user explicitly requests an unbranded or logo-free concept, a
  brand-specific final requires a verified logo. If none is available, ask for
  it and keep any interim output clearly labeled as a concept draft. Do not mark
  the requested branded asset complete, and never typeset or invent a substitute
  logo.

Separate stable identity cues from temporary campaign styling. Stable cues may
include the official logo, recurring color relationships, typography character,
product imagery, graphic proportions, and tone of voice. Temporary styling may
include a particular gradient, metallic 3D type, bevel, glow, ribbon, confetti,
swoosh, or seasonal campaign motif. Use several recent official references when
available to infer what recurs; do not copy every effect from one old banner.

When official references look dated, familiarity still matters, but datedness
is not a brand requirement. Preserve recognition through the stable cues and
modernize hierarchy, whitespace, type discipline, image quality, depth, and the
number of competing effects. A brand-safe direction should feel like the next
campaign from the same brand, not a replica of its oldest promotion.

## Generate the complete creative

Use `generate_image` to create the full designed asset, including the intended
composition, typography, hierarchy, graphic elements, and user-supplied copy.
Prompt with:

- the organizing visual idea and emotional tone;
- the chosen structure, focal subject, and specific visual device;
- exact text to render, quoted clearly;
- hierarchy and approximate placement, without over-constraining every pixel;
- target aspect ratio and viewing context;
- relevant brand, product, campaign, and style references, with a clear statement
  of what to borrow from each;
- the intended production finish, such as documentary photography, tactile
  collage, screen print, editorial type, clean product render, or bold flat art;
- any quiet zones required around identity-critical asset placement;
- exclusions such as fake logos, fake QR codes, watermarks, and unrelated text.

Each call must request one finished placement on one continuous canvas. Render
each direction separately. Explicitly exclude contact sheets, moodboards,
option grids, multiple versions, presentation mockups, repeated layouts,
split-frame compositions, and duplicated headlines. Do not put words such as
"variations," "option A/B," or "layout exploration" in a render prompt.

Keep copy load proportional to the format. A typical feed ad should have one
exact headline of no more than two short lines, at most one short support line,
one concise CTA, and only required fine print. The offer is expressed through
the headline or the CTA, never as an additional standalone "offer highlight"
element; if a brief or step description enumerates headline, support, offer,
and CTA as four separate text blocks, merge them back into this budget before
rendering. Never send alternative copy, strategy labels, markdown, rationale,
or production notes for rendering.

Write a concept-specific prompt for every direction. Do not reuse a generic
"modern professional ad" prompt with only the colors and headline changed.

Before generation, compare the locked briefs as a set. Each direction must
differ on at least three of these axes: focal subject, structural approach,
visual metaphor, production finish, palette balance, and type strategy. Do not
reuse the same gradient, display treatment, and decorative motif across the
set. Compare the actual render prompts, not merely their direction names or
strategy labels. If the prompts describe the same focal mechanism and layout,
revise them before rendering. Distinct propositions do not make distinct
directions when every focal subject is the same device: a set in which each
direction's hero is a large typographic number or headline is one direction
wearing three copy variations. For an open brief, at most one direction in a
set may use pure typography as its focal subject; the others must be carried
by a scene, person, place, object, or material device. The same limit applies
to production depth: a deliberately flat, minimal treatment with an empty
environment and atmosphere is one direction at most, never the unexamined
default for the whole set. When the user explicitly requests a typographic,
minimalist, or otherwise constrained system, follow the brief and create the
variation inside that system — through composition, scale, material, and
color — instead of forcing scenes into it.

For campaign directions developed above and for plural creative requests, make
the concepts materially different before making size variants. Vary the idea,
composition, subject, image treatment, and hierarchy—not merely crop or accent
color. Cosmetic resizes are not distinct concepts.

Choose each aspect ratio from the actual placement, never as a habit, and let
the medium set the vocabulary: social campaigns default to 4:5 feed and 9:16
story (1:1 only when a channel specifically requires square); print, poster,
out-of-home, and display work follows its placement's actual dimensions, such
as standard banner and billboard sizes. When the user has not named channels,
either ask or cover the likely placements deliberately and say which asset
serves which placement. Do not silently default every render to one square
canvas. Compose natively for the chosen ratio: a square
canvas wants a centered or radial composition built around its focal device,
not a vertical poster stack with dead side margins.

Generate each materially different aspect ratio for that format. Do not force a
landscape master into square, portrait, or story placements when a fresh
composition would be stronger.

Do not forbid familiar devices merely because they are common in generated ads.
A gradient, portrait, phone, giant headline, glow, wave, confetti field, or
sparkle can be the right choice. Reject it only when it is an unexamined default,
does no communication work, or could move unchanged to a competitor. Favor one
clear focal point, a recognizable silhouette at thumbnail size, and a visual
idea specific enough that another brand could not use it unchanged.

Treat stacked display effects as a warning sign: metallic 3D lettering plus
bevels, glow, drop shadows, ribbons, swooshes, confetti, and multiple headline
blocks rarely become stronger by accumulation. Keep only the effects that serve
the chosen visual device and remove the rest.

## Inspect and iterate with image tools

Inspect every candidate with `understand_media`, checking:

- exact spelling, numbers, dates, CTA, offer, and disclaimer;
- whether the copy reads naturally and expresses the intended campaign angle;
- hierarchy, contrast, and thumbnail-size legibility;
- whether the headline and image complement rather than repeat each other;
- whether the visual device feels ownable instead of stock or interchangeable;
- crop, balance, edge clearance, and platform safe zones;
- consistency with the supplied references and recognizable brand language;
- accidental pseudo-logos, fake QR codes, watermarks, malformed objects, or
  unrelated lettering.

Automatically reject contact sheets, multiple ads in one image, duplicated
layouts or headlines, fake or duplicated logos, wrong or invented copy,
unverified claims, unclear hierarchy, and clipped or overlapping essentials.
Do not rationalize these as stylistic choices.

Use `edit_image` only to refine a strong, single-canvas candidate with a
localized defect. Regenerate from the locked design specification when the
organizing idea, focal subject, hierarchy, copy load, or canvas structure is
wrong, or when several text errors appear. Correct permitted copy errors and
inspect again. A successful generation call alone is not proof that the asset
is finished. Compare the candidates side by side and discard the safest generic
option even when it is technically clean.

## Handle identity assets without blind post-processing

Use official logos and other brand assets as generation or editing references
when the image model supports them. Inspect the result closely. Never add a
second logo over a generated pseudo-logo, and never treat a generic typed brand
name as proof of fidelity; remove the artifact or regenerate the creative.

Do not add a deterministic compositing step for ordinary branded visuals.
Apply the identity-asset rule above whenever exact reproduction is required;
do not pretend a generative result is exact.

Do not use HTML/CSS plus browser screenshots for ordinary poster, ad, banner,
or social-creative generation. Use HTML only when the user explicitly requests
an editable HTML/template deliverable; it is not the default fallback for text
layout.

## Apply the completion gate

Do not enter `final_answer` until every requested visual exists as a successful
tool result and the final files pass inspection. A successful `generate_image`
or `edit_image` call is never completion evidence by itself, in any execution
form: when render work is defined as a plan step, the step's completion
criteria and termination condition must require that the result passed
`understand_media` inspection with no automatic rejection — never "the
generation tool returned success". When no plan is involved, apply the same
rule to the direct decision to stop. For a brand-specific final, reject fake,
duplicated, misspelled, or visibly distorted identity marks. When the user
explicitly requires an exact logo, a brand name rendered as ordinary text does
not satisfy the requirement.

Reject or continue iterating on an output that is merely polished but generic,
uses two disconnected visual ideas, weakens the supplied fact into awkward
copy, omits a required brand asset, or provides fewer meaningful directions
than the open brief calls for. Tool success is evidence that an image was
created, not that the campaign deliverable is complete.

## Deliver

Return only final PNG or JPEG files that were actually created successfully.
Lead with the files, then identify the communication angle and dimensions of
each asset in one concise line so the user can compare candidates. Do not
present a prompt, brief, HTML intermediate, or claimed file path instead of the
requested image.
