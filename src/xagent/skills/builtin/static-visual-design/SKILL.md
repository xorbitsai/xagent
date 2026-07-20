---
name: static-visual-design
description: |
  Generate polished static visual designs as complete PNG or JPEG assets using
  image generation and editing as the primary creative engine. Use for posters,
  social media graphics, advertising creatives, campaign key visuals, event or
  announcement cards, banners, and placement variants where art direction,
  typography, hierarchy, brand fidelity, and visual quality matter. Do not use
  it to invent a logo, create video, manage ad accounts, generate a standalone
  illustration or photo with no designed layout, or answer copy-only requests.
---

# Static Visual Design

Produce the finished visual with image generation and editing. Let the image
model solve the composition, type treatment, atmosphere, and graphic language
together instead of reducing the work to a generated background plus an HTML
layout.

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

Give every direction one single-minded proposition and one coherent organizing
visual idea. Evaluate directions for brand fit, stopping power, glance-level
clarity, offer comprehension, and factual safety. Do not let a creative become
two disconnected ads stacked in one canvas.

Do not interpret the singular nouns "an ad," "a poster," or "a social post" as
an instruction to explore only one direction. When the creative direction is
open, render two or three candidates so the user can choose. When the user
explicitly requests exactly one final asset, still compare possible directions
and render the strongest one without exposing unnecessary internal deliberation.

## Use brand and reference assets intentionally

Inspect relevant uploaded or workspace images with `understand_media`. Pass
useful product, campaign, style, or layout references to image generation or
editing so the result belongs to the intended visual world.

For work naming a real brand, resolve the brand identity before treating the
asset as final. Look first in user uploads and the task workspace. When browsing
is available, use only an official brand site or brand kit as an external source;
otherwise ask the user for the asset. A visually plausible search result is not
proof that a logo is authentic.

Treat identity-critical assets differently:

- Use an official supplied logo as the source of truth. It may be included as a
  generation reference for brand language, proportions, and reserved placement,
  but do not trust a generated or edited recreation as the final logo.
- Preserve QR codes, certification marks, sponsor marks, UI screenshots, and
  other exact assets pixel-for-pixel.
- Unless the user explicitly requests an unbranded or logo-free concept, a
  brand-specific final requires a verified logo. If none is available, ask for
  it and keep any interim output clearly labeled as a concept draft. Do not mark
  the requested branded asset complete, and never typeset or invent a substitute
  logo.

## Generate the complete creative

Use `generate_image` to create the full designed asset, including the intended
composition, typography, hierarchy, graphic elements, and user-supplied copy.
Prompt with:

- the organizing visual idea and emotional tone;
- exact text to render, quoted clearly;
- hierarchy and approximate placement, without over-constraining every pixel;
- target aspect ratio and viewing context;
- relevant reference images;
- any quiet zones required for an exact logo or QR overlay;
- exclusions such as fake logos, fake QR codes, watermarks, and unrelated text.

For campaign directions developed above and for plural creative requests, make
the concepts materially different before making size variants. Vary the idea,
composition, subject, image treatment, and hierarchy—not merely crop or accent
color. Cosmetic resizes are not distinct concepts.

Generate each materially different aspect ratio for that format. Do not force a
landscape master into square, portrait, or story placements when a fresh
composition would be stronger.

Avoid generic AI-ad decoration unless the brief calls for it: purple-blue
gradients, giant centered white type, heavy shadows, neon glows, arbitrary
waves, confetti, floating spheres, and excessive sparkles. Favor one clear
focal point and glance-level comprehension.

## Inspect and iterate with image tools

Inspect every candidate with `understand_media`, checking:

- exact spelling, numbers, dates, CTA, offer, and disclaimer;
- whether the copy reads naturally and expresses the intended campaign angle;
- hierarchy, contrast, and thumbnail-size legibility;
- crop, balance, edge clearance, and platform safe zones;
- consistency with the supplied references and recognizable brand language;
- accidental pseudo-logos, fake QR codes, watermarks, malformed objects, or
  unrelated lettering.

Use `edit_image` to refine a strong candidate or regenerate when the organizing
idea is weak. Correct copy errors through editing or regeneration and inspect
again. Shorten nonessential copy only when the brief permits it. A successful
generation call alone is not proof that the asset is finished.

## Add exact assets as the final layer

After the creative itself passes inspection, composite official logos and QR
codes from their original files as a deterministic final layer. Use
`logo_overlay` when it fits the placement. Never leave a generated pseudo-logo
underneath the official mark; remove it or regenerate the creative first.

Do not use HTML/CSS plus browser screenshots for ordinary poster, ad, banner,
or social-creative generation. Use HTML only when the user explicitly requests
an editable HTML/template deliverable; it is not the default fallback for text
layout.

## Apply the completion gate

Do not enter `final_answer` until every requested visual exists as a successful
tool result and the final files pass inspection. For a brand-specific final,
also require the exact verified logo to be visibly composited unless the user
explicitly requested a logo-free result. A brand name rendered as ordinary text
does not satisfy this requirement.

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
