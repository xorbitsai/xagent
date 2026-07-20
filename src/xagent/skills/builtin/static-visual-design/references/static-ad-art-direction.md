# Static Ad and Poster Art Direction

Use this reference for advertising creatives, campaign posters, promotional
social posts, announcement graphics, and other static visuals that must feel
designed rather than merely illustrated.

## Lock the communication system

Before rendering, reduce the brief to four decisions:

1. **Promise** — the one idea the audience should remember.
2. **Proof** — the fact, product, scene, or social evidence that makes it
   believable.
3. **Action** — the one next step the audience should take.
4. **Brand code** — the verified identity cues that make the work recognizable
   before the logo is read.

If a direction needs two promises, two unrelated hero images, or two CTAs, it
is not yet a direction. Resolve that conflict before generation.

## Choose a communication structure

Choose one structure because it serves the proposition. Do not combine several
structures to make the canvas look busier.

### Dominant proof

Use one number, milestone, result, short quote, or recognized fact as the main
visual mass. Pair it with imagery that demonstrates scale or consequence. Best
for social proof, launches, milestones, and concrete results.

### Offer reveal

Make the tangible benefit the fastest-read element, then show who it is for and
what action unlocks it. Avoid gift-box, confetti, price-tag, or coupon imagery
unless that metaphor fits the brand and does more than restate the word
"free."

### Product or object hero

Use one product, device, package, interface, or symbolic object as the focal
subject. Supporting callouts must point to real differentiators and remain
subordinate. Best when the thing itself is recognizable and desirable.

### Human outcome

Show the change in a person's life rather than a generic spokesperson holding
the product. Casting, setting, gesture, and moment must be specific to the
audience and promise. The person is evidence of the outcome, not decoration.

### Tension and resolution

Express a before/after, problem/solution, old/new, or friction/release contrast
inside one continuous composition. Do not request a split frame, comparison
grid, diptych, or multiple panels unless the final format explicitly requires
them; image models often turn those words into a contact sheet.

### Editorial provocation

Let one sharp line of copy and one surprising image or typographic gesture carry
the idea. Use this for cultural relevance, attitude, challenger positioning, or
an ownable campaign thought. The image and headline should create a third
meaning together instead of captioning one another.

### Testimonial or proof card

Build the composition around one short, credible statement, recognizable
source, and restrained supporting proof. Do not fabricate people, ratings,
press marks, or quotations.

### Information poster

Use a clear modular grid when dates, venues, schedules, speakers, features, or
instructions are genuinely required. Information posters may carry more text
than feed ads, but still need one dominant entry point and an unmistakable scan
order.

## Specify the layout before prompting

Every direction needs a compact design specification:

- canvas ratio and placement context;
- chosen communication structure;
- focal subject and its approximate share of the canvas;
- primary, secondary, and tertiary information roles;
- intended scan path, such as top-left to center to CTA;
- image zone, type zone, brand zone, and deliberate negative space;
- type character, headline line count, alignment, and contrast strategy;
- dominant, supporting, and accent color roles;
- production finish and material qualities;
- exact reference assets and what each reference controls;
- explicit exclusions that protect this concept from generic defaults.

Use proportions, relationships, and zones rather than micromanaging every
pixel. A useful default visual-weight budget is roughly 55–70% for the focal
idea, 20–30% for supporting communication, and the remainder for brand, CTA,
and mandatory legal text. Break this when the concept demands it, but never let
all elements compete at the same weight.

Design for a three-pass read:

- **one second:** proposition or visual hook;
- **three seconds:** proof or offer comprehension;
- **five seconds:** brand, action, and required qualification.

If the scan order is not obvious in the written specification, generation will
not fix it.

## Control typography and copy load

Treat text as a limited visual resource. For a typical feed ad or promotional
poster, prefer:

- one exact headline, ideally no more than two short lines;
- zero or one short support line;
- one CTA of roughly two to five words;
- only legally or operationally required fine print.

Do not put strategy labels, markdown, quotation marks used only for prompting,
alternative headlines, rationale, or production notes on the canvas. Never ask
the image model to choose between copy options. Longer event information belongs
in a deliberate information-poster grid, not in an ad layout.

Use no more than three obvious hierarchy levels. Contrast levels by scale,
weight, position, color, and whitespace; do not depend on glow, bevel, shadow,
outline, and extrusion simultaneously. Display type may be expressive, but body
and qualification text must remain calm and legible.

## Use a one-canvas generation contract

Each generation call must request one finished composition for one placement.
State this positively and negatively:

> Create one single final ad on one continuous canvas. Show one composition
> only. Do not create a contact sheet, moodboard, grid of options, multiple
> versions, before-and-after panels, repeated layouts, mockup presentation, or
> duplicated headline.

Do not use prompt phrases such as "three concepts," "two variations,"
"split-frame," "option A/B," or "layout exploration" inside a render call.
Those belong in planning; render each locked direction in a separate call.

References must already be resolved before rendering begins. Every generation
step must depend on the shared brand-and-brief step and receive the locked
direction, exact copy, and relevant reference assets. Do not search for the logo
in parallel with generation.

## Make the concept brand-specific

Apply the substitution test: mentally replace the named brand with its closest
competitor. If the image, proposition, and art direction still work unchanged,
the direction is generic.

Brand specificity can come from a real product truth, audience behavior,
recognizable setting, distinctive visual code, ownable metaphor, or verified
campaign language. A brand-colored gradient, generic crowd, smiling model,
phone mockup, glowing particles, city skyline, ribbon, or confetti does not make
an idea specific by itself.

For people-led work, define a credible moment instead of demographic shorthand.
For abstract work, explain what each form or motion represents. Do not translate
"community," "technology," or "growth" automatically into purple particles,
data streams, glowing faces, or orbiting dots.

## Review like a creative director

Automatic rejection overrides subjective scoring. Reject a candidate when it
has any of these failures:

- more than one ad, a contact sheet, multiple panels, or presentation mockups
  when one asset was requested;
- duplicated, omitted, invented, or misspelled copy;
- a fake, misspelled, duplicated, or visibly distorted identity mark;
- an unverified claim, offer detail, URL, date, rating, quote, or legal line;
- no dominant entry point, unclear scan order, or unreadable essential text;
- a focal image that merely decorates or repeats the headline;
- a concept that passes the competitor-substitution test unchanged;
- clipped elements, accidental overlaps, malformed subjects, watermarks, or
  unrelated lettering.

For candidates without an automatic failure, score five dimensions from 1–5:

1. proposition clarity;
2. hierarchy and scan performance;
3. brand specificity and factual integrity;
4. visual craft and restraint;
5. placement fitness and action clarity.

A technically clean result with a weak proposition or interchangeable visual
idea is not production-ready.

## Decide whether to edit or regenerate

Use `edit_image` only for a localized defect on an otherwise strong,
single-canvas composition: a small crop issue, one contained object problem, a
minor color imbalance, or removable artifact.

Regenerate from the locked design specification when the failure is structural:
contact-sheet output, duplicated layout, wrong hierarchy, excessive copy,
generic concept, wrong focal subject, incoherent visual metaphor, or multiple
text errors. Editing a structurally wrong image usually compounds the failure.
