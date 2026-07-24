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

## Invent the campaign device

A proposition becomes an ad only when it finds its device: the one image that
dramatizes the promise. Do not jump from an angle name ("milestone pride",
"offer-led") straight to a layout; generate device candidates deliberately.
These are generative moves, not a menu — each move can produce many different
devices from the same brief:

- **Dramatize the verb.** Take the action word inside the proposition —
  switch, join, port, save, unlock, upgrade — and stage it literally: a crowd
  mid-movement, a door standing open, a queue crossing over.
- **Make the number a thing.** Give proof physical form in a real place: a
  human sea, a monument, a stadium filled to the exact count, product units
  assembled into the figure.
- **Literalize the offer.** Render the benefit as a tangible object or event —
  a gift revealed, a meter refilling, a bill torn in half — but only when the
  object adds drama or meaning beyond restating the offer word.
- **Stage a recognizable moment.** Borrow a civic or cultural scene the
  audience lives in — a rally, a festival, a countdown, a homecoming — and
  let the brand host it.
- **Shift the scale.** Make the small cause huge or the huge consequence
  small: a thumb-sized product casting a city-block shadow, a city skyline
  balanced on one small object the brand sells.
- **Let a brand asset act.** Give the mascot, logo form, or brand color a
  role in the story instead of a corner placement.

For one brief, generate more device candidates than direction slots — five to
eight sketches for a set of three — then keep the ones that dramatize
genuinely different propositions and discard any that merely decorate. A kept
direction states its device in one sentence, such as "the 1.4M milestone
staged as a rally the viewer is invited to join." Every device must still
pass the competitor-substitution test and carry verified claims only.

## Choose a communication structure

Choose one structure because it serves the proposition. Do not combine several
structures to make the canvas look busier.

### Dominant proof

Use one number, milestone, result, short quote, or recognized fact as the main
visual mass. Pair it with imagery that demonstrates scale or consequence. Best
for social proof, launches, milestones, and concrete results.

The pairing is mandatory, not decorative: the number must be physically
integrated with a scene, material, environment, or crowd that makes the scale
believable — built into a real location, towering over a gathering, assembled
from product units, carved into a material the brand owns. A large numeral
floating on a plain or gradient background is typography, not proof, and is an
incomplete direction. When the proposition is local social proof, include one
recognizable cue of the actual market (a landmark, streetscape, or audience
context) so the claim reads as "here", not anywhere. The headline must then add
meaning the numeral does not already carry; restating the number in words is a
wasted line.

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
- the depth stack: environment, atmosphere, focal subject, supporting
  graphics, typography, and brand (see below);
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

## Build the canvas as a depth stack

Think in layers the way a working designer builds the file. Every direction
must consciously decide six layers:

1. **Environment** — the world behind the message: a place, a surface, a
   material, a crowd, or a deliberate flat field.
2. **Atmosphere** — light direction, depth of field, haze, particles, weather;
   whatever binds the subject to the environment and gives the canvas air.
3. **Focal subject** — the hero device, ideally physically situated in the
   environment rather than pasted over it.
4. **Supporting graphics** — callouts, secondary proof, texture, pattern.
5. **Typography** — the headline, support line, and CTA hierarchy.
6. **Brand** — logo lockup, mandatories, fine print.

Leaving a layer empty is a legitimate design decision only when the
specification records it as one. For an open brief, a deliberately flat,
minimal treatment — empty environment and atmosphere, typography carrying
everything — may appear at most once in a set of directions; if two or more
specifications read "solid or gradient background, no atmosphere", the set
has collapsed into layout variants of a single idea. When the brief itself
calls for a typographic or minimalist system, keep the stack discipline but
create the depth inside that system: paper texture, ink behavior, embossing,
shadow, and material light are atmosphere too. Write the render prompt in the same order
the stack is built — environment, light, subject, supporting graphics, exact
copy, brand — so the image model receives a scene to construct instead of a
list of text blocks to place.

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

## Follow the main skill's one-canvas generation contract

Use the one-canvas render contract from `SKILL.md` without adding a competing
version here. Compare multiple directions during planning, then render each
locked direction in a separate generation call.

References must already be resolved before rendering begins. Every generation
step must depend on the shared brand-and-brief step and receive the locked
direction, exact copy, and relevant reference assets. Do not search for the logo
in parallel with generation. Attach acquired references through the image
tool's actual image-input parameter; a filename mentioned only inside prose is
not a model input.

## Write the render prompt as an art direction brief

Describe the ad the way an art director briefs a photographer or illustrator:
the scene, the focal subject and what it is made of, the light, the material
and production finish, the mood, and the exact copy in quotation marks. Give
placement as loose spatial relationships ("the numeral anchors the lower left,
the skyline recedes behind it") rather than engineering zones.

Do not write the prompt as a UI specification. Percentage zone maps ("top 15%:
empty, center 15-55%: hero"), CSS-like color-and-weight tables, and web
interface vocabulary produce flat screen layouts instead of designed
advertising. Never describe the CTA as a "button", "pill", or other interface
control: in a static ad the CTA is a graphic device — a color band, a painted
block, a ticket stub, a stamped mark — sized as one short line that cannot
wrap.

Keep the negative prompt short and factual. Some image providers ignore the
negative prompt entirely, so any exclusion the direction depends on must also
be expressed positively in the prompt ("empty sky above the skyline", "the
only text on the canvas is..."). It may contain render-quality defects
(misspelled or distorted text, watermarks, contact sheets, extra panels,
malformed anatomy, fake logos or QR codes) plus at most two or three
exclusions that the locked direction specifically requires. Do not blanket-ban
whole families of imagery — people, crowds, skylines, silhouettes, glow, 3D,
confetti — as a routine safety list; stripping every scene-making device from
the model is how an ad collapses into a bare text layout. Exclude a device only
when the design specification records why it conflicts with this direction.

## Make the concept brand-specific

Apply the substitution test: mentally replace the named brand with its closest
competitor. If the image, proposition, and art direction still work unchanged,
the direction is generic.

Brand specificity can come from a real product truth, audience behavior,
recognizable setting, distinctive visual code, ownable metaphor, or verified
campaign language. A brand-colored gradient, generic crowd, smiling model,
phone mockup, glowing particles, city skyline, ribbon, or confetti does not make
an idea specific by itself. These devices are not prohibited; require each one
to perform a clear strategic or compositional role.

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
- a focal image that merely decorates or repeats the headline, or a headline
  that restates in words what the focal element already shows;
- more than one CTA competing on the canvas, or a CTA whose text wraps or
  renders as a web interface button;
- an environment or atmosphere layer the specification called for that the
  render dropped, leaving a flat layout the direction did not choose;
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
