---
name: presentation
description: "Generate and edit PowerPoint presentations (.pptx). Use for: creating slide decks, pitch decks, or presentations from scratch; reading, parsing, or extracting content from .pptx files; editing, modifying, or updating existing presentations; combining or splitting slide files; working with templates, layouts, speaker notes, or comments. Trigger when user mentions 'deck,' 'slides,' 'presentation,' or references a .pptx filename."
---

# Presentation Generator

Generate PowerPoint presentations using JavaScript code with the pptxgenjs library.

## ⚠️ CRITICAL REQUIREMENTS - READ FIRST

**YOU MUST FOLLOW THESE RULES - NO EXCEPTIONS:**

1. **ONLY use the 4 predefined themes below** (NOVA, ORBIT, PULSE, MINIMA)
2. **NEVER create custom color variables** like `const colorAccent = "0078D7"`
3. **NEVER use hardcoded hex values** in your code - always reference `theme.xxx`
4. **ALWAYS include the `#` prefix** in hex colors (e.g., `#EF4444` not `EF4444`)
5. **Slide 0 MUST use `theme.cover`** - first slide is the cover with dark/high-contrast background
6. **Slides 1+ MUST use `theme.content`** - all other slides use light/readable background

**WRONG:**
```javascript
const colorAccent = "0078D7";  // ❌ WRONG - custom color
slide1.addText("Title", { color: "363636" });  // ❌ WRONG - hardcoded hex
pres.addSlide();
pres.background = { color: theme.cover.background };  // ❌ WRONG - cover on non-first slide
```

**CORRECT:**
```javascript
const theme = {
  cover: {
    background: '#0A0F1C',
    title: '#FFFFFF',
    subtitle: '#94A3B8',
    accent: '#7C3AED'
  },
  content: {
    background: '#F6F7FB',
    primary: '#0A0F1C',
    secondary: '#5B6475',
    accent: '#7C3AED',
    text: '#0A0F1C'
  }
};
// Slide 0 (first slide, no addSlide) - use theme.cover
pres.background = { color: theme.cover.background };
pres.addText("Title", { color: theme.cover.title });

// Slide 1+ (after addSlide) - use theme.content
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText("Content", { color: theme.content.primary });
```

---

## Execution Rules

**CRITICAL: MANDATORY constraints for stable presentation generation:**

1. **USE PREDEFINED THEMES ONLY**: You MUST use one of the 4 predefined themes (NOVA, ORBIT, PULSE, MINIMA) from the "Predefined Themes" section below. NEVER define custom colors or use hardcoded hex values like `"003366"` or `color: "FF0000"`.

2. **Theme Object Format**: Always define theme as:
   ```javascript
   const theme = {
     background: '#XXXXXX',
     primary: '#XXXXXX',
     secondary: '#XXXXXX',
     accent: '#XXXXXX',
     text: '#XXXXXX'
   };
   ```
   Then reference colors as `theme.background`, `theme.primary`, etc.

3. **Slide Creation**: Always call `pres.addSlide()` before adding content to a new slide (except the first slide)

4. **Font Consistency**: Keep font sizes consistent: titles 40-56pt, subtitles 24-32pt, body text 16-20pt

5. **Slide Bounds**: Never position elements outside 10 x 7.5 inch slide area (x: 0-10, y: 0-7.5)

6. **Layout Simplicity**: Prefer clean, simple layouts over dense or decorative designs

7. **Theme Selection**: Choose theme based on presentation context (see Theme Selection Guide below)

8. **Visual Hierarchy**: Slides must follow clear visual hierarchy: Title → Section Heading → Body → Accent Highlight. Avoid mixing too many font sizes on a single slide.

9. **Image Sizing**: ALWAYS specify both width AND height, or use sizing to ensure images stay within bounds:
   - Use `w` and `h` together to control exact dimensions
   - Or use `sizing: { type: 'contain', w: 8, h: 5 }` to fit within bounds while maintaining aspect ratio
   - NEVER use only `w` or only `h` without the other - image may overflow
   - Keep images within content area: x: 0-10, y: 0-7.5 inches

## Layout Zones

For consistent slide layouts:

| Zone | X/Y Range | Purpose |
|------|-----------|---------|
| Safe content area | x: 0.5-9.5, y: 0.5-7 | Main content, images, data |
| Title area | y: 0.5 - 1.5 | Slide titles and headings |
| Content area | y: 1.5 - 5.5 | Main content, bullets, data |
| Footer area | y: 6.0 - 7.0 | Footer text, page numbers, notes |

**Note**: Title slides may use vertically centered positioning (y ≈ 3). All other slides must follow layout zones.

## Theme Selection Guide

| Context | Recommended Theme | Why |
|---------|------------------|-----|
| Strategy / AI narrative / Investor deck | **NOVA** | Large titles, generous whitespace, authoritative feel |
| Technical deep dive / Architecture / Dev | **ORBIT** | Dark background, strong contrast, clean technical aesthetic |
| Metrics-heavy / Growth / Business performance | **PULSE** | Bold KPI emphasis, high-contrast numbers, data-focused |
| Founder story / Brand / Minimalist | **MINIMA** | Extremely clean, typography-driven, minimal decoration |

**Default**: Use NOVA if context is unclear or not specified.

## Predefined Themes

All presentations MUST use one of these themes.

**Each theme has TWO variants:**
- **Cover (slide 0)**: Dark background, high visual impact, emotional
- **Content (slides 1+)**: Light background, clean, readable

| Theme | Positioning | Cover Colors | Content Colors | Style |
|-------|------------|-------------|----------------|-------|
| **NOVA** (Default) | Strategy / AI / Investor | `bg: #0A0F1C`, `title: #FFFFFF`, `accent: #7C3AED` | `bg: #F6F7FB`, `primary: #0A0F1C`, `accent: #7C3AED` | Dark cover, light content |
| **ORBIT** | Technical / Architecture / Dev | `bg: #0B1220`, `title: #F1F5F9`, `accent: #22D3EE` | `bg: #0F172A`, `primary: #F1F5F9`, `accent: #22D3EE` | Dark cover, lighter dark content |
| **PULSE** | Metrics / Growth / Performance | `bg: #111827`, `title: #FFFFFF`, `accent: #EF4444` | `bg: #FFFFFF`, `primary: #111827`, `accent: #EF4444` | Dark cover, white content |
| **MINIMA** | Founder / Brand / Minimalist | `bg: #111111`, `title: #FFFFFF` | `bg: #FAFAFA`, `primary: #111111`, `accent: #000000` | Black cover, off-white content |

### Color Reference Table

| Purpose | NOVA Cover | NOVA Content | ORBIT Cover | ORBIT Content |
|---------|-----------|--------------|-------------|----------------|
| Background | `#0A0F1C` | `#F6F7FB` | `#0B1220` | `#0F172A` |
| Title/Primary | `#FFFFFF` | `#0A0F1C` | `#F1F5F9` | `#F1F5F9` |
| Subtitle/Secondary | `#94A3B8` | `#5B6475` | `#94A3B8` | `#94A3B8` |
| Accent | `#7C3AED` | `#7C3AED` | `#22D3EE` | `#22D3EE` |
| Text | - | `#0A0F1C` | - | `#F1F5F9` |

| Purpose | PULSE Cover | PULSE Content | MINIMA Cover | MINIMA Content |
|---------|-------------|---------------|--------------|-----------------|
| Background | `#111827` | `#FFFFFF` | `#111111` | `#FAFAFA` |
| Title/Primary | `#FFFFFF` | `#111827` | `#FFFFFF` | `#111111` |
| Subtitle/Secondary | `#9CA3AF` | `#6B7280` | `#999999` | `#777777` |
| Accent | `#EF4444` | `#EF4444` | `#FFFFFF` | `#000000` |
| Success | - | `#10B981` | - | - |
| Warning | - | `#F59E0B` | - | - |
| Secondary | `#5B6475` | `#94A3B8` | `#6B7280` | `#777777` |
| Accent | `#7C3AED` | `#22D3EE` | `#EF4444` | `#000000` |
| Success | - | `#34D399` | `#10B981` | - |
| Warning | - | - | `#F59E0B` | - |
| Highlight | `#22D3EE` | - | - | - |

### Theme Code Templates (Copy & Use)

**IMPORTANT**: Each theme has TWO variants - Cover (slide 0) and Content (slides 1+).

**NOVA Theme** (Strategy / AI / Investor):
```javascript
const theme = {
  // Cover slide (slide 0) - dark, high impact
  cover: {
    background: '#0A0F1C',
    title: '#FFFFFF',
    subtitle: '#94A3B8',
    accent: '#7C3AED'
  },
  // Content slides (slides 1+) - light, readable
  content: {
    background: '#F6F7FB',
    primary: '#0A0F1C',
    secondary: '#5B6475',
    accent: '#7C3AED',
    text: '#0A0F1C'
  }
};
```

**ORBIT Theme** (Technical / Architecture / Dev):
```javascript
const theme = {
  // Cover slide (slide 0) - dark, tech aesthetic
  cover: {
    background: '#0B1220',
    title: '#F1F5F9',
    subtitle: '#94A3B8',
    accent: '#22D3EE'
  },
  // Content slides (slides 1+) - slightly lighter dark
  content: {
    background: '#0F172A',
    primary: '#F1F5F9',
    secondary: '#94A3B8',
    accent: '#22D3EE',
    success: '#34D399',
    text: '#F1F5F9'
  }
};
```

**PULSE Theme** (Metrics / Growth / Performance):
```javascript
const theme = {
  // Cover slide (slide 0) - dark, bold
  cover: {
    background: '#111827',
    title: '#FFFFFF',
    subtitle: '#9CA3AF',
    accent: '#EF4444'
  },
  // Content slides (slides 1+) - light, clean
  content: {
    background: '#FFFFFF',
    primary: '#111827',
    secondary: '#6B7280',
    accent: '#EF4444',
    success: '#10B981',
    warning: '#F59E0B',
    text: '#111827'
  }
};
```

**MINIMA Theme** (Founder / Brand / Minimalist):
```javascript
const theme = {
  // Cover slide (slide 0) - pure black & white
  cover: {
    background: '#111111',
    title: '#FFFFFF',
    subtitle: '#999999',
    accent: '#FFFFFF'
  },
  // Content slides (slides 1+) - off-white, clean
  content: {
    background: '#FAFAFA',
    primary: '#111111',
    secondary: '#777777',
    accent: '#000000',
    text: '#111111'
  }
};
```

**Usage Example**:
```javascript
const pres = new PptxGenJS();

// Slide 0: Cover (use theme.cover)
pres.background = { color: theme.cover.background };
pres.addText('My Presentation', { x: 1, y: 3, fontSize: 60, bold: true, color: theme.cover.title });
pres.addText('Company Name', { x: 1, y: 4.2, fontSize: 28, color: theme.cover.subtitle });

// Slide 1+: Content (use theme.content)
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('Key Points', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.content.primary });
['Point 1', 'Point 2', 'Point 3'].forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2 + i * 0.7, fontSize: 18, color: theme.content.text, bullet: true });
});
```

## Quick Start

**IMPORTANT**: You MUST use one of the predefined themes (NOVA, ORBIT, PULSE, MINIMA) below. Never define custom colors or use hardcoded hex values.

**CRITICAL**: Slide 0 (cover) uses `theme.cover`, all other slides use `theme.content`.

```javascript
execute_javascript_code("""
const PptxGenJS = require('pptxgenjs');

const theme = {
  cover: {
    background: '#0A0F1C',
    title: '#FFFFFF',
    subtitle: '#94A3B8',
    accent: '#7C3AED'
  },
  content: {
    background: '#F6F7FB',
    primary: '#0A0F1C',
    secondary: '#5B6475',
    accent: '#7C3AED',
    text: '#0A0F1C'
  }
};

const pres = new PptxGenJS();

// Slide 0: Cover slide (use theme.cover)
pres.background = { color: theme.cover.background };
pres.addText('My Presentation', { x: 1, y: 3, fontSize: 60, bold: true, color: theme.cover.title });
pres.addText('Company Name', { x: 1, y: 4.2, fontSize: 28, color: theme.cover.subtitle });

// Slide 1: Content slide (use theme.content)
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('Key Points', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.content.primary });
['Point 1', 'Point 2', 'Point 3'].forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2 + i * 0.7, fontSize: 18, color: theme.content.text, bullet: true });
});

pres.writeFile({ fileName: 'my_presentation.pptx' });
""", packages='pptxgenjs')
```

**Note**: Generated files are automatically saved to the workspace output directory.

## Core Slide Patterns

### Cover + Content Slides (NOVA - Strategy Deck)

```javascript
const pres = new PptxGenJS();

const theme = {
  cover: {
    background: '#0A0F1C',
    title: '#FFFFFF',
    subtitle: '#94A3B8',
    accent: '#7C3AED'
  },
  content: {
    background: '#F6F7FB',
    primary: '#0A0F1C',
    secondary: '#5B6475',
    accent: '#7C3AED',
    text: '#0A0F1C'
  }
};

// Slide 0: Cover (dark background, white title)
pres.background = { color: theme.cover.background };
pres.addText('Annual Report 2024', { x: 1, y: 3, fontSize: 64, bold: true, color: theme.cover.title });
pres.addText('Company Name', { x: 1, y: 4.2, fontSize: 28, color: theme.cover.subtitle });

// Slide 1: Content (light background, readable)
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('Key Points', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.content.primary });
['Point 1', 'Point 2', 'Point 3'].forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2 + i * 0.7, fontSize: 18, color: theme.content.text, bullet: true });
});

pres.writeFile({ fileName: 'strategy.pptx' });
```

### Content Slide with Bullets (ORBIT - Technical)

```javascript
const pres = new PptxGenJS();

const theme = {
  cover: {
    background: '#0B1220',
    title: '#F1F5F9',
    subtitle: '#94A3B8',
    accent: '#22D3EE'
  },
  content: {
    background: '#0F172A',
    primary: '#F1F5F9',
    secondary: '#94A3B8',
    accent: '#22D3EE',
    success: '#34D399',
    text: '#F1F5F9'
  }
};

// Content slide
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('System Architecture', { x: 1, y: 0.8, fontSize: 48, bold: true, color: theme.content.primary });

const bullets = [
  'Microservices architecture',
  'Event-driven communication',
  'Scalable infrastructure'
];

bullets.forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2 + i * 0.7, fontSize: 18, color: theme.content.text, bullet: true });
});

pres.writeFile({ fileName: 'technical.pptx' });
```

### Metrics Slide (PULSE - Business Performance)

```javascript
const pres = new PptxGenJS();

const theme = {
  cover: {
    background: '#111827',
    title: '#FFFFFF',
    subtitle: '#9CA3AF',
    accent: '#EF4444'
  },
  content: {
    background: '#FFFFFF',
    primary: '#111827',
    secondary: '#6B7280',
    accent: '#EF4444',
    success: '#10B981',
    warning: '#F59E0B',
    text: '#111827'
  }
};

// Content slide with metrics
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('Q4 Key Metrics', { x: 1, y: 0.8, fontSize: 52, bold: true, color: theme.content.primary });

const metrics = [
  { label: 'Revenue', value: '$1.5M', color: theme.content.success },
  { label: 'Growth', value: '+25%', color: theme.content.accent },
  { label: 'Customers', value: '86', color: theme.content.warning }
];

metrics.forEach((metric, i) => {
  const x = 1 + (i % 3) * 3;
  const y = 2.5 + Math.floor(i / 3) * 2;
  pres.addText(metric.label, { x, y: y, fontSize: 16, color: theme.content.secondary });
  pres.addText(metric.value, { x, y: y + 0.4, fontSize: 36, bold: true, color: metric.color });
});

pres.writeFile({ fileName: 'metrics.pptx' });
```
};

pres.background = { color: theme.background };

pres.addText('Q4 Key Metrics', { x: 1, y: 0.8, fontSize: 52, bold: true, color: theme.primary });

const metrics = [
  { label: 'Revenue', value: '$1.5M', color: theme.success },
  { label: 'Growth', value: '+25%', color: theme.accent },
  { label: 'Customers', value: '86', color: theme.warning }
];

metrics.forEach((metric, i) => {
  const x = 1 + (i % 3) * 3;
  const y = 2.5 + Math.floor(i / 3) * 2;
  pres.addText(metric.label, { x, y: y, fontSize: 16, color: theme.secondary });
  pres.addText(metric.value, { x, y: y + 0.4, fontSize: 36, bold: true, color: metric.color });
});

pres.writeFile({ fileName: 'metrics.pptx' });
```

### Minimalist Content (MINIMA - Founder Story)

```javascript
const pres = new PptxGenJS();

const theme = {
  background: '#FAFAFA',
  primary: '#111111',
  secondary: '#777777',
  accent: '#000000',
  text: '#111111'
};

pres.background = { color: theme.background };

pres.addText('Our Journey', { x: 1, y: 0.8, fontSize: 56, bold: true, color: theme.primary });

['Founded in 2020', 'Team of 10', 'Bootstrapped'].forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2.5 + i * 0.8, fontSize: 20, color: theme.text });
});

pres.writeFile({ fileName: 'minimal.pptx' });
```

### Multi-Slide Presentation (NOVA - Strategy Context)

```javascript
const pres = new PptxGenJS();

const theme = {
  background: '#F6F7FB',
  primary: '#0A0F1C',
  secondary: '#5B6475',
  accent: '#7C3AED',
  highlight: '#22D3EE',
  text: '#0A0F1C'
};

// Slide 1: Title
pres.background = { color: theme.background };
pres.addText('Strategic Vision 2025', { x: 1, y: 3, fontSize: 64, bold: true, color: theme.primary });
pres.addText('Company Name', { x: 1, y: 4.2, fontSize: 28, color: theme.secondary });

// Slide 2: Content
pres.addSlide();
pres.background = { color: theme.background };
pres.addText('Key Initiatives', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.primary });
['AI Platform Launch', 'Market Expansion', 'Team Growth'].forEach((text, i) => {
  pres.addText(text, { x: 1, y: 2 + i * 0.7, fontSize: 18, color: theme.text, bullet: true });
});

// Slide 3: Metrics
pres.addSlide();
pres.background = { color: theme.background };
pres.addText('Performance Targets', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.primary });
pres.addText('Revenue: $5M', { x: 1, y: 2.5, fontSize: 28, color: theme.accent });
pres.addText('Growth: 150%', { x: 5, y: 2.5, fontSize: 28, color: theme.highlight });

pres.writeFile({ fileName: 'strategy.pptx' });
```

## Working with Images

When adding images to presentations, use relative paths (code runs in workspace/output directory):

```javascript
const pres = new PptxGenJS();
const fs = require('fs');

// Slide with image
pres.addSlide();
pres.addText('Revenue Chart', { x: 0.5, y: 0.8, fontSize: 40, bold: true, color: theme.content.primary });

// Check if image exists
if (fs.existsSync('revenue_chart.png')) {
  // GOOD: Specify both w and h to control exact dimensions
  pres.addImage({ path: 'revenue_chart.png', x: 1, y: 1.5, w: 8, h: 4.5 });

  // BETTER: Use sizing: 'contain' to fit within bounds while maintaining aspect ratio
  // pres.addImage({ path: 'revenue_chart.png', x: 1, y: 1.5, sizing: { type: 'contain', w: 8, h: 4.5 } });
} else {
  pres.addText('Image not available: revenue_chart.png', { x: 1, y: 3, fontSize: 18, color: theme.content.secondary || theme.content.warning });
}

pres.writeFile({ fileName: 'with_image.pptx' });
```

**Image Sizing Best Practices:**

| Method | When to Use | Example |
|--------|-------------|---------|
| `w: 8, h: 4.5` | Exact dimensions needed | `{ x: 1, y: 2, w: 8, h: 4 }` |
| `sizing: { type: 'contain', w: 8, h: 5 }` | Maintain aspect ratio, fit in bounds | `{ x: 1, y: 1.5, sizing: { type: 'contain', w: 8, h: 5 } }` |
| `sizing: { type: 'cover', w: 8, h: 5 }` | Fill bounds, crop if needed | `{ x: 1, y: 1.5, sizing: { type: 'cover', w: 8, h: 5 } }` |

**WRONG - May overflow slide:**
```javascript
// ❌ Only width - height depends on aspect ratio, may exceed slide bounds
pres.addImage({ path: 'chart.png', x: 1, y: 2, w: 9 });

// ❌ Only height - width may exceed slide width
pres.addImage({ path: 'photo.png', x: 1, y: 1, h: 6 });
```

**CORRECT - Stay within bounds:**
```javascript
// ✅ Both w and h specified
pres.addImage({ path: 'chart.png', x: 1, y: 1.5, w: 8, h: 4.5 });

// ✅ Using contain with max bounds
pres.addImage({ path: 'photo.png', x: 0.5, y: 1, sizing: { type: 'contain', w: 9, h: 6 } });
```

**Important Notes:**
- Use relative paths like `'revenue_chart.png'` - code runs in workspace/output directory
- Supported formats: PNG, JPG, JPEG, GIF, PDF
- Keep images within safe area: x: 0.5-9.5, y: 0.5-7 inches
- Use `sizing: { type: 'contain' }` when you want to preserve aspect ratio
- Use `sizing: { type: 'cover' }` when you want to fill the entire area

## Working with Existing Presentations

### Read Presentation Structure

```
read_pptx("presentation.pptx", extract_text=False)
```

Returns:
```json
{
  "slide_count": 5,
  "slides": [
    {"index": 0, "filename": "slide1.xml", "hidden": false}
  ],
  "titles": ["Title", "Content", "Summary", "Q&A", "Thank You"]
}
```

### Extract Text Content

```
read_pptx("presentation.pptx", extract_text=True)
```

Returns all text content from the presentation.

## Available Tools

| Tool | Purpose |
|-------|---------|
| `execute_javascript_code` | Generate presentations using JavaScript (use packages='pptxgenjs') |
| `read_pptx` | Extract content/structure from existing PPTX files |
| `unpack_pptx` | Extract PPTX files to directory for inspection/learning from templates |
| `pack_pptx` | Package directory back into PPTX file after manual editing |
| `clean_pptx` | Clean orphaned files from unpacked PPTX directory |
