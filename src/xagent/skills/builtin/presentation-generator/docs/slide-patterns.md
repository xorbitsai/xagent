# Slide Patterns & Templates

Ready-to-use slide templates for different presentation contexts.

## Basic Cover + Content Slides (NOVA - Strategy Deck)

**Note**: This is a basic template. For Image-Text Mixed Mode presentations, cover titles must be embedded in the image (see Mixed Mode Guide).

Perfect for simple strategy presentations, investor decks, and business reviews.

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
// For Mixed Mode: Use generated image with embedded title instead
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

## Content Slide with Bullets (ORBIT - Technical)

Ideal for technical presentations, architecture reviews, and development updates.

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

## Metrics Slide (PULSE - Business Performance)

Perfect for business reviews, quarterly reports, and performance dashboards.

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

## Minimalist Content (MINIMA - Founder Story)

Great for brand stories, founder presentations, and minimalist designs.

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

## Multi-Slide Presentation (NOVA - Strategy Context)

Complete multi-slide presentation template for strategic planning.

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

## Customizing Slide Patterns

### 1. Change Theme
Replace the theme object with any of the predefined themes:
- **NOVA**: Strategy/AI presentations
- **ORBIT**: Technical/architecture presentations
- **PULSE**: Business metrics presentations
- **MINIMA**: Minimalist/brand presentations

### 2. Modify Content
Update the text arrays with your specific content:
```javascript
// Original
['Point 1', 'Point 2', 'Point 3']

// Customized
['Increase market share by 15%', 'Launch new product line', 'Expand to 3 new countries']
```

### 3. Adjust Layout
Change positioning and sizing:
```javascript
// Original positioning
pres.addText('Title', { x: 1, y: 3, fontSize: 64 })

// Adjusted positioning
pres.addText('Title', { x: 0.5, y: 2.5, fontSize: 72 })
```

### 4. Add Images
Incorporate generated images:
```javascript
// Add generated chart image
pres.addImage({ path: 'generated_chart.png', x: 1, y: 1.5, w: 8, h: 4.5 });
```

## Best Practices

1. **Consistency**: Use the same theme throughout your presentation
2. **Hierarchy**: Maintain clear visual hierarchy (Title > Subtitle > Body)
3. **Whitespace**: Leave adequate whitespace for readability
4. **Alignment**: Keep elements properly aligned
5. **Contrast**: Ensure sufficient contrast between text and background

---

**Navigation**: [Back to Theme System](themes.md) | [Next: Image Guide](image-guide.md) | [Back to Main Documentation](../SKILL.md)
