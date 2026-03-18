# Mixed Mode Guide (Image-Text Mixed Mode)

## Overview

Image-Text Mixed Mode is the **default presentation mode**, combining the advantages of images and text, suitable for most business and technical presentations.

## Mode Selection Rules

### Default Mode
- **Image-Text Mixed Mode** is the default mode
- Used when the user does not explicitly specify a mode

### Full Image Mode Trigger Conditions
- Full Image Mode is used ONLY when the user explicitly requests "全图", "全图片", "all images", "full image"
- Example trigger words: "生成全图的ppt", "创建全图片演示", "all image presentation"

### Layout Assignment Rules
- In Image-Text Mixed Mode, layouts are **randomly assigned**
- A single PPT can contain multiple layouts (left-image-right-text, left-text-right-image)
- Ensures visual diversity and avoids monotony

## Mode Requirements

### B. Mixed Mode (Mixed Image-Text Template) - **Default Mode**

#### Image Coverage
- **30% full-image slides** + **70% mixed image-text slides**
- In every 10 slides: 3 full-image, 7 mixed

#### Text Processing
- **Text is editable** (except cover)
- Use standard text formats and fonts
- Supports bullet points and numbered lists

#### Layout Options
- **Left Image, Right Text**
- **Left Text, Right Image**
- **Random layout assignment**: Multiple layouts can be mixed in a single PPT
- Detailed layout explanations below

#### Cover Rules
- **Cover must be pure image**
- **Title embedded in the image** (non-editable text)
- Use high-quality background images
- Can add semi-transparent overlay for better text readability

#### Data Visualization
- **All data content must be converted to 3D charts**
- Use `generate_image` to generate 3D charts
- Ensure charts are clear and readable
- Match presentation theme colors

## Available Layouts

### 1. Left Image, Right Text
Image on the left, text on the right. Suitable for showing visual content first, then explanation.

```javascript
// Left Image, Right Text layout example with divider and centered alignment
const slide = pres.addSlide();
slide.background = { color: theme.content.background };

// Left image (50% width) - centered vertically
slide.addImage({
    path: 'generated_image.png',
    x: 0.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});

// Right text (50% width) - centered alignment
slide.addText('Title Text', {
    x: 6.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});

// Add divider line under title
slide.addShape(pres.ShapeType.line, {
    x: 6.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});

// Body text - centered alignment with proper spacing
slide.addText('Body content here...', {
    x: 6.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    bullet: true,
    align: 'center'
});
```

### 2. Left Text, Right Image
Text on the left, image on the right. Suitable for explaining concepts first, then showing visual content.

```javascript
// Left Text, Right Image layout example with divider and centered alignment
const slide = pres.addSlide();
slide.background = { color: theme.content.background };

// Left text (50% width) - centered alignment
slide.addText('Title Text', {
    x: 0.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});

// Add divider line under title
slide.addShape(pres.ShapeType.line, {
    x: 0.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});

// Body text - centered alignment with proper spacing
slide.addText('Body content here...', {
    x: 0.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    bullet: true,
    align: 'center'
});

// Right image (50% width) - centered vertically
slide.addImage({
    path: 'generated_image.png',
    x: 6.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});
```

### Layout Parameters

| Parameter | Left Image, Right Text | Left Text, Right Image | Description |
|-----------|------------------------|------------------------|-------------|
| Image position (x) | 0.5 | 6.5 | Left or right side |
| Image width (w) | 5.5 | 5.5 | 50% of slide width |
| Image height (h) | 4.5 | 4.5 | Appropriate height |
| Image alignment | `align: 'center', valign: 'middle'` | `align: 'center', valign: 'middle'` | Center image vertically |
| Text position (x) | 6.5 | 0.5 | Opposite side of image |
| Text width (w) | 3.5 | 3.5 | Appropriate width |
| Text alignment | `align: 'center'` | `align: 'center'` | Center text horizontally |
| Divider position (y) | 1.3 | 1.3 | Below title |
| Divider width (w) | 3.5 | 3.5 | Same as text width |
| Divider color | `theme.content.accent` | `theme.content.accent` | Theme accent color |

### Usage Recommendations

#### Factors to consider when choosing layout:
1. **Information flow order**: What to show first? Image or text?
2. **Visual focus**: Which element is more important?
3. **Content type**: Data chart vs concept diagram vs photo
4. **Reading habits**: Target audience reading habits

#### Best practices:
1. **Consistency**: Maintain consistent layout patterns throughout the presentation
2. **Balance**: Ensure visual balance between image and text areas
3. **Whitespace**: Maintain adequate whitespace
4. **Alignment**: Ensure elements are properly aligned

## Workflow

### 1. Plan Slide Structure
```python
import random

# Example: 10-slide presentation
# Random layout assignment: left-image-right-text or left-text-right-image
slide_plan = [
    {"type": "cover", "layout": "full_image", "content": "Cover"},
]

# Add 7 mixed image-text slides (70%), randomly assign layouts
for i in range(7):
    # Randomly choose layout: left-image-right-text or left-text-right-image
    layout = random.choice(["image_left_text_right", "text_left_image_right"])
    slide_plan.append({
        "type": "content",
        "layout": layout,
        "content": f"Content {i+1}"
    })

# Add 3 full-image slides (30%)
for i in range(3):
    slide_plan.append({
        "type": "full_image",
        "layout": "full_image",
        "content": f"Full Image {i+1}"
    })

print("Slide plan (random layouts):")
for i, slide in enumerate(slide_plan):
    print(f"Slide {i+1}: {slide['type']} - {slide['layout']}")
```

### 2. Generate Images
```python
# Generate cover image
generate_image(
    prompt="Professional presentation cover with dark background, futuristic design, embedded title 'Annual Report 2024'",
    size="1920x1080"
)

# Generate 3D data chart
generate_image(
    prompt="3D bar chart showing quarterly sales data: Q1 $1.2M, Q2 $1.5M, Q3 $1.8M, Q4 $2.1M. Professional design, gradient colors, clear labels",
    size="1024x768"
)

# Generate full-image slide image
generate_image(
    prompt="Conceptual illustration of business growth, abstract design, professional style, full slide image",
    size="1920x1080"
)
```

### 3. Create Presentation (10-slide example following 7:3 ratio)
```javascript
const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// --- Slide 0: Cover (pure image with embedded title) ---
// First slide doesn't need addSlide()
pres.addImage({
    path: 'cover_image.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});
// Optional: Add semi-transparent overlay for better readability
pres.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: '100%', h: '100%',
    fill: { color: '000000', transparency: 40 }
});
// Title is embedded in the image, no addText() needed

// --- 7 Mixed Image-Text Slides (70%) ---
// Slide 1: Left Image, Right Text
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };
slide1.addImage({
    path: 'data_chart_3d.png',
    x: 0.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});
slide1.addText('Quarterly Sales Data', {
    x: 6.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});
slide1.addShape(pres.ShapeType.line, {
    x: 6.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});
slide1.addText('Detailed analysis content...', {
    x: 6.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    align: 'center'
});

// Slide 2: Left Text, Right Image
const slide2 = pres.addSlide();
slide2.background = { color: theme.content.background };
slide2.addText('Market Analysis', {
    x: 0.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});
slide2.addShape(pres.ShapeType.line, {
    x: 0.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});
slide2.addText('Analysis content...', {
    x: 0.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    align: 'center'
});
slide2.addImage({
    path: 'market_analysis_3d.png',
    x: 6.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});

// Slides 3-7: Additional mixed slides (follow similar pattern)
// You would continue adding 5 more mixed slides here...
// For brevity, we show just 2 examples above

// --- 3 Full-Image Slides (30%) ---
// Slide 8: Full-image slide
const slide8 = pres.addSlide();
slide8.addImage({
    path: 'full_image_1.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});

// Slide 9: Full-image slide
const slide9 = pres.addSlide();
slide9.addImage({
    path: 'full_image_2.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});

// Slide 10: Full-image slide
const slide10 = pres.addSlide();
slide10.addImage({
    path: 'full_image_3.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});

// Total: 1 cover + 7 mixed + 3 full-image = 11 slides
// Ratio: 7 mixed / 10 content slides = 70%, 3 full-image / 10 = 30%
```

## Complete Example (5-slide demo following 7:3 ratio)

```javascript
// Create mixed-mode presentation with cover and proper ratio
const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// --- Slide 0: Cover (pure image with embedded title) ---
// First slide doesn't need addSlide()
pres.addImage({
    path: 'generated_cover.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});
// Optional overlay for better readability
pres.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: '100%', h: '100%',
    fill: { color: '000000', transparency: 40 }
});
// Title is embedded in the image (non-editable)

// --- Mixed Image-Text Slides (70%) ---
// Slide 1: Left Image, Right Text
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };

// Left image - centered vertically
slide1.addImage({
    path: 'data_chart_3d.png',
    x: 0.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});

// Right text - centered alignment
slide1.addText('Quarterly Sales Data', {
    x: 6.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});

// Add divider line under title
slide1.addShape(pres.ShapeType.line, {
    x: 6.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});

// Body text - centered alignment
slide1.addText('• Q1: $1.2M\n• Q2: $1.5M\n• Q3: $1.8M\n• Q4: $2.1M', {
    x: 6.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    align: 'center'
});

// Slide 2: Left Text, Right Image
const slide2 = pres.addSlide();
slide2.background = { color: theme.content.background };

// Left text - centered alignment
slide2.addText('Market Trend Analysis', {
    x: 0.5, y: 0.8,
    w: 3.5, fontSize: 32, bold: true,
    color: theme.content.primary,
    align: 'center'
});

// Add divider line under title
slide2.addShape(pres.ShapeType.line, {
    x: 0.5, y: 1.3,
    w: 3.5, h: 0,
    line: { color: theme.content.accent, width: 2 }
});

// Body text - centered alignment
slide2.addText('• Mobile growth rapid\n• Enterprise customers increasing\n• International market expansion', {
    x: 0.5, y: 1.6,
    w: 3.5, fontSize: 18,
    color: theme.content.text,
    bullet: true,
    align: 'center'
});

// Right image - centered vertically
slide2.addImage({
    path: 'market_trend_3d.png',
    x: 6.5, y: 1.5,
    w: 5.5, h: 4.5,
    align: 'center', valign: 'middle'
});

// --- Full-Image Slides (30%) ---
// Slide 3: Full-image slide
const slide3 = pres.addSlide();
slide3.addImage({
    path: 'full_image_concept.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});

// For a complete 10-slide presentation, you would add:
// - 5 more mixed slides (slides 4-8)
// - 2 more full-image slides (slides 9-10)
// Total: 1 cover + 7 mixed + 3 full-image = 11 slides (7:3 ratio)

pres.writeFile({ fileName: 'mixed_mode_presentation.pptx' });
```

## Complete Example with Five-Level Title Hierarchy & Key Text Emphasis

Example showing proper five-level title hierarchy with key text emphasis:

```javascript
const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

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

// --- Slide: Five-Level Title Hierarchy Example with Key Text Emphasis ---
const slide = pres.addSlide();
slide.background = { color: theme.content.background };

// Level 1: 大标题 (Main Title) - 48-56pt, bold, primary color with underline
slide.addText('2024年第四季度业务绩效评审', {
  x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

// Accent line under main title (thicker, accent color)
slide.addShape(pres.ShapeType.line, {
  x: 0.5, y: 1.4, w: 9, h: 0,
  line: { color: theme.content.accent, width: 3 }
});

// Level 2: 小标题 (Sub Title) - 36-44pt, bold, accent color
slide.addText('财务绩效深度分析', {
  x: 0.5, y: 1.8, fontSize: 40, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// Light divider under section heading
slide.addShape(pres.ShapeType.line, {
  x: 0.5, y: 2.3, w: 4, h: 0,
  line: { color: theme.content.secondary, width: 1, dashType: 'dash' }
});

// Level 3: 二级标题 (Secondary Title) - 28-32pt, semi-bold, secondary color
slide.addText('关键收入指标分析:', {
  x: 0.5, y: 2.6, fontSize: 30, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Level 4: 内容标题 (Content Title) - 22-26pt, semi-bold, text color
slide.addText('季度财务表现:', {
  x: 0.8, y: 3.2, fontSize: 24, bold: true,
  color: theme.content.text, fontFace: 'Arial'
});

// Level 5: 内容 (Content) with key text emphasis - 18-22pt, normal weight, text color
const metrics = [
  '总收入: $15.2M',
  '同比增长: 25%',
  '毛利率: 68%',
  '营业利润: $4.8M',
  '净利润: $3.2M'
];

metrics.forEach((text, i) => {
  slide.addText(text, {
    x: 0.8, y: 3.6 + i * 0.6,
    fontSize: 20, color: theme.content.text,
    bullet: true, lineSpacing: 24, fontFace: 'Arial'
  });
});

// Key text emphasis - highlight important data with accent colors
slide.addText('$15.2M', {
  x: 2.5, y: 3.6, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('+25%', {
  x: 2.5, y: 4.2, fontSize: 20, bold: true,
  color: theme.content.success, fontFace: 'Arial'
});

slide.addText('68%', {
  x: 2.5, y: 4.8, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('$4.8M', {
  x: 2.5, y: 5.4, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('$3.2M', {
  x: 2.5, y: 6.0, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// Another Level 2 heading for different section
slide.addText('市场地位与增长趋势', {
  x: 5.5, y: 1.8, fontSize: 38, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// Level 3 under second section
slide.addText('市场份额分析:', {
  x: 5.5, y: 2.6, fontSize: 28, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Level 4 for second section
slide.addText('区域市场表现:', {
  x: 5.8, y: 3.2, fontSize: 24, bold: true,
  color: theme.content.text, fontFace: 'Arial'
});

// Body text in second column with key emphasis
const marketData = [
  '北美市场: 42% 市场份额',
  '欧洲市场: 28% 市场份额',
  '亚太市场: 18% 市场份额',
  '其他地区: 12% 市场份额'
];

marketData.forEach((text, i) => {
  slide.addText(text, {
    x: 5.8, y: 3.6 + i * 0.6,
    fontSize: 18, color: theme.content.text,
    bullet: { code: '•' }, lineSpacing: 22, fontFace: 'Arial'
  });
});

// Key text emphasis for market data
slide.addText('42%', {
  x: 7.8, y: 3.6, fontSize: 18, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('28%', {
  x: 7.8, y: 4.2, fontSize: 18, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('18%', {
  x: 7.8, y: 4.8, fontSize: 18, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

slide.addText('12%', {
  x: 7.8, y: 5.4, fontSize: 18, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Key insight with warning color
slide.addText('关键发现: 北美市场占据主导地位，但亚太市场增长最快', {
  x: 0.5, y: 6.2, fontSize: 20, bold: true,
  color: theme.content.warning, fontFace: 'Arial'
});

// Caption text at bottom (14-16pt, secondary color, italic)
slide.addText('数据来源: 内部财务报告 2024年第四季度 | 注: 所有数据未经审计，仅供参考', {
  x: 0.5, y: 6.8, fontSize: 14,
  color: theme.content.secondary, italic: true, fontFace: 'Arial'
});

pres.writeFile({ fileName: 'title_hierarchy_example.pptx' });
```

## Quality Checklist

### Image Requirements
- [ ] Cover is pure image with title embedded
- [ ] All data charts are in 3D format
- [ ] Image quality is clear, size appropriate
- [ ] Images match theme colors

### Text Requirements
- [ ] Text is editable (except cover)
- [ ] Consistent fonts and sizes used
- [ ] Text and images are balanced
- [ ] Adequate whitespace

### Layout Requirements
- [ ] 30% full-image slides + 70% mixed slides
- [ ] Correct use of left-image-right-text/left-text-right-image layouts
- [ ] Title divider lines added under each title
- [ ] Text centered horizontally (`align: 'center'`)
- [ ] Images centered vertically (`align: 'center', valign: 'middle'`)
- [ ] Elements properly aligned
- [ ] Clear visual hierarchy

### Technical Requirements
- [ ] All images generated with `generate_image`
- [ ] Follow theme system
- [ ] Reasonable file size
- [ ] Good compatibility

---

**Navigation**: [Back to Main Documentation](../SKILL.md) | [Next: Full Image Mode](full-image-mode.md) | [Previous: Image Generation Workflow](image-generation-workflow.md)
