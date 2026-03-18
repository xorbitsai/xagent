---
name: presentation
description: "Generate and edit PowerPoint presentations (.pptx) with integrated image tools. Use for: creating slide decks, pitch decks, or presentations from scratch; generating custom images, searching for visuals, adding logos; reading, parsing, or extracting content from .pptx files; editing, modifying, or updating existing presentations; combining or splitting slide files; working with templates, layouts, speaker notes, or comments. Trigger when user mentions 'deck,' 'slides,' 'presentation,' or references a .pptx filename."
---

# Presentation Generator

Generate PowerPoint presentations using JavaScript code with the pptxgenjs library and integrated image tools.

## ⚠️ CRITICAL REQUIREMENTS - READ FIRST

**YOU MUST FOLLOW THESE RULES - NO EXCEPTIONS:**

1. **ONLY use the 4 predefined themes** (NOVA, ORBIT, PULSE, MINIMA)
2. **NEVER create custom color variables** like `const colorAccent = "0078D7"`
3. **NEVER use hardcoded hex values** in your code - always reference `theme.xxx`
4. **ALWAYS include the `#` prefix** in hex colors (e.g., `#EF4444` not `EF4444`)
5. **Slide 0 MUST use `theme.cover`** - first slide is the cover with dark/high-contrast background
6. **Slides 1+ MUST use `theme.content`** - all other slides use light/readable background
7. **MUST generate all images with `generate_image`** - NO external URLs or pre-existing images allowed
8. **DEFAULT to Image-Text Mixed Mode** - Use Full Image Mode ONLY when user explicitly requests "全图", "full image", etc.
9. **RANDOMIZE layouts in Mixed Mode** - Mix left-image-right-text and left-text-right-image layouts
10. **DATA VISUALIZATION MUST use specific chart types** - Use appropriate chart types (line, bar, pie, etc.) with clear data labels, not abstract images
11. **CLEAR TITLE HIERARCHY** - Use distinct font sizes, colors, and styles for Level 1, Level 2, Level 3 titles and body text
12. **PROPER CHART LABELING** - All charts must include axis labels, data labels, legends, and grid lines where appropriate

**WRONG:**
```javascript
const colorAccent = "0078D7";  // ❌ WRONG - custom color
slide1.addText("Title", { color: "363636" });  // ❌ WRONG - hardcoded hex
pres.addSlide();
pres.background = { color: theme.cover.background };  // ❌ WRONG - cover on non-first slide
slide1.addImage({ path: 'https://external.com/image.jpg', x: 1, y: 1, w: 8, h: 4.5 });  // ❌ WRONG - external URL
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

## Quick Start

**IMPORTANT**: You MUST use one of the predefined themes (NOVA, ORBIT, PULSE, MINIMA). Never define custom colors or use hardcoded hex values.

**CRITICAL**: Slide 0 (cover) uses `theme.cover`, all other slides use `theme.content`.

### Key Improvements:
1. **Enhanced Data Visualization**: Use specific chart types (line, bar, pie) with clear and accurate data labels instead of abstract images
2. **Five-Level Title Hierarchy**: Distinct styling for 大标题 (Level 1), 小标题 (Level 2), 二级标题 (Level 3), 内容标题 (Level 4), 内容 (Level 5)
3. **Key Text Emphasis**: Highlight important text with accent colors and bold styling
4. **Professional Chart Labeling**: All charts must include proper axis labels, data labels, legends, and accurate data representation

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
    success: '#10B981',
    warning: '#F59E0B',
    text: '#0A0F1C'
  }
};

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// Slide 0: Cover slide - pure image with embedded title (non-editable)
pres.addImage({
    path: 'generated_cover_image.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});
// Optional: Add semi-transparent overlay for better readability
pres.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: '100%', h: '100%',
    fill: { color: '000000', transparency: 40 }
});

// Slide 1: Content slide with five-level title hierarchy and key text emphasis
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };

// Level 1: 大标题 (Main Title)
slide1.addText('2024年度业务绩效报告', {
    x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
    color: theme.content.primary, fontFace: 'Arial'
});
// Accent line under main title
slide1.addShape(pres.ShapeType.line, {
    x: 0.5, y: 1.4, w: 9, h: 0,
    line: { color: theme.content.accent, width: 3 }
});

// Level 2: 小标题 (Sub Title)
slide1.addText('财务绩效分析', {
    x: 0.5, y: 1.8, fontSize: 40, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
});
// Light divider line
slide1.addShape(pres.ShapeType.line, {
    x: 0.5, y: 2.3, w: 4, h: 0,
    line: { color: theme.content.secondary, width: 1, dashType: 'dash' }
});

// Level 3: 二级标题 (Secondary Title)
slide1.addText('关键收入指标:', {
    x: 0.5, y: 2.6, fontSize: 30, bold: true,
    color: theme.content.secondary, fontFace: 'Arial'
});

// Level 4: 内容标题 (Content Title)
slide1.addText('季度表现:', {
    x: 0.8, y: 3.2, fontSize: 24, bold: true,
    color: theme.content.text, fontFace: 'Arial'
});

// Level 5: 内容 (Content) with key text emphasis
const metrics = [
    '总收入: $15.2M',
    '同比增长: 25%',
    '毛利率: 68%',
    '营业利润: $4.8M'
];

metrics.forEach((text, i) => {
    slide1.addText(text, {
        x: 0.8, y: 3.6 + i * 0.6,
        fontSize: 20, color: theme.content.text,
        bullet: true, lineSpacing: 24, fontFace: 'Arial'
    });
});

// Key text emphasis - highlight important data
slide1.addText('$15.2M', {
    x: 2.5, y: 3.6, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
});

slide1.addText('+25%', {
    x: 2.5, y: 4.2, fontSize: 20, bold: true,
    color: theme.content.success, fontFace: 'Arial'
});

slide1.addText('68%', {
    x: 2.5, y: 4.8, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
});

slide1.addText('$4.8M', {
    x: 2.5, y: 5.4, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
});

// Caption with data source
slide1.addText('数据来源: 内部财务报告 Q4 2024 | 注: 所有数据未经审计', {
    x: 0.5, y: 6.8, fontSize: 14,
    color: theme.content.secondary, italic: true, fontFace: 'Arial'
});

pres.writeFile({ fileName: 'my_presentation_with_hierarchy.pptx' });
""", packages='pptxgenjs')
```

**Note**: Generated files are automatically saved to the workspace output directory.

### Quick Start with Images

Create a presentation with custom-generated images:

```python
# Step 1: Generate a custom cover image
generate_image(
    prompt="Abstract futuristic background with blue and purple gradients, tech pattern, clean design, presentation cover",
    size="1920x1080"
)

# Step 2: Generate a 3D chart image (REQUIRED for data)
generate_image(
    prompt="3D精美 bar chart showing growth metrics: 2022 75%, 2023 85%, 2024 95%. Professional 3D design, gradient bars, data labels, realistic lighting, high quality",
    size="1024x768"
)

# Step 3: Create presentation with images
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

// Slide 0: Cover with generated image (first slide, no addSlide)
pres.addImage({ path: 'generated_cover_image.png', x: 0, y: 0, w: '100%', h: '100%' });
pres.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: '100%', h: '100%', fill: { color: '000000', transparency: 40 } });
pres.addText('Annual Strategy Review', { x: 1, y: 3, fontSize: 64, bold: true, color: theme.cover.title });
pres.addText('2024 Performance & 2025 Outlook', { x: 1, y: 4, fontSize: 28, color: theme.cover.subtitle });

// Slide 1: Content with chart
pres.addSlide();
pres.background = { color: theme.content.background };
pres.addText('Growth Metrics', { x: 1, y: 0.8, fontSize: 44, bold: true, color: theme.content.primary });
pres.addImage({ path: 'generated_chart_image.png', x: 1, y: 1.5, w: 8, h: 4.5 });

pres.writeFile({ fileName: 'strategy_with_images.pptx' });
""", packages='pptxgenjs')
```

### Improved Example with Enhanced Data Visualization, Five-Level Title Hierarchy & Key Text Emphasis

This example demonstrates the improved approach with specific chart types, clear five-level title hierarchy, and key text emphasis:

```python
# Step 1: Generate detailed line chart with proper labeling
generate_image(
    prompt="Professional line chart showing quarterly revenue growth. X-axis: Q1 2023 to Q4 2024. Y-axis: Revenue in $M. Data: Q1 2023: 1.2, Q2 2023: 1.5, Q3 2023: 1.8, Q4 2023: 2.1, Q1 2024: 2.4, Q2 2024: 2.7, Q3 2024: 3.0, Q4 2024: 3.3. Include grid lines, axis labels, data point markers, trend line, legend. Professional design with blue gradient line (#7C3AED), gray grid, white background.",
    size="1024x768"
)

# Step 2: Generate detailed bar chart for comparison
generate_image(
    prompt="3D bar chart comparing regional performance. Regions: North America, Europe, Asia, South America. Sales: NA: $5.2M, Europe: $3.8M, Asia: $4.5M, SA: $2.1M. Each bar labeled with region and value. Include Y-axis scale, grid lines, color-coded bars (gradient blue to purple). Professional 3D design.",
    size="1024x768"
)

# Step 3: Create presentation with enhanced title hierarchy
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
pres.layout = 'LAYOUT_WIDE';

// Slide 1: Cover
pres.addImage({ path: 'generated_cover_image.png', x: 0, y: 0, w: '100%', h: '100%' });
pres.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: '100%', h: '100%', fill: { color: '000000', transparency: 40 } });
pres.addText('2024 Annual Performance Review', { x: 1, y: 3, fontSize: 64, bold: true, color: theme.cover.title });
pres.addText('Data-Driven Insights & Strategic Outlook', { x: 1, y: 4, fontSize: 28, color: theme.cover.subtitle });

// Slide 2: Revenue Analysis with clear title hierarchy
const slide2 = pres.addSlide();
slide2.background = { color: theme.content.background };

// Level 1 Title
slide2.addText('Financial Performance Analysis', {
  x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

// Accent line under title
slide2.addShape(pres.ShapeType.line, {
  x: 0.5, y: 1.4, w: 9, h: 0,
  line: { color: theme.content.accent, width: 3 }
});

// Level 2 Title
slide2.addText('Quarterly Revenue Trends', {
  x: 0.5, y: 1.8, fontSize: 40, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// Detailed line chart
slide2.addImage({ path: 'generated_line_chart.png', x: 0.5, y: 2.2, w: 9, h: 4 });

// Level 3 Title for insights
slide2.addText('Key Insights:', {
  x: 0.5, y: 6.3, fontSize: 30, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Body text with bullet points
const insights = [
  'Steady quarterly growth averaging 12.5%',
  'Q4 2024 shows strongest performance at $3.3M',
  'Year-over-year growth of 57% from 2023',
  'Consistent upward trend across all quarters'
];

insights.forEach((text, i) => {
  slide2.addText(text, {
    x: 0.8, y: 6.8 + i * 0.6,
    fontSize: 20, color: theme.content.text,
    bullet: true, lineSpacing: 24
  });
});

// Slide 3: Regional Comparison
const slide3 = pres.addSlide();
slide3.background = { color: theme.content.background };

// Level 1 Title
slide3.addText('Regional Performance Comparison', {
  x: 0.5, y: 0.8, w: 9, fontSize: 48, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

// Level 2 Title
slide3.addText('Sales by Region (2024)', {
  x: 0.5, y: 1.6, fontSize: 36, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// Detailed bar chart
slide3.addImage({ path: 'generated_bar_chart.png', x: 0.5, y: 2.2, w: 9, h: 4 });

// Level 3 Title for analysis
slide3.addText('Regional Analysis:', {
  x: 0.5, y: 6.3, fontSize: 28, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Body text in two columns
const leftColumn = [
  'North America leads with $5.2M',
  'Europe shows strong growth potential',
  'Asia demonstrates rapid expansion'
];

const rightColumn = [
  'South America emerging market',
  'NA accounts for 42% of total revenue',
  'International markets growing 35% YoY'
];

leftColumn.forEach((text, i) => {
  slide3.addText(text, {
    x: 0.8, y: 6.8 + i * 0.6,
    fontSize: 18, color: theme.content.text,
    bullet: true, lineSpacing: 22
  });
});

rightColumn.forEach((text, i) => {
  slide3.addText(text, {
    x: 5.0, y: 6.8 + i * 0.6,
    fontSize: 18, color: theme.content.text,
    bullet: true, lineSpacing: 22
  });
});

// Caption at bottom
slide3.addText('Data Source: Internal Sales Reports Q1-Q4 2024 | Note: All figures in USD millions', {
  x: 0.5, y: 7.2, fontSize: 14,
  color: theme.content.secondary, italic: true
});

pres.writeFile({ fileName: 'enhanced_presentation.pptx' });
""", packages='pptxgenjs')
```

---

## Documentation Navigation

### 📚 Core Documentation
- **[Core Rules & Themes](docs/core-rules-and-themes.md)** - Combined execution rules, complete theme system, **five-level title hierarchy guidelines**, and **key text emphasis guidelines**
- **[Slide Patterns](docs/slide-patterns.md)** - Ready-to-use slide templates and patterns
- **[Five-Level Hierarchy Example](docs/five-level-hierarchy-example.md)** - Complete example of five-level title hierarchy with key text emphasis

### 🖼️ Image Generation Workflow
- **[Image Generation Workflow](docs/image-generation-workflow.md)** - Combined image tools, mandatory workflow, **detailed chart generation guidelines**, and **data accuracy requirements**

### 🎨 Presentation Modes
- **[Mixed Mode Guide](docs/mixed-mode-guide.md)** - Default mode with 30% full-image + 70% mixed slides, includes layouts
- **[Full Image Mode](docs/full-image-mode.md)** - 100% full-image slides with embedded text

### 🔧 Advanced Features
- **[Existing Presentations](docs/existing-presentations.md)** - Reading and editing existing PPTX files

---

## Available Tools

### Presentation Tools
| Tool | Purpose |
|-------|---------|
| `execute_javascript_code` | Generate presentations using JavaScript (use packages='pptxgenjs') |
| `read_pptx` | Extract content/structure from existing PPTX files |
| `unpack_pptx` | Extract PPTX files to directory for inspection/learning from templates |
| `pack_pptx` | Package directory back into PPTX file after manual editing |
| `clean_pptx` | Clean orphaned files from unpacked PPTX directory |

### Image Tools for Presentations
| Tool | Purpose | Use Case in Presentations |
|------|---------|---------------------------|
| `generate_image` | Generate high-quality images from text prompts | Create custom visuals, charts, diagrams, illustrations for slides |
| `image_web_search` | Search and download images from the web | Find relevant photos, icons, backgrounds for presentation content |
| `logo_overlay` | Overlay logos on images with customizable position/size | Add company logos to slide images, create branded visuals |
| `edit_image` | Edit existing images using text prompts | Modify images to fit presentation theme, adjust colors, add text |

### File Tools for Presentations
| Tool | Purpose | Use Case in Presentations |
|------|---------|---------------------------|
| `read_file` | Read file content from workspace | Read configuration files, data files, templates |
| `write_file` | Write content to file in workspace | Save generated content, create configuration files |
| `list_files` | List files in workspace directory | Check available images, templates, data files |
| `file_exists` | Check if file exists in workspace | Verify image files exist before adding to presentation |
| `get_file_info` | Get detailed file information | Check file sizes, modification times for images |

**重要提示**: 所有文件工具都在工作空间内操作，使用相对路径（如`'generated_image.png'`），而不是绝对路径。

**Note**: All generated/downloaded images are automatically saved to the workspace output directory, ready to be used in presentations.

---

## Getting Help

- Start with **Quick Start** above for basic usage
- Check **Available Tools** for tool references
- Use the **Documentation Navigation** to explore specific topics
- Follow **CRITICAL REQUIREMENTS** to avoid common mistakes
