# Image Generation Workflow

## Overview

This document combines image tool usage guidelines with the mandatory workflow for generating all presentation images using `generate_image`.

## Available Image Tools

| Tool | Purpose | Use Case in Presentations |
|------|---------|---------------------------|
| `generate_image` | Generate high-quality images from text prompts | Create custom visuals, charts, diagrams, illustrations for slides |
| `image_web_search` | Search and download images from the web | Find relevant photos, icons, backgrounds for presentation content |
| `logo_overlay` | Overlay logos on images with customizable position/size | Add company logos to slide images, create branded visuals |
| `edit_image` | Edit existing images using text prompts | Modify images to fit presentation theme, adjust colors, add text |

**Note**: All generated/downloaded images are automatically saved to the workspace output directory, ready to be used in presentations.

## Mandatory Image Generation Requirement

**CRITICAL**: All presentations MUST use generated images from `generate_image`. External URLs or pre-existing images are NOT allowed unless absolutely necessary.

### Why Mandatory Image Generation?
1. **Consistency**: All images follow the same visual style and quality
2. **Customization**: Images are tailored to your specific content
3. **Brand Safety**: No risk of copyright issues from external sources
4. **Quality Control**: Consistent resolution, aspect ratios, and styling

### Workflow Rules
1. **NO EXTERNAL URLS**: Never use `https://` URLs in `pres.addImage()`
2. **GENERATE ALL VISUALS**: Use `generate_image` for charts, diagrams, illustrations, backgrounds
3. **THEME MATCHING**: Generate images that match your presentation theme colors
4. **PROPER SIZING**: Use appropriate dimensions for slide layouts

## Using Image Tools in Presentations

### Basic Image Generation

```python
# Generate a cover image
generate_image(
    prompt="Abstract futuristic background with blue and purple gradients, tech pattern, clean design, presentation cover",
    size="1920x1080"
)

# Generate a 3D chart image (REQUIRED for data)
generate_image(
    prompt="3D精美 bar chart showing growth metrics: 2022 75%, 2023 85%, 2024 95%. Professional 3D design, gradient bars, data labels, realistic lighting, high quality",
    size="1024x768"
)
```

### Web Image Search (Use with Caution)

```python
# Search for relevant images (only when absolutely necessary)
image_web_search(
    query="business team collaboration modern office",
    count=3,
    image_size="large"
)
```

### Logo Overlay

```python
# Add logo to generated image
logo_overlay(
    base_image_path="generated_chart.png",
    logo_path="company_logo.png",
    position="bottom-right",
    size="small"
)
```

### Image Editing

```python
# Edit image to match theme
edit_image(
    image_path="downloaded_image.jpg",
    prompt="Change background to dark blue (#0A0F1C) and add purple (#7C3AED) accents to match NOVA theme"
)
```

## Image Sizing Guidelines

| Slide Type | Recommended Size | Aspect Ratio | Use Case |
|------------|-----------------|--------------|----------|
| Full background | 1920x1080 | 16:9 | Cover slides, full-slide images |
| Content image | 1024x768 | 4:3 | Standard slide images |
| Wide content | 1280x720 | 16:9 | Widescreen presentations |
| Small graphic | 800x600 | 4:3 | Icons, small illustrations |

## 数据可视化图像生成 (Data Visualization Image Generation)

**CRITICAL REQUIREMENT**: All data visualization MUST use detailed, specific chart types with clear and accurate data representation. Avoid abstract or conceptual images for data.

**数据准确性要求**:
1. **数据必须准确**：图表中的数值必须与描述完全一致
2. **标签必须清晰**：坐标轴、数据点、图例必须有清晰可读的标签
3. **比例必须合理**：坐标轴比例尺必须能准确反映数据关系
4. **单位必须明确**：所有数据必须包含明确的单位（如$、%、M等）
5. **来源必须注明**：重要数据应注明来源或说明

### 图表类型选择指南

根据数据类型选择合适的图表类型：

| 数据类型 | 推荐图表类型 | 用途 |
|----------|-------------|------|
| 趋势变化 | **折线图 (Line Chart)** | 显示随时间变化的趋势 |
| 类别比较 | **柱状图 (Bar Chart)** | 比较不同类别的数值 |
| 比例分布 | **饼图/环形图 (Pie/Doughnut Chart)** | 显示各部分占总体的比例 |
| 相关性 | **散点图 (Scatter Plot)** | 显示两个变量之间的关系 |
| 分布密度 | **热力图 (Heatmap)** | 显示数据密度或强度分布 |
| 多指标 | **组合图 (Combination Chart)** | 同时显示多种数据类型 |

### 详细图表生成示例（增强数据准确性）

```python
# 1. 折线图 - 显示趋势变化 (必须包含准确数据、坐标轴、单位)
generate_image(
    prompt="Professional 3D line chart showing accurate quarterly revenue growth from Q1 2023 to Q4 2024. X-axis: Q1 2023, Q2 2023, Q3 2023, Q4 2023, Q1 2024, Q2 2024, Q3 2024, Q4 2024. Y-axis: Revenue (in millions USD). Exact data points: Q1 2023: $1.2M, Q2 2023: $1.5M, Q3 2023: $1.8M, Q4 2023: $2.1M, Q1 2024: $2.4M, Q2 2024: $2.7M, Q3 2024: $3.0M, Q4 2024: $3.3M. Include: grid lines, axis labels with units, data point markers showing exact values, trend line, clear legend, title 'Quarterly Revenue Growth 2023-2024'. Design: Professional 3D design with gradient colors matching NOVA theme (#0A0F1C background, #7C3AED accent), realistic lighting. Data accuracy: All values must be accurately plotted at correct positions.",
    size="1024x768"
)

# 2. 柱状图 - 类别比较 (必须包含准确数值、清晰标签)
generate_image(
    prompt="Accurate 3D bar chart comparing regional sales performance in Q4 2024. Regions: North America, Europe, Asia Pacific, South America. Exact sales data: North America: $5.2M, Europe: $3.8M, Asia Pacific: $4.5M, South America: $2.1M. Each bar must be clearly labeled with region name and exact value (e.g., 'North America: $5.2M'). Include: Y-axis labeled 'Sales (in millions USD)' with appropriate scale, grid lines, color-coded bars using gradient blue to purple, legend, title 'Regional Sales Performance Q4 2024'. Design: Professional 3D design with realistic lighting, shadows, and depth. Data accuracy: Bars must accurately represent the values, height proportional to data.",
    size="1024x768"
)

# 3. 饼图 - 比例分布 (必须包含准确百分比、清晰标签)
generate_image(
    prompt="Accurate 3D pie chart showing market share distribution for 2024. Exact segments: Company A: 35%, Company B: 25%, Company C: 20%, Other Competitors: 20%. Each segment must be clearly labeled with company name and exact percentage (e.g., 'Company A: 35%'). Include: legend explaining each segment, 3D depth effect, gradient colors (use theme colors: #0A0F1C, #7C3AED, #22D3EE, #94A3B8), title 'Market Share Distribution 2024'. Design: Professional visualization with clear contrast between segments, readable labels. Data accuracy: Segment sizes must accurately represent percentages.",
    size="1024x768"
)

# 4. 组合图 - 多指标分析 (必须包含双坐标轴、准确数据)
generate_image(
    prompt="Accurate combination chart showing revenue (bars) and growth rate (line) by quarter for 2024. Primary Y-axis (left): Revenue in millions USD. Secondary Y-axis (right): Growth rate in percentage. Quarters: Q1, Q2, Q3, Q4. Exact revenue data: Q1: $2.4M, Q2: $2.7M, Q3: $3.0M, Q4: $3.3M. Exact growth rate data: Q1: 15%, Q2: 25%, Q3: 20%, Q4: 17%. Include: dual Y-axes with clear labels and units, grid lines, clear legend distinguishing 'Revenue (bars)' and 'Growth Rate (line)', data labels on each bar and data point showing exact values, title 'Revenue & Growth Rate by Quarter 2024'. Design: Professional design with theme colors, bars in blue gradient, line in purple. Data accuracy: All values must be accurately represented on respective axes.",
    size="1024x768"
)

# 5. 散点图 - 相关性分析 (必须包含准确坐标点)
generate_image(
    prompt="Accurate scatter plot showing correlation between marketing spend and sales revenue. X-axis: Marketing Spend (in thousands USD). Y-axis: Sales Revenue (in millions USD). Exact data points: (50, 1.2), (75, 1.8), (100, 2.5), (125, 3.2), (150, 3.8), (175, 4.3), (200, 4.7). Each point labeled with coordinates. Include: trend line showing correlation, grid lines, axis labels with units, legend, title 'Marketing Spend vs Sales Revenue Correlation'. Design: Professional design with theme colors. Data accuracy: Points must be accurately plotted at correct coordinates.",
    size="1024x768"
)
```

### 图表生成最佳实践与数据准确性要求

```python
# 必须包含的关键元素和数据准确性要求
required_elements = [
    "clear axis labels with units",      # 清晰的坐标轴标签（含单位）
    "accurate data labels/values",       # 准确的数据标签/数值
    "grid lines for readability",        # 提高可读性的网格线
    "legend for data series",           # 数据系列图例
    "appropriate and consistent scale",  # 合适且一致的比例尺
    "professional theme colors",         # 专业主题配色
    "data source notation if applicable", # 数据来源标注（如适用）
]

# 数据准确性验证清单
data_accuracy_checklist = [
    "所有数值与描述一致",
    "单位明确且正确",
    "比例尺合理反映数据关系",
    "数据标签无歧义",
    "图例清晰解释数据系列",
    "坐标轴标签包含单位",
    "数据点标记准确",
]

# 好的图表提示词结构（增强数据准确性）
good_chart_prompt = """
{chart_type} showing {data_description}.
Data: {specific_data_points_with_units}.
Include: {required_elements}.
Design: {design_requirements}.
Theme: {theme_colors_if_applicable}.
Data accuracy: All values must be accurately represented, labels must be clear and readable.
"""

# 示例：完整的折线图提示词（增强版）
complete_line_chart_prompt = """
Professional 3D line chart showing monthly user growth from January to December 2024.
Months: Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec.
Growth data: Jan: 1,000 users, Feb: 1,200, Mar: 1,500, Apr: 1,800, May: 2,200, Jun: 2,600, Jul: 3,100, Aug: 3,700, Sep: 4,400, Oct: 5,200, Nov: 6,100, Dec: 7,100.
Include: X-axis labeled "Month", Y-axis labeled "Number of Users", grid lines, data point markers with exact values, trend line, clear legend, title "Monthly User Growth 2024".
Design: Clean modern 3D design with blue gradient line (#7C3AED), light gray grid, white background, professional typography.
Data accuracy: All values must be accurately plotted, labels must show exact numbers.
Size: 1024x768.
"""

# 示例：准确的柱状图提示词
accurate_bar_chart_prompt = """
Professional 3D bar chart comparing regional sales performance in Q4 2024.
Regions: North America, Europe, Asia Pacific, South America.
Sales data: North America: $5.2M, Europe: $3.8M, Asia Pacific: $4.5M, South America: $2.1M.
Include: X-axis labeled "Region", Y-axis labeled "Sales (in millions USD)", grid lines, each bar labeled with exact value and region name, color-coded bars, legend, title "Regional Sales Performance Q4 2024".
Design: 3D design with gradient colors matching NOVA theme, realistic lighting and shadows, professional layout.
Data accuracy: Bars must accurately represent the values, labels must show exact amounts with $ and M notation.
Size: 1024x768.
"""

# 避免的抽象提示词
abstract_prompt = "Chart showing growth"  # ❌ 太抽象，缺少具体数据
abstract_prompt2 = "Beautiful data visualization"  # ❌ 没有具体图表类型和数据
vague_prompt = "Chart with some data"  # ❌ 数据不明确
inaccurate_prompt = "Chart showing growth around 20%"  # ❌ 数据不准确
```

### 数据可视化质量与准确性检查清单

**数据准确性检查：**
- [ ] **数据准确无误**：图表中的数值与描述完全一致
- [ ] **单位明确正确**：所有数据包含正确单位（$、%、M等）
- [ ] **标签清晰可读**：坐标轴、数据点、图例都有清晰可读的标签
- [ ] **比例合理一致**：坐标轴比例尺能准确反映数据关系
- [ ] **数据来源注明**：重要数据注明来源或说明

**图表质量检查：**
- [ ] **图表类型合适**：根据数据类型选择了正确的图表类型
- [ ] **数据完整呈现**：包含了所有需要展示的数据点
- [ ] **颜色专业协调**：使用主题颜色或专业配色方案
- [ ] **设计易于理解**：图表能直观传达数据信息
- [ ] **无歧义误导**：不会引起误解或错误解读
- [ ] **3D效果适当**：3D设计增强可读性而非造成扭曲

**可读性检查：**
- [ ] **字体大小合适**：所有文字在幻灯片上清晰可读
- [ ] **对比度足够**：文字与背景有足够对比度
- [ ] **布局平衡**：图表元素布局平衡美观
- [ ] **重点突出**：关键数据得到适当突出
- [ ] **整体协调**：图表与幻灯片整体风格协调

## Step-by-Step Mandatory Workflow

### Step 1: Plan Your Image Needs

For each slide, identify what images are needed:
- Cover slide: Background image
- Content slides: Charts, diagrams, illustrations
- Data slides: Infographics, metrics visualizations (MUST be 3D)
- Team/About slides: Conceptual team illustrations

### Step 2: Generate Images with Theme-Aware Prompts

```python
# Example: Generate theme-aware images for NOVA theme
# NOVA theme colors: background #0A0F1C (dark blue), accent #7C3AED (purple)

# Cover background image
generate_image(
    prompt="Abstract futuristic background with dark blue (#0A0F1C) and purple (#7C3AED) gradients, geometric patterns, clean professional design, presentation cover, high quality",
    size="1920x1080"
)

# Business chart for metrics slide (MUST be 3D)
generate_image(
    prompt="3D精美 business chart showing quarterly growth with blue and purple bars matching NOVA theme colors, 3D design, realistic lighting, professional data visualization, grid lines, data labels, high quality",
    size="1024x768"
)

# Architecture diagram for technical slide
generate_image(
    prompt="Technology architecture diagram with blue (#0A0F1C) and purple (#7C3AED) color scheme, clean lines, boxes and arrows, professional tech illustration, white background",
    size="1280x720"
)
```

### Step 3: Create Presentation with Generated Images

**重要提示**: 在JavaScript代码中引用图像文件时，使用相对路径（如`'generated_image.png'`）。所有生成的图像会自动保存到工作空间输出目录。

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
pres.layout = 'LAYOUT_WIDE';

// Helper function to add image with fallback text
function addImageWithFallback(slide, path, x, y, w, h) {
  try {
    slide.addImage({ path: path, x: x, y: y, w: w, h: h });
    return true;
  } catch (error) {
    console.log(`Image not found or error loading: ${path}, error: ${error.message}`);
    slide.addText(`[Image: ${path}]`, {
      x: x, y: y, w: w, h: h,
      fontSize: 14, color: theme.content.secondary,
      align: 'center', valign: 'middle'
    });
    return false;
  }
}

// --- Slide 0: Cover with generated background (title embedded in image) ---
// Note: First slide (cover) doesn't need pres.addSlide()
// Title should be embedded in the generated cover image, not added as text
if (addImageIfExists(pres, 'generated_cover_background.png', 0, 0, 13.33, 7.5)) {
  // Add semi-transparent overlay for better readability if needed
  pres.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: '100%', h: '100%',
    fill: { color: '000000', transparency: 40 }
  });
}
// IMPORTANT: For mixed mode, cover title MUST be embedded in the image
// Do NOT use pres.addText() for cover slides in mixed mode
// The generated cover image should include the title text

// --- Slide 1: Business Metrics with generated chart ---
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };
slide1.addText('Quarterly Performance Metrics', {
  x: 0.5, y: 0.5, w: 12.33, fontSize: 44, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});
slide1.addShape(pres.ShapeType.line, {
  x: 0.5, y: 1.2, w: 12.33, h: 0,
  line: { color: theme.content.accent, width: 2 }
});

// Add generated chart
addImageIfExists(slide1, 'generated_business_chart.png', 0.5, 1.5, 12.33, 5);

pres.writeFile({ fileName: 'strategy_with_generated_images.pptx' })
  .then(fileName => console.log('Successfully saved: ' + fileName))
  .catch(err => console.error('Error saving presentation: ' + err));
""", packages='pptxgenjs')
```

## Theme-Aware Image Generation

### Include Theme Colors in Prompts

```python
# Example for NOVA theme
generate_image(
    prompt="Infographic with dark blue (#0A0F1C) background and purple (#7C3AED) highlights, matching NOVA theme",
    size="1280x720"
)

# Example for ORBIT theme
generate_image(
    prompt="Technology diagram with dark background (#0B1220) and cyan (#22D3EE) accents, ORBIT theme style",
    size="1024x768"
)
```

### Slide-Specific Prompts

```python
# Cover slide images
generate_image(
    prompt="Abstract professional background for presentation cover, {theme_colors}, minimalist, high quality",
    size="1920x1080"
)

# Data visualization slides
generate_image(
    prompt="Modern data chart showing growth trends, clean design, {theme_colors}, white background",
    size="1024x768"
)

# Process/flow slides
generate_image(
    prompt="Process flow diagram with arrows and boxes, {theme_colors}, professional illustration",
    size="1280x720"
)
```

## Adding Images to Presentations

### Basic Image Placement

```javascript
// Add full background image
pres.addImage({
    path: 'generated_cover_background.png',
    x: 0, y: 0,
    w: '100%', h: '100%'
});

// Add content image with specific dimensions
pres.addImage({
    path: 'generated_chart.png',
    x: 1, y: 1.5,
    w: 8, h: 4.5
});

// Add image with aspect ratio preservation
pres.addImage({
    path: 'team_photo.jpg',
    x: 1, y: 1.5,
    sizing: { type: 'contain', w: 8, h: 4.5 }
});
```

### Image with Overlay for Readability

```javascript
// Add image with semi-transparent overlay for text readability
pres.addImage({ path: 'background_image.png', x: 0, y: 0, w: '100%', h: '100%' });
pres.addShape(pres.ShapeType.rect, {
    x: 0, y: 0, w: '100%', h: '100%',
    fill: { color: '000000', transparency: 40 }
});
pres.addText('Title Text', {
    x: 1, y: 3, fontSize: 64, bold: true,
    color: theme.cover.title
});
```

## Complete Mandatory Workflow Template

```python
# TEMPLATE: Mandatory Image Generation Workflow
# Replace {presentation_topic} and {theme_name} with your values

# 1. Generate cover background
generate_image(
    prompt="Abstract professional background for {presentation_topic} presentation cover, {theme_name} theme colors, minimalist design",
    size="1920x1080"
)

# 2. Generate key visuals (adjust based on your slide count)
visuals_needed = [
    "business chart showing growth metrics",
    "architecture diagram for system overview",
    "team collaboration illustration",
    "process flow diagram",
    "infographic with key statistics"
]

for i, visual_desc in enumerate(visuals_needed):
    generate_image(
        prompt=f"{visual_desc}, {theme_name} theme colors, professional design for presentation slide",
        size="1024x768"
    )

# 3. Optional: Check if images were generated successfully
# You can use file_exists tool to verify images
image_files = [
    'generated_cover_background.png',
    'generated_business_chart.png',
    'generated_architecture_diagram.png',
    'generated_team_collaboration.png',
    'generated_process_flow.png',
    'generated_infographic.png'
]

for image_file in image_files:
    if file_exists(image_file):
        print(f"✓ Image generated: {image_file}")
    else:
        print(f"⚠ Image not found: {image_file}")

# 4. Create presentation with generated images
execute_javascript_code("""
const PptxGenJS = require('pptxgenjs');

// Select theme based on {theme_name}
const theme = {theme_configuration};

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// Helper function to add image with fallback
function addImageWithFallback(slide, path, x, y, w, h) {
  try {
    slide.addImage({ path: path, x: x, y: y, w: w, h: h });
    return true;
  } catch (error) {
    console.log(`Image error: ${path} - ${error.message}`);
    slide.addText(`[Image: ${path}]`, {
      x: x, y: y, w: w, h: h,
      fontSize: 14, color: theme.content.secondary,
      align: 'center', valign: 'middle'
    });
    return false;
  }
}

// Add slides with generated images
// Slide 0: Cover (first slide, no addSlide)
addImageWithFallback(pres, 'generated_cover_background.png', 0, 0, '100%', '100%');

// Slide 1: Content with chart
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };
slide1.addText('Business Metrics', {
  x: 0.5, y: 0.8, w: 12, fontSize: 44, bold: true,
  color: theme.content.primary
});
addImageWithFallback(slide1, 'generated_business_chart.png', 0.5, 1.5, 12, 6);

// Add more slides as needed...

pres.writeFile({ fileName: '{presentation_topic}_with_generated_images.pptx' });
""", packages='pptxgenjs')
```

## Quality Checklist

### Image Generation
- [ ] All images generated with `generate_image`
- [ ] No external URLs in `pres.addImage()`
- [ ] Images match presentation theme colors
- [ ] Proper image sizes for slide layouts
- [ ] All data visualizations use 3D精美图片 with accurate data

### File Management
- [ ] Image filenames referenced correctly in JavaScript
- [ ] Use relative paths (e.g., `'generated_image.png'`) not absolute paths
- [ ] Check image existence with `file_exists` before presentation generation
- [ ] Fallback text for missing images in presentation code

### Data Accuracy
- [ ] Chart data is accurate and clearly labeled
- [ ] Units are specified for all numerical data
- [ ] Data sources are cited where appropriate
- [ ] Charts use appropriate scales and proportions

## Best Practices

1. **Theme Consistency**: Generate images that match your presentation theme colors
2. **Proper Sizing**: Use appropriate dimensions for different slide layouts
3. **Quality Prompts**: Be specific in image generation prompts for better results
4. **File Management**: Generated images are automatically saved to workspace
5. **Fallback Handling**: Consider adding fallback text for missing images in presentations

---

**Navigation**: [Back to Main Documentation](../SKILL.md) | [Next: Mixed Mode Guide](mixed-mode-guide.md) | [Previous: Core Rules & Themes](core-rules-and-themes.md)
