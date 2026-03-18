# Core Rules & Theme System

## Execution Rules

**CRITICAL: MANDATORY constraints for stable presentation generation:**

1. **USE PREDEFINED THEMES ONLY**: You MUST use one of the 4 predefined themes (NOVA, ORBIT, PULSE, MINIMA) from the "Predefined Themes" section below. NEVER define custom colors or use hardcoded hex values like `"003366"` or `color: "FF0000"`.

2. **Theme Object Format**: Always define theme with cover and content variants:
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
   Then reference colors as `theme.cover.background`, `theme.content.primary`, etc.

3. **Slide Creation**: Always call `pres.addSlide()` before adding content to a new slide (except the first slide)

4. **Font Consistency & Hierarchy**: Maintain clear visual hierarchy with consistent font sizes and styles:
   - **Level 1: 大标题 (Main Title)**: 48-56pt, bold, theme.primary color, with underline or accent line
   - **Level 2: 小标题 (Sub Title)**: 36-44pt, bold, theme.accent color, may have lighter accent line
   - **Level 3: 二级标题 (Secondary Title)**: 28-32pt, semi-bold, theme.secondary color
   - **Level 4: 内容标题 (Content Title)**: 22-26pt, semi-bold, theme.text or theme.secondary color
   - **Level 5: 内容 (Content)**: 18-22pt, normal weight, theme.text color
   - **重点文字强调 (Key Text Emphasis)**: Same size as content, bold, theme.accent or theme.success/theme.warning color
   - **Caption/Small Text**: 14-16pt, normal weight, theme.secondary color
   - **Bullet Points**: 18-20pt, normal weight, theme.text color with bullet symbols

5. **Slide Bounds**: Never position elements outside 10 x 7.5 inch slide area (x: 0-10, y: 0-7.5)

6. **Layout Simplicity**: Prefer clean, simple layouts over dense or decorative designs

7. **Theme Selection**: Choose theme based on presentation context (see Theme Selection Guide below)

8. **Visual Hierarchy**: Slides must follow clear visual hierarchy: 大标题 → 小标题 → 二级标题 → 内容标题 → 内容 → 重点强调. Use appropriate font sizes and colors for each level. Avoid mixing too many font sizes on a single slide.

### 五级标题层级系统 (Five-Level Title Hierarchy System)

为了确保PPT内容的清晰层次和视觉区分，必须使用以下五级标题系统：

#### Level 1: 大标题 (Main Title) - 幻灯片主标题
- **Font Size**: 48-56pt
- **Weight**: Bold (700)
- **Color**: `theme.primary` (e.g., `#0A0F1C` in NOVA)
- **Style**: 通常带有下划线或强调线，使用主题主色
- **Position**: 幻灯片顶部 (y: 0.5-1.0)
- **Example**:
  ```javascript
  slide.addText('2024年度业务绩效报告', {
    x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
    color: theme.content.primary, fontFace: 'Arial'
  });
  // 添加强调线下划线
  slide.addShape(pres.ShapeType.line, {
    x: 0.5, y: 1.4, w: 9, h: 0,
    line: { color: theme.content.accent, width: 3 }
  });
  ```

#### Level 2: 小标题 (Sub Title) - 章节标题
- **Font Size**: 36-44pt
- **Weight**: Bold (700)
- **Color**: `theme.accent` (强调色)
- **Style**: 可使用较细的强调线或虚线分隔
- **Position**: 主标题下方 (y: 1.8-2.5)
- **Example**:
  ```javascript
  slide.addText('财务绩效分析', {
    x: 0.5, y: 2.0, fontSize: 40, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
  });
  // 可选：添加虚线分隔线
  slide.addShape(pres.ShapeType.line, {
    x: 0.5, y: 2.5, w: 4, h: 0,
    line: { color: theme.content.secondary, width: 1, dashType: 'dash' }
  });
  ```

#### Level 3: 二级标题 (Secondary Title) - 子章节标题
- **Font Size**: 28-32pt
- **Weight**: Semi-bold (600) 或 Bold
- **Color**: `theme.secondary` (次要色)
- **Style**: 无特殊装饰，清晰可读
- **Position**: 章节标题下方 (y: 2.6-3.2)
- **Example**:
  ```javascript
  slide.addText('关键收入指标:', {
    x: 0.5, y: 2.8, fontSize: 30, bold: true,
    color: theme.content.secondary, fontFace: 'Arial'
  });
  ```

#### Level 4: 内容标题 (Content Title) - 内容区块标题
- **Font Size**: 22-26pt
- **Weight**: Semi-bold (600)
- **Color**: `theme.text` 或 `theme.secondary`
- **Style**: 可使用项目符号或缩进
- **Position**: 内容区域顶部 (y: 3.2-4.0)
- **Example**:
  ```javascript
  slide.addText('季度收入趋势', {
    x: 0.8, y: 3.4, fontSize: 24, bold: true,
    color: theme.content.text, fontFace: 'Arial'
  });
  ```

#### Level 5: 内容 (Content) - 正文内容
- **Font Size**: 18-22pt
- **Weight**: Normal (400)
- **Color**: `theme.text`
- **Style**: 标准段落或项目符号
- **Line Spacing**: 1.2-1.5倍字体大小
- **Position**: 内容标题下方 (y: 3.8+)
- **Example**:
  ```javascript
  slide.addText('• 总收入: $15.2M (+25% YoY)\n• 毛利率: 68% (+5个百分点)\n• 营业利润: $4.8M (+32% YoY)', {
    x: 0.8, y: 3.8, fontSize: 20,
    color: theme.content.text, fontFace: 'Arial',
    bullet: true, lineSpacing: 24
  });
  ```

#### 重点文字强调 (Emphasis for Key Text)
- **Font Size**: 与所在内容相同
- **Weight**: Bold (700) 或保持原样
- **Color**: `theme.accent` (强调色) 或 `theme.success`/`theme.warning` (如有)
- **Style**: 使用强调色突出显示关键数据、重要结论
- **Example**:
  ```javascript
  // 在正文中突出关键数据
  slide.addText('本季度收入达到$15.2M，同比增长25%，创历史新高。', {
    x: 0.8, y: 4.8, fontSize: 20,
    color: theme.content.text, fontFace: 'Arial'
  });
  // 单独突出关键数据
  slide.addText('$15.2M', {
    x: 3.5, y: 4.8, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
  });
  slide.addText('+25%', {
    x: 5.0, y: 4.8, fontSize: 20, bold: true,
    color: theme.content.success, fontFace: 'Arial'
  });
  ```

#### 重点文字强调指南 (Key Text Emphasis Guidelines)

为了增强PPT的表现力和重点突出，必须对关键文字进行特殊强调：

**何时使用重点强调：**
- 关键数据指标（如收入、增长率、市场份额）
- 重要结论和发现
- 核心竞争优势
- 关键行动项和建议
- 风险警示和注意事项

**强调方式：**
1. **颜色强调**：使用`theme.accent`颜色突出显示关键文字
2. **加粗强调**：在原有基础上加粗显示
3. **组合强调**：同时使用特殊颜色和加粗
4. **独立显示**：将关键数据单独显示并强调

**示例：混合强调方式**
```javascript
// 在正文中嵌入强调
slide.addText('本季度总收入达到$15.2M，同比增长25%，超出市场预期。', {
  x: 0.8, y: 4.8, fontSize: 20,
  color: theme.content.text, fontFace: 'Arial'
});

// 单独强调关键数据
slide.addText('$15.2M', {
  x: 3.5, y: 4.8, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

slide.addText('+25%', {
  x: 5.0, y: 4.8, fontSize: 20, bold: true,
  color: theme.content.success, fontFace: 'Arial'
});

// 使用不同颜色表示不同含义
slide.addText('高风险', {
  x: 6.5, y: 4.8, fontSize: 20, bold: true,
  color: theme.content.warning, fontFace: 'Arial'
});
```

**避免过度强调：**
- 每页幻灯片重点强调不超过3-5处
- 确保强调内容真正重要
- 保持整体视觉平衡
- 避免使用过多颜色造成混乱

#### Caption & Small Text
- **Font Size**: 14-16pt
- **Weight**: Normal (400)
- **Color**: `theme.secondary`
- **Style**: Italic for captions, normal for footnotes
- **Example**:
  ```javascript
  slide.addText('Source: Internal analytics Q4 2024', {
    x: 0.5, y: 6.5, fontSize: 14,
    color: theme.content.secondary,
    italic: true
  });
  ```

9. **Image Sizing**: ALWAYS specify both width AND height, or use sizing to ensure images stay within bounds:
   - Use `w` and `h` together to control exact dimensions
   - Or use `sizing: { type: 'contain', w: 8, h: 5 }` to fit within bounds while maintaining aspect ratio
   - NEVER use only `w` or only `h` without the other - image may overflow
   - Keep images within content area: x: 0-10, y: 0-7.5 inches

10. **Mandatory Image Generation**: ALL images MUST be generated using `generate_image`:
    - NO external URLs (e.g., `https://example.com/image.jpg`)
    - NO pre-existing image files unless absolutely necessary
    - Generate theme-aware images that match your presentation colors
    - Use appropriate sizes: 1920x1080 for backgrounds, 1024x768 for content
    - Include theme colors in image prompts for visual consistency

## Layout Zones

For consistent slide layouts:

| Zone | X/Y Range | Purpose |
|------|-----------|---------|
| Safe content area | x: 0.5-9.5, y: 0.5-7 | Main content, images, data |
| Title area | y: 0.5 - 1.5 | Slide titles and headings |
| Content area | y: 1.5 - 5.5 | Main content, bullets, data |
| Footer area | y: 6.0 - 7.0 | Footer text, page numbers, notes |

**Note**: Title slides may use vertically centered positioning (y ≈ 3). All other slides must follow layout zones.

---

## Complete Theme System

### Predefined Themes

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

#### NOVA & ORBIT Themes

| Purpose | NOVA Cover | NOVA Content | ORBIT Cover | ORBIT Content |
|---------|-----------|--------------|-------------|----------------|
| Background | `#0A0F1C` | `#F6F7FB` | `#0B1220` | `#0F172A` |
| Title/Primary | `#FFFFFF` | `#0A0F1C` | `#F1F5F9` | `#F1F5F9` |
| Subtitle/Secondary | `#94A3B8` | `#5B6475` | `#94A3B8` | `#94A3B8` |
| Accent | `#7C3AED` | `#7C3AED` | `#22D3EE` | `#22D3EE` |
| Success | - | `#10B981` | - | `#34D399` |
| Warning | - | `#F59E0B` | - | `#FBBF24` |
| Text | - | `#0A0F1C` | - | `#F1F5F9` |

#### PULSE & MINIMA Themes

| Purpose | PULSE Cover | PULSE Content | MINIMA Cover | MINIMA Content |
|---------|-------------|---------------|--------------|-----------------|
| Background | `#111827` | `#FFFFFF` | `#111111` | `#FAFAFA` |
| Title/Primary | `#FFFFFF` | `#111827` | `#FFFFFF` | `#111111` |
| Subtitle/Secondary | `#9CA3AF` | `#6B7280` | `#999999` | `#777777` |
| Accent | `#EF4444` | `#EF4444` | `#FFFFFF` | `#000000` |
| Success | - | `#10B981` | - | - |
| Warning | - | `#F59E0B` | - | - |

### Theme Code Templates (Copy & Use)

**IMPORTANT**: Each theme has TWO variants - Cover (slide 0) and Content (slides 1+).

#### NOVA Theme (Strategy / AI / Investor)

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
    success: '#10B981',    // For positive emphasis (growth, success)
    warning: '#F59E0B',    // For caution/warning emphasis
    text: '#0A0F1C'
  }
};
```

#### ORBIT Theme (Technical / Architecture / Dev)

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
    success: '#34D399',    // For positive emphasis
    warning: '#FBBF24',    // For caution/warning emphasis
    text: '#F1F5F9'
  }
};
```

#### PULSE Theme (Metrics / Growth / Performance)

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

#### MINIMA Theme (Founder / Brand / Minimalist)

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

### Usage Example

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

## Theme Selection Guide

| Context | Recommended Theme | Why |
|---------|------------------|-----|
| Strategy / AI narrative / Investor deck | **NOVA** | Large titles, generous whitespace, authoritative feel |
| Technical deep dive / Architecture / Dev | **ORBIT** | Dark background, strong contrast, clean technical aesthetic |
| Metrics-heavy / Growth / Business performance | **PULSE** | Bold KPI emphasis, high-contrast numbers, data-focused |
| Founder story / Brand / Minimalist | **MINIMA** | Extremely clean, typography-driven, minimal decoration |

**Default**: Use NOVA if context is unclear or not specified.

---

**Navigation**: [Back to Main Documentation](../SKILL.md) | [Next: Image Generation Workflow](image-generation-workflow.md)
