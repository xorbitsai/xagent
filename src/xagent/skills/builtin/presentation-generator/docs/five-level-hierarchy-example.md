# 五级标题层级与重点强调示例

## 概述

本文档展示如何使用五级标题层级系统和重点强调功能创建专业PPT。

## 五级标题层级系统

### 1. 大标题 (Level 1: Main Title)
- **字体大小**: 48-56pt
- **字体重量**: Bold (700)
- **颜色**: `theme.primary`
- **位置**: 幻灯片顶部 (y: 0.5-1.0)
- **样式**: 通常带有强调线下划线

### 2. 小标题 (Level 2: Sub Title)
- **字体大小**: 36-44pt
- **字体重量**: Bold (700)
- **颜色**: `theme.accent`
- **位置**: 大标题下方 (y: 1.8-2.5)
- **样式**: 可使用虚线分隔线

### 3. 二级标题 (Level 3: Secondary Title)
- **字体大小**: 28-32pt
- **字体重量**: Semi-bold (600) 或 Bold
- **颜色**: `theme.secondary`
- **位置**: 小标题下方 (y: 2.6-3.2)
- **样式**: 无特殊装饰

### 4. 内容标题 (Level 4: Content Title)
- **字体大小**: 22-26pt
- **字体重量**: Semi-bold (600)
- **颜色**: `theme.text` 或 `theme.secondary`
- **位置**: 内容区域顶部 (y: 3.2-4.0)
- **样式**: 可使用项目符号或缩进

### 5. 内容 (Level 5: Content)
- **字体大小**: 18-22pt
- **字体重量**: Normal (400)
- **颜色**: `theme.text`
- **位置**: 内容标题下方 (y: 3.8+)
- **样式**: 标准段落或项目符号

## 重点文字强调

### 强调方式
1. **颜色强调**: 使用`theme.accent`颜色突出显示关键文字
2. **加粗强调**: 在原有基础上加粗显示
3. **成功/警告强调**: 使用`theme.success`(绿色)表示积极数据，`theme.warning`(橙色)表示需要注意的数据
4. **独立显示**: 将关键数据单独显示并强调

### 强调原则
- 每页幻灯片重点强调不超过3-5处
- 确保强调内容真正重要
- 保持整体视觉平衡
- 避免使用过多颜色造成混乱

## 完整示例代码

```javascript
execute_javascript_code("""
const PptxGenJS = require('pptxgenjs');

// 使用NOVA主题
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
    success: '#10B981',    // 成功/积极数据
    warning: '#F59E0B',    // 警告/需要注意的数据
    text: '#0A0F1C'
  }
};

const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// --- 幻灯片1: 五级标题层级示例 ---
const slide1 = pres.addSlide();
slide1.background = { color: theme.content.background };

// Level 1: 大标题
slide1.addText('2024年度数字化转型报告', {
  x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

// 强调线下划线
slide1.addShape(pres.ShapeType.line, {
  x: 0.5, y: 1.4, w: 9, h: 0,
  line: { color: theme.content.accent, width: 3 }
});

// Level 2: 小标题
slide1.addText('技术投资与业务成果', {
  x: 0.5, y: 1.8, fontSize: 40, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// 虚线分隔线
slide1.addShape(pres.ShapeType.line, {
  x: 0.5, y: 2.3, w: 4, h: 0,
  line: { color: theme.content.secondary, width: 1, dashType: 'dash' }
});

// Level 3: 二级标题
slide1.addText('关键绩效指标分析:', {
  x: 0.5, y: 2.6, fontSize: 30, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

// Level 4: 内容标题 (左栏)
slide1.addText('技术投资回报:', {
  x: 0.8, y: 3.2, fontSize: 24, bold: true,
  color: theme.content.text, fontFace: 'Arial'
});

// Level 5: 内容 (左栏)
const leftContent = [
  '云计算投资: $2.5M',
  'AI/ML平台: $1.8M',
  '网络安全: $1.2M',
  '数字化转型: $3.0M'
];

leftContent.forEach((text, i) => {
  slide1.addText(text, {
    x: 0.8, y: 3.6 + i * 0.6,
    fontSize: 20, color: theme.content.text,
    bullet: true, lineSpacing: 24, fontFace: 'Arial'
  });
});

// Level 4: 内容标题 (右栏)
slide1.addText('业务成果指标:', {
  x: 5.0, y: 3.2, fontSize: 24, bold: true,
  color: theme.content.text, fontFace: 'Arial'
});

// Level 5: 内容 (右栏)
const rightContent = [
  '运营效率提升: 35%',
  '客户满意度: +28%',
  '收入增长: $15.2M',
  '成本节约: $4.8M'
];

rightContent.forEach((text, i) => {
  slide1.addText(text, {
    x: 5.0, y: 3.6 + i * 0.6,
    fontSize: 20, color: theme.content.text,
    bullet: true, lineSpacing: 24, fontFace: 'Arial'
  });
});

// --- 重点文字强调 ---
// 使用强调色突出关键数据
slide1.addText('$15.2M', {
  x: 6.5, y: 4.8, fontSize: 20, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// 使用成功色表示积极数据
slide1.addText('+35%', {
  x: 6.5, y: 3.6, fontSize: 20, bold: true,
  color: theme.content.success, fontFace: 'Arial'
});

slide1.addText('+28%', {
  x: 6.5, y: 4.2, fontSize: 20, bold: true,
  color: theme.content.success, fontFace: 'Arial'
});

// 使用警告色表示需要注意的数据
slide1.addText('$4.8M', {
  x: 6.5, y: 6.0, fontSize: 20, bold: true,
  color: theme.content.warning, fontFace: 'Arial'
});

// 关键结论强调
slide1.addText('关键结论: 技术投资带来显著业务回报，ROI达到215%', {
  x: 0.5, y: 6.2, fontSize: 22, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// 数据来源标注
slide1.addText('数据来源: 内部财务与技术报告 2024年度 | 注: 所有数据基于实际业务指标', {
  x: 0.5, y: 6.8, fontSize: 14,
  color: theme.content.secondary, italic: true, fontFace: 'Arial'
});

// --- 幻灯片2: 数据准确性示例 ---
const slide2 = pres.addSlide();
slide2.background = { color: theme.content.background };

// Level 1: 大标题
slide2.addText('季度销售数据准确分析', {
  x: 0.5, y: 0.8, w: 9, fontSize: 52, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

slide2.addShape(pres.ShapeType.line, {
  x: 0.5, y: 1.4, w: 9, h: 0,
  line: { color: theme.content.accent, width: 3 }
});

// Level 2: 小标题
slide2.addText('2024年各季度表现', {
  x: 0.5, y: 1.8, fontSize: 40, bold: true,
  color: theme.content.accent, fontFace: 'Arial'
});

// 准确的数据表格
const quarterlyData = [
  { quarter: 'Q1 2024', revenue: '$2.4M', growth: '+15%', profit: '$0.8M' },
  { quarter: 'Q2 2024', revenue: '$2.7M', growth: '+25%', profit: '$1.0M' },
  { quarter: 'Q3 2024', revenue: '$3.0M', growth: '+20%', profit: '$1.2M' },
  { quarter: 'Q4 2024', revenue: '$3.3M', growth: '+17%', profit: '$1.4M' }
];

// 表头
slide2.addText('季度', {
  x: 0.8, y: 2.5, fontSize: 22, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

slide2.addText('收入', {
  x: 2.5, y: 2.5, fontSize: 22, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

slide2.addText('同比增长', {
  x: 4.2, y: 2.5, fontSize: 22, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

slide2.addText('利润', {
  x: 6.0, y: 2.5, fontSize: 22, bold: true,
  color: theme.content.primary, fontFace: 'Arial'
});

// 表格数据
quarterlyData.forEach((row, i) => {
  const yPos = 3.0 + i * 0.7;

  // 季度
  slide2.addText(row.quarter, {
    x: 0.8, y: yPos, fontSize: 20,
    color: theme.content.text, fontFace: 'Arial'
  });

  // 收入 - 使用强调色
  slide2.addText(row.revenue, {
    x: 2.5, y: yPos, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
  });

  // 增长率 - 使用成功色
  slide2.addText(row.growth, {
    x: 4.2, y: yPos, fontSize: 20, bold: true,
    color: theme.content.success, fontFace: 'Arial'
  });

  // 利润 - 使用强调色
  slide2.addText(row.profit, {
    x: 6.0, y: yPos, fontSize: 20, bold: true,
    color: theme.content.accent, fontFace: 'Arial'
  });
});

// 数据准确性说明
slide2.addText('数据准确性说明:', {
  x: 0.5, y: 5.8, fontSize: 24, bold: true,
  color: theme.content.secondary, fontFace: 'Arial'
});

const accuracyNotes = [
  '所有数据基于实际财务报表',
  '收入数据精确到十万位',
  '增长率计算基于同比数据',
  '利润为税后净利润'
];

accuracyNotes.forEach((text, i) => {
  slide2.addText(text, {
    x: 0.8, y: 6.2 + i * 0.6,
    fontSize: 18, color: theme.content.text,
    bullet: true, lineSpacing: 22, fontFace: 'Arial'
  });
});

// 关键发现强调
slide2.addText('关键发现: Q2增长最快(+25%)，Q4收入最高($3.3M)', {
  x: 0.5, y: 7.0, fontSize: 22, bold: true,
  color: theme.content.warning, fontFace: 'Arial'
});

pres.writeFile({ fileName: 'five_level_hierarchy_example.pptx' });
""", packages='pptxgenjs')
```

## 数据准确性最佳实践

### 1. 明确数据单位
- 货币数据: `$15.2M` (明确显示货币单位和百万单位)
- 百分比数据: `+25%` (明确显示百分比符号)
- 数量数据: `1,500 units` (明确显示单位)

### 2. 准确的数据标签
- 坐标轴标签包含单位
- 数据点显示精确数值
- 图例清晰解释数据系列

### 3. 合理的数据比例
- 坐标轴比例尺能准确反映数据关系
- 避免扭曲的数据可视化
- 保持数据可比性

### 4. 数据来源标注
- 重要数据注明来源
- 说明数据时间范围
- 标注数据限制条件

## 使用建议

1. **层级一致性**: 在整个PPT中保持标题层级的一致性
2. **重点适度**: 每页重点强调3-5处关键信息
3. **颜色协调**: 使用主题颜色保持视觉协调
4. **数据准确**: 确保所有数据准确无误
5. **可读性优先**: 确保所有文字在投影时清晰可读

通过使用五级标题层级系统和重点强调功能，可以创建出结构清晰、重点突出、数据准确的专业PPT。

---
**导航**: [返回主文档](../SKILL.md) | [核心规则与主题](core-rules-and-themes.md) | [图像生成工作流](image-generation-workflow.md)
