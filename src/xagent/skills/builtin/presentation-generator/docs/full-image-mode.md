# 全图模式 (Full Image Mode)

## 概述 (Overview)

全图模式创建完全由图像组成的演示文稿，所有内容都嵌入在图像中，无可编辑文本。适合视觉演示、艺术展示或需要高度视觉冲击力的场合。

## 模式选择规则 (Mode Selection Rules)

### 触发条件
- **只有当用户明确要求时才使用全图模式**
- 触发关键词："全图"、"全图片"、"all images"、"full image"、"纯图片"
- 示例："生成全图的ppt"、"创建全图片演示"、"all image presentation"

### 默认模式
- 图文混合模式是默认模式
- 当用户没有明确指定时，使用图文混合模式

### 使用场景
- 视觉艺术作品集
- 摄影展示
- 品牌视觉识别
- 概念演示
- 情绪板展示

## 模式要求 (Mode Requirements)

### A. 全图模式 (Full Image Template)

#### 图像覆盖率 (Image Coverage)
- **100% 全图幻灯片**
- 每张幻灯片都是完整的图像
- 无单独的可编辑文本元素

#### 文本处理 (Text Processing)
- **所有文本都嵌入在图像中**
- **无可编辑文本**
- 文本必须在图像生成时包含在提示中

#### 布局选项 (Layout Options)
- **全屏图像** - 图像填充整个幻灯片
- **无混合布局** - 只有全图模式

#### 封面规则 (Cover Rules)
- **封面也是全图**
- 标题和其他文本都嵌入在封面图像中
- 使用高质量背景图像

#### 数据可视化 (Data Visualization)
- **所有数据内容必须转换为3D图表并嵌入图像中**
- 图表和解释文本都在同一图像中
- 确保图表在图像中清晰可见

## 工作流程 (Workflow)

### 1. 规划幻灯片内容
```python
# 示例：5张幻灯片的全图演示文稿
slide_content = [
    {
        "type": "cover",
        "prompt": "Professional presentation cover with title 'Annual Report 2024' and subtitle 'Visual Summary', dark background, futuristic design",
        "size": "1920x1080"
    },
    {
        "type": "data",
        "prompt": "3D infographic showing quarterly sales data: Q1 $1.2M, Q2 $1.5M, Q3 $1.8M, Q4 $2.1M. Include title 'Sales Performance' and data labels in the image",
        "size": "1920x1080"
    },
    {
        "type": "analysis",
        "prompt": "Market analysis visualization with charts and key insights text embedded. Professional design, clear data presentation",
        "size": "1920x1080"
    },
    {
        "type": "strategy",
        "prompt": "Strategic roadmap illustration with timeline and milestones. Include explanatory text within the image",
        "size": "1920x1080"
    },
    {
        "type": "conclusion",
        "prompt": "Conclusion slide with key takeaways and thank you message embedded in elegant design",
        "size": "1920x1080"
    }
]
```

### 2. 生成全图图像
```python
# 生成所有幻灯片图像
for i, slide in enumerate(slide_content):
    generate_image(
        prompt=slide["prompt"],
        size=slide["size"]
    )
    # 图像将保存为 generated_image_0.png, generated_image_1.png 等
```

### 3. 创建全图演示文稿
```javascript
const pres = new PptxGenJS();
pres.layout = 'LAYOUT_WIDE';

// 所有幻灯片都是全图
for (let i = 0; i < 5; i++) {
    const slide = pres.addSlide();
    slide.addImage({
        path: `generated_image_${i}.png`,
        x: 0, y: 0,
        w: '100%', h: '100%'
    });
}

pres.writeFile({ fileName: 'full_image_presentation.pptx' });
```

## 图像生成提示技巧 (Image Generation Prompt Tips)

### 包含所有文本内容
```python
# 好的提示 - 包含所有文本
generate_image(
    prompt="Business presentation slide with title 'Q4 Results' and bullet points: • Revenue: $2.1M • Growth: +25% • Profit Margin: 18%. Professional design, dark background, white text",
    size="1920x1080"
)

# 不好的提示 - 缺少文本
generate_image(
    prompt="Business chart with data",
    size="1920x1080"
)
```

### 指定文本样式
```python
generate_image(
    prompt="Infographic with large title 'MARKET TRENDS' in bold font, subtitle '2024 Analysis' in smaller font, and 3 key points in bullet format. Clean layout, professional colors",
    size="1920x1080"
)
```

### 数据可视化
```python
generate_image(
    prompt="3D pie chart showing market share: Company A 35%, Company B 25%, Company C 20%, Others 20%. Include legend and percentage labels within the chart. Title: 'Market Share Distribution 2024'",
    size="1920x1080"
)
```

## 质量检查清单 (Quality Checklist)

### 图像要求
- [ ] 每张幻灯片都是完整的图像
- [ ] 所有文本都嵌入在图像中
- [ ] 图像质量高，分辨率适当
- [ ] 图像尺寸匹配幻灯片尺寸

### 文本要求
- [ ] 所有必要文本都包含在图像提示中
- [ ] 文本在图像中清晰可读
- [ ] 文本大小和颜色适当
- [ ] 无遗漏的关键信息

### 布局要求
- [ ] 图像填充整个幻灯片
- [ ] 重要内容在安全区域内
- [ ] 视觉层次清晰
- [ ] 颜色对比度足够

### 技术要求
- [ ] 使用 `generate_image` 生成所有图像
- [ ] 图像尺寸一致
- [ ] 文件大小合理
- [ ] 加载性能良好

## 优点和限制 (Advantages and Limitations)

### 优点
- **高度视觉一致性** - 所有元素都在同一图像中
- **无格式问题** - 在不同设备上显示一致
- **创意自由** - 可以创建独特的视觉设计
- **文件管理简单** - 只有图像文件

### 限制
- **文本不可编辑** - 需要重新生成图像来修改文本
- **文件大小较大** - 高分辨率图像增加文件大小
- **可访问性挑战** - 屏幕阅读器无法读取嵌入文本
- **更新困难** - 修改内容需要重新生成图像

## 使用场景 (Use Cases)

### 适合全图模式
- 视觉艺术作品集
- 摄影展示
- 品牌视觉识别
- 概念演示
- 情绪板展示

### 不适合全图模式
- 需要频繁更新的报告
- 数据密集的财务报告
- 需要可访问性的文档
- 需要协作编辑的演示文稿

## 示例模板 (Example Template)

完整的全图模式示例可在 [full-image-example.md](full-image-example.md) 中找到。
