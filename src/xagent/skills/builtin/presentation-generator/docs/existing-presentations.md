# Working with Existing Presentations

Tools for reading, editing, and modifying existing PowerPoint presentations.

## Available Tools for Existing Presentations

| Tool | Purpose |
|------|---------|
| `read_pptx` | Extract content/structure from existing PPTX files |
| `unpack_pptx` | Extract PPTX files to directory for inspection/learning from templates |
| `pack_pptx` | Package directory back into PPTX file after manual editing |
| `clean_pptx` | Clean orphaned files from unpacked PPTX directory |

## Reading Presentation Structure

### Basic Structure Extraction

```python
# Read presentation structure without extracting text
read_pptx("presentation.pptx", extract_text=False)
```

**Returns:**
```json
{
  "slide_count": 5,
  "slides": [
    {"index": 0, "filename": "slide1.xml", "hidden": false},
    {"index": 1, "filename": "slide2.xml", "hidden": false},
    {"index": 2, "filename": "slide3.xml", "hidden": false},
    {"index": 3, "filename": "slide4.xml", "hidden": false},
    {"index": 4, "filename": "slide5.xml", "hidden": false}
  ],
  "titles": ["Title", "Content", "Summary", "Q&A", "Thank You"]
}
```

### Extract Text Content

```python
# Read presentation with text extraction
read_pptx("presentation.pptx", extract_text=True)
```

**Returns all text content from the presentation**, organized by slide.

## Unpacking Presentations for Inspection

### Unpack PPTX to Directory

```python
# Extract PPTX file to directory for inspection
unpack_pptx("presentation.pptx", output_dir="presentation_unpacked")
```

This creates a directory structure:
```
presentation_unpacked/
├── [Content_Types].xml
├── _rels/
├── docProps/
│   ├── app.xml
│   ├── core.xml
│   └── custom.xml
└── ppt/
    ├── presentation.xml
    ├── presProps.xml
    ├── tableStyles.xml
    ├── viewProps.xml
    ├── _rels/
    ├── charts/
    ├── media/
    ├── slideLayouts/
    ├── slideMasters/
    └── slides/
        ├── slide1.xml
        ├── slide2.xml
        └── ...
```

### Learning from Templates

Unpacking is useful for:
1. **Learning slide layouts** from existing templates
2. **Extracting images and media** for reuse
3. **Understanding XML structure** for custom modifications
4. **Creating custom templates** based on existing designs

## Editing Existing Presentations

### Manual Editing Workflow

1. **Unpack the presentation:**
   ```python
   unpack_pptx("template.pptx", output_dir="template_edit")
   ```

2. **Manually edit files:**
   - Edit slide XML files in `template_edit/ppt/slides/`
   - Replace images in `template_edit/ppt/media/`
   - Modify layouts in `template_edit/ppt/slideLayouts/`

3. **Repack the presentation:**
   ```python
   pack_pptx("template_edit", output_file="template_modified.pptx")
   ```

### Cleaning Orphaned Files

After manual editing, clean up orphaned files:
```python
clean_pptx("template_edit")
```

This removes files that are no longer referenced in the presentation XML.

## Creating Presentations from Templates

### Extract and Modify Template

```python
# Step 1: Unpack template
unpack_pptx("company_template.pptx", output_dir="my_presentation")

# Step 2: Generate custom images
generate_image(
    prompt="Custom chart for Q1 sales data, company branding colors",
    size="1024x768"
)

# Step 3: Replace template images
# Manually replace images in my_presentation/ppt/media/

# Step 4: Update slide content
# Edit XML files in my_presentation/ppt/slides/

# Step 5: Repack presentation
pack_pptx("my_presentation", output_file="q1_report.pptx")
```

## Complete Example: Update Existing Presentation

```python
# 1. Read existing presentation
presentation_info = read_pptx("quarterly_report.pptx", extract_text=True)
print(f"Presentation has {presentation_info['slide_count']} slides")

# 2. Unpack for editing
unpack_pptx("quarterly_report.pptx", output_dir="report_edit")

# 3. Generate new images for updated slides
generate_image(
    prompt="Updated Q4 sales chart with latest data, professional design",
    size="1024x768"
)

generate_image(
    prompt="Market share infographic with current percentages",
    size="1280x720"
)

# 4. (Manual step) Replace images in report_edit/ppt/media/

# 5. (Manual step) Update text in slide XML files

# 6. Clean up orphaned files
clean_pptx("report_edit")

# 7. Repack updated presentation
pack_pptx("report_edit", output_file="quarterly_report_updated.pptx")
```

## Best Practices for Working with Existing Presentations

1. **Always backup** original files before unpacking
2. **Use version control** for unpacked directories if making significant changes
3. **Test repacking** with small changes first to ensure workflow works
4. **Clean up** after editing to remove orphaned files
5. **Document changes** made to template files for future reference

## Common Use Cases

### 1. Template Customization
- Start with company template
- Replace placeholder images with generated content
- Update text while preserving layout

### 2. Content Updates
- Update data visualizations in existing reports
- Replace outdated images with current ones
- Refresh text content while keeping design

### 3. Style Migration
- Extract color schemes from existing presentations
- Apply consistent styling to new presentations
- Reuse layout patterns across different content

### 4. Learning and Analysis
- Study successful presentation structures
- Extract best practices from well-designed slides
- Understand how complex layouts are implemented

---

**Navigation**: [Back to Main Documentation](../SKILL.md) | [Previous: Mandatory Workflow](mandatory-workflow.md)
