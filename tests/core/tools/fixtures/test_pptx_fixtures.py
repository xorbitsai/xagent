"""
Test fixtures for PPTX tool tests.

Provides sample data and expected outputs for testing.
"""

# Sample theme configurations
VALID_THEME_CONFIG = {
    "colors": {
        "background": "#1E2761",
        "surface": "#F5F5F5",
        "primary": "#CADCFC",
        "secondary": "#E8E8D1",
        "accent": "#FFFFFF",
        "text": "#1E2761",
    },
    "typography": {
        "title_size": 44,
        "body_size": 18,
        "quote_size": 24,
        "title_weight": "bold",
        "body_weight": "normal",
    },
    "layout": {
        "title_bar": True,
        "content_padding": 0.5,
        "card_style": "flat",
        "rounded_corners": False,
    },
    "visual": {"background_style": "solid", "shadow": False, "rounded_corners": False},
}

INVALID_THEME_CONFIG_MISSING_COLOR = {
    "colors": {
        "background": "#1E2761",
        "primary": "#CADCFC",
        # Missing: surface, secondary, accent, text
    },
    "typography": {"title_size": 44, "body_size": 18},
    "layout": {"title_bar": True},
}

INVALID_THEME_CONFIG_BAD_HEX = {
    "colors": {
        "background": "not_a_hex_color",
        "primary": "#CADCFC",
        "surface": "#FFFFFF",
        "secondary": "#E8E8D1",
        "accent": "#FFFFFF",
        "text": "#1E2761",
    },
    "typography": {"title_size": 44, "body_size": 18},
    "layout": {"title_bar": True},
    "visual": {"background_style": "solid"},
}

INVALID_THEME_CONFIG_BAD_TYPO_VALUE = {
    "colors": {"background": "#1E2761", "text": "#1E2761"},
    "typography": {
        "title_size": "not_a_number",
        "body_size": 18,
        "title_weight": "bold",
        "body_weight": "normal",
    },
    "layout": {"title_bar": True},
    "visual": {"background_style": "solid"},
}

INVALID_THEME_CONFIG_BAD_WEIGHT = {
    "colors": {"background": "#1E2761", "text": "#1E2761"},
    "typography": {
        "title_size": 44,
        "body_size": 18,
        "title_weight": "invalid_font_weight",
        "body_weight": "normal",
    },
    "layout": {"title_bar": True},
    "visual": {"background_style": "solid"},
}


# Sample slide definitions
SAMPLE_SLIDES_MINIMAL = [
    {"type": "title", "title": "Test Presentation", "subtitle": "Minimal Theme Test"},
    {
        "type": "content",
        "title": "First Point",
        "bullets": ["Bullet one", "Bullet two", "Bullet three"],
    },
    {
        "type": "content",
        "title": "Second Point",
        "bullets": ["Bullet one", "Bullet two"],
    },
    {"type": "thank_you", "message": "Thank You"},
]

SAMPLE_SLIDES_WITH_TWO_COLUMN = [
    {"type": "title", "title": "Two Column Test"},
    {
        "type": "two_column",
        "title": "Comparison",
        "left": ["Feature A", "Feature B", "Feature C", "Feature D", "Feature E"],
        "right": ["Benefit 1", "Benefit 2", "Benefit 3"],
    },
]

SAMPLE_SLIDES_WITH_ALL_TYPES = [
    {"type": "title", "title": "All Types Test", "subtitle": "Comprehensive"},
    {
        "type": "content",
        "title": "Content Slide",
        "bullets": ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"],
    },
    {
        "type": "two_column",
        "title": "Two Columns",
        "left": ["Left 1", "Left 2", "Left 3"],
        "right": ["Right 1", "Right 2"],
    },
    {"type": "section_divider"},
    {"type": "quote", "text": "This is a sample quote slide."},
    {"type": "thank_you", "message": "Questions?"},
]

SAMPLE_SLIDES_WITHOUT_VISUAL = [
    {"type": "title", "title": "Text Only"},
    {
        "type": "content",
        "title": "Text Content",
        "bullets": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4", "Bullet 5"],
    },
]

# New enterprise slide type fixtures
SAMPLE_SLIDES_WITH_METRICS = [
    {"type": "title", "title": "Q3 Performance"},
    {
        "type": "metrics",
        "title": "Key Performance Indicators",
        "items": [
            {"label": "ARR", "value": "$2.1M"},
            {"label": "Growth", "value": "120%"},
            {"label": "Customers", "value": "86"},
        ],
    },
]

SAMPLE_SLIDES_WITH_TIMELINE = [
    {
        "type": "timeline",
        "title": "Product Roadmap",
        "milestones": [
            {"title": "Q1", "description": "Launch MVP"},
            {"title": "Q2", "description": "Expand to SEA"},
            {"title": "Q3", "description": "Enterprise deals"},
        ],
    }
]

SAMPLE_SLIDES_WITH_COMPARISON = [
    {
        "type": "comparison",
        "title": "Our Advantage",
        "left_title": "Traditional",
        "left_items": ["Manual", "Slow", "Expensive"],
        "right_title": "XAgent",
        "right_items": ["Automated", "Fast", "Scalable"],
    }
]

SAMPLE_SLIDES_WITH_STATEMENT = [
    {"type": "statement", "text": "AI Infrastructure Will Become the New Cloud"}
]

SAMPLE_SLIDES_WITH_IMAGE_HIGHLIGHT = [
    {
        "type": "image_highlight",
        "title": "Platform Overview",
        "caption": "Unified agent orchestration platform",
    }
]

SAMPLE_SLIDES_WITH_FLOW = [
    {
        "type": "flow",
        "title": "How It Works",
        "steps": [
            "User Input",
            "Agent Reasoning",
            "Tool Execution",
            "Structured Output",
        ],
    }
]

# All 12 slide types
SAMPLE_SLIDES_ALL_TWELVE_TYPES = [
    {"type": "title", "title": "All 12 Types"},
    {"type": "content", "title": "Content", "bullets": ["A", "B", "C"]},
    {
        "type": "two_column",
        "title": "Two Column",
        "left": ["L1", "L2"],
        "right": ["R1", "R2"],
    },
    {"type": "section_divider"},
    {"type": "quote", "text": "Quote text"},
    {"type": "thank_you", "message": "Thank You"},
    {"type": "metrics", "title": "KPIs", "items": [{"label": "A", "value": "100"}]},
    {
        "type": "timeline",
        "title": "Timeline",
        "milestones": [{"title": "Q1", "description": "Launch"}],
    },
    {
        "type": "comparison",
        "title": "Compare",
        "left_title": "A",
        "left_items": ["1"],
        "right_title": "B",
        "right_items": ["2"],
    },
    {"type": "statement", "text": "Big Statement"},
    {"type": "image_highlight", "title": "Image", "caption": "Caption"},
    {"type": "flow", "title": "Process", "steps": ["Step 1", "Step 2"]},
]
