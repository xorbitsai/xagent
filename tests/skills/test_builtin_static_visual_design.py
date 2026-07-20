from pathlib import Path

from src.xagent.skills.parser import SkillParser


def test_static_visual_design_skill_routes_designed_graphics_and_brand_assets() -> None:
    skill_dir = (
        Path(__file__).parents[2]
        / "src"
        / "xagent"
        / "skills"
        / "builtin"
        / "static-visual-design"
    )

    skill = SkillParser.parse(skill_dir)

    assert skill["name"] == "static-visual-design"
    assert "complete PNG or JPEG assets" in skill["description"]
    assert "posters" in skill["description"]
    assert "advertising creatives" in skill["description"]
    assert "placement variants" in skill["description"]

    content = " ".join(skill["content"].split())
    assert "Use `generate_image` to create the full designed asset" in content
    assert "concepts materially different" in content
    assert "two or three genuinely different communication angles" in content
    assert "Do not interpret the singular nouns" in content
    assert "render two or three candidates" in content
    assert "one visual device, and one structural approach" in content
    assert "headline and image divide the communication work" in content
    assert "generic business portraits holding phones" in content
    assert "prior winning creative, and performance evidence" in content
    assert "a brand-specific final requires a verified logo" in content
    assert "Include it as a generation reference" in content
    assert "download_web_asset" in content
    assert "Cosmetic resizes are not distinct concepts" in content
    assert "never invent prices, milestones, performance" in content
    assert "Never add a second logo over a generated pseudo-logo" in content
    assert "Do not make deterministic compositing an automatic final step" in content
    assert "Do not use HTML/CSS plus browser screenshots" in content
    assert "Do not enter `final_answer`" in content
    assert "merely polished but generic" in content
    assert "Return only final PNG or JPEG files" in content
