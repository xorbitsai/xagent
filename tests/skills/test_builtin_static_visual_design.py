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
    assert "references/static-ad-art-direction.md" in content
    assert "concepts materially different" in content
    assert "two or three genuinely different communication angles" in content
    assert "Do not interpret the singular nouns" in content
    assert "render two or three candidates" in content
    assert "one visual device, and one structural approach" in content
    assert "one coherent planning pass" in content
    assert "do not delegate open-ended ideation" in content
    assert "creative-risk ladder" in content
    assert "Brand/reference acquisition is a shared prerequisite" in content
    assert "one finished placement on one continuous canvas" in content
    assert "contact sheets, moodboards, option grids" in content
    assert "Automatically reject contact sheets" in content
    assert "Use `edit_image` only" in content
    assert "Brand-safe evolution" in content
    assert "headline and image divide the communication work" in content
    assert "generic business portraits holding phones" in content
    assert "prior winning creative, and performance evidence" in content
    assert "a brand-specific final requires a verified logo" in content
    assert "Include it as a generation reference" in content
    assert "Stable cues may include the official logo" in content
    assert "datedness is not a brand requirement" in content
    assert "download_web_asset" in content
    assert "Cosmetic resizes are not distinct concepts" in content
    assert "Each direction must differ on at least three" in content
    assert "stacked display effects as a warning sign" in content
    assert "never invent prices, milestones, performance" in content
    assert "Never add a second logo over a generated pseudo-logo" in content
    assert "Do not make deterministic compositing an automatic final step" in content
    assert "Do not use HTML/CSS plus browser screenshots" in content
    assert "Do not enter `final_answer`" in content
    assert "merely polished but generic" in content
    assert "Return only final PNG or JPEG files" in content


def test_static_visual_design_includes_art_direction_reference() -> None:
    reference_path = (
        Path(__file__).parents[2]
        / "src"
        / "xagent"
        / "skills"
        / "builtin"
        / "static-visual-design"
        / "references"
        / "static-ad-art-direction.md"
    )

    content = " ".join(reference_path.read_text().split())

    assert "Choose a communication structure" in content
    assert "Dominant proof" in content
    assert "Offer reveal" in content
    assert "Editorial provocation" in content
    assert "Design for a three-pass read" in content
    assert "Use a one-canvas generation contract" in content
    assert "Apply the substitution test" in content
    assert "Automatic rejection overrides subjective scoring" in content
    assert "Regenerate from the locked design specification" in content
