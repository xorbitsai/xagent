from pathlib import Path

from xagent.core.agent.context.skill_tool import _index_text
from xagent.skills.parser import SkillParser


def test_static_visual_design_skill_routes_only_commercial_creatives() -> None:
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
    description = " ".join(skill["description"].split())
    when_to_use = " ".join(skill["when_to_use"].split())
    assert "complete PNG or JPEG assets" in description
    assert "commercial and brand-facing" in description
    assert "campaign posters" in description
    assert "advertising creatives" in description
    assert "placement variants" in description
    assert "marketing, campaign, event, or brand communication" in when_to_use
    assert "educational infographics" in when_to_use
    assert "technical diagrams" in when_to_use
    assert "concept explainers" in when_to_use

    # Auto routing sees bounded one-line versions of these fields. Keep the
    # positive commercial scope and the important exclusions inside that
    # actual routing surface instead of only in the full skill body.
    routing_description = _index_text(skill["description"])
    routing_when_to_use = _index_text(skill["when_to_use"])
    assert "commercial and brand-facing" in routing_description
    assert "advertising creatives" in routing_description
    assert "Use only for marketing" in routing_when_to_use
    assert "educational infographics" in routing_when_to_use
    assert "concept explainers" in routing_when_to_use

    content = " ".join(skill["content"].split())
    assert "Stay within the commercial-creative scope" in content
    assert "Use `generate_image` to create the full designed asset" in content
    assert "references/static-ad-art-direction.md" in content
    assert "two or three genuinely different communication angles" in content
    assert "one finished placement on one continuous canvas" in content
    assert "a brand-specific final requires a verified logo" in content
    assert "This runtime does not provide deterministic compositing" in content
    assert "download_web_asset" not in content
    assert "SVG is source text" not in content
    assert "Do not use HTML/CSS plus browser screenshots" in content
    assert "Do not enter `final_answer`" in content
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
    assert "Design for a three-pass read" in content
    assert "Follow the main skill's one-canvas generation contract" in content
    assert "Automatic rejection overrides subjective scoring" in content
