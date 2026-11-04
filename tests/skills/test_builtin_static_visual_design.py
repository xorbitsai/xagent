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
    assert "one PNG/JPEG" in description
    assert "brand styling" in description
    assert "finished ad, poster, banner, social post" in description
    assert "another placement size or aspect ratio" in description
    assert "marketing, promotion, campaign, event, or brand-facing image" in when_to_use
    assert "explanatory diagrams" in when_to_use
    assert "infographics" in when_to_use
    assert "plain illustrations" in when_to_use

    # Auto routing sees bounded one-line versions of these fields, so every
    # assertion above has to hold on the routing surface, not only in the body.
    assert _index_text(skill["description"]) == description
    assert _index_text(skill["when_to_use"]) == when_to_use

    content = " ".join(skill["content"].split())
    assert "Stay within the commercial-creative scope" in content
    assert "Use `generate_image` to create the full designed asset" in content
    assert "references/static-ad-art-direction.md" in content
    assert "two or three genuinely different communication angles" in content
    assert "one finished placement on one continuous canvas" in content
    assert "a brand-specific final requires a verified logo" in content
    assert "No image tool composites deterministically" in content
    # #942 removed logo_overlay as brittle; hand-rolled compositing is the same work.
    assert "composite it yourself" not in content
    assert "download_web_asset" not in content
    assert "SVG is source text" not in content
    assert "Do not use HTML/CSS plus browser screenshots" in content
    assert "Do not enter `final_answer`" in content
    assert "Return only final PNG or JPEG files" in content
    assert "The user's other tasks and earlier outputs are not a source" in content
    assert "recover by finding it with `list_all_user_files`" in content
    # One anchor per behavior an earlier round had to fix and a later rewrite lost.
    # Coverage means absent, not wrong — otherwise any defect can be relabeled
    # into an unbudgeted render.
    assert "producing the first candidate for a required asset that has none" in content
    assert "Only one failure is a coverage failure" in content
    assert "do not reclassify a defect as missing coverage" in content
    # Repairs are what the budget counts, and three relabeling routes stay closed.
    assert "any `generate_image` or `edit_image` call on an asset that" in content
    assert "any call on an optional asset, including its first" in content
    assert "even at another required placement" in content
    # The gate's coverage half never yields to the budget.
    assert "**Coverage is unconditional.**" in content
    # Two rejection reasons a rewrite dropped, and the brand rule they serve.
    assert "a render that omits a required brand asset" in content
    assert "labeled a concept draft" in content
    # The hand-back has to pass the completion check that precedes it.
    assert "not in the check's `missing_verification` field" in content
    assert "let the user decide whether to spend another round" in content
    assert "Reject them while the repair budget lasts" in content
    assert "While the budget lasts, regenerate from the locked design" in content
    assert "while the repair budget lasts, remove the artifact or" in content
    assert "iterations is not. While the budget lasts," in content
    assert "at most two on the same asset, and four across the run" in content
    assert "a variant, a fresh angle, or a retry" in content
    assert "When the run budget is spent, stop repairing altogether" in content
    assert "Where the two definitions both fit one call, coverage wins" in content
    assert "sketches are sentences, not renders" in content
    # The hand-back is a workaround around the gate, not a supported path.
    assert "This is a workaround, not a supported" in content
    assert "may be replanned and the task may still fail" in content


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
    # Mandatory reading that lands in the same context as the skill, so its
    # rejection rules need the budget too.
    assert "within the limits the main skill sets on repairs" in content
    # Every rejection here is a quality failure by the skill's definitions, so the
    # budget governs all of them; only an absent asset is coverage.
    assert "Every rejection in this document is a quality failure" in content
    assert "subordinate to that skill's repair budget" in content
    assert "Only a required asset with no candidate at all is coverage" in content
    assert "While the repair budget lasts, regenerate from the locked design" in content
    # `generate ... sketches` read as a render-driving instruction with no budget.
    assert "one-sentence sketches for a set of three, none of them rendered" in content
