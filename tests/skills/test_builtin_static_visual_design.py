from pathlib import Path

from xagent.core.agent.context.skill_tool import INDEX_ENTRY_MAX_CHARS, _index_text
from xagent.skills.parser import SkillParser

SKILL_DIR = (
    Path(__file__).parents[2]
    / "src"
    / "xagent"
    / "skills"
    / "builtin"
    / "static-visual-design"
)


def _skill() -> dict:
    return SkillParser.parse(SKILL_DIR)


def test_static_visual_design_index_entries_survive_truncation() -> None:
    """The index line is the only routing surface; a cut field loses triggers."""
    skill = _skill()

    for field in ("description", "when_to_use"):
        raw = " ".join(skill[field].split())
        assert raw, f"{field} must not be empty"
        assert len(raw) <= INDEX_ENTRY_MAX_CHARS, (
            f"{field} is {len(raw)} chars; the skill index truncates at "
            f"{INDEX_ENTRY_MAX_CHARS} and the tail never reaches the model"
        )
        assert raw == _index_text(skill[field])


def test_static_visual_design_skill_routes_only_commercial_creatives() -> None:
    skill = _skill()

    assert skill["name"] == "static-visual-design"
    description = " ".join(skill["description"].split())
    when_to_use = " ".join(skill["when_to_use"].split())

    # Positive triggers and exclusions both have to sit in the routing surface.
    assert "ad, poster, banner" in description
    assert "PNG/JPEG" in description
    assert "marketing, promotion, campaign, event, or brand-facing" in when_to_use
    assert "Not for explanatory diagrams" in when_to_use
    assert "infographics" in when_to_use

    content = " ".join(skill["content"].split())

    # Scope and medium.
    assert "belongs to the general image-generation workflow instead" in content
    assert (
        "Produce the finished visual with `generate_image` and `edit_image`" in content
    )
    assert "Reserve HTML/CSS plus browser screenshots" in content
    assert "references/static-ad-art-direction.md" in content
    assert "two or three directions that differ" in content
    assert "one finished placement on one continuous canvas" in content

    # #942 removed logo_overlay as brittle; hand-rolled compositing is the same work.
    assert "cannot survive generative rendering pixel-for-pixel" in content
    assert "composite it yourself" not in content
    assert "SVG is source text" not in content

    # Brand provenance: sanctioned sources, and every substitute path closed.
    assert "a brand-specific final requires a verified logo" in content
    assert "recoverable with `list_all_user_files`" in content
    assert "other tasks and earlier outputs are never a source" in content
    assert "a logo you reconstruct is a logo you invented" in content
    assert "label it a concept draft" in content
    # Searching belongs to the user, and the run stops rather than guessing.
    assert "stop before rendering finals" in content
    assert "take it only when they tell you to" in content
    assert "download_web_asset" not in content

    # Coverage means absent, not wrong — otherwise any defect can be relabeled
    # into an unbudgeted render.
    assert "producing the first candidate for a required asset that has none" in content
    assert "Coverage means the asset is absent, not that an existing" in content
    assert "Coverage is unconditional" in content
    # Repairs are what the budget counts, and the relabeling routes stay closed.
    assert "any render call on an asset that already has a candidate" in content
    assert "at most two per asset and four per run" in content
    assert "anything you would call a variant or retry" in content
    assert "Counters do not carry across plan steps" in content
    # The one case where finishing with nothing rendered is correct.
    assert "ends in the question above, with nothing rendered" in content
    assert "do not list or describe files that do not exist" in content
    # A spent budget hands the asset back with its defect named, not silently.
    assert "name the defect concretely" in content
    assert "decide whether to spend another round" in content


def test_static_visual_design_includes_art_direction_reference() -> None:
    reference_path = SKILL_DIR / "references" / "static-ad-art-direction.md"

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
    # Craft guidance demoted out of SKILL.md keeps its anchors here.
    assert "Build a set of directions" in content
    assert "creative-risk ladder" in content
    assert "differ on at least three of" in content
