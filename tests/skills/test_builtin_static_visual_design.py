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

    # Brand provenance: both sanctioned sources named, every substitute closed.
    assert "a brand-specific final requires a verified logo" in content
    assert "recoverable with `list_all_user_files`" in content
    assert "current task workspace" in content
    assert "other tasks and earlier outputs are never a source" in content
    assert "a logo you reconstruct is a logo you invented" in content
    assert "label it a concept draft" in content
    # Stable brand cues are not one campaign's effects (#1411 review C7).
    assert "Separate stable identity cues from temporary campaign styling" in content
    # Searching belongs to the user, and the run stops rather than guessing.
    assert "stop before rendering finals" in content
    assert "take it only when they tell you to" in content
    assert "`ask_user_question`" in content
    # No wording may reintroduce unprompted retrieval as a sanctioned route.
    assert "download_web_asset" not in content
    assert "official web presence" not in content
    assert "fetch_web_content" not in content
    assert "web tools available to you" not in content

    # Planning order has to survive in SKILL.md: LLMPlanGenerator sees the skill
    # body but not the reference, which is only read later (#1411 review C1).
    assert "represent that order in any execution plan" in content
    assert "acquisition is a shared prerequisite" in content
    assert "Never plan a render, or an identity search, to run alongside" in content

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
    # A supplied logo the render dropped is a rejection (#1411 review C4).
    assert "omits a required brand asset is a rejection" in content
    assert "does not prove the image model placed it" in content
    # The one case where finishing with nothing rendered is correct.
    assert "ends in the question above, with nothing rendered" in content
    assert "do not list or describe files that do not exist" in content
    # A spent budget hands the asset back with its defect named, not silently,
    # and the hand-back is terminal rather than a replan trigger (C3).
    assert "name the defect concretely" in content
    assert "decide whether to spend another round" in content
    assert "spent budget with complete coverage is terminal" in content
    assert "`missing_verification` stays empty" in content
    assert "`outcome=partial`" in content

    # The file list is conditional on files existing, never unconditional.
    finish = content.split("## Finish", 1)[1]
    assert "Otherwise lead with the files" in finish
    assert finish.index(
        "do not list or describe files that do not exist"
    ) < finish.index("Otherwise lead with the files")


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


def test_web_asset_tool_descriptions_do_not_invite_unprompted_logo_search() -> None:
    """Tool descriptions are always in context; the skill policy is not.

    A description that tells the model to go find a brand's logo overrides the
    skill's ask-the-user rule, because the skill is only present once loaded
    (#1411 review C2).
    """
    from xagent.core.tools.adapters.vibe.download_web_asset import DownloadWebAssetArgs
    from xagent.core.tools.adapters.vibe.fetch_web_content import (
        FetchWebContentArgs,
        FetchWebContentTool,
    )

    def field_text(model: type, name: str) -> str:
        return model.model_fields[name].description or ""

    surfaces = [
        FetchWebContentTool().description,
        field_text(FetchWebContentArgs, "include_assets"),
        field_text(DownloadWebAssetArgs, "url"),
    ]
    for text in surfaces:
        lowered = text.lower()
        assert "when looking for logos" not in lowered
        assert "usually asset_query='logo'" not in lowered
        assert "prefer the official brand domain" not in lowered
