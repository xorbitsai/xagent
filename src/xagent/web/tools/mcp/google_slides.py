import json
import logging
import os
import re
import uuid
from typing import Any, NamedTuple

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP

from .utils import resolve_id_from_url, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-slides-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-slides-mcp")

_PRESENTATION_URL_ID_PATTERN = re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)")


class _LayoutSpec(NamedTuple):
    """(title_placeholder, body_placeholder) types Slides creates for a
    predefined layout, plus the two policies that depend on that shape:
    `bulleted` (does body land in a real list-style BODY placeholder?) and
    `body_required` (must body be non-empty?). Kept as explicit fields
    instead of comparing `body_placeholder == "BODY"` at each call site, so
    a future layout can't silently pick up the wrong policy just because
    its body placeholder happens to be named "BODY"."""

    title_placeholder: str | None
    body_placeholder: str | None
    bulleted: bool
    body_required: bool


# Predefined layouts we know how to fill in. `None` means that slot doesn't
# exist on the layout at all.
#
# Deliberately limited to the layouts whose placeholder composition is well
# established (matching Google's official Apps Script PredefinedLayout docs
# and this file's pre-existing TITLE_AND_BODY mapping). Google's docs
# explicitly warn a predefined layout's placeholder set "may have been
# changed" and don't enumerate it for every layout, so less-common ones
# (SECTION_TITLE_AND_DESCRIPTION, ONE_COLUMN_TEXT, MAIN_POINT, BIG_NUMBER)
# are intentionally left out rather than guessed at — use
# google_slides_batch_update for those until verified against a live API.
_LAYOUT_PLACEHOLDERS: dict[str, _LayoutSpec] = {
    "TITLE": _LayoutSpec(
        "CENTERED_TITLE", "SUBTITLE", bulleted=False, body_required=False
    ),
    "TITLE_AND_BODY": _LayoutSpec("TITLE", "BODY", bulleted=True, body_required=True),
    "TITLE_ONLY": _LayoutSpec("TITLE", None, bulleted=False, body_required=False),
    "SECTION_HEADER": _LayoutSpec("TITLE", None, bulleted=False, body_required=False),
    "BLANK": _LayoutSpec(None, None, bulleted=False, body_required=False),
}

# Leading "•"/"-"/"*" marker (with or without a following space, so a marker
# glued to text like "•First" is still recognized) at the start of a line.
# Captures any leading whitespace so it survives the substitution — Slides'
# createParagraphBullets infers nesting level from leading tab characters
# in the inserted text, so stripping only the marker (not the indentation
# in front of it) keeps multi-level bullets intact.
_BULLET_PREFIX_PATTERN = re.compile(r"^([ \t]*)[•\-\*][ \t]*")


def _strip_bullet_prefixes(text: str) -> str:
    """Drop a leading "•"/"-"/"*" marker from each line, preserving any
    leading whitespace before it.

    The public docstrings tell callers not to type literal bullet glyphs,
    but callers may do so anyway; once we ask Slides to render real
    bulleted paragraphs (see createParagraphBullets below) those glyphs
    would double up with Slides' own bullet, so strip them first.
    """
    return "\n".join(
        _BULLET_PREFIX_PATTERN.sub(r"\1", line) for line in text.split("\n")
    )


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def get_slides_service() -> Any:
    token = os.environ.get("GOOGLE_ACCESS_TOKEN")
    refresh_token = os.environ.get("GOOGLE_REFRESH_TOKEN")
    client_id = os.environ.get("GOOGLE_CLIENT_ID")
    client_secret = os.environ.get("GOOGLE_CLIENT_SECRET")

    if not token:
        raise ValueError("GOOGLE_ACCESS_TOKEN environment variable is missing")

    creds_kwargs = {"token": token}
    if refresh_token and client_id and client_secret:
        creds_kwargs.update(
            {
                "refresh_token": refresh_token,
                "client_id": client_id,
                "client_secret": client_secret,
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )

    credentials = Credentials(**creds_kwargs)
    return build("slides", "v1", credentials=credentials)


def _resolve_presentation_id(presentation_id: str) -> str:
    """Accept either a bare presentation id or a full Google Slides URL."""
    return resolve_id_from_url(
        presentation_id, _PRESENTATION_URL_ID_PATTERN, "presentation_id"
    )


def _element_text(element: dict[str, Any]) -> str:
    text_elements = element.get("shape", {}).get("text", {}).get("textElements", [])
    return "".join(
        text_element.get("textRun", {}).get("content", "")
        for text_element in text_elements
    )


def _slide_summary(slide: dict[str, Any], index: int) -> dict[str, Any]:
    texts = [
        text
        for element in slide.get("pageElements", [])
        if (text := _element_text(element).strip())
    ]
    return {
        "slide_number": index + 1,
        "object_id": slide.get("objectId"),
        "text": texts,
    }


@mcp.tool()
def google_slides_get_presentation(presentation_id: str) -> str:
    """
    Read a Google Slides presentation by id or full URL.
    Returns the title and the text content of each slide.
    """
    try:
        pres_id = _resolve_presentation_id(presentation_id)
        service = get_slides_service()
        presentation = service.presentations().get(presentationId=pres_id).execute()

        slides = [
            _slide_summary(slide, index)
            for index, slide in enumerate(presentation.get("slides", []))
        ]
        return json.dumps(
            {
                "status": "success",
                "presentation_id": presentation.get("presentationId"),
                "title": presentation.get("title"),
                "slide_count": len(slides),
                "slides": slides,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error getting presentation: {e}")
        return _error(str(e))


@mcp.tool()
def google_slides_create_presentation(title: str) -> str:
    """
    Create a new, empty Google Slides presentation with the given title.
    Use google_slides_add_slide to add content slides afterwards.
    """
    try:
        service = get_slides_service()
        presentation = service.presentations().create(body={"title": title}).execute()
        pres_id = presentation.get("presentationId")

        return json.dumps(
            {
                "status": "success",
                "presentation_id": pres_id,
                "title": presentation.get("title"),
                "link": f"https://docs.google.com/presentation/d/{pres_id}/edit",
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error creating presentation: {e}")
        return _error(str(e))


@mcp.tool()
def google_slides_add_slide(
    presentation_id: str,
    title: str = "",
    body: str = "",
    layout: str = "TITLE_AND_BODY",
) -> str:
    """
    Append a slide with a title and body text to a Google Slides presentation.
    The body supports plain text; use newlines to separate bullet lines — each
    line is rendered as its own bulleted paragraph (don't type literal "•"/"-"
    markers, Slides adds the bullet glyph itself).

    layout picks the slide's predefined layout, which decides which of
    title/body actually have a placeholder to land in:
      - "TITLE_AND_BODY" (default): title + full bulleted body content.
        body is required here — this layout rejects an empty body rather
        than silently creating a title-only slide.
      - "TITLE": a cover/section-opening slide — title is the big centered
        title, body (optional) becomes the subtitle line (not bulleted).
      - "TITLE_ONLY", "SECTION_HEADER": title only, no body placeholder —
        pass body="" or the call is rejected.
      - "BLANK": no placeholders at all; use google_slides_batch_update to
        add free-form text boxes/images instead.

    Every layout with a title placeholder also needs at least a non-empty
    title (or, for "TITLE", a non-empty body/subtitle instead) — a call
    that would otherwise produce a completely empty slide is rejected.
    """
    try:
        if layout not in _LAYOUT_PLACEHOLDERS:
            return _error(
                f"Unknown layout '{layout}'. Supported layouts: "
                f"{', '.join(sorted(_LAYOUT_PLACEHOLDERS))}"
            )

        title_placeholder, body_placeholder, is_bulleted, body_required = (
            _LAYOUT_PLACEHOLDERS[layout]
        )
        if is_bulleted and body:
            body = _strip_bullet_prefixes(body)

        if title and title_placeholder is None:
            return _error(
                f"layout '{layout}' has no title placeholder, so "
                "'title' would be silently dropped. Use a different "
                "layout, or google_slides_batch_update for a custom "
                "text box."
            )
        if body and body_placeholder is None:
            return _error(
                f"layout '{layout}' has no body placeholder, so "
                "'body' would be silently dropped. Use a layout "
                "with a body/subtitle placeholder (e.g. "
                "TITLE_AND_BODY, TITLE) or omit body."
            )
        if body_required and not body.strip():
            return _error(
                f"layout '{layout}' expects body content but none "
                "was provided. Include this slide's full bullet/"
                "detail text in 'body' — don't create the slide "
                "with just a title."
            )
        if (
            title_placeholder is not None
            and not body_required
            and not title.strip()
            and not body.strip()
        ):
            return _error(
                f"layout '{layout}' needs at least a non-empty 'title'"
                + (" or 'body'" if body_placeholder is not None else "")
                + " — as given, this call would create a completely "
                "empty slide."
            )

        pres_id = _resolve_presentation_id(presentation_id)
        service = get_slides_service()

        slide_id = f"slide_{uuid.uuid4().hex[:12]}"
        title_id = f"{slide_id}_title"
        body_id = f"{slide_id}_body"

        placeholder_mappings: list[dict[str, Any]] = []
        if title_placeholder is not None:
            placeholder_mappings.append(
                {
                    "layoutPlaceholder": {"type": title_placeholder},
                    "objectId": title_id,
                }
            )
        if body_placeholder is not None:
            placeholder_mappings.append(
                {
                    "layoutPlaceholder": {"type": body_placeholder},
                    "objectId": body_id,
                }
            )

        create_slide: dict[str, Any] = {
            "objectId": slide_id,
            "slideLayoutReference": {"predefinedLayout": layout},
        }
        if placeholder_mappings:
            create_slide["placeholderIdMappings"] = placeholder_mappings

        requests: list[dict[str, Any]] = [{"createSlide": create_slide}]
        if title.strip():
            requests.append({"insertText": {"objectId": title_id, "text": title}})
        if body:
            requests.append({"insertText": {"objectId": body_id, "text": body}})
            if is_bulleted:
                requests.append(
                    {
                        "createParagraphBullets": {
                            "objectId": body_id,
                            "textRange": {"type": "ALL"},
                            "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                        }
                    }
                )

        service.presentations().batchUpdate(
            presentationId=pres_id, body={"requests": requests}
        ).execute()

        return json.dumps(
            {
                "status": "success",
                "presentation_id": pres_id,
                "slide_id": slide_id,
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error adding slide: {e}")
        return _error(str(e))


@mcp.tool()
def google_slides_batch_update(presentation_id: str, requests_json: str) -> str:
    """
    Advanced: apply raw Google Slides API batchUpdate requests to a presentation.
    requests_json must be a JSON array of request objects following the Slides API
    schema (e.g. createShape, insertText, updateTextStyle, createImage).
    Use this only when the simpler tools cannot express the required change.
    """
    try:
        pres_id = _resolve_presentation_id(presentation_id)
        requests = json.loads(requests_json)
        if not isinstance(requests, list):
            raise ValueError("requests_json must be a JSON array of request objects")

        service = get_slides_service()
        result = (
            service.presentations()
            .batchUpdate(presentationId=pres_id, body={"requests": requests})
            .execute()
        )

        return json.dumps(
            {
                "status": "success",
                "presentation_id": result.get("presentationId", pres_id),
                "replies": result.get("replies", []),
            },
            ensure_ascii=False,
        )
    except Exception as e:
        logger.error(f"Error applying batch update: {e}")
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
