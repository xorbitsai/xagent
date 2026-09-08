import json
import logging
import os
import re
import uuid
from typing import Annotated, Any, Literal

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore[import-not-found]
from mcp.server.fastmcp import FastMCP
from pydantic import BeforeValidator

from .utils import resolve_id_from_url, setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("google-slides-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("google-slides-mcp")

_PRESENTATION_URL_ID_PATTERN = re.compile(r"/presentation/d/([a-zA-Z0-9_-]+)")


def _normalize_layout(value: object) -> object:
    """Case/whitespace-normalize `layout` before FastMCP's Pydantic layer
    validates it against the Literal below. Without this, an MCP tool
    call (the only way a real agent ever invokes this tool — direct
    Python calls, used by this file's own tests, bypass FastMCP
    validation entirely) with e.g. "title_and_body" would be rejected by
    Pydantic's case-sensitive Literal check *before* the function body's
    own normalization ever ran, making that normalization dead code for
    every real caller."""
    return value.strip().upper() if isinstance(value, str) else value


_Layout = Annotated[
    Literal["TITLE", "TITLE_AND_BODY", "TITLE_ONLY", "SECTION_HEADER", "BLANK"],
    BeforeValidator(_normalize_layout),
]

# Predefined layouts we know how to fill in, mapped to the (title_placeholder,
# body_placeholder) types Slides creates for each one. `None` means that slot
# doesn't exist on the layout at all. This table only records that factual
# placeholder shape; `is_bulleted`/`body_required` policy is derived from it
# once in google_slides_add_slide (both happen to be True only for
# TITLE_AND_BODY today, i.e. exactly when body_placeholder == "BODY").
#
# Deliberately limited to the layouts whose placeholder composition is well
# established (matching Google's official Apps Script PredefinedLayout docs
# and this file's pre-existing TITLE_AND_BODY mapping). Google's docs
# explicitly warn a predefined layout's placeholder set "may have been
# changed" and don't enumerate it for every layout, so less-common ones
# (SECTION_TITLE_AND_DESCRIPTION, ONE_COLUMN_TEXT, MAIN_POINT, BIG_NUMBER)
# are intentionally left out rather than guessed at — use
# google_slides_batch_update for those until verified against a live API.
_LAYOUT_PLACEHOLDERS: dict[str, tuple[str | None, str | None]] = {
    "TITLE": ("CENTERED_TITLE", "SUBTITLE"),
    "TITLE_AND_BODY": ("TITLE", "BODY"),
    "TITLE_ONLY": ("TITLE", None),
    "SECTION_HEADER": ("TITLE", None),
    "BLANK": (None, None),
}

# Leading "•"/"-"/"*" marker at the start of a line, with or without a
# following space (so a marker glued to a word, e.g. "•First", or with
# nothing after it at all, e.g. a bare "•" on its own line, is still
# recognized). Captures any leading Unicode whitespace so it can be
# converted to nesting-level tabs by _leading_whitespace_to_tabs below.
#
# "-" and "*" also have common non-bullet meanings, so the "glued, no
# space" case is deliberately narrower for them than for "•" (which has no
# other meaning in plain text):
#   - "-" is only glued-recognized when followed directly by a letter
#     ("-Nospace"); anything else — a digit, another "-", a currency
#     symbol, a decimal point — is left alone, so a negative number's
#     sign ("-5%", "-$5m", "-.5%") or an em-dash/en-dash convention
#     ("-- Author", "-– Author") is never silently eaten.
#   - "*" is only treated as a marker when followed by real whitespace
#     ("* item") or nothing; a bare glued "*" ("*emphasis* text") is left
#     alone so markdown-style emphasis isn't corrupted.
#
# The "glued to a letter" lookahead is Unicode-aware (Python's default
# \w), so e.g. "-中文条目" is recognized and stripped the same way
# "-Nospace" is — intentional, not an ASCII-only oversight.
#
# Only "•"/"-"/"*" are recognized. Other list conventions (numbered lists
# like "1.", or other dash/arrow glyphs like "–"/"—"/"→") are left as
# literal text and will visually double up with Slides' own bullet glyph
# once createParagraphBullets is applied.
_BULLET_PREFIX_PATTERN = re.compile(
    r"^(\s*)(?:"
    r"•(?:[ \t]+|(?=\S)|$)"
    r"|-(?:[ \t]+|(?=[^\W\d_])|$)"
    r"|\*(?:[ \t]+|$)"
    r")"
)


def _leading_whitespace_to_tabs(leading: str) -> str:
    """Convert captured leading whitespace into the literal tab
    characters Slides' createParagraphBullets actually uses to infer
    nesting level (plain spaces are otherwise inserted as literal,
    non-nesting text). Each literal tab in the input counts as one level;
    otherwise every 2 spaces — the typical hand-typed nested-bullet
    convention — counts as one level, with any non-empty indentation
    counting as at least one level. Mixing tabs and spaces in the same
    run is not combined: if any tab is present, only the tabs are
    counted and any spaces in that same leading run are ignored."""
    if not leading:
        return ""
    level = leading.count("\t") or max(1, len(leading) // 2)
    return "\t" * level


def _strip_bullet_prefixes(text: str) -> str:
    """Drop a leading "•"/"-"/"*" marker from each line — repeatedly, so a
    line with more than one leading marker (e.g. "• - nested") is fully
    unwrapped rather than leaving a residual marker as literal text — and
    convert any leading indentation into nesting-level tabs. See
    _BULLET_PREFIX_PATTERN for the narrower rules that keep this from
    corrupting non-bullet content.

    The public docstrings tell callers not to type literal bullet glyphs,
    but callers may do so anyway; once we ask Slides to render real
    bulleted paragraphs (see createParagraphBullets below) those glyphs
    would double up with Slides' own bullet, so strip them first.
    """

    def _strip_line(line: str) -> str:
        while True:
            match = _BULLET_PREFIX_PATTERN.match(line)
            if match is None:
                return line
            stripped = _leading_whitespace_to_tabs(match.group(1)) + line[match.end() :]
            if stripped == line:
                return line
            line = stripped

    return "\n".join(_strip_line(line) for line in text.split("\n"))


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
    layout: _Layout = "TITLE_AND_BODY",
) -> str:
    """
    Append a slide with a title and body text to a Google Slides presentation.
    The body supports plain text; use newlines to separate bullet lines — each
    line is rendered as its own bulleted paragraph (don't type literal "•"/"-"
    markers, Slides adds the bullet glyph itself — only "•"/"-"/"*" are
    recognized as markers to strip; other conventions like numbered lists
    ("1.") or other dash/arrow glyphs ("–"/"—"/"→") are inserted as
    literal text and will visually double up with Slides' own bullet).

    layout picks the slide's predefined layout, which decides which of
    title/body actually have a placeholder to land in:
      - "TITLE_AND_BODY" (default): title + full bulleted body content.
        body is required here — this layout rejects an empty body rather
        than silently creating a title-only slide. An empty title is
        still allowed as long as body has content (e.g. a title-less
        continuation slide) — this is intentional, not a gap.
      - "TITLE": a cover/section-opening slide — title is the big centered
        title, body (optional) becomes the subtitle line (not bulleted).
      - "TITLE_ONLY", "SECTION_HEADER": title only, no body placeholder —
        pass body="" or the call is rejected.
      - "BLANK": no placeholders at all; use google_slides_batch_update to
        add free-form text boxes/images instead.

    Layouts with a title placeholder that don't already require a body
    (TITLE, TITLE_ONLY, SECTION_HEADER — not TITLE_AND_BODY, whose own
    body-required rule above already covers it, and not BLANK, which has
    no title placeholder) still need at least a non-empty title — or, for
    "TITLE", a non-empty body/subtitle instead — since otherwise the call
    would produce a completely empty slide; that combination is rejected.

    This assumes the presentation uses a standard/default theme. A custom
    theme whose master doesn't expose the placeholder types listed above
    for a given layout will fail with a raw Slides API error from the
    underlying batchUpdate call rather than a friendly one — use
    google_slides_batch_update directly for presentations with a
    non-standard theme.
    """
    try:
        if not isinstance(layout, str):
            return _error(f"'layout' must be a string, got {type(layout).__name__}.")

        title = title.replace("\r\n", "\n").replace("\r", "\n")
        body = body.replace("\r\n", "\n").replace("\r", "\n")
        # A title is expected to be a single line; collapse any embedded
        # newline (and surrounding whitespace) into a space rather than
        # silently producing a multi-paragraph title placeholder.
        title = re.sub(r"\s*\n\s*", " ", title)
        normalized_layout = layout.strip().upper()

        if normalized_layout not in _LAYOUT_PLACEHOLDERS:
            return _error(
                f"Unknown layout '{layout}'. Supported layouts: "
                f"{', '.join(sorted(_LAYOUT_PLACEHOLDERS))}"
            )

        title_placeholder, body_placeholder = _LAYOUT_PLACEHOLDERS[normalized_layout]
        is_bulleted = body_required = body_placeholder == "BODY"
        if is_bulleted and body:
            body = _strip_bullet_prefixes(body)
        if body:
            # Drop blank lines (a bare marker stripped down to nothing,
            # blank lines already in the input, or a trailing newline) so
            # no empty paragraph survives — for bulleted content this
            # avoids createParagraphBullets putting a bullet glyph on an
            # empty paragraph (a visibly floating bullet point); for
            # non-bulleted content (e.g. TITLE's subtitle) it avoids
            # stray blank lines in the placeholder.
            body = "\n".join(line for line in body.split("\n") if line.strip())

        if title.strip() and title_placeholder is None:
            return _error(
                f"layout '{normalized_layout}' has no title placeholder, "
                "so 'title' would be silently dropped. Use a different "
                "layout, or google_slides_batch_update for a custom "
                "text box."
            )
        if body.strip() and body_placeholder is None:
            return _error(
                f"layout '{normalized_layout}' has no body placeholder, "
                "so 'body' would be silently dropped. Use a layout "
                "with a body/subtitle placeholder (e.g. "
                "TITLE_AND_BODY, TITLE) or omit body."
            )
        if body_required and not body.strip():
            return _error(
                f"layout '{normalized_layout}' expects body content but "
                "none was provided. Include this slide's full bullet/"
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
                f"layout '{normalized_layout}' needs at least a non-empty "
                "'title'"
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
            "slideLayoutReference": {"predefinedLayout": normalized_layout},
        }
        if placeholder_mappings:
            create_slide["placeholderIdMappings"] = placeholder_mappings

        requests: list[dict[str, Any]] = [{"createSlide": create_slide}]
        if title.strip():
            requests.append(
                {"insertText": {"objectId": title_id, "text": title.strip()}}
            )
        if body.strip():
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
                "layout": normalized_layout,
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
