"""Redact personal data from web search queries.

The model writes search queries from conversation content, so a query can carry
an email address, phone number, street address or government identifier. This
module replaces those with typed placeholders before the query leaves the
process, so the query still reads as a search (``[EMAIL]``, ``[PHONE]``,
``[ADDRESS]``, ``[TFN]``, ``[MEDICARE]``, ``[ABN]``).

Design constraints:

* Pure and cheap: regular expressions only, no model, no network, no new
  dependency. Every pattern is bounded (no nested open-ended quantifiers) so a
  hostile query cannot trigger catastrophic backtracking.
* Names of people and organisations are deliberately kept: searching for
  public figures and businesses is normal use.
* TFN, Medicare and ABN candidates are replaced only when their checksum
  passes, so years, prices, order numbers and product codes survive.
* :func:`redact_search_query` never raises. If redaction itself fails, it fails
  closed: the whole query is replaced by a fixed placeholder instead of being
  passed through unredacted.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, List, Match, NamedTuple, Tuple

logger = logging.getLogger(__name__)

EMAIL = "email"
ADDRESS = "address"
ABN = "abn"
TFN = "tfn"
MEDICARE = "medicare"
PHONE = "phone"
REDACTION_ERROR = "redaction_error"

_PLACEHOLDERS = {
    EMAIL: "[EMAIL]",
    ADDRESS: "[ADDRESS]",
    ABN: "[ABN]",
    TFN: "[TFN]",
    MEDICARE: "[MEDICARE]",
    PHONE: "[PHONE]",
}
_FAILED_PLACEHOLDER = "[REDACTED]"


class RedactionResult(NamedTuple):
    """Outcome of :func:`redact_search_query`."""

    text: str
    # Entity types that were replaced, de-duplicated, in the order the
    # recognisers ran. Values are the module constants (``"email"``, ...).
    entities: Tuple[str, ...]


# --- checksums -------------------------------------------------------------


def _digits(value: str) -> List[int]:
    return [int(c) for c in value if c.isdigit()]


def valid_abn(value: str) -> bool:
    """ABN check: subtract 1 from the first digit, weight, sum, mod 89 == 0.

    Algorithm as published by the Australian Business Register:
    https://abr.business.gov.au/Help/AbnFormat
    (weights 10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19).
    """
    d = _digits(value)
    if len(d) != 11:
        return False
    d[0] -= 1
    weights = (10, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19)
    return sum(w * x for w, x in zip(weights, d)) % 89 == 0


def valid_tfn(value: str) -> bool:
    """9-digit TFN check: weights 1,4,3,7,5,8,6,9,10, sum mod 11 == 0.

    The ATO does not publish the TFN algorithm itself; the weights and modulus
    are the widely implemented ones, e.g. python-stdnum's ``stdnum.au.tfn``:
    https://arthurdejong.org/git/python-stdnum/tree/stdnum/au/tfn.py
    Older 8-digit TFNs are not recognised (no authoritative weights found).
    """
    d = _digits(value)
    if len(d) != 9:
        return False
    weights = (1, 4, 3, 7, 5, 8, 6, 9, 10)
    return sum(w * x for w, x in zip(weights, d)) % 11 == 0


def valid_medicare(value: str) -> bool:
    """Medicare card number check (10 digits, issue number is the last).

    The first digit is 2-6; the first 8 digits are weighted 1,3,7,9,1,3,7,9,
    and the sum mod 10 must equal the 9th digit (the check digit); the 10th
    digit is the card issue number and is never 0. Algorithm as implemented by
    the ``starling`` package, which cites Services Australia:
    https://rdrr.io/cran/starling/src/R/check_medicare.R
    and described at https://doccy.com.au/medicare-number-validation
    """
    d = _digits(value)
    if len(d) != 10 or d[0] not in range(2, 7) or d[9] == 0:
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9)
    return sum(w * x for w, x in zip(weights, d[:8])) % 10 == d[8]


# --- patterns --------------------------------------------------------------

# A numeric identifier must stand alone: not glued to letters/digits, a hyphen
# ("ORD-123456782"), a currency sign, or a decimal/thousands separator.
_NUM_START = r"(?<![\w\-./$€£,])"
_NUM_END = r"(?![\w\-]|[.,]\d)"

_EMAIL_RE = re.compile(
    r"[A-Za-z0-9._%+\-]{1,64}@[A-Za-z0-9\-]{1,63}(?:\.[A-Za-z0-9\-]{1,63}){1,8}"
)

_ABN_RE = re.compile(_NUM_START + r"\d{2}[ -]?\d{3}[ -]?\d{3}[ -]?\d{3}" + _NUM_END)
_TFN_RE = re.compile(_NUM_START + r"\d{3}[ -]?\d{3}[ -]?\d{3}" + _NUM_END)
_MEDICARE_RE = re.compile(_NUM_START + r"[2-6]\d{3}[ -]?\d{5}[ -]?\d" + _NUM_END)

_STREET_TYPES = (
    r"Street|St|Road|Rd|Avenue|Ave|Drive|Dr|Lane|Ln|Court|Ct|Place|Pl|Crescent|"
    r"Cres|Boulevard|Blvd|Terrace|Tce|Parade|Pde|Highway|Hwy|Way|Close|Circuit|"
    r"Cct|Esplanade|Esp|Grove|Square|Sq|Mews|Walk|Row"
)
_AU_STATES = r"NSW|VIC|QLD|SA|WA|TAS|NT|ACT"
# "12", "12A", "4/12", "Unit 4/12", "Flat 2, 12", "12-14"
_STREET_NO = (
    r"(?:(?:Unit|Apt|Apartment|Flat|Suite|Level|Lot)\s+\d{1,4}[A-Za-z]?[,/ ]\s*)?"
    r"\d{1,4}[A-Za-z]?(?:\s?[/-]\s?\d{1,4}[A-Za-z]?)?"
)
_SUBURB_TAIL = r"(?:,?\s+(?:[A-Za-z'\-]{2,20}\s+){0,3}(?:" + _AU_STATES + r")\s+\d{4})?"
# Capitalised street name: matches without further context ("12 Smith Street").
_ADDRESS_CAP_RE = re.compile(
    r"(?<![\w/])" + _STREET_NO + r"\s+(?:[A-Z][A-Za-z'’\-]{1,24}\s+){1,3}"
    r"(?:" + _STREET_TYPES + r")\b\.?" + _SUBURB_TAIL
)
# Any-case street name: only with a trailing "<suburb> <STATE> <postcode>", so
# "10 ways drive growth" and "5 best street food" are left alone.
_ADDRESS_CTX_RE = re.compile(
    r"(?<![\w/])" + _STREET_NO + r"\s+(?:[A-Za-z'’\-]{2,24}\s+){1,3}"
    r"(?:" + _STREET_TYPES + r")\b\.?,?\s+(?:[A-Za-z'\-]{2,20}\s+){0,3}"
    r"(?:" + _AU_STATES + r")\s+\d{4}\b",
    re.IGNORECASE,
)
_PO_BOX_RE = re.compile(r"\bP\.?\s?O\.?\s+Box\s+\d{1,6}\b", re.IGNORECASE)

# Phone numbers. International needs a leading "+"; Australian national forms
# are mobile 04xx xxx xxx and landline (0x) xxxx xxxx; NANP needs separators.
_PHONE_INTL_RE = re.compile(
    r"(?<![\w+])\+\d{1,3}[ .\-]?(?:\(\d{1,4}\)[ .\-]?)?\d{1,4}(?:[ .\-]?\d{2,5}){1,4}(?!\d)"
)
_PHONE_AU_RE = re.compile(
    r"(?<![\w\-./$+])(?:\(0[2378]\)|0[2378])[ \-]?\d{4}[ \-]?\d{4}(?!\d)"
    r"|(?<![\w\-./$+])04\d{2}[ \-]?\d{3}[ \-]?\d{3}(?!\d)"
)
_PHONE_NANP_RE = re.compile(
    r"(?<![\w\-./$+])(?:\(\d{3}\)[ ]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\w\-]|[.,]\d)"
)


def _intl_ok(m: Match[str]) -> bool:
    return 8 <= len(_digits(m.group(0))) <= 15


def _sub(
    text: str,
    pattern: "re.Pattern[str]",
    entity: str,
    found: List[str],
    accept: Callable[[Match[str]], bool] = lambda m: True,
) -> str:
    def repl(m: Match[str]) -> str:
        if not accept(m):
            return m.group(0)
        if entity not in found:
            found.append(entity)
        return _PLACEHOLDERS[entity]

    return pattern.sub(repl, text)


def redact_search_query(query: str) -> RedactionResult:
    """Return ``query`` with personal data replaced by typed placeholders.

    Never raises. On an internal failure the whole query is replaced by
    ``[REDACTED]`` (fail closed) and ``entities`` is ``("redaction_error",)``.
    """
    try:
        if not isinstance(query, str) or not query:
            return RedactionResult(query if isinstance(query, str) else "", ())
        found: List[str] = []
        text = query
        text = _sub(text, _EMAIL_RE, EMAIL, found)
        text = _sub(text, _ADDRESS_CAP_RE, ADDRESS, found)
        text = _sub(text, _ADDRESS_CTX_RE, ADDRESS, found)
        text = _sub(text, _PO_BOX_RE, ADDRESS, found)
        # "+" numbers first so their trailing digits are never read as a TFN.
        text = _sub(text, _PHONE_INTL_RE, PHONE, found, _intl_ok)
        # Longest identifier first so an 11-digit ABN is never read as a TFN.
        text = _sub(text, _ABN_RE, ABN, found, lambda m: valid_abn(m.group(0)))
        text = _sub(
            text, _MEDICARE_RE, MEDICARE, found, lambda m: valid_medicare(m.group(0))
        )
        text = _sub(text, _TFN_RE, TFN, found, lambda m: valid_tfn(m.group(0)))
        text = _sub(text, _PHONE_AU_RE, PHONE, found)
        text = _sub(text, _PHONE_NANP_RE, PHONE, found)
        return RedactionResult(text, tuple(found))
    except Exception:
        # Deliberately no query in the log line.
        logger.error("Search query redaction failed; query withheld", exc_info=False)
        return RedactionResult(_FAILED_PLACEHOLDER, (REDACTION_ERROR,))
