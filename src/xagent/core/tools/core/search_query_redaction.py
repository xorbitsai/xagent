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

Recognised and not recognised
-----------------------------
Email: the local part and each domain label may contain any Unicode letter or
digit (RFC 6531), the domain may use an ASCII or ideographic dot, and a quoted
local part is accepted. A local part or label is either a run of non-CJK
characters or a run of CJK characters (Han, kana, hangul), never a mix, so an
address inside CJK text does not absorb the words beside it; a local part or
label made only of CJK characters still absorbs adjacent CJK text. The last
domain label must be letters only (at least two) or ``xn--`` punycode, so a
version specifier such as ``lodash@4.17.21`` is not an address, but one whose
last dotted part is alphabetic (``react@18.2.0-rc.alpha``) is replaced. The
domain must contain at least one dot, so ``user@localhost`` and ``user@[IP]``
are not recognised, nor is an address with spaces around the ``@``. Not
replaced: ``git@host`` (the ssh user of git hosts) and an address followed by
``:`` and a path (scp-style remotes such as ``deploy@host:/srv``).

Phone: an international number starting with ``+`` (or the full-width ``＋``),
checked against the structure of its country code (+61: 9 national digits,
+1: 10, +44: 10; any other code: at most 15 digits in total), with an optional
trunk prefix (``(0)`` or a leading ``0``). Exactly the digits that belong to
the number are replaced; digits that follow it stay in the query. Also
Australian mobile (04xx xxx xxx, 04 xxxx xxxx, separators space, dot or
hyphen) and landline ((0x) xxxx xxxx), and US/Canada style ten-digit numbers
with separators. An extension (``ext 12``, ``x12``) is kept. 13xx/1300/1800
business numbers are not recognised.

TFN, ABN, Medicare: replaced only when the number stands alone (not glued to
letters, digits or a hyphen, and not part of a decimal or thousands figure),
is written with single spaces, hyphens or nothing between its groups, and its
checksum passes. Numbers glued to letters, written with dots, or grouped
irregularly are not recognised, so order numbers, versions and prices survive.

Address: an English-style street address (house number with optional unit or
level, a capitalised street name and a street-type word), or any-case street
name followed by ``<suburb> <STATE> <postcode>``; and PO Box, GPO Box, Locked
Bag and Private Bag numbers. A suburb or postcode on its own is not replaced.
Lowercase street names without a suburb and state, addresses in other
languages, and addresses that do not start with a house number are not
recognised.

Not covered: names of people and organisations; text the model restates in its
reply; search tool arguments other than ``query`` (for example Exa's
``include_text``); results returned by a provider. In console trace lines, a
search call's arguments that were cut or cannot be parsed are replaced whole by
a fixed marker, never partly shown.
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

# Email. ``\w`` is Unicode-aware, so internationalised mailboxes (RFC 6531)
# match. Scripts written without spaces (Han, kana, hangul) must not swallow the
# text beside the address, so the local part and each domain label are a run of
# either non-CJK word characters or CJK characters, never a mix. The last
# domain label is the TLD: letters only (at least two), or ``xn--`` punycode, so
# version specifiers such as ``lodash@4.17.21`` are not addresses.
_CJK = (
    "\u1100-\u11ff\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af"
    "\uf900-\ufaff\uff66-\uff9f"
)
_NOT_CJK = f"(?![{_CJK}])"
_NON_CJK_WORD_END = f"(?!{_NOT_CJK}[\\w\\-])"
_LOCAL = (
    r'(?:"[^"\\\r\n@]{1,64}"'
    + f"|(?:{_NOT_CJK}[\\w.%+\\-]){{1,64}}"
    + f"|[{_CJK}]{{1,64}})"
)
_LABEL = f"(?:(?:{_NOT_CJK}[^\\W_]|-){{1,63}}|[{_CJK}]{{1,63}})"
_TLD = (
    r"(?:xn--[A-Za-z0-9\-]{2,59}(?![\w\-])"
    + f"|(?:{_NOT_CJK}[^\\W\\d_]){{2,63}}{_NON_CJK_WORD_END}"
    + f"|[{_CJK}]{{2,63}}(?![{_CJK}]))"
)
_EMAIL_RE = re.compile(_LOCAL + "@(?:" + _LABEL + r"[.\u3002\uff0e\uff61]){1,8}" + _TLD)
# scp-style remote ``user@host:path``: the address is followed by ``:`` and a
# path, not by prose.
_SCP_PATH_RE = re.compile(r":(?:[/~]|[\w.\-]+/)")


def _email_ok(m: Match[str]) -> bool:
    if m.group(0).partition("@")[0] == "git":
        return False  # ssh user of every git host, never a person's mailbox
    return _SCP_PATH_RE.match(m.string, m.end()) is None


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
_PO_BOX_RE = re.compile(
    r"\b(?:P\.?\s?O\.?\s+Box|GPO\s+Box|(?:Locked|Private)\s+Bag)\s+\d{1,6}\b",
    re.IGNORECASE,
)

# Phone numbers. International needs a leading "+"; Australian national forms
# are mobile 04xx xxx xxx and landline (0x) xxxx xxxx; NANP needs separators.
_PHONE_INTL_RE = re.compile(
    r"(?<![\w+\uff0b])[+\uff0b]\d{1,3}[ .\-]?(?:\(\d{1,4}\)[ .\-]?)?\d{1,4}"
    r"(?:[ .\-]?\d{2,5}){1,4}(?!\d)"
)
_PHONE_AU_RE = re.compile(
    r"(?<![\w\-./$+\uff0b])(?:\(0[2378]\)|0[2378])[ \-]?\d{4}[ \-]?\d{4}(?!\d)"
    r"|(?<![\w\-./$+\uff0b])(?:\(04\)|04)"
    r"(?:\d{2}[ .\-]?\d{3}[ .\-]?\d{3}|[ .\-]?\d{4}[ .\-]?\d{4})(?!\d)"
)
_PHONE_NANP_RE = re.compile(
    r"(?<![\w\-./$+\uff0b])(?:\(\d{3}\)[ ]?|\d{3}[ .\-])\d{3}[ .\-]\d{4}(?![\w\-]|[.,]\d)"
)


# National significant number length by country code: +61 9 digits, +1 10,
# +44 10 (UK numbers are 9-10; the longer is taken). The trunk prefix after the
# country code ("(0)" or a leading 0 of +61/+44 numbers) is not part of it.
_NATIONAL_LEN = {"61": 9, "44": 10, "1": 10}


def _intl_prefix_len(text: str) -> int:
    """Length of the phone-number prefix of an international-shaped match.

    The pattern is greedy, so digits that follow a number (a TFN, an order id)
    can be glued onto it. For a known country code the structure fixes how many
    national digits belong to the number (the trunk ``0`` is skipped) and
    exactly that many are consumed, even in the middle of a group. For other
    country codes the longest prefix of digit groups with at most 15 digits
    (E.164) and at least 8 is taken. 0 means no usable prefix.
    """
    digits = [(i, c) for i, c in enumerate(text) if c.isdigit()]
    num = "".join(c for _, c in digits)
    for cc in ("61", "44", "1"):
        if num.startswith(cc):
            national = num[len(cc) :]
            skip = 1 if cc != "1" and national.startswith("0") else 0
            need = _NATIONAL_LEN[cc]
            have = len(national) - skip
            if len(cc) + skip + min(have, need) < 8:
                return 0
            last = len(cc) + skip + min(have, need) - 1
            return digits[last][0] + 1
    groups = list(re.finditer(r"\d+", text))
    total = 0
    best = 0
    for g in groups:
        total += len(g.group(0))
        if total > 15:
            break
        if total >= 8:
            best = g.end()
    return best


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


def _sub_intl_phone(text: str, found: List[str]) -> str:
    def repl(m: Match[str]) -> str:
        raw = m.group(0)
        cut = _intl_prefix_len(raw)
        if cut == 0:
            return raw
        if PHONE not in found:
            found.append(PHONE)
        # Whatever follows the number stays for the later recognisers.
        return _PLACEHOLDERS[PHONE] + raw[cut:]

    return _PHONE_INTL_RE.sub(repl, text)


def _spans(pattern: "re.Pattern[str]", text: str) -> List[Tuple[int, int]]:
    return [m.span() for m in pattern.finditer(text)]


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
        text = _sub(text, _EMAIL_RE, EMAIL, found, _email_ok)
        text = _sub(text, _ADDRESS_CAP_RE, ADDRESS, found)
        text = _sub(text, _ADDRESS_CTX_RE, ADDRESS, found)
        text = _sub(text, _PO_BOX_RE, ADDRESS, found)
        # "+" numbers first so their trailing digits are never read as a TFN.
        text = _sub_intl_phone(text, found)
        # Longest identifier first so an 11-digit ABN is never read as a TFN.
        text = _sub(text, _ABN_RE, ABN, found, lambda m: valid_abn(m.group(0)))
        text = _sub(
            text, _MEDICARE_RE, MEDICARE, found, lambda m: valid_medicare(m.group(0))
        )
        # What is still ABN- or Medicare-shaped here failed its checksum. A TFN
        # candidate that starts inside one is just its tail ("ABN 50 123 456 782"
        # must not become "ABN 50 [TFN]").
        rejected = _spans(_ABN_RE, text) + _spans(_MEDICARE_RE, text)
        text = _sub(
            text,
            _TFN_RE,
            TFN,
            found,
            lambda m: valid_tfn(m.group(0))
            and not any(a <= m.start() < b for a, b in rejected),
        )
        text = _sub(text, _PHONE_AU_RE, PHONE, found)
        text = _sub(text, _PHONE_NANP_RE, PHONE, found)
        return RedactionResult(text, tuple(found))
    except Exception:
        # Deliberately no query in the log line.
        logger.error("Search query redaction failed; query withheld", exc_info=False)
        return RedactionResult(_FAILED_PLACEHOLDER, (REDACTION_ERROR,))
