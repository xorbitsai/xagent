"""Corpus tests for web search query redaction (true and false positives)."""

import pytest

from xagent.core.tools.core.search_query_redaction import (
    redact_search_query,
    valid_abn,
    valid_medicare,
    valid_tfn,
)


@pytest.mark.parametrize(
    "query,expected,entities",
    [
        ("mail jane.doe@example.com now", "mail [EMAIL] now", ("email",)),
        ("a+tag@sub.example.co.uk", "[EMAIL]", ("email",)),
        ("call +61 412 345 678", "call [PHONE]", ("phone",)),
        ("call 0412 345 678", "call [PHONE]", ("phone",)),
        ("call 0412345678", "call [PHONE]", ("phone",)),
        ("call (02) 9999 8888", "call [PHONE]", ("phone",)),
        ("call 02 9999 8888", "call [PHONE]", ("phone",)),
        ("call +44 20 7946 0958", "call [PHONE]", ("phone",)),
        ("call +1 (415) 555-0132", "call [PHONE]", ("phone",)),
        ("call 415-555-0132", "call [PHONE]", ("phone",)),
        ("lives at 12 Smith Street", "lives at [ADDRESS]", ("address",)),
        (
            "12 Smith Street, Surry Hills NSW 2010",
            "[ADDRESS]",
            ("address",),
        ),
        ("unit 4/12 george st sydney nsw 2000", "[ADDRESS]", ("address",)),
        ("Unit 4/12 George St", "[ADDRESS]", ("address",)),
        ("PO Box 1234", "[ADDRESS]", ("address",)),
        ("TFN 123 456 782", "TFN [TFN]", ("tfn",)),
        ("TFN 123456782", "TFN [TFN]", ("tfn",)),
        ("ABN 51 824 753 556", "ABN [ABN]", ("abn",)),
        ("ABN 51824753556", "ABN [ABN]", ("abn",)),
        ("Medicare 2123 45670 1", "Medicare [MEDICARE]", ("medicare",)),
        ("Medicare 2123456701", "Medicare [MEDICARE]", ("medicare",)),
        (
            "jane.doe@example.com 0412 345 678",
            "[EMAIL] [PHONE]",
            ("email", "phone"),
        ),
    ],
)
def test_redacts_true_positives(query, expected, entities):
    result = redact_search_query(query)
    assert result.text == expected
    assert result.entities == entities


@pytest.mark.parametrize(
    "query",
    [
        # years, dates, prices, versions
        "best laptops 2024",
        "released 2024-11-14 and 14/11/2024",
        "price $1,234.56 or 1234567 units",
        "python 3.12.1 release notes",
        "windows 11 22H2 build 22621.3007",
        # order numbers and product codes
        "order ORD-2024-118233 status",
        "invoice INV123456782",
        "ORD-123456782",
        "$123456782",
        "part number 4520-998877-X",
        # checksum failures survive
        "TFN 123 456 783",
        "ABN 51 824 753 557",
        "Medicare 2123 45671 1",
        "Medicare 2123456711",
        "number 98765432101",
        # names of people and organisations survive
        "Elon Musk Tesla Inc Sydney",
        "Jane Doe Acme Pty Ltd annual report",
        # street-type words without a house number or context
        "best street food in Sydney",
        "5 best street food places",
        "10 ways drive growth",
        "Smith Street Melbourne shops",
        "",
    ],
)
def test_leaves_false_positives_alone(query):
    result = redact_search_query(query)
    assert result.text == query
    assert result.entities == ()


def test_checksums():
    assert valid_tfn("123 456 782") and not valid_tfn("123 456 783")
    assert valid_abn("51 824 753 556") and not valid_abn("51 824 753 557")
    assert valid_medicare("2123 45670 1") and not valid_medicare("2123 45671 1")
    # first digit outside 2-6 and issue number 0 are never Medicare numbers
    assert not valid_medicare("1123 45670 1")
    assert not valid_medicare("2123 45670 0")
    # wrong length
    assert not valid_tfn("12345678") and not valid_abn("1234567890")


def test_fails_closed_without_raising(monkeypatch):
    import xagent.core.tools.core.search_query_redaction as mod

    def boom(*a, **k):
        raise RuntimeError("jane.doe@example.com")

    monkeypatch.setattr(mod, "_sub", boom)
    result = mod.redact_search_query("jane.doe@example.com")
    assert result.text == "[REDACTED]"
    assert result.entities == ("redaction_error",)


def test_non_string_input_does_not_raise():
    assert redact_search_query(None).text == ""  # type: ignore[arg-type]


def test_long_input_is_fast():
    import time

    query = ("12 " + "a" * 40 + " ") * 2000
    start = time.perf_counter()
    redact_search_query(query)
    assert time.perf_counter() - start < 2.0
