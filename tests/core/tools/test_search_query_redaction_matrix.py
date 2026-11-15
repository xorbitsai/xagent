"""Shape matrix for the redactor, one row per input form.

``FIXED`` rows are recognised and must come out exactly as listed.
``OUT_OF_SCOPE`` rows are the documented exclusions (see the scope statement in
``search_query_redaction``): they must come out unchanged, so a later change of
scope is a visible change in this file.
"""

import pytest

from xagent.core.tools.core.search_query_redaction import redact_search_query

FIXED = [
    # email
    ("alice@例子.公司", "[EMAIL]"),
    ("alice@münchen.de", "[EMAIL]"),
    ("alice@xn--fsqu00a.xn--55qx5d", "[EMAIL]"),
    ("josé.garcía@example.com", "[EMAIL]"),
    ("用户@example.com", "[EMAIL]"),
    ("jane.doe@sub.例子.公司", "[EMAIL]"),
    ("jane@EXAMPLE。COM", "[EMAIL]"),
    ("a.b+tag@example.co.uk", "[EMAIL]"),
    ('"john doe"@example.com', "[EMAIL]"),
    ("mail jane@example.com.", "mail [EMAIL]."),
    ("(jane@example.com)", "([EMAIL])"),
    ("<jane@example.com>", "<[EMAIL]>"),
    ("mailto:jane@example.com?subject=hi", "mailto:[EMAIL]?subject=hi"),
    ("JANE@EXAMPLE.COM", "[EMAIL]"),
    ("jane@example.com,bob@example.org", "[EMAIL],[EMAIL]"),
    # phone: international, trunk forms, separators
    ("call +61 (0) 412 345 678", "call [PHONE]"),
    ("call +61 (0)412 345 678", "call [PHONE]"),
    ("call +61 0412 345 678", "call [PHONE]"),
    ("call +61 (0)2 9999 8888", "call [PHONE]"),
    ("call +61 2 9999 8888", "call [PHONE]"),
    ("call +61412345678", "call [PHONE]"),
    ("call +61 412-345-678", "call [PHONE]"),
    ("call +61.412.345.678", "call [PHONE]"),
    ("call ＋61 412 345 678", "call [PHONE]"),
    ("tel:+61412345678", "tel:[PHONE]"),
    ("call +61 412 345 678123", "call [PHONE]123"),
    ("call +61 412 345 678 and 0412 345 678", "call [PHONE] and [PHONE]"),
    ("call +44 (0) 20 7946 0958", "call [PHONE]"),
    ("call +44 7911 123456", "call [PHONE]"),
    ("call +1 415 555 0132 x12", "call [PHONE] x12"),
    ("call 0412 345 678 ext 123", "call [PHONE] ext 123"),
    # phone: Australian national, NANP
    ("call 0412.345.678", "call [PHONE]"),
    ("call (04) 1234 5678", "call [PHONE]"),
    ("call 04 1234 5678", "call [PHONE]"),
    ("call 0412 345 678.", "call [PHONE]."),
    ("call 415.555.0132", "call [PHONE]"),
    ("call (415) 555-0132", "call [PHONE]"),
    # TFN / ABN / Medicare
    ("TFN 123 456 782", "TFN [TFN]"),
    ("TFN: 123-456-782", "TFN: [TFN]"),
    ("TFN 123 456 782.", "TFN [TFN]."),
    ("(123 456 782)", "([TFN])"),
    ("ABN 51 824 753 556", "ABN [ABN]"),
    ("ABN: 51-824-753-556", "ABN: [ABN]"),
    ("Medicare 2123 45670 1", "Medicare [MEDICARE]"),
    ("Medicare 2123-45670-1", "Medicare [MEDICARE]"),
    ("Medicare 2123456701/2", "Medicare [MEDICARE]/2"),
    ("Medicare 2123 45670 1 ref 1", "Medicare [MEDICARE] ref 1"),
    # addresses
    ("12 Smith Street", "[ADDRESS]"),
    ("12 Smith St.", "[ADDRESS]"),
    ("12a Smith Street", "[ADDRESS]"),
    ("Unit 4, 12 Smith Street", "[ADDRESS]"),
    ("4/12 Smith Street Surry Hills NSW 2010", "[ADDRESS]"),
    ("12 St Kilda Road Melbourne VIC 3004", "[ADDRESS]"),
    ("12 O'Connell Street", "[ADDRESS]"),
    ("Level 5, 100 Collins Street Melbourne VIC 3000", "[ADDRESS]"),
    ("Flat 2, 10 Downing Street", "[ADDRESS]"),
    ("PO Box 1234 Sydney NSW 2001", "[ADDRESS] Sydney NSW 2001"),
    ("GPO Box 99", "[ADDRESS]"),
    ("Locked Bag 5", "[ADDRESS]"),
    ("Private Bag 7", "[ADDRESS]"),
    # suburb and postcode alone are not personal data and stay
    ("12 Smith Street, Surry Hills, NSW, 2010", "[ADDRESS], Surry Hills, NSW, 2010"),
    ("12 Smith Street Surry Hills 2010", "[ADDRESS] Surry Hills 2010"),
    ("12 Smith Street Sydney", "[ADDRESS] Sydney"),
]

OUT_OF_SCOPE = [
    # email: the domain needs a dot; no address literals; no spaces around "@"
    "jane@localhost",
    "jane@[192.168.0.1]",
    "jane @ example.com",
    # phone: business numbers
    "call 1300 123 456",
    "call 13 12 34",
    # TFN / ABN: glued to letters, dot-separated, irregular grouping
    "tfn123456782",
    "TFN 123.456.782",
    "TFN 12 345 6782",
    "abn51824753556",
    "ABN 51.824.753.556",
    "company number 004 085 616",
    # addresses: lowercase without suburb context, non-English, not number-first
    "12 smith street",
    "12 Rue de la Paix Paris",
    "Hauptstraße 5 Berlin",
    "中山路 12 号",
]


@pytest.mark.parametrize("query,expected", FIXED)
def test_matrix_fixed_rows(query, expected):
    assert redact_search_query(query).text == expected


@pytest.mark.parametrize("query", OUT_OF_SCOPE)
def test_matrix_out_of_scope_rows_are_left_unchanged(query):
    result = redact_search_query(query)
    assert result.text == query
    assert result.entities == ()
