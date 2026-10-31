from __future__ import annotations

import pytest

from xagent.core.utils.timezones import RAILS_ZONE_NAMES, resolve_zone


def _key(name: object) -> str | None:
    zone = resolve_zone(name)  # type: ignore[arg-type]
    return zone.key if zone is not None else None


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Australia/Melbourne", "Australia/Melbourne"),
        ("australia/melbourne", "Australia/Melbourne"),
        ("  America/New_York\t", "America/New_York"),
        ("UTC", "UTC"),
        ("utc", "UTC"),
        ("EST", "EST"),
        ("us/eastern", "US/Eastern"),
        ("Etc/GMT+10", "Etc/GMT+10"),
    ],
)
def test_canonical_names_resolve_to_their_canonical_key(
    supplied: str, expected: str
) -> None:
    assert _key(supplied) == expected


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Wellington", "Pacific/Auckland"),
        ("Pacific Time (US & Canada)", "America/Los_Angeles"),
        ("Canberra", "Australia/Canberra"),
        ("Melbourne", "Australia/Melbourne"),
        ("Sydney", "Australia/Sydney"),
        ("Cape Verde Is.", "Atlantic/Cape_Verde"),
        ("Nuku'alofa", "Pacific/Tongatapu"),
        ("pacific time (us & canada)", "America/Los_Angeles"),
        ("  Pacific   Time (US & Canada) ", "America/Los_Angeles"),
        ("WELLINGTON", "Pacific/Auckland"),
    ],
)
def test_rails_zone_names_resolve_to_their_iana_zone(
    supplied: str, expected: str
) -> None:
    assert _key(supplied) == expected


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        ("Toronto", "America/Toronto"),
        ("toronto", "America/Toronto"),
        (" Vancouver ", "America/Vancouver"),
        ("Los_Angeles", "America/Los_Angeles"),
        ("los angeles", "America/Los_Angeles"),
        ("Port   of Spain", "America/Port_of_Spain"),
    ],
)
def test_a_city_naming_exactly_one_zone_resolves_to_it(
    supplied: str, expected: str
) -> None:
    assert _key(supplied) == expected


def test_a_canonical_name_wins_over_a_rails_name() -> None:
    # Both are Rails keys too ("UTC" -> Etc/UTC, "Singapore" -> Asia/Singapore),
    # but an exact IANA name is taken as the caller wrote it.
    assert _key("UTC") == "UTC"
    assert _key("Singapore") == "Singapore"


def test_a_rails_name_wins_over_an_ambiguous_city() -> None:
    # As a bare city both are ambiguous (two zones each end in the same name),
    # but the Rails table names one zone for them.
    assert _key("Buenos Aires") == "America/Argentina/Buenos_Aires"
    assert _key("Istanbul") == "Europe/Istanbul"


@pytest.mark.parametrize(
    "supplied",
    [
        "Indianapolis",  # America/Indianapolis, America/Indiana/Indianapolis
        "Buenos_Aires",  # America/Buenos_Aires, America/Argentina/Buenos_Aires
        "Nicosia",  # Asia/Nicosia, Europe/Nicosia
    ],
)
def test_an_ambiguous_city_stays_unresolved(supplied: str) -> None:
    assert _key(supplied) is None


@pytest.mark.parametrize(
    "supplied",
    [
        "East",  # only Brazil/East ends in it
        "General",  # only Mexico/General ends in it
        "Eastern",
    ],
)
def test_a_legacy_country_zone_suffix_is_not_a_city(supplied: str) -> None:
    assert _key(supplied) is None


@pytest.mark.parametrize(
    "supplied",
    [
        "Not/AZone",
        "Gotham",
        "",
        "   ",
        None,
        42,
        "A" * 5000,
        "/etc/passwd",
        "../../../etc/passwd",
        "Australia/../../etc/hosts",
        "Melbourne\x00",
        "Australia/Melbourne\x00",
    ],
)
def test_an_unknown_name_stays_unresolved(supplied: object) -> None:
    assert _key(supplied) is None


@pytest.mark.parametrize(
    ("alias", "zone"),
    [
        (alias, zone)
        for alias, zone in sorted(RAILS_ZONE_NAMES.items())
        # Exact IANA names of their own, resolved before the table (see above).
        if alias not in ("UTC", "Singapore")
    ],
)
def test_every_rails_name_resolves_to_its_mapped_zone(alias: str, zone: str) -> None:
    assert _key(alias) == zone
