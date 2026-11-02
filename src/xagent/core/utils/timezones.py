"""Resolve a caller-supplied timezone name to a canonical IANA zone.

Names reach the agent from integrations and from the model in more forms than
the IANA one: a bare city ("Melbourne") or a Ruby on Rails zone name
("Pacific Time (US & Canada)"). ``resolve_zone`` accepts those as well, and
always returns a zone whose ``key`` is the canonical IANA name, so callers can
report which zone was actually used.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

# ActiveSupport::TimeZone::MAPPING from Ruby on Rails v8.1.4
# (activesupport/lib/active_support/values/time_zone.rb), copied verbatim:
# Rails zone names ("Eastern Time (US & Canada)", Time.zone.name in a Rails
# application) mapped to the IANA zone each one stands for.
RAILS_ZONE_NAMES: Mapping[str, str] = MappingProxyType(
    {
        "International Date Line West": "Etc/GMT+12",
        "Midway Island": "Pacific/Midway",
        "American Samoa": "Pacific/Pago_Pago",
        "Hawaii": "Pacific/Honolulu",
        "Alaska": "America/Juneau",
        "Pacific Time (US & Canada)": "America/Los_Angeles",
        "Tijuana": "America/Tijuana",
        "Mountain Time (US & Canada)": "America/Denver",
        "Arizona": "America/Phoenix",
        "Chihuahua": "America/Chihuahua",
        "Mazatlan": "America/Mazatlan",
        "Central Time (US & Canada)": "America/Chicago",
        "Saskatchewan": "America/Regina",
        "Guadalajara": "America/Mexico_City",
        "Mexico City": "America/Mexico_City",
        "Monterrey": "America/Monterrey",
        "Central America": "America/Guatemala",
        "Eastern Time (US & Canada)": "America/New_York",
        "Indiana (East)": "America/Indiana/Indianapolis",
        "Bogota": "America/Bogota",
        "Lima": "America/Lima",
        "Quito": "America/Lima",
        "Atlantic Time (Canada)": "America/Halifax",
        "Caracas": "America/Caracas",
        "La Paz": "America/La_Paz",
        "Santiago": "America/Santiago",
        "Asuncion": "America/Asuncion",
        "Newfoundland": "America/St_Johns",
        "Brasilia": "America/Sao_Paulo",
        "Buenos Aires": "America/Argentina/Buenos_Aires",
        "Montevideo": "America/Montevideo",
        "Georgetown": "America/Guyana",
        "Puerto Rico": "America/Puerto_Rico",
        "Greenland": "America/Nuuk",
        "Mid-Atlantic": "Atlantic/South_Georgia",
        "Azores": "Atlantic/Azores",
        "Cape Verde Is.": "Atlantic/Cape_Verde",
        "Dublin": "Europe/Dublin",
        "Edinburgh": "Europe/London",
        "Lisbon": "Europe/Lisbon",
        "London": "Europe/London",
        "Casablanca": "Africa/Casablanca",
        "Monrovia": "Africa/Monrovia",
        "UTC": "Etc/UTC",
        "Belgrade": "Europe/Belgrade",
        "Bratislava": "Europe/Bratislava",
        "Budapest": "Europe/Budapest",
        "Ljubljana": "Europe/Ljubljana",
        "Prague": "Europe/Prague",
        "Sarajevo": "Europe/Sarajevo",
        "Skopje": "Europe/Skopje",
        "Warsaw": "Europe/Warsaw",
        "Zagreb": "Europe/Zagreb",
        "Brussels": "Europe/Brussels",
        "Copenhagen": "Europe/Copenhagen",
        "Madrid": "Europe/Madrid",
        "Paris": "Europe/Paris",
        "Amsterdam": "Europe/Amsterdam",
        "Berlin": "Europe/Berlin",
        "Bern": "Europe/Zurich",
        "Zurich": "Europe/Zurich",
        "Rome": "Europe/Rome",
        "Stockholm": "Europe/Stockholm",
        "Vienna": "Europe/Vienna",
        "West Central Africa": "Africa/Algiers",
        "Bucharest": "Europe/Bucharest",
        "Cairo": "Africa/Cairo",
        "Helsinki": "Europe/Helsinki",
        "Kyiv": "Europe/Kyiv",
        "Riga": "Europe/Riga",
        "Sofia": "Europe/Sofia",
        "Tallinn": "Europe/Tallinn",
        "Vilnius": "Europe/Vilnius",
        "Athens": "Europe/Athens",
        "Istanbul": "Europe/Istanbul",
        "Minsk": "Europe/Minsk",
        "Jerusalem": "Asia/Jerusalem",
        "Harare": "Africa/Harare",
        "Pretoria": "Africa/Johannesburg",
        "Kaliningrad": "Europe/Kaliningrad",
        "Moscow": "Europe/Moscow",
        "St. Petersburg": "Europe/Moscow",
        "Volgograd": "Europe/Volgograd",
        "Samara": "Europe/Samara",
        "Kuwait": "Asia/Kuwait",
        "Riyadh": "Asia/Riyadh",
        "Nairobi": "Africa/Nairobi",
        "Baghdad": "Asia/Baghdad",
        "Tehran": "Asia/Tehran",
        "Abu Dhabi": "Asia/Muscat",
        "Muscat": "Asia/Muscat",
        "Baku": "Asia/Baku",
        "Tbilisi": "Asia/Tbilisi",
        "Yerevan": "Asia/Yerevan",
        "Kabul": "Asia/Kabul",
        "Ekaterinburg": "Asia/Yekaterinburg",
        "Islamabad": "Asia/Karachi",
        "Karachi": "Asia/Karachi",
        "Tashkent": "Asia/Tashkent",
        "Chennai": "Asia/Kolkata",
        "Kolkata": "Asia/Kolkata",
        "Mumbai": "Asia/Kolkata",
        "New Delhi": "Asia/Kolkata",
        "Kathmandu": "Asia/Kathmandu",
        "Dhaka": "Asia/Dhaka",
        "Sri Jayawardenepura": "Asia/Colombo",
        "Almaty": "Asia/Almaty",
        "Astana": "Asia/Almaty",
        "Novosibirsk": "Asia/Novosibirsk",
        "Rangoon": "Asia/Yangon",
        "Bangkok": "Asia/Bangkok",
        "Hanoi": "Asia/Bangkok",
        "Jakarta": "Asia/Jakarta",
        "Krasnoyarsk": "Asia/Krasnoyarsk",
        "Beijing": "Asia/Shanghai",
        "Chongqing": "Asia/Chongqing",
        "Hong Kong": "Asia/Hong_Kong",
        "Urumqi": "Asia/Urumqi",
        "Kuala Lumpur": "Asia/Kuala_Lumpur",
        "Singapore": "Asia/Singapore",
        "Taipei": "Asia/Taipei",
        "Perth": "Australia/Perth",
        "Irkutsk": "Asia/Irkutsk",
        "Ulaanbaatar": "Asia/Ulaanbaatar",
        "Seoul": "Asia/Seoul",
        "Osaka": "Asia/Tokyo",
        "Sapporo": "Asia/Tokyo",
        "Tokyo": "Asia/Tokyo",
        "Yakutsk": "Asia/Yakutsk",
        "Darwin": "Australia/Darwin",
        "Adelaide": "Australia/Adelaide",
        "Canberra": "Australia/Canberra",
        "Melbourne": "Australia/Melbourne",
        "Sydney": "Australia/Sydney",
        "Brisbane": "Australia/Brisbane",
        "Hobart": "Australia/Hobart",
        "Vladivostok": "Asia/Vladivostok",
        "Guam": "Pacific/Guam",
        "Port Moresby": "Pacific/Port_Moresby",
        "Magadan": "Asia/Magadan",
        "Srednekolymsk": "Asia/Srednekolymsk",
        "Solomon Is.": "Pacific/Guadalcanal",
        "New Caledonia": "Pacific/Noumea",
        "Fiji": "Pacific/Fiji",
        "Kamchatka": "Asia/Kamchatka",
        "Marshall Is.": "Pacific/Majuro",
        "Auckland": "Pacific/Auckland",
        "Wellington": "Pacific/Auckland",
        "Nuku'alofa": "Pacific/Tongatapu",
        "Tokelau Is.": "Pacific/Fakaofo",
        "Chatham Is.": "Pacific/Chatham",
        "Samoa": "Pacific/Apia",
    }
)

_RAILS_ZONE_BY_LOWERCASE = {
    name.lower(): zone for name, zone in RAILS_ZONE_NAMES.items()
}

# The AREA of tzdata's AREA/LOCATION names: a continent or ocean. A bare city
# is matched only against these, not against legacy country groupings such as
# US/, Canada/, Brazil/ or Mexico/, whose last segment is a region ("East" in
# Brazil/East, "General" in Mexico/General).
_CITY_AREAS = frozenset(
    {
        "Africa",
        "America",
        "Antarctica",
        "Arctic",
        "Asia",
        "Atlantic",
        "Australia",
        "Europe",
        "Indian",
        "Pacific",
    }
)

# Legacy names under Australia/ that tzdata keeps as links to current zones
# (Australia/Victoria -> Australia/Melbourne). Their last segment is a state,
# territory, island, county or compass direction, not a city, so they are left
# out of city matching like the country groupings above: "Victoria" would
# otherwise answer for Victoria, BC with Melbourne time. Written in full, each
# still resolves as an exact IANA name.
_LEGACY_REGION_ZONES = frozenset(
    {
        "Australia/ACT",
        "Australia/LHI",
        "Australia/NSW",
        "Australia/North",
        "Australia/Queensland",
        "Australia/South",
        "Australia/Tasmania",
        "Australia/Victoria",
        "Australia/West",
        "Australia/Yancowinna",
    }
)


@lru_cache(maxsize=1)
def _zone_by_lowercase() -> dict[str, str]:
    return {name.lower(): name for name in available_timezones()}


@lru_cache(maxsize=1)
def _zone_by_city() -> dict[str, str]:
    # Keyed by the lowercased last segment of each name under a city area,
    # legacy region names excluded. A city resolves only when exactly one of
    # those names ends in it and that name has two segments (AREA/CITY).
    #
    # A city that ends more than one name stays unresolved
    # (America/Indianapolis, America/Indiana/Indianapolis). Such a pair is
    # usually an old and a current name for one zone rather than two different
    # clocks, but zoneinfo does not expose which name links to which, so this
    # table does not choose between them; the full name always resolves.
    #
    # A three-segment name (America/Argentina/San_Juan,
    # America/North_Dakota/Center) ends in a town standing for a sub-region,
    # while the bare name usually means somewhere better known (San Juan,
    # Puerto Rico), so such a zone is reached only through its full name. It
    # still counts towards its last segment above, so Indianapolis stays
    # unresolved instead of matching America/Indianapolis alone.
    zones_by_city: defaultdict[str, list[str]] = defaultdict(list)
    for name in available_timezones():
        area, _, rest = name.partition("/")
        if area in _CITY_AREAS and rest and name not in _LEGACY_REGION_ZONES:
            zones_by_city[name.rsplit("/", 1)[1].lower()].append(name)
    return {
        city: zones[0]
        for city, zones in zones_by_city.items()
        if len(zones) == 1 and zones[0].count("/") == 1
    }


def _canonical_zone_name(name: str) -> Optional[str]:
    # Tiers in order, each consulted only when the one before has no entry:
    # an exact IANA name, a Rails zone name, then a bare city as
    # _zone_by_city matches it.
    by_lowercase = _zone_by_lowercase()
    lowered = name.strip().lower()
    canonical = by_lowercase.get(lowered)
    if canonical is not None:
        return canonical
    words = " ".join(lowered.split())
    rails_zone = _RAILS_ZONE_BY_LOWERCASE.get(words)
    if rails_zone is not None:
        return by_lowercase.get(rails_zone.lower())
    # tzdata writes a space in a city name as "_" (America/Los_Angeles).
    return _zone_by_city().get(words.replace(" ", "_"))


def resolve_zone(name: Optional[str]) -> Optional[ZoneInfo]:
    """Return the zone a caller-supplied name refers to, or None.

    None covers a missing, unknown or ambiguous name, and a bare name the
    city tier does not match (see _zone_by_city); what that means (fall back
    to UTC, or refuse) is the caller's decision.
    """
    # Every tier ends in a name taken from available_timezones() rather than
    # ZoneInfo(name) directly: ZoneInfo.key echoes whatever string it was
    # given, and tzdata lookup is case-sensitive on Linux but not on macOS,
    # so the reported zone would otherwise vary by input case and filesystem.
    # Those names guarantee a canonical key (e.g. "australia/melbourne" ->
    # "Australia/Melbourne") and identical behavior across platforms.
    if not isinstance(name, str) or not name.strip():
        return None
    canonical = _canonical_zone_name(name)
    if canonical is None:
        return None
    try:
        return ZoneInfo(canonical)
    except (ZoneInfoNotFoundError, ValueError, OSError, KeyError):
        return None
