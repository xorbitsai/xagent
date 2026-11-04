import pytest
from fastapi import HTTPException

from xagent.web.services.widget_domains import (
    domain_allowed,
    origin_to_domain,
    require_domain_allowed,
)


@pytest.mark.parametrize(
    ("origin", "expected_domain"),
    [
        ("https://EXAMPLE.com:8443/widget", "example.com:8443"),
        ("EXAMPLE.com", "example.com"),
        ("", ""),
    ],
)
def test_origin_to_domain_preserves_the_current_host_port_contract(
    origin: str, expected_domain: str
) -> None:
    assert origin_to_domain(origin) == expected_domain


@pytest.mark.parametrize(
    ("origin_domain", "allowed_domains", "expected"),
    [
        ("app.example.com", ["example.com"], True),
        ("evil-example.com", ["example.com"], False),
        ("example.com:443", ["example.com"], False),
        ("example.com:443", ["example.com:443"], True),
        ("example.com", [], False),
        ("", ["*"], True),
    ],
)
def test_domain_allowed_preserves_widget_allowlist_matching(
    origin_domain: str, allowed_domains: list[str], expected: bool
) -> None:
    assert domain_allowed(origin_domain, allowed_domains) is expected


@pytest.mark.parametrize(
    ("origin_domain", "allowed_domains"),
    [
        ("example.com", None),
        ("e", "example.com"),
        ("example.com", {"example.com": True}),
        ("example.com", ["example.com", None]),
        ("example.com", [None, "example.com"]),
        ("123", [123]),
        ("true", [True]),
        ("example.com.", [""]),
        ("example.com.", ["   "]),
        ("example.com.", ["\u001c"]),
        ("trusted.example", ["trusted.example", ""]),
    ],
)
def test_domain_allowed_rejects_malformed_persisted_allowlists(
    origin_domain: str, allowed_domains: object
) -> None:
    assert not domain_allowed(origin_domain, allowed_domains)


def test_normalized_domain_composition_preserves_case_insensitive_allowlists() -> None:
    raw_origin = "https://APP.EXAMPLE.COM/widget"
    raw_allowed_domain = " APP.example.COM "

    assert domain_allowed(origin_to_domain(raw_origin), [raw_allowed_domain])


def test_domain_allowed_does_not_normalize_direct_origin_input() -> None:
    assert not domain_allowed("APP.EXAMPLE.COM", ["app.example.com"])


def test_domain_allowed_does_not_parse_scheme_form_allowlist_entries() -> None:
    assert not domain_allowed("example.com", ["https://example.com"])


def test_require_domain_allowed_accepts_case_insensitive_stored_allowlist_entries() -> (
    None
):
    require_domain_allowed(
        origin_to_domain("https://APP.EXAMPLE.COM/widget"),
        [" APP.example.COM "],
        owner_type="agent",
        owner_id=42,
    )


def test_require_domain_allowed_preserves_forbidden_response() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_domain_allowed(
            "untrusted.example",
            ["trusted.example"],
            owner_type="agent",
            owner_id=42,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Domain not allowed: untrusted.example"


def test_require_domain_allowed_maps_malformed_allowlist_to_forbidden() -> None:
    with pytest.raises(HTTPException) as exc_info:
        require_domain_allowed(
            "trusted.example",
            [None, "trusted.example"],
            owner_type="agent",
            owner_id=42,
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Domain not allowed: trusted.example"


@pytest.mark.parametrize(
    ("allowed_domains", "expected_reason"),
    [
        ({"do-not-log-this-policy-value.example": True}, "not_list"),
        (
            ["do-not-log-this-policy-value.example", None],
            "non_string_entry",
        ),
        (["do-not-log-this-policy-value.example", "   "], "blank_entry"),
        (["do-not-log-this-policy-value.example", "\u001c"], "blank_entry"),
    ],
)
def test_require_domain_allowed_logs_bounded_owner_context_for_malformed_policy(
    caplog: pytest.LogCaptureFixture,
    allowed_domains: object,
    expected_reason: str,
) -> None:
    caplog.set_level("WARNING", logger="xagent.web.services.widget_domains")
    sensitive_policy_value = "do-not-log-this-policy-value.example"

    with pytest.raises(HTTPException):
        require_domain_allowed(
            "trusted.example",
            allowed_domains,
            owner_type="agent",
            owner_id=42,
        )

    assert (
        "Rejected malformed widget allowed-domains policy: "
        f"owner_type=agent owner_id=42 reason={expected_reason}"
    ) in caplog.text
    assert sensitive_policy_value not in caplog.text


def test_require_domain_allowed_does_not_log_expected_origin_mismatch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("WARNING", logger="xagent.web.services.widget_domains")

    with pytest.raises(HTTPException):
        require_domain_allowed(
            "untrusted.example",
            ["trusted.example"],
            owner_type="workforce",
            owner_id=84,
        )

    assert "Rejected malformed widget allowed-domains policy" not in caplog.text
