import json
import socket
import sys
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import magento
from xagent.web.tools.mcp import utils as mcp_utils


def _fake_getaddrinfo(*ips):
    def _impl(host, port, *args, **kwargs):
        return [
            (
                socket.AF_INET6 if ":" in ip else socket.AF_INET,
                socket.SOCK_STREAM,
                6,
                "",
                (ip, port),
            )
            for ip in ips
        ]

    return _impl


_UNSET = object()


class MockResponse:
    def __init__(
        self,
        json_data=_UNSET,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        json_raises: bool = False,
        content: bytes | None = None,
    ):
        # `json_data` defaults to a sentinel (not None/False/{}) so a caller
        # that explicitly passes a falsy JSON value -- None, False, 0, "",
        # {} -- gets that value actually serialized into .text/.content and
        # round-tripped through .json() below, rather than being silently
        # treated the same as "no body was configured for this test".
        self._json_data = {} if json_data is _UNSET else json_data
        self._json_raises = json_raises
        self.status_code = status_code
        if text:
            self.text = text
        elif json_data is _UNSET:
            self.text = ""
        else:
            self.text = json.dumps(self._json_data)
        self.content = content if content is not None else self.text.encode()
        self.url = url

    def json(self):
        if self._json_raises:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error for url: {self.url}", response=self
            )


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://store.example.com")
    monkeypatch.setenv("MAGENTO_ACCESS_TOKEN", "test-token")
    monkeypatch.delenv("MAGENTO_STORE_CODE", raising=False)
    # _base_url() resolves DNS to catch a hostname that rebinds to a private
    # address; tests must not depend on real network/DNS, so every test
    # gets a fake resolver returning an unambiguously public IP by default.
    monkeypatch.setattr(magento.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1"))


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("MAGENTO_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="MAGENTO_ACCESS_TOKEN"):
        magento._headers()


def test_headers_include_bearer_token_and_json_content_type():
    assert magento._headers() == {
        "Authorization": "Bearer test-token",
        "Content-Type": "application/json",
    }


def test_headers_strips_padded_access_token(monkeypatch):
    monkeypatch.setenv("MAGENTO_ACCESS_TOKEN", "  test-token  ")

    assert magento._headers()["Authorization"] == "Bearer test-token"


def test_base_url_requires_env(monkeypatch):
    monkeypatch.delenv("MAGENTO_BASE_URL")

    with pytest.raises(ValueError, match="MAGENTO_BASE_URL"):
        magento._base_url()


def test_base_url_accepts_bare_https_origin():
    assert magento._base_url() == "https://store.example.com"


def test_base_url_adds_https_scheme_when_missing(monkeypatch):
    monkeypatch.setenv("MAGENTO_BASE_URL", "store.example.com")

    assert magento._base_url() == "https://store.example.com"


def test_base_url_preserves_custom_port(monkeypatch):
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://store.example.com:8443")

    assert magento._base_url() == "https://store.example.com:8443"


def test_base_url_brackets_ipv6_literal_host(monkeypatch):
    # urlsplit(...).hostname strips the brackets from a bare IPv6 literal
    # ("[::1]:8443" -> "::1"), so rebuilding the origin without re-adding
    # them would collide the address's own colons with the port separator
    # ("https://::1:8443", not a valid authority).
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://[::1]:8443")

    assert magento._base_url() == "https://[::1]:8443"


@pytest.mark.parametrize(
    "value, match",
    [
        ("http://store.example.com", "https"),
        ("user:pass@store.example.com", "credentials"),
        ("store.example.com/rest", "path"),
        ("store.example.com?x=1", "path"),
        ("store.example.com#frag", "path"),
        ("https://store.example.com:not-a-port", "port"),
        ("https://", "hostname"),
    ],
)
def test_base_url_rejects_invalid_url(monkeypatch, value, match):
    monkeypatch.setenv("MAGENTO_BASE_URL", value)

    with pytest.raises(ValueError, match=match):
        magento._base_url()


def test_base_url_raises_when_no_addresses_are_returned(monkeypatch):
    monkeypatch.setattr(magento.socket, "getaddrinfo", lambda *a, **k: [])

    with pytest.raises(ValueError, match="could not be resolved"):
        magento._base_url()


def test_base_url_rejects_host_resolving_to_private_ip(monkeypatch):
    monkeypatch.setattr(magento.socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))

    with pytest.raises(ValueError, match="not allowed"):
        magento._base_url()


def test_base_url_rejects_when_any_resolved_address_is_private(monkeypatch):
    monkeypatch.setattr(
        magento.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "10.0.0.5")
    )

    with pytest.raises(ValueError, match="not allowed"):
        magento._base_url()


def test_base_url_raises_when_dns_resolution_fails(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(magento.socket, "getaddrinfo", _raise)

    with pytest.raises(ValueError, match="could not be resolved"):
        magento._base_url()


def test_resolve_store_host_returns_hostname_port_and_first_valid_ip(monkeypatch):
    monkeypatch.setattr(
        magento.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "2.2.2.2")
    )

    hostname, port, pinned_ip = magento._resolve_store_host()

    assert hostname == "store.example.com"
    assert port is None
    assert pinned_ip == "1.1.1.1"


def test_resolve_store_host_idna_encodes_non_ascii_hostname(monkeypatch):
    # urllib3's own URL parsing IDNA-encodes a non-ASCII host to punycode
    # before it ever reaches the connection pool -- if `hostname` here
    # stayed raw Unicode, _pinned_dns's `host == hostname` comparison would
    # never match the punycode `host` urllib3 actually dials, silently
    # skipping the pinning defense for exactly this input.
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://münchen.example.com")

    hostname, _port, _pinned_ip = magento._resolve_store_host()

    assert hostname == "xn--mnchen-3ya.example.com"


def test_resolve_store_host_idna_encodes_using_urllib3s_actual_encoder(monkeypatch):
    # Python's stdlib `str.encode("idna")` codec (IDNA2003) and the
    # third-party `idna` package urllib3 actually uses (IDNA2008/UTS46)
    # disagree on some real hostnames -- e.g. this one case-folds "ß"
    # differently, so the stdlib codec would produce "fass.de" (a
    # different, unrelated domain) while urllib3 would dial "xn--fa-hia.de"
    # for the exact same MAGENTO_BASE_URL. _resolve_store_host() must match
    # what urllib3 will actually connect to, not just produce *some*
    # ASCII-looking encoding of the host.
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://faß.de")

    hostname, _port, _pinned_ip = magento._resolve_store_host()

    assert hostname == "xn--fa-hia.de"


def test_resolve_store_host_raises_clear_error_for_invalid_idna_hostname(monkeypatch):
    # idna.IDNAError (a UnicodeError subclass) for a non-ASCII hostname
    # that's also invalid under IDNA's own rules (an underscore, not
    # allowed under std3_rules) -- must surface as a clear ValueError,
    # not propagate the raw IDNAError.
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://café_.example.com")

    with pytest.raises(ValueError, match="invalid hostname"):
        magento._resolve_store_host()


def test_resolve_store_host_raises_clear_error_when_idna_package_missing(monkeypatch):
    monkeypatch.setenv("MAGENTO_BASE_URL", "https://münchen.example.com")
    monkeypatch.setitem(sys.modules, "idna", None)

    with pytest.raises(ValueError, match="'idna' package is not installed"):
        magento._resolve_store_host()


def test_pinned_dns_redirects_matching_host_to_pinned_ip(monkeypatch):
    calls = []

    def _fake_create_connection(address, *args, **kwargs):
        calls.append(address)
        return "sentinel-socket"

    monkeypatch.setattr(
        magento.urllib3_connection, "create_connection", _fake_create_connection
    )

    with magento._pinned_dns("store.example.com", "9.9.9.9"):
        result = magento.urllib3_connection.create_connection(
            ("store.example.com", 443)
        )

    assert result == "sentinel-socket"
    assert calls == [("9.9.9.9", 443)]


def test_pinned_dns_leaves_other_hosts_untouched(monkeypatch):
    calls = []

    def _fake_create_connection(address, *args, **kwargs):
        calls.append(address)
        return "sentinel-socket"

    monkeypatch.setattr(
        magento.urllib3_connection, "create_connection", _fake_create_connection
    )

    with magento._pinned_dns("store.example.com", "9.9.9.9"):
        magento.urllib3_connection.create_connection(("other-host.example.com", 443))

    assert calls == [("other-host.example.com", 443)]


def test_pinned_dns_restores_original_function_on_exit(monkeypatch):
    original = magento.urllib3_connection.create_connection

    with magento._pinned_dns("store.example.com", "9.9.9.9"):
        assert magento.urllib3_connection.create_connection is not original

    assert magento.urllib3_connection.create_connection is original


def test_pinned_dns_restores_original_function_on_exception(monkeypatch):
    original = magento.urllib3_connection.create_connection

    with pytest.raises(RuntimeError):
        with magento._pinned_dns("store.example.com", "9.9.9.9"):
            raise RuntimeError("boom")

    assert magento.urllib3_connection.create_connection is original


def test_request_pins_connection_to_validated_ip(monkeypatch):
    # End-to-end: _request() must pin the same IP _resolve_store_host()
    # validated, not let requests/urllib3 re-resolve independently.
    connect_calls = []

    def _fake_create_connection(address, *args, **kwargs):
        connect_calls.append(address)
        raise OSError("no real network in tests")

    monkeypatch.setattr(
        magento.urllib3_connection, "create_connection", _fake_create_connection
    )

    with pytest.raises(RuntimeError):
        magento._request("GET", "/products/abc")

    assert connect_calls
    assert all(addr[0] == "1.1.1.1" for addr in connect_calls)


def test_api_base_url_defaults_to_v1(monkeypatch):
    assert magento._api_base_url() == "https://store.example.com/rest/V1"


def test_api_base_url_uses_store_code_when_set(monkeypatch):
    monkeypatch.setenv("MAGENTO_STORE_CODE", "default")

    assert magento._api_base_url() == "https://store.example.com/rest/default/V1"


@pytest.mark.parametrize("store_code", ["../V1", "default/foo", "default?x=1", "a b"])
def test_api_base_url_rejects_invalid_store_code(monkeypatch, store_code):
    monkeypatch.setenv("MAGENTO_STORE_CODE", store_code)

    with pytest.raises(ValueError, match="MAGENTO_STORE_CODE"):
        magento._api_base_url()


@pytest.mark.parametrize("store_code", ["V1", "v1"])
def test_api_base_url_rejects_v1_as_store_code(monkeypatch, store_code):
    # A literal "V1" store code would build the confusing, wrong
    # /rest/V1/V1/... path (V1 is the API version segment, not a store
    # view) -- reject it with a clear reason instead.
    monkeypatch.setenv("MAGENTO_STORE_CODE", store_code)

    with pytest.raises(ValueError, match="MAGENTO_STORE_CODE"):
        magento._api_base_url()


@pytest.mark.parametrize(
    "limit, expected",
    [
        (0, 1),
        (-5, 1),
        (1, 1),
        (magento.MAX_LIMIT, magento.MAX_LIMIT),
        (magento.MAX_LIMIT + 1, magento.MAX_LIMIT),
        (10_000, magento.MAX_LIMIT),
    ],
)
def test_clamp_limit_boundaries(limit, expected):
    assert magento._clamp_limit(limit) == expected


def test_search_criteria_params_ands_filters_across_separate_groups():
    # Magento ORs filters *within* one filter_groups entry and ANDs *across*
    # filter_groups entries -- every filter here must land in its own group
    # so multiple filters are required simultaneously (AND), not OR'd.
    params = magento._search_criteria_params(
        [("sku", "like", "%shirt%"), ("status", "eq", "1")],
        page_size=10,
        current_page=2,
    )

    assert params == {
        "searchCriteria[pageSize]": 10,
        "searchCriteria[currentPage]": 2,
        "searchCriteria[filter_groups][0][filters][0][field]": "sku",
        "searchCriteria[filter_groups][0][filters][0][value]": "%shirt%",
        "searchCriteria[filter_groups][0][filters][0][condition_type]": "like",
        "searchCriteria[filter_groups][1][filters][0][field]": "status",
        "searchCriteria[filter_groups][1][filters][0][value]": "1",
        "searchCriteria[filter_groups][1][filters][0][condition_type]": "eq",
    }


def test_search_criteria_params_with_no_filters():
    params = magento._search_criteria_params([], page_size=25, current_page=1)

    assert params == {
        "searchCriteria[pageSize]": 25,
        "searchCriteria[currentPage]": 1,
    }


def test_paginated_result_has_more_when_more_pages_remain():
    items, has_more = magento._paginated_result(
        {"items": [{"sku": "a"}], "total_count": 30}, page_size=10, current_page=1
    )

    assert items == [{"sku": "a"}]
    assert has_more is True


def test_paginated_result_no_more_on_last_page():
    _items, has_more = magento._paginated_result(
        {"items": [{"sku": "a"}], "total_count": 5}, page_size=10, current_page=1
    )

    assert has_more is False


def test_paginated_result_rejects_non_dict_payload():
    with pytest.raises(ValueError, match="JSON object"):
        magento._paginated_result([], page_size=10, current_page=1)


def test_paginated_result_rejects_non_list_items():
    with pytest.raises(ValueError, match="items"):
        magento._paginated_result({"items": "nope"}, page_size=10, current_page=1)


@pytest.mark.parametrize("items_value", [0, "", False])
def test_paginated_result_rejects_falsy_wrong_type_items(items_value):
    # A naive `payload.get("items") or []` would silently treat any of
    # these as "no items" instead of the malformed response they actually
    # are, since they're falsy but not a list.
    with pytest.raises(ValueError, match="items"):
        magento._paginated_result({"items": items_value}, page_size=10, current_page=1)


def test_paginated_result_defaults_missing_items_to_empty_list():
    items, has_more = magento._paginated_result(
        {"total_count": 0}, page_size=10, current_page=1
    )

    assert items == []
    assert has_more is False


def test_paginated_result_rejects_missing_total_count():
    # A missing/malformed total_count must not be indistinguishable from
    # "this is genuinely the last page" -- silently defaulting has_more to
    # False would truncate results for any caller paginating mechanically,
    # against a non-compliant fork/gateway that violates Magento's own
    # SearchResultsInterface contract.
    with pytest.raises(ValueError, match="total_count"):
        magento._paginated_result({"items": []}, page_size=10, current_page=1)


@pytest.mark.parametrize("total_count_value", ["30", None, 12.5, True, False])
def test_paginated_result_rejects_non_int_total_count(total_count_value):
    # isinstance(x, int) alone would accept a bool here too (bool is an
    # int subclass in Python) -- a non-compliant "total_count": true/false
    # must still be rejected, not silently evaluated as 1/0.
    with pytest.raises(ValueError, match="total_count"):
        magento._paginated_result(
            {"items": [], "total_count": total_count_value},
            page_size=10,
            current_page=1,
        )


def test_escape_like_escapes_percent_and_underscore():
    assert magento._escape_like("ABC_1%off") == "ABC\\_1\\%off"


def test_escape_like_escapes_backslash_first():
    assert magento._escape_like("a\\b_c") == "a\\\\b\\_c"


def test_stringify_param_renders_scalars_as_str():
    assert magento._stringify_param(42) == "42"
    assert magento._stringify_param("sku") == "sku"


def test_stringify_param_renders_dict_and_list_as_json():
    assert magento._stringify_param({"a": 1}) == '{"a": 1}'
    assert magento._stringify_param([1, "x"]) == '[1, "x"]'


def test_extract_error_detail_substitutes_named_parameters():
    response = MockResponse(
        json_data={
            "message": "Consumer is not authorized to access %resources",
            "parameters": {"resources": "self"},
        }
    )

    assert (
        magento._extract_error_detail(response)
        == "Consumer is not authorized to access self"
    )


def test_extract_error_detail_substitutes_named_parameters_with_prefix_collision():
    # Substituting "%field" before "%fieldName" is reached would corrupt
    # "%fieldName" (since "%field" is a prefix of it) -- longer keys must
    # be substituted first regardless of dict iteration order.
    response = MockResponse(
        json_data={
            "message": 'Invalid "%field" value for %fieldName field.',
            "parameters": {"field": "sku", "fieldName": "SKU"},
        }
    )

    assert (
        magento._extract_error_detail(response) == 'Invalid "sku" value for SKU field.'
    )


def test_extract_error_detail_substitutes_positional_parameters():
    response = MockResponse(
        json_data={
            "message": 'Invalid value of "%1" provided for the %2 field.',
            "parameters": ["foo", "sku"],
        }
    )

    assert (
        magento._extract_error_detail(response)
        == 'Invalid value of "foo" provided for the sku field.'
    )


def test_extract_error_detail_substitutes_ten_plus_positional_parameters_correctly():
    # Substituting "%1" before "%10" is reached would corrupt "%10" (since
    # "%1" is a substring of "%10") -- this must substitute high-to-low.
    params = [chr(ord("a") + i) for i in range(10)]
    message = " ".join(f"%{i}" for i in range(1, 11))
    response = MockResponse(json_data={"message": message, "parameters": params})

    assert magento._extract_error_detail(response) == " ".join(params)


def test_extract_error_detail_leaves_out_of_range_positional_placeholder_unchanged():
    response = MockResponse(
        json_data={
            "message": 'Field "%5" is invalid for %1.',
            "parameters": ["foo", "bar"],
        }
    )

    assert magento._extract_error_detail(response) == 'Field "%5" is invalid for foo.'


def test_extract_error_detail_ignores_non_list_non_dict_parameters():
    response = MockResponse(
        json_data={"message": "Invalid value for %1.", "parameters": "not-a-container"}
    )

    assert magento._extract_error_detail(response) == "Invalid value for %1."


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert magento._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_when_message_missing():
    assert magento._extract_error_detail(MockResponse(json_data={"other": 1})) is None


def test_request_uses_configured_host_and_bearer_token(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = magento._request("GET", "/products/abc")

    assert result == {"ok": True}
    assert (
        mock_request.call_args.kwargs["url"]
        == "https://store.example.com/rest/V1/products/abc"
    )
    assert (
        mock_request.call_args.kwargs["headers"]["Authorization"] == "Bearer test-token"
    )
    assert mock_request.call_args.kwargs["allow_redirects"] is False


def test_request_uses_store_code_in_the_real_request_url(monkeypatch):
    # _base_url()/_api_base_url() have their own store-code coverage above,
    # but _request() builds the URL independently (for the single-DNS-
    # lookup/pinning guarantee) -- this asserts the store-code prefix
    # actually reaches a real _request() call, not just the unused helpers.
    monkeypatch.setenv("MAGENTO_STORE_CODE", "default")
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento._request("GET", "/products/abc")

    assert (
        mock_request.call_args.kwargs["url"]
        == "https://store.example.com/rest/default/V1/products/abc"
    )


def test_request_passes_empty_proxies_when_none_configured(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento._request("GET", "/products/abc")

    assert mock_request.call_args.kwargs["proxies"] == {}


def test_request_raises_when_proxy_is_configured_but_not_trusted(monkeypatch):
    # An ambient proxy makes the *proxy* resolve DNS for the real
    # connection, silently defeating _pinned_dns -- this must fail loudly
    # instead of quietly connecting through an unvetted proxy.
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.delenv("XAGENT_TRUSTED_EGRESS_PROXY", raising=False)
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    with pytest.raises(ValueError, match="not marked as trusted"):
        magento._request("GET", "/products/abc")

    mock_request.assert_not_called()


def test_request_passes_proxy_explicitly_when_trusted(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.internal:8080")
    monkeypatch.setenv("XAGENT_TRUSTED_EGRESS_PROXY", "1")
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = magento._request("GET", "/products/abc")

    assert result == {"ok": True}
    assert mock_request.call_args.kwargs["proxies"] == {
        "http": "http://proxy.internal:8080",
        "https": "http://proxy.internal:8080",
    }


def test_make_request_disables_trust_env(monkeypatch):
    # requests.request() always runs on a fresh trust_env=True Session,
    # which -- regardless of what proxies= is passed -- still falls back
    # to get_environ_proxies() -> urllib.request.getproxies(), which
    # itself falls back *past* HTTP_PROXY/HTTPS_PROXY/ALL_PROXY to the
    # OS's own proxy configuration (getproxies_macosx_sysconf() on macOS,
    # getproxies_registry() on Windows) once none of those env vars are
    # set. Scrubbing named env vars closes the env-var vector but not
    # this OS-native fallback layer. trust_env=False skips the entire
    # environment-and-OS-native lookup in one place (verified directly:
    # Session.merge_environment_settings only calls get_environ_proxies()
    # inside `if self.trust_env:`), so assert _make_request actually sets
    # it before issuing the real request, rather than trusting the
    # env-var-scrubbing approach a prior round used instead.
    captured = {}

    class FakeSession:
        def __enter__(self):
            self.trust_env = True
            return self

        def __exit__(self, *exc_info):
            return False

        def request(self, **kwargs):
            captured["trust_env"] = self.trust_env
            return MockResponse(json_data={"ok": True})

    monkeypatch.setattr(magento.requests, "Session", FakeSession)

    response = magento._make_request(
        method="GET",
        url="https://store.example.com/rest/V1/products/abc",
        headers={},
        params=None,
        json=None,
        timeout=1,
        proxies={},
        allow_redirects=False,
    )

    assert captured["trust_env"] is False
    assert response.json() == {"ok": True}


def test_make_request_reapplies_ca_bundle_env_var(monkeypatch):
    # trust_env=False also disables requests' REQUESTS_CA_BUNDLE/
    # CURL_CA_BUNDLE lookup (gated behind the same `if self.trust_env:` as
    # the proxy lookup) -- without re-applying it explicitly, a self-hosted
    # store behind a private/internal CA that relied on this env var would
    # silently start failing TLS verification after the trust_env fix.
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/etc/ssl/internal-ca.pem")
    captured = {}

    class FakeSession:
        def __enter__(self):
            self.trust_env = True
            self.verify = True
            return self

        def __exit__(self, *exc_info):
            return False

        def request(self, **kwargs):
            captured["verify"] = self.verify
            return MockResponse(json_data={"ok": True})

    monkeypatch.setattr(magento.requests, "Session", FakeSession)

    magento._make_request(
        method="GET",
        url="https://store.example.com/rest/V1/products/abc",
        headers={},
        params=None,
        json=None,
        timeout=1,
        proxies={},
        allow_redirects=False,
    )

    assert captured["verify"] == "/etc/ssl/internal-ca.pem"


def test_request_pins_dns_within_a_lock(monkeypatch):
    # _pinned_dns serializes its global urllib3 monkeypatch through
    # _PINNED_DNS_LOCK so two overlapping calls can never race on it --
    # confirm the lock is actually held (not just present/unused) while
    # the block that patches/restores create_connection runs.
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)
    held_during_call = {}

    def _check_locked(*args, **kwargs):
        held_during_call["locked"] = magento._PINNED_DNS_LOCK.locked()
        return mock_request.return_value

    mock_request.side_effect = _check_locked

    magento._request("GET", "/products/abc")

    assert held_during_call["locked"] is True
    assert not magento._PINNED_DNS_LOCK.locked()


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_request_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=status_code, url="https://store.example.com/x"
            )
        ),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        magento._request("GET", "/products/abc")


def test_request_passes_configured_timeout(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento._request("GET", "/products/abc")

    assert mock_request.call_args.kwargs["timeout"] == magento.DEFAULT_TIMEOUT_SECONDS


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert magento._request("DELETE", "/products/abc") == {}


def test_request_redacts_connection_error_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(magento, "_make_request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        magento._request("GET", "/products/abc")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=404,
                json_data={"message": "Requested product doesn't exist"},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="doesn't exist"):
        magento._request("GET", "/products/missing")


def test_request_hints_at_the_bearer_token_setting_on_401(monkeypatch):
    # The single most common cause of a 401 from an otherwise-correctly-
    # configured integration: Magento 2.4.4+ disabled standalone-Bearer
    # usage by default. That's currently only documented in a docstring
    # nobody reading a runtime error sees -- surface it in the error too.
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(status_code=401, text="Unauthorized")),
    )

    with pytest.raises(RuntimeError, match="standalone Bearer tokens"):
        magento._request("GET", "/products/abc")


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        magento._request("GET", "/products/abc")


def test_request_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        magento._request("GET", "/products/abc")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_list_products_sends_sku_like_and_status_filters(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_products(sku_like="shirt", status=1, limit=10, page=2)

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "%shirt%"
    # A separate filter_groups entry, not a second filter in group 0 -- see
    # test_search_criteria_params_ands_filters_across_separate_groups.
    assert params["searchCriteria[filter_groups][1][filters][0][field]"] == "status"
    assert params["searchCriteria[filter_groups][1][filters][0][value]"] == "1"
    assert params["searchCriteria[pageSize]"] == 10
    assert params["searchCriteria[currentPage]"] == 2


def test_list_products_returns_pagination_metadata_at_top_level(monkeypatch):
    # success_with_capped_dict can only cap a collection *nested inside* a
    # dict, so the summarized list is intentionally wrapped one level
    # deeper under `result_key` (result["products"]["products"]) to give
    # it something to shrink -- but has_more/next_page must still land as
    # top-level siblings, not trapped inside that same nested value, or
    # pagination continuation breaks. Assert the actual top-level shape,
    # not just the outbound request params (which every other test in
    # this file checks instead).
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "items": [{"sku": "abc", "name": "Shirt"}],
                    "total_count": 100,
                }
            )
        ),
    )

    result = json.loads(magento.magento_list_products(limit=10, page=1))

    assert result["status"] == "success"
    assert result["products"]["products"] == [
        {
            "sku": "abc",
            "name": "Shirt",
            "price": None,
            "status": None,
            "visibility": None,
            "type_id": None,
            "attribute_set_id": None,
            "created_at": None,
            "updated_at": None,
        }
    ]
    assert result["has_more"] is True
    assert result["next_page"] == 2


def test_list_products_caps_output_when_the_page_is_oversized(monkeypatch):
    # Confirmed bug: passing a bare list straight to success_with_capped_dict
    # short-circuits its `not isinstance(data, dict)` guard, returning the
    # payload uncapped no matter how large -- output capping was silently
    # disabled for every list tool. Force a small limit and a page that
    # exceeds it to prove capping actually engages again.
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 2000)
    items = [{"sku": f"SKU-{i}", "name": "x" * 100, "price": 19.99} for i in range(100)]
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(json_data={"items": items, "total_count": 100})),
    )

    raw = magento.magento_list_products(limit=100, page=1)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert len(result["products"]["products"]) < len(items)
    assert len(raw) <= 2000 + 400  # last halving step can overshoot


def test_list_products_survives_an_aggressively_low_output_limit(monkeypatch):
    # Confirmed bug: under an extremely low XAGENT_TOOL_MAX_OUTPUT_LENGTH,
    # success_with_capped_dict's phase-2 fallback can drop the wrapper's
    # sole key entirely once list-halving alone isn't enough, leaving
    # capped["products"] == {} instead of {"products": []} --
    # result["products"]["products"] then raised KeyError instead of
    # returning an empty list.
    monkeypatch.setattr(mcp_utils, "get_tool_max_output_length", lambda: 30)
    items = [{"sku": f"SKU-{i}", "name": "x" * 50} for i in range(50)]
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(json_data={"items": items, "total_count": 50})),
    )

    result = json.loads(magento.magento_list_products(limit=50, page=1))

    assert result["status"] == "success"
    assert result["products"]["products"] == []


def test_list_products_drops_malformed_items_instead_of_phantom_records(monkeypatch):
    # A malformed item (not a dict) would otherwise summarize to {} via
    # _as_record() and appear as a phantom all-None record indistinguishable
    # from a real product -- it must be dropped instead, matching
    # _category_summary's treatment of a malformed child.
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "items": [{"sku": "abc", "name": "Shirt"}, "not-a-product", None],
                    "total_count": 3,
                }
            )
        ),
    )

    result = json.loads(magento.magento_list_products(limit=10, page=1))

    assert len(result["products"]["products"]) == 1
    assert result["products"]["products"][0]["sku"] == "abc"


def test_list_products_logs_a_warning_when_dropping_malformed_items(
    monkeypatch, caplog
):
    # Dropping a malformed item silently would look identical to "this
    # page just happens to be short" -- a warning is the only signal a
    # non-compliant response occurred, since there's no cursor to retry
    # with once it's dropped.
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "items": [{"sku": "abc", "name": "Shirt"}, "not-a-product"],
                    "total_count": 2,
                }
            )
        ),
    )

    with caplog.at_level("WARNING", logger="magento-mcp"):
        magento.magento_list_products(limit=10, page=1)

    assert any("Dropped 1 malformed" in record.message for record in caplog.records)


def test_list_products_rejects_invalid_status(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_list_products(status=9))

    assert result["status"] == "error"
    mock_request.assert_not_called()


@pytest.mark.parametrize("value", [True, False])
def test_validate_choice_rejects_bool(value):
    # bool is an int subclass in Python, so `True in {1, 2}` compares equal
    # to 1 and would silently pass -- the same gap _paginated_result's
    # total_count check was hardened against.
    assert magento._validate_choice(value, magento._PRODUCT_STATUSES, "status")


def test_list_products_rejects_bool_status(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_list_products(status=True))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_list_products_escapes_like_wildcards_in_sku_like(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_products(sku_like="ABC_1%off")

    params = mock_request.call_args.kwargs["params"]
    assert (
        params["searchCriteria[filter_groups][0][filters][0][value]"]
        == "%ABC\\_1\\%off%"
    )


def test_list_products_escapes_backslash_percent_and_underscore_together(
    monkeypatch,
):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_products(sku_like="a\\b_c%d")

    params = mock_request.call_args.kwargs["params"]
    assert (
        params["searchCriteria[filter_groups][0][filters][0][value]"]
        == "%a\\\\b\\_c\\%d%"
    )


def test_get_product_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"sku": "abc", "name": "Shirt", "price": 19.99, "status": 1}
            )
        ),
    )

    result = json.loads(magento.magento_get_product("abc"))

    assert result["status"] == "success"
    assert result["product"]["name"] == "Shirt"


def test_get_product_requires_non_blank_sku(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_get_product("  "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_get_product_percent_encodes_sku_containing_slash(monkeypatch):
    # A SKU containing "/" (legitimate for some configurable-product
    # variants) must not be split into an extra path segment or let a
    # crafted value like "../orders/5" redirect the request to a different
    # resource entirely.
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "SHIRT-RED/M"}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_get_product("SHIRT-RED/M")

    assert mock_request.call_args.kwargs["url"].endswith("/products/SHIRT-RED%2FM")


def test_get_product_rejects_dot_dot_sku():
    result = json.loads(magento.magento_get_product(".."))

    assert result["status"] == "error"


def test_create_product_rejects_invalid_status(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(
        magento.magento_create_product("sku1", "Shirt", 19.99, status=9)
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_product_rejects_invalid_visibility(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(
        magento.magento_create_product("sku1", "Shirt", 19.99, visibility=9)
    )

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_product_rejects_whitespace_padded_sku(monkeypatch):
    # magento_get_product/magento_update_product reject a whitespace-padded
    # sku (via url_path_id) -- create must reject it too, or a caller could
    # create a product with a sku it can never look back up with the same
    # string.
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_create_product(" sku1 ", "Shirt", 19.99))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_product_sends_expected_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"sku": "sku1", "name": "Shirt"})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_create_product("sku1", "Shirt", 19.99))

    assert result["status"] == "success"
    body = mock_request.call_args.kwargs["json"]["product"]
    assert body == {
        "sku": "sku1",
        "name": "Shirt",
        "price": 19.99,
        "attribute_set_id": 4,
        "type_id": "simple",
        "status": 1,
        "visibility": 4,
    }
    assert mock_request.call_args.kwargs["method"] == "POST"
    assert mock_request.call_args.kwargs["url"].endswith("/products")


def test_create_product_strips_padded_name(monkeypatch):
    # Unlike sku (an identifier, rejected if padded via require_clean_identifier),
    # name is a free-text display field -- incidental padding should be
    # stripped rather than silently stored, matching _require_non_blank's
    # contract.
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "sku1"}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_create_product("sku1", "  Shirt  ", 19.99)

    assert mock_request.call_args.kwargs["json"]["product"]["name"] == "Shirt"


def test_create_product_rejects_negative_price(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_create_product("sku1", "Shirt", -1))

    assert result["status"] == "error"
    mock_request.assert_not_called()


@pytest.mark.parametrize("value", [True, False, float("nan"), float("inf")])
def test_validate_non_negative_rejects_bool_and_non_finite(value):
    # bool is an int subclass in Python (False/True < 0 are both False),
    # and NaN/inf also silently pass `value < 0` -- the same gap
    # _validate_choice/_paginated_result's total_count check were
    # hardened against, missed when this helper was first extracted.
    assert magento._validate_non_negative(value, "price")


def test_update_product_requires_at_least_one_field(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_update_product("sku1"))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_product_sends_only_provided_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"sku": "sku1", "price": 29.99})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_update_product("sku1", price=29.99)

    body = mock_request.call_args.kwargs["json"]["product"]
    assert body == {"sku": "sku1", "price": 29.99}
    assert mock_request.call_args.kwargs["url"].endswith("/products/sku1")
    assert mock_request.call_args.kwargs["method"] == "PUT"


def test_update_product_strips_padded_name(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "sku1"}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_update_product("sku1", name="  Shirt  ")

    assert mock_request.call_args.kwargs["json"]["product"]["name"] == "Shirt"


def test_update_product_rejects_blank_name(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_update_product("sku1", name="   "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_product_rejects_negative_price(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_update_product("sku1", price=-1))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_update_product_percent_encodes_sku_in_path(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "SHIRT-RED/M"}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_update_product("SHIRT-RED/M", price=29.99)

    assert mock_request.call_args.kwargs["url"].endswith("/products/SHIRT-RED%2FM")
    # The request *body*'s sku field stays unencoded -- only the URL path
    # segment needs escaping.
    assert mock_request.call_args.kwargs["json"]["product"]["sku"] == "SHIRT-RED/M"


def test_list_orders_sends_status_filter_and_sort(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_orders(status="processing")

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "processing"
    assert params["searchCriteria[sortOrders][0][field]"] == "created_at"
    assert params["searchCriteria[sortOrders][0][direction]"] == "DESC"


def test_list_orders_sends_increment_id_filter(monkeypatch):
    # The only way to look up an order by the customer-facing number
    # (e.g. "000000123") -- magento_get_order requires the internal
    # entity id instead.
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_orders(increment_id="000000123")

    params = mock_request.call_args.kwargs["params"]
    assert (
        params["searchCriteria[filter_groups][0][filters][0][field]"] == "increment_id"
    )
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "000000123"


def test_list_orders_sends_status_and_increment_id_filters_as_separate_groups(
    monkeypatch,
):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_orders(status="processing", increment_id="000000123")

    params = mock_request.call_args.kwargs["params"]
    # Separate filter_groups entries (ANDed), not a second filter in group 0
    # (which Magento would OR together instead).
    assert params["searchCriteria[filter_groups][0][filters][0][field]"] == "status"
    assert (
        params["searchCriteria[filter_groups][1][filters][0][field]"] == "increment_id"
    )


def test_list_orders_returns_pagination_metadata_at_top_level(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"items": [{"entity_id": 1}], "total_count": 1}
            )
        ),
    )

    result = json.loads(magento.magento_list_orders())

    assert isinstance(result["orders"]["orders"], list)
    assert result["orders"]["orders"][0]["entity_id"] == 1
    assert result["has_more"] is False
    assert "next_page" in result


def test_get_order_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={
                    "entity_id": 1,
                    "increment_id": "000000001",
                    "status": "processing",
                    "grand_total": 42.5,
                    "order_currency_code": "USD",
                }
            )
        ),
    )

    result = json.loads(magento.magento_get_order(1))

    assert result["status"] == "success"
    assert result["order"]["increment_id"] == "000000001"
    assert result["order"]["currency"] == "USD"


def test_get_order_builds_the_expected_path(monkeypatch):
    # Renamed from "..._percent_encodes_id_in_path": a plain "1" has
    # nothing to encode, so this only covers the happy-path URL shape --
    # actual percent-encoding is exercised by
    # test_get_order_neutralizes_path_traversal_id below.
    mock_request = Mock(return_value=MockResponse(json_data={"entity_id": 1}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_get_order("1")

    assert mock_request.call_args.kwargs["url"].endswith("/orders/1")


async def test_get_order_rejects_non_numeric_id_at_the_wire_boundary(monkeypatch):
    # The test below calls magento_get_order(...) directly with a string,
    # bypassing FastMCP's own pydantic validation entirely -- this is the
    # one test in the group that actually proves that path is unreachable
    # through the real MCP interface: a real tool call arrives as JSON and
    # is validated against order_id: int before the function body runs.
    from mcp.server.fastmcp.exceptions import ToolError

    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    with pytest.raises(ToolError, match="validation error"):
        await magento.mcp.call_tool(
            "magento_get_order", {"order_id": "1/../../customers/5"}
        )
    mock_request.assert_not_called()


def test_get_order_neutralizes_path_traversal_id(monkeypatch):
    # order_id is typed int, so a real MCP call can never reach this value
    # (FastMCP validates the wire argument first, confirmed by the test
    # above) -- this is a defense-in-depth check on the underlying
    # function, matching the protection magento_get_product's sku already
    # has. url_path_id() percent-encodes rather than rejects a value
    # merely containing "/", so this must not hit a real network call to
    # prove anything: mock the request and assert the traversal segment
    # landed encoded in the path, never as a raw path separator.
    mock_request = Mock(return_value=MockResponse(json_data={"entity_id": 1}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_get_order("1/../../customers/5"))

    assert result["status"] == "success"
    url = mock_request.call_args.kwargs["url"]
    assert url.endswith("/orders/1%2F..%2F..%2Fcustomers%2F5")


def test_add_order_comment_requires_non_blank_comment(monkeypatch):
    mock_request = Mock()
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_add_order_comment(1, "   "))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_add_order_comment_strips_padded_comment(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_add_order_comment(1, "  Shipped today  ")

    body = mock_request.call_args.kwargs["json"]["statusHistory"]
    assert body["comment"] == "Shipped today"


def test_add_order_comment_sends_status_history(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(
        magento.magento_add_order_comment(
            1,
            "Shipped today",
            status="complete",
            notify_customer=True,
            visible_to_customer=True,
        )
    )

    assert result["status"] == "success"
    assert result["added"] is True
    body = mock_request.call_args.kwargs["json"]["statusHistory"]
    assert body == {
        "comment": "Shipped today",
        "is_customer_notified": True,
        "is_visible_on_front": True,
        "status": "complete",
    }
    assert mock_request.call_args.kwargs["url"].endswith("/orders/1/comments")


def test_add_order_comment_neutralizes_path_traversal_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(
        magento.magento_add_order_comment("1/../../customers/5", "Note")
    )

    assert result["status"] == "success"
    url = mock_request.call_args.kwargs["url"]
    assert url.endswith("/orders/1%2F..%2F..%2Fcustomers%2F5/comments")


def test_add_order_comment_omits_status_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_add_order_comment(1, "Internal note")

    body = mock_request.call_args.kwargs["json"]["statusHistory"]
    assert "status" not in body
    assert body["is_customer_notified"] is False
    # visible_to_customer defaults to False: an agent adding a note without
    # explicitly opting in shouldn't publish it to the customer by default.
    assert body["is_visible_on_front"] is False


def test_add_order_comment_reports_added_false_on_falsy_result(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(json_data=False)),
    )

    result = json.loads(magento.magento_add_order_comment(1, "Note"))

    assert result["status"] == "success"
    assert result["added"] is False


def test_add_order_comment_reports_added_true_on_empty_body_response(monkeypatch):
    # _request()'s generic "204 or empty content" shortcut returns {} --
    # bool({}) is False, so a legitimate empty-body success response (a
    # non-compliant fork/gateway, or a 204) would previously be reported
    # as added=False, indistinguishable from Magento explicitly returning
    # JSON `false`. Only an explicit `false` should mean failure.
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(status_code=204, content=b"")),
    )

    result = json.loads(magento.magento_add_order_comment(1, "Note"))

    assert result["status"] == "success"
    assert result["added"] is True


@pytest.mark.parametrize("value", [None, 0, "", []])
def test_add_order_comment_surfaces_unexpected_response_shapes(monkeypatch, value):
    # `result is not False` would treat any of these non-compliant values
    # (a fork/gateway deviating from Magento's "always a JSON bool"
    # contract) as success, silently hiding an actual failure -- only an
    # explicit true/false/{} (the 204/empty-body case) are legitimate.
    monkeypatch.setattr(
        magento, "_make_request", Mock(return_value=MockResponse(json_data=value))
    )

    result = json.loads(magento.magento_add_order_comment(1, "Note"))

    assert result["status"] == "error"
    assert "unexpected" in result["message"]


def test_list_customers_sends_email_like_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_customers(email_like="jane")

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "%jane%"
    assert mock_request.call_args.kwargs["url"].endswith("/customers/search")


def test_list_customers_returns_pagination_metadata_at_top_level(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"items": [{"id": 5}], "total_count": 1}
            )
        ),
    )

    result = json.loads(magento.magento_list_customers())

    assert isinstance(result["customers"]["customers"], list)
    assert result["customers"]["customers"][0]["id"] == 5
    assert result["has_more"] is False
    assert "next_page" in result


def test_list_customers_escapes_like_wildcards_in_email_like(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    magento.magento_list_customers(email_like="j_ane%")

    params = mock_request.call_args.kwargs["params"]
    assert (
        params["searchCriteria[filter_groups][0][filters][0][value]"] == "%j\\_ane\\%%"
    )


def test_get_customer_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"id": 5, "email": "jane@example.com", "firstname": "Jane"}
            )
        ),
    )

    result = json.loads(magento.magento_get_customer(5))

    assert result["status"] == "success"
    assert result["customer"]["email"] == "jane@example.com"


def test_get_customer_neutralizes_path_traversal_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_get_customer("1/../../orders/5"))

    assert result["status"] == "success"
    url = mock_request.call_args.kwargs["url"]
    assert url.endswith("/customers/1%2F..%2F..%2Forders%2F5")


def test_get_category_tree_sends_optional_params(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(
            json_data={
                "id": 2,
                "name": "Root",
                "children_data": [{"id": 3, "name": "Shirts", "children_data": []}],
            }
        )
    )
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_get_category_tree(root_category_id=2, depth=2))

    assert result["status"] == "success"
    assert result["category"]["name"] == "Root"
    assert result["category"]["children_data"][0]["name"] == "Shirts"
    assert mock_request.call_args.kwargs["params"] == {"rootCategoryId": 2, "depth": 2}


def test_get_category_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(
            return_value=MockResponse(
                json_data={"id": 3, "name": "Shirts", "children_data": []}
            )
        ),
    )

    result = json.loads(magento.magento_get_category(3))

    assert result["status"] == "success"
    assert result["category"]["name"] == "Shirts"


def test_get_category_neutralizes_path_traversal_id(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"id": 1, "name": "Root"}))
    monkeypatch.setattr(magento, "_make_request", mock_request)

    result = json.loads(magento.magento_get_category("1/../../orders/5"))

    assert result["status"] == "success"
    url = mock_request.call_args.kwargs["url"]
    assert url.endswith("/categories/1%2F..%2F..%2Forders%2F5")


_ALL_SUMMARY_FUNCTIONS = [
    magento._product_summary,
    magento._order_summary,
    magento._customer_summary,
    magento._category_summary,
]


@pytest.mark.parametrize("summary_fn", _ALL_SUMMARY_FUNCTIONS)
def test_summary_functions_return_empty_dict_for_none(summary_fn):
    # A malformed 2xx body (JSON null, or an array where an object is
    # expected) would otherwise raise AttributeError on .get(), surfaced
    # by each tool's blanket "except Exception" as an opaque "'NoneType'
    # object has no attribute 'get'" -- all four summary functions share
    # this guard via _as_record().
    assert summary_fn(None) == {}


@pytest.mark.parametrize("summary_fn", _ALL_SUMMARY_FUNCTIONS)
def test_summary_functions_return_empty_dict_for_non_dict(summary_fn):
    assert summary_fn("not-a-record") == {}


def test_category_summary_filters_out_falsy_children():
    # None and {} entries in children_data are dropped outright (skipped
    # before recursing), not turned into an empty-dict placeholder.
    summary = magento._category_summary(
        {
            "id": 1,
            "name": "Root",
            "children_data": [None, {"id": 2, "name": "Child"}, {}],
        }
    )

    assert summary["name"] == "Root"
    assert [child.get("name") for child in summary["children_data"]] == ["Child"]


def test_category_summary_filters_out_truthy_non_dict_children():
    # A truthy-but-malformed child (a stray string/int from a non-compliant
    # proxy, not just None/{}) must be dropped too, not turned into a
    # phantom empty-dict category -- `if child` alone wouldn't catch this
    # since a non-empty string is truthy.
    summary = magento._category_summary(
        {
            "id": 1,
            "name": "Root",
            "children_data": ["not-a-category", {"id": 2, "name": "Child"}, 42],
        }
    )

    assert [child.get("name") for child in summary["children_data"]] == ["Child"]


@pytest.mark.parametrize("children_data_value", ["not-a-list", {"0": {"id": 2}}, 42])
def test_category_summary_rejects_non_list_children_data(children_data_value):
    # A truthy non-list children_data (e.g. a non-compliant proxy
    # re-encoding a sparse PHP array as a JSON object) must not be
    # silently iterated -- that would walk characters/dict keys instead
    # of category records, each dropped with no error, silently reporting
    # "no children" for a category that actually has some.
    with pytest.raises(ValueError, match="children_data"):
        magento._category_summary(
            {"id": 1, "name": "Root", "children_data": children_data_value}
        )


def _make_nested_category(depth: int) -> dict:
    node: dict = {"id": depth, "name": f"Level {depth}", "children_data": []}
    for level in range(depth - 1, -1, -1):
        node = {"id": level, "name": f"Level {level}", "children_data": [node]}
    return node


def test_category_summary_accepts_a_deeply_nested_but_bounded_tree():
    # Comfortably below the safety cap -- a real, if unusually deep,
    # category tree must not be rejected.
    tree = _make_nested_category(depth=10)

    summary = magento._category_summary(tree)

    # Walk down to confirm the whole chain summarized correctly.
    node = summary
    for level in range(10):
        assert node["name"] == f"Level {level}"
        node = node["children_data"][0]


def test_category_summary_rejects_a_pathologically_deep_tree():
    # A pathological or cyclic children_data would otherwise only ever be
    # stopped by an incidental RecursionError -- assert the explicit,
    # documented cap raises a clear error instead.
    tree = _make_nested_category(depth=magento._MAX_CATEGORY_SUMMARY_DEPTH + 5)

    with pytest.raises(ValueError, match="maximum supported depth"):
        magento._category_summary(tree)


def test_get_category_handles_null_category_response(monkeypatch):
    monkeypatch.setattr(
        magento,
        "_make_request",
        Mock(return_value=MockResponse(json_data=None)),
    )

    result = json.loads(magento.magento_get_category(3))

    assert result["status"] == "success"
    assert result["category"] == {}


def test_magento_app_registry_requires_base_url_and_token():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    magento_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "magento"
    )
    assert magento_app["provider_name"] is None
    assert magento_app["category"] == "Commerce"
    assert magento_app["transport"] == "stdio"
    assert magento_app["launch_config"]["required_env"] == [
        "MAGENTO_BASE_URL",
        "MAGENTO_ACCESS_TOKEN",
        "MAGENTO_STORE_CODE",
    ]
