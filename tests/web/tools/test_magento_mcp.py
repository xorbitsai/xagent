import json
import socket
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import magento


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


class MockResponse:
    def __init__(
        self,
        json_data=None,
        status_code: int = 200,
        text: str = "",
        url: str = "",
        json_raises: bool = False,
        content: bytes | None = None,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
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


@pytest.mark.parametrize(
    "value, match",
    [
        ("http://store.example.com", "https"),
        ("user:pass@store.example.com", "credentials"),
        ("store.example.com/rest", "path"),
        ("store.example.com?x=1", "path"),
        ("https://", "hostname"),
    ],
)
def test_base_url_rejects_invalid_url(monkeypatch, value, match):
    monkeypatch.setenv("MAGENTO_BASE_URL", value)

    with pytest.raises(ValueError, match=match):
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


def test_extract_error_detail_returns_none_for_non_json_body():
    response = MockResponse(status_code=500, text="not json", json_raises=True)

    assert magento._extract_error_detail(response) is None


def test_extract_error_detail_returns_none_when_message_missing():
    assert magento._extract_error_detail(MockResponse(json_data={"other": 1})) is None


def test_request_uses_configured_host_and_bearer_token(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"ok": True}))
    monkeypatch.setattr(magento.requests, "request", mock_request)

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


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_request_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        magento.requests,
        "request",
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
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento._request("GET", "/products/abc")

    assert mock_request.call_args.kwargs["timeout"] == magento.DEFAULT_TIMEOUT_SECONDS


def test_request_returns_empty_dict_for_204(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(return_value=MockResponse(status_code=204, text="")),
    )

    assert magento._request("DELETE", "/products/abc") == {}


def test_request_redacts_connection_error_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(magento.requests, "request", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        magento._request("GET", "/products/abc")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_request_raises_with_structured_error_detail(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(
            return_value=MockResponse(
                status_code=404,
                json_data={"message": "Requested product doesn't exist"},
            )
        ),
    )

    with pytest.raises(RuntimeError, match="doesn't exist"):
        magento._request("GET", "/products/missing")


def test_request_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
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
        magento.requests,
        "request",
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
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_list_products(sku_like="shirt", status=1, limit=10, page=2)

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "%shirt%"
    # A separate filter_groups entry, not a second filter in group 0 -- see
    # test_search_criteria_params_ands_filters_across_separate_groups.
    assert params["searchCriteria[filter_groups][1][filters][0][field]"] == "status"
    assert params["searchCriteria[filter_groups][1][filters][0][value]"] == "1"
    assert params["searchCriteria[pageSize]"] == 10
    assert params["searchCriteria[currentPage]"] == 2


def test_list_products_rejects_invalid_status():
    result = json.loads(magento.magento_list_products(status=9))

    assert result["status"] == "error"


def test_get_product_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"sku": "abc", "name": "Shirt", "price": 19.99, "status": 1}
            )
        ),
    )

    result = json.loads(magento.magento_get_product("abc"))

    assert result["status"] == "success"
    assert result["product"]["name"] == "Shirt"


def test_get_product_requires_non_blank_sku():
    result = json.loads(magento.magento_get_product("  "))

    assert result["status"] == "error"


def test_get_product_percent_encodes_sku_containing_slash(monkeypatch):
    # A SKU containing "/" (legitimate for some configurable-product
    # variants) must not be split into an extra path segment or let a
    # crafted value like "../orders/5" redirect the request to a different
    # resource entirely.
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "SHIRT-RED/M"}))
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_get_product("SHIRT-RED/M")

    assert mock_request.call_args.kwargs["url"].endswith("/products/SHIRT-RED%2FM")


def test_get_product_rejects_dot_dot_sku():
    result = json.loads(magento.magento_get_product(".."))

    assert result["status"] == "error"


def test_create_product_rejects_invalid_status():
    result = json.loads(
        magento.magento_create_product("sku1", "Shirt", 19.99, status=9)
    )

    assert result["status"] == "error"


def test_create_product_rejects_invalid_visibility():
    result = json.loads(
        magento.magento_create_product("sku1", "Shirt", 19.99, visibility=9)
    )

    assert result["status"] == "error"


def test_create_product_rejects_whitespace_padded_sku(monkeypatch):
    # magento_get_product/magento_update_product reject a whitespace-padded
    # sku (via url_path_id) -- create must reject it too, or a caller could
    # create a product with a sku it can never look back up with the same
    # string.
    mock_request = Mock()
    monkeypatch.setattr(magento.requests, "request", mock_request)

    result = json.loads(magento.magento_create_product(" sku1 ", "Shirt", 19.99))

    assert result["status"] == "error"
    mock_request.assert_not_called()


def test_create_product_sends_expected_body(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"sku": "sku1", "name": "Shirt"})
    )
    monkeypatch.setattr(magento.requests, "request", mock_request)

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


def test_update_product_requires_at_least_one_field():
    result = json.loads(magento.magento_update_product("sku1"))

    assert result["status"] == "error"


def test_update_product_sends_only_provided_fields(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"sku": "sku1", "price": 29.99})
    )
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_update_product("sku1", price=29.99)

    body = mock_request.call_args.kwargs["json"]["product"]
    assert body == {"sku": "sku1", "price": 29.99}
    assert mock_request.call_args.kwargs["url"].endswith("/products/sku1")
    assert mock_request.call_args.kwargs["method"] == "PUT"


def test_update_product_percent_encodes_sku_in_path(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data={"sku": "SHIRT-RED/M"}))
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_update_product("SHIRT-RED/M", price=29.99)

    assert mock_request.call_args.kwargs["url"].endswith("/products/SHIRT-RED%2FM")
    # The request *body*'s sku field stays unencoded -- only the URL path
    # segment needs escaping.
    assert mock_request.call_args.kwargs["json"]["product"]["sku"] == "SHIRT-RED/M"


def test_list_orders_sends_status_filter_and_sort(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_list_orders(status="processing")

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "processing"
    assert params["searchCriteria[sortOrders][0][field]"] == "created_at"
    assert params["searchCriteria[sortOrders][0][direction]"] == "DESC"


def test_get_order_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
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


def test_add_order_comment_requires_non_blank_comment():
    result = json.loads(magento.magento_add_order_comment(1, "   "))

    assert result["status"] == "error"


def test_add_order_comment_sends_status_history(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento.requests, "request", mock_request)

    result = json.loads(
        magento.magento_add_order_comment(
            1, "Shipped today", status="complete", notify_customer=True
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


def test_add_order_comment_omits_status_when_not_provided(monkeypatch):
    mock_request = Mock(return_value=MockResponse(json_data=True))
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_add_order_comment(1, "Internal note")

    body = mock_request.call_args.kwargs["json"]["statusHistory"]
    assert "status" not in body
    assert body["is_customer_notified"] is False
    assert body["is_visible_on_front"] is True


def test_add_order_comment_reports_added_false_on_falsy_result(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(return_value=MockResponse(json_data=False)),
    )

    result = json.loads(magento.magento_add_order_comment(1, "Note"))

    assert result["status"] == "success"
    assert result["added"] is False


def test_list_customers_sends_email_like_filter(monkeypatch):
    mock_request = Mock(
        return_value=MockResponse(json_data={"items": [], "total_count": 0})
    )
    monkeypatch.setattr(magento.requests, "request", mock_request)

    magento.magento_list_customers(email_like="jane")

    params = mock_request.call_args.kwargs["params"]
    assert params["searchCriteria[filter_groups][0][filters][0][value]"] == "%jane%"
    assert mock_request.call_args.kwargs["url"].endswith("/customers/search")


def test_get_customer_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"id": 5, "email": "jane@example.com", "firstname": "Jane"}
            )
        ),
    )

    result = json.loads(magento.magento_get_customer(5))

    assert result["status"] == "success"
    assert result["customer"]["email"] == "jane@example.com"


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
    monkeypatch.setattr(magento.requests, "request", mock_request)

    result = json.loads(magento.magento_get_category_tree(root_category_id=2, depth=2))

    assert result["status"] == "success"
    assert result["category"]["name"] == "Root"
    assert result["category"]["children_data"][0]["name"] == "Shirts"
    assert mock_request.call_args.kwargs["params"] == {"rootCategoryId": 2, "depth": 2}


def test_get_category_returns_summary(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
        Mock(
            return_value=MockResponse(
                json_data={"id": 3, "name": "Shirts", "children_data": []}
            )
        ),
    )

    result = json.loads(magento.magento_get_category(3))

    assert result["status"] == "success"
    assert result["category"]["name"] == "Shirts"


def test_category_summary_returns_empty_dict_for_none():
    assert magento._category_summary(None) == {}


def test_category_summary_returns_empty_dict_for_non_dict():
    assert magento._category_summary("not-a-category") == {}


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


def test_get_category_handles_null_category_response(monkeypatch):
    monkeypatch.setattr(
        magento.requests,
        "request",
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
    ]
