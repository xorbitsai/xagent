import json
import socket
from unittest.mock import Mock

import pytest
import requests

from xagent.web.tools.mcp import shopify


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
        headers: dict | None = None,
        json_raises: bool = False,
    ):
        self._json_data = json_data if json_data is not None else {}
        self._json_raises = json_raises
        self.status_code = status_code
        self.text = text or (json.dumps(self._json_data) if json_data else "")
        self.headers = headers or {}

    def json(self):
        if self._json_raises:
            raise json.JSONDecodeError("Expecting value", self.text, 0)
        return self._json_data


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("SHOPIFY_STORE_DOMAIN", "acme")
    monkeypatch.setenv("SHOPIFY_ACCESS_TOKEN", "shpat_test_token")
    # _graphql_url() resolves DNS to catch a hostname that rebinds to a
    # private address; tests must not depend on real network/DNS, so every
    # test gets a fake resolver returning an unambiguously public IP.
    monkeypatch.setattr(shopify.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1"))


def test_headers_require_access_token(monkeypatch):
    monkeypatch.delenv("SHOPIFY_ACCESS_TOKEN")

    with pytest.raises(ValueError, match="SHOPIFY_ACCESS_TOKEN"):
        shopify._headers()


def test_headers_include_access_token_and_json_content_type():
    assert shopify._headers() == {
        "X-Shopify-Access-Token": "shpat_test_token",
        "Content-Type": "application/json",
    }


def test_graphql_url_requires_store_domain(monkeypatch):
    monkeypatch.delenv("SHOPIFY_STORE_DOMAIN")

    with pytest.raises(ValueError, match="SHOPIFY_STORE_DOMAIN"):
        shopify._graphql_url()


def test_graphql_url_builds_from_store_domain():
    assert shopify._graphql_url() == (
        f"https://acme.myshopify.com/admin/api/{shopify.SHOPIFY_API_VERSION}/graphql.json"
    )


def test_graphql_url_lowercases_store_domain(monkeypatch):
    monkeypatch.setenv("SHOPIFY_STORE_DOMAIN", "ACME")

    assert shopify._graphql_url().startswith("https://acme.myshopify.com/")


@pytest.mark.parametrize(
    "subdomain",
    [
        "",
        "   ",
        "acme.evil.com",
        "acme/../evil",
        "https://acme",
        "acme:8080",
        "-acme",
        "acme-",
        "acme evil",
    ],
)
def test_graphql_url_rejects_invalid_store_domain(monkeypatch, subdomain):
    monkeypatch.setenv("SHOPIFY_STORE_DOMAIN", subdomain)

    with pytest.raises(ValueError, match="SHOPIFY_STORE_DOMAIN"):
        shopify._graphql_url()


def test_graphql_url_rejects_store_domain_resolving_to_private_ip(monkeypatch):
    monkeypatch.setattr(shopify.socket, "getaddrinfo", _fake_getaddrinfo("10.0.0.5"))

    with pytest.raises(ValueError, match="not allowed"):
        shopify._graphql_url()


def test_graphql_url_rejects_when_any_resolved_address_is_private(monkeypatch):
    monkeypatch.setattr(
        shopify.socket, "getaddrinfo", _fake_getaddrinfo("1.1.1.1", "10.0.0.5")
    )

    with pytest.raises(ValueError, match="not allowed"):
        shopify._graphql_url()


def test_graphql_url_raises_when_dns_resolution_fails(monkeypatch):
    def _raise(*args, **kwargs):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(shopify.socket, "getaddrinfo", _raise)

    with pytest.raises(ValueError, match="could not be resolved"):
        shopify._graphql_url()


@pytest.mark.parametrize("value", ["", "   ", None])
def test_require_non_blank_rejects_empty_values(value):
    with pytest.raises(ValueError, match="field"):
        shopify._require_non_blank(value, "field")


@pytest.mark.parametrize(
    "value, expected",
    [
        ("123", "gid://shopify/Product/123"),
        (123, "gid://shopify/Product/123"),
        ("gid://shopify/Product/123", "gid://shopify/Product/123"),
        ("  123  ", "gid://shopify/Product/123"),
    ],
)
def test_gid_normalizes_numeric_and_passes_through_full_gid(value, expected):
    assert shopify._gid("Product", value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "gid://shopify/Order/123",
        "12.5",
        "١٢٣",
        # A full gid with non-ASCII (Arabic-Indic) digits must be rejected
        # the same way a bare non-ASCII numeric id is -- \\d in a str
        # pattern matches any Unicode decimal digit, not just ASCII ones.
        "gid://shopify/Product/١٢٣",
    ],
)
def test_gid_rejects_invalid_or_mismatched_values(value):
    with pytest.raises(ValueError, match="product_id"):
        shopify._gid("Product", value)


def test_user_errors_message_joins_field_path_and_message():
    message = shopify._user_errors_message(
        [{"field": ["product", "title"], "message": "can't be blank"}]
    )

    assert message == "product.title: can't be blank"


def test_user_errors_message_handles_integer_array_index_in_field_path():
    # Shopify's field path can mix strings with integer array indices for a
    # list-input error (e.g. a variant's price).
    message = shopify._user_errors_message(
        [{"field": ["variants", 0, "price"], "message": "must be positive"}]
    )

    assert message == "variants.0.price: must be positive"


def test_user_errors_message_handles_missing_field():
    message = shopify._user_errors_message([{"field": None, "message": "failed"}])

    assert message == "failed"


def test_user_errors_message_defaults_when_empty():
    assert shopify._user_errors_message([]) == "Shopify reported a validation error"


def test_run_mutation_folds_top_level_errors_into_user_errors_message(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "productCreate": {
                            "product": None,
                            "userErrors": [{"field": ["title"], "message": "too long"}],
                        }
                    },
                    "errors": [{"message": "a sub-field failed"}],
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_create_product("Shirt"))

    assert result["status"] == "error"
    assert result["message"] == "title: too long (a sub-field failed)"


def test_split_tags_strips_and_drops_empty_entries():
    assert shopify._split_tags("vip, , priority,  ") == ["vip", "priority"]


def test_throttle_wait_seconds_true_for_429():
    assert shopify._throttle_wait_seconds(MockResponse(status_code=429)) is not None


def test_throttle_wait_seconds_true_for_200_with_throttled_message():
    response = MockResponse(json_data={"errors": [{"message": "Throttled"}]})

    assert shopify._throttle_wait_seconds(response) is not None


def test_throttle_wait_seconds_true_for_200_with_throttled_extension_code():
    response = MockResponse(
        json_data={"errors": [{"message": "x", "extensions": {"code": "THROTTLED"}}]}
    )

    assert shopify._throttle_wait_seconds(response) is not None


def test_throttle_wait_seconds_none_for_normal_success():
    assert (
        shopify._throttle_wait_seconds(MockResponse(json_data={"data": {"ok": True}}))
        is None
    )


def test_throttle_wait_seconds_ignores_non_dict_extensions():
    # A malformed error entry (extensions present but not a dict) must not
    # crash the throttle check.
    response = MockResponse(
        json_data={"errors": [{"message": "x", "extensions": ["not", "a", "dict"]}]}
    )

    assert shopify._throttle_wait_seconds(response) is None


def test_throttle_wait_seconds_computes_from_cost_extension():
    response = MockResponse(
        status_code=429,
        json_data={
            "extensions": {
                "cost": {
                    "requestedQueryCost": 200,
                    "throttleStatus": {"currentlyAvailable": 100, "restoreRate": 50},
                }
            }
        },
    )

    assert shopify._throttle_wait_seconds(response) == 2.0


def test_throttle_wait_seconds_falls_back_to_one_second_without_cost_data():
    assert shopify._throttle_wait_seconds(MockResponse(status_code=429)) == 1.0


def test_graphql_sends_query_and_variables_as_json_body(monkeypatch):
    mock_post = Mock(return_value=MockResponse(json_data={"data": {"ok": True}}))
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    data, errors = shopify._graphql("query { ok }", {"a": 1})

    assert data == {"ok": True}
    assert errors == []
    assert mock_post.call_args.args[0] == shopify._graphql_url()
    assert mock_post.call_args.kwargs["json"] == {
        "query": "query { ok }",
        "variables": {"a": 1},
    }
    assert mock_post.call_args.kwargs["headers"]["X-Shopify-Access-Token"] == (
        "shpat_test_token"
    )


def test_graphql_raises_on_top_level_errors_with_null_data(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"product": None},
                    "errors": [{"message": "not found"}],
                }
            )
        ),
    )

    with pytest.raises(RuntimeError, match="not found"):
        shopify._graphql('query { product(id: "x") { id } }')


def test_graphql_returns_warnings_for_partial_success(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"product": {"id": "gid://shopify/Product/1"}},
                    "errors": [{"message": "a sub-field failed"}],
                }
            )
        ),
    )

    data, errors = shopify._graphql('query { product(id: "x") { id } }')

    assert data == {"product": {"id": "gid://shopify/Product/1"}}
    assert errors == [{"message": "a sub-field failed"}]


def test_graphql_raises_with_structured_error_body_on_http_error(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=400, json_data={"errors": [{"message": "Invalid token"}]}
            )
        ),
    )

    with pytest.raises(RuntimeError, match="Invalid token"):
        shopify._graphql("query { shop { id } }")


def test_graphql_raises_with_plain_string_error_body_on_http_error(monkeypatch):
    # Shopify's own auth-failure responses (e.g. an invalid access token)
    # put a plain string in "errors", not the GraphQL {"errors": [{...}]}
    # shape -- this must not be iterated character-by-character.
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=401,
                json_data={
                    "errors": "[API] Invalid API key or access token "
                    "(unrecognized login or wrong password)"
                },
            )
        ),
    )

    with pytest.raises(RuntimeError) as excinfo:
        shopify._graphql("query { shop { id } }")

    message = str(excinfo.value)
    assert "Invalid API key or access token" in message
    assert "A; P; I" not in message


def test_graphql_truncates_unstructured_error_body(monkeypatch):
    long_body = "x" * 5000
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(status_code=500, text=long_body)),
    )

    with pytest.raises(RuntimeError) as excinfo:
        shopify._graphql("query { shop { id } }")

    assert "[truncated]" in str(excinfo.value)
    assert len(str(excinfo.value)) < len(long_body)


def test_graphql_raises_clear_error_for_non_json_2xx_body(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                status_code=200, text="<html>not json</html>", json_raises=True
            )
        ),
    )

    with pytest.raises(RuntimeError, match="non-JSON"):
        shopify._graphql("query { shop { id } }")


def test_graphql_rejects_response_with_more_than_one_top_level_field(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"a": 1, "b": 2}})),
    )

    with pytest.raises(RuntimeError, match="exactly one"):
        shopify._graphql("query { a b }")


@pytest.mark.parametrize("status_code", [301, 302, 303, 307, 308])
def test_graphql_rejects_redirect_response(monkeypatch, status_code):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(status_code=status_code)),
    )

    with pytest.raises(RuntimeError, match="redirect"):
        shopify._graphql("query { shop { id } }")


def test_graphql_retries_once_on_throttled_and_then_succeeds(monkeypatch):
    responses = [
        MockResponse(status_code=429),
        MockResponse(json_data={"data": {"ok": True}}),
    ]
    mock_post = Mock(side_effect=responses)
    monkeypatch.setattr(shopify.requests, "post", mock_post)
    monkeypatch.setattr(shopify.time, "sleep", Mock())

    data, errors = shopify._graphql("query { ok }")

    assert data == {"ok": True}
    assert mock_post.call_count == 2
    shopify.time.sleep.assert_called_once()


def test_graphql_does_not_retry_a_second_throttle(monkeypatch):
    mock_post = Mock(return_value=MockResponse(status_code=429))
    monkeypatch.setattr(shopify.requests, "post", mock_post)
    monkeypatch.setattr(shopify.time, "sleep", Mock())

    with pytest.raises(RuntimeError):
        shopify._graphql("query { ok }")

    assert mock_post.call_count == 2


def test_graphql_redacts_connection_error_message(monkeypatch):
    def _raise(*args, **kwargs):
        raise requests.exceptions.ProxyError(
            "Unable to connect to proxy: "
            "https://user:sp-secret-proxy-pass@proxy.internal:8080/"
        )

    monkeypatch.setattr(shopify.requests, "post", _raise)

    with pytest.raises(RuntimeError) as excinfo:
        shopify._graphql("query { ok }")

    assert "sp-secret-proxy-pass" not in str(excinfo.value)


def test_extract_connection_returns_end_cursor_when_has_more():
    items, has_more, after_cursor = shopify._extract_connection(
        {
            "products": {
                "nodes": [{"id": "1"}],
                "pageInfo": {"hasNextPage": True, "endCursor": "cur1"},
            }
        },
        "products",
        lambda n: n,
    )

    assert items == [{"id": "1"}]
    assert has_more is True
    assert after_cursor == "cur1"


def test_extract_connection_skips_null_nodes():
    # A connection's individual nodes can be null (e.g. failed to resolve
    # due to permissions) even when the list itself is present.
    items, _has_more, _after_cursor = shopify._extract_connection(
        {"products": {"nodes": [{"id": "1"}, None, {"id": "2"}], "pageInfo": {}}},
        "products",
        lambda n: n,
    )

    assert items == [{"id": "1"}, {"id": "2"}]


def test_extract_connection_no_cursor_when_not_truncated():
    _items, has_more, after_cursor = shopify._extract_connection(
        {"products": {"nodes": [], "pageInfo": {"hasNextPage": False}}},
        "products",
        lambda n: n,
    )

    assert has_more is False
    assert after_cursor is None


def test_extract_connection_treats_missing_end_cursor_as_no_more_pages():
    # hasNextPage=true with no endCursor would otherwise tell a caller to
    # retry with after=None -- the first page again, forever.
    _items, has_more, after_cursor = shopify._extract_connection(
        {"products": {"nodes": [{"id": "1"}], "pageInfo": {"hasNextPage": True}}},
        "products",
        lambda n: n,
    )

    assert has_more is False
    assert after_cursor is None


def test_shop_summary_snake_cases_fields():
    assert shopify._shop_summary(
        {
            "name": "Acme",
            "myshopifyDomain": "acme.myshopify.com",
            "email": "owner@acme.com",
            "currencyCode": "USD",
            "ianaTimezone": "America/New_York",
        }
    ) == {
        "name": "Acme",
        "domain": "acme.myshopify.com",
        "email": "owner@acme.com",
        "currency": "USD",
        "timezone": "America/New_York",
    }


def test_get_shop_returns_shop_info(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "shop": {
                            "name": "Acme",
                            "myshopifyDomain": "acme.myshopify.com",
                            "currencyCode": "USD",
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_get_shop())

    assert result["status"] == "success"
    assert result["shop"]["name"] == "Acme"
    assert result["shop"]["domain"] == "acme.myshopify.com"
    assert result["shop"]["currency"] == "USD"


def test_list_products_sends_query_limit_and_after(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "products": {
                        "nodes": [{"id": "gid://shopify/Product/1", "title": "Shirt"}],
                        "pageInfo": {"hasNextPage": False},
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(
        shopify.shopify_list_products(query="status:active", limit=10, after="cur0")
    )

    assert result["status"] == "success"
    assert result["products"][0]["title"] == "Shirt"
    body = mock_post.call_args.kwargs["json"]
    assert body["variables"] == {"first": 10, "query": "status:active", "after": "cur0"}
    assert "$query: String!" in body["query"]
    assert "$after: String!" in body["query"]


def test_list_products_omits_query_and_after_when_not_provided(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"products": {"nodes": [], "pageInfo": {"hasNextPage": False}}}
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    shopify.shopify_list_products()

    body = mock_post.call_args.kwargs["json"]
    assert body["variables"] == {"first": 25}
    assert "query" not in body["query"] or "$query" not in body["query"]


def test_list_products_clamps_limit_to_max(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"products": {"nodes": [], "pageInfo": {"hasNextPage": False}}}
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    shopify.shopify_list_products(limit=10_000)

    assert mock_post.call_args.kwargs["json"]["variables"]["first"] == shopify.MAX_LIMIT


def test_list_products_surfaces_partial_success_warnings(monkeypatch):
    # A genuine partial GraphQL success (data still returned alongside a
    # top-level errors entry, e.g. a nullable nested field that failed to
    # resolve) must surface as a warning at the tool boundary, not be
    # silently dropped.
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": [
                                {"id": "gid://shopify/Product/1", "title": "Shirt"}
                            ],
                            "pageInfo": {"hasNextPage": False},
                        }
                    },
                    "errors": [{"message": "a sub-field failed"}],
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_list_products())

    assert result["status"] == "success"
    assert result["warnings"] == ["a sub-field failed"]


def test_list_products_caps_output_size(monkeypatch):
    big_products = [
        {"id": f"gid://shopify/Product/{i}", "title": "x" * 1000} for i in range(50)
    ]
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": big_products,
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(shopify, "get_tool_max_output_length", lambda: 2000)

    raw = shopify.shopify_list_products(limit=100)
    result = json.loads(raw)

    assert result["status"] == "success"
    assert result["truncated"] is True
    assert 0 < len(result["products"]) < len(big_products)
    assert result["has_more"] is True
    # first page (no `after` was passed in) -- a truncated page must report
    # a dead end, not a real (unreachable) cursor.
    assert result["after_cursor"] is None
    assert len(raw) <= 2000 + 200  # last halving step can overshoot


def test_list_products_truncation_retries_same_page_not_shopifys_next_page(monkeypatch):
    # A truncated page must report ITS OWN input cursor so the caller
    # retries the same starting point with a smaller limit -- returning
    # Shopify's real endCursor instead would skip every item this call
    # couldn't fit, silently dropping them for good.
    big_products = [
        {"id": f"gid://shopify/Product/{i}", "title": "x" * 1000} for i in range(50)
    ]
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": big_products,
                            "pageInfo": {
                                "hasNextPage": True,
                                "endCursor": "shopifys_real_next_cursor",
                            },
                        }
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(shopify, "get_tool_max_output_length", lambda: 2000)

    result = json.loads(shopify.shopify_list_products(limit=100, after="cur0"))

    assert result["truncated"] is True
    assert result["has_more"] is True
    assert result["after_cursor"] == "cur0"


def test_list_products_truncates_to_empty_when_single_item_still_oversized(monkeypatch):
    """Regression for an off-by-one in the halving loop: a page that
    shrinks to exactly one item which is STILL over budget on its own must
    still truncate down to zero items with truncated=True, not silently
    return the oversized single item untouched."""
    huge_product = {"id": "gid://shopify/Product/1", "title": "x" * 5000}
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": [huge_product],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(shopify, "get_tool_max_output_length", lambda: 200)

    result = json.loads(shopify.shopify_list_products(limit=1))

    assert result["status"] == "success"
    assert result["products"] == []
    assert result["truncated"] is True
    assert result["has_more"] is True


def test_list_products_explains_the_dead_end_when_collapsed_to_empty(monkeypatch):
    huge_product = {"id": "gid://shopify/Product/1", "title": "x" * 5000}
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": [huge_product],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(shopify, "get_tool_max_output_length", lambda: 400)

    result = json.loads(shopify.shopify_list_products(limit=1))

    assert "message" in result
    assert "too large" in result["message"]


def test_list_products_drops_message_when_it_alone_exceeds_the_limit(monkeypatch):
    huge_product = {"id": "gid://shopify/Product/1", "title": "x" * 5000}
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "products": {
                            "nodes": [huge_product],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )
    monkeypatch.setattr(shopify, "get_tool_max_output_length", lambda: 150)

    raw = shopify.shopify_list_products(limit=1)
    result = json.loads(raw)

    assert len(raw) <= 150
    assert result["status"] == "success"
    assert result["truncated"] is True
    assert result["products"] == []
    assert "message" not in result


def test_get_product_normalizes_id_and_returns_summary(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "product": {
                        "id": "gid://shopify/Product/1",
                        "title": "Shirt",
                        "status": "ACTIVE",
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(shopify.shopify_get_product("1"))

    assert result["status"] == "success"
    assert result["product"]["title"] == "Shirt"
    assert mock_post.call_args.kwargs["json"]["variables"] == {
        "id": "gid://shopify/Product/1"
    }


def test_get_product_returns_error_when_not_found(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"product": None}})),
    )

    result = json.loads(shopify.shopify_get_product("999"))

    assert result["status"] == "error"


def test_create_product_requires_title():
    result = json.loads(shopify.shopify_create_product(""))

    assert result["status"] == "error"


def test_create_product_rejects_invalid_status():
    result = json.loads(shopify.shopify_create_product("Shirt", status="PUBLISHED"))

    assert result["status"] == "error"


def test_create_product_rejects_lowercase_status():
    # Status values are matched exactly against Shopify's enum spelling;
    # a caller passing "active" instead of "ACTIVE" gets a clear local
    # error rather than a Shopify-side enum validation failure.
    result = json.loads(shopify.shopify_create_product("Shirt", status="active"))

    assert result["status"] == "error"


def test_create_product_accepts_unlisted_status(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "productCreate": {
                        "product": {
                            "id": "gid://shopify/Product/1",
                            "status": "UNLISTED",
                        },
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(shopify.shopify_create_product("Shirt", status="UNLISTED"))

    assert result["status"] == "success"
    assert mock_post.call_args.kwargs["json"]["variables"]["product"]["status"] == (
        "UNLISTED"
    )


def test_create_product_sends_expected_input(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "productCreate": {
                        "product": {"id": "gid://shopify/Product/1", "title": "Shirt"},
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(
        shopify.shopify_create_product(
            "Shirt", description="<p>Nice</p>", vendor="Acme", tags="new, summer"
        )
    )

    assert result["status"] == "success"
    product_input = mock_post.call_args.kwargs["json"]["variables"]["product"]
    assert product_input["title"] == "Shirt"
    assert product_input["descriptionHtml"] == "<p>Nice</p>"
    assert product_input["vendor"] == "Acme"
    assert product_input["tags"] == ["new", "summer"]
    assert product_input["status"] == "DRAFT"


def test_create_product_surfaces_user_errors(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "productCreate": {
                            "product": None,
                            "userErrors": [{"field": ["title"], "message": "too long"}],
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_create_product("Shirt"))

    assert result["status"] == "error"
    assert "title: too long" in result["message"]


def test_update_product_requires_at_least_one_field():
    result = json.loads(shopify.shopify_update_product("1"))

    assert result["status"] == "error"


def test_update_product_rejects_invalid_status(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(shopify.shopify_update_product("1", status="PUBLISHED"))

    assert result["status"] == "error"
    mock_post.assert_not_called()


def test_update_product_sends_only_provided_fields(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "productUpdate": {
                        "product": {"id": "gid://shopify/Product/1", "title": "New"},
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    shopify.shopify_update_product("1", title="New")

    product_input = mock_post.call_args.kwargs["json"]["variables"]["product"]
    assert product_input == {"id": "gid://shopify/Product/1", "title": "New"}


def test_update_product_clears_field_with_explicit_empty_string(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "productUpdate": {
                        "product": {"id": "gid://shopify/Product/1"},
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    # vendor="" is an explicit "clear this field", distinct from the
    # default None ("leave unchanged") -- both must be forwarded, not
    # dropped by a truthiness check.
    result = json.loads(shopify.shopify_update_product("1", vendor=""))

    assert result["status"] == "success"
    product_input = mock_post.call_args.kwargs["json"]["variables"]["product"]
    assert product_input == {"id": "gid://shopify/Product/1", "vendor": ""}


def test_update_product_rejects_blank_title(monkeypatch):
    mock_post = Mock()
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(shopify.shopify_update_product("1", title="   "))

    assert result["status"] == "error"
    mock_post.assert_not_called()


def test_create_product_fails_closed_when_mutation_field_missing(monkeypatch):
    # A malformed/empty response for the requested mutation field must be
    # treated as a failure, not silently reported as success with an empty
    # product.
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {}})),
    )

    result = json.loads(shopify.shopify_create_product("Shirt"))

    assert result["status"] == "error"


def test_create_product_fails_closed_when_object_null_despite_empty_user_errors(
    monkeypatch,
):
    # userErrors is empty, but the product itself is null (e.g. a
    # nested field GraphQL couldn't resolve, null-propagating the whole
    # object) -- this must not be reported as success with an all-null
    # product.
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {"productCreate": {"product": None, "userErrors": []}},
                    "errors": [{"message": "Access denied for tags field"}],
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_create_product("Shirt"))

    assert result["status"] == "error"
    assert "Access denied for tags field" in result["message"]


def test_errors_detail_returns_string_as_is():
    assert shopify._errors_detail("plain string error") == "plain string error"


def test_errors_detail_joins_list_of_error_objects():
    assert (
        shopify._errors_detail([{"message": "bad"}, {"message": "worse"}])
        == "bad; worse"
    )


def test_errors_detail_stringifies_unexpected_shape():
    # Not per the GraphQL spec (errors should be a list), but must not
    # silently iterate a dict's keys and drop the actual diagnostic text.
    detail = shopify._errors_detail({"query": ["failed to parse query"]})

    assert "failed to parse query" in detail


def test_list_orders_sends_query_and_limit(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {"orders": {"nodes": [], "pageInfo": {"hasNextPage": False}}}
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    shopify.shopify_list_orders(query="financial_status:paid", limit=5)

    assert mock_post.call_args.kwargs["json"]["variables"] == {
        "first": 5,
        "query": "financial_status:paid",
    }


def test_get_order_returns_summary(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "order": {
                            "id": "gid://shopify/Order/1",
                            "name": "#1001",
                            "totalPriceSet": {
                                "shopMoney": {"amount": "10.00", "currencyCode": "USD"}
                            },
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_get_order("1"))

    assert result["status"] == "success"
    assert result["order"]["name"] == "#1001"
    assert result["order"]["total_price"] == "10.00"
    assert result["order"]["currency"] == "USD"


def test_update_order_requires_tags_or_note():
    result = json.loads(shopify.shopify_update_order("1"))

    assert result["status"] == "error"


def test_update_order_sends_tags_and_note(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "orderUpdate": {
                        "order": {"id": "gid://shopify/Order/1"},
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    shopify.shopify_update_order("1", tags="vip, priority", note="Handle with care")

    order_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert order_input == {
        "id": "gid://shopify/Order/1",
        "tags": ["vip", "priority"],
        "note": "Handle with care",
    }


def test_update_order_clears_note_with_explicit_empty_string(monkeypatch):
    mock_post = Mock(
        return_value=MockResponse(
            json_data={
                "data": {
                    "orderUpdate": {
                        "order": {"id": "gid://shopify/Order/1"},
                        "userErrors": [],
                    }
                }
            }
        )
    )
    monkeypatch.setattr(shopify.requests, "post", mock_post)

    result = json.loads(shopify.shopify_update_order("1", note=""))

    assert result["status"] == "success"
    order_input = mock_post.call_args.kwargs["json"]["variables"]["input"]
    assert order_input == {"id": "gid://shopify/Order/1", "note": ""}


def test_update_order_surfaces_user_errors(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "orderUpdate": {
                            "order": None,
                            "userErrors": [{"field": None, "message": "not found"}],
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_update_order("999", note="x"))

    assert result["status"] == "error"
    assert "not found" in result["message"]


def test_list_customers_returns_summaries(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "customers": {
                            "nodes": [
                                {
                                    "id": "gid://shopify/Customer/1",
                                    "firstName": "Jane",
                                    "defaultEmailAddress": {
                                        "emailAddress": "jane@example.com"
                                    },
                                    "numberOfOrders": "3",
                                }
                            ],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_list_customers())

    customer = result["customers"][0]
    assert customer["first_name"] == "Jane"
    assert customer["email"] == "jane@example.com"
    assert customer["number_of_orders"] == "3"


def test_get_customer_returns_summary(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "customer": {
                            "id": "gid://shopify/Customer/1",
                            "firstName": "Jane",
                            "lastName": "Doe",
                            "defaultEmailAddress": {"emailAddress": "jane@example.com"},
                            "numberOfOrders": "2",
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_get_customer("1"))

    assert result["status"] == "success"
    assert result["customer"]["email"] == "jane@example.com"
    assert result["customer"]["number_of_orders"] == "2"


def test_get_customer_returns_error_when_not_found(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(return_value=MockResponse(json_data={"data": {"customer": None}})),
    )

    result = json.loads(shopify.shopify_get_customer("999"))

    assert result["status"] == "error"


def test_list_collections_returns_summaries_with_products_count(monkeypatch):
    monkeypatch.setattr(
        shopify.requests,
        "post",
        Mock(
            return_value=MockResponse(
                json_data={
                    "data": {
                        "collections": {
                            "nodes": [
                                {
                                    "id": "gid://shopify/Collection/1",
                                    "title": "Summer",
                                    "productsCount": {"count": 12},
                                }
                            ],
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            )
        ),
    )

    result = json.loads(shopify.shopify_list_collections())

    collection = result["collections"][0]
    assert collection["title"] == "Summer"
    assert collection["products_count"] == 12


def test_shopify_app_registry_requires_store_domain_and_token():
    from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows

    shopify_app = next(
        row for row in get_builtin_public_mcp_app_rows() if row["app_id"] == "shopify"
    )
    assert shopify_app["provider_name"] is None
    assert shopify_app["category"] == "Commerce"
    assert shopify_app["transport"] == "stdio"
    assert shopify_app["launch_config"]["required_env"] == [
        "SHOPIFY_STORE_DOMAIN",
        "SHOPIFY_ACCESS_TOKEN",
    ]
