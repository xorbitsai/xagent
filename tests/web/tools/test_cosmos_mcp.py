"""Tests for the Azure Cosmos DB MCP connector (SDK mocked at its client boundary)."""

import asyncio
import inspect
import io
import json
import logging
import typing
from typing import Any

import pytest
import requests
from azure.core.exceptions import ServiceRequestError, ServiceResponseTimeoutError
from azure.core.pipeline.transport import HttpRequest, RequestsTransport
from azure.cosmos.exceptions import CosmosClientTimeoutError, CosmosHttpResponseError

from xagent.core.utils import public_requests
from xagent.web.builtin_mcp_registry import get_builtin_public_mcp_app_rows
from xagent.web.connector_probes import get_connector_probe
from xagent.web.tools.mcp import cosmos

KEY = "dGhpcy1pcy1hLXNlY3JldC1rZXk="
ENDPOINT = "https://acct.documents.azure.com:443/"
SENTINEL = "SENTINEL-RAW-SDK-TEXT"
TOOLS = [
    "cosmos_list_databases",
    "cosmos_list_containers",
    "cosmos_query_container",
    "cosmos_read_item",
]


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("COSMOS_ENDPOINT", ENDPOINT)
    monkeypatch.setenv("COSMOS_KEY", KEY)
    monkeypatch.setenv("COSMOS_DATABASE", "appdb")


class FakePager:
    def __init__(self, pages, calls):
        self._pages = list(pages)
        self._calls = calls
        self.continuation_token = "TOKEN-SENT"  # the token the caller passed

    def __next__(self):
        self._calls.append("next")
        if not self._pages:
            raise StopIteration  # the real pager does this on an empty block
        items, token = self._pages.pop(0)
        self.continuation_token = token
        return iter(items)


class FakeItems:
    def __init__(self, pages, calls, hook):
        self._pages, self._calls, self._hook = pages, calls, hook

    def by_page(self, token=None):
        self._calls.append(("by_page", token))
        for charge in ("2.5", "1.0"):
            self._hook({"x-ms-request-charge": charge}, {})
        return FakePager(self._pages, self._calls)


class FakeContainer:
    def __init__(self, owner):
        self.owner = owner

    def query_items(self, **kwargs):
        self.owner.query_kwargs = kwargs
        self.owner.timeouts.append(kwargs.get("timeout"))
        return FakeItems(self.owner.pages, self.owner.calls, kwargs["response_hook"])

    def read_item(self, item, partition_key, response_hook=None, **kwargs):
        self.owner.read_args = (item, partition_key)
        self.owner.timeouts.append(kwargs.get("timeout"))
        response_hook({"x-ms-request-charge": "1"}, {})
        return self.owner.item


class FakeDatabase:
    def __init__(self, owner):
        self.owner = owner

    def read(self, **kwargs):
        self.owner.calls.append("db.read")
        self.owner.timeouts.append(kwargs.get("timeout"))
        return {}

    def list_containers(self, max_item_count=None, **kwargs):
        self.owner.timeouts.append(kwargs.get("timeout"))
        self.owner.list_containers_max = max_item_count
        return iter(self.owner.containers)

    def get_container_client(self, name):
        self.owner.container_name = name
        return FakeContainer(self.owner)


class FakeClient:
    """Stands in for ``azure.cosmos.CosmosClient``."""

    last: "FakeClient"
    default_pages: list[Any] = [([{"id": "1"}], None)]
    default_containers: list[dict[str, Any]] = []
    default_item: Any = {"id": "1"}

    def __init__(self, url, credential=None, **kwargs):
        FakeClient.last = self
        self.url, self.credential, self.kwargs = url, credential, kwargs
        self.calls: list[Any] = []
        self.pages = list(self.default_pages)
        self.containers = list(self.default_containers)
        self.databases: list[dict[str, Any]] = [{"id": "appdb"}, {"id": "other"}]
        self.item = self.default_item
        self.closed = False
        self.timeouts: list[Any] = []

    def list_databases(self, max_item_count=None, **kwargs):
        self.timeouts.append(kwargs.get("timeout"))
        return iter(self.databases)

    def get_database_client(self, name):
        self.database_name = name
        return FakeDatabase(self)

    def close(self):
        self.closed = True


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(cosmos, "CosmosClient", FakeClient)
    monkeypatch.setattr(FakeClient, "last", None, raising=False)
    return FakeClient


def run(tool, **kwargs):
    return json.loads(getattr(cosmos, tool)(**kwargs))


# --- endpoint validation ---------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://acct.documents.azure.com:443/", "https://acct.documents.azure.com/"),
        ("https://acct.documents.azure.com", "https://acct.documents.azure.com/"),
        ("  https://ACCT.documents.azure.com/\n", "https://acct.documents.azure.com/"),
        ("https://a-b.documents.azure.com:443", "https://a-b.documents.azure.com/"),
        ("https://acct.documents.azure.us:443/", "https://acct.documents.azure.us/"),
        ("https://acct.documents.azure.cn/", "https://acct.documents.azure.cn/"),
    ],
)
def test_validate_endpoint_accepts_account_hosts(raw, expected):
    assert cosmos.validate_endpoint(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "http://acct.documents.azure.com/",
        "acct.documents.azure.com",
        "https://documents.azure.com/",
        "https://a.b.documents.azure.com/",
        "https://evil.example.com/",
        "https://acct.documents.azure.com.evil.com/",
        "https://evildocuments.azure.com/",
        "https://acct.documents.azure.de/",
        "https://127.0.0.1/",
        "https://169.254.169.254/",
        "https://[::1]/",
        "https://localhost/",
        "https://user:pw@acct.documents.azure.com/",
        "https://acct.documents.azure.com@evil.com/",
        "https://evil.com\\@acct.documents.azure.com/",
        "https://acct.documents.azure.com:8443/",
        "https://acct.documents.azure.com:80/",
        "https://acct.documents.azure.com/dbs",
        "https://acct.documents.azure.com/?key=secret",
        "https://acct.documents.azure.com/#frag",
        "https://acct.documents.azure.com/ x",
        "https://аcct.documents.azure.com/",
        "",
    ],
)
def test_validate_endpoint_rejects(raw):
    with pytest.raises(cosmos.CosmosConfigError):
        cosmos.validate_endpoint(raw)


def test_rejected_endpoint_never_creates_a_client(client, monkeypatch):
    monkeypatch.setenv("COSMOS_ENDPOINT", "https://169.254.169.254/")
    result = run("cosmos_list_containers")
    assert result["error_code"] == "invalid_configuration"
    assert client.last is None


def test_guarded_transport_blocks_other_hosts(monkeypatch, caplog):
    sent = []
    monkeypatch.setattr(
        RequestsTransport, "send", lambda self, request, **kw: sent.append(request.url)
    )
    transport = cosmos._GuardedTransport()
    req = lambda url: type("R", (), {"url": url})()  # noqa: E731
    transport.send(req("https://acct.documents.azure.com/dbs/appdb/colls?x=1"))
    for bad in (
        "https://evil.example.net:443/",
        "https://acct.documents.azure.com:8443/dbs",
        "http://acct.documents.azure.com/dbs",
        "https://10.0.0.5/dbs",
    ):
        with pytest.raises(cosmos.BlockedDestinationError):
            transport.send(req(bad))
    assert sent == ["https://acct.documents.azure.com/dbs/appdb/colls?x=1"]
    assert "evil.example.net" in caplog.text


def test_blocked_destination_is_its_own_error_code(client, monkeypatch, caplog):
    def blocked(self, *a, **kw):
        raise cosmos.BlockedDestinationError

    monkeypatch.setattr(FakeClient, "list_databases", blocked)
    result = run("cosmos_list_databases")
    assert result["error_code"] == "blocked_destination"
    assert KEY not in json.dumps(result)


def test_transport_uses_the_public_only_session():
    """Dropping ``session=`` would remove the private-address guard."""
    transport = cosmos._GuardedTransport()
    assert isinstance(transport.session, public_requests.Session)


def test_transport_does_not_follow_redirects(monkeypatch):
    seen = {}

    def fake_request(method, url, **kw):
        seen.update(kw)
        response = requests.Response()
        response.status_code = 204
        response.raw = io.BytesIO(b"")
        return response

    transport = cosmos._GuardedTransport()
    monkeypatch.setattr(transport.session, "request", fake_request)
    transport.send(HttpRequest("GET", "https://acct.documents.azure.com/dbs"))
    assert seen["allow_redirects"] is False


def test_real_sdk_never_contacts_a_discovered_regional_endpoint(monkeypatch):
    """Account metadata names another host; with discovery off the real SDK
    must stay on the validated endpoint."""
    urls = []

    def fake_request(self, method, url, **kw):
        urls.append(url)
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "application/json"
        if url.rstrip("/").endswith("/dbs"):
            body: dict[str, Any] = {"Databases": [{"id": "appdb"}], "_count": 1}
        else:
            region = [
                {
                    "name": "West US",
                    "databaseAccountEndpoint": "https://evil.example.net:443/",
                }
            ]
            body = {
                "id": "acct",
                "writableLocations": region,
                "readableLocations": region,
                "userConsistencyPolicy": {"defaultConsistencyLevel": "Session"},
            }
        response._content = json.dumps(body).encode()
        response.raw = io.BytesIO(response._content)
        return response

    monkeypatch.setattr(public_requests.Session, "request", fake_request)
    result = run("cosmos_list_databases")
    assert result["databases"] == ["appdb"]
    assert urls and all(u.startswith("https://acct.documents.azure.com/") for u in urls)


def test_client_closes_the_transport_when_construction_fails(monkeypatch):
    closed = []
    monkeypatch.setattr(
        cosmos._GuardedTransport, "close", lambda self: closed.append(True)
    )

    def failing(*a, **kw):
        raise ServiceRequestError("network down")

    monkeypatch.setattr(cosmos, "CosmosClient", failing)
    assert run("cosmos_list_databases")["error_code"] == "unreachable"
    assert closed == [True]


def test_client_is_built_bounded_and_without_discovery(client):
    run("cosmos_list_databases")
    created = client.last
    assert created.url == "https://acct.documents.azure.com/"
    assert created.credential == KEY
    assert isinstance(created.kwargs["transport"], cosmos._GuardedTransport)
    assert created.kwargs["enable_endpoint_discovery"] is False
    assert created.kwargs["connection_timeout"] == cosmos.CONNECTION_TIMEOUT_SECONDS
    assert created.kwargs["read_timeout"] == cosmos.READ_TIMEOUT_SECONDS
    assert created.kwargs["timeout"] == cosmos.TOTAL_TIMEOUT_SECONDS
    assert created.kwargs["retry_total"] == cosmos.RETRY_TOTAL
    assert created.kwargs["retry_throttle_total"] == cosmos.THROTTLE_RETRY_TOTAL
    assert created.closed


@pytest.mark.parametrize("name", ["COSMOS_ENDPOINT", "COSMOS_KEY", "COSMOS_DATABASE"])
def test_missing_env(client, monkeypatch, name):
    monkeypatch.delenv(name)
    result = run("cosmos_list_containers")
    assert result["error_code"] == "invalid_configuration"
    assert name in result["message"]


# --- tools -----------------------------------------------------------------


def test_list_databases(client):
    result = run("cosmos_list_databases")
    assert result == {
        "status": "success",
        "databases": ["appdb", "other"],
        "configured_database": "appdb",
        "truncated": False,
    }


def test_list_containers_reads_database_and_is_bounded(client, monkeypatch):
    monkeypatch.setattr(
        FakeClient,
        "default_containers",
        [
            {"id": f"c{i}", "partitionKey": {"paths": ["/pk"], "kind": "Hash"}}
            for i in range(cosmos.MAX_LIST_ITEMS + 5)
        ],
    )
    result = run("cosmos_list_containers")
    assert client.last.database_name == "appdb"
    assert client.last.calls == ["db.read"]
    assert len(result["containers"]) == cosmos.MAX_LIST_ITEMS
    assert result["containers"][0] == {"id": "c0", "partition_key_paths": ["/pk"]}
    assert result["truncated"] is True


def test_query_one_page_with_continuation_and_charge(client, monkeypatch):
    monkeypatch.setattr(
        FakeClient, "default_pages", [([{"id": "1"}, {"id": "2"}], "TOKEN-2")]
    )
    result = run(
        "cosmos_query_container",
        container="orders",
        query="SELECT * FROM c WHERE c.s = @s",
        parameters={"s": "open", "@n": 3},
        continuation_token="TOKEN-1",
    )
    fake = client.last
    assert fake.calls == [("by_page", "TOKEN-1"), "next"]  # one page, never drained
    assert fake.query_kwargs["parameters"] == [
        {"name": "@s", "value": "open"},
        {"name": "@n", "value": 3},
    ]
    assert fake.query_kwargs["enable_cross_partition_query"] is True
    assert "partition_key" not in fake.query_kwargs
    assert fake.container_name == "orders"
    assert fake.database_name == "appdb"
    assert result["items"] == [{"id": "1"}, {"id": "2"}]
    assert result["continuation_token"] == "TOKEN-2"
    assert result["has_more"] is True
    assert result["request_charge"] == 3.5
    assert result["truncated"] is False


def test_query_last_page_has_no_token(client):
    result = run("cosmos_query_container", container="c", query="select 1")
    assert result["continuation_token"] is None
    assert result["has_more"] is False


def test_query_partition_key_targets_one_partition(client):
    run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c",
        partition_key="p1",
    )
    kwargs = client.last.query_kwargs
    assert kwargs["partition_key"] == "p1"
    assert "enable_cross_partition_query" not in kwargs


@pytest.mark.parametrize(
    ("requested", "effective"), [(0, 1), (-5, 1), (7, 7), (100, 100), (10_000, 100)]
)
def test_query_page_size_is_clamped(client, requested, effective):
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT 1",
        max_item_count=requested,
    )
    assert client.last.query_kwargs["max_item_count"] == effective
    assert result["max_item_count"] == effective


def test_query_exhausted_pager_ends_the_paging(client, monkeypatch):
    """A continuation that yields no further page must not echo the token."""
    monkeypatch.setattr(FakeClient, "default_pages", [])
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c",
        continuation_token="TOKEN-SENT",
    )
    assert result["items"] == []
    assert result["continuation_token"] is None
    assert result["has_more"] is False


def test_query_unchanged_token_is_not_offered_again(client, monkeypatch):
    monkeypatch.setattr(FakeClient, "default_pages", [([], "TOKEN-SENT")])
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c",
        continuation_token="TOKEN-SENT",
    )
    assert (result["continuation_token"], result["has_more"]) == (None, False)


def _limit(monkeypatch, chars):
    monkeypatch.setenv("XAGENT_TOOL_MAX_OUTPUT_LENGTH", str(chars))


def test_query_response_is_capped_as_a_whole_and_keeps_paging_fields(
    client, monkeypatch
):
    _limit(monkeypatch, 5000)
    big = "x" * 1500
    monkeypatch.setattr(
        FakeClient,
        "default_pages",
        [([{"id": f"i{n}", "v": big} for n in range(6)], "NEXT")],
    )
    raw = cosmos.cosmos_query_container(container="c", query="SELECT * FROM c")
    assert len(raw) <= 5000
    result = json.loads(raw)
    assert result["truncated"] is True
    kept = len(result["items"])
    assert 0 < kept < 6
    assert result["omitted_count"] == 6 - kept
    assert result["omitted_ids"] == [f"i{n}" for n in range(kept, 6)]
    assert (result["continuation_token"], result["has_more"]) == ("NEXT", True)
    # the paging fields are the last keys of the JSON, the first a blind cut drops
    assert raw.index('"has_more"') > raw.index('"items"')


def test_query_first_item_alone_over_the_limit_still_makes_progress(
    client, monkeypatch
):
    _limit(monkeypatch, 3000)
    monkeypatch.setattr(
        FakeClient, "default_pages", [([{"id": "huge", "v": "x" * 20_000}], "NEXT")]
    )
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c",
        continuation_token="CURRENT",
    )
    assert result["items"] == [] and result["omitted_ids"] == ["huge"]
    assert result["continuation_token"] == "NEXT"  # past the page, not the input
    assert result["truncated"] is True


def test_list_tools_are_capped_as_a_whole_too(client, monkeypatch):
    _limit(monkeypatch, 1500)
    monkeypatch.setattr(
        FakeClient,
        "default_containers",
        [{"id": "c" * 100, "partitionKey": {"paths": ["/pk"]}} for _ in range(60)],
    )
    raw = cosmos.cosmos_list_containers()
    assert len(raw) <= 1500
    result = json.loads(raw)
    assert result["truncated"] is True and result["database"] == "appdb"
    assert result["omitted_count"] == 60 - len(result["containers"])


def test_list_databases_is_capped_as_a_whole_too(client, monkeypatch):
    _limit(monkeypatch, 600)
    monkeypatch.setattr(
        FakeClient, "__init__", _init_many_databases(FakeClient.__init__)
    )
    raw = cosmos.cosmos_list_databases()
    assert len(raw) <= 600
    assert json.loads(raw)["truncated"] is True


def _init_many_databases(original):
    def init(self, *a, **kw):
        original(self, *a, **kw)
        self.databases = [{"id": "d" * 50 + str(n)} for n in range(40)]

    return init


def test_read_item_is_capped_as_a_whole(client, monkeypatch):
    _limit(monkeypatch, 2000)
    monkeypatch.setattr(
        FakeClient,
        "default_item",
        {"id": "1", "tags": [f"t{n}" * 20 for n in range(200)], "name": "n"},
    )
    raw = cosmos.cosmos_read_item(container="c", id="1", partition_key="p")
    assert len(raw) <= 2000
    result = json.loads(raw)
    assert result["truncated"] is True
    assert result["request_charge"] == 1.0
    assert result["database"] == "appdb" and result["container"] == "c"


@pytest.mark.parametrize(
    "query",
    [
        "SELECT * FROM c ORDER BY c.ts",
        "select c.a, count(1) from c group by c.a",
        "SELECT DISTINCT c.a FROM c",
        "SELECT TOP 5 * FROM c",
        "SELECT * FROM c OFFSET 5 LIMIT 5",
        "SELECT VALUE COUNT(1) FROM c",
        "SELECT VALUE MAX (c.ts) FROM c",
    ],
)
def test_cross_partition_non_resumable_queries_give_no_token(
    client, monkeypatch, query
):
    monkeypatch.setattr(FakeClient, "default_pages", [([{"id": "1"}], "UNUSABLE")])
    result = run("cosmos_query_container", container="c", query=query)
    assert result["continuation_token"] is None
    assert result["has_more"] is False
    assert result["resumable"] is False
    assert result["may_be_incomplete"] is False


def test_non_resumable_full_page_says_it_may_be_incomplete(client, monkeypatch):
    monkeypatch.setattr(
        FakeClient, "default_pages", [([{"id": "1"}, {"id": "2"}], "UNUSABLE")]
    )
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c ORDER BY c.ts",
        max_item_count=2,
    )
    assert result["may_be_incomplete"] is True and result["has_more"] is False


def test_non_resumable_query_rejects_a_token(client):
    result = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c ORDER BY c.ts",
        continuation_token="T",
    )
    assert result["error_code"] == "invalid_configuration"
    assert client.last is None


def test_partition_scoped_or_streamable_queries_keep_their_token(client, monkeypatch):
    monkeypatch.setattr(FakeClient, "default_pages", [([{"id": "1"}], "NEXT")])
    ordered = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c ORDER BY c.ts",
        partition_key="p",
    )
    streamable = run(
        "cosmos_query_container",
        container="c",
        query="SELECT * FROM c WHERE c.note = 'order by top'",
    )
    assert ordered["continuation_token"] == "NEXT" == streamable["continuation_token"]
    assert "resumable" not in streamable


def test_every_operation_gets_the_total_timeout(client):
    run("cosmos_list_databases")
    assert client.last.timeouts == [cosmos.TOTAL_TIMEOUT_SECONDS]
    run("cosmos_list_containers")
    assert client.last.timeouts == [cosmos.TOTAL_TIMEOUT_SECONDS] * 2
    run("cosmos_query_container", container="c", query="SELECT 1")
    assert client.last.timeouts == [cosmos.TOTAL_TIMEOUT_SECONDS]
    run("cosmos_read_item", container="c", id="1", partition_key="p")
    assert client.last.timeouts == [cosmos.TOTAL_TIMEOUT_SECONDS]


@pytest.mark.parametrize(
    "query",
    [
        "DELETE FROM c",
        "UPSERT INTO c VALUES (1)",
        "EXEC sproc",
        "",
        "-- SELECT 1\nDELETE",
        "x" * (cosmos.MAX_QUERY_CHARS + 1),
    ],
)
def test_query_rejects_non_select(client, query):
    result = run("cosmos_query_container", container="c", query=query)
    assert result["error_code"] == "invalid_configuration"
    assert client.last is None  # rejected before any client was built


@pytest.mark.parametrize(
    "kwargs",
    [
        {"parameters": {"bad name": 1}},
        {"parameters": {str(i): i for i in range(cosmos.MAX_PARAMETERS + 1)}},
        {"continuation_token": "x" * (cosmos.MAX_CONTINUATION_TOKEN_CHARS + 1)},
        {"continuation_token": "a\r\nInjected: header"},
        {"container": "a/b"},
        {"container": ""},
    ],
)
def test_query_rejects_bad_arguments(client, kwargs):
    args = {"container": "c", "query": "SELECT 1", **kwargs}
    assert (
        run("cosmos_query_container", **args)["error_code"] == "invalid_configuration"
    )


def test_read_item(client):
    result = run("cosmos_read_item", container="c", id="1", partition_key=["a", 2])
    assert client.last.read_args == ("1", ["a", 2])
    assert result["item"] == {"id": "1"}
    assert result["request_charge"] == 1.0
    assert result["truncated"] is False


@pytest.mark.parametrize(
    "bad_id", ["", "a/b", "a\\b", "a?b", "a#b", "a\nb", " ", ".", "..", "é" * 600]
)
def test_read_item_rejects_bad_id(client, bad_id):
    result = run("cosmos_read_item", container="c", id=bad_id, partition_key="p")
    assert result["error_code"] == "invalid_configuration"


def test_item_ids_may_be_up_to_1023_bytes_but_names_only_255_characters(client):
    ok = run("cosmos_read_item", container="c", id="x" * 1023, partition_key="p")
    assert ok["status"] == "success"
    assert (
        run("cosmos_read_item", container="c", id="x" * 1024, partition_key="p")[
            "status"
        ]
        == "error"
    )
    assert (
        run("cosmos_read_item", container="c", id="é" * 511, partition_key="p")[
            "status"
        ]
        == "success"
    )  # 1,022 bytes
    long_name = "c" * 256
    assert (
        run("cosmos_read_item", container=long_name, id="1", partition_key="p")[
            "error_code"
        ]
        == "invalid_configuration"
    )


@pytest.mark.parametrize("name", [".", ".."])
def test_container_name_rejects_dot_segments(client, name):
    result = run("cosmos_query_container", container=name, query="SELECT 1")
    assert result["error_code"] == "invalid_configuration"


def test_read_item_partition_key_cannot_be_null():
    """``None`` means a cross-partition query elsewhere; here it would silently
    mean "the item whose key is null", so the type does not allow it."""
    hint = typing.get_type_hints(cosmos.cosmos_read_item)["partition_key"]
    assert type(None) not in typing.get_args(hint)
    assert (
        inspect.signature(cosmos.cosmos_read_item).parameters["partition_key"].default
        is inspect.Parameter.empty
    )


def test_not_found_messages_are_written_per_call_site(client, monkeypatch):
    def not_found(self, *a, **kw):
        raise _http(404)

    monkeypatch.setattr(FakeClient, "list_databases", not_found)
    assert "account was not found" in run("cosmos_list_databases")["message"]
    monkeypatch.setattr(FakeClient, "get_database_client", not_found)
    assert "Database 'appdb' was not found" in run("cosmos_list_containers")["message"]
    query = run("cosmos_query_container", container="orders", query="SELECT 1")
    assert "Container 'orders' or the configured database" in query["message"]
    item = run("cosmos_read_item", container="orders", id="42", partition_key="p")
    assert "Item '42' was not found in container 'orders'" in item["message"]


def test_container_tools_are_pinned_to_the_configured_database():
    for name in TOOLS:
        params = inspect.signature(getattr(cosmos, name)).parameters
        assert not {"database", "database_name", "db"} & set(params)


def test_no_unlisted_tools_and_all_read_only():
    tools = asyncio.run(cosmos.mcp.list_tools())
    assert sorted(t.name for t in tools) == sorted(TOOLS)
    assert all(t.annotations and t.annotations.readOnlyHint for t in tools)


# --- error mapping and secrecy ---------------------------------------------


def _http(status, message=SENTINEL, headers=None):
    exc = CosmosHttpResponseError(status_code=status, message=f"{message} {KEY}")
    exc.headers = headers or {}  # type: ignore[assignment]
    return exc


@pytest.mark.parametrize(
    ("exc", "code"),
    [
        (_http(401), "authentication_failed"),
        (_http(403), "forbidden"),
        (
            _http(401, "Local Authorization is disabled. Use an AAD token"),
            "key_auth_disabled",
        ),
        (
            _http(403, "Local Authorization is disabled. Use an AAD token"),
            "key_auth_disabled",
        ),
        (_http(404), "not_found"),
        (_http(429, headers={"x-ms-retry-after-ms": "250"}), "throttled"),
        (_http(400), "bad_request"),
        (_http(408), "timeout"),
        (_http(503), "service_unavailable"),
        (_http(418), "request_failed"),
        (ServiceResponseTimeoutError(f"{SENTINEL} {KEY}"), "timeout"),
        (CosmosClientTimeoutError(f"{SENTINEL} {KEY}"), "timeout"),
        (ServiceRequestError(f"{SENTINEL} {KEY}"), "unreachable"),
        (RuntimeError(f"{SENTINEL} {KEY}"), "internal_error"),
        (ValueError(f"{SENTINEL} {KEY}"), "internal_error"),
    ],
)
@pytest.mark.parametrize(
    "call",
    [
        {"tool": "cosmos_list_databases"},
        {"tool": "cosmos_list_containers"},
        {"tool": "cosmos_query_container", "container": "c", "query": "SELECT 1"},
        {"tool": "cosmos_read_item", "container": "c", "id": "1", "partition_key": "p"},
    ],
)
def test_errors_are_mapped_and_never_leak(client, monkeypatch, caplog, exc, code, call):
    def boom(self, *a, **kw):
        raise exc

    monkeypatch.setattr(FakeClient, "get_database_client", boom)
    monkeypatch.setattr(FakeClient, "list_databases", boom)
    caplog.set_level(logging.DEBUG)
    call = dict(call)
    raw = getattr(cosmos, call.pop("tool"))(**call)
    result = json.loads(raw)
    assert result["status"] == "error"
    assert result["error_code"] == code
    for secret in (KEY, SENTINEL, "acct.documents"):
        assert secret not in raw
        assert secret not in caplog.text


def test_throttled_reports_retry_after(client, monkeypatch):
    monkeypatch.setattr(
        FakeClient,
        "list_databases",
        lambda self, **kw: (_ for _ in ()).throw(
            _http(429, headers={"x-ms-retry-after-ms": "250"})
        ),
    )
    assert run("cosmos_list_databases")["retry_after_ms"] == "250"


def test_bad_query_surfaces_parsed_service_message_only(client, monkeypatch):
    body = {"errors": [{"message": "Syntax error, incorrect syntax near 'x'."}]}
    exc = _http(400, f"Message: {json.dumps(body)}\r\nActivityId: {SENTINEL}")
    monkeypatch.setattr(
        FakeClient, "get_database_client", lambda self, n: (_ for _ in ()).throw(exc)
    )
    result = run("cosmos_query_container", container="c", query="SELECT x")
    assert "Syntax error, incorrect syntax near 'x'." in result["message"]
    assert SENTINEL not in json.dumps(result)


def test_key_never_logged_on_success(client, caplog):
    caplog.set_level(logging.DEBUG)
    run("cosmos_list_containers")
    assert KEY not in caplog.text


# --- registry and probe ----------------------------------------------------


def _row():
    return next(r for r in get_builtin_public_mcp_app_rows() if r["app_id"] == "cosmos")


def test_registry_entry():
    row = _row()
    launch = row["launch_config"]
    assert row["name"] == "Azure Cosmos DB"
    assert row["transport"] == "stdio"
    assert launch["command"] == "python"
    assert launch["args"] == ["-m", "xagent.web.tools.mcp.cosmos"]
    assert launch["required_env"] == [
        "COSMOS_ENDPOINT",
        "COSMOS_KEY",
        "COSMOS_DATABASE",
    ]
    assert launch["read_only_tools"] == TOOLS


def test_registry_allowlist_names_exactly_the_servers_tools():
    listed = {t.name for t in asyncio.run(cosmos.mcp.list_tools())}
    assert set(_row()["launch_config"]["read_only_tools"]) == listed


def test_probe_is_a_declared_read_tool():
    probe = get_connector_probe("cosmos")
    assert probe is not None
    assert probe.tool in TOOLS
    assert probe.tool in _row()["launch_config"]["read_only_tools"]
