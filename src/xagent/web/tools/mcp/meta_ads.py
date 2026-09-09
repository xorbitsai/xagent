import json
import logging
import re
from typing import Any
from urllib.parse import quote

from mcp.server.fastmcp import FastMCP

from . import meta_graph
from .meta_graph import GraphAPIError
from .meta_graph import bounded_limit as _bounded_limit
from .meta_graph import error_response as _error
from .meta_graph import graph_error_response as _graph_error
from .meta_graph import graph_request as _graph_request
from .meta_graph import success_response as _success
from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meta-ads-mcp")

setup_proxy_env()

mcp = FastMCP("meta-ads-mcp")
requests = meta_graph.requests  # exposed for test monkeypatching

AD_ACCOUNT_FIELDS = "id,name,account_status,currency,timezone_name,business_name"
CAMPAIGN_FIELDS = (
    "id,name,objective,status,effective_status,daily_budget,lifetime_budget,"
    "start_time,stop_time,created_time"
)
AD_SET_FIELDS = (
    "id,name,campaign_id,status,effective_status,daily_budget,lifetime_budget,"
    "billing_event,optimization_goal,start_time,end_time"
)
AD_FIELDS = "id,name,adset_id,campaign_id,status,effective_status,creative,created_time"
INSIGHTS_FIELDS = "impressions,clicks,spend,ctr,cpc,cpm,reach,frequency,actions"

_VALID_INSIGHTS_LEVELS = {"account", "campaign", "adset", "ad"}
_VALID_DATE_PRESETS = {
    "today",
    "yesterday",
    "this_month",
    "last_month",
    "this_quarter",
    "maximum",
    "data_maximum",
    "last_3d",
    "last_7d",
    "last_14d",
    "last_28d",
    "last_30d",
    "last_90d",
    "last_week_mon_sun",
    "last_week_sun_sat",
    "last_quarter",
    "last_year",
    "this_week_mon_sun",
    "this_week_sun_sat",
    "this_year",
}

# re.ASCII: bare \d matches any Unicode decimal digit (Arabic-Indic,
# fullwidth, etc.), not just 0-9, which would otherwise let a non-ASCII
# "numeric" id slip past validation and into a Graph API request.
_NUMERIC_ID_PATTERN = re.compile(r"^\d+\Z", re.ASCII)
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}\Z", re.ASCII)


def _normalize_ad_account_id(ad_account_id: str) -> str:
    """Validate and normalize an ad account id to the "act_<digits>" form the
    Graph API expects, accepting the id with or without that prefix.

    Rejects anything that isn't purely digits (after stripping an optional
    "act_" prefix) rather than silently sanitizing it, since the value is
    interpolated directly into a URL path -- a malformed value could
    otherwise redirect the request to an unintended API path.
    """
    value = str(ad_account_id).strip()
    if value.startswith("act_"):
        value = value[len("act_") :]
    if not _NUMERIC_ID_PATTERN.match(value):
        raise ValueError(
            "ad_account_id must be numeric, optionally prefixed with 'act_'"
        )
    return f"act_{value}"


def _numeric_id(value: str, name: str) -> str:
    """Validate a bare numeric Graph object id (a campaign/ad set/ad id).

    Rejects anything non-numeric rather than silently sanitizing it: the
    value is sent to the Graph API as a "filtering" clause, so a malformed
    id should fail fast with a clear error instead of round-tripping to the
    API for a confusing rejection there.
    """
    stripped = str(value).strip()
    if not _NUMERIC_ID_PATTERN.match(stripped):
        raise ValueError(f"{name} must be numeric")
    return stripped


def _insights_object_id(object_id: str) -> str:
    """Validate an insights target id: either an "act_<digits>" ad account,
    or a bare numeric campaign/ad set/ad id -- matching exactly the id shapes
    the other meta_ads_* tools return, so callers never have to reformat one.

    Delegates to _normalize_ad_account_id/_numeric_id for the actual "digits,
    optionally act_-prefixed" check rather than a third copy of that regex,
    so the two id shapes stay validated by exactly one rule each.
    """
    value = str(object_id).strip()
    if not value:
        raise ValueError("object_id is required")
    try:
        if value.startswith("act_"):
            return _normalize_ad_account_id(value)
        return _numeric_id(value, "object_id")
    except ValueError:
        raise ValueError("object_id must be 'act_<digits>' or a numeric id") from None


def _graph_path(*segments: str) -> str:
    return "/" + "/".join(quote(segment, safe="") for segment in segments)


def _equal_filter(field: str, value: str) -> dict[str, str]:
    return {"field": field, "operator": "EQUAL", "value": value}


def _log_graph_error(message: str, error: GraphAPIError) -> None:
    """Log a GraphAPIError with the same token redaction its JSON response
    gets, instead of the raw str(error) -- which can otherwise write the
    access token straight to application logs before graph_error_response()
    has a chance to redact it (the Graph API can echo the token back in an
    OAuth error message)."""
    logger.error(
        "%s: %s",
        message,
        meta_graph.redact_secrets(str(error), sensitive_values=error.sensitive_values),
    )


@mcp.tool()
def meta_ads_auth_status() -> str:
    """Check whether the injected Meta access token is usable."""
    try:
        me = _graph_request("GET", "/me", params={"fields": "id,name"})
        return _success(
            authenticated=True,
            user={"id": me.get("id"), "name": me.get("name")},
        )
    except GraphAPIError as e:
        _log_graph_error("Error checking Meta Ads auth status", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error checking Meta Ads auth status: %s", e)
        return _error(str(e))


@mcp.tool()
def meta_ads_list_ad_accounts(limit: int = 25) -> str:
    """List Meta ad accounts accessible to the connected user. Use this first
    to discover which ad_account_id values are available for the other
    meta_ads_* tools."""
    try:
        result = _graph_request(
            "GET",
            "/me/adaccounts",
            params={"fields": AD_ACCOUNT_FIELDS, "limit": _bounded_limit(limit)},
        )
        return _success(
            ad_accounts=result.get("data", []),
            next_link=(result.get("paging") or {}).get("next"),
        )
    except GraphAPIError as e:
        _log_graph_error("Error listing Meta ad accounts", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing Meta ad accounts: %s", e)
        return _error(str(e))


@mcp.tool()
def meta_ads_get_ad_account(ad_account_id: str) -> str:
    """Get details for one Meta ad account. Accepts the id with or without
    the "act_" prefix."""
    try:
        account_id = _normalize_ad_account_id(ad_account_id)
        result = _graph_request(
            "GET", _graph_path(account_id), params={"fields": AD_ACCOUNT_FIELDS}
        )
        return _success(ad_account=result)
    except GraphAPIError as e:
        _log_graph_error(f"Error getting Meta ad account {ad_account_id}", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error getting Meta ad account %s: %s", ad_account_id, e)
        return _error(str(e))


@mcp.tool()
def meta_ads_list_campaigns(ad_account_id: str, limit: int = 25) -> str:
    """List campaigns for a Meta ad account."""
    try:
        account_id = _normalize_ad_account_id(ad_account_id)
        result = _graph_request(
            "GET",
            _graph_path(account_id, "campaigns"),
            params={"fields": CAMPAIGN_FIELDS, "limit": _bounded_limit(limit)},
        )
        return _success(
            campaigns=result.get("data", []),
            next_link=(result.get("paging") or {}).get("next"),
        )
    except GraphAPIError as e:
        _log_graph_error(f"Error listing campaigns for {ad_account_id}", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing campaigns for %s: %s", ad_account_id, e)
        return _error(str(e))


@mcp.tool()
def meta_ads_list_ad_sets(
    ad_account_id: str, campaign_id: str | None = None, limit: int = 25
) -> str:
    """List ad sets for a Meta ad account, optionally filtered to one
    campaign_id."""
    try:
        account_id = _normalize_ad_account_id(ad_account_id)
        params: dict[str, Any] = {
            "fields": AD_SET_FIELDS,
            "limit": _bounded_limit(limit),
        }
        if campaign_id:
            params["filtering"] = json.dumps(
                [_equal_filter("campaign.id", _numeric_id(campaign_id, "campaign_id"))]
            )
        result = _graph_request("GET", _graph_path(account_id, "adsets"), params=params)
        return _success(
            ad_sets=result.get("data", []),
            next_link=(result.get("paging") or {}).get("next"),
        )
    except GraphAPIError as e:
        _log_graph_error(f"Error listing ad sets for {ad_account_id}", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing ad sets for %s: %s", ad_account_id, e)
        return _error(str(e))


@mcp.tool()
def meta_ads_list_ads(
    ad_account_id: str,
    campaign_id: str | None = None,
    ad_set_id: str | None = None,
    limit: int = 25,
) -> str:
    """List ads for a Meta ad account, optionally filtered to one campaign_id
    or ad_set_id."""
    try:
        account_id = _normalize_ad_account_id(ad_account_id)
        params: dict[str, Any] = {"fields": AD_FIELDS, "limit": _bounded_limit(limit)}
        filters = []
        if campaign_id:
            filters.append(
                _equal_filter("campaign.id", _numeric_id(campaign_id, "campaign_id"))
            )
        if ad_set_id:
            filters.append(
                _equal_filter("adset.id", _numeric_id(ad_set_id, "ad_set_id"))
            )
        if filters:
            params["filtering"] = json.dumps(filters)
        result = _graph_request("GET", _graph_path(account_id, "ads"), params=params)
        return _success(
            ads=result.get("data", []),
            next_link=(result.get("paging") or {}).get("next"),
        )
    except GraphAPIError as e:
        _log_graph_error(f"Error listing ads for {ad_account_id}", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error listing ads for %s: %s", ad_account_id, e)
        return _error(str(e))


@mcp.tool()
def meta_ads_get_insights(
    object_id: str,
    level: str | None = None,
    date_preset: str = "last_30d",
    since: str | None = None,
    until: str | None = None,
    fields: str | None = None,
    limit: int = 25,
) -> str:
    """Get performance insights (impressions, clicks, spend, etc.) for a Meta
    ad account, campaign, ad set, or ad. object_id is an "act_<id>" ad
    account id, or a campaign/ad set/ad id as returned by the other
    meta_ads_* tools. level optionally breaks the result down to
    "account", "campaign", "adset", or "ad" rows. Defaults to
    date_preset="last_30d"; pass since and until (both "YYYY-MM-DD") instead
    for an explicit range.
    """
    try:
        target_id = _insights_object_id(object_id)
        if level is not None and level not in _VALID_INSIGHTS_LEVELS:
            raise ValueError(f"level must be one of {sorted(_VALID_INSIGHTS_LEVELS)}")

        params: dict[str, Any] = {
            "fields": fields or INSIGHTS_FIELDS,
            "limit": _bounded_limit(limit),
        }
        if level:
            params["level"] = level

        if since or until:
            if not (since and until):
                raise ValueError("since and until must be provided together")
            if not (_DATE_PATTERN.match(since) and _DATE_PATTERN.match(until)):
                raise ValueError("since/until must be in YYYY-MM-DD format")
            params["time_range"] = json.dumps({"since": since, "until": until})
        else:
            if date_preset not in _VALID_DATE_PRESETS:
                raise ValueError(
                    f"date_preset must be one of {sorted(_VALID_DATE_PRESETS)}"
                )
            params["date_preset"] = date_preset

        result = _graph_request(
            "GET", _graph_path(target_id, "insights"), params=params
        )
        return _success(
            insights=result.get("data", []),
            next_link=(result.get("paging") or {}).get("next"),
        )
    except GraphAPIError as e:
        _log_graph_error(f"Error getting Meta Ads insights for {object_id}", e)
        return _graph_error(e)
    except Exception as e:
        logger.error("Error getting Meta Ads insights for %s: %s", object_id, e)
        return _error(str(e))


if __name__ == "__main__":
    mcp.run()
