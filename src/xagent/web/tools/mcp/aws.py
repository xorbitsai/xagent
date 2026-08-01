import json
import logging
import os
import re
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aws-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("aws-mcp")

ASSUME_ROLE_SESSION_NAME = "xagent-aws-mcp"
# Set by xagent.web.services.mcp_runtime.caller_id_env for every stdio MCP
# connector — kept as a literal here (not imported) since this module runs as
# a standalone subprocess entrypoint.
CALLER_ID_ENV_VAR = "XAGENT_MCP_CALLER_ID"
# RoleSessionName's allowed charset per AWS STS: [\w+=,.@-], max 64 chars.
_SESSION_NAME_UNSAFE_CHARS = re.compile(r"[^\w+=,.@-]")
MAX_LOG_EVENTS = 100
# CloudWatch's own server-side default (100,800) is far more than an LLM
# needs handed back in one response; cap to roughly a day of 1-minute data.
MAX_METRIC_DATAPOINTS = 1440
# CloudWatch log messages routinely carry a full stack trace or a serialized
# payload; MAX_LOG_EVENTS bounds event *count* but not size, so up to 100
# uncapped messages could still produce a multi-megabyte tool result that
# blows out the model's context. ~4000 chars is generous for a log line
# while keeping a full batch's worst case bounded.
MAX_LOG_MESSAGE_CHARS = 4000

# role_arn is a model-supplied ARN with no operator-side allowlist, so an
# assumed session's actual permissions could otherwise be as broad as
# whatever policy the target role happens to carry -- there's no IAM-level
# guarantee that role is itself read-only. An inline session policy caps
# what the *assumed* session can do to exactly the API calls this module
# makes, regardless of the target role's own permissions: read-only becomes
# an enforced property of every role_arn call, not just a documentation
# convention the operator has to trust. Session policies never grant more
# than the target role's own policy already allows; they only narrow it.
_READ_ONLY_SESSION_POLICY = json.dumps(
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Action": [
                    "sts:GetCallerIdentity",
                    "cloudwatch:DescribeAlarms",
                    "cloudwatch:GetMetricData",
                    "logs:DescribeLogGroups",
                    "logs:FilterLogEvents",
                    "dynamodb:ListTables",
                    "dynamodb:DescribeTable",
                    "sqs:ListQueues",
                    "sqs:GetQueueAttributes",
                    # Reading from a CloudWatch Logs group encrypted with a
                    # customer-managed KMS key requires these in addition to
                    # logs:FilterLogEvents -- without them, this session
                    # policy would regress an existing role_arn caller's
                    # access to an encrypted log group even though the
                    # target role's own permissions are unchanged.
                    "kms:Decrypt",
                    "kms:DescribeKey",
                ],
                "Resource": "*",
            }
        ],
    }
)

# boto3 defaults (60s connect/read timeout, up to 5 legacy retries) are too
# generous for a synchronous tool call the LLM is waiting on.
DEFAULT_CLIENT_CONFIG = Config(
    connect_timeout=10,
    read_timeout=30,
    retries={"max_attempts": 3, "mode": "standard"},
)


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False, default=str)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _require_base_credentials(resolved_region: str) -> None:
    missing = [
        name
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    # AWS_REGION is only required when the caller didn't already supply an
    # explicit region= argument - resolved_region already reflects that
    # override (see _resolve_region), so checking the env var directly here
    # would make an explicit region unable to satisfy the requirement.
    if not resolved_region:
        missing.append("AWS_REGION")
    if missing:
        raise ValueError(f"Missing environment variable(s): {', '.join(missing)}")


def _resolve_region(region: str | None) -> str:
    return region or os.environ.get("AWS_REGION", "")


def _role_session_name() -> str:
    """Build a RoleSessionName that attributes an AssumeRole call to the
    xagent user who triggered it, so the target account's CloudTrail can tell
    different users' cross-account reads apart. Falls back to the bare
    connector name if the caller id wasn't threaded through (e.g. a manual
    `python -m xagent.web.tools.mcp.aws` invocation outside the platform).

    This only covers the role_arn (cross-account) path — see _client. A call
    without role_arn goes straight through the connector's own base
    credentials, one shared IAM principal with no per-call session name, so
    same-account calls remain indistinguishable from each other in
    CloudTrail.
    """
    caller_id = os.environ.get(CALLER_ID_ENV_VAR)
    if not caller_id:
        return ASSUME_ROLE_SESSION_NAME
    sanitized = _SESSION_NAME_UNSAFE_CHARS.sub("_", caller_id)
    return f"{ASSUME_ROLE_SESSION_NAME}-{sanitized}"[:64]


def _assume_role_credentials(role_arn: str, region: str) -> dict[str, str | None]:
    # No caching: each MCP tool call runs in its own short-lived subprocess
    # (see _execute_mcp_call in mcp_adapter.py, which opens and tears down a
    # fresh stdio session per call), so a module-level cache never survives
    # past the single call that populated it.
    sts = boto3.Session().client(
        "sts", region_name=region, config=DEFAULT_CLIENT_CONFIG
    )
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName=_role_session_name(),
        Policy=_READ_ONLY_SESSION_POLICY,
    )
    creds = response.get("Credentials") or {}
    access_key_id = creds.get("AccessKeyId")
    secret_access_key = creds.get("SecretAccessKey")
    session_token = creds.get("SessionToken")
    # A 200 response with Credentials present but AccessKeyId/SecretAccessKey
    # both absent would otherwise spread as **{..., "aws_access_key_id": None,
    # "aws_secret_access_key": None, ...} into session.client(), which
    # botocore treats as "no explicit credentials supplied" and silently
    # falls back to the connector's own base/ambient credentials -- a
    # cross-account call would then run under the wrong identity with no
    # error at all. AWS's documented AssumeRole contract says this can't
    # happen on a 200, but failing loudly here costs nothing and closes the
    # gap if it ever does (a partial response, e.g. only one field missing,
    # already raises botocore's own PartialCredentialsError upstream).
    if not access_key_id or not secret_access_key or not session_token:
        raise ValueError(
            f"AssumeRole for {role_arn!r} returned an incomplete credential set"
        )
    return {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
        "aws_session_token": session_token,
    }


def _client(
    service: str, region: str | None = None, role_arn: str | None = None
) -> Any:
    """Build a boto3 client from the connector's env credentials.

    With role_arn set, base credentials are exchanged for temporary
    assumed-role credentials first — this is how cross-account read access
    works without storing a second key set. The exchange happens on every
    call: see the comment on _assume_role_credentials for why caching it
    would be dead code.

    A fresh boto3.Session per call, rather than the module-level
    boto3.client shortcut: boto3 Sessions are documented as not safe to
    share across concurrent client creation, so a fresh one per call is a
    low-cost precaution regardless of how this module happens to be invoked.
    """
    resolved_region = _resolve_region(region)
    _require_base_credentials(resolved_region)
    session = boto3.Session()
    if role_arn:
        credentials = _assume_role_credentials(role_arn, resolved_region)
        return session.client(
            service,
            region_name=resolved_region,
            config=DEFAULT_CLIENT_CONFIG,
            **credentials,
        )
    return session.client(
        service, region_name=resolved_region, config=DEFAULT_CLIENT_CONFIG
    )


def _aws_error_message(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error") or {}
        code = error.get("Code", "")
        message = error.get("Message", "")
        if code or message:
            return f"{code}: {message}" if code else message
    return str(exc)


def _truncate_message(
    message: str | None, max_chars: int = MAX_LOG_MESSAGE_CHARS
) -> str | None:
    # CloudWatch Logs documents `message` as always a string, but guard the
    # type anyway rather than let a TypeError from `+` on a non-str value
    # escape uncaught (the caller's except clause doesn't include TypeError).
    if message is None or not isinstance(message, str) or len(message) <= max_chars:
        return message
    return message[:max_chars] + f"...[truncated, {len(message)} chars total]"


@mcp.tool()
def aws_get_caller_identity(
    region: str | None = None, role_arn: str | None = None
) -> str:
    """
    Verify AWS connectivity and show which account/principal the connector is
    acting as. Use this first if any other AWS call fails with access errors.
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        result = _client("sts", region, role_arn).get_caller_identity()
        return _success(
            account=result.get("Account"),
            arn=result.get("Arn"),
            user_id=result.get("UserId"),
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error getting caller identity: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_cloudwatch_describe_alarms(
    state: str | None = None,
    alarm_name_prefix: str | None = None,
    region: str | None = None,
    role_arn: str | None = None,
) -> str:
    """
    List CloudWatch alarms — the triage entry point for "what is wrong right now".
    Returns both metric alarms and composite alarms (account-level rollups
    aggregating other alarms) separately.
    state: optional filter, one of "ALARM", "OK", "INSUFFICIENT_DATA".
    alarm_name_prefix: optional name prefix filter.
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        kwargs: dict[str, Any] = {"MaxRecords": 100}
        if state:
            kwargs["StateValue"] = state
        if alarm_name_prefix:
            kwargs["AlarmNamePrefix"] = alarm_name_prefix
        result = _client("cloudwatch", region, role_arn).describe_alarms(**kwargs)
        alarms = [
            {
                "name": alarm.get("AlarmName"),
                "state": alarm.get("StateValue"),
                "state_reason": alarm.get("StateReason"),
                "state_updated": alarm.get("StateUpdatedTimestamp"),
                "metric": alarm.get("MetricName"),
                "namespace": alarm.get("Namespace"),
                "dimensions": alarm.get("Dimensions"),
                "threshold": alarm.get("Threshold"),
                "comparison": alarm.get("ComparisonOperator"),
            }
            for alarm in (result.get("MetricAlarms") or [])
        ]
        # DescribeAlarms returns composite alarms (account-level "is the
        # service healthy" rollups aggregating leaf alarms) in a separate
        # top-level field sharing the same MaxRecords budget. Omitting them
        # would let this triage tool report "nothing is wrong" while the
        # account's top rollup alarm is firing.
        composite_alarms = [
            {
                "name": alarm.get("AlarmName"),
                "state": alarm.get("StateValue"),
                "state_reason": alarm.get("StateReason"),
                "state_updated": alarm.get("StateUpdatedTimestamp"),
                "alarm_rule": alarm.get("AlarmRule"),
            }
            for alarm in (result.get("CompositeAlarms") or [])
        ]
        return _success(
            alarms=alarms,
            composite_alarms=composite_alarms,
            truncated="NextToken" in result,
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error describing CloudWatch alarms: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_cloudwatch_get_metric_data(
    namespace: str,
    metric_name: str,
    start_time: str,
    end_time: str,
    period_seconds: int = 300,
    stat: str = "Average",
    dimensions: list[dict[str, str]] | None = None,
    region: str | None = None,
    role_arn: str | None = None,
) -> str:
    """
    Fetch a CloudWatch metric timeseries, e.g. CPUUtilization or queue depth over time.
    namespace: e.g. "AWS/EC2", "AWS/SQS", "AWS/DynamoDB".
    metric_name: e.g. "CPUUtilization", "ApproximateNumberOfMessagesVisible".
    start_time / end_time: ISO 8601 timestamps, e.g. "2026-07-30T00:00:00Z".
    stat: one of "Average", "Sum", "Minimum", "Maximum", "SampleCount".
    dimensions: e.g. [{"Name": "QueueName", "Value": "my-queue"}].
    period_seconds: CloudWatch's own retention tiers cap the resolution
    actually available for older data regardless of this value -- 1-minute
    for the last 15 days, 5-minute for 15-63 days, 1-hour beyond that --
    and CloudWatch rounds start_time down to align with whichever tier
    applies, so a small period_seconds for an old start_time won't add
    resolution back that was never retained.
    Returns at most 1440 datapoints — narrow the time range or widen
    period_seconds rather than expecting more in one call.
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        metric: dict[str, Any] = {"Namespace": namespace, "MetricName": metric_name}
        if dimensions:
            metric["Dimensions"] = dimensions
        result = _client("cloudwatch", region, role_arn).get_metric_data(
            MetricDataQueries=[
                {
                    "Id": "m1",
                    "MetricStat": {
                        "Metric": metric,
                        "Period": period_seconds,
                        "Stat": stat,
                    },
                }
            ],
            # botocore natively accepts ISO 8601 strings for timestamp
            # params (same as the AWS CLI); an unparsable string raises
            # ParamValidationError (a BotoCoreError), which the except
            # below already surfaces. Converting via datetime.fromisoformat
            # here would only narrow the accepted formats.
            StartTime=start_time,
            EndTime=end_time,
            MaxDatapoints=MAX_METRIC_DATAPOINTS,
        )
        series = result.get("MetricDataResults") or []
        first = (series[0] or {}) if series else {}
        status_code = first.get("StatusCode")
        return _success(
            timestamps=first.get("Timestamps", []),
            values=first.get("Values", []),
            label=first.get("Label"),
            status_code=status_code,
            # PartialData/InternalError mean CloudWatch could not fully
            # answer this query — surface that distinctly from "no data".
            partial=status_code is not None and status_code != "Complete",
            messages=result.get("Messages") or [],
            truncated="NextToken" in result,
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error getting CloudWatch metric data: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_cloudwatch_describe_log_groups(
    name_prefix: str | None = None,
    region: str | None = None,
    role_arn: str | None = None,
) -> str:
    """
    List CloudWatch Logs log groups (optionally filtered by name prefix), to
    discover where a service's logs live before searching them.
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        # 50, not 100 like the other list-style tools: DescribeLogGroups'
        # `limit` has a hard server-side max of 50 (unlike, e.g., MaxRecords
        # on describe_alarms), so 100 here would fail remotely.
        kwargs: dict[str, Any] = {"limit": 50}
        if name_prefix:
            kwargs["logGroupNamePrefix"] = name_prefix
        result = _client("logs", region, role_arn).describe_log_groups(**kwargs)
        groups = [
            {
                "name": group.get("logGroupName"),
                "stored_bytes": group.get("storedBytes"),
                "retention_days": group.get("retentionInDays"),
            }
            for group in (result.get("logGroups") or [])
        ]
        return _success(log_groups=groups, truncated="nextToken" in result)
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error describing log groups: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_cloudwatch_filter_log_events(
    log_group_name: str,
    filter_pattern: str | None = None,
    start_time_ms: int | None = None,
    end_time_ms: int | None = None,
    limit: int = MAX_LOG_EVENTS,
    region: str | None = None,
    role_arn: str | None = None,
) -> str:
    """
    Search a CloudWatch Logs group for matching events, e.g. errors during an incident window.
    filter_pattern: CloudWatch filter syntax, e.g. "ERROR" or "{ $.level = \"error\" }".
    start_time_ms / end_time_ms: unix epoch milliseconds bounding the search window.
    Returns at most `limit` events (default 100) — narrow the window or pattern rather
    than raising the limit. Each event's message is truncated past 4000 characters
    (MAX_LOG_MESSAGE_CHARS).
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        kwargs: dict[str, Any] = {
            "logGroupName": log_group_name,
            # Clamp to [1, MAX_LOG_EVENTS]: a zero/negative limit passes the
            # int signature validation but AWS rejects it remotely.
            "limit": max(1, min(int(limit), MAX_LOG_EVENTS)),
        }
        if filter_pattern:
            kwargs["filterPattern"] = filter_pattern
        if start_time_ms is not None:
            kwargs["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            kwargs["endTime"] = int(end_time_ms)
        result = _client("logs", region, role_arn).filter_log_events(**kwargs)
        events = [
            {
                "timestamp_ms": event.get("timestamp"),
                "log_stream": event.get("logStreamName"),
                "message": _truncate_message(event.get("message")),
            }
            for event in (result.get("events") or [])
        ]
        return _success(events=events, truncated="nextToken" in result)
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error filtering log events: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_dynamodb_list_tables(
    region: str | None = None, role_arn: str | None = None
) -> str:
    """
    List DynamoDB table names in the region.
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        result = _client("dynamodb", region, role_arn).list_tables(Limit=100)
        return _success(
            tables=result.get("TableNames") or [],
            truncated="LastEvaluatedTableName" in result,
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error listing DynamoDB tables: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_dynamodb_describe_table(
    table_name: str, region: str | None = None, role_arn: str | None = None
) -> str:
    """
    Get a DynamoDB table's status and health signals: state, item count, size,
    billing mode, and provisioned throughput (for throttling diagnostics).
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        result = _client("dynamodb", region, role_arn).describe_table(
            TableName=table_name
        )
        table = result.get("Table") or {}
        return _success(
            table={
                "name": table.get("TableName"),
                "status": table.get("TableStatus"),
                "item_count": table.get("ItemCount"),
                "size_bytes": table.get("TableSizeBytes"),
                "billing_mode": (table.get("BillingModeSummary") or {}).get(
                    "BillingMode"
                ),
                "provisioned_throughput": table.get("ProvisionedThroughput"),
                "global_secondary_indexes": [
                    {
                        "name": index.get("IndexName"),
                        "status": index.get("IndexStatus"),
                    }
                    for index in (table.get("GlobalSecondaryIndexes") or [])
                ],
            }
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error describing DynamoDB table {table_name}: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_sqs_list_queues(
    name_prefix: str | None = None,
    region: str | None = None,
    role_arn: str | None = None,
) -> str:
    """
    List SQS queue URLs in the region (optionally filtered by name prefix).
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        kwargs: dict[str, Any] = {"MaxResults": 100}
        if name_prefix:
            kwargs["QueueNamePrefix"] = name_prefix
        result = _client("sqs", region, role_arn).list_queues(**kwargs)
        return _success(
            queue_urls=result.get("QueueUrls") or [],
            truncated="NextToken" in result,
        )
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error listing SQS queues: {e}")
        return _error(_aws_error_message(e))


@mcp.tool()
def aws_sqs_get_queue_attributes(
    queue_url: str, region: str | None = None, role_arn: str | None = None
) -> str:
    """
    Get an SQS queue's attributes — most importantly backlog depth
    (ApproximateNumberOfMessages), in-flight count, oldest message age, and
    the redrive/DLQ policy — the key signals for "is this queue backed up".
    region: optional AWS region override; defaults to the connector's
    configured AWS_REGION.
    role_arn: optional IAM role ARN to assume (cross-account access) before
    the call -- reachable with no operator-side allowlist if the base
    credentials permit sts:AssumeRole, but the assumed session is scoped to
    this connector's own read-only calls regardless of the target role's
    own permissions.
    """
    try:
        result = _client("sqs", region, role_arn).get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=["All"]
        )
        return _success(attributes=result.get("Attributes") or {})
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error getting SQS queue attributes for {queue_url}: {e}")
        return _error(_aws_error_message(e))


if __name__ == "__main__":
    mcp.run()
