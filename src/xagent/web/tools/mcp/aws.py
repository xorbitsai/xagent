import json
import logging
import os
import time
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from mcp.server.fastmcp import FastMCP

from .utils import setup_proxy_env

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aws-mcp")

# Ensure standard proxy environment variables are set to prevent hanging requests
setup_proxy_env()

mcp = FastMCP("aws-mcp")

# Refresh assumed-role credentials this many seconds before they expire, so a
# long-running tool call never starts with credentials about to lapse.
ASSUME_ROLE_EXPIRY_MARGIN_SECONDS = 300
ASSUME_ROLE_SESSION_NAME = "xagent-aws-mcp"
MAX_LOG_EVENTS = 100

# role_arn -> (credentials dict, unix expiry timestamp)
_assumed_role_cache: dict[str, tuple[dict[str, str], float]] = {}


def _success(**payload: Any) -> str:
    return json.dumps({"status": "success", **payload}, ensure_ascii=False, default=str)


def _error(message: str) -> str:
    return json.dumps({"status": "error", "message": message}, ensure_ascii=False)


def _require_base_credentials() -> None:
    missing = [
        name
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(f"Missing environment variable(s): {', '.join(missing)}")


def _resolve_region(region: str | None) -> str:
    return region or os.environ.get("AWS_REGION", "")


def _assume_role_credentials(role_arn: str, region: str) -> dict[str, str]:
    cached = _assumed_role_cache.get(role_arn)
    now = time.time()
    if cached and cached[1] - ASSUME_ROLE_EXPIRY_MARGIN_SECONDS > now:
        return cached[0]

    sts = boto3.client("sts", region_name=region)
    response = sts.assume_role(
        RoleArn=role_arn, RoleSessionName=ASSUME_ROLE_SESSION_NAME
    )
    creds = response["Credentials"]
    credentials = {
        "aws_access_key_id": creds["AccessKeyId"],
        "aws_secret_access_key": creds["SecretAccessKey"],
        "aws_session_token": creds["SessionToken"],
    }
    expiry = creds["Expiration"].timestamp()
    _assumed_role_cache[role_arn] = (credentials, expiry)
    return credentials


def _client(
    service: str, region: str | None = None, role_arn: str | None = None
) -> Any:
    """Build a boto3 client from the connector's env credentials.

    With role_arn set, base credentials are exchanged for temporary
    assumed-role credentials first (cached per ARN until shortly before
    expiry) — this is how cross-account read access works without storing a
    second key set.
    """
    _require_base_credentials()
    resolved_region = _resolve_region(region)
    if role_arn:
        credentials = _assume_role_credentials(role_arn, resolved_region)
        return boto3.client(service, region_name=resolved_region, **credentials)
    return boto3.client(service, region_name=resolved_region)


def _aws_error_message(exc: Exception) -> str:
    if isinstance(exc, ClientError):
        error = exc.response.get("Error", {})
        code = error.get("Code", "")
        message = error.get("Message", "")
        if code or message:
            return f"{code}: {message}" if code else message
    return str(exc)


@mcp.tool()
def aws_get_caller_identity(
    region: str | None = None, role_arn: str | None = None
) -> str:
    """
    Verify AWS connectivity and show which account/principal the connector is
    acting as. Use this first if any other AWS call fails with access errors.
    Optional role_arn assumes that IAM role (cross-account) before the call.
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
    state: optional filter, one of "ALARM", "OK", "INSUFFICIENT_DATA".
    alarm_name_prefix: optional name prefix filter.
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
            for alarm in result.get("MetricAlarms", [])
        ]
        return _success(alarms=alarms)
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
            StartTime=start_time,
            EndTime=end_time,
        )
        series = result.get("MetricDataResults", [])
        first = series[0] if series else {}
        return _success(
            timestamps=first.get("Timestamps", []),
            values=first.get("Values", []),
            label=first.get("Label"),
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
    """
    try:
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
            for group in result.get("logGroups", [])
        ]
        return _success(log_groups=groups)
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
    than raising the limit.
    """
    try:
        kwargs: dict[str, Any] = {
            "logGroupName": log_group_name,
            "limit": min(int(limit), MAX_LOG_EVENTS),
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
                "message": event.get("message"),
            }
            for event in result.get("events", [])
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
    """
    try:
        result = _client("dynamodb", region, role_arn).list_tables(Limit=100)
        return _success(tables=result.get("TableNames", []))
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
    """
    try:
        result = _client("dynamodb", region, role_arn).describe_table(
            TableName=table_name
        )
        table = result.get("Table", {})
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
                    for index in table.get("GlobalSecondaryIndexes", [])
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
    """
    try:
        kwargs: dict[str, Any] = {"MaxResults": 100}
        if name_prefix:
            kwargs["QueueNamePrefix"] = name_prefix
        result = _client("sqs", region, role_arn).list_queues(**kwargs)
        return _success(queue_urls=result.get("QueueUrls", []))
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
    """
    try:
        result = _client("sqs", region, role_arn).get_queue_attributes(
            QueueUrl=queue_url, AttributeNames=["All"]
        )
        return _success(attributes=result.get("Attributes", {}))
    except (ClientError, BotoCoreError, ValueError) as e:
        logger.error(f"Error getting SQS queue attributes for {queue_url}: {e}")
        return _error(_aws_error_message(e))


if __name__ == "__main__":
    mcp.run()
