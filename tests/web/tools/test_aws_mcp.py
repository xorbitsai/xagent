import json
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from xagent.web.tools.mcp import aws


@pytest.fixture(autouse=True)
def _credentials(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA-test")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-test")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


def _client_error(code: str, message: str, operation: str = "Op") -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": message}}, operation)


def test_missing_credentials_reported(monkeypatch):
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY")

    result = json.loads(aws.aws_get_caller_identity())

    assert result["status"] == "error"
    assert "AWS_SECRET_ACCESS_KEY" in result["message"]


def test_get_caller_identity_returns_account_and_arn(monkeypatch):
    sts = Mock()
    sts.get_caller_identity.return_value = {
        "Account": "123456789012",
        "Arn": "arn:aws:iam::123456789012:user/devops-readonly",
        "UserId": "AIDATEST",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=sts))

    result = json.loads(aws.aws_get_caller_identity())

    assert result["status"] == "success"
    assert result["account"] == "123456789012"
    assert "devops-readonly" in result["arn"]


def test_client_error_surfaces_code_and_message(monkeypatch):
    sts = Mock()
    sts.get_caller_identity.side_effect = _client_error(
        "AccessDenied", "User is not authorized"
    )
    monkeypatch.setattr(aws, "_client", Mock(return_value=sts))

    result = json.loads(aws.aws_get_caller_identity())

    assert result["status"] == "error"
    assert "AccessDenied" in result["message"]
    assert "not authorized" in result["message"]


def test_describe_alarms_passes_state_filter_and_shapes_output(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.describe_alarms.return_value = {
        "MetricAlarms": [
            {
                "AlarmName": "high-cpu",
                "StateValue": "ALARM",
                "StateReason": "Threshold crossed",
                "MetricName": "CPUUtilization",
                "Namespace": "AWS/EC2",
                "Threshold": 90.0,
                "ComparisonOperator": "GreaterThanThreshold",
            }
        ]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(aws.aws_cloudwatch_describe_alarms(state="ALARM"))

    assert result["status"] == "success"
    assert result["alarms"][0]["name"] == "high-cpu"
    assert result["alarms"][0]["state"] == "ALARM"
    assert result["truncated"] is False
    assert cloudwatch.describe_alarms.call_args.kwargs["StateValue"] == "ALARM"


def test_describe_alarms_flags_truncated_when_next_token_present(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.describe_alarms.return_value = {
        "MetricAlarms": [{"AlarmName": "high-cpu"}],
        "NextToken": "more",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(aws.aws_cloudwatch_describe_alarms())

    assert result["truncated"] is True


def test_describe_alarms_omits_filters_when_not_given(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.describe_alarms.return_value = {"MetricAlarms": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(aws.aws_cloudwatch_describe_alarms())

    assert result["status"] == "success"
    assert result["alarms"] == []
    kwargs = cloudwatch.describe_alarms.call_args.kwargs
    assert "StateValue" not in kwargs
    assert "AlarmNamePrefix" not in kwargs


def test_get_metric_data_builds_query_and_returns_series(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.get_metric_data.return_value = {
        "MetricDataResults": [
            {
                "Label": "CPUUtilization",
                "Timestamps": ["2026-07-30T00:00:00Z"],
                "Values": [42.5],
                "StatusCode": "Complete",
            }
        ]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(
        aws.aws_cloudwatch_get_metric_data(
            namespace="AWS/EC2",
            metric_name="CPUUtilization",
            start_time="2026-07-30T00:00:00Z",
            end_time="2026-07-30T01:00:00Z",
            stat="Maximum",
            dimensions=[{"Name": "InstanceId", "Value": "i-123"}],
        )
    )

    assert result["status"] == "success"
    assert result["values"] == [42.5]
    assert result["status_code"] == "Complete"
    assert result["partial"] is False
    query = cloudwatch.get_metric_data.call_args.kwargs["MetricDataQueries"][0]
    assert query["MetricStat"]["Stat"] == "Maximum"
    assert query["MetricStat"]["Metric"]["Dimensions"] == [
        {"Name": "InstanceId", "Value": "i-123"}
    ]
    assert (
        cloudwatch.get_metric_data.call_args.kwargs["MaxDatapoints"]
        == aws.MAX_METRIC_DATAPOINTS
    )


def test_get_metric_data_handles_empty_series(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.get_metric_data.return_value = {"MetricDataResults": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(
        aws.aws_cloudwatch_get_metric_data(
            namespace="AWS/EC2",
            metric_name="CPUUtilization",
            start_time="2026-07-30T00:00:00Z",
            end_time="2026-07-30T01:00:00Z",
        )
    )

    assert result["status"] == "success"
    assert result["timestamps"] == []
    assert result["values"] == []
    assert result["partial"] is False


def test_get_metric_data_flags_partial_data_and_surfaces_messages(monkeypatch):
    """StatusCode other than "Complete" (PartialData/InternalError) means
    CloudWatch could not fully answer the query — this must be distinguishable
    from a clean "no data in this range" result."""
    cloudwatch = Mock()
    cloudwatch.get_metric_data.return_value = {
        "MetricDataResults": [
            {
                "Label": "CPUUtilization",
                "Timestamps": [],
                "Values": [],
                "StatusCode": "PartialData",
            }
        ],
        "Messages": [{"Code": "PartialData", "Value": "Some data unavailable"}],
        "NextToken": "next-page",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(
        aws.aws_cloudwatch_get_metric_data(
            namespace="AWS/EC2",
            metric_name="CPUUtilization",
            start_time="2026-07-30T00:00:00Z",
            end_time="2026-07-30T01:00:00Z",
        )
    )

    assert result["status"] == "success"
    assert result["status_code"] == "PartialData"
    assert result["partial"] is True
    assert result["messages"] == [
        {"Code": "PartialData", "Value": "Some data unavailable"}
    ]
    assert result["truncated"] is True


def test_tools_tolerate_explicit_none_values_in_responses(monkeypatch):
    """Keys present with an explicit None value (as opposed to absent) must
    not crash list/dict handling — the `or []` / `or {}` fallbacks apply."""
    client = Mock()
    client.describe_alarms.return_value = {"MetricAlarms": None}
    client.get_metric_data.return_value = {"MetricDataResults": None}
    client.list_tables.return_value = {"TableNames": None}
    client.describe_table.return_value = {"Table": None}
    client.list_queues.return_value = {"QueueUrls": None}
    client.get_queue_attributes.return_value = {"Attributes": None}
    monkeypatch.setattr(aws, "_client", Mock(return_value=client))

    assert json.loads(aws.aws_cloudwatch_describe_alarms())["alarms"] == []
    metric = json.loads(
        aws.aws_cloudwatch_get_metric_data(
            namespace="AWS/EC2",
            metric_name="CPUUtilization",
            start_time="2026-07-30T00:00:00Z",
            end_time="2026-07-30T01:00:00Z",
        )
    )
    assert metric["status"] == "success" and metric["values"] == []
    assert json.loads(aws.aws_dynamodb_list_tables())["tables"] == []
    assert json.loads(aws.aws_dynamodb_describe_table("orders"))["status"] == "success"
    assert json.loads(aws.aws_sqs_list_queues())["queue_urls"] == []
    assert json.loads(aws.aws_sqs_get_queue_attributes("https://q"))["attributes"] == {}


def test_filter_log_events_caps_limit_and_passes_window(monkeypatch):
    logs = Mock()
    logs.filter_log_events.return_value = {
        "events": [
            {"timestamp": 1753900000000, "logStreamName": "s1", "message": "ERROR boom"}
        ],
        "nextToken": "more",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(
            log_group_name="/app/prod",
            filter_pattern="ERROR",
            start_time_ms=1753890000000,
            end_time_ms=1753900000000,
            limit=5000,
        )
    )

    assert result["status"] == "success"
    assert result["events"][0]["message"] == "ERROR boom"
    assert result["truncated"] is True
    kwargs = logs.filter_log_events.call_args.kwargs
    assert kwargs["limit"] == aws.MAX_LOG_EVENTS  # capped, not 5000
    assert kwargs["filterPattern"] == "ERROR"
    assert kwargs["startTime"] == 1753890000000


def test_filter_log_events_not_truncated_without_next_token(monkeypatch):
    logs = Mock()
    logs.filter_log_events.return_value = {"events": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(log_group_name="/app/prod")
    )

    assert result["truncated"] is False


def test_describe_log_groups_filters_by_prefix(monkeypatch):
    logs = Mock()
    logs.describe_log_groups.return_value = {
        "logGroups": [{"logGroupName": "/app/prod", "storedBytes": 1024}]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(aws.aws_cloudwatch_describe_log_groups(name_prefix="/app"))

    assert result["status"] == "success"
    assert result["log_groups"][0]["name"] == "/app/prod"
    assert result["truncated"] is False
    assert logs.describe_log_groups.call_args.kwargs["logGroupNamePrefix"] == "/app"


def test_describe_log_groups_flags_truncated_when_next_token_present(monkeypatch):
    logs = Mock()
    logs.describe_log_groups.return_value = {
        "logGroups": [{"logGroupName": "/app/prod"}],
        "nextToken": "more",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(aws.aws_cloudwatch_describe_log_groups())

    assert result["truncated"] is True


def test_dynamodb_list_tables(monkeypatch):
    dynamodb = Mock()
    dynamodb.list_tables.return_value = {
        "TableNames": ["orders", "users"],
        "LastEvaluatedTableName": "users",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=dynamodb))

    result = json.loads(aws.aws_dynamodb_list_tables())

    assert result["status"] == "success"
    assert result["tables"] == ["orders", "users"]
    assert result["truncated"] is True


def test_dynamodb_list_tables_not_truncated_without_last_evaluated_key(monkeypatch):
    dynamodb = Mock()
    dynamodb.list_tables.return_value = {"TableNames": ["orders"]}
    monkeypatch.setattr(aws, "_client", Mock(return_value=dynamodb))

    result = json.loads(aws.aws_dynamodb_list_tables())

    assert result["truncated"] is False


def test_dynamodb_describe_table_shapes_health_fields(monkeypatch):
    dynamodb = Mock()
    dynamodb.describe_table.return_value = {
        "Table": {
            "TableName": "orders",
            "TableStatus": "ACTIVE",
            "ItemCount": 12345,
            "TableSizeBytes": 987654,
            "BillingModeSummary": {"BillingMode": "PAY_PER_REQUEST"},
            "GlobalSecondaryIndexes": [
                {"IndexName": "by-user", "IndexStatus": "ACTIVE"}
            ],
        }
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=dynamodb))

    result = json.loads(aws.aws_dynamodb_describe_table("orders"))

    assert result["status"] == "success"
    assert result["table"]["status"] == "ACTIVE"
    assert result["table"]["billing_mode"] == "PAY_PER_REQUEST"
    assert result["table"]["global_secondary_indexes"] == [
        {"name": "by-user", "status": "ACTIVE"}
    ]


def test_sqs_list_queues_passes_prefix(monkeypatch):
    sqs = Mock()
    sqs.list_queues.return_value = {
        "QueueUrls": ["https://sqs.us-east-1.amazonaws.com/1/jobs"]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=sqs))

    result = json.loads(aws.aws_sqs_list_queues(name_prefix="jobs"))

    assert result["status"] == "success"
    assert result["queue_urls"] == ["https://sqs.us-east-1.amazonaws.com/1/jobs"]
    assert result["truncated"] is False
    assert sqs.list_queues.call_args.kwargs["QueueNamePrefix"] == "jobs"


def test_sqs_list_queues_flags_truncated_when_next_token_present(monkeypatch):
    sqs = Mock()
    sqs.list_queues.return_value = {
        "QueueUrls": ["https://sqs.us-east-1.amazonaws.com/1/jobs"],
        "NextToken": "more",
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=sqs))

    result = json.loads(aws.aws_sqs_list_queues())

    assert result["truncated"] is True


def test_sqs_get_queue_attributes_requests_all(monkeypatch):
    sqs = Mock()
    sqs.get_queue_attributes.return_value = {
        "Attributes": {"ApproximateNumberOfMessages": "42"}
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=sqs))

    result = json.loads(
        aws.aws_sqs_get_queue_attributes("https://sqs.us-east-1.amazonaws.com/1/jobs")
    )

    assert result["status"] == "success"
    assert result["attributes"]["ApproximateNumberOfMessages"] == "42"
    assert sqs.get_queue_attributes.call_args.kwargs["AttributeNames"] == ["All"]


# --- assume-role plumbing -------------------------------------------------


def _sts_with_assume_role(expires_in_seconds: int) -> Mock:
    sts = Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIA-temp",
            "SecretAccessKey": "temp-secret",
            "SessionToken": "temp-token",
            "Expiration": datetime.now(timezone.utc)
            + timedelta(seconds=expires_in_seconds),
        }
    }
    return sts


def _patch_boto3_session(monkeypatch, client_factory) -> Mock:
    """Patch boto3.Session so every new session's .client() delegates to
    client_factory; returns the shared client Mock for call assertions."""
    session_client = Mock(side_effect=client_factory)
    session = Mock()
    session.client = session_client
    monkeypatch.setattr(aws.boto3, "Session", Mock(return_value=session))
    return session_client


def test_client_with_role_arn_assumes_role_on_every_call(monkeypatch):
    """Each MCP tool call runs in its own short-lived subprocess, so there is
    no in-process caching layer to test — every _client() call with a
    role_arn must exchange it for fresh temporary credentials."""
    sts = _sts_with_assume_role(expires_in_seconds=3600)
    service_client = Mock()
    session_client = _patch_boto3_session(
        monkeypatch,
        lambda service, **kwargs: sts if service == "sts" else service_client,
    )

    first = aws._client("sqs", role_arn="arn:aws:iam::2:role/read")
    second = aws._client("sqs", role_arn="arn:aws:iam::2:role/read")

    assert first is service_client and second is service_client
    assert sts.assume_role.call_count == 2
    sqs_calls = [c for c in session_client.call_args_list if c.args[0] == "sqs"]
    assert sqs_calls[0].kwargs["aws_access_key_id"] == "ASIA-temp"
    assert sqs_calls[0].kwargs["aws_session_token"] == "temp-token"
    assert sqs_calls[0].kwargs["config"] == aws.DEFAULT_CLIENT_CONFIG
    sts_calls = [c for c in session_client.call_args_list if c.args[0] == "sts"]
    assert sts_calls[0].kwargs["config"] == aws.DEFAULT_CLIENT_CONFIG


def test_role_session_name_falls_back_without_caller_id(monkeypatch):
    monkeypatch.delenv(aws.CALLER_ID_ENV_VAR, raising=False)

    assert aws._role_session_name() == aws.ASSUME_ROLE_SESSION_NAME


def test_role_session_name_includes_sanitized_caller_id(monkeypatch):
    monkeypatch.setenv(aws.CALLER_ID_ENV_VAR, "42")

    assert aws._role_session_name() == f"{aws.ASSUME_ROLE_SESSION_NAME}-42"


def test_role_session_name_sanitizes_unsafe_characters(monkeypatch):
    """RoleSessionName's allowed charset is [\\w+=,.@-]; anything else (e.g. a
    slash or space) would make sts.assume_role reject the call outright."""
    monkeypatch.setenv(aws.CALLER_ID_ENV_VAR, "weird/id with space")

    name = aws._role_session_name()

    assert "/" not in name and " " not in name
    assert name.startswith(f"{aws.ASSUME_ROLE_SESSION_NAME}-")


def test_role_session_name_truncates_to_64_chars(monkeypatch):
    monkeypatch.setenv(aws.CALLER_ID_ENV_VAR, "x" * 100)

    name = aws._role_session_name()

    assert len(name) == 64


def test_assume_role_credentials_uses_caller_attributed_session_name(monkeypatch):
    monkeypatch.setenv(aws.CALLER_ID_ENV_VAR, "7")
    sts = _sts_with_assume_role(expires_in_seconds=3600)
    _patch_boto3_session(monkeypatch, lambda service, **kwargs: sts)

    aws._assume_role_credentials("arn:aws:iam::2:role/read", "us-east-1")

    assert (
        sts.assume_role.call_args.kwargs["RoleSessionName"]
        == f"{aws.ASSUME_ROLE_SESSION_NAME}-7"
    )


def test_client_without_role_arn_uses_env_credentials(monkeypatch):
    session_client = _patch_boto3_session(monkeypatch, lambda service, **kwargs: Mock())

    aws._client("cloudwatch")

    kwargs = session_client.call_args.kwargs
    # no explicit creds → env chain; a timeout/retry config is still applied
    assert kwargs == {
        "region_name": "us-east-1",
        "config": aws.DEFAULT_CLIENT_CONFIG,
    }


def test_client_creates_a_fresh_session_per_call(monkeypatch):
    """The default shared boto3 session is not thread-safe for concurrent
    client creation; each _client call must build its own Session."""
    session_factory = Mock(side_effect=lambda: Mock(client=Mock(return_value=Mock())))
    monkeypatch.setattr(aws.boto3, "Session", session_factory)

    aws._client("cloudwatch")
    aws._client("sqs")

    assert session_factory.call_count == 2


def test_filter_log_events_clamps_non_positive_limit(monkeypatch):
    logs = Mock()
    logs.filter_log_events.return_value = {"events": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(log_group_name="/app/prod", limit=0)
    )

    assert result["status"] == "success"
    assert logs.filter_log_events.call_args.kwargs["limit"] == 1
