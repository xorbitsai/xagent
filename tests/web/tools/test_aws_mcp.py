import json
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


def test_describe_alarms_surfaces_composite_alarms(monkeypatch):
    """DescribeAlarms returns composite (rollup) alarms in a separate
    top-level field from MetricAlarms; a firing composite alarm must not be
    silently dropped from this triage tool's output."""
    cloudwatch = Mock()
    cloudwatch.describe_alarms.return_value = {
        "MetricAlarms": [],
        "CompositeAlarms": [
            {
                "AlarmName": "service-health-rollup",
                "StateValue": "ALARM",
                "StateReason": "1 out of 3 alarms in ALARM",
                "AlarmRule": "ALARM(high-cpu) OR ALARM(high-latency)",
            }
        ],
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    result = json.loads(aws.aws_cloudwatch_describe_alarms())

    assert result["status"] == "success"
    assert result["alarms"] == []
    assert result["composite_alarms"][0]["name"] == "service-health-rollup"
    assert result["composite_alarms"][0]["state"] == "ALARM"
    assert result["composite_alarms"][0]["alarm_rule"] == (
        "ALARM(high-cpu) OR ALARM(high-latency)"
    )


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


def _sts_with_assume_role() -> Mock:
    sts = Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": "ASIA-temp",
            "SecretAccessKey": "temp-secret",
            "SessionToken": "temp-token",
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
    sts = _sts_with_assume_role()
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
    sts = _sts_with_assume_role()
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


def test_client_creates_independent_sessions_for_sts_and_service_on_role_arn_path(
    monkeypatch,
):
    """On the role_arn (cross-account) path, a single _client() call must
    create two independent boto3.Session objects -- one for the STS
    assume_role exchange (inside _assume_role_credentials), one for the
    final service client -- neither reused from the other.
    test_client_creates_a_fresh_session_per_call never passes role_arn, and
    _patch_boto3_session's shared-Session mock can't tell "one Session
    reused across both calls" from "two independent Sessions"."""
    sessions: list[Mock] = []

    def make_client(service: str, **kwargs) -> Mock:
        client = Mock()
        if service == "sts":
            client.assume_role.return_value = {
                "Credentials": {
                    "AccessKeyId": "ASIA-temp",
                    "SecretAccessKey": "temp-secret",
                    "SessionToken": "temp-token",
                }
            }
        return client

    def session_factory() -> Mock:
        session = Mock()
        session.client = Mock(side_effect=make_client)
        sessions.append(session)
        return session

    monkeypatch.setattr(aws.boto3, "Session", Mock(side_effect=session_factory))

    aws._client("cloudwatch", role_arn="arn:aws:iam::2:role/read")

    assert len(sessions) == 2
    assert sessions[0] is not sessions[1]


def test_filter_log_events_clamps_non_positive_limit(monkeypatch):
    logs = Mock()
    logs.filter_log_events.return_value = {"events": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(log_group_name="/app/prod", limit=0)
    )

    assert result["status"] == "success"
    assert logs.filter_log_events.call_args.kwargs["limit"] == 1


def test_filter_log_events_truncates_oversized_messages(monkeypatch):
    """CloudWatch messages can carry a full stack trace or serialized
    payload; MAX_LOG_EVENTS bounds count but not size, so an individual
    message must still be capped to keep a full batch's worst case
    bounded (PR review finding R3-m2)."""
    long_message = "x" * (aws.MAX_LOG_MESSAGE_CHARS + 500)
    logs = Mock()
    logs.filter_log_events.return_value = {
        "events": [{"timestamp": 1, "logStreamName": "s1", "message": long_message}]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(log_group_name="/app/prod")
    )

    message = result["events"][0]["message"]
    assert len(message) < len(long_message)
    assert message.startswith("x" * 100)
    assert "truncated" in message


def test_filter_log_events_leaves_short_messages_untouched(monkeypatch):
    logs = Mock()
    logs.filter_log_events.return_value = {
        "events": [{"timestamp": 1, "logStreamName": "s1", "message": "short"}]
    }
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    result = json.loads(
        aws.aws_cloudwatch_filter_log_events(log_group_name="/app/prod")
    )

    assert result["events"][0]["message"] == "short"


def test_truncate_message_passes_through_non_str_without_raising():
    """CloudWatch Logs documents `message` as always a string, but
    _truncate_message must not raise TypeError on a non-str value - the
    caller's except clause doesn't include TypeError, so an uncaught one
    would crash the tool call instead of degrading gracefully."""
    assert aws._truncate_message(None) is None
    assert aws._truncate_message(12345) == 12345


# --- server-side page-size caps (PR review finding R3-m6) -----------------


def test_describe_log_groups_sends_the_50_item_hard_cap(monkeypatch):
    logs = Mock()
    logs.describe_log_groups.return_value = {"logGroups": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=logs))

    aws.aws_cloudwatch_describe_log_groups()

    assert logs.describe_log_groups.call_args.kwargs["limit"] == 50


def test_describe_alarms_sends_the_100_record_cap(monkeypatch):
    cloudwatch = Mock()
    cloudwatch.describe_alarms.return_value = {"MetricAlarms": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=cloudwatch))

    aws.aws_cloudwatch_describe_alarms()

    assert cloudwatch.describe_alarms.call_args.kwargs["MaxRecords"] == 100


def test_dynamodb_list_tables_sends_the_100_item_cap(monkeypatch):
    dynamodb = Mock()
    dynamodb.list_tables.return_value = {"TableNames": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=dynamodb))

    aws.aws_dynamodb_list_tables()

    assert dynamodb.list_tables.call_args.kwargs["Limit"] == 100


def test_sqs_list_queues_sends_the_100_result_cap(monkeypatch):
    sqs = Mock()
    sqs.list_queues.return_value = {"QueueUrls": []}
    monkeypatch.setattr(aws, "_client", Mock(return_value=sqs))

    aws.aws_sqs_list_queues()

    assert sqs.list_queues.call_args.kwargs["MaxResults"] == 100


# --- explicit region= plumbing (PR review finding R3-m6) -------------------


def test_client_passes_explicit_region_to_boto3(monkeypatch):
    session_client = _patch_boto3_session(monkeypatch, lambda service, **kwargs: Mock())

    aws._client("cloudwatch", region="eu-west-1")

    assert session_client.call_args.kwargs["region_name"] == "eu-west-1"


def test_tool_call_with_explicit_region_reaches_boto3(monkeypatch):
    """Regression test for R3-m6: an explicit region= argument passed to a
    real tool function (not just _client() directly) must reach the
    eventual boto3 client call. The no-AWS_REGION-env case is covered
    separately by
    test_explicit_region_satisfies_the_region_requirement_without_env_var."""
    dynamodb = Mock()
    dynamodb.list_tables.return_value = {"TableNames": []}
    session_client = _patch_boto3_session(
        monkeypatch, lambda service, **kwargs: dynamodb
    )

    result = json.loads(aws.aws_dynamodb_list_tables(region="ap-southeast-2"))

    assert result["status"] == "success"
    assert session_client.call_args.kwargs["region_name"] == "ap-southeast-2"


def test_explicit_region_satisfies_the_region_requirement_without_env_var(monkeypatch):
    """Regression test for R3-m1: _require_base_credentials used to check
    the AWS_REGION env var directly, so an explicit region= argument could
    not satisfy it - _client would raise even though the caller supplied
    everything it needed."""
    monkeypatch.delenv("AWS_REGION")
    _patch_boto3_session(monkeypatch, lambda service, **kwargs: Mock())

    # Must not raise.
    aws._client("sts", region="us-west-2")


def test_missing_region_without_explicit_override_still_reported(monkeypatch):
    monkeypatch.delenv("AWS_REGION")

    result = json.loads(aws.aws_get_caller_identity())

    assert result["status"] == "error"
    assert "AWS_REGION" in result["message"]


# --- malformed AssumeRole response (PR review finding R3-M1) ---------------


def test_assume_role_credentials_rejects_null_credential_fields(monkeypatch):
    """AWS's documented AssumeRole contract says a 200 always returns a
    complete Credentials block, but if it ever didn't, spreading
    {"aws_access_key_id": None, ...} into session.client() would make
    botocore silently fall back to the connector's own base/ambient
    credentials - a cross-account call would then run under the wrong
    identity with no error at all. Must fail loudly instead."""
    sts = Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": None,
            "SecretAccessKey": None,
            "SessionToken": None,
        }
    }
    _patch_boto3_session(monkeypatch, lambda service, **kwargs: sts)

    with pytest.raises(ValueError, match="incomplete credential set"):
        aws._assume_role_credentials("arn:aws:iam::2:role/read", "us-east-1")


def test_client_with_role_arn_surfaces_null_credentials_as_a_clean_error(monkeypatch):
    """End-to-end: a tool call must translate the malformed-response
    ValueError into the standard error envelope, not an unhandled crash."""
    sts = Mock()
    sts.assume_role.return_value = {
        "Credentials": {
            "AccessKeyId": None,
            "SecretAccessKey": "temp-secret",
            "SessionToken": "temp-token",
        }
    }
    _patch_boto3_session(monkeypatch, lambda service, **kwargs: sts)

    result = json.loads(
        aws.aws_get_caller_identity(role_arn="arn:aws:iam::2:role/read")
    )

    assert result["status"] == "error"
    assert "incomplete credential set" in result["message"]
