"""Offline reports for shared_worker_benchmark; no database/service calls.

Keep unsuccessful attempts and absent observations visible. First-model latency
is measured only by the timing launcher, never replaced by llm_call_start.
Warmup IDs are excluded, and completion is persisted task completion rather than
browser rendering or time-to-first-token. Quantiles use floor((n - 1) * q).
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict


def stats(values):
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": values[math.floor((len(values) - 1) * 0.5)],
        "p95": values[math.floor((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def metrics(rows):
    return {
        key: stats(row.get(key) for row in rows)
        for key in ("api_s", "first_model_s", "total_s")
    }


def read_case(folder):
    manifest = json.loads((folder / "manifest.json").read_text())
    records = [
        json.loads(line)
        for line in (folder / "workload.jsonl").read_text().splitlines()
        if line.strip()
    ]

    def one(kind):
        found = [row for row in records if row.get("type") == kind]
        if len(found) != 1:
            raise ValueError(f"Expected one {kind} record")
        return found[0]

    admission, done, probes = one("admission"), one("completed"), one("probes")
    task_rows = done["tasks"] + probes["tasks"]
    tasks = {row["id"]: row for row in task_rows}
    entries = defaultdict(list)
    for path in sorted(folder.glob("model-entries-*.json")):
        pid = int(path.stem.split("-")[-1])
        for row in json.loads(path.read_text()):
            entries[row["task_id"]].append({"time": row["time"], "pid": pid})
    issues = []

    def enrich(row):
        result = dict(row)
        result["api_s"] = row["api_ms"] / 1000
        persisted = tasks.get(row.get("id"))
        result["status"] = persisted["status"] if persisted else "unconfirmed"
        result["total_s"] = persisted["updated"] - row["sent"] if persisted else None
        first = min(
            entries.get(row.get("id"), []), key=lambda item: item["time"], default=None
        )
        result["first_model_s"] = first["time"] - row["sent"] if first else None
        result["first_model_pid"] = first["pid"] if first else None
        if first is None:
            issues.append("missing_model_entry")
        if any(
            result[key] is not None and result[key] < 0
            for key in ("api_s", "first_model_s", "total_s")
        ):
            issues.append("negative_timestamp_delta")
        if row.get("http") != 202 or result["status"] != "completed":
            issues.append("failed_or_unconfirmed_request")
        if len({entry["pid"] for entry in entries.get(row.get("id"), [])}) > 1:
            issues.append("multiple_execution_processes")
        return result

    background = [enrich(row) for row in admission["background"]]
    initial = [enrich(row) for row in admission["probes"]]
    continuous = [enrich(row) for row in probes["rows"]]
    all_rows = background + initial + continuous
    ids = {row.get("id") for row in all_rows}
    if len(ids) != len(all_rows) or None in ids:
        issues.append("duplicate_or_unknown_task_id")
    if (
        len(background) != manifest["background_tasks"]
        or len(initial) != 10
        or len(continuous) != manifest["continuous_probes"]
        or {row.get("ordinal") for row in continuous}
        != set(range(manifest["continuous_probes"]))
        or len(task_rows) != len(all_rows)
        or set(tasks) != ids
    ):
        issues.append("missing_or_duplicate_samples")
    if not manifest["complete"] or done.get("timeout") or probes.get("timeout"):
        issues.append("incomplete_case")
    if any(manifest["client_exit_codes"]) or manifest["forced_shutdown_pids"]:
        issues.append("unsuccessful_shutdown_or_client")
    if any(row.get("type") == "safety_stop" for row in records):
        issues.append("safety_stop")
    web_pids = {host["pid"] for host in manifest["hosts"] if host["role"] == "web"}
    if any(row["first_model_pid"] in web_pids for row in all_rows):
        issues.append("model_executed_in_web_process")
    warm_ids = {row["id"] for row in one("warm")["tasks"]}
    warm = manifest.get("prewarm")
    if manifest["prewarm_requested"]:
        if (
            not warm
            or not warm["complete"]
            or warm["warmed_pids"] != warm["expected_pids"]
        ):
            issues.append("incomplete_worker_warmup")
        else:
            tool_ids = set()
            for detail in warm["rounds"]:
                for record in [
                    json.loads(line)
                    for line in (folder / detail["file"]).read_text().splitlines()
                    if line.strip()
                ]:
                    if record["type"] == "admission":
                        tool_ids.update(row["id"] for row in record["background"])
                    if record["type"] in ("warm", "completed"):
                        warm_ids.update(row["id"] for row in record["tasks"])
                        if any(
                            row["updated"] > manifest["workload_started_at"]
                            for row in record["tasks"]
                        ):
                            issues.append("warm_tasks_overlapped_measurement")
            observed = {
                entry["pid"]
                for key in tool_ids
                for entry in entries.get(key, [])
                if entry["time"] < manifest["workload_started_at"]
            }
            if observed != set(warm["expected_pids"]):
                issues.append("missing_warm_model_observations")
    if warm_ids & ids:
        issues.append("warmup_ids_in_measurement")
    tools = {
        row["id"]: row for row in done["events"] if row["type"] == "tool_execution_end"
    }
    deviations = [
        row["id"]
        for row in background
        if tools.get(row["id"], {}).get("count") != 10
        or tools.get(row["id"], {}).get("success_count") != 10
    ]
    if deviations:
        issues.append("tool_workload_deviation")
    return {
        "manifest": manifest,
        "valid": not issues,
        "issues": dict(Counter(issues)),
        "background": metrics(background),
        "initial_probes": metrics(initial),
        "continuous_all": metrics(continuous),
        "background_by_worker": dict(
            Counter(row["first_model_pid"] for row in background)
        ),
        "tool_count_deviations": deviations,
        "raw": {
            "background": background,
            "initial_probes": initial,
            "continuous": continuous,
        },
    }


def compare_cases(cases):
    compatible = bool(cases) and all(case["valid"] for case in cases)
    # Do not silently compare unlike workloads or a cold case to a warm one.
    fields = (
        "background_tasks",
        "continuous_probes",
        "prewarm_requested",
        "probe_interval_s",
        "model_id",
        "prompt_sha256",
        "harness_sha256",
        "task_timeout_s",
        "source_head",
        "source_diff_sha256",
        "source_status",
    )
    if cases:
        compatible = compatible and all(
            not case["manifest"].get("source_status")
            and all(
                key in case["manifest"]
                and case["manifest"][key] == cases[0]["manifest"].get(key)
                for key in fields
            )
            for case in cases
        )
    common = (
        set.intersection(
            *[
                {
                    row["ordinal"]
                    for row in case["raw"]["continuous"]
                    if row["active_background"] == case["manifest"]["background_tasks"]
                }
                for case in cases
            ]
        )
        if compatible
        else set()
    )
    return {
        "valid": compatible and bool(common),
        "matched_full_load_ordinals": sorted(common),
        "matched_full_load": [
            metrics(
                [row for row in case["raw"]["continuous"] if row["ordinal"] in common]
            )
            for case in cases
        ],
        "cases": cases,
        "limitations": [
            "Sequential real-model runs; repeat and retain failed attempts.",
            "No enforced CPU quota; multiple workers can use more host cores.",
            "Warmup does not fix production cold starts or guarantee fair assignment.",
            "First-model means adapter entry, not network send or first token.",
            "Full load means all background tasks are marked RUNNING (admitted), not concurrent execution.",
            "Full-load matching is by submission ordinal, not identical wall-clock exposure.",
            "Comparison requires matching clean source revisions and nonempty matched samples.",
            "A completed run is not a production timeout SLO pass.",
        ],
    }
