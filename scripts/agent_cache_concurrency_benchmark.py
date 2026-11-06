"""Concurrent real ReAct audits and fixed-arrival foreground requests.

This optional benchmark never launches/reconfigures a server. ``baseline`` only
suppresses agent hints in this harness. ``lifecycle`` and ``lmcache`` send the
same hints; the operator must configure and record the storage backend separately.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import math
import statistics
import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from scripts import agent_cache_benchmark as audit
from xagent.core.agent import PatternRuntime, ReActPattern
from xagent.core.agent.runner import _session_routing_key
from xagent.core.model.chat.routing_context import xinference_routing_key


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class AdmissionGate:
    """Shared request limit; drain in-flight work before a thermal cooldown.

    All agents and foreground calls share this gate. Its queue is client-side,
    not an engine scheduler. Only intervals with zero in-flight requests count
    as cooling idle time; per-request admission waits must never be summed and
    subtracted from concurrent batch wall time.
    """

    def __init__(
        self,
        limit: int,
        telemetry: Path | None = None,
        *,
        trigger: float = 70,
        resume: float = 60,
    ):
        self.limit = limit
        self.telemetry = telemetry
        self.trigger = trigger
        self.resume = resume
        self.active = 0
        self.condition = asyncio.Condition()
        self.waiters: deque[object] = deque()
        self.cooling = False
        self.idle_started: float | None = None
        self.cooling_intervals: list[dict[str, Any]] = []

    def temperatures(self) -> list[dict[str, Any]]:
        if self.telemetry is None:
            return []
        rows = list(json.loads(self.telemetry.read_text()).values())
        now = datetime.datetime.now(datetime.timezone.utc)
        if not rows or any(
            (now - datetime.datetime.fromisoformat(row["utc"])).total_seconds() > 10
            for row in rows
        ):
            raise RuntimeError("Missing or stale GPU telemetry")
        return rows

    def can_enter(self) -> bool:
        rows = self.temperatures()
        thermal = any(
            row["sw_thermal_throttle"] or row["hw_thermal_throttle"] for row in rows
        )
        if thermal or any(row["temperature_C"] > self.trigger for row in rows):
            self.cooling = True
        if self.cooling:
            if not self.active and self.idle_started is None:
                self.idle_started = time.perf_counter()
            if (
                not self.active
                and not thermal
                and all(row["temperature_C"] <= self.resume for row in rows)
            ):
                self.cooling = False
                self.finish_idle()
            else:
                if (
                    self.idle_started is not None
                    and time.perf_counter() - self.idle_started > 600
                ):
                    raise RuntimeError("GPU cooling exceeded ten minutes")
                return False
        return self.active < self.limit

    def finish_idle(self) -> None:
        if self.idle_started is not None:
            self.cooling_intervals.append(
                {
                    "elapsed_s": time.perf_counter() - self.idle_started,
                    "utc_finished": utc(),
                }
            )
            self.idle_started = None

    @asynccontextmanager
    async def slot(self):
        requested = time.perf_counter()
        ticket = object()
        async with self.condition:
            self.waiters.append(ticket)
            try:
                while self.waiters[0] is not ticket or not self.can_enter():
                    try:
                        await asyncio.wait_for(self.condition.wait(), timeout=0.5)
                    except asyncio.TimeoutError:
                        pass
                self.waiters.popleft()
                self.active += 1
                self.condition.notify_all()
            except BaseException:
                self.waiters.remove(ticket)
                self.condition.notify_all()
                raise
        try:
            yield {"admission_wait_s": time.perf_counter() - requested}
        finally:
            async with self.condition:
                self.active -= 1
                if self.cooling and not self.active and self.idle_started is None:
                    self.idle_started = time.perf_counter()
                self.condition.notify_all()


class ConcurrentLLM(audit.BenchmarkLLM):
    def __init__(self, *, gate: AdmissionGate, hints: bool, **kwargs: Any):
        super().__init__(**kwargs)
        self.gate = gate
        self.hints = hints
        self.protocol: dict[str, bool] = {}
        self.admission: dict[str, float] = {}
        self.control: list[dict[str, Any]] = []

    def _apply_session_affinity(self, config: dict[str, Any]) -> None:
        if self.hints:
            super()._apply_session_affinity(config)
        self.protocol["routing_key_sent"] = "routing_key" in config

    def _apply_cache_lifecycle(self, config: dict[str, Any]) -> None:
        if self.hints:
            super()._apply_cache_lifecycle(config)
        self.protocol["cache_session_sent"] = "cache_session" in config

    async def _update_cache_session(self, session: dict[str, Any]) -> None:
        row = {"phase": session["phase"], "revision": session["revision"], "utc": utc()}
        try:
            await super()._update_cache_session(session)
            row["success"] = True
        except Exception:
            row["success"] = False
            raise
        finally:
            self.control.append(row)

    def record(self, row: dict[str, Any]) -> None:
        row.update(self.protocol, **self.admission, utc_recorded=utc())
        super().record(row)

    async def stream_chat(self, **kwargs: Any):
        async with self.gate.slot() as admission:
            self.admission = admission
            async for chunk in super().stream_chat(**kwargs):
                yield chunk

    async def chat(self, **kwargs: Any):
        async with self.gate.slot() as admission:
            self.admission = admission
            return await super().chat(**kwargs)


class WaitingAuditTool(audit.AuditTool):
    def __init__(self, workload: audit.AuditWorkload, *, delay: float, **kwargs: Any):
        super().__init__(workload, **kwargs)
        self.delay = delay
        self.waited = False

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        result = await super().run_json_async(args)
        if (
            not self.save
            and not self.read
            and not self.waited
            and result.get("success")
        ):
            self.waited = True
            await asyncio.sleep(self.delay)
        return result


def model(
    args: argparse.Namespace, output: Path, gate: AdmissionGate, *, hints: bool
) -> ConcurrentLLM:
    return ConcurrentLLM(
        output=output,
        gate=gate,
        hints=hints,
        context_window=args.context_window,
        model_name=args.model,
        model_uid=args.model,
        base_url=args.endpoint,
        api_key=args.api_key_file.read_text().strip() if args.api_key_file else None,
        default_temperature=0,
        default_max_tokens=1024,
        timeout=args.timeout,
        timeout_config=audit.TimeoutConfig(args.timeout, 60),
        abilities=["chat", "tool_calling"],
    )


async def run_agent(
    args: argparse.Namespace,
    index: int,
    ledger: str,
    rows: int,
    gate: AdmissionGate,
    scope: str,
) -> dict[str, Any]:
    output = args.output / f"agent-{index:02d}"
    output.mkdir()
    (output / "ledger.txt").write_text(ledger)
    workload = audit.AuditWorkload([1, rows], output, fail_at=2)
    trace = audit.BenchmarkTrace(output)
    runtime = PatternRuntime(tracer=trace)
    llm = model(args, output, gate, hints=args.variant != "baseline")
    context = audit.build_context(ledger, 1, args.context_window, compaction=False)
    context.system_prompt = (
        f"Experiment scope {scope}-{index}.\n" + context.system_prompt
    )
    routing_key = _session_routing_key(context, f"{scope}-{index}")
    token = xinference_routing_key.set(routing_key)
    delay = args.tool_delays[index % len(args.tool_delays)]
    start = time.perf_counter()
    started_utc = utc()
    result: dict[str, Any] = {}
    error = None
    try:
        result = await ReActPattern(
            max_iterations=14,
            user_interaction_enabled=False,
            repeated_tool_decision_after_consecutive_tool_calls=None,
            repeated_tool_decision_after_consecutive_work_tool_calls=None,
        ).run(
            context=context,
            tools=[
                WaitingAuditTool(workload, delay=delay),
                audit.AuditTool(workload, save=True),
                audit.SumAmountsTool(),
            ],
            llm=llm,
            compact_llm=llm,
            runtime=runtime,
        )
    except Exception as exc:
        error = type(exc).__name__
    finally:
        try:
            await runtime.finish_cache_session()
            publisher = runtime._cache_session._publisher
            if publisher is not None:
                await asyncio.wait_for(asyncio.shield(publisher), timeout=2)
        finally:
            xinference_routing_key.reset(token)
            await llm.close()
    calls = llm.calls
    expected_hints = args.variant != "baseline"
    protocol = bool(calls) and all(
        call.get("routing_key_sent") is expected_hints
        and call.get("cache_session_sent") is expected_hints
        for call in calls
    )
    first_usage = (calls[0].get("usage") or {}) if calls else {}
    first_cached = (first_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    waiting = any(row["phase"] == "waiting" and row["success"] for row in llm.control)
    finished = any(row["phase"] == "finished" and row["success"] for row in llm.control)
    tools = {
        row.get("tool_name")
        for row in trace.events
        if row["category"] == "tool" and row["action"] == "end"
    }
    saved = json.loads((output / "audit.json").read_text()) if workload.saved else {}
    verified = (
        bool(saved)
        and saved.get("verified") is True
        and saved.get("targets") == [1, rows]
        and saved.get("receipts") == workload.receipts
        and saved.get("total_cents") == workload.total
        and len(set(workload.receipts)) == 2
    )
    report = {
        "agent": index,
        "scope": f"{scope}-{index}",
        "routing_key": routing_key,
        "utc_started": started_utc,
        "utc_finished": utc(),
        "elapsed_s": time.perf_counter() - start,
        "tool_delay_s": delay,
        "passed": result.get("success") is True
        and verified
        and protocol
        and first_cached == 0
        and workload.failed_once
        and {"check_invoice", "sum_amounts", "save_audit"} <= tools
        and (not expected_hints or waiting and finished),
        "audit_verified": verified,
        "protocol_valid": protocol,
        "initial_cached_tokens": first_cached,
        "transient_failure_exercised": workload.failed_once,
        "error_type": error,
        "waiting_acknowledged": waiting,
        "finished_acknowledged": finished,
        "control_updates": llm.control,
        "calls": calls,
        **audit.cache_summary(calls),
    }
    audit.write_json(output / "report.json", report)
    return report


async def run_probe(
    args: argparse.Namespace,
    index: int,
    start: float,
    tokenizer: Any,
    gate: AdmissionGate,
    scope: str,
) -> dict[str, Any]:
    scheduled = start + args.probe_start + index * args.probe_interval
    await asyncio.sleep(max(0, scheduled - time.perf_counter()))
    output = args.output / f"foreground-{index:02d}"
    output.mkdir()
    llm = model(args, output, gate, hints=False)
    text = (
        f"Independent request {scope}-{index}.\n"
        + "Inventory and delivery records are complete. " * args.probe_tokens
    )
    text = tokenizer.decode(tokenizer.encode(text).ids[: args.probe_tokens])
    text += "\nReply with OK only."
    error = None
    passed = False
    try:
        content = []
        async for chunk in llm.stream_chat(
            messages=[{"role": "user", "content": text}], max_tokens=8
        ):
            content.append(chunk.content or "")
        passed = "OK" in "".join(content).upper()
    except Exception as exc:
        error = type(exc).__name__
    finally:
        await llm.close()
    cold_probe = len(llm.calls) == 1 and (
        ((llm.calls[0].get("usage") or {}).get("prompt_tokens_details") or {}).get(
            "cached_tokens"
        )
        == 0
        and not llm.calls[0].get("routing_key_sent")
        and not llm.calls[0].get("cache_session_sent")
    )
    report = {
        "foreground": index,
        "passed": passed and cold_probe,
        "error_type": error,
        "scheduled_offset_s": scheduled - start,
        "utc_finished": utc(),
        "scheduled_to_finish_s": time.perf_counter() - scheduled,
        "calls": llm.calls,
    }
    audit.write_json(output / "report.json", report)
    return report


def summarize(
    agents: list[dict[str, Any]],
    probes: list[dict[str, Any]],
    elapsed: float,
    cooling: float,
) -> dict[str, Any]:
    valid_agents = [row for row in agents if row.get("passed")]
    latencies = [row["scheduled_to_finish_s"] for row in probes if row.get("passed")]
    adjusted = max(0, elapsed - cooling)
    return {
        "passed": len(valid_agents) == len(agents)
        and all(row.get("passed") for row in probes),
        "successful_agents": len(valid_agents),
        "attempted_agents": len(agents),
        "successful_foreground": len(latencies),
        "attempted_foreground": len(probes),
        "batch_wall_s": elapsed,
        "cooling_idle_s": cooling,
        "batch_without_cooling_s": adjusted,
        "successful_agents_per_minute_wall": len(valid_agents) * 60 / elapsed
        if elapsed
        else None,
        "successful_agents_per_minute_without_cooling": len(valid_agents)
        * 60
        / adjusted
        if adjusted
        else None,
        "foreground_scheduled_latency_median_s": statistics.median(latencies)
        if latencies
        else None,
        "foreground_scheduled_latency_max_s": max(latencies) if latencies else None,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from tokenizers import Tokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    ledger, rows = audit.build_ledger(args.context_tokens, tokenizer)
    if rows < 2:
        raise ValueError("Corpus needs at least two invoices")
    scope = uuid4().hex
    gate = AdmissionGate(
        args.max_inflight,
        args.gpu_telemetry_json,
        trigger=args.cooldown_trigger,
        resume=args.cooldown_resume,
    )
    source = audit.source_manifest()
    source["source_sha256"]["scripts/agent_cache_concurrency_benchmark.py"] = (
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    )
    audit.write_json(args.output / "source.json", source)
    start_utc = utc()
    start = time.perf_counter()
    jobs = [
        asyncio.create_task(run_agent(args, index, ledger, rows, gate, scope))
        for index in range(args.agents)
    ]
    jobs.extend(
        asyncio.create_task(run_probe(args, index, start, tokenizer, gate, scope))
        for index in range(args.probes)
    )
    try:
        outcomes = await asyncio.gather(*jobs, return_exceptions=True)
    except BaseException:
        for job in jobs:
            job.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        raise
    gate.finish_idle()
    elapsed = time.perf_counter() - start

    def outcome(value: Any) -> dict[str, Any]:
        return (
            {"passed": False, "error_type": type(value).__name__}
            if isinstance(value, BaseException)
            else value
        )

    agents = [outcome(row) for row in outcomes[: args.agents]]
    probes = [outcome(row) for row in outcomes[args.agents :]]
    cooling = sum(row["elapsed_s"] for row in gate.cooling_intervals)
    report = {
        "variant": args.variant,
        "source": source,
        "scope": scope,
        "utc_started": start_utc,
        "utc_finished": utc(),
        "configuration": {
            key: getattr(args, key)
            for key in [
                "model",
                "agents",
                "context_tokens",
                "context_window",
                "max_inflight",
                "tool_delays",
                "probes",
                "probe_tokens",
                "probe_interval",
                "probe_start",
                "cooldown_trigger",
                "cooldown_resume",
            ]
        },
        **summarize(agents, probes, elapsed, cooling),
        "cooling_intervals": gate.cooling_intervals,
        "agents": agents,
        "foreground": probes,
    }
    audit.write_json(args.output / "report.json", report)
    return report


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--endpoint", required=True)
    value.add_argument("--model", required=True)
    value.add_argument("--api-key-file", type=Path)
    value.add_argument("--tokenizer-json", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument(
        "--variant", choices=["baseline", "lifecycle", "lmcache"], required=True
    )
    value.add_argument("--agents", type=int, choices=[2, 4], required=True)
    value.add_argument("--context-tokens", type=int, default=65536)
    value.add_argument("--context-window", type=int, default=131072)
    value.add_argument("--max-inflight", type=int, default=2)
    value.add_argument("--tool-delays", type=float, nargs="+", default=[2, 20])
    value.add_argument("--probes", type=int, default=8)
    value.add_argument("--probe-tokens", type=int, default=8192)
    value.add_argument("--probe-interval", type=float, default=10)
    value.add_argument("--probe-start", type=float, default=5)
    value.add_argument("--gpu-telemetry-json", type=Path)
    value.add_argument("--cooldown-trigger", type=float, default=70)
    value.add_argument("--cooldown-resume", type=float, default=60)
    value.add_argument("--timeout", type=float, default=1200)
    return value


def main() -> None:
    cli = parser()
    args = cli.parse_args()
    if (
        args.context_tokens <= 4096
        or args.context_tokens + 2048 > args.context_window
        or args.max_inflight < 1
        or args.probes < 1
        or args.probe_tokens < 1
        or args.probe_tokens + 2048 > args.context_window
        or args.probe_interval <= 0
        or args.probe_start < 0
        or args.timeout <= 0
        or not 0 < args.cooldown_resume < args.cooldown_trigger
        or any(delay < 0 for delay in args.tool_delays)
        or not all(
            math.isfinite(value)
            for value in [
                args.probe_interval,
                args.probe_start,
                args.timeout,
                args.cooldown_trigger,
                args.cooldown_resume,
                *args.tool_delays,
            ]
        )
    ):
        cli.error(
            "Require valid context headroom, positive request limits/intervals, and nonnegative tool delays/start"
        )
    report = asyncio.run(run(args))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"agents", "foreground"}
            }
        ),
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
