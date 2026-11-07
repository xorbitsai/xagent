"""Opt-in real ReAct/Xinference long-context benchmark; see docs/agent-cache-benchmark.md.

Runs real model calls and local, synthetic invoice-audit tools. It does not launch
models, change server settings, touch a production database, or execute shell tools.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import statistics
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from xagent.core.agent import ExecutionContext, PatternRuntime, ReActPattern
from xagent.core.agent.context.execution import CompactConfig
from xagent.core.agent.runtime import load_pattern_checkpoint
from xagent.core.agent.trace import TraceCategory
from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.timeout_config import TimeoutConfig
from xagent.core.tools.adapters.vibe.base import ToolMetadata


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def source_manifest() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True
    )
    paths = [
        "scripts/benchmarks/agent_cache_benchmark.py",
        "src/xagent/core/agent/runtime.py",
        "src/xagent/core/agent/prompt_prefix.py",
        "src/xagent/core/agent/trace.py",
    ]
    return {
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in paths
        },
    }


def invoice_amount(index: int) -> int:
    return (
        100
        + int(hashlib.sha256(f"invoice-{index}".encode()).hexdigest()[:6], 16) % 99900
    )


def cache_summary(calls: list[dict[str, Any]]) -> dict[str, Any]:
    inputs = 0
    cached = 0
    samples = 0
    hit_latencies: list[float] = []
    for call in calls:
        usage = call.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        count = details.get("cached_tokens")
        total = usage.get("prompt_tokens")
        if (
            not isinstance(count, int)
            or not isinstance(total, int)
            or not 0 <= count <= total
        ):
            continue
        samples += 1
        inputs += total
        cached += count
        latency = call.get("adapter_first_payload_s")
        if count > 0 and isinstance(latency, (int, float)):
            hit_latencies.append(latency)
    return {
        "cache_usage_samples": samples,
        "cache_observed_input_tokens": inputs,
        "reported_cached_tokens": cached if samples else None,
        "cached_input_fraction": cached / inputs if inputs else None,
        "cache_hit_latency_samples": len(hit_latencies),
        "cache_hit_first_payload_median_s": statistics.median(hit_latencies)
        if hit_latencies
        else None,
    }


def build_ledger(target_tokens: int, tokenizer: Any) -> tuple[str, int]:
    """Size the synthetic corpus with the caller's local model tokenizer.

    Reserve 4096 tokens for the harness, schemas and task. Actual request token
    counts come from serving usage, and are reported separately from this target.
    """

    def ledger(rows: int) -> str:
        return "\n".join(
            f"invoice-{i:06d} | amount_cents={invoice_amount(i)} | currency=USD | status=posted"
            for i in range(1, rows + 1)
        )

    low, high = 1, target_tokens
    budget = target_tokens - 4096
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenizer.encode(ledger(middle)).ids) <= budget:
            low = middle
        else:
            high = middle - 1
    return ledger(low), low


class CheckInvoiceArgs(BaseModel):
    invoice_id: str
    amount_cents: int
    previous_receipt: str


class ReadInvoiceArgs(BaseModel):
    invoice_id: str


class SumAmountsArgs(BaseModel):
    amounts_cents: list[int]


class SaveAuditArgs(BaseModel):
    receipts: list[str]
    total_cents: int


@dataclass
class AuditWorkload:
    targets: list[int]
    output: Path
    fail_at: int = 2
    receipts: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    failed_once: bool = False
    saved: bool = False

    @property
    def total(self) -> int:
        return sum(invoice_amount(i) for i in self.targets)

    def check(self, args: dict[str, Any]) -> dict[str, Any]:
        position = len(self.receipts)
        self.attempts.append(dict(args))
        if position >= len(self.targets):
            return {"success": False, "error": "All invoices checked; save the audit."}
        expected_id = f"invoice-{self.targets[position]:06d}"
        previous = self.receipts[-1] if self.receipts else "START"
        if (args["invoice_id"], args["amount_cents"], args["previous_receipt"]) != (
            expected_id,
            invoice_amount(self.targets[position]),
            previous,
        ):
            return {
                "success": False,
                "error": "Invoice, amount or prior receipt mismatch. Re-read the ledger and last receipt.",
            }
        if position + 1 == self.fail_at and not self.failed_once:
            self.failed_once = True
            return {
                "success": False,
                "retryable": True,
                "error": "Synthetic transient failure; nothing committed. Retry these arguments.",
            }
        receipt = uuid4().hex
        self.receipts.append(receipt)
        return {
            "success": True,
            "receipt": receipt,
            "next_invoice": f"invoice-{self.targets[position + 1]:06d}"
            if position + 1 < len(self.targets)
            else None,
            "instruction": "Check the next invoice using this receipt. When next_invoice is null, use sum_amounts to compute the exact total, then save_audit with all successful receipts in order.",
        }

    def save(self, args: dict[str, Any]) -> dict[str, Any]:
        if len(self.receipts) != len(self.targets):
            return {
                "success": False,
                "error": "Audit incomplete; follow next_invoice until all invoices are checked.",
            }
        if args["receipts"] != self.receipts:
            return {
                "success": False,
                "error": "Receipt order mismatch; copy each successful tool receipt in order.",
            }
        if args["total_cents"] != self.total:
            return {
                "success": False,
                "error": "Incorrect total; use sum_amounts with the successfully verified invoice amounts, then retry.",
            }
        payload = {
            "verified": True,
            "targets": self.targets,
            "receipts": self.receipts,
            "total_cents": self.total,
        }
        write_json(self.output / "audit.json", payload)
        self.saved = True
        return {
            "success": True,
            "saved_to_workspace": True,
            "file_path": str(self.output / "audit.json"),
            **payload,
        }


class SumAmountsTool:
    metadata = ToolMetadata(
        name="sum_amounts",
        description="Compute the exact sum of verified invoice amounts in cents. Use this tool before saving the audit total.",
        read_only=True,
    )

    def args_type(self) -> type[BaseModel]:
        return SumAmountsArgs

    async def run_json_async(self, args: dict[str, Any]) -> dict[str, Any]:
        return {"success": True, "total_cents": sum(args["amounts_cents"])}


class AuditTool:
    def __init__(
        self, workload: AuditWorkload, *, save: bool = False, read: bool = False
    ):
        self.workload = workload
        self.save = save
        self.read = read
        self.metadata = ToolMetadata(
            name="save_audit" if save else "check_invoice",
            description="Save the completed audit with all receipts in order and total cents."
            if save
            else "Verify a ledger invoice amount. Requires previous receipt (START for first); returns a fresh receipt and the next invoice id. Retry transient errors with the same arguments.",
        )
        if read:
            self.metadata = ToolMetadata(
                name="read_ledger_invoice",
                description="Read one invoice from the saved reference ledger if compaction removed its text from context.",
                read_only=True,
            )

    def args_type(self) -> type[BaseModel]:
        if self.read:
            return ReadInvoiceArgs
        return SaveAuditArgs if self.save else CheckInvoiceArgs

    async def run_json_async(self, args: dict[str, Any]) -> Any:
        if self.read:
            for line in (self.workload.output / "ledger.txt").read_text().splitlines():
                invoice_id, amount, *_ = line.split(" | ")
                if invoice_id == args["invoice_id"]:
                    return {
                        "success": True,
                        "invoice_id": invoice_id,
                        "amount_cents": int(amount.split("=")[1]),
                    }
            return {
                "success": False,
                "error": "Invoice not found in the reference ledger",
            }
        return self.workload.save(args) if self.save else self.workload.check(args)


class BenchmarkLLM(XinferenceLLM):
    def __init__(self, *, output: Path, context_window: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.output = output
        self.context_window = context_window
        self.calls: list[dict[str, Any]] = []

    def record(self, row: dict[str, Any]) -> None:
        self.calls.append(row)
        with (self.output / "calls.jsonl").open("a") as fd:
            fd.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    async def stream_chat(self, **kwargs: Any):
        start = time.perf_counter()
        row: dict[str, Any] = {
            "call": len(self.calls) + 1,
            "mode": "stream",
            "adapter_first_payload_s": None,
        }
        try:
            kwargs.setdefault("thinking", {"type": "disabled"})
            async for chunk in super().stream_chat(**kwargs):
                if row["adapter_first_payload_s"] is None and (
                    chunk.content or chunk.tool_calls
                ):
                    row["adapter_first_payload_s"] = time.perf_counter() - start
                if chunk.usage:
                    row["usage"] = chunk.usage
                yield chunk
            row["success"] = True
        except (Exception, asyncio.CancelledError, GeneratorExit) as exc:
            row.update(success=False, error_type=type(exc).__name__)
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - start
            self.record(row)

    async def chat(self, **kwargs: Any):
        start = time.perf_counter()
        row: dict[str, Any] = {"call": len(self.calls) + 1, "mode": "nonstream"}
        try:
            kwargs.setdefault("thinking", {"type": "disabled"})
            result = await super().chat(**kwargs)
            row.update(
                success=True,
                usage=result.get("usage", {}) if isinstance(result, dict) else {},
            )
            return result
        except (Exception, asyncio.CancelledError, GeneratorExit) as exc:
            row.update(success=False, error_type=type(exc).__name__)
            raise
        finally:
            row["elapsed_s"] = time.perf_counter() - start
            self.record(row)


class BenchmarkTrace:
    # This local harness persists checkpoints, not the web service's durable
    # tool-outcome/event store. Do not claim its exactly-once recovery contract.
    records_execution_events = False

    def __init__(self, output: Path):
        self.output = output
        self.events: list[dict[str, Any]] = []

    async def checkpoint(self, **payload: Any) -> None:
        temporary = self.output / "checkpoint.tmp"
        write_json(temporary, payload)
        temporary.replace(self.output / "checkpoint.json")

    async def trace_event(self, event_type: Any, **kwargs: Any) -> None:
        if event_type.category not in {
            TraceCategory.LLM,
            TraceCategory.TOOL,
            TraceCategory.COMPACT,
        }:
            return
        data = kwargs.get("data", {})
        fields = (
            "iteration",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "prompt_prefix",
            "prompt_composition",
            "success",
            "tool_name",
            "strategy",
            "removed_count",
        )
        row = {
            "category": event_type.category.value,
            "action": event_type.action.value,
            **{key: data[key] for key in fields if key in data},
        }
        self.events.append(row)
        with (self.output / "events.jsonl").open("a") as fd:
            fd.write(json.dumps(row) + "\n")


class BenchmarkRuntime(PatternRuntime):
    def __init__(
        self,
        *,
        workload: AuditWorkload,
        compact_after: int,
        compaction_attempted: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.workload = workload
        self.compact_after = compact_after
        self.compaction_attempted = compaction_attempted

    async def compact_context_if_needed(self, *, context: Any, **kwargs: Any):
        force = (
            self.compact_after > 0
            and not self.compaction_attempted
            and len(self.workload.receipts) >= self.compact_after
        )
        if not force:
            return await super().compact_context_if_needed(context=context, **kwargs)
        self.compaction_attempted = True
        threshold = context.compact_config.threshold
        context.compact_config.threshold = min(threshold, 8192)
        try:
            return await super().compact_context_if_needed(context=context, **kwargs)
        finally:
            context.compact_config.threshold = threshold


def build_context(
    ledger: str, first_invoice: int, context_window: int, *, compaction: bool
) -> ExecutionContext:
    context = ExecutionContext(
        system_prompt="You audit invoices using the supplied ledger and tools. Complete the entire audit and save it before answering. Never invent amounts or receipts.",
        compact_config=CompactConfig(threshold=int(context_window * 0.9)),
    )
    context.add_user_message(
        f"Reference ledger for the following audit task:\n{ledger}"
    )
    request = f"Audit the invoice chain starting with invoice-{first_invoice:06d}. Read each amount_cents from the reference ledger. Call check_invoice once at a time, passing previous_receipt=START initially and the returned receipt thereafter. Follow next_invoice until null; retry explicitly retryable failures. Then use sum_amounts to compute the total of successfully verified amounts, call save_audit with every successful receipt in order and that total, and report the saved audit."
    if compaction:
        request += " If compaction removes ledger text, read_ledger_invoice can read it from the saved reference file."
    context.add_user_message(request)
    return context


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from tokenizers import Tokenizer

    args.output.mkdir(parents=True, exist_ok=False)
    source = source_manifest()
    write_json(args.output / "source.json", source)
    tokenizer = Tokenizer.from_file(str(args.tokenizer_json))
    ledger, rows = build_ledger(args.context_tokens, tokenizer)
    (args.output / "ledger.txt").write_text(ledger)
    if rows < args.rounds:
        raise ValueError("Corpus must contain at least one row per audit round")
    targets = [
        1 + i * (rows - 1) // max(1, args.rounds - 1) for i in range(args.rounds)
    ]
    workload = AuditWorkload(targets=targets, output=args.output, fail_at=args.fail_at)
    trace = BenchmarkTrace(args.output)
    llm = BenchmarkLLM(
        output=args.output,
        context_window=args.context_window,
        model_name=args.model,
        model_uid=args.model,
        base_url=args.endpoint,
        api_key=args.api_key_file.read_text().strip() if args.api_key_file else None,
        default_temperature=0,
        default_max_tokens=1024,
        timeout=args.timeout,
        timeout_config=TimeoutConfig(args.timeout, 60),
        abilities=["chat", "tool_calling"],
    )
    context = build_context(
        ledger, targets[0], args.context_window, compaction=bool(args.compact_after)
    )

    def tools() -> list[Any]:
        items: list[Any] = [
            AuditTool(workload),
            AuditTool(workload, save=True),
            SumAmountsTool(),
        ]
        if args.compact_after:
            items.append(AuditTool(workload, read=True))
        return items

    def pattern() -> ReActPattern:
        return ReActPattern(
            max_iterations=args.rounds * 3 + 8,
            user_interaction_enabled=False,
            repeated_tool_decision_after_consecutive_tool_calls=None,
            repeated_tool_decision_after_consecutive_work_tool_calls=None,
        )

    runtime = BenchmarkRuntime(
        workload=workload,
        compact_after=args.compact_after,
        tracer=trace,
        interrupt_checker=(lambda: len(workload.receipts) >= args.resume_after)
        if args.resume_after
        else None,
    )
    started = time.perf_counter()
    resumed = False
    result: dict[str, Any] = {}
    error_type = None
    try:
        result = await pattern().run(
            context=context,
            tools=tools(),
            llm=llm,
            compact_llm=llm,
            runtime=runtime,
        )
        if args.resume_after and result.get("status") == "interrupted":
            checkpoint = runtime.last_checkpoint
            if checkpoint is None:
                raise RuntimeError("Interrupted without checkpoint")
            write_json(args.output / "resume-checkpoint.json", checkpoint)
            write_json(
                args.output / "audit-state.json",
                {
                    "receipts": workload.receipts,
                    "attempts": workload.attempts,
                    "failed_once": workload.failed_once,
                    "saved": workload.saved,
                    "compaction_attempted": runtime.compaction_attempted,
                },
            )
            checkpoint = json.loads(
                (args.output / "resume-checkpoint.json").read_text()
            )
            saved_state = json.loads((args.output / "audit-state.json").read_text())
            compaction_attempted = saved_state.pop("compaction_attempted")
            workload = AuditWorkload(
                targets=targets, output=args.output, fail_at=args.fail_at, **saved_state
            )
            context = ExecutionContext.from_dict(checkpoint["context"])
            next_pattern = pattern()
            load_pattern_checkpoint(next_pattern, checkpoint)
            runtime = BenchmarkRuntime(
                workload=workload,
                compact_after=args.compact_after,
                compaction_attempted=compaction_attempted,
                tracer=trace,
            )
            resumed = True
            result = await next_pattern.run(
                context=context,
                tools=tools(),
                llm=llm,
                compact_llm=llm,
                runtime=runtime,
            )
    except Exception as exc:
        error_type = type(exc).__name__
        traceback.print_exc()
    finally:
        await llm.close()
    latencies = sorted(
        c["adapter_first_payload_s"]
        for c in llm.calls
        if c.get("adapter_first_payload_s") is not None
    )
    compactions = [
        event
        for event in trace.events
        if event["category"] == "compact"
        and event["action"] == "end"
        and event.get("removed_count", 0) > 0
    ]
    sum_tool_exercised = any(
        event["category"] == "tool"
        and event["action"] == "end"
        and event.get("tool_name") == "sum_amounts"
        for event in trace.events
    )
    report = {
        "model": args.model,
        "source": source,
        "context_window": args.context_window,
        "context_target_tokens": args.context_tokens,
        "corpus_tokens": len(tokenizer.encode(ledger).ids),
        "corpus_rows": rows,
        "rounds": args.rounds,
        "elapsed_s": time.perf_counter() - started,
        "passed": result.get("success") is True
        and workload.saved
        and sum_tool_exercised
        and (not args.resume_after or resumed)
        and (not args.compact_after or bool(compactions)),
        "result_status": result.get("status"),
        "error_type": error_type,
        "verified_invoices": len(workload.receipts),
        "tool_attempts": len(workload.attempts),
        "transient_failure_exercised": workload.failed_once,
        "sum_tool_exercised": sum_tool_exercised,
        "resumed": resumed,
        "compaction_attempted": runtime.compaction_attempted,
        "compactions": compactions,
        "calls": llm.calls,
        "adapter_first_payload_p95_s": latencies[math.ceil(0.95 * len(latencies)) - 1]
        if latencies
        else None,
        "latency_samples": len(latencies),
        **cache_summary(llm.calls),
    }
    write_json(args.output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key-file", type=Path)
    parser.add_argument("--tokenizer-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-tokens", type=int, default=65536)
    parser.add_argument("--context-window", type=int, default=131072)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--fail-at", type=int, default=2)
    parser.add_argument("--resume-after", type=int, default=0)
    parser.add_argument("--compact-after", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=600)
    args = parser.parse_args()
    if (
        args.context_tokens <= 4096
        or args.context_tokens + 2048 > args.context_window
        or args.rounds < 1
        or not 0 <= args.resume_after <= args.rounds
        or not 0 <= args.compact_after <= args.rounds
        or not 0 <= args.fail_at <= args.rounds
    ):
        parser.error(
            "Require context-tokens > 4096 with 2048 tokens of window headroom, rounds >= 1, and failure/resume/compaction positions between 0 and rounds"
        )
    report = asyncio.run(run(args))
    print(
        json.dumps({key: value for key, value in report.items() if key != "calls"}),
        flush=True,
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
