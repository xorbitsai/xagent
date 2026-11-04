from __future__ import annotations

import argparse
import asyncio
import json
import sys

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from scripts import agent_cache_benchmark
from scripts.agent_cache_benchmark import (
    AuditTool,
    AuditWorkload,
    BenchmarkLLM,
    build_context,
    cache_summary,
    invoice_amount,
)
from xagent.core.agent import PatternRuntime
from xagent.core.agent.trace import (
    TraceAction,
    TraceCategory,
    TraceEventType,
    TraceScope,
)
from xagent.core.model.chat.basic.xinference import XinferenceLLM
from xagent.core.model.chat.types import ChunkType, StreamChunk


@pytest.mark.parametrize("compaction", [False, True])
def test_context_setup_keeps_audit_request_and_recovery_guidance(compaction):
    context = build_context("reference-ledger", 42, 131072, compaction=compaction)
    request = context.current_user_request_text()
    assert "Audit the invoice chain starting with invoice-000042" in request
    assert ("read_ledger_invoice" in request) is compaction
    assert len(context.messages) == 2
    assert "reference-ledger" in context.messages[0].content


def test_audit_requires_real_ordered_receipts_and_retries(tmp_path):
    workload = AuditWorkload([1, 100], tmp_path)
    assert (
        workload.save({"receipts": [], "total_cents": workload.total})["success"]
        is False
    )
    assert (
        workload.check(
            {
                "invoice_id": "invoice-000001",
                "amount_cents": 0,
                "previous_receipt": "START",
            }
        )["success"]
        is False
    )
    first = workload.check(
        {
            "invoice_id": "invoice-000001",
            "amount_cents": invoice_amount(1),
            "previous_receipt": "START",
        }
    )
    args = {
        "invoice_id": "invoice-000100",
        "amount_cents": invoice_amount(100),
        "previous_receipt": first["receipt"],
    }
    assert workload.check(args)["retryable"] is True
    assert len(workload.receipts) == 1
    assert workload.check(args)["success"] is True
    assert (
        workload.save(
            {
                "receipts": list(reversed(workload.receipts)),
                "total_cents": workload.total,
            }
        )["success"]
        is False
    )
    result = workload.save(
        {"receipts": workload.receipts, "total_cents": workload.total}
    )
    assert result["saved_to_workspace"] is True
    assert json.loads((tmp_path / "audit.json").read_text())["verified"] is True


def test_cache_summary_distinguishes_missing_zero_and_weighted_hits():
    assert (
        cache_summary([{"usage": {"prompt_tokens": 10}}])["reported_cached_tokens"]
        is None
    )
    missing = {"usage": {"prompt_tokens": 100}}
    cold = {
        "usage": {"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 0}}
    }
    hit = {
        "usage": {"prompt_tokens": 90, "prompt_tokens_details": {"cached_tokens": 80}},
        "adapter_first_payload_s": 2.0,
    }
    result = cache_summary([missing, cold, hit])
    assert result["cache_usage_samples"] == 2
    assert result["cached_input_fraction"] == 0.8
    assert result["cache_hit_first_payload_median_s"] == 2.0


@pytest.mark.asyncio
@pytest.mark.parametrize("save_audit", [False, True])
async def test_benchmark_run_reconstructs_audit_and_compaction_state(
    tmp_path, monkeypatch, save_audit
):
    workloads = []

    class TrackingWorkload(AuditWorkload):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            workloads.append(self)

    class AuditLLM:
        model_name = "fixture"

        def __init__(self, *, context_window, **kwargs):
            self.context_window = context_window
            self.calls = []
            self.summed = False

        async def close(self):
            pass

        async def chat(self, **kwargs):
            workload = workloads[-1]
            position = len(workload.receipts)
            if workload.saved or (self.summed and not save_audit):
                name = "final_answer"
                args = {
                    "answer": "Audit saved and verified.",
                    "response_language": "English",
                }
            elif position == len(workload.targets) and not self.summed:
                name = "sum_amounts"
                args = {"amounts_cents": [invoice_amount(i) for i in workload.targets]}
                self.summed = True
            elif position == len(workload.targets):
                name = "save_audit"
                args = {"receipts": workload.receipts, "total_cents": workload.total}
            else:
                name = "check_invoice"
                args = {
                    "invoice_id": f"invoice-{workload.targets[position]:06d}",
                    "amount_cents": invoice_amount(workload.targets[position]),
                    "previous_receipt": workload.receipts[-1] if position else "START",
                }
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": f"call-{position}-{len(workload.attempts)}-{name}",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                ],
            }

    forced_thresholds = []

    async def compact(runtime, *, context, **kwargs):
        if context.compact_config.threshold == 8192:
            forced_thresholds.append(context.compact_config.threshold)
            await runtime.tracer.trace_event(
                TraceEventType(
                    TraceScope.ACTION, TraceAction.END, TraceCategory.COMPACT
                ),
                data={"removed_count": 1},
            )

    monkeypatch.setattr(agent_cache_benchmark, "AuditWorkload", TrackingWorkload)
    monkeypatch.setattr(agent_cache_benchmark, "BenchmarkLLM", AuditLLM)
    monkeypatch.setattr(PatternRuntime, "compact_context_if_needed", compact)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    tokenizer_file = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_file))
    output = tmp_path / "audit"
    report = await agent_cache_benchmark.run(
        argparse.Namespace(
            output=output,
            tokenizer_json=tokenizer_file,
            model="fixture",
            endpoint="http://127.0.0.1:9997",
            api_key_file=None,
            context_window=16384,
            context_tokens=4200,
            rounds=3,
            fail_at=2,
            resume_after=2,
            compact_after=1,
            timeout=30,
        )
    )
    assert report["passed"] is save_audit
    assert report["resumed"] and report["compaction_attempted"]
    assert report["transient_failure_exercised"] and report["sum_tool_exercised"]
    assert report["verified_invoices"] == 3
    assert report["tool_attempts"] == 4
    assert forced_thresholds == [8192]
    assert len(report["compactions"]) == 1
    original, resumed = workloads
    assert original is not resumed and original.receipts is not resumed.receipts
    saved = json.loads((output / "audit-state.json").read_text())
    assert saved["compaction_attempted"] is True
    assert saved["receipts"] == original.receipts == resumed.receipts[:2]
    assert (output / "resume-checkpoint.json").is_file()
    assert (output / "audit.json").exists() is save_audit
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    assert any(
        e.get("prompt_prefix", {}).get("change") == "append_only" for e in events
    )
    tool = AuditTool(resumed, read=True)
    assert await tool.run_json_async({"invoice_id": "invoice-000001"}) == {
        "success": True,
        "invoice_id": "invoice-000001",
        "amount_cents": invoice_amount(1),
    }


@pytest.mark.asyncio
async def test_benchmark_records_cancelled_and_closed_calls(tmp_path, monkeypatch):
    async def chat(self, **kwargs):
        raise asyncio.CancelledError()

    async def stream(self, **kwargs):
        yield StreamChunk(type=ChunkType.TOKEN, content="partial")

    monkeypatch.setattr(XinferenceLLM, "chat", chat)
    monkeypatch.setattr(XinferenceLLM, "stream_chat", stream)
    llm = BenchmarkLLM(output=tmp_path, context_window=16384)
    with pytest.raises(asyncio.CancelledError):
        await llm.chat()
    generator = llm.stream_chat()
    await anext(generator)
    await generator.aclose()
    assert [(row["success"], row["error_type"]) for row in llm.calls] == [
        (False, "CancelledError"),
        (False, "GeneratorExit"),
    ]


def test_benchmark_cli_rejects_out_of_range_resume(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "agent_cache_benchmark.py",
            "--endpoint",
            "http://127.0.0.1:9997",
            "--model",
            "fixture",
            "--tokenizer-json",
            str(tmp_path / "tokenizer.json"),
            "--output",
            str(tmp_path / "audit"),
            "--rounds",
            "2",
            "--resume-after",
            "3",
        ],
    )
    with pytest.raises(SystemExit) as error:
        agent_cache_benchmark.main()
    assert error.value.code == 2
    assert "positions between 0 and rounds" in capsys.readouterr().err
