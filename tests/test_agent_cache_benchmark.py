from __future__ import annotations

import json

import pytest

from scripts.agent_cache_benchmark import (
    AuditTool,
    AuditWorkload,
    BenchmarkRuntime,
    BenchmarkTrace,
    SumAmountsTool,
    build_context,
    cache_summary,
    invoice_amount,
)
from xagent.core.agent import ExecutionContext, ReActPattern
from xagent.core.agent.runtime import load_pattern_checkpoint


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
async def test_benchmark_tracer_supports_real_runtime_and_checkpoint_resume(tmp_path):
    workload = AuditWorkload([1, 100], tmp_path)
    trace = BenchmarkTrace(tmp_path)

    class AuditLLM:
        context_window = 131072
        model_name = "fixture"
        summed = False

        async def chat(self, **kwargs):
            position = len(workload.receipts)
            if workload.saved:
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

    def pattern():
        return ReActPattern(
            max_iterations=10,
            user_interaction_enabled=False,
            repeated_tool_decision_after_consecutive_tool_calls=None,
            repeated_tool_decision_after_consecutive_work_tool_calls=None,
        )

    context = ExecutionContext()
    context.add_user_message("Audit the invoice chain and save the verified file.")
    tools = [AuditTool(workload), AuditTool(workload, save=True), SumAmountsTool()]
    runtime = BenchmarkRuntime(
        workload=workload,
        compact_after=0,
        tracer=trace,
        interrupt_checker=lambda: len(workload.receipts) == 1,
    )
    result = await pattern().run(
        context=context, tools=tools, llm=AuditLLM(), runtime=runtime
    )
    assert result["status"] == "interrupted"
    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())
    context = ExecutionContext.from_dict(checkpoint["context"])
    resumed = pattern()
    load_pattern_checkpoint(resumed, checkpoint)
    runtime = BenchmarkRuntime(workload=workload, compact_after=0, tracer=trace)
    result = await resumed.run(
        context=context, tools=tools, llm=AuditLLM(), runtime=runtime
    )
    assert result["success"] is True
    assert workload.saved and workload.failed_once
    assert len(workload.attempts) == 3
    assert any(
        event.get("prompt_prefix", {}).get("change") == "append_only"
        for event in trace.events
    )
