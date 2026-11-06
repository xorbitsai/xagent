from __future__ import annotations

import argparse
import asyncio
import datetime
import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit

from scripts.benchmarks import agent_cache_concurrency_benchmark as bench
from xagent.core.model.chat.cache_lifecycle import agent_cache_session
from xagent.core.model.chat.routing_context import xinference_routing_key


def test_summary_keeps_failed_work_in_the_denominator():
    result = bench.summarize(
        [{"passed": True}, {"passed": False}],
        [{"passed": True, "scheduled_to_finish_s": 12}, {"passed": False}],
        elapsed=120,
        cooling=20,
    )
    assert result["passed"] is False
    assert result["successful_agents"] == 1
    assert result["attempted_agents"] == 2
    assert result["successful_agents_per_minute_wall"] == 0.5
    assert result["successful_agents_per_minute_without_cooling"] == 0.6
    assert result["successful_foreground"] == 1
    assert result["attempted_foreground"] == 2


@pytest.mark.asyncio
async def test_admission_cancellation_does_not_leak_capacity():
    gate = bench.AdmissionGate(1)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def hold():
        async with gate.slot():
            entered.set()
            await release.wait()

    first = asyncio.create_task(hold())
    await entered.wait()
    second = asyncio.create_task(hold())
    await asyncio.sleep(0)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    assert gate.active == 1
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert gate.active == 0
    async with asyncio.timeout(1):
        async with gate.slot():
            assert gate.active == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger,resume,hot", [(70, 60, 75), (60, 45, 65)])
async def test_thermal_gate_drains_work_before_counting_idle(
    tmp_path, trigger, resume, hot
):
    file = tmp_path / "telemetry.json"

    def temperature(value):
        file.write_text(
            json.dumps(
                {
                    "0": {
                        "utc": bench.utc(),
                        "temperature_C": value,
                        "sw_thermal_throttle": False,
                        "hw_thermal_throttle": False,
                    }
                }
            )
        )

    temperature(50)
    gate = bench.AdmissionGate(2, file, trigger=trigger, resume=resume)
    async with gate.slot():
        temperature(hot)
        assert not gate.can_enter()
        assert gate.cooling and gate.idle_started is None
    assert gate.idle_started is not None
    temperature(resume)
    async with gate.slot():
        assert not gate.cooling
    assert len(gate.cooling_intervals) == 1
    assert gate.cooling_intervals[0]["elapsed_s"] >= 0
    stale = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
        seconds=30
    )
    row = json.loads(file.read_text())
    row["0"]["utc"] = stale.isoformat()
    file.write_text(json.dumps(row))
    with pytest.raises(RuntimeError, match="stale"):
        async with gate.slot():
            pass
    assert gate.active == 0


@pytest.mark.asyncio
async def test_simultaneous_audits_keep_receipts_and_sessions_isolated(
    tmp_path, monkeypatch
):
    instances = []
    workloads = {}
    original_workload = bench.audit.AuditWorkload

    class TrackingWorkload(original_workload):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            workloads[self.output] = self

    monkeypatch.setattr(bench.audit, "AuditWorkload", TrackingWorkload)
    first_calls = asyncio.Event()
    active = 0
    peak = 0

    class FixtureLLM:
        def __init__(self, *, output, hints, gate, **kwargs):
            self.output = output
            self.hints = hints
            self.gate = gate
            self.calls = []
            self.control = []
            self.summed = False
            self.session = None
            instances.append(self)

        async def _update_cache_session(self, payload):
            self.control.append({**payload, "success": True})

        async def close(self):
            pass

        async def chat(self, **kwargs):
            nonlocal active, peak
            async with self.gate.slot():
                session = agent_cache_session.get()
                assert session is not None
                self.session = session.id
                if self.hints:
                    session.register(
                        (str(self.output), "fixture"), self._update_cache_session
                    )
                self.routing_key = xinference_routing_key.get()
                active += 1
                peak = max(peak, active)
                try:
                    if not self.calls:
                        if peak == 2:
                            first_calls.set()
                        await asyncio.wait_for(first_calls.wait(), 1)
                    workload = workloads[self.output]
                    checked = [{"receipt": receipt} for receipt in workload.receipts]
                    saved = workload.saved
                    if saved:
                        name = "final_answer"
                        args = {"answer": "Saved.", "response_language": "English"}
                    elif len(checked) == 2 and not self.summed:
                        name = "sum_amounts"
                        args = {
                            "amounts_cents": [
                                bench.audit.invoice_amount(1),
                                bench.audit.invoice_amount(2),
                            ]
                        }
                        self.summed = True
                    elif len(checked) == 2:
                        name = "save_audit"
                        args = {
                            "receipts": [r["receipt"] for r in checked],
                            "total_cents": bench.audit.invoice_amount(1)
                            + bench.audit.invoice_amount(2),
                        }
                    else:
                        pos = len(checked)
                        name = "check_invoice"
                        args = {
                            "invoice_id": f"invoice-{pos + 1:06d}",
                            "amount_cents": bench.audit.invoice_amount(pos + 1),
                            "previous_receipt": checked[-1]["receipt"]
                            if checked
                            else "START",
                        }
                    self.calls.append(
                        {
                            "routing_key_sent": self.hints,
                            "cache_session_sent": self.hints,
                            "usage": {
                                "prompt_tokens": 6000,
                                "prompt_tokens_details": {
                                    "cached_tokens": 0 if not self.calls else 5500
                                },
                            },
                        }
                    )
                    return {
                        "content": "",
                        "tool_calls": [
                            {
                                "id": f"call-{len(self.calls)}",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(args),
                                },
                            }
                        ],
                    }
                finally:
                    active -= 1

    monkeypatch.setattr(bench, "ConcurrentLLM", FixtureLLM)
    args = argparse.Namespace(
        output=tmp_path,
        variant="lifecycle",
        tool_delays=[0.02],
        model="fixture",
        context_window=16384,
        endpoint="http://unused",
        api_key_file=None,
        timeout=30,
    )
    gate = bench.AdmissionGate(2)
    ledger = "\n".join(
        f"invoice-{i:06d} | amount_cents={bench.audit.invoice_amount(i)} | currency=USD | status=posted"
        for i in [1, 2]
    )
    results = await asyncio.gather(
        *(bench.run_agent(args, i, ledger, 2, gate, "scope") for i in [0, 1])
    )
    assert all(r["passed"] for r in results), results
    assert peak == 2
    assert len({instance.session for instance in instances}) == 2
    assert len({instance.routing_key for instance in instances}) == 2
    receipts = [
        json.loads((tmp_path / f"agent-{i:02d}" / "audit.json").read_text())["receipts"]
        for i in [0, 1]
    ]
    assert not set(receipts[0]) & set(receipts[1])
    assert xinference_routing_key.get() is None


@pytest.mark.asyncio
async def test_batch_retains_failed_agent_and_still_finishes_foreground(
    tmp_path, monkeypatch
):
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    file = tmp_path / "tokenizer.json"
    tokenizer.save(str(file))
    monkeypatch.setattr(bench.audit, "build_ledger", lambda *a: ("ledger", 2))

    async def agent(args, index, *rest):
        if index == 0:
            raise RuntimeError("fixture")
        return {"passed": True}

    async def probe(*args):
        return {"passed": True, "scheduled_to_finish_s": 1}

    monkeypatch.setattr(bench, "run_agent", agent)
    monkeypatch.setattr(bench, "run_probe", probe)
    args = bench.parser().parse_args(
        [
            "--endpoint",
            "http://unused",
            "--model",
            "fixture",
            "--tokenizer-json",
            str(file),
            "--output",
            str(tmp_path / "run"),
            "--variant",
            "baseline",
            "--agents",
            "2",
            "--probes",
            "1",
        ]
    )
    result = await bench.run(args)
    assert not result["passed"] and result["successful_agents"] == 1
    assert result["successful_foreground"] == 1
    assert result["agents"][0] == {"passed": False, "error_type": "RuntimeError"}
    assert (args.output / "report.json").exists()
    with pytest.raises(FileExistsError):
        await bench.run(args)


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_tokens,expected", [(0, True), (32, False)])
async def test_foreground_stream_preserves_usage_and_rejects_warm_probe(
    tmp_path, monkeypatch, cached_tokens, expected
):
    from xagent.core.model.chat.basic.xinference import XinferenceLLM
    from xagent.core.model.chat.types import ChunkType, StreamChunk

    async def stream(self, **kwargs):
        config = {}
        self._apply_session_affinity(config)
        self._apply_cache_lifecycle(config)
        yield StreamChunk(type=ChunkType.TOKEN, content="OK")
        yield StreamChunk(
            type=ChunkType.USAGE,
            usage={
                "prompt_tokens": 500,
                "prompt_tokens_details": {"cached_tokens": cached_tokens},
            },
        )

    monkeypatch.setattr(XinferenceLLM, "stream_chat", stream)
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = WhitespaceSplit()
    args = argparse.Namespace(
        output=tmp_path,
        model="fixture",
        context_window=16384,
        endpoint="http://unused",
        api_key_file=None,
        timeout=30,
        probe_start=0,
        probe_interval=1,
        probe_tokens=256,
    )
    result = await bench.run_probe(
        args, 0, bench.time.perf_counter(), tokenizer, bench.AdmissionGate(1), "scope"
    )
    assert result["passed"] is expected
    assert (
        result["calls"][0]["usage"]["prompt_tokens_details"]["cached_tokens"]
        == cached_tokens
    )
    assert not result["calls"][0]["routing_key_sent"]
    assert not result["calls"][0]["cache_session_sent"]
