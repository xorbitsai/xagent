"""Timing-only launcher for benchmark-owned hosts, not a production entry point.

Records task ID and first OpenAI-compatible model-adapter entry per process.
This is NOT socket-send time, token latency, or the earlier llm_call_start trace.
Non-OpenAI-compatible adapters are deliberately unsupported: missing observations
fail validation instead of being substituted with a different clock.
"""

from __future__ import annotations

import atexit
import json
import os
import runpy
import sys
import time
from functools import wraps
from pathlib import Path


def install(output):
    from xagent.core.model.chat.basic.openai import OpenAICompatibleLLM
    from xagent.web.services.task_lease_service import current_task_lease

    entries = {}

    def record():
        lease = current_task_lease()
        if lease is not None:
            entries.setdefault(lease.task_id, time.time())

    original_chat = OpenAICompatibleLLM.chat
    original_stream = OpenAICompatibleLLM.stream_chat

    @wraps(original_chat)
    async def chat(self, *args, **kwargs):
        record()
        return await original_chat(self, *args, **kwargs)

    @wraps(original_stream)
    async def stream(self, *args, **kwargs):
        record()
        iterator = original_stream(self, *args, **kwargs)
        try:
            async for chunk in iterator:
                yield chunk
        finally:
            await iterator.aclose()

    OpenAICompatibleLLM.chat = chat
    OpenAICompatibleLLM.stream_chat = stream

    @atexit.register
    def dump():
        with (Path(output) / f"model-entries-{os.getpid()}.json").open("x") as handle:
            json.dump(
                [{"task_id": key, "time": value} for key, value in entries.items()],
                handle,
            )


def main():
    module, output, *arguments = sys.argv[1:]
    if module not in {"xagent.web", "xagent.web.worker"}:
        raise SystemExit("Unsupported benchmark host")
    install(output)
    sys.argv = [module, *arguments]
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
