"""Explicit encoded artifacts share real FileRef delivery, not text heuristics."""

import asyncio
import base64
import json
import textwrap
from pathlib import Path

import pytest

from xagent.core.inline_file_delivery import InlineFileDelivery, InlineFileStreamGuard


class Workspace:
    def __init__(self, root):
        self.workspace_dir = root
        self.output_dir = root / "output"
        self.files = {}

    def get_file_id_from_path(self, path):
        return self.files.get(path)

    def register_delivery_file(self, path):
        file_id = f"registered-{len(self.files)}"
        self.files[path] = file_id
        return file_id


def link(data=b"order,rate\r\nA01,0.15\r\n", name="report.csv", mime="text/csv"):
    return f"[{name}](data:{mime};base64,{base64.b64encode(data).decode()})"


@pytest.fixture
def delivery(tmp_path):
    return InlineFileDelivery(Workspace(tmp_path))


def test_registers_exact_bytes_and_deduplicates_repeated_deliveries(delivery):
    data = b"order,rate\r\nA01,0.15\r\n\x1a\x00\xff"
    source = f"Download: {link(data)}\n\nAgain: {link(data, name='copy.csv')}"
    result = delivery.transform(source)
    assert (
        result
        == "Download: [report.csv](file:registered-0)\n\nAgain: [report.csv](file:registered-0)"
    )
    assert len(delivery.workspace.files) == 1
    assert Path(next(iter(delivery.workspace.files))).read_bytes() == data
    assert delivery.transform(source) == result
    assert delivery.transform(result) == result
    assert len(delivery.workspace.files) == 1


@pytest.mark.parametrize("fence", ["```", "~~~~"])
@pytest.mark.parametrize("filename", ["report.csv", "report.CSV"])
def test_explicit_named_base64_fence(delivery, fence, filename):
    source = f'{fence}base64 filename="{filename}"\nYSwK\nYiwK\n{fence}\n'
    assert delivery.transform(source) == f"[{filename}](file:registered-0)\n"
    assert Path(next(iter(delivery.workspace.files))).read_bytes() == b"a,\nb,\n"


@pytest.mark.parametrize(
    "parameters",
    [
        ";name=x",
        ";charset=utf-8;name=my%20file",
        ";name=x;charset=UTF-8",
        ";name=",
        ";x-custom=base64",
    ],
)
def test_data_uri_parameters_share_stream_and_delivery_grammar(delivery, parameters):
    source = link().replace(";base64,", parameters + ";base64,")
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = (
            guard.feed(source[:offset]) + guard.feed(source[offset:]) + guard.flush()
        )
        assert "b3JkZX" not in emitted
    assert delivery.transform(source) == "[report.csv](file:registered-0)"


@pytest.mark.parametrize("extension", ["jpg", "jpeg", "JPEG"])
def test_named_jpeg_alias_is_a_registered_download(delivery, extension):
    source = f"```base64 filename=photo.{extension}\nYQ==\n```"
    assert delivery.transform(source) == "[photo.jpg](file:registered-0)"
    assert Path(next(iter(delivery.workspace.files))).read_bytes() == b"a"


@pytest.mark.parametrize(
    "source",
    [
        "A long token: " + "YQ==" * 100,
        "`" + link() + "`",
        "``" + link() + "``",
        "`multi\n" + link() + "\nline`",
        "```python\n" + link() + "\n```",
        "~~~~text\n" + link() + "\n~~~~",
        "````markdown\n```base64 filename=report.csv\nYQ==\n```\n````",
        "```base64\nYQ==\n```",
        "```\n" + link(),
        "    " + link(),
        "> " + link(),
        "  > " + link(),
        "\\" + link(),
    ],
)
def test_does_not_turn_code_or_ambiguous_text_into_files(delivery, source):
    assert delivery.transform(source) == source
    assert not delivery.workspace.files


@pytest.mark.parametrize(
    "prefix",
    [
        "- Files:\n    - ",
        "1. Files:\n    1. ",
        "- Files:\n  - More:\n      - ",
        "- Files:\n\n    ",
    ],
)
def test_nested_list_deliveries_are_not_indented_code(delivery, prefix):
    assert (
        delivery.transform(prefix + link())
        == prefix + "[report.csv](file:registered-0)"
    )


@pytest.mark.parametrize(
    "source",
    [
        "    - " + link(),
        "- Example:\n\n      " + link(),
        "- Example:\n    ```markdown\n    " + link() + "\n    ```",
        "- Example:\n    > " + link(),
    ],
)
def test_nested_code_and_quotes_stay_literal(delivery, source):
    assert delivery.transform(source) == source
    assert not delivery.workspace.files


@pytest.mark.parametrize("payload", ["not-base64!!!", "Y", "YQ=", "", "YQ==garbage"])
def test_malformed_payload_has_no_raw_blob_or_fake_file(delivery, payload):
    result = delivery.transform(f"[report.csv](data:text/csv;base64,{payload})")
    assert result == "report.csv (attachment unavailable)"
    assert not delivery.workspace.files


def test_mime_controls_suffix_and_filename_cannot_escape_workspace(delivery):
    result = delivery.transform(link(name="../../outside.exe"))
    assert result == "[outside.csv](file:registered-0)"
    path = Path(next(iter(delivery.workspace.files)))
    assert path.is_relative_to(delivery.workspace.output_dir)
    assert path.name == "outside.csv"


def test_image_delivery_uses_inline_file_ref(delivery):
    result = delivery.transform("!" + link(b"image bytes", "figure.png", "image/png"))
    assert result == "![figure.png](file:registered-0)"


@pytest.mark.parametrize(
    "mime", ["text/html", "image/svg+xml", "application/x-executable"]
)
def test_active_or_unknown_types_are_not_materialized(delivery, mime):
    assert "attachment unavailable" in delivery.transform(link(mime=mime))
    assert not delivery.workspace.files


@pytest.mark.parametrize("budget", ["0", "-1", "invalid", "3"])
def test_budget_rejects_before_registration(delivery, monkeypatch, budget):
    monkeypatch.setenv("XAGENT_INLINE_FILE_DELIVERY_MAX_BYTES", budget)
    assert "attachment unavailable" in delivery.transform(link())
    assert not delivery.workspace.output_dir.exists()


def test_total_bytes_and_file_count_are_bounded(delivery, monkeypatch):
    monkeypatch.setenv("XAGENT_INLINE_FILE_DELIVERY_MAX_BYTES", "5")
    assert "file:" in delivery.transform(link(b"abc"))
    assert "attachment unavailable" in delivery.transform(link(b"def"))
    monkeypatch.setenv("XAGENT_INLINE_FILE_DELIVERY_MAX_BYTES", "1000")
    for index in range(7):
        assert "file:" in delivery.transform(link(bytes([index])))
    assert "attachment unavailable" in delivery.transform(link(b"ninth"))
    assert len(delivery.workspace.files) == 8


def test_registration_failure_is_logged_without_exposing_details(
    delivery, monkeypatch, caplog
):
    def fail(path):
        raise ValueError("server private detail")

    monkeypatch.setattr(delivery.workspace, "register_delivery_file", fail)
    assert delivery.transform(link()) == "report.csv (attachment unavailable)"
    assert not list(delivery.workspace.output_dir.iterdir())
    assert caplog.records[-1].exc_info[1].args == ("server private detail",)


def test_output_symlink_cannot_redirect_delivery(delivery, tmp_path):
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    delivery.workspace.output_dir.symlink_to(outside, target_is_directory=True)
    assert "attachment unavailable" in delivery.transform(link())
    assert not list(outside.iterdir())


@pytest.mark.parametrize("source", [link()[:-1], "```base64 filename=report.csv\nYQ=="])
def test_incomplete_explicit_deliveries_do_not_dump_encoded_content(delivery, source):
    assert delivery.transform(source) == "report.csv (attachment unavailable)"
    assert not delivery.workspace.files


def test_incomplete_link_does_not_swallow_following_text_or_deliveries(delivery):
    source = link()[:-1] + " See " + link(name="second.csv")
    assert (
        delivery.transform(source)
        == "report.csv (attachment unavailable) See [second.csv](file:registered-0)"
    )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("closed", [True, False])
def test_wrapped_data_link_consumes_encoded_continuations(delivery, newline, closed):
    payload = base64.b64encode(b"some exact file bytes\n").decode()
    wrapped = newline.join(textwrap.wrap(payload, 8))
    source = f"Get [report.csv](data:text/csv;base64,{wrapped}"
    if closed:
        source += newline + ")"
    source += newline + "Next paragraph: " + link(name="next.csv")
    result = delivery.transform(source)
    expected = (
        "[report.csv](file:registered-0)"
        if closed
        else "report.csv (attachment unavailable)"
    )
    next_id = 1 if closed else 0
    assert (
        result
        == f"Get {expected}{newline}Next paragraph: [next.csv](file:registered-{next_id})"
    )
    if closed:
        assert (
            Path(next(iter(delivery.workspace.files))).read_bytes()
            == b"some exact file bytes\n"
        )


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_budget_accepts_mime_wrapped_named_fence(delivery, monkeypatch, newline):
    data = b"a" * (512 * 1024)
    monkeypatch.setenv("XAGENT_INLINE_FILE_DELIVERY_MAX_BYTES", str(len(data)))
    encoded = newline.join(textwrap.wrap(base64.b64encode(data).decode(), 76))
    assert len(encoded) - len(encoded.replace(newline, "")) > 4096
    source = f"```base64 filename=report.csv{newline}{encoded}{newline}```"
    assert delivery.transform(source) == "[report.csv](file:registered-0)"
    assert Path(next(iter(delivery.workspace.files))).read_bytes() == data


def test_excessive_whitespace_remains_bounded(delivery, monkeypatch):
    monkeypatch.setenv("XAGENT_INLINE_FILE_DELIVERY_MAX_BYTES", "10")
    source = "```base64 filename=report.csv\n" + " " * 5000 + "YQ==\n```"
    assert delivery.transform(source) == "report.csv (attachment unavailable)"
    assert not delivery.workspace.files


def test_legacy_registration_is_not_a_delivery_fallback(delivery, monkeypatch):
    monkeypatch.setattr(delivery.workspace, "register_delivery_file", None)
    legacy_calls = []
    monkeypatch.setattr(
        delivery.workspace, "register_file", legacy_calls.append, raising=False
    )
    assert delivery.transform(link()) == "report.csv (attachment unavailable)"
    assert not legacy_calls
    assert not list(delivery.workspace.output_dir.iterdir())


def test_missing_delivery_file_id_does_not_fall_back(delivery, monkeypatch):
    monkeypatch.setattr(delivery.workspace, "register_delivery_file", lambda path: None)
    assert delivery.transform(link()) == "report.csv (attachment unavailable)"
    assert not list(delivery.workspace.output_dir.iterdir())


def test_every_possible_stream_split_withholds_encoded_tail():
    source = "Download: " + link()
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = (
            guard.feed(source[:offset]) + guard.feed(source[offset:]) + guard.flush()
        )
        assert "base64" not in emitted
        assert "b3JkZX" not in emitted
        assert source.startswith(emitted)


def test_ordinary_stream_text_flushes_without_loss():
    guard = InlineFileStreamGuard()
    source = "Ordinary answer, no attachment."
    emitted = "".join(guard.feed(c) for c in source) + guard.flush()
    assert emitted == source


@pytest.mark.parametrize(
    "prefix",
    [
        "Use `df to load.\n",
        "Use `df to load.\n\n",
        "Use ``df to load.\n",
        "An unmatched ` tick.\n\nA later ` paragraph.\n",
    ],
)
def test_unmatched_backtick_does_not_hide_later_delivery(delivery, prefix):
    assert (
        delivery.transform(prefix + link())
        == prefix + "[report.csv](file:registered-0)"
    )


def test_inline_code_does_not_cross_paragraphs(delivery):
    source = "`Example\n\n" + link() + "\n`"
    assert (
        delivery.transform(source) == "`Example\n\n[report.csv](file:registered-0)\n`"
    )


@pytest.mark.parametrize(
    "source",
    [
        "Config uses metadata: values",
        "This explains base64 encoding without an attachment.",
        "Normal data: values and base64.",
    ],
)
def test_prose_markers_do_not_stall_stream(source):
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = guard.feed(source[:offset]) + guard.feed(source[offset:])
        assert not guard.held
        assert len(emitted) >= len(source) - 6
        assert emitted + guard.flush() == source


@pytest.mark.parametrize(
    "source",
    [
        "Use `df` to inspect the result. " * 10,
        "`df` is the result. " * 10,
        "An unmatched ` tick followed by prose. " * 10,
        "Read [text](data:text/plain,hello) then continue. " * 10,
    ],
)
def test_prose_backticks_and_plain_data_urls_keep_streaming(source):
    for chunks in ([source], list(source), [source[:20], source[20:]]):
        guard = InlineFileStreamGuard()
        emitted = "".join(guard.feed(chunk) for chunk in chunks)
        assert not guard.held
        assert len(guard.pending) <= 6
        assert emitted + guard.flush() == source


@pytest.mark.parametrize("fence", ["```", "~~~~"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_unnamed_base64_example_keeps_streaming(fence, newline):
    source = f"Example:{newline}{fence}base64{newline}YSwK{newline}{fence}{newline}More prose."
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = guard.feed(source[:offset]) + guard.feed(source[offset:])
        assert not guard.held
        assert emitted + guard.flush() == source


@pytest.mark.parametrize("fence", ["```", "~~~~", "````````"])
def test_named_fence_stream_splits_withhold_payload(fence):
    source = f"Download:\n{fence}base64 filename=report.csv\nYSwK\n{fence}"
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = (
            guard.feed(source[:offset]) + guard.feed(source[offset:]) + guard.flush()
        )
        assert "YSwK" not in emitted
        assert source.startswith(emitted)


def test_nested_named_fence_is_delivered_without_streaming_payload(delivery):
    source = "- Files:\n    ```base64 filename=report.csv\n    YSwK\n    ```\n"
    for offset in range(len(source) + 1):
        guard = InlineFileStreamGuard()
        emitted = (
            guard.feed(source[:offset]) + guard.feed(source[offset:]) + guard.flush()
        )
        assert "YSwK" not in emitted
    assert delivery.transform(source) == "- Files:\n  [report.csv](file:registered-0)\n"


def test_long_unicode_filename_fits_byte_budget(delivery):
    result = delivery.transform(link(name="报" * 100 + ".csv"))
    assert "file:registered-0" in result
    path = Path(next(iter(delivery.workspace.files)))
    assert len(path.name.encode("utf-8")) <= 240
    assert path.suffix == ".csv"
    assert path.read_bytes() == b"order,rate\r\nA01,0.15\r\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_child_runtime", [False, True])
async def test_runtime_stream_end_and_buffered_result_share_registered_file(
    delivery, use_child_runtime
):
    from xagent.core.agent import PatternRuntime
    from xagent.core.agent.pattern.final_answer_stream import FinalAnswerStreamSession

    events = []
    runtime = PatternRuntime(
        outbound_message_handler=events.append, inline_file_delivery=delivery
    )
    stream_runtime = runtime
    if use_child_runtime:
        from xagent.core.agent.pattern.auto.auto import AutoPattern, _AutoChildRuntime

        stream_runtime = _AutoChildRuntime(
            parent=runtime, auto_pattern=AutoPattern(), root_context=None
        )
    stream = FinalAnswerStreamSession(stream_runtime)
    source = "Download: " + link()
    for char in source:
        await stream.emit_delta(char)
    await stream.finish(source)
    assert events[-1]["type"] == "final_answer_end"
    assert events[-1]["content"] == "Download: [report.csv](file:registered-0)"
    assert all("base64" not in str(e) for e in events)
    assert await stream_runtime.prepare_final_answer(source) == events[-1]["content"]
    assert len(delivery.workspace.files) == 1
    assert not runtime._inline_stream_guards


@pytest.mark.asyncio
@pytest.mark.parametrize("restored", [False, True])
async def test_runner_wires_delivery_and_normalizes_context(delivery, restored):
    from xagent.core.agent import Agent, ExecutionContext
    from xagent.core.agent.runner import AgentRunner

    class Manager:
        def get_or_create_workspace(self, **kwargs):
            delivery.workspace.id = "inline-test"
            delivery.workspace.input_dir = delivery.workspace.workspace_dir / "input"
            delivery.workspace.temp_dir = delivery.workspace.workspace_dir / "temp"
            return delivery.workspace

    class Pattern:
        async def run(self, context, **kwargs):
            if restored:
                assert context.messages[-1].content == link()
            context.add_assistant_message(link())
            return {"success": True, "output": link()}

    runner = AgentRunner(
        Agent(name="inline-test", patterns=[Pattern()]), workspace_manager=Manager()
    )
    checkpoint = None
    if restored:
        saved_context = ExecutionContext(execution_id="inline-test")
        saved_context.add_assistant_message(link())
        checkpoint = {"context": saved_context.to_dict()}
    result = await runner.run(
        "Create a CSV",
        execution_id="inline-test",
        resume=restored,
        checkpoint=checkpoint,
    )
    assert result["output"] == "[report.csv](file:registered-0)"
    assert result["context"].messages[-1].content == result["output"]
    assert len(delivery.workspace.files) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["llm", "finish", "cancel"])
async def test_runtime_stream_failure_releases_guard_without_flushing_payload(
    delivery, monkeypatch, failure
):
    from xagent.core.agent import PatternRuntime

    events = []
    runtime = PatternRuntime(
        outbound_message_handler=events.append, inline_file_delivery=delivery
    )
    monkeypatch.setattr(runtime, "_has_native_stream_chat", lambda llm: True)
    started = asyncio.Event()

    async def stream_call(llm, *, on_chunk, **kwargs):
        message_id = next(iter(runtime._inline_stream_guards))
        await runtime.emit_final_answer_delta(message_id, link())
        started.set()
        if failure == "cancel":
            await asyncio.Event().wait()
        if failure == "llm":
            raise RuntimeError("LLM failed")
        return link()

    async def fail_finish(content):
        raise RuntimeError("Delivery failed")

    monkeypatch.setattr(runtime, "run_streaming_llm_call", stream_call)
    if failure == "finish":
        monkeypatch.setattr(runtime, "prepare_final_answer", fail_finish)
    task = asyncio.create_task(runtime.stream_final_answer(object()))
    if failure == "cancel":
        await started.wait()
        task.cancel()
    with pytest.raises(asyncio.CancelledError if failure == "cancel" else RuntimeError):
        await task
    assert not runtime._inline_stream_guards
    assert events[-1]["type"] == "final_answer_error"
    assert not any("base64" in str(event) or "b3JkZX" in str(event) for event in events)


@pytest.mark.asyncio
async def test_start_callback_failure_releases_guard(delivery):
    from xagent.core.agent import PatternRuntime

    def fail(event):
        raise RuntimeError("Disconnected")

    runtime = PatternRuntime(
        outbound_message_handler=fail, inline_file_delivery=delivery
    )
    with pytest.raises(RuntimeError, match="Disconnected"):
        await runtime.start_final_answer_stream()
    assert not runtime._inline_stream_guards


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "cancel", "error"])
async def test_terminal_broadcast_never_gets_contradictory_error(
    delivery, monkeypatch, outcome
):
    from xagent.core.agent import PatternRuntime

    events = []
    end_sent = asyncio.Event()

    async def outbound(event):
        events.append(event)
        if event["type"] == "final_answer_end":
            end_sent.set()
            if outcome == "cancel":
                await asyncio.Event().wait()
            if outcome == "error":
                raise RuntimeError("Broadcast interrupted after delivery")

    runtime = PatternRuntime(
        outbound_message_handler=outbound, inline_file_delivery=delivery
    )
    monkeypatch.setattr(runtime, "_has_native_stream_chat", lambda llm: True)

    async def response(*args, **kwargs):
        return link()

    monkeypatch.setattr(runtime, "run_streaming_llm_call", response)
    task = asyncio.create_task(runtime.stream_final_answer(object()))
    if outcome == "cancel":
        await asyncio.wait_for(end_sent.wait(), timeout=5)
        task.cancel()
    if outcome == "success":
        await task
    else:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancel" else RuntimeError
        ):
            await task
    assert [event["type"] for event in events].count("final_answer_end") == 1
    assert not any(event["type"] == "final_answer_error" for event in events)
    assert events[-1]["content"] == "[report.csv](file:registered-0)"
    assert not runtime._inline_stream_guards


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer_key", ["response", "answer", "output", "content", "message"]
)
@pytest.mark.parametrize("legacy", [False, True])
async def test_delivery_changes_only_the_selected_result_field(
    delivery, answer_key, legacy
):
    from xagent.core.agent import Agent, PatternRuntime
    from xagent.core.agent.runner import AgentRunner

    keys = ["response", "answer", "output", "content", "message"]
    original = {
        key: "" if keys.index(key) < keys.index(answer_key) else f"unrelated {key}"
        for key in keys
    }
    original[answer_key] = json.dumps({"final_answer": link()}) if legacy else link()

    class Pattern:
        async def run(self, **kwargs):
            return dict(original)

    runner = AgentRunner(
        Agent(name="field-test", patterns=[Pattern()]), workspace_enabled=False
    )
    result = await runner.run(
        "Deliver", runtime=PatternRuntime(inline_file_delivery=delivery)
    )
    assert result[answer_key] == "[report.csv](file:registered-0)"
    for key in keys:
        if key != answer_key:
            assert result[key] == original[key]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_runner_releases_abandoned_candidate_guards(delivery, outcome):
    from xagent.core.agent import Agent, PatternRuntime
    from xagent.core.agent.runner import AgentRunner

    events = []
    runtime = PatternRuntime(
        outbound_message_handler=events.append, inline_file_delivery=delivery
    )
    started = asyncio.Event()

    class Pattern:
        async def run(self, **kwargs):
            # Two retries can leave distinct candidate IDs pending. Keep the
            # keyed ownership model, but release all buffers at run teardown.
            for _ in range(2):
                message_id = await runtime.start_final_answer_stream()
                await runtime.emit_final_answer_delta(message_id, link())
            assert len(runtime._inline_stream_guards) == 2
            started.set()
            if outcome == "cancel":
                await asyncio.Event().wait()
            if outcome == "error":
                raise RuntimeError("Candidate parse failed")
            return {"success": True, "output": "Final answer"}

    runner = AgentRunner(
        Agent(name="inline-test", patterns=[Pattern()]), workspace_enabled=False
    )
    task = asyncio.create_task(
        runner.run("Answer", execution_id="inline-test", runtime=runtime)
    )
    if outcome == "cancel":
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        result = await task
        assert result["success"] == (outcome == "success")
    assert not runtime._inline_stream_guards
    assert not any("base64" in str(event) for event in events)
