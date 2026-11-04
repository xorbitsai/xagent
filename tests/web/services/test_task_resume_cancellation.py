"""An uncertain reply is reported as unknown without swallowing cancellation."""

import asyncio

import pytest

from xagent.web.services.task_resume import (
    TaskResumeOutcomeUnknownError,
    _raise_outcome_unknown,
)


def _report(exc: BaseException) -> None:
    try:
        raise exc
    except BaseException as caught:
        _raise_outcome_unknown(caught, "reply-1")


@pytest.mark.asyncio
async def test_stray_cancelled_error_from_the_write_is_outcome_unknown():
    # The write raised CancelledError while this task was not being cancelled.
    with pytest.raises(TaskResumeOutcomeUnknownError) as unknown:
        _report(asyncio.CancelledError())
    assert unknown.value.command_id == "reply-1"
    assert isinstance(unknown.value.__cause__, asyncio.CancelledError)
    assert asyncio.current_task().cancelling() == 0


@pytest.mark.asyncio
async def test_a_cancelled_task_settles_as_unknown_and_consumes_the_request():
    entered = asyncio.Event()

    async def resume() -> tuple[str | None, int]:
        try:
            try:
                entered.set()
                await asyncio.sleep(10)
            except asyncio.CancelledError as exc:
                _raise_outcome_unknown(exc, "reply-1")
        except TaskResumeOutcomeUnknownError as unknown:
            # Settled as unknown: no cancellation request is left pending for
            # structured-concurrency code later in this task.
            return unknown.command_id, asyncio.current_task().cancelling()
        raise AssertionError("expected an outcome-unknown settlement")

    task = asyncio.create_task(resume())
    await entered.wait()
    task.cancel()
    assert await task == ("reply-1", 0)
    assert not task.cancelled()


@pytest.mark.asyncio
async def test_non_exception_base_exceptions_propagate_unchanged():
    class Stop(BaseException):
        pass

    with pytest.raises(Stop):
        _report(Stop())


@pytest.mark.asyncio
async def test_ordinary_errors_become_outcome_unknown():
    with pytest.raises(TaskResumeOutcomeUnknownError) as unknown:
        _report(RuntimeError("lost acknowledgement"))
    assert isinstance(unknown.value.__cause__, RuntimeError)
