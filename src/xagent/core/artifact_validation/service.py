"""Bounded host-side validation, cached by bytes rather than a mutable filename.

Authorization/path ownership remains with the caller. This module neither
registers nor deletes files and must never run inside a database transaction.
"""

import hashlib
import json
import logging
import os
import stat
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from threading import BoundedSemaphore, Lock

from ...config import (
    get_artifact_validation_max_bytes,
    get_artifact_validation_timeout_seconds,
    in_sandbox_tool_runner,
)
from .defaults import default_registry
from .models import CheckResult, ValidationReport, unchecked

logger = logging.getLogger(__name__)

_slots = BoundedSemaphore(2)
# Public capability URLs may be shared widely. Reserve at least one worker
# slot for authenticated/tool callers and never queue public requests here.
_public_slots = BoundedSemaphore(1)
_cache_lock = Lock()
_cache: OrderedDict[tuple[str, str, int], ValidationReport] = OrderedDict()


def _run_checks(
    filename: str, data: bytes, max_bytes: int, timeout: float
) -> ValidationReport:
    try:
        # A subprocess timeout actually terminates a stuck parser; timing out a
        # thread would leave it consuming resources. Bytes, not paths, cross
        # this boundary, so the report describes exactly the parent snapshot.
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "xagent.core.artifact_validation.worker",
                filename,
                str(max_bytes),
            ],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=True,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
        )
        value = json.loads(completed.stdout)
        statuses = {"valid", "invalid", "unchecked"}
        if value["status"] not in statuses or not value["checks"]:
            raise ValueError("Invalid validation report")
        checks = tuple(CheckResult(**item) for item in value["checks"])
        if any(c.status not in statuses for c in checks):
            raise ValueError("Invalid check status")
        return ValidationReport(value["status"], checks, value["sha256"])
    except subprocess.TimeoutExpired:
        return unchecked("File validation exceeded its time budget.")
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError):
        logger.exception("Artifact validator process failed unexpectedly")
        return unchecked("Validator process could not complete.")


def validate_artifact(
    path: str | Path, *, filename: str | None = None, public: bool = False
) -> ValidationReport:
    if in_sandbox_tool_runner():
        return unchecked("Awaiting host-side file validation.")
    path = Path(path)
    filename = Path(filename or path.name).name
    if not default_registry().supports(filename):
        return unchecked("No validator is installed for this format.")
    try:
        max_bytes = get_artifact_validation_max_bytes()
        timeout = get_artifact_validation_timeout_seconds()
    except ValueError:
        logger.warning("Invalid artifact validation configuration; checks are disabled")
        return unchecked("Validation configuration is invalid.")
    if public and not _public_slots.acquire(blocking=False):
        return unchecked("Public validation capacity is busy; retry later.")
    # Acquire before loading bytes, not merely before launching the parser.
    # Concurrent preview requests must not each retain a max-sized snapshot.
    try:
        if not _slots.acquire(timeout=timeout):
            return unchecked("Validation capacity is busy; file has not been checked.")
        try:
            try:
                return _validate_snapshot(path, filename, max_bytes, timeout)
            except Exception:
                # Validation is advisory: snapshot/host failures must not turn
                # a successfully registered output into a failed tool result.
                logger.exception("Artifact snapshot validation failed unexpectedly")
                return unchecked("File validation could not complete.")
        finally:
            _slots.release()
    finally:
        if public:
            _public_slots.release()


def _validate_snapshot(
    path: Path, filename: str, max_bytes: int, timeout: float
) -> ValidationReport:
    try:
        if path.stat().st_size > max_bytes:
            return unchecked("File exceeds the validation byte budget.")
        # Nonblocking open prevents a replaced file/FIFO from hanging before
        # the killable parser process even starts. Validate the opened inode.
        with os.fdopen(
            os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)), "rb"
        ) as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                return unchecked("Only regular files can be validated.")
            data = stream.read(max_bytes + 1)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError:
        return unchecked("File bytes are unavailable for validation.")
    if len(data) > max_bytes:
        return unchecked("File exceeds the validation byte budget.")

    def identity(stat_result: os.stat_result) -> tuple[int, ...]:
        return (
            stat_result.st_dev,
            stat_result.st_ino,
            stat_result.st_size,
            stat_result.st_mtime_ns,
            stat_result.st_ctime_ns,
        )

    if identity(before) != identity(after) or identity(after) != identity(current):
        return unchecked("File changed during validation; check its new version.")
    digest = hashlib.sha256(data).hexdigest()
    key = (digest, filename, max_bytes)
    with _cache_lock:
        cached = _cache.get(key)
        if cached:
            _cache.move_to_end(key)
    report = cached or _run_checks(filename, data, max_bytes, timeout)
    try:
        if identity(path.stat()) != identity(after):
            return unchecked("File changed during validation; check its new version.")
    except OSError:
        return unchecked("File bytes are unavailable for validation.")
    if report.sha256 not in (None, digest):
        return unchecked("Validation report did not match the file snapshot.")
    report = ValidationReport(report.status, report.checks, digest)
    if report.status != "unchecked":
        with _cache_lock:
            _cache[key] = report
            _cache.move_to_end(key)
            while len(_cache) > 128:
                _cache.popitem(last=False)
    return report
