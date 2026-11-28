"""Windows file operations that reject reparse points throughout a path."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from ctypes import wintypes
from functools import cache
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    from ctypes import WinDLL, WinError
    from msvcrt import open_osfhandle

_OBJ_CASE_INSENSITIVE = 0x0040
_OBJ_DONT_REPARSE = 0x1000
_FILE_SHARE_ALL = 0x0007
_FILE_READ_DATA = 0x0001
_FILE_WRITE_DATA = 0x0002
_FILE_READ_ATTRIBUTES = 0x0080
_SYNCHRONIZE = 0x00100000
_FILE_DIRECTORY_FILE = 0x0001
_FILE_SYNCHRONOUS_IO_NONALERT = 0x0020
_FILE_NON_DIRECTORY_FILE = 0x0040
_FILE_OPEN = 1
_FILE_CREATE = 2
_FILE_OPEN_IF = 3
_FILE_TYPE_DISK = 1


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.USHORT),
        ("maximum_length", wintypes.USHORT),
        ("buffer", ctypes.c_void_p),
    ]


class _ObjectAttributes(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.ULONG),
        ("root_directory", wintypes.HANDLE),
        ("object_name", ctypes.POINTER(_UnicodeString)),
        ("attributes", wintypes.ULONG),
        ("security_descriptor", ctypes.c_void_p),
        ("security_quality_of_service", ctypes.c_void_p),
    ]


class _IOStatusBlock(ctypes.Structure):
    _fields_ = [
        ("status_or_pointer", ctypes.c_void_p),
        ("information", ctypes.c_size_t),
    ]


@cache
def _apis() -> tuple[Any, Any]:
    if sys.platform == "win32":
        ntdll = WinDLL("ntdll")
        ntdll.RtlDosPathNameToNtPathName_U_WithStatus.argtypes = [
            wintypes.LPCWSTR,
            ctypes.POINTER(_UnicodeString),
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        ntdll.RtlDosPathNameToNtPathName_U_WithStatus.restype = wintypes.LONG
        ntdll.RtlFreeUnicodeString.argtypes = [ctypes.POINTER(_UnicodeString)]
        ntdll.RtlFreeUnicodeString.restype = None
        ntdll.RtlNtStatusToDosError.argtypes = [wintypes.LONG]
        ntdll.RtlNtStatusToDosError.restype = wintypes.ULONG
        ntdll.NtCreateFile.argtypes = [
            ctypes.POINTER(wintypes.HANDLE),
            wintypes.DWORD,
            ctypes.POINTER(_ObjectAttributes),
            ctypes.POINTER(_IOStatusBlock),
            ctypes.c_void_p,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            wintypes.ULONG,
            ctypes.c_void_p,
            wintypes.ULONG,
        ]
        ntdll.NtCreateFile.restype = wintypes.LONG

        kernel32 = WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel32.CloseHandle.restype = wintypes.BOOL
        kernel32.GetFileType.argtypes = [wintypes.HANDLE]
        kernel32.GetFileType.restype = wintypes.DWORD
        return ntdll, kernel32
    raise NotImplementedError("Windows filesystem operation")


def _open(path: Path, access: int, disposition: int, options: int) -> int:
    if sys.platform == "win32":
        if (
            not path.is_absolute()
            or path.is_reserved()
            or str(path).startswith("\\\\.\\")
        ):
            raise OSError(errno.EINVAL, "Expected an absolute filesystem path", path)

        ntdll, kernel32 = _apis()
        native_name = _UnicodeString()
        status = ntdll.RtlDosPathNameToNtPathName_U_WithStatus(
            str(path), ctypes.byref(native_name), None, None
        )
        if status < 0:
            raise WinError(ntdll.RtlNtStatusToDosError(status))
        try:
            # OPEN_REPARSE_POINT only applies to the final component. Asking the
            # object manager to stop at any reparse also covers parent directories.
            attributes = _ObjectAttributes(
                length=ctypes.sizeof(_ObjectAttributes),
                object_name=ctypes.pointer(native_name),
                attributes=_OBJ_CASE_INSENSITIVE | _OBJ_DONT_REPARSE,
            )
            handle = wintypes.HANDLE()
            status = ntdll.NtCreateFile(
                ctypes.byref(handle),
                access | _SYNCHRONIZE,
                ctypes.byref(attributes),
                ctypes.byref(_IOStatusBlock()),
                None,
                0,
                _FILE_SHARE_ALL,
                disposition,
                options | _FILE_SYNCHRONOUS_IO_NONALERT,
                None,
                0,
            )
            if status < 0:
                raise WinError(ntdll.RtlNtStatusToDosError(status))
            assert handle.value is not None
            if kernel32.GetFileType(handle) != _FILE_TYPE_DISK:
                kernel32.CloseHandle(handle)
                raise OSError(errno.EINVAL, "Expected a disk file or directory", path)
            return handle.value
        finally:
            ntdll.RtlFreeUnicodeString(ctypes.byref(native_name))
    raise NotImplementedError("Windows filesystem operation")


def open_readonly(path: Path) -> int:
    """Open a regular binary file without following reparse points."""
    if sys.platform == "win32":
        handle = _open(
            path,
            _FILE_READ_DATA | _FILE_READ_ATTRIBUTES,
            _FILE_OPEN,
            _FILE_NON_DIRECTORY_FILE,
        )
        try:
            return open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
        except BaseException:
            _apis()[1].CloseHandle(handle)
            raise
    raise NotImplementedError("Windows filesystem operation")


def create_file(path: Path) -> int:
    """Create a new binary file without following reparse points."""
    if sys.platform == "win32":
        handle = _open(path, _FILE_WRITE_DATA, _FILE_CREATE, _FILE_NON_DIRECTORY_FILE)
        try:
            return open_osfhandle(handle, os.O_WRONLY | os.O_BINARY)
        except BaseException:
            _apis()[1].CloseHandle(handle)
            raise
    raise NotImplementedError("Windows filesystem operation")


def create_directory(path: Path, *, exist_ok: bool = False) -> None:
    """Create a directory while rejecting reparse points at any component."""
    handle = _open(
        path,
        _FILE_READ_ATTRIBUTES,
        _FILE_OPEN_IF if exist_ok else _FILE_CREATE,
        _FILE_DIRECTORY_FILE,
    )
    _apis()[1].CloseHandle(handle)
