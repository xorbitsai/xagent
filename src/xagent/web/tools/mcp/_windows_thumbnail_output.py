"""Handle-relative thumbnail output for Windows, where os.open has no dir_fd."""

import ctypes
import msvcrt
import os
import sys
from ctypes import wintypes
from pathlib import Path

assert sys.platform == "win32"

_FILE_TRAVERSE = 0x20
_FILE_WRITE_DATA = 0x2
_DELETE = 0x10000
_SYNCHRONIZE = 0x100000
_FILE_SHARE_READ_WRITE = 0x3  # Excluding FILE_SHARE_DELETE pins the directory name.
_OPEN_EXISTING = 3
_FILE_FLAG_BACKUP_SEMANTICS = 0x02000000
_FILE_CREATE = 2
_FILE_OPEN_IF = 3
_FILE_DIRECTORY_FILE = 0x1
_FILE_NON_DIRECTORY_FILE = 0x40
_FILE_SYNCHRONOUS_IO_NONALERT = 0x20
_OBJ_CASE_INSENSITIVE = 0x40
_OBJ_DONT_REPARSE = 0x1000
_FILE_DISPOSITION_INFO = 4
_STATUS_FILE_IS_A_DIRECTORY = 0xC00000BA
_STATUS_REPARSE_POINT_ENCOUNTERED = 0xC000050B
_INVALID_HANDLE = ctypes.c_void_p(-1).value


class _UnicodeString(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.USHORT),
        ("maximum_length", wintypes.USHORT),
        ("buffer", wintypes.LPWSTR),
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


class _IoStatusBlock(ctypes.Structure):
    # The first field is a union of NTSTATUS and a pointer.
    _fields_ = [
        ("status_or_pointer", ctypes.c_void_p),
        ("information", ctypes.c_size_t),
    ]


_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_ntdll = ctypes.WinDLL("ntdll")
_create_file = _kernel32.CreateFileW
_create_file.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_create_file.restype = wintypes.HANDLE
_close_handle = _kernel32.CloseHandle
_close_handle.argtypes = [wintypes.HANDLE]
_close_handle.restype = wintypes.BOOL
_set_file_information = _kernel32.SetFileInformationByHandle
_set_file_information.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_set_file_information.restype = wintypes.BOOL
_nt_create_file = _ntdll.NtCreateFile
_nt_create_file.argtypes = [
    ctypes.POINTER(wintypes.HANDLE),
    wintypes.DWORD,
    ctypes.POINTER(_ObjectAttributes),
    ctypes.POINTER(_IoStatusBlock),
    ctypes.c_void_p,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    wintypes.ULONG,
    ctypes.c_void_p,
    wintypes.ULONG,
]
_nt_create_file.restype = wintypes.LONG
_status_to_error = _ntdll.RtlNtStatusToDosError
_status_to_error.argtypes = [wintypes.LONG]
_status_to_error.restype = wintypes.ULONG


def _open_relative(root: int, name: str, *, directory: bool) -> int:
    # Supplying an open RootDirectory anchors the lookup. OBJ_DONT_REPARSE
    # rejects junctions and other reparse points before traversing them.
    name_buffer = ctypes.create_unicode_buffer(name)
    name_bytes = len(name.encode("utf-16-le"))
    object_name = _UnicodeString(
        name_bytes, name_bytes + 2, ctypes.cast(name_buffer, wintypes.LPWSTR)
    )
    attributes = _ObjectAttributes(
        ctypes.sizeof(_ObjectAttributes),
        root,
        ctypes.pointer(object_name),
        _OBJ_CASE_INSENSITIVE | _OBJ_DONT_REPARSE,
        None,
        None,
    )
    handle = wintypes.HANDLE()
    status = _nt_create_file(
        ctypes.byref(handle),
        (_FILE_TRAVERSE if directory else _FILE_WRITE_DATA | _DELETE) | _SYNCHRONIZE,
        ctypes.byref(attributes),
        ctypes.byref(_IoStatusBlock()),
        None,
        0,
        _FILE_SHARE_READ_WRITE if directory else 0,
        _FILE_OPEN_IF if directory else _FILE_CREATE,
        (_FILE_DIRECTORY_FILE if directory else _FILE_NON_DIRECTORY_FILE)
        | _FILE_SYNCHRONOUS_IO_NONALERT,
        None,
        0,
    )
    if status < 0:
        if not directory and (status & 0xFFFFFFFF) in {
            _STATUS_FILE_IS_A_DIRECTORY,
            _STATUS_REPARSE_POINT_ENCOUNTERED,
        }:
            # These statuses also indicate an occupied candidate name.
            raise FileExistsError(name)
        raise ctypes.WinError(_status_to_error(status))
    if handle.value is None:
        raise OSError("NtCreateFile returned no handle")
    return int(handle.value)


def open_output_directory(workspace: Path) -> int:
    path = str(workspace)
    if not path.startswith("\\\\?\\"):
        path = (
            "\\\\?\\UNC\\" + path[2:] if path.startswith("\\\\") else "\\\\?\\" + path
        )
    workspace_handle = _create_file(
        path,
        _FILE_TRAVERSE,
        _FILE_SHARE_READ_WRITE,
        None,
        _OPEN_EXISTING,
        _FILE_FLAG_BACKUP_SEMANTICS,
        None,
    )
    if workspace_handle == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return _open_relative(workspace_handle, "output", directory=True)
    finally:
        close_directory(workspace_handle)


def close_directory(handle: int) -> None:
    _close_handle(handle)


def _discard_handle(handle: int) -> None:
    # FileDispositionInfo's DeleteFile field is a one-byte BOOLEAN.
    disposition = ctypes.c_ubyte(1)
    if not _set_file_information(
        handle,
        _FILE_DISPOSITION_INFO,
        ctypes.byref(disposition),
        ctypes.sizeof(disposition),
    ):
        raise ctypes.WinError(ctypes.get_last_error())


def open_new_file(directory: int, name: str) -> int:
    handle = _open_relative(directory, name, directory=False)
    try:
        return msvcrt.open_osfhandle(handle, os.O_WRONLY | os.O_BINARY | os.O_NOINHERIT)
    except OSError:
        try:
            _discard_handle(handle)
        finally:
            _close_handle(handle)
        raise


def discard_file(descriptor: int) -> None:
    _discard_handle(msvcrt.get_osfhandle(descriptor))
