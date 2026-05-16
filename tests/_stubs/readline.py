"""Test-only readline stub.

Some local macOS/Codex Python environments segfault while importing the native
readline extension. Pytest and pdb can import readline opportunistically, so the
test runner uses this stub through pytest's pythonpath setting.
"""


def parse_and_bind(_: str) -> None:
    return None


def set_completer(_: object | None) -> None:
    return None


def get_completer() -> None:
    return None


def set_completer_delims(_: str) -> None:
    return None


def get_completer_delims() -> str:
    return ""


def set_history_length(_: int) -> None:
    return None


def get_history_length() -> int:
    return 0


def read_history_file(_: str | None = None) -> None:
    return None


def write_history_file(_: str | None = None) -> None:
    return None


def clear_history() -> None:
    return None


def add_history(_: str) -> None:
    return None


def get_current_history_length() -> int:
    return 0


def get_history_item(_: int) -> None:
    return None


def remove_history_item(_: int) -> None:
    return None


def replace_history_item(_: int, __: str) -> None:
    return None


def set_startup_hook(_: object | None = None) -> None:
    return None


def set_pre_input_hook(_: object | None = None) -> None:
    return None


def redisplay() -> None:
    return None


def insert_text(_: str) -> None:
    return None


def get_line_buffer() -> str:
    return ""


def get_begidx() -> int:
    return 0


def get_endidx() -> int:
    return 0


def __getattr__(_: str) -> object:
    def noop(*args: object, **kwargs: object) -> None:
        del args, kwargs
        return None

    return noop
