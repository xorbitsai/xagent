"""Type stubs for dateutil.parser (only the pieces this repo uses)."""

from datetime import datetime
from typing import Any

def isoparse(dt_str: str) -> datetime: ...
def parse(timestr: str, *args: Any, **kwargs: Any) -> datetime: ...
