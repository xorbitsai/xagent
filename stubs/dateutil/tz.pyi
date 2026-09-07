"""Type stubs for dateutil.tz (only the pieces this repo uses)."""

from datetime import tzinfo

def gettz(name: str | None = ...) -> tzinfo | None: ...
