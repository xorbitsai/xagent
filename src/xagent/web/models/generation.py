"""Cross-dialect database defaults for durable UUID generations."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import Uuid
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import FunctionElement


class RandomUUID(FunctionElement[UUID]):
    """A database-generated random UUID on every supported dialect."""

    type = Uuid(as_uuid=True)
    inherit_cache = True


@compiles(RandomUUID, "postgresql")
def _compile_postgresql_random_uuid(
    _element: RandomUUID, _compiler: object, **_kwargs: object
) -> str:
    return "gen_random_uuid()"


@compiles(RandomUUID, "sqlite")
def _compile_sqlite_random_uuid(
    _element: RandomUUID, _compiler: object, **_kwargs: object
) -> str:
    # SQLite has no UUID function. Assemble 32 hexadecimal digits while
    # pinning RFC 4122 version/variant bits so SQLAlchemy's emulated Uuid type
    # reads the database-generated value with the same semantics as uuid4().
    return (
        "(lower(hex(randomblob(4))) || lower(hex(randomblob(2))) || "
        "'4' || substr(lower(hex(randomblob(2))), 2) || "
        "substr('89ab', (random() & 3) + 1, 1) || "
        "substr(lower(hex(randomblob(2))), 2) || lower(hex(randomblob(6))))"
    )
