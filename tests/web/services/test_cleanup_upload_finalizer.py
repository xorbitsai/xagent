"""An uncertain staged-path owner cannot abandon the rest of an upload batch."""

import io

import pytest
import sqlalchemy as sa
from fastapi import HTTPException, UploadFile

from tests.web.services.test_complete_upload_cleanup import (
    lifecycle as cleanup_lifecycle_fixture,
)
from xagent.web.api.files import store_uploaded_files

lifecycle = cleanup_lifecycle_fixture


@pytest.mark.asyncio
async def test_finalizer_continues_after_one_path_ownership_query_fails(lifecycle):
    sessions, source, *_ = lifecycle
    failed = []

    def fail_first_staged_owner(_conn, _cursor, statement, parameters, *_):
        if (
            not failed
            and "SELECT" in statement.upper()
            and "uploaded_files" in statement
            and "uploaded_files.storage_path =" in statement
            and "first.txt" in str(parameters)
            and list(source.parent.rglob("*first.txt"))
        ):
            failed.append(True)
            raise sa.exc.OperationalError(
                statement, parameters, RuntimeError("lost DB")
            )

    engine = sessions.kw["bind"]
    sa.event.listen(engine, "before_cursor_execute", fail_first_staged_owner)
    try:
        with pytest.raises(HTTPException) as fault:
            await store_uploaded_files(
                upload_items=[
                    UploadFile(filename="first.txt", file=io.BytesIO(b"first")),
                    UploadFile(filename="second.txt", file=io.BytesIO(b"second")),
                    UploadFile(filename="", file=io.BytesIO(b"invalid")),
                ],
                task_type="general",
                task_id=None,
                folder=None,
                user_id=1,
                single_file_mode=False,
            )
        assert fault.value.status_code == 422
        assert failed
        assert list(source.parent.rglob("*first.txt"))
        assert not list(source.parent.rglob("*second.txt"))
    finally:
        sa.event.remove(engine, "before_cursor_execute", fail_first_staged_owner)
