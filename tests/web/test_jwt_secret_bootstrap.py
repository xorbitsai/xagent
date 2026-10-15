import stat
from concurrent.futures import ThreadPoolExecutor

import pytest

from xagent.web.jwt_secret_bootstrap import load_or_create_jwt_secret


def test_bootstrap_creates_private_secret_and_reuses_it(tmp_path):
    path = tmp_path / "secrets" / "jwt.secret"

    first, first_created = load_or_create_jwt_secret(path)
    second, second_created = load_or_create_jwt_secret(path)

    assert second == first
    assert len(first) >= 32
    assert first_created is True
    assert second_created is False
    assert path.read_text(encoding="utf-8").strip() == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "contents", ["", "too-short", "replace-with-a-long-random-secret"]
)
def test_bootstrap_rejects_weak_existing_secret(tmp_path, contents):
    path = tmp_path / "jwt.secret"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="weak or too short"):
        load_or_create_jwt_secret(path)


def test_concurrent_bootstrap_calls_share_one_secret(tmp_path):
    path = tmp_path / "jwt.secret"

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: load_or_create_jwt_secret(path), range(8))
        )

    assert len({secret for secret, _ in results}) == 1
    assert sum(created for _, created in results) == 1
