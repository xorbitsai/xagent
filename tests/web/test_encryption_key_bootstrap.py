import stat

import pytest
from cryptography.fernet import Fernet

from xagent.web.encryption_key_bootstrap import load_or_create_encryption_key


def test_bootstrap_creates_private_key_and_reuses_it(tmp_path):
    path = tmp_path / "secrets" / "encryption.key"

    first, first_created = load_or_create_encryption_key(path)
    second, second_created = load_or_create_encryption_key(path)

    assert second == first
    assert first_created is True
    assert second_created is False
    assert path.read_text(encoding="utf-8").strip() == first
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    Fernet(first.encode())


@pytest.mark.parametrize("contents", ["", "not-a-fernet-key"])
def test_bootstrap_rejects_invalid_existing_key(tmp_path, contents):
    path = tmp_path / "encryption.key"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises((ValueError, TypeError)):
        load_or_create_encryption_key(path)
