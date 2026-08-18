import pytest
from cryptography.fernet import Fernet

from xagent.core.utils.encryption import (
    EncryptionDecodeError,
    _get_encryption_key,
    decrypt_env_dict,
    decrypt_env_dict_strict,
    decrypt_value,
    decrypt_value_strict,
    encrypt_env_dict,
    encrypt_value,
    get_cipher,
)


def test_encrypt_decrypt_roundtrip():
    original_value = "my_super_secret_value"
    encrypted = encrypt_value(original_value)

    assert encrypted != original_value
    assert isinstance(encrypted, str)

    decrypted = decrypt_value(encrypted)
    assert decrypted == original_value


def test_encrypt_empty_value():
    assert encrypt_value("") == ""
    assert encrypt_value(None) is None


def test_encrypt_value_idempotent():
    """Re-encrypting an already-encrypted value is a no-op (no double-encryption)."""
    once = encrypt_value("secret")
    assert encrypt_value(once) == once
    assert decrypt_value(encrypt_value(once)) == "secret"


def test_encrypt_value_encrypts_fake_ciphertext_prefix():
    """A plaintext that merely looks like a Fernet token is still encrypted.

    Guards against the old prefix-only heuristic that would store such a value
    in plaintext at rest.
    """
    looks_like_token = "gAAAAABnot-a-real-token"
    encrypted = encrypt_value(looks_like_token)
    assert encrypted != looks_like_token
    assert decrypt_value(encrypted) == looks_like_token


def test_decrypt_empty_value():
    assert decrypt_value("") == ""
    assert decrypt_value(None) is None


def test_decrypt_invalid_token():
    # Provide an invalid token, should catch InvalidToken and return the original string
    invalid_encrypted = "invalid_token_value"
    result = decrypt_value(invalid_encrypted)
    assert result == "invalid_token_value"


def test_get_encryption_key_no_env(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "development")
    key = _get_encryption_key()
    assert key == "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="


def test_get_encryption_key_production_missing_key(monkeypatch):
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    with pytest.raises(
        ValueError, match="ENCRYPTION_KEY environment variable is not set"
    ):
        _get_encryption_key()


def test_get_encryption_key_with_env(monkeypatch):
    test_key = "some_test_key_base64_encoded="
    monkeypatch.setenv("ENCRYPTION_KEY", test_key)
    key = _get_encryption_key()
    assert key == test_key


STRICT_KEY_A = "RQMpe38gK3m0szjpSmTNw_sP3Y54r6hDc6JewBoPKXc="
STRICT_KEY_B = Fernet.generate_key().decode()


@pytest.fixture
def use_key(monkeypatch):
    """Switch the module to a given encryption key for one test.

    Setting ENCRYPTION_KEY alone has no effect once get_cipher() has run:
    it is lru_cached. Clear on every switch, and again on teardown so no
    test key survives into another test in the same process.
    """

    def _use(key):
        monkeypatch.setenv("ENCRYPTION_KEY", key)
        get_cipher.cache_clear()

    yield _use
    get_cipher.cache_clear()


def test_decrypt_value_strict_raises_on_foreign_token(use_key):
    """A token from another key raises, where the lenient helper stays silent."""
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    assert decrypt_value(token) == token
    with pytest.raises(EncryptionDecodeError):
        decrypt_value_strict(token)


def test_decrypt_value_strict_roundtrip(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict(encrypt_value("secret")) == "secret"


@pytest.mark.parametrize(
    "plaintext",
    ["invalid_token_value", "gAAAAABnot-a-real-token", "sk-abc123", "plain text"],
)
def test_decrypt_value_strict_passes_plaintext_through(use_key, plaintext):
    """Values that are not token-shaped are returned unchanged, as before.

    No key can open them, so classifying them as plaintext loses nothing.
    """
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict(plaintext) == plaintext


def test_decrypt_value_strict_empty_value(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_value_strict("") == ""
    assert decrypt_value_strict(None) is None


def test_decrypt_value_strict_error_is_value_error(use_key):
    """Callers already catching ValueError keep working."""
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    with pytest.raises(ValueError):
        decrypt_value_strict(token)


def test_decrypt_value_strict_error_omits_the_value(use_key):
    use_key(STRICT_KEY_A)
    token = encrypt_value("secret")

    use_key(STRICT_KEY_B)
    with pytest.raises(EncryptionDecodeError) as excinfo:
        decrypt_value_strict(token)
    assert token not in str(excinfo.value)


def test_decrypt_env_dict_strict(use_key):
    use_key(STRICT_KEY_A)
    assert decrypt_env_dict_strict(encrypt_env_dict({"A": "1", "B": "2"})) == {
        "A": "1",
        "B": "2",
    }
    assert decrypt_env_dict_strict(None) is None
    assert decrypt_env_dict_strict({}) == {}

    mixed = {"A": encrypt_value("1"), "B": "plain", "C": 5, "D": None}
    assert decrypt_env_dict_strict(mixed) == {
        "A": "1",
        "B": "plain",
        "C": 5,
        "D": None,
    }


def test_decrypt_env_dict_strict_raises_on_foreign_token(use_key):
    """One unreadable entry fails the whole map, instead of leaking ciphertext."""
    use_key(STRICT_KEY_A)
    foreign = encrypt_value("1")

    use_key(STRICT_KEY_B)
    env = {"A": encrypt_value("ok"), "B": foreign}
    assert decrypt_env_dict(env)["B"] == foreign
    with pytest.raises(EncryptionDecodeError):
        decrypt_env_dict_strict(env)


def test_decrypt_value_strict_reports_missing_key_for_a_token(monkeypatch):
    """A key configuration fault must surface for a token-shaped value.

    With no usable key at all, the lenient helper quietly returns the input;
    the strict helper must instead let get_cipher()'s ValueError out, because
    "this deployment has no key" and "this token cannot be opened" call for
    entirely different responses.
    """
    token = Fernet(STRICT_KEY_A.encode()).encrypt(b"secret").decode()
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_cipher.cache_clear()
    try:
        with pytest.raises(ValueError, match="ENCRYPTION_KEY"):
            decrypt_value_strict(token)
    finally:
        get_cipher.cache_clear()


def test_decrypt_value_strict_passes_plaintext_through_without_a_key(monkeypatch):
    """Plaintext never needs a key, so a missing key must not turn it into an error."""
    monkeypatch.delenv("ENCRYPTION_KEY", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    get_cipher.cache_clear()
    try:
        assert decrypt_value_strict("sk-abc123") == "sk-abc123"
    finally:
        get_cipher.cache_clear()


def test_decrypt_value_strict_passes_unencodable_plaintext_through(use_key):
    """A lone surrogate cannot be UTF-8 encoded; it is plaintext, not a token."""
    use_key(STRICT_KEY_A)
    value = "sk-\ud800-key"
    assert decrypt_value_strict(value) == value


def test_decrypt_value_strict_binary_plaintext_raises_without_the_bytes(use_key):
    """A token whose plaintext is not UTF-8 raises the typed error and keeps
    the decrypted bytes out of the exception entirely -- not in the message,
    not in args, and not reachable through __cause__ or __context__."""
    use_key(STRICT_KEY_A)
    secret = b"\xff\xfe\x00secret-bytes"
    token = get_cipher().encrypt(secret).decode()
    with pytest.raises(EncryptionDecodeError) as excinfo:
        decrypt_value_strict(token)
    exc = excinfo.value
    assert exc.__cause__ is None
    assert exc.__context__ is None
    assert b"secret-bytes" not in repr(exc.args).encode()
    assert "secret-bytes" not in str(exc)


@pytest.mark.parametrize("cut", [4, 8, 12, 16, 20])
def test_decrypt_value_strict_truncated_token_still_raises(use_key, cut):
    """A token that lost a few base64 characters in transit is still reported.

    Its decoded length is no longer a whole number of AES blocks, so no key
    could open it -- but it is ciphertext, and handing it back as if it were
    plaintext is exactly the outcome the strict helper exists to prevent.
    The shape floor is a minimum, not an alignment rule, on purpose: a
    two-block token (89 raw bytes) cut by 4..20 base64 characters decodes to
    87/84/81/78/75 bytes, all still above the floor and all still reported.
    (A one-block token is already at the floor, so a truncated one falls
    below it under any rule; this test uses the two-block case.)
    """
    use_key(STRICT_KEY_A)
    token = get_cipher().encrypt(b"a-secret-of-20-bytes").decode()
    truncated = token[:-cut]
    with pytest.raises(EncryptionDecodeError):
        decrypt_value_strict(truncated)
