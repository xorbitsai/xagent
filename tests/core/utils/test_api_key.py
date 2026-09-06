"""Unit tests for ``xagent.core.utils.api_key``.

Covers the three key contracts callers rely on:

  - Generation produces a syntactically correct key (brand + alphabet
    + lengths) and a bcrypt hash that verifies against it.
  - Parse is strict -- any deviation from the format returns None
    rather than partial / lenient parses, so a bad header never
    reaches the bcrypt step.
  - verify_api_key matches bcrypt's contract (true on match, false
    on miss, false on garbage input rather than raising).

Plus a couple of robustness checks:

  - verify_dummy performs a bcrypt check with the same cost as real API
    keys, so missing prefixes do not skip the expensive verification.
  - generate_api_key retries on prefix collision and gives up cleanly
    if a mock keeps colliding.
"""

import re
from unittest.mock import MagicMock, patch

import pytest

from xagent.core.utils import api_key as api_key_module
from xagent.core.utils.api_key import (
    BCRYPT_COST,
    KEY_ALPHABET,
    KEY_BRAND,
    KEY_PREFIX_LENGTH,
    KEY_SECRET_LENGTH,
    PREFIX_COLLISION_RETRIES,
    ApiKeyKind,
    generate_api_key,
    parse_api_key,
    verify_api_key,
    verify_dummy,
)

# ===== generate_api_key =====


def test_generate_format() -> None:
    """Generated key matches xag_<6 alnum>_<32 alnum>; halves stay within alphabet."""
    full, prefix, key_hash = generate_api_key(db=None)

    # Brand + segment lengths
    assert full.startswith(f"{KEY_BRAND}_")
    parts = full.split("_")
    assert len(parts) == 3
    assert parts[0] == KEY_BRAND
    assert len(parts[1]) == KEY_PREFIX_LENGTH
    assert len(parts[2]) == KEY_SECRET_LENGTH

    # Returned prefix matches the embedded prefix segment
    assert parts[1] == prefix
    assert len(prefix) == KEY_PREFIX_LENGTH

    # Alphabet constraint -- no underscores, dashes, or other glyphs slip in
    alphabet_re = re.compile(f"^[{re.escape(KEY_ALPHABET)}]+$")
    assert alphabet_re.fullmatch(parts[1])
    assert alphabet_re.fullmatch(parts[2])

    # bcrypt hash is the standard $2b$ prefix with our cost factor
    assert key_hash.startswith(f"$2b${BCRYPT_COST:02d}$")


def test_generate_persists_only_hash() -> None:
    """The returned hash verifies the full key; the secret is never returned twice."""
    full, _prefix, key_hash = generate_api_key(db=None)
    # Round-trip: hash verifies its source
    assert verify_api_key(full, key_hash) is True
    # Sanity: hash does NOT verify a different key
    assert verify_api_key(full + "X", key_hash) is False


def test_generate_prefix_collision_retry() -> None:
    """First call collides (mock returns existing row), second succeeds."""
    mock_db = MagicMock()
    # Sequence of .filter(...).first() results:
    #   1st call -> simulated collision (returns a truthy "row" object)
    #   2nd call -> no collision (returns None)
    mock_db.query.return_value.filter.return_value.first.side_effect = [
        object(),  # collision on first prefix
        None,  # second prefix is free
    ]

    full, prefix, key_hash = generate_api_key(db=mock_db)
    assert len(prefix) == KEY_PREFIX_LENGTH
    assert verify_api_key(full, key_hash) is True
    # We called the DB exactly twice (one collision + one success)
    assert mock_db.query.return_value.filter.return_value.first.call_count == 2


def test_generate_gives_up_after_retry_cap() -> None:
    """All PREFIX_COLLISION_RETRIES draws colliding -> RuntimeError, not infinite loop."""
    mock_db = MagicMock()
    # Always return a truthy "existing row" to force perpetual collision
    mock_db.query.return_value.filter.return_value.first.return_value = object()

    with pytest.raises(RuntimeError, match="unique key prefix"):
        generate_api_key(db=mock_db)

    # Exhausted the retry budget exactly
    assert (
        mock_db.query.return_value.filter.return_value.first.call_count
        == PREFIX_COLLISION_RETRIES
    )


# ===== parse_api_key =====


def test_parse_valid() -> None:
    """Well-formed key splits cleanly into (prefix, secret)."""
    full, prefix, _hash = generate_api_key(db=None)
    parsed = parse_api_key(full)
    assert parsed is not None
    assert parsed.kind == ApiKeyKind.AGENT
    assert parsed.prefix == prefix
    assert len(parsed.secret) == KEY_SECRET_LENGTH
    # Reassembly round-trip
    assert f"{KEY_BRAND}_{parsed.prefix}_{parsed.secret}" == full


def test_generate_and_parse_personal_key() -> None:
    """Personal keys include an explicit kind segment."""
    full, prefix, key_hash = generate_api_key(db=None, kind=ApiKeyKind.PERSONAL)
    assert full.startswith(f"{KEY_BRAND}_{ApiKeyKind.PERSONAL.value}_")
    parts = full.split("_")
    assert len(parts) == 4
    assert parts[2] == prefix
    assert verify_api_key(full, key_hash) is True

    parsed = parse_api_key(full)
    assert parsed is not None
    assert parsed.kind == ApiKeyKind.PERSONAL
    assert parsed.prefix == prefix
    assert len(parsed.secret) == KEY_SECRET_LENGTH


@pytest.mark.parametrize(
    "bad",
    [
        "",  # empty
        "no_brand_prefix",  # wrong brand
        "xag_only_two_parts",  # 3 underscores -> 4 parts? actually wrong split
        "xag_short_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # prefix len != 6
        "xag_ABCDEF_short",  # secret len != 32
        "xag_ABCDEF_" + "!" * KEY_SECRET_LENGTH,  # secret contains forbidden char
    ],
)
def test_parse_invalid(bad: str) -> None:
    """Any deviation from xag_<6>_<32 alnum> returns None, not a partial parse."""
    assert parse_api_key(bad) is None


def test_parse_non_string_input() -> None:
    """Non-string / None input returns None rather than raising."""
    assert parse_api_key(None) is None  # type: ignore[arg-type]
    assert parse_api_key(123) is None  # type: ignore[arg-type]


# ===== verify_api_key =====


def test_verify_correct() -> None:
    """A freshly generated key verifies against its own hash."""
    full, _prefix, key_hash = generate_api_key(db=None)
    assert verify_api_key(full, key_hash) is True


def test_verify_wrong_secret() -> None:
    """Tampering with even a single secret char fails verification."""
    full, _prefix, key_hash = generate_api_key(db=None)
    # Flip the last char of the secret to something else still in the alphabet
    flipped = full[:-1] + ("A" if full[-1] != "A" else "B")
    assert verify_api_key(flipped, key_hash) is False


def test_verify_empty_inputs_return_false() -> None:
    """Empty / malformed inputs return False rather than raising."""
    assert verify_api_key("", "") is False
    assert verify_api_key("xag_ABCDEF_anything", "") is False
    assert verify_api_key("", "$2b$12$notarealhash") is False


def test_verify_garbage_hash_returns_false() -> None:
    """Malformed bcrypt hash strings produce False, not ValueError leaking out."""
    full, _prefix, _hash = generate_api_key(db=None)
    assert verify_api_key(full, "not-a-bcrypt-hash") is False


# ===== verify_dummy =====


def test_verify_dummy_runs_without_raising() -> None:
    """Dummy verification is callable and returns None."""
    assert verify_dummy() is None


def test_verify_dummy_uses_the_same_bcrypt_cost_as_real_verification() -> None:
    """Both paths execute bcrypt at the configured cost, without timing CI."""
    full, _prefix, key_hash = generate_api_key(db=None)
    with patch.object(
        api_key_module.bcrypt, "checkpw", wraps=api_key_module.bcrypt.checkpw
    ) as checkpw:
        assert verify_api_key(full, key_hash) is True
        checkpw.assert_called_once_with(full.encode(), key_hash.encode())
        checkpw.reset_mock()

        assert verify_dummy() is None
        checkpw.assert_called_once()
        dummy_password, dummy_hash = checkpw.call_args.args
        assert isinstance(dummy_password, bytes)
        assert isinstance(dummy_hash, bytes)
        assert int(dummy_hash.split(b"$")[2]) == BCRYPT_COST
        assert int(key_hash.split("$")[2]) == BCRYPT_COST
