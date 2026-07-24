from xagent.core.ssh.egress import EgressPolicyConfig, check_ip


def test_public_ipv4_allowed_by_default() -> None:
    decision = check_ip("93.184.216.34", EgressPolicyConfig())
    assert decision.allowed is True


def test_loopback_denied() -> None:
    assert check_ip("127.0.0.1", EgressPolicyConfig()).allowed is False
    assert check_ip("::1", EgressPolicyConfig()).allowed is False


def test_link_local_denied() -> None:
    assert check_ip("169.254.0.5", EgressPolicyConfig()).allowed is False
    assert check_ip("fe80::1", EgressPolicyConfig()).allowed is False


def test_multicast_denied() -> None:
    # Multicast used to fall through to default_allow_public (m6); it's denied
    # unconditionally — even when deny_private is off, since the two are
    # orthogonal.
    assert check_ip("224.0.0.1", EgressPolicyConfig()).allowed is False
    assert (
        check_ip("224.0.0.1", EgressPolicyConfig(deny_private=False)).allowed is False
    )


def test_cloud_metadata_denied_even_if_link_local_check_disabled() -> None:
    config = EgressPolicyConfig(deny_link_local=False)
    decision = check_ip("169.254.169.254", config)
    assert decision.allowed is False
    assert "metadata" in decision.reason


def test_private_ranges_denied_v4_and_v6() -> None:
    for ip in ("10.0.0.1", "172.16.5.4", "192.168.1.1", "fc00::1"):
        assert check_ip(ip, EgressPolicyConfig()).allowed is False, ip


def test_extra_denied_cidr_blocks_platform_internal() -> None:
    config = EgressPolicyConfig(extra_denied_cidrs=("100.64.0.0/10",))
    assert check_ip("100.64.1.1", config).allowed is False


def test_allowlist_overrides_private_denial() -> None:
    # A customer private host explicitly allowlisted via deployment policy.
    config = EgressPolicyConfig(allow_cidrs=("10.10.0.0/16",))
    assert check_ip("10.10.5.5", config).allowed is True
    # But an address outside the allowlist is still denied.
    assert check_ip("10.20.5.5", config).allowed is False


def test_invalid_ip_denied() -> None:
    decision = check_ip("not-an-ip", EgressPolicyConfig())
    assert decision.allowed is False
    assert "invalid" in decision.reason


def test_public_denied_when_default_deny_public_enabled() -> None:
    # Deny-by-default deployments only permit explicit allowlist.
    config = EgressPolicyConfig(default_allow_public=False)
    assert check_ip("93.184.216.34", config).allowed is False
    config_allowed = EgressPolicyConfig(
        default_allow_public=False, allow_cidrs=("93.184.216.0/24",)
    )
    assert check_ip("93.184.216.34", config_allowed).allowed is True


def test_ipv4_mapped_loopback_denied() -> None:
    # ::ffff:127.0.0.1 is the dual-stack form of loopback and must be denied.
    assert check_ip("::ffff:127.0.0.1", EgressPolicyConfig()).allowed is False


def test_ipv4_mapped_metadata_denied_even_with_private_and_link_local_off() -> None:
    config = EgressPolicyConfig(deny_link_local=False, deny_private=False)
    assert check_ip("::ffff:169.254.169.254", config).allowed is False


def test_ipv4_mapped_public_still_allowed() -> None:
    # A mapped public address should behave like the bare public IPv4.
    assert check_ip("::ffff:93.184.216.34", EgressPolicyConfig()).allowed is True


def test_metadata_denied_even_when_broadly_allowlisted() -> None:
    # A 0.0.0.0/0 allowlist opens private ranges, but the metadata deny sits
    # ahead of the allowlist so instance credentials stay unreachable (#4).
    config = EgressPolicyConfig(allow_cidrs=("0.0.0.0/0",))
    assert check_ip("169.254.169.254", config).allowed is False
    assert check_ip("10.0.0.1", config).allowed is True  # allowlist still works


def test_nat64_embedded_private_denied() -> None:
    # 64:ff9b::/96 wrapping 10.0.0.1 must be judged as the private IPv4 (#5).
    assert check_ip("64:ff9b::0a00:0001", EgressPolicyConfig()).allowed is False


def test_nat64_embedded_public_allowed() -> None:
    # 64:ff9b:: wrapping a public IPv4 (93.184.216.34) stays allowed.
    assert check_ip("64:ff9b::5db8:d822", EgressPolicyConfig()).allowed is True


def test_6to4_embedded_metadata_denied() -> None:
    # 2002:a9fe:a9fe::/48 embeds 169.254.169.254 — decode and deny it (#5).
    assert check_ip("2002:a9fe:a9fe::1", EgressPolicyConfig()).allowed is False
