from xagent.core.ssh.errors import SshError, SshErrorCode


def test_error_codes_have_stable_string_values() -> None:
    # These strings are a public contract with the xagent-cloud layer and clients.
    assert SshErrorCode.TARGET_NOT_FOUND.value == "ssh_target_not_found"
    assert SshErrorCode.HOST_KEY_MISMATCH.value == "ssh_host_key_mismatch"
    assert SshErrorCode.EGRESS_DENIED.value == "ssh_egress_denied"
    assert SshErrorCode.APPROVAL_REQUIRED.value == "ssh_approval_required"
    assert SshErrorCode.CONCURRENT_UPDATE.value == "ssh_concurrent_update"


def test_cross_seam_consumer_raised_codes_are_stable() -> None:
    # These four are raised by out-of-repo consumers across the provider /
    # secret-store / audit seam and only *recorded* by core, so an in-repo grep
    # sees no raise-site and can wrongly flag them as dead (this already happened
    # once). Pinning their values here makes a delete/rename fail CI in-repo.
    assert SshErrorCode.TARGET_IN_USE.value == "ssh_target_in_use"
    assert SshErrorCode.CREDENTIAL_REVOKED.value == "ssh_credential_revoked"
    assert SshErrorCode.CREDENTIAL_IN_USE.value == "ssh_credential_in_use"
    assert SshErrorCode.HOST_KEY_UNVERIFIED.value == "ssh_host_key_unverified"


def test_all_codes_are_lowercase_ssh_prefixed() -> None:
    for code in SshErrorCode:
        assert code.value.startswith("ssh_")
        assert code.value == code.value.lower()


def test_ssh_error_exposes_code_and_message() -> None:
    err = SshError(SshErrorCode.TARGET_DISABLED, "target is disabled")
    assert err.code is SshErrorCode.TARGET_DISABLED
    assert str(err) == "target is disabled"


def test_ssh_error_to_dict_shape() -> None:
    err = SshError(
        SshErrorCode.EGRESS_DENIED,
        "destination not allowed",
        context={"target": "t-1"},
    )
    data = err.to_dict()
    assert data == {
        "error_code": "ssh_egress_denied",
        "message": "destination not allowed",
        "context": {"target": "t-1"},
    }


def test_ssh_error_preserves_cause() -> None:
    root = ValueError("boom")
    err = SshError(SshErrorCode.SECRET_UNAVAILABLE, "secret store down", cause=root)
    assert err.cause is root


def test_default_context_is_isolated_between_instances() -> None:
    a = SshError(SshErrorCode.TARGET_NOT_FOUND, "a")
    b = SshError(SshErrorCode.TARGET_NOT_FOUND, "b")
    a.context["x"] = 1
    assert b.context == {}


def test_ssh_error_chains_cause_for_traceback() -> None:
    root = ValueError("boom")
    err = SshError(SshErrorCode.SECRET_UNAVAILABLE, "secret store down", cause=root)
    assert err.__cause__ is root
