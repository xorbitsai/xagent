"""Tests for the Chromium privilege-dropping launcher."""

import grp
import os
import pwd
import shlex
import shutil
import stat
import subprocess

import pytest

from xagent.core.tools.core import browser_privdrop, browser_use


def _non_root_identity() -> tuple[int, int] | None:
    if os.geteuid() != 0:
        return os.geteuid(), os.getegid()
    for name in ("nobody", "daemon"):
        try:
            entry = pwd.getpwnam(name)
        except KeyError:
            continue
        try:
            group = grp.getgrgid(entry.pw_gid)
        except KeyError:
            continue
        if entry.pw_uid != 0 and group.gr_gid != 0:
            return entry.pw_uid, group.gr_gid
    return None


def _profile(tmp_path):
    profile = tmp_path / "playwright_chromiumdev_profile-test"
    profile.mkdir()
    return profile


def _argv(profile, *, uid="1000", gid="1000"):
    return [
        uid,
        gid,
        "/opt/chrome",
        f"--user-data-dir={profile}",
        "--headless=new",
    ]


def _identity_fakes(*, allow_setuid_zero: bool = False):
    state = {"uid": 0, "gid": 0}
    calls: list[tuple[str, object]] = []

    def setgroups(groups):
        calls.append(("setgroups", list(groups)))

    def setgid(gid):
        calls.append(("setgid", gid))
        state["gid"] = gid

    def setuid(uid):
        calls.append(("setuid", uid))
        if uid == 0 and state["uid"] != 0 and not allow_setuid_zero:
            raise PermissionError("setuid(0) denied")
        state["uid"] = uid

    return (
        {
            "setgroups": setgroups,
            "setgid": setgid,
            "setuid": setuid,
            "getuid": lambda: state["uid"],
            "geteuid": lambda: state["uid"],
            "getgid": lambda: state["gid"],
            "getegid": lambda: state["gid"],
        },
        calls,
    )


def _stat(mode, *, nlink=1, uid=0):
    return os.stat_result((mode, 1, 1, nlink, uid, 0, 0, 0, 0, 0))


class _FakeTree:
    def __init__(self):
        self.root_fd = 10
        self.next_fd = 11
        self.stats = {self.root_fd: _stat(stat.S_IFDIR)}
        self.children = {self.root_fd: []}
        self.entries = {}
        self.child_fds = {}
        self.open_calls = []
        self.chown_calls = []
        self.fchown_calls = []
        self.close_calls = []
        self.stat_calls = []
        self.events = []

    def add(self, parent_fd, name, entry_stat, *, directory=False):
        self.children[parent_fd].append(name)
        self.entries[(parent_fd, name)] = entry_stat
        if directory:
            child_fd = self.next_fd
            self.next_fd += 1
            self.stats[child_fd] = entry_stat
            self.children[child_fd] = []
            self.child_fds[(parent_fd, name)] = child_fd
            return child_fd
        return None

    def open_fn(self, path, flags, dir_fd=None):
        self.open_calls.append((path, flags, dir_fd))
        if path == "/profile":
            return self.root_fd
        return self.child_fds[(dir_fd, path)]

    def fstat(self, fd):
        return self.stats[fd]

    def stat(self, name, *, dir_fd, follow_symlinks):
        assert follow_symlinks is False
        self.stat_calls.append((name, dir_fd))
        return self.entries[(dir_fd, name)]

    def chown(self, name, uid, gid, *, dir_fd, follow_symlinks):
        assert follow_symlinks is False
        self.chown_calls.append((name, uid, gid, dir_fd))
        self.events.append(("chown", name))

    def fchown(self, fd, uid, gid):
        self.fchown_calls.append((fd, uid, gid))
        self.events.append(("fchown", fd))

    def close(self, fd):
        self.close_calls.append(fd)

    def listdir(self, fd):
        return list(self.children[fd])

    def run(self):
        browser_privdrop.chown_tree(
            "/profile",
            1234,
            5678,
            open_fn=self.open_fn,
            fstat=self.fstat,
            stat=self.stat,
            chown=self.chown,
            fchown=self.fchown,
            close=self.close,
            listdir=self.listdir,
            is_dir=stat.S_ISDIR,
        )


def test_parse_accepts_playwright_profile(tmp_path):
    profile = _profile(tmp_path)

    spec = browser_privdrop.parse_launch_argv(_argv(profile))

    assert spec.uid == 1000
    assert spec.gid == 1000
    assert spec.real_chrome == "/opt/chrome"
    assert spec.chrome_args[0] == f"--user-data-dir={profile}"
    assert spec.user_data_dir == str(profile)


@pytest.mark.parametrize("uid, gid", [("0", "1000"), ("1000", "4294967295")])
def test_parse_rejects_out_of_range_ids(tmp_path, uid, gid):
    argv = [uid, gid, *_argv(_profile(tmp_path))[2:]]

    with pytest.raises(browser_privdrop.LaunchError, match="between 1 and"):
        browser_privdrop.parse_launch_argv(argv)


def test_parse_rejects_missing_user_data_dir():
    argv = ["1000", "1000", "/opt/chrome", "--headless=new"]

    with pytest.raises(browser_privdrop.LaunchError, match="exactly one"):
        browser_privdrop.parse_launch_argv(argv)


def test_parse_rejects_duplicate_user_data_dir(tmp_path):
    profile = _profile(tmp_path)
    argv = [
        "1000",
        "1000",
        "/opt/chrome",
        f"--user-data-dir={profile}",
        f"--user-data-dir={profile}",
    ]

    with pytest.raises(browser_privdrop.LaunchError, match="exactly one"):
        browser_privdrop.parse_launch_argv(argv)


def test_parse_rejects_relative_user_data_dir():
    argv = [
        "1000",
        "1000",
        "/opt/chrome",
        "--user-data-dir=playwright_chromiumdev_profile-relative",
    ]

    with pytest.raises(browser_privdrop.LaunchError, match="absolute"):
        browser_privdrop.parse_launch_argv(argv)


def test_parse_rejects_symlink_user_data_dir(tmp_path):
    profile = _profile(tmp_path)
    link = tmp_path / "playwright_chromiumdev_profile-link"
    link.symlink_to(profile, target_is_directory=True)

    with pytest.raises(browser_privdrop.LaunchError, match="symlink"):
        browser_privdrop.parse_launch_argv(_argv(link))


def test_parse_rejects_wrong_prefix(tmp_path):
    profile = tmp_path / "not-a-playwright-profile"
    profile.mkdir()

    with pytest.raises(browser_privdrop.LaunchError, match="Playwright"):
        browser_privdrop.parse_launch_argv(_argv(profile))


def test_chown_tree_rejects_root_not_owned_by_root():
    tree = _FakeTree()
    tree.stats[tree.root_fd] = _stat(stat.S_IFDIR, uid=1234)

    with pytest.raises(browser_privdrop.LaunchError, match="owned by root"):
        tree.run()

    assert tree.chown_calls == []
    assert tree.fchown_calls == []
    assert tree.close_calls == [tree.root_fd]


def test_chown_tree_rejects_hard_linked_entry():
    tree = _FakeTree()
    tree.add(tree.root_fd, "hardlink", _stat(stat.S_IFREG, nlink=2))

    with pytest.raises(browser_privdrop.LaunchError, match="hard-linked"):
        tree.run()

    assert tree.chown_calls == []
    assert tree.fchown_calls == []
    assert tree.close_calls == [tree.root_fd]


def test_chown_tree_chowns_symlink_without_descending():
    tree = _FakeTree()
    tree.add(tree.root_fd, "link", _stat(stat.S_IFLNK))

    tree.run()

    assert tree.open_calls == [("/profile", tree.open_calls[0][1], None)]
    assert tree.chown_calls == [("link", 1234, 5678, tree.root_fd)]
    assert tree.fchown_calls == [(tree.root_fd, 1234, 5678)]


def test_chown_tree_opens_subdirectory_with_nofollow_and_chowns_root_last():
    tree = _FakeTree()
    tree.add(tree.root_fd, "file", _stat(stat.S_IFREG))
    sub_fd = tree.add(tree.root_fd, "sub", _stat(stat.S_IFDIR), directory=True)
    tree.add(sub_fd, "nested", _stat(stat.S_IFREG))

    tree.run()

    root_open = tree.open_calls[0]
    assert root_open[0] == "/profile"
    assert root_open[1] & os.O_NOFOLLOW
    assert root_open[1] & os.O_DIRECTORY
    assert root_open[1] & os.O_CLOEXEC

    sub_open = next(call for call in tree.open_calls if call[0] == "sub")
    assert sub_open[1] & os.O_NOFOLLOW
    assert sub_open[1] & os.O_DIRECTORY
    assert sub_open[2] == tree.root_fd

    assert tree.events.index(("chown", "nested")) < tree.events.index(("chown", "sub"))
    assert tree.events[-1] == ("fchown", tree.root_fd)
    assert tree.close_calls[-1] == tree.root_fd


def test_launch_drops_identity_then_execs_chromium(tmp_path):
    profile = _profile(tmp_path)
    argv = _argv(profile)
    fakes, calls = _identity_fakes()
    exec_calls = []
    environ = {"PATH": "/usr/bin"}

    browser_privdrop.launch(
        argv,
        **fakes,
        chown_profile=lambda *_args, **_kwargs: None,
        execve=lambda chrome, args, env: exec_calls.append((chrome, args, env)),
        environ=environ,
    )

    assert calls[:3] == [
        ("setgroups", []),
        ("setgid", 1000),
        ("setuid", 1000),
    ]
    assert calls[3] == ("setuid", 0)
    assert exec_calls == [
        (
            "/opt/chrome",
            ["/opt/chrome", *argv[3:]],
            {"PATH": "/usr/bin", "HOME": str(profile)},
        )
    ]
    assert "HOME" not in environ


def test_launch_execs_with_allowlisted_environment(tmp_path):
    profile = _profile(tmp_path)
    fakes, _ = _identity_fakes()
    exec_calls = []
    parent_env = {
        "PATH": "/usr/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LANGUAGE": "en",
        "TZ": "UTC",
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "LC_ALL": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "FONTCONFIG_PATH": "/etc/fonts",
        "SECRET_KEY": "secret",
        "DATABASE_URL": "postgres://secret",
    }

    browser_privdrop.launch(
        _argv(profile),
        **fakes,
        chown_profile=lambda *_args, **_kwargs: None,
        execve=lambda chrome, args, env: exec_calls.append((chrome, args, env)),
        environ=parent_env,
    )

    assert exec_calls[0][2] == {
        "PATH": "/usr/bin",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LANGUAGE": "en",
        "TZ": "UTC",
        "DISPLAY": ":0",
        "WAYLAND_DISPLAY": "wayland-0",
        "LC_ALL": "C.UTF-8",
        "XDG_RUNTIME_DIR": "/run/user/1000",
        "FONTCONFIG_PATH": "/etc/fonts",
        "HOME": str(profile),
    }


def test_launch_aborts_when_setuid_zero_still_succeeds(tmp_path):
    profile = _profile(tmp_path)
    fakes, _ = _identity_fakes(allow_setuid_zero=True)
    exec_calls = []

    with pytest.raises(browser_privdrop.LaunchError, match=r"setuid\(0\)"):
        browser_privdrop.launch(
            _argv(profile),
            **fakes,
            chown_profile=lambda *_args, **_kwargs: None,
            execve=lambda chrome, args, env: exec_calls.append((chrome, args, env)),
            environ={},
        )

    assert exec_calls == []


def test_launch_aborts_on_gid_mismatch(tmp_path):
    profile = _profile(tmp_path)
    fakes, _ = _identity_fakes()
    fakes["getgid"] = lambda: 4321
    fakes["getegid"] = lambda: 4321
    exec_calls = []

    with pytest.raises(browser_privdrop.LaunchError, match="gid mismatch"):
        browser_privdrop.launch(
            _argv(profile),
            **fakes,
            chown_profile=lambda *_args, **_kwargs: None,
            execve=lambda chrome, args, env: exec_calls.append((chrome, args, env)),
            environ={},
        )

    assert exec_calls == []


def test_main_reports_validation_error(capsys):
    assert browser_privdrop.main(["1000"]) == 2
    assert "usage" in capsys.readouterr().err


def test_chown_tree_real_tree_as_non_root(tmp_path):
    identity = _non_root_identity()
    if identity is None:
        pytest.skip("no non-root identity available")
    uid, gid = identity
    profile = tmp_path / "playwright_chromiumdev_profile-real"
    nested = profile / "nested"
    nested.mkdir(parents=True)
    payload = nested / "file"
    payload.write_text("data", encoding="utf-8")
    target = tmp_path / "target.txt"
    target.write_text("secret", encoding="utf-8")
    link = profile / "link"
    link.symlink_to(target)
    target_owner = os.lstat(target).st_uid

    if os.geteuid() == 0:
        if not hasattr(os, "fork"):
            pytest.skip("fork unavailable")
        for path in (profile, nested, payload, link):
            os.chown(path, uid, gid, follow_symlinks=False)
        pid = os.fork()
        if pid == 0:
            try:
                os.chdir(profile)
                os.setgroups([])
                os.setgid(gid)
                os.setuid(uid)
                browser_privdrop.chown_tree(".", uid, gid, expected_owner=uid)
            except BaseException:
                os._exit(1)
            os._exit(0)
        _, status = os.waitpid(pid, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        assert target_owner == 0
    else:
        browser_privdrop.chown_tree(str(profile), uid, gid, expected_owner=uid)

    assert os.lstat(target).st_uid == target_owner
    assert os.lstat(link).st_uid == uid
    assert os.lstat(payload).st_uid == uid
    assert os.lstat(profile).st_uid == uid


def test_real_launcher_stops_at_privilege_drop(tmp_path):
    identity = _non_root_identity()
    if identity is None:
        pytest.skip("no non-root identity available")
    uid, gid = identity
    fake_chrome = tmp_path / "fake chrome"
    launcher_dir, launcher = browser_use._create_browser_launcher(
        uid, gid, str(fake_chrome)
    )
    try:
        profile = launcher_dir / "playwright_chromiumdev_profile-real"
        profile.mkdir()
        marker = profile / "chrome-ran"
        fake_chrome.write_text(
            f"#!/bin/sh\nprintf ran > {shlex.quote(str(marker))}\n",
            encoding="utf-8",
        )
        fake_chrome.chmod(0o755)
        if os.geteuid() == 0:
            os.chown(profile, uid, gid)
            profile.chmod(0o700)
            launcher_dir.chmod(0o755)
            launcher.chmod(0o755)
            (launcher_dir / "browser_privdrop.py").chmod(0o644)
            run_kwargs = {"user": uid, "group": gid, "extra_groups": []}
        else:
            run_kwargs = {}

        result = subprocess.run(
            [
                str(launcher),
                f"--user-data-dir={profile}",
                "--headless=new",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(launcher_dir),
            **run_kwargs,
        )

        assert result.returncode != 0
        assert not marker.exists()
        assert "browser_privdrop:" in result.stderr
        assert "Operation not permitted" in result.stderr
    finally:
        shutil.rmtree(launcher_dir, ignore_errors=True)
