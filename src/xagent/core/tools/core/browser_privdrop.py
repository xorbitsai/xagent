"""Launch Chromium under a dedicated non-root identity.

Invoked by a generated launcher as ``uid gid real_chrome chrome_args...``. The
Playwright driver stays root; only the browser process drops privileges.
"""

import os
import stat as stat_module
import sys
from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass

_USER_DATA_DIR_ARG = "--user-data-dir="
_USER_DATA_DIR_PREFIX = "playwright_chromiumdev_profile-"
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class LaunchError(RuntimeError):
    """A launch decision failed its validation."""


@dataclass(frozen=True)
class LaunchSpec:
    """Validated launcher inputs."""

    uid: int
    gid: int
    real_chrome: str
    chrome_args: tuple[str, ...]
    user_data_dir: str


def parse_launch_argv(
    argv: Sequence[str],
    *,
    isabs: Callable[[str], bool] = os.path.isabs,
    isdir: Callable[[str], bool] = os.path.isdir,
    islink: Callable[[str], bool] = os.path.islink,
) -> LaunchSpec:
    """Validate wrapper arguments and return the launch plan."""
    if len(argv) < 3:
        raise LaunchError("usage: browser_privdrop <uid> <gid> <chrome> [args...]")
    try:
        uid = int(argv[0])
        gid = int(argv[1])
    except ValueError:
        raise LaunchError("uid and gid must be integers") from None
    if uid < 1 or gid < 1:
        raise LaunchError("uid and gid must be positive")
    chrome_args = tuple(argv[3:])
    flag_values = [arg for arg in chrome_args if arg.startswith(_USER_DATA_DIR_ARG)]
    if len(flag_values) != 1:
        raise LaunchError("exactly one --user-data-dir argument is required")
    user_data_dir = flag_values[0][len(_USER_DATA_DIR_ARG) :]
    if not isabs(user_data_dir):
        raise LaunchError("--user-data-dir must be an absolute path")
    if islink(user_data_dir):
        raise LaunchError("--user-data-dir must not be a symlink")
    if not isdir(user_data_dir):
        raise LaunchError("--user-data-dir must be an existing directory")
    if not os.path.basename(user_data_dir).startswith(_USER_DATA_DIR_PREFIX):
        raise LaunchError("--user-data-dir is not a Playwright temporary profile")
    return LaunchSpec(uid, gid, argv[2], chrome_args, user_data_dir)


def chown_tree(
    root: str,
    uid: int,
    gid: int,
    *,
    open_fn: Callable[..., int] = os.open,
    fstat: Callable[[int], os.stat_result] = os.fstat,
    stat: Callable[..., os.stat_result] = os.stat,
    chown: Callable[..., None] = os.chown,
    fchown: Callable[[int, int, int], None] = os.fchown,
    close: Callable[[int], None] = os.close,
    listdir: Callable[[int], list[str]] = os.listdir,
    is_dir: Callable[[int], bool] = stat_module.S_ISDIR,
) -> None:
    """Chown the profile fd-relatively without following symlinks, root last."""
    root_fd = open_fn(root, _DIRECTORY_OPEN_FLAGS)
    try:
        root_stat = fstat(root_fd)
        if not is_dir(root_stat.st_mode) or root_stat.st_uid != 0:
            raise LaunchError("--user-data-dir must be a directory owned by root")
        _chown_children(
            root_fd,
            uid,
            gid,
            open_fn=open_fn,
            stat=stat,
            chown=chown,
            close=close,
            listdir=listdir,
            is_dir=is_dir,
        )
        fchown(root_fd, uid, gid)
    finally:
        close(root_fd)


def _chown_children(
    fd: int,
    uid: int,
    gid: int,
    *,
    open_fn: Callable[..., int],
    stat: Callable[..., os.stat_result],
    chown: Callable[..., None],
    close: Callable[[int], None],
    listdir: Callable[[int], list[str]],
    is_dir: Callable[[int], bool],
) -> None:
    for name in listdir(fd):
        entry = stat(name, dir_fd=fd, follow_symlinks=False)
        if is_dir(entry.st_mode):
            child_fd = open_fn(name, _DIRECTORY_OPEN_FLAGS, dir_fd=fd)
            try:
                _chown_children(
                    child_fd,
                    uid,
                    gid,
                    open_fn=open_fn,
                    stat=stat,
                    chown=chown,
                    close=close,
                    listdir=listdir,
                    is_dir=is_dir,
                )
            finally:
                close(child_fd)
            chown(name, uid, gid, dir_fd=fd, follow_symlinks=False)
            continue
        if entry.st_nlink > 1:
            raise LaunchError(f"refusing to chown hard-linked entry {name!r}")
        chown(name, uid, gid, dir_fd=fd, follow_symlinks=False)


def drop_privileges(
    uid: int,
    gid: int,
    *,
    setgroups: Callable[[Sequence[int]], None] = os.setgroups,
    setgid: Callable[[int], None] = os.setgid,
    setuid: Callable[[int], None] = os.setuid,
    getuid: Callable[[], int] = os.getuid,
    geteuid: Callable[[], int] = os.geteuid,
    getgid: Callable[[], int] = os.getgid,
    getegid: Callable[[], int] = os.getegid,
) -> None:
    """Drop groups, gid and uid, then verify the new identity is irreversible."""
    setgroups([])
    setgid(gid)
    setuid(uid)
    if getuid() != uid or geteuid() != uid:
        raise LaunchError("privilege drop verification failed: uid mismatch")
    if getgid() != gid or getegid() != gid:
        raise LaunchError("privilege drop verification failed: gid mismatch")
    try:
        setuid(0)
    except OSError:
        return
    raise LaunchError("privilege drop verification failed: setuid(0) succeeded")


def launch(
    argv: Sequence[str],
    *,
    isabs: Callable[[str], bool] = os.path.isabs,
    isdir: Callable[[str], bool] = os.path.isdir,
    islink: Callable[[str], bool] = os.path.islink,
    chown_profile: Callable[[str, int, int], None] = chown_tree,
    setgroups: Callable[[Sequence[int]], None] = os.setgroups,
    setgid: Callable[[int], None] = os.setgid,
    setuid: Callable[[int], None] = os.setuid,
    getuid: Callable[[], int] = os.getuid,
    geteuid: Callable[[], int] = os.geteuid,
    getgid: Callable[[], int] = os.getgid,
    getegid: Callable[[], int] = os.getegid,
    execv: Callable[[str, list[str]], object] = os.execv,
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Validate, chown the profile, drop privileges and exec Chromium."""
    spec = parse_launch_argv(argv, isabs=isabs, isdir=isdir, islink=islink)
    chown_profile(spec.user_data_dir, spec.uid, spec.gid)
    drop_privileges(
        spec.uid,
        spec.gid,
        setgroups=setgroups,
        setgid=setgid,
        setuid=setuid,
        getuid=getuid,
        geteuid=geteuid,
        getgid=getgid,
        getegid=getegid,
    )
    (os.environ if environ is None else environ)["HOME"] = spec.user_data_dir
    execv(spec.real_chrome, [spec.real_chrome, *spec.chrome_args])


def main(argv: Sequence[str] | None = None) -> int:
    """Run the launcher, reporting validation failures on stderr."""
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        launch(args)
    except (LaunchError, OSError) as exc:
        print(f"browser_privdrop: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
