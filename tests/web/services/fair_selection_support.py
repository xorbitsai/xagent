"""Queued task commands for the fair-share selection tests, on any engine."""

import copy
import random
import threading
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import sessionmaker

from xagent.web.models.task import Task, TaskStatus
from xagent.web.models.task_admission import TaskAdmissionBucket, TaskAdmissionTicket
from xagent.web.models.task_command import TaskExecutionCommand
from xagent.web.models.user import User
from xagent.web.services import task_command_transport as transport
from xagent.web.services.task_dispatch_cap import DispatchCapSnapshot, DispatchPolicy


class World:
    """Commands queued for tenants on any engine; ``add`` gives each its own task,
    ``follow_up`` adds to one."""

    def __init__(self, engine) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        with self.sessions() as db:
            user = User(username="owner", password_hash="unused")
            db.add(user)
            db.commit()
            self.user = user.id
        self.count = 0

    def add(
        self,
        tenant,
        dispatch_class="batch",
        kind="start",
        status="pending",
        *,
        pinned=False,
        unavailable=False,
        priority=0,
    ) -> int:
        """Queue one command of a new task; ``pinned`` puts the task's lease elsewhere."""
        self.count += 1
        later = datetime.now(timezone.utc) + timedelta(hours=1)
        with self.sessions() as db:
            task = Task(
                user_id=self.user,
                title="t",
                status=TaskStatus.PENDING,
                runner_id="other" if pinned else None,
                lease_expires_at=later if pinned else None,
            )
            db.add(task)
            db.flush()
            command = TaskExecutionCommand(
                task_id=task.id,
                command_id=f"c{self.count}",
                kind=kind,
                payload={},
                status=status,
                dispatch_class=dispatch_class,
                fairness_key=tenant,
                priority=priority,
                retry_available_at=later if unavailable else None,
            )
            db.add(command)
            db.commit()
            return command.id

    def follow_up(
        self, task_id, tenant, kind="message", status="pending", priority=0
    ) -> int:
        """Queue one more command of an existing task."""
        self.count += 1
        with self.sessions() as db:
            command = TaskExecutionCommand(
                task_id=task_id,
                command_id=f"f{self.count}",
                kind=kind,
                payload={},
                status=status,
                dispatch_class="batch",
                fairness_key=tenant,
                priority=priority,
            )
            db.add(command)
            db.commit()
            return command.id

    def reset(self) -> None:
        """Empty the queue and the admission tables, keeping the user."""
        with self.engine.begin() as connection:
            for table in (
                "task_admission_tickets",
                "task_admission_buckets",
                "task_execution_commands",
                "tasks",
            ):
                connection.exec_driver_sql(f"DELETE FROM {table}")

    def task_of(self, command_id: int) -> int:
        with self.sessions() as db:
            return db.get(TaskExecutionCommand, command_id).task_id


class FakeClock:
    """The time the skip table reads, moved by the test."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class RecordingLock:
    """A lock that knows which thread holds it and how often it was taken."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._holder: int | None = None
        self.taken = 0

    def held_by_me(self) -> bool:
        return self._holder == threading.get_ident()

    def __enter__(self):
        self._lock.acquire()
        self._holder = threading.get_ident()
        self.taken += 1
        return self

    def __exit__(self, *exc) -> None:
        self._holder = None
        self._lock.release()


# (tenant key, dispatch class, kind) of the pending commands of each world, oldest first.
WORLDS = {
    "three": [
        ("user:1", "batch", "start"),
        ("user:2", "interactive", "start"),
        ("user:3", "batch", "start"),
        ("user:2", "batch", "start"),
        ("user:1", "interactive", "message"),
    ],
}


# --- random worlds and a brute-force reading of the selection rules --------------------

CLASSES = ("batch", "interactive")
CONTROL_KINDS = {"pause", "cancel"}
OPENING_KINDS = {"start", "resume", "resume_input", "message"}
QUEUED = ("pending", "processing")


@dataclass
class Cmd:
    """One command of a random world; ``id`` and ``task`` are filled in when stored.

    Commands with the same ``group`` share a task, and -1 is a task of its own.
    ``claimed`` is a command claimed once and back in pending.
    """

    tenant: str
    dispatch_class: str
    kind: str
    status: str
    pinned: bool = False
    unavailable: bool = False
    ticketed: bool = False
    priority: int = 0
    claimed: bool = False
    group: int = -1
    id: int = 0
    task: int = 0


@dataclass
class RandomWorld:
    commands: list[Cmd]
    full_tenants: set[str] = field(default_factory=set)


def random_world(
    rng: random.Random,
    *,
    tenants: int,
    size: int,
    answers: bool = True,
    admission: bool = True,
    heavy: bool = True,
) -> RandomWorld:
    """A queue of ``size`` commands over ``tenants`` tenants, with the awkward cases in it.

    Some commands wait on a task leased elsewhere, some are not yet due, some
    belong to a tenant whose admission bucket is full, some are settled, and
    some were claimed once and are pending again. Many tasks hold several
    commands, so a command can wait behind an earlier one of its task. With
    ``heavy``, a tenant may have sixteen pinned commands ahead of its first
    claimable one.
    """
    names = [f"user:{n}" for n in range(1, tenants + 1)]
    kinds = ["start"] * 6 + ["message"] * 2 + ["pause"]
    if answers:
        kinds.append("resume_input")
    full = {name for name in names if admission and rng.random() < 0.2}
    commands: list[Cmd] = []
    # One tenant and class with sixteen commands that wait for another process, then
    # one that does not: only a window of seventeen would reach it.
    leading: list[Cmd] = []
    if heavy and rng.random() < 0.4:
        name, klass = rng.choice(names), rng.choice(CLASSES)
        leading = [
            Cmd(name, klass, "start", "pending", pinned=True, priority=10)
            for _ in range(16)
        ]
        leading.append(Cmd(name, klass, "start", "pending", priority=-3))
    for _ in range(size):
        name = rng.choice(names)
        kind = rng.choice(kinds)
        status = rng.choices(["pending", "processing", "completed"], [8, 1, 2])[0]
        commands.append(
            Cmd(
                name,
                "interactive" if kind == "resume_input" else rng.choice(CLASSES),
                kind,
                status,
                pinned=rng.random() < 0.1,
                unavailable=rng.random() < 0.07,
                ticketed=name in full and kind not in CONTROL_KINDS,
                priority=rng.choice((-3, 0, 0, 0, 5, 10)),
                claimed=status == "pending" and rng.random() < 0.1,
            )
        )
    # Pauses and cancels whose key is not the key of any tenant with work, as when
    # a task changed hands after the command that was copied from.
    if rng.random() < 0.4:
        commands += [
            Cmd(
                f"stale:{n}",
                rng.choice(CLASSES),
                rng.choice(["pause", "cancel"]),
                "pending",
            )
            for n in range(rng.randint(1, 3))
        ]
    if rng.random() < 0.3:
        rng.shuffle(commands)
    group_tasks(rng, commands)
    return RandomWorld(leading + commands, full)


def group_tasks(rng: random.Random, commands: list[Cmd]) -> None:
    """Put many commands into the task of an earlier command of their tenant.

    A pause or cancel with a stale key may join any task. A task's lease is the
    task's, so its commands take the first one's ``pinned``.
    """
    groups: list[list[Cmd]] = []
    for cmd in commands:
        stale = cmd.tenant.startswith("stale:")
        joinable = [g for g in groups if stale or g[0].tenant == cmd.tenant]
        if joinable and rng.random() < 0.6:
            members = rng.choice(joinable)
            cmd.group, cmd.pinned = members[0].group, members[0].pinned
            members.append(cmd)
        else:
            cmd.group = len(groups)
            groups.append([cmd])


def store_world(world: "World", spec: RandomWorld) -> None:
    """Write a random world; the tasks of ``active`` commands are chosen by the caller."""
    later = datetime.now(timezone.utc) + timedelta(hours=1)
    with world.sessions() as db:
        for tenant in sorted(spec.full_tenants):
            db.add(
                TaskAdmissionBucket(
                    key=f"bucket:{tenant}", capacity=1, max_pending=99999
                )
            )
        db.flush()
        for tenant in sorted(spec.full_tenants):
            holder = Task(
                user_id=world.user,
                title="holder",
                status=TaskStatus.PENDING,
                runner_id="other",
                lease_attempt_id=f"held-{tenant}",
                lease_expires_at=later,
            )
            db.add(holder)
            db.flush()
            command = TaskExecutionCommand(
                task_id=holder.id,
                command_id=f"held-{tenant}",
                kind="start",
                payload={},
                status="completed",
                fairness_key=tenant,
            )
            db.add(command)
            db.flush()
            db.add(
                TaskAdmissionTicket(
                    command_id=command.id,
                    task_id=holder.id,
                    bucket_key=f"bucket:{tenant}",
                    runner_id="other",
                    owner_attempt_id=f"held-{tenant}",
                )
            )
        tasks: dict[int, int] = {}
        for number, cmd in enumerate(spec.commands):
            if cmd.group >= 0 and cmd.group in tasks:
                task_id = tasks[cmd.group]
            else:
                task = Task(
                    user_id=world.user,
                    title="t",
                    status=TaskStatus.PENDING,
                    runner_id="other" if cmd.pinned else None,
                    lease_expires_at=later if cmd.pinned else None,
                )
                db.add(task)
                db.flush()
                task_id = task.id
                if cmd.group >= 0:
                    tasks[cmd.group] = task_id
            row = TaskExecutionCommand(
                task_id=task_id,
                command_id=f"c{number}",
                kind=cmd.kind,
                payload={},
                status=cmd.status,
                dispatch_class=cmd.dispatch_class,
                fairness_key=cmd.tenant,
                priority=cmd.priority,
                retry_available_at=later if cmd.unavailable else None,
                attempt_count=1 if cmd.claimed else 0,
            )
            db.add(row)
            db.flush()
            cmd.id, cmd.task = row.id, row.task_id
            if cmd.ticketed:
                db.add(
                    TaskAdmissionTicket(
                        command_id=row.id,
                        task_id=row.task_id,
                        bucket_key=f"bucket:{cmd.tenant}",
                    )
                )
        db.commit()


class Reference:
    """The documented selection rules, rewritten over plain objects.

    It shares no code or constants with the dispatcher. The order of the commands of
    one task follows the docstrings of ``_unfinished_earlier_command`` and
    ``_held_below_cap`` clause for clause, so a misreading shared with them is caught
    by the hand-written cases in test_dispatch_batch_line_overtake.py and
    test_dispatch_cap_cancel_overtake.py, not here. ``clock`` is the time source the
    test moves; the reference keeps its own fallback balance and skip table.
    """

    def __init__(
        self,
        commands: list[Cmd],
        clock: Callable[[], float],
        full: set[str] | frozenset[str] = frozenset(),
    ) -> None:
        self.commands = commands
        self.clock = clock
        self.full = full
        self.since_fallback = 0
        self.fallback = False
        self.skips: dict[tuple[str, str], float] = {}
        # Overridable so that a test can pin a rule to a different value.
        self.tenants_per_scan = 4
        self.window = 16
        self.fallback_every = 8
        self.tenant_cap = True
        self.last_capped: str | None = None
        # Counted per call, every scan: calls of claimable that turn a command down
        # only for the order of its own task, and calls of behind_own_task in which
        # a control passes an earlier opening command at the batch line or the limit.
        self.held_back = 0
        self.passed = 0

    def claimable(self, cmd: Cmd, cap, busy) -> bool:
        if cmd.status not in QUEUED or cmd.pinned or cmd.unavailable:
            return False
        if cmd.task in busy or cmd.ticketed and cmd.tenant in self.full:
            return False
        if cap.batch_closed:
            needs_no_slot = cmd.kind in CONTROL_KINDS or cmd.task in cap.active_ids
            if cap.closed:
                allowed = needs_no_slot
            else:
                allowed = needs_no_slot or cmd.dispatch_class == "interactive"
            if not allowed:
                return False
        # Last, so that ``held_back`` counts only commands no other check here holds.
        if self.behind_own_task(cmd, cap):
            self.held_back += 1
            return False
        return True

    def behind_own_task(self, cmd: Cmd, cap) -> bool:
        """Whether an earlier queued command of ``cmd``'s task keeps it back.

        A pause or cancel passes an earlier pending command that waits for its
        admission ticket. With the batch line reached it also passes an earlier
        pending, never-claimed command that would open an execution and that this
        process would not start now, unless the control's own task holds a slot:
        at the limit any such command, below it one held at the batch line.
        """
        passed_batch_line = False
        for earlier in self.commands:
            if (
                earlier.task != cmd.task
                or earlier.id >= cmd.id
                or earlier.status not in QUEUED
            ):
                continue
            if cmd.kind in CONTROL_KINDS and earlier.status == "pending":
                if earlier.ticketed:
                    continue
                if (
                    cap.batch_closed
                    and earlier.kind in OPENING_KINDS
                    and not earlier.claimed
                    and cmd.task not in cap.active_ids
                    and (cap.closed or self.held_at_batch_line(earlier))
                ):
                    passed_batch_line = True
                    continue
            return True
        if passed_batch_line:
            # Counts only a pass over an opening command by the batch line or the
            # limit rule, not a pass over a command that waits for its ticket.
            self.passed += 1
        return False

    def held_at_batch_line(self, cmd: Cmd) -> bool:
        """A batch command, or one behind a pending, never-claimed batch command of
        its task that would open an execution: commands of a task run in order."""
        return cmd.dispatch_class == "batch" or any(
            head.task == cmd.task
            and head.id <= cmd.id
            and head.status == "pending"
            and not head.claimed
            and head.kind in OPENING_KINDS
            and head.dispatch_class == "batch"
            for head in self.commands
        )

    def capped_tenant(self, cap) -> str | None:
        if not self.tenant_cap or not cap.batch_closed or cap.closed:
            return None
        now = self.clock()
        skips = {k for k, until in self.skips.items() if until > now}
        counted: dict[str, int] = {}
        classes: dict[str, set[str]] = {}
        for c in self.commands:
            if c.status in QUEUED and c.kind not in CONTROL_KINDS:
                counted[c.tenant] = min(counted.get(c.tenant, c.id), c.id)
                classes.setdefault(c.tenant, set()).add(c.dispatch_class)
        counted = {
            t: first
            for t, first in counted.items()
            if any((t, k) not in skips for k in classes[t])
        }
        top = max(cap.occupancy.values(), default=0)
        if len(counted) < 2 or top <= 0:
            return None
        holders = [t for t, n in cap.occupancy.items() if n == top and t in counted]
        return min(holders, key=lambda t: counted[t]) if holders else None

    def count(self, cap, cmd: Cmd) -> None:
        """Move the balance for the pick just made: a scan that had room, whose
        command opens an execution of a task not yet active, took a new slot."""
        if cap.closed or cmd.kind in CONTROL_KINDS or cmd.task in cap.active_ids:
            return
        if self.fallback:
            self.since_fallback -= self.fallback_every - 1
        else:
            self.since_fallback += 1

    def pick(self, cap, busy=()) -> int | None:
        has_room = not cap.closed
        fallback = False
        if has_room:
            fallback = self.since_fallback >= self.fallback_every - 1
        self.fallback = fallback
        capped = self.capped_tenant(cap) if has_room else None
        self.last_capped = capped
        usable = [
            c
            for c in self.commands
            if self.claimable(c, cap, busy)
            and not (
                capped is not None
                and c.tenant == capped
                and c.kind not in CONTROL_KINDS
                and c.task not in cap.active_ids
            )
        ]
        today = min((c.id for c in usable), default=None)
        if not has_room:
            return today
        replies = [c.id for c in usable if c.kind == "resume_input"]
        if replies:
            return min(replies)
        if fallback:
            return today
        queued = [c for c in self.commands if c.status in QUEUED]
        oldest: dict[tuple[str, str], int] = {}
        for c in queued:
            if c.kind in CONTROL_KINDS:
                continue  # takes no slot, so it makes no tenant of its own
            key = (c.tenant, c.dispatch_class)
            oldest[key] = min(oldest.get(key, c.id), c.id)
        if cap.batch_closed:
            oldest = {k: v for k, v in oldest.items() if k[1] == "interactive"}
        tenants: dict[str, dict[str, int]] = {}
        for (tenant, klass), first in oldest.items():
            tenants.setdefault(tenant, {})[klass] = first
        if not tenants:
            return today
        # A tenant ranked alone is asked every scan: no skip is read or written.
        alone = len(tenants) == 1
        now = self.clock()
        self.skips = {k: until for k, until in self.skips.items() if until > now}
        skipped = {} if alone else self.skips
        ranked = []
        for tenant, classes in tenants.items():
            remaining = sorted(k for k in classes if (tenant, k) not in skipped)
            if remaining:
                ranked.append(
                    (
                        cap.occupancy.get(tenant, 0),
                        min(classes.values()),
                        tenant,
                        remaining,
                    )
                )
        ranked = [entry for entry in ranked if entry[2] != capped]
        ranked.sort()
        best = None
        for _, _, tenant, remaining in ranked[: self.tenants_per_scan]:
            hits = []
            for klass in remaining:
                window = sorted(
                    (
                        c
                        for c in queued
                        if c.tenant == tenant and c.dispatch_class == klass
                    ),
                    key=lambda c: (-c.priority, c.id),
                )[: self.window]
                found = [c.id for c in window if self.claimable(c, cap, busy)]
                if found:
                    # The interactive side counts from the oldest claimable
                    # interactive command of its window.
                    age = found[0]
                    if klass == "interactive":
                        age = min(found)
                    hits.append((age, found[0]))
                elif not alone:
                    self.skips[(tenant, klass)] = now + 2.0
            if hits:
                best = min(hits)[1]
                break
        if best is None:
            return today
        older = [
            c.id
            for c in usable
            if (c.kind in CONTROL_KINDS or c.task in cap.active_ids) and c.id < best
        ]
        return min(older) if older else best


# --- scanning the queue and playing a random world through the dispatcher -------------

BIG = DispatchPolicy(limit=40, interactive_reserve=0)


def cap_of(active=(), occupancy=None, policy=BIG):
    return DispatchCapSnapshot(
        policy=policy, active_ids=frozenset(active), occupancy=occupancy or {}
    )


def scan(cap, *, attempt=None, busy=(), command_db_id=None):
    """The id one scan picks; ``attempt`` is one dispatch attempt, shared by rescans."""
    candidate = transport._find_command_candidate(
        command_db_id, busy, cap, attempt=attempt
    )
    return None if candidate is None else candidate.id


def attempt():
    return transport._FairAttempt()


def play(
    world, clock, seed, *, tenants, reserves, attempts=20, stats=None, **world_options
):
    """Run ``attempts`` scans of one random world and compare each with the reference.

    Fails with the seed and the whole world. After every scan the chosen command is
    settled and its task counted as holding a slot for its tenant. A pause or cancel
    that passed earlier commands of its task leaves them pending, as a rejected
    control does in production; the dispatcher reads only that the control is
    finished.
    """
    # The reference keeps no attempt count: a world runs too few attempts to pass
    # FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS.
    assert attempts <= transport.FAIR_SHARE_FALLBACK_AFTER_ATTEMPTS
    rng = random.Random(seed)
    spec = random_world(rng, tenants=tenants, size=rng.randint(6, 45), **world_options)
    world.reset()
    store_world(world, spec)
    transport._fair_state = transport._FairState()
    clock.now = 1000.0
    limit = rng.randint(3, 8)
    policy = DispatchPolicy(
        limit=limit, interactive_reserve=rng.choice(reserves(limit))
    )
    queued = [c for c in spec.commands if c.status in QUEUED]
    holds: list[tuple[int, str]] = []
    for c in rng.sample(queued, k=rng.randint(0, min(limit - 1, len(queued)))):
        # A task holds one slot, for the tenant of the first command drawn from it.
        if all(task != c.task for task, _ in holds):
            holds.append((c.task, c.tenant))
    reference = Reference(spec.commands, clock, spec.full_tenants)
    by_id = {c.id: c for c in spec.commands}
    for step in range(attempts):
        clock.now += rng.uniform(0, 1.5)
        cap = DispatchCapSnapshot(
            policy=policy,
            active_ids=frozenset(task for task, _ in holds),
            occupancy=dict(Counter(tenant for _, tenant in holds)),
        )
        busy = tuple(c.task for c in spec.commands if rng.random() < 0.03)
        shadow = None
        if stats is not None:
            shadow = copy.deepcopy(reference)
            shadow.clock, shadow.tenant_cap = clock, False
        expected = reference.pick(cap, busy)
        if shadow is not None and reference.last_capped is not None:
            stats["capped"] = stats.get("capped", 0) + 1
            if shadow.pick(cap, busy) != expected:
                stats["changed"] = stats.get("changed", 0) + 1
        one = attempt()
        chosen = scan(cap, attempt=one, busy=busy)
        assert chosen == expected, (
            f"seed {seed} step {step}: dispatcher {chosen}, reference {expected}\n"
            f"limit {limit} reserve {policy.interactive_reserve} holds {holds} "
            f"busy {busy}\nworld {spec.commands}"
        )
        if chosen is not None:
            done = by_id[chosen]
            if not cap.closed and done.kind not in CONTROL_KINDS:
                # What the dispatch loop does once the pick holds a slot token: the
                # tasks of ``holds`` are the ones an execution holds.
                transport._fair_count_dispatch(one, done.task not in cap.active_ids)
                reference.count(cap, done)
            done.status = "completed"
            with world.sessions() as db:
                db.get(TaskExecutionCommand, chosen).status = "completed"
                db.commit()
            if done.kind not in CONTROL_KINDS and done.task not in cap.active_ids:
                holds.append((done.task, done.tenant))
        if holds and rng.random() < 0.2:
            holds.pop(rng.randrange(len(holds)))
    if stats is not None:
        for name in ("held_back", "passed"):
            stats[name] = stats.get(name, 0) + getattr(reference, name)
    return spec
