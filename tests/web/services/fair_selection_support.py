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
QUEUED = ("pending", "processing")


@dataclass
class Cmd:
    """One command of a random world; ``id`` and ``task`` are filled in when stored."""

    tenant: str
    dispatch_class: str
    kind: str
    status: str
    pinned: bool = False
    unavailable: bool = False
    ticketed: bool = False
    priority: int = 0
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
    belong to a tenant whose admission bucket is full, some are settled. With
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
    if heavy and tenants > 1 and rng.random() < 0.4:
        name, klass = rng.choice(names), rng.choice(CLASSES)
        leading = [
            Cmd(name, klass, "start", "pending", pinned=True, priority=10)
            for _ in range(16)
        ]
        leading.append(Cmd(name, klass, "start", "pending", priority=-3))
    for _ in range(size):
        name = rng.choice(names)
        kind = rng.choice(kinds)
        commands.append(
            Cmd(
                name,
                "interactive" if kind == "resume_input" else rng.choice(CLASSES),
                kind,
                rng.choices(["pending", "processing", "completed"], [8, 1, 2])[0],
                pinned=rng.random() < 0.1,
                unavailable=rng.random() < 0.07,
                ticketed=name in full and kind not in CONTROL_KINDS,
                priority=rng.choice((-3, 0, 0, 0, 5, 10)),
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
    return RandomWorld(leading + commands, full)


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
        for number, cmd in enumerate(spec.commands):
            task = Task(
                user_id=world.user,
                title="t",
                status=TaskStatus.PENDING,
                runner_id="other" if cmd.pinned else None,
                lease_expires_at=later if cmd.pinned else None,
            )
            db.add(task)
            db.flush()
            row = TaskExecutionCommand(
                task_id=task.id,
                command_id=f"c{number}",
                kind=cmd.kind,
                payload={},
                status=cmd.status,
                dispatch_class=cmd.dispatch_class,
                fairness_key=cmd.tenant,
                priority=cmd.priority,
                retry_available_at=later if cmd.unavailable else None,
            )
            db.add(row)
            db.flush()
            cmd.id, cmd.task = row.id, task.id
            if cmd.ticketed:
                db.add(
                    TaskAdmissionTicket(
                        command_id=row.id,
                        task_id=task.id,
                        bucket_key=f"bucket:{cmd.tenant}",
                    )
                )
        db.commit()


class Reference:
    """The selection rules read straight off the specification, over plain objects.

    It shares no code with the dispatcher. ``clock`` is the time source the test
    moves; the reference keeps its own counter and skip table.
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
        self.attempts = 0
        self.skips: dict[tuple[str, str], float] = {}
        # Overridable so that a test can pin a rule to a different value.
        self.tenants_per_scan = 4
        self.window = 16
        self.fallback_every = 8
        self.tenant_cap = True
        self.last_capped: str | None = None

    def claimable(self, cmd: Cmd, cap, busy) -> bool:
        if cmd.status not in QUEUED or cmd.pinned or cmd.unavailable:
            return False
        if cmd.task in busy or cmd.ticketed and cmd.tenant in self.full:
            return False
        if cap.batch_closed:
            needs_no_slot = cmd.kind in CONTROL_KINDS or cmd.task in cap.active_ids
            if cap.closed:
                return needs_no_slot
            return needs_no_slot or cmd.dispatch_class == "interactive"
        return True

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

    def pick(self, cap, busy=()) -> int | None:
        has_room = not cap.closed
        fallback = False
        if has_room:
            self.attempts += 1
            fallback = self.attempts % self.fallback_every == 0
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
        if len(tenants) <= 1:
            return today
        now = self.clock()
        self.skips = {k: until for k, until in self.skips.items() if until > now}
        ranked = []
        for tenant, classes in tenants.items():
            remaining = sorted(k for k in classes if (tenant, k) not in self.skips)
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
                else:
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
    settled and its task counted as holding a slot for its tenant.
    """
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
    holds = [
        (c.task, c.tenant)
        for c in rng.sample(queued, k=rng.randint(0, min(limit - 1, len(queued))))
    ]
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
        chosen = scan(cap, attempt=attempt(), busy=busy)
        assert chosen == expected, (
            f"seed {seed} step {step}: dispatcher {chosen}, reference {expected}\n"
            f"limit {limit} reserve {policy.interactive_reserve} holds {holds} "
            f"busy {busy}\nworld {spec.commands}"
        )
        if chosen is not None:
            done = by_id[chosen]
            done.status = "completed"
            with world.sessions() as db:
                db.get(TaskExecutionCommand, chosen).status = "completed"
                db.commit()
            if done.kind not in CONTROL_KINDS and done.task not in cap.active_ids:
                holds.append((done.task, done.tenant))
        if holds and rng.random() < 0.2:
            holds.pop(rng.randrange(len(holds)))
    return spec
