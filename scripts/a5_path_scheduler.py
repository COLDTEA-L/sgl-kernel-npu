"""Single-owner, event-driven collective admission; no CANN dependency.

Resource names describe configured contention domains, not discovered hardware.
Reservations start BEFORE GO and end only after every participant's DONE.
"""
from dataclasses import dataclass, field
import time


@dataclass(frozen=True)
class JobSpec:
    name: str
    ranks: int
    resources: frozenset
    paths: int
    nbytes: int
    policy: str = 'mixed'

    def split(self, iteration, op):
        choices = {'direct': 0, 'mixed': self.paths}
        choices.update({f'relay{i}': i + 1 for i in range(self.paths - 1)})
        selected = (iteration + op) % (self.paths + 1) if self.policy == 'cycle' else choices[self.policy]
        result = [0] * self.paths
        if selected < self.paths:
            result[selected] = self.nbytes
        else:
            result[0] = self.nbytes // 2
            for i in range(1, self.paths):
                result[i] = self.nbytes // (2 * (self.paths - 1))
        if sum(result) != self.nbytes:
            raise ValueError('policy split does not cover payload')
        return result


@dataclass
class Task:
    key: tuple
    resources: frozenset
    first_arrival_ns: int
    arrived: set = field(default_factory=set)
    done: set = field(default_factory=set)
    ready_ns: int = 0
    eligible_ns: int = 0
    grant_ns: int = 0
    end_ns: int = 0
    ticket: int = 0
    blocked: bool = False
    state: str = 'ARRIVING'
    rank_release_to_done_ns: dict = field(default_factory=dict)


class Scheduler:
    def __init__(self, jobs, iterations, mode='serialize', cohort_barrier=False,
                 clock=time.monotonic_ns, emit=None):
        if mode not in ('serialize', 'uncontrolled') or iterations < 1:
            raise ValueError('invalid scheduler mode/iterations')
        self.jobs = {j.name: j for j in jobs}
        if not self.jobs or len(self.jobs) != len(jobs):
            raise ValueError('empty/duplicate jobs')
        for j in jobs:
            if j.ranks < 1 or not j.resources or not 2 <= j.paths <= 3:
                raise ValueError('invalid job specification')
            j.split(0, 0)
        self.iterations, self.mode, self.cohort_barrier = iterations, mode, cohort_barrier
        self.clock, self.emit = clock, emit or (lambda event: None)
        self.tasks, self.pending, self.active = {}, [], {}
        self.next_sequence = {(j.name, r): 0 for j in jobs for r in range(j.ranks)}
        self.next_ticket = 1

    def event(self, kind, task, **extra):
        self.emit(dict(event=kind, ns=self.clock(), job=task.key[0],
                       iteration=task.key[1], op=task.key[2], ticket=task.ticket, **extra))

    def _key(self, job, rank, iteration, op):
        if job not in self.jobs or not 0 <= rank < self.jobs[job].ranks:
            raise ValueError('unknown job/rank')
        if not 0 <= iteration < self.iterations or op not in (0, 1):
            raise ValueError('invalid iteration/op')
        return job, iteration, op

    def arrive(self, job, rank, iteration, op):
        key = self._key(job, rank, iteration, op)
        if self.next_sequence[job, rank] != 2 * iteration + op:
            raise ValueError('out-of-order ARRIVED')
        task = self.tasks.get(key)
        if task is None:
            task = self.tasks[key] = Task(key, self.jobs[job].resources, self.clock())
        if rank in task.arrived or task.state != 'ARRIVING':
            raise ValueError('duplicate/unexpected ARRIVED')
        task.arrived.add(rank)
        self.event('ARRIVED', task, rank=rank)
        if len(task.arrived) == self.jobs[job].ranks:
            task.ready_ns = self.clock()
            task.state = 'WAITING'
            self.pending.append(key)
            self.event('READY', task)
        return self.schedule()

    def _eligible(self, task):
        if not self.cohort_barrier:
            return True
        return all((j, task.key[1], task.key[2]) in self.tasks and
                   self.tasks[j, task.key[1], task.key[2]].state != 'ARRIVING'
                   for j in self.jobs)

    def schedule(self):
        grants, older_resources = [], set()
        for key in list(self.pending):
            task = self.tasks[key]
            if not self._eligible(task):
                continue
            if not task.eligible_ns:
                task.eligible_ns = self.clock()
            owners = [other.ticket for other in self.active.values()
                      if task.resources & other.resources]
            blocked_by_fifo = bool(task.resources & older_resources)
            if self.mode == 'serialize' and (owners or blocked_by_fifo):
                older_resources.update(task.resources)
                if not task.blocked:
                    task.blocked = True
                    self.event('BLOCKED', task, owner_tickets=owners,
                               older_waiter=blocked_by_fifo)
                continue
            task.ticket, self.next_ticket = self.next_ticket, self.next_ticket + 1
            task.grant_ns, task.state = self.clock(), 'RUNNING'
            # Atomic in this event loop: reserve ALL resources before returning GO.
            self.active[key] = task
            self.pending.remove(key)
            self.event('GRANT', task, resources=sorted(task.resources),
                       overlap_tickets=owners, split=self.jobs[key[0]].split(key[1], key[2]))
            grants.append(task)
        return grants

    def done(self, job, rank, iteration, op, ticket, release_to_done_ns):
        key = self._key(job, rank, iteration, op)
        task = self.tasks.get(key)
        if task is None or task.state != 'RUNNING' or ticket != task.ticket or rank in task.done:
            raise ValueError('stale/duplicate/ungranted DONE')
        if release_to_done_ns < 0:
            raise ValueError('negative timing')
        task.done.add(rank)
        task.rank_release_to_done_ns[rank] = release_to_done_ns
        self.next_sequence[job, rank] += 1
        self.event('DONE', task, rank=rank, release_to_done_ns=release_to_done_ns)
        if len(task.done) == self.jobs[job].ranks:
            task.end_ns, task.state = self.clock(), 'DONE'
            del self.active[key]
            self.event('RELEASE', task, resources=sorted(task.resources))
        return self.schedule()

    def rank_finished(self, job, rank):
        return self.next_sequence[job, rank] == self.iterations * 2

    def complete(self):
        return (not self.active and not self.pending and
                all(n == self.iterations * 2 for n in self.next_sequence.values()))

    def metrics(self):
        rows = []
        for key, task in sorted(self.tasks.items()):
            if task.state != 'DONE':
                continue
            rows.append(dict(job=key[0], iteration=key[1], op=key[2], ticket=task.ticket,
                             arrival_skew_ns=task.ready_ns - task.first_arrival_ns,
                             cohort_wait_ns=task.eligible_ns - task.ready_ns,
                             resource_wait_ns=task.grant_ns - task.eligible_ns,
                             reservation_ns=task.end_ns - task.grant_ns,
                             ready_to_done_ns=task.end_ns - task.ready_ns,
                             blocked=task.blocked,
                             rank_release_to_done_ns=task.rank_release_to_done_ns))
        return rows
