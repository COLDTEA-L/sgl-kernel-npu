#!/usr/bin/env python3
"""Layout oracle and asynchronous stream/event/collective DAG execution on CPU.

Unlike the serial threaded mock, operations are deferred and different streams
can progress out of submission order. This catches premature slot reuse and
missing consumer joins; it does not claim to validate a hardware CCU scheduler.
"""
import contextlib
import importlib.util
from pathlib import Path
import random
import sys
import unittest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import make_plan
from pipelined_alltoallv import (build_pipeline_schedule, PipelineContext, PipelineLane,
                                PreparedPipelinedAllToAllV)

ALL = [[k for k in range(8) if k not in (s, d)] if s != d else []
       for s in range(8) for d in range(8)]


def collective(send, counts):
    out = [[] for _ in range(8)]
    for s in range(8):
        offset = 0
        for d in range(8):
            n = counts[s * 8 + d]
            out[d].extend(send[s][offset:offset + n]); offset += n
        assert offset == len(send[s])
    return out


def simulate(plan, chunk_bytes):
    schedule = build_pipeline_schedule(plan, chunk_bytes=chunk_bytes)
    send = [[(s, i) for i in range(sum(plan.counts[s * 8:s * 8 + 8]))] for s in range(8)]
    expected = collective(send, plan.counts)
    actual = [[None] * len(v) for v in expected]
    direct = [[None] * sum(schedule.direct_counts[s * 8:s * 8 + 8]) for s in range(8)]
    for s, d, n, src, dst, ps, pr in schedule.direct_pieces:
        direct[s][ps:ps + n] = send[s][src:src + n]
    dr = collective(direct, schedule.direct_counts)
    for s, d, n, src, dst, ps, pr in schedule.direct_pieces:
        actual[d][dst:dst + n] = dr[d][pr:pr + n]
    for chunk in schedule.chunks:
        first = [[None] * chunk.sizes(r)[0] for r in range(8)]
        second = [[None] * chunk.sizes(r)[2] for r in range(8)]
        for p in chunk.pieces:
            first[p.src][p.first_send:p.first_send + p.rows] = send[p.src][p.source:p.source + p.rows]
        fr = collective(first, chunk.first_counts)
        for p in chunk.pieces:
            second[p.relay][p.second_send:p.second_send + p.rows] = fr[p.relay][p.first_recv:p.first_recv + p.rows]
        sr = collective(second, chunk.second_counts)
        for p in chunk.pieces:
            actual[p.dst][p.final:p.final + p.rows] = sr[p.dst][p.second_recv:p.second_recv + p.rows]
        assert all(x is not None for buf in first + second for x in buf)
    assert actual == expected


class Node:
    def __init__(self, tag, action=lambda: None):
        self.tag, self.action, self.deps, self.calls = tag, action, set(), {}


class Engine:
    def __init__(self):
        self.nodes, self.tails, self.collectives = [], {}, {}

    def append(self, stream, tag, action=lambda: None, extra=()):
        node = Node(tag, action)
        node.deps.update(extra)
        if stream in self.tails: node.deps.add(self.tails[stream])
        self.nodes.append(node); self.tails[stream] = node
        return node

    def ancestors(self, node):
        found, todo = set(), list(node.deps)
        while todo:
            dep = todo.pop()
            if dep not in found:
                found.add(dep); todo.extend(dep.deps)
        return found

    def run(self, seed):
        rng = random.Random(seed)
        pending, done, order = set(self.nodes), set(), []
        while pending:
            ready = [n for n in pending if n.deps <= done]
            if not ready: raise AssertionError('deadlock/cyclic stream dependencies')
            # Deliberately delay direct when another lane can progress. A
            # second hop completing before direct is a legal execution order.
            relay = [n for n in ready if n.tag[:2] != ('collective', 0)]
            node = rng.choice(relay or ready)
            node.action(); pending.remove(node); done.add(node); order.append(node.tag)
        return order


class FakeRuntime:
    def __init__(self, engine, rank):
        self.engine, self.rank, self.active = engine, rank, (rank, 0)
        self.sequences = [0, 0, 0]
        self.sync_calls = 0
    def current(self): return self.active
    def synchronize(self, stream):
        # Deferred DAG nodes retain old storage. This mock validates ordering;
        # real stream completion/reclamation must also be tested on the NPU.
        self.sync_calls += 1
    @contextlib.contextmanager
    def stream(self, stream):
        previous = self.active; self.active = stream
        try: yield
        finally: self.active = previous
    def event(self): return [None]
    def record(self, event, stream): event[0] = self.engine.append(stream, ('record', stream))
    def wait(self, stream, event):
        assert event[0] is not None
        self.engine.append(stream, ('wait', stream), extra=(event[0],))
    def retain(self, tensor, stream): pass
    def copy(self, dst, di, src, si, n):
        self.engine.append(self.active, ('copy', self.active),
                           lambda: dst.narrow(0, di, n).copy_(src.narrow(0, si, n)))
    def transfer(self, lane, send, recv, counts, storage_counts=None):
        import torch
        li = lane.stream[1]; seq = self.sequences[li]; self.sequences[li] += 1
        key = (li, seq)
        node = self.engine.collectives.get(key)
        if node is None:
            node = Node(('collective', li, seq))
            def action():
                assert len(node.calls) == 8
                assert all(c[2] == counts for c in node.calls.values())
                assert all(c[3] == storage_counts for c in node.calls.values())
                if storage_counts is None:
                    result = collective([node.calls[r][0].tolist() for r in range(8)], counts)
                    for r in range(8):
                        target = node.calls[r][1]
                        target.copy_(torch.tensor(result[r], dtype=target.dtype).reshape(target.shape))
                else:
                    for src in range(8):
                        for dst in range(8):
                            n = counts[src * 8 + dst]
                            si = sum(storage_counts[src * 8:src * 8 + dst])
                            di = sum(storage_counts[s * 8 + dst] for s in range(src))
                            node.calls[dst][1][di:di+n].copy_(node.calls[src][0][si:si+n])
            node.action = action
            self.engine.collectives[key] = node; self.engine.nodes.append(node)
        node.calls[self.rank] = (send, recv, counts, storage_counts)
        if self.active in self.engine.tails: node.deps.add(self.engine.tails[self.active])
        self.engine.tails[self.active] = node


class Buffer:
    group_size = 8
    def __init__(self, rank, lane):
        self.rank = rank; self.moe_all_to_all_group_name = 'group' + str(lane)


class PipelineTests(unittest.TestCase):
    def test_layout_random_zero_self_nondivisible_and_chunk_caps(self):
        rng = random.Random(20261028)
        matrices = [[0] * 64, [4 if s == d else 0 for s in range(8) for d in range(8)]]
        c = [0] * 64; c[4] = 65; c[0] = 3; matrices.append(c)
        matrices += [[rng.randrange(40) if rng.random() < .4 else 0 for _ in range(64)] for _ in range(100)]
        for c in matrices:
            p = make_plan(c, 14, ALL)
            for size in (14, 7 * 14, 2 << 20): simulate(p, size)
        p = make_plan(matrices[2], 14, ALL)
        sch = build_pipeline_schedule(p, chunk_bytes=7 * 14)
        self.assertEqual(len(sch.chunks), 5)
        self.assertLessEqual(max(p.rows for c in sch.chunks for p in c.pieces), 7)
        with self.assertRaises(ValueError): build_pipeline_schedule(p, chunk_bytes=14, max_chunks=2)
        with self.assertRaises(ValueError): build_pipeline_schedule(p, chunk_bytes=0)

    def test_context_rejects_shared_comm_and_stream(self):
        rt = FakeRuntime(Engine(), 0)
        lanes = [PipelineLane(Buffer(0, i), i + 1, (0, i)) for i in range(3)]
        PipelineContext(lanes, _runtime=rt)
        lanes[1].buffer.moe_all_to_all_group_name = 'group0'
        with self.assertRaises(ValueError): PipelineContext(lanes, _runtime=rt)
        lanes[1].buffer.moe_all_to_all_group_name = 'group1'
        lanes[1].stream = lanes[0].stream
        with self.assertRaises(ValueError): PipelineContext(lanes, _runtime=rt)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed')
    def test_async_executor_overlap_dependencies_slot_reuse_and_changed_payloads(self):
        import torch
        torch.set_num_threads(1)
        rng = random.Random(9)
        matrices = []
        hot = [0] * 64; hot[4] = 65; hot[0] = 3; hot[27] = 5
        matrices.append(hot)
        matrices += [[rng.randrange(35) if s != 7 and d != 6 else 0
                      for s in range(8) for d in range(8)] for _ in range(4)]
        matrices += [[0] * 64, [2] * 64]
        for trial, counts in enumerate(matrices):
            plan = make_plan(counts, 14, ALL)
            engine, results = Engine(), []
            for rank in range(8):
                rt = FakeRuntime(engine, rank)
                lanes = [PipelineLane(Buffer(rank, i), i + 1, (rank, i)) for i in range(3)]
                context = PipelineContext(lanes, _runtime=rt)
                n, m = sum(counts[rank * 8:rank * 8 + 8]), sum(counts[rank::8])
                istore = torch.full((n * 7 + 74,), -19, dtype=torch.float16)
                ostore = torch.full((m * 7 + 74,), -17, dtype=torch.float16)
                send, recv = istore[37:-37].reshape(n, 7), ostore[37:-37].reshape(m, 7)
                prepared = PreparedPipelinedAllToAllV(context, plan, send, chunk_bytes=7 * 14)
                self.assertEqual(sum(x.numel() * x.element_size() for x in prepared.workspace),
                                 prepared.schedule.scratch_rows(rank) * 14)
                for epoch in range(4):
                    source = (torch.arange(n * 7).reshape(n, 7) % 97 + rank + epoch).to(torch.float16)
                    rt.copy(send, 0, source, 0, n)
                    prepared.execute(send, recv)
                    snapshot = torch.empty_like(recv)
                    rt.copy(snapshot, 0, recv, 0, m)
                    expected = []
                    for s in range(8):
                        start = sum(counts[s * 8:s * 8 + rank])
                        for row in range(start, start + counts[s * 8 + rank]):
                            expected.append([(row * 7 + col) % 97 + s + epoch for col in range(7)])
                    results.append((snapshot, torch.tensor(expected).reshape(m, 7), istore, ostore))
                with rt.stream((rank, 1)):
                    with self.assertRaises(ValueError): prepared.execute(send, recv)
                if n == m and n:
                    with self.assertRaises(ValueError): prepared.execute(send, send)
                if plan.uses_relay:
                    with self.assertRaises(ValueError):
                        PreparedPipelinedAllToAllV(context, plan, send, max_scratch_bytes=0)
            if plan.uses_relay:
                direct = engine.collectives[(0, 0)]
                first0 = engine.collectives[(1, 0)]
                second0 = engine.collectives[(2, 0)]
                self.assertIn(first0, engine.ancestors(second0))
                self.assertNotIn(direct, engine.ancestors(second0))
                if len(prepared.chunks) > 1:
                    self.assertNotIn(second0, engine.ancestors(engine.collectives[(1, 1)]))
                if len(prepared.chunks) > 2:
                    self.assertIn(second0, engine.ancestors(engine.collectives[(1, 2)]))
            order = engine.run(trial)
            if trial == 0:
                self.assertLess(order.index(('collective', 2, 0)), order.index(('collective', 0, 0)))
            for actual, expected, istore, ostore in results:
                self.assertTrue(torch.equal(actual, expected), (trial, actual, expected))
                self.assertTrue(bool((istore[:37] == -19).all() and (istore[-37:] == -19).all()))
                self.assertTrue(bool((ostore[:37] == -17).all() and (ostore[-37:] == -17).all()))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed')
    def test_persistent_context_with_changed_matrices_and_tensor_storage(self):
        import torch
        engine, results, keepalive = Engine(), [], []
        hot = [0] * 64; hot[4] = 43; hot[0] = 5
        reverse = [0] * 64; reverse[32] = 59; reverse[9] = 3
        matrices = (hot, [0] * 64, reverse, [2] * 64, hot)
        for rank in range(8):
            rt = FakeRuntime(engine, rank)
            context = PipelineContext([PipelineLane(Buffer(rank, i), i + 1, (rank, i))
                                       for i in range(3)], _runtime=rt)
            for iteration, counts in enumerate(matrices):
                plan = make_plan(counts, 4, ALL)
                send = torch.empty((sum(counts[rank * 8:rank * 8 + 8]), 1), dtype=torch.float32)
                recv = torch.empty((sum(counts[rank::8]), 1), dtype=torch.float32)
                values = torch.arange(send.numel(), dtype=torch.float32).reshape(-1, 1) + rank * 100 + iteration
                rt.copy(send, 0, values, 0, len(send))
                prepared = PreparedPipelinedAllToAllV(context, plan, send, chunk_bytes=12)
                keepalive.append(prepared)
                prepared.execute(send, recv)
                snapshot = torch.empty_like(recv)
                rt.copy(snapshot, 0, recv, 0, len(recv))
                expected = []
                for s in range(8):
                    start = sum(counts[s * 8:s * 8 + rank])
                    expected.extend([[i + s * 100 + iteration] for i in range(start, start + counts[s * 8 + rank])])
                results.append((snapshot, torch.tensor(expected).reshape(recv.shape), send, values))
        engine.run(91)
        for snapshot, expected, send, values in results:
            self.assertTrue(torch.equal(snapshot, expected))
            self.assertTrue(torch.equal(send, values))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed')
    def test_dynamic_pool_stays_bounded_across_64_changed_matrices(self):
        import torch
        engine = Engine()
        contexts = [PipelineContext([PipelineLane(Buffer(r, i), i + 1, (r, i)) for i in range(3)],
                                    _runtime=FakeRuntime(engine, r)) for r in range(8)]
        baseline = None
        retained = []
        for iteration in range(64):
            counts = [0] * 64
            counts[4] = 65 - iteration % 3
            counts[0] = 3
            plan = make_plan(counts, 4, ALL)
            results = []
            for rank, context in enumerate(contexts):
                n, m = sum(counts[rank * 8:rank * 8 + 8]), sum(counts[rank::8])
                send = (torch.arange(n, dtype=torch.float32) + 100 * rank + iteration).reshape(n, 1)
                recv = torch.full((m, 1), -1.)
                prepared = PreparedPipelinedAllToAllV(context, plan, send, chunk_bytes=28)
                # A zero-row view still owns stable, nonempty backing.
                self.assertTrue(all(t.untyped_storage().nbytes() >= 4 for t in prepared.workspace))
                prepared.execute(send, recv)
                expected = []
                for src in range(8):
                    start = sum(counts[src * 8:src * 8 + rank])
                    expected.extend([[row + 100 * src + iteration]
                                     for row in range(start, start + counts[src * 8 + rank])])
                results.append((recv, torch.tensor(expected).reshape(m, 1)))
                # Keep an earlier prepared object alive through pool reuse.
                if iteration == 0: retained.append(prepared)
            engine.run(iteration)
            for actual, expected in results: self.assertTrue(torch.equal(actual, expected))
            stats = [c.workspace_stats() for c in contexts]
            if baseline is None: baseline = stats
            self.assertEqual(stats, baseline, 'workspace addresses/events/allocations grew without capacity growth')
            # Each iteration represents completed work before the next batch.
            engine.nodes.clear(); engine.tails.clear(); engine.collectives.clear()
        self.assertTrue(all(s['tensor_allocations'] == 8 for s in baseline))
        self.assertTrue(all(s['event_allocations'] == 11 for s in baseline))

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'Torch not installed')
    def test_capacity_growth_rebinds_old_prepared_after_completion(self):
        import torch
        engine = Engine()
        contexts = [PipelineContext([PipelineLane(Buffer(r, i), i + 1, (r, i)) for i in range(3)],
                                    _runtime=FakeRuntime(engine, r)) for r in range(8)]
        old = []
        for iteration, rows in enumerate((17, 129, 17)):
            counts = [0] * 64; counts[4] = rows
            plan = make_plan(counts, 4, ALL)
            results = []
            for rank, context in enumerate(contexts):
                n, m = sum(counts[rank * 8:rank * 8 + 8]), sum(counts[rank::8])
                send = (torch.arange(n, dtype=torch.float32) + iteration).reshape(n, 1)
                recv = torch.empty((m, 1))
                prepared = old[rank] if iteration == 2 else PreparedPipelinedAllToAllV(
                    context, plan, send, chunk_bytes=128)
                if iteration == 0: old.append(prepared)
                prepared.execute(send, recv)
                self.assertEqual(prepared._bound_generation, context.generation)
                results.append((recv, (torch.arange(m) + iteration).reshape(m, 1)))
            engine.run(iteration)
            for actual, expected in results: self.assertTrue(torch.equal(actual, expected))
            engine.nodes.clear(); engine.tails.clear(); engine.collectives.clear()
        self.assertTrue(all(c.runtime.sync_calls == 1 for c in contexts))
        self.assertTrue(all(c.allocations == 16 for c in contexts))


if __name__ == '__main__': unittest.main()
