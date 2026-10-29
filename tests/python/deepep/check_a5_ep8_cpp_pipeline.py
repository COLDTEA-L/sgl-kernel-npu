#!/usr/bin/env python3
"""Run the actual C++ pipeline driver with CPU tensors and a deferred DAG runtime.

No NPU allocation. This checks dependencies and memory ownership, not hardware
stream/CCU behavior. The NPU worker must still validate on the target machine.
"""
import argparse
import importlib.util
from pathlib import Path
import random
import sys
import unittest
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu
    import test_a5_ccu_pipeline_cpu as ref
    from heuristic_alltoallv import RelayPlanner
    spec = importlib.util.spec_from_file_location('deep_ep_cpp', args.extension.resolve(strict=True))
    ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    assert ext.ep8_pipeline_host_abi == 1
    planner = RelayPlanner(8, ref.ALL, native_module=ext)

    def context(rank, chunk=24, limit=1 << 20):
        return ext.ep8_pipeline_create(planner._native, rank, chunk, 128, limit, True)

    def replay(ctx, rt, events):
        for op in ctx.take_trace():
            tag, lane = op[:2]; stream = (rt.rank, lane)
            if tag in (0, 1):
                event = events.setdefault(op[2], rt.event())
                if tag == 0: rt.record(event, stream)
                else: rt.wait(stream, event)
            else:
                with rt.stream(stream):
                    if tag == 2:
                        _, _, src, dst, spans = op
                        for si, di, n in spans: rt.copy(dst, di, src, si, n)
                    else:
                        _, _, src, dst, counts, storage = op
                        rt.transfer(SimpleNamespace(stream=stream), src, dst, tuple(counts), storage)

    def payloads(c, epoch):
        sends = [(torch.arange(sum(c[r*8:r*8+8])*2).reshape(-1, 2).float() + 1000*r + epoch)
                 for r in range(8)]
        expected = ref.collective([t.tolist() for t in sends], c)
        recv = [torch.full((len(v), 2), -77.) for v in expected]
        return sends, recv, [torch.tensor(v).reshape(-1, 2) for v in expected]

    class PipelineTests(unittest.TestCase):
        def test_changed_matrices_all_ranks_and_reused_storage(self):
            rng = random.Random(84531)
            contexts = [context(r) for r in range(8)]
            matrices = [[0]*64, [3 if f//8 == f%8 else 0 for f in range(64)]]
            hot = [0]*64; hot[4] = 35; hot[0] = 7; matrices.append(hot)
            matrices += [[rng.randrange(12) if rng.random() < .3 else 0 for _ in range(64)] for _ in range(64)]
            for step, c in enumerate(matrices):
                sends, recv, expected = payloads(c, step)
                engine = ref.Engine()
                for r in range(8):
                    runtime = ref.FakeRuntime(engine, r)
                    times = contexts[r].run(c, sends[r], recv[r])
                    self.assertTrue(all(v >= 0 for v in times))
                    replay(contexts[r], runtime, {})
                engine.run(step)
                for a, b in zip(recv, expected): self.assertTrue(torch.equal(a, b))
            allocations = [ctx.stats()['tensor_allocations'] for ctx in contexts]
            # Steady same-shape traffic must not grow pool/event resources.
            for _ in range(10):
                sends, recv, expected = payloads(matrices[-1], 13)
                engine = ref.Engine()
                for r, ctx in enumerate(contexts):
                    ctx.run(matrices[-1], sends[r], recv[r]); replay(ctx, ref.FakeRuntime(engine, r), {})
                engine.run(3)
            self.assertEqual(allocations, [ctx.stats()['tensor_allocations'] for ctx in contexts])
            for ctx in contexts: ctx.close()

        def test_queued_producer_consumer_overlap_and_slot_reuse(self):
            c = [0]*64; c[4] = 37; c[0] = 3; c[63] = 2
            contexts = [context(r, 8) for r in range(8)]
            sends, recv, _ = payloads(c, 0)
            engine = ref.Engine(); runtimes = [ref.FakeRuntime(engine, r) for r in range(8)]
            events = [{} for _ in range(8)]
            plans = [ctx.prepare(c, sends[r]) for r, ctx in enumerate(contexts)]
            self.assertGreater(plans[0].chunks, 2)
            snapshots = []
            for epoch in (1, 9, 17):
                values, _, expected = payloads(c, epoch)
                out = [torch.empty_like(v) for v in recv]
                for r in range(8):
                    engine.append((r,0), ('producer',r), lambda r=r, values=values: sends[r].copy_(values[r]))
                    contexts[r].execute(plans[r], sends[r], recv[r]); replay(contexts[r], runtimes[r], events[r])
                    engine.append((r,0), ('consumer',r), lambda r=r, out=out: out[r].copy_(recv[r]))
                snapshots.append((out, expected))
            first_second = next(n for n in engine.nodes if n.tag[:2] == ('collective',2))
            self.assertFalse(any(n.tag[:2] == ('collective',0) for n in engine.ancestors(first_second)))
            engine.run(7)
            for out, expected in snapshots:
                for a,b in zip(out, expected): self.assertTrue(torch.equal(a,b))
            for ctx in contexts: ctx.close()

        def test_growth_rebinds_old_prepared_and_validation_before_submit(self):
            small = [0]*64; small[4] = 8
            large = [0]*64; large[4] = 100
            contexts = [context(r, 1024) for r in range(8)]
            sends, recv, expected = payloads(small, 2)
            old = [ctx.prepare(small, sends[r]) for r,ctx in enumerate(contexts)]
            big, _, _ = payloads(large, 0)
            for r,ctx in enumerate(contexts): ctx.prepare(large, big[r])
            self.assertGreater(contexts[0].stats()['generation'], 1)
            engine = ref.Engine()
            for r,ctx in enumerate(contexts):
                ctx.execute(old[r], sends[r], recv[r]); replay(ctx, ref.FakeRuntime(engine,r), {})
            engine.run(1)
            for a,b in zip(recv, expected): self.assertTrue(torch.equal(a,b))
            with self.assertRaises(RuntimeError): contexts[0].execute(old[1], sends[0], recv[0])
            with self.assertRaises(RuntimeError): contexts[0].execute(old[0], sends[0], torch.empty((10,2)))
            self.assertFalse(contexts[0].take_trace())
            self.assertFalse(contexts[0].stats()['poisoned'])
            for ctx in contexts:
                ctx.close()
                self.assertEqual(ctx.stats()['reserved_bytes'], 0)
                self.assertTrue(ctx.stats()['closed'])
            with self.assertRaises(RuntimeError): contexts[0].prepare(small, sends[0])

        def test_global_capacity_and_failed_submission_poison_context(self):
            c = [0]*64; c[4] = 40
            sends, recv, _ = payloads(c, 0)
            # Even idle rank rejects globally oversized workspace before launch.
            tiny = context(7, 24, 1)
            with self.assertRaises(RuntimeError): tiny.prepare(c, sends[7])
            self.assertFalse(tiny.take_trace()); tiny.close()
            ctx = context(0)
            plan = ctx.prepare(c, sends[0]); ctx.trace_fail_after(2)
            with self.assertRaisesRegex(RuntimeError, 'injected'): ctx.execute(plan, sends[0], recv[0])
            self.assertTrue(ctx.stats()['poisoned'])
            ctx.take_trace()
            with self.assertRaisesRegex(RuntimeError, 'failed'): ctx.execute(plan, sends[0], recv[0])
            self.assertFalse(ctx.take_trace()); ctx.close()

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PipelineTests))
    if not result.wasSuccessful(): sys.exit(1)
    print('EP8_CPP_PIPELINE_DEFERRED_CPU_PASS')


if __name__ == '__main__': main()
