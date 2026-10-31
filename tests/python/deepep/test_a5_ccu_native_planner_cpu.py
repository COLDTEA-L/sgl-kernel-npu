"""Build the production planner binding without Torch/HCCL; compare to Python."""
import importlib.util
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import replace

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import RelayPlanner, FastRelayPlanner, make_plan, MAX_I64, _coalesce_spans
from pipelined_alltoallv import build_pipeline_schedule
from test_a5_ccu_pipeline_cpu import ALL, simulate


def load_native_planner(directory):
    """Same binding header as deep_ep_cpp; no NPU runtime or device needed."""
    directory = Path(directory)
    compiler = shutil.which('c++') or shutil.which('g++')
    if not compiler:
        raise RuntimeError('a C++17 compiler is required for planner regression')
    try:
        import pybind11
        include = pybind11.get_include()
    except ImportError:
        spec = importlib.util.find_spec('torch')
        if spec is None:
            raise RuntimeError('install pybind11 or Torch headers for planner regression')
        include = str(Path(spec.origin).parent / 'include')
    source = directory / 'planner.cpp'
    source.write_text('#include "ep8_relay_planner_bindings.hpp"\n'
                      'PYBIND11_MODULE(_ep8_planner_cpu, m) { deep_ep::ep8::BindPlanner(m); }\n')
    target = directory / ('_ep8_planner_cpu' + sysconfig.get_config_var('EXT_SUFFIX'))
    command = [compiler, '-O3', '-std=c++17', '-shared', '-fPIC', str(source),
               '-I' + str(ROOT / 'csrc/deepep'), '-I' + include,
               '-I' + sysconfig.get_path('include'), '-o', str(target)]
    if sys.platform == 'darwin': command += ['-undefined', 'dynamic_lookup']
    subprocess.run(command, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location('_ep8_planner_cpu', target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NativePlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix='ep8-native-planner-')
        cls.native = load_native_planner(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_dense_observed_volume_split_and_diagnostics(self):
        from a5_ccu_dense_cases import dense_counts
        for divisor, expected in ((2,31),(4,15),(8,7),(16,3)):
            planner=RelayPlanner(5120,ALL,native_module=self.native,dense_split_divisor=divisor)
            observed=planner.plan(dense_counts('observed_dense'))
            self.assertEqual(sum(observed.counts),196608)
            self.assertEqual(observed.actions[0][:3],(0,7,2))
            self.assertEqual(sum(observed.relay_rows),expected)
            self.assertEqual(max(observed.link_rows),3165)
            diag=observed.volume_diagnostics()
            self.assertEqual((diag['base_rows'],diag['excess_rows']),(3072,62))
            self.assertEqual((diag['edge_peak_rows_before'],diag['edge_peak_rows_after'],diag['edge_peak_rows_delta']),(3165,3165,0))
            self.assertEqual(diag['edge_peak_bytes_after'],3165*5120)
            self.assertLessEqual(observed.candidate_evaluations,336)
            # Helper is NOT a lighter card: it also sends 3165 rows to rank 1.
            self.assertEqual(observed.counts[2*8+1],3165)
            for case in ('dense_near','dense_hot'):
                c=dense_counts(case);p=planner.plan(c)
                self.assertEqual(len(p.actions),1)
                self.assertEqual(p.actions[0][:2],(0,1))
                self.assertEqual(sum(p.relay_rows),(c[1]-sum(c[:8])//8)//divisor)
                self.assertLess(max(p.link_rows),max(c))
        for bad in (1,1025):
            with self.assertRaises(ValueError): RelayPlanner(5120,ALL,native_module=self.native,dense_split_divisor=bad)
        with self.assertRaises(RuntimeError): RelayPlanner(5120,ALL,dense_split_divisor=8)

    def test_dense_rejects_new_bottleneck_and_honors_direct(self):
        from a5_ccu_dense_cases import dense_counts
        c=dense_counts('observed_dense')
        # Only flows to the hottest receiver are eligible: every second hop
        # already carries the global maximum, so any positive split is unsafe.
        allowed=[ks if f%8==1 else () for f,ks in enumerate(ALL)]
        for opts in (dict(),dict(max_actions=0)):
            p=RelayPlanner(5120,allowed,native_module=self.native,dense_split_divisor=2,**opts).plan(c)
            self.assertFalse(p.uses_relay)
            self.assertEqual(p.volume_diagnostics()['excess_rows'],0)
        p=RelayPlanner(5120,ALL,native_module=self.native,dense_split_divisor=2,min_gain_bytes=1).plan(c)
        self.assertFalse(p.uses_relay)  # strict improvement remains an optional threshold
        for c in ([3000]*64,[0]*64,[r+1 if f//8==f%8 else 0 for f in range(64) for r in [f//8]]):
            self.assertFalse(RelayPlanner(5120,ALL,native_module=self.native,dense_split_divisor=8).plan(c).uses_relay)

    def test_dense_randomized_bounds_and_capacity(self):
        rng=random.Random(412)
        for trial in range(150):
            c=[rng.randrange(1,80) for _ in range(64)]
            divisor=rng.choice((2,4,8,16));cap=[rng.randrange(10)*5120 for _ in range(8)]
            allowed=[tuple(k for k in ks if rng.randrange(2)) for ks in ALL]
            p=RelayPlanner(5120,allowed,native_module=self.native,dense_split_divisor=divisor,
                           relay_capacity_bytes=cap).plan(c)
            original=[c[f] if f//8!=f%8 else 0 for f in range(64)]
            peak=max(original)
            feasible=[]
            # Independent brute-force oracle constructs all changed edges.
            for f in range(64):
                excess=max(0,c[f]-sum(c[f//8*8:f//8*8+8])//8)
                for k in allowed[f]:
                    x=min(excess//divisor,cap[k]//5120)
                    if not x: continue
                    loads=list(original);loads[f]-=x;loads[f//8*8+k]+=x;loads[k*8+f%8]+=x
                    if max(loads)<=peak: feasible.append(((-excess,max(loads),f,k),x,loads))
            self.assertEqual(bool(p.actions),bool(feasible))
            self.assertLessEqual(len(p.actions),1)
            if p.actions:
                key,x,loads=min(feasible)
                s,d,k,moved,gain=p.actions[0]
                self.assertEqual((s*8+d,k,moved),(key[2],key[3],x))
                self.assertEqual(list(p.link_rows),loads)
                self.assertEqual(gain,peak-max(loads))
                self.assertTrue(all(not p.relay_rows[k*8+j] and not p.relay_rows[j*8+k] for j in range(8)))
                self.assertEqual(sum(p.link_rows),sum(original)+x)
                self.assertLessEqual(max(p.link_rows),peak)

    def test_fast_admission_defaults_and_large_hotspot(self):
        from test_a5_ccu_heuristic_npu import synthetic
        planner = FastRelayPlanner(8192, ALL, native_module=self.native)
        for name in ('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut'):
            result = planner.select(synthetic(name, 128))
            self.assertFalse(result.uses_relay, name)
            self.assertEqual(result.candidate_evaluations, 0, name)
        # Decisions use counts, never a synthetic-case label. 64 MiB original
        # hotspot admits at these configurable costs; it has exactly one relay.
        c = synthetic('single_hot', 256)
        decision = planner.select(c)
        self.assertTrue(decision.uses_relay)
        self.assertGreater(decision.predicted_saving_us, 20)
        self.assertLessEqual(decision.candidate_evaluations, 24)
        plan = planner.materialize(decision)
        self.assertEqual(len(plan.actions), 1)
        schedule = build_pipeline_schedule(plan, chunk_bytes=decision.chunk_bytes)
        self.assertLessEqual(len(schedule.chunks), 2)
        self.assertGreater(decision.chunk_bytes, 8 << 20)
        self.assertEqual(plan.counts, c)

    def test_fast_randomized_conservation_topology_capacity_and_payload(self):
        rng = random.Random(8971)
        for trial in range(300):
            c = [rng.randrange(4) for _ in range(64)]
            f = rng.choice([i for i in range(64) if i // 8 != i % 8])
            c[f] += rng.randrange(10, 61)
            allowed = [sorted(rng.sample(v, rng.randrange(len(v) + 1))) for v in ALL]
            caps = [rng.randrange(1, 41) * 8 for _ in range(8)]
            planner = FastRelayPlanner(8, allowed, native_module=self.native,
                        chunk_bytes=24, extra_us=0, min_saving_us=0, relay_capacity_bytes=caps)
            decision = planner.select(c)
            plan = planner.materialize(decision)
            self.assertLessEqual(len(plan.actions), 1)
            self.assertLessEqual(decision.candidate_evaluations, 24)
            original = [n if i // 8 != i % 8 else 0 for i, n in enumerate(c)]
            self.assertEqual(sum(plan.link_rows), sum(original) + sum(plan.relay_rows))
            if decision.uses_relay:
                self.assertIn(decision.relay, allowed[decision.flow])
                self.assertLessEqual(decision.rows * 8, caps[decision.relay])
                self.assertLess(max(plan.link_rows), max(original))
                self.assertLessEqual(len(build_pipeline_schedule(plan, chunk_bytes=decision.chunk_bytes).chunks), 2)
            if trial < 80:
                simulate(plan, decision.chunk_bytes or 24)
            # Same static selector processes a different real matrix every call.
            changed = list(c); changed[f] = 0
            second = planner.select(changed)
            self.assertEqual(planner.materialize(second).counts, tuple(changed))
            self.assertEqual(planner.materialize(decision).counts, tuple(c))

    def test_fast_boundaries_and_no_implicit_tensor_conversion(self):
        planner = FastRelayPlanner(1, ALL, native_module=self.native,
                                   extra_us=0, min_saving_us=0, chunk_bytes=1)
        c = [0] * 64; c[7] = MAX_I64
        d = planner.select(c)
        p = planner.materialize(d)
        self.assertTrue(d.uses_relay)
        self.assertLessEqual(d.chunk_bytes, MAX_I64)
        self.assertEqual(p.counts, tuple(c))
        for value in (-1, True, 1.5, MAX_I64 + 1):
            invalid = [0] * 64; invalid[1] = value
            with self.assertRaises((ValueError, OverflowError)): planner.select(invalid)
        for invalid in ([0] * 63, [0] * 8, [[0] * 7] * 8, object()):
            with self.assertRaises(ValueError): planner.select(invalid)
        self.assertEqual(planner.select([[0] * 8 for _ in range(8)]).reason, 'no_remote_traffic')
        c = [0] * 64; c[1] = 90
        none = FastRelayPlanner(1, [()] * 64, native_module=self.native, extra_us=0, min_saving_us=0)
        self.assertFalse(none.select(c).uses_relay)
        with self.assertRaises(ValueError): none.materialize(planner.select(c))
        for opts in (dict(link_gbps=0), dict(extra_us=-1), dict(min_saving_us=float('nan'))):
            with self.assertRaises(ValueError): FastRelayPlanner(1, ALL, native_module=self.native, **opts)

    def compare(self, c, rb, allowed, options, chunk_bytes):
        reference = make_plan(c, rb, allowed, **options)
        planner = RelayPlanner(rb, allowed, native_module=self.native, **options)
        result = planner.plan(c)
        for key in ('counts', 'relay_rank', 'relay_rows', 'link_rows', 'actions', 'stop_reason'):
            self.assertEqual(getattr(result, key), getattr(reference, key), key)
        # Different scan order may evaluate extra candidates, but cannot alter a decision.
        a = build_pipeline_schedule(reference, chunk_bytes=chunk_bytes)
        b = build_pipeline_schedule(result, chunk_bytes=chunk_bytes)
        for key in ('chunk_rows', 'direct_counts', 'direct_pieces', 'chunks', 'rank_pool_sizes'):
            self.assertEqual(getattr(a, key), getattr(b, key), key)
        self.assertEqual(a.report(), b.report())
        return result

    def test_randomized_plans_and_all_layout_fields(self):
        rng = random.Random(42081)
        for trial in range(1000):
            c = [rng.randrange(41) if rng.random() < .7 else 0 for _ in range(64)]
            allowed = [sorted(rng.sample(row, rng.randrange(len(row) + 1))) for row in ALL]
            rb = rng.choice((1, 14, 4096, 8192))
            options = dict(max_actions=rng.choice((0, 1, 4, 31, 56)),
                           min_chunk_bytes=rng.choice((1, rb, 3 * rb)),
                           min_gain_bytes=rng.choice((0, rb, 2 * rb)),
                           relay_capacity_bytes=[rng.randrange(80) * rb for _ in range(8)])
            p = self.compare(c, rb, allowed, options, rng.choice((1, 3, 9, 25)) * rb)
            if trial < 30: simulate(p, 3 * rb)

    def test_persistent_planner_changing_counts_and_mutable_inputs(self):
        allowed = [list(row) for row in ALL]
        planner = RelayPlanner(8192, allowed, native_module=self.native)
        for row in allowed: row.clear()  # Constructor takes a snapshot of topology.
        for iteration in range(200):
            c = [0] * 64
            c[(iteration % 7) * 8 + 7] = iteration * 9
            result = planner.plan(c)
            expected = make_plan(c, 8192, ALL)
            self.assertEqual(result.actions, expected.actions)
            self.assertEqual(result.link_rows, expected.link_rows)
            c[:] = [0] * 64
            self.assertEqual(result.counts, expected.counts)
            simulate(result, 7 * 8192)

    def test_limits_zero_self_uniform_and_overflow(self):
        for c in ([0] * 64, [7] * 64, [f + 1 if f // 8 == f % 8 else 0 for f in range(64)]):
            p = self.compare(c, 8192, ALL, {}, 8 << 20)
            self.assertFalse(p.uses_relay)
        c = [0] * 64; c[4] = MAX_I64
        p = self.compare(c, 1, ALL, {}, MAX_I64)
        self.assertEqual(sum(p.relay_rows), MAX_I64 // 2)
        planner = RelayPlanner(8192, ALL, native_module=self.native)
        for bad in (c, [-1] * 64, [True] * 64, [1.5] * 64, [0] * 63):
            with self.assertRaises((ValueError, TypeError)): planner.plan(bad)
        c = [0] * 64; c[4] = 100
        p = planner.plan(c)
        with self.assertRaises(ValueError): build_pipeline_schedule(p, chunk_bytes=8192, max_chunks=1)
        with self.assertRaises(ValueError): build_pipeline_schedule(p, chunk_bytes=0)
        bad = replace(p, relay_rank=tuple(0 if n else -1 for n in p.relay_rows))
        with self.assertRaises(ValueError): build_pipeline_schedule(bad)

    def test_static_configuration_and_missing_abi_rejected(self):
        for options in (dict(row_bytes=0), dict(max_actions=57), dict(min_chunk_bytes=0),
                        dict(relay_capacity_bytes=[0] * 7)):
            args = dict(row_bytes=8192, allowed_relays=ALL, native_module=self.native)
            args.update(options)
            with self.assertRaises(ValueError): RelayPlanner(**args)
        invalid = [list(row) for row in ALL]; invalid[4] = [1, 1]
        with self.assertRaises(ValueError): RelayPlanner(8192, invalid, native_module=self.native)
        with self.assertRaises(RuntimeError): RelayPlanner(8192, ALL, native_module=object())
        with self.assertRaises(ValueError): self.native.Ep8RelayPlanner(invalid, 8192, 1, 0, 56, [])

    def test_native_copy_descriptors_match_reference_all_ranks(self):
        rng = random.Random(91)
        planner = RelayPlanner(14, ALL, native_module=self.native)
        for _ in range(100):
            p = planner.plan([rng.randrange(31) if rng.random() < .8 else 0 for _ in range(64)])
            for rank in range(8):
                schedule = build_pipeline_schedule(p, chunk_bytes=7 * 14, rank=rank)
                for chunk in schedule.chunks:
                    expected = (
                        _coalesce_spans([(v.source, v.first_send, v.rows) for v in chunk.pieces if v.src == rank]),
                        _coalesce_spans([(v.first_recv, v.second_send, v.rows) for v in chunk.pieces if v.relay == rank]),
                        _coalesce_spans([(v.second_recv, v.final, v.rows) for v in chunk.pieces if v.dst == rank]))
                    self.assertEqual(tuple([tuple(s) for s in c] for c in chunk.copy_spans), expected)
                    for copies in chunk.copy_spans:
                        self.assertFalse(hasattr(copies, 'append'))
        with self.assertRaises(ValueError): build_pipeline_schedule(p, rank=8)
        with self.assertRaises(ValueError): build_pipeline_schedule(p, rank=-1)

    def test_native_layout_in_async_executor_and_growing_reused_workspace(self):
        import test_a5_ccu_pipeline_cpu as pipeline_tests
        def native_plan(c, rb, allowed, **options):
            return RelayPlanner(rb, allowed, native_module=self.native, **options).plan(c)
        methods = ('test_async_executor_overlap_dependencies_slot_reuse_and_changed_payloads',
                   'test_persistent_context_with_changed_matrices_and_tensor_storage',
                   'test_dynamic_pool_stays_bounded_across_64_changed_matrices',
                   'test_capacity_growth_rebinds_old_prepared_after_completion')
        with patch.object(pipeline_tests, 'make_plan', native_plan):
            for name in methods:
                with self.subTest(method=name):
                    getattr(pipeline_tests.PipelineTests(name), name)()


if __name__ == '__main__':
    unittest.main()
