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
from heuristic_alltoallv import RelayPlanner, make_plan, MAX_I64, _coalesce_spans
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
