#!/usr/bin/env python3
"""CPU tensors exercise the actual built DeepEP copy binding; no NPU allocation."""
import argparse
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extension', type=Path, required=True)
    args = parser.parse_args()
    import torch
    import torch_npu  # Load dependencies for the production extension; no set_device.
    spec = importlib.util.spec_from_file_location('deep_ep_cpp', args.extension.resolve(strict=True))
    extension = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    assert extension.ep8_relay_planner_abi == 2
    assert extension.ep8_fast_selector_abi == 1
    def copy(src, dst, spans):
        extension.ep8_copy_spans_out(src, dst, extension.Ep8CopySpans(spans))

    class CopyTests(unittest.TestCase):
        def test_native_helpers_in_pipeline_dependency_simulator(self):
            import test_a5_ccu_pipeline_cpu as pipeline_tests
            from heuristic_alltoallv import RelayPlanner
            class NativeRuntime(pipeline_tests.FakeRuntime):
                def bind_workspace_views(self, arena, sizes, chunks):
                    return extension.ep8_bind_workspace_views(arena, sizes, chunks)
                def copy_spans(self, dst, src, spans):
                    self.engine.append(self.active, ('copy', self.active),
                                       lambda: extension.ep8_copy_spans_out(src, dst, spans))
            def native_plan(c, rb, allowed, **options):
                return RelayPlanner(rb, allowed, native_module=extension, **options).plan(c)
            methods = ('test_async_executor_overlap_dependencies_slot_reuse_and_changed_payloads',
                       'test_persistent_context_with_changed_matrices_and_tensor_storage',
                       'test_dynamic_pool_stays_bounded_across_64_changed_matrices',
                       'test_capacity_growth_rebinds_old_prepared_after_completion')
            with patch.object(pipeline_tests, 'FakeRuntime', NativeRuntime), patch.object(pipeline_tests, 'make_plan', native_plan):
                for name in methods:
                    with self.subTest(method=name):
                        getattr(pipeline_tests.PipelineTests(name), name)()

        def test_production_fast_selector_changing_counts(self):
            import test_a5_ccu_pipeline_cpu as pipeline_tests
            from heuristic_alltoallv import FastRelayPlanner
            fast = FastRelayPlanner(8, pipeline_tests.ALL, native_module=extension,
                                    chunk_bytes=24, extra_us=0, min_saving_us=0)
            for i in range(24):
                c = [0] * 64
                c[(i % 7) * 8 + 7] = 60 + i
                decision = fast.select(c)
                self.assertTrue(decision.uses_relay)
                plan = fast.materialize(decision)
                self.assertEqual(plan.counts, tuple(c))
                pipeline_tests.simulate(plan, decision.chunk_bytes)
            self.assertFalse(fast.select([0] * 64).uses_relay)

        def test_workspace_views_slots_storage_and_bounds(self):
            arena = [torch.zeros((10, 4)) for _ in range(8)]
            sizes = (9, 7, 6, 8)
            chunks = [(8, 6, 4, 7), (0, 4, 5, 0), (3, 2, 1, 5)]
            pool, views = extension.ep8_bind_workspace_views(arena, sizes, chunks)
            self.assertEqual([v.shape[0] for v in pool], list(sizes) * 2)
            for j, row in enumerate(chunks):
                for i, n in enumerate(row):
                    self.assertEqual(views[j][i].shape, (n, 4))
                    self.assertEqual(views[j][i].untyped_storage().data_ptr(), arena[j % 2 * 4 + i].data_ptr())
            views[2][0].fill_(7)
            self.assertTrue((arena[0][:3] == 7).all())
            self.assertTrue((arena[4] == 0).all())
            # Existing views retain their original storage across pool replacement.
            new_arena = [torch.zeros((20, 4)) for _ in range(8)]
            _, new_views = extension.ep8_bind_workspace_views(new_arena, sizes, chunks)
            new_views[0][0].fill_(2)
            self.assertTrue((arena[0][:3] == 7).all())
            for bad_sizes, bad_chunks in (((11, 7, 6, 8), chunks), (sizes, [(10, 1, 1, 1)])):
                with self.assertRaises(RuntimeError): extension.ep8_bind_workspace_views(arena, bad_sizes, bad_chunks)
            with self.assertRaises(RuntimeError): extension.ep8_bind_workspace_views(arena[:4], sizes, chunks)

        def test_multiple_spans_gaps_and_repeated_payloads(self):
            for dtype in (torch.float16, torch.float32, torch.int8):
                for epoch in range(3):
                    src = (torch.arange(80).reshape(20, 4) + epoch).to(dtype)
                    dst = torch.full((18, 4), -7, dtype=dtype)
                    spans = [(0, 1, 3), (7, 5, 4), (19, 16, 1)]
                    expected = dst.clone()
                    for s, d, n in spans: expected[d:d+n].copy_(src[s:s+n])
                    copy(src, dst, spans)
                    self.assertTrue(torch.equal(dst, expected))

        def test_invalid_later_span_submits_no_earlier_copy(self):
            src = torch.ones((10, 4)); dst = torch.zeros((10, 4))
            for bad in ((10, 0, 1), (0, 10, 1), (0, 0, 11), ((1 << 64)-1, 0, 1)):
                with self.assertRaises(RuntimeError): copy(src, dst, [(0, 0, 1), bad])
                self.assertEqual(dst.count_nonzero().item(), 0)

        def test_empty_and_zero_length(self):
            src = torch.empty((0, 4)); dst = torch.empty((0, 4))
            copy(src, dst, [])
            copy(src, dst, [(0, 0, 0)])

        def test_alias_shape_dtype_and_strides(self):
            base = torch.arange(48, dtype=torch.float32).reshape(12, 4)
            before = base.clone()
            with self.assertRaises(RuntimeError): copy(base[:8], base[1:9], [(0, 0, 1)])
            self.assertTrue(torch.equal(base, before))
            # Disjoint views in the same allocation are valid.
            copy(base[:2], base[10:], [(0, 0, 2)])
            self.assertTrue(torch.equal(base[10:], before[:2]))
            with self.assertRaises(RuntimeError): copy(base, torch.zeros((12, 5)), [(0, 0, 1)])
            with self.assertRaises(RuntimeError): copy(base, torch.zeros((12, 4), dtype=torch.float16), [(0, 0, 1)])
            with self.assertRaises(RuntimeError): copy(base[:, ::2], torch.zeros((12, 2)), [(0, 0, 1)])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CopyTests))
    if not result.wasSuccessful(): sys.exit(1)
    print('EP8_NATIVE_COPY_SPANS_BUILT_EXTENSION_PASS')


if __name__ == '__main__': main()
