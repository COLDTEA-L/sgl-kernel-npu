#!/usr/bin/env python3
"""CPU-only timing of production native bindings vs Python; no route cache/NPU."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/python/deepep'))
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from test_a5_ccu_native_planner_cpu import load_native_planner, ALL
from test_a5_ccu_heuristic_npu import synthetic
from heuristic_alltoallv import RelayPlanner, FastRelayPlanner
from pipelined_alltoallv import PipelineContext, PreparedPipelinedAllToAllV, build_pipeline_schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/home/l00934901/profiling/a5-ep8-planner.json'))
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--fast-extra-us', type=float, default=500.)
    parser.add_argument('--extension', type=Path,
                        help='optional production DeepEP SO; also measures native workspace-view binding on CPU')
    args = parser.parse_args()
    if args.iters < 1: parser.error('--iters must be positive')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # This diagnostic must not initialize the device backend.
    os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
    import torch
    torch.set_num_threads(1)
    class Runtime:
        def current(self): return 0
        def event(self): return object()
        def synchronize(self, stream): pass
    def measure(fn):
        for i in range(10): fn(i)
        start = time.perf_counter()
        for i in range(args.iters): fn(i)
        return (time.perf_counter() - start) * 1e6 / args.iters
    results = []
    with tempfile.TemporaryDirectory(prefix='planner-build-', dir=args.output.parent) as directory:
        if args.extension:
            import torch_npu
            spec = importlib.util.spec_from_file_location('deep_ep_cpp', args.extension.resolve(strict=True))
            native = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(native)
        else:
            native = load_native_planner(directory)
        for name in ('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut',
                     'single_hot_64mib', 'single_hot_128mib'):
            c = synthetic('single_hot', 256 if name.endswith('64mib') else 512) if name.endswith('mib') else synthetic(name, 128)
            row = dict(case=name)
            for backend, module in (('python', None), ('native', native)):
                planner = RelayPlanner(8192, ALL, native_module=module,
                                       relay_capacity_bytes=[1024 << 20] * 8)
                plan = planner.plan(c)
                # Alternating actual counts proves the planner does not just
                # return a cache hit. Preserve total input rows for layout tests.
                changed = list(c)
                f = next((i for i, n in enumerate(c) if n), None)
                if f is not None:
                    changed[f] -= 1; changed[(f + 1) % 64] += 1
                matrices = (c, tuple(changed))
                plans = tuple(planner.plan(v) for v in matrices)
                lanes = [SimpleNamespace(buffer=SimpleNamespace(rank=0, group_size=8,
                                        moe_all_to_all_group_name='lane' + str(i)),
                                         handle=i + 1, stream=i) for i in range(3)]
                context = PipelineContext(lanes, _runtime=Runtime())
                context.runtime.bind_workspace_views = getattr(module, 'ep8_bind_workspace_views', None)
                inputs = tuple(torch.empty((sum(v[:8]), 4096), dtype=torch.float16) for v in matrices)
                def prepare(i):
                    return PreparedPipelinedAllToAllV(context, plans[i % 2], inputs[i % 2], chunk_bytes=8 << 20)
                prepare(0); prepare(1)
                row[backend] = dict(
                    planner_us=measure(lambda i: planner.plan(matrices[i % 2])),
                    layout_us=measure(lambda i: build_pipeline_schedule(plans[i % 2], chunk_bytes=8 << 20)),
                    prepare_cpu_rank0_us=measure(prepare),
                    actions=len(plan.actions), candidate_evaluations=plan.candidate_evaluations)
            fast = FastRelayPlanner(8192, ALL, native_module=native, extra_us=args.fast_extra_us,
                                    relay_capacity_bytes=[1024 << 20] * 8)
            # Use a separate persistent context; rejected calls must not prepare
            # any pipeline object or mutate its arena allocation counters.
            fast_context = PipelineContext(lanes, _runtime=Runtime())
            fast_context.runtime.bind_workspace_views = getattr(native, 'ep8_bind_workspace_views', None)
            def fast_prepare(i):
                decision = fast.select(matrices[i % 2])
                if not decision.uses_relay: return None
                return PreparedPipelinedAllToAllV(fast_context, fast.materialize(decision), inputs[i % 2],
                                                  chunk_bytes=decision.chunk_bytes)
            decision = fast.select(c)
            row['fast'] = dict(select_us=measure(lambda i: fast.select(matrices[i % 2])),
                              select_prepare_cpu_rank0_us=measure(fast_prepare),
                              selection=fast.report(decision), workspace=fast_context.workspace_stats())
            if args.extension and getattr(native, 'ep8_pipeline_host_abi', None) in (2, 3):
                opaque_planner = RelayPlanner(8192, ALL, native_module=native,
                                              relay_capacity_bytes=[1024 << 20] * 8)
                opaque = native.ep8_pipeline_create(opaque_planner._native, 0, 8 << 20, 128, 1 << 30, True)
                row['cpp_host'] = dict(plan_prepare_cpu_rank0_us=measure(
                    lambda i: opaque.prepare(matrices[i % 2], inputs[i % 2])), workspace=opaque.stats())
                outputs = tuple(torch.empty((sum(v[::8]), 4096), dtype=torch.float16) for v in matrices)
                phase_samples = []
                for i in range(args.iters + 10):
                    phases = opaque.run(matrices[i % 2], inputs[i % 2], outputs[i % 2])
                    opaque.take_trace()  # do not retain tensors/trace entries across iterations
                    if i >= 10: phase_samples.append(phases[0] + phases[1])
                row['cpp_host']['dynamic_plan_layout_cpu_rank0_us'] = sum(phase_samples) / len(phase_samples)
                row['cpp_host']['copy_report_rank0'] = opaque.prepare(c, inputs[0]).copy_report()
                row['cpp_host']['workspace'] = opaque.stats()
                opaque.close()
            results.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(dict(scope='CPU only; layout_us is INCLUDED in prepare_cpu_rank0_us',
                                          native_workspace_views=bool(args.extension),
                                          alternating_counts=True, fast_extra_us=args.fast_extra_us, iters=args.iters, cases=results), indent=2) + '\n')


if __name__ == '__main__': main()
