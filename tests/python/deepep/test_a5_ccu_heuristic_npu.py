#!/usr/bin/env python3
"""EP8 native HCCL / custom direct / relay correctness and payload timing."""
import argparse
import ctypes
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import (make_plan, RelayPlanner, FastRelayPlanner, build_schedule, normalize_counts, normalize_relays,
                                PreparedAllToAllV, gather_routing_counts)
from pipelined_alltoallv import PipelineContext, PreparedPipelinedAllToAllV, build_pipeline_schedule


def synthetic(name, scale):
    c = [0] * 64
    if name == 'self':
        for r in range(8): c[r * 8 + r] = r + 1
    elif name == 'single_hot':
        c[4] = 32; c[0] = 3; c[27] = 1
    elif name == 'competing':
        c[1] = 40; c[2] = 30; c[25] = 24; c[46] = 17; c[18] = 3
    elif name == 'asymmetric':
        c = [(s * 17 + d * 7) % 11 if s != 7 and d != 6 else 0
             for s in range(8) for d in range(8)]
    elif name == 'uniform':
        c = [4] * 64
    elif name == 'cut':
        c = [7 if s < 4 and d >= 4 else 0 for s in range(8) for d in range(8)]
    elif name != 'zero':
        raise ValueError('unknown synthetic case')
    return tuple(v * scale for v in c)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--topology-json', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--matrix-json', type=Path, help='{"counts":8x8 or flat64}; packed rows, no histogram')
    p.add_argument('--case', choices=('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut'), default='single_hot')
    p.add_argument('--suite', choices=('benchmark', 'correctness'), default='benchmark')
    p.add_argument('--mode', choices=('all', 'native', 'both', 'direct', 'multipath'), default='both')
    p.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32', 'int8', 'int32'), default='float16')
    p.add_argument('--width', type=int, default=4096)
    p.add_argument('--row-scale', type=int, default=128)
    p.add_argument('--min-chunk-bytes', type=int, default=1)
    p.add_argument('--min-gain-bytes', type=int, default=0)
    p.add_argument('--max-actions', type=int, default=56)
    p.add_argument('--planner', choices=('native', 'python'), default='native',
                   help='native C++ planner/layout/copy submission or the Python reference')
    p.add_argument('--routing-policy', choices=('fast', 'greedy'), default='fast',
                   help='bounded single-hotspot admission with native fallback, or legacy greedy')
    p.add_argument('--fast-link-gbps', type=float, default=50.)
    p.add_argument('--fast-extra-us', type=float, default=500.,
                   help='estimated incremental dynamic pipeline/host cost; NOT a measured guarantee')
    p.add_argument('--fast-min-saving-us', type=float, default=20.)
    p.add_argument('--max-scratch-mib', type=int, default=1024)
    p.add_argument('--relay-capacity-mib', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--dynamic-iters', type=int, default=10)
    p.add_argument('--executor', choices=('pipeline', 'serial'), default='pipeline')
    p.add_argument('--cpp-copy-mode', choices=('auto', 'staged'), default='auto',
                   help='C++ auto aliases contiguous intervals; staged keeps all pack/forward/unpack copies')
    p.add_argument('--pipeline-host', choices=('python', 'cpp'), default='python',
                   help='cpp keeps greedy plan/layout/views and stream/event submission inside one extension call')
    p.add_argument('--chunk-bytes', type=int, default=8 << 20,
                   help='greedy: per-flow chunk cap; fast: base chunk size, may grow to limit relay to two batches')
    p.add_argument('--thread-bridge', choices=('event', 'hcomm'), default='event',
                   help='EP8 die-1 caller/route stream bridge; hcomm is the legacy diagnostic control')
    p.add_argument('--max-chunks', type=int, default=128)
    p.add_argument('--executor-sync', choices=('stream', 'device'), default='stream',
                   help='stream ordering or old four device fences for A/B diagnosis')
    p.add_argument('--multipath-policy', choices=('force', 'auto'), default='force',
                   help='force really measures relaying; auto may explicitly reject serial forwarding')
    args = p.parse_args()
    if (int(os.environ.get('WORLD_SIZE', 0)) != 8 or args.width < 1 or args.row_scale < 1 or
            args.warmup < 0 or min(args.iters, args.rounds, args.dynamic_iters, args.max_scratch_mib) < 1
            or args.relay_capacity_mib < 0):
        p.error('requires EP8, positive sizes/iterations and nonnegative warmup/capacity')
    if args.pipeline_host == 'cpp' and (args.executor != 'pipeline' or args.planner != 'native' or args.routing_policy != 'greedy'):
        p.error('cpp host comparison requires --executor pipeline --planner native --routing-policy greedy')
    if args.routing_policy == 'fast' and (args.executor != 'pipeline' or args.planner != 'native' or args.max_chunks < 2):
        p.error('fast requires native planner, pipeline executor and max-chunks >= 2; use --routing-policy greedy for legacy controls')
    if args.routing_policy == 'fast' and (args.max_actions != 56 or args.min_chunk_bytes != 1 or args.min_gain_bytes != 0):
        p.error('max-actions/min-chunk-bytes/min-gain-bytes belong to --routing-policy greedy; fast uses its cost options')
    if args.executor == 'pipeline' and (args.multipath_policy != 'force' or args.executor_sync != 'stream'):
        p.error('pipeline requires --multipath-policy force --executor-sync stream; serial auto model is inapplicable')
    if args.chunk_bytes < 1 or args.max_chunks < 1:
        p.error('chunk-bytes/max-chunks must be positive')
    topology = json.loads(args.topology_json.read_text())
    if topology['version'] != 1 or topology['executor'] != 'participant_hbm_two_stage_ccu':
        raise ValueError('wrong topology schema/executor')
    if topology['devices'] != [int(v) for v in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]:
        raise ValueError('topology device order differs from visible devices')
    allowed = normalize_relays(topology['allowed_relays'])
    if args.matrix_json:
        data = json.loads(args.matrix_json.read_text())
        cases = [('external', normalize_counts(data['counts']))]
        if 'devices' in data and data['devices'] != topology['devices']:
            raise ValueError('external matrix devices differ from topology rank mapping')
    else:
        names = ('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut') if args.suite == 'correctness' else (args.case,)
        cases = [(name, synthetic(name, args.row_scale)) for name in names]
    os.environ['A5_CCU_EP8_THREAD_BRIDGE'] = args.thread_bridge
    from a5_ccu_test_runtime import prepare_runtime, import_deep_ep, ccu_pg_kwargs, log_ccu_communicator
    extension, route_path = prepare_runtime(__file__)
    import torch
    import torch.distributed as dist
    import torch_npu
    deep_ep, _ = import_deep_ep(extension)
    native_planner_module = sys.modules.get('deep_ep_cpp') if args.planner == 'native' else None
    if args.planner == 'native' and (getattr(native_planner_module, 'ep8_relay_planner_abi', None) != 2
                                   or not hasattr(native_planner_module, 'ep8_copy_spans_out')
                                   or not hasattr(native_planner_module, 'ep8_bind_workspace_views')):
        raise RuntimeError('rebuild/install DeepEP wheel: planner ABI 2, span copies and workspace views required')
    if args.routing_policy == 'fast' and getattr(native_planner_module, 'ep8_fast_selector_abi', None) != 1:
        raise RuntimeError('rebuild/install DeepEP: fast selector ABI 1 required')
    if args.pipeline_host == 'cpp' and getattr(native_planner_module, 'ep8_pipeline_host_abi', None) != 2:
        raise RuntimeError('rebuild/install DeepEP: ep8_pipeline_host_abi=2 required')
    route = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    public_api = route.A5CcuPublicApiAbiVersion
    public_api.restype = ctypes.c_int
    if public_api() != 1: raise RuntimeError('public CCU SDK backend ABI 1 is required')
    marker = route.A5CcuBoundDirectAbiVersion
    marker.restype = ctypes.c_int
    if marker() != 1: raise RuntimeError('rebuild/install route: bound-direct ABI missing')
    thread_abi = route.A5CcuEp8ThreadBridgeAbiVersion
    thread_abi.restype = ctypes.c_int
    if thread_abi() != 1: raise RuntimeError('rebuild/install route: EP8 thread bridge ABI 1 required')
    if args.executor == 'pipeline':
        try:
            layout_abi = route.A5CcuAllToAllVLayoutAbiVersion
            layout_abi.restype = ctypes.c_int
            if layout_abi() != 1: raise AttributeError('wrong layout ABI')
        except AttributeError as exc:
            raise RuntimeError('rebuild/install route: AllToAllV layout ABI 1 is required') from exc
        bridge = ctypes.CDLL(str(extension), mode=ctypes.RTLD_GLOBAL)
        try:
            workspace_abi = bridge.A5DeepEpPipelineWorkspaceAbiVersion
            workspace_abi.restype = ctypes.c_int
            if workspace_abi() != 2: raise AttributeError('wrong workspace ABI')
        except AttributeError as exc:
            raise RuntimeError('rebuild/install DeepEP extension: pipeline workspace ABI 2 is required') from exc
    try:
        dfx_abi = route.A5CcuEp8DfxAbiVersion
        dfx_abi.restype = ctypes.c_int
        if dfx_abi() != 2: raise AttributeError('public SDK reporting ABI 2 required')
    except AttributeError as exc:
        raise RuntimeError('rebuild/install route: public SDK reporting ABI 2 is required') from exc
    rank = int(os.environ['RANK'])
    def phase(name, **fields):
        print('EP8_PHASE ' + json.dumps(dict(rank=rank, pid=os.getpid(), phase=name,
              time=datetime.now().astimezone().isoformat(timespec='microseconds'), **fields)), flush=True)
    phase('process_group_begin')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180), **ccu_pg_kwargs(torch_npu))
    # Reject mismatched test configuration before preparing any CCU plan.
    identity = json.dumps(dict(cases=cases, allowed=allowed, options=vars(args)), default=str, sort_keys=True)
    digest = torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.int32, device='npu')
    gathered = [torch.empty_like(digest) for _ in range(8)]
    dist.all_gather(gathered, digest)
    if any(not torch.equal(v.cpu(), digest.cpu()) for v in gathered):
        raise ValueError('ranks disagree on counts/topology/options')
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    log_ccu_communicator(buffer.moe_all_to_all_group_name)
    phase('direct_plan_begin')
    handle = buffer.prepare_ccu_urma_peer_plan_alltoallv('ep8-heuristic', str(args.manifest.resolve()))
    phase('direct_plan_ready')
    pipeline = None
    if args.executor == 'pipeline':
        phase('relay_context_begin')
        pipeline = PipelineContext.create(buffer, handle, args.manifest.resolve(),
                                          group_ranks=list(range(8)), pg_kwargs=ccu_pg_kwargs(torch_npu))
        phase('relay_context_ready')
        print('PIPELINE_LANES ' + json.dumps(dict(rank=rank, groups=[
            lane.buffer.moe_all_to_all_group_name for lane in pipeline.lanes],
            lanes=['direct', 'first_hop', 'second_hop'])), flush=True)
    dtype = getattr(torch, args.dtype)
    row_bytes = torch.empty((), dtype=dtype).element_size() * args.width
    planner_options = dict(min_chunk_bytes=args.min_chunk_bytes, min_gain_bytes=args.min_gain_bytes,
                           max_actions=args.max_actions, relay_capacity_bytes=[args.relay_capacity_mib << 20] * 8)
    planner = RelayPlanner(row_bytes, allowed, native_module=native_planner_module, **planner_options)
    fast = (FastRelayPlanner(row_bytes, allowed, native_module=native_planner_module,
                chunk_bytes=args.chunk_bytes, link_gbps=args.fast_link_gbps,
                extra_us=args.fast_extra_us, min_saving_us=args.fast_min_saving_us,
                relay_capacity_bytes=planner_options['relay_capacity_bytes'])
            if args.routing_policy == 'fast' else None)
    def create_cpp_context(chunk_bytes):
        ctx = native_planner_module.ep8_pipeline_create(planner._native, rank, chunk_bytes,
                                                        args.max_chunks, args.max_scratch_mib << 20,
                                                        bypass_copies=args.cpp_copy_mode == 'auto')
        for li, lane in enumerate(pipeline.lanes):
            with torch.npu.stream(lane.stream):
                ctx.attach_lane(li, lane.buffer.runtime, lane.handle)
        return ctx
    cpp = create_cpp_context(args.chunk_bytes) if args.pipeline_host == 'cpp' else None
    selected = (('native', 'direct', 'multipath') if args.mode == 'all' else
                ('direct', 'multipath') if args.mode == 'both' else (args.mode,))
    def measurement_order(round_id):
        # all: six rounds cover all six permutations; both retains AB/BA.
        if len(selected) == 3:
            offset = round_id % 3
            order = selected[offset:] + selected[:offset]
        else:
            order = selected
        return order if round_id % 2 == 0 else tuple(reversed(order))
    def native_alltoallv(send, recv, input_splits, output_splits):
        # Same payload API / Work.wait contract as vLLM Ascend async_all_to_all.
        # Outputs are preallocated for EVERY mode; wrapper allocation is excluded.
        work = dist.all_to_all_single(recv, send, input_split_sizes=input_splits,
                                      output_split_sizes=output_splits, group=dist.group.WORLD,
                                      async_op=True)
        work.wait()

    reports = []
    def memory_sample(phase):
        # Small host-side diagnostic snapshots, outside timed regions. No profiler.
        rss = int(Path('/proc/self/statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
        sample = dict(phase=phase, rss_bytes=rss, npu_allocated_bytes=torch.npu.memory_allocated(),
                      npu_reserved_bytes=torch.npu.memory_reserved())
        if pipeline: sample['workspace'] = pipeline.workspace_stats()
        if cpp is not None: sample['cpp_workspace'] = cpp.stats()
        print('PIPELINE_MEMORY ' + json.dumps(dict(rank=rank, **sample)), flush=True)
        return sample
    for case_name, counts in cases:
        phase('correctness_begin', case=case_name)
        begin = time.perf_counter()
        selection = fast.select(counts) if fast else None
        plan = fast.materialize(selection) if fast else planner.plan(counts)
        case_chunk_bytes = selection.chunk_bytes if selection is not None and selection.uses_relay else args.chunk_bytes
        planner_us = (time.perf_counter() - begin) * 1e6
        if args.planner == 'native' and fast is None:
            reference = make_plan(counts, row_bytes, allowed, **planner_options)
            for key in ('counts', 'relay_rank', 'relay_rows', 'link_rows', 'actions', 'stop_reason'):
                if getattr(plan, key) != getattr(reference, key):
                    raise AssertionError(f'native/reference planner disagreement: {case_name} {key}')
            print(f'EP8_PLANNER_EQUIVALENCE_PASS rank={rank} case={case_name}', flush=True)
        schedule = (build_pipeline_schedule(plan, chunk_bytes=case_chunk_bytes, max_chunks=args.max_chunks)
                    if pipeline else build_schedule(plan))
        if plan.uses_relay and max(schedule.report()['scratch_bytes_per_rank']) > (args.max_scratch_mib << 20):
            raise ValueError('workspace exceeds per-rank bound before launching data collectives')
        sends = list(counts[rank * 8:rank * 8 + 8]); recvs = list(counts[rank::8])
        def payload(s):
            n = sum(counts[s * 8:s * 8 + 8])
            return ((torch.arange(n * args.width, dtype=torch.int64).reshape(n, args.width) * 7 + s * 11) % 97).to(dtype)
        source = payload(rank)
        expected = torch.cat([payload(s)[sum(counts[s * 8:s * 8 + rank]):sum(counts[s * 8:s * 8 + rank]) + counts[s * 8 + rank]] for s in range(8)], dim=0)
        guard = 37
        input_store = torch.full((source.numel() + 2 * guard,), -19, dtype=dtype, device='npu')
        send = input_store[guard:-guard].reshape(sum(sends), args.width)
        stores = {name: torch.full((expected.numel() + 2 * guard,), -17, dtype=dtype, device='npu')
                  for name in ('direct', 'multipath', 'native')}
        outputs = {name: storage[guard:-guard].reshape(sum(recvs), args.width) for name, storage in stores.items()}
        prepare_options = dict(max_scratch_bytes=args.max_scratch_mib << 20)
        if pipeline:
            prepare_options.update(chunk_bytes=case_chunk_bytes, max_chunks=args.max_chunks)
            def prepare(current_plan, selected_chunk_bytes=None):
                opts = dict(prepare_options)
                if selected_chunk_bytes is not None: opts['chunk_bytes'] = selected_chunk_bytes
                return PreparedPipelinedAllToAllV(pipeline, current_plan, send, **opts)
        else:
            prepare_options.update(sync_mode=args.executor_sync, policy=args.multipath_policy)
            def prepare(current_plan):
                return PreparedAllToAllV(buffer, handle, rank, current_plan, send, **prepare_options)
        prepared = prepare(plan)
        cpp_prepared = cpp.prepare(counts, send) if cpp is not None else None
        if cpp_prepared is not None:
            expected_plan = (plan.counts, plan.relay_rank, plan.relay_rows, plan.actions)
            if cpp_prepared.describe() != expected_plan:
                raise AssertionError('C++ opaque plan differs from reference')
        forced = (prepared if pipeline or args.multipath_policy == 'force' else
                  PreparedAllToAllV(buffer, handle, rank, plan, send,
                                   **dict(prepare_options, policy='force')))
        if fast is not None:
            # Correctness must exercise actual relay even if admission rejects it.
            forced = prepare(planner.plan(counts), args.chunk_bytes)
        execution = dict(prepared.decision)
        if cpp_prepared is not None:
            cpp_copies = cpp_prepared.copy_report()
            print('EP8_CPP_COPIES ' + json.dumps(dict(rank=rank, case=case_name,
                  mode=args.cpp_copy_mode, **cpp_copies)), flush=True)
        execution['selected_backend'] = (('pipeline' if pipeline else 'serial') if prepared.uses_relay else
                                         'native' if fast is not None else 'custom_direct')
        if fast is not None:
            execution.update(policy='fast', reason=selection.reason, admission=fast.report(selection))
        # Fast rejection uses native/native A/A. Legacy no-relay uses custom
        # direct/direct A/A. Dynamic includes selection/planning respectively.
        def selected_multi():
            if cpp is not None:
                return cpp.execute(cpp_prepared, send, outputs['multipath'])
            if prepared.uses_relay:
                return prepared.execute(send, outputs['multipath'])
            if fast is not None:
                return native_alltoallv(send, outputs['multipath'], sends, recvs)
            return buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs['multipath'], handle, list(counts))
        ops = dict(direct=lambda: buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs['direct'], handle, list(counts)),
                   multipath=selected_multi,
                   native=lambda: native_alltoallv(send, outputs['native'], sends, recvs))
        def set_payload(epoch):
            send.copy_(source + epoch)
            for output in outputs.values(): output.fill_(-17)
            torch.npu.synchronize()
        def check(name, epoch):
            if not torch.equal(outputs[name].cpu(), expected + epoch):
                raise AssertionError(f'{case_name} {name} mismatch rank={rank} epoch={epoch}')
            if not torch.equal(send.cpu(), source + epoch): raise AssertionError('input modified')
            for storage, sentinel in ((input_store, -19), (stores[name], -17)):
                cpu = storage.cpu()
                if not ((cpu[:guard] == sentinel).all() and (cpu[-guard:] == sentinel).all()):
                    raise AssertionError('guard modified')
        # All modes prove against native and an independently built host oracle.
        for epoch in range(2):
            set_payload(epoch)
            for name in ('direct', 'multipath', 'native'):
                ops[name](); torch.npu.synchronize(); check(name, epoch)
            # Auto fallback must not hide a broken actual relay executor.
            forced.execute(send, outputs['multipath'])
            torch.npu.synchronize(); check('multipath', epoch)
            # Producer -> relay -> Torch consumer -> next invocation, WITHOUT
            # an intermediate device sync. Detects stream ordering regressions
            # that the ordinary per-call correctness sync would otherwise hide.
            snapshots = []
            for queued_epoch in (epoch + 7, epoch + 13, epoch + 19):
                send.copy_(source + queued_epoch)
                if cpp is not None: cpp.execute(cpp_prepared, send, outputs['multipath'])
                else: forced.execute(send, outputs['multipath'])
                snapshots.append((queued_epoch, outputs['multipath'].clone()))
            torch.npu.synchronize()
            for queued_epoch, snapshot in snapshots:
                if not torch.equal(snapshot.cpu(), expected + queued_epoch):
                    raise AssertionError(f'queued stream ordering failure rank={rank}')
            if cpp is not None:
                set_payload(epoch)
                cpp.run(counts, send, outputs['multipath'])
                torch.npu.synchronize(); check('multipath', epoch)
            dist.barrier()
        # Small correctness cases must still exercise >2 chunks and both slot
        # reuses. Default 8 MiB chunks would make them a one-round pipeline.
        checked_chunks = [len(prepared.schedule.chunks)] if pipeline else []
        if pipeline and forced.uses_relay and len(forced.schedule.chunks) < min(5, args.max_chunks):
            target = min(5, args.max_chunks)
            probe_plan = forced.plan
            probe_rows = max(1, (max(probe_plan.relay_rows) + target - 1) // target)
            probe = PreparedPipelinedAllToAllV(pipeline, probe_plan, send,
                         **dict(prepare_options, chunk_bytes=probe_rows * row_bytes))
            set_payload(23)
            probe.execute(send, outputs['multipath'])
            torch.npu.synchronize(); check('multipath', 23)
            checked_chunks.append(len(probe.schedule.chunks))
            if cpp is not None:
                cpp_probe = create_cpp_context(probe_rows * row_bytes)
                cpp_probe.run(counts, send, outputs['multipath'])
                torch.npu.synchronize(); check('multipath', 23)
                cpp_probe.close()
            del probe
        phase('prepared_begin', case=case_name)
        timings = {name: [] for name in selected}
        for round_id in range(args.rounds):
            order = measurement_order(round_id)
            epoch = 2 + round_id % 5
            for name in order:
                set_payload(epoch)
                for _ in range(args.warmup): ops[name]()
                torch.npu.synchronize()
                outputs[name].fill_(-17)
                dist.barrier(); torch.npu.synchronize()
                begin = time.perf_counter()
                for _ in range(args.iters): ops[name]()
                torch.npu.synchronize()
                timings[name].append((time.perf_counter() - begin) * 1e6 / args.iters)
                check(name, epoch)
        # Same device-resident packed row counters feed both dynamic paths.
        local_counts = torch.tensor(sends, dtype=torch.int64, device='npu')
        phase('prepared_complete', case=case_name)
        memory_samples = [memory_sample(case_name + ':before_dynamic')]
        dynamic_rounds = {name: [] for name in selected}
        components = {name: [] for name in selected}
        dynamic_backends = dict(native=0, pipeline=0, custom_direct=0)
        def dynamic_once(name):
            start = time.perf_counter()
            current = gather_routing_counts(local_counts)
            gathered_at = time.perf_counter()
            if name == 'native':
                current_sends = list(current[rank * 8:(rank + 1) * 8])
                current_recvs = list(current[rank::8])
                ready_at = time.perf_counter()
                planned_at = gathered_at
                native_alltoallv(send, outputs[name], current_sends, current_recvs)
            elif name == 'direct':
                current_counts = list(current)
                ready_at = time.perf_counter()
                planned_at = gathered_at
                buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs[name], handle, current_counts)
            elif cpp is not None:
                phase_us = cpp.run(current, send, outputs[name])
                # C++ returns host plan/layout phases; residual includes binding,
                # submission and the device completion wait below.
                planned_at = gathered_at + phase_us[0] * 1e-6
                ready_at = planned_at + phase_us[1] * 1e-6
                dynamic_backends['pipeline' if plan.uses_relay else 'custom_direct'] += 1
            elif fast is not None:
                admission = fast.select(current)
                if not admission.uses_relay:
                    # Same native call and split construction as the baseline.
                    # No RoutePlan, layout, workspace, events or copy descriptors.
                    planned_at = time.perf_counter()
                    current_sends = list(current[rank * 8:(rank + 1) * 8])
                    current_recvs = list(current[rank::8])
                    ready_at = time.perf_counter()
                    native_alltoallv(send, outputs[name], current_sends, current_recvs)
                    dynamic_backends['native'] += 1
                else:
                    current_plan = fast.materialize(admission)
                    planned_at = time.perf_counter()
                    current_prepared = prepare(current_plan, admission.chunk_bytes)
                    ready_at = time.perf_counter()
                    current_prepared.execute(send, outputs[name])
                    dynamic_backends['pipeline'] += 1
            else:
                current_plan = planner.plan(current)
                planned_at = time.perf_counter()
                current_prepared = prepare(current_plan)
                ready_at = time.perf_counter()
                current_prepared.execute(send, outputs[name])
                dynamic_backends['pipeline' if current_plan.uses_relay else 'custom_direct'] += 1
            # Completion is part of BOTH paths; temporary workspace remains live.
            torch.npu.synchronize()
            finished = time.perf_counter()
            return ((gathered_at - start) * 1e6, (ready_at - gathered_at) * 1e6,
                    (finished - ready_at) * 1e6, (planned_at - gathered_at) * 1e6,
                    (ready_at - planned_at) * 1e6)
        for round_id in range(args.rounds):
            order = measurement_order(round_id)
            for name in order:
                set_payload(0)
                # Warm the SAME counts collective + planner/allocator + executor
                # path, rather than only warming prepared payload communication.
                phase('dynamic_warmup_begin', case=case_name, mode=name, round=round_id)
                for _ in range(args.warmup): dynamic_once(name)
                dist.barrier(); torch.npu.synchronize()
                phase('dynamic_measure_begin', case=case_name, mode=name, round=round_id)
                totals = [0.] * 5
                begin = time.perf_counter()
                for _ in range(args.dynamic_iters):
                    sample = dynamic_once(name)
                    for i in range(5): totals[i] += sample[i]
                dynamic_rounds[name].append((time.perf_counter() - begin) * 1e6 / args.dynamic_iters)
                components[name].append(dict(zip(('counts_gather_sync_us', 'plan_workspace_us',
                                                  'execute_completion_us', 'planner_us', 'layout_workspace_us'),
                                                 (v / args.dynamic_iters for v in totals))))
                check(name, 0)
                phase('dynamic_round_complete', case=case_name, mode=name, round=round_id)
                memory_samples.append(memory_sample(f'{case_name}:dynamic_round_{round_id}:{name}'))
        dynamic = {name: sum(v) / len(v) for name, v in dynamic_rounds.items()}
        report = dict(case=case_name, correctness='PASS', plan=plan.report(), schedule=schedule.report(),
                      planner_us=planner_us, prepared_round_us=timings, dynamic_mean_us=dynamic,
                      dynamic_round_us=dynamic_rounds, dynamic_components_round_us=components,
                      execution=execution, dynamic_backend_calls=dynamic_backends, executor_sync=args.executor_sync,
                      pipeline_correctness_chunk_counts=checked_chunks, memory_samples=memory_samples,
                      fingerprint=plan.fingerprint,
                      cpp_copies=(cpp_copies if cpp_prepared is not None else None))
        reports.append(report)
        print('EP8_HEURISTIC_CASE ' + json.dumps(dict(rank=rank, case=case_name, actions=len(plan.actions),
                                                    prepared_round_us=timings, dynamic_mean_us=dynamic)), flush=True)
    torch.npu.synchronize()
    phase('teardown_begin')
    destroy = route.HcclCcuUrmaPeerPlanDestroy
    destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
    def destroy_checked(value):
        if destroy(value) != 0: raise RuntimeError('plan destroy failed')
    if cpp is not None:
        cpp.close()
    if pipeline:
        pipeline.close(destroy_checked)
    destroy_checked(handle)
    dist.barrier(); dist.destroy_process_group()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = dict(rank=rank, ranks=8, correctness='PASS', dtype=args.dtype, width=args.width,
                  benchmark=dict(protocol=11, pipeline_host=args.pipeline_host,
                                 cpp_copy_mode=(args.cpp_copy_mode if cpp is not None else 'unused'), planner=args.planner, routing_policy=args.routing_policy,
                                 fast_link_gbps=args.fast_link_gbps, fast_extra_us=args.fast_extra_us,
                                 fast_min_saving_us=args.fast_min_saving_us, warmup=args.warmup, iters=args.iters, rounds=args.rounds,
                                 copy_submission=(('cpp_contiguous_alias' if args.cpp_copy_mode == 'auto' else 'cpp_staged')
                                                  if cpp is not None else
                                                  'native_spans' if pipeline and args.planner == 'native' else 'python'),
                                 dynamic_iters=args.dynamic_iters, executor_sync=args.executor_sync,
                                 multipath_policy=args.multipath_policy, executor=args.executor,
                                 chunk_bytes=args.chunk_bytes, max_chunks=args.max_chunks, thread_bridge=args.thread_bridge,
                                 native_api='all_to_all_single_async_work_wait_preallocated',
                                 hccl_expansion_mode=os.environ.get('HCCL_OP_EXPANSION_MODE'),
                                 hccl_algo=os.environ.get('HCCL_ALGO'),
                                 hccl_buffsize=os.environ.get('HCCL_BUFFSIZE'),
                                 torch_version=str(torch.__version__), torch_npu_version=str(torch_npu.__version__)),
                  mode=args.mode, backend=('participant_hbm_pipelined_ccu' if pipeline else 'participant_hbm_two_stage_ccu'), cases=reports,
                  timing_scope='host_wall_submit_and_device_completion',
                  dynamic_scope=('device_counts_gather_cpu_sync_replan_pooled_workspace_execute_completion' if pipeline else
                                 'device_counts_gather_cpu_sync_plan_workspace_execute_completion'))
    temp = args.run_dir / f'rank{rank}.json.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.run_dir / f'rank{rank}.json')


if __name__ == '__main__':
    main()
