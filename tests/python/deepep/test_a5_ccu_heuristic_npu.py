#!/usr/bin/env python3
"""EP8 direct vs participant-relay CCU correctness and eager timing, no profiler."""
import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import (make_plan, build_schedule, normalize_counts, normalize_relays,
                                PreparedAllToAllV, gather_routing_counts)


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
    p.add_argument('--mode', choices=('both', 'direct', 'multipath'), default='both')
    p.add_argument('--dtype', choices=('float16', 'bfloat16', 'float32', 'int8', 'int32'), default='float16')
    p.add_argument('--width', type=int, default=4096)
    p.add_argument('--row-scale', type=int, default=128)
    p.add_argument('--min-chunk-bytes', type=int, default=1)
    p.add_argument('--min-gain-bytes', type=int, default=0)
    p.add_argument('--max-actions', type=int, default=56)
    p.add_argument('--max-scratch-mib', type=int, default=1024)
    p.add_argument('--relay-capacity-mib', type=int, default=1024)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--dynamic-iters', type=int, default=10)
    p.add_argument('--executor-sync', choices=('stream', 'device'), default='stream',
                   help='stream ordering or old four device fences for A/B diagnosis')
    p.add_argument('--multipath-policy', choices=('force', 'auto'), default='force',
                   help='force really measures relaying; auto may explicitly reject serial forwarding')
    args = p.parse_args()
    if (int(os.environ.get('WORLD_SIZE', 0)) != 8 or args.width < 1 or args.row_scale < 1 or
            args.warmup < 0 or min(args.iters, args.rounds, args.dynamic_iters, args.max_scratch_mib) < 1
            or args.relay_capacity_mib < 0):
        p.error('requires EP8, positive sizes/iterations and nonnegative warmup/capacity')
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
    from a5_ccu_test_runtime import prepare_runtime, import_deep_ep, ccu_pg_kwargs, log_ccu_communicator
    extension, route_path = prepare_runtime(__file__)
    import torch
    import torch.distributed as dist
    import torch_npu
    deep_ep, _ = import_deep_ep(extension)
    route = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    marker = route.A5CcuBoundDirectAbiVersion
    marker.restype = ctypes.c_int
    if marker() != 1: raise RuntimeError('rebuild/install route: bound-direct ABI missing')
    rank = int(os.environ['RANK'])
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
    handle = buffer.prepare_ccu_urma_peer_plan_alltoallv('ep8-heuristic', str(args.manifest.resolve()))
    dtype = getattr(torch, args.dtype)
    row_bytes = torch.empty((), dtype=dtype).element_size() * args.width
    planner_options = dict(min_chunk_bytes=args.min_chunk_bytes, min_gain_bytes=args.min_gain_bytes,
                           max_actions=args.max_actions, relay_capacity_bytes=[args.relay_capacity_mib << 20] * 8)
    selected = ('direct', 'multipath') if args.mode == 'both' else (args.mode,)
    reports = []
    for case_name, counts in cases:
        begin = time.perf_counter()
        plan = make_plan(counts, row_bytes, allowed, **planner_options)
        planner_us = (time.perf_counter() - begin) * 1e6
        schedule = build_schedule(plan)
        if plan.uses_relay and any(sum(schedule.scratch_rows(r)) * row_bytes > (args.max_scratch_mib << 20) for r in range(8)):
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
        prepare_options = dict(max_scratch_bytes=args.max_scratch_mib << 20,
                               sync_mode=args.executor_sync, policy=args.multipath_policy)
        prepared = PreparedAllToAllV(buffer, handle, rank, plan, send, **prepare_options)
        forced = (prepared if args.multipath_policy == 'force' else
                  PreparedAllToAllV(buffer, handle, rank, plan, send,
                                   **dict(prepare_options, policy='force')))
        # No-relay prepared timing is an explicit A/A control, using the same
        # direct call on both labels. Dynamic still includes policy planning.
        def selected_multi():
            if prepared.uses_relay:
                return prepared.execute(send, outputs['multipath'])
            return buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs['multipath'], handle, list(counts))
        ops = dict(direct=lambda: buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs['direct'], handle, list(counts)),
                   multipath=selected_multi,
                   native=lambda: dist.all_to_all_single(outputs['native'], send, input_split_sizes=sends, output_split_sizes=recvs))
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
                forced.execute(send, outputs['multipath'])
                snapshots.append((queued_epoch, outputs['multipath'].clone()))
            torch.npu.synchronize()
            for queued_epoch, snapshot in snapshots:
                if not torch.equal(snapshot.cpu(), expected + queued_epoch):
                    raise AssertionError(f'queued stream ordering failure rank={rank}')
            dist.barrier()
        timings = {name: [] for name in selected}
        for round_id in range(args.rounds):
            order = selected if round_id % 2 == 0 else tuple(reversed(selected))
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
        dynamic_rounds = {name: [] for name in selected}
        components = {name: [] for name in selected}
        def dynamic_once(name):
            start = time.perf_counter()
            current = gather_routing_counts(local_counts)
            gathered_at = time.perf_counter()
            if name == 'direct':
                current_counts = list(current)
                ready_at = time.perf_counter()
                buffer.ccu_urma_peer_plan_alltoallv_out(send, outputs[name], handle, current_counts)
            else:
                current_plan = make_plan(current, row_bytes, allowed, **planner_options)
                current_prepared = PreparedAllToAllV(buffer, handle, rank, current_plan, send,
                                                    **prepare_options)
                ready_at = time.perf_counter()
                current_prepared.execute(send, outputs[name])
            # Completion is part of BOTH paths; temporary workspace remains live.
            torch.npu.synchronize()
            finished = time.perf_counter()
            return ((gathered_at - start) * 1e6, (ready_at - gathered_at) * 1e6,
                    (finished - ready_at) * 1e6)
        for round_id in range(args.rounds):
            order = selected if round_id % 2 == 0 else tuple(reversed(selected))
            for name in order:
                set_payload(0)
                # Warm the SAME counts collective + planner/allocator + executor
                # path, rather than only warming prepared payload communication.
                for _ in range(args.warmup): dynamic_once(name)
                dist.barrier(); torch.npu.synchronize()
                totals = [0., 0., 0.]
                begin = time.perf_counter()
                for _ in range(args.dynamic_iters):
                    sample = dynamic_once(name)
                    for i in range(3): totals[i] += sample[i]
                dynamic_rounds[name].append((time.perf_counter() - begin) * 1e6 / args.dynamic_iters)
                components[name].append(dict(zip(('counts_gather_sync_us', 'plan_workspace_us',
                                                  'execute_completion_us'),
                                                 (v / args.dynamic_iters for v in totals))))
                check(name, 0)
        dynamic = {name: sum(v) / len(v) for name, v in dynamic_rounds.items()}
        report = dict(case=case_name, correctness='PASS', plan=plan.report(), schedule=schedule.report(),
                      planner_us=planner_us, prepared_round_us=timings, dynamic_mean_us=dynamic,
                      dynamic_round_us=dynamic_rounds, dynamic_components_round_us=components,
                      execution=prepared.decision, executor_sync=args.executor_sync,
                      fingerprint=plan.fingerprint)
        reports.append(report)
        print('EP8_HEURISTIC_CASE ' + json.dumps(dict(rank=rank, case=case_name, actions=len(plan.actions),
                                                    prepared_round_us=timings, dynamic_mean_us=dynamic)), flush=True)
    torch.npu.synchronize()
    destroy = route.HcclCcuUrmaPeerPlanDestroy
    destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
    if destroy(handle) != 0: raise RuntimeError('plan destroy failed')
    dist.barrier(); dist.destroy_process_group()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    result = dict(rank=rank, ranks=8, correctness='PASS', dtype=args.dtype, width=args.width,
                  benchmark=dict(protocol=2, warmup=args.warmup, iters=args.iters, rounds=args.rounds,
                                 dynamic_iters=args.dynamic_iters, executor_sync=args.executor_sync,
                                 multipath_policy=args.multipath_policy),
                  mode=args.mode, backend='participant_hbm_two_stage_ccu', cases=reports,
                  timing_scope='host_wall_submit_and_device_completion',
                  dynamic_scope='device_counts_gather_cpu_sync_plan_workspace_execute_completion')
    temp = args.run_dir / f'rank{rank}.json.tmp'
    temp.write_text(json.dumps(result, indent=2) + '\n')
    temp.replace(args.run_dir / f'rank{rank}.json')


if __name__ == '__main__':
    main()
