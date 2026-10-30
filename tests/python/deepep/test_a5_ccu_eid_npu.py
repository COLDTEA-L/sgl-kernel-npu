#!/usr/bin/env python3
"""EP8 end-to-end explicit-EID AllToAllV; one communicator, no relay HBM."""
import argparse
import ctypes
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import RelayPlanner, normalize_counts, normalize_relays, gather_routing_counts
from test_a5_ccu_heuristic_npu import synthetic
from a5_ccu_payload_test_utils import (PAYLOAD_DTYPE_NAMES, make_payload, full_payload,
    cat_payload, byte_payload_view, copy_payload_, fill_payload_, equal_payload,
    guards_intact, shifted_payload, clone_payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--topology-json', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--matrix-json', type=Path)
    p.add_argument('--suite', choices=('create', 'correctness', 'benchmark'), default='benchmark')
    p.add_argument('--case', choices=('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut'), default='single_hot')
    p.add_argument('--mode', choices=('all', 'native', 'direct', 'multipath'), default='all')
    p.add_argument('--dtype', choices=PAYLOAD_DTYPE_NAMES, default='bfloat16')
    p.add_argument('--width', type=int, default=4096)
    p.add_argument('--row-scale', type=int, default=128)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--rounds', type=int, default=6)
    p.add_argument('--dynamic-iters', type=int, default=100)
    p.add_argument('--bidirectional-iters', type=int, default=0)
    p.add_argument('--allow-no-relay', action='store_true', help='explicit direct-only diagnostic, never a relay PASS')
    args = p.parse_args()
    if (int(os.environ.get('WORLD_SIZE', 0)) != 8 or min(args.width, args.row_scale, args.iters,
            args.rounds, args.dynamic_iters) < 1 or min(args.warmup, args.bidirectional_iters) < 0):
        p.error('requires EP8, positive sizes/iterations and nonnegative warmup')
    if args.bidirectional_iters and not args.dtype.startswith('float8_'):
        p.error('bidirectional test requires an FP8 dispatch dtype')
    topology = json.loads(args.topology_json.read_text())
    if topology['version'] != 2 or topology['executor'] != 'participant_eid_ccu_urma':
        raise ValueError('requires generated EP8 explicit-EID topology v2')
    if topology['devices'] != [int(v) for v in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]:
        raise ValueError('topology rank/device mapping mismatch')
    allowed = normalize_relays(topology['allowed_relays'])
    if args.matrix_json:
        data = json.loads(args.matrix_json.read_text())
        if 'devices' in data and data['devices'] != topology['devices']:
            raise ValueError('matrix physical-device order mismatch')
        cases = [('external', normalize_counts(data['counts']))]
    else:
        names = ('zero', 'self', 'single_hot', 'competing', 'asymmetric', 'uniform', 'cut') if args.suite == 'correctness' else (args.case,)
        cases = [(name, synthetic(name, args.row_scale)) for name in names]
    os.environ['A5_CCU_EP8_THREAD_BRIDGE'] = 'event'
    from a5_ccu_test_runtime import prepare_runtime, import_deep_ep, ccu_pg_kwargs
    extension, route_path = prepare_runtime(__file__)
    import torch
    import torch.distributed as dist
    import torch_npu
    deep_ep, _ = import_deep_ep(extension)
    native = sys.modules['deep_ep_cpp']
    if getattr(native, 'ep8_eid_host_abi', None) != 1:
        raise RuntimeError('rebuild/install DeepEP: ep8_eid_host_abi=1 required')
    route = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    for symbol, expected in (('A5CcuEidAllToAllVAbiVersion', 1), ('A5CcuEidV1CompatAbiVersion', 1),
                             ('A5CcuEidPeerResourceAbiVersion', 1),
                             ('A5CcuPublicApiAbiVersion', 1), ('A5CcuEp8DfxAbiVersion', 2)):
        if not hasattr(route, symbol): raise RuntimeError('rebuild/install route: missing ' + symbol)
        fn = getattr(route, symbol); fn.restype = ctypes.c_int
        if fn() != expected: raise RuntimeError('rebuild route: ' + symbol)
    rank = int(os.environ['RANK'])
    def phase(name, **fields):
        print('EP8_EID_PHASE ' + json.dumps(dict(rank=rank, pid=os.getpid(), phase=name,
              time=datetime.now().astimezone().isoformat(timespec='microseconds'), **fields)), flush=True)
    phase('process_group_begin')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('hccl', timeout=timedelta(seconds=180), **ccu_pg_kwargs(torch_npu))
    identity = json.dumps(dict(options=vars(args), cases=cases, topology=topology,
                               manifest=args.manifest.read_text()), default=str, sort_keys=True)
    digest = torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.int32, device='npu')
    hashes = [torch.empty_like(digest) for _ in range(8)]
    dist.all_gather(hashes, digest)
    if any(not torch.equal(h.cpu(), digest.cpu()) for h in hashes): raise ValueError('rank configuration mismatch')
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    phase('catalog_register_begin', manifest=str(args.manifest.resolve()),
          manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
          channels=topology.get('channels_per_rank'), executor=topology['executor'])
    handle = buffer.prepare_ccu_urma_eid_alltoallv('ep8-eid', str(args.manifest.resolve()))
    phase('catalog_register_complete', handle=handle)
    if args.suite == 'create':
        # No payload host, FP8 allocation, AllToAllV execution or timed loop.
        destroy = route.HcclCcuUrmaPeerPlanDestroy
        destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
        status = destroy(handle)
        if status != 0: raise RuntimeError(f'EID create-only destroy failed status={status}')
        dist.barrier()
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,
            backend='participant_eid_ccu_urma', options=vars(args), reports=[]), default=str))
        dist.destroy_process_group()
        print(f'EP8_EID_CREATE_RANK_PASS rank={rank}', flush=True)
        return
    dtype = getattr(torch, args.dtype)
    row_bytes = args.width * torch.empty((), dtype=dtype).element_size()
    # Same registered channels/kernel for direct and multipath; only policy differs.
    planners = {mode: RelayPlanner(row_bytes, allowed, max_actions=0 if mode == 'direct' else 56,
                                   native_module=native) for mode in ('direct', 'multipath')}
    hosts = {mode: native.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
             for mode, planner in planners.items()}
    def native_transfer(send, recv, sends, recvs):
        work = dist.all_to_all_single(byte_payload_view(recv), byte_payload_view(send),
            input_split_sizes=sends, output_split_sizes=recvs, group=dist.group.WORLD, async_op=True)
        work.wait()
    if args.bidirectional_iters:
        from a5_ccu_bidirectional_test import check_bidirectional
        rows = check_bidirectional(hosts['multipath'], cases, rank, dtype, args.width,
                                   args.bidirectional_iters, native_transfer)
        print('EP8_EID_BIDIRECTIONAL_PASS ' + json.dumps(dict(rank=rank, cases=rows)), flush=True)
    selected = ('native', 'direct', 'multipath') if args.mode == 'all' else (args.mode,)
    reports = []
    for case, counts in cases:
        phase('correctness_begin', case=case)
        sends, recvs = list(counts[rank*8:(rank+1)*8]), list(counts[rank::8])
        source = make_payload(sum(sends), args.width, rank, dtype)
        expected = cat_payload([make_payload(sum(counts[s*8:(s+1)*8]), args.width, s, dtype)[
            sum(counts[s*8:s*8+rank]):sum(counts[s*8:s*8+rank])+counts[s*8+rank]] for s in range(8)])
        guard = 37
        input_store = full_payload((source.numel()+2*guard,), -19, dtype, 'npu')
        send = input_store[guard:-guard].reshape(sum(sends), args.width)
        stores = {mode: full_payload((expected.numel()+2*guard,), -17, dtype, 'npu') for mode in ('native', 'direct', 'multipath')}
        outputs = {mode: store[guard:-guard].reshape(sum(recvs), args.width) for mode, store in stores.items()}
        prepared = {mode: host.prepare(counts, send) for mode, host in hosts.items()}
        plan = planners['multipath'].plan(counts)
        if prepared['multipath'].describe() != (plan.counts, plan.relay_rank, plan.relay_rows, plan.actions):
            raise AssertionError('host/reference plan mismatch')
        if case == 'single_hot' and not plan.uses_relay and not args.allow_no_relay:
            raise RuntimeError('single_hot has no usable relay: inspect topology exclusions; not a multipath validation')
        print('EP8_EID_CASE ' + json.dumps(dict(rank=rank, case=case, executed_relay=plan.uses_relay,
              relay_rows=sum(plan.relay_rows), cross_bytes=sum(counts[s*8+d] for s in range(8) for d in range(8) if s != d)*row_bytes,
              relay_hbm_bytes=0, ccu_launches_per_call=1)), flush=True)
        def op(mode):
            if mode == 'native': native_transfer(send, outputs[mode], sends, recvs)
            else: hosts[mode].execute(prepared[mode], send, outputs[mode])
        def setup(epoch):
            copy_payload_(send, shifted_payload(source, epoch))
            for out in outputs.values(): fill_payload_(out, -17)
            torch.npu.synchronize()
        def check(mode, epoch):
            if not equal_payload(outputs[mode], shifted_payload(expected, epoch)): raise AssertionError(f'{case} {mode} payload mismatch rank={rank}')
            if not equal_payload(send, shifted_payload(source, epoch)): raise AssertionError('input modified')
            if not guards_intact(input_store, guard, -19) or not guards_intact(stores[mode], guard, -17): raise AssertionError('guard corruption')
        for epoch in range(2):
            setup(epoch)
            for mode in ('native', 'direct', 'multipath'):
                op(mode); torch.npu.synchronize(); check(mode, epoch)
            hosts['multipath'].run(counts, send, outputs['multipath'])
            torch.npu.synchronize(); check('multipath', epoch)
        # Producer/consumer ordering and repeated notification epochs, no fence between calls.
        snapshots = []
        for epoch in (7, 13, 19):
            copy_payload_(send, shifted_payload(source, epoch))
            op('multipath'); snapshots.append((epoch, clone_payload(outputs['multipath'])))
        torch.npu.synchronize()
        for epoch, snapshot in snapshots:
            if not equal_payload(snapshot, shifted_payload(expected, epoch)): raise AssertionError('queued ordering mismatch')
        local_counts = torch.tensor(sends, dtype=torch.int64, device='npu')
        def dynamic(mode):
            current = gather_routing_counts(local_counts)
            if mode == 'native': native_transfer(send, outputs[mode], list(current[rank*8:(rank+1)*8]), list(current[rank::8]))
            else: hosts[mode].run(current, send, outputs[mode])
        samples = {metric: {mode: [] for mode in selected} for metric in ('prepared', 'dynamic')}
        for metric, iterations in (('prepared', args.iters), ('dynamic', args.dynamic_iters)):
            call = op if metric == 'prepared' else dynamic
            for round_id in range(args.rounds):
                order = selected[round_id % len(selected):] + selected[:round_id % len(selected)]
                if round_id % 2: order = tuple(reversed(order))
                for mode in order:
                    phase('measure_begin', case=case, metric=metric, mode=mode, round=round_id)
                    setup(round_id)
                    for _ in range(args.warmup): call(mode)
                    torch.npu.synchronize(); dist.barrier(); torch.npu.synchronize()
                    start = time.perf_counter()
                    for _ in range(iterations): call(mode)
                    torch.npu.synchronize()
                    elapsed = (time.perf_counter()-start)*1e6/iterations
                    check(mode, round_id)
                    # Slowest participant, outside measured interval.
                    value = torch.tensor([elapsed], dtype=torch.float64, device='npu')
                    dist.all_reduce(value, op=dist.ReduceOp.MAX)
                    samples[metric][mode].append(value.item())
                    phase('measure_complete', case=case, metric=metric, mode=mode, round=round_id)
        report = dict(case=case, row_bytes=row_bytes, counts=counts, relay_rows=sum(plan.relay_rows),
                      executed_relay=plan.uses_relay, samples_us=samples,
                      median_us={metric: {mode: statistics.median(values) for mode, values in modes.items()} for metric, modes in samples.items()},
                      workspace={mode: host.stats() for mode, host in hosts.items()})
        reports.append(report)
        print('EP8_EID_CASE_PASS ' + json.dumps(dict(rank=rank, **report)), flush=True)
    for host in hosts.values(): host.close()
    destroy = route.HcclCcuUrmaPeerPlanDestroy
    destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
    if destroy(handle) != 0: raise RuntimeError('EID plan destroy failed')
    dist.barrier()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, backend='participant_eid_ccu_urma',
        options=vars(args), reports=reports), default=str, indent=2))
    dist.destroy_process_group()
    print(f'EP8_EID_RANK_PASS rank={rank}', flush=True)


if __name__ == '__main__':
    main()
