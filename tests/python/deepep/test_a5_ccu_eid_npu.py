#!/usr/bin/env python3
"""EP8 end-to-end explicit-EID AllToAllV; one communicator, no relay HBM."""
import argparse
import ctypes
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

_last_phase = {}

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python/deep_ep/deep_ep'))
from heuristic_alltoallv import RelayPlanner, normalize_counts, normalize_relays, normalize_l1_budget, gather_routing_counts
from test_a5_ccu_heuristic_npu import synthetic
from ccu_l1_config import planner_l1_options, l1_enabled, validate_l1_schema
from a5_ccu_payload_test_utils import (PAYLOAD_DTYPE_NAMES, make_payload, full_payload,
    cat_payload, byte_payload_view, copy_payload_, fill_payload_, equal_payload,
    guards_intact, shifted_payload, clone_payload)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--topology-json', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--matrix-json', type=Path)
    p.add_argument('--planner', choices=('greedy', 'bottleneck', 'l1_bottleneck'), default='bottleneck')
    p.add_argument('--max-actions', type=int, default=56, help='L1 action budget 0..56; legacy bottleneck remains capped at 8')
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
    p.add_argument('--combination-audit', action='store_true',
                   help='extra host-boundary/control measurements; fixed matrix only, not a model fast path')
    p.add_argument('--dynamic-stage-audit', action='store_true',
                   help='focused interleaved legacy-count diagnostics; adds host timers, no new synchronization')
    p.add_argument('--counts-path', choices=('legacy', 'reuse', 'stream'), default='legacy',
                   help='dynamic/counts_only: old adapter, reusable buffers, or reusable dedicated count stream')
    p.add_argument('--bidirectional-iters', type=int, default=0)
    p.add_argument('--allow-no-relay', action='store_true', help='explicit direct-only diagnostic, never a relay PASS')
    p.add_argument('--minimal', action='store_true', help='create only: stop after configuration all_gather; no Buffer or EID plan')
    p.add_argument('--config-prewarm', type=int, default=0, help='optional startup all_gather count; fail immediately on error, no retries')
    p.add_argument('--register-stage', choices=('minimal', 'args', 'mask', 'channel_vars', 'sync', 'direct', 'relay', 'self', 'full', 'mask_init', 'mask_negative', 'mask_compare', 'mask_add', 'mask_static', 'mask_equal', 'mask_not_equal', 'mask_compare_zero', 'mask_compare_single'), default='full', help='create-only registration graph; mask probes are isolated from production')
    args = p.parse_args()
    if args.dynamic_stage_audit:
        if args.suite != "benchmark" or args.counts_path != "legacy" or args.matrix_json is None:
            p.error("--dynamic-stage-audit requires benchmark, legacy counts and --matrix-json")
        args.combination_audit = True
    if not 0 <= args.max_actions <= 56:
        p.error('--max-actions must be in [0,56]')
    if args.combination_audit and args.suite == 'create':
        p.error('--combination-audit requires payload measurements, not create')
    if args.register_stage != 'full' and (args.suite != 'create' or args.minimal):
        p.error('--register-stage probes require --suite create without --minimal')
    os.environ['A5_EID_REGISTER_STAGE'] = args.register_stage
    if args.minimal and args.suite != 'create':
        p.error('--minimal requires --suite create')
    if args.config_prewarm < 0:
        p.error('--config-prewarm must be nonnegative')
    if (int(os.environ.get('WORLD_SIZE', 0)) != 8 or min(args.width, args.row_scale, args.iters,
            args.rounds, args.dynamic_iters) < 1 or min(args.warmup, args.bidirectional_iters) < 0):
        p.error('requires EP8, positive sizes/iterations and nonnegative warmup')
    if args.bidirectional_iters and not args.dtype.startswith('float8_'):
        p.error('bidirectional test requires an FP8 dispatch dtype')
    topology = json.loads(args.topology_json.read_text())
    if topology['version'] not in (2,3) or topology['executor'] != 'participant_eid_ccu_urma':
        raise ValueError('requires generated EP8 explicit-EID topology v2')
    if topology['devices'] != [int(v) for v in os.environ['ASCEND_RT_VISIBLE_DEVICES'].split(',')]:
        raise ValueError('topology rank/device mapping mismatch')
    allowed = normalize_relays(topology['allowed_relays'])
    l1_resources = topology.get('l1_resources')
    if l1_resources is not None and not l1_enabled(True):
        raise ValueError('L1 disabled but manifest/topology contains L1; regenerate with A5_ENABLE_L1=0')
    args.requested_planner = args.planner
    args.planner, l1_resources, _ = planner_l1_options(args.planner, l1_resources, topology.get('l1_budget'))
    print(f'L1_PLANNER_SWITCH requested={args.requested_planner} effective={args.planner}', flush=True)
    if (args.planner == 'l1_bottleneck') != (l1_resources is not None):
        raise ValueError('l1_bottleneck requires --l1-config topology and vice versa')
    if l1_resources is not None:
        validate_l1_schema(topology['l1_policy']['schema'])
    if topology.get('l1_budget') is not None:
        if l1_resources is None: raise ValueError('L1 budget requires L1 paths')
        normalize_l1_budget(topology['l1_budget'])
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
    if args.counts_path != 'legacy' and getattr(native, 'ep8_eid_host_counts_tensor_abi', None) != 1:
        raise RuntimeError('rebuild/install DeepEP: ep8_eid_host_counts_tensor_abi=1 required')
    route = ctypes.CDLL(str(route_path), mode=ctypes.RTLD_GLOBAL)
    if l1_resources is not None and route.A5CcuSinglePortL1AbiVersion()!=1:
        raise RuntimeError("rebuild route: A5CcuSinglePortL1AbiVersion=1 required")
    if l1_resources is not None and topology['l1_policy']['schema'] == 'a5.rank-graph-l1.v1':
        if getattr(route, 'A5CcuRankGraphL1AbiVersion', lambda: 0)() != 1:
            raise RuntimeError('rebuild route: A5CcuRankGraphL1AbiVersion=1 required')
    required_abis = [
        ('A5CcuEidAllToAllVAbiVersion', 1), ('A5CcuEidV1CompatAbiVersion', 1),
        ('A5CcuEidPeerResourceAbiVersion', 1), ('A5CcuEidRegisterProbeAbiVersion', 1),
        ('A5CcuEidDirectControlAbiVersion', 1), ('A5CcuPublicApiAbiVersion', 1),
        ('A5CcuEp8DfxAbiVersion', 2),
    ]
    # Existing direct-control routes can run full/correctness unchanged. Only
    # the new comparison probes require their additional capability marker.
    if args.register_stage in ('mask_equal', 'mask_not_equal', 'mask_compare_zero', 'mask_compare_single'):
        required_abis.append(('A5CcuEidCompareProbeAbiVersion', 1))
    for symbol, expected in required_abis:
        if not hasattr(route, symbol): raise RuntimeError('rebuild/install route: missing ' + symbol)
        fn = getattr(route, symbol); fn.restype = ctypes.c_int
        if fn() != expected: raise RuntimeError('rebuild route: ' + symbol)
    rank = int(os.environ['RANK'])
    args.run_dir.mkdir(parents=True, exist_ok=True)
    phase_path = args.run_dir / f'rank{rank}.phases.jsonl'
    from a5_ccu_startup_probe import make_phase_logger, traced_all_gather
    phase = make_phase_logger(phase_path, dict(
        rank=rank, pid=os.getpid(), planner=args.planner, dtype=args.dtype,
        payload_dtype=args.dtype, width=args.width, register_stage=args.register_stage), _last_phase)
    phase('startup_resources', custom_channels_acquired=0, custom_plans=0,
          process_groups_requested=1, hccl_buffsize_mb=os.environ.get('HCCL_BUFFSIZE'),
          minimal=args.minimal, config_prewarm=args.config_prewarm, register_stage=args.register_stage)
    phase('device_init_begin')
    torch.npu.set_device(int(os.environ['LOCAL_RANK']))
    phase('device_init_complete')
    phase('process_group_begin')
    dist.init_process_group('hccl', timeout=timedelta(seconds=180), **ccu_pg_kwargs(torch_npu))
    phase('process_group_complete')
    identity = json.dumps(dict(options=vars(args), cases=cases, topology=topology,
                               manifest=args.manifest.read_text()), default=str, sort_keys=True)
    phase('config_digest_begin')
    digest = torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.int32, device='npu')
    hashes = [torch.empty_like(digest) for _ in range(8)]
    torch.npu.synchronize()
    phase('config_digest_complete')
    for warmup in range(args.config_prewarm):
        traced_all_gather(torch, dist, hashes, digest, phase, f'config_prewarm_{warmup}')
    traced_all_gather(torch, dist, hashes, digest, phase)
    phase('config_check_begin')
    expected_digest=digest.cpu()
    if any(not torch.equal(h.cpu(), expected_digest) for h in hashes): raise ValueError('rank configuration mismatch')
    phase('config_check_complete')
    if args.minimal:
        # Same imports, process group and config check as full create, but no
        # Buffer, EID channel acquisition or kernel registration.
        phase('process_group_destroy_begin')
        dist.destroy_process_group()
        phase('process_group_destroy_complete')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,
            backend='participant_eid_ccu_urma', options=vars(args), reports=[]), default=str))
        os.write(1, f'EP8_EID_MINIMAL_RANK_PASS rank={rank}\n'.encode())
        return
    phase('buffer_init_begin')
    buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
    torch.npu.synchronize()
    phase('buffer_init_complete')
    phase('catalog_register_begin', manifest=str(args.manifest.resolve()),
          manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
          channels=topology.get('channels_per_rank'), executor=topology['executor'])
    handle = buffer.prepare_ccu_urma_eid_alltoallv('ep8-eid', str(args.manifest.resolve()))
    phase('catalog_register_complete', handle=handle)
    if args.suite == 'create':
        # No payload host, FP8 allocation, AllToAllV execution or timed loop.
        destroy = route.HcclCcuUrmaPeerPlanDestroy
        destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
        phase('plan_destroy_begin', handle=handle)
        status = destroy(handle)
        if status != 0: raise RuntimeError(f'EID create-only destroy failed status={status}')
        phase('plan_destroy_complete', handle=handle, channels_released_by_plan=False)
        phase('create_exit_barrier_begin')
        dist.barrier()
        torch.npu.synchronize()
        phase('create_exit_barrier_complete')
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank,
            backend='participant_eid_ccu_urma', options=vars(args), reports=[]), default=str))
        phase('process_group_destroy_begin')
        dist.destroy_process_group()
        phase('process_group_destroy_complete')
        marker = 'EP8_EID_CREATE_RANK_PASS' if args.register_stage == 'full' else 'EP8_EID_REGISTER_PROBE_RANK_PASS'
        os.write(1, f'{marker} rank={rank}\n'.encode())
        return
    dtype = getattr(torch, args.dtype)
    row_bytes = args.width * torch.empty((), dtype=dtype).element_size()
    # Same registered channels/kernel for direct and multipath; only policy differs.
    planners = {mode: RelayPlanner(row_bytes, allowed, max_actions=0 if mode == 'direct' else args.max_actions,
                                   native_module=native, strategy=args.planner, l1_resources=l1_resources, l1_budget=topology.get('l1_budget')) for mode in ('direct', 'multipath')}
    hosts = {mode: native.ep8_eid_create(planner._native, rank, buffer.runtime, handle)
             for mode, planner in planners.items()}
    from a5_ccu_dynamic_stage_audit import StageRecorder, read_legacy_counts, measurement_schedule, STAGE_METRICS
    stages = StageRecorder() if args.dynamic_stage_audit else None
    def native_transfer(send, recv, sends, recvs):
        if stages is not None:
            out_bytes, in_bytes = stages.measure("native_views", lambda: (byte_payload_view(recv), byte_payload_view(send)))
            work = stages.measure("native_enqueue", dist.all_to_all_single, out_bytes, in_bytes,
                input_split_sizes=sends, output_split_sizes=recvs, group=dist.group.WORLD, async_op=True)
            stages.measure("native_work_wait", work.wait)
            return
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
        phase('payload_allocate_begin', case=case, row_bytes=row_bytes, send_bytes=sum(sends)*row_bytes, recv_bytes=sum(recvs)*row_bytes)
        source = make_payload(sum(sends), args.width, rank, dtype)
        expected = cat_payload([make_payload(sum(counts[s*8:(s+1)*8]), args.width, s, dtype)[
            sum(counts[s*8:s*8+rank]):sum(counts[s*8:s*8+rank])+counts[s*8+rank]] for s in range(8)])
        guard = 37
        input_store = full_payload((source.numel()+2*guard,), -19, dtype, 'npu')
        send = input_store[guard:-guard].reshape(sum(sends), args.width)
        stores = {mode: full_payload((expected.numel()+2*guard,), -17, dtype, 'npu') for mode in ('native', 'direct', 'multipath')}
        outputs = {mode: store[guard:-guard].reshape(sum(recvs), args.width) for mode, store in stores.items()}
        phase('payload_allocate_complete', case=case)
        plan = planners['multipath'].plan(counts)
        actions_bytes = []
        for src, dst, relay, rows, _ in plan.actions:
            direct_rows = counts[src*8+dst] - rows
            actions_bytes.append(dict(src=src, dst=dst, relay=relay, rows=rows,
                bytes=rows*row_bytes,
                send_offset=(sum(counts[src*8:src*8+dst])+direct_rows)*row_bytes,
                recv_offset=(sum(counts[s*8+dst] for s in range(src))+direct_rows)*row_bytes))
        plan_record = dict(rank=rank, case=case, planner=args.planner, dtype=args.dtype,
            plan=plan.report(), relay_capacity_bytes=None, actions_bytes=actions_bytes,
            send_bytes=send.nbytes, recv_bytes=outputs['multipath'].nbytes,
            send_pointer=hex(send.data_ptr()), recv_pointer=hex(outputs['multipath'].data_ptr()))
        plan_path = args.run_dir / f'rank{rank}.{case}.plan.json'
        plan_path.write_text(json.dumps(plan_record, indent=2))
        phase('prepare_begin', case=case, row_bytes=row_bytes,
              plan_fingerprint=plan.fingerprint, plan_file=str(plan_path))
        prepared = {mode: host.prepare(counts, send) for mode, host in hosts.items()}
        phase('prepare_complete', case=case)
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
                phase('validate_submit_begin', case=case, mode=mode, epoch=epoch, api='execute')
                op(mode)
                phase('validate_sync_begin', case=case, mode=mode, epoch=epoch, api='execute')
                torch.npu.synchronize(); check(mode, epoch)
                phase('validate_complete', case=case, mode=mode, epoch=epoch, api='execute')
            phase('validate_submit_begin', case=case, mode='multipath', epoch=epoch, api='run')
            hosts['multipath'].run(counts, send, outputs['multipath'])
            phase('validate_sync_begin', case=case, mode='multipath', epoch=epoch, api='run')
            torch.npu.synchronize(); check('multipath', epoch)
            phase('validate_complete', case=case, mode='multipath', epoch=epoch, api='run')
        # Producer/consumer ordering and repeated notification epochs, no fence between calls.
        phase('queued_validation_begin', case=case)
        snapshots = []
        for epoch in (7, 13, 19):
            copy_payload_(send, shifted_payload(source, epoch))
            op('multipath'); snapshots.append((epoch, clone_payload(outputs['multipath'])))
        torch.npu.synchronize()
        for epoch, snapshot in snapshots:
            if not equal_payload(snapshot, shifted_payload(expected, epoch)): raise AssertionError('queued ordering mismatch')
        phase('queued_validation_complete', case=case)
        local_counts = torch.tensor(sends, dtype=torch.int64, device='npu')
        exchange = None
        if args.counts_path != 'legacy':
            from routing_counts import RoutingCountsExchange
            exchange = RoutingCountsExchange(local_counts, dedicated_stream=args.counts_path == 'stream')
            phase('counts_exchange_validate_begin', case=case, counts_path=args.counts_path)
            # Changing producer values prove that buffer reuse does not cache C.
            # begin/finish bracket an unrelated current-stream operation, without
            # moving a collective across any other collective on the EP group.
            for epoch in range(3):
                probe = torch.arange(8, dtype=torch.int64, device='npu') + rank*100 + epoch*1000
                exchange.begin(probe)
                independent = probe + 1
                actual = exchange.finish().tolist()
                if actual != [r*100 + epoch*1000 + j for r in range(8) for j in range(8)]:
                    raise AssertionError('reusable counts are stale or rank order is wrong')
                if independent.cpu().tolist() != [rank*100 + epoch*1000 + j+1 for j in range(8)]:
                    raise AssertionError('producer stream ordering failed')
            phase('counts_exchange_validate_complete', case=case, counts_path=args.counts_path)
        last_counts = None
        def read_counts():
            nonlocal last_counts
            if stages is not None:
                last_counts = read_legacy_counts(local_counts, stages, normalize_counts, torch=torch, dist=dist)
            else:
                last_counts = exchange.gather(local_counts) if exchange is not None else gather_routing_counts(local_counts)
            return last_counts
        def submit_counts(mode, current):
            if mode == 'native':
                # HCCL's public split API still requires host integer lists.
                if exchange is not None: current = current.tolist()
                if stages is not None:
                    ss, rs = stages.measure('native_splits', lambda: (list(current[rank*8:(rank+1)*8]), list(current[rank::8])))
                    native_transfer(send, outputs[mode], ss, rs)
                else:
                    native_transfer(send, outputs[mode], list(current[rank*8:(rank+1)*8]), list(current[rank::8]))
            elif exchange is not None:
                hosts[mode].run_counts_tensor(current, send, outputs[mode])
            else:
                if stages is not None:
                    result = stages.measure('custom_run', hosts[mode].run, current, send, outputs[mode])
                    stages.cpp_run(result)
                else:
                    hosts[mode].run(current, send, outputs[mode])
        def dynamic(mode):
            submit_counts(mode, read_counts())
        def host_counts(mode):
            if mode == 'native': native_transfer(send, outputs[mode], sends, recvs)
            else: hosts[mode].run(counts, send, outputs[mode])
        def host_counts_serial(mode):
            host_counts(mode)
            torch.npu.synchronize()
        def counts_only(mode):
            nonlocal last_counts
            last_counts = read_counts()
        def check_counts():
            actual = last_counts.tolist() if exchange is not None else last_counts
            if tuple(actual) != counts: raise AssertionError('count exchange mismatch')
        from a5_ccu_combination_audit import AUDIT_METRICS, CombinationAudit
        prepared_controls = {}
        def prepare_only(mode, current):
            if mode == 'native':
                if exchange is not None: current = current.tolist()
                prepared_controls[mode] = (list(current[rank*8:(rank+1)*8]), list(current[rank::8]))
            elif exchange is not None:
                prepared_controls[mode] = hosts[mode].prepare_counts_tensor(current, send)
            else:
                prepared_controls[mode] = hosts[mode].prepare(current, send)
        def audit_execute(mode):
            if stages is not None and mode != 'native':
                stages.measure('custom_execute', op, mode)
            else:
                op(mode)
        def audit_prepare(mode, current):
            if stages is not None:
                stages.measure('prepare_only', prepare_only, mode, current)
            else:
                prepare_only(mode, current)
        audit = CombinationAudit(read_counts, submit_counts, audit_execute, audit_prepare,
                                 torch.npu.synchronize, stages=stages)
        rank_local_audit = []
        metrics = ('prepared', 'host_counts', 'host_counts_serial', 'counts_only', 'dynamic')
        if args.combination_audit: metrics += AUDIT_METRICS
        if args.dynamic_stage_audit: metrics = STAGE_METRICS
        samples = {metric: {mode: [] for mode in selected} for metric in metrics}
        schedule = measurement_schedule(metrics, selected, args.rounds, interleave=args.dynamic_stage_audit)
        for metric, round_id, mode in schedule:
            iterations = args.iters if metric == 'prepared' else args.dynamic_iters
            call = (lambda mode: audit.call(metric, mode)) if metric.startswith('audit_') else dict(
                prepared=op, host_counts=host_counts, host_counts_serial=host_counts_serial,
                counts_only=counts_only, dynamic=dynamic)[metric]
            phase('warmup_begin', case=case, metric=metric, mode=mode, round=round_id)
            setup(round_id)
            for _ in range(args.warmup): call(mode)
            phase('warmup_sync_begin', case=case, metric=metric, mode=mode, round=round_id)
            torch.npu.synchronize()
            phase('measure_barrier_begin', case=case, metric=metric, mode=mode, round=round_id)
            dist.barrier(); torch.npu.synchronize()
            phase('measure_begin', case=case, metric=metric, mode=mode, round=round_id)
            audit.reset()  # discard warmup samples before the measured loop
            start = time.perf_counter()
            for _ in range(iterations): call(mode)
            drain_start = time.perf_counter() if metric.startswith('audit_') else None
            torch.npu.synchronize()
            stop = time.perf_counter()
            elapsed = (stop-start)*1e6/iterations
            if metric.startswith('audit_'):
                rank_local_audit.append(dict(metric=metric, mode=mode, round=round_id,
                    iterations=iterations, elapsed_host_us=elapsed,
                    final_drain_host_us=(stop-drain_start)*1e6,
                    samples=list(audit.samples)))
            phase('measure_execution_complete', case=case, metric=metric, mode=mode, round=round_id)
            if metric == 'counts_only' or metric.startswith('audit_'):
                check_counts()
            if metric == 'audit_counts_prepare_only':
                control = prepared_controls[mode]
                if mode == 'native':
                    if control != (sends, recvs): raise AssertionError('native split control mismatch')
                elif control.describe() != prepared[mode].describe():
                    raise AssertionError('prepare-only plan differs from reference')
            elif metric not in ('counts_only', 'audit_counts_only'):
                check(mode, round_id)
            phase('measure_check_complete', case=case, metric=metric, mode=mode, round=round_id)
            # Slowest participant, outside measured interval. HCCL rejects
            # FP64; FP32 is sufficient for microsecond timing samples.
            value = torch.tensor([elapsed], dtype=torch.float32, device='npu')
            phase('timing_reduce_begin', case=case, metric=metric, mode=mode, round=round_id)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            samples[metric][mode].append(value.item())
            phase('measure_complete', case=case, metric=metric, mode=mode, round=round_id)
        if exchange is not None: exchange.close()
        report = dict(case=case, counts_path=args.counts_path, row_bytes=row_bytes, counts=counts, relay_rows=sum(plan.relay_rows),
                      executed_relay=plan.uses_relay, planner=args.planner, plan=plan.report(), samples_us=samples,
                      median_us={metric: {mode: statistics.median(values) for mode, values in modes.items()} for metric, modes in samples.items()},
                      workspace={mode: host.stats() for mode, host in hosts.items()})
        if args.combination_audit:
            report['rank_local_audit'] = rank_local_audit
        reports.append(report)
        # Keep per-iteration diagnostics in rank JSON; avoid flooding shared stdout.
        print('EP8_EID_CASE_PASS ' + json.dumps(dict(rank=rank, **{k:v for k,v in report.items() if k != 'rank_local_audit'})), flush=True)
    for host in hosts.values(): host.close()
    destroy = route.HcclCcuUrmaPeerPlanDestroy
    destroy.argtypes = [ctypes.c_uint64]; destroy.restype = ctypes.c_int
    if destroy(handle) != 0: raise RuntimeError('EID plan destroy failed')
    dist.barrier()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / f'rank{rank}.json').write_text(json.dumps(dict(rank=rank, backend='participant_eid_ccu_urma',
        options=vars(args), reports=reports), default=str, indent=2))
    dist.destroy_process_group()
    os.write(1, f'EP8_EID_RANK_PASS rank={rank}\n'.encode())


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        print('EP8_EID_FAILURE ' + json.dumps(dict(rank=os.environ.get('RANK'), pid=os.getpid(),
            error_type=type(error).__name__, error=str(error), last_phase=_last_phase)), file=sys.stderr, flush=True)
        raise
