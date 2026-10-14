#!/usr/bin/env python3
"""Two-rank native-HCCL or CCU+URMA multi-route AllToAll benchmark."""

import argparse
import ctypes
from contextlib import nullcontext
import faulthandler
import json
import os
import re
import sys
import time
from pathlib import Path
from a5_ccu_test_runtime import (
    import_deep_ep as shared_import_deep_ep,
    prepare_runtime as shared_prepare_runtime,
)
from a5_ccu_peer_test_support import file_barrier, make_profiler, measurement_start
from a5_ccu_path_contention_support import (
    contention_measurement_start,
    safe_name,
    write_iteration_trace,
)

os.environ.setdefault("HCCL_OP_EXPANSION_MODE", "CCU_SCHED")
CURRENT_PHASE = "module_import"


def mark_phase(name):
    global CURRENT_PHASE
    CURRENT_PHASE = name
    print(
        f"CASE_PHASE pid={os.getpid()} rank={os.environ.get('RANK', 'NA')} phase={name}",
        flush=True,
    )


def start_phase_watchdog():
    text = os.environ.get("A5_CCU_PHASE_WATCHDOG_SECONDS", "0")
    try:
        seconds = int(text)
    except ValueError as error:
        raise RuntimeError(f"invalid A5_CCU_PHASE_WATCHDOG_SECONDS={text!r}") from error
    faulthandler.enable(all_threads=True)
    if seconds > 0:
        faulthandler.dump_traceback_later(seconds, repeat=True)


start_phase_watchdog()


def sanitize(value):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))


def encode_path_policy(weights):
    """Pack ABI-1 runtime path weights into one signed-int64-safe word."""
    if not 2 <= len(weights) <= 13:
        raise ValueError("runtime path policy requires 2..13 catalog weights")
    if any(weight < 0 or weight > 15 for weight in weights):
        raise ValueError("every runtime path weight must be in [0, 15]")
    if not any(weights):
        raise ValueError("runtime path policy must enable at least one path")
    word = 1 << 60
    for index, weight in enumerate(weights):
        word |= weight << (index * 4)
    return word


def prepare_runtime(require_route_library=True):
    return shared_prepare_runtime(__file__, require_route_library)


def requested_implementation():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--implementation", choices=("native", "multiroute", "explicit", "standard", "prepared"),
                        default="multiroute")
    return parser.parse_known_args()[0].implementation


mark_phase("requested_implementation")
IMPLEMENTATION = requested_implementation()
EXTENSION = None
ROUTE_LIB = None
if IMPLEMENTATION != "native":
    mark_phase("prepare_runtime")
    EXTENSION, ROUTE_LIB = prepare_runtime(IMPLEMENTATION != "standard")
    mark_phase(f"prepare_runtime_done_extension_{sanitize(EXTENSION)}")

mark_phase("import_torch")
import torch
mark_phase("import_torch_distributed")
import torch.distributed as dist
mark_phase("import_torch_npu")
import torch_npu
mark_phase("import_torch_npu_done")
if IMPLEMENTATION != "native":
    mark_phase("import_deep_ep")
    deep_ep, DEEP_EP_CPP = shared_import_deep_ep(EXTENSION)
    mark_phase("import_deep_ep_done")
    if IMPLEMENTATION in ("standard", "prepared"):
        extension = ctypes.CDLL(str(EXTENSION))
        try:
            attr_abi = extension.A5DeepEpExplicitMultipathAttrAbiVersion
        except AttributeError as error:
            raise RuntimeError(
                f"worker loaded stale DeepEP extension without attr ABI marker: {EXTENSION}"
            ) from error
        attr_abi.restype = ctypes.c_int
        version = attr_abi()
        required_version = 12 if IMPLEMENTATION == "prepared" else 8
        if version < required_version:
            raise RuntimeError(
                f"worker loaded DeepEP attr ABI {version}, expected >= {required_version}: {EXTENSION}"
            )
        print(f"CASE_DEEP_EP_ABI rank={os.environ.get('RANK', 'NA')} "
              f"version={version} extension={EXTENSION}", flush=True)
        if IMPLEMENTATION == "prepared":
            route_extension = ctypes.CDLL(str(ROUTE_LIB))
            try:
                plan_abi = route_extension.A5CcuUrmaPreparedPlanAbiVersion
            except AttributeError as error:
                raise RuntimeError(
                    f"route library lacks prepared-plan ABI marker: {ROUTE_LIB}"
                ) from error
            plan_abi.restype = ctypes.c_int
            if plan_abi() < 14:
                raise RuntimeError(f"invalid prepared-plan ABI in {ROUTE_LIB}")
            print(f"CASE_PREPARED_PLAN_ABI rank={os.environ.get('RANK', 'NA')} "
                  f"version={plan_abi()} route_library={ROUTE_LIB}", flush=True)
    if IMPLEMENTATION == "multiroute":
        route_extension = ctypes.CDLL(str(ROUTE_LIB))
        route_abi = route_extension.A5CcuUrmaPreparedPlanAbiVersion
        route_abi.restype = ctypes.c_int
        if route_abi() < 12:
            raise RuntimeError(
                f"legacy baseline requires route ABI >= 12 (single-path layout fix): {ROUTE_LIB}"
            )
        print(f"CASE_ROUTE_RUNTIME_ABI rank={os.environ.get('RANK', 'NA')} "
              f"version={route_abi()} route_library={ROUTE_LIB}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("native", "multiroute", "explicit", "standard", "prepared"),
                        default="multiroute")
    parser.add_argument("--bytes", type=int, default=2 * 1024 * 1024,
                        help="bytes in each destination slice")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32", "int8", "int32"),
                        default="float32", help="operator infers dtype from the input tensor")
    parser.add_argument("--route-index", type=int, default=0)
    parser.add_argument("--route-indices", default="")
    parser.add_argument("--path-uids", default="",
                        help="comma-separated PATH_CATALOG path_uid values; preferred over route ordinals")
    parser.add_argument("--path-weights", default="",
                        help="comma-separated positive weights, one per selected path")
    parser.add_argument("--replay-path-weights", default="",
                        help=("standard-op ACLGraph replay policy; comma-separated 0..15 weights, "
                              "where zero disables one pre-provisioned path"))
    parser.add_argument("--synthetic-route-manifest", default="",
                        help="TSV of explicitly resolved relay EID pairs")
    parser.add_argument("--direct-route", type=int, default=0,
                        help="HCCL-discovered direct candidate prepended in explicit mode")
    parser.add_argument("--relay-manifest", default="",
                        help="ordered explicit relay TSV for the formal direct+relay operator")
    parser.add_argument("--plan-id", default="a5-explicit-multipath",
                        help="stable control-plane key used by the standard graph operator")
    parser.add_argument("--schedule", choices=("concurrent", "serial"),
                        default="concurrent")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--completion-mode", choices=("per-iteration", "batch"),
        default="per-iteration",
        help=("per-iteration waits after every prepared call; batch enqueues the "
              "whole loop and synchronizes once after the loop"),
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--validate-every-iteration", action="store_true",
        help="change input, poison output, and validate every repeated call; diagnostic only",
    )
    parser.add_argument("--profile-iters", type=int, default=20)
    parser.add_argument("--profile-root", default="/home/l00934901/profiling")
    parser.add_argument("--trace-job", default="",
                        help="stable P/Q label written into per-iteration trace files")
    parser.add_argument("--iteration-trace-dir", default="",
                        help="write rank0/rank1 host-monotonic intervals as JSON")
    parser.add_argument("--contention-sync-dir", default="",
                        help="shared directory used to release jobs P and Q together")
    parser.add_argument(
        "--compile-backend",
        choices=("none", "eager", "aot_eager", "npu", "npugraphs", "npugraph_ex", "inductor", "aclgraph"),
        default="none",
        help=("compile the prepared graph op; eager checks Dynamo fullgraph, npugraphs checks framework "
              "integration, and aclgraph performs explicit torch.npu.NPUGraph capture/replay"),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--runtime-check", action="store_true",
        help="validate DeepEP and route-library loading without initializing NPU devices",
    )
    args = parser.parse_args()
    element_bytes = {"float16": 2, "bfloat16": 2, "float32": 4, "int8": 1, "int32": 4}[args.dtype]
    if args.bytes <= 0 or args.bytes % element_bytes:
        parser.error("--bytes must be a positive multiple of dtype element size")
    if args.implementation == "standard" and args.dtype != "float32":
        parser.error("the historical ACLNN standard experiment is outside the multi-dtype prepared-plan change")
    if args.warmup < 0 or args.iters <= 0 or args.profile_iters <= 0:
        parser.error("invalid warmup/iters/profile-iters")
    if args.profile and args.profile_iters != args.iters:
        parser.error(
            "profiling measures the same formal iterations as --iters; "
            "set --profile-iters equal to --iters"
        )
    if args.completion_mode == "batch" and args.validate_every_iteration:
        parser.error("--validate-every-iteration is incompatible with batch completion")
    if args.runtime_check:
        return args
    if args.trace_job:
        try:
            safe_name(args.trace_job)
        except ValueError as error:
            parser.error(str(error))
    if bool(args.iteration_trace_dir) != bool(args.trace_job):
        parser.error("--iteration-trace-dir and --trace-job must be supplied together")
    if args.contention_sync_dir and args.trace_job not in ("P", "Q"):
        parser.error("--contention-sync-dir requires --trace-job P or Q")
    if args.iteration_trace_dir and args.implementation != "prepared":
        parser.error("per-iteration contention traces require --implementation prepared")
    if args.implementation == "native" and args.schedule != "concurrent":
        parser.error("--schedule applies only to --implementation multiroute")
    if args.compile_backend != "none" and args.implementation not in ("prepared", "standard"):
        parser.error("--compile-backend requires --implementation prepared or standard")
    if args.implementation == "standard" and args.compile_backend not in ("none", "aclgraph"):
        parser.error("the standard dynamic-policy operator currently validates graph replay with aclgraph")
    if args.implementation in ("explicit", "standard", "prepared"):
        if not args.relay_manifest:
            parser.error(f"--implementation {args.implementation} requires --relay-manifest")
        if args.schedule != "concurrent":
            parser.error(
                "the formal explicit-path operator is concurrent-only; "
                "use --implementation multiroute for serialized control experiments"
            )
        if not args.path_weights:
            parser.error(f"--implementation {args.implementation} requires direct+relay --path-weights")
        weights = [item for item in args.path_weights.split(",") if item]
        if len(weights) < 2 or len(weights) > 64 or any(
                not item.isdigit() or int(item) <= 0 for item in weights):
            parser.error("explicit --path-weights must contain 2..64 positive integers")
        if args.implementation == "standard":
            if len(weights) > 13 or any(int(item) > 15 for item in weights):
                parser.error("standard runtime PathPolicy supports 2..13 weights in [1, 15]")
            if args.replay_path_weights:
                replay = args.replay_path_weights.split(",")
                if len(replay) != len(weights) or any(
                        not item.isdigit() or int(item) > 15 for item in replay
                ) or not any(int(item) for item in replay):
                    parser.error("--replay-path-weights must match path count, use 0..15, and enable a path")
    return args


def select_routes(args):
    if args.synthetic_route_manifest:
        manifest = Path(args.synthetic_route_manifest).resolve()
        if not manifest.is_file():
            raise RuntimeError(f"synthetic route manifest not found: {manifest}")
        os.environ["A5_CCU_SYNTHETIC_ROUTE_MANIFEST"] = str(manifest)
        os.environ["A5_CCU_REBUILD_PUBLIC_FIELDS"] = "1"
        os.environ.pop("A5_CCU_SOURCE_ROUTE_MANIFEST", None)
        os.environ.pop("A5_CCU_SOURCE_ROUTE_PROVIDER", None)
        os.environ.pop("A5_CCU_SYNTHETIC_RANK0_LOCAL_EID", None)
        os.environ.pop("A5_CCU_SYNTHETIC_RANK0_REMOTE_EID", None)
    else:
        os.environ.pop("A5_CCU_SYNTHETIC_ROUTE_MANIFEST", None)
    if args.path_uids:
        os.environ["A5_CCU_PATH_UIDS"] = args.path_uids
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ.pop("A5_CCU_ROUTE_INDEX", None)
    elif args.route_indices:
        os.environ.pop("A5_CCU_PATH_UIDS", None)
        os.environ["A5_CCU_ROUTE_INDICES"] = args.route_indices
        os.environ.pop("A5_CCU_ROUTE_INDEX", None)
    else:
        os.environ.pop("A5_CCU_PATH_UIDS", None)
        os.environ.pop("A5_CCU_ROUTE_INDICES", None)
        os.environ["A5_CCU_ROUTE_INDEX"] = str(args.route_index)
    os.environ["A5_CCU_ROUTE_SCHEDULE"] = args.schedule
    if args.path_weights:
        os.environ["A5_CCU_PATH_WEIGHTS"] = args.path_weights
    else:
        os.environ.pop("A5_CCU_PATH_WEIGHTS", None)


def main():
    mark_phase("parse_args")
    args = parse_args()
    if args.runtime_check:
        print(
            f"MULTIPATH_RUNTIME_VERIFIED extension={EXTENSION} "
            f"route_library={ROUTE_LIB}",
            flush=True,
        )
        return
    if args.implementation == "multiroute":
        select_routes(args)
    elif args.implementation in ("explicit", "standard", "prepared"):
        os.environ.pop("A5_CCU_ROUTE_SCHEDULE", None)
        for name in ("A5_CCU_SYNTHETIC_ROUTE_MANIFEST", "A5_CCU_PATH_UIDS",
                     "A5_CCU_ROUTE_INDICES", "A5_CCU_ROUTE_INDEX",
                     "A5_CCU_PATH_WEIGHTS"):
            os.environ.pop(name, None)
        if args.implementation == "standard":
            manifest = str(Path(args.relay_manifest).resolve())
            os.environ["A5_CCU_EXPLICIT_MULTIPATH_MANIFEST"] = manifest
            os.environ["A5_CCU_EXPLICIT_MULTIPATH_PLAN_ID"] = args.plan_id
    if args.debug:
        os.environ["A5_CCU_DEBUG"] = "1"
    else:
        os.environ.pop("A5_CCU_DEBUG", None)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 2:
        raise RuntimeError("this test requires WORLD_SIZE=2")
    mark_phase("set_device")
    torch.npu.set_device(local_rank)
    mark_phase("init_process_group")
    pg_init_file = os.environ.get("A5_CCU_PG_INIT_FILE", "")
    if pg_init_file:
        pg_init_path = Path(pg_init_file).resolve()
        pg_init_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"CASE_PG_INIT rank={rank} method=file path={pg_init_path}",
            flush=True,
        )
        dist.init_process_group(
            "hccl",
            init_method=f"file://{pg_init_path}",
            rank=rank,
            world_size=world_size,
        )
    else:
        print(f"CASE_PG_INIT rank={rank} method=env", flush=True)
        dist.init_process_group("hccl")
    mark_phase("init_process_group_done")
    buffer = None
    if args.implementation != "native":
        mark_phase("deep_ep_buffer_init")
        buffer = deep_ep.Buffer(dist.group.WORLD, num_nvl_bytes=0, num_rdma_bytes=0)
        mark_phase("deep_ep_buffer_init_done")

    explicit_weights = [int(item) for item in args.path_weights.split(",") if item]
    replay_weights = ([int(item) for item in args.replay_path_weights.split(",")]
                      if args.replay_path_weights else explicit_weights)
    relay_manifest = str(Path(args.relay_manifest).resolve()) if args.relay_manifest else ""
    if relay_manifest and not Path(relay_manifest).is_file():
        raise RuntimeError(f"relay manifest not found: {relay_manifest}")

    capture_stream = torch_npu.npu.Stream() if args.compile_backend == "aclgraph" else None
    plan_handle = 0
    if args.implementation == "prepared":
        mark_phase("prepare_explicit_multipath_plan")
        if capture_stream is None:
            plan_handle = buffer.prepare_ccu_urma_explicit_multipath_plan(
                args.plan_id, relay_manifest, args.direct_route, explicit_weights
            )
        else:
            # HcclThreadAcquireWithStream binds the prepared CCU resources to
            # the stream. Explicit ACLGraph validation therefore prepares the
            # plan while its capture stream is current.
            with torch_npu.npu.stream(capture_stream):
                plan_handle = buffer.prepare_ccu_urma_explicit_multipath_plan(
                    args.plan_id, relay_manifest, args.direct_route, explicit_weights
                )
        if plan_handle <= 0:
            raise RuntimeError(f"invalid prepared plan handle: {plan_handle}")
        print(f"CASE_PREPARED_PLAN rank={rank} plan_id={args.plan_id} "
              f"plan_handle={plan_handle} paths={len(explicit_weights)}", flush=True)
        mark_phase("prepare_explicit_multipath_plan_done")

    dtype = getattr(torch, args.dtype)
    elements = args.bytes // torch.empty((), dtype=dtype).element_size()
    # Allocate the final two-rank tensors directly.  Building a list of peer
    # tensors and stacking it adds a temporary bytes_per_peer * world_size
    # peak, which is avoidable at the 256 MiB endpoint.
    send = torch.empty((world_size, elements), dtype=dtype, device="npu")
    for dst in range(world_size):
        send[dst].fill_(float(rank * 100 + dst + 1))
    recv = torch.empty_like(send)
    path_policy = torch.tensor(
        [encode_path_policy(explicit_weights)], dtype=torch.int64, device="npu"
    ) if args.implementation == "standard" else None
    expected = torch.empty_like(send)
    for src in range(world_size):
        expected[src].fill_(float(src * 100 + rank + 1))

    def check_result():
        try:
            torch.testing.assert_close(recv, expected, rtol=0, atol=0)
        except AssertionError:
            mismatch_by_row = torch.count_nonzero(
                recv != expected, dim=1
            ).detach().cpu().tolist()
            print(
                f"CASE_DATA_DIAGNOSTIC rank={rank} phase={CURRENT_PHASE} "
                f"mismatch_by_row={mismatch_by_row} "
                f"recv_heads={recv[:, 0].detach().cpu().tolist()} "
                f"recv_tails={recv[:, -1].detach().cpu().tolist()} "
                f"expected_heads={expected[:, 0].detach().cpu().tolist()} "
                f"send_heads={send[:, 0].detach().cpu().tolist()}",
                flush=True,
            )
            raise

    compiled_prepared_op = None
    acl_graph = None
    if args.compile_backend not in ("none", "aclgraph"):
        mark_phase(f"compile_graph_{args.compile_backend}")

        def prepared_graph_op(send_tensor):
            return torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
                send_tensor, plan_handle, explicit_weights
            )

        compiled_prepared_op = torch.compile(
            prepared_graph_op,
            backend=args.compile_backend,
            fullgraph=True,
            dynamic=False,
        )
        print(
            f"CASE_GRAPH_CAPTURE rank={rank} backend={args.compile_backend} "
            f"fullgraph=1 dynamic=0 plan_handle={plan_handle}",
            flush=True,
        )

    def run_op():
        nonlocal recv, acl_graph
        if acl_graph is not None:
            acl_graph.replay()
            return
        if args.implementation == "native":
            dist.all_to_all_single(recv, send)
        elif args.implementation == "standard":
            recv = buffer.explicit_multipath_all2all_ccu(
                send, path_policy, args.plan_id
            )
        elif args.implementation == "prepared":
            if compiled_prepared_op is None:
                buffer.ccu_urma_prepared_multipath_alltoall_policy_out(
                    send, recv, plan_handle, explicit_weights
                )
            else:
                recv = compiled_prepared_op(send)
        elif args.implementation == "explicit":
            buffer.ccu_urma_explicit_multipath_alltoall_out(
                send, recv, relay_manifest, args.direct_route, explicit_weights
            )
        else:
            buffer.ccu_urma_multiroute_alltoall_out(send, recv)

    complete_each_invocation = (
        args.implementation == "prepared" and
        args.completion_mode == "per-iteration"
    )
    print(
        f"CASE_LAUNCH_COMPLETION rank={rank} mode={args.completion_mode} "
        f"synchronize_inside_loop={int(complete_each_invocation)}",
        flush=True,
    )

    iteration_records = []

    def run_repeated(count, stage):
        for index in range(count):
            if args.validate_every_iteration:
                # Leave these producers in the normal torch_npu host queue.
                # The bridge must order its raw ACL/HCOMM launch after them.
                # Fresh input and poisoned output also reject stale replay.
                send.add_(17)
                expected.add_(17)
                recv.fill_(-128)
            trace_iteration = stage == "measure" and bool(args.iteration_trace_dir)
            started_ns = time.monotonic_ns() if trace_iteration else None
            marker = (
                torch.autograd.profiler.record_function(
                    f"A5_PATH_CONTENTION_{args.trace_job}_ITER_{index + 1:04d}"
                )
                if trace_iteration and args.profile else nullcontext()
            )
            with marker:
                run_op()
                if complete_each_invocation:
                    torch.npu.synchronize()
            if trace_iteration:
                ended_ns = time.monotonic_ns()
                iteration_records.append({
                    "iteration": index + 1,
                    "started_ns": started_ns,
                    "ended_ns": ended_ns,
                    "duration_us": (ended_ns - started_ns) / 1000.0,
                    "timing_semantics": (
                        "completed_invocation" if complete_each_invocation
                        else "host_submission_only"
                    ),
                })
            if args.validate_every_iteration:
                try:
                    check_result()
                except AssertionError:
                    mismatch_count = int(torch.count_nonzero(recv != expected).cpu().item())
                    recv_heads = recv[:, 0].detach().cpu().tolist()
                    expected_heads = expected[:, 0].detach().cpu().tolist()
                    print(
                        f"CASE_ITERATION_FAILURE rank={rank} stage={stage} "
                        f"iteration={index + 1} mismatch_count={mismatch_count} "
                        f"recv_heads={recv_heads} expected_heads={expected_heads} "
                        f"send_heads={send[:, 0].detach().cpu().tolist()}",
                        flush=True,
                    )
                    raise
                print(
                    f"CASE_ITERATION_PASS rank={rank} stage={stage} iteration={index + 1}",
                    flush=True,
                )

    run_id = sanitize(
        f"{os.environ.get('TORCHELASTIC_RUN_ID', 'standalone')}_"
        f"{os.environ.get('MASTER_PORT', '0')}")
    case_tag = sanitize(
        "native" if args.implementation == "native" else
        (f"{args.implementation}_direct{args.direct_route}_{len(explicit_weights) - 1}relay_{args.schedule}"
         if args.implementation in ("explicit", "standard", "prepared") else
         f"multiroute_{args.path_uids or args.route_indices or args.route_index}_{args.schedule}"))
    sync_dir = Path("/tmp") / f"a5_ccu_urma_a2a_{case_tag}_{run_id}"

    if args.compile_backend == "aclgraph":
        # Prime all lazy host/runtime state on the exact capture stream. Plan
        # creation already installed stream-bound HCOMM resources, so neither
        # capture nor replay may acquire a Channel or register a CCU kernel.
        mark_phase("aclgraph_stream_warmup")
        # Inputs were produced on the caller stream. A new capture stream has
        # no automatic dependency on those producers.
        capture_stream.wait_stream(torch_npu.npu.current_stream())
        with torch_npu.npu.stream(capture_stream):
            if args.implementation == "standard":
                recv = buffer.explicit_multipath_all2all_ccu(
                    send, path_policy, args.plan_id
                )
            else:
                recv = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
                    send, plan_handle, explicit_weights
                )
        torch.npu.synchronize()
        check_result()
        file_barrier(sync_dir, "aclgraph_stream_warmup", rank, world_size)

        mark_phase("aclgraph_capture")
        file_barrier(sync_dir, "aclgraph_capture_begin", rank, world_size)
        acl_graph = torch.npu.NPUGraph()
        with torch_npu.npu.graph(
            acl_graph, stream=capture_stream, auto_dispatch_capture=True
        ):
            if args.implementation == "standard":
                recv = buffer.explicit_multipath_all2all_ccu(
                    send, path_policy, args.plan_id
                )
            else:
                recv = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy(
                    send, plan_handle, explicit_weights
                )
        torch.npu.synchronize()
        file_barrier(sync_dir, "aclgraph_capture_done", rank, world_size)
        print(f"CASE_ACLGRAPH_CAPTURE rank={rank} PASS", flush=True)

        # Mutating the stable captured input makes stale-output replay fail.
        send.add_(7)
        expected.add_(7)
        if args.implementation == "standard":
            path_policy.fill_(encode_path_policy(replay_weights))
            print(f"CASE_DYNAMIC_PATH_POLICY rank={rank} weights={replay_weights}", flush=True)
        torch.npu.synchronize()
        mark_phase("aclgraph_replay")
        acl_graph.replay()
        torch.npu.synchronize()
        check_result()
        file_barrier(sync_dir, "aclgraph_replay", rank, world_size)
        print(f"CASE_ACLGRAPH_REPLAY rank={rank} PASS", flush=True)
    elif compiled_prepared_op is not None:
        # torch.compile is lazy. Separate the first compile/execute from a
        # second replay so logs and profiler results can distinguish them.
        mark_phase(f"graph_first_execute_{args.compile_backend}")
        run_op()
        torch.npu.synchronize()
        check_result()
        file_barrier(sync_dir, "graph_first_execute", rank, world_size)
        print(f"CASE_GRAPH_FIRST_EXECUTE rank={rank} backend={args.compile_backend} PASS", flush=True)

        # Change the captured input storage before the second execution. A
        # successful equality check now proves that replay consumed current
        # data instead of returning the first execution's stale output.
        send.add_(7)
        expected.add_(7)
        torch.npu.synchronize()
        mark_phase(f"graph_replay_{args.compile_backend}")
        run_op()
        torch.npu.synchronize()
        check_result()
        file_barrier(sync_dir, "graph_replay", rank, world_size)
        print(f"CASE_GRAPH_REPLAY rank={rank} backend={args.compile_backend} PASS", flush=True)

    mark_phase("warmup")
    run_repeated(args.warmup, "warmup")
    torch.npu.synchronize()
    mark_phase("warmup_correctness")
    check_result()
    file_barrier(sync_dir, "warmup_done", rank, world_size)

    if args.compile_backend == "aclgraph":
        measurement_mode = "aclgraph_replay"
    elif args.compile_backend == "none":
        measurement_mode = "prepared_eager" if args.implementation == "prepared" else "eager"
    else:
        measurement_mode = f"torch_compile_{args.compile_backend}"
    print(
        f"CASE_MEASUREMENT_MODE rank={rank} mode={measurement_mode} "
        f"capture_in_timing=0",
        flush=True,
    )

    profiler = None
    profile_dir = None
    if args.profile:
        mark_phase("profiling")
        profile_dir = (Path(args.profile_root) /
                       f"a5_ccu_alltoall_{case_tag}_{run_id}" / f"rank{rank}")
        profiler = make_profiler(profile_dir, rank)
        print(
            f"CASE_PROFILE_SCOPE rank={rank} warmup_excluded={args.warmup} "
            f"measured_iterations={args.iters}",
            flush=True,
        )
        profiler.start()
    else:
        mark_phase("timing")
    try:
        torch.npu.synchronize()
        if args.contention_sync_dir:
            start_lateness_us = contention_measurement_start(
                args.contention_sync_dir, args.trace_job, rank, world_size
            )
        else:
            start_lateness_us = measurement_start(sync_dir, rank, world_size)
        batch_started_ns = time.monotonic_ns()
        run_repeated(args.iters, "measure")
        torch.npu.synchronize()
        batch_ended_ns = time.monotonic_ns()
        batch_total_us = (batch_ended_ns - batch_started_ns) / 1000.0
        average_us = batch_total_us / args.iters
    finally:
        if profiler is not None:
            profiler.stop()
            print(f"[rank={rank}] profiling={profile_dir.resolve()}", flush=True)
    check_result()

    if args.iteration_trace_dir:
        trace_path = write_iteration_trace(
            args.iteration_trace_dir,
            args.trace_job,
            rank,
            {
                "bytes_per_peer": args.bytes,
                "warmup_iterations": args.warmup,
                "iterations": args.iters,
                "implementation": args.implementation,
                "completion_mode": args.completion_mode,
                "iteration_timing_semantics": (
                    "completed_invocation" if complete_each_invocation
                    else "host_submission_only"
                ),
                "batch_started_ns": batch_started_ns,
                "batch_ended_ns": batch_ended_ns,
                "batch_total_us": batch_total_us,
                "batch_average_us": average_us,
                "profile_enabled": bool(args.profile),
                "profile_scope": "measure_only_after_warmup",
                "start_lateness_us": start_lateness_us,
            },
            iteration_records,
        )
        print(f"CASE_ITERATION_TRACE rank={rank} path={trace_path}", flush=True)

    (sync_dir / f"average.rank{rank}.json").write_text(json.dumps(average_us))
    file_barrier(sync_dir, "timing_results", rank, world_size)
    averages = [json.loads((sync_dir / f"average.rank{item}.json").read_text())
                for item in range(world_size)]
    result_us = max(averages)

    if rank == 0:
        result = {
            "self_copy_mode": os.environ.get("A5_CCU_SELF_COPY_MODE", "overlap"),
            "dtype": args.dtype,
            "implementation": args.implementation,
            "paths": ((f"direct{args.direct_route}+{len(explicit_weights) - 1}relay")
                      if args.implementation in ("explicit", "standard", "prepared") else
                      (args.path_uids or args.route_indices or str(args.route_index)
                       if args.implementation == "multiroute" else "native")),
            "weights": args.path_weights or "1",
            "schedule": args.schedule if args.implementation != "native" else "native",
            "bytes_per_peer": args.bytes,
            "warmup": args.warmup,
            "iterations": args.iters,
            "host_batch_avg_us": result_us,
            "synthetic_route_manifest": args.synthetic_route_manifest or "",
            "relay_manifest": relay_manifest,
            "compile_backend": args.compile_backend,
            "measurement_mode": measurement_mode,
            "measurement_start_lateness_us": start_lateness_us,
            "trace_job": args.trace_job,
            "iteration_trace_dir": args.iteration_trace_dir,
        }
        print("RESULT_JSON " + json.dumps(result, sort_keys=True), flush=True)
        print(f"PASS: implementation={result['implementation']} paths={result['paths']} "
              f"schedule={result['schedule']} bytes_per_peer={args.bytes} "
              f"host_batch_avg_us={result_us:.3f}", flush=True)
        if args.implementation == "multiroute":
            print(f"deep_ep_cpp={EXTENSION}", flush=True)
            print(f"route_library={ROUTE_LIB}", flush=True)
    mark_phase("reported_barrier")
    file_barrier(sync_dir, "reported", rank, world_size)
    if args.implementation == "prepared" and args.compile_backend in ("none", "aclgraph"):
        mark_phase("release_copy_plan")
        torch.npu.synchronize()
        if acl_graph is not None:
            acl_graph.reset()
            acl_graph = None
        runtime = ctypes.CDLL(str(ROUTE_LIB))
        release = runtime.HcclCcuUrmaExplicitMultipathPlanDestroy
        release.argtypes = [ctypes.c_uint64]
        release.restype = ctypes.c_int
        status = release(plan_handle)
        if status != 0:
            raise RuntimeError(f"prepared plan destroy failed: status={status}")
    mark_phase("destroy_process_group")
    dist.destroy_process_group()
    faulthandler.cancel_dump_traceback_later()
    mark_phase("complete")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            f"CASE_FAILURE rank={os.environ.get('RANK', 'NA')} "
            f"phase={CURRENT_PHASE} type={type(error).__name__} error={error}",
            flush=True,
        )
        raise
