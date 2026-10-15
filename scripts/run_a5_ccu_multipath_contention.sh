#!/usr/bin/env bash
# Two prepared direct+relay AllToAll jobs sharing a pair of physical directions.
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)

p_devices=2,3
p_relay=4
p_relay_plane=all
q_devices=4,5
q_relay=2
q_relay_plane=all
sizes=16384,65536,262144,1048576,4194304,16777216,67108864,268435456
weights=1,1
warmup=10
iterations=100
repeats=1
mindstudio_profile=1
timeout_seconds=900
hccl_buffsize_mb=auto
self_copy_mode=ccu
channel_sync_mode=baseline
topology="${repo_root}/docs/topology/a5_hccn_device_topology_raw.txt"
topology_json=/usr/local/Ascend/driver/topo/950/atlas_950_1.json
output_root=/home/l00934901/profiling
cann_root=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.1.T560}

while (($#)); do
    case "$1" in
        --p-devices) p_devices=$2; shift 2;;
        --p-relay) p_relay=$2; shift 2;;
        --p-relay-plane) p_relay_plane=$2; shift 2;;
        --q-devices) q_devices=$2; shift 2;;
        --q-relay) q_relay=$2; shift 2;;
        --q-relay-plane) q_relay_plane=$2; shift 2;;
        --sizes) sizes=$2; shift 2;;
        --weights) weights=$2; shift 2;;
        --warmup) warmup=$2; shift 2;;
        --iters) iterations=$2; shift 2;;
        --repeats) repeats=$2; shift 2;;
        --mindstudio-profile) mindstudio_profile=1; shift;;
        --no-mindstudio-profile) mindstudio_profile=0; shift;;
        --timeout-seconds) timeout_seconds=$2; shift 2;;
        --hccl-buffsize-mb) hccl_buffsize_mb=$2; shift 2;;
        --topology) topology=$2; shift 2;;
        --topology-json) topology_json=$2; shift 2;;
        --output-root) output_root=$2; shift 2;;
        --cann-root) cann_root=$2; shift 2;;
        --help)
            echo 'A5 direct+relay contention; defaults: P=2,3 relay4; Q=4,5 relay2.'
            echo "P/Q endpoints must be disjoint; each relay must be the other job's first endpoint."
            echo 'Defaults: 16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB; warmup=10 iters=100 repeats=1.'
            echo 'Data plane is fixed to CCU GroupCopy8x32K self-copy and baseline Channel synchronization; D2D overlap/packed modes are rejected.'
            echo 'P/Q workers, process groups and prepared plans stay alive across every requested size and phase.'
            echo 'HCCL buffer defaults to the maximum requirement across all sizes: max(200MB, two-rank payload in MiB + 64MB).'
            echo 'MindStudio profiling is enabled by default and starts only after warmup; use --no-mindstudio-profile for timing-only runs.'
            exit 0;;
        *) echo "unsupported argument: $1" >&2; exit 2;;
    esac
done

python3 - "$repo_root" "$p_devices" "$p_relay" "$q_devices" "$q_relay" \
    "$p_relay_plane" "$q_relay_plane" "$sizes" "$weights" \
    "$warmup" "$iterations" "$repeats" "$timeout_seconds" "$hccl_buffsize_mb" <<'PY'
import sys
repo, p_devices, p_relay, q_devices, q_relay, p_plane, q_plane, sizes, weights, warmup, iters, repeats, timeout, hccl_buffer = sys.argv[1:]
sys.path.insert(0, repo + "/tests/python/deepep")
from a5_ccu_path_contention_support import (
    parse_sizes, resolve_hccl_buffer_mb, validate_contention_devices,
)

try:
    validate_contention_devices(p_devices, p_relay, q_devices, q_relay)
except ValueError as error:
    raise SystemExit(str(error)) from error
if p_plane not in ("all", "0", "1") or q_plane not in ("all", "0", "1"):
    raise SystemExit("relay plane must be all, 0 or 1")
parsed_sizes = parse_sizes(sizes)
for size in parsed_sizes:
    resolve_hccl_buffer_mb(size, hccl_buffer)
weight_values = weights.split(',')
if len(weight_values) != 2 or any(not value.isdecimal() or int(value) <= 0 for value in weight_values):
    raise SystemExit("--weights must contain positive direct,relay weights")
for value, name in ((warmup, "warmup"), (iters, "iters"), (repeats, "repeats"), (timeout, "timeout")):
    if not value.isdecimal() or int(value) <= 0:
        raise SystemExit(f"{name} must be positive")
PY

[[ -f "$topology" && -f "$topology_json" ]] || {
    echo "topology input missing: $topology or $topology_json" >&2
    exit 2
}

cd "$repo_root"
source "${cann_root}/set_env.sh"
# The vendor environment belongs to the installed DeepEP wheel and is not a
# tracked source-tree file.  Each worker's a5_ccu_test_runtime bootstrap finds
# that wheel, adds its OPP/library paths, and re-execs before importing torch.
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export PYTHONUNBUFFERED=1
unset ASCEND_LAUNCH_BLOCKING A5_CCU_SOURCE_ROUTE_MANIFEST A5_CCU_SOURCE_ROUTE_PROVIDER
unset A5_CCU_SYNTHETIC_ROUTE_MANIFEST A5_CCU_PATH_UIDS A5_CCU_ROUTE_INDICES A5_CCU_ROUTE_INDEX
export A5_CCU_SELF_COPY_MODE="$self_copy_mode"
export A5_CCU_CHANNEL_SYNC_MODE="$channel_sync_mode"

# Validate the actual Python package, extension and route-library pair before
# resolving topology or creating any device communicator.
python3 tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py \
    --implementation prepared --self-copy-mode "$self_copy_mode" \
    --channel-sync-mode "$channel_sync_mode" --runtime-check

mkdir -p "$output_root"
IFS=',' read -r p_src p_dst <<<"$p_devices"
IFS=',' read -r q_src q_dst <<<"$q_devices"
run_dir=$(mktemp -d "${output_root}/a5_path_contention_${p_src}_${p_dst}_${q_src}_${q_dst}_$(date +%Y%m%d_%H%M%S).XXXXXX")
mkdir -p "$run_dir/manifests" "$run_dir/logs" "$run_dir/traces" "$run_dir/profiles"

format_size_label() {
    local byte_count=$1
    if ((byte_count % (1024 * 1024) == 0)); then
        printf '%dMB' "$((byte_count / (1024 * 1024)))"
    elif ((byte_count % 1024 == 0)); then
        printf '%dKB' "$((byte_count / 1024))"
    else
        printf '%dB' "$byte_count"
    fi
}

resolve_hccl_buffsize_mb() {
    local byte_count=$1
    if [[ "$hccl_buffsize_mb" != auto ]]; then
        printf '%d' "$hccl_buffsize_mb"
        return
    fi
    local mib=$((1024 * 1024))
    local rank_payload_mb=$(((2 * byte_count + mib - 1) / mib))
    local value=$((rank_payload_mb + 64))
    ((value < 200)) && value=200
    printf '%d' "$value"
}

resolve_manifest() {
    local src=$1 dst=$2 relay=$3 plane=$4 output=$5
    python3 "${script_dir}/resolve_a5_explicit_multirelay_eids.py" \
        --topology "$topology" --topology-json "$topology_json" \
        --src-phy "$src" --dst-phy "$dst" --relay-phys "$relay" \
        --relay-planes "$plane" --weights 1 >"$output"
}
resolve_manifest "$p_src" "$p_dst" "$p_relay" "$p_relay_plane" "$run_dir/manifests/P.tsv"
resolve_manifest "$q_src" "$q_dst" "$q_relay" "$q_relay_plane" "$run_dir/manifests/Q.tsv"

IFS=',' read -ra size_array <<<"$sizes"
suite_hccl_buffsize_mb=0
for size in "${size_array[@]}"; do
    candidate=$(resolve_hccl_buffsize_mb "$size")
    ((candidate > suite_hccl_buffsize_mb)) && suite_hccl_buffsize_mb=$candidate
done
echo "PERSISTENT_ENVIRONMENT sizes=${#size_array[@]} HCCL_BUFFSIZE_MB=${suite_hccl_buffsize_mb} plan_scope=all_sizes"

python3 - "$run_dir" "$sizes" "$warmup" "$iterations" "$repeats" \
    "$weights" "$mindstudio_profile" "$p_relay_plane" "$q_relay_plane" \
    "$hccl_buffsize_mb" "$repo_root" "$self_copy_mode" "$channel_sync_mode" \
    "$suite_hccl_buffsize_mb" "$p_devices" "$p_relay" "$q_devices" "$q_relay" <<'PY'
import json, pathlib, sys
root, sizes, warmup, iterations, repeats, weights, profile, p_plane, q_plane, hccl_buffer, repo, self_copy, channel_sync, suite_buffer, p_devices, p_relay, q_devices, q_relay = sys.argv[1:]
sys.path.insert(0, repo + "/tests/python/deepep")
from a5_ccu_path_contention_support import resolve_hccl_buffer_mb, validate_contention_devices
size_values = list(map(int, sizes.split(',')))
p, p_relay_id, q, q_relay_id = validate_contention_devices(
    p_devices, p_relay, q_devices, q_relay
)
settings = dict(
    experiment="two_prepared_multipath_jobs_shared_directed_hops",
    jobs={
        "P": {"endpoints": list(p), "relay": p_relay_id, "relay_plane": p_plane},
        "Q": {"endpoints": list(q), "relay": q_relay_id, "relay_plane": q_plane},
    },
    expected_shared_physical_directions=[
        f"{q_relay_id}->{p_relay_id}", f"{p_relay_id}->{q_relay_id}"
    ],
    sizes=size_values, weights=list(map(int, weights.split(','))),
    warmup=int(warmup), iterations=int(iterations), repeats=int(repeats),
    mindstudio_profile=bool(int(profile)), implementation="prepared",
    self_copy_mode=self_copy,
    channel_sync_mode=channel_sync,
    completion_mode="batch",
    execution_lifecycle={
        "worker_processes": "persistent_across_all_sizes_and_phases",
        "process_groups_per_job": 1,
        "prepared_plans_per_rank": 1,
        "plan_reused_across_sizes": True,
    },
    hccl_buffsize_policy=dict(
        setting=hccl_buffer,
        unit="MB",
        auto_formula="max(200, ceil(2 * bytes_per_peer / MiB) + 64)",
        persistent_runtime_mb=int(suite_buffer),
        persistent_runtime_basis="maximum requirement across all requested sizes",
        per_size_mb={str(size): resolve_hccl_buffer_mb(size, hccl_buffer)
                     for size in size_values},
    ),
    size_semantics="bytes_per_peer_slice",
    profile_scope={
        "warmup_excluded": True,
        "warmup_iterations": int(warmup),
        "measured_iterations_per_case": int(iterations),
    },
    per_size_results="results/size_<KB_or_MB_label>",
)
(pathlib.Path(root) / "run_settings.json").write_text(json.dumps(settings, indent=2) + "\n")
PY

printf 'size\trepeat\tphase\tjob\tstatus\tresult\n' >"$run_dir/case_status.tsv"
test_script=tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py

run_persistent_job() {
    local job=$1 log=$2
    local devices manifest profile_args=()
    if [[ "$job" == P ]]; then
        devices=$p_devices
        manifest="$run_dir/manifests/P.tsv"
    else
        devices=$q_devices
        manifest="$run_dir/manifests/Q.tsv"
    fi
    if ((mindstudio_profile != 0)); then
        profile_args=(--profile --profile-iters "$iterations")
    fi
    (
        export ASCEND_RT_VISIBLE_DEVICES="$devices"
        export HCCL_BUFFSIZE="$suite_hccl_buffsize_mb"
        echo "CASE_MEMORY_PLAN scope=all_sizes sizes=${sizes} HCCL_BUFFSIZE_MB=${HCCL_BUFFSIZE}" >&2
        timeout --signal=TERM --kill-after=10 "$suite_timeout_seconds" \
            python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
            "$test_script" --implementation prepared --relay-manifest "$manifest" \
            --direct-route 0 --path-weights "$weights" \
            --plan-id "path-contention-${job}-persistent" \
            --dtype float32 --warmup "$warmup" --iters "$iterations" \
            --completion-mode batch \
            --self-copy-mode "$self_copy_mode" --channel-sync-mode "$channel_sync_mode" \
            --suite-sizes "$sizes" --suite-repeats "$repeats" \
            --suite-job "$job" --suite-run-dir "$run_dir" \
            "${profile_args[@]}"
    ) >"$log" 2>&1
}

print_size_complete() {
    local size=$1 repeat=$2 size_label=$3
    python3 - "$run_dir" "$size" "$repeat" "$size_label" <<'PY'
import json, pathlib, sys
root, size, repeat, label = sys.argv[1:]
base = pathlib.Path(root) / "status" / f"size_{label}" / f"repeat_{repeat}"
cases = {
    "baseline_P": base / "baseline_P",
    "baseline_Q": base / "baseline_Q",
    "concurrent_P": base / "concurrent_PQ" / "P",
    "concurrent_Q": base / "concurrent_PQ" / "Q",
}
values = {}
for name, directory in cases.items():
    rows = [json.loads((directory / f"rank{rank}.json").read_text()) for rank in (0, 1)]
    values[name] = max(float(row["host_batch_avg_us"]) for row in rows)
print(
    f"SIZE_COMPLETE size={label} bytes_per_peer={size} repeat={repeat} "
    f"baseline_P_us={values['baseline_P']:.3f} baseline_Q_us={values['baseline_Q']:.3f} "
    f"concurrent_P_us={values['concurrent_P']:.3f} concurrent_Q_us={values['concurrent_Q']:.3f} "
    f"artifacts={root}/traces/size_{label}/repeat_{repeat}",
    flush=True,
)
PY
}

suite_timeout_seconds=$((timeout_seconds + 180 * ${#size_array[@]} * repeats))
p_log="$run_dir/logs/persistent_P.log"
q_log="$run_dir/logs/persistent_Q.log"
run_persistent_job P "$p_log" &
p_pid=$!
run_persistent_job Q "$q_log" &
q_pid=$!

cleanup_children() {
    kill "$p_pid" "$q_pid" 2>/dev/null || true
    wait "$p_pid" "$q_pid" 2>/dev/null || true
}
trap cleanup_children EXIT INT TERM

for size in "${size_array[@]}"; do
    size_label=$(format_size_label "$size")
    for ((repeat=1; repeat<=repeats; ++repeat)); do
        progress="$run_dir/progress/size_${size_label}_repeat_${repeat}.json"
        while [[ ! -f "$progress" ]]; do
            if ! kill -0 "$p_pid" 2>/dev/null || ! kill -0 "$q_pid" 2>/dev/null; then
                echo "persistent worker exited before size=${size_label} repeat=${repeat} completed" >&2
                cleanup_children
                tail -n 120 "$p_log" >&2 || true
                tail -n 120 "$q_log" >&2 || true
                exit 1
            fi
            sleep 1
        done
        print_size_complete "$size" "$repeat" "$size_label"
    done
done

set +e
wait "$p_pid"; p_status=$?
wait "$q_pid"; q_status=$?
set -e
trap - EXIT INT TERM
if ((p_status != 0 || q_status != 0)); then
    echo "persistent jobs failed: P=${p_status} Q=${q_status}" >&2
    tail -n 120 "$p_log" >&2 || true
    tail -n 120 "$q_log" >&2 || true
    exit 1
fi

for job in P Q; do
    log_var="${job,,}_log"
    log=${!log_var}
    policy_count=$(grep -c 'CASE_DATA_PLANE_POLICY .*self_copy=ccu channel_sync=baseline' "$log" || true)
    group_copy_count=$(grep -c '^CCU_LOCAL_REMOTE_PLAN .*self_copy=GroupCopy8x32K' "$log" || true)
    plan_count=$(grep -c '^CASE_PREPARED_PLAN ' "$log" || true)
    if ((policy_count != 2 || group_copy_count != 2 || plan_count != 2)) || \
       grep -q '^SELF_COPY_LANE ' "$log" || \
       ! grep -q "^PASS: persistent_contention_suite job=${job} " "$log"; then
        echo "persistent ${job} log failed lifecycle/data-plane validation" >&2
        tail -n 120 "$log" >&2
        exit 1
    fi
done

for size in "${size_array[@]}"; do
    size_label=$(format_size_label "$size")
    for ((repeat=1; repeat<=repeats; ++repeat)); do
        for job in P Q; do
            baseline_status="$run_dir/status/size_${size_label}/repeat_${repeat}/baseline_${job}"
            concurrent_status="$run_dir/status/size_${size_label}/repeat_${repeat}/concurrent_PQ/${job}"
            for rank in 0 1; do
                [[ -f "$baseline_status/rank${rank}.json" ]] || { echo "missing $baseline_status/rank${rank}.json" >&2; exit 1; }
                [[ -f "$concurrent_status/rank${rank}.json" ]] || { echo "missing $concurrent_status/rank${rank}.json" >&2; exit 1; }
            done
            printf '%s\t%s\tbaseline\t%s\t0\tPASS\n' "$size" "$repeat" "$job" >>"$run_dir/case_status.tsv"
            printf '%s\t%s\tconcurrent\t%s\t0\tPASS\n' "$size" "$repeat" "$job" >>"$run_dir/case_status.tsv"
        done
    done
done

python3 "${script_dir}/analyze_a5_ccu_multipath_contention.py" --run-dir "$run_dir"
echo "Result directory: $run_dir"
column -s $'\t' -t "$run_dir/case_status.tsv" 2>/dev/null || cat "$run_dir/case_status.tsv"
sed -n '1,260p' "$run_dir/path_contention_report.md"
