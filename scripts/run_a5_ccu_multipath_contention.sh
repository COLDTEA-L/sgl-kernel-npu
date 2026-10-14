#!/usr/bin/env bash
# Two prepared direct+relay AllToAll jobs sharing physical directions 2->4 and 4->2.
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
            echo 'A5 direct+relay contention: P=2,3 relay4; Q=4,5 relay2.'
            echo 'Defaults: 16KiB,64KiB,256KiB,1MiB,4MiB,16MiB,64MiB,256MiB; warmup=10 iters=100 repeats=1.'
            echo 'Data plane is fixed to CCU GroupCopy8x32K self-copy and baseline Channel synchronization; D2D overlap/packed modes are rejected.'
            echo 'HCCL buffer defaults to auto: max(200MB, two-rank payload in MiB + 64MB); override with --hccl-buffsize-mb N.'
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
from a5_ccu_path_contention_support import parse_sizes, resolve_hccl_buffer_mb

def pair(text, name):
    items = text.split(',')
    if len(items) != 2 or len(set(items)) != 2 or any(not item.isdecimal() for item in items):
        raise SystemExit(f"{name} must contain two distinct physical card IDs")
    return tuple(map(int, items))

p = pair(p_devices, "--p-devices")
q = pair(q_devices, "--q-devices")
if not p_relay.isdecimal() or not q_relay.isdecimal():
    raise SystemExit("relay IDs must be nonnegative integers")
if int(p_relay) in p or int(q_relay) in q:
    raise SystemExit("a relay cannot also be an endpoint of its own job")
if p != (2, 3) or q != (4, 5) or int(p_relay) != 4 or int(q_relay) != 2:
    raise SystemExit("this first experiment is intentionally pinned to P=2,3/relay4 and Q=4,5/relay2")
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
run_dir=$(mktemp -d "${output_root}/a5_path_contention_2_3_4_5_$(date +%Y%m%d_%H%M%S).XXXXXX")
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
resolve_manifest 2 3 "$p_relay" "$p_relay_plane" "$run_dir/manifests/P.tsv"
resolve_manifest 4 5 "$q_relay" "$q_relay_plane" "$run_dir/manifests/Q.tsv"

python3 - "$run_dir" "$sizes" "$warmup" "$iterations" "$repeats" \
    "$weights" "$mindstudio_profile" "$p_relay_plane" "$q_relay_plane" \
    "$hccl_buffsize_mb" "$repo_root" "$self_copy_mode" "$channel_sync_mode" <<'PY'
import json, pathlib, sys
root, sizes, warmup, iterations, repeats, weights, profile, p_plane, q_plane, hccl_buffer, repo, self_copy, channel_sync = sys.argv[1:]
sys.path.insert(0, repo + "/tests/python/deepep")
from a5_ccu_path_contention_support import resolve_hccl_buffer_mb
size_values = list(map(int, sizes.split(',')))
settings = dict(
    experiment="two_prepared_multipath_jobs_shared_directed_hops",
    jobs={
        "P": {"endpoints": [2, 3], "relay": 4, "relay_plane": p_plane},
        "Q": {"endpoints": [4, 5], "relay": 2, "relay_plane": q_plane},
    },
    expected_shared_physical_directions=["2->4", "4->2"],
    sizes=size_values, weights=list(map(int, weights.split(','))),
    warmup=int(warmup), iterations=int(iterations), repeats=int(repeats),
    mindstudio_profile=bool(int(profile)), implementation="prepared",
    self_copy_mode=self_copy,
    channel_sync_mode=channel_sync,
    completion_mode="batch",
    hccl_buffsize_policy=dict(
        setting=hccl_buffer,
        unit="MB",
        auto_formula="max(200, ceil(2 * bytes_per_peer / MiB) + 64)",
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

run_job() {
    local job=$1 phase=$2 size=$3 repeat=$4 sync_dir=$5 log=$6
    local devices manifest profile_args=() case_path size_label case_hccl_buffsize_mb
    size_label=$(format_size_label "$size")
    case_hccl_buffsize_mb=$(resolve_hccl_buffsize_mb "$size")
    if [[ "$phase" == concurrent ]]; then
        case_path="concurrent_PQ/${job}"
    else
        case_path="baseline_${job}"
    fi
    if [[ "$job" == P ]]; then
        devices=$p_devices
        manifest="$run_dir/manifests/P.tsv"
    else
        devices=$q_devices
        manifest="$run_dir/manifests/Q.tsv"
    fi
    if ((mindstudio_profile != 0)); then
        profile_args=(--profile --profile-iters "$iterations"
            --profile-root "$run_dir/profiles/size_${size_label}/repeat_${repeat}/${case_path}")
    fi
    local sync_args=()
    [[ -z "$sync_dir" ]] || sync_args=(--contention-sync-dir "$sync_dir")
    (
        export ASCEND_RT_VISIBLE_DEVICES="$devices"
        export HCCL_BUFFSIZE="$case_hccl_buffsize_mb"
        echo "CASE_MEMORY_PLAN size=${size_label} bytes_per_peer=${size} HCCL_BUFFSIZE_MB=${HCCL_BUFFSIZE}" >&2
        timeout --signal=TERM --kill-after=10 "$timeout_seconds" \
            python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
            "$test_script" --implementation prepared --relay-manifest "$manifest" \
            --direct-route 0 --path-weights "$weights" \
            --plan-id "path-contention-${job}-${phase}-${size_label}-${repeat}" \
            --bytes "$size" --dtype float32 --warmup "$warmup" --iters "$iterations" \
            --completion-mode batch \
            --self-copy-mode "$self_copy_mode" --channel-sync-mode "$channel_sync_mode" \
            --trace-job "$job" \
            --iteration-trace-dir "$run_dir/traces/size_${size_label}/repeat_${repeat}/${case_path}" \
            "${sync_args[@]}" "${profile_args[@]}"
    ) >"$log" 2>&1
}

record_status() {
    local size=$1 repeat=$2 phase=$3 job=$4 status=$5 log=$6 result=FAIL
    local policy_count group_copy_count
    policy_count=$(grep -c 'CASE_DATA_PLANE_POLICY .*self_copy=ccu channel_sync=baseline' "$log" || true)
    group_copy_count=$(grep -c '^CCU_LOCAL_REMOTE_PLAN .*self_copy=GroupCopy8x32K' "$log" || true)
    if ((status == 0 && policy_count == 2 && group_copy_count == 2)) && \
       grep -q '^PASS: implementation=prepared' "$log" && \
       ! grep -q '^SELF_COPY_LANE ' "$log"; then
        result=PASS
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$size" "$repeat" "$phase" "$job" "$status" "$result" \
        >>"$run_dir/case_status.tsv"
    [[ "$result" == PASS ]]
}

IFS=',' read -ra size_array <<<"$sizes"
for size in "${size_array[@]}"; do
    size_label=$(format_size_label "$size")
    for ((repeat=1; repeat<=repeats; ++repeat)); do
        trace_root="$run_dir/traces/size_${size_label}/repeat_${repeat}"
        mkdir -p "$trace_root"
        for job in P Q; do
            log="$run_dir/logs/size_${size_label}_repeat_${repeat}_baseline_${job}.log"
            status=0
            run_job "$job" baseline "$size" "$repeat" "" "$log" || status=$?
            if ! record_status "$size" "$repeat" baseline "$job" "$status" "$log"; then
                echo "baseline ${job} failed: size=${size} repeat=${repeat} status=${status}" >&2
                tail -n 100 "$log" >&2
                exit 1
            fi
        done

        sync_dir="$run_dir/sync/size_${size_label}/repeat_${repeat}"
        mkdir -p "$sync_dir"
        p_log="$run_dir/logs/size_${size_label}_repeat_${repeat}_concurrent_PQ_P.log"
        q_log="$run_dir/logs/size_${size_label}_repeat_${repeat}_concurrent_PQ_Q.log"
        set +e
        run_job P concurrent "$size" "$repeat" "$sync_dir" "$p_log" &
        p_pid=$!
        run_job Q concurrent "$size" "$repeat" "$sync_dir" "$q_log" &
        q_pid=$!
        wait "$p_pid"; p_status=$?
        wait "$q_pid"; q_status=$?
        set -e
        p_ok=0; q_ok=0
        record_status "$size" "$repeat" concurrent P "$p_status" "$p_log" && p_ok=1 || true
        record_status "$size" "$repeat" concurrent Q "$q_status" "$q_log" && q_ok=1 || true
        if ((p_ok == 0 || q_ok == 0)); then
            echo "concurrent jobs failed: size=${size} repeat=${repeat} P=${p_status} Q=${q_status}" >&2
            tail -n 100 "$p_log" >&2
            tail -n 100 "$q_log" >&2
            exit 1
        fi
    done
done

python3 "${script_dir}/analyze_a5_ccu_multipath_contention.py" --run-dir "$run_dir"
echo "Result directory: $run_dir"
column -s $'\t' -t "$run_dir/case_status.tsv" 2>/dev/null || cat "$run_dir/case_status.tsv"
sed -n '1,260p' "$run_dir/path_contention_report.md"
