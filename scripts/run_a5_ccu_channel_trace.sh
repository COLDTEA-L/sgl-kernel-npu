#!/usr/bin/env bash
set -euo pipefail

devices="4,5"
routes="0,2"
hold_seconds=20
output_root=/home/l00934901/profiling
urma_include=/usr/include/ub/umdk/urma

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --routes) routes=$2; shift 2 ;;
        --hold-seconds) hold_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --urma-include) urma_include=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
trace_dir=examples/a5_urma_tp_trace
make -C "$trace_dir" URMA_INCLUDE="$urma_include"
trace_so=$(readlink -f "$trace_dir/liba5_urma_tp_trace.so")

run_dir="${output_root}/a5_ccu_channel_trace_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"
env -u LD_PRELOAD urma_admin show -w >"$run_dir/urma_admin_show.txt" 2>&1 || true

snapshot_resources() {
    local case_dir=$1
    local phase=$2
    local devices_file="$run_dir/urma_admin_show.txt"
    mapfile -t ub_devices < <(awk 'NR > 2 && $1 ~ /^[0-9]+$/ {print $2}' "$devices_file" | sort -u)
    if (( ${#ub_devices[@]} == 0 )); then
        echo "URMA_RESOURCE_SNAPSHOT status=NO_DEVICE_LIST" >"$case_dir/${phase}_resources.txt"
        return
    fi
    : >"$case_dir/${phase}_resources.txt"
    for dev in "${ub_devices[@]}"; do
        {
            echo "===== dev=${dev} resource=TP ====="
            env -u LD_PRELOAD urma_admin list_res -d "$dev" -R 2 2>&1 || true
            echo "===== dev=${dev} resource=TPG ====="
            env -u LD_PRELOAD urma_admin list_res -d "$dev" -R 3 2>&1 || true
        } >>"$case_dir/${phase}_resources.txt"
    done
}

run_case() {
    local route=$1
    local case_dir="$run_dir/candidate_${route}"
    mkdir -p "$case_dir"
    snapshot_resources "$case_dir" before
    (
        export A5_CCU_TRACE_LINK=1
        export A5_CCU_CHANNEL_HOLD_SECONDS="$hold_seconds"
        export A5_URMA_TP_TRACE_PREFIX="$case_dir/urma_tp"
        export LD_PRELOAD="$trace_so${LD_PRELOAD:+:$LD_PRELOAD}"
        bash scripts/run_a5_ccu_urma_route_probe.sh \
            --devices "$devices" --route-index "$route" --bytes 4096 \
            --warmup 1 --iters 1 --channel-only
    ) >"$case_dir/channel.log" 2>&1 &
    local probe_pid=$!

    local ready=0
    for _ in $(seq 1 600); do
        if grep -q 'CHANNEL_TRACE_HOLD' "$case_dir/channel.log" 2>/dev/null; then
            ready=1
            break
        fi
        if ! kill -0 "$probe_pid" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done
    if (( ready == 0 )); then
        wait "$probe_pid" || true
        echo "candidate ${route} did not reach channel hold; inspect $case_dir/channel.log" >&2
        return 1
    fi
    snapshot_resources "$case_dir" active
    wait "$probe_pid"
    snapshot_resources "$case_dir" after
}

IFS=',' read -ra route_list <<<"$routes"
for route in "${route_list[@]}"; do
    run_case "${route//[[:space:]]/}"
done

python3 scripts/compare_a5_ccu_channel_trace.py \
    --run-dir "$run_dir" --output "$run_dir/channel_trace_comparison.md"
echo "Trace result: $run_dir"

