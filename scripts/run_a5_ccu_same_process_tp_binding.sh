#!/usr/bin/env bash
set -euo pipefail

devices="6,7"
routes="0,2"
repeats=3
output_root=/home/l00934901/profiling
urma_include=/usr/include/ub/umdk/urma

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --routes) routes=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --urma-include) urma_include=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -r route_a route_b extra <<<"$routes"
[[ -n "${route_a:-}" && -n "${route_b:-}" && -z "${extra:-}" ]] || {
    echo "--routes must contain exactly two ordinals, for example 0,2" >&2
    exit 2
}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || {
    echo "--repeats must be a positive integer" >&2
    exit 2
}

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
trace_dir=examples/a5_urma_tp_trace
make -C "$trace_dir" URMA_INCLUDE="$urma_include"
trace_so=$(readlink -f "$trace_dir/liba5_urma_tp_trace.so")

run_dir="${output_root}/a5_ccu_same_process_tp_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"

run_order() {
    local order=$1
    local repeat=$2
    local case_dir="$run_dir/order_${order//,/_}_r${repeat}"
    mkdir -p "$case_dir"
    (
        export A5_CCU_TRACE_LINK=1
        export A5_CCU_SEQUENTIAL_ACQUIRE_TRACE=1
        export A5_CCU_CHANNEL_HOLD_SECONDS=0
        export A5_URMA_TP_TRACE_PREFIX="$case_dir/urma_tp"
        export LD_PRELOAD="$trace_so${LD_PRELOAD:+:$LD_PRELOAD}"
        bash scripts/run_a5_ccu_urma_route_probe.sh \
            --devices "$devices" --route-indices "$order" --bytes 4096 \
            --warmup 1 --iters 1 --channel-only
    ) >"$case_dir/channel.log" 2>&1
}

for repeat in $(seq 1 "$repeats"); do
    run_order "${route_a},${route_b}" "$repeat"
    run_order "${route_b},${route_a}" "$repeat"
done

python3 scripts/analyze_a5_ccu_same_process_tp_binding.py \
    --run-dir "$run_dir" --output "$run_dir/tp_binding_by_path.tsv"
echo "Same-process TP binding result: $run_dir"
