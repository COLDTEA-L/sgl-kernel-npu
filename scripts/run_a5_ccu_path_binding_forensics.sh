#!/usr/bin/env bash
set -uo pipefail

devices="6,7"
candidates="0,2"
bytes=4194304
warmup=10
iterations=30
repeats=2
output_root=/home/l00934901/profiling
urma_include=/usr/include/ub/umdk/urma
hccn_devices="0,1,2,3,4,5,6,7"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --candidates) candidates=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --urma-include) urma_include=$2; shift 2 ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
IFS=',' read -r candidate_a candidate_b extra <<<"$candidates"
[[ -n "${candidate_a:-}" && -n "${candidate_b:-}" && -z "${extra:-}" ]] || {
    echo "--candidates must contain exactly two route-probe ordinals" >&2; exit 2;
}

run_dir="${output_root}/a5_ccu_path_binding_forensics_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"/{control,same_process,footprint,inventory}
status_file="$run_dir/case_status.tsv"
printf 'case\tstatus\n' >"$status_file"

record_case() {
    local name=$1; shift
    echo "===== ${name} ====="
    "$@"
    local status=$?
    printf '%s\t%s\n' "$name" "$status" >>"$status_file"
    return 0
}

make -C examples/a5_urma_tp_trace clean all URMA_INCLUDE="$urma_include" \
    >"$run_dir/inventory/tracer_build.log" 2>&1 || {
    echo "tracer build failed; see $run_dir/inventory/tracer_build.log" >&2; exit 1;
}

for library in \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhccl.so \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhcomm.so \
    /usr/local/Ascend/driver/lib64/driver/libascend_hal.so \
    /lib64/liburma.so /lib64/urma/liburma-udma.so; do
    [[ -e "$library" ]] || continue
    base=$(basename "$library")
    { echo "===== nm -D $library ====="; nm -D "$library" 2>&1 || true;
      echo "===== readelf -Ws $library ====="; readelf -Ws "$library" 2>&1 || true;
    } >"$run_dir/inventory/${base}.symbols.txt"
done

record_case control_candidates bash scripts/run_a5_ccu_channel_trace.sh \
    --devices "$devices" --routes "$candidates" --hold-seconds 1 \
    --output-root "$run_dir/control" --urma-include "$urma_include" \
    --skip-resource-snapshot

record_case same_process bash scripts/run_a5_ccu_same_process_tp_binding.sh \
    --devices "$devices" --routes "$candidates" --repeats "$repeats" \
    --output-root "$run_dir/same_process" --urma-include "$urma_include"

run_footprint() {
    local name=$1
    local route_args=$2
    local output="$run_dir/footprint/$name"
    mkdir -p "$output"
    local args=(--devices "$devices" --bytes "$bytes" --warmup "$warmup"
        --iters "$iterations" --remote-only --hccn-stat
        --hccn-devices "$hccn_devices" --hccn-stat-root "$output")
    if [[ "$route_args" == *,* ]]; then
        args+=(--route-indices "$route_args")
    else
        args+=(--route-index "$route_args")
    fi
    bash scripts/run_a5_ccu_urma_route_probe.sh "${args[@]}" >"$output/workload.log" 2>&1
}

record_case "candidate_${candidate_a}_footprint" run_footprint "candidate_${candidate_a}" "$candidate_a"
record_case "candidate_${candidate_b}_footprint" run_footprint "candidate_${candidate_b}" "$candidate_b"
record_case concurrent_footprint run_footprint concurrent "$candidates"

python3 scripts/analyze_a5_ccu_path_binding_forensics.py \
    --run-dir "$run_dir" --candidates "$candidates" \
    >"$run_dir/analyzer.log" 2>&1
analysis_status=$?
printf 'analysis\t%s\n' "$analysis_status" >>"$status_file"

echo "Forensic result: $run_dir"
echo "Primary report: $run_dir/path_binding_report.md"
exit "$analysis_status"
