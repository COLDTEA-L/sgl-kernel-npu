#!/usr/bin/env bash
set -uo pipefail

devices="6,7"
candidates="0,2"
bytes=4194304
warmup=3
iterations=20
repeats=1
timeout_seconds=180
output_root=/home/l00934901/profiling
urma_include=/usr/include/ub/umdk/urma
hccn_devices="0,1,2,3,4,5,6,7"
skip_footprint=0
syscall_trace=0
perf_callgraph=0
urma_trace=0
cases_filter=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --candidates) candidates=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --urma-include) urma_include=$2; shift 2 ;;
        --hccn-devices) hccn_devices=$2; shift 2 ;;
        --skip-footprint) skip_footprint=1; shift ;;
        --syscall-trace) syscall_trace=1; shift ;;
        --perf-callgraph) perf_callgraph=1; shift ;;
        --urma-trace) urma_trace=1; shift ;;
        --cases) cases_filter=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

IFS=',' read -r candidate_a candidate_b extra <<<"$candidates"
[[ -n "${candidate_a:-}" && -n "${candidate_b:-}" && -z "${extra:-}" ]] || {
    echo "--candidates must contain exactly two ordinals, for example 0,2" >&2
    exit 2
}
[[ "$repeats" =~ ^[1-9][0-9]*$ ]] || { echo "--repeats must be positive" >&2; exit 2; }
[[ "$timeout_seconds" =~ ^[1-9][0-9]*$ ]] || { echo "--timeout-seconds must be positive" >&2; exit 2; }

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
run_dir="${output_root}/a5_ccu_channel_selector_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"/{inventory,cases}
status_file="$run_dir/case_status.tsv"
printf 'case\trepeat\tbase\tdonor\tmutation\tchannel_status\tdata_status\n' >"$status_file"

trace_so=""
if (( urma_trace )); then
    make -C examples/a5_urma_tp_trace clean all URMA_INCLUDE="$urma_include" \
        >"$run_dir/inventory/urma_trace_build.log" 2>&1 || {
        echo "URMA tracer build failed: $run_dir/inventory/urma_trace_build.log" >&2
        exit 1
    }
    trace_so=$(readlink -f examples/a5_urma_tp_trace/liba5_urma_tp_trace.so)
else
    printf '%s\n' 'disabled: use --urma-trace only for targeted follow-up runs' \
        >"$run_dir/inventory/urma_trace_build.log"
fi

{
    echo "timestamp=$(date --iso-8601=ns)"
    echo "devices=$devices"
    echo "candidates=$candidates"
    echo "urma_trace=$urma_trace"
    echo "cases=${cases_filter:-all}"
    echo "kernel=$(uname -r)"
    echo "machine=$(uname -m)"
    echo "cann=${ASCEND_HOME_PATH:-unset}"
    command -v urma_admin || true
    command -v hccn_tool || true
    command -v perf || true
    command -v strace || true
} >"$run_dir/inventory/environment.txt"
env -u LD_PRELOAD urma_admin show >"$run_dir/inventory/urma_admin_show.txt" 2>&1 || true

for library in \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhccl.so \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhcomm.so \
    /usr/local/Ascend/driver/lib64/driver/libascend_hal.so \
    /lib64/liburma.so /lib64/urma/liburma-udma.so \
    /usr/lib64/liburma.so /usr/lib64/urma/liburma-udma.so; do
    [[ -e "$library" ]] || continue
    base=$(basename "$library")
    {
        echo "path=$library"
        sha256sum "$library" || true
        readelf -n "$library" 2>/dev/null | grep -A2 'Build ID' || true
        readelf -Ws "$library" 2>/dev/null | grep -E -i \
            'ChannelAcquire|ChannelCreate|ChannelGetStatus|ConnectChannels|CcuUrma|Endpoint|LinkData|TpInfo|HdcSession|import_jfr' || true
    } >"$run_dir/inventory/${base}.inventory.txt"
done

declare -a case_specs=(
    "c00_base_a|${candidate_a}||none"
    "c01_base_b|${candidate_b}||none"
    "c02_b_local_endpoint_from_a|${candidate_b}|${candidate_a}|local_endpoint"
    "c03_b_remote_endpoint_from_a|${candidate_b}|${candidate_a}|remote_endpoint"
    "c04_b_both_endpoints_from_a|${candidate_b}|${candidate_a}|both_endpoints"
    "c05_a_local_endpoint_from_b|${candidate_a}|${candidate_b}|local_endpoint"
    "c06_a_remote_endpoint_from_b|${candidate_a}|${candidate_b}|remote_endpoint"
    "c07_a_both_endpoints_from_b|${candidate_a}|${candidate_b}|both_endpoints"
    "c08_b_local_comm_addr_from_a|${candidate_b}|${candidate_a}|local_comm_addr"
    "c09_b_remote_comm_addr_from_a|${candidate_b}|${candidate_a}|remote_comm_addr"
    "c10_b_both_comm_addrs_from_a|${candidate_b}|${candidate_a}|both_comm_addrs"
    "c11_b_locations_from_a|${candidate_b}|${candidate_a}|locations"
)

if [[ -n "$cases_filter" ]]; then
    IFS=',' read -r -a requested_cases <<<"$cases_filter"
    for requested_case in "${requested_cases[@]}"; do
        found=0
        for spec in "${case_specs[@]}"; do
            [[ "${spec%%|*}" == "$requested_case" ]] && { found=1; break; }
        done
        (( found == 1 )) || {
            echo "unknown case in --cases: $requested_case" >&2
            exit 2
        }
    done
fi

run_command() {
    local output_prefix=$1
    shift
    local -a command=("$@")
    if (( syscall_trace )); then
        command=(strace -ff -ttt -e trace=ioctl,connect -o "${output_prefix}.strace" "${command[@]}")
    fi
    if (( perf_callgraph )); then
        command=(perf record -o "${output_prefix}.perf.data" -e cpu-clock -g \
            --call-graph dwarf -- "${command[@]}")
    fi
    timeout --signal=TERM --kill-after=5 "${timeout_seconds}" "${command[@]}"
}

run_one_case() {
    local case_name=$1
    local base=$2
    local donor=$3
    local mutation=$4
    local repeat=$5
    local case_dir="$run_dir/cases/${case_name}_r${repeat}"
    mkdir -p "$case_dir"
    printf 'case\trepeat\tbase\tdonor\tmutation\tdevices\n%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$repeat" "$base" "${donor:-NA}" "$mutation" "$devices" \
        >"$case_dir/case_meta.tsv"

    local -a mutation_args=()
    if [[ "$mutation" != "none" ]]; then
        mutation_args=(--descriptor-mutation "$mutation" --descriptor-donor "$donor")
    fi
    local channel_status=0
    (
        export A5_CCU_TRACE_LINK=1
        export A5_CCU_SEQUENTIAL_ACQUIRE_TRACE=1
        export A5_CCU_CHANNEL_HOLD_SECONDS=0
        unset A5_URMA_TP_TRACE_PREFIX LD_PRELOAD
        if (( urma_trace )); then
            export A5_URMA_TP_TRACE_PREFIX="$case_dir/urma_tp"
            export LD_PRELOAD="$trace_so"
        fi
        run_command "$case_dir/channel" stdbuf -oL -eL \
            bash scripts/run_a5_ccu_urma_route_probe.sh \
            --devices "$devices" --route-index "$base" --bytes 4096 \
            --warmup 1 --iters 1 --channel-only "${mutation_args[@]}"
    ) >"$case_dir/channel.log" 2>&1 || channel_status=$?
    printf '%s\n' "$channel_status" >"$case_dir/channel_status.txt"

    local data_status=125
    if (( channel_status == 0 && skip_footprint == 0 )); then
        data_status=0
        (
            export A5_CCU_TRACE_LINK=1
            unset A5_URMA_TP_TRACE_PREFIX LD_PRELOAD
            if (( urma_trace )); then
                export A5_URMA_TP_TRACE_PREFIX="$case_dir/data_urma_tp"
                export LD_PRELOAD="$trace_so"
            fi
            run_command "$case_dir/data" stdbuf -oL -eL \
                bash scripts/run_a5_ccu_urma_route_probe.sh \
                --devices "$devices" --route-index "$base" --bytes "$bytes" \
                --warmup "$warmup" --iters "$iterations" --remote-only \
                --hccn-stat --hccn-devices "$hccn_devices" \
                --hccn-stat-root "$case_dir" "${mutation_args[@]}"
        ) >"$case_dir/data.log" 2>&1 || data_status=$?
    fi
    printf '%s\n' "$data_status" >"$case_dir/data_status.txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$case_name" "$repeat" "$base" "${donor:-NA}" "$mutation" \
        "$channel_status" "$data_status" >>"$status_file"

    if [[ -f "$case_dir/channel.perf.data" ]]; then
        perf script -i "$case_dir/channel.perf.data" >"$case_dir/channel.perf.txt" 2>&1 || true
    fi
}

for repeat in $(seq 1 "$repeats"); do
    for spec in "${case_specs[@]}"; do
        IFS='|' read -r case_name base donor mutation <<<"$spec"
        if [[ -n "$cases_filter" ]]; then
            case ",$cases_filter," in
                *",$case_name,"*) ;;
                *) continue ;;
            esac
        fi
        echo "===== ${case_name} repeat=${repeat} base=${base} donor=${donor:-NA} mutation=${mutation} ====="
        run_one_case "$case_name" "$base" "$donor" "$mutation" "$repeat"
    done
done

python3 scripts/analyze_a5_ccu_channel_path_selector.py \
    --run-dir "$run_dir" --candidates "$candidates" --devices "$devices"
analysis_status=$?
echo "Selector forensic result: $run_dir"
echo "Primary report: $run_dir/channel_path_selector_report.md"
exit "$analysis_status"
