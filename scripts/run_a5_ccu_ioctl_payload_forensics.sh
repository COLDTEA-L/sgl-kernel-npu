#!/usr/bin/env bash
set -euo pipefail

devices="2,3"
repeats=3
timeout_seconds=180
output_root=/home/l00934901/profiling
requests="0x402c5801,0x402c580a,0x40345810,0x40806b05,0x40806b06,0x40806b07,0xc0085707,0xc0085708,0xc008570a,0xc0104500,0xc0105501,0xc020550c,0xc02c5800,0xc0345812,0xc060580f"
max_events=2048
max_per_request=64

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --repeats) repeats=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --requests) requests=$2; shift 2 ;;
        --max-events) max_events=$2; shift 2 ;;
        --max-per-request) max_per_request=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
run_dir="${output_root}/a5_ccu_ioctl_payload_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir/cases"
printf 'case\trepeat\tbase\tdonor\tmutation\tstatus\n' >"$run_dir/case_status.tsv"

# Never preload the outer shell/build process.  The route runner injects this
# library only into its two final worker executables.
unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX
make -C examples/a5_urma_tp_trace >"$run_dir/tracer_build.log" 2>&1
make -C examples/a5_ccu_urma_route_probe/testcase >"$run_dir/testcase_build.log" 2>&1
trace_so=$(readlink -f examples/a5_urma_tp_trace/liba5_urma_tp_trace.so)

export A5_CCU_TRACE_LINK=1
export A5_CCU_SEQUENTIAL_ACQUIRE_TRACE=1
export A5_CCU_CHANNEL_HOLD_SECONDS=0
export A5_IOCTL_PAYLOAD_REQUESTS="$requests"
export A5_IOCTL_PAYLOAD_MAX_EVENTS="$max_events"
export A5_IOCTL_PAYLOAD_MAX_PER_REQUEST="$max_per_request"
export A5_IOCTL_PAYLOAD_CAPTURE_BYTES=512

run_case() {
    local name=$1 repeat=$2 base=$3 donor=$4 mutation=$5
    local case_dir="$run_dir/cases/${name}_r${repeat}"
    mkdir -p "$case_dir"
    local -a command=(bash scripts/run_a5_ccu_urma_route_probe.sh
        --devices "$devices" --route-index "$base" --bytes 4096 --warmup 1 --iters 1
        --channel-only --skip-build --worker-preload "$trace_so"
        --worker-trace-prefix "$case_dir/ioctl_payload")
    if [[ "$mutation" != none ]]; then
        command+=(--descriptor-mutation "$mutation" --descriptor-donor "$donor")
    fi
    local status=0
    echo "[$(date --iso-8601=seconds)] BEGIN ${name}_r${repeat}"
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
        stdbuf -oL -eL "${command[@]}" >"$case_dir/run.log" 2>&1 || status=$?
    echo "[$(date --iso-8601=seconds)] END   ${name}_r${repeat} status=${status}"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$repeat" "$base" "$donor" "$mutation" "$status" >>"$run_dir/case_status.tsv"
    printf '%s\n' "$status" >"$case_dir/status.txt"
}

for ((repeat=1; repeat<=repeats; ++repeat)); do
    run_case candidate0 "$repeat" 0 NA none
    run_case candidate2 "$repeat" 2 NA none
    run_case c20_addr0 "$repeat" 2 0 both_comm_addrs
    run_case c21_addr2 "$repeat" 0 2 both_comm_addrs
done

python3 scripts/analyze_a5_ccu_ioctl_payload.py --run-dir "$run_dir"
echo "Payload forensic result: $run_dir"
echo "Primary report: $run_dir/ioctl_payload_report.md"
