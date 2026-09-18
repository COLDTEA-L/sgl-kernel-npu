#!/usr/bin/env bash
set -uo pipefail

devices="2,3"
candidates="0,1,2"
timeout_seconds=180
output_root=/home/l00934901/profiling
with_strace=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --candidates) candidates=$2; shift 2 ;;
        --timeout-seconds) timeout_seconds=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        --no-strace) with_strace=0; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
run_dir="${output_root}/a5_ccu_path_provisioning_${devices//,/_}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$run_dir"/{cases,inventory}
printf 'case\tkind\tstatus\n' >"$run_dir/case_status.tsv"

unset LD_PRELOAD A5_URMA_TP_TRACE_PREFIX
make -C examples/a5_ccu_urma_route_probe/testcase \
    >"$run_dir/inventory/testcase_build.log" 2>&1 || {
    echo "testcase build failed: $run_dir/inventory/testcase_build.log" >&2
    exit 1
}

{
    echo "timestamp=$(date --iso-8601=ns)"
    echo "devices=$devices"
    echo "candidates=$candidates"
    echo "kernel=$(uname -r)"
    echo "machine=$(uname -m)"
    echo "cann=${ASCEND_HOME_PATH:-unset}"
    command -v strace || true
    command -v perf || true
} >"$run_dir/inventory/environment.txt"

for library in \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhccl.so \
    /usr/local/Ascend/cann-9.1.T560/lib64/libhcomm.so \
    /usr/local/Ascend/driver/lib64/driver/libascend_hal.so \
    /usr/local/Ascend/driver/lib64/driver/libhccp.so \
    /lib64/liburma.so /lib64/urma/liburma-udma.so \
    /usr/lib64/liburma.so /usr/lib64/urma/liburma-udma.so; do
    [[ -e "$library" ]] || continue
    base=$(basename "$library")
    {
        echo "path=$library"
        sha256sum "$library" || true
        readelf -n "$library" 2>/dev/null | grep -A2 'Build ID' || true
        readelf -Ws "$library" 2>/dev/null | grep -E -i \
            'channel|endpoint|comm.?addr|path|route|hccp|mue|tp.?list|hdc|ctrlq' || true
    } >"$run_dir/inventory/${base}.inventory.txt"
    strings -a "$library" 2>/dev/null | grep -E -i \
        'channel|endpoint|comm.?addr|path|route|hccp|mue|tp.?list|hdc|ctrlq|next.?hop|relay' \
        >"$run_dir/inventory/${base}.strings.txt" || true
done

run_case() {
    local name=$1 kind=$2
    shift 2
    local case_dir="$run_dir/cases/$name"
    mkdir -p "$case_dir"
    local -a command=(bash scripts/run_a5_ccu_urma_route_probe.sh
        --devices "$devices" --bytes 4096 --warmup 1 --iters 1 --skip-build "$@")
    if (( with_strace )); then
        command=(strace -ff -ttt -T -yy -k -s 512
            -e trace=ioctl,connect,sendto,recvfrom,sendmsg,recvmsg,read,write
            -o "$case_dir/syscall.strace" "${command[@]}")
    fi
    local status=0
    timeout --signal=TERM --kill-after=5 "$timeout_seconds" \
        stdbuf -oL -eL "${command[@]}" >"$case_dir/run.log" 2>&1 || status=$?
    printf '%s\t%s\t%s\n' "$name" "$kind" "$status" >>"$run_dir/case_status.tsv"
    printf '%s\n' "$status" >"$case_dir/status.txt"
}

# Window A: communicator construction/provisioning only.  No RankGraph query and
# no ChannelAcquire are issued by the testcase after HcclCommInitRootInfo.
run_case c00_comm_init_only comm_init --comm-init-only --route-index 0

# Window B: one fresh communicator per candidate, followed by exactly one
# ChannelAcquire.  Candidate 1 is intentionally retained as the failed/partial
# hop-2 control when the current topology exposes it.
IFS=',' read -ra candidate_list <<<"$candidates"
for candidate in "${candidate_list[@]}"; do
    candidate=${candidate//[[:space:]]/}
    [[ "$candidate" =~ ^[0-9]+$ ]] || { echo "invalid candidate: $candidate" >&2; exit 2; }
    run_case "c1_candidate_${candidate}" channel --route-index "$candidate" --channel-only
done

# Causal controls proven by the previous phase: keep the base ordinal fixed but
# replace the complete CommAddr pair with the other working candidate.
if [[ ",$candidates," == *,0,* && ",$candidates," == *,2,* ]]; then
    run_case c20_base2_addrpair_from0 crossover --route-index 2 --channel-only \
        --descriptor-mutation both_comm_addrs --descriptor-donor 0
    run_case c21_base0_addrpair_from2 crossover --route-index 0 --channel-only \
        --descriptor-mutation both_comm_addrs --descriptor-donor 2
fi

python3 scripts/analyze_a5_ccu_path_provisioning.py \
    --run-dir "$run_dir" --candidates "$candidates" --devices "$devices"
status=$?
echo "Provisioning forensic result: $run_dir"
echo "Primary report: $run_dir/path_provisioning_report.md"
exit "$status"
