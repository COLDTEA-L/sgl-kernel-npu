#!/usr/bin/env bash
set -euo pipefail

devices="4,5"
bytes=$((2 * 1024 * 1024))
warmup=10
iters=100
rounds=3
profile=0
profile_iters=20
profile_root="/home/l00934901/profiling"
require_overlap_ratio=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices="$2"; shift 2 ;;
        --bytes) bytes="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --rounds) rounds="$2"; shift 2 ;;
        --profile) profile=1; shift ;;
        --profile-iters) profile_iters="$2"; shift 2 ;;
        --profile-root) profile_root="$2"; shift 2 ;;
        --require-overlap-ratio) require_overlap_ratio="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
source /usr/local/Ascend/cann-9.1.T560/set_env.sh
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
export ASCEND_RT_VISIBLE_DEVICES="$devices"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-2300}
export PYTHONUNBUFFERED=1
unset A5_CCU_DEBUG ASCEND_LAUNCH_BLOCKING

if (( bytes <= 0 || bytes % 256 != 0 )); then
    echo "--bytes must be a positive multiple of 256" >&2
    exit 2
fi
if (( rounds <= 0 )); then
    echo "--rounds must be positive" >&2
    exit 2
fi

route0_bytes=$((bytes * 2 / 3 / 256 * 256))
route2_bytes=$((bytes - route0_bytes))
work_dir=$(mktemp -d /tmp/a5_ccu_a2a_concurrency.XXXXXX)
cleanup() {
    find "$work_dir" -type f -delete 2>/dev/null || true
    rmdir "$work_dir" 2>/dev/null || true
}
trap cleanup EXIT

test_script="tests/python/deepep/test_a5_ccu_urma_multiroute_all2all.py"
run_case() {
    local name=$1
    shift
    local log_file="$work_dir/${name}.log"
    echo "===== ${name} =====" >&2
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "$test_script" "$@" --warmup "$warmup" --iters "$iters" \
        2>&1 | tee "$log_file" >&2
    local result
    result=$(sed -n 's/^RESULT_JSON //p' "$log_file" | tail -n 1)
    if [[ -z "$result" ]]; then
        echo "missing RESULT_JSON in ${name}" >&2
        return 1
    fi
    printf '%s' "$result"
}

route0_result=$(run_case route0_share --implementation multiroute \
    --route-index 0 --bytes "$route0_bytes")
route2_result=$(run_case route2_share --implementation multiroute \
    --route-index 2 --bytes "$route2_bytes")

native_results=()
serial_results=()
concurrent_results=()
for ((round = 1; round <= rounds; ++round)); do
    native_results+=("$(run_case native_${round} --implementation native --bytes "$bytes")")
    if (( round % 2 == 1 )); then
        serial_results+=("$(run_case serial_${round} --implementation multiroute \
            --route-indices 0,2 --schedule serial --bytes "$bytes")")
        concurrent_results+=("$(run_case concurrent_${round} --implementation multiroute \
            --route-indices 0,2 --schedule concurrent --bytes "$bytes")")
    else
        concurrent_results+=("$(run_case concurrent_${round} --implementation multiroute \
            --route-indices 0,2 --schedule concurrent --bytes "$bytes")")
        serial_results+=("$(run_case serial_${round} --implementation multiroute \
            --route-indices 0,2 --schedule serial --bytes "$bytes")")
    fi
done

summary=(python3 scripts/summarize_a5_ccu_urma_concurrency.py
    --bytes "$bytes" --route0-bytes "$route0_bytes" --route2-bytes "$route2_bytes"
    --route0 "$route0_result" --route2 "$route2_result"
    --native "${native_results[@]}"
    --serial "${serial_results[@]}"
    --concurrent "${concurrent_results[@]}")
if [[ -n "$require_overlap_ratio" ]]; then
    summary+=(--require-overlap-ratio "$require_overlap_ratio")
fi
"${summary[@]}"

if (( profile )); then
    mkdir -p "$profile_root"
    test -w "$profile_root"
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "$test_script" --implementation native --bytes "$bytes" \
        --warmup "$warmup" --iters 1 --profile --profile-iters "$profile_iters" \
        --profile-root "$profile_root"
    python3 -m torch.distributed.run --standalone --nproc-per-node=2 \
        "$test_script" --implementation multiroute --route-indices 0,2 \
        --schedule concurrent --bytes "$bytes" --warmup "$warmup" --iters 1 \
        --profile --profile-iters "$profile_iters" --profile-root "$profile_root"
fi
