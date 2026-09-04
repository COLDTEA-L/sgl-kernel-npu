#!/usr/bin/env bash
set -eo pipefail

visible_devices="${ASCEND_RT_VISIBLE_DEVICES:-2,3}"
comm_ranks="0,1"
elements_per_peer=11804800
warmup=10
iters=100
metrics="PipeUtilization"
output_root="/home/l00934901/profiling"

usage() {
    echo "Usage: $0 [options]"
    echo "  --visible-devices IDS   Physical device IDs, default: ${visible_devices}"
    echo "  --comm-ranks IDS        Logical communication ranks, default: ${comm_ranks}"
    echo "  --elements-per-peer N   int32 elements sent to each peer, default: ${elements_per_peer}"
    echo "  --warmup N              Warmup iterations, default: ${warmup}"
    echo "  --iters N               Profiled test iterations, default: ${iters}"
    echo "  --metrics GROUP         msprof AIC metric group, default: ${metrics}"
    echo "  --output-root DIR       Result root, default: ${output_root}"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --visible-devices) visible_devices="$2"; shift 2 ;;
        --comm-ranks) comm_ranks="$2"; shift 2 ;;
        --elements-per-peer) elements_per_peer="$2"; shift 2 ;;
        --warmup) warmup="$2"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --metrics) metrics="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

IFS=',' read -r -a device_list <<< "${visible_devices}"
nproc=${#device_list[@]}
if (( nproc < 2 )); then
    echo "At least two visible devices are required" >&2
    exit 2
fi
if (( elements_per_peer <= 0 || warmup < 0 || iters <= 0 )); then
    echo "Invalid elements/warmup/iters value" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

set +u
if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
fi
source python/deep_ep/deep_ep/vendors/hwcomputing/bin/set_env.bash
set -u

command -v msprof >/dev/null || { echo "msprof not found in PATH" >&2; exit 1; }
mkdir -p "${output_root}"
timestamp="$(date +%Y%m%d_%H%M%S)"
metric_tag="${metrics//[^[:alnum:]_-]/_}"
run_dir="${output_root}/a5_all2all_${nproc}card_${metric_tag}_${timestamp}"
mkdir -p "${run_dir}"

export ASCEND_RT_VISIBLE_DEVICES="${visible_devices}"
export HCCL_BUFFSIZE=2300
export DEEP_USE_MODE=default
unset HCCL_OP_EXPANSION_MODE
unset ASCEND_LAUNCH_BLOCKING
unset A5_DETOUR_DEBUG_WINDOWS

echo "msprof output : ${run_dir}"
echo "devices       : ${visible_devices} (${nproc} processes)"
echo "comm ranks    : ${comm_ranks}"
echo "elements/peer : ${elements_per_peer} int32"
echo "AIC metrics   : ${metrics}"

msprof \
    --output="${run_dir}" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=l2 \
    --ai-core=on \
    --aic-mode=task-based \
    --aic-metrics="${metrics}" \
    --hccl=on \
    --type=text \
    python3 -m torch.distributed.run \
        --standalone \
        --nproc-per-node="${nproc}" \
        tests/python/deepep/test_a5_aiv_urma_all2all_detour.py \
        --comm-ranks "${comm_ranks}" \
        --elements-per-peer "${elements_per_peer}" \
        --warmup "${warmup}" \
        --iters "${iters}"

prof_dir="$({
    find "${run_dir}" -maxdepth 2 -type d -name 'PROF_*' -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | awk 'NR==1 {print $2}')"
if [[ -n "${prof_dir}" ]]; then
    msprof --export=on --output="${prof_dir}"
    echo "Exported profile: ${prof_dir}"
    echo "CSV files:"
    find "${prof_dir}" -type f -name '*.csv' | sort
else
    echo "Collection finished, but no PROF_* directory was found under ${run_dir}" >&2
    exit 1
fi
