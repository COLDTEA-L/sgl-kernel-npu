#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
test_dir="${repo_root}/examples/a5_ccu_urma_route_probe/testcase"

devices="2,3"
route_index=0
bytes=2097152
warmup=10
iterations=100
profile=0
remote_only=0
channel_only=0
sweep=0
profile_root=/home/l00934901/profiling

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices) devices=$2; shift 2 ;;
        --route-index) route_index=$2; shift 2 ;;
        --bytes) bytes=$2; shift 2 ;;
        --warmup) warmup=$2; shift 2 ;;
        --iters) iterations=$2; shift 2 ;;
        --remote-only) remote_only=1; shift ;;
        --channel-only) channel_only=1; shift ;;
        --sweep) sweep=1; shift ;;
        --profile) profile=1; shift ;;
        --profile-root) profile_root=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

(( remote_only == 0 || channel_only == 0 )) || {
    echo "--remote-only and --channel-only cannot be used together" >&2
    exit 2
}
(( sweep == 0 || channel_only == 0 )) || {
    echo "--sweep and --channel-only cannot be used together" >&2
    exit 2
}
(( sweep == 0 || profile == 0 )) || {
    echo "--sweep and --profile cannot be used together; profile one size at a time" >&2
    exit 2
}

if [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

# This is an HCCL custom package installed under the active CANN tree.  A
# DeepEP vendor environment may redirect ASCEND_CUSTOM_OPP_PATH elsewhere.
unset ASCEND_CUSTOM_OPP_PATH

[[ "${devices}" == *,* && "${devices}" != *,*,* ]] || {
    echo "--devices must contain exactly two physical device IDs, for example 2,3" >&2
    exit 2
}

export ASCEND_RT_VISIBLE_DEVICES="${devices}"
export HCCL_OP_EXPANSION_MODE=CCU_SCHED
export A5_CCU_ROUTE_INDEX="${route_index}"
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"

make -C "${test_dir}"

build_command() {
    local payload_bytes=$1
    command=("${test_dir}/a5_ccu_urma_route_probe_test"
        --bytes "${payload_bytes}"
        --warmup "${warmup}"
        --iters "${iterations}"
        --route-index "${route_index}")
    (( remote_only == 0 )) || command+=(--remote-only)
    (( channel_only == 0 )) || command+=(--channel-only)
}

if (( sweep != 0 )); then
    echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "Selected route  : ${A5_CCU_ROUTE_INDEX}"
    echo "Mode            : $([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather)"
    for sweep_bytes in 65536 262144 1048576 2097152 8388608 33554432; do
        echo "===== payload ${sweep_bytes} bytes ====="
        build_command "${sweep_bytes}"
        "${command[@]}"
    done
    exit 0
fi

build_command "${bytes}"

echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Selected route  : ${A5_CCU_ROUTE_INDEX}"
echo "Payload/rank    : ${bytes} bytes"
echo "Mode            : $([[ ${channel_only} -eq 1 ]] && echo channel-only || \
    ([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather))"

if (( profile == 0 )); then
    "${command[@]}"
    exit 0
fi

command -v msprof >/dev/null || { echo "msprof not found in PATH" >&2; exit 1; }
run_dir="${profile_root}/a5_ccu_urma_route_${route_index}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${run_dir}"
msprof \
    --output="${run_dir}" \
    --ascendcl=on \
    --runtime-api=on \
    --task-time=l2 \
    --hccl=on \
    --type=text \
    "${command[@]}"

mapfile -d '' prof_dirs < <(find "${run_dir}" -type d -name 'PROF_*' -print0 2>/dev/null)
(( ${#prof_dirs[@]} > 0 )) || { echo "No PROF_* directory under ${run_dir}" >&2; exit 1; }
for prof_dir in "${prof_dirs[@]}"; do
    msprof --export=on --output="${prof_dir}"
done
echo "Profile output: ${run_dir}"
find "${run_dir}" -type f -name '*.csv' | sort
