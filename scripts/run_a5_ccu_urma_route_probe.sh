#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/.." && pwd)
test_dir="${repo_root}/examples/a5_ccu_urma_route_probe/testcase"

devices="2,3"
route_index=0
route_indices=""
source_route_manifest=""
source_route_provider=""
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
        --route-indices) route_indices=$2; shift 2 ;;
        --source-route-manifest) source_route_manifest=$2; shift 2 ;;
        --source-route-provider) source_route_provider=$2; shift 2 ;;
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
if [[ -n "${route_indices}" ]]; then
    export A5_CCU_ROUTE_INDICES="${route_indices}"
else
    unset A5_CCU_ROUTE_INDICES
fi
if [[ -n "${source_route_manifest}" ]]; then
    export A5_CCU_SOURCE_ROUTE_MANIFEST="${source_route_manifest}"
    export A5_CCU_SOURCE_ROUTE_PROVIDER="${source_route_provider}"
else
    unset A5_CCU_SOURCE_ROUTE_MANIFEST A5_CCU_SOURCE_ROUTE_PROVIDER
fi
export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/cust/lib64:${LD_LIBRARY_PATH:-}"

make -C "${test_dir}"

build_command() {
    local payload_bytes=$1
    local worker_rank=$2
    local root_info_file=$3
    command=("${test_dir}/a5_ccu_urma_route_probe_test"
        --bytes "${payload_bytes}"
        --warmup "${warmup}"
        --iters "${iterations}"
        --route-index "${route_index}"
        --worker-rank "${worker_rank}"
        --root-info-file "${root_info_file}")
    [[ -z "${route_indices}" ]] || command+=(--route-indices "${route_indices}")
    (( remote_only == 0 )) || command+=(--remote-only)
    (( channel_only == 0 )) || command+=(--channel-only)
}

run_pair() {
    local payload_bytes=$1
    local pair_state_dir
    pair_state_dir=$(mktemp -d /tmp/a5_ccu_urma_pair.XXXXXX)
    local root_info_file="${pair_state_dir}/root_info.bin"
    local status0=0
    local status1=0

    build_command "${payload_bytes}" 0 "${root_info_file}"
    local rank0_command=("${command[@]}")
    build_command "${payload_bytes}" 1 "${root_info_file}"
    local rank1_command=("${command[@]}")

    if (( profile == 0 )); then
        "${rank0_command[@]}" &
        local rank0_pid=$!
        "${rank1_command[@]}" &
        local rank1_pid=$!
        wait "${rank0_pid}" || status0=$?
        wait "${rank1_pid}" || status1=$?
    else
        command -v msprof >/dev/null || { echo "msprof not found in PATH" >&2; return 1; }
        local run_dir="${profile_root}/a5_ccu_urma_routes_${route_indices:-${route_index}}_$(date +%Y%m%d_%H%M%S)"
        mkdir -p "${run_dir}/rank0" "${run_dir}/rank1"
        msprof --output="${run_dir}/rank0" --ascendcl=on --runtime-api=on \
            --task-time=l2 --hccl=on --type=text "${rank0_command[@]}" &
        local rank0_pid=$!
        msprof --output="${run_dir}/rank1" --ascendcl=on --runtime-api=on \
            --task-time=l2 --hccl=on --type=text "${rank1_command[@]}" &
        local rank1_pid=$!
        wait "${rank0_pid}" || status0=$?
        wait "${rank1_pid}" || status1=$?
        if (( status0 == 0 && status1 == 0 )); then
            mapfile -d '' prof_dirs < <(find "${run_dir}" -type d -name 'PROF_*' -print0 2>/dev/null)
            (( ${#prof_dirs[@]} > 0 )) || { echo "No PROF_* directory under ${run_dir}" >&2; return 1; }
            for prof_dir in "${prof_dirs[@]}"; do
                msprof --export=on --output="${prof_dir}"
            done
            echo "Profile output: ${run_dir}"
            find "${run_dir}" -type f -name '*.csv' | sort
        fi
    fi
    find "${pair_state_dir}" -type f -delete
    rmdir "${pair_state_dir}"
    (( status0 == 0 && status1 == 0 )) || {
        echo "rank workers failed: rank0=${status0}, rank1=${status1}" >&2
        return 1
    }
}

if (( sweep != 0 )); then
    echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
    echo "Selected routes : ${route_indices:-${A5_CCU_ROUTE_INDEX}}"
    echo "Mode            : $([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather)"
    for sweep_bytes in 65536 262144 1048576 2097152 8388608 33554432; do
        echo "===== payload ${sweep_bytes} bytes ====="
        run_pair "${sweep_bytes}"
    done
    exit 0
fi

echo "Physical devices : ${ASCEND_RT_VISIBLE_DEVICES}"
echo "Selected routes : ${route_indices:-${A5_CCU_ROUTE_INDEX}}"
echo "Payload/rank    : ${bytes} bytes"
echo "Mode            : $([[ ${channel_only} -eq 1 ]] && echo channel-only || \
    ([[ ${remote_only} -eq 1 ]] && echo remote-only || echo allgather))"

run_pair "${bytes}"
